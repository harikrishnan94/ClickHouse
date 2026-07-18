#pragma once

#include <Interpreters/HashJoin/AddedColumns.h>
#include <Interpreters/HashJoin/HashJoinMethodsImpl.h>
#include <Interpreters/HashJoin/HashJoinResult.h>
#include <Interpreters/HashJoin/JoinUsedFlags.h>
#include <Interpreters/HashJoin/KeyGetter.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/PartitionedHashJoin/AmacRing.h>
#include <Interpreters/PartitionedHashJoin/JoinRouteHashing.h>
#include <Interpreters/PartitionedHashJoin/PartitionedHashJoin.h>
#include <Interpreters/TableJoin.h>
#include <base/scope_guard.h>
#include <Common/HashTable/HashTable.h>
#include <Common/PODArray.h>

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int UNSUPPORTED_JOIN_KEYS;
}

/** The AMAC find policy of the two-phase probe (phase A): out-of-order lookups that only fill
  * the per-row result arrays - the matched cell's mapped value (null = no match) and its
  * used-flags offset, shifted into the shared flag space. Nothing is emitted here; phase B (the
  * sequential in-order loop) consumes the results through the standard `processMatch`.
  * Unlike the build ring, one ring serves MANY maps - each row's leaf - so the slot carries the
  * leaf id and the policy re-resolves the (cache-hot) map header per visit. Cell prefetches use
  * read intent and low locality: a probed cell is not revisited.
  */
template <typename KeyGetter, typename Map, bool need_flags>
struct RoutedAmacFindPolicy
{
    using MapNonConst = std::remove_const_t<Map>;
    using Cell = MapNonConst::cell_type;
    static constexpr bool store_hash = cell_stores_hash<Cell>;
    static constexpr bool may_grow = false;

    struct Slot : public AmacRingSlot<store_hash>
    {
        UInt16 leaf = 0;
    };

    KeyGetter & key_getter;
    const void * const * leaf_maps_data = nullptr;
    const UInt16 * leaf_ids = nullptr; /// null at the single-leaf plan
    const ScatteredBlock::Selector & selector;
    const UInt8 * skip_data = nullptr; /// null on the fast path
    const UInt64 * flag_base_data = nullptr;
    Arena & pool;
    const void ** found_mapped = nullptr;
    UInt64 * found_offset = nullptr;

    ALWAYS_INLINE const MapNonConst & mapAt(size_t leaf) const { return *static_cast<Map *>(leaf_maps_data[leaf]); }

    ALWAYS_INLINE void record(size_t row, size_t leaf, const Cell * cell, const MapNonConst & map)
    {
        if (!cell)
        {
            found_mapped[row] = nullptr;
            return;
        }
        found_mapped[row] = &cell->getMapped();
        size_t offset = map.offsetInternal(cell);
        if constexpr (need_flags)
            offset += flag_base_data[leaf];
        found_offset[row] = offset;
    }

    ALWAYS_INLINE bool start(Slot & slot, size_t i)
    {
        const size_t ind = selector[i];
        if (skip_data && skip_data[ind])
        {
            found_mapped[i] = nullptr;
            return false;
        }
        auto && key_holder = key_getter.getKeyHolder(ind, pool);
        const auto & key = keyHolderGetKey(key_holder);
        const size_t leaf = leaf_ids ? leaf_ids[ind] : 0;
        const MapNonConst & map = mapAt(leaf);
        if (unlikely(map.isZeroKey(key)))
        {
            /// The zero key lives in the dedicated zero-value cell - nothing to overlap.
            record(i, leaf, map.find(key), map);
            return false;
        }
        const size_t hash = map.hash(key);
        slot.pos = map.cursorPlace(hash);
        slot.row = static_cast<UInt32>(i);
        slot.leaf = static_cast<UInt16>(leaf);
        if constexpr (store_hash)
            slot.hash = hash;
        __builtin_prefetch(map.cursorCell(slot.pos), 0, 1);
        return true;
    }

    ALWAYS_INLINE AmacStepResult step(Slot & slot)
    {
        const MapNonConst & map = mapAt(slot.leaf);
        const Cell * cell = map.cursorCell(slot.pos);
        if (map.cursorCellIsEmpty(cell))
        {
            found_mapped[slot.row] = nullptr;
            return AmacStepResult::Done;
        }
        const size_t ind = selector[slot.row];
        auto && key_holder = key_getter.getKeyHolder(ind, pool);
        const auto & key = keyHolderGetKey(key_holder);
        size_t hash = 0;
        if constexpr (store_hash)
            hash = slot.hash;
        else
            hash = map.hash(key);
        if (map.cursorKeyEquals(cell, key, hash))
        {
            record(slot.row, slot.leaf, cell, map);
            return AmacStepResult::Done;
        }
        slot.pos = map.cursorNext(slot.pos);
        __builtin_prefetch(map.cursorCell(slot.pos), 0, 1);
        return AmacStepResult::Advance;
    }
};

template <typename Grower>
inline constexpr bool is_power_of_two_linear_grower = false;
template <size_t initial_size_degree>
inline constexpr bool is_power_of_two_linear_grower<HashTableGrowerWithPrecalculation<initial_size_degree>> = true;

/** The compile-time gate of the flat-descriptor lookup: an open-addressing map whose find needs
  * nothing from the map object itself - a power-of-two linear-probing grower, so the cell
  * address is `hash & mask` and the walk step `(pos + 1) & mask`, both computable from the
  * 16-byte leaf descriptor, and stateless cells, whose `isZero`/`keyEquals` read only the cell
  * and the key. The fixed-size maps (`key8`/`key16`) have no cursor API; the string, `hashed`
  * and LowCardinality getters are excluded by the caller's cheap-key gate.
  */
template <typename Map>
concept FlatLookupMap = AmacResumableMap<Map> && requires {
    requires is_power_of_two_linear_grower<typename Map::grower_type>;
    requires std::is_same_v<typename Map::cell_type::State, HashTableNoState>;
};

/** The routed probe: the single-map `joinRightColumns` loop with one difference - each row's map
  * is the leaf its recomputed route word points at. Probe blocks are never scattered, buffered or
  * materialized (G2); everything around the loop (`AddedColumns`, `processMatch`, the lazy
  * `HashJoinResult` emit) is the standard `HashJoin` machinery over the shared row store.
  *
  * Above the AMAC engagement threshold the lookups run in two phases per block: phase A is an
  * AMAC find ring completing rows out of order into the reused per-row result scratch; phase B
  * is the same sequential in-order loop with the lookup replaced by the precomputed result, so
  * replication offsets, used-flags semantics and every join kind's logic are untouched. Below
  * the threshold the plain routed loop runs.
  *
  * `MapsShape` is the standard maps shape (`HashJoin::MapsOne`/`MapsAll`/`MapsAsof`) driving
  * `JoinFeatures`/`processMatch` semantics; `Map` is the partitioned leaf map holding identical
  * cells. Right-side used flags are per-offset over ONE flag space spanning all leaves: a found
  * cell's offset is shifted by its leaf's `flag_base` before `processMatch` consumes it, so the
  * standard `JoinUsedFlags` machinery keeps its single-map semantics.
  */
template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsShape, typename KeyGetter, typename Map, typename AddedColumnsType>
size_t PartitionedHashJoin::routedJoinRightColumns(AddedColumnsType & added_columns, const ScatteredBlock & block)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsShape> join_features;
    /// Single disjunct only: the per-row-flags shapes run the delegated standard path.
    constexpr bool flag_per_row = false;

    if (added_columns.additional_filter_expression)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Additional filter expression is not supported for PartitionedHashJoin");

    const auto & join_keys = added_columns.join_on_keys.at(0);
    const auto & selector = block.getSelector();
    const size_t rows = selector.size();
    JoinStuff::JoinUsedFlags & used_flags = *leaf_join->used_flags;
    const UInt64 * flag_base_data [[maybe_unused]] = flag_base.data();

    /// The leaf maps in a flat pointer table, built once after the build phase; the entries are
    /// exactly `Map` (stored by the same `data->type` + maps-variant dispatch this template was
    /// selected by), so the cast is a round trip.
    const void * const * leaf_maps_data = leaf_map_ptrs.data();
    auto map_at = [&](size_t leaf) -> Map & { return *static_cast<Map *>(leaf_maps_data[leaf]); };

    /// Pooled per-probe-stream scratch, reused across blocks; acquired only on paths that need
    /// it (routing at bits > 0, the AMAC result arrays), so the degenerate single-leaf plan pays
    /// no extra allocation at all.
    std::unique_ptr<ProbeScratch> scratch;
    SCOPE_EXIT({
        if (scratch)
            releaseProbeScratch(std::move(scratch));
    });
    auto ensure_scratch = [&]() -> ProbeScratch &
    {
        if (!scratch)
            scratch = acquireProbeScratch();
        return *scratch;
    };

    /// One route word per probe row (over the whole source block: continuation chunks share it).
    /// ASOF would route by the equi-key prefix, but its build plan is always single-leaf.
    const size_t source_rows = block.getSourceBlock().rows();
    const UInt16 * leaf_ids = nullptr;
    if (bits > 0 && source_rows > 0)
    {
        chassert(!join_features.is_asof_join);
        auto & routing = ensure_scratch();
        routing.route_words.resize(source_rows);
        computeJoinRouteWords(join_keys.key_columns, source_rows, routing.route_words.data());
        routing.leaf_ids.resize(source_rows);
        const auto shift = static_cast<UInt32>(32 - bits);
        for (size_t i = 0; i < source_rows; ++i)
            routing.leaf_ids[i] = static_cast<UInt16>(routing.route_words[i] >> shift);
        leaf_ids = routing.leaf_ids.data();
    }

    /// Mirrors `createKeyGetter`: the ASOF getter excludes the trailing inequality column.
    auto key_getter = [&]
    {
        if constexpr (join_features.is_asof_join)
        {
            ColumnRawPtrs equi_columns(join_keys.key_columns.begin(), join_keys.key_columns.end() - 1);
            Sizes equi_sizes(join_keys.key_sizes.begin(), join_keys.key_sizes.end() - 1);
            return KeyGetter(equi_columns, equi_sizes, nullptr);
        }
        else
        {
            return KeyGetter(join_keys.key_columns, join_keys.key_sizes, nullptr);
        }
    }();

    /// The skip byte merges the null map and the ON mask, exactly like the single-map loop;
    /// the fast path compiles the check out.
    const bool fast_path = !join_keys.null_map && join_keys.join_mask_column.getKind() == JoinCommon::JoinMask::Kind::AllTrue;

    if constexpr (!flag_per_row && (STRICTNESS == JoinStrictness::All || (STRICTNESS == JoinStrictness::Semi && KIND == JoinKind::Right)))
        added_columns.lazy_output.output_by_row_list = true;

    if constexpr (join_features.need_replication)
        added_columns.offsets_to_replicate = IColumn::Offsets(rows);

    Arena pool;

    const UInt8 * skip_data = nullptr;
    IColumn::Filter skip_buffer;
    if (!fast_path)
    {
        if (selector.isContinuousRange())
            skip_data = join_keys.buildRowSkipData(skip_buffer, selector.getRange().first, rows);
        else
            skip_data = join_keys.buildRowSkipData(skip_buffer, selector.getIndexes());
    }

    /// AMAC probe engagement, measured on this machine (see the P5 phase report): for the
    /// cheap-key getters the adaptive look-ahead prefetch of the plain loop already extracts the
    /// same memory-level parallelism, and the ring's extra pass costs ~9-10 ns/row at every
    /// thread count - so those types keep the plain loop. The ring engages where the prefetcher
    /// CANNOT run (`join_prefetch_supported` is false - the string-key maps, whose per-visit key
    /// fetch is too expensive for the look-ahead heuristic): there it is the only mechanism that
    /// overlaps the cell misses, measured ~5% (T16) to ~12% (T1) faster end-to-end. The runtime
    /// conditions mirror the software-prefetch heuristics: the user toggle, the aggregate table
    /// size outgrowing L2 (below it the tables are cache resident and the ring is pure
    /// overhead), and a row floor so tiny blocks keep the plain loop (G6).
    using MapNonConst = std::remove_const_t<Map>;
    constexpr bool amac_supported = amac_join_supported<KeyGetter, MapNonConst>;
    constexpr bool prefetch_supported = join_prefetch_supported<KeyGetter, Map>;
    /// The cheap-key open-addressing shapes run the flat-descriptor loop below instead of the
    /// plain routed loop.
    constexpr bool flat_lookup_supported = prefetch_supported && FlatLookupMap<MapNonConst>;
    bool use_amac = false;
    if constexpr (amac_supported)
        use_amac = (!prefetch_supported || amac_probe_forced_for_tests) && amac_enabled && added_columns.enable_prefetch
            && ht_slab_bytes > getMinBytesForPrefetchInJoin() && rows >= amac_min_rows && rows < AmacRingSlot<false>::inactive_row;

    /// Routed look-ahead software prefetch of the plain loop, mutually exclusive with the AMAC
    /// pass; same engagement threshold (the leaf tables are cache-sized by design, so both fire
    /// only when the aggregate table size outgrows it).
    constexpr bool can_prefetch = prefetch_supported;
    bool use_prefetch = false;
    if constexpr (can_prefetch)
        use_prefetch = !use_amac && added_columns.enable_prefetch && ht_slab_bytes > getMinBytesForPrefetchInJoin();

    auto prefetcher = makeJoinPrefetcher(
        use_prefetch,
        rows,
        [&](size_t k) __attribute__((always_inline))
        {
            if constexpr (can_prefetch)
            {
                const size_t ind = selector[k];
                map_at(leaf_ids ? leaf_ids[ind] : 0).prefetch(key_getter.getKeyHolder(ind, pool));
            }
        });

    /// Phase B / the plain loop. With `precomputed` the lookup is replaced by the phase-A result
    /// (skipped rows were recorded as misses there, so the skip check is compiled out); without
    /// it this is the sequential routed lookup. Everything downstream is shared and standard.
    auto loop = [&]<bool need_filter, bool with_skip, bool precomputed>(const ProbeScratch * results)
    {
        if constexpr (need_filter)
        {
            added_columns.filter = IColumn::Filter(rows, 0);
            added_columns.matched_rows.reserve(rows);
        }

        /// The probe-side mapped type (const-qualified: probe maps are immutable).
        using Mapped = std::remove_reference_t<decltype(std::declval<typename KeyGetter::FindResult &>().getMapped())>;

        IColumn::Offset current_offset = 0;
        for (size_t i = 0; i < rows; ++i)
        {
            if constexpr (can_prefetch && !precomputed)
                prefetcher.prefetchAt(i);

            const size_t ind = selector[i];

            bool right_row_found = false;
            KnownRowsHolder<flag_per_row> dummy_known_rows;

            if constexpr (precomputed)
            {
                if (const void * mapped_ptr = results->found_mapped[i])
                {
                    right_row_found = true;
                    typename KeyGetter::FindResult find_result(static_cast<Mapped *>(mapped_ptr), true, results->found_offset[i]);
                    processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsShape, Map, KeyGetter>(
                        find_result, added_columns, used_flags, i, ind, current_offset, dummy_known_rows);
                }
            }
            else
            {
                bool skip_row = false;
                if constexpr (with_skip)
                    skip_row = skip_data && skip_data[ind];

                if (!skip_row)
                {
                    const size_t leaf = leaf_ids ? leaf_ids[ind] : 0;
                    auto find_result = key_getter.findKey(map_at(leaf), ind, pool);
                    if (find_result.isFound())
                    {
                        right_row_found = true;
                        if constexpr (join_features.need_flags)
                        {
                            /// Shift the cell offset into the shared flag space before the standard
                            /// used-flags machinery consumes it.
                            find_result = typename KeyGetter::FindResult(
                                &find_result.getMapped(), true, find_result.getOffset() + flag_base_data[leaf]);
                        }
                        processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsShape, Map, KeyGetter>(
                            find_result, added_columns, used_flags, i, ind, current_offset, dummy_known_rows);
                    }
                }
            }

            if (!right_row_found)
            {
                if constexpr (join_features.is_anti_join && join_features.left)
                    setUsed<need_filter>(added_columns.filter, i, added_columns.matched_rows);
                addNotFoundRow<join_features.add_missing, join_features.need_replication>(added_columns, current_offset);
            }

            if constexpr (join_features.need_replication)
                added_columns.offsets_to_replicate[i] = current_offset;
        }
    };

    /// The flat routed loop for the cheap-key open-addressing maps - the measured hot shape.
    /// Two structural fixes over the plain loop above, per the probe disassembly: the per-row
    /// cell address comes from the contiguous 16-byte-per-leaf descriptor array (one L1 load)
    /// instead of `leaf_map_ptrs[leaf]` and then the map header (three dependent loads on the
    /// address-generation critical path, mirroring `parallel_hash`'s contiguous bucket-header
    /// array); and every loop invariant - the selector base, the leaf ids, the descriptors, the
    /// key getter - is snapshotted into locals of the loop body, because the closure fields sit
    /// behind a pointer the compiler must conservatively re-load after every opaque call
    /// (`appendFromBlock`), which showed up as ~10 per-row loads. The selector variant is a
    /// template parameter for the same reason - its per-row kind check was a per-row load. The
    /// lookup itself is exactly `HashMapTable::find` with identical offset semantics; keys equal
    /// to the zero sentinel resolve through the map object (rare). Everything downstream of the
    /// `FindResult` matches the plain loop.
    auto flat_loop = [&]<bool need_filter, bool with_skip, bool selector_is_range>()
    {
        /// The call sites are gated on the same constant, but the guard must also be here:
        /// instantiating the enclosing function substitutes the enclosing template arguments
        /// into this body even when the lambda is never called, and the lookup below is only
        /// well-formed for the gated map types.
        if constexpr (flat_lookup_supported)
        {
            using Cell = typename MapNonConst::cell_type;

            /// Unlike the plain loop, the invariant snapshots below touch the leaf tables before
            /// the first row; an empty probe block may legally arrive with no leaf maps at all.
            if (rows == 0)
                return;

            if constexpr (need_filter)
            {
                added_columns.filter = IColumn::Filter(rows, 0);
                added_columns.matched_rows.reserve(rows);
            }

            [[maybe_unused]] size_t selector_base = 0;
            [[maybe_unused]] const UInt64 * selector_indexes = nullptr;
            if constexpr (selector_is_range)
                selector_base = selector.getRange().first;
            else
                selector_indexes = selector.getIndexes().getData().data();
            auto index_at = [&](size_t k) __attribute__((always_inline))
            {
                if constexpr (selector_is_range)
                    return selector_base + k;
                else
                    return static_cast<size_t>(selector_indexes[k]);
            };

            const UInt16 * const leaf_ids_local = leaf_ids;
            [[maybe_unused]] const UInt8 * const skip_local = skip_data;
            [[maybe_unused]] const UInt64 * const flag_base_local = flag_base_data;
            const LeafMapDesc * const descs = leaf_map_descs.data();
            /// The gate guarantees the cells' zero-check and key-compare read no map state.
            const HashTableNoState no_state{};
            /// Hash provider; reads nothing through the object (the hash functor is an empty base).
            const MapNonConst & map0 = map_at(0);
            /// A private copy of a cheap key getter keeps its column pointer in a register.
            std::conditional_t<std::is_trivially_copyable_v<KeyGetter>, KeyGetter, KeyGetter &> keys = key_getter;

            auto flat_prefetcher = makeJoinPrefetcher(
                use_prefetch,
                rows,
                [&](size_t k) __attribute__((always_inline))
                {
                    const size_t ind = index_at(k);
                    const auto & desc = descs[leaf_ids_local ? leaf_ids_local[ind] : 0];
                    auto && key_holder = keys.getKeyHolder(ind, pool);
                    const size_t hash = map0.hash(keyHolderGetKey(key_holder));
                    __builtin_prefetch(static_cast<const Cell *>(desc.buf) + (hash & desc.mask));
                });

            IColumn::Offset current_offset = 0;
            for (size_t i = 0; i < rows; ++i)
            {
                flat_prefetcher.prefetchAt(i);

                const size_t ind = index_at(i);

                bool right_row_found = false;
                KnownRowsHolder<flag_per_row> dummy_known_rows;

                bool skip_row = false;
                if constexpr (with_skip)
                    skip_row = skip_local && skip_local[ind];

                if (!skip_row)
                {
                    const size_t leaf = leaf_ids_local ? leaf_ids_local[ind] : 0;
                    auto && key_holder = keys.getKeyHolder(ind, pool);
                    const auto & key = keyHolderGetKey(key_holder);
                    const Cell * cell = nullptr;
                    size_t offset = 0;
                    if (unlikely(Cell::isZero(key, no_state)))
                    {
                        /// The zero key lives in the map's dedicated zero-value cell, whose
                        /// `offsetInternal` is 0.
                        cell = map_at(leaf).find(key);
                    }
                    else
                    {
                        const auto & desc = descs[leaf];
                        const size_t hash = map0.hash(key);
                        const Cell * buf = static_cast<const Cell *>(desc.buf);
                        size_t pos = hash & desc.mask;
                        while (!buf[pos].isZero(no_state) && !buf[pos].keyEquals(key, hash, no_state))
                            pos = (pos + 1) & desc.mask;
                        if (!buf[pos].isZero(no_state))
                        {
                            cell = buf + pos;
                            offset = pos + 1;
                        }
                    }
                    if (cell)
                    {
                        right_row_found = true;
                        if constexpr (join_features.need_flags)
                            offset += flag_base_local[leaf];
                        typename KeyGetter::FindResult find_result(&cell->getMapped(), true, offset);
                        processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsShape, Map, KeyGetter>(
                            find_result, added_columns, used_flags, i, ind, current_offset, dummy_known_rows);
                    }
                }

                if (!right_row_found)
                {
                    if constexpr (join_features.is_anti_join && join_features.left)
                        setUsed<need_filter>(added_columns.filter, i, added_columns.matched_rows);
                    addNotFoundRow<join_features.add_missing, join_features.need_replication>(added_columns, current_offset);
                }

                if constexpr (join_features.need_replication)
                    added_columns.offsets_to_replicate[i] = current_offset;
            }
        }
    };

    bool amac_ran = false;
    if constexpr (amac_supported)
    {
        if (use_amac)
        {
            /// Phase A: the AMAC find pass. Every row gets a result: `start` records skipped and
            /// zero-key rows synchronously, `step` records hits and misses - so the arrays need
            /// no pre-fill and phase B needs no skip logic.
            auto & results = ensure_scratch();
            results.found_mapped.resize(rows);
            results.found_offset.resize(rows);
            RoutedAmacFindPolicy<KeyGetter, Map, join_features.need_flags> policy{
                .key_getter = key_getter,
                .leaf_maps_data = leaf_maps_data,
                .leaf_ids = leaf_ids,
                .selector = selector,
                .skip_data = skip_data,
                .flag_base_data = flag_base_data,
                .pool = pool,
                .found_mapped = results.found_mapped.data(),
                .found_offset = results.found_offset.data()};
            amacRun(policy, rows);

            if (added_columns.need_filter)
                loop.template operator()<true, false, true>(&results);
            else
                loop.template operator()<false, false, true>(&results);
            amac_ran = true;
        }
    }

    if (!amac_ran)
    {
        if constexpr (flat_lookup_supported)
        {
            auto flat_dispatch = [&]<bool need_filter, bool with_skip>()
            {
                if (selector.isContinuousRange())
                    flat_loop.template operator()<need_filter, with_skip, true>();
                else
                    flat_loop.template operator()<need_filter, with_skip, false>();
            };
            if (added_columns.need_filter)
            {
                if (fast_path)
                    flat_dispatch.template operator()<true, false>();
                else
                    flat_dispatch.template operator()<true, true>();
            }
            else
            {
                if (fast_path)
                    flat_dispatch.template operator()<false, false>();
                else
                    flat_dispatch.template operator()<false, true>();
            }
        }
        else
        {
            if (added_columns.need_filter)
            {
                if (fast_path)
                    loop.template operator()<true, false, false>(nullptr);
                else
                    loop.template operator()<true, true, false>(nullptr);
            }
            else
            {
                if (fast_path)
                    loop.template operator()<false, false, false>(nullptr);
                else
                    loop.template operator()<false, true, false>(nullptr);
            }
        }
    }

    added_columns.applyLazyDefaults();
    return 0;
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsShape>
JoinResultPtr PartitionedHashJoin::probeImpl(Block block)
{
    HashJoin & join = *leaf_join;

    for (const auto & onexpr : table_join->getClauses())
    {
        auto cond_column_name = onexpr.condColumnNames();
        JoinCommon::checkTypesOfKeys(
            block, onexpr.key_names_left, cond_column_name.first, join.right_sample_block, onexpr.key_names_right, cond_column_name.second);
    }

    join.materializeColumnsFromLeftBlock(block);
    ScatteredBlock scattered_block{std::move(block)};

    if (leaf_maps.empty() && scattered_block.rows() > 0)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "PartitionedHashJoin: probe started before the build phase finished");

    constexpr JoinFeatures<KIND, STRICTNESS, MapsShape> join_features;

    const auto & clause = table_join->getOnlyClause();
    std::vector<JoinOnKeyColumns> join_on_keys;
    join_on_keys.emplace_back(
        scattered_block,
        clause.key_names_left,
        clause.condColumnNames().first,
        join.key_sizes[0],
        HashJoin::isLowCardinalityType(join.data->type));

    AddedColumns<!join_features.is_any_join> added_columns(
        scattered_block,
        join.sample_block_with_columns_to_add,
        join.savedBlockSample(),
        join,
        std::move(join_on_keys),
        join.table_join->getMixedJoinExpression(),
        join.additional_filter_required_rhs_pos,
        join_features.is_asof_join,
        /*is_join_get=*/false);

    /// Emit the fixed-width right columns through the direct typed gather (see
    /// `LazyOutput::buildOutputFromBlocks`) instead of the generic pair-expansion path. Gated on
    /// the lazy shapes whose emit consumes ref words; ASOF keeps the generic path (its
    /// `AddedColumns` does not resolve the emit table). Partitioned-only by construction:
    /// `hash`/`parallel_hash` never set the flag.
    if constexpr (!join_features.is_any_join && !join_features.is_asof_join)
    {
        added_columns.lazy_output.use_direct_typed_gather = true;
        added_columns.lazy_output.stored_blocks_count = join.getJoinedData()->stored_columns_index->blocksCount();
    }

    const bool has_required_right_keys = join.required_right_keys.columns() != 0;
    added_columns.need_filter = join_features.need_filter || has_required_right_keys;
    added_columns.max_joined_block_rows = join.max_joined_block_rows;
    if (!added_columns.max_joined_block_rows)
        added_columns.max_joined_block_rows = std::numeric_limits<size_t>::max();
    else
        added_columns.reserve(join_features.need_replication);

    using OurMaps = PartitionedMapsFor<MapsShape>::Type;

    switch (join.data->type)
    {
#define M(TYPE) \
    case HashJoin::Type::TYPE: { \
        using Map = const decltype(OurMaps::TYPE)::element_type; \
        using KeyGetter = typename KeyGetterForType<HashJoin::Type::TYPE, Map>::Type; \
        routedJoinRightColumns<KIND, STRICTNESS, MapsShape, KeyGetter, Map>(added_columns, scattered_block); \
        break; \
    }
        APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M)
#undef M
        default:
            throw Exception(
                ErrorCodes::UNSUPPORTED_JOIN_KEYS, "Unsupported JOIN keys for the partitioned join (type: {})", join.data->type);
    }

    added_columns.join_on_keys.clear();

    return std::make_unique<HashJoinResult>(
        std::move(added_columns.lazy_output),
        std::move(added_columns.columns),
        std::move(added_columns.offsets_to_replicate),
        std::move(added_columns.filter),
        std::move(added_columns.matched_rows),
        std::move(scattered_block),
        HashJoinResult::Properties{
            *join.table_join,
            join.required_right_keys,
            join.required_right_keys_sources,
            join.max_joined_block_rows,
            join.max_joined_block_bytes,
            join.data->allocated_size / std::max<size_t>(1, join.data->rows_to_join),
            join_features.need_filter,
            /*is_join_get=*/false,
            join.joined_block_split_single_row,
            join.enable_lazy_columns_replication,
            join.enable_lazy_columns_indexing});
}

}
