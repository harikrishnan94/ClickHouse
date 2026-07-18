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

    /// AMAC engagement mirrors the software-prefetch heuristics (the user toggle + the aggregate
    /// table size outgrowing L2 - below it the tables are cache resident and the ring is pure
    /// overhead), with a row floor so tiny blocks keep the plain loop (G6).
    constexpr bool amac_supported = amac_join_supported<KeyGetter, std::remove_const_t<Map>>;
    bool use_amac = false;
    if constexpr (amac_supported)
        use_amac = amac_enabled && added_columns.enable_prefetch && ht_slab_bytes > getMinBytesForPrefetchInJoin() && rows >= amac_min_rows
            && rows < AmacRingSlot<false>::inactive_row;

    /// Routed look-ahead software prefetch of the plain loop, mutually exclusive with the AMAC
    /// pass; same engagement threshold (the leaf tables are cache-sized by design, so both fire
    /// only when the aggregate table size outgrows it).
    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;
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
