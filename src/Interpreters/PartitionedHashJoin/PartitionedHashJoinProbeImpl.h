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
#include <Interpreters/RowRefs.h>
#include <Interpreters/TableJoin.h>
#include <base/scope_guard.h>
#include <Common/ElapsedTimeProfileEventIncrement.h>
#include <Common/HashTable/HashTable.h>
#include <Common/PODArray.h>
#include <Common/ProfileEvents.h>

namespace ProfileEvents
{
extern const Event PartitionedHashJoinProbeLookupMicroseconds;
}

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int UNSUPPORTED_JOIN_KEYS;
}

/// Mapped values the AMAC find pass records BY VALUE: both are 8-byte words (`RowRef` encodes to
/// its ref word, `RowRefList` IS a tagged word) that are never 0 for a built cell - a `RowRef` is
/// always constructed with INLINE_FLAG in bit 63, and a `RowRefList` word is an inline ref (bit
/// 63 set) or a non-null node pointer - so 0 can encode a miss. Everything phase B does with a
/// match (`appendFromBlock`, `rows`, `refsOf`, `firstRefWord`) consumes only the word, and the
/// probe maps are immutable, so the copy is semantically the cell itself.
template <typename Mapped>
inline constexpr bool amac_mapped_fits_word = std::is_same_v<Mapped, RowRef> || std::is_same_v<Mapped, RowRefList>;

template <typename Mapped>
requires amac_mapped_fits_word<Mapped>
ALWAYS_INLINE UInt64 mappedWordOf(const Mapped & mapped)
{
    if constexpr (std::is_same_v<Mapped, RowRefList>)
        return mapped.word;
    else
        return mapped.encode();
}

template <typename Mapped>
requires amac_mapped_fits_word<Mapped>
ALWAYS_INLINE Mapped mappedFromWord(UInt64 word)
{
    if constexpr (std::is_same_v<Mapped, RowRefList>)
        return RowRefList::fromWord(word);
    else
        return RowRef::fromWord(word);
}

template <typename Grower>
inline constexpr bool is_power_of_two_linear_grower = false;
template <size_t initial_size_degree>
inline constexpr bool is_power_of_two_linear_grower<HashTableGrowerWithPrecalculation<initial_size_degree>> = true;

/** The AMAC find policy of the two-phase probe (phase A): out-of-order lookups that only fill
  * the per-row result arrays - the matched cell's mapped value copied by value into `found_word`
  * (0 = no match) and, for the flagged shapes only, its used-flags offset shifted into the
  * shared flag space. Copying the word in the same visit that reads the cell means phase B never
  * touches the cell again - by the time the sequential loop reaches the row, a block later, the
  * cell line has usually left the cache and re-reading it through a recorded pointer was a
  * second random miss per row. Mapped types that do not fit a word (ASOF) keep the pointer
  * scheme, storing its bits in the same array. Nothing is emitted here; phase B consumes the
  * results in row order - the flagless word-mapped lazy shapes through the degenerate cursor
  * pass (`word_loop`), the rest through the standard `processMatch`.
  * One ring serves MANY maps - each row's leaf. The leaf's address material - the cell buffer
  * and the grower mask - is resolved once at admit from the flat descriptor array and carried in
  * the slot, so a steady visit dereferences nothing but the cell itself and the probe key: the
  * map headers, scattered across as many heap objects as there are partitions, would otherwise
  * sit on the address chain of every visit (2-3 dependent loads; the flat loop got the same fix
  * in descriptor form). The selector variant is a template parameter for the same reason - it
  * was a per-visit kind branch. Cell prefetches use read intent and low locality: a probed cell
  * is not revisited.
  */
template <typename KeyGetter, typename Map, bool need_flags, bool selector_is_range>
struct RoutedAmacFindPolicy
{
    using MapNonConst = std::remove_const_t<Map>;
    using Cell = MapNonConst::cell_type;
    static constexpr bool store_hash = cell_stores_hash<Cell>;
    static constexpr bool may_grow = false;
    static constexpr bool copy_into_frame = true; /// results live in the arrays; no state survives the run
    static constexpr bool mapped_by_value = amac_mapped_fits_word<typename MapNonConst::mapped_type>;

    /// The slot-register walk below is `HashMapTable::find` only under the contract of the flat
    /// lookup (`FlatLookupMap`): a power-of-two linear-probing grower - the home cell is
    /// `hash & mask`, the walk step `(pos + 1) & mask` - and stateless cells, whose zero-check
    /// and key-compare read nothing through the map object. Every map the AMAC gate admits
    /// satisfies it; one that does not must keep the map-resolved cursor instead.
    static_assert(is_power_of_two_linear_grower<typename MapNonConst::grower_type>);
    static_assert(std::is_same_v<typename Cell::State, HashTableNoState>);
    static constexpr HashTableNoState no_state{};

    /// The leaf id stays for the record path (the used-flags base of the leaf).
    struct Slot : public AmacRingSlot<store_hash>
    {
        UInt16 leaf = 0;
        const Cell * buf = nullptr;
        size_t mask = 0;
    };
    static_assert(sizeof(Slot) == (store_hash ? 48 : 32));

    /// A by-value copy of a trivially copyable key getter keeps its key-column pointer a plain
    /// field of the (frame-local) policy instead of two dependent loads behind a reference.
    std::conditional_t<std::is_trivially_copyable_v<KeyGetter>, KeyGetter, KeyGetter &> key_getter;
    /// Hash provider and zero-key checker; reads nothing through the object (the hash functor is
    /// an empty base and the cells are stateless), so any leaf serves.
    const MapNonConst & map0;
    const void * const * leaf_maps_data = nullptr; /// the zero-key sentinel path only
    const LeafMapDesc * leaf_descs = nullptr;
    const UInt16 * leaf_ids = nullptr; /// null at the single-leaf plan
    size_t selector_base = 0; /// the first row of a continuous-range selector
    const UInt64 * selector_indexes = nullptr; /// the data of an explicit-indexes selector
    const UInt8 * skip_data = nullptr; /// null on the fast path
    const UInt64 * flag_base_data = nullptr;
    Arena & pool;
    UInt64 * found_word = nullptr;
    UInt64 * found_offset = nullptr; /// null unless `need_flags`

    ALWAYS_INLINE size_t indexAt(size_t i) const
    {
        if constexpr (selector_is_range)
            return selector_base + i;
        else
            return selector_indexes[i];
    }

    ALWAYS_INLINE const MapNonConst & mapAt(size_t leaf) const { return *static_cast<Map *>(leaf_maps_data[leaf]); }

    /// The synchronous zero-key path of `start`: the cell (the map's dedicated zero-value cell,
    /// or null) came from the map object, and so must its used-flags offset.
    ALWAYS_INLINE void record(size_t row, size_t leaf [[maybe_unused]], const Cell * cell, const MapNonConst & map [[maybe_unused]])
    {
        if (!cell)
        {
            found_word[row] = 0;
            return;
        }
        if constexpr (mapped_by_value)
            found_word[row] = mappedWordOf(cell->getMapped());
        else
            found_word[row] = reinterpret_cast<UInt64>(&cell->getMapped());
        if constexpr (need_flags)
            found_offset[row] = map.offsetInternal(cell) + flag_base_data[leaf];
    }

    /// A ring hit: the cell is `slot.buf + slot.pos` and known non-zero, so its used-flags
    /// offset is `slot.pos + 1` - `offsetInternal` without touching the map.
    ALWAYS_INLINE void recordHit(const Slot & slot, const Cell * cell)
    {
        if constexpr (mapped_by_value)
            found_word[slot.row] = mappedWordOf(cell->getMapped());
        else
            found_word[slot.row] = reinterpret_cast<UInt64>(&cell->getMapped());
        if constexpr (need_flags)
            found_offset[slot.row] = slot.pos + 1 + flag_base_data[slot.leaf];
    }

    ALWAYS_INLINE bool start(Slot & slot, size_t i)
    {
        const size_t ind = indexAt(i);
        if (skip_data && skip_data[ind])
        {
            found_word[i] = 0;
            return false;
        }
        auto && key_holder = key_getter.getKeyHolder(ind, pool);
        const auto & key = keyHolderGetKey(key_holder);
        const size_t leaf = leaf_ids ? leaf_ids[ind] : 0;
        if (unlikely(map0.isZeroKey(key)))
        {
            /// The zero key lives in the dedicated zero-value cell - nothing to overlap.
            const MapNonConst & map = mapAt(leaf);
            record(i, leaf, map.find(key), map);
            return false;
        }
        const size_t hash = map0.hash(key);
        const LeafMapDesc & desc = leaf_descs[leaf];
        slot.pos = hash & desc.mask;
        slot.row = static_cast<UInt32>(i);
        if constexpr (store_hash)
            slot.hash = hash;
        slot.leaf = static_cast<UInt16>(leaf);
        slot.buf = static_cast<const Cell *>(desc.buf);
        slot.mask = desc.mask;
        __builtin_prefetch(slot.buf + slot.pos, 0, 1);
        return true;
    }

    ALWAYS_INLINE AmacStepResult step(Slot & slot)
    {
        const Cell * cell = slot.buf + slot.pos;
        if (cell->isZero(no_state))
        {
            found_word[slot.row] = 0;
            return AmacStepResult::Done;
        }
        const size_t ind = indexAt(slot.row);
        auto && key_holder = key_getter.getKeyHolder(ind, pool);
        const auto & key = keyHolderGetKey(key_holder);
        size_t hash = 0;
        if constexpr (store_hash)
            hash = slot.hash;
        else
            hash = map0.hash(key);
        if (cell->keyEquals(key, hash, no_state))
        {
            recordHit(slot, cell);
            return AmacStepResult::Done;
        }
        slot.pos = (slot.pos + 1) & slot.mask;
        __builtin_prefetch(slot.buf + slot.pos, 0, 1);
        return AmacStepResult::Advance;
    }
};

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
  * consumes the results in row order. On the flagless word-mapped lazy shapes it degenerates to
  * a dispatch-free cursor pass over the recorded words (`word_loop`); the other shapes run the
  * same sequential in-order loop with the lookup replaced by the precomputed result - either
  * way replication offsets, used-flags semantics and every join kind's logic are untouched.
  * Below the threshold the plain routed loop runs.
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
        routing.leaf_ids.resize(source_rows);
        computeJoinLeafIds(join_keys.key_columns, source_rows, bits, routing.leaf_ids.data());
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

    /// AMAC probe engagement, measured on this machine (see the P5 phase report): above the
    /// threshold the ring is the single engaged probe path for every AMAC-capable shape. On the
    /// string-key maps - where the look-ahead prefetcher cannot run at all (`getKeyHolder` per
    /// look-ahead is too expensive for its heuristic) - it is the only mechanism that overlaps
    /// the cell misses, measured ~20% (T16 and T1) faster end-to-end than the plain loop. On the
    /// cheap-key getters it beats the adaptive look-ahead prefetch of the flat loop too (~11%
    /// probe thread time at T16, parity-to-win at T1) since the slot-carried address chain and
    /// the degenerate emit pass removed the ring's former per-visit metadata and two-phase
    /// costs. The runtime conditions mirror the software-prefetch heuristics: the user toggle,
    /// the aggregate table size outgrowing L2 (below it the tables are cache resident and the
    /// ring is pure overhead), and a row floor so tiny blocks keep the plain loop (G6); the
    /// loops below serve those regimes and the AMAC-incapable getters.
    using MapNonConst = std::remove_const_t<Map>;
    constexpr bool amac_supported = amac_join_supported<KeyGetter, MapNonConst>;
    constexpr bool prefetch_supported = join_prefetch_supported<KeyGetter, Map>;
    /// The cheap-key open-addressing shapes run the flat-descriptor loop below instead of the
    /// plain routed loop.
    constexpr bool flat_lookup_supported = prefetch_supported && FlatLookupMap<MapNonConst>;
    bool use_amac = false;
    if constexpr (amac_supported)
        use_amac = amac_enabled && added_columns.enable_prefetch && ht_slab_bytes > getMinBytesForPrefetchInJoin() && rows >= amac_min_rows
            && rows < AmacRingSlot<false>::inactive_row;

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
                if (const UInt64 word = results->found_word[i])
                {
                    right_row_found = true;
                    size_t offset = 0;
                    if constexpr (join_features.need_flags)
                        offset = results->found_offset[i];
                    /// Phase A decided by-value recording from the map's mapped type; this side
                    /// decides from the FindResult's - they must be the same type, or a word
                    /// would be reinterpreted as a pointer.
                    static_assert(std::is_same_v<std::remove_const_t<Mapped>, typename std::remove_const_t<Map>::mapped_type>);
                    if constexpr (amac_mapped_fits_word<std::remove_const_t<Mapped>>)
                    {
                        /// The mapped value phase A copied out of the cell, rebuilt on the stack:
                        /// the cell itself is never dereferenced again.
                        auto mapped_value = mappedFromWord<std::remove_const_t<Mapped>>(word);
                        typename KeyGetter::FindResult find_result(&mapped_value, true, offset);
                        processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsShape, Map, KeyGetter>(
                            find_result, added_columns, used_flags, i, ind, current_offset, dummy_known_rows);
                    }
                    else
                    {
                        typename KeyGetter::FindResult find_result(
                            reinterpret_cast<Mapped *>(word), true, offset); /// NOLINT(performance-no-int-to-ptr)
                        processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsShape, Map, KeyGetter>(
                            find_result, added_columns, used_flags, i, ind, current_offset, dummy_known_rows);
                    }
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

    /// Whether phase B can degenerate to the dispatch-free `word_loop` below: the recorded word
    /// must be the mapped value itself, the emit must be the lazy ref-word append, and the shape
    /// must consume no per-row state beyond the filter, the appended words and the replication
    /// offsets. The flagged shapes (RIGHT/FULL used flags, incl. `setUsedOnce` - every shape
    /// with first-match-only semantics is flagged) and ASOF keep the full loop above.
    constexpr bool degenerate_phase_b = AddedColumnsType::isLazy() && amac_mapped_fits_word<typename MapNonConst::mapped_type>
        && !join_features.need_flags && !join_features.is_asof_join && !join_features.is_any_join;

    /// The degenerate phase B of the two-phase AMAC probe. On the shapes gated above,
    /// `processMatch` reduces to: mark the row matched (filter + `matched_rows`), append the
    /// recorded word (ALL: the list word, advancing the replication offset by its row count;
    /// RightAny/Semi: its first ref) or the default on an added miss - so this pass consumes
    /// `found_word` directly instead of rebuilding a `FindResult` and dispatching `processMatch`
    /// per row, whose outlined `appendFromBlock` call forced the loop-carried state to spill.
    /// Every row appends at most one entry, so the cursors write into pre-sized arrays with no
    /// per-append capacity check. Row order, filter, offsets and `row_count` are exactly the
    /// full loop's; the 3-way parity suite pins that.
    auto word_loop = [&]<bool need_filter, bool with_refs>(const ProbeScratch & results [[maybe_unused]])
    {
        if constexpr (degenerate_phase_b)
        {
            using Mapped = MapNonConst::mapped_type;

            if constexpr (need_filter)
            {
                added_columns.filter = IColumn::Filter(rows, 0);
                added_columns.matched_rows.resize(rows);
            }

            const UInt64 * const words = results.found_word.data();
            [[maybe_unused]] UInt8 * filter_data = nullptr;
            [[maybe_unused]] IColumn::Offset * matched_cur = nullptr;
            if constexpr (need_filter)
            {
                filter_data = added_columns.filter.data();
                matched_cur = added_columns.matched_rows.data();
            }
            [[maybe_unused]] UInt64 * ref_cur = nullptr;
            if constexpr (with_refs)
            {
                auto & row_refs = added_columns.lazy_output.row_refs;
                const size_t refs_begin = row_refs.size();
                row_refs.resize(refs_begin + rows);
                ref_cur = row_refs.data() + refs_begin;
            }
            [[maybe_unused]] IColumn::Offset * offsets = nullptr;
            if constexpr (join_features.need_replication)
                offsets = added_columns.offsets_to_replicate.data();

            [[maybe_unused]] IColumn::Offset current_offset = 0;
            [[maybe_unused]] UInt64 appended_row_count = 0;
            /// A local copy: the loop bound would otherwise reload through the closure per
            /// iteration - the filter's byte stores may alias anything the closure points at.
            const size_t rows_local = rows;
            for (size_t i = 0; i < rows_local; ++i)
            {
                const UInt64 word = words[i];
                if (word)
                {
                    /// A flagless anti match only leaves its row unmatched in the filter.
                    if constexpr (!join_features.is_anti_join)
                    {
                        if constexpr (need_filter)
                        {
                            filter_data[i] = 1;
                            *matched_cur++ = i;
                        }
                        if constexpr (join_features.is_all_join)
                        {
                            const UInt32 match_rows = refWordRows(word);
                            current_offset += match_rows;
                            if constexpr (with_refs)
                            {
                                *ref_cur++ = word;
                                appended_row_count += match_rows;
                            }
                        }
                        else if constexpr (with_refs)
                        {
                            *ref_cur++ = firstRefWord(mappedFromWord<Mapped>(word));
                            ++appended_row_count;
                        }
                    }
                }
                else
                {
                    if constexpr (join_features.is_anti_join && join_features.left && need_filter)
                    {
                        filter_data[i] = 1;
                        *matched_cur++ = i;
                    }
                    if constexpr (join_features.add_missing)
                    {
                        if constexpr (with_refs)
                        {
                            *ref_cur++ = 0;
                            ++appended_row_count;
                        }
                        if constexpr (join_features.need_replication)
                            ++current_offset;
                    }
                }
                if constexpr (join_features.need_replication)
                    offsets[i] = current_offset;
            }

            if constexpr (need_filter)
                added_columns.matched_rows.resize(matched_cur - added_columns.matched_rows.data());
            if constexpr (with_refs)
            {
                auto & row_refs = added_columns.lazy_output.row_refs;
                row_refs.resize(ref_cur - row_refs.data());
                added_columns.lazy_output.row_count += appended_row_count;
            }
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
            /// no pre-fill and phase B needs no skip logic. The offsets are only recorded (and
            /// only sized) for the flagged shapes - they have no other consumer.
            auto & results = ensure_scratch();
            results.found_word.resize(rows);
            UInt64 * found_offset_data = nullptr;
            if constexpr (join_features.need_flags)
            {
                results.found_offset.resize(rows);
                found_offset_data = results.found_offset.data();
            }
            auto amac_find = [&]<bool selector_is_range>()
            {
                size_t selector_base = 0;
                const UInt64 * selector_indexes = nullptr;
                if constexpr (selector_is_range)
                    selector_base = selector.getRange().first;
                else
                    selector_indexes = selector.getIndexes().getData().data();
                RoutedAmacFindPolicy<KeyGetter, Map, join_features.need_flags, selector_is_range> policy{
                    .key_getter = key_getter,
                    .map0 = map_at(0),
                    .leaf_maps_data = leaf_maps_data,
                    .leaf_descs = leaf_map_descs.data(),
                    .leaf_ids = leaf_ids,
                    .selector_base = selector_base,
                    .selector_indexes = selector_indexes,
                    .skip_data = skip_data,
                    .flag_base_data = flag_base_data,
                    .pool = pool,
                    .found_word = results.found_word.data(),
                    .found_offset = found_offset_data};
                amacRun(policy, rows);
            };
            if (selector.isContinuousRange())
                amac_find.template operator()<true>();
            else
                amac_find.template operator()<false>();

            if constexpr (degenerate_phase_b)
            {
                auto word_dispatch = [&]<bool need_filter>()
                {
                    if (added_columns.has_columns_to_add)
                        word_loop.template operator()<need_filter, true>(results);
                    else
                        word_loop.template operator()<need_filter, false>(results);
                };
                if (added_columns.need_filter)
                    word_dispatch.template operator()<true>();
                else
                    word_dispatch.template operator()<false>();
            }
            else
            {
                if (added_columns.need_filter)
                    loop.template operator()<true, false, true>(&results);
                else
                    loop.template operator()<false, false, true>(&results);
            }
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
        added_columns.lazy_output.use_direct_typed_gather = true;

    const bool has_required_right_keys = join.required_right_keys.columns() != 0;
    added_columns.need_filter = join_features.need_filter || has_required_right_keys;
    added_columns.max_joined_block_rows = join.max_joined_block_rows;
    if (!added_columns.max_joined_block_rows)
        added_columns.max_joined_block_rows = std::numeric_limits<size_t>::max();
    else
        added_columns.reserve(join_features.need_replication);

    using OurMaps = PartitionedMapsFor<MapsShape>::Type;

    {
        /// The routed hash-table lookup (leaf routing plus the AMAC find ring or plain/flat
        /// lookup loop and match bookkeeping - filter, matched rows, replication offsets). Does
        /// NOT gather any column values yet; that is deferred to the lazy `HashJoinResult::next`
        /// (`HashJoinResultFilterLeftMicroseconds`/`HashJoinResultBuildOutputMicroseconds`, shared
        /// with `hash`/`parallel_hash`/`grace_hash`).
        ProfileEventTimeIncrement<Microseconds> lookup_watch(ProfileEvents::PartitionedHashJoinProbeLookupMicroseconds);
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
