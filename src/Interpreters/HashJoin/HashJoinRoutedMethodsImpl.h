#pragma once

#include <Interpreters/HashJoin/AmacMode.h>
#include <Interpreters/HashJoin/AmacProbe.h>
#include <Interpreters/HashJoin/HashJoinMethodsImpl.h>
#include <Interpreters/HashJoin/HashJoinRoutedMethods.h>
#include <Interpreters/HashJoin/JoinProbeScratch.h>

#include <bit>

namespace DB
{

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
JoinResultPtr RoutedHashJoinMethods<KIND, STRICTNESS, MapsTemplate>::joinBlockImpl(
    const std::vector<const HashJoin *> & slot_joins,
    const RoutedProbePlan & plan,
    ScatteredBlock block,
    const Block & block_with_columns_to_add,
    JoinProbeScratch & scratch,
    std::vector<JoinOnKeyColumns> join_on_keys)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;

    chassert(!slot_joins.empty());
    const HashJoin & join = *slot_joins[0];

    chassert(join_on_keys.size() == 1, "parallel_hash supports a single join clause");

    /// The eager slot ids ride in the scratch; the per-row loops below take a raw pointer,
    /// null when there is a single slot AND for the hash-routed families, which derive the
    /// slot inline from the lookup's own hash (`joinBlock` fills the array for them only when
    /// the mixed ON-expression path below consumes it).
    const UInt8 * slot_ids = scratch.slot_ids.empty() ? nullptr : scratch.slot_ids.data();

    /// Slot 0 is the representative for everything the emit machinery reads through the join:
    /// the saved-block sample, the limits, and - crucially - the `StoredColumnsIndex`, which is
    /// SHARED across the slots, so the one `AddedColumns` below resolves every slot's stored
    /// blocks at emit time.
    AddedColumns<!join_features.is_any_join> added_columns(
        block,
        block_with_columns_to_add,
        join.savedBlockSample(),
        join,
        std::move(join_on_keys),
        join.table_join->getMixedJoinExpression(),
        join.additional_filter_required_rhs_pos,
        join_features.is_asof_join,
        /*is_join_get=*/false);

    bool has_required_right_keys = (join.required_right_keys.columns() != 0);
    added_columns.need_filter = join_features.need_filter || has_required_right_keys;
    added_columns.max_joined_block_rows = join.max_joined_block_rows;
    if (!added_columns.max_joined_block_rows)
        added_columns.max_joined_block_rows = std::numeric_limits<size_t>::max();
    else
        added_columns.reserve(join_features.need_replication);

    size_t processed_rows = switchJoinRightColumns(slot_joins, plan, added_columns, block.getSelector(), slot_ids, scratch);
    /// Do not hold memory for join_on_keys anymore
    added_columns.join_on_keys.clear();

    std::optional<ScatteredBlock> next_scattered_block;
    if (0 < processed_rows && processed_rows < block.rows())
    {
        auto [raw_block, raw_selector] = std::move(block).detachData();
        auto split_selector = raw_selector.split(processed_rows);
        block = ScatteredBlock(raw_block, std::move(split_selector.first));
        next_scattered_block = ScatteredBlock(std::move(raw_block), std::move(split_selector.second));
    }

    auto join_result = std::make_unique<HashJoinResult>(
        std::move(added_columns.lazy_output),
        std::move(added_columns.columns),
        std::move(added_columns.offsets_to_replicate),
        std::move(added_columns.filter),
        std::move(added_columns.matched_rows),
        std::move(block),
        HashJoinResult::Properties{
            *join.table_join,
            join.required_right_keys,
            join.required_right_keys_sources,
            join.max_joined_block_rows,
            join.max_joined_block_bytes,
            /// The whole-join output-splitting estimate (never slot 0's alone: with few
            /// distinct keys most slots are empty and a zero estimate would disable
            /// `max_joined_block_bytes` splitting entirely).
            plan.avg_joined_bytes_per_row,
            join_features.need_filter,
            /*is_join_get=*/false,
            join.joined_block_split_single_row,
            join.enable_lazy_columns_replication,
            join.enable_lazy_columns_indexing});

    if (next_scattered_block)
        join_result->setNextBlock(std::move(next_scattered_block.value()));
    return join_result;
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename AddedColumns>
size_t RoutedHashJoinMethods<KIND, STRICTNESS, MapsTemplate>::switchJoinRightColumns(
    const std::vector<const HashJoin *> & slot_joins,
    const RoutedProbePlan & plan,
    AddedColumns & added_columns,
    const ScatteredBlock::Selector & selector,
    const UInt8 * slot_ids,
    JoinProbeScratch & scratch)
{
    constexpr bool is_asof_join = STRICTNESS == JoinStrictness::Asof;
    const HashJoin & join0 = *slot_joins[0];

    chassert(plan.map_by_slot.size() == slot_joins.size());
    chassert(added_columns.join_on_keys.size() == 1);
    const auto & join_on_key = added_columns.join_on_keys[0];

    /// The map type is uniform across the slots: it is chosen from the shared right sample
    /// block, `ConcurrentHashJoin` never runs the post-build fixed-map conversion, and the
    /// All -> RightAny promotion is synchronized before the per-slot `onBuildPhaseFinish`.
    switch (join0.data->type)
    {
/// The once-per-build plan holds the type-erased map pointers; this switch is the same one that
/// erased them, so the cast back is exact.
#define M(TYPE) \
    case HashJoin::Type::TYPE: { \
        using MapTypeVal = const typename std::remove_reference_t<decltype(MapsTemplate::TYPE)>::element_type; \
        using KeyGetter = typename KeyGetterForType<HashJoin::Type::TYPE, MapTypeVal>::Type; \
        const auto * const * maps_by_slot = reinterpret_cast<const MapTypeVal * const *>(plan.map_by_slot.data()); \
        return joinRightColumnsRouted<KeyGetter, MapTypeVal>( \
            slot_joins, \
            plan, \
            createKeyGetter<KeyGetter, is_asof_join>(join_on_key.key_columns, join_on_key.key_sizes, join0.data->key_range), \
            maps_by_slot, \
            added_columns, \
            selector, \
            slot_ids, \
            scratch); \
    }
        APPLY_FOR_JOIN_VARIANTS(M)
#undef M
    }
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename KeyGetter, typename Map, typename AddedColumns>
size_t RoutedHashJoinMethods<KIND, STRICTNESS, MapsTemplate>::joinRightColumnsRouted(
    const std::vector<const HashJoin *> & slot_joins,
    const RoutedProbePlan & plan,
    KeyGetter && key_getter,
    const Map * const * maps_by_slot,
    AddedColumns & added_columns,
    const ScatteredBlock::Selector & selector,
    const UInt8 * slot_ids,
    JoinProbeScratch & scratch)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;

    if constexpr (join_features.is_maps_all)
    {
        if (added_columns.additional_filter_expression)
        {
            /// The mixed ON-condition path: the shared filter machinery of `HashJoinMethods`
            /// with the per-row map/flags selection routed by slot. Single clause, so per-row
            /// (block-keyed) flags are needed only for the flagged RIGHT/FULL shapes.
            /// This path consumes a slot-ids ARRAY for every family - `joinBlock` fills it
            /// eagerly whenever the join carries a mixed expression, hash-routed or not.
            chassert(slot_ids || plan.map_by_slot.size() == 1);
            const bool mark_per_row_used = join_features.right || join_features.full;
            const RoutedProbeContext<Map> routed_ctx{
                .slot_ids = slot_ids,
                .maps_by_slot = maps_by_slot,
                .flags_by_slot = plan.flags_by_slot.data(),
                .total_map_bytes = plan.total_map_bytes};
            std::vector<KeyGetter> key_getter_vector;
            key_getter_vector.push_back(std::forward<KeyGetter>(key_getter));
            const std::vector<const Map *> mapv{maps_by_slot[0]};
            return HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::template joinRightColumnsWithAdditionalFilter<KeyGetter, Map>(
                std::move(key_getter_vector),
                mapv,
                added_columns,
                *plan.flags_by_slot[0],
                selector,
                added_columns.need_filter,
                mark_per_row_used,
                &routed_ctx);
        }
    }

    if (added_columns.additional_filter_expression)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Additional filter expression is not supported for this JOIN");

    if (selector.isContinuousRange())
        return joinRightColumns<KeyGetter, Map>(
            slot_joins, plan, key_getter, maps_by_slot, added_columns, selector.getRange(), slot_ids, scratch);
    else
        return joinRightColumns<KeyGetter, Map>(
            slot_joins, plan, key_getter, maps_by_slot, added_columns, selector.getIndexes(), slot_ids, scratch);
}

/// The key families the routed probe routes by the maps' own hash (see `joinHashRouteSlot`):
/// every cursor-capable open-addressing family. Their lookups hash the key anyway, so the row's
/// slot falls out of the hash's top route bits with no separate per-row slot-ids pass, and the
/// lookup itself is `flatFindKey` when the ring is not engaged. Unlike the ring, the flat find
/// serves every mapped shape (ASOF included - the match is consumed in place, so the mapped
/// value never needs to fit a word) and the wrapped-chain builds the ring must refuse.
/// `key8`/`key16` and the range maps (`FixedHashMap` - no hash at all) keep the map-resolved
/// `findKey` under the slot ids `joinBlock` derives eagerly (`computeDispatchSlotIds`).
template <typename Map>
constexpr bool hash_routed_lookup = AmacResumableMap<std::remove_const_t<Map>>;

/// The flat find of the routed plain loop: address material from the once-per-build plan, the
/// map object only for its hash/equality functors. The walk matches the standard grower's
/// masked linear probing. The offset is the slot-local used-flags offset
/// (`offsetInternal` semantics).
template <typename KeyGetter, typename Map, typename KeyType>
ALWAYS_INLINE typename KeyGetter::FindResult flatFindKey(const SlotMapDesc & desc, const KeyType & key, size_t hash)
{
    using Cell = typename std::remove_const_t<Map>::cell_type;
    using Mapped = typename std::remove_const_t<Map>::mapped_type;
    static constexpr HashTableNoState no_state{};

    const Cell * buf = static_cast<const Cell *>(desc.buf);
    const Cell * end = buf + desc.mask + 1;
    const Cell * cell = buf + (hash & desc.mask);
    while (!cell->isZero(no_state))
    {
        if (cell->keyEquals(key, hash, no_state))
        {
            /// The probe maps are immutable; the const cast only recovers `findKey`'s own
            /// `FindResult` shape (its mapped pointer is non-const).
            auto * mapped = const_cast<Mapped *>(&cell->getMapped());
            return typename KeyGetter::FindResult(mapped, true, static_cast<size_t>(cell - buf) + 1);
        }
        if (unlikely(++cell == end))
            cell = buf;
    }
    return typename KeyGetter::FindResult(nullptr, false, 0);
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename KeyGetter, typename Map, typename AddedColumnsType, typename Selector>
size_t RoutedHashJoinMethods<KIND, STRICTNESS, MapsTemplate>::joinRightColumns(
    const std::vector<const HashJoin *> & slot_joins [[maybe_unused]],
    const RoutedProbePlan & plan,
    KeyGetter & key_getter,
    const Map * const * maps_by_slot,
    AddedColumnsType & added_columns,
    const Selector & selector,
    const UInt8 * slot_ids,
    JoinProbeScratch & scratch)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;
    /// Single join clause (`parallel_hash` supports no disjuncts), so right-row used flags are
    /// per-offset, never per-row.
    constexpr bool flag_per_row = false;

    const size_t rows = ScatteredBlock::Selector::size(selector);
    const auto & join_keys = added_columns.join_on_keys.at(0);

    /// The skip pointer is a local so that it can stay in a register across the calls in
    /// the loop body (see `JoinOnKeyColumns::buildRowSkipData`).
    const UInt8 * skip_data = nullptr;
    IColumn::Filter skip_buffer;
    const bool fast_path = !join_keys.null_map && join_keys.join_mask_column.getKind() == JoinCommon::JoinMask::Kind::AllTrue;
    if (!fast_path)
    {
        if constexpr (std::is_same_v<std::decay_t<Selector>, ScatteredBlock::Indexes>)
            skip_data = join_keys.buildRowSkipData(skip_buffer, selector);
        else
            skip_data = join_keys.buildRowSkipData(skip_buffer, selector.first, rows);
    }

    if constexpr (!flag_per_row && (STRICTNESS == JoinStrictness::All || (STRICTNESS == JoinStrictness::Semi && KIND == JoinKind::Right)))
        added_columns.lazy_output.output_by_row_list = true;

    Arena pool;

    if constexpr (join_features.need_replication)
        added_columns.offsets_to_replicate = IColumn::Offsets(rows);

    /// Snapshots of the plan's arrays: the loop bodies below make opaque calls
    /// (`appendFromBlock`), after which the compiler must conservatively reload anything
    /// reachable through a captured object.
    JoinStuff::JoinUsedFlags * const * flags_data = plan.flags_by_slot.data();
    /// Sized for every cursor-capable map type (`hash_routed_lookup` implies
    /// cursor-capable), empty for the rest - where the hash-routed arms are compiled out.
    [[maybe_unused]] const SlotMapDesc * descs_data = plan.desc_by_slot.data();
    /// Hash provider and zero-key checker of the hash-routed arms; stateless, so any slot's
    /// map serves (see `map0` in `AmacProbeImpl.h`).
    [[maybe_unused]] const Map & map0 = *maps_by_slot[0];
    /// The hash-derived route of the cursor families (see `joinHashRouteSlot`); the slot
    /// count is a power of two, so this is exact (32 at the single-slot plan).
    [[maybe_unused]] const auto route_shift = static_cast<UInt32>(32 - std::countr_zero(plan.map_by_slot.size()));

    using MapNonConst = std::remove_const_t<Map>;

    /// AMAC find-ring engagement (see `AmacProbe.h`): the process hook, the per-join opt-in
    /// (only `ConcurrentHashJoin` sets it), and - under `Auto` - the size thresholds over the
    /// AGGREGATE map bytes; `Force` bypasses the thresholds so tests and A/B harnesses can pin
    /// the path.
    bool use_amac = false;
    if constexpr (amac_probe_supported<KeyGetter, Map>)
    {
        const AmacMode amac_mode = joinAmacMode();
        /// Cheap-key ASOF stays out under `Auto`: the fleet measured the pointer-recording
        /// ring a net loss there (the sorted-vector search dominates a numeric key's cheap
        /// hash, and the ring's two-phase cost stacks on top), while string ASOF keys are
        /// expensive enough that the ring's memory-level parallelism still pays. `Force`
        /// engages every supported shape - tests drive it there.
        constexpr bool auto_engageable = !join_features.is_asof_join || !KeyGetter::has_cheap_key_calculation;
        use_amac = amac_mode != AmacMode::Off && slot_joins[0]->amacEnabled()
            && (amac_mode == AmacMode::Force
                || (auto_engageable && plan.total_map_bytes > getMinBytesForPrefetchInJoin() && rows >= amac_min_rows));
    }

    /// Look-ahead software prefetch of the plain loop, mutually exclusive with the AMAC pass,
    /// gated on the AGGREGATE map bytes across the slots: each row misses in its own slot's
    /// map, and the set of maps one block touches is the whole fleet of them, so a per-slot
    /// size check would under-gate the routed loop.
    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;
    bool use_prefetch = false;
    if constexpr (can_prefetch)
        use_prefetch = !use_amac && added_columns.enable_prefetch && plan.total_map_bytes > getMinBytesForPrefetchInJoin();

    auto prefetcher = makeJoinPrefetcher(use_prefetch, rows,
        [&](size_t k) __attribute__((always_inline))
        {
            if constexpr (hash_routed_lookup<Map>)
            {
                /// The flat find's look-ahead: the home-cell address from the descriptor -
                /// the hash routes AND places, so no map-header loads and no slot-ids array
                /// on the prefetch path either.
                const size_t ind = selectorIndexAt(selector, k);
                auto && key_holder = key_getter.getKeyHolder(ind, pool);
                const auto & key = keyHolderGetKey(key_holder);
                const size_t hash = map0.hash(key);
                const SlotMapDesc & desc = descs_data[joinHashRouteSlot(hash, route_shift)];
                using Cell = typename MapNonConst::cell_type;
                __builtin_prefetch(static_cast<const Cell *>(desc.buf) + (hash & desc.mask));
            }
            else if constexpr (can_prefetch)
            {
                const size_t ind = selectorIndexAt(selector, k);
                maps_by_slot[slot_ids ? slot_ids[ind] : 0]->prefetch(key_getter.getKeyHolder(ind, pool));
            }
        });

    /// The in-order find/emit loop. With `precomputed` the lookup is replaced by the AMAC find
    /// pass's per-row result (0 = miss; skipped rows were recorded as misses there, so the skip
    /// check compiles out); without it this is the sequential routed lookup. Everything
    /// downstream of the lookup is shared and standard.
    auto loop = [&]<bool need_filter, bool with_skip, bool precomputed>(
        const UInt64 * found_words [[maybe_unused]],
        const UInt64 * found_offsets [[maybe_unused]],
        const UInt8 * found_slots [[maybe_unused]])
    {
        if constexpr (need_filter)
        {
            added_columns.filter = IColumn::Filter(rows, 0);
            added_columns.matched_rows.reserve(rows);
        }

        IColumn::Offset current_offset = 0;
        for (size_t i = 0; i < rows; ++i)
        {
            if constexpr (can_prefetch && !precomputed)
                prefetcher.prefetchAt(i);

            const size_t ind = selectorIndexAt(selector, i);

            bool right_row_found = false;
            KnownRowsHolder<flag_per_row> dummy_known_rows;

            if constexpr (precomputed)
            {
                using Mapped = std::remove_reference_t<decltype(std::declval<typename KeyGetter::FindResult &>().getMapped())>;
                /// The find pass recorded the map's mapped value - by word where it fits one,
                /// by pointer for ASOF; `amac_probe_supported` is exactly that disjunction, so
                /// the arms below are exhaustive. This side rebuilds the `FindResult` from its
                /// own mapped type; the type equality keeps a word from being rebuilt as the
                /// wrong mapped type.
                static_assert(std::is_same_v<std::remove_const_t<Mapped>, typename std::remove_const_t<Map>::mapped_type>);
                if (const UInt64 word = found_words[i])
                {
                    right_row_found = true;
                    size_t offset = 0;
                    /// The flagged shapes read the row's route slot as the find pass derived
                    /// it; the flagless ones never consume the flags object, so slot 0's
                    /// reference serves as the placeholder.
                    size_t slot = 0;
                    if constexpr (join_features.need_flags)
                    {
                        offset = found_offsets[i];
                        slot = found_slots[i];
                    }
                    auto emit = [&](Mapped * mapped) ALWAYS_INLINE
                    {
                        typename KeyGetter::FindResult find_result(mapped, true, offset);
                        processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsTemplate, Map, KeyGetter>(
                            find_result, added_columns, *flags_data[slot], i, ind, current_offset, dummy_known_rows);
                    };
                    if constexpr (amac_mapped_fits_word<std::remove_const_t<Mapped>>)
                    {
                        auto mapped_value = mappedFromWord<std::remove_const_t<Mapped>>(word);
                        emit(&mapped_value);
                    }
                    else
                    {
                        /// ASOF: the find pass recorded the mapped value's address (see
                        /// `amac_mapped_by_pointer`); never flagged, so `offset` stays 0.
                        emit(reinterpret_cast<Mapped *>(word));
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
                    size_t slot = 0;
                    auto find_result = [&]() ALWAYS_INLINE
                    {
                        /// The trait check keeps the hash-routed arm out of the non-cursor
                        /// families' instantiations.
                        if constexpr (hash_routed_lookup<Map>)
                        {
                            auto && key_holder = key_getter.getKeyHolder(ind, pool);
                            const auto & key = keyHolderGetKey(key_holder);
                            /// One hash per row: it derives the route slot AND places the
                            /// cell - exactly the hash a map-resolved `findKey` would burn
                            /// internally.
                            const size_t hash = map0.hash(key);
                            slot = joinHashRouteSlot(hash, route_shift);
                            /// The zero key lives in the dedicated zero-value cell, outside the
                            /// descriptor's buffer; the map-resolved find serves it.
                            if (unlikely(map0.isZeroKey(key)))
                                return key_getter.findKey(*maps_by_slot[slot], ind, pool);
                            return flatFindKey<KeyGetter, Map>(descs_data[slot], key, hash);
                        }
                        else
                        {
                            slot = slot_ids ? slot_ids[ind] : 0;
                            return key_getter.findKey(*maps_by_slot[slot], ind, pool);
                        }
                    }();
                    if (find_result.isFound())
                    {
                        right_row_found = true;
                        processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsTemplate, Map, KeyGetter>(
                            find_result, added_columns, *flags_data[slot], i, ind, current_offset, dummy_known_rows);
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
    /// must be the mapped value itself, the emit must be the lazy ref-word append, and the
    /// shape must consume no per-row state beyond the filter, the appended words and the
    /// replication offsets. The flagged shapes (RIGHT/FULL used flags, incl. `setUsedOnce` -
    /// every shape with first-match-only semantics is flagged or ANY) and ASOF keep the full
    /// loop above.
    constexpr bool degenerate_phase_b = AddedColumnsType::isLazy()
        && amac_mapped_fits_word<typename MapNonConst::mapped_type> && !join_features.need_flags && !join_features.is_asof_join
        && !join_features.is_any_join;

    /// The degenerate phase B of the two-phase AMAC probe. On the shapes gated above,
    /// `processMatch` reduces to: mark the row matched (filter + `matched_rows`), append the
    /// recorded word (ALL: the list word, advancing the replication offset by its row count;
    /// RightAny/Semi: its first ref) or the default on an added miss - so this pass consumes
    /// `found_word` directly instead of rebuilding a `FindResult` and dispatching
    /// `processMatch` per row, whose outlined `appendFromBlock` call forces the loop-carried
    /// state to spill. Every row appends at most one entry, so the cursors write into pre-sized
    /// arrays with no per-append capacity check. Row order, filter, offsets and `row_count`
    /// are exactly the full loop's.
    [[maybe_unused]] auto word_loop = [&]<bool need_filter, bool with_refs>(const UInt64 * words [[maybe_unused]])
    {
        if constexpr (degenerate_phase_b)
        {
            using Mapped = MapNonConst::mapped_type;

            if constexpr (need_filter)
            {
                added_columns.filter = IColumn::Filter(rows, 0);
                added_columns.matched_rows.resize(rows);
            }

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

    bool amac_ran = false;
    if constexpr (amac_probe_supported<KeyGetter, Map>)
    {
        if (use_amac)
        {
            /// Phase A: the AMAC find pass. Every row gets a result - `start` records skipped
            /// and zero-key rows synchronously, `step` records hits and misses - so the arrays
            /// need no pre-fill and phase B needs no skip logic. The offsets and route slots
            /// are recorded (and sized) only for the flagged shapes - they have no other
            /// consumer. The arrays live in the lane's pooled scratch; `resize` never shrinks
            /// a `PaddedPODArray`, so the capacity survives across blocks.
            scratch.found_word.resize(rows);
            UInt64 * found_word_data = scratch.found_word.data();
            UInt64 * found_offset_data = nullptr;
            UInt8 * found_slot_data = nullptr;
            if constexpr (join_features.need_flags)
            {
                scratch.found_offset.resize(rows);
                found_offset_data = scratch.found_offset.data();
                scratch.found_slot.resize(rows);
                found_slot_data = scratch.found_slot.data();
            }

            constexpr bool selector_is_range = !std::is_same_v<std::decay_t<Selector>, ScatteredBlock::Indexes>;
            size_t range_first = 0;
            const UInt64 * sel_indexes = nullptr;
            if constexpr (selector_is_range)
                range_first = selector.first;
            else
                sel_indexes = selector.getData().data();

            amacFindPass<KeyGetter, MapNonConst, join_features.need_flags, selector_is_range>(
                key_getter,
                maps_by_slot,
                descs_data,
                route_shift,
                rows,
                range_first,
                sel_indexes,
                skip_data,
                pool,
                found_word_data,
                found_offset_data,
                found_slot_data);

            if constexpr (degenerate_phase_b)
            {
                auto word_dispatch = [&]<bool need_filter>()
                {
                    if (added_columns.has_columns_to_add)
                        word_loop.template operator()<need_filter, true>(found_word_data);
                    else
                        word_loop.template operator()<need_filter, false>(found_word_data);
                };
                if (added_columns.need_filter)
                    word_dispatch.template operator()<true>();
                else
                    word_dispatch.template operator()<false>();
            }
            else
            {
                if (added_columns.need_filter)
                    loop.template operator()<true, false, true>(found_word_data, found_offset_data, found_slot_data);
                else
                    loop.template operator()<false, false, true>(found_word_data, found_offset_data, found_slot_data);
            }
            amac_ran = true;
        }
    }

    if (!amac_ran)
    {
        /// The cursor-capable families run the hash-routed flat find (`hash_routed_lookup`)
        /// instead of the map-resolved `findKey`; everything else (`LowCardinality`,
        /// `key8`/`key16`, the range maps) keeps the plain lookup under the eager slot ids.
        if (added_columns.need_filter)
        {
            if (fast_path)
                loop.template operator()<true, false, false>(nullptr, nullptr, nullptr);
            else
                loop.template operator()<true, true, false>(nullptr, nullptr, nullptr);
        }
        else
        {
            if (fast_path)
                loop.template operator()<false, false, false>(nullptr, nullptr, nullptr);
            else
                loop.template operator()<false, true, false>(nullptr, nullptr, nullptr);
        }
    }

    added_columns.applyLazyDefaults();
    return 0;
}
}
