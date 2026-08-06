#pragma once

#include <Columns/IColumn.h>
#include <Common/HashTable/Prefetching.h>
#include <Interpreters/ExpressionActions.h>
#include <Interpreters/UnifiedHashJoin/AddedColumns.h>
#include <Interpreters/UnifiedHashJoin/HashJoinMethods.h>
#include <Interpreters/UnifiedHashJoin/HashJoinResult.h>
#include <Interpreters/UnifiedHashJoin/ProbeLookup.h>
#include <Interpreters/JoinUtils.h>

#include <algorithm>
#include <type_traits>

namespace DB
{
namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}

namespace Unified
{
/// Check if the hash table type supports the prefetch interface.
template <typename Map, typename KeyHolder>
concept HasPrefetchMemberFunc = requires
{
    {std::declval<Map>().prefetch(std::declval<KeyHolder>())};
};

/// True when software prefetch in the JOIN probe loop is feasible for the given key
/// getter / map combination: the key getter must compute keys cheaply and the map
/// must expose a `prefetch` member that takes the key holder produced by the getter.
template <typename KeyGetter, typename Map>
constexpr bool join_prefetch_supported = KeyGetter::has_cheap_key_calculation
    && HasPrefetchMemberFunc<
        std::remove_const_t<Map>,
        decltype(std::declval<KeyGetter &>().getKeyHolder(std::declval<size_t>(), std::declval<Arena &>()))>;

/// Decide at runtime whether prefetching should actually fire for a given map: the user
/// must have it enabled and the map must be large enough that we expect non-trivial
/// cache misses to amortize the prefetch cost.
template <typename Map>
ALWAYS_INLINE bool shouldUseJoinPrefetch(bool enable_prefetch, const Map * map)
{
    return enable_prefetch && map != nullptr
        && map->getBufferSizeInBytes() > getMinBytesForPrefetchInJoin();
}

/** The probe's look-ahead software prefetch as a named type. A lambda's closure type is
  * minted where the lambda is written - inside `joinRightColumns`, which is instantiated
  * per (join variant x need_filter) - and every lookup body templated on it multiplied the
  * same way. This struct's type depends only on (Map, KeyGetter, Selector), which is what
  * lets the single-clause lookup instantiate per key type alone.
  *
  * Semantics are exactly `JoinPrefetcher` driving a `map->prefetch(getKeyHolder(...))`
  * action: called with the ABSOLUTE row (P5), and the look-ahead is calibrated once when
  * the absolute row reaches `iterationsToMeasure()`. One instance must therefore live for
  * the whole probe call - an instance constructed per batch would never calibrate for
  * `begin > 0` (F11). Members may be null when `use_prefetch` is false; they are only
  * dereferenced behind that flag.
  */
template <typename Map, typename KeyGetter, typename Selector>
struct ProbePrefetch
{
    const Map * map = nullptr;
    KeyGetter * key_getter = nullptr;
    const Selector * selector = nullptr;
    Arena * pool = nullptr;
    bool use_prefetch = false;
    size_t total = 0;
    PrefetchingHelper prefetching{};
    size_t prefetch_look_ahead = PrefetchingHelper::getInitialLookAheadValue();

    ALWAYS_INLINE void operator()(size_t absolute_row)
    {
        if constexpr (join_prefetch_supported<KeyGetter, Map>)
        {
            if (!use_prefetch)
                return;

            /// Estimate optimal look-ahead distance once we have measured iteration latency.
            if (absolute_row == PrefetchingHelper::iterationsToMeasure())
                prefetch_look_ahead = prefetching.calcPrefetchLookAhead();

            const size_t prefetch_idx = absolute_row + prefetch_look_ahead;
            if (prefetch_idx < total)
                map->prefetch(key_getter->getKeyHolder(selectorIndexAt(*selector, prefetch_idx), *pool));
        }
    }
};

/// Drives the adaptive software prefetch logic in the hash join probe loop.
template <typename PrefetchAction>
struct JoinPrefetcher
{
    bool use_prefetch = false;
    size_t total = 0;
    PrefetchAction prefetch_action;
    PrefetchingHelper prefetching{};
    size_t prefetch_look_ahead = PrefetchingHelper::getInitialLookAheadValue();

    ALWAYS_INLINE void prefetchAt(size_t i)
    {
        if (!use_prefetch)
            return;

        /// Estimate optimal look-ahead distance once we have measured iteration latency.
        if (i == PrefetchingHelper::iterationsToMeasure())
            prefetch_look_ahead = prefetching.calcPrefetchLookAhead();

        const size_t prefetch_idx = i + prefetch_look_ahead;
        if (prefetch_idx < total)
            prefetch_action(prefetch_idx);
    }
};

template <typename PrefetchAction>
ALWAYS_INLINE auto makeJoinPrefetcher(bool use_prefetch, size_t total, PrefetchAction && prefetch_action)
{
    return JoinPrefetcher<std::decay_t<PrefetchAction>>{
        use_prefetch, total, std::forward<PrefetchAction>(prefetch_action)};
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
void HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::insertFromBlockImpl(
    HashJoin & join,
    HashJoin::Type type,
    MapsTemplate & maps,
    BlockKeyGetter & block_key_getter,
    const ColumnRawPtrs & key_columns,
    const Sizes & key_sizes,
    UInt32 stored_block_no,
    const ScatteredBlock::Selector & selector,
    const Columns * dense_keys,
    ConstNullMapPtr null_map,
    const JoinCommon::JoinMask & join_mask,
    Arena & pool,
    BuildResult & result)
{
    switch (type)
    {
#define M(TYPE) \
    case HashJoin::Type::TYPE: \
        if (selector.isContinuousRange()) \
            insertFromBlockImplTypeCase< \
                typename KeyGetterForType<HashJoin::Type::TYPE, std::remove_reference_t<decltype(*maps.TYPE)>, needs_offset>::Type>( \
                join, \
                *maps.TYPE, \
                block_key_getter, \
                key_columns, \
                key_sizes, \
                stored_block_no, \
                selector.getRange(), \
                dense_keys, \
                null_map, \
                join_mask, \
                pool, \
                result); \
        else \
            insertFromBlockImplTypeCase< \
                typename KeyGetterForType<HashJoin::Type::TYPE, std::remove_reference_t<decltype(*maps.TYPE)>, needs_offset>::Type>( \
                join, \
                *maps.TYPE, \
                block_key_getter, \
                key_columns, \
                key_sizes, \
                stored_block_no, \
                selector.getIndexes(), \
                dense_keys, \
                null_map, \
                join_mask, \
                pool, \
                result); \
        break;

            UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
#undef M
    }
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::SlotScatter
HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::scatterBySlot(
    HashJoin::Type type,
    MapsTemplate & maps,
    BlockKeyGetter & block_key_getter,
    const ColumnRawPtrs & key_columns,
    const Sizes & key_sizes,
    const ScatteredBlock::Selector & selector,
    size_t num_slots)
{
    switch (type)
    {
#define M(TYPE) \
    case HashJoin::Type::TYPE: \
        return scatterBySlotTypeCase< \
            typename KeyGetterForType<HashJoin::Type::TYPE, std::remove_reference_t<decltype(*maps.TYPE)>, needs_offset>::Type>( \
            *maps.TYPE, block_key_getter, key_columns, key_sizes, selector, num_slots);

            UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
#undef M
    }
    UNREACHABLE();
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename KeyGetter, typename HashMap>
HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::SlotScatter
HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::scatterBySlotTypeCase(
    const HashMap & map,
    BlockKeyGetter & block_key_getter,
    const ColumnRawPtrs & key_columns,
    const Sizes & key_sizes,
    const ScatteredBlock::Selector & selector,
    size_t num_slots)
{
    /// Bucket selection must use the same hash as insertion; a precomputed hash could disagree with map placement.
    if constexpr (requires { KeyGetter::has_pre_computed_hashes; })
        static_assert(!KeyGetter::has_pre_computed_hashes, "Bucket routing assumes the map computes the hash it places by");

    constexpr bool is_asof_join = STRICTNESS == JoinStrictness::Asof;
    std::optional<KeyGetter> own_key_getter;
    auto & key_getter = blockKeyGetter<KeyGetter, is_asof_join>(block_key_getter, own_key_getter, key_columns, key_sizes);

    /// Key holders are only read here, never persisted into the map, so nothing allocated through
    /// this outlives the call. String key holders do not touch it at all.
    Arena scratch_pool;

    const size_t rows = selector.size();

    PODArray<UInt32> row_to_slot(rows);
    std::vector<size_t> counts(num_slots, 0);
    for (size_t i = 0; i < rows; ++i)
    {
        auto key_holder = key_getter.getKeyHolder(selector[i], scratch_pool);
        const auto & key = keyHolderGetKey(key_holder);

        size_t hash_value = 0;
        if constexpr (requires { key_getter.routingHashForRow(map, selector[i], scratch_pool); })
            hash_value = key_getter.routingHashForRow(map, selector[i], scratch_pool);
        else
            hash_value = map.hash(key);

        const size_t bucket = map.getBucketFromHash(map.bucketRoutingHash(key, hash_value));
        const auto slot = static_cast<UInt32>(slotForBucket(bucket, num_slots));
        row_to_slot[i] = slot;
        ++counts[slot];
    }

    std::vector<ScatteredBlock::Selector::IndexesPtr> indexes;
    indexes.reserve(num_slots);
    for (size_t slot = 0; slot < num_slots; ++slot)
    {
        auto column = ScatteredBlock::Selector::Indexes::create();
        column->getData().reserve(counts[slot]);
        indexes.push_back(std::move(column));
    }

    for (size_t i = 0; i < rows; ++i)
        indexes[row_to_slot[i]]->getData().push_back(selector[i]);

    SlotScatter result;
    result.selectors.reserve(num_slots);
    for (auto & column : indexes)
        result.selectors.emplace_back(std::move(column));

    constexpr size_t threshold = sizeof(IColumn::Selector::value_type);
    size_t max_bytes_per_row = 0;
    for (const auto * column : key_columns)
        max_bytes_per_row += (column->valuesHaveFixedSize() && !column->lowCardinality()) ? column->sizeOfValueIfFixed() : threshold + 1;

    const bool selector_is_identity = selector.isContinuousRange() && selector.getRange().first == 0
        && !key_columns.empty() && selector.getRange().second == key_columns[0]->size();

    if (max_bytes_per_row <= threshold && selector_is_identity)
    {
        IColumn::Selector column_selector(rows);
        for (size_t i = 0; i < rows; ++i)
            column_selector[i] = row_to_slot[i];

        result.dense_keys.resize(num_slots);
        for (const auto * column : key_columns)
        {
            auto parts = column->scatter(num_slots, column_selector);
            chassert(parts.size() == num_slots);
            for (size_t slot = 0; slot < num_slots; ++slot)
                result.dense_keys[slot].push_back(std::move(parts[slot]));
        }
    }

    return result;
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
JoinResultPtr HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::joinBlockImpl(
    const HashJoin & join, Block block, const Block & block_with_columns_to_add, const MapsTemplateVector & maps_)
{
    ScatteredBlock scattered_block{std::move(block)};
    return joinBlockImpl(join, std::move(scattered_block), block_with_columns_to_add, maps_);
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
JoinResultPtr HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::joinBlockImpl(
    const HashJoin & join,
    ScatteredBlock block,
    const Block & block_with_columns_to_add,
    const MapsTemplateVector & maps_)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;

    std::vector<JoinOnKeyColumns> join_on_keys;
    const auto & onexprs = join.table_join->getClauses();
    for (size_t i = 0; i < onexprs.size(); ++i)
    {
        join_on_keys.emplace_back(
            block, onexprs[i].key_names_left, onexprs[i].condColumnNames().first, join.key_sizes[i],
            HashJoin::isLowCardinalityType(join.data->type));
    }


    /** For LEFT/INNER JOIN, the saved blocks do not contain keys.
      * For FULL/RIGHT JOIN, the saved blocks contain keys;
      *  but they will not be used at this stage of joining (and will be in `CollectorNonJoined`), and they need to be skipped.
      * For ASOF, the last column is used as the ASOF column
      */
    AddedColumns<!join_features.is_any_join> added_columns(
        block,
        block_with_columns_to_add,
        join.savedBlockSample(),
        join,
        std::move(join_on_keys),
        join.table_join->getMixedJoinExpression(),
        join.additional_filter_required_rhs_pos,
        join_features.is_asof_join);

    bool has_required_right_keys = (join.required_right_keys.columns() != 0);
    added_columns.need_filter = join_features.need_filter || has_required_right_keys;
    added_columns.max_joined_block_rows = join.max_joined_block_rows;
    if (!added_columns.max_joined_block_rows)
        added_columns.max_joined_block_rows = std::numeric_limits<size_t>::max();
    else
        added_columns.reserve(join_features.need_replication);

    size_t processed_rows = switchJoinRightColumns(maps_, added_columns, block.getSelector(), join.data->type, *join.used_flags, join.data->key_range);
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
            join.data->allocated_size / std::max<size_t>(1, join.data->rows_to_join),
            join_features.need_filter,
            join.joined_block_split_single_row,
            join.enable_lazy_columns_replication,
            join.enable_lazy_columns_indexing
        });

    if (next_scattered_block)
        join_result->setNextBlock(std::move(next_scattered_block.value()));
    return join_result;
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename KeyGetter, bool is_asof_join>
KeyGetter HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::createKeyGetter(const ColumnRawPtrs & key_columns, const Sizes & key_sizes, HashJoin::RightTableData::KeyRange key_range)
{
    KeyGetter getter = [&]()
    {
        if constexpr (is_asof_join)
        {
            auto key_column_copy = key_columns;
            auto key_size_copy = key_sizes;
            key_column_copy.pop_back();
            key_size_copy.pop_back();
            return KeyGetter(key_column_copy, key_size_copy, nullptr);
        }
        else
        {
            return KeyGetter(key_columns, key_sizes, nullptr);
        }
    }();

    if constexpr (ColumnsHashing::IsHashMethodInRange<KeyGetter>::value)
    {
        getter.min_key = static_cast<decltype(getter.min_key)>(key_range.min_key);
        getter.range_size = static_cast<decltype(getter.range_size)>(key_range.size);
    }

    return getter;
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename KeyGetter, bool is_asof_join>
KeyGetter & HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::blockKeyGetter(
    BlockKeyGetter & block_key_getter, std::optional<KeyGetter> & own, const ColumnRawPtrs & key_columns, const Sizes & key_sizes)
{
    const auto create = [&] { return createKeyGetter<KeyGetter, is_asof_join>(key_columns, key_sizes); };

    if constexpr (shareKeyGetterAcrossBuckets<KeyGetter>())
        return block_key_getter.getOrBuild<KeyGetter>(create);
    else
        return own.emplace(create());
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename KeyGetter, typename HashMap, typename Selector>
void HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::insertFromBlockImplTypeCase(
    HashJoin & join,
    HashMap & map,
    BlockKeyGetter & block_key_getter,
    const ColumnRawPtrs & key_columns,
    const Sizes & key_sizes,
    UInt32 stored_block_no,
    const Selector & selector,
    const Columns * dense_keys,
    ConstNullMapPtr null_map,
    const JoinCommon::JoinMask & join_mask,
    Arena & pool,
    BuildResult & result)
{
    [[maybe_unused]] constexpr bool mapped_one = std::is_same_v<typename HashMap::mapped_type, RowRef>;
    constexpr bool is_asof_join = STRICTNESS == JoinStrictness::Asof;

    const IColumn * asof_column [[maybe_unused]] = nullptr;
    if constexpr (is_asof_join)
        asof_column = key_columns.back();

    const size_t rows = ScatteredBlock::Selector::size(selector);

    std::optional<KeyGetter> own_key_getter;
    ColumnRawPtrs dense_key_ptrs;
    KeyGetter * key_getter_ptr = nullptr;
    if (dense_keys)
    {
        chassert(!dense_keys->empty() && dense_keys->front()->size() == rows);
        dense_key_ptrs.reserve(dense_keys->size());
        for (const auto & column : *dense_keys)
            dense_key_ptrs.push_back(column.get());
        key_getter_ptr = &own_key_getter.emplace(createKeyGetter<KeyGetter, is_asof_join>(dense_key_ptrs, key_sizes));
    }
    else
    {
        key_getter_ptr = &blockKeyGetter<KeyGetter, is_asof_join>(block_key_getter, own_key_getter, key_columns, key_sizes);
    }
    auto & key_getter = *key_getter_ptr;

    /// For ALL and ASOF join always insert values
    result.is_inserted = !mapped_one || is_asof_join;

    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, HashMap>;

    bool use_prefetch = false;
    if constexpr (can_prefetch)
        use_prefetch = shouldUseJoinPrefetch(join.enable_prefetch, &map);

    const bool keys_are_dense = dense_keys != nullptr;

    auto prefetcher = makeJoinPrefetcher(use_prefetch, rows,
        [&](size_t k) __attribute__((always_inline))
        {
            if constexpr (can_prefetch)
                map.prefetch(key_getter.getKeyHolder(keys_are_dense ? k : selectorIndexAt(selector, k), pool));
        });

    for (size_t i = 0; i < rows; ++i)
    {
        if constexpr (can_prefetch)
            prefetcher.prefetchAt(i);

        const size_t ind = selectorIndexAt(selector, i);
        const size_t key_row = keys_are_dense ? i : ind;

        chassert(!null_map || ind < null_map->size());
        if (null_map && (*null_map)[ind])
        {
            /// nulls are not inserted into hash table,
            /// keep them for RIGHT and FULL joins
            result.is_inserted = true;
            continue;
        }

        /// ON-filtered rows stay out of the map; NULL rows above still mark RIGHT/FULL output.
        if (join_mask.isRowFiltered(ind))
            continue;

        if constexpr (is_asof_join)
            Inserter<HashMap, KeyGetter>::insertAsof(
                join, map, key_getter, stored_block_no, key_row, ind, pool, result.new_keys, *asof_column);
        else if constexpr (mapped_one)
            result.is_inserted |= Inserter<HashMap, KeyGetter>::insertOne(
                join, map, key_getter, stored_block_no, key_row, ind, pool, result.new_keys);
        else
            result.all_values_unique &= Inserter<HashMap, KeyGetter>::insertAll(
                join, map, key_getter, stored_block_no, key_row, ind, pool, result.new_keys);
    }
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename AddedColumns>
size_t HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::switchJoinRightColumns(
    const std::vector<const MapsTemplate *> & mapv,
    AddedColumns & added_columns,
    const ScatteredBlock::Selector & selector,
    HashJoin::Type type,
    JoinStuff::JoinUsedFlags & used_flags,
    HashJoin::RightTableData::KeyRange key_range)
{
    constexpr bool is_asof_join = STRICTNESS == JoinStrictness::Asof;
    switch (type)
    {
#define M(TYPE) \
    case HashJoin::Type::TYPE: { \
        using MapTypeVal = const typename std::remove_reference_t<decltype(MapsTemplate::TYPE)>::element_type; \
        using KeyGetter = typename KeyGetterForType<HashJoin::Type::TYPE, MapTypeVal, needs_offset>::Type; \
        std::vector<const MapTypeVal *> a_map_type_vector(mapv.size()); \
        std::vector<KeyGetter> key_getter_vector; \
        for (size_t d = 0; d < added_columns.join_on_keys.size(); ++d) \
        { \
            const auto & join_on_key = added_columns.join_on_keys[d]; \
            a_map_type_vector[d] = mapv[d]->TYPE.get(); \
            key_getter_vector.push_back( \
                std::move(createKeyGetter<KeyGetter, is_asof_join>(join_on_key.key_columns, join_on_key.key_sizes, key_range))); \
        } \
        return joinRightColumnsSwitchNullability<KeyGetter>(std::move(key_getter_vector), a_map_type_vector, added_columns, selector, used_flags); \
    }
            UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
#undef M

    }
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename KeyGetter, typename Map, typename AddedColumns>
size_t HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::joinRightColumnsSwitchNullability(
    std::vector<KeyGetter> && key_getter_vector,
    const std::vector<const Map *> & mapv,
    AddedColumns & added_columns,
    const ScatteredBlock::Selector & selector,
    JoinStuff::JoinUsedFlags & used_flags)
{
    if (added_columns.need_filter)
        return joinRightColumnsSwitchMultipleDisjuncts<KeyGetter, Map, true>(
            std::forward<std::vector<KeyGetter>>(key_getter_vector), mapv, added_columns, selector, used_flags);
    else
        return joinRightColumnsSwitchMultipleDisjuncts<KeyGetter, Map, false>(
            std::forward<std::vector<KeyGetter>>(key_getter_vector), mapv, added_columns, selector, used_flags);
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename KeyGetter, typename Map, bool need_filter, typename AddedColumns>
size_t HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::joinRightColumnsSwitchMultipleDisjuncts(
    std::vector<KeyGetter> && key_getter_vector,
    const std::vector<const Map *> & mapv,
    AddedColumns & added_columns,
    const ScatteredBlock::Selector & selector,
    JoinStuff::JoinUsedFlags & used_flags)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;
    if constexpr (join_features.is_maps_all)
    {
        if (added_columns.additional_filter_expression)
        {
            const bool mark_per_row_used = join_features.right || join_features.full || mapv.size() > 1;
            return joinRightColumnsWithAdditionalFilter<KeyGetter, Map>(
                std::forward<std::vector<KeyGetter>>(key_getter_vector),
                mapv,
                added_columns,
                used_flags,
                selector,
                need_filter,
                mark_per_row_used);
        }
    }

    if (added_columns.additional_filter_expression)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Additional filter expression is not supported for this JOIN");

    /// Skip vs no-skip is folded inside the lookup driver (P4), so this call site no longer
    /// doubles instantiations on a `fast_path` template axis.
    if (selector.isContinuousRange())
    {
        if (mapv.size() > 1 || added_columns.join_on_keys.empty())
            return joinRightColumns<KeyGetter, Map, need_filter>(
                std::move(key_getter_vector), mapv, added_columns, used_flags, selector.getRange());
        chassert(key_getter_vector.size() == 1);
        return joinRightColumns<KeyGetter, Map, need_filter>(
            key_getter_vector.at(0), mapv.at(0), added_columns, used_flags, selector.getRange());
    }
    if (mapv.size() > 1 || added_columns.join_on_keys.empty())
        return joinRightColumns<KeyGetter, Map, need_filter>(
            std::move(key_getter_vector), mapv, added_columns, used_flags, selector.getIndexes());
    chassert(key_getter_vector.size() == 1);
    return joinRightColumns<KeyGetter, Map, need_filter>(
        key_getter_vector.at(0), mapv.at(0), added_columns, used_flags, selector.getIndexes());
}

template <bool need_filter>
void setUsed(IColumn::Filter & filter [[maybe_unused]], size_t pos [[maybe_unused]], IColumn::Offsets & matched_rows [[maybe_unused]])
{
    if constexpr (need_filter)
    {
        filter[pos] = 1;
        matched_rows.push_back(pos);
    }
}

template <
    JoinKind KIND,
    JoinStrictness STRICTNESS,
    bool need_filter,
    bool flag_per_row,
    typename MapsTemplate,
    typename AddedColumns>
void processMatch(
    const ColumnsHashing::columns_hashing_impl::FindResultImpl<
        const typename MapsTemplate::MappedType,
        JoinFeatures<KIND, STRICTNESS, MapsTemplate>::need_flags> & find_result,
    AddedColumns & added_columns,
    JoinStuff::JoinUsedFlags & used_flags,
    size_t i,
    size_t ind [[maybe_unused]],
    IColumn::Offset & current_offset,
    KnownRowsHolder<flag_per_row> & known_rows,
    bool is_last_disjunct)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;
    using Mapped = typename MapsTemplate::MappedType;

    auto & mapped = find_result.getMapped();
    if constexpr (join_features.is_asof_join)
    {
        const IColumn & left_asof_key = added_columns.leftAsofKey();

        const auto * row_ref = mapped->findAsof(left_asof_key, ind);
        if (row_ref)
        {
            setUsed<need_filter>(added_columns.filter, i, added_columns.matched_rows);
            added_columns.appendFromBlock(row_ref->encode(), join_features.add_missing);
        }
        else
        {
            addNotFoundRow<join_features.add_missing, join_features.need_replication>(added_columns, current_offset);
        }
    }
    else if constexpr (join_features.is_all_join)
    {
        setUsed<need_filter>(added_columns.filter, i, added_columns.matched_rows);
        used_flags.template setUsed<join_features.need_flags, flag_per_row>(find_result);
        /// setUsed already marked every row of the key's list, so addFoundRowAll does not need to call setUsedOnce on the rows it emits.
        addFoundRowAll<Mapped, join_features.add_missing>(mapped, added_columns, current_offset, known_rows, nullptr, is_last_disjunct);
    }
    else if constexpr ((join_features.is_any_join || join_features.is_semi_join) && join_features.right)
    {
        /// Use first appeared left key + it needs left columns replication
        bool used_once = used_flags.template setUsedOnce<join_features.need_flags, flag_per_row>(find_result);
        if (used_once)
        {
            auto used_flags_opt = join_features.need_flags ? &used_flags : nullptr;
            setUsed<need_filter>(added_columns.filter, i, added_columns.matched_rows);
            addFoundRowAll<Mapped, join_features.add_missing>(mapped, added_columns, current_offset, known_rows, used_flags_opt, is_last_disjunct);
        }
    }
    else if constexpr (join_features.is_any_join && join_features.inner)
    {
        bool used_once = used_flags.template setUsedOnce<join_features.need_flags, flag_per_row>(find_result);

        /// Use first appeared left key only
        if (used_once)
        {
            setUsed<need_filter>(added_columns.filter, i, added_columns.matched_rows);
            added_columns.appendFromBlock(firstRefWord(mapped), join_features.add_missing);
        }
    }
    else if constexpr (join_features.is_any_join && join_features.full)
    {
        /// TODO
    }
    else if constexpr (join_features.is_anti_join)
    {
        if constexpr (join_features.right && join_features.need_flags)
            used_flags.template setUsed<join_features.need_flags, flag_per_row>(find_result);
    }
    else /// ANY LEFT, SEMI LEFT, old ANY (RightAny)
    {
        setUsed<need_filter>(added_columns.filter, i, added_columns.matched_rows);
        used_flags.template setUsed<join_features.need_flags, flag_per_row>(find_result);
        added_columns.appendFromBlock(firstRefWord(mapped), join_features.add_missing);
    }
}

/** Sink: do the whole of the join's match processing for a row as its lookup completes.
  *
  * This is the fused probe - there is no second pass and nothing is recorded, because each
  * row's output is produced between its own lookup and the next row's. That is exactly what a
  * recording sink exists to avoid, and exactly what costs nothing when the emit is cheap.
  */
template <
    JoinKind KIND,
    JoinStrictness STRICTNESS,
    bool need_filter,
    typename MapsTemplate,
    typename AddedColumns>
struct EmitSink
{
    using Features = JoinFeatures<KIND, STRICTNESS, MapsTemplate>;
    static constexpr bool flag_per_row = false; /// single-map probe

    AddedColumns & added_columns;
    JoinStuff::JoinUsedFlags & used_flags;
    IColumn::Offset & current_offset;

    ALWAYS_INLINE void miss(size_t /* j */, size_t row)
    {
        if constexpr (Features::is_anti_join && Features::left)
            setUsed<need_filter>(added_columns.filter, row, added_columns.matched_rows);
        addNotFoundRow<Features::add_missing, Features::need_replication>(added_columns, current_offset);
        finish(row);
    }

    template <typename FindResult>
    ALWAYS_INLINE void result(size_t j, size_t row, size_t ind, const FindResult & find_result)
    {
        if (!find_result.isFound())
        {
            miss(j, row);
            return;
        }

        KnownRowsHolder<flag_per_row> dummy_known_rows;
        processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsTemplate>(
            find_result, added_columns, used_flags, row, ind, current_offset, dummy_known_rows, /*is_last_disjunct=*/true);
        finish(row);
    }

private:
    ALWAYS_INLINE void finish(size_t row)
    {
        if constexpr (Features::need_replication)
            added_columns.offsets_to_replicate[row] = current_offset;
    }
};

/** Sink for the multi-clause probe (C15). Flag-per-row is true; the driver owns clause
  * iteration and `KnownRowsHolder`, and calls `finish` once per consumed row so C16's
  * `offsets_to_replicate.push_back` truncates correctly on early exit.
  */
template <
    JoinKind KIND,
    JoinStrictness STRICTNESS,
    bool need_filter,
    typename MapsTemplate,
    typename AddedColumns>
struct MultiEmitSink
{
    using Features = JoinFeatures<KIND, STRICTNESS, MapsTemplate>;
    static constexpr bool flag_per_row = true;

    AddedColumns & added_columns;
    JoinStuff::JoinUsedFlags & used_flags;
    IColumn::Offset & current_offset;

    ALWAYS_INLINE void miss(size_t /* j */, size_t row)
    {
        if constexpr (Features::is_anti_join && Features::left)
            setUsed<need_filter>(added_columns.filter, row, added_columns.matched_rows);
        addNotFoundRow<Features::add_missing, Features::need_replication>(added_columns, current_offset);
    }

    template <typename FindResult, typename KnownRows>
    ALWAYS_INLINE void result(
        size_t /* j */,
        size_t row,
        size_t ind,
        const FindResult & find_result,
        KnownRows & known_rows,
        bool is_last_disjunct)
    {
        processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsTemplate>(
            find_result, added_columns, used_flags, row, ind, current_offset, known_rows, is_last_disjunct);
    }

    ALWAYS_INLINE void finish(size_t /* row */)
    {
        if constexpr (Features::need_replication)
            added_columns.offsets_to_replicate.push_back(current_offset);
    }
};

/// Adapter so `addFoundRowAll` can append into the pre-select buffer without emitting.
struct PreSelectedRows
{
    explicit PreSelectedRows(PODArray<UInt64> & container_) : container(container_) { }
    void appendFromBlock(UInt64 ref_word, bool /* has_default */) { container.push_back(ref_word); }
    static constexpr bool isLazy() { return false; }

    PODArray<UInt64> & container;
};

/** Sink for the additional-filter path (Stage 5): record matching right refs for a later
  * filter pass. MapsAll-only. Stays free of KeyGetter/Map so Stage 3's Mapped decoupling is
  * not undone. `need_filter` / `flag_per_row` remain runtime bools at the call site; the
  * `KnownRowsHolder` type is fixed by the `SequentialMultiLookup` instantiation.
  *
  * C13: the driver stops when `current_added_rows >= max_joined_rows` and returns `i`.
  * C16: `finish` push_backs into `row_replicate_offset`; final `offsets_to_replicate` is
  * still pre-sized then `resize(left_block_rows)` after the filter pass.
  */
template <typename FindResult>
struct PreSelectSink
{
    PODArray<UInt64> & selected_rows;
    std::vector<FindResult> & find_results;
    IColumn::Offset & current_added_rows;
    IColumn::Offsets & row_replicate_offset;

    ALWAYS_INLINE void miss(size_t /* j */, size_t /* row */) { }

    template <typename FindResultArg, typename KnownRows>
    ALWAYS_INLINE void result(
        size_t /* j */,
        size_t /* row */,
        size_t /* ind */,
        const FindResultArg & find_result,
        KnownRows & known_rows,
        bool is_last_disjunct)
    {
        using Mapped = std::remove_reference_t<decltype(std::declval<FindResultArg &>().getMapped())>;
        auto & mapped = find_result.getMapped();
        find_results.push_back(find_result);
        PreSelectedRows view{selected_rows};
        /// We don't add missing in addFoundRowAll here; missing rows are added after the
        /// additional filter is applied (different from `joinRightColumns`).
        if constexpr (std::is_same_v<KnownRows, KnownRowsHolder<true>>)
            addFoundRowAll<Mapped, false, true>(
                mapped, view, current_added_rows, known_rows, nullptr, is_last_disjunct);
        else
            addFoundRowAll<Mapped, false, false>(
                mapped, view, current_added_rows, known_rows, nullptr, is_last_disjunct);
    }

    ALWAYS_INLINE void finish(size_t /* row */) { row_replicate_offset.push_back(current_added_rows); }
};

/** Phase 2 of the two-phase probe: the fused loop's body with the lookup replaced by the
  * outcome phase 1 recorded for the row. It reads only `ProbeOutcomes`, never the hash table
  * and never the skip bytes (phase 1 records a skipped row as a miss, which is what the fused
  * loop does with it anyway), so it is the same code whichever backend filled the batch.
  * `current_offset` is carried across batches by the caller.
  *
  * Parameterised on MapsTemplate's Mapped type (not Map/KeyGetter), so the emit is not
  * multiplied by the 32 key types. NO_INLINE so clang cannot re-inline it into every
  * per-KeyGetter probeTwoPhase instantiation (P7). Non-ASOF drops Selector (C9: processMatch
  * reads ind only in the ASOF branch); the ASOF overload below keeps it.
  */
template <
    JoinKind KIND,
    JoinStrictness STRICTNESS,
    bool need_filter,
    typename MapsTemplate,
    typename AddedColumns>
NO_INLINE void consumeProbeBatch(
    const ProbeOutcomes & outcomes,
    AddedColumns & added_columns,
    JoinStuff::JoinUsedFlags & used_flags,
    size_t begin,
    size_t count,
    IColumn::Offset & current_offset)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;
    static_assert(!join_features.is_asof_join);
    static constexpr bool flag_per_row = false; /// the two-phase probe is single-map only

    using Mapped = const typename MapsTemplate::MappedType;
    using MappedValue = typename MapsTemplate::MappedType;
    using FindResult = ColumnsHashing::columns_hashing_impl::FindResultImpl<Mapped, join_features.need_flags>;

    const UInt64 * const found = outcomes.found;
    const UInt64 * const offsets [[maybe_unused]] = join_features.need_flags ? outcomes.offset.data() : nullptr;

    for (size_t j = 0; j < count; ++j)
    {
        const size_t i = begin + j;

        bool right_row_found = false;
        KnownRowsHolder<flag_per_row> dummy_known_rows;

        if (const UInt64 word = found[j])
        {
            right_row_found = true;

            size_t offset = 0;
            if constexpr (join_features.need_flags)
                offset = offsets[j];

            if constexpr (probe_mapped_fits_word<MappedValue>)
            {
                /// The mapped value phase 1 copied out of the cell, rebuilt on the stack: the
                /// cell itself is never dereferenced again.
                auto mapped_value = mappedFromWord<MappedValue>(word);
                FindResult find_result(&mapped_value, true, offset);
                processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsTemplate>(
                    find_result, added_columns, used_flags, i, /*ind=*/0, current_offset, dummy_known_rows, /*is_last_disjunct=*/true);
            }
            else
            {
                FindResult find_result(reinterpret_cast<Mapped *>(word), true, offset); /// NOLINT(performance-no-int-to-ptr)
                processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsTemplate>(
                    find_result, added_columns, used_flags, i, /*ind=*/0, current_offset, dummy_known_rows, /*is_last_disjunct=*/true);
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

/// ASOF phase-2 consume: needs the selector so processMatch can read the probe row's asof key.
template <
    JoinKind KIND,
    JoinStrictness STRICTNESS,
    bool need_filter,
    typename MapsTemplate,
    typename AddedColumns,
    typename Selector>
NO_INLINE void consumeProbeBatch(
    const ProbeOutcomes & outcomes,
    AddedColumns & added_columns,
    JoinStuff::JoinUsedFlags & used_flags,
    const Selector & selector,
    size_t begin,
    size_t count,
    IColumn::Offset & current_offset)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;
    static_assert(join_features.is_asof_join);
    static constexpr bool flag_per_row = false;

    using Mapped = const typename MapsTemplate::MappedType;
    using MappedValue = typename MapsTemplate::MappedType;
    using FindResult = ColumnsHashing::columns_hashing_impl::FindResultImpl<Mapped, join_features.need_flags>;

    const UInt64 * const found = outcomes.found;
    const UInt64 * const offsets [[maybe_unused]] = join_features.need_flags ? outcomes.offset.data() : nullptr;

    for (size_t j = 0; j < count; ++j)
    {
        const size_t i = begin + j;
        const size_t ind = selectorIndexAt(selector, i);

        bool right_row_found = false;
        KnownRowsHolder<flag_per_row> dummy_known_rows;

        if (const UInt64 word = found[j])
        {
            right_row_found = true;

            size_t offset = 0;
            if constexpr (join_features.need_flags)
                offset = offsets[j];

            if constexpr (probe_mapped_fits_word<MappedValue>)
            {
                auto mapped_value = mappedFromWord<MappedValue>(word);
                FindResult find_result(&mapped_value, true, offset);
                processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsTemplate>(
                    find_result, added_columns, used_flags, i, ind, current_offset, dummy_known_rows, /*is_last_disjunct=*/true);
            }
            else
            {
                FindResult find_result(reinterpret_cast<Mapped *>(word), true, offset); /// NOLINT(performance-no-int-to-ptr)
                processMatch<KIND, STRICTNESS, need_filter, flag_per_row, MapsTemplate>(
                    find_result, added_columns, used_flags, i, ind, current_offset, dummy_known_rows, /*is_last_disjunct=*/true);
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

/** Whether phase 1 can write its outcomes straight into the join's output array.
  *
  * It can when the output is one word per probe row and that word is the outcome. A lazy ALL
  * join that adds missing rows (LEFT, FULL) is exactly that case: a match appends the cell's
  * word through `addRef`, and a miss appends zero through `addDefault` - which is the same
  * zero phase 1 records for a miss. So `LazyOutput::row_refs` and the outcome buffer would
  * hold the identical sequence, and phase 2's only remaining job is the bookkeeping that
  * `processMatch` and `addNotFoundRow` do around the append.
  *
  * It cannot when the correspondence is not one word per row: INNER and RIGHT drop misses
  * instead of defaulting them, ANY appends only behind a `setUsedOnce` gate, SEMI and ANTI
  * append nothing at all, and ASOF emits the matched row's ref rather than the outcome. The
  * non-lazy `AddedColumns` materializes columns inside `appendFromBlock`, so there is no
  * array to share.
  */
template <typename AddedColumns, typename JoinFeaturesT>
constexpr bool outputIsProbeOutcomes(const JoinFeaturesT & join_features)
{
    return AddedColumns::isLazy() && join_features.is_all_join && join_features.add_missing
        && join_features.need_replication && !join_features.is_asof_join;
}

/** Phase 2 for the case above: the words are already in place, so nothing is appended.
  *
  * This deliberately does not go through `processMatch`. Everything it would do here reduces
  * to a prefix sum over `refWordRows` plus the used flags, and routing it through the general
  * emit path would mean writing every word a second time, which is the whole point of the
  * fusion. The two paths are held to the same answers by the harness's cross-arm agreement
  * check rather than by sharing code.
  */
template <
    JoinKind KIND,
    JoinStrictness STRICTNESS,
    bool need_filter,
    typename MapsTemplate,
    typename AddedColumns>
NO_INLINE void consumeFusedBatch(
    const ProbeOutcomes & outcomes,
    AddedColumns & added_columns,
    JoinStuff::JoinUsedFlags & used_flags,
    size_t begin,
    size_t count,
    IColumn::Offset & current_offset)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;
    static_assert(outputIsProbeOutcomes<AddedColumns>(join_features));

    const UInt64 * const found = outcomes.found;
    const UInt64 * const offsets [[maybe_unused]] = join_features.need_flags ? outcomes.offset.data() : nullptr;

    size_t rows_added = 0;
    for (size_t j = 0; j < count; ++j)
    {
        const size_t i = begin + j;
        const UInt64 word = found[j];

        /// A zero word is the default row `addDefault` would have appended: one output row.
        UInt32 rows_of_key = 1;
        if (word)
        {
            setUsed<need_filter>(added_columns.filter, i, added_columns.matched_rows);
            /// `flag_per_row` is false throughout the single-map probe, so this reads only the
            /// offset; the block and row arguments are unused on that path.
            if constexpr (join_features.need_flags)
                used_flags.template setUsed<true, false>(0, 0, offsets[j]);
            rows_of_key = refWordRows(word);
        }

        rows_added += rows_of_key;
        current_offset += rows_of_key;
        added_columns.offsets_to_replicate[i] = current_offset;
    }

    /// `addRef`/`addDefault` maintain this per append; here it is one update per batch.
    added_columns.lazy_output.row_count += rows_added;
}

/** One batch of the single-clause probe's phase 1: fill `outcomes` for rows
  * `[begin, begin + count)` through the sequential driver. Keyed only on the types the
  * lookup body needs - (Map, KeyGetter, Selector), the TU-constant `need_flags`, and a
  * prefetch type that no longer carries `need_filter` (`ProbePrefetch`) - so it
  * instantiates 64 bodies per TU instead of 128. NO_INLINE (P2): the batch boundary is an
  * outlined call on purpose, and `runImpl` below it stays NO_INLINE too (P1).
  */
template <bool need_flags, typename Map, typename KeyGetter, typename Selector, typename PrefetchAt>
NO_INLINE void lookupBatch(
    KeyGetter & key_getter,
    const Map & map,
    const Selector & selector,
    const UInt8 * skip_data,
    Arena & pool,
    size_t begin,
    size_t count,
    PrefetchAt && prefetch_at,
    ProbeOutcomes & outcomes)
{
    RecordOutcomeSink<need_flags> sink{outcomes};
    SequentialLookup::run(
        key_getter, map, selector, skip_data, pool, begin, count,
        std::forward<PrefetchAt>(prefetch_at), sink);
}

/** The two-phase probe over one block: `lookupBatch` fills the outcomes of a batch of
  * consecutive rows, phase 2 consumes them, and the scratch is reused by the next batch.
  * Keeps `need_filter` only because the consume needs it; both callees are NO_INLINE, so
  * the shell stays small.
  */
template <
    JoinKind KIND,
    JoinStrictness STRICTNESS,
    bool need_filter,
    typename MapsTemplate,
    typename Map,
    typename KeyGetter,
    typename AddedColumns,
    typename Selector,
    typename PrefetchAt>
void probeTwoPhase(
    KeyGetter & key_getter,
    const Map & map,
    AddedColumns & added_columns,
    JoinStuff::JoinUsedFlags & used_flags,
    const Selector & selector,
    const UInt8 * skip_data,
    Arena & pool,
    PrefetchAt && prefetch_at,
    ProbeOutcomes & outcomes,
    size_t rows,
    size_t batch_rows,
    IColumn::Offset & current_offset)
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;
    const size_t scratch_rows = std::min(rows, batch_rows);

    if constexpr (outputIsProbeOutcomes<AddedColumns>(join_features))
    {
        /// `has_columns_to_add` is what decides whether `appendFromBlock` appends at all; with
        /// nothing to add there is no output array to write into and no `row_count` to keep.
        if (added_columns.has_columns_to_add)
        {
            auto & row_refs = added_columns.lazy_output.row_refs;
            const size_t base = row_refs.size();
            /// Sized once for the whole block, so no batch can reallocate it under the
            /// pointer phase 1 holds. Every position is written by phase 1, which is why the
            /// uninitialized resize of a POD array is safe here.
            row_refs.resize(base + rows);
            outcomes.useExternal(row_refs.data() + base, scratch_rows, join_features.need_flags);

            for (size_t begin = 0; begin < rows; begin += batch_rows)
            {
                const size_t count = std::min(batch_rows, rows - begin);
                outcomes.found = row_refs.data() + base + begin;
                lookupBatch<join_features.need_flags>(
                    key_getter, map, selector, skip_data, pool, begin, count, prefetch_at, outcomes);
                consumeFusedBatch<KIND, STRICTNESS, need_filter, MapsTemplate>(
                    outcomes, added_columns, used_flags, begin, count, current_offset);
            }
            return;
        }
    }

    outcomes.useScratch(scratch_rows, join_features.need_flags);

    for (size_t begin = 0; begin < rows; begin += batch_rows)
    {
        const size_t count = std::min(batch_rows, rows - begin);
        lookupBatch<join_features.need_flags>(
            key_getter, map, selector, skip_data, pool, begin, count, prefetch_at, outcomes);
        if constexpr (join_features.is_asof_join)
        {
            consumeProbeBatch<KIND, STRICTNESS, need_filter, MapsTemplate>(
                outcomes, added_columns, used_flags, selector, begin, count, current_offset);
        }
        else
        {
            consumeProbeBatch<KIND, STRICTNESS, need_filter, MapsTemplate>(
                outcomes, added_columns, used_flags, begin, count, current_offset);
        }
    }
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <
    typename KeyGetter,
    typename Map,
    bool need_filter,
    typename AddedColumns,
    typename Selector>
size_t HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::joinRightColumns(
    KeyGetter & key_getter, const Map * map, AddedColumns & added_columns, JoinStuff::JoinUsedFlags & used_flags, const Selector & selector)
{
    static constexpr bool flag_per_row = false; // Always false in single map case
    const auto & join_keys = added_columns.join_on_keys.at(0);

    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;

    size_t rows = ScatteredBlock::Selector::size(selector);

    /// The skip pointer is a local so that it can stay in a register across the calls in
    /// the loop body (see `JoinOnKeyColumns::buildRowSkipData`). nullptr means the fast path;
    /// SequentialLookup folds that into two inner loops (P4).
    const UInt8 * skip_data = nullptr;
    IColumn::Filter skip_buffer;
    if (join_keys.null_map || join_keys.join_mask_column.getKind() != JoinCommon::JoinMask::Kind::AllTrue)
    {
        if constexpr (std::is_same_v<std::decay_t<Selector>, ScatteredBlock::Indexes>)
            skip_data = join_keys.buildRowSkipData(skip_buffer, selector);
        else
            skip_data = join_keys.buildRowSkipData(skip_buffer, selector.first, rows);
    }
    if constexpr (need_filter)
    {
        added_columns.filter = IColumn::Filter(rows, 0);
        added_columns.matched_rows.reserve(rows);
    }
    if constexpr (!flag_per_row && (STRICTNESS == JoinStrictness::All || (STRICTNESS == JoinStrictness::Semi && KIND == JoinKind::Right)))
        added_columns.lazy_output.output_by_row_list = true;

    Arena pool;

    if constexpr (join_features.need_replication)
        added_columns.offsets_to_replicate = IColumn::Offsets(rows);

    /// Software prefetch during the probe phase. One instance for the whole probe call: the
    /// look-ahead calibration fires once at an absolute row, so a per-batch instance would
    /// never calibrate past the first batch (F11).
    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;

    bool use_prefetch = false;
    if constexpr (can_prefetch)
        use_prefetch = shouldUseJoinPrefetch(added_columns.enable_prefetch, map);

    ProbePrefetch<Map, KeyGetter, Selector> prefetch_at{
        map, &key_getter, &selector, &pool, use_prefetch, rows};

    /// A second pass can only pay for itself when the emit it defers is more than a flag or a
    /// filter bit. SEMI LEFT and ANTI have nothing else per row, so recording an outcome to
    /// hand over is pure overhead - they were the two worst kinds in the matrix, at +1.0% and
    /// +1.6%, before this exclusion. Compile-time so `(SEMI && left) || ANTI` TUs never
    /// instantiate `probeTwoPhase` and the other kinds never instantiate `EmitSink` in this path.
    constexpr bool split_can_pay = !((join_features.is_semi_join && join_features.left) || join_features.is_anti_join);

    IColumn::Offset current_offset = 0;
    if constexpr (split_can_pay)
    {
        ProbeOutcomes outcomes;
        probeTwoPhase<KIND, STRICTNESS, need_filter, MapsTemplate, Map, KeyGetter>(
            key_getter, *map, added_columns, used_flags, selector, skip_data, pool, prefetch_at, outcomes, rows, PROBE_BATCH_ROWS, current_offset);
    }
    else
    {
        EmitSink<KIND, STRICTNESS, need_filter, MapsTemplate, AddedColumns> sink{
            added_columns, used_flags, current_offset};
        SequentialLookup::run(
            key_getter, *map, selector, skip_data, pool, /*begin=*/0, rows, prefetch_at, sink);
    }

    added_columns.applyLazyDefaults();
    return 0;
}

/// Joins right table columns which indexes are present in right_indexes using specified map.
/// Makes filter (1 if row presented in right table) and returns offsets to replicate (for ALL JOINS).
template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <
    typename KeyGetter,
    typename Map,
    bool need_filter,
    typename AddedColumns,
    typename Selector>
size_t HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::joinRightColumns(
    std::vector<KeyGetter> && key_getter_vector,
    const std::vector<const Map *> & mapv,
    AddedColumns & added_columns,
    JoinStuff::JoinUsedFlags & used_flags,
    const Selector & selector)
{
    static constexpr bool flag_per_row = true; // Always true in multiple maps case

    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;

    size_t rows = ScatteredBlock::Selector::size(selector);

    /// Per-clause skip bytes, prepared once per call (see `JoinOnKeyColumns::buildRowSkipData`).
    /// Empty `skip_datas` means the fast path; SequentialMultiLookup folds that into two inner
    /// loops (P4). A skipped clause is not a missed row (C15).
    /// Clause count matches `key_getter_vector` / `join_on_keys` (not `mapv`): empty ON keys
    /// still enter this overload with an empty getter vector.
    chassert(key_getter_vector.size() == added_columns.join_on_keys.size());
    chassert(key_getter_vector.size() == mapv.size() || key_getter_vector.empty());
    std::vector<const UInt8 *> skip_datas;
    std::vector<IColumn::Filter> skip_buffers;
    const size_t num_clauses = key_getter_vector.size();
    bool any_skip = false;
    for (size_t d = 0; d < num_clauses; ++d)
    {
        const auto & keys = added_columns.join_on_keys[d];
        if (keys.null_map || keys.join_mask_column.getKind() != JoinCommon::JoinMask::Kind::AllTrue)
        {
            any_skip = true;
            break;
        }
    }
    if (any_skip)
    {
        skip_datas.resize(num_clauses);
        skip_buffers.resize(num_clauses);
        for (size_t d = 0; d < num_clauses; ++d)
        {
            const auto & keys = added_columns.join_on_keys[d];
            if (keys.null_map || keys.join_mask_column.getKind() != JoinCommon::JoinMask::Kind::AllTrue)
            {
                if constexpr (std::is_same_v<std::decay_t<Selector>, ScatteredBlock::Indexes>)
                    skip_datas[d] = keys.buildRowSkipData(skip_buffers[d], selector);
                else
                    skip_datas[d] = keys.buildRowSkipData(skip_buffers[d], selector.first, rows);
            }
            else
            {
                skip_datas[d] = nullptr;
            }
        }
    }
    if constexpr (need_filter)
    {
        added_columns.filter = IColumn::Filter(rows, 0);
        added_columns.matched_rows.reserve(rows);
    }
    /// C17: output_by_row_list is true iff !flag_per_row && (All || (Semi && Right)).
    /// Multi-clause has flag_per_row=true, so this is always false here - kept for clarity.
    if constexpr (!flag_per_row && (STRICTNESS == JoinStrictness::All || (STRICTNESS == JoinStrictness::Semi && KIND == JoinKind::Right)))
        added_columns.lazy_output.output_by_row_list = true;

    Arena pool;

    if constexpr (join_features.need_replication)
    {
        added_columns.offsets_to_replicate.clear();
        added_columns.offsets_to_replicate.reserve(rows);
    }

    /// Software prefetch for multi-map variant. Only prefetch the first map when there is a
    /// clause to look up (empty join_on_keys leaves key_getter_vector empty). One instance
    /// for the whole probe call, as in the single-clause probe (F11).
    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;

    bool use_prefetch = false;
    if constexpr (can_prefetch)
        use_prefetch = !key_getter_vector.empty() && !mapv.empty()
            && shouldUseJoinPrefetch(added_columns.enable_prefetch, mapv[0]);

    ProbePrefetch<Map, KeyGetter, Selector> prefetch_at{
        mapv.empty() ? nullptr : mapv[0],
        key_getter_vector.empty() ? nullptr : &key_getter_vector[0],
        &selector,
        &pool,
        use_prefetch,
        rows};

    size_t max_joined_rows
        = added_columns.max_joined_block_rows > 0 ? added_columns.max_joined_block_rows : std::numeric_limits<size_t>::max();

    IColumn::Offset current_offset = 0;
    constexpr bool stop_after_first_match = join_features.is_any_or_semi_join
        && !(join_features.is_any_join && (join_features.right || join_features.full));

    MultiEmitSink<KIND, STRICTNESS, need_filter, MapsTemplate, AddedColumns> sink{
        added_columns, used_flags, current_offset};

    /// C13: driver stops at a row boundary when current_offset >= max_joined_rows and returns i.
    /// C15: driver owns clause iteration + KnownRowsHolder; sink stays free of KeyGetter/Map.
    /// C16: MultiEmitSink::finish push_backs offsets_to_replicate so early exit truncates.
    const size_t i = SequentialMultiLookup::run<stop_after_first_match, KnownRowsHolder<flag_per_row>>(
        key_getter_vector,
        mapv,
        selector,
        skip_datas,
        pool,
        rows,
        max_joined_rows,
        current_offset,
        prefetch_at,
        sink);

    added_columns.applyLazyDefaults();
    return i;
}

template <typename AddedColumns, typename Selector>
static ColumnPtr buildAdditionalFilter(
    const Selector & selector,
    const PODArray<UInt64> & selected_rows,
    const IColumn::Offsets & row_replicate_offset,
    const AddedColumns & added_columns)
{
    ColumnPtr result_column;
    do
    {
        if (selected_rows.empty())
        {
            result_column = ColumnUInt8::create();
            break;
        }

        if (!added_columns.additional_filter_expression)
        {
            auto filter = ColumnUInt8::create();
            filter->insertMany(1, selected_rows.size());
            result_column = std::move(filter);
            break;
        }

        auto required_cols = added_columns.additional_filter_expression->getRequiredColumnsWithTypes();
        if (required_cols.empty())
        {
            Block block;
            added_columns.additional_filter_expression->execute(block);
            result_column = block.getByPosition(0).column->cloneResized(selected_rows.size());
            break;
        }

        ColumnsWithTypeAndName required_columns;
        required_columns.reserve(required_cols.size());
        auto rhs_pos_it = added_columns.additional_filter_required_rhs_pos.begin();
        auto req_cols_it = required_cols.begin();
        for (size_t pos = 0; pos < required_cols.size(); ++pos, ++req_cols_it)
        {
            if (rhs_pos_it != added_columns.additional_filter_required_rhs_pos.end() && pos == rhs_pos_it->first)
            {
                const auto & req_col = *req_cols_it;
                required_columns.emplace_back(nullptr, req_col.type, req_col.name);

                auto col = req_col.type->createColumn();
                for (const UInt64 selected_row : selected_rows)
                {
                    const auto * block = added_columns.lazy_output.stored_columns[refWordBlockNo(selected_row)];
                    const auto [src_col, row_pos] = getBlockColumnAndRow(block, refWordRowNo(selected_row), rhs_pos_it->second);
                    col->insertFrom(*src_col, row_pos);
                }
                required_columns[pos].column = std::move(col);
                ++rhs_pos_it;
            }
            else
            {
                const auto & col_name = req_cols_it->name;
                const auto * src_col = added_columns.left_block.findByName(col_name);
                if (!src_col)
                    throw Exception(
                        ErrorCodes::LOGICAL_ERROR,
                        "required columns: [{}], but not found any in left table. left table: {}, required column: {}",
                        required_cols.toString(),
                        added_columns.left_block.dumpNames(),
                        col_name);

                auto new_col = src_col->column->cloneEmpty();
                for (size_t i = 0; i < row_replicate_offset.size(); ++i)
                {
                    size_t rows = row_replicate_offset[i] - row_replicate_offset[i - 1];
                    if (rows)
                    {
                        new_col->insertManyFrom(*src_col->column, selectorIndexAt(selector, i), rows);
                    }
                }
                required_columns.push_back({std::move(new_col), src_col->type, col_name});
            }
        }

        Block executed_block(std::move(required_columns));

        for (const auto & col : executed_block.getColumnsWithTypeAndName())
            if (!col.column || !col.type)
                throw Exception(ErrorCodes::LOGICAL_ERROR, "Illegal nullptr column in input block: {}", executed_block.dumpStructure());

        added_columns.additional_filter_expression->execute(executed_block);
        result_column = executed_block.getByPosition(0).column->convertToFullColumnIfConst();
        executed_block.clear();
    } while (false);

    result_column = result_column->convertToFullIfWrapped()->convertToFullColumnIfLowCardinality();
    if (result_column->isNullable())
    {
        /// Convert Nullable(UInt8) to UInt8 ensuring that nulls are zeros
        /// Trying to avoid copying data, since we are the only owner of the column.
        ColumnPtr mask_column = assert_cast<const ColumnNullable &>(*result_column).getNullMapColumnPtr();

        MutableColumnPtr mutable_column;
        {
            ColumnPtr nested_column = assert_cast<const ColumnNullable &>(*result_column).getNestedColumnPtr();
            result_column.reset();
            mutable_column = IColumn::mutate(std::move(nested_column));
        }

        auto & column_data = assert_cast<ColumnUInt8 &>(*mutable_column).getData();
        const auto & mask_column_data = assert_cast<const ColumnUInt8 &>(*mask_column).getData();
        for (size_t i = 0; i < column_data.size(); ++i)
        {
            if (mask_column_data[i])
                column_data[i] = 0;
        }
        return mutable_column;
    }
    return result_column;
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
template <typename KeyGetter, typename Map, typename AddedColumns>
size_t HashJoinMethods<KIND, STRICTNESS, MapsTemplate>::joinRightColumnsWithAdditionalFilter(
    std::vector<KeyGetter> && key_getter_vector,
    const std::vector<const Map *> & mapv,
    AddedColumns & added_columns,
    JoinStuff::JoinUsedFlags & used_flags [[maybe_unused]],
    const ScatteredBlock::Selector & selector,
    bool need_filter [[maybe_unused]],
    bool flag_per_row [[maybe_unused]])
{
    constexpr JoinFeatures<KIND, STRICTNESS, MapsTemplate> join_features;
    size_t left_block_rows = selector.size();
    if (need_filter)
    {
        added_columns.filter = IColumn::Filter(left_block_rows, 0);
        added_columns.matched_rows.reserve(left_block_rows);
    }

    /// C16: pre-size offsets_to_replicate to the selector size, then resize(left_block_rows)
    /// after the filter pass (early exit may leave fewer processed left rows).
    if constexpr (join_features.need_replication)
        added_columns.offsets_to_replicate = IColumn::Offsets(left_block_rows);

    using FindResult = KeyGetter::FindResult;

    PODArray<UInt64> selected_rows;
    selected_rows.reserve(left_block_rows);
    std::vector<FindResult> find_results;
    find_results.reserve(left_block_rows);
    IColumn::Offset total_added_rows = 0;

    IColumn::Offsets row_replicate_offset;
    row_replicate_offset.reserve(left_block_rows);

    size_t max_joined_rows = added_columns.max_joined_block_rows;
    if (max_joined_rows == 0)
        max_joined_rows = std::numeric_limits<size_t>::max();

    Arena pool;
    IColumn::Offset current_added_rows = 0;

    /// Per-clause skip bytes, same prep as the multi-clause probe. Empty `skip_datas` means
    /// the fast path; SequentialMultiLookup folds that into two inner loops (P4).
    chassert(key_getter_vector.size() == added_columns.join_on_keys.size());
    chassert(key_getter_vector.size() == mapv.size() || key_getter_vector.empty());
    chassert(!mapv.empty());
    std::vector<const UInt8 *> skip_datas;
    std::vector<IColumn::Filter> skip_buffers;
    const size_t num_clauses = key_getter_vector.size();
    bool any_skip = false;
    for (size_t d = 0; d < num_clauses; ++d)
    {
        const auto & keys = added_columns.join_on_keys[d];
        if (keys.null_map || keys.join_mask_column.getKind() != JoinCommon::JoinMask::Kind::AllTrue)
        {
            any_skip = true;
            break;
        }
    }

    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;

    /// Dispatch Range / Indexes so SequentialMultiLookup can reuse selectorIndexAt without
    /// holding the variant Selector (same split as the non-filter multi-clause probe).
    auto run_preselect = [&]<typename Sel>(const Sel & sel) -> size_t
    {
        const size_t rows = ScatteredBlock::Selector::size(sel);
        if (any_skip)
        {
            skip_datas.assign(num_clauses, nullptr);
            skip_buffers.resize(num_clauses);
            for (size_t d = 0; d < num_clauses; ++d)
            {
                const auto & keys = added_columns.join_on_keys[d];
                if (keys.null_map || keys.join_mask_column.getKind() != JoinCommon::JoinMask::Kind::AllTrue)
                {
                    if constexpr (std::is_same_v<std::decay_t<Sel>, ScatteredBlock::Indexes>)
                        skip_datas[d] = keys.buildRowSkipData(skip_buffers[d], sel);
                    else
                        skip_datas[d] = keys.buildRowSkipData(skip_buffers[d], sel.first, rows);
                }
            }
        }
        else
        {
            skip_datas.clear();
        }

        bool use_prefetch = false;
        if constexpr (can_prefetch)
            use_prefetch = !key_getter_vector.empty()
                && shouldUseJoinPrefetch(added_columns.enable_prefetch, mapv[0]);

        auto prefetcher = makeJoinPrefetcher(
            use_prefetch,
            rows,
            [&](size_t k) __attribute__((always_inline))
            {
                if constexpr (can_prefetch)
                    mapv[0]->prefetch(key_getter_vector[0].getKeyHolder(selectorIndexAt(sel, k), pool));
            });

        auto prefetch_at = [&](size_t k) __attribute__((always_inline))
        {
            if constexpr (can_prefetch)
                prefetcher.prefetchAt(k);
        };

        PreSelectSink<FindResult> sink{selected_rows, find_results, current_added_rows, row_replicate_offset};

        /// Do not stop after the first matching clause: the filter pass still needs every
        /// pre-selected right ref (SEMI/ANY first-match is applied after filtering).
        /// flag_per_row stays a runtime bool; KnownRowsHolder is picked per branch.
        if (flag_per_row)
        {
            return SequentialMultiLookup::run</*stop_after_first_match=*/false, KnownRowsHolder<true>>(
                key_getter_vector,
                mapv,
                sel,
                skip_datas,
                pool,
                rows,
                max_joined_rows,
                current_added_rows,
                prefetch_at,
                sink);
        }
        return SequentialMultiLookup::run</*stop_after_first_match=*/false, KnownRowsHolder<false>>(
            key_getter_vector,
            mapv,
            sel,
            skip_datas,
            pool,
            rows,
            max_joined_rows,
            current_added_rows,
            prefetch_at,
            sink);
    };

    /// C13: driver stops at a row boundary when current_added_rows >= max_joined_rows and
    /// returns rows consumed (`i`).
    const size_t processed_rows = selector.isContinuousRange() ? run_preselect(selector.getRange())
                                                               : run_preselect(selector.getIndexes());

    if (selected_rows.size() != current_added_rows)
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "Sizes are mismatched. selected_rows.size:{}, current_added_rows:{}, row_replicate_offset.size:{}",
            selected_rows.size(),
            current_added_rows,
            row_replicate_offset.size());

    left_block_rows = processed_rows;
    chassert(left_block_rows == row_replicate_offset.size());

    {
        auto filter_col = buildAdditionalFilter(selector, selected_rows, row_replicate_offset, added_columns);

        const PaddedPODArray<UInt8> & filter_flags = assert_cast<const ColumnUInt8 &>(*filter_col).getData();

        size_t prev_replicated_row = 0;
        auto * selected_right_row_it = selected_rows.begin();
        size_t find_result_index = 0;
        for (size_t i = 0, n = row_replicate_offset.size(); i < n; ++i)
        {
            bool any_matched = false;
            /// right/full join or multiple disjuncts, we need to mark used flags for each row.
            if (flag_per_row)
            {
                for (size_t replicated_row = prev_replicated_row; replicated_row < row_replicate_offset[i]; ++replicated_row)
                {
                    if (filter_flags[replicated_row])
                    {
                        const UInt64 selected_ref = *selected_right_row_it;
                        if constexpr (join_features.is_semi_join || join_features.is_any_join)
                        {
                            /// For LEFT/INNER SEMI/ANY JOIN, we need to add only first appeared row from left,
                            if constexpr (join_features.left || join_features.inner)
                            {
                                if (!any_matched)
                                {
                                    // For inner join, we need mark each right row'flag, because we only use each right row once.
                                    auto used_once = used_flags.template setUsedOnce<join_features.need_flags, true>(
                                        refWordBlockNo(selected_ref), refWordRowNo(selected_ref), 0);
                                    if (used_once)
                                    {
                                        any_matched = true;
                                        total_added_rows += 1;
                                        added_columns.appendFromBlock(selected_ref, join_features.add_missing);
                                    }
                                }
                            }
                            else
                            {
                                auto used_once = used_flags.template setUsedOnce<join_features.need_flags, true>(
                                    refWordBlockNo(selected_ref), refWordRowNo(selected_ref), 0);
                                if (used_once)
                                {
                                    any_matched = true;
                                    total_added_rows += 1;
                                    added_columns.appendFromBlock(selected_ref, join_features.add_missing);
                                }
                            }
                        }
                        else if constexpr (join_features.is_anti_join)
                        {
                            any_matched = true;
                            if constexpr (join_features.right && join_features.need_flags)
                                used_flags.template setUsed<true, true>(refWordBlockNo(selected_ref), refWordRowNo(selected_ref), 0);
                        }
                        else
                        {
                            any_matched = true;
                            total_added_rows += 1;
                            added_columns.appendFromBlock(selected_ref, join_features.add_missing);
                            used_flags.template setUsed<join_features.need_flags, true>(refWordBlockNo(selected_ref), refWordRowNo(selected_ref), 0);
                        }
                    }

                    ++selected_right_row_it;
                }
            }
            else
            {
                for (size_t replicated_row = prev_replicated_row; replicated_row < row_replicate_offset[i]; ++replicated_row)
                {
                    if constexpr (join_features.is_anti_join)
                    {
                        any_matched |= filter_flags[replicated_row];
                    }
                    else if constexpr (join_features.need_replication)
                    {
                        if (filter_flags[replicated_row])
                        {
                            any_matched = true;
                            added_columns.appendFromBlock(*selected_right_row_it, join_features.add_missing);
                            total_added_rows += 1;
                        }
                        ++selected_right_row_it;
                    }
                    else
                    {
                        if (filter_flags[replicated_row])
                        {
                            any_matched = true;
                            added_columns.appendFromBlock(*selected_right_row_it, join_features.add_missing);
                            total_added_rows += 1;
                            selected_right_row_it = selected_right_row_it + row_replicate_offset[i] - replicated_row;
                            break;
                        }
                        ++selected_right_row_it;
                    }
                }
            }


            if constexpr (join_features.is_anti_join)
            {
                if (!any_matched)
                {
                    if constexpr (join_features.left)
                        if (need_filter)
                            setUsed<true>(added_columns.filter, i, added_columns.matched_rows);
                    addNotFoundRow<join_features.add_missing, join_features.need_replication>(added_columns, total_added_rows);
                }
            }
            else
            {
                if (!any_matched)
                {
                    addNotFoundRow<join_features.add_missing, join_features.need_replication>(added_columns, total_added_rows);
                }
                else
                {
                    if (!flag_per_row)
                        used_flags.template setUsed<join_features.need_flags, false>(find_results[find_result_index]);
                    if (need_filter)
                        setUsed<true>(added_columns.filter, i, added_columns.matched_rows);
                    if constexpr (join_features.add_missing)
                        added_columns.applyLazyDefaults();
                }
            }
            find_result_index += (prev_replicated_row != row_replicate_offset[i]);

            if constexpr (join_features.need_replication)
            {
                added_columns.offsets_to_replicate[i] = total_added_rows;
            }
            prev_replicated_row = row_replicate_offset[i];
        }
    }

    if constexpr (join_features.need_replication)
    {
        added_columns.offsets_to_replicate.resize(left_block_rows);
        added_columns.filter.resize(left_block_rows);
    }
    else if (need_filter)
    {
        /// The loop above may break early at max_joined_block_rows, producing fewer left rows
        /// than the selector size the filter was allocated for. Trim the filter to the number of
        /// processed rows so the required right key column built from it matches the left block,
        /// which is cut to left_block_rows downstream.
        added_columns.filter.resize(left_block_rows);
    }
    added_columns.applyLazyDefaults();
    return left_block_rows;
}

}

}
