#include <Interpreters/HashJoin/AddedColumns.h>
#include <Interpreters/HashJoin/HashJoinMethodsImpl.h>
#include <Interpreters/HashJoin/HashJoinResult.h>
#include <Interpreters/HashJoin/JoinUsedFlags.h>
#include <Interpreters/HashJoin/KeyGetter.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/PartitionedHashJoin/JoinRouteHashing.h>
#include <Interpreters/PartitionedHashJoin/PartitionedHashJoin.h>
#include <Interpreters/TableJoin.h>
#include <Common/PODArray.h>

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int UNSUPPORTED_JOIN_KEYS;
}

/** The routed probe: the single-map `joinRightColumns` loop with one difference - each row's map
  * is the leaf its recomputed route word points at. Probe blocks are never scattered, buffered or
  * materialized (G2); everything around the loop (`AddedColumns`, `processMatch`, the lazy
  * `HashJoinResult` emit) is the standard `HashJoin` machinery over the shared row store.
  */
template <JoinKind KIND, JoinStrictness STRICTNESS, typename KeyGetter, typename Map, typename AddedColumnsType>
size_t PartitionedHashJoin::routedJoinRightColumns(
    const std::vector<const Map *> & leaf_maps_vector, AddedColumnsType & added_columns, const ScatteredBlock & block)
{
    constexpr JoinFeatures<KIND, STRICTNESS, HashJoin::MapsAll> join_features;
    constexpr bool flag_per_row = false; /// single disjunct, INNER/LEFT: per-row used flags are never needed

    if (added_columns.additional_filter_expression)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Additional filter expression is not supported for PartitionedHashJoin");

    const auto & join_keys = added_columns.join_on_keys.at(0);
    const auto & selector = block.getSelector();
    const size_t rows = selector.size();
    JoinStuff::JoinUsedFlags & used_flags = *leaf_join->used_flags;

    /// One route word per probe row (over the whole source block: continuation chunks share it).
    const size_t source_rows = block.getSourceBlock().rows();
    PaddedPODArray<UInt16> leaf_ids;
    if (bits > 0 && source_rows > 0)
    {
        PaddedPODArray<UInt32> words(source_rows);
        computeJoinRouteWords(join_keys.key_columns, source_rows, words.data());
        leaf_ids.resize(source_rows);
        const auto shift = static_cast<UInt32>(32 - bits);
        for (size_t i = 0; i < source_rows; ++i)
            leaf_ids[i] = static_cast<UInt16>(words[i] >> shift);
    }
    else
    {
        leaf_ids.resize_fill(source_rows, 0);
    }

    KeyGetter key_getter(join_keys.key_columns, join_keys.key_sizes, nullptr);

    /// The skip byte merges the null map and the (absent here) ON mask, exactly like the
    /// single-map loop; the fast path compiles the check out.
    const bool fast_path = !join_keys.null_map && join_keys.join_mask_column.getKind() == JoinCommon::JoinMask::Kind::AllTrue;

    if constexpr (!flag_per_row && STRICTNESS == JoinStrictness::All)
        added_columns.lazy_output.output_by_row_list = true;

    if constexpr (join_features.need_replication)
        added_columns.offsets_to_replicate = IColumn::Offsets(rows);

    Arena pool;

    /// Routed look-ahead software prefetch: the leaf tables are cache-sized by design, so this
    /// fires only when the aggregate table size outgrows the threshold (e.g. the single-leaf
    /// degenerate plan or many leaves probed in a mixed stream).
    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;
    bool use_prefetch = false;
    if constexpr (can_prefetch)
        use_prefetch = added_columns.enable_prefetch && ht_slab_bytes > getMinBytesForPrefetchInJoin();

    auto prefetcher = makeJoinPrefetcher(
        use_prefetch,
        rows,
        [&](size_t k) __attribute__((always_inline))
        {
            if constexpr (can_prefetch)
            {
                const size_t ind = selector[k];
                leaf_maps_vector[leaf_ids[ind]]->prefetch(key_getter.getKeyHolder(ind, pool));
            }
        });

    auto loop = [&]<bool need_filter, bool with_skip>(const UInt8 * skip_data)
    {
        if constexpr (need_filter)
        {
            added_columns.filter = IColumn::Filter(rows, 0);
            added_columns.matched_rows.reserve(rows);
        }

        IColumn::Offset current_offset = 0;
        for (size_t i = 0; i < rows; ++i)
        {
            if constexpr (can_prefetch)
                prefetcher.prefetchAt(i);

            const size_t ind = selector[i];

            bool right_row_found = false;
            KnownRowsHolder<flag_per_row> dummy_known_rows;

            bool skip_row = false;
            if constexpr (with_skip)
                skip_row = skip_data && skip_data[ind];

            if (!skip_row)
            {
                auto find_result = key_getter.findKey(*leaf_maps_vector[leaf_ids[ind]], ind, pool);
                if (find_result.isFound())
                {
                    right_row_found = true;
                    processMatch<KIND, STRICTNESS, need_filter, flag_per_row, HashJoin::MapsAll, Map, KeyGetter>(
                        find_result, added_columns, used_flags, i, ind, current_offset, dummy_known_rows);
                }
            }

            if (!right_row_found)
                addNotFoundRow<join_features.add_missing, join_features.need_replication>(added_columns, current_offset);

            if constexpr (join_features.need_replication)
                added_columns.offsets_to_replicate[i] = current_offset;
        }
    };

    const UInt8 * skip_data = nullptr;
    IColumn::Filter skip_buffer;
    if (!fast_path)
    {
        if (selector.isContinuousRange())
            skip_data = join_keys.buildRowSkipData(skip_buffer, selector.getRange().first, rows);
        else
            skip_data = join_keys.buildRowSkipData(skip_buffer, selector.getIndexes());
    }

    if (added_columns.need_filter)
    {
        if (fast_path)
            loop.template operator()<true, false>(nullptr);
        else
            loop.template operator()<true, true>(skip_data);
    }
    else
    {
        if (fast_path)
            loop.template operator()<false, false>(nullptr);
        else
            loop.template operator()<false, true>(skip_data);
    }

    added_columns.applyLazyDefaults();
    return 0;
}

template <JoinKind KIND, JoinStrictness STRICTNESS>
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

    constexpr JoinFeatures<KIND, STRICTNESS, HashJoin::MapsAll> join_features;

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

    switch (join.data->type)
    {
#define M(TYPE) \
    case HashJoin::Type::TYPE: { \
        using Map = const typename decltype(PartitionedJoinMaps::TYPE)::element_type; \
        using KeyGetter = typename KeyGetterForType<HashJoin::Type::TYPE, Map>::Type; \
        std::vector<Map *> leaf_maps_vector(std::max<size_t>(partitions, 1), nullptr); \
        for (size_t leaf = 0; leaf < leaf_maps.size(); ++leaf) \
            leaf_maps_vector[leaf] = leaf_maps[leaf].TYPE.get(); \
        routedJoinRightColumns<KIND, STRICTNESS, KeyGetter, Map>(leaf_maps_vector, added_columns, scattered_block); \
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

JoinResultPtr PartitionedHashJoin::probeDispatch(Block block)
{
    /// Dispatch on the leaf delegate's kind/strictness: the build barrier may have promoted
    /// ALL to RightAny when every build key turned out unique (the probe then skips the
    /// replication machinery, like `hash`/`parallel_hash` do).
    const JoinKind kind = leaf_join->getKind();
    const JoinStrictness strictness = leaf_join->getStrictness();

    if (kind == JoinKind::Inner && strictness == JoinStrictness::All)
        return probeImpl<JoinKind::Inner, JoinStrictness::All>(std::move(block));
    if (kind == JoinKind::Left && strictness == JoinStrictness::All)
        return probeImpl<JoinKind::Left, JoinStrictness::All>(std::move(block));
    if (kind == JoinKind::Inner && strictness == JoinStrictness::RightAny)
        return probeImpl<JoinKind::Inner, JoinStrictness::RightAny>(std::move(block));
    if (kind == JoinKind::Left && strictness == JoinStrictness::RightAny)
        return probeImpl<JoinKind::Left, JoinStrictness::RightAny>(std::move(block));

    throw Exception(ErrorCodes::LOGICAL_ERROR, "Wrong JOIN combination for PartitionedHashJoin: {} {}", strictness, kind);
}

}
