#pragma once
#include <Interpreters/UnifiedHashJoin/HashJoin.h>
#include <Interpreters/UnifiedHashJoin/KeyGetter.h>
#include <Interpreters/UnifiedHashJoin/JoinFeatures.h>
#include <Interpreters/UnifiedHashJoin/AddedColumns.h>
#include <Interpreters/UnifiedHashJoin/KnownRowsHolder.h>
#include <Interpreters/UnifiedHashJoin/JoinUsedFlags.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/TableJoin.h>
#include <Interpreters/castColumn.h>
#include <base/types.h>

#include <memory>
#include <optional>
#include <typeinfo>

namespace DB
{
namespace Unified
{

/// Prefetching doesn't make sense for small hash tables, because they fit in caches entirely.
/// Returns the threshold (in bytes) above which prefetching is enabled in JOIN.
size_t getMinBytesForPrefetchInJoin();

/// Inserting an element into a hash table of the form `key -> reference to a row`, which will then be used by JOIN.
template <typename HashMap, typename KeyGetter>
struct Inserter
{
    /// `new_keys` counts keys the map did not have before, which is what the size limits are
    /// checked against. It is not always what these functions return: with `any_take_last_row` a
    /// duplicate overwrites the mapped value, so the row becomes reachable without adding a key.
    static ALWAYS_INLINE bool insertOne(
        const HashJoin & join, HashMap & map, KeyGetter & key_getter, UInt32 stored_block_no, size_t i, Arena & pool, size_t & new_keys)
    {
        auto emplace_result = key_getter.emplaceKey(map, i, pool);

        const bool inserted = emplace_result.isInserted();
        new_keys += inserted;
        if (inserted || join.anyTakeLastRow())
            new (&emplace_result.getMapped()) typename HashMap::mapped_type(stored_block_no, i);
        return inserted || join.anyTakeLastRow();
    }

    static ALWAYS_INLINE bool insertAll(
        const HashJoin &, HashMap & map, KeyGetter & key_getter, UInt32 stored_block_no, size_t i, Arena & pool, size_t & new_keys)
    {
        auto emplace_result = key_getter.emplaceKey(map, i, pool);

        const bool inserted = emplace_result.isInserted();
        new_keys += inserted;
        if (inserted)
            new (&emplace_result.getMapped()) typename HashMap::mapped_type(stored_block_no, i);
        else
        {
            /// A single ref is stored inline in the value of the hash table; the first duplicate
            /// switches the value to a pointer to an arena-allocated list of refs.
            emplace_result.getMapped().insert(RowRef(stored_block_no, i).encode(), pool);
        }
        return inserted;
    }

    static ALWAYS_INLINE bool insertAsof(
        HashJoin & join,
        HashMap & map,
        KeyGetter & key_getter,
        UInt32 stored_block_no,
        size_t i,
        Arena & pool,
        size_t & new_keys,
        const IColumn & asof_column)
    {
        auto emplace_result = key_getter.emplaceKey(map, i, pool);
        typename HashMap::mapped_type * time_series_map = &emplace_result.getMapped();

        const bool inserted = emplace_result.isInserted();
        new_keys += inserted;
        TypeIndex asof_type = *join.getAsofType();
        if (inserted)
            time_series_map = new (time_series_map) typename HashMap::mapped_type(createAsofRowRef(asof_type, join.getAsofInequality()));
        (*time_series_map)->insert(asof_column, stored_block_no, i);
        return inserted;
    }
};
/// A key getter for one block, built once and handed to everything that reads that block's keys.
///
/// Only for a getter whose construction reads the whole block: `HashMethodKeysFixed` packs the
/// block's fixed-width keys in its constructor - sized by the key column, taking no notice of the
/// selector it will be used with - so a build that splits the block by bucket would otherwise repack
/// every row of it once per bucket. A getter that only latches a couple of column pointers is built
/// per bucket, because a shared one has to live behind a pointer and buys nothing to pay for it. See
/// `shareKeyGetterAcrossBuckets`.
///
/// Sharing is safe on the build path: the getters a join can reach write nothing after their
/// constructor (the last-element cache is compiled out for every join key getter, and the
/// LowCardinality getter's probe caches are read by `findKey` only, never by `emplaceKey`).
///
/// The type is not known where the holder is created - it follows from the join's map kind, which is
/// a runtime value - so it is erased here and recovered by the caller that knows it. Every user of
/// one holder asks for the same type, because they all derive it from the same map.
class BlockKeyGetter
{
public:
    template <typename KeyGetter, typename Build>
    KeyGetter & getOrBuild(Build && build)
    {
        if (!getter)
        {
            getter = std::make_shared<KeyGetter>(build());
            built_type = &typeid(KeyGetter);
        }
        chassert(*built_type == typeid(KeyGetter));
        return *static_cast<KeyGetter *>(getter.get());
    }

private:
    std::shared_ptr<void> getter;
    const std::type_info * built_type = nullptr;
};

/// Whether a block's buckets share one key getter or each build their own. Sharing removes a
/// per-bucket pass over the block's keys, and costs one allocation per block plus a pointer
/// indirection the row loop cannot always see through, so it is worth it exactly when construction
/// is what reads the block.
template <typename KeyGetter>
constexpr bool shareKeyGetterAcrossBuckets()
{
    if constexpr (requires { KeyGetter::reads_whole_block_at_construction; })
        return KeyGetter::reads_whole_block_at_construction;
    else
        return false;
}

/// MapsTemplate is one of MapsOne, MapsAll and MapsAsof
template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
class HashJoinMethods
{
    /// Whether the key getters have to report a matched cell's offset: only a join that keeps
    /// per-offset used flags reads one (see `JoinUsedFlags`), and on a partitioned map producing an
    /// offset means placing the cell among the other buckets' cells rather than reading a pointer
    /// difference. The build never reads offsets, but it shares the constant so that a join
    /// instantiates one key getter rather than two.
    static constexpr bool needs_offset = JoinFeatures<KIND, STRICTNESS, MapsTemplate>::need_flags;

public:
    /// Insert `selector`'s rows into `bucket` of every map in `maps`. The caller routed those rows
    /// there (see `scatterByBucket`) and holds that bucket's lock, so the insert addresses the
    /// bucket's cells directly rather than re-deriving the bucket from each row's key.
    ///
    /// `block_key_getter` belongs to the block, not to the bucket: the caller passes the same one to
    /// every bucket of a block and to the scatter pass that split it, so the block's keys are read
    /// once. See `BlockKeyGetter`.
    static void insertFromBlockImpl(
        HashJoin & join,
        HashJoin::Type type,
        MapsTemplate & maps,
        size_t bucket,
        BlockKeyGetter & block_key_getter,
        const ColumnRawPtrs & key_columns,
        const Sizes & key_sizes,
        UInt32 stored_block_no,
        const ScatteredBlock::Selector & selector,
        ConstNullMapPtr null_map,
        const JoinCommon::JoinMask & join_mask,
        Arena & pool,
        BuildResult & result);

    /// Split `selector`'s rows by the bucket of `maps` each row's key routes to, returning one
    /// selector per bucket. Inserts lock the bucket `emplace` routes each row to, which is the same
    /// routing function used here.
    static std::vector<ScatteredBlock::Selector> scatterByBucket(
        HashJoin::Type type,
        MapsTemplate & maps,
        BlockKeyGetter & block_key_getter,
        const ColumnRawPtrs & key_columns,
        const Sizes & key_sizes,
        const ScatteredBlock::Selector & selector,
        size_t num_buckets);

    using MapsTemplateVector = std::vector<const MapsTemplate *>;

    static JoinResultPtr joinBlockImpl(
        const HashJoin & join,
        Block block,
        const Block & block_with_columns_to_add,
        const MapsTemplateVector & maps_,
        bool is_join_get = false);

    static JoinResultPtr joinBlockImpl(
        const HashJoin & join,
        ScatteredBlock block,
        const Block & block_with_columns_to_add,
        const MapsTemplateVector & maps_,
        bool is_join_get = false);

private:
    template <typename KeyGetter, bool is_asof_join>
    static KeyGetter createKeyGetter(const ColumnRawPtrs & key_columns, const Sizes & key_sizes, HashJoin::RightTableData::KeyRange key_range = {});

    /// The key getter for this block's keys: `block_key_getter`'s, built on first use and then
    /// reused, or one constructed into `own` - see `shareKeyGetterAcrossBuckets`.
    template <typename KeyGetter, bool is_asof_join>
    static KeyGetter & blockKeyGetter(
        BlockKeyGetter & block_key_getter,
        std::optional<KeyGetter> & own,
        const ColumnRawPtrs & key_columns,
        const Sizes & key_sizes);

    template <typename KeyGetter, typename HashMap, typename Selector>
    static void insertFromBlockImplTypeCase(
        HashJoin & join,
        HashMap & map,
        size_t bucket,
        BlockKeyGetter & block_key_getter,
        const ColumnRawPtrs & key_columns,
        const Sizes & key_sizes,
        UInt32 stored_block_no,
        const Selector & selector,
        ConstNullMapPtr null_map,
        const JoinCommon::JoinMask & join_mask,
        Arena & pool,
        BuildResult & result);

    template <typename KeyGetter, typename HashMap>
    static std::vector<ScatteredBlock::Selector> scatterByBucketTypeCase(
        const HashMap & map,
        BlockKeyGetter & block_key_getter,
        const ColumnRawPtrs & key_columns,
        const Sizes & key_sizes,
        const ScatteredBlock::Selector & selector,
        size_t num_buckets);

    template <typename AddedColumns>
    static size_t switchJoinRightColumns(
        const std::vector<const MapsTemplate *> & mapv,
        AddedColumns & added_columns,
        const ScatteredBlock::Selector & selector,
        HashJoin::Type type,
        JoinStuff::JoinUsedFlags & used_flags,
        HashJoin::RightTableData::KeyRange key_range);

    template <typename KeyGetter, typename Map, typename AddedColumns>
    static size_t joinRightColumnsSwitchNullability(
        std::vector<KeyGetter> && key_getter_vector,
        const std::vector<const Map *> & mapv,
        AddedColumns & added_columns,
        const ScatteredBlock::Selector & selector,
        JoinStuff::JoinUsedFlags & used_flags);

    template <typename KeyGetter, typename Map, bool need_filter, typename AddedColumns>
    static size_t joinRightColumnsSwitchMultipleDisjuncts(
        std::vector<KeyGetter> && key_getter_vector,
        const std::vector<const Map *> & mapv,
        AddedColumns & added_columns,
        const ScatteredBlock::Selector & selector,
        JoinStuff::JoinUsedFlags & used_flags);

    /// Joins right table columns which indexes are present in right_indexes using specified map.
    /// Makes filter (1 if row presented in right table) and returns offsets to replicate (for ALL JOINS).
    /// `fast_path` compiles out the per-row null-map and join-mask checks for the common case of
    /// non-nullable keys and no ON-section condition (the checks are done at runtime otherwise).
    template <
        typename KeyGetter,
        typename Map,
        bool need_filter,
        bool fast_path,
        typename AddedColumns,
        typename Selector>
    static size_t joinRightColumns(
        std::vector<KeyGetter> && key_getter_vector,
        const std::vector<const Map *> & mapv,
        AddedColumns & added_columns,
        JoinStuff::JoinUsedFlags & used_flags,
        const Selector & selector);

    template <
        typename KeyGetter,
        typename Map,
        bool need_filter,
        bool fast_path,
        typename AddedColumns,
        typename Selector>
    static size_t joinRightColumns(
        KeyGetter & key_getter,
        const Map * map,
        AddedColumns & added_columns,
        JoinStuff::JoinUsedFlags & used_flags,
        const Selector & selector);

    /// First to collect all matched rows refs by join keys, then filter out rows which are not true in additional filter expression.
    template <typename KeyGetter, typename Map, typename AddedColumns>
    static size_t joinRightColumnsWithAdditionalFilter(
        std::vector<KeyGetter> && key_getter_vector,
        const std::vector<const Map *> & mapv,
        AddedColumns & added_columns,
        JoinStuff::JoinUsedFlags & used_flags [[maybe_unused]],
        const ScatteredBlock::Selector & selector,
        bool need_filter [[maybe_unused]],
        bool flag_per_row [[maybe_unused]]);

    /// Cut first num_rows rows from block in place and returns block with remaining rows
    static Block sliceBlock(Block & block, size_t num_rows);

    /** Since we do not store right key columns,
      * this function is used to copy left key columns to right key columns.
      * If the user requests some right columns, we just copy left key columns to right, since they are equal.
      * Example: SELECT t1.key, t2.key FROM t1 FULL JOIN t2 ON t1.key = t2.key;
      * In that case for matched rows in t2.key we will use values from t1.key.
      * However, in some cases we might need to adjust the type of column, e.g. t1.key :: LowCardinality(String) and t2.key :: String
      * Also, the nullability of the column might be different.
      * Returns the right column after with necessary adjustments.
      */
    static ColumnWithTypeAndName copyLeftKeyColumnToRight(
        const DataTypePtr & right_key_type,
        const String & renamed_right_column,
        const ColumnWithTypeAndName & left_column,
        const IColumn::Filter * null_map_filter = nullptr);

    static void correctNullabilityInplace(ColumnWithTypeAndName & column, bool nullable);

    static void correctNullabilityInplace(ColumnWithTypeAndName & column, bool nullable, const IColumn::Filter & negative_null_map);
};

/// Instantiate template class ahead in different .cpp files to avoid `too large translation unit`.
extern template class HashJoinMethods<JoinKind::Left, JoinStrictness::RightAny, HashJoin::MapsOne>;
extern template class HashJoinMethods<JoinKind::Left, JoinStrictness::RightAny, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Left, JoinStrictness::Any, HashJoin::MapsOne>;
extern template class HashJoinMethods<JoinKind::Left, JoinStrictness::Any, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Left, JoinStrictness::All, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Left, JoinStrictness::Semi, HashJoin::MapsOne>;
extern template class HashJoinMethods<JoinKind::Left, JoinStrictness::Semi, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Left, JoinStrictness::Anti, HashJoin::MapsOne>;
extern template class HashJoinMethods<JoinKind::Left, JoinStrictness::Anti, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Left, JoinStrictness::Asof, HashJoin::MapsAsof>;

extern template class HashJoinMethods<JoinKind::Right, JoinStrictness::RightAny, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Right, JoinStrictness::Any, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Right, JoinStrictness::All, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Right, JoinStrictness::Semi, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Right, JoinStrictness::Anti, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Right, JoinStrictness::Asof, HashJoin::MapsAsof>;

extern template class HashJoinMethods<JoinKind::Inner, JoinStrictness::RightAny, HashJoin::MapsOne>;
extern template class HashJoinMethods<JoinKind::Inner, JoinStrictness::RightAny, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Inner, JoinStrictness::Any, HashJoin::MapsOne>;
extern template class HashJoinMethods<JoinKind::Inner, JoinStrictness::Any, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Inner, JoinStrictness::All, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Inner, JoinStrictness::Semi, HashJoin::MapsOne>;
extern template class HashJoinMethods<JoinKind::Inner, JoinStrictness::Anti, HashJoin::MapsOne>;
extern template class HashJoinMethods<JoinKind::Inner, JoinStrictness::Asof, HashJoin::MapsAsof>;

extern template class HashJoinMethods<JoinKind::Full, JoinStrictness::RightAny, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Full, JoinStrictness::Any, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Full, JoinStrictness::All, HashJoin::MapsAll>;
extern template class HashJoinMethods<JoinKind::Full, JoinStrictness::Semi, HashJoin::MapsOne>;
extern template class HashJoinMethods<JoinKind::Full, JoinStrictness::Anti, HashJoin::MapsOne>;
extern template class HashJoinMethods<JoinKind::Full, JoinStrictness::Asof, HashJoin::MapsAsof>;
}

}
