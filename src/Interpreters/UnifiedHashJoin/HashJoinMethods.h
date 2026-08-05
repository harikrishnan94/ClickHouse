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
    ///
    /// `key_row` is the row the key getter reads and `row_no` the row the ref records. They differ
    /// when the scatter copied the keys into per-bucket dense columns (see `scatterByBucket`): the
    /// getter then reads the dense copy at its own position while the ref still points at the row
    /// of the stored block.
    static ALWAYS_INLINE bool insertOne(
        const HashJoin & join,
        HashMap & map,
        KeyGetter & key_getter,
        UInt32 stored_block_no,
        size_t key_row,
        size_t row_no,
        Arena & pool,
        size_t & new_keys)
    {
        auto emplace_result = key_getter.emplaceKey(map, key_row, pool);

        const bool inserted = emplace_result.isInserted();
        new_keys += inserted;
        if (inserted || join.anyTakeLastRow())
            new (&emplace_result.getMapped()) typename HashMap::mapped_type(stored_block_no, row_no);
        return inserted || join.anyTakeLastRow();
    }

    static ALWAYS_INLINE bool insertAll(
        const HashJoin &,
        HashMap & map,
        KeyGetter & key_getter,
        UInt32 stored_block_no,
        size_t key_row,
        size_t row_no,
        Arena & pool,
        size_t & new_keys)
    {
        auto emplace_result = key_getter.emplaceKey(map, key_row, pool);

        const bool inserted = emplace_result.isInserted();
        new_keys += inserted;
        if (inserted)
            new (&emplace_result.getMapped()) typename HashMap::mapped_type(stored_block_no, row_no);
        else
        {
            /// A single ref is stored inline in the value of the hash table; the first duplicate
            /// switches the value to a pointer to an arena-allocated list of refs.
            emplace_result.getMapped().insert(RowRef(stored_block_no, row_no).encode(), pool);
        }
        return inserted;
    }

    static ALWAYS_INLINE bool insertAsof(
        HashJoin & join,
        HashMap & map,
        KeyGetter & key_getter,
        UInt32 stored_block_no,
        size_t key_row,
        size_t row_no,
        Arena & pool,
        size_t & new_keys,
        const IColumn & asof_column)
    {
        auto emplace_result = key_getter.emplaceKey(map, key_row, pool);
        typename HashMap::mapped_type * time_series_map = &emplace_result.getMapped();

        const bool inserted = emplace_result.isInserted();
        new_keys += inserted;
        TypeIndex asof_type = *join.getAsofType();
        if (inserted)
            time_series_map = new (time_series_map) typename HashMap::mapped_type(createAsofRowRef(asof_type, join.getAsofInequality()));
        (*time_series_map)->insert(asof_column, stored_block_no, row_no);
        return inserted;
    }
};
/// A key getter for one block, built once and handed to everything that reads that block's keys.
/// Share only when construction packs the whole block (`HashMethodKeysFixed`); otherwise build per
/// bucket so the row loop keeps a stack-local getter. Safe on the build path: getters write nothing
/// after construction. Type-erased here because the map kind is runtime.
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
    static constexpr bool needs_offset = JoinFeatures<KIND, STRICTNESS, MapsTemplate>::need_flags;

public:
    /// The scatter's output: one selector per slot, and - when the keys were scattered by copying
    /// (see `scatterBySlot`) - one dense copy of the key columns per slot, parallel to that slot's
    /// selector. `dense_keys` is empty when the keys kept the zero-copy selectors.
    struct SlotScatter
    {
        std::vector<ScatteredBlock::Selector> selectors;
        std::vector<Columns> dense_keys;
    };

    /// Insert `selector`'s rows into `maps`. Caller holds the slot lock; `block_key_getter` is shared
    /// across the block's slots. `dense_keys`, when not null, is this slot's dense key-column copies.
    static void insertFromBlockImpl(
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
        BuildResult & result);

    /// Split rows by the slot that owns each key's bucket. Mirrors `ConcurrentHashJoin::dispatchBlock`:
    /// narrow fixed-size keys are also copied into `SlotScatter::dense_keys` for sequential insert.
    static SlotScatter scatterBySlot(
        HashJoin::Type type,
        MapsTemplate & maps,
        BlockKeyGetter & block_key_getter,
        const ColumnRawPtrs & key_columns,
        const Sizes & key_sizes,
        const ScatteredBlock::Selector & selector,
        size_t num_slots);

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

    template <typename KeyGetter, bool is_asof_join>
    static KeyGetter & blockKeyGetter(
        BlockKeyGetter & block_key_getter, std::optional<KeyGetter> & own, const ColumnRawPtrs & key_columns, const Sizes & key_sizes);

    template <typename KeyGetter, typename HashMap, typename Selector>
    static void insertFromBlockImplTypeCase(
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
        BuildResult & result);

    template <typename KeyGetter, typename HashMap>
    static SlotScatter scatterBySlotTypeCase(
        const HashMap & map,
        BlockKeyGetter & block_key_getter,
        const ColumnRawPtrs & key_columns,
        const Sizes & key_sizes,
        const ScatteredBlock::Selector & selector,
        size_t num_slots);

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
