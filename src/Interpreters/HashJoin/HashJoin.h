#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <Interpreters/HashTablesStatistics.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/RowRefs.h>

#include <Core/Block_fwd.h>
#include <Interpreters/HashJoin/ScatteredBlock.h>
#include <QueryPipeline/SizeLimits.h>
#include <Storages/IStorage_fwd.h>
#include <Storages/TableLockHolder.h>
#include <Common/Arena.h>
#include <Common/HashTable/FixedHashMap.h>
#include <Common/HashTable/HashMap.h>
#include <Common/HashTable/HashTableTraits.h>
#include <Interpreters/HashJoin/ResumableHashMap.h>

namespace DB
{

class TableJoin;
class ExpressionActions;
struct JoinOnKeyColumns;
struct JoinProbeScratch;
struct RoutedProbePlan;
using Sizes = std::vector<size_t>;

namespace JoinStuff
{
/// Flags needed to implement RIGHT and FULL JOINs.
class JoinUsedFlags;
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
class HashJoinMethods;

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
class RoutedHashJoinMethods;

/** Data structure for implementation of hash JOIN.
  * It is a hash table: keys -> rows of joined ("right") table.
  *
  * JOIN-s could be of these types:
  * - ALL × LEFT/INNER/RIGHT/FULL
  * - ANY × LEFT/INNER/RIGHT
  * - SEMI/ANTI x LEFT/RIGHT
  * - ASOF x LEFT/INNER
  *
  * ALL means usual JOIN, when rows are multiplied by number of matching rows from the "right" table.
  * ANY uses one line per unique key from right table. For LEFT JOIN it would be any row (with needed joined key) from the right table,
  * for RIGHT JOIN it would be any row from the left table and for INNER one it would be any row from right and any row from left.
  * SEMI JOIN filter left table by keys that are present in right table for LEFT JOIN, and filter right table by keys from left table
  * for RIGHT JOIN. In other words SEMI JOIN returns only rows which joining keys present in another table.
  * ANTI JOIN is the same as SEMI JOIN but returns rows with joining keys that are NOT present in another table.
  * SEMI/ANTI JOINs allow to get values from both tables. For filter table it gets any row with joining same key. For ANTI JOIN it returns
  * defaults other table columns.
  * ASOF JOIN is not-equi join. For one key column it finds nearest value to join according to join inequality.
  * It's expected that ANY|SEMI LEFT JOIN is more efficient that ALL one.
  *
  * If INNER is specified - leave only rows that have matching rows from "right" table.
  * If LEFT is specified - in case when there is no matching row in "right" table, fill it with default values instead.
  * If RIGHT is specified - first process as INNER, but track what rows from the right table was joined,
  *  and at the end, add rows from right table that was not joined and substitute default values for columns of left table.
  * If FULL is specified - first process as LEFT, but track what rows from the right table was joined,
  *  and at the end, add rows from right table that was not joined and substitute default values for columns of left table.
  *
  * Thus, LEFT and RIGHT JOINs are not symmetric in terms of implementation.
  *
  * All JOINs are done by equality condition on keys (equijoin).
  * Non-equality and other conditions are not supported.
  *
  * Implementation:
  *
  * 1. Build hash table in memory from "right" table.
  * This hash table is in form of keys -> row in case of ANY or keys -> [rows...] in case of ALL.
  * This is done in insertFromBlock method.
  *
  * 2. Process "left" table and join corresponding rows from "right" table by lookups in the map.
  * This is done in joinBlock methods.
  *
  * In case of ANY LEFT JOIN - form new columns with found values or default values.
  * This is the most simple. Number of rows in left table does not change.
  *
  * In case of ANY INNER JOIN - form new columns with found values,
  *  and also build a filter - in what rows nothing was found.
  * Then filter columns of "left" table.
  *
  * In case of ALL ... JOIN - form new columns with all found rows,
  *  and also fill 'offsets' array, describing how many times we need to replicate values of "left" table.
  * Then replicate columns of "left" table.
  *
  * How Nullable keys are processed:
  *
  * NULLs never join to anything, even to each other.
  * During building of map, we just skip keys with NULL value of any component.
  * During joining, we simply treat rows with any NULLs in key as non joined.
  *
  * Default values for outer joins (LEFT, RIGHT, FULL):
  *
  * Behaviour is controlled by 'join_use_nulls' settings.
  * If it is false, we substitute (global) default value for the data type, for non-joined rows
  *  (zero, empty string, etc. and NULL for Nullable data types).
  * If it is true, we always generate Nullable column and substitute NULLs for non-joined rows,
  *  as in standard SQL.
  */
class HashJoin : public IJoin
{
public:
    HashJoin(
        std::shared_ptr<TableJoin> table_join_,
        SharedHeader right_sample_block,
        bool any_take_last_row_ = false,
        size_t reserve_num_ = 0,
        const String & instance_id_ = "",
        const StatsCollectingParams & stats_collecting_params_ = {},
        /// Set by `ConcurrentHashJoin` for its per-slot joins; see the LowCardinality map choice.
        bool is_parallel_hash_slot = false);

    ~HashJoin() override;

    std::string getName() const override { return "HashJoin"; }

    const TableJoin & getTableJoin() const override { return *table_join; }

    bool isCloneSupported() const override
    {
        return getTotals().empty() && getTotalRowCount() == 0;
    }

    std::shared_ptr<IJoin> clone(const std::shared_ptr<TableJoin> & table_join_,
        SharedHeader,
        SharedHeader right_sample_block_) const override
    {
        return std::make_shared<HashJoin>(table_join_, right_sample_block_, any_take_last_row, reserve_num, instance_id);
    }

    /** Add block of data from right hand of JOIN to the map.
      * Returns false, if some limit was exceeded and you should not insert more data.
      */
    bool addBlockToJoin(const Block & source_block_, bool check_limits) override;

    using IJoin::addBlockToJoin;

    /// Called directly from ConcurrentJoin::addBlockToJoin
    bool addBlockToJoin(const Block & block, ScatteredBlock::Selector selector, bool check_limits);

    void checkTypesOfKeys(const Block & block) const override;

    using IJoin::joinBlock;

    /** Join data from the map (that was previously built by calls to addBlockToJoin) to the block with data from "left" table.
      * Could be called from different threads in parallel.
      */
    JoinResultPtr joinBlock(Block block) override;

    /// The order-preserving routed probe of `ConcurrentHashJoin` (`parallel_hash`): one lookup
    /// pass over the ORIGINAL left block, with each row's slot map selected by the row's route
    /// - derived inline from the lookup's own hash for the open-addressing families, or read
    /// from the scratch's eager `slot_ids` for the rest (see `RoutedHashJoinMethods`).
    /// `join_on_keys` holds the block's prepared key columns; the caller builds them once and
    /// any eager slot ids come from the same columns. The `slot_joins` share one
    /// `StoredColumnsIndex`, so the result emits every slot's matches through slot 0's
    /// machinery - in left-row order.
    static JoinResultPtr joinRoutedBlock(
        const std::vector<const HashJoin *> & slot_joins,
        const RoutedProbePlan & plan,
        ScatteredBlock block,
        JoinProbeScratch & scratch,
        std::vector<JoinOnKeyColumns> join_on_keys);

    /// Check joinGet arguments and infer the return type.
    DataTypePtr joinGetCheckAndGetReturnType(const DataTypes & data_types, const String & column_name, bool or_null) const;

    /// Used by joinGet function that turns StorageJoin into a dictionary.
    ColumnWithTypeAndName joinGet(const Block & block, const Block & block_with_columns_to_add) const;

    bool isFilled() const override { return from_storage_join; }

    JoinPipelineType pipelineType() const override
    {
        /// No need to process anything in the right stream if hash table was already filled
        if (from_storage_join)
            return JoinPipelineType::FilledRight;

        /// Default pipeline processes right stream at first and then left.
        return JoinPipelineType::FillRightFirst;
    }

    /** For RIGHT and FULL JOINs.
      * A stream that will contain default values from left table, joined with rows from right table, that was not joined before.
      * Use only after all calls to joinBlock was done.
      * left_sample_block is passed without account of 'use_nulls' setting (columns will be converted to Nullable inside).
      */
    IBlocksStreamPtr getNonJoinedBlocks(
        const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const override;

    void onBuildPhaseFinish() override;

    bool hasPostBuildPhase() const override;
    void runPostBuildPhase() override;

    /// Number of keys in all built JOIN maps.
    size_t getTotalRowCount() const final;
    /// Sum size in bytes of all buffers, used for JOIN maps and for all memory pools.
    size_t getTotalByteCount() const final;

    bool alwaysReturnsEmptySet() const final;

    JoinKind getKind() const { return kind; }
    JoinStrictness getStrictness() const { return strictness; }
    const std::optional<TypeIndex> & getAsofType() const { return asof_type; }
    ASOFJoinInequality getAsofInequality() const { return asof_inequality; }
    bool anyTakeLastRow() const { return any_take_last_row; }

    const ColumnWithTypeAndName & rightAsofKeyColumn() const;

    /// Different types of keys for maps.
    #define APPLY_FOR_JOIN_VARIANTS(M) \
        M(key8)                        \
        M(key16)                       \
        M(key32)                       \
        M(key64)                       \
        M(key_string)                  \
        M(key_fixed_string)            \
        M(keys32)                      \
        M(keys64)                      \
        M(keys128)                     \
        M(keys256)                     \
        M(hashed)                      \
        M(low_cardinality_key_string)       \
        M(low_cardinality_key_fixed_string) \
        M(range8_key32)                \
        M(range16_key32)               \
        M(range17_key32)               \
        M(range18_key32)               \
        M(range8_key64)                \
        M(range16_key64)               \
        M(range17_key64)               \
        M(range18_key64)

    /// Used for reading from StorageJoin and applying joinGet function. The single-LowCardinality-key
    /// maps store key values in maps physically identical to their non-LowCardinality counterparts, so
    /// they are read back the same way (the output key column is the parent LowCardinality type).
    #define APPLY_FOR_JOIN_VARIANTS_LIMITED(M) \
        M(key8)                                \
        M(key16)                               \
        M(key32)                               \
        M(key64)                               \
        M(key_string)                          \
        M(key_fixed_string)                    \
        M(low_cardinality_key_string)          \
        M(low_cardinality_key_fixed_string)

    enum class Type : uint8_t
    {
        #define M(NAME) NAME,
            APPLY_FOR_JOIN_VARIANTS(M)
        #undef M
    };

    /// True for the single-LowCardinality-column maps, whose key getter consumes the live
    /// ColumnLowCardinality (so the key column must not be materialized for them).
    static bool isLowCardinalityType(Type type)
    {
        switch (type)
        {
            case Type::low_cardinality_key_string:
            case Type::low_cardinality_key_fixed_string:
                return true;
            default:
                return false;
        }
    }

    /** Different data structures, that are used to perform JOIN.
      */
    template <typename Mapped>
    struct MapsTemplate
    {
        /// NOLINTBEGIN(bugprone-macro-parentheses)
        using MappedType = Mapped;
        /// The nine maps `key32`..`hashed` are rebound through `WithJoinCursor` so the
        /// `parallel_hash` AMAC rings can drive their cursor API; `hash` uses them through the
        /// unchanged public interface. The low-cardinality strings are not ring targets and
        /// stay standard: their key getter memoises `findKey` per dictionary index, which the
        /// ring would bypass.
        std::shared_ptr<FixedHashMap<UInt8, Mapped>>                          key8;
        std::shared_ptr<FixedHashMap<UInt16, Mapped>>                         key16;
        std::shared_ptr<typename WithJoinCursor<HashMap<UInt32, Mapped, HashCRC32<UInt32>>>::Type>      key32;
        std::shared_ptr<typename WithJoinCursor<HashMap<UInt64, Mapped, HashCRC32<UInt64>>>::Type>      key64;
        std::shared_ptr<typename WithJoinCursor<HashMapWithSavedHash<std::string_view, Mapped>>::Type>  key_string;
        std::shared_ptr<typename WithJoinCursor<HashMapWithSavedHash<std::string_view, Mapped>>::Type>  key_fixed_string;
        std::shared_ptr<typename WithJoinCursor<HashMap<UInt32, Mapped, HashCRC32<UInt32>>>::Type>      keys32;
        std::shared_ptr<typename WithJoinCursor<HashMap<UInt64, Mapped, HashCRC32<UInt64>>>::Type>      keys64;
        std::shared_ptr<typename WithJoinCursor<HashMap<UInt128, Mapped, UInt128HashCRC32>>::Type>      keys128;
        std::shared_ptr<typename WithJoinCursor<HashMap<UInt256, Mapped, UInt256HashCRC32>>::Type>      keys256;
        std::shared_ptr<typename WithJoinCursor<HashMap<UInt128, Mapped, UInt128TrivialHash>>::Type> hashed;
        std::shared_ptr<HashMapWithSavedHash<std::string_view, Mapped>>      low_cardinality_key_string;
        std::shared_ptr<HashMapWithSavedHash<std::string_view, Mapped>>      low_cardinality_key_fixed_string;
        std::shared_ptr<FixedHashMapWithSizeBits<UInt32, Mapped, 8>>          range8_key32;
        std::shared_ptr<FixedHashMapWithSizeBits<UInt32, Mapped, 16>>         range16_key32;
        std::shared_ptr<FixedHashMapWithSizeBits<UInt32, Mapped, 17>>         range17_key32;
        std::shared_ptr<FixedHashMapWithSizeBits<UInt32, Mapped, 18>>         range18_key32;
        std::shared_ptr<FixedHashMapWithSizeBits<UInt64, Mapped, 8>>          range8_key64;
        std::shared_ptr<FixedHashMapWithSizeBits<UInt64, Mapped, 16>>         range16_key64;
        std::shared_ptr<FixedHashMapWithSizeBits<UInt64, Mapped, 17>>         range17_key64;
        std::shared_ptr<FixedHashMapWithSizeBits<UInt64, Mapped, 18>>         range18_key64;

        void create(Type which, size_t reserve)
        {
            switch (which)
            {
            #define M(NAME)                                                                                       \
                case Type::NAME:                                                                                  \
                    if constexpr (HasConstructorOfNumberOfElements<typename decltype(NAME)::element_type>::value) \
                        NAME = reserve ? std::make_shared<typename decltype(NAME)::element_type>(reserve)         \
                                       : std::make_shared<typename decltype(NAME)::element_type>();               \
                    else                                                                                          \
                        NAME = std::make_shared<typename decltype(NAME)::element_type>();                         \
                    break;

                APPLY_FOR_JOIN_VARIANTS(M)
            #undef M
            }
        }

        size_t getTotalRowCount(Type which) const
        {
            switch (which)
            {
            #define M(NAME) \
                case Type::NAME: return NAME ? NAME->size() : 0;
                APPLY_FOR_JOIN_VARIANTS(M)
            #undef M
            }
        }

        size_t getTotalByteCountImpl(Type which) const
        {
            switch (which)
            {
            #define M(NAME) \
                case Type::NAME: return NAME ? NAME->getBufferSizeInBytes() : 0;
                APPLY_FOR_JOIN_VARIANTS(M)
            #undef M
            }
        }

        size_t getBufferSizeInCells(Type which) const
        {
            switch (which)
            {
            #define M(NAME) \
                case Type::NAME: return NAME ? NAME->getBufferSizeInCells() : 0;
                APPLY_FOR_JOIN_VARIANTS(M)
            #undef M
            }
        }
/// NOLINTEND(bugprone-macro-parentheses)
    };

    using MapsOne = MapsTemplate<RowRef>;
    using MapsAll = MapsTemplate<RowRefList>;
    using MapsAsof = MapsTemplate<AsofRowRefs>;

    using MapsVariant = std::variant<MapsOne, MapsAll, MapsAsof>;

    struct NullMapHolder
    {
        const StoredBlock * columns{};
        ColumnPtr column;
        size_t selector_rows = 0;

        NullMapHolder() = default;
        explicit NullMapHolder(const StoredBlock * columns_, ColumnPtr column_)
            : columns(columns_), column(column_)
        {
            // we can cache the selector size at construction to make the holder robust
            // even if columns are moved/cleared later
            selector_rows = columns ? columns->selector.size() : (this->column ? this->column->size() : 0);
        }

        size_t allocatedBytes() const;
    };

    using NullmapList = std::deque<NullMapHolder>;
    using StoredBlocksList = std::list<StoredBlock>;

    struct RightTableData
    {
        Type type = Type::hashed;

        /// tab1 join tab2 on t1.x = t2.x or t1.y = t2.y
        /// =>
        /// tab1 join tab2 on t1.x = t2.x
        /// join tab2 on [not_joined(t1.x = t2.x)] and t1.y = t2.y
        std::vector<MapsVariant> maps;
        Block sample_block; /// Block as it would appear in the BlockList
        StoredBlocksList columns; /// Columns of "right" table.
        NullmapList nullmaps; /// Nullmaps for blocks of "right" table (if needed)

        /// Resolves RowRef::block_no to the stored block.
        StoredColumnsIndexPtr stored_columns_index = std::make_shared<StoredColumnsIndex>();

        /// Additional data - strings for string keys and continuation elements of single-linked lists of references to rows.
        Arena pool;

        size_t allocated_size = 0;
        size_t nullmaps_allocated_size = 0;
        /// Number of rows of right table to join
        size_t rows_to_join = 0;
        /// Number of keys of right table to join
        size_t keys_to_join = 0;
        /// Whether the right table reranged by key
        bool sorted = false;

        /// For range types: the minimum key value and the range size from min_key to max_key.
        struct KeyRange
        {
            UInt64 min_key = 0;
            UInt64 size = 0;
        };

        KeyRange key_range;

        size_t avgPerKeyRows() const
        {
            if (keys_to_join == 0)
                return 0;
            return rows_to_join / keys_to_join;
        }
    };

    /// For INNER/LEFT ALL JOINs, if the right side has no duplicates inside the join key columns,
    /// we can switch from ALL to RightAny strictness for better performance.
    bool all_values_unique = true;
    bool all_join_was_promoted_to_right_any = false;

    using RightTableDataPtr = std::shared_ptr<RightTableData>;

    /// We keep correspondence between used_flags and hash table internal buffer.
    /// Hash table cannot be modified during HashJoin lifetime and must be protected with lock.
    void setLock(TableLockHolder rwlock_holder)
    {
        storage_join_lock = rwlock_holder;
    }

    void reuseJoinedData(const HashJoin & join);

    RightTableDataPtr getJoinedData() const { return data; }
    BlocksList releaseJoinedBlocks(bool restructure = false);

    /// Modify right block (update structure according to sample block) to save it in block list
    static Block prepareRightBlock(const Block & block, const Block & saved_block_sample_);
    Block prepareRightBlock(const Block & block) const;

    const Block & savedBlockSample() const { return data->sample_block; }

    bool isUsed(size_t off) const;
    bool isUsed(UInt32 block_no, size_t row_idx) const;

    void debugKeys() const;

    void shrinkStoredBlocksToFit(size_t & total_bytes_in_join, bool force_optimize = false);

    void setMaxJoinedBlockRows(size_t value) { max_joined_block_rows = value; }
    void setMaxJoinedBlockBytes(size_t value) { max_joined_block_bytes = value; }

    void materializeColumnsFromLeftBlock(Block & block) const;
    Block materializeColumnsFromRightBlock(Block block) const;

    size_t getAndSetRightTableKeys() const;

    bool hasNonJoinedRows();
    void updateNonJoinedRowsStatus();

    const std::vector<Sizes> & getKeySizes() const { return key_sizes; }

    bool enableLazyColumnsReplication() const { return enable_lazy_columns_replication; }
    bool enableSoftwarePrefetch() const { return enable_prefetch; }

    /// Whether the AMAC build-insert ring (see `AmacRing.h`) may engage for this join's maps.
    /// Flipped only by `ConcurrentHashJoin` for its per-slot instances - the large, insert-bound
    /// builds the ring targets; plain `hash`, `StorageJoin` and grace hash keep the sequential
    /// insert loop.
    void setAmacEnabled(bool value) { amac_enabled = value; }
    bool amacEnabled() const { return amac_enabled; }

    void setEnableLazyColumnsIndexing(bool value) override { enable_lazy_columns_indexing = value; }

    static bool isUsedByAnotherAlgorithm(const TableJoin & table_join);
    static bool canRemoveColumnsFromLeftBlock(const TableJoin & table_join);

private:
    friend class NotJoinedHash;
    friend class JoinSource;
    /// Collects the once-per-build `RoutedProbePlan` (maps, descriptors, used flags) across
    /// its slots.
    friend class ConcurrentHashJoin;

    template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
    friend class HashJoinMethods;

    template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
    friend class RoutedHashJoinMethods;

    std::shared_ptr<TableJoin> table_join;
    JoinKind kind;
    JoinStrictness strictness;

    bool has_non_joined_rows_checked = false;
    bool has_non_joined_rows = false;

    /// This join was created from StorageJoin and it is already filled.
    bool from_storage_join = false;

    const bool any_take_last_row; /// Overwrite existing values when encountering the same key again
    const size_t reserve_num;
    const String instance_id;
    std::optional<TypeIndex> asof_type;
    const ASOFJoinInequality asof_inequality;

    /// Right table data. StorageJoin shares it between many Join objects.
    /// Flags that indicate that particular row already used in join.
    /// Flag is stored for every record in hash map.
    /// Number of this flags equals to hashtable buffer size (plus one for zero value).
    /// Changes in hash table broke correspondence,
    /// so we must guarantee constantness of hash table during HashJoin lifetime (using method setLock)
    mutable std::unique_ptr<JoinStuff::JoinUsedFlags> used_flags;
    RightTableDataPtr data;

    std::vector<Sizes> key_sizes;

    /// Block with columns from the right-side table.
    Block right_sample_block;
    /// Block with columns from the right-side table except key columns.
    Block sample_block_with_columns_to_add;
    /// Block with key columns in the same order they appear in the right-side table (duplicates appear once).
    Block right_table_keys;
    /// Block with key columns right-side table keys that are needed in result (would be attached after joined columns).
    Block required_right_keys;
    /// Left table column names that are sources for required_right_keys columns
    std::vector<String> required_right_keys_sources;

    std::vector<std::pair<size_t, size_t>> additional_filter_required_rhs_pos;

    /// Maximum number of rows in result block. If it is 0, then no limits.
    size_t max_joined_block_rows = 0;
    size_t max_joined_block_bytes = 0;
    bool joined_block_split_single_row = false;
    bool enable_lazy_columns_replication = false;
    bool enable_lazy_columns_indexing = false;
    bool enable_prefetch = true;
    bool amac_enabled = false;

    /// When tracked memory consumption is more than a threshold, we will shrink to fit stored blocks.
    bool shrink_blocks = false;
    Int64 memory_usage_before_adding_blocks = 0;

    /// Track if conversion to fixed hash map was already attempted to prevent repeated checks.
    bool conversion_to_fixed_hash_map_attempted = false;

    /// Track if shared runtime filters were already published to keep publication one-shot.
    bool shared_runtime_filters_publish_attempted = false;

    const StatsCollectingParams stats_collecting_params;
    bool build_phase_finished = false;

    /// Identifier to distinguish different HashJoin instances in logs
    /// Several instances can be created, for example, in GraceHashJoin to handle different buckets
    String instance_log_id;

    LoggerPtr log;

    /// Should be set via setLock to protect hash table from modification from StorageJoin
    /// If set HashJoin instance is not available for modification (addBlockToJoin)
    TableLockHolder storage_join_lock = nullptr;

    void dataMapInit(MapsVariant & map);

    void initRightBlockStructure(Block & saved_block_sample);

    bool preferUseMapsAll() const;

    bool isUsedByAnotherAlgorithm() const;
    bool canRemoveColumnsFromLeftBlock() const;

    void validateAdditionalFilterExpression(std::shared_ptr<ExpressionActions> additional_filter_expression);
    bool needUsedFlagsForPerRightTableRow(std::shared_ptr<TableJoin> table_join_) const;

    bool rightTableCanBeReranged() const;
    void tryRerangeRightTableData();

    template <JoinKind KIND, typename Map, JoinStrictness STRICTNESS>
    void tryRerangeRightTableDataImpl(Map & map);

    bool canConvertToFixedHashMap() const;

    /// Publish a SharedFixedHashTableRuntimeFilter that replaces the Set/BloomFilter
    /// installed by BuildRuntimeFilterStep, when the build side is a FixedHashMap.
    void publishSharedRuntimeFilters();
    void tryConvertToFixedHashMap();

    template <bool is_signed, typename Key, typename MapsTemplate>
    void tryConvertToFixedHashMapImpl(MapsTemplate & maps);

    void reinitUsedFlags();

    void doDebugAsserts() const;
};
}
