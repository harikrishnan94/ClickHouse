#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <variant>
#include <vector>

#include <Interpreters/HashTablesStatistics.h>
#include <Interpreters/IInMemoryHashJoin.h>
#include <Interpreters/RowRefs.h>

#include <Core/Block_fwd.h>
#include <Interpreters/HashJoin/ScatteredBlock.h>
#include <QueryPipeline/SizeLimits.h>
#include <Storages/IStorage_fwd.h>
#include <Storages/TableLockHolder.h>
#include <Common/Arena.h>
#include <Common/HashTable/BucketPartitionedTable.h>
#include <Common/HashTable/FixedHashMap.h>
#include <Common/HashTable/HashMap.h>
#include <Common/HashTable/PartitionedFixedHashMap.h>
#include <Common/HashTable/TwoLevelHashMap.h>

namespace DB
{

class TableJoin;
class ExpressionActions;
/// Reads the built maps back for StorageJoin; see StorageJoin.cpp.
class JoinSource;
using Sizes = std::vector<size_t>;

namespace Unified
{

namespace JoinStuff
{
/// Flags needed to implement RIGHT and FULL JOINs.
class JoinUsedFlags;
}

template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
class HashJoinMethods;

/// The join's hash maps are two-level, in the runtime-sized mode (`bits_for_bucket == -1`), and are
/// currently built with a single first-level bucket, so bucket selection always yields 0 and the
/// offsets the per-row used-flags are indexed by come out identical to the single-level ones. The
/// build is still serialized by `build_mutex`. Making `NUM_BUCKETS` a real, per-join value is the
/// next step of the sharded build that lets `hash` subsume `parallel_hash`.
///
/// Behaviour is unchanged, but cost is not. Runtime mode keeps the bucket count in a member rather
/// than in the type, so the compiler cannot fold the bucket arithmetic away: `offsetInternal` really
/// does recompute the cell hash to derive a bucket that is always 0, the buckets live behind a
/// `std::vector` instead of an inline array, and `emplace` calls a `refreshDesc` that is no longer
/// an empty function.
constexpr Int32 BITS_FOR_BUCKET = -1;

/// Must be a power of two and at least 1; see `TwoLevelHashTable::RuntimeStorage`.
constexpr size_t NUM_BUCKETS = 1;

/// The grower is spelled out because `TwoLevelHashMap` defaults to `TwoLevelHashTableGrower`, which
/// stops quadrupling at 2^15 cells. That is the right trade when a bucket holds a 256th of the rows,
/// but here a bucket holds all of them, so keep the single-level grower.
template <typename Key, typename Mapped, typename Hash = DefaultHash<Key>>
using JoinHashMap
    = TwoLevelHashMap<Key, Mapped, Hash, HashTableGrowerWithPrecalculation<>, HashTableAllocator, HashMapTable, BITS_FOR_BUCKET>;

template <typename Key, typename Mapped, typename Hash = DefaultHash<Key>>
using JoinHashMapWithSavedHash = TwoLevelHashMapWithSavedHash<
    Key,
    Mapped,
    Hash,
    HashTableGrowerWithPrecalculation<>,
    HashTableAllocator,
    HashMapTable,
    BITS_FOR_BUCKET>;

/// The direct-addressed maps (`key8`/`key16`, and `range8_key32`..`range18_key64` after the
/// post-build range conversion). Same bucket protocol, same `NUM_BUCKETS`, but the buckets route
/// into one flat `buf[key]` buffer rather than owning their cells - see `FixedRangeStorage`.
template <typename Key, typename Mapped, size_t size_bits = sizeof(Key) * 8>
using JoinFixedHashMap = PartitionedFixedHashMap<Key, Mapped, size_bits>;

static_assert(BucketPartitionedMap<JoinHashMap<UInt64, RowRefList>>);
static_assert(BucketPartitionedMap<JoinHashMapWithSavedHash<std::string_view, RowRefList>>);
static_assert(BucketPartitionedMap<JoinFixedHashMap<UInt8, RowRefList>>);
static_assert(BucketPartitionedMap<JoinFixedHashMap<UInt64, RowRefList, 18>>);

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
class HashJoin : public IInMemoryHashJoin
{
public:
    HashJoin(
        std::shared_ptr<TableJoin> table_join_,
        SharedHeader right_sample_block,
        bool any_take_last_row_ = false,
        size_t reserve_num_ = 0,
        const String & instance_id_ = "",
        const StatsCollectingParams & stats_collecting_params_ = {});

    ~HashJoin() override;

    std::string getName() const override { return "UnifiedHashJoin"; }

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

    void checkTypesOfKeys(const Block & block) const override;

    using IJoin::joinBlock;

    /** Join data from the map (that was previously built by calls to addBlockToJoin) to the block with data from "left" table.
      * Could be called from different threads in parallel.
      */
    JoinResultPtr joinBlock(Block block) override;

    /// Check joinGet arguments and infer the return type.
    DataTypePtr joinGetCheckAndGetReturnType(const DataTypes & data_types, const String & column_name, bool or_null) const;

    /// Used by joinGet function that turns StorageJoin into a dictionary.
    ColumnWithTypeAndName joinGet(const Block & block, const Block & block_with_columns_to_add) const;

    bool isFilled() const override { return from_storage_join; }

    /** The right side may be filled from several threads at once. Unlike `ConcurrentHashJoin`, which
      * splits the right side across one hash map per thread, there is still a single hash map here and
      * `build_mutex` serializes the insertion into it. Only the per-block preparation - materialization
      * of the key columns and of the columns to store - runs outside the critical section.
      */
    bool supportParallelJoin() const override { return true; }

    void setTotals(const Block & block) override;
    const Block & getTotals() const override;

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
    #define UNIFIED_APPLY_FOR_JOIN_VARIANTS(M) \
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
    #define UNIFIED_APPLY_FOR_JOIN_VARIANTS_LIMITED(M) \
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
            UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
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
        std::shared_ptr<JoinFixedHashMap<UInt8, Mapped>>                      key8;
        std::shared_ptr<JoinFixedHashMap<UInt16, Mapped>>                     key16;
        std::shared_ptr<JoinHashMap<UInt32, Mapped, HashCRC32<UInt32>>>       key32;
        std::shared_ptr<JoinHashMap<UInt64, Mapped, HashCRC32<UInt64>>>       key64;
        std::shared_ptr<JoinHashMapWithSavedHash<std::string_view, Mapped>>          key_string;
        std::shared_ptr<JoinHashMapWithSavedHash<std::string_view, Mapped>>          key_fixed_string;
        std::shared_ptr<JoinHashMap<UInt32, Mapped, HashCRC32<UInt32>>>       keys32;
        std::shared_ptr<JoinHashMap<UInt64, Mapped, HashCRC32<UInt64>>>       keys64;
        std::shared_ptr<JoinHashMap<UInt128, Mapped, UInt128HashCRC32>>       keys128;
        std::shared_ptr<JoinHashMap<UInt256, Mapped, UInt256HashCRC32>>       keys256;
        std::shared_ptr<JoinHashMap<UInt128, Mapped, UInt128TrivialHash>>     hashed;
        std::shared_ptr<JoinHashMapWithSavedHash<std::string_view, Mapped>>  low_cardinality_key_string;
        std::shared_ptr<JoinHashMapWithSavedHash<std::string_view, Mapped>>  low_cardinality_key_fixed_string;
        std::shared_ptr<JoinFixedHashMap<UInt32, Mapped, 8>>                  range8_key32;
        std::shared_ptr<JoinFixedHashMap<UInt32, Mapped, 16>>                 range16_key32;
        std::shared_ptr<JoinFixedHashMap<UInt32, Mapped, 17>>                 range17_key32;
        std::shared_ptr<JoinFixedHashMap<UInt32, Mapped, 18>>                 range18_key32;
        std::shared_ptr<JoinFixedHashMap<UInt64, Mapped, 8>>                  range8_key64;
        std::shared_ptr<JoinFixedHashMap<UInt64, Mapped, 16>>                 range16_key64;
        std::shared_ptr<JoinFixedHashMap<UInt64, Mapped, 17>>                 range17_key64;
        std::shared_ptr<JoinFixedHashMap<UInt64, Mapped, 18>>                 range18_key64;

        /// Every variant is a bucket-partitioned table taking the same `(num_buckets, size_hint)`,
        /// so there is one construction path rather than one per map family. The `static_assert`
        /// below is what keeps it that way.
        #define M(NAME) static_assert(BucketPartitionedMap<typename decltype(NAME)::element_type>);
            UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
        #undef M

        void create(Type which, size_t reserve)
        {
            switch (which)
            {
            #define M(NAME)                                                                        \
                case Type::NAME:                                                                   \
                    NAME = std::make_shared<typename decltype(NAME)::element_type>(NUM_BUCKETS, reserve); \
                    break;

                UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
            #undef M
            }
        }

        size_t getTotalRowCount(Type which) const
        {
            switch (which)
            {
            #define M(NAME) \
                case Type::NAME: return NAME ? NAME->size() : 0;
                UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
            #undef M
            }
        }

        size_t getTotalByteCountImpl(Type which) const
        {
            switch (which)
            {
            #define M(NAME) \
                case Type::NAME: return NAME ? NAME->getBufferSizeInBytes() : 0;
                UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
            #undef M
            }
        }

        size_t getBufferSizeInCells(Type which) const
        {
            switch (which)
            {
            #define M(NAME) \
                case Type::NAME: return NAME ? NAME->getBufferSizeInCells() : 0;
                UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
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

        /// Resolves RowRef::block_no to the stored block. Block numbers are assigned as blocks
        /// arrive, so they stay unique across the build threads that share this instance.
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
    BlocksList releaseJoinedBlocks(bool restructure = false) override;

    /// Modify right block (update structure according to sample block) to save it in block list
    static Block prepareRightBlock(const Block & block, const Block & saved_block_sample_);
    Block prepareRightBlock(const Block & block) const override;

    const Block & savedBlockSample() const override { return data->sample_block; }

    bool isUsed(size_t off) const;
    bool isUsed(UInt32 block_no, size_t row_idx) const;

    void debugKeys() const;

    void shrinkStoredBlocksToFit(size_t & total_bytes_in_join, bool force_optimize = false);

    void setMaxJoinedBlockRows(size_t value) { max_joined_block_rows = value; }
    void setMaxJoinedBlockBytes(size_t value) { max_joined_block_bytes = value; }

    void materializeColumnsFromLeftBlock(Block & block) const;
    Block materializeColumnsFromRightBlock(Block block) const;

    size_t getAndSetRightTableKeys() const override;

    const std::vector<Sizes> & getKeySizes() const { return key_sizes; }

    bool enableLazyColumnsReplication() const { return enable_lazy_columns_replication; }
    bool enableSoftwarePrefetch() const { return enable_prefetch; }

    void setEnableLazyColumnsIndexing(bool value) override { enable_lazy_columns_indexing = value; }

    static bool isUsedByAnotherAlgorithm(const TableJoin & table_join);
    static bool canRemoveColumnsFromLeftBlock(const TableJoin & table_join);

private:
    friend class NotJoinedHash;
    friend class DB::JoinSource;

    template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
    friend class HashJoinMethods;

    /// The build implementation. `selector` restricts insertion to a subset of the block's rows;
    /// it is narrowed for ASOF joins to drop rows with a NULL ASOF key.
    bool addBlockToJoin(const Block & block, ScatteredBlock::Selector selector, bool check_limits);

    std::shared_ptr<TableJoin> table_join;
    JoinKind kind;
    JoinStrictness strictness;

    /// This join was created from StorageJoin and it is already filled.
    bool from_storage_join = false;

    const bool any_take_last_row; /// Overwrite existing values when encountering the same key again
    const size_t reserve_num;
    const String instance_id;
    std::optional<TypeIndex> asof_type;
    const ASOFJoinInequality asof_inequality;

    /// Serializes the build phase, which may run on several threads. Guards `data`, `used_flags`,
    /// `all_values_unique`, `memory_usage_before_adding_blocks` and `shrink_blocks`. Joining a block does
    /// not take it: once the build phase is over the hash table is immutable and the used flags are atomic.
    mutable std::mutex build_mutex;

    /// Guards the totals block, which every parallel `FillingRightJoinSideTransform` writes.
    mutable std::mutex totals_mutex;

    /// Right table data. StorageJoin shares it between many Join objects.
    /// Flags that indicate that particular row already used in join.
    /// Flag is stored for every record in hash map.
    /// Number of this flags equals to hashtable buffer size (plus one for zero value).
    /// Changes in hash table broke correspondence,
    /// so we must guarantee constantness of hash table during HashJoin lifetime (using method setLock)
    mutable std::shared_ptr<JoinStuff::JoinUsedFlags> used_flags;
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

    /// Counterparts of the public accessors for callers that already hold `build_mutex`,
    /// and for the probe path, which must not contend on it.
    size_t getTotalRowCountUnlocked() const;
    size_t getTotalByteCountUnlocked() const;

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

using UnifiedHashJoin = Unified::HashJoin;

}
