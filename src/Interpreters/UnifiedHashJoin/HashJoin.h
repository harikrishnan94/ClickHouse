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
#include <Common/CacheLine.h>
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

/// A bucket is the unit of lock granularity for the build. A build thread scatters its block's rows
/// by slot once, then inserts each group while holding only that slot's lock, so several threads
/// mutate one shared map concurrently - this is what lets `unified_hash` subsume `parallel_hash`
/// without replicating the whole join per thread.
///
/// The bucket count is part of the map's type, as it is for `HashJoin`: a serial build gets the
/// one-bucket map and a parallel build the 256-bucket one, chosen by `Type` exactly where
/// `HashJoin` chooses between `key64` and `two_level_key64`. One bucket is the serial case of the
/// same template rather than a separate kind of table - routing folds to a constant and the single
/// sub-table is stored inline, so those maps are single-table maps in everything but their name.
constexpr Int32 BITS_FOR_BUCKET_SERIAL = 0;
constexpr Int32 BITS_FOR_BUCKET_TWO_LEVEL = DEFAULT_BITS_FOR_BUCKET;

/// Hash-table bucket count of the two-level maps, and the cap on the slot count that sizes locks
/// and arenas: more HT buckets improve key-space partitioning and cache locality of each sub-table,
/// while the slot count only needs to track build concurrency.
constexpr size_t NUM_HASH_TABLE_BUCKETS = 1ull << BITS_FOR_BUCKET_TWO_LEVEL;

/// Whether a right side built by `max_threads` threads uses the two-level maps.
inline bool useTwoLevelMaps(size_t max_threads)
{
    return max_threads > 1;
}

/// Lock/arena slot count for a join whose right side is built by `max_threads` threads. A power of
/// two, as in `ConcurrentHashJoin`, capped at `NUM_HASH_TABLE_BUCKETS` so every slot owns at least
/// one HT bucket; one slot for a serial build.
size_t slotCountForThreads(size_t max_threads);

/// Slot that owns HT bucket `bucket`. Both counts are powers of two, so this is a mask.
inline size_t slotForBucket(size_t bucket, size_t num_slots)
{
    return bucket & (num_slots - 1);
}

/// The lock guarding one slot: every HT bucket mapped to that slot in every clause's map, and that
/// slot's arena. One lock covers all clauses because the arenas are per slot and shared between
/// them, so two threads working on different clauses of the same slot would otherwise race on the
/// arena. Padded, because two threads inserting into neighbouring slots would otherwise contend on
/// the cache line holding both mutexes even though they never contend on the lock itself.
struct alignas(DB::CH_CACHE_LINE_SIZE) BucketLock
{
    std::mutex mutex;
};

/// What inserting one block's rows into one clause's map produced. A bucket-parallel build makes
/// one insert call per bucket and reduces these across the buckets, so every field has to be
/// reducible: OR, AND and addition respectively.
struct BuildResult
{
    /// Whether this block became reachable at all - either a key was inserted, or a NULL key was
    /// seen that a RIGHT/FULL join still has to emit. If no clause sets it the block is dropped.
    bool is_inserted = false;
    /// Cleared as soon as one key is seen twice. Uniqueness is a property of the whole right side,
    /// so this is only meaningful once reduced across every bucket and every block.
    bool all_values_unique = true;
    /// Keys the map did not have before.
    size_t new_keys = 0;
};

/// `TwoLevelHashTableGrower` - the same growth policy `HashJoin` uses - so that a bucket holding a
/// given number of keys ends up with the same buffer as its `parallel_hash` counterpart. It stops
/// quadrupling at 2^15 cells and doubles from there, which matters now that a multi-threaded build
/// always splits the rows over `NUM_HASH_TABLE_BUCKETS` buckets: quadrupling all the way overshoots
/// by one step for a bucket of that size and doubles the buffer, which is paid for in page faults
/// during the build rather than in probe time.
template <typename Key, typename Mapped, typename Hash = DefaultHash<Key>>
using JoinHashMap
    = TwoLevelHashMap<Key, Mapped, Hash, TwoLevelHashTableGrower<>, HashTableAllocator, HashMapTable, BITS_FOR_BUCKET_SERIAL>;

template <typename Key, typename Mapped, typename Hash = DefaultHash<Key>>
using JoinHashMapWithSavedHash = TwoLevelHashMapWithSavedHash<
    Key,
    Mapped,
    Hash,
    TwoLevelHashTableGrower<>,
    HashTableAllocator,
    HashMapTable,
    BITS_FOR_BUCKET_SERIAL>;

/// The 256-bucket counterparts, for a parallel build. These are the same instantiations
/// `HashJoin::MapsTemplate` names `two_level_*`.
template <typename Key, typename Mapped, typename Hash = DefaultHash<Key>>
using TwoLevelJoinHashMap
    = TwoLevelHashMap<Key, Mapped, Hash, TwoLevelHashTableGrower<>, HashTableAllocator, HashMapTable, BITS_FOR_BUCKET_TWO_LEVEL>;

template <typename Key, typename Mapped, typename Hash = DefaultHash<Key>>
using TwoLevelJoinHashMapWithSavedHash = TwoLevelHashMapWithSavedHash<
    Key,
    Mapped,
    Hash,
    TwoLevelHashTableGrower<>,
    HashTableAllocator,
    HashMapTable,
    BITS_FOR_BUCKET_TWO_LEVEL>;

/// The direct-addressed maps (`key8`/`key16`, and `range8_key32`..`range18_key64` after the
/// post-build range conversion). Same bucket protocol, same bucket count, but the buckets route
/// into one flat `buf[key]` buffer rather than owning their cells - see `FixedRangeStorage`.
template <typename Key, typename Mapped, size_t size_bits = sizeof(Key) * 8>
using JoinFixedHashMap = PartitionedFixedHashMap<Key, Mapped, size_bits>;

static_assert(BucketPartitionedMap<JoinHashMap<UInt64, RowRefList>>);
static_assert(BucketPartitionedMap<JoinHashMapWithSavedHash<std::string_view, RowRefList>>);
static_assert(BucketPartitionedMap<TwoLevelJoinHashMap<UInt64, RowRefList>>);
static_assert(BucketPartitionedMap<TwoLevelJoinHashMapWithSavedHash<std::string_view, RowRefList>>);
static_assert(BucketPartitionedMap<JoinFixedHashMap<UInt8, RowRefList>>);
static_assert(BucketPartitionedMap<JoinFixedHashMap<UInt64, RowRefList, 18>>);

/// The serial maps must cost no more than a single-level table does: one sub-table, stored inline.
static_assert(JoinHashMap<UInt64, RowRefList>::bucketCount() == 1);
static_assert(TwoLevelJoinHashMap<UInt64, RowRefList>::bucketCount() == NUM_HASH_TABLE_BUCKETS);

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
        const StatsCollectingParams & stats_collecting_params_ = {},
        size_t max_threads_ = 1);

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
        /// `max_threads` is carried over because it only sizes the buckets, which is this join's
        /// analogue of the `use_two_level_maps` the baseline's `clone` also settles here.
        return std::make_shared<HashJoin>(
            table_join_, right_sample_block_, any_take_last_row, reserve_num, instance_id, StatsCollectingParams{}, max_threads);
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

    /** The right side may be filled from several threads at once, into one shared map per clause.
      * Unlike `ConcurrentHashJoin`, which gives each thread its own `HashJoin` and merges them
      * afterwards, a build thread here routes its block's rows to buckets and inserts each group
      * holding only that bucket's lock, so there is nothing to merge and a block is stored once
      * rather than once per thread that received a row from it.
      */
    bool supportParallelJoin() const override { return true; }

    /** The non-joined rows of a RIGHT/FULL join are emitted by several streams, each taking the
      * buckets it owns. `ConcurrentHashJoin` does the same on a two-level map; the maps here are
      * always two-level, so the same partitioning applies.
      */
    bool supportParallelNonJoinedBlocksProcessing() const override;

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

    IBlocksStreamPtr getNonJoinedBlocks(
        const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size,
        size_t stream_idx, size_t num_streams) const override;

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

    /// The open-addressing maps of a serial build: one bucket each.
    #define UNIFIED_APPLY_FOR_SINGLE_LEVEL_JOIN_VARIANTS(M) \
        M(key32)                       \
        M(key64)                       \
        M(key_string)                  \
        M(key_fixed_string)            \
        M(keys32)                      \
        M(keys64)                      \
        M(keys128)                     \
        M(keys256)                     \
        M(hashed)                      \
        M(low_cardinality_key_string)  \
        M(low_cardinality_key_fixed_string)

    /// Their 256-bucket counterparts, chosen for a parallel build. Same key kinds, same key getters,
    /// same names as `HashJoin`'s.
    #define UNIFIED_APPLY_FOR_TWO_LEVEL_JOIN_VARIANTS(M) \
        M(two_level_key32)                       \
        M(two_level_key64)                       \
        M(two_level_key_string)                  \
        M(two_level_key_fixed_string)            \
        M(two_level_keys32)                      \
        M(two_level_keys64)                      \
        M(two_level_keys128)                     \
        M(two_level_keys256)                     \
        M(two_level_hashed)                      \
        M(two_level_low_cardinality_key_string)  \
        M(two_level_low_cardinality_key_fixed_string)

    /// The direct-addressed maps. One flat `buf[key]` buffer whatever the bucket count, so they have
    /// no serial counterpart: their buckets route a lock and nothing else.
    #define UNIFIED_APPLY_FOR_FIXED_JOIN_VARIANTS(M) \
        M(key8)                        \
        M(key16)                       \
        M(range8_key32)                \
        M(range16_key32)               \
        M(range17_key32)               \
        M(range18_key32)               \
        M(range8_key64)                \
        M(range16_key64)               \
        M(range17_key64)               \
        M(range18_key64)

    /// Different types of keys for maps.
    #define UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)          \
        UNIFIED_APPLY_FOR_FIXED_JOIN_VARIANTS(M)        \
        UNIFIED_APPLY_FOR_SINGLE_LEVEL_JOIN_VARIANTS(M) \
        UNIFIED_APPLY_FOR_TWO_LEVEL_JOIN_VARIANTS(M)

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
        M(low_cardinality_key_fixed_string)    \
        M(two_level_key32)                     \
        M(two_level_key64)                     \
        M(two_level_key_string)                \
        M(two_level_key_fixed_string)          \
        M(two_level_low_cardinality_key_string)\
        M(two_level_low_cardinality_key_fixed_string)

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
            case Type::two_level_low_cardinality_key_string:
            case Type::two_level_low_cardinality_key_fixed_string:
                return true;
            default:
                return false;
        }
    }

    /// True for the 256-bucket maps, which is what a parallel build uses.
    static bool isTwoLevelType(Type type)
    {
        switch (type)
        {
            #define M(NAME) \
                case Type::NAME: \
                    return true;
                UNIFIED_APPLY_FOR_TWO_LEVEL_JOIN_VARIANTS(M)
            #undef M
            default:
                return false;
        }
    }

    /// The 256-bucket counterpart of a key kind, for a parallel build. The direct-addressed maps are
    /// their own counterpart - see `UNIFIED_APPLY_FOR_FIXED_JOIN_VARIANTS`.
    static Type toTwoLevelType(Type type)
    {
        switch (type)
        {
            #define M(NAME) \
                case Type::NAME: \
                    return Type::two_level_##NAME;
                UNIFIED_APPLY_FOR_SINGLE_LEVEL_JOIN_VARIANTS(M)
            #undef M
            default:
                return type;
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
        std::shared_ptr<TwoLevelJoinHashMap<UInt32, Mapped, HashCRC32<UInt32>>>       two_level_key32;
        std::shared_ptr<TwoLevelJoinHashMap<UInt64, Mapped, HashCRC32<UInt64>>>       two_level_key64;
        std::shared_ptr<TwoLevelJoinHashMapWithSavedHash<std::string_view, Mapped>>          two_level_key_string;
        std::shared_ptr<TwoLevelJoinHashMapWithSavedHash<std::string_view, Mapped>>          two_level_key_fixed_string;
        std::shared_ptr<TwoLevelJoinHashMap<UInt32, Mapped, HashCRC32<UInt32>>>       two_level_keys32;
        std::shared_ptr<TwoLevelJoinHashMap<UInt64, Mapped, HashCRC32<UInt64>>>       two_level_keys64;
        std::shared_ptr<TwoLevelJoinHashMap<UInt128, Mapped, UInt128HashCRC32>>       two_level_keys128;
        std::shared_ptr<TwoLevelJoinHashMap<UInt256, Mapped, UInt256HashCRC32>>       two_level_keys256;
        std::shared_ptr<TwoLevelJoinHashMap<UInt128, Mapped, UInt128TrivialHash>>     two_level_hashed;
        std::shared_ptr<TwoLevelJoinHashMapWithSavedHash<std::string_view, Mapped>>  two_level_low_cardinality_key_string;
        std::shared_ptr<TwoLevelJoinHashMapWithSavedHash<std::string_view, Mapped>>  two_level_low_cardinality_key_fixed_string;
        std::shared_ptr<JoinFixedHashMap<UInt32, Mapped, 8>>                  range8_key32;
        std::shared_ptr<JoinFixedHashMap<UInt32, Mapped, 16>>                 range16_key32;
        std::shared_ptr<JoinFixedHashMap<UInt32, Mapped, 17>>                 range17_key32;
        std::shared_ptr<JoinFixedHashMap<UInt32, Mapped, 18>>                 range18_key32;
        std::shared_ptr<JoinFixedHashMap<UInt64, Mapped, 8>>                  range8_key64;
        std::shared_ptr<JoinFixedHashMap<UInt64, Mapped, 16>>                 range16_key64;
        std::shared_ptr<JoinFixedHashMap<UInt64, Mapped, 17>>                 range17_key64;
        std::shared_ptr<JoinFixedHashMap<UInt64, Mapped, 18>>                 range18_key64;

        /// Every variant is a bucket-partitioned table taking the same `(size_hint)`, so there is one
        /// construction path rather than one per map family. The `static_assert` below is what keeps
        /// it that way.
        #define M(NAME) static_assert(BucketPartitionedMap<typename decltype(NAME)::element_type>);
            UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
        #undef M

        /// A reserve taken from the statistics of a previous, larger run of the same query can be
        /// far bigger than the memory this one is allowed to use, and it is claimed before
        /// `SpillingHashJoin` ever gets to check its threshold. `max_reserve_bytes` bounds it; zero
        /// means unbounded. Buffers run at a 0.5 load factor and round up to a power of two, so a
        /// reserved entry can cost up to four cells, and we aim at half the budget so that the
        /// spill trigger and the hand-over peak still fit under it - the same arithmetic as
        /// `reserveSpaceInHashMaps` in `ConcurrentHashJoin`.
        template <typename Table>
        static size_t clampReserve(size_t reserve, size_t max_reserve_bytes)
        {
            if (!max_reserve_bytes)
                return reserve;
            /// Direct-addressed tables size their buffer from the key range, not from the reserve.
            if constexpr (requires { sizeof(typename Table::cell_type); })
                return std::min(reserve, max_reserve_bytes / (8 * sizeof(typename Table::cell_type)));
            else
                return reserve;
        }

        void create(Type which, size_t reserve, size_t max_reserve_bytes = 0)
        {
            switch (which)
            {
            #define M(NAME)                                                                        \
                case Type::NAME:                                                                   \
                {                                                                                  \
                    using Table = typename decltype(NAME)::element_type;                           \
                    NAME = std::make_shared<Table>(clampReserve<Table>(reserve, max_reserve_bytes)); \
                    break;                                                                         \
                }

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

        /// The bucket count of the chosen variant. Every variant's is a compile-time constant; this
        /// picks the one the runtime `Type` names, for code that is not templated on the map.
        size_t getBucketCount(Type which) const
        {
            switch (which)
            {
            #define M(NAME) \
                case Type::NAME: return decltype(NAME)::element_type::bucketCount();
                UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
            #undef M
            }
        }

        /// Bytes of one bucket's own buffer. A storage whose buckets share one flat buffer reports
        /// that whole buffer for every bucket, so this is only meaningful as a delta taken around
        /// an insert into that bucket - such a buffer has a fixed capacity, so its delta is zero.
        size_t getBucketBufferSizeInBytes(Type which, size_t bucket) const
        {
            switch (which)
            {
            #define M(NAME) \
                case Type::NAME: return NAME ? NAME->impls[bucket].getBufferSizeInBytes() : 0;
                UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
            #undef M
            }
        }

        /// Sum up the buckets' capacities into the prefix the global cell offsets are numbered from.
        /// The build does not maintain it, precisely so that a bucket-parallel build writes nothing
        /// shared between buckets, so this must run once after the last insert and before any probe.
        void computeBucketPrefix(Type which) const
        {
            switch (which)
            {
            #define M(NAME) \
                case Type::NAME: if (NAME) NAME->computeBucketPrefix(); break;
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
        explicit RightTableData(size_t slots)
            : num_slots(slots)
        {
            pools.reserve(slots);
            for (size_t i = 0; i < slots; ++i)
                pools.push_back(std::make_unique<Arena>());
        }

        /// How many lock/arena slots this build uses. Lives beside the maps rather than on the join,
        /// because `StorageJoin` hands one `RightTableData` to joins created with a different
        /// `max_threads`, and the locks must match the maps, not the join that took them over. HT
        /// buckets map onto slots by `slotForBucket`.
        const size_t num_slots;

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

        /// Additional data - strings for string keys and continuation elements of single-linked
        /// lists of references to rows.
        ///
        /// One arena per slot. `Arena` is a plain bump allocator with no synchronization, so a
        /// single shared one could not be filled by several build threads at once; a slot's arena
        /// is covered by that slot's lock, exactly like the HT buckets mapped to it. Splitting is
        /// safe because the only allocations here are `keyHolderPersistKey` for string keys and
        /// `RowRefList` continuation nodes, and neither needs allocations of different keys to be
        /// contiguous or to be rolled back.
        std::vector<std::unique_ptr<Arena>> pools;

        Arena & poolForBucket(size_t bucket) { return *pools[slotForBucket(bucket, num_slots)]; }

        size_t poolsAllocatedBytes() const
        {
            size_t res = 0;
            for (const auto & pool : pools)
                res += pool->allocatedBytes();
            return res;
        }

        /// Bytes of the stored blocks and of the nullmaps. Both are per-block quantities and are
        /// only ever written under `blocks_mutex`, which is what lets `doDebugAsserts` recompute
        /// and compare them; they are atomic so that the size-limit check can read them without
        /// taking that lock.
        std::atomic<size_t> allocated_size = 0;
        std::atomic<size_t> nullmaps_allocated_size = 0;

        /// Number of rows of right table to join
        std::atomic<size_t> rows_to_join = 0;
        /// Number of keys of right table to join. Maintained incrementally from the number of keys
        /// each insert added, rather than by asking the maps: during a bucket-parallel build the
        /// buckets are being mutated, so summing their sizes would both race and cost O(buckets)
        /// per block.
        std::atomic<size_t> keys_to_join = 0;
        /// Bytes owned by the buckets - their map buffers plus their arenas. Seeded with the maps'
        /// initial size and then advanced by the delta each block's inserts produced, measured
        /// bucket by bucket under that bucket's slot lock, for the same reason as `keys_to_join`.
        /// The deltas of one block's buckets are summed by the inserting thread and added here once,
        /// after it has let go of the last slot lock (see `insertIntoBuckets`).
        std::atomic<size_t> bucket_bytes = 0;

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
            const size_t keys = keys_to_join.load(std::memory_order_relaxed);
            if (keys == 0)
                return 0;
            return rows_to_join.load(std::memory_order_relaxed) / keys;
        }
    };

    /// For INNER/LEFT ALL JOINs, if the right side has no duplicates inside the join key columns,
    /// we can switch from ALL to RightAny strictness for better performance. Only ever goes from
    /// true to false, so build threads can clear it with a relaxed store and no coordination.
    std::atomic<bool> all_values_unique = true;
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

    /// How many threads may call `addBlockToJoin` concurrently, and the slot count derived from it.
    /// Locks and arenas are sized by `num_slots`, which may be smaller than the maps' bucket count:
    /// several HT buckets then share one slot.
    const size_t max_threads;
    const size_t num_slots;

    std::optional<TypeIndex> asof_type;
    const ASOFJoinInequality asof_inequality;

    /// The build phase runs on several threads and is split across two levels of locking.
    ///
    /// `blocks_mutex` covers the per-BLOCK bookkeeping: the stored-block list and its block-number
    /// index, the nullmaps, their byte counters, the pending per-row used flags, and the
    /// shrink decision. All of it is O(1) per block, so holding one lock for it costs nothing -
    /// unlike the rows, which are the actual work.
    ///
    /// `bucket_locks[slot]` covers the rows of every HT bucket mapped to that slot: a build thread
    /// routes its block's rows to HT buckets once and then inserts each group holding only that
    /// slot's lock for the whole group, so threads inserting into HT buckets of different slots run
    /// concurrently. The lock is taken per group and not per row - a per-row acquisition costs more
    /// than the insert it protects and makes the build scale negatively with the thread count.
    ///
    /// A thread holds at most one slot lock at a time and never holds one across `blocks_mutex`,
    /// so there is no lock order to get wrong. Joining a block takes neither: once the build phase
    /// is over the hash table is immutable and the used flags are atomic.
    mutable std::mutex blocks_mutex;

    mutable std::vector<BucketLock> bucket_locks;

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

    /// When tracked memory consumption is more than a threshold, we will shrink to fit stored
    /// blocks. Set under `blocks_mutex`, but read without it on the per-block path, where a stale
    /// `false` only means one more block is stored unshrunk before the next one shrinks it.
    std::atomic<bool> shrink_blocks = false;
    /// The query's tracked memory usage as it stood before this join stored its first block. Atomic
    /// so that the per-block path can sample it - once per join, on whichever build thread gets
    /// there first - without taking `blocks_mutex`, which the reading does not need.
    std::atomic<Int64> memory_usage_before_adding_blocks = 0;

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

    /// Counterparts of the public accessors for callers that already hold `blocks_mutex`,
    /// and for the probe path, which must not contend on it.
    size_t getTotalRowCountUnlocked() const;
    size_t getTotalByteCountUnlocked() const;

    /// Recompute `data->bucket_bytes` from the maps and arenas as they now stand. The running sum
    /// the build maintains cannot survive the post-build surgery that replaces a whole map, so
    /// call this after anything that does.
    void recomputeBucketBytes();

    void dataMapInit(MapsVariant & map);

    /// How many entries to preallocate across all buckets: the caller's explicit reserve if there
    /// is one, otherwise the size the hash-table statistics recorded for this query shape.
    size_t sizeHintForMaps() const;

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

    template <bool is_signed, typename Key, typename SourcePtr, typename MapsTemplate>
    void tryConvertToFixedHashMapImpl(MapsTemplate & maps, SourcePtr & source_ptr);

    /// Publishes everything the probe reads that the build deliberately left stale, so that a
    /// bucket-parallel build never writes state shared between buckets. Must run after the last
    /// insert and before the first probe.
    void freezeMapsForProbing();

    void reinitUsedFlags();

    void doDebugAsserts() const;
};
}

using UnifiedHashJoin = Unified::HashJoin;

}
