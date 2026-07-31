#include <Columns/IColumn.h>
#include <Core/Names.h>
#include <Core/NamesAndTypes.h>
#include <DataTypes/IDataType.h>
#include <DataTypes/Serializations/ISerialization.h>
#include <Interpreters/ConcurrentHashJoin.h>
#include <Interpreters/HashJoin/ScatteredBlock.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/PreparedSets.h>
#include <Interpreters/TableJoin.h>
#include <Parsers/ASTSelectQuery.h>
#include <Parsers/IAST_fwd.h>
#include <Storages/SelectQueryInfo.h>
#include <Common/CurrentThread.h>
#include <Common/ElapsedTimeProfileEventIncrement.h>
#include <Common/Exception.h>
#include <Common/ProfileEvents.h>
#include <Common/ThreadPool.h>
#include <Common/AllocatorWithMemoryTracking.h>
#include <Common/setThreadName.h>
#include <Common/ThreadGroupSwitcher.h>

#include <Interpreters/HashJoin/AddedColumns.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/HashJoin/JoinProbeScratch.h>
#include <Interpreters/HashJoin/JoinSlotRouting.h>
#include <Interpreters/HashJoin/KeyGetter.h>
#include <DataTypes/NullableUtils.h>
#include <base/defines.h>
#include <base/types.h>

#include <algorithm>
#include <bit>
#include <numeric>
#include <deque>
#include <iterator>
#include <thread>
#include <tuple>
#include <utility>

using namespace DB;

namespace ProfileEvents
{
extern const Event HashJoinPreallocatedElementsInHashTables;
extern const Event ConcurrentHashJoinBuildMicroseconds;
extern const Event ConcurrentHashJoinBuildDispatchMicroseconds;
extern const Event ConcurrentHashJoinBuildInsertMicroseconds;
extern const Event ConcurrentHashJoinBuildMergeMicroseconds;
extern const Event ConcurrentHashJoinProbeMicroseconds;
extern const Event ConcurrentHashJoinProbeDispatchMicroseconds;
extern const Event ConcurrentHashJoinProbeLookupMicroseconds;
}

namespace CurrentMetrics
{
extern const Metric ConcurrentHashJoinPoolThreads;
extern const Metric ConcurrentHashJoinPoolThreadsActive;
extern const Metric ConcurrentHashJoinPoolThreadsScheduled;
}

namespace
{

void updateStatistics(const auto & hash_joins, const DB::StatsCollectingParams & params)
{
    if (!params.isCollectionAndUseEnabled())
        return;

    /// Each `HashJoin` instance ("slot") holds a disjoint subset of the keys, so the whole join's
    /// size is the sum over the slots.
    size_t ht_size = 0;
    size_t source_rows = 0;
    for (const auto & hash_join : hash_joins)
    {
        ht_size += hash_join->data->getTotalRowCount();
        /// The joined data is released when the blocks are handed over to GraceHashJoin.
        if (const auto & joined_data = hash_join->data->getJoinedData())
            source_rows += joined_data->rows_to_join;
    }

    if (ht_size)
        DB::getHashTablesStatistics<DB::HashJoinEntry>().update({.ht_size = ht_size, .source_rows = source_rows}, params);
}

UInt32 toPowerOfTwo(UInt32 x)
{
    if (x <= 1)
        return 1;
    return static_cast<UInt32>(1) << (32 - std::countl_zero(x - 1));
}

void reserveSpaceInHashMaps(
    HashJoin & hash_join,
    const StatsCollectingParams & stats_collecting_params,
    size_t slots,
    size_t external_join_threshold)
{
    auto hint = getSizeHint(stats_collecting_params);
    if (!hint)
        return;

    /// The size hint describes the whole join, and every `HashJoin` instance holds a disjoint
    /// subset of the keys, so each slot reserves its `1/slots` share.
    const size_t reserve_size = hint->ht_size;

    auto reserve_space = [&](auto & map_ptr)
    {
        using Map = typename std::remove_cvref_t<decltype(map_ptr)>::element_type;
        /// Fixed-size maps (e.g. FixedHashMap) are allocated at full capacity and have nothing to reserve.
        if constexpr (requires(Map & map) { map.reserve(size_t{}); })
        {
            size_t actual_reserve_size = reserve_size;

            /// When a `SpillingHashJoin` wraps us, `external_join_threshold` is the auto-spill memory cap.
            /// Statistics-driven preallocation can reserve many gigabytes up front based on a previous larger
            /// query, blowing past that cap before `SpillingHashJoin` ever runs its threshold check. We still
            /// want preallocation - just bounded by the memory budget. When running standalone
            /// (`external_join_threshold == 0`), the original full reserve is used.
            if (external_join_threshold > 0)
            {
                /// Hash table buffers run at ~0.5 load factor (`maxFill = bufSize / 2`), so each
                /// stored entry consumes 2 cells of capacity, and `bufSize` is then rounded up to
                /// the next power of two - a factor of up to 4x in the worst case. So each reserved
                /// entry occupies up to 4 × cell_size bytes of buffer. We keep total preallocated
                /// bytes (summed across all slots) under `threshold / 2`, leaving headroom for the
                /// eventual SpillingHashJoin trigger (also at `threshold / 2`) and for the conversion
                /// peak when handing data over to GraceHashJoin.
                constexpr size_t cell_size = sizeof(typename Map::cell_type);
                const size_t budget_entries = external_join_threshold / (8 * cell_size);
                actual_reserve_size = std::min(reserve_size, budget_entries);
            }

            map_ptr->reserve(actual_reserve_size / slots);
            ProfileEvents::increment(ProfileEvents::HashJoinPreallocatedElementsInHashTables, actual_reserve_size / slots);
        }
    };

    auto reserve_space_in_map = [&](auto & maps, HashJoin::Type type)
    {
        switch (type)
        {
        #define M(NAME)                    \
            case HashJoin::Type::NAME:     \
                reserve_space(maps.NAME);  \
                break;
            APPLY_FOR_JOIN_VARIANTS(M)
        #undef M
        }
    };

    const auto & right_data = hash_join.getJoinedData();
    std::visit([&](auto & maps) { reserve_space_in_map(maps, right_data->type); }, right_data->maps.at(0));
}
}

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int SET_SIZE_LIMIT_EXCEEDED;
}


ConcurrentHashJoin::ConcurrentHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    size_t slots_,
    SharedHeader right_sample_block,
    const StatsCollectingParams & stats_collecting_params_,
    bool any_take_last_row_,
    size_t external_join_threshold_,
    bool use_two_level_key64_poc_)
    : table_join(table_join_)
    /// The requested count is honored (tests cover the single- and few-slot plans); production
    /// callers pass `max_slots` - see its comment for why the count is not thread-derived.
    , slots(toPowerOfTwo(std::min<UInt32>(static_cast<UInt32>(slots_), max_slots)))
    , any_take_last_row(any_take_last_row_)
    , pool(std::make_unique<ThreadPool>(
          CurrentMetrics::ConcurrentHashJoinPoolThreads,
          CurrentMetrics::ConcurrentHashJoinPoolThreadsActive,
          CurrentMetrics::ConcurrentHashJoinPoolThreadsScheduled,
          /*max_threads_*/ slots,
          /*max_free_threads_*/ 0,
          /*queue_size_*/ slots))
    , stats_collecting_params(stats_collecting_params_)
    , external_join_threshold(external_join_threshold_)
    /// 2x because probe lanes are pipeline streams, which are not guaranteed to stay below
    /// the slot count in every pipeline shape (see `IJoin::joinBlock`'s lane contract).
    , probe_scratch_by_lane(2 * slots)
    , use_two_level_key64_poc(use_two_level_key64_poc_)
{
    hash_joins.resize(slots);

    /// The bucketed map needs exactly `slots` buckets so that `computeDispatchSlotIds`'s slot id
    /// and the table's own `getBucketFromHash()` bucket id are the SAME number (see
    /// `computeHashRouteSlotIds`'s bucketed branch below) - a slot always exclusively owns the
    /// identically-numbered bucket, nothing is folded/interleaved.
    const size_t two_level_buckets = use_two_level_key64_poc ? slots : 1;

    try
    {
        for (size_t i = 0; i < slots; ++i)
        {
            pool->scheduleOrThrow(
                [&, i, thread_group = CurrentThread::getGroup()]()
                {
                    ThreadGroupSwitcher switcher(thread_group, ThreadName::CONCURRENT_JOIN);

                    auto inner_hash_join = std::make_shared<InternalHashJoin>();
                    /// Statistics-driven preallocation is done lazily on the first insert into the
                    /// slot (see `reserveSpaceInHashMaps`), so no space is reserved here.
                    inner_hash_join->data = std::make_unique<HashJoin>(
                        table_join_,
                        right_sample_block,
                        any_take_last_row_,
                        /*reserve_num_=*/0,
                        fmt::format("concurrent{}", i),
                        /*stats_collecting_params_=*/StatsCollectingParams{},
                        /*is_parallel_hash_slot=*/true,
                        two_level_buckets);
                    inner_hash_join->data->setMaxJoinedBlockRows(table_join->maxJoinedBlockRows());
                    inner_hash_join->data->setMaxJoinedBlockBytes(table_join->maxJoinedBlockBytes());
                    /// Opt the per-slot maps into the AMAC build-insert ring (see `AmacRing.h`):
                    /// only `parallel_hash` does this, because its per-slot builds are exactly
                    /// the large, insert-bound case the ring targets.
                    inner_hash_join->data->setAmacEnabled(true);
                    inner_hash_join->local_total_bytes = inner_hash_join->data->getTotalByteCount();
                    global_total_bytes.fetch_add(inner_hash_join->local_total_bytes, std::memory_order_relaxed);
                    hash_joins[i] = std::move(inner_hash_join);
                });
        }
        pool->wait();

        /// Share one `StoredColumnsIndex` across all slots so that `RowRef::block_no` is
        /// globally unique: any slot's probe can then resolve refs into any slot's stored
        /// blocks at emit time (the routed probe emits every slot's matches through one
        /// `AddedColumns`). Registration stays safe under the concurrent build because
        /// `StoredColumnsIndex::add` is mutex-protected.
        auto shared_index = hash_joins[0]->data->getJoinedData()->stored_columns_index;
        for (size_t i = 1; i < slots; ++i)
            hash_joins[i]->data->getJoinedData()->stored_columns_index = shared_index;

        slot_joins.reserve(slots);
        for (const auto & hash_join : hash_joins)
            slot_joins.push_back(hash_join->data.get());

        /// Plan-time header probes run before any build and need sized plan arrays.
        collectRoutedProbePlan();
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
        pool->wait();
        throw;
    }
}

/// Runs twice: at the end of the constructor, because the freshly built (empty) maps already
/// have valid buffers and plan-time header probes need sized plan arrays, and again from
/// `onBuildPhaseFinish`, when shrink-to-fit has finalized the buffers - the probes may trust
/// addresses and sizes only from that second collection.
void ConcurrentHashJoin::collectRoutedProbePlan()
{
    RoutedProbePlan plan;
    plan.map_by_slot.reserve(slots);
    plan.desc_by_slot.reserve(slots);
    plan.flags_by_slot.reserve(slots);

    size_t total_allocated_size = 0;
    size_t total_rows_to_join = 0;

    const auto type = hash_joins[0]->data->getJoinedData()->type;
    auto collect_map = [&](auto & maps)
    {
        switch (type)
        {
/// The descriptor exists only for the cursor-capable open-addressing map types; the rest keep
/// map-resolved lookups.
#define M(NAME) \
    case HashJoin::Type::NAME: { \
        const auto & map = *maps.NAME; \
        plan.map_by_slot.push_back(&map); \
        plan.total_map_bytes += map.getBufferSizeInBytes(); \
        if constexpr (requires { map.cursorCells(); }) \
            plan.desc_by_slot.push_back({map.cursorCells(), map.getBufferSizeInCells() - 1}); \
        break; \
    }
            APPLY_FOR_JOIN_VARIANTS(M)
#undef M
        }
    };

    for (const auto & hash_join : hash_joins)
    {
        HashJoin & join = *hash_join->data;
        auto & joined = *join.getJoinedData();
        plan.flags_by_slot.push_back(join.used_flags.get());
        total_allocated_size += joined.allocated_size;
        total_rows_to_join += joined.rows_to_join;
        std::visit(collect_map, joined.maps.at(0));
    }

    plan.avg_joined_bytes_per_row = total_allocated_size / std::max<size_t>(1, total_rows_to_join);
    routed_probe_plan = std::move(plan);
}

ConcurrentHashJoin::~ConcurrentHashJoin()
{
    /// No probe result may be alive here, so every parked scratch is owned by `probe_scratch_by_lane`.
    for (auto & parked : probe_scratch_by_lane)
        delete parked.load(std::memory_order_acquire);

    try
    {
        updateStatistics(hash_joins, stats_collecting_params);

        for (size_t i = 0; i < slots; ++i)
        {
            // Hash tables destruction may be very time-consuming.
            // Without the following code, they would be destroyed in the current thread (i.e. sequentially).
            pool->scheduleOrThrow(
                [join = std::move(hash_joins[i]), thread_group = CurrentThread::getGroup()]() mutable
                {
                    ThreadGroupSwitcher switcher(thread_group, ThreadName::CONCURRENT_JOIN);
                    join.reset();
                });
        }
        pool->wait();
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
        pool->wait();
    }
}

std::unique_ptr<JoinProbeScratch> ConcurrentHashJoin::acquireProbeScratch(size_t lane)
{
    /// Lane fast path.
    if (lane < probe_scratch_by_lane.size())
        if (JoinProbeScratch * parked = probe_scratch_by_lane[lane].exchange(nullptr, std::memory_order_acquire))
            return std::unique_ptr<JoinProbeScratch>(parked);

    {
        std::lock_guard lock(probe_scratch_mutex);
        if (!probe_scratch_pool.empty())
        {
            auto scratch = std::move(probe_scratch_pool.back());
            probe_scratch_pool.pop_back();
            return scratch;
        }
    }
    return std::make_unique<JoinProbeScratch>();
}

void ConcurrentHashJoin::releaseProbeScratch(std::unique_ptr<JoinProbeScratch> scratch, size_t lane)
{
    /// Park back under the lane when its entry is free; a collision or an out-of-range lane
    /// falls through to the pool.
    if (lane < probe_scratch_by_lane.size())
    {
        JoinProbeScratch * expected = nullptr;
        if (probe_scratch_by_lane[lane].compare_exchange_strong(expected, scratch.get(), std::memory_order_release))
        {
            scratch.release(); /// NOLINT(bugprone-unused-return-value): ownership moved into the entry
            return;
        }
    }

    std::lock_guard lock(probe_scratch_mutex);
    probe_scratch_pool.push_back(std::move(scratch));
}

bool ConcurrentHashJoin::addBlockToJoin(const Block & right_block_, bool check_limits)
{
    ProfileEventTimeIncrement<Microseconds> build_watch(ProfileEvents::ConcurrentHashJoinBuildMicroseconds);

    /// We materialize columns here to avoid materializing them multiple times on different threads
    /// (inside different `hash_join`-s) because the block will be shared.
    Block right_block = hash_joins[0]->data->materializeColumnsFromRightBlock(right_block_);

    ScatteredBlocks dispatched_blocks;
    {
        ProfileEventTimeIncrement<Microseconds> dispatch_watch(ProfileEvents::ConcurrentHashJoinBuildDispatchMicroseconds);
        dispatched_blocks = dispatchBlock(table_join->getOnlyClause().key_names_right, std::move(right_block));
    }
    size_t blocks_left = 0;
    for (const auto & block : dispatched_blocks)
    {
        if (block.rows())
        {
            ++blocks_left;
        }
    }

    size_t post_join_total_rows = 0;
    size_t post_join_total_bytes = 0;

    {
        ProfileEventTimeIncrement<Microseconds> insert_watch(ProfileEvents::ConcurrentHashJoinBuildInsertMicroseconds);
        while (blocks_left > 0)
        {
            bool made_progress = false;

            /// insert blocks into corresponding HashJoin instances
            for (size_t i = 0; i < dispatched_blocks.size(); ++i)
            {
                auto & hash_join = hash_joins[i];
                auto & dispatched_block = dispatched_blocks[i];

                if (dispatched_block.rows())
                {
                    /// if current hash_join is already processed by another thread, skip it and try later
                    std::unique_lock<std::mutex> lock(hash_join->mutex, std::try_to_lock);
                    if (!lock.owns_lock())
                        continue;

                    made_progress = true;

                    if (!hash_join->space_was_preallocated)
                    {
                        reserveSpaceInHashMaps(*hash_join->data, stats_collecting_params, slots, external_join_threshold);
                        hash_join->space_was_preallocated = true;
                    }

                    auto [block, selector] = std::move(dispatched_block).detachData();
                    bool limit_exceeded = !hash_join->data->addBlockToJoin(block, std::move(selector), check_limits);

                    std::tie(post_join_total_rows, post_join_total_bytes) = updateTotalRowsAndBytesUnlocked(hash_join);

                    dispatched_block = {};
                    blocks_left--;

                    if (limit_exceeded)
                        return false;
                }
            }

            /// If no slot was available in this pass, yield to avoid burning CPU while waiting
            /// for other threads to finish inserting into their respective hash join slots
            if (!made_progress)
                std::this_thread::yield();
        }
    }

    if (check_limits && table_join->sizeLimits().hasLimits())
        return table_join->sizeLimits().check(post_join_total_rows, post_join_total_bytes, "JOIN", ErrorCodes::SET_SIZE_LIMIT_EXCEEDED);
    return true;
}

static ColumnRawPtrs routeKeyColumns(const HashJoin & join, const JoinOnKeyColumns & keys);
static void computeDispatchSlotIds(
    const HashJoin & join, const ColumnRawPtrs & route_columns, size_t rows, size_t num_shards, UInt8 * slot_ids);

class ConcatStreams final : public IBlocksStream
{
public:
    using Deque = std::deque<IBlocksStreamPtr, AllocatorWithMemoryTracking<IBlocksStreamPtr>>;

    explicit ConcatStreams(std::vector<IBlocksStreamPtr> children_)
    {
        children = Deque(std::make_move_iterator(children_.begin()), std::make_move_iterator(children_.end()),
                         Deque::allocator_type{});
    }

    Block nextImpl() override
    {
        while (!children.empty())
        {
            auto & child = children.front();
            if (!child)
            {
                children.pop_front();
                continue;
            }
            Block b = child->next();
            if (!b.empty())
                return b;
            children.pop_front();
        }
        return {};
    }

private:
    Deque children;
};

/// The lazy result of the routed probe. The routed lookup itself is deferred to the first
/// `next()` call, keeping `joinBlock` cheap (it only prepares the key columns and derives
/// the slot ids); after that this is a thin timing shim around the inner `HashJoinResult`.
/// A `max_joined_block_rows` remainder (`next_block`) is passed through to the caller -
/// `JoiningTransform` re-feeds it through `joinBlock`, which re-prepares its keys and slot ids.
class RoutedJoinResult : public IJoinResult
{
    /// The parent join outlives every probe result (the pipeline holds it for as long as
    /// results are drained), so the reference into its slot table and the scratch release
    /// below are safe.
    ConcurrentHashJoin & parent;
    const std::vector<const HashJoin *> & slot_joins;
    /// The once-per-build probe address material; owned by `parent`, stable while any probe
    /// result is alive (like `slot_joins`).
    const RoutedProbePlan & plan;
    ScatteredBlock block;
    /// Owns the lane's pooled scratch (slot ids + the AMAC find-pass result arrays) for the
    /// lifetime of the lookup; parked back into the pool by the destructor - the lookup is
    /// lazy, so the release point cannot be `joinBlock`'s scope.
    std::unique_ptr<JoinProbeScratch> scratch;
    const size_t lane;
    /// The block's prepared key columns (materialized keys, null map, ON-section mask),
    /// built in `joinBlock`.
    std::vector<JoinOnKeyColumns> join_on_keys;
    /// Kept alive until destruction even after the last block: the `next_block` pointer
    /// returned to the caller points into it.
    JoinResultPtr inner;

public:
    RoutedJoinResult(
        ConcurrentHashJoin & parent_,
        const std::vector<const HashJoin *> & slot_joins_,
        const RoutedProbePlan & plan_,
        ScatteredBlock && block_,
        std::unique_ptr<JoinProbeScratch> && scratch_,
        size_t lane_,
        std::vector<JoinOnKeyColumns> && join_on_keys_)
        : parent(parent_)
        , slot_joins(slot_joins_)
        , plan(plan_)
        , block(std::move(block_))
        , scratch(std::move(scratch_))
        , lane(lane_)
        , join_on_keys(std::move(join_on_keys_))
    {
    }

    ~RoutedJoinResult() override
    {
        parent.releaseProbeScratch(std::move(scratch), lane);
    }

    JoinResultBlock next() override
    {
        /// Accumulates the whole lazy probe cost (the routed lookup below plus the gather/emit
        /// that runs inside `inner->next()`) into the probe total, on top of the route
        /// derivation `joinBlock` already charged before this result was created.
        ProfileEventTimeIncrement<Microseconds> probe_watch(ProfileEvents::ConcurrentHashJoinProbeMicroseconds);
        if (!inner)
        {
            /// The routed hash-map lookup: `joinRoutedBlock` -> `RoutedHashJoinMethods`'
            /// per-row `findKey` in the row's slot map (or the AMAC find pass) plus recording
            /// cheap match row-refs. It does NOT gather any column values yet (that is
            /// deferred to `HashJoinResult::next`).
            ProfileEventTimeIncrement<Microseconds> lookup_watch(ProfileEvents::ConcurrentHashJoinProbeLookupMicroseconds);
            inner = HashJoin::joinRoutedBlock(slot_joins, plan, std::move(block), *scratch, std::move(join_on_keys));
        }

        auto data = inner->next();
        return {std::move(data.block), data.next_block, data.is_last};
    }
};

JoinResultPtr ConcurrentHashJoin::joinBlock(Block block, size_t lane)
{
    ProfileEventTimeIncrement<Microseconds> probe_watch(ProfileEvents::ConcurrentHashJoinProbeMicroseconds);

    /// Once `mergeTwoLevelKey64BucketsIfUsed()` has moved every slot's bucket into
    /// `hash_joins[0]`'s table, that ONE instance holds every row - route through its plain,
    /// non-routed `joinBlock` directly (per decision #9 of `tmp/two_level_hashjoin_plan.md`: no
    /// probe-side scatter, bucket dispatch happens inside the table's own `find()`). The routed
    /// path below would find nothing for any row whose route lands on a slot other than 0 - their
    /// buckets are empty post-merge.
    if (two_level_key64_merged)
        return hash_joins[0]->data->joinBlock(std::move(block));

    const HashJoin & join0 = *hash_joins[0]->data;
    ScatteredBlock scattered;
    std::vector<JoinOnKeyColumns> join_on_keys;
    std::unique_ptr<JoinProbeScratch> scratch;
    {
        ProfileEventTimeIncrement<Microseconds> probe_dispatch_watch(ProfileEvents::ConcurrentHashJoinProbeDispatchMicroseconds);
        join0.materializeColumnsFromLeftBlock(block);
        scattered = ScatteredBlock{std::move(block)};

        /// The block's key columns are prepared here so that the slot-id derivation below and
        /// the routed lookup (through `RoutedJoinResult`) read the same `JoinOnKeyColumns`.
        const auto & onexpr = table_join->getOnlyClause();
        join_on_keys.emplace_back(
            scattered,
            onexpr.key_names_left,
            onexpr.condColumnNames().first,
            join0.getKeySizes().at(0),
            HashJoin::isLowCardinalityType(join0.getJoinedData()->type));

        scratch = acquireProbeScratch(lane);

        /// At most the slot ids are derived here; the block itself is NOT scattered - the
        /// routed probe (see `RoutedHashJoinMethods`) follows the route per row and emits in
        /// left-row order. The hash-routed families (every cursor-capable map; the plan's
        /// descriptors exist exactly for them) skip even the eager pass: the lookup derives
        /// the slot from the hash it computes anyway. The pass remains for the FixedHashMap
        /// families and for the mixed ON-expression path, whose shared filter loop consumes
        /// a slot-ids array (see `RoutedProbeContext`).
        const bool routes_by_hash = !routed_probe_plan.desc_by_slot.empty();
        if (slots > 1 && (!routes_by_hash || table_join->getMixedJoinExpression()))
        {
            const size_t rows = scattered.rows();
            scratch->slot_ids.resize(rows);
            computeDispatchSlotIds(join0, routeKeyColumns(join0, join_on_keys.front()), rows, slots, scratch->slot_ids.data());
        }
        else
            scratch->slot_ids.clear();
    }

    return std::make_unique<RoutedJoinResult>(
        *this, slot_joins, routed_probe_plan, std::move(scattered), std::move(scratch), lane, std::move(join_on_keys));
}

void ConcurrentHashJoin::checkTypesOfKeys(const Block & block) const
{
    hash_joins[0]->data->checkTypesOfKeys(block);
}

void ConcurrentHashJoin::setTotals(const Block & block)
{
    if (!block.empty())
    {
        std::lock_guard lock(totals_mutex);
        totals = block;
    }
}

const Block & ConcurrentHashJoin::getTotals() const
{
    return totals;
}

size_t ConcurrentHashJoin::getTotalRowCount() const
{
    return global_total_rows.load(std::memory_order_relaxed);
}

size_t ConcurrentHashJoin::getTotalByteCount() const
{
    return global_total_bytes.load(std::memory_order_relaxed);
}

bool ConcurrentHashJoin::alwaysReturnsEmptySet() const
{
    for (const auto & hash_join : hash_joins)
    {
        std::lock_guard lock(hash_join->mutex);
        if (!hash_join->data->alwaysReturnsEmptySet())
            return false;
    }
    return true;
}

IBlocksStreamPtr ConcurrentHashJoin::getNonJoinedBlocks(
        const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const
{
    return getNonJoinedBlocks(left_sample_block, result_sample_block, max_block_size, 0, 1);
}

bool ConcurrentHashJoin::supportParallelNonJoinedBlocksProcessing() const
{
    return table_join->allowParallelNonJoinedRowsProcessing()
        && JoinCommon::hasNonJoinedBlocks(*table_join)
        && !table_join->getOnlyClause().key_names_right.empty();
}

///   1) always-false condition (no keys): stream 0 returns all right rows
///   2) each stream scans slots where (slot % num_streams == stream_idx)
IBlocksStreamPtr ConcurrentHashJoin::getNonJoinedBlocks(
    const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size,
    size_t stream_idx, size_t num_streams) const
{
    if (!JoinCommon::hasNonJoinedBlocks(*table_join))
        return {};

    if (!isRightOrFull(table_join->kind()))
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Invalid join type. join kind: {}, strictness: {}",
                        table_join->kind(), table_join->strictness());

    /// no join keys (always false-condition), all right rows are non-joined, only stream 0 emits them
    if (table_join->getOnlyClause().key_names_right.empty())
    {
        if (stream_idx != 0)
            return {};
        std::lock_guard lock(hash_joins[0]->mutex);
        return hash_joins[0]->data->getNonJoinedBlocks(left_sample_block, result_sample_block, max_block_size);
    }

    std::vector<IBlocksStreamPtr> streams;
    for (size_t i = stream_idx; i < slots; i += num_streams)
    {
        const auto & hash_join = hash_joins[i];
        std::lock_guard lock(hash_join->mutex);
        if (hash_join->data->hasNonJoinedRows())
        {
            if (auto s = hash_join->data->getNonJoinedBlocks(left_sample_block, result_sample_block, max_block_size))
                streams.push_back(std::move(s));
        }
    }
    if (streams.empty())
        return {};
    if (streams.size() == 1)
        return streams[0];
    return std::make_shared<ConcatStreams>(std::move(streams));
}

/// How many leading key columns participate in routing. For ASOF the trailing
/// key is the inequality column and must NOT participate: the per-slot
/// `HashJoin`'s bucket key is the equality-only prefix (see `HashJoin`'s
/// constructor, where `key_columns.pop_back()` runs for ASOF before
/// `chooseMethod` picks a key getter). Routing by the full list would send
/// rows with equal equality keys but different asof values to different
/// slots, and they would never meet.
static size_t routeKeyColumnCount(const HashJoin & join, size_t total_key_columns)
{
    if (join.getTableJoin().strictness() == JoinStrictness::Asof && total_key_columns > 0)
        return total_key_columns - 1;
    return total_key_columns;
}

/// The routed prefix of the probe block's prepared key columns.
static ColumnRawPtrs routeKeyColumns(const HashJoin & join, const JoinOnKeyColumns & keys)
{
    const size_t count = routeKeyColumnCount(join, keys.key_columns.size());
    return ColumnRawPtrs(keys.key_columns.begin(), keys.key_columns.begin() + count);
}

/// The map-hash route pass of the cursor-capable open-addressing families: one slot id per
/// row, `joinHashRouteSlot` over the map's own hash of the row's key - the SAME hash/key
/// packing the per-slot inserts and the probe lookups compute, which is what makes the route
/// a build/probe contract (a plain `String` probe column against a `LowCardinality` build
/// side hashes the same value bytes). Returns false for the non-cursor map types
/// (`FixedHashMap`, the dictionary-aware `LowCardinality` maps) - the caller falls back.
///
/// Also engages for a runtime-bucket-count `TwoLevelHashTable` (`getBucketFromHash()`, e.g.
/// `key64_two_level` - not `WithJoinCursor`-wrapped, so it lacks `cursorCells()`, but its own
/// hash is just as usable for routing): the constructor sizes such a map's bucket count to
/// `slots` (`ConcurrentHashJoin::ConcurrentHashJoin`'s `two_level_buckets`), which makes
/// `joinHashRouteSlot(hash, route_shift)` and `map.getBucketFromHash(hash)` compute the exact
/// same bucket number (proven for every hash bit pattern and every count 1..256 by
/// `TwoLevelHashTableDynamic.BucketSelectionMatchesJoinHashRouteSlot`) - so this reuses the
/// identical body below rather than a separate `map.getBucketFromHash()` call, and dispatch and
/// the table's own internal routing are self-consistent by construction, not by a separate proof.
static bool computeHashRouteSlotIds(
    const HashJoin & join, const ColumnRawPtrs & route_columns, size_t rows, size_t num_shards, UInt8 * slot_ids)
{
    const auto route_shift = static_cast<UInt32>(32 - std::countr_zero(num_shards));
    const auto & joined = *join.getJoinedData();

    /// The key sizes exactly as the build/probe getters run on them (`createKeyGetter`): for
    /// ASOF the trailing entry belongs to the inequality column, excluded like its column.
    Sizes key_sizes = join.getKeySizes().at(0);
    if (join.getTableJoin().strictness() == JoinStrictness::Asof && !key_sizes.empty())
        key_sizes.pop_back();

    bool routed = false;
    auto compute = [&](const auto & maps)
    {
        switch (joined.type)
        {
#define M(TYPE) \
    case HashJoin::Type::TYPE: \
        if constexpr (requires { maps.TYPE->cursorCells(); } || requires { maps.TYPE->getBucketFromHash(size_t{}); }) \
        { \
            const auto & map = *maps.TYPE; \
            using KeyGetter = typename KeyGetterForType<HashJoin::Type::TYPE, std::remove_cvref_t<decltype(map)>>::Type; \
            Arena pool; \
            KeyGetter key_getter(route_columns, key_sizes, nullptr); \
            for (size_t i = 0; i < rows; ++i) \
            { \
                auto && key_holder = key_getter.getKeyHolder(i, pool); \
                const size_t hash = map.hash(keyHolderGetKey(key_holder)); \
                slot_ids[i] = static_cast<UInt8>(joinHashRouteSlot(hash, route_shift)); \
            } \
            routed = true; \
        } \
        break;
            APPLY_FOR_JOIN_VARIANTS(M)
#undef M
        }
    };
    std::visit(compute, joined.maps.at(0));
    return routed;
}

/// One slot id per row, shared by the build scatter and the probe dispatch (though the probe
/// runs this pass only for the families below and for the mixed ON-expression path - the
/// hash-routed families derive slots inline; see `hash_routed_lookup`):
/// - `FixedHashMap` types (`key8`/`key16`) index cells directly by the key value and have no
///   collision chains, so the key's low bits are the natural route for them;
/// - the cursor-capable open-addressing families route by the top bits of the maps' own hash
///   (`computeHashRouteSlotIds`), the same word their lookups derive per row;
/// - everything else (the range maps, the dictionary-aware `LowCardinality` maps - neither
///   reachable under `parallel_hash` today) keeps the value-byte fold of `JoinSlotRouting`.
static void computeDispatchSlotIds(
    const HashJoin & join, const ColumnRawPtrs & route_columns, size_t rows, size_t num_shards, UInt8 * slot_ids)
{
    chassert(isPowerOf2(num_shards) && num_shards > 1 && num_shards <= 256);

    const auto type = join.getJoinedData()->type;
    if (type == HashJoin::Type::key8 || type == HashJoin::Type::key16)
    {
        const IColumn & column = *route_columns.at(0);
        const char * data = column.getRawData().data();
        const size_t width = column.sizeOfValueIfFixed();
        const UInt8 mask = static_cast<UInt8>(num_shards - 1);
        /// The mask fits the value's low byte; on little-endian targets (the only ones this
        /// code runs on), the value's low byte is its first byte.
        for (size_t i = 0; i < rows; ++i)
            slot_ids[i] = static_cast<UInt8>(data[i * width]) & mask;
        return;
    }

    if (computeHashRouteSlotIds(join, route_columns, rows, num_shards, slot_ids))
        return;

    JoinSlotRouting::computeJoinSlotIds(route_columns, rows, static_cast<size_t>(std::countr_zero(num_shards)), slot_ids);
}

/// Build-side preparation of the routed key-column prefix: the same unwrap
/// chain `JoinOnKeyColumns` applies on the probe side (const/sparse unwrap,
/// `LowCardinality` removal unless the map type consumes the live dictionary
/// column, nullable-to-nested extraction). The fold is value-based, so the
/// route words agree with the probe's even where the physical
/// representations differ.
struct DispatchKeyColumns
{
    Columns holders;
    ColumnRawPtrs columns;
};

static DispatchKeyColumns prepareDispatchKeyColumns(
    const HashJoin & join, const Strings & key_columns_names, const Block & from_block)
{
    const size_t count = routeKeyColumnCount(join, key_columns_names.size());
    const bool keep_lowcardinality = HashJoin::isLowCardinalityType(join.getJoinedData()->type);

    const Names route_names(key_columns_names.begin(), key_columns_names.begin() + count);
    DispatchKeyColumns result;
    result.holders = keep_lowcardinality ? JoinCommon::materializeColumnsKeepLowCardinality(from_block, route_names)
                                         : JoinCommon::materializeColumns(from_block, route_names);
    result.columns = JoinCommon::getRawPointers(result.holders);
    ConstNullMapPtr null_map{};
    extractNestedColumnsAndNullMap(result.columns, null_map);
    return result;
}

static ScatteredBlocks scatterBlocksByCopying(size_t num_shards, const IColumn::Selector & selector, const Block & from_block)
{
    Blocks blocks(num_shards);
    for (size_t i = 0; i < num_shards; ++i)
        blocks[i] = from_block.cloneEmpty();

    for (size_t i = 0; i < from_block.columns(); ++i)
    {
        auto dispatched_columns = from_block.getByPosition(i).column->scatter(num_shards, selector);
        chassert(blocks.size() == dispatched_columns.size());
        for (size_t block_index = 0; block_index < num_shards; ++block_index)
        {
            blocks[block_index].getByPosition(i).column = std::move(dispatched_columns[block_index]);
        }
    }

    ScatteredBlocks result;
    result.reserve(num_shards);
    for (size_t i = 0; i < num_shards; ++i)
        result.emplace_back(std::move(blocks[i]));
    return result;
}

static ScatteredBlocks scatterBlocksWithSelector(size_t num_shards, const PaddedPODArray<UInt8> & slot_ids, const Block & from_block)
{
    std::vector<ScatteredBlock::IndexesPtr> selectors(num_shards);
    for (size_t i = 0; i < num_shards; ++i)
    {
        selectors[i] = ScatteredBlock::Indexes::create();
        selectors[i]->reserve(slot_ids.size() / num_shards + 1);
    }
    for (size_t i = 0; i < slot_ids.size(); ++i)
    {
        const size_t shard = slot_ids[i];
        selectors[shard]->getData().push_back(i);
    }
    ScatteredBlocks result;
    result.reserve(num_shards);
    for (size_t i = 0; i < num_shards; ++i)
        result.emplace_back(from_block, std::move(selectors[i]));
    return result;
}

ScatteredBlocks ConcurrentHashJoin::dispatchBlock(const Strings & key_columns_names, Block && from_block)
{
    const size_t num_shards = hash_joins.size();
    if (num_shards == 1)
    {
        ScatteredBlocks res;
        res.emplace_back(std::move(from_block));
        return res;
    }

    const HashJoin & join0 = *hash_joins[0]->data;
    const size_t rows = from_block.rows();
    PaddedPODArray<UInt8> slot_ids(rows);
    {
        const auto route_columns = prepareDispatchKeyColumns(join0, key_columns_names, from_block);
        computeDispatchSlotIds(join0, route_columns.columns, rows, num_shards, slot_ids.data());
    }

    /// With zero-copy approach we won't copy the source columns, but will create a new one with indices.
    /// This is not beneficial when the whole set of columns is e.g. a single small column.
    constexpr auto threshold = sizeof(IColumn::Selector::value_type);
    const auto & data_types = from_block.getDataTypes();
    const bool use_zero_copy_approach
        = std::accumulate(
              data_types.begin(),
              data_types.end(),
              0u,
              [](size_t sum, const DataTypePtr & type)
              { return sum + (type->haveMaximumSizeOfValue() ? type->getMaximumSizeOfValueInMemory() : threshold + 1); })
        > threshold;

    if (use_zero_copy_approach)
        return scatterBlocksWithSelector(num_shards, slot_ids, from_block);

    /// `IColumn::scatter` takes a `UInt64` selector; widen once here, on the narrow-row path
    /// where the block is copied anyway.
    IColumn::Selector selector(rows);
    for (size_t i = 0; i < rows; ++i)
        selector[i] = slot_ids[i];
    return scatterBlocksByCopying(num_shards, selector, from_block);
}

std::pair<size_t, size_t> ConcurrentHashJoin::updateTotalRowsAndBytesUnlocked(std::shared_ptr<InternalHashJoin> & hash_join)
{
    /// Update total rows for the current hash join instance and for the overall concurrent hash join
    const size_t rows_delta = hash_join->data->getTotalRowCount() - hash_join->local_total_rows;
    const size_t updated_global_rows = global_total_rows.fetch_add(rows_delta, std::memory_order_relaxed) + rows_delta;
    hash_join->local_total_rows += rows_delta;

    /// Update total bytes for the current hash join instance and for the overall concurrent hash join, taking
    /// into account that bytes could shrink
    const size_t updated_local_bytes = hash_join->data->getTotalByteCount();
    size_t updated_global_bytes = 0;
    if (updated_local_bytes >= hash_join->local_total_bytes)
    {
        const size_t bytes_delta = updated_local_bytes - hash_join->local_total_bytes;
        updated_global_bytes = global_total_bytes.fetch_add(bytes_delta, std::memory_order_relaxed) + bytes_delta;
    }
    else
    {
        const size_t bytes_delta = hash_join->local_total_bytes - updated_local_bytes;
        updated_global_bytes = global_total_bytes.fetch_sub(bytes_delta, std::memory_order_relaxed) - bytes_delta;
    }
    hash_join->local_total_bytes = updated_local_bytes;
    return {updated_global_rows, updated_global_bytes};
}

void ConcurrentHashJoin::resetTotalRowsAndBytesUnlocked(std::shared_ptr<InternalHashJoin> & hash_join)
{
    /// Reset global and local total rows and bytes
    global_total_rows.fetch_sub(hash_join->local_total_rows, std::memory_order_relaxed);
    global_total_bytes.fetch_sub(hash_join->local_total_bytes, std::memory_order_relaxed);
    hash_join->local_total_rows = 0;
    hash_join->local_total_bytes = 0;
}

BlocksList ConcurrentHashJoin::releaseSlotBlocks(size_t slot_idx)
{
    chassert(slot_idx < hash_joins.size());
    auto & hash_join = hash_joins[slot_idx];
    std::lock_guard lock(hash_join->mutex);
    if (!hash_join->data || !hash_join->data->getJoinedData())
        return {};
    resetTotalRowsAndBytesUnlocked(hash_join);
    return hash_join->data->releaseJoinedBlocks(/*restructure=*/ false);
}

/// Post-build, single-threaded (see `onBuildPhaseFinish`'s own comment: it cannot run concurrently
/// with other `IJoin` methods). Mirrors `onBuildPhaseFinish`'s `move_buckets` step in the
/// reference worktree `ClickHouse-concurrent-hash-join-profile-events`
/// (`ConcurrentHashJoin.cpp`, commit `a05f3ee81ff`), simplified for the 1:1 slot/bucket mapping
/// this PoC uses (`two_level_buckets == slots`, so slot `i` owns EXACTLY bucket `i`, no
/// interleaving/folding needed).
void ConcurrentHashJoin::mergeTwoLevelKey64BucketsIfUsed()
{
    if (!use_two_level_key64_poc || slots == 1)
        return;
    if (hash_joins[0]->data->getJoinedData()->type != HashJoin::Type::key64_two_level)
        return;

    auto & dst_data = *hash_joins[0]->data->getJoinedData();
    for (size_t i = 1; i < slots; ++i)
    {
        auto & src_data = *hash_joins[i]->data->getJoinedData();

        std::visit(
            [&](auto & dst_maps)
            {
                using T = std::decay_t<decltype(dst_maps)>;
                auto & src_maps = std::get<T>(src_data.maps.at(0));
                /// An O(1) ownership transfer, not a re-insertion: bucket `i` was never touched
                /// by any slot other than `i` (dispatch routes every row to the SAME bucket
                /// number as its slot, see `computeHashRouteSlotIds`'s bucketed branch), so this
                /// move is the entire merge for this bucket.
                dst_maps.key64_two_level->impls[i] = std::move(src_maps.key64_two_level->impls[i]);
            },
            dst_data.maps.at(0));

        dst_data.columns.splice(dst_data.columns.end(), src_data.columns);
        dst_data.allocated_size += src_data.allocated_size;
        dst_data.rows_to_join += src_data.rows_to_join;
        dst_data.keys_to_join += src_data.keys_to_join;
        src_data.allocated_size = 0;
        src_data.rows_to_join = 0;
        src_data.keys_to_join = 0;
    }

    two_level_key64_merged = true;
}

void ConcurrentHashJoin::onBuildPhaseFinish()
{
    ProfileEventTimeIncrement<Microseconds> build_watch(ProfileEvents::ConcurrentHashJoinBuildMicroseconds);
    /// The whole function is post-build consolidation across the slots, so it is charged wholesale
    /// as the "merge" build sub-phase.
    ProfileEventTimeIncrement<Microseconds> merge_watch(ProfileEvents::ConcurrentHashJoinBuildMergeMicroseconds);

    /// Synchronize all `HashJoin`s on the `all_values_unique` flag.
    bool all_values_unique = true;
    for (const auto & hash_join : hash_joins)
        all_values_unique &= hash_join->data->all_values_unique;

    for (const auto & hash_join : hash_joins)
        hash_join->data->all_values_unique = all_values_unique;

    // `onBuildPhaseFinish` cannot be called concurrently with other IJoin methods, so we don't need a lock to access internal joins.
    for (const auto & hash_join : hash_joins)
        hash_join->data->onBuildPhaseFinish();

    mergeTwoLevelKey64BucketsIfUsed();

    /// Per-slot `onBuildPhaseFinish` includes shrink-to-fit, which moves buffers, so re-collect
    /// the final addresses. Skipped once merged - the routed-probe plan is unused from then on
    /// (`joinBlock` short-circuits to `hash_joins[0]` directly, see there).
    if (!two_level_key64_merged)
        collectRoutedProbePlan();
}
}
