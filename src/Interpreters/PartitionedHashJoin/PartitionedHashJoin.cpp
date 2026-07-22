#include <Interpreters/PartitionedHashJoin/PartitionedHashJoin.h>

#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnsScatter.h>
#include <DataTypes/NullableUtils.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/PartitionedHashJoin/JoinRouteHashing.h>
#include <Interpreters/TableJoin.h>
#include <base/getL1CacheSize.h>
#include <base/getL2CacheSize.h>
#include <Common/CurrentMetrics.h>
#include <Common/CurrentThread.h>
#include <Common/ElapsedTimeProfileEventIncrement.h>
#include <Common/ProfileEvents.h>
#include <Common/ThreadGroupSwitcher.h>
#include <Common/logger_useful.h>

#include <fmt/ranges.h>

#include <bit>
#include <cmath>

namespace ProfileEvents
{
extern const Event PartitionedHashJoinBuildMicroseconds;
extern const Event PartitionedHashJoinBuildFillMicroseconds;
extern const Event PartitionedHashJoinProbeMicroseconds;
extern const Event PartitionedHashJoinPartitions;
extern const Event PartitionedHashJoinLeafRows;
extern const Event PartitionedHashJoinTeardownMicroseconds;
extern const Event PartitionedHashJoinDistinctEstimateReused;
}

namespace CurrentMetrics
{
extern const Metric PartitionedHashJoinPoolThreads;
extern const Metric PartitionedHashJoinPoolThreadsActive;
extern const Metric PartitionedHashJoinPoolThreadsScheduled;
}

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int NOT_IMPLEMENTED;
extern const int SET_SIZE_LIMIT_EXCEEDED;
}

namespace
{

/// Accumulates the time spent producing result blocks into `event`. `joinBlock` only sets up
/// a lazy result; the actual matching runs inside `IJoinResult::next`, so the probe time must
/// be accounted there.
class TimedJoinResult : public IJoinResult
{
public:
    TimedJoinResult(JoinResultPtr result_, ProfileEvents::Event event_)
        : result(std::move(result_))
        , event(event_)
    {
    }

    JoinResultBlock next() override
    {
        ProfileEventTimeIncrement<Microseconds> watch(event);
        return result->next();
    }

private:
    JoinResultPtr result;
    ProfileEvents::Event event;
};

}

PartitionedHashJoin::PartitionedHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader right_sample_block_,
    size_t num_threads_,
    bool any_take_last_row_,
    const StatsCollectingParams & stats_collecting_params_)
    : table_join(std::move(table_join_))
    , right_sample_block(std::move(right_sample_block_))
    , any_take_last_row(any_take_last_row_)
    , num_threads(std::max<size_t>(1, num_threads_))
    , leaf_join(std::make_unique<HashJoin>(table_join, right_sample_block, any_take_last_row))
    , delegate_mode(!table_join->oneDisjunct())
    , maps_variant_index(leaf_join->data->maps.empty() ? 1 : leaf_join->data->maps.front().index())
    , max_fanout_per_pass(ColumnsScatter::MAX_FANOUT_PER_PASS)
    , stats_collecting_params(stats_collecting_params_)
    , log(getLogger("PartitionedHashJoin"))
{
    if (!PartitionedJoinMaps::isSupportedType(leaf_join->data->type))
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "PartitionedHashJoin was created for an unsupported map type {}; the plan-time gate must reject this shape",
            leaf_join->data->type);

    /// Sized once, never resized: the lock-free fast paths index these without synchronizing
    /// against growth. Lanes past the table (rare pipeline shapes) take the legacy fallbacks.
    fill_lane_slots = std::vector<std::atomic<FillLane *>>(2 * num_threads);
    probe_scratch_slots = std::vector<std::atomic<ProbeScratch *>>(2 * num_threads);

    /// A cached per-partition distinct-key breakdown from a previous run of the same query
    /// replaces the sketch estimate wholesale: the decision is per build (all lanes fill the
    /// same way), so it is made once here. A stale or differently-shaped entry only mis-sizes
    /// the leaf reserves - the maps grow past an under-reserve (counted, never silent) - and the
    /// post-build always republishes the fresh exact counts.
    if (!delegate_mode && stats_collecting_params.isCollectionAndUseEnabled())
        cached_stats = getHashTablesStatistics<PartitionedHashJoinEntry>().getSizeHint(stats_collecting_params);
}

PartitionedHashJoin::~PartitionedHashJoin()
{
    /// The heavy state is destroyed explicitly inside the timed scope - members are otherwise
    /// destroyed after the destructor body, outside any timer. Order matters: the leaf maps'
    /// cells reference arena memory (string keys, duplicate-list nodes) and the row store, so
    /// the maps go first.
    ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinTeardownMicroseconds);

    /// The per-leaf maps are the bulk of the teardown (one exact-reserved buffer per leaf, up to
    /// tens of thousands of them); destroy them work-stealing over a short-lived pool, mirroring
    /// `ConcurrentHashJoin`'s teardown rationale (hash-table destruction can be very
    /// time-consuming). `post_build_pool` itself cannot be reused here: it is torn down right
    /// after the post-build phase finishes (`PartitionedHashJoinBuild.cpp`, well before probing
    /// even starts), so a fresh pool is spun up purely for this teardown. Destructors must not
    /// throw: a scheduling failure just leaves the remaining leaves to the serial clear below.
    if (!delegate_mode && leaf_maps.size() >= 64)
    {
        try
        {
            const size_t workers = std::min<size_t>(num_threads, leaf_maps.size());
            ThreadPool teardown_pool(
                CurrentMetrics::PartitionedHashJoinPoolThreads,
                CurrentMetrics::PartitionedHashJoinPoolThreadsActive,
                CurrentMetrics::PartitionedHashJoinPoolThreadsScheduled,
                /*max_threads_*/ workers,
                /*max_free_threads_*/ 0,
                /*queue_size_*/ workers);
            std::atomic<size_t> claim{0};
            for (size_t w = 0; w < workers; ++w)
                teardown_pool.scheduleOrThrow(
                    [this, &claim, thread_group = CurrentThread::getGroup()]
                    {
                        ThreadGroupSwitcher switcher(thread_group, ThreadName::PARTITIONED_JOIN);
                        while (true)
                        {
                            const size_t leaf = claim.fetch_add(1, std::memory_order_relaxed);
                            if (leaf >= leaf_maps.size())
                                break;
                            leaf_maps[leaf] = PartitionedJoinMaps(maps_variant_index);
                        }
                    });
            teardown_pool.wait();
        }
        catch (...) /// NOLINT(bugprone-empty-catch): fall through to the serial teardown below
        {
        }
    }
    leaf_maps.clear();
    build_arenas.clear();
    leaf_join.reset();
    probe_scratch_pool.clear();
    for (auto & slot : probe_scratch_slots)
        delete slot.load(std::memory_order_acquire);
}

bool PartitionedHashJoin::isSupported(const TableJoin & table_join)
{
    /// Everything the single-level `HashJoin` machinery serves: INNER/LEFT/RIGHT/FULL x
    /// ALL/ANY/RightAny/SEMI/ANTI plus ASOF, with null maps, per-clause ON-section filter
    /// conditions, USING, and single or multiple disjuncts (the latter run the standard
    /// machinery whole - see the class comment). Shapes that stay outside: special storages
    /// (StorageJoin / dictionaries), Cross/Comma/Paste and ON-constant (routed before the
    /// algorithm loop), spilling contexts (the SpillingHashJoin branch keeps plan-time
    /// priority), and mixed non-equi ON conditions attached to the join - `parallel_hash`
    /// serves those better than a delegated single-threaded build would.
    const JoinKind kind = table_join.kind();
    const JoinStrictness strictness = table_join.strictness();

    if (!isInner(kind) && !isLeft(kind) && !isRight(kind) && !isFull(kind))
        return false;

    switch (strictness)
    {
        case JoinStrictness::All:
        case JoinStrictness::Any:
        case JoinStrictness::RightAny:
        case JoinStrictness::Semi:
        case JoinStrictness::Anti:
        case JoinStrictness::Asof: break;
        default: return false;
    }

    if (table_join.isSpecialStorage())
        return false;
    if (table_join.getMixedJoinExpression())
        return false;

    if (strictness == JoinStrictness::Asof)
    {
        /// Mirrors the HashJoin restrictions: LEFT/INNER only, one disjunct, at least one
        /// equi-join column besides the trailing inequality column.
        if (!isInnerOrLeft(kind) || !table_join.oneDisjunct())
            return false;
        if (table_join.getOnlyClause().key_names_right.size() <= 1)
            return false;
    }

    /// Keyless clauses (ON-constant shapes) are handled by dedicated plan-time routing.
    for (const auto & clause : table_join.getClauses())
        if (clause.key_names_right.empty())
            return false;

    return true;
}

const TableJoin & PartitionedHashJoin::getTableJoin() const
{
    return *table_join;
}

PartitionedHashJoin::FillLane & PartitionedHashJoin::getFillLane()
{
    std::lock_guard lock(fill_mutex);
    auto [it, inserted] = lane_by_thread.try_emplace(std::this_thread::get_id(), nullptr);
    if (inserted)
        it->second = &lanes.emplace_back();
    return *it->second;
}

PartitionedHashJoin::FillLane & PartitionedHashJoin::getFillLane(size_t build_lane)
{
    if (build_lane >= fill_lane_slots.size())
        return getFillLane();

    if (FillLane * fast = fill_lane_slots[build_lane].load(std::memory_order_acquire))
        return *fast;

    /// First block of this lane: one mutexed emplace into the owning deque (whose elements are
    /// stable), then every later block takes the atomic load above. A lane index is unique per
    /// filling transform and a transform's work is serialized, so after publication the slot's
    /// state is single-writer even when executor threads migrate.
    std::lock_guard lock(fill_mutex);
    if (FillLane * raced = fill_lane_slots[build_lane].load(std::memory_order_relaxed))
        return *raced;
    FillLane * fresh = &lanes.emplace_back();
    fill_lane_slots[build_lane].store(fresh, std::memory_order_release);
    return *fresh;
}

bool PartitionedHashJoin::addBlockToJoin(const Block & source_block, bool check_limits)
{
    return addBlockToJoinImpl(source_block, check_limits, invalid_lane);
}

bool PartitionedHashJoin::addBlockToJoin(const Block & source_block, size_t /*num_rows*/, bool check_limits, size_t build_lane)
{
    /// num_rows only matters for columnless CROSS-join blocks, a shape this algorithm never
    /// plans; the row count comes from the block itself, as in the lane-less entry point.
    return addBlockToJoinImpl(source_block, check_limits, build_lane);
}

bool PartitionedHashJoin::addBlockToJoinImpl(const Block & source_block, bool check_limits, size_t build_lane)
{
    ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);

    if (build_phase_finished)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "PartitionedHashJoin: addBlockToJoin called after the build phase finished");

    if (delegate_mode)
    {
        /// The standard machinery runs the join whole (single fill stream, see `supportParallelJoin`).
        ProfileEvents::increment(ProfileEvents::PartitionedHashJoinLeafRows, source_block.rows());
        return leaf_join->addBlockToJoin(source_block, check_limits);
    }

    /// The "fill" build sub-phase: right-side key preparation plus the per-row route hash and
    /// HLL sketch update (`computeJoinRoutesForFill`); the partitioned/single-leaf decision itself
    /// is made later, at the build barrier, so every plan pays this cost identically.
    ProfileEventTimeIncrement<Microseconds> fill_watch(ProfileEvents::PartitionedHashJoinBuildFillMicroseconds);

    Block materialized = leaf_join->materializeColumnsFromRightBlock(source_block);
    const size_t rows = materialized.rows();
    if (rows == 0)
        return true;

    /// RowRef::row_no is 32-bit; same restriction as HashJoin's.
    if (rows > std::numeric_limits<UInt32>::max()) [[unlikely]]
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Too many rows in right table block for PartitionedHashJoin: {}", rows);

    FillBlock fill;
    fill.rows = rows;

    /// Prepare the key columns the same way the probe side does (`JoinOnKeyColumns`): materialize,
    /// keep a live LowCardinality column only for the dictionary-aware map types, extract the
    /// merged null map and strip the key columns to their nested form. For ASOF the merged null
    /// map covers the trailing inequality column too - rows with a NULL ASOF key never join.
    const auto & clause = table_join->getOnlyClause();
    fill.keys_holder = HashJoin::isLowCardinalityType(leaf_join->data->type)
        ? JoinCommon::materializeColumnsKeepLowCardinality(materialized, clause.key_names_right)
        : JoinCommon::materializeColumns(materialized, clause.key_names_right);
    fill.key_columns = JoinCommon::getRawPointers(fill.keys_holder);
    fill.null_map_holder = extractNestedColumnsAndNullMap(fill.key_columns, fill.null_map);

    /// The right-side ON-section condition: rows it filters are not inserted into the leaf maps
    /// (they are still saved for RIGHT/FULL non-joined output, see `storeBlocksInRowStore`).
    fill.join_mask = JoinCommon::getColumnAsMask(materialized, clause.condColumnNames().second);
    if (fill.join_mask.hasData() && fill.join_mask.getKind() != JoinCommon::JoinMask::Kind::AllTrue)
    {
        fill.skip_bytes.resize_exact(rows);
        const NullMap * nulls = fill.null_map;
        for (size_t i = 0; i < rows; ++i)
            fill.skip_bytes[i] = ((nulls && (*nulls)[i]) || fill.join_mask.isRowFiltered(i)) ? 1 : 0;
    }

    /// One route word per row (R4, R6): the top 16 bits saved per row and the full word fed to
    /// the lane sketch, fused into the word loop (no 32-bit word transient). Skipped rows (null
    /// keys, mask-filtered) are never inserted, so they do not contribute to the estimate; their
    /// routes are still written - the scatter's bucket derivation reads them. ASOF routes and
    /// sketches by the equi-key prefix only - the trailing inequality column goes into the
    /// per-key sorted lookup, not into the map key.
    fill.routes.resize_exact(rows);
    FillLane & lane = build_lane == invalid_lane ? getFillLane() : getFillLane(build_lane);
    if (leaf_join->getStrictness() == JoinStrictness::Asof)
    {
        ColumnRawPtrs equi_columns(fill.key_columns.begin(), fill.key_columns.end() - 1);
        if (cached_stats)
            computeJoinRoutesForFill(equi_columns, rows, fill.routes.data());
        else
            computeJoinRoutesForFill(equi_columns, rows, fill.skipData(), fill.routes.data(), lane.hll);
    }
    else if (cached_stats)
    {
        /// Warm run: a previous run published this query's distinct-key counts, so the sketch
        /// estimate is not needed and the per-row sketch feed is skipped.
        computeJoinRoutesForFill(fill.key_columns, rows, fill.routes.data());
    }
    else
    {
        computeJoinRoutesForFill(fill.key_columns, rows, fill.skipData(), fill.routes.data(), lane.hll);
    }

    /// The block in row-store form (payload untouched, G1); appended zero-copy to the lane.
    fill.stored = HashJoin::prepareRightBlock(materialized, leaf_join->savedBlockSample());

    accumulated_rows.fetch_add(rows, std::memory_order_relaxed);
    accumulated_bytes.fetch_add(fill.stored.allocatedBytes() + fill.routes.allocated_bytes(), std::memory_order_relaxed);
    lane.blocks.push_back(std::move(fill));

    if (!check_limits)
        return true;

    /// Fill-phase analog of HashJoin's per-block limit check. The row count is the accumulated
    /// input rows (an upper bound of the hash-table keys the map-based algorithms check), the
    /// byte count covers the stored blocks and the route transients.
    return table_join->sizeLimits().check(
        accumulated_rows.load(std::memory_order_relaxed),
        accumulated_bytes.load(std::memory_order_relaxed),
        "JOIN",
        ErrorCodes::SET_SIZE_LIMIT_EXCEEDED);
}

void PartitionedHashJoin::checkTypesOfKeys(const Block & block) const
{
    leaf_join->checkTypesOfKeys(block);
}

void PartitionedHashJoin::setTotals(const Block & block)
{
    if (!block.empty())
    {
        std::lock_guard lock(totals_mutex);
        totals = block;
    }
}

const Block & PartitionedHashJoin::getTotals() const
{
    return totals;
}

void PartitionedHashJoin::storeBlocksInRowStore()
{
    const bool right_or_full = isRightOrFull(leaf_join->getKind());
    auto & data = *leaf_join->data;
    for (auto & fill : build_blocks)
    {
        assertBlocksHaveEqualStructureAllowReplicated(data.sample_block, fill.stored, "joined block");
        auto & stored = data.columns.emplace_back(fill.stored.getColumns(), ScatteredBlock::Selector(fill.rows));
        stored.block_no = data.stored_columns_index->add(&stored);
        data.allocated_size += stored.allocatedBytes();
        data.rows_to_join += fill.rows;
        fill.block_no = stored.block_no;
        fill.stored = Block{};

        if (!right_or_full)
            continue;

        /// RIGHT/FULL non-joined output needs the rows that were never inserted into the maps:
        /// null-key rows (the key null map) and rows filtered by the right-side ON condition
        /// (a mask of filtered-and-not-null rows), exactly as the standard build saves them.
        bool save_nullmap = false;
        if (fill.null_map)
            for (size_t i = 0; i < fill.rows && !save_nullmap; ++i)
                save_nullmap = (*fill.null_map)[i];
        if (save_nullmap)
        {
            auto & holder = data.nullmaps.emplace_back(&stored, fill.null_map_holder);
            data.nullmaps_allocated_size += holder.allocatedBytes();
        }

        if (fill.join_mask.hasData() && fill.join_mask.getKind() != JoinCommon::JoinMask::Kind::AllTrue)
        {
            auto not_joined_map = ColumnUInt8::create(fill.rows, static_cast<UInt8>(0));
            bool has_right_not_joined = false;
            for (size_t i = 0; i < fill.rows; ++i)
            {
                if (!fill.join_mask.isRowFiltered(i))
                    continue;
                if (save_nullmap && (*fill.null_map)[i])
                    continue; /// already covered by the null-keys map
                not_joined_map->getData()[i] = 1;
                has_right_not_joined = true;
            }
            if (has_right_not_joined)
            {
                auto & holder = data.nullmaps.emplace_back(&stored, std::move(not_joined_map));
                data.nullmaps_allocated_size += holder.allocatedBytes();
            }
        }
    }
}

void PartitionedHashJoin::decidePartitionPlan()
{
    const HashJoin::Type type = leaf_join->data->type;

    /// ASOF builds stay at the single-leaf plan: the mapped values are per-key sorted lookup
    /// vectors whose insert wants the original (block, row) order, and the sorted-vector work
    /// dominates the build - partitioning the equi-key map buys nothing worth a scattered
    /// insert order. The single-leaf path inserts straight from the stored blocks.
    bits = 0;
    if (!PartitionedJoinMaps::isFixedSizeType(type) && leaf_join->getStrictness() != JoinStrictness::Asof)
    {
        /// The L2 rule with grower-exact rounding: the smallest number of bits such that the
        /// worst-case per-leaf reserve (the histogram clamp can only shrink it) produces a
        /// bucket array within the leaf budget through the map's own grower math. Evaluating
        /// at the safety-scaled reserve also hedges estimates landing exactly on a grower
        /// boundary, where per-leaf spread would otherwise double half the leaves.
        const size_t l2_bytes = std::max<size_t>(getL2CacheSize(), 1 << 20);
        const auto leaf_budget_bytes = static_cast<size_t>(0.8 * static_cast<double>(l2_bytes));
        const auto reserve_for = [&](size_t fanout)
        { return std::max<size_t>(1, static_cast<size_t>(std::ceil(hll_estimate * reserve_safety / static_cast<double>(fanout)))); };

        while (bits < 16
               && PartitionedJoinMaps::predictedBufferBytes(maps_variant_index, type, reserve_for(1uz << bits)) > leaf_budget_bytes)
            ++bits;

        /// The single-pass descriptor cap: past this many leaves, the flat per-leaf lookup
        /// descriptor array (`LeafMapDesc`, gathered once per probe row at AMAC-ring admit) stops
        /// being a single cache-resident load, so growing further to keep leaf hash-table buckets
        /// L2-resident only adds probe-side traffic and a second scatter pass for it. Budgeted
        /// against L1 - the descriptor gather is meant to cost one L1-latency load per row - with
        /// only a quarter of it charged to the array, leaving headroom for the rest of the probe's
        /// per-row working set (the hash-bucket cell it points at, key and result columns) that
        /// shares L1 alongside it.
        const size_t l1_bytes = std::max<size_t>(getL1CacheSize(), 32 << 10);
        const size_t max_leaves_for_descs = std::max<size_t>(1, l1_bytes / 4 / sizeof(LeafMapDesc));
        const auto descriptor_cap_bits = static_cast<size_t>(std::bit_width(max_leaves_for_descs) - 1);
        bits = std::min(bits, descriptor_cap_bits);

        if (bits > 0)
        {
            /// Once partitioning pays for itself, take at least one leaf per worker so the leaf
            /// builds parallelize; a small build stays at the degenerate single-leaf plan (G6).
            const auto parallelism_floor = static_cast<size_t>(std::bit_width(std::bit_ceil(num_threads) - 1));
            bits = std::max(bits, parallelism_floor);
        }
    }

    partitions = 1uz << bits;

    /// When the L2 rule wants more partitions than one scatter pass's fanout ceiling sustains,
    /// the plan splits the bits into MSB-first passes (a first pass plus refine passes) instead
    /// of capping the fanout. The 16-bit plan-loop bound above is what the saved 16-bit routes
    /// and the UInt16 probe leaf ids cover, so no route widening is needed at any reachable plan.
    pass_bits = bits > 0 ? ColumnsScatter::computePassBits(partitions, max_fanout_per_pass) : std::vector<size_t>{};
}

void PartitionedHashJoin::onBuildPhaseFinish()
{
    ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);

    if (delegate_mode)
    {
        /// The standard machinery ran the whole build during the fill; only its own barrier
        /// (used-flags init, promotion, non-joined status) remains. The partition plan is 1.
        leaf_join->onBuildPhaseFinish();
        ProfileEvents::increment(ProfileEvents::PartitionedHashJoinPartitions, partitions);
        return;
    }

    /// The cheap barrier (R5), run once by the last fill thread: concatenate the lanes, assign
    /// row-store block numbers, merge the sketches and pick the partition plan. The heavy work
    /// (scatter, hash-table allocation, leaf builds) runs in `runPostBuildPhase`.
    DenseHyperLogLog merged;
    size_t total_blocks = 0;
    for (const auto & lane : lanes)
        total_blocks += lane.blocks.size();
    build_blocks.reserve(total_blocks);
    for (auto & lane : lanes)
    {
        merged.merge(lane.hll);
        for (auto & block : lane.blocks)
            build_blocks.push_back(std::move(block));
        lane.blocks.clear();
    }
    lanes.clear();
    lane_by_thread.clear();

    if (cached_stats)
    {
        /// The lanes' sketches were never fed (see `addBlockToJoinImpl`); the cached total from
        /// the previous run replaces the estimate driving the partition-count decision below.
        /// The per-leaf sizing in `planHashTables` separately consumes the cached per-partition
        /// breakdown. Downstream sizing is clamped per leaf by exact row counts, so a stale
        /// value cannot inflate a leaf past its rows and an under-estimate only triggers counted
        /// map growth.
        hll_estimate = static_cast<double>(std::max<size_t>(1, cached_stats->total_distinct));
        stats.distinct_estimate_reused = true;
        ProfileEvents::increment(ProfileEvents::PartitionedHashJoinDistinctEstimateReused);
    }
    else
    {
        hll_estimate = merged.estimate();
    }
    storeBlocksInRowStore();

    /// The packed 4-byte locator encoding applies when every (block_no, row_no) fits 16+16 bits;
    /// typical pipelines deliver blocks under 65536 rows, so this halves the locator transient.
    narrow_locators = build_blocks.size() <= (1uz << 16);
    for (const auto & fill : build_blocks)
        narrow_locators = narrow_locators && fill.block_no < (1u << 16) && fill.rows <= (1uz << 16);

    decidePartitionPlan();
    ProfileEvents::increment(ProfileEvents::PartitionedHashJoinPartitions, partitions);

    LOG_TRACE(
        log,
        "Partition plan: bits = {}, partitions = {}, {} scatter pass(es) (bits per pass [{}]), {} rows in {} blocks, "
        "estimated {} distinct keys",
        bits,
        partitions,
        std::max<size_t>(pass_bits.size(), 1),
        fmt::join(pass_bits, ", "),
        accumulated_rows.load(std::memory_order_relaxed),
        build_blocks.size(),
        static_cast<size_t>(hll_estimate));
}

JoinResultPtr PartitionedHashJoin::joinBlock(Block block)
{
    return joinBlock(std::move(block), invalid_lane);
}

JoinResultPtr PartitionedHashJoin::joinBlock(Block block, size_t lane)
{
    JoinResultPtr result;
    {
        ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinProbeMicroseconds);
        result = delegate_mode ? leaf_join->joinBlock(std::move(block)) : probeDispatch(std::move(block), lane);
    }
    return std::make_unique<TimedJoinResult>(std::move(result), ProfileEvents::PartitionedHashJoinProbeMicroseconds);
}

size_t PartitionedHashJoin::getTotalRowCount() const
{
    if (delegate_mode)
        return leaf_join->getTotalRowCount();

    if (!build_phase_finished)
        return accumulated_rows.load(std::memory_order_relaxed);

    const HashJoin::Type type = leaf_join->data->type;
    size_t res = 0;
    for (const auto & maps : leaf_maps)
        res += maps.getTotalRowCount(type);
    return res;
}

size_t PartitionedHashJoin::getTotalByteCount() const
{
    if (delegate_mode)
        return leaf_join->getTotalByteCount();

    size_t res = accumulated_bytes.load(std::memory_order_relaxed);
    const HashJoin::Type type = leaf_join->data->type;
    for (const auto & maps : leaf_maps)
        res += maps.getBufferSizeInBytes(type);
    for (const auto & arena : build_arenas)
        res += arena.allocatedBytes();
    return res;
}

bool PartitionedHashJoin::alwaysReturnsEmptySet() const
{
    if (delegate_mode)
        return leaf_join->alwaysReturnsEmptySet();
    return isInnerOrRight(table_join->kind()) && accumulated_rows.load(std::memory_order_relaxed) == 0;
}

PartitionedHashJoin::BuildStats PartitionedHashJoin::getBuildStats() const
{
    BuildStats res = stats;
    res.bits = bits;
    res.partitions = partitions;
    res.pass_bits = pass_bits;
    res.hll_estimate = hll_estimate;
    res.ht_total_bytes = ht_total_bytes;
    res.amac_ring_growths = amac_ring_growths.load(std::memory_order_relaxed);
    res.amac_build_engaged = amac_build_engaged;
    res.flag_base = flag_base;
    return res;
}

std::unique_ptr<PartitionedHashJoin::ProbeScratch> PartitionedHashJoin::acquireProbeScratch(size_t lane)
{
    /// Lane fast path: take the parked scratch of this probe stream with one atomic exchange.
    if (lane < probe_scratch_slots.size())
        if (ProbeScratch * parked = probe_scratch_slots[lane].exchange(nullptr, std::memory_order_acquire))
            return std::unique_ptr<ProbeScratch>(parked);

    {
        std::lock_guard lock(probe_scratch_mutex);
        if (!probe_scratch_pool.empty())
        {
            auto scratch = std::move(probe_scratch_pool.back());
            probe_scratch_pool.pop_back();
            return scratch;
        }
    }
    return std::make_unique<ProbeScratch>();
}

void PartitionedHashJoin::releaseProbeScratch(std::unique_ptr<ProbeScratch> scratch, size_t lane)
{
    /// Park back into the lane's slot when it is free; a collision (or an out-of-range lane)
    /// falls through to the pool, so the scratch is never lost and never double-owned.
    if (lane < probe_scratch_slots.size())
    {
        ProbeScratch * expected = nullptr;
        if (probe_scratch_slots[lane].compare_exchange_strong(expected, scratch.get(), std::memory_order_release))
        {
            scratch.release(); /// NOLINT(bugprone-unused-return-value): ownership moved into the slot
            return;
        }
    }

    std::lock_guard lock(probe_scratch_mutex);
    probe_scratch_pool.push_back(std::move(scratch));
}

bool PartitionedHashJoin::isCloneSupported() const
{
    return getTotals().empty() && getTotalRowCount() == 0;
}

std::shared_ptr<IJoin>
PartitionedHashJoin::clone(const std::shared_ptr<TableJoin> & table_join_, SharedHeader, SharedHeader right_sample_block_) const
{
    /// Every reachable clone path preserves a supported shape (e.g. a table swap of an
    /// INNER ALL one-disjunct clause); re-check so a future caller that violates the invariant
    /// surfaces as an exception instead of wrong results.
    if (!isSupported(*table_join_))
        throw Exception(
            ErrorCodes::LOGICAL_ERROR, "PartitionedHashJoin: attempt to clone with a join shape the algorithm does not support");
    return std::make_shared<PartitionedHashJoin>(table_join_, right_sample_block_, num_threads, any_take_last_row, stats_collecting_params);
}

std::shared_ptr<IJoin>
PartitionedHashJoin::cloneNoParallel(const std::shared_ptr<TableJoin> & table_join_, SharedHeader, SharedHeader right_sample_block_) const
{
    return std::make_shared<HashJoin>(table_join_, right_sample_block_, any_take_last_row);
}

void PartitionedHashJoin::setEnableLazyColumnsIndexing(bool value)
{
    leaf_join->setEnableLazyColumnsIndexing(value);
}

}
