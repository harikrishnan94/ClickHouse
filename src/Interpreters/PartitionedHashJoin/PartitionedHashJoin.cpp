#include <Interpreters/PartitionedHashJoin/PartitionedHashJoin.h>

#include <Columns/ColumnsScatter.h>
#include <DataTypes/NullableUtils.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/PartitionedHashJoin/JoinRouteHashing.h>
#include <Interpreters/TableJoin.h>
#include <base/getL2CacheSize.h>
#include <Common/ElapsedTimeProfileEventIncrement.h>
#include <Common/ProfileEvents.h>
#include <Common/logger_useful.h>

#include <bit>
#include <cmath>

namespace ProfileEvents
{
extern const Event PartitionedHashJoinBuildMicroseconds;
extern const Event PartitionedHashJoinProbeMicroseconds;
extern const Event PartitionedHashJoinPartitions;
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
    std::shared_ptr<TableJoin> table_join_, SharedHeader right_sample_block_, size_t num_threads_, bool any_take_last_row_)
    : table_join(std::move(table_join_))
    , right_sample_block(std::move(right_sample_block_))
    , any_take_last_row(any_take_last_row_)
    , num_threads(std::max<size_t>(1, num_threads_))
    , leaf_join(std::make_unique<HashJoin>(table_join, right_sample_block, any_take_last_row))
    , log(getLogger("PartitionedHashJoin"))
{
    if (!PartitionedJoinMaps::isSupportedType(leaf_join->data->type))
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "PartitionedHashJoin was created for an unsupported map type {}; the plan-time gate must reject this shape",
            leaf_join->data->type);
}

PartitionedHashJoin::~PartitionedHashJoin()
{
    /// The leaf maps carve their buffers from the slab; release them before the slab.
    leaf_maps.clear();
    if (ht_slab)
        slab_allocator.free(ht_slab, ht_slab_bytes);
}

bool PartitionedHashJoin::isSupported(const TableJoin & table_join)
{
    /// The supported set is deliberately narrow while the partitioned build/probe paths are
    /// being brought up: INNER/LEFT ALL equi-joins with a single conjunction of keys and no
    /// extra ON conditions. Everything else is planned with another algorithm.
    if (!isInnerOrLeft(table_join.kind()))
        return false;
    if (table_join.strictness() != JoinStrictness::All)
        return false;
    if (!table_join.oneDisjunct())
        return false;
    if (table_join.isSpecialStorage())
        return false;
    if (table_join.getMixedJoinExpression())
        return false;

    const auto & clause = table_join.getOnlyClause();
    if (clause.key_names_right.empty())
        return false;
    if (clause.on_filter_condition_left || clause.on_filter_condition_right || !clause.analyzer_left_filter_condition_column_name.empty()
        || !clause.analyzer_right_filter_condition_column_name.empty())
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

bool PartitionedHashJoin::addBlockToJoin(const Block & source_block, bool check_limits)
{
    ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);

    if (build_phase_finished)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "PartitionedHashJoin: addBlockToJoin called after the build phase finished");

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
    /// merged null map and strip the key columns to their nested form.
    const auto & key_names_right = table_join->getOnlyClause().key_names_right;
    fill.keys_holder = HashJoin::isLowCardinalityType(leaf_join->data->type)
        ? JoinCommon::materializeColumnsKeepLowCardinality(materialized, key_names_right)
        : JoinCommon::materializeColumns(materialized, key_names_right);
    fill.key_columns = JoinCommon::getRawPointers(fill.keys_holder);
    fill.null_map_holder = extractNestedColumnsAndNullMap(fill.key_columns, fill.null_map);

    /// One route word per row (R4, R6): save the top 16 bits, feed the full word to the lane
    /// sketch. Null-key rows are never inserted, so they do not contribute to the estimate.
    fill.routes.resize_exact(rows);
    {
        PaddedPODArray<UInt32> words(rows);
        computeJoinRouteWords(fill.key_columns, rows, words.data());
        FillLane & lane = getFillLane();
        const NullMap * nulls = fill.null_map;
        for (size_t i = 0; i < rows; ++i)
        {
            fill.routes[i] = static_cast<UInt16>(words[i] >> 16);
            if (!nulls || !(*nulls)[i])
                lane.hll.add(words[i]);
        }

        /// The block in row-store form (payload untouched, G1); appended zero-copy to the lane.
        fill.stored = HashJoin::prepareRightBlock(materialized, leaf_join->savedBlockSample());

        accumulated_rows.fetch_add(rows, std::memory_order_relaxed);
        accumulated_bytes.fetch_add(fill.stored.allocatedBytes() + fill.routes.allocated_bytes(), std::memory_order_relaxed);
        lane.blocks.push_back(std::move(fill));
    }

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

void PartitionedHashJoin::storeBlocksInRowStore()
{
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
    }
}

void PartitionedHashJoin::decidePartitionPlan()
{
    const HashJoin::Type type = leaf_join->data->type;

    bits = 0;
    if (!PartitionedJoinMaps::isFixedSizeType(type))
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

        while (bits < 16 && PartitionedJoinMaps::predictedBufferBytes(type, reserve_for(1uz << bits)) > leaf_budget_bytes)
            ++bits;

        if (bits > 0)
        {
            /// Once partitioning pays for itself, take at least one leaf per worker so the leaf
            /// builds parallelize; a small build stays at the degenerate single-leaf plan (G6).
            const auto parallelism_floor = static_cast<size_t>(std::bit_width(std::bit_ceil(num_threads) - 1));
            bits = std::max(bits, parallelism_floor);
        }

        /// Single-pass scatter only: the per-pass fanout ceiling caps the bits. Reaching the cap
        /// needs an estimate above ~400M distinct keys (2^13 leaves of ~50K); the multi-pass
        /// refine that lifts it is deferred (see the phase report). Correctness is unaffected -
        /// the leaves just exceed the L2 budget.
        const auto pass_cap_bits = static_cast<size_t>(std::countr_zero(ColumnsScatter::MAX_FANOUT_PER_PASS));
        if (bits > pass_cap_bits)
        {
            LOG_WARNING(
                log,
                "Partition plan capped at {} bits by the single-pass fanout ceiling (the L2 rule asked for {}); "
                "leaf hash tables will exceed the cache budget",
                pass_cap_bits,
                bits);
            bits = pass_cap_bits;
        }
    }

    partitions = 1uz << bits;
}

void PartitionedHashJoin::onBuildPhaseFinish()
{
    ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);

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

    hll_estimate = merged.estimate();
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
        "Partition plan: bits = {}, partitions = {}, {} rows in {} blocks, estimated {} distinct keys",
        bits,
        partitions,
        accumulated_rows.load(std::memory_order_relaxed),
        build_blocks.size(),
        static_cast<size_t>(hll_estimate));
}

JoinResultPtr PartitionedHashJoin::joinBlock(Block block)
{
    JoinResultPtr result;
    {
        ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinProbeMicroseconds);
        result = probeDispatch(std::move(block));
    }
    return std::make_unique<TimedJoinResult>(std::move(result), ProfileEvents::PartitionedHashJoinProbeMicroseconds);
}

size_t PartitionedHashJoin::getTotalRowCount() const
{
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
    return isInnerOrRight(table_join->kind()) && accumulated_rows.load(std::memory_order_relaxed) == 0;
}

PartitionedHashJoin::BuildStats PartitionedHashJoin::getBuildStats() const
{
    BuildStats res = stats;
    res.bits = bits;
    res.partitions = partitions;
    res.hll_estimate = hll_estimate;
    res.slab_bytes = ht_slab_bytes;
    res.region_carves = region_carves.load(std::memory_order_relaxed);
    res.heap_fallbacks = heap_fallbacks.load(std::memory_order_relaxed);
    return res;
}

IBlocksStreamPtr
PartitionedHashJoin::getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const
{
    /// Non-joined rows exist only for RIGHT/FULL kinds, which the plan-time gate rejects.
    return leaf_join->getNonJoinedBlocks(left_sample_block, result_sample_block, max_block_size);
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
    return std::make_shared<PartitionedHashJoin>(table_join_, right_sample_block_, num_threads, any_take_last_row);
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
