#include <Interpreters/RadixHashJoin/RadixHashJoin.h>

#include <Interpreters/TableJoin.h>

#include <Common/Exception.h>

namespace DB
{

namespace ErrorCodes
{
extern const int NOT_IMPLEMENTED;
}

/// All radix-path state (partition plan, build side, leaf tables, probe scratch, output plan)
/// lives here once implemented.
struct RadixHashJoin::State
{
};

RadixHashJoin::RadixHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader right_sample_block_,
    size_t max_threads_,
    std::optional<UInt64> rhs_size_estimation_,
    UInt64 max_partitions_per_pass_,
    bool size_tables_by_distinct_estimate_,
    double probe_buffer_fraction_,
    UInt64 probe_buffer_min_bytes_,
    UInt64 probe_buffer_max_bytes_,
    const StatsCollectingParams & stats_collecting_params_)
    : table_join(std::move(table_join_))
    , right_sample_block(right_sample_block_)
    , max_threads(std::max<size_t>(max_threads_, 1))
    , rhs_size_estimation(rhs_size_estimation_)
    , max_partitions_per_pass(max_partitions_per_pass_)
    , size_tables_by_distinct_estimate(size_tables_by_distinct_estimate_)
    , stats_collecting_params(stats_collecting_params_)
    , state(std::make_unique<State>())
{
    /// Implementation point: re-check the planner-gate invariants (single disjunct, fixed-width keys,
    /// packed key width a multiple of 4 in [4, 64]), validate the probe-buffer knobs, precompute the
    /// key layout and output plan, choose the partition plan, and create the build side and the pool.
    (void)probe_buffer_fraction_;
    (void)probe_buffer_min_bytes_;
    (void)probe_buffer_max_bytes_;

    /// Silence -Wunused-private-field until the implementation consumes these.
    (void)max_partitions_per_pass;
    (void)size_tables_by_distinct_estimate;
}

RadixHashJoin::~RadixHashJoin() = default;

const TableJoin & RadixHashJoin::getTableJoin() const
{
    return *table_join;
}

bool RadixHashJoin::addBlockToJoin(const Block & block, bool check_limits)
{
    return addBlockToJoin(block, block.rows(), check_limits, 0);
}

bool RadixHashJoin::addBlockToJoin(const Block & block, size_t num_rows, bool check_limits)
{
    return addBlockToJoin(block, num_rows, check_limits, 0);
}

bool RadixHashJoin::addBlockToJoin(const Block & /*block*/, size_t /*num_rows*/, bool /*check_limits*/, size_t /*build_lane*/)
{
    /// Implementation point: normalise the right block to the sample structure, materialise it, and
    /// accumulate it in the build side (per-lane block store + per-leaf histogram; no scatter, no copy).
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "RadixHashJoin::addBlockToJoin is not implemented");
}

void RadixHashJoin::setTotals(const Block & /*block*/)
{
    /// Implementation point: serialize the assignment under `totals_mutex` and delegate to IJoin::setTotals.
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "RadixHashJoin::setTotals is not implemented");
}

void RadixHashJoin::checkTypesOfKeys(const Block & /*block*/) const
{
    /// Implementation point: check the left key types against the right sample (JoinCommon::checkTypesOfKeys).
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "RadixHashJoin::checkTypesOfKeys is not implemented");
}

void RadixHashJoin::onBuildPhaseFinish()
{
    /// Implementation point: the cheap build barrier only — concatenate the per-lane block stores and
    /// fold the per-lane histograms (D-0003: runs inside the last filling transform's prepare()).
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "RadixHashJoin::onBuildPhaseFinish is not implemented");
}

void RadixHashJoin::runPostBuildPhase()
{
    /// Implementation point: the heavy parallel post-build — the radix scatter of every `[ref | key]`
    /// record to its leaf array, the leaf-group layout/sizing, and the payload-resolution index.
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "RadixHashJoin::runPostBuildPhase is not implemented");
}

void RadixHashJoin::ensureTouchedGroupsBuilt(const char * /*keys*/, size_t /*rows*/)
{
    /// Implementation point (D-0004): route the block's probe keys to leaf groups and build any touched
    /// group whose tables do not exist yet — exactly once per group, parallelised over the join's pool.
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "RadixHashJoin::ensureTouchedGroupsBuilt is not implemented");
}

JoinResultPtr RadixHashJoin::joinBlock(Block block)
{
    return joinBlock(std::move(block), 0);
}

JoinResultPtr RadixHashJoin::joinBlock(Block /*block*/, size_t /*lane*/)
{
    /// Implementation point: the probe — prepare the packed keys, build the touched leaf groups, run
    /// the AMAC lookup, then gather the left columns and the right payload into the output block.
    /// Before the build barrier (the header/planning path) it must emit the output schema only.
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "RadixHashJoin::joinBlock is not implemented");
}

size_t RadixHashJoin::getTotalRowCount() const
{
    /// Implementation point: the total number of accumulated right rows.
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "RadixHashJoin::getTotalRowCount is not implemented");
}

size_t RadixHashJoin::getTotalByteCount() const
{
    /// Implementation point: the post-build byte-count snapshot (the scattered fused-record arrays).
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "RadixHashJoin::getTotalByteCount is not implemented");
}

bool RadixHashJoin::alwaysReturnsEmptySet() const
{
    /// Implementation point: true once the build completed with zero rows (inner join).
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "RadixHashJoin::alwaysReturnsEmptySet is not implemented");
}

IBlocksStreamPtr RadixHashJoin::getNonJoinedBlocks(
    const Block & /*left_sample_block*/, const Block & /*result_sample_block*/, UInt64 /*max_block_size*/) const
{
    /// Inner join only: no non-joined right rows.
    return {};
}

}
