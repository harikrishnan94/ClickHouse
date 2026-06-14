#pragma once

#include <Core/Block_fwd.h>
#include <Interpreters/HashTablesStatistics.h>
#include <Interpreters/IJoin.h>

#include <memory>
#include <mutex>
#include <optional>

namespace DB
{

class TableJoin;

/** RadixHashJoin — a radix-partitioned hash join exposed as `join_algorithm = 'radix_hash'`.
  *
  * It targets the case where the build-side hash table working set exceeds last-level cache, for a
  * fixed-width join key whose packed width is a multiple of 4 in [4, 64]. The idea: never copy build
  * payload (only the key and an 8-byte build reference are partitioned), do all the partitioning as one
  * deferred, exactly-sized scatter, and probe small per-leaf hash tables that each stay L2-resident.
  * Where `parallel_hash` probes one shared map that has fallen out of LLC (a cold miss per lookup),
  * this probes a cache-hot leaf — that lookup locality is the win; the conscious trade-off is that the
  * payload gather stays random across the build blocks (payload is not co-located).
  *
  * The planner gate (`radixHashJoinApplicable` in PlannerJoins.cpp) admits only:
  *   - a single-disjunct inner ALL equi-join with no special storage, and
  *   - join key columns that are all fixed-width, non-nullable, non-LowCardinality, whose packed width
  *     (the sum of the column widths) is a multiple of 4 in [4, 64].
  * Anything else falls back to `parallel_hash`. The constructor re-checks these and throws a
  * LOGICAL_ERROR if violated (rather than silently degrading).
  *
  * Lifecycle:
  *   addBlockToJoin     accumulate the right block (move) + count rows per leaf; no scatter, no copy.
  *   onBuildPhaseFinish merge + prefix-sum the per-thread histograms (the build barrier), then run the
  *                      whole post-build eagerly on a dedicated `ThreadPool` (scatter + leaf-table build
  *                      + payload-resolution index). The join is fully built when this returns.
  *   joinBlock          probe and emit; never builds (before the build barrier it emits schema only).
  */
class RadixHashJoin : public IJoin
{
public:
    RadixHashJoin(
        std::shared_ptr<TableJoin> table_join_,
        SharedHeader right_sample_block_,
        size_t max_threads_,
        std::optional<UInt64> rhs_size_estimation_,
        UInt64 max_partitions_per_pass_,
        bool size_tables_by_distinct_estimate_,
        const StatsCollectingParams & stats_collecting_params_);

    ~RadixHashJoin() override;

    std::string getName() const override { return "RadixHashJoin"; }
    const TableJoin & getTableJoin() const override;

    /// Build is parallel: the radix build path is lock-free (one build-store slot per build thread).
    bool supportParallelJoin() const override { return true; }

    bool addBlockToJoin(const Block & block, bool check_limits) override;
    bool addBlockToJoin(const Block & block, size_t num_rows, bool check_limits) override;
    bool addBlockToJoin(const Block & block, size_t num_rows, bool check_limits, size_t build_lane) override;

    void checkTypesOfKeys(const Block & block) const override;
    JoinResultPtr joinBlock(Block block) override;
    JoinResultPtr joinBlock(Block block, size_t lane) override;

    /// The parallel build transforms each call setTotals concurrently on this shared object; serialize
    /// the assignment (the base does an unguarded `totals = block`). getTotals stays unlocked (read
    /// only after the build completes).
    void setTotals(const Block & block) override;

    size_t getTotalRowCount() const override;
    size_t getTotalByteCount() const override;
    bool alwaysReturnsEmptySet() const override;

    IBlocksStreamPtr getNonJoinedBlocks(
        const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const override;

    void onBuildPhaseFinish() override;

private:
    /// The eager post-build, run once from `onBuildPhaseFinish` and parallelised over `State::pool`:
    /// scatter to leaves + leaf-table build + the build-row payload-resolution index.
    void runPostBuild();

    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample_block;

    size_t max_threads;
    std::optional<UInt64> rhs_size_estimation;
    UInt64 max_partitions_per_pass;
    /// When true, leaf hash tables are sized by a per-leaf HLL distinct-key estimate (only ever smaller)
    /// rather than by row count. Gated by setting `radix_hash_join_size_tables_by_distinct_estimate`.
    bool size_tables_by_distinct_estimate;

    /// Cross-run hash-table statistics ("the stats"): keyed by the query plan, this lets a warm run reuse
    /// the previous run's distinct-key estimate and skip the per-leaf HLL estimation entirely. Disabled
    /// (key == 0) when `collect_hash_table_stats_during_joins` is off, in which case every run runs the HLL.
    StatsCollectingParams stats_collecting_params;

    std::mutex totals_mutex;

    /// All radix-path state lives in the .cpp so this header stays free of the internals.
    struct State;
    std::unique_ptr<State> state;
};

}
