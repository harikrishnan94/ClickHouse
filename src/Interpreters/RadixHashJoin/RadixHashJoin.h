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

/** RadixHashJoin — a radix-partitioned hash join exposed as `join_algorithm = 'radix_join'`.
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
  * Anything else falls back to `parallel_hash` (or plain `hash` where even that shape does not hold).
  * The constructor re-checks these and throws a LOGICAL_ERROR if violated (rather than silently
  * degrading).
  *
  * Lifecycle:
  *   addBlockToJoin      accumulate the right block (move) + count rows per leaf; no scatter, no copy.
  *   onBuildPhaseFinish  the cheap build barrier only: concatenate the per-lane block stores and fold
  *                       the per-lane histograms. Runs inside the last filling transform's prepare(),
  *                       which must stay cheap for the executor (D-0003).
  *   runPostBuildPhase   the heavy post-build, parallelised over a dedicated `ThreadPool`: the radix
  *                       scatter of every `[ref | key]` record to its leaf array, the leaf GROUP
  *                       layout/sizing (`prepareLeafTables`), and the payload-resolution index. Leaf
  *                       hash tables are NOT built here.
  *   joinBlock           probe and emit; never accumulates build rows (before the build barrier it
  *                       emits schema only). Leaf tables are built lazily at GROUP granularity on the
  *                       first probe touch of the group (D-0004): a cheap route pre-pass over the
  *                       block's keys derives the touched groups, missing ones are built exactly once
  *                       (first toucher wins, contenders spin), then the AMAC probe runs.
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
        double probe_buffer_fraction_,
        UInt64 probe_buffer_min_bytes_,
        UInt64 probe_buffer_max_bytes_,
        const StatsCollectingParams & stats_collecting_params_);

    ~RadixHashJoin() override;

    std::string getName() const override { return "RadixHashJoin"; }
    const TableJoin & getTableJoin() const override;

    /// Build is parallel: the radix build path is lock-free (one build-store slot per build lane).
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

    /// D-0003 split: `onBuildPhaseFinish` runs in the last filling transform's prepare() on this tree,
    /// so it does only the cheap build barrier; the heavy scatter runs in `runPostBuildPhase` (a work()
    /// quantum, timed under `JoinBuildPostProcessingMicroseconds`).
    void onBuildPhaseFinish() override;
    bool hasPostBuildPhase() const override { return true; }
    void runPostBuildPhase() override;

private:
    /// D-0004: given the packed (or raw single-column) probe keys of one block, route them to leaf
    /// groups and build any touched group whose tables do not exist yet — exactly once per group,
    /// parallelised over `State::pool`. Runs on the probing thread, before the AMAC lookup.
    void ensureTouchedGroupsBuilt(const char * keys, size_t rows);

    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample_block;

    size_t max_threads;
    std::optional<UInt64> rhs_size_estimation;
    UInt64 max_partitions_per_pass;
    /// When true, leaf hash tables are sized by a per-leaf HLL distinct-key estimate (only ever smaller)
    /// rather than by row count. Gated by setting `radix_join_size_tables_by_distinct_estimate`.
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
