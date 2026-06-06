#pragma once

#include <Core/Block_fwd.h>
#include <Interpreters/IJoin.h>

#include <memory>
#include <optional>

namespace DB
{

class TableJoin;

/** RadixHashJoin — a radix-partitioned hash join exposed as `join_algorithm = 'radix_hash'`.
  *
  * It targets the case where the build-side hash table working set exceeds last-level cache, for a
  * fixed-width join key whose packed width is a multiple of 4 in [4, 64], by never copying build payload
  * (only key + ref are partitioned) and doing all partitioning as one deferred, exactly-sized SWWC/NT
  * scatter, then probing small per-leaf 16-byte-ish-cell hash tables that stay L2-resident. See the spec
  * `radix_hash_join_spec.md` for the full design.
  *
  * The planner gate (`radixHashJoinApplicable` in `PlannerJoins.cpp`) enforces:
  *   - single-disjunct inner ALL equi-join, no special storage
  *   - all join-key columns are fixed-width, non-nullable, non-LowCardinality
  *   - packed key width (Σ column widths) is a multiple of 4 in [4, 64]
  *
  * Only joins that satisfy every condition above reach this class; the constructor throws
  * `LOGICAL_ERROR` if any invariant is unexpectedly violated.
  *
  * Data path:
  *   addBlockToJoin    -> BuildStore::add        (move + select; no scatter, no payload copy)
  *   onBuildPhaseFinish-> BuildStore::finishBuild (merge histograms, prefix sums)
  *   runPostBuildPhase -> BuildStore::scatterToLeaves + build per-leaf HTs + next_chain + colptr
  *   joinBlock         -> probe the leaf HTs (chain traversal for JOIN ALL) and emit matches
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
        bool scatter_thp_ = false);

    ~RadixHashJoin() override;

    std::string getName() const override { return "RadixHashJoin"; }

    const TableJoin & getTableJoin() const override;

    /// Build is parallel: `FillingRightJoinSideTransform` calls `addBlockToJoin` from multiple threads.
    /// The radix build path is lock-free (one `BuildStore` slot per build thread).
    bool supportParallelJoin() const override { return true; }

    bool addBlockToJoin(const Block & block, bool check_limits) override;
    bool addBlockToJoin(const Block & block, size_t num_rows, bool check_limits) override;

    void checkTypesOfKeys(const Block & block) const override;

    JoinResultPtr joinBlock(Block block) override;

    size_t getTotalRowCount() const override;
    size_t getTotalByteCount() const override;
    bool alwaysReturnsEmptySet() const override;

    IBlocksStreamPtr getNonJoinedBlocks(
        const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const override;

    void onBuildPhaseFinish() override;
    bool hasPostBuildPhase() const override;
    void runPostBuildPhase() override;

private:
    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample_block;

    size_t max_threads;
    std::optional<UInt64> rhs_size_estimation;
    UInt64 max_partitions_per_pass;
    bool scatter_thp;

    /// All radix-path state (build store, leaf HTs, colptr tables, output plan). Defined in the
    /// .cpp so this header stays free of the RadixHash internals.
    struct RadixState;
    std::unique_ptr<RadixState> state;
};

}
