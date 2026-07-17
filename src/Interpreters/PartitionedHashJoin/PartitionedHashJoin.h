#pragma once

#include <Core/Block_fwd.h>
#include <Interpreters/IJoin.h>
#include <Common/Logger.h>

#include <memory>

namespace DB
{

class TableJoin;
class HashJoin;

/** Partitioned hash join (`join_algorithm = 'partitioned_hash'`).
  *
  * Key-only scatter + partitioned hash-table build, with an unpartitioned probe:
  * - Build: right-side blocks are accumulated as-is; a build barrier picks the partition count,
  *   then the post-build phase scatters only the key columns plus an 8-byte row locator into
  *   per-partition chunks (payload columns stay in the shared row store) and builds one hash
  *   table per partition.
  * - Probe: probe blocks are never scattered or buffered; each row is routed to its leaf table
  *   by a separate routing hash, decorrelated from the hash the leaf tables bucket by.
  *
  * Current state: only the degenerate `bits = 0` plan - a single leaf that is a plain `HashJoin`,
  * which also serves as the schema delegate (`prepareRightBlock` / `savedBlockSample` /
  * `checkTypesOfKeys`) for the partitioned paths. The phase structure (fill -> barrier ->
  * post-build -> probe) and the ProfileEvents instrumentation are in place; the fill phase is
  * single-threaded (`supportParallelJoin` is false) until the partitioned build lands.
  */
class PartitionedHashJoin : public IJoin
{
public:
    PartitionedHashJoin(std::shared_ptr<TableJoin> table_join_, SharedHeader right_sample_block_, bool any_take_last_row_ = false);

    ~PartitionedHashJoin() override;

    /// Plan-time gate: whether this join shape is implemented by the partitioned algorithm.
    /// Shapes outside the predicate must be planned with another enabled algorithm instead
    /// (see `tryCreateJoin` in `Planner/PlannerJoins.cpp`) - never fail at execution time.
    static bool isSupported(const TableJoin & table_join);

    std::string getName() const override { return "PartitionedHashJoin"; }
    const TableJoin & getTableJoin() const override;

    bool addBlockToJoin(const Block & block, bool check_limits) override;
    void checkTypesOfKeys(const Block & block) const override;
    JoinResultPtr joinBlock(Block block) override;

    size_t getTotalRowCount() const override;
    size_t getTotalByteCount() const override;
    bool alwaysReturnsEmptySet() const override;

    void onBuildPhaseFinish() override;
    bool hasPostBuildPhase() const override { return true; }
    void runPostBuildPhase() override;

    IBlocksStreamPtr
    getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const override;

    bool isCloneSupported() const override;

    std::shared_ptr<IJoin>
    clone(const std::shared_ptr<TableJoin> & table_join_, SharedHeader left_sample_block_, SharedHeader right_sample_block_) const override;

    std::shared_ptr<IJoin> cloneNoParallel(
        const std::shared_ptr<TableJoin> & table_join_, SharedHeader left_sample_block_, SharedHeader right_sample_block_) const override;

    void setEnableLazyColumnsIndexing(bool value) override;

private:
    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample_block;
    const bool any_take_last_row;

    /// Partition plan, decided at the build barrier (`onBuildPhaseFinish`).
    /// Always the degenerate plan for now: one leaf, nothing to scatter.
    size_t bits = 0;
    size_t partitions = 1;

    /// The `bits = 0` leaf: a plain `HashJoin` over the whole right side. It doubles as the
    /// schema delegate (block preparation, saved block sample, key type checks) that the
    /// partitioned build/probe paths are built around.
    std::unique_ptr<HashJoin> leaf_join;

    LoggerPtr log;
};

}
