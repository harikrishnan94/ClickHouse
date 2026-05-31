#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <vector>

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/PartitionedHashConfig.h>
#include <base/types.h>

namespace DB
{

class TableJoin;

/** PartitionedHashJoin — `join_algorithm = 'partitioned_hash'`.
  *
  * Best-Effort Partitioned Hash Join (RFC #106023). The design radix-partitions the build side into
  * cache-local leaves and streams the probe side under a dynamic, memory-pressure-driven buffer
  * budget, so a partitioned build working set beats the merged shared map of `parallel_hash` once the
  * hash table exceeds last-level cache.
  *
  * P0 (this revision): a passthrough skeleton. The class is selectable end-to-end and forwards the
  * whole `IJoin` surface to a single internal plain `HashJoin`, so results are identical to the `hash`
  * algorithm. The radix shuffle, eager leaf-HT build, custom probe transform, hysteresis eviction, and
  * auto-select land in later phases (P1..P9). `supportParallelJoin()` therefore returns `false` for now
  * (the internal `HashJoin` is not safe for concurrent build); it flips to `true` in P3 once the
  * lock-free per-leaf build exists.
  */
class PartitionedHashJoin : public IJoin
{
public:
    PartitionedHashJoin(
        std::shared_ptr<TableJoin> table_join_,
        SharedHeader right_sample_block_,
        size_t max_threads_,
        std::optional<UInt64> rhs_size_estimation_,
        size_t max_partitions_per_pass_,
        size_t shard_by_hash_input_batch_bytes_,
        bool debug_skip_passthrough_,
        bool any_take_last_row_ = false);

    std::string getName() const override { return "PartitionedHashJoin"; }
    const TableJoin & getTableJoin() const override { return *table_join; }

    bool addBlockToJoin(const Block & block, bool check_limits) override;
    void checkTypesOfKeys(const Block & block) const override;
    void initialize(const Block & left_sample_block) override;
    void onBuildPhaseFinish() override;
    JoinResultPtr joinBlock(Block block) override;

    /// The derived partition configuration (leaf count + pass schedule). For tests / diagnostics.
    const PartitionConfig & getPartitionConfig() const { return partition_config; }

    void setTotals(const Block & block) override;
    const Block & getTotals() const override;

    size_t getTotalRowCount() const override;
    size_t getTotalByteCount() const override;
    bool alwaysReturnsEmptySet() const override;

    /// See class comment: P0 keeps the build single-threaded for correctness.
    bool supportParallelJoin() const override { return false; }

    IBlocksStreamPtr
    getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const override;

private:
    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample_block;
    /// Stored for the lock-free parallel build in P3; unused until then.
    [[maybe_unused]] size_t max_threads;
    bool any_take_last_row;
    /// Input-batching threshold for the build shuffle (the `shard_by_hash_input_batch_bytes` setting that
    /// BufferedShardByHashTransform uses; 0 = flush per block).
    size_t shard_by_hash_input_batch_bytes;
    /// Diagnostic only: when true, skip the passthrough HashJoin build so the shuffle is timed alone
    /// (results are incorrect; for measurement only).
    bool debug_skip_passthrough;

    /// Partition config (leaf count + pass schedule) derived at construction from rhs_size_estimation
    /// (BYTES, §2.3) + selected-column widths + max_partitions_per_pass (spec §5).
    PartitionConfig partition_config;
    /// Right-block positions of the join key columns (for hashing/routing in addBlockToJoin).
    std::vector<size_t> key_indices;

    /// P0/P2 correctness target. Replaced by per-leaf HashJoin instances in later phases.
    std::unique_ptr<HashJoin> hash_join;

    // ── Deferred-cascade build shuffle state (single-threaded build in P2) ────────────────────────────
    // Hash is computed once per block and carried as a trailing column. Pass 0 is eager (batched input);
    // trailing passes are deferred and run per partition once it has accumulated the byte threshold, so
    // every scatter operates on a large input. The per-thread [slot] dimension is added in P3.

    /// Raw input blocks accumulated before the eager pass-0 flush.
    std::vector<Columns> pending_input;
    size_t pending_input_bytes = 0;

    /// Intermediate stage buffers. `stage_buffers[s][prefix]` is a chain of carried-hash column groups
    /// that have had passes 0..s-1 applied (valid for s in 1..numPasses-1; stage numPasses == leaves).
    std::vector<std::vector<std::vector<Columns>>> stage_buffers;
    std::vector<std::vector<size_t>> stage_buffer_bytes;

    /// Final per-leaf moved scatter-output column groups (hash dropped). Indexed by leaf in [0,total_leaves).
    std::vector<std::vector<Columns>> leaf_chains;

    std::atomic<size_t> ingested_rows{0};

    void flushPass0();
    void pushToStage(size_t stage, size_t prefix, Columns group);
    void refineBuffer(size_t stage, size_t prefix);
};

}
