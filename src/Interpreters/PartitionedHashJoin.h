#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/PartitionedHashConfig.h>
#include <Interpreters/PartitionedHashShuffle.h>
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
  * P3 (this revision): the build side is real. `addBlockToJoin` runs the lock-free per-build-thread
  * (slot) radix shuffle into per-slot leaf chains, and `runPostBuildPhase` work-steals whole leaves to
  * build one read-only per-leaf `HashJoin` each (move-not-copy). `supportParallelJoin()` is therefore
  * `true`. Query results still ride a passthrough `HashJoin` rebuilt single-threaded in
  * `onBuildPhaseFinish` (the `[PROXY]` path) until the custom probe transform lands (P4..P8). The probe
  * scatter, hysteresis eviction, and auto-select land in later phases.
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

    /// P3: the eager leaf-HT build is a post-build step (work-stealing over the query/pipeline pool).
    bool hasPostBuildPhase() const override { return true; }
    void runPostBuildPhase() override;

    JoinResultPtr joinBlock(Block block) override;

    /// The derived partition configuration (leaf count + pass schedule). For tests / diagnostics.
    const PartitionConfig & getPartitionConfig() const { return partition_config; }

    /// Per-leaf built-HT row counts (0 for empty/unbuilt leaves), valid after runPostBuildPhase.
    /// For tests / diagnostics: cell conservation + leaf-membership checks.
    std::vector<size_t> getLeafRowCounts() const;

    void setTotals(const Block & block) override;
    const Block & getTotals() const override;

    size_t getTotalRowCount() const override;
    size_t getTotalByteCount() const override;
    bool alwaysReturnsEmptySet() const override;

    /// P3: the build is lock-free per-slot, so `FillingRightJoinSideTransform` may run in parallel.
    bool supportParallelJoin() const override { return true; }

    IBlocksStreamPtr
    getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const override;

private:
    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample_block;
    /// Number of lock-free build slots == max_threads (each concurrent build thread owns one slot).
    size_t slots;
    bool any_take_last_row;
    /// Input-batching threshold for the build shuffle (the `shard_by_hash_input_batch_bytes` setting that
    /// BufferedShardByHashTransform uses; 0 = flush per block).
    size_t shard_by_hash_input_batch_bytes;
    /// Diagnostic only: when true, skip the passthrough HashJoin build so the shuffle/leaf build is timed
    /// alone (query results become incorrect; for measurement only).
    bool debug_skip_passthrough;
    /// Measurement-only: batch each radix pass, exact-allocate output columns from histograms, then scatter.
    bool debug_prealloc_pass_scatter;
    /// Measurement-only: like prealloc pass scatter, but flush each stage when one output reaches 64K rows.
    bool debug_prealloc_stream_scatter;

    /// Process-unique id used to map a build thread to its slot (see slot handout in the .cpp). Never
    /// reused, so a stale thread-local cache entry can never alias a different join instance.
    size_t instance_id;

    /// Partition config (leaf count + pass schedule) derived at construction from rhs_size_estimation
    /// (right-side ROW count, §2.3) + selected-column widths + max_partitions_per_pass (spec §5).
    PartitionConfig partition_config;
    /// Right-block positions of the join key columns (for hashing/routing in addBlockToJoin).
    std::vector<size_t> key_indices;

    /// The `[PROXY]` query-result path: a passthrough plain HashJoin rebuilt single-threaded in
    /// onBuildPhaseFinish from the leaf chains. Removed once the custom probe transform lands.
    std::unique_ptr<HashJoin> hash_join;

    // ── Lock-free per-slot deferred-cascade build shuffle state (P3) ──────────────────────────────────
    // Each concurrent build thread owns exactly one BuildSlot, so addBlockToJoin needs no locks. The
    // deferred cascade runs inside a slot: pass 0 is eager (batched input); trailing passes are deferred
    // and run per partition once it accumulates the byte threshold. runPostBuildPhase gathers, per leaf,
    // that leaf's fragments across all slots and builds the leaf HT.

    struct BuildSlot
    {
        /// Raw input blocks accumulated before the eager pass-0 flush.
        std::vector<Columns> pending_input;
        size_t pending_input_bytes = 0;

        /// Intermediate stage buffers. `stage_buffers[s][prefix]` is a chain of column groups that have
        /// had passes 0..s-1 applied (valid for s in 1..numPasses-1; stage numPasses == leaves).
        std::vector<std::vector<std::vector<Columns>>> stage_buffers;
        std::vector<std::vector<size_t>> stage_buffer_bytes;

        /// Final per-leaf moved scatter-output column groups, indexed by leaf in [0, total_leaves).
        std::vector<std::vector<Columns>> leaf_chains;

        /// P1: reusable transient buffers for the radix scatter, allocated once and reused across every
        /// pass/partition flush of this slot (owned by one thread; no locking). See `ScatterScratch`.
        ScatterScratch scatter_scratch;
        /// P2: reusable scatter-output containers, one per cascade stage (== pass index, [0, numPasses)).
        /// A stage's children are consumed while deeper stages scatter into their own buffer, so each
        /// stage needs its own; reusing them avoids reallocating the fanout-sized container every flush.
        std::vector<std::vector<Columns>> stage_children;

        /// Measurement-only PHJ_PREALLOC_PASS_SCATTER state. Pass 0 is collected in addBlockToJoin:
        /// raw input groups, their pass-0 pids, and one exact histogram over pass-0 partitions.
        std::vector<Columns> prealloc_inputs;
        std::vector<PaddedPODArray<UInt32>> prealloc_pass0_pids;
        PaddedPODArray<UInt32> prealloc_pass0_hist;

        struct PreallocPending
        {
            std::vector<Columns> inputs;
            std::vector<PaddedPODArray<UInt32>> pids;
            PaddedPODArray<UInt32> hist;
        };

        /// Measurement-only PHJ_PREALLOC_STREAM_SCATTER state. `stream_pending[s][prefix]` buffers
        /// sources for one pass/prefix until any output partition reaches the row threshold.
        std::vector<std::vector<PreallocPending>> stream_pending;
    };

    std::vector<BuildSlot> build_slots;
    /// Hands out a fresh slot index to each distinct build thread (lock-free, §3).
    std::atomic<size_t> next_slot{0};
    /// Work-steal cursor over slots for the parallel end-of-build drain (lock-free, §3).
    std::atomic<size_t> next_drain_slot{0};
    std::atomic<size_t> ingested_rows{0};

    /// Read-only per-leaf hash tables built eagerly in runPostBuildPhase. Indexed by leaf; empty leaves
    /// stay null. Shared but read-only during the probe phase (spec invariant #3).
    std::vector<std::unique_ptr<HashJoin>> leaf_joins;
    /// Work-steal cursor over leaves for the eager build (lock-free, §3).
    std::atomic<size_t> next_leaf{0};

    /// Build-shuffle wall span: first addBlockToJoin -> end of onBuildPhaseFinish drain (ProfileEvents, once).
    std::atomic<Int64> scatter_wall_begin_ns{0};
    std::once_flag scatter_wall_begin_flag;
    std::once_flag scatter_wall_end_flag;

    /// Returns this thread's build slot, allocating one on first use (fail-close if slots exhausted).
    BuildSlot & slotForCurrentThread();
    void allocateSlotState(BuildSlot & slot) const;

    void flushPass0(BuildSlot & slot);
    void pushToStage(BuildSlot & slot, size_t stage, size_t prefix, Columns group);
    void refineBuffer(BuildSlot & slot, size_t stage, size_t prefix);

    /// End-of-build drain of one slot: flush its residual pending input through pass 0, then cascade every
    /// remaining stage buffer down to leaves. One worker owns a slot, so its scratch is used race-free.
    void drainSlot(BuildSlot & slot);

    /// Measurement-only exact-allocation pass cascade for PHJ_PREALLOC_PASS_SCATTER.
    void preallocScatterSlot(BuildSlot & slot);
    void preallocStreamPushToStage(BuildSlot & slot, size_t stage, size_t prefix, Columns group);
    void preallocStreamFlushPending(BuildSlot & slot, size_t stage, size_t prefix);
    void preallocStreamDrainSlot(BuildSlot & slot);

    /// Eager build of one leaf: gather its fragments across all slots and build the leaf HashJoin.
    /// Returns the number of build rows inserted (for cell conservation). Updates blocks_moved.
    size_t buildLeaf(size_t leaf, std::atomic<size_t> & blocks_moved);
};

}
