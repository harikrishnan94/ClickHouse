#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <Core/Block.h>
#include <Core/Block_fwd.h>
#include <Core/Names.h>
#include <Interpreters/HashTablesStatistics.h>
#include <Interpreters/IJoin.h>
#include <base/types.h>
#include <Common/ThreadPool_fwd.h>

namespace DB
{

class TableJoin;
class HashJoin;

/**
 * Best-effort partitioned hash join (Zukowski, Heman, Boncz, *Architecture-Conscious Hashing*, DaMoN 2006, sec 3.1).
 *
 * The build side is radix-partitioned to leaf depth and a small per-leaf `HashJoin` is built for every
 * partition. All leaf hash tables are constructed eagerly, in parallel, in `onBuildPhaseFinish` (mirroring the
 * way `ConcurrentHashJoin` builds its slot hash tables) and are read-only for the entire probe phase.
 *
 * The probe side follows the phj-bench `idea-1-eager-build-no-sync` design: every pipeline worker keeps its OWN
 * private probe buffers, so there is no synchronization on the probe hot path. A worker is identified by its OS
 * thread id (`getThreadId()`), mapped to a dense worker slot on first touch. Each worker maintains:
 *   - `unrefined`: pass-1 (coarse) probe chains, one per coarse partition;
 *   - `leaves`:    refined probe chains, one per leaf partition.
 * On every probe block the worker scatters the block into its own coarse chains. When its buffered probe bytes
 * exceed the per-worker budget (`max_bytes_in_join_probe_buffer / max_threads`, split 1/4 unrefined / 3/4
 * leaves with hysteresis), it refines its largest coarse chain into its leaf chains and/or probes its largest
 * leaf chain against the shared read-only leaf hash table and drops it. The output of an eviction is streamed
 * back through the `joinBlock` `JoinResult`. Whatever remains buffered at end of input is drained in
 * `getDelayedBlocks()`.
 *
 * v1 supports only INNER ALL joins with a single equi-join disjunct (see `isSupported`); the planner throws
 * NOT_IMPLEMENTED otherwise.
 */
class BestEffortPartitionJoin final : public IJoin
{
public:
    BestEffortPartitionJoin(
        std::shared_ptr<TableJoin> table_join_,
        size_t max_threads_,
        SharedHeader right_sample_block_,
        size_t probe_buffer_budget_,
        size_t max_partitions_per_pass_,
        const StatsCollectingParams & stats_collecting_params_);

    ~BestEffortPartitionJoin() override;

    std::string getName() const override { return "BestEffortPartitionJoin"; }
    const TableJoin & getTableJoin() const override { return *table_join; }

    bool addBlockToJoin(const Block & block, bool check_limits) override;
    void checkTypesOfKeys(const Block & block) const override;
    JoinResultPtr joinBlock(Block block) override;

    size_t getTotalRowCount() const override { return total_rows.load(std::memory_order_relaxed); }
    size_t getTotalByteCount() const override { return total_bytes.load(std::memory_order_relaxed); }
    bool alwaysReturnsEmptySet() const override { return total_rows.load(std::memory_order_relaxed) == 0; }

    bool supportParallelJoin() const override { return true; }

    void onBuildPhaseFinish() override;

    /// INNER ALL has no non-joined right rows.
    IBlocksStreamPtr getNonJoinedBlocks(const Block &, const Block &, UInt64) const override { return nullptr; }

    bool hasDelayedBlocks() const override { return true; }
    IBlocksStreamPtr getDelayedBlocks() override;

    /// v1 scope check: INNER ALL, single equi-join disjunct, no mixed ON expression.
    static bool isSupported(const std::shared_ptr<TableJoin> & table_join);

private:
    class DrainStream;
    friend class DrainStream;

    /// Per-build-thread shard: scattered right blocks bucketed by leaf. Owned by a single build thread, so
    /// no locking is needed on the per-call append.
    struct BuildShard
    {
        explicit BuildShard(size_t total_leaves_)
            : leaf_blocks(total_leaves_)
        {
        }
        std::vector<Blocks> leaf_blocks;
    };

    /// Per-probe-worker private buffers (idea-1: touched only by the owning worker thread).
    struct ProbeWorker
    {
        ProbeWorker(size_t coarse_count_, size_t total_leaves_)
            : unrefined(coarse_count_)
            , unrefined_bytes(coarse_count_, 0)
            , leaves(total_leaves_)
            , leaf_bytes(total_leaves_, 0)
        {
        }

        std::vector<Blocks> unrefined;
        std::vector<size_t> unrefined_bytes;
        size_t total_unrefined_bytes = 0;

        std::vector<Blocks> leaves;
        std::vector<size_t> leaf_bytes;
        size_t total_leaf_bytes = 0;
    };

    struct DrainTask
    {
        size_t leaf;
        Block block;
    };

    BuildShard & getBuildShard();
    ProbeWorker & getProbeWorker();

    std::unique_ptr<HashJoin> makeLeafJoin(size_t leaf_idx) const;

    /// Move one coarse chain into the worker's leaf chains (the trailing radix passes).
    void refineCoarse(ProbeWorker & worker, size_t coarse);
    /// Probe one leaf chain against the read-only leaf hash table and append the output, then drop the chain.
    void probeAndDropLeaf(ProbeWorker & worker, size_t leaf, Blocks & out);
    /// Hysteresis trigger check after a probe block has been scattered into a worker's coarse chains.
    void evictAsNeeded(ProbeWorker & worker, Blocks & out);

    /// Monotonically increasing identifier assigned at construction; used by the thread-local
    /// ProbeWorker cache to avoid the ABA problem when a new BEP instance happens to be
    /// allocated at the same address as a recently-destroyed one.
    const uint64_t instance_id;
    static std::atomic<uint64_t> instance_counter;

    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample_block;
    size_t max_threads;
    size_t probe_buffer_budget;
    StatsCollectingParams stats_collecting_params;

    Names key_names_right;
    Names key_names_left;

    /// Radix layout. Both build and probe use the canonical `JoinCommon::scatterBlockByHash` (hash & (n-1) for
    /// power-of-two n), so a coarse partition is the low bits of a leaf partition and the levels are consistent.
    size_t coarse_count;
    size_t total_leaves;
    size_t leaves_per_coarse;

    std::unique_ptr<ThreadPool> pool;

    /// Empty hash join used for type checks and output-header derivation (never fed any rows).
    std::shared_ptr<HashJoin> sample_join;

    /// Per-leaf hash tables, built once in onBuildPhaseFinish, read-only afterwards.
    std::vector<std::shared_ptr<HashJoin>> leaf_joins;

    std::mutex build_registry_mutex;
    std::unordered_map<UInt64, size_t> build_tid_to_idx;
    std::vector<std::unique_ptr<BuildShard>> build_shards;

    std::mutex probe_registry_mutex;
    std::unordered_map<UInt64, size_t> probe_tid_to_idx;
    std::vector<std::unique_ptr<ProbeWorker>> probe_workers;

    std::atomic<size_t> total_rows{0};
    std::atomic<size_t> total_bytes{0};

    /// Drain (residual probe rows) state.
    std::once_flag drain_init_flag;
    std::vector<DrainTask> drain_tasks;
    std::atomic<size_t> drain_cursor{0};
    std::atomic<bool> drain_stream_handed{false};
    std::mutex drain_out_mutex;
    std::deque<Block> drain_pending;
};

}
