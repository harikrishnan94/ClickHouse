#include <Interpreters/BestEffortPartitionJoin.h>

#include <Core/Block.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/HashJoin/ScatteredBlock.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/TableJoin.h>
#include <base/defines.h>
#include <base/getThreadId.h>
#include <Common/CurrentThread.h>
#include <Common/Exception.h>
#include <Common/ThreadGroupSwitcher.h>
#include <Common/ThreadPool.h>
#include <Common/setThreadName.h>

#include <fmt/format.h>

#include <algorithm>
#include <bit>

namespace CurrentMetrics
{
extern const Metric ConcurrentHashJoinPoolThreads;
extern const Metric ConcurrentHashJoinPoolThreadsActive;
extern const Metric ConcurrentHashJoinPoolThreadsScheduled;
}

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}

namespace
{

size_t toPowerOfTwo(size_t x)
{
    if (x <= 1)
        return 1;
    return size_t{1} << (sizeof(size_t) * 8 - std::countl_zero(x - 1));
}

/// Upper bound on the number of per-leaf hash tables (and therefore HashJoin instances) we create. Keeps the
/// fixed per-query overhead bounded; comparable to ConcurrentHashJoin's 256-slot cap.
constexpr size_t LEAF_CAP = 128;

/// Fully consume the result of probing one block against a leaf hash join, appending every produced output
/// block to `out`. Handles the `IJoinResult` continuation protocol (multi-block output and `next_block`).
void drainResult(HashJoin & leaf_join, Block block, Blocks & out)
{
    Blocks pending;
    pending.push_back(std::move(block));

    while (!pending.empty())
    {
        Block cur = std::move(pending.back());
        pending.pop_back();
        if (cur.rows() == 0)
            continue;

        auto result = leaf_join.joinBlock(std::move(cur));
        while (true)
        {
            auto data = result->next();
            if (data.block.rows() != 0)
                out.push_back(std::move(data.block));

            if (data.is_last)
            {
                if (data.next_block)
                {
                    data.next_block->filterBySelector();
                    Block next_block = std::move(*data.next_block).getSourceBlock();
                    if (next_block.rows() > 0)
                        pending.push_back(std::move(next_block));
                }
                break;
            }
        }
    }
}

}

class BestEffortPartitionJoin::DrainStream : public IBlocksStream
{
public:
    explicit DrainStream(BestEffortPartitionJoin & parent_)
        : parent(parent_)
    {
    }

protected:
    Block nextImpl() override
    {
        /// Shared across all DelayedJoinedBlocksWorkerTransform threads: tasks are claimed via an atomic cursor
        /// and the read-only leaf hash tables are safe for concurrent probing.
        {
            std::lock_guard lock(parent.drain_out_mutex);
            if (!parent.drain_pending.empty())
            {
                Block block = std::move(parent.drain_pending.front());
                parent.drain_pending.pop_front();
                return block;
            }
        }

        while (true)
        {
            const size_t idx = parent.drain_cursor.fetch_add(1, std::memory_order_relaxed);
            if (idx >= parent.drain_tasks.size())
                return {};

            auto & task = parent.drain_tasks[idx];
            Blocks produced;
            drainResult(*parent.leaf_joins[task.leaf], std::move(task.block), produced);
            if (produced.empty())
                continue;

            std::lock_guard lock(parent.drain_out_mutex);
            for (size_t i = 1; i < produced.size(); ++i)
                parent.drain_pending.push_back(std::move(produced[i]));
            return std::move(produced[0]);
        }
    }

private:
    BestEffortPartitionJoin & parent;
};

class BestEffortPartitionJoinResult : public IJoinResult
{
public:
    explicit BestEffortPartitionJoinResult(Blocks blocks_)
        : blocks(std::move(blocks_))
    {
    }

    JoinResultBlock next() override
    {
        if (idx >= blocks.size())
            return {Block(), nullptr, true};

        Block block = std::move(blocks[idx]);
        ++idx;
        const bool is_last = idx >= blocks.size();
        return {std::move(block), nullptr, is_last};
    }

private:
    Blocks blocks;
    size_t idx = 0;
};

std::atomic<uint64_t> BestEffortPartitionJoin::instance_counter{0};

BestEffortPartitionJoin::BestEffortPartitionJoin(
    std::shared_ptr<TableJoin> table_join_,
    size_t max_threads_,
    SharedHeader right_sample_block_,
    size_t probe_buffer_budget_,
    size_t max_partitions_per_pass_,
    const StatsCollectingParams & stats_collecting_params_)
    : instance_id(instance_counter.fetch_add(1, std::memory_order_relaxed))
    , table_join(std::move(table_join_))
    , right_sample_block(right_sample_block_)
    , max_threads(std::max<size_t>(max_threads_, 1))
    , probe_buffer_budget(probe_buffer_budget_)
    , stats_collecting_params(stats_collecting_params_)
{
    const auto & clause = table_join->getOnlyClause();
    key_names_right = clause.key_names_right;
    key_names_left = clause.key_names_left;

    /// Coarse fan-out per pass is capped by `max_partitions_per_pass`; the leaf count is the coarse count
    /// multiplied by the number of trailing radix passes, bounded by LEAF_CAP. Both build and probe partition
    /// with the same canonical hash function, so coarse == leaf & (coarse_count - 1).
    coarse_count = toPowerOfTwo(std::clamp<size_t>(max_partitions_per_pass_, 1, LEAF_CAP));
    leaves_per_coarse = std::max<size_t>(1, toPowerOfTwo(LEAF_CAP / coarse_count));
    total_leaves = coarse_count * leaves_per_coarse;

    sample_join = std::make_shared<HashJoin>(table_join, right_sample_block, /*any_take_last_row*/ false);
    leaf_joins.resize(total_leaves);

    pool = std::make_unique<ThreadPool>(
        CurrentMetrics::ConcurrentHashJoinPoolThreads,
        CurrentMetrics::ConcurrentHashJoinPoolThreadsActive,
        CurrentMetrics::ConcurrentHashJoinPoolThreadsScheduled,
        max_threads);
}

BestEffortPartitionJoin::~BestEffortPartitionJoin() = default;

bool BestEffortPartitionJoin::isSupported(const std::shared_ptr<TableJoin> & table_join)
{
    if (table_join->getMixedJoinExpression())
        return false;
    if (!table_join->oneDisjunct())
        return false;

    const auto & clauses = table_join->getClauses();
    if (clauses.size() != 1)
        return false;
    /// Need at least one equi-join key column (JOIN ON constant / cross-like is unsupported).
    if (clauses[0].key_names_left.empty() || clauses[0].key_names_right.empty())
        return false;

    if (table_join->kind() != JoinKind::Inner)
        return false;

    const auto strictness = table_join->strictness();
    return strictness == JoinStrictness::All || strictness == JoinStrictness::Unspecified;
}

std::unique_ptr<HashJoin> BestEffortPartitionJoin::makeLeafJoin(size_t leaf_idx) const
{
    return std::make_unique<HashJoin>(
        table_join, right_sample_block, /*any_take_last_row*/ false, /*reserve_num*/ 0, fmt::format("bep_leaf{}", leaf_idx));
}

BestEffortPartitionJoin::BuildShard & BestEffortPartitionJoin::getBuildShard()
{
    const UInt64 tid = getThreadId();
    std::lock_guard lock(build_registry_mutex);
    auto it = build_tid_to_idx.find(tid);
    if (it != build_tid_to_idx.end())
        return *build_shards[it->second];

    build_tid_to_idx.emplace(tid, build_shards.size());
    build_shards.push_back(std::make_unique<BuildShard>(total_leaves));
    return *build_shards.back();
}

BestEffortPartitionJoin::ProbeWorker & BestEffortPartitionJoin::getProbeWorker()
{
    /// Thread-local cache makes the steady-state lookup lock-free (the registry mutex is taken only on the first
    /// probe block a given thread processes for this join instance).
    ///
    /// We key the cache by `instance_id` (not `this`) to avoid the ABA problem: a new BEP instance can be
    /// allocated at the same address as a recently-destroyed one, which would make the `this` comparison
    /// falsely succeed and return a freed ProbeWorker — causing a segfault on the first push_back into its
    /// now-invalid vectors. The monotonically increasing `instance_id` is unique per BEP lifetime.
    static thread_local uint64_t cached_id = std::numeric_limits<uint64_t>::max();
    static thread_local ProbeWorker * cached_worker = nullptr;
    if (cached_id == instance_id && cached_worker)
        return *cached_worker;

    const UInt64 tid = getThreadId();
    std::lock_guard lock(probe_registry_mutex);
    ProbeWorker * worker = nullptr;
    if (auto it = probe_tid_to_idx.find(tid); it != probe_tid_to_idx.end())
    {
        worker = probe_workers[it->second].get();
    }
    else
    {
        probe_tid_to_idx.emplace(tid, probe_workers.size());
        probe_workers.push_back(std::make_unique<ProbeWorker>(coarse_count, total_leaves));
        worker = probe_workers.back().get();
    }

    cached_id = instance_id;
    cached_worker = worker;
    return *worker;
}

bool BestEffortPartitionJoin::addBlockToJoin(const Block & block, bool /*check_limits*/)
{
    if (block.rows() == 0)
        return true;

    Block right_block = block;
    materializeBlockInplace(right_block);

    total_rows.fetch_add(right_block.rows(), std::memory_order_relaxed);
    total_bytes.fetch_add(right_block.bytes(), std::memory_order_relaxed);

    BuildShard & shard = getBuildShard();
    Blocks parts = JoinCommon::scatterBlockByHash(key_names_right, right_block, total_leaves);
    for (size_t leaf = 0; leaf < total_leaves; ++leaf)
    {
        if (parts[leaf].rows() != 0)
            shard.leaf_blocks[leaf].push_back(std::move(parts[leaf]));
    }

    return true;
}

void BestEffortPartitionJoin::checkTypesOfKeys(const Block & block) const
{
    sample_join->checkTypesOfKeys(block);
}

void BestEffortPartitionJoin::onBuildPhaseFinish()
{
    std::atomic<size_t> next_leaf{0};
    const size_t num_workers = std::min(max_threads, total_leaves);

    auto build_one_range = [&]
    {
        while (true)
        {
            const size_t leaf = next_leaf.fetch_add(1, std::memory_order_relaxed);
            if (leaf >= total_leaves)
                break;

            auto leaf_join = makeLeafJoin(leaf);
            for (auto & shard : build_shards)
            {
                for (auto & right_block : shard->leaf_blocks[leaf])
                    leaf_join->addBlockToJoin(right_block, /*check_limits*/ false);
                shard->leaf_blocks[leaf].clear();
                shard->leaf_blocks[leaf].shrink_to_fit();
            }
            leaf_join->onBuildPhaseFinish();
            leaf_joins[leaf] = std::move(leaf_join);
        }
    };

    if (num_workers <= 1)
    {
        build_one_range();
        return;
    }

    for (size_t i = 0; i < num_workers; ++i)
    {
        pool->scheduleOrThrowOnError(
            [&, thread_group = CurrentThread::getGroup()]
            {
                ThreadGroupSwitcher switcher(thread_group, ThreadName::CONCURRENT_JOIN);
                build_one_range();
            });
    }
    pool->wait();
}

void BestEffortPartitionJoin::refineCoarse(ProbeWorker & worker, size_t coarse)
{
    Blocks chunks = std::move(worker.unrefined[coarse]);
    worker.unrefined[coarse].clear();
    worker.total_unrefined_bytes -= worker.unrefined_bytes[coarse];
    worker.unrefined_bytes[coarse] = 0;

    for (auto & chunk : chunks)
    {
        if (chunk.rows() == 0)
            continue;

        /// Single radix pass: coarse partitions already are leaf partitions, so the refinement is a plain move.
        if (leaves_per_coarse == 1)
        {
            const size_t bytes = chunk.bytes();
            worker.leaf_bytes[coarse] += bytes;
            worker.total_leaf_bytes += bytes;
            worker.leaves[coarse].push_back(std::move(chunk));
            continue;
        }

        /// Trailing passes: rows of this coarse chain belong only to leaves whose low bits equal `coarse`.
        Blocks leaf_parts = JoinCommon::scatterBlockByHash(key_names_left, chunk, total_leaves);
        for (size_t leaf = 0; leaf < total_leaves; ++leaf)
        {
            if (leaf_parts[leaf].rows() == 0)
                continue;
            const size_t bytes = leaf_parts[leaf].bytes();
            worker.leaf_bytes[leaf] += bytes;
            worker.total_leaf_bytes += bytes;
            worker.leaves[leaf].push_back(std::move(leaf_parts[leaf]));
        }
    }
}

void BestEffortPartitionJoin::probeAndDropLeaf(ProbeWorker & worker, size_t leaf, Blocks & out)
{
    Blocks chunks = std::move(worker.leaves[leaf]);
    worker.leaves[leaf].clear();
    worker.total_leaf_bytes -= worker.leaf_bytes[leaf];
    worker.leaf_bytes[leaf] = 0;

    auto & leaf_join = *leaf_joins[leaf];
    for (auto & chunk : chunks)
    {
        if (chunk.rows() != 0)
            drainResult(leaf_join, std::move(chunk), out);
    }
}

void BestEffortPartitionJoin::evictAsNeeded(ProbeWorker & worker, Blocks & out)
{
    /// A budget of 0 means "buffer the entire probe stream"; eviction never fires and everything is drained at
    /// end of input (the algorithm degrades to a fully partitioned hash join).
    if (probe_buffer_budget == 0)
        return;

    const size_t per_worker = std::max<size_t>(1, probe_buffer_budget / max_threads);
    const size_t unrefined_high = per_worker / 4;
    const size_t unrefined_low = per_worker / 8;
    const size_t leaves_high = (per_worker * 3) / 4;
    const size_t leaves_low = (per_worker * 15) / 32;

    auto argmax = [](const std::vector<size_t> & bytes) -> long
    {
        long best = -1;
        size_t best_bytes = 0;
        for (size_t i = 0; i < bytes.size(); ++i)
        {
            if (bytes[i] > best_bytes)
            {
                best_bytes = bytes[i];
                best = static_cast<long>(i);
            }
        }
        return best;
    };

    if (worker.total_unrefined_bytes >= unrefined_high)
    {
        while (worker.total_unrefined_bytes > unrefined_low)
        {
            const long coarse = argmax(worker.unrefined_bytes);
            if (coarse < 0)
                break;
            refineCoarse(worker, static_cast<size_t>(coarse));
        }
    }

    if (worker.total_leaf_bytes >= leaves_high)
    {
        while (worker.total_leaf_bytes > leaves_low)
        {
            const long leaf = argmax(worker.leaf_bytes);
            if (leaf < 0)
                break;
            probeAndDropLeaf(worker, static_cast<size_t>(leaf), out);
        }
    }
}

JoinResultPtr BestEffortPartitionJoin::joinBlock(Block block)
{
    /// Zero-row probe (e.g. header derivation in JoiningTransform::transformHeader): delegate to the empty
    /// sample join, which produces the correct output header.
    if (block.rows() == 0)
        return sample_join->joinBlock(std::move(block));

    materializeBlockInplace(block);

    ProbeWorker & worker = getProbeWorker();
    Blocks parts = JoinCommon::scatterBlockByHash(key_names_left, block, coarse_count);
    for (size_t coarse = 0; coarse < coarse_count; ++coarse)
    {
        if (parts[coarse].rows() == 0)
            continue;
        const size_t bytes = parts[coarse].bytes();
        worker.unrefined_bytes[coarse] += bytes;
        worker.total_unrefined_bytes += bytes;
        worker.unrefined[coarse].push_back(std::move(parts[coarse]));
    }

    Blocks out;
    evictAsNeeded(worker, out);
    return std::make_unique<BestEffortPartitionJoinResult>(std::move(out));
}

IBlocksStreamPtr BestEffortPartitionJoin::getDelayedBlocks()
{
    /// First call builds the residual work list: refine every worker's leftover coarse chains and flatten the
    /// remaining leaf chains into per-(leaf, block) drain tasks.
    std::call_once(
        drain_init_flag,
        [&]
        {
            for (auto & worker_ptr : probe_workers)
            {
                ProbeWorker & worker = *worker_ptr;
                for (size_t coarse = 0; coarse < coarse_count; ++coarse)
                {
                    if (!worker.unrefined[coarse].empty())
                        refineCoarse(worker, coarse);
                }
                for (size_t leaf = 0; leaf < total_leaves; ++leaf)
                {
                    for (auto & chunk : worker.leaves[leaf])
                    {
                        if (chunk.rows() != 0)
                            drain_tasks.push_back({leaf, std::move(chunk)});
                    }
                    worker.leaves[leaf].clear();
                }
            }
        });

    /// Hand out a single shared drain stream once; it is broadcast to every worker transform, which then
    /// concurrently claim tasks from `drain_cursor`. Subsequent calls return nullptr to signal completion.
    bool expected = false;
    if (drain_stream_handed.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    {
        if (drain_tasks.empty())
            return nullptr;
        return std::make_shared<DrainStream>(*this);
    }

    return nullptr;
}

}
