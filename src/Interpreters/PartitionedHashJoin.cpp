#include <Interpreters/PartitionedHashJoin.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include <Columns/ColumnVector.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/PartitionedHashShuffle.h>
#include <Interpreters/TableJoin.h>
#include <Common/CurrentThread.h>
#include <Common/Exception.h>
#include <Common/ProfileEvents.h>
#include <Common/Stopwatch.h>
#include <Common/ThreadGroupSwitcher.h>
#include <Common/ThreadPool.h>
#include <Common/assert_cast.h>
#include <Common/logger_useful.h>
#include <Common/setThreadName.h>

namespace ProfileEvents
{
extern const Event PartitionedHashBuildShuffleMicroseconds;
extern const Event PartitionedHashBuildShufflePass0Microseconds;
extern const Event PartitionedHashBuildShuffleTrailingMicroseconds;
extern const Event PartitionedHashBuildScatterRows;
extern const Event PartitionedHashBuildBlocksMoved;
extern const Event PartitionedHashBuildHTMicroseconds;
extern const Event PartitionedHashBuildFinishDrainMicroseconds;
}

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}

namespace
{

/// Per-row fixed width of a (sample, empty) column; falls back to a coarse estimate for variable types.
size_t fixedRowBytes(const ColumnPtr & col)
{
    if (col->valuesHaveFixedSize())
        return col->sizeOfValueIfFixed();
    return 16;
}

size_t groupRows(const Columns & group)
{
    return group.empty() ? 0 : group[0]->size();
}

size_t groupBytes(const Columns & group)
{
    size_t bytes = 0;
    for (const auto & col : group)
        bytes += col->byteSize();
    return bytes;
}

/// MEASUREMENT-ONLY: collect per-subtable cell-buffer sizes from a HashJoin map (single-level or two-level).
struct SubtableSizeStats
{
    size_t count = 0;
    size_t nonempty = 0;
    size_t sum_bytes = 0;
    size_t min_bytes = std::numeric_limits<size_t>::max();
    size_t max_bytes = 0;
    std::vector<size_t> sizes;

    void add(size_t bytes)
    {
        ++count;
        if (bytes > 0)
            ++nonempty;
        sum_bytes += bytes;
        min_bytes = std::min(min_bytes, bytes);
        max_bytes = std::max(max_bytes, bytes);
        sizes.push_back(bytes);
    }

    size_t medianBytes() const
    {
        if (sizes.empty())
            return 0;
        auto sorted = sizes;
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() / 2];
    }
};

void collectHashJoinSubtableSizes(const HashJoin & join, SubtableSizeStats & stats)
{
    const auto & data = join.getJoinedData();
    if (!data || data->maps.empty())
        return;

    const auto type = data->type;
    std::visit(
        [&](const auto & maps)
        {
            switch (type)
            {
                case HashJoin::Type::key64:
                    if (maps.key64)
                        stats.add(maps.key64->getBufferSizeInBytes());
                    break;
                case HashJoin::Type::two_level_key64:
                    if (maps.two_level_key64)
                        for (const auto & impl : maps.two_level_key64->impls)
                            stats.add(impl.getBufferSizeInBytes());
                    break;
                default:
                    break;
            }
        },
        data->maps.at(0));
}

void prefaultHashJoinMapBuffers(HashJoin & join)
{
    const auto & data = join.getJoinedData();
    if (!data || data->maps.empty())
        return;

    const auto type = data->type;
    std::visit(
        [&](auto & maps)
        {
            switch (type)
            {
                case HashJoin::Type::key64:
                    if (maps.key64)
                        maps.key64->prefaultBufferPages();
                    break;
                case HashJoin::Type::two_level_key64:
                    if (maps.two_level_key64)
                        for (auto & impl : maps.two_level_key64->impls)
                            impl.prefaultBufferPages();
                    break;
                default:
                    break;
            }
        },
        data->maps.at(0));
}

void logSubtableSizeStats(const char * label, const SubtableSizeStats & stats)
{
    if (stats.count == 0)
    {
        LOG_INFO(getLogger("PartitionedHashJoin"), "{} subtable sizes: (none)", label);
        return;
    }
    size_t at_min = 0;
    size_t at_2mib = 0;
    size_t at_max = 0;
    for (size_t b : stats.sizes)
    {
        if (b == stats.min_bytes)
            ++at_min;
        if (b == 2097152)
            ++at_2mib;
        if (b == stats.max_bytes)
            ++at_max;
    }
    LOG_INFO(
        getLogger("PartitionedHashJoin"),
        "{} subtable sizes: count={} nonempty={} min={} median={} max={} sum={} avg={} at_min={} at_2MiB={} at_max={}",
        label,
        stats.count,
        stats.nonempty,
        stats.min_bytes == std::numeric_limits<size_t>::max() ? 0 : stats.min_bytes,
        stats.medianBytes(),
        stats.max_bytes,
        stats.sum_bytes,
        stats.count ? stats.sum_bytes / stats.count : 0,
        at_min,
        at_2mib,
        at_max);
}

/// Process-unique id generator: a join id is never reused, so a stale thread-local slot-cache entry can
/// never alias a different join at the same address.
std::atomic<size_t> g_instance_counter{0};

/// Single-entry per-thread cache mapping the join currently being built to this thread's build slot.
/// A pipeline-executor worker thread runs one processor's `work()` to completion at a time, so it builds
/// at most one join instance at any instant; the single entry is sufficient (and never grows).
struct SlotCacheEntry
{
    size_t instance_id = std::numeric_limits<size_t>::max();
    size_t slot = 0;
};
thread_local SlotCacheEntry slot_cache;

/// Run `worker(worker_idx)` on `num_workers` threads, reusing the query/pipeline global pool: spawn
/// `num_workers-1` ThreadFromGlobalPool workers (they borrow threads from the same global pool the
/// pipeline executor uses — no private ThreadPool) plus one inline on the current thread; each worker
/// attaches to the query thread group for memory / ProfileEvents accounting. Exceptions are captured
/// per worker and rethrown after the join, never escaping a pool thread. The actual work-stealing
/// (an atomic cursor over the items) lives inside `worker`. All threads are joined before returning.
template <typename Worker>
void runParallelWorkers(size_t num_workers, Worker && worker)
{
    num_workers = std::max<size_t>(num_workers, 1);
    std::vector<std::exception_ptr> exceptions(num_workers);
    auto run = [&](size_t idx)
    {
        try
        {
            worker(idx);
        }
        catch (...)
        {
            exceptions[idx] = std::current_exception();
        }
    };

    auto thread_group = CurrentThread::getGroup();
    std::vector<ThreadFromGlobalPool> threads;
    threads.reserve(num_workers - 1);
    try
    {
        for (size_t i = 1; i < num_workers; ++i)
            threads.emplace_back(
                [&run, i, thread_group]
                {
                    ThreadGroupSwitcher switcher(thread_group, ThreadName::PARTITIONED_JOIN);
                    run(i);
                });
    }
    catch (...)
    {
        /// Could not spawn all workers: the remaining items are still processed by the spawned workers
        /// and the inline worker via work-stealing (just with less parallelism). Already-spawned threads
        /// are joined below; never leave a ThreadFromGlobalPool unjoined.
        LOG_WARNING(
            getLogger("PartitionedHashJoin"),
            "Could not spawn all {} build workers ({}); continuing with fewer via work-stealing",
            num_workers,
            getCurrentExceptionMessage(false));
    }

    /// Run one worker inline on the current thread (already attached to the query group), then join.
    run(0);
    for (auto & t : threads)
        t.join();

    for (const auto & e : exceptions)
        if (e)
            std::rethrow_exception(e);
}

}

PartitionedHashJoin::PartitionedHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader right_sample_block_,
    size_t max_threads_,
    std::optional<UInt64> rhs_size_estimation_,
    size_t max_partitions_per_pass_,
    size_t shard_by_hash_input_batch_bytes_,
    bool debug_skip_passthrough_,
    bool any_take_last_row_)
    : table_join(std::move(table_join_))
    , right_sample_block(std::move(right_sample_block_))
    , slots(std::max<size_t>(max_threads_, 1))
    , any_take_last_row(any_take_last_row_)
    , shard_by_hash_input_batch_bytes(shard_by_hash_input_batch_bytes_)
    , debug_skip_passthrough(debug_skip_passthrough_)
    , instance_id(g_instance_counter.fetch_add(1, std::memory_order_relaxed))
    , hash_join(std::make_unique<HashJoin>(table_join, right_sample_block, any_take_last_row))
{
    /// Map the right-side join key names to positions in the right sample block.
    const auto & key_names_right = table_join->getOnlyClause().key_names_right;
    std::vector<bool> is_key(right_sample_block->columns(), false);
    for (const auto & name : key_names_right)
    {
        const size_t pos = right_sample_block->getPositionByName(name);
        key_indices.push_back(pos);
        is_key[pos] = true;
    }

    /// Per-row byte widths of the selected right columns (key vs payload), for leaf-count derivation.
    /// rhs_size_estimation from the planner is the right-side ROW count (right_rows_estimation).
    PartitionConfigInputs cfg_inputs;
    cfg_inputs.rhs_rows_estimation = rhs_size_estimation_;
    cfg_inputs.max_partitions_per_pass = max_partitions_per_pass_;
    cfg_inputs.key_bytes = 0;
    cfg_inputs.payload_bytes = 0;
    const Columns sample_cols = right_sample_block->getColumns();
    for (size_t i = 0; i < sample_cols.size(); ++i)
    {
        const size_t w = fixedRowBytes(sample_cols[i]);
        if (is_key[i])
            cfg_inputs.key_bytes += w;
        else
            cfg_inputs.payload_bytes += w;
    }
    if (cfg_inputs.key_bytes == 0)
        cfg_inputs.key_bytes = 8;

    partition_config = derivePartitionConfig(cfg_inputs);

    /// One lock-free build slot per concurrent build thread.
    build_slots.resize(slots);
    for (auto & slot : build_slots)
        allocateSlotState(slot);

    LOG_DEBUG(
        getLogger("PartitionedHashJoin"),
        "Partition config: {} leaves, {} passes, key_bytes={}, payload_bytes={}, batch_bytes={}, slots={}",
        partition_config.total_leaves,
        partition_config.numPasses(),
        cfg_inputs.key_bytes,
        cfg_inputs.payload_bytes,
        shard_by_hash_input_batch_bytes,
        slots);
}

void PartitionedHashJoin::allocateSlotState(BuildSlot & slot) const
{
    slot.leaf_chains.resize(partition_config.total_leaves);

    /// Intermediate stage buffers for stages 1..numPasses-1 (stage numPasses == leaves).
    const size_t num_passes = partition_config.numPasses();
    slot.stage_buffers.resize(num_passes);
    slot.stage_buffer_bytes.resize(num_passes);
    /// One reusable scatter-output buffer per pass (P2). flushPass0 uses index 0; refineBuffer(stage) uses
    /// index `stage`. Sized to numPasses (empty when there is no partitioning; never indexed then).
    slot.stage_children.resize(num_passes);
    UInt8 bits_so_far = 0;
    for (size_t s = 1; s < num_passes; ++s)
    {
        bits_so_far = static_cast<UInt8>(bits_so_far + partition_config.pass_bits[s - 1]);
        const size_t count = size_t{1} << bits_so_far;
        slot.stage_buffers[s].resize(count);
        slot.stage_buffer_bytes[s].assign(count, 0);
    }
}

PartitionedHashJoin::BuildSlot & PartitionedHashJoin::slotForCurrentThread()
{
    if (slot_cache.instance_id != instance_id)
    {
        const size_t slot = next_slot.fetch_add(1, std::memory_order_relaxed);
        /// Fail-close: the build pipeline must not run more concurrent threads than slots (== max_threads).
        if (slot >= slots)
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "PartitionedHashJoin build-slot overflow: at least {} distinct build threads, but only {} slots",
                slot + 1,
                slots);
        slot_cache.instance_id = instance_id;
        slot_cache.slot = slot;
    }
    return build_slots[slot_cache.slot];
}

bool PartitionedHashJoin::addBlockToJoin(const Block & block, bool /*check_limits*/)
{
    /// P3: shuffle-only. The right block is radix-shuffled, lock-free, into THIS thread's build slot.
    /// The passthrough HashJoin (the `[PROXY]` query-result path) is rebuilt single-threaded later, in
    /// onBuildPhaseFinish, because the plain HashJoin is not safe for concurrent build.
    const size_t rows = block.rows();
    if (rows == 0)
        return true;

    ingested_rows.fetch_add(rows, std::memory_order_relaxed);

    BuildSlot & slot = slotForCurrentThread();

    /// Degenerate single-leaf config: no partitioning, collect raw blocks into leaf 0.
    if (partition_config.numPasses() == 0)
    {
        slot.leaf_chains[0].push_back(block.getColumns());
        return true;
    }

    /// Pass 0 is eager and batched like BufferedShardByHashTransform: accumulate input, flush per block
    /// when the setting is 0, else once the accumulated byte threshold is crossed.
    slot.pending_input.push_back(block.getColumns());
    slot.pending_input_bytes += block.bytes();
    if (shard_by_hash_input_batch_bytes == 0 || slot.pending_input_bytes >= shard_by_hash_input_batch_bytes)
        flushPass0(slot);

    return true;
}

void PartitionedHashJoin::flushPass0(BuildSlot & slot)
{
    if (slot.pending_input.empty())
        return;

    /// One multi-source pass-0 scatter over the accumulated input batch into the fanout0 stage-1
    /// partitions (re-deriving the hash from the key columns; nothing carried).
    const UInt32 fanout0 = UInt32{1} << partition_config.pass_bits[0];
    const UInt32 shift0 = partition_config.shiftForPass(0);

    size_t scattered_rows = 0;
    Stopwatch watch;
    auto & children = slot.stage_children[0];
    scatterGroupsByKeyHash(slot.pending_input, key_indices, shift0, fanout0, scattered_rows, slot.scatter_scratch, children);
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildShufflePass0Microseconds, watch.elapsedMicroseconds());
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildShuffleMicroseconds, watch.elapsedMicroseconds());
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildScatterRows, scattered_rows);
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildBlocksMoved, slot.pending_input.size());

    slot.pending_input.clear();
    slot.pending_input_bytes = 0;

    for (UInt32 pid = 0; pid < fanout0; ++pid)
        pushToStage(slot, 1, pid, std::move(children[pid]));
}

void PartitionedHashJoin::pushToStage(BuildSlot & slot, size_t stage, size_t prefix, Columns group)
{
    if (groupRows(group) == 0)
        return;

    /// Leaf: append to this slot's leaf chain.
    if (stage == partition_config.numPasses())
    {
        slot.leaf_chains[prefix].push_back(std::move(group));
        return;
    }

    /// Intermediate stage: accumulate; refine when the partition reaches the batch threshold (or always
    /// when the setting is 0).
    slot.stage_buffer_bytes[stage][prefix] += groupBytes(group);
    slot.stage_buffers[stage][prefix].push_back(std::move(group));
    if (shard_by_hash_input_batch_bytes == 0 || slot.stage_buffer_bytes[stage][prefix] >= shard_by_hash_input_batch_bytes)
        refineBuffer(slot, stage, prefix);
}

void PartitionedHashJoin::refineBuffer(BuildSlot & slot, size_t stage, size_t prefix)
{
    auto & chain = slot.stage_buffers[stage][prefix];
    if (chain.empty())
        return;

    const UInt32 fanout = UInt32{1} << partition_config.pass_bits[stage];
    const UInt32 shift = partition_config.shiftForPass(stage);

    size_t scattered_rows = 0;
    Stopwatch watch;
    auto & children = slot.stage_children[stage];
    scatterGroupsByKeyHash(chain, key_indices, shift, fanout, scattered_rows, slot.scatter_scratch, children);
    const auto us = watch.elapsedMicroseconds();
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildShuffleTrailingMicroseconds, us);
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildShuffleMicroseconds, us);
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildScatterRows, scattered_rows);

    chain.clear();
    slot.stage_buffer_bytes[stage][prefix] = 0;

    for (UInt32 pid = 0; pid < fanout; ++pid)
        pushToStage(slot, stage + 1, prefix * fanout + pid, std::move(children[pid]));
}

void PartitionedHashJoin::drainSlot(BuildSlot & slot)
{
    /// Flush this slot's residual pending input through pass 0, then cascade every remaining stage buffer
    /// down to leaves. Each slot is owned by a single drain worker, so its scratch / stage_children are
    /// used race-free (no locks).
    flushPass0(slot);
    for (size_t stage = 1; stage < partition_config.numPasses(); ++stage)
        for (size_t prefix = 0; prefix < slot.stage_buffers[stage].size(); ++prefix)
            refineBuffer(slot, stage, prefix);
}

void PartitionedHashJoin::onBuildPhaseFinish()
{
    /// Drain the residual per-slot shuffle buffers down to leaves. At high thread counts most trailing-pass
    /// work lands here (per-slot, per-partition batches stay below the flush threshold during streaming),
    /// so this used to be a single-threaded Amdahl tail dominating the build. It is now parallelised by
    /// work-stealing whole slots over the query/pipeline pool: one worker fully drains a slot, so its
    /// scratch is exclusive and the path stays lock-free (spec §4.2, §9.3).
    Stopwatch drain_watch;
    next_drain_slot.store(0, std::memory_order_relaxed);
    runParallelWorkers(
        slots,
        [this](size_t)
        {
            size_t s;
            while ((s = next_drain_slot.fetch_add(1, std::memory_order_relaxed)) < slots)
                drainSlot(build_slots[s]);
        });
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildFinishDrainMicroseconds, drain_watch.elapsedMicroseconds());

    /// Runtime conservation check on the REAL workload: rows across all slot leaf chains == ingested.
    size_t leaf_rows = 0;
    for (const auto & slot : build_slots)
        for (const auto & chain : slot.leaf_chains)
            for (const auto & group : chain)
                leaf_rows += groupRows(group);

    const size_t ingested = ingested_rows.load(std::memory_order_relaxed);
    if (leaf_rows != ingested)
        LOG_ERROR(
            getLogger("PartitionedHashJoin"),
            "Build shuffle row conservation VIOLATED: leaf chains have {} rows but {} were ingested",
            leaf_rows,
            ingested);
    else
        LOG_DEBUG(
            getLogger("PartitionedHashJoin"),
            "Build shuffle row conservation OK: {} rows across {} leaves, {} slots",
            leaf_rows,
            partition_config.total_leaves,
            slots);

    /// Fill the passthrough HashJoin (constructed empty for header/type-check use) single-threaded for the
    /// `[PROXY]` query-result path, reusing the already-shuffled leaf blocks (copies only the shared
    /// ColumnPtrs; no payload copy). It runs BEFORE runPostBuildPhase moves those chains into the leaf
    /// HTs. Skipped for isolated build timing (query results then incorrect; measurement only).
    if (!debug_skip_passthrough)
    {
        for (const auto & slot : build_slots)
            for (const auto & chain : slot.leaf_chains)
                for (const auto & group : chain)
                {
                    if (groupRows(group) == 0)
                        continue;
                    Block block = right_sample_block->cloneWithColumns(group);
                    hash_join->addBlockToJoin(block, /*check_limits=*/false);
                }
        hash_join->onBuildPhaseFinish();
    }
}

size_t PartitionedHashJoin::buildLeaf(size_t leaf, std::atomic<size_t> & blocks_moved)
{
    /// Gather this leaf's fragments across all slots (disjoint leaf ownership -> no locks).
    size_t leaf_rows = 0;
    for (auto & slot : build_slots)
        for (auto & group : slot.leaf_chains[leaf])
            leaf_rows += groupRows(group);

    if (leaf_rows == 0)
        return 0;

    /// MEASUREMENT-ONLY (revert before commit): PHJ_SIMPLE_HT builds each leaf with a minimal 16-byte-cell
    /// open-addressing table (inner join, single UInt64 key) instead of the general `HashJoin`/`RowRefList`,
    /// to isolate the cold per-leaf HT-fill cost from the HashJoin machinery. Result is NOT probe-usable
    /// (build timing only); use with an empty probe + skip-passthrough.
    static const bool simple_ht = std::getenv("PHJ_SIMPLE_HT") != nullptr;
    if (simple_ht)
    {
        struct Cell
        {
            UInt64 key;
            UInt64 val;
        };
        static constexpr UInt64 EMPTY = std::numeric_limits<UInt64>::max();
        const size_t cap = std::bit_ceil(std::max<size_t>(leaf_rows * 2, size_t{16}));
        const size_t mask = cap - 1;
        std::vector<Cell> cells(cap, Cell{0, EMPTY});
        auto mix = [](UInt64 x) noexcept
        {
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53ULL;
            x ^= x >> 33;
            return x;
        };
        const size_t key_pos = key_indices[0];
        size_t n = 0;
        for (auto & slot : build_slots)
        {
            for (auto & group : slot.leaf_chains[leaf])
            {
                if (groupRows(group) == 0)
                    continue;
                const auto & keycol = assert_cast<const ColumnVector<UInt64> &>(*group[key_pos]);
                const UInt64 * keys = keycol.getData().data();
                const size_t rows = keycol.size();
                for (size_t i = 0; i < rows; ++i)
                {
                    const UInt64 k = keys[i];
                    size_t pos = mix(k) & mask;
                    while (cells[pos].val != EMPTY)
                    {
                        if (cells[pos].key == k)
                            break;
                        pos = (pos + 1) & mask;
                    }
                    if (cells[pos].val == EMPTY)
                    {
                        cells[pos].key = k;
                        cells[pos].val = i;
                        ++n;
                    }
                }
            }
            slot.leaf_chains[leaf].clear();
            slot.leaf_chains[leaf].shrink_to_fit();
        }
        /// Make the inserts observable so the build cannot be elided; leaf_joins[leaf] stays null (no probe).
        blocks_moved.fetch_add(n, std::memory_order_relaxed);
        return leaf_rows;
    }

    /// Reserve the leaf HT to its exact final size: the leaf row count is already known here, so the map
    /// is allocated once instead of growing through several resize() reallocations. This both removes the
    /// resize cost and collapses many small allocations into one, which sharply reduces the allocator
    /// (jemalloc extent) contention that otherwise serialises the concurrent per-leaf build.
    /// Each per-leaf HT is sized to ~L2 (well below the global 8 MB prefetch threshold), so software prefetch
    /// would never engage with the default heuristic — this matches the landed phj-p3 behaviour (no prefetch
    /// on the build side). MEASUREMENT-ONLY (revert before commit): PHJ_FORCE_PREFETCH=1 lowers each leaf HT's
    /// prefetch threshold to 0 so the insert kernel issues software prefetches, for an A/B against the default.
    static const bool force_prefetch = std::getenv("PHJ_FORCE_PREFETCH") != nullptr;
    /// MEASUREMENT-ONLY (revert before commit): PHJ_LEAF_TWOLEVEL=1 builds each leaf HashJoin with a
    /// TwoLevelHashMap (256 sub-buckets) — the same map type `parallel_hash`/ConcurrentHashJoin uses — to
    /// isolate whether the single-level-vs-two-level insert kernel (and its prefetch gating, since a
    /// two-level map reports the aggregate >8 MB buffer size) explains the build-CPU gap vs `parallel_hash`.
    static const bool leaf_twolevel = std::getenv("PHJ_LEAF_TWOLEVEL") != nullptr;
    /// PHJ_PREWARM_MAPS / PHJ_SERIAL_ALLOC: the map was already allocated and reserved in the pre-pass; reuse it.
    static const bool prealloc_insert = std::getenv("PHJ_PREWARM_MAPS") != nullptr || std::getenv("PHJ_SERIAL_ALLOC") != nullptr;
    /// MEASUREMENT-ONLY: PHJ_MADV_POPULATE=1 prefaults each leaf map's cell array via MADV_POPULATE_WRITE
    /// after reserve, before insert — tests whether eager prefault removes the parallel CoW-fault IPI storm.
    static const bool madv_populate = std::getenv("PHJ_MADV_POPULATE") != nullptr;
    std::unique_ptr<HashJoin> leaf_join;
    if (prealloc_insert && leaf_joins[leaf])
        leaf_join = std::move(leaf_joins[leaf]);
    else
    {
        leaf_join = std::make_unique<HashJoin>(
            table_join,
            right_sample_block,
            any_take_last_row,
            /*reserve_num=*/leaf_rows,
            /*instance_id=*/"",
            /*use_two_level_maps=*/leaf_twolevel,
            /*force_enable_prefetch=*/force_prefetch);
        if (madv_populate)
            prefaultHashJoinMapBuffers(*leaf_join);
    }

    size_t moved = 0;
    for (auto & slot : build_slots)
    {
        for (auto & group : slot.leaf_chains[leaf])
        {
            if (groupRows(group) == 0)
                continue;
            /// Hand the scattered columns to the leaf HashJoin's block store: cloneWithColumns shares the
            /// ColumnPtrs (no payload memcpy / insertRangeFrom) and clearing our chain right after leaves
            /// the leaf HT as the sole owner -> effective move, the column DATA is never copied (ZC gate,
            /// §9.4).
            Block block = right_sample_block->cloneWithColumns(group);
            leaf_join->addBlockToJoin(block, /*check_limits=*/false);
            ++moved;
        }
        slot.leaf_chains[leaf].clear();
        slot.leaf_chains[leaf].shrink_to_fit();
    }

    leaf_join->onBuildPhaseFinish();
    blocks_moved.fetch_add(moved, std::memory_order_relaxed);
    leaf_joins[leaf] = std::move(leaf_join);
    return leaf_rows;
}

void PartitionedHashJoin::runPostBuildPhase()
{
    const size_t total_leaves = partition_config.total_leaves;
    leaf_joins.clear();
    leaf_joins.resize(total_leaves);
    next_leaf.store(0, std::memory_order_relaxed);

    /// MEASUREMENT-ONLY (revert before commit): PHJ_SKIP_LEAF_BUILD returns before the eager leaf-HT build,
    /// leaving the scattered per-slot chains in place. Together with debug_skip_passthrough this isolates the
    /// build-side scatter + source scan from the leaf-HT construction, so a perf/iMC A/B (full vs skip) yields
    /// the leaf-HT build's integral DRAM bytes, faults, TLB, IPC, etc. Result is NOT probe-usable.
    static const bool skip_leaf_build = std::getenv("PHJ_SKIP_LEAF_BUILD") != nullptr;
    if (skip_leaf_build)
        return;

    /// MEASUREMENT-ONLY: PHJ_PREWARM_MAPS=1 splits runPostBuildPhase into two parallel passes:
    ///   pass 1: count rows per leaf + construct+reserve the leaf HashJoin (warms all cell-array pages).
    ///   pass 2: insert rows into the already-warmed maps.
    /// Hypothesis: if the IPI-storm root cause is 16 workers simultaneously writing to cold CoW pages
    /// WHILE ALSO doing insertions, separating the two phases should show whether pre-warming helps.
    /// If IPIs still dominate after pre-warming, the cause is the parallel allocation itself (not the
    /// interleaving with inserts). Result is probe-usable.
    static const bool prewarm_maps = std::getenv("PHJ_PREWARM_MAPS") != nullptr;
    /// MEASUREMENT-ONLY: PHJ_SERIAL_ALLOC=1 allocates+reserves all leaf maps on one thread (optionally
    /// with PHJ_MADV_POPULATE), then parallel workers insert only — tests whether serializing allocation
    /// removes cross-core TLB-shootdown IPIs while keeping parallel insert.
    static const bool serial_alloc = std::getenv("PHJ_SERIAL_ALLOC") != nullptr;
    static const bool madv_populate_build = std::getenv("PHJ_MADV_POPULATE") != nullptr;
    static const bool leaf_twolevel_build = std::getenv("PHJ_LEAF_TWOLEVEL") != nullptr;
    static const bool force_prefetch_build = std::getenv("PHJ_FORCE_PREFETCH") != nullptr;
    if (prewarm_maps || serial_alloc)
    {
        auto alloc_one_leaf = [&](size_t leaf)
        {
            size_t cnt = 0;
            for (auto & slot : build_slots)
                for (const auto & group : slot.leaf_chains[leaf])
                    cnt += groupRows(group);
            if (cnt > 0)
            {
                leaf_joins[leaf] = std::make_unique<HashJoin>(
                    table_join,
                    right_sample_block,
                    any_take_last_row,
                    /*reserve_num=*/cnt,
                    /*instance_id=*/"",
                    /*use_two_level_maps=*/leaf_twolevel_build,
                    /*force_enable_prefetch=*/force_prefetch_build);
                if (madv_populate_build)
                    prefaultHashJoinMapBuffers(*leaf_joins[leaf]);
            }
        };

        if (serial_alloc)
        {
            for (size_t leaf = 0; leaf < total_leaves; ++leaf)
                alloc_one_leaf(leaf);
        }
        else
        {
            runParallelWorkers(
                slots,
                [&](size_t /*worker_idx*/)
                {
                    size_t leaf;
                    while ((leaf = next_leaf.fetch_add(1, std::memory_order_relaxed)) < total_leaves)
                        alloc_one_leaf(leaf);
                });
        }
        /// Reset cursor and do the insert pass into already-allocated maps.
        next_leaf.store(0, std::memory_order_relaxed);
    }

    std::atomic<size_t> blocks_moved{0};

    /// Per-worker stats (PB / NS gates): how many leaves/rows each worker actually built.
    std::vector<size_t> worker_rows(slots, 0);
    std::vector<size_t> worker_leaves(slots, 0);

    /// Work-steal whole leaves over the query/pipeline pool: a worker owns a leaf, gathers its fragments
    /// across all slots and builds its read-only leaf HashJoin (move-not-copy). Disjoint leaf ownership
    /// keeps the path lock-free (spec §4.2, §9.3).
    runParallelWorkers(
        slots,
        [&](size_t worker_idx)
        {
            Stopwatch watch;
            size_t local_rows = 0;
            size_t local_leaves = 0;
            size_t leaf;
            while ((leaf = next_leaf.fetch_add(1, std::memory_order_relaxed)) < total_leaves)
            {
                const size_t rows = buildLeaf(leaf, blocks_moved);
                if (rows != 0)
                {
                    local_rows += rows;
                    ++local_leaves;
                }
            }
            worker_rows[worker_idx] = local_rows;
            worker_leaves[worker_idx] = local_leaves;
            /// Per-worker CPU time accumulates into PartitionedHashBuildHTMicroseconds; the sum across
            /// workers is the denominator for build-HT ns/row (§9.1).
            ProfileEvents::increment(ProfileEvents::PartitionedHashBuildHTMicroseconds, watch.elapsedMicroseconds());
        });

    /// Cell conservation (operation-local, REAL): Sum of leaf HT row counts == ingested build rows.
    size_t total_cells = 0;
    for (const auto & rows : worker_rows)
        total_cells += rows;
    const size_t ingested = ingested_rows.load(std::memory_order_relaxed);
    if (total_cells != ingested)
        LOG_ERROR(
            getLogger("PartitionedHashJoin"),
            "Leaf-HT cell conservation VIOLATED: {} cells built but {} rows ingested",
            total_cells,
            ingested);

    /// Parallel-build balance log (PB / NS gates): rows per worker.
    size_t built_leaves = 0;
    size_t max_worker_rows = 0;
    for (size_t i = 0; i < slots; ++i)
    {
        built_leaves += worker_leaves[i];
        max_worker_rows = std::max(max_worker_rows, worker_rows[i]);
    }
    LOG_DEBUG(
        getLogger("PartitionedHashJoin"),
        "Eager leaf-HT build done: {} cells, {} non-empty leaves, {} workers, max-worker-share={:.1f}%, blocks_moved={}",
        total_cells,
        built_leaves,
        slots,
        total_cells == 0 ? 0.0 : 100.0 * static_cast<double>(max_worker_rows) / static_cast<double>(total_cells),
        blocks_moved.load(std::memory_order_relaxed));

    /// MEASUREMENT-ONLY: JOIN_LOG_SUBTABLE_SIZES=1 logs per-leaf cell-buffer size distribution after build.
    static const bool log_subtable_sizes = std::getenv("JOIN_LOG_SUBTABLE_SIZES") != nullptr;
    if (log_subtable_sizes)
    {
        SubtableSizeStats stats;
        for (const auto & leaf_join : leaf_joins)
            if (leaf_join)
                collectHashJoinSubtableSizes(*leaf_join, stats);
        logSubtableSizeStats("PHJ leaf", stats);
    }
}

void PartitionedHashJoin::checkTypesOfKeys(const Block & block) const
{
    hash_join->checkTypesOfKeys(block);
}

void PartitionedHashJoin::initialize(const Block & left_sample_block)
{
    hash_join->initialize(left_sample_block);
}

JoinResultPtr PartitionedHashJoin::joinBlock(Block block)
{
    /// The `[PROXY]` path until the custom probe transform lands; also serves transformHeader's
    /// empty-block call at pipeline-build time. Under debug_skip_passthrough the passthrough is empty.
    return hash_join->joinBlock(std::move(block));
}

void PartitionedHashJoin::setTotals(const Block & block)
{
    hash_join->setTotals(block);
}

const Block & PartitionedHashJoin::getTotals() const
{
    return hash_join->getTotals();
}

size_t PartitionedHashJoin::getTotalRowCount() const
{
    /// After the eager build, the real right-side data lives in the leaf HTs.
    if (!leaf_joins.empty())
    {
        size_t rows = 0;
        for (const auto & leaf_join : leaf_joins)
            if (leaf_join)
                rows += leaf_join->getTotalRowCount();
        return rows;
    }
    return hash_join->getTotalRowCount();
}

size_t PartitionedHashJoin::getTotalByteCount() const
{
    if (!leaf_joins.empty())
    {
        size_t bytes = 0;
        for (const auto & leaf_join : leaf_joins)
            if (leaf_join)
                bytes += leaf_join->getTotalByteCount();
        return bytes;
    }
    return hash_join->getTotalByteCount();
}

bool PartitionedHashJoin::alwaysReturnsEmptySet() const
{
    return ingested_rows.load(std::memory_order_relaxed) == 0;
}

std::vector<size_t> PartitionedHashJoin::getLeafRowCounts() const
{
    std::vector<size_t> counts(leaf_joins.size(), 0);
    for (size_t leaf = 0; leaf < leaf_joins.size(); ++leaf)
        if (leaf_joins[leaf])
            counts[leaf] = leaf_joins[leaf]->getTotalRowCount();
    return counts;
}

IBlocksStreamPtr
PartitionedHashJoin::getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const
{
    return hash_join->getNonJoinedBlocks(left_sample_block, result_sample_block, max_block_size);
}

}
