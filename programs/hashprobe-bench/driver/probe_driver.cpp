/// hashprobe-bench/driver/probe_driver.cpp
///
/// ProbeDriver implementation.
///
/// Tasks implemented:
///   C.1  — tight-loop drain (joinBlock once, next() until is_last, filterBySelector at end)
///   C.2  — ProfileEvents parity (JoinProbeTableRowCount / JoinResultRowCount)
///   C.3  — per-call TID logging (gettid, CLOCK_MONOTONIC_RAW timestamps)
///   C.4  — per-probe-block and per-output-block timing log
///   C.5  — DRAIN_MODE constant = "tight_loop" (see probe_driver.h)

#include "probe_driver.h"

#include "../instrumentation/perf_probes.h"

#include <Interpreters/HashJoin/HashJoinProbePhaseHooks.h>
#include <Interpreters/HashJoin/ScatteredBlock.h>
#include <Common/ProfileEvents.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <exception>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <sys/syscall.h>

namespace ProfileEvents
{
extern const Event JoinProbeTableRowCount;
extern const Event JoinResultRowCount;
}

namespace DB::HashProbeBench
{

namespace
{

/// Monotonic-raw nanosecond clock (C.3, C.4).
static uint64_t clock_ns_raw()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

/// Per-thread CPU-time nanosecond clock (C.4).
static uint64_t thread_cpu_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

/// Linux thread-ID via syscall (C.3, G2).
static uint64_t get_tid()
{
#if defined(__linux__)
    return static_cast<uint64_t>(::syscall(SYS_gettid));
#else
    return static_cast<uint64_t>(::getpid());
#endif
}

struct PhaseSnapshot
{
    uint64_t tid = 0;
    uint64_t wall_ns = 0;
    uint64_t cpu_ns = 0;
    uint64_t cycles = 0;
    uint64_t instructions = 0;
    uint64_t llc_miss = 0;
    uint64_t branch_miss = 0;
    uint64_t dtlb_miss = 0;
    uint64_t branches = 0;
    uint64_t llc_load = 0;
    uint64_t dtlb_load = 0;
    bool valid = false;
    bool hw_valid = false;
};

struct PhaseHookContext
{
    HwCounters * hw = nullptr;
    PhaseSnapshot probe_start;
    PhaseSnapshot generate_start;
    PhaseMetrics probe;
    PhaseMetrics generate;
};

static PhaseSnapshot takePhaseSnapshot(HwCounters * hw)
{
    PhaseSnapshot s;
    s.tid = get_tid();
    s.wall_ns = clock_ns_raw();
    s.cpu_ns = thread_cpu_ns();
    s.valid = true;
    if (hw && hw->isAvailable())
    {
        s.hw_valid = hw->snapshot(s.cycles, s.instructions, s.llc_miss, s.branch_miss, s.dtlb_miss, s.branches, s.llc_load, s.dtlb_load);
    }
    return s;
}

static uint64_t nonNegativeDelta(uint64_t end, uint64_t begin)
{
    return (end >= begin) ? (end - begin) : 0;
}

static void accumulatePhase(PhaseMetrics & dst, const PhaseSnapshot & begin, const PhaseSnapshot & end)
{
    if (!begin.valid || !end.valid)
        return;

    dst.wall_ns += static_cast<double>(nonNegativeDelta(end.wall_ns, begin.wall_ns));

    if (begin.tid != end.tid)
        return;

    dst.cpu_ns += static_cast<double>(nonNegativeDelta(end.cpu_ns, begin.cpu_ns));

    if (!begin.hw_valid || !end.hw_valid)
        return;

    dst.hw_available = true;
    dst.hw_cycles += nonNegativeDelta(end.cycles, begin.cycles);
    dst.hw_instructions += nonNegativeDelta(end.instructions, begin.instructions);
    dst.hw_llc_miss += nonNegativeDelta(end.llc_miss, begin.llc_miss);
    dst.hw_branch_miss += nonNegativeDelta(end.branch_miss, begin.branch_miss);
    dst.hw_dtlb_miss += nonNegativeDelta(end.dtlb_miss, begin.dtlb_miss);
    dst.hw_branches += nonNegativeDelta(end.branches, begin.branches);
    dst.hw_llc_load += nonNegativeDelta(end.llc_load, begin.llc_load);
    dst.hw_dtlb_load += nonNegativeDelta(end.dtlb_load, begin.dtlb_load);
}

class ProbePointCallbackGuard
{
public:
    ProbePointCallbackGuard(ProbePointCallback callback, void * context) { setProbePointCallback(callback, context); }

    ~ProbePointCallbackGuard() { clearProbePointCallback(); }

    ProbePointCallbackGuard(const ProbePointCallbackGuard &) = delete;
    ProbePointCallbackGuard & operator=(const ProbePointCallbackGuard &) = delete;
};

static void phaseProbeHook(ProbePoint point, void * raw_context)
{
    if (!raw_context)
        return;

    auto & ctx = *static_cast<PhaseHookContext *>(raw_context);
    const PhaseSnapshot now = takePhaseSnapshot(ctx.hw);

    switch (point)
    {
        case ProbePoint::probe_loop_start:
            ctx.probe_start = now;
            break;
        case ProbePoint::probe_loop_end:
            accumulatePhase(ctx.probe, ctx.probe_start, now);
            break;
        case ProbePoint::generate_block_start:
            ctx.generate_start = now;
            break;
        case ProbePoint::generate_block_end:
            accumulatePhase(ctx.generate, ctx.generate_start, now);
            break;
        default:
            break;
    }
}

} // anonymous namespace

// ── ProbeDriver ────────────────────────────────────────────────────────────────

ProbeDriver::ProbeDriver(std::shared_ptr<IJoin> join, OutputSink sink)
    : join_(std::move(join))
    , sink_(std::move(sink))
{
}

ProbeBlockEntry ProbeDriver::drainBlock(Block block, uint64_t probe_block_idx, HwCounters * hw_counters)
{
    ProbeBlockEntry entry;
    entry.probe_block_idx = probe_block_idx;
    entry.probe_block_rows = block.rows();

    // C.3: TID and joinblock_start_ns captured before the joinBlock call
    const uint64_t tid = get_tid();
    const uint64_t joinblock_start_ns = clock_ns_raw();

    entry.caller_tid = tid;
    entry.joinblock_start_ns = static_cast<double>(joinblock_start_ns);

    PhaseHookContext phase_context;
    phase_context.hw = hw_counters;
    ProbePointCallbackGuard phase_hook_guard(phaseProbeHook, &phase_context);

    // C.2: match production JoiningTransform::readExecute line 245
    // Increment exactly once per probe block before joinBlock (matches production).
    ProfileEvents::increment(ProfileEvents::JoinProbeTableRowCount, block.rows());

    HARNESS_PROBE_BLOCK_START();

    // C.1: call joinBlock exactly once per probe block; block is moved (production line 246)
    const uint64_t t0_join_wall = clock_ns_raw();
    const uint64_t t0_join_cpu = thread_cpu_ns();
    auto join_result = join_->joinBlock(std::move(block));
    entry.joinblock_probe_wall_ns = static_cast<double>(clock_ns_raw() - t0_join_wall);
    entry.joinblock_probe_cpu_ns = static_cast<double>(thread_cpu_ns() - t0_join_cpu);

    // C.1: tight drain loop — NO BATCHING, NO SKIPPING next(), NO EARLY RESET
    double result_emit_wall_ns = 0.0;
    double result_emit_cpu_ns = 0.0;
    uint32_t output_block_idx = 0;

    while (true)
    {
        // C.4: per-output-block timing (H2) — wall and CPU time for every next() call
        const uint64_t t0_next_wall = clock_ns_raw();
        const uint64_t t0_next_cpu = thread_cpu_ns();
        auto data = join_result->next();
        const double next_wall_ns = static_cast<double>(clock_ns_raw() - t0_next_wall);
        const double next_cpu_ns = static_cast<double>(thread_cpu_ns() - t0_next_cpu);

        result_emit_wall_ns += next_wall_ns;
        result_emit_cpu_ns += next_cpu_ns;

        // Save row count before block is moved into the sink
        const uint64_t out_rows = data.block.rows();
        const bool is_last = data.is_last;

        // C.2: match production JoiningTransform::process() line 235
        if (out_rows > 0)
            ProfileEvents::increment(ProfileEvents::JoinResultRowCount, out_rows);

        // Emit output block to downstream consumer
        sink_(std::move(data.block));

        // H2: per-output-block log entry (C.4)
        {
            OutputBlockEntry out_entry;
            out_entry.probe_block_idx = probe_block_idx;
            out_entry.output_block_idx = output_block_idx;
            out_entry.output_block_rows = out_rows;
            out_entry.next_wall_ns = next_wall_ns;
            out_entry.next_cpu_ns = next_cpu_ns;
            out_entry.is_last = is_last;
            output_block_log_.push_back(out_entry);
        }

        output_block_idx++;

        // C.1: apply pending selector to next probe block when present
        // (matches production JoiningTransform::readExecute lines 251-253)
        if (is_last && data.next_block)
            data.next_block->filterBySelector();

        // C.1: reset only at is_last (matches production lines 257-258)
        if (is_last)
        {
            join_result.reset();
            HARNESS_PROBE_BLOCK_END();
            break;
        }
    }

    // C.3: last_next_end_ns captured after the final next() call
    const uint64_t last_next_end_ns = clock_ns_raw();
    entry.last_next_end_ns = static_cast<double>(last_next_end_ns);

    entry.result_emit_wall_ns = result_emit_wall_ns;
    entry.result_emit_cpu_ns = result_emit_cpu_ns;
    entry.output_block_count = output_block_idx;
    entry.phase_probe = phase_context.probe;
    entry.phase_generate = phase_context.generate;

    // Append to per-probe-block log (H2, C.4)
    probe_block_log_.push_back(entry);

    // Append to per-call TID log (G2, C.3)
    CallerTidEntry tid_entry;
    tid_entry.probe_block_idx = probe_block_idx;
    tid_entry.tid = tid;
    tid_entry.joinblock_start_ns = joinblock_start_ns;
    tid_entry.last_next_end_ns = last_next_end_ns;
    caller_tid_log_.push_back(tid_entry);

    return entry;
}

// ── PartitionedHashJoin scatter phase ─────────────────────────────────────────

ProbeBlockEntry ProbeDriver::scatterPhjBlock(Block block, uint64_t probe_block_idx)
{
    const uint64_t joinblock_start_ns = clock_ns_raw();
    const uint64_t tid = static_cast<uint64_t>(syscall(SYS_gettid));

    ProfileEvents::increment(ProfileEvents::JoinProbeTableRowCount, block.rows());

    const auto t0_wall = static_cast<double>(clock_ns_raw());
    const auto t0_cpu = static_cast<double>(thread_cpu_ns());

    // PHJ::joinBlock scatters the probe block into per-partition slices.
    // It returns an empty block (no result rows yet; they come via getDelayedBlocks).
    auto result = join_->joinBlock(std::move(block));
    // Drain the (empty) result to satisfy the interface contract.
    while (result)
    {
        auto data = result->next();
        if (data.is_last)
            break;
    }

    const double scatter_wall_ns = static_cast<double>(clock_ns_raw()) - t0_wall;
    const double scatter_cpu_ns = static_cast<double>(thread_cpu_ns()) - t0_cpu;

    ProbeBlockEntry entry;
    entry.probe_block_idx = probe_block_idx;
    entry.probe_block_rows = 0; // actual rows not known yet (will be counted in drainDelayedBlocks)
    entry.joinblock_probe_wall_ns = scatter_wall_ns;
    entry.joinblock_probe_cpu_ns = scatter_cpu_ns;
    entry.result_emit_wall_ns = 0.0;
    entry.result_emit_cpu_ns = 0.0;
    entry.output_block_count = 0;
    entry.joinblock_start_ns = static_cast<double>(joinblock_start_ns);
    entry.last_next_end_ns = static_cast<double>(clock_ns_raw());
    entry.caller_tid = tid;

    probe_block_log_.push_back(entry);

    CallerTidEntry tid_entry;
    tid_entry.probe_block_idx = probe_block_idx;
    tid_entry.tid = tid;
    tid_entry.joinblock_start_ns = joinblock_start_ns;
    tid_entry.last_next_end_ns = static_cast<uint64_t>(entry.last_next_end_ns);
    caller_tid_log_.push_back(tid_entry);

    return entry;
}

// ── PartitionedHashJoin delayed-blocks drain ───────────────────────────────────

double ProbeDriver::drainDelayedBlocksImpl(
    IBlocksStreamPtr delayed,
    uint32_t max_threads,
    bool use_hw_counters,
    HwCounters * single_thread_hw_counters)
{
    if (!delayed)
        return 0.0;

    max_threads = std::max<uint32_t>(1, max_threads);

    struct SharedPartitionContext
    {
        struct PartitionState
        {
            PhjPartitionEntry entry;
            PhaseSnapshot build_ht_start{};
            PhaseSnapshot probe_start{};
            PhaseSnapshot gen_start{};
        };

        std::mutex mutex;
        std::map<size_t, PartitionState> partitions;

        void onProbePoint(ProbePoint pt, size_t partition, HwCounters * hw)
        {
            const PhaseSnapshot now = takePhaseSnapshot(hw);
            std::lock_guard lock(mutex);

            auto & state = partitions[partition];
            state.entry.partition_idx = static_cast<uint32_t>(partition);

            switch (pt)
            {
                case ProbePoint::phj_build_ht_start:
                    state = PartitionState{};
                    state.entry.partition_idx = static_cast<uint32_t>(partition);
                    state.build_ht_start = now;
                    break;
                case ProbePoint::phj_build_ht_end:
                    accumulatePhase(state.entry.phase_build_ht, state.build_ht_start, now);
                    break;
                case ProbePoint::phj_probe_start:
                    state.probe_start = now;
                    break;
                case ProbePoint::phj_probe_end:
                    accumulatePhase(state.entry.phase_probe, state.probe_start, now);
                    break;
                case ProbePoint::phj_gen_start:
                    state.gen_start = now;
                    break;
                case ProbePoint::phj_gen_end:
                    accumulatePhase(state.entry.phase_gen, state.gen_start, now);
                    break;
                default:
                    break;
            }
        }

        void addOutputRows(size_t partition, uint64_t rows)
        {
            std::lock_guard lock(mutex);
            auto & state = partitions[partition];
            state.entry.partition_idx = static_cast<uint32_t>(partition);
            state.entry.output_rows += rows;
        }

        std::vector<PhjPartitionEntry> finish()
        {
            std::lock_guard lock(mutex);
            std::vector<PhjPartitionEntry> result;
            result.reserve(partitions.size());
            for (const auto & [_, state] : partitions)
                result.push_back(state.entry);
            return result;
        }
    };

    struct ThreadPartitionContext
    {
        SharedPartitionContext * shared = nullptr;
        HwCounters * hw = nullptr;
    };

    auto phjCallback = [](ProbePoint pt, void * raw)
    {
        const size_t partition = getProbePointPartition();
        if (partition == INVALID_PROBE_POINT_PARTITION)
            return;

        auto & c = *static_cast<ThreadPartitionContext *>(raw);
        c.shared->onProbePoint(pt, partition, c.hw);
    };

    SharedPartitionContext shared_context;
    std::vector<double> worker_cpu_ns(max_threads, 0.0);
    std::vector<std::thread> threads;
    threads.reserve(max_threads);

    std::mutex exception_mutex;
    std::exception_ptr first_exception;
    auto captureException = [&]()
    {
        std::lock_guard lock(exception_mutex);
        if (!first_exception)
            first_exception = std::current_exception();
    };

    auto drainWorker = [&](uint32_t thread_idx, HwCounters * external_hw)
    {
        try
        {
            HwCounters local_hw;
            HwCounters * hw = external_hw;
            if (!hw && use_hw_counters)
            {
                if (local_hw.open())
                {
                    local_hw.start();
                    hw = &local_hw;
                }
            }

            ThreadPartitionContext ctx{&shared_context, hw};
            ProbePointCallbackGuard phase_hook_guard(+phjCallback, &ctx);

            const double t0_cpu = static_cast<double>(thread_cpu_ns());
            while (!delayed->isFinished())
            {
                clearProbePointPartition();
                Block out = delayed->next();
                if (out.empty())
                    continue;

                const size_t partition = getProbePointPartition();
                if (partition == INVALID_PROBE_POINT_PARTITION)
                    throw std::runtime_error("[hashprobe-bench] PHJ delayed block had no partition attribution");

                shared_context.addOutputRows(partition, out.rows());
                sink_(std::move(out));
            }
            worker_cpu_ns[thread_idx] = static_cast<double>(thread_cpu_ns()) - t0_cpu;
            clearProbePointPartition();
        }
        catch (...)
        {
            captureException();
        }
    };

    if (max_threads == 1)
    {
        drainWorker(0, single_thread_hw_counters);
    }
    else
    {
        for (uint32_t t = 0; t < max_threads; ++t)
            threads.emplace_back(drainWorker, t, nullptr);
        for (auto & thread : threads)
            thread.join();
    }

    if (first_exception)
        std::rethrow_exception(first_exception);

    phj_partition_log_ = shared_context.finish();
    return std::accumulate(worker_cpu_ns.begin(), worker_cpu_ns.end(), 0.0);
}

void ProbeDriver::drainDelayedBlocks(HwCounters * hw_counters)
{
    drainDelayedBlocksImpl(join_->getDelayedBlocks(), 1, false, hw_counters);
}

double ProbeDriver::drainDelayedBlocksParallel(uint32_t max_threads, bool use_hw_counters)
{
    return drainDelayedBlocksImpl(join_->getDelayedBlocks(), max_threads, use_hw_counters, nullptr);
}

} // namespace DB::HashProbeBench
