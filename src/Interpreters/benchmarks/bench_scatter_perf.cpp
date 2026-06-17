/// bench_scatter_perf — phase-attributed PMU instrumentation of the RHJ build-side data movement, to
/// explain the T_scatter (BuildSide::scatterToLeaves) vs T_memcpy (flat key+ref copy) gap measured by
/// bench_build_bandwidth. It drives the GENUINE DB::RadixJoin::BuildSide scatter unchanged and a memcpy
/// baseline carved exactly like allocExactPartitions, over the SAME pre-generated blocks, and wraps each
/// phase's ParallelFor workers with per-worker perf_event_open groups + getrusage(RUSAGE_THREAD) deltas
/// so the hardware counters and page-fault/kernel-time attribute to scatter vs memcpy only.
///
/// The PMU group machinery (EvDef / PerfGroup / perfGroup) is ported verbatim from bench_rhj_vs_chj.cpp
/// (SPR-verified raw configs), extended with store-side TLB walks, retired load/store counts, and L2 RFO
/// to characterise the scattered-write pattern that the load-side probe groups do not cover.
///
/// Research/diagnostic harness only; the production join path is untouched.

#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <DataTypes/DataTypesNumber.h>

#include <Interpreters/RadixHashJoin/Arena.h>
#include <Interpreters/RadixHashJoin/BuildSide.h>
#include <Interpreters/RadixHashJoin/KeyRefScatter.h>
#include <Interpreters/RadixHashJoin/ParallelFor.h>
#include <Interpreters/RadixHashJoin/PartitionPlan.h>

#include <Common/HashTable/Hash.h>
#include <Common/ThreadStatus.h>

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sched.h>
#include <unistd.h>

namespace
{

using namespace DB;
using namespace DB::RadixJoin;

constexpr size_t KEY_WIDTH = 8;

void pinToCore(int core)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

UInt64 buildKey(UInt64 i)
{
    return intHash64(i);
}

double nowMs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double minOf(const std::vector<double> & v)
{
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

double medianOf(std::vector<double> v)
{
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double gibPerSec(UInt64 bytes, double ms)
{
    return ms > 0.0 ? (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0) : 0.0;
}

/// ── PMU groups (raw configs SPR-verified via `perf stat -vv`) ────────────────────────────────────────
struct EvDef
{
    const char * name;
    uint32_t type;
    uint64_t config;
};

inline std::vector<EvDef> perfGroup(const std::string & g)
{
    constexpr uint32_t raw = PERF_TYPE_RAW;
    constexpr uint32_t hw = PERF_TYPE_HARDWARE;
    const EvDef cyc{"cycles", hw, PERF_COUNT_HW_CPU_CYCLES};
    const EvDef ins{"instructions", hw, PERF_COUNT_HW_INSTRUCTIONS};
    /// Top-down L1+L2 (PERF_METRICS; slots is the leader).
    if (g == "td")
        return {{"slots", raw, 0x400}, {"retiring", raw, 0x8000}, {"bad_spec", raw, 0x8100},
                {"fe_bound", raw, 0x8200}, {"be_bound", raw, 0x8300}, {"heavy_ops", raw, 0x8400},
                {"br_mispred", raw, 0x8500}, {"fetch_lat", raw, 0x8600}, {"mem_bound", raw, 0x8700}};
    /// LFB-level MLP + fill-buffer-full stalls.
    if (g == "lfb")
        return {{"pending", raw, 0x148}, {"pending_cycles", raw, 0x1000148}, {"fb_full", raw, 0x248}, cyc, ins};
    /// Offcore read occupancy (demand+prefetch) -> DRAM-level MLP.
    if (g == "off2")
        return {{"data_rd", raw, 0x820}, {"cycles_with_data_rd", raw, 0x1000820}, cyc, ins};
    /// Execution-stall attribution by outstanding-miss level.
    if (g == "stall")
        return {{"stalls_total", raw, 0x40004a3}, {"stalls_l2_miss", raw, 0x50005a3},
                {"stalls_l3_miss", raw, 0x60006a3}, cyc, ins};
    /// Retired-load cache-hierarchy distribution.
    if (g == "loc")
        return {{"l1_hit", raw, 0x1d1}, {"l2_hit", raw, 0x2d1}, {"l3_hit", raw, 0x4d1},
                {"l3_miss", raw, 0x20d1}, cyc, ins};
    /// dTLB LOAD page walks.
    if (g == "ltlb")
        return {{"walk_pending", raw, 0x1012}, {"walk_active", raw, 0x1001012}, {"walk_completed", raw, 0xe12}, cyc, ins};
    /// dTLB STORE page walks (the scattered-write TLB pressure; not covered by the probe-side groups).
    if (g == "stlb")
        return {{"walk_pending", raw, 0x1013}, {"walk_active", raw, 0x1001013}, {"walk_completed", raw, 0xe13}, cyc, ins};
    /// Retired loads / stores (per-byte instruction-traffic asymmetry).
    if (g == "mem")
        return {{"all_loads", raw, 0x81d0}, {"all_stores", raw, 0x82d0}, cyc, ins};
    /// L2 demand traffic incl. RFO: distinguishes memcpy's temporal-store RFO from scatter's NT stores.
    if (g == "rfo")
        return {{"l2_refs", raw, 0xff24}, {"l2_miss", raw, 0x3f24}, {"all_rfo", raw, 0xe224},
                {"rfo_miss", raw, 0x2224}, cyc};
    return {};
}

struct PerfGroup
{
    std::vector<int> fds;
    int leader = -1;
    bool ok = false;

    static int64_t sysOpen(perf_event_attr * a, pid_t pid, int cpu, int grp, uint64_t flags)
    {
        return syscall(__NR_perf_event_open, a, pid, cpu, grp, flags);
    }

    void open(const std::vector<EvDef> & evs)
    {
        for (size_t i = 0; i < evs.size(); ++i)
        {
            perf_event_attr a{};
            a.size = sizeof(a);
            a.type = evs[i].type;
            a.config = evs[i].config;
            a.disabled = (i == 0) ? 1 : 0;
            a.exclude_kernel = 1;
            a.exclude_hv = 1;
            a.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
            int fd = static_cast<int>(sysOpen(&a, 0, -1, (i == 0) ? -1 : leader, 0));
            if (fd < 0)
            {
                for (int f : fds)
                    ::close(f);
                fds.clear();
                leader = -1;
                ok = false;
                return;
            }
            if (i == 0)
                leader = fd;
            fds.push_back(fd);
        }
        ok = !fds.empty();
    }

    void reset() const { ioctl(leader, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP); }
    void enable() const { ioctl(leader, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP); }
    void disable() const { ioctl(leader, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP); }

    /// Returns {scaled values..., multiplex_ok(0/1)}. scale = enabled/running corrects multiplexing.
    std::vector<uint64_t> read(double & worst_ratio) const
    {
        const size_t n = fds.size();
        std::vector<uint64_t> buf(3 + n, 0);
        std::vector<uint64_t> out(n, 0);
        const ssize_t r = ::read(leader, buf.data(), buf.size() * sizeof(uint64_t));
        if (r <= 0)
            return out;
        const double te = static_cast<double>(buf[1]);
        const double tr = static_cast<double>(buf[2]);
        const double scale = tr > 0 ? te / tr : 1.0;
        const double ratio = te > 0 ? tr / te : 0.0;
        worst_ratio = std::min(worst_ratio, ratio);
        for (size_t i = 0; i < n; ++i)
            out[i] = static_cast<uint64_t>(static_cast<double>(buf[3 + i]) * scale);
        return out;
    }

    void closeAll()
    {
        for (int fd : fds)
            if (fd >= 0)
                ::close(fd);
        fds.clear();
        leader = -1;
    }
};

/// ── Global capture state consulted by the instrumented ParallelFor ───────────────────────────────────
struct Capture
{
    std::atomic<bool> on{false};
    const std::vector<EvDef> * group = nullptr;     /// set before a captured phase
    std::vector<std::atomic<uint64_t>> sums;        /// per-event, summed across all workers in the window
    std::atomic<uint64_t> minflt{0};
    std::atomic<uint64_t> majflt{0};
    std::atomic<uint64_t> utime_us{0};
    std::atomic<uint64_t> stime_us{0};
    std::atomic<uint64_t> worker_invocations{0};    /// number of worker steal-loops that measured
    std::atomic<uint64_t> mux_bad{0};               /// workers whose group multiplexed (<99% running)

    void begin(const std::vector<EvDef> & g)
    {
        group = &g;
        sums = std::vector<std::atomic<uint64_t>>(g.size());
        for (auto & s : sums)
            s.store(0);
        minflt = 0;
        majflt = 0;
        utime_us = 0;
        stime_us = 0;
        worker_invocations = 0;
        mux_bad = 0;
        on.store(true, std::memory_order_release);
    }
    void end() { on.store(false, std::memory_order_release); }
};

Capture g_cap;

/// Instrumented ParallelFor: identical work-stealing contract to bench_build_bandwidth's makeParallelFor,
/// but when g_cap.on each worker opens g_cap.group on its pinned core, counts ONLY its steal loop, and
/// folds scaled counts + RUSAGE_THREAD fault/time deltas into the global accumulators.
ParallelFor makeInstrumentedParallelFor(size_t num_workers, int first_core)
{
    return [num_workers, first_core](size_t total, const UnitFn & fn)
    {
        if (total == 0)
            return;

        /// Work-stealing loop with embedded measurement (kept inline so the captured region is exactly
        /// this worker's steal loop, mirroring bench_rhj_vs_chj::probePass).
        const size_t workers = std::min(num_workers, total);
        std::atomic<size_t> next{0};
        std::mutex exc_mutex;
        std::exception_ptr first_exc;
        std::vector<std::thread> ts;
        ts.reserve(workers);
        for (size_t w = 0; w < workers; ++w)
            ts.emplace_back([&, w]
            {
                pinToCore(first_core + static_cast<int>(w));

                const bool cap = g_cap.on.load(std::memory_order_acquire);
                PerfGroup pg;
                bool measuring = false;
                rusage r0{};
                if (cap)
                {
                    if (g_cap.group && !g_cap.group->empty())
                    {
                        pg.open(*g_cap.group);
                        if (pg.ok)
                        {
                            pg.reset();
                            pg.enable();
                            measuring = true;
                        }
                    }
                    getrusage(RUSAGE_THREAD, &r0);
                }

                while (true)
                {
                    const size_t unit = next.fetch_add(1);
                    if (unit >= total)
                        break;
                    try
                    {
                        fn(unit, w);
                    }
                    catch (...)
                    {
                        std::lock_guard lock(exc_mutex);
                        if (!first_exc)
                            first_exc = std::current_exception();
                        next.store(total);
                        break;
                    }
                }

                if (cap)
                {
                    rusage r1{};
                    getrusage(RUSAGE_THREAD, &r1);
                    auto us = [](const timeval & a, const timeval & b)
                    { return static_cast<uint64_t>((b.tv_sec - a.tv_sec) * 1000000LL + (b.tv_usec - a.tv_usec)); };
                    g_cap.minflt.fetch_add(static_cast<uint64_t>(r1.ru_minflt - r0.ru_minflt), std::memory_order_relaxed);
                    g_cap.majflt.fetch_add(static_cast<uint64_t>(r1.ru_majflt - r0.ru_majflt), std::memory_order_relaxed);
                    g_cap.utime_us.fetch_add(us(r0.ru_utime, r1.ru_utime), std::memory_order_relaxed);
                    g_cap.stime_us.fetch_add(us(r0.ru_stime, r1.ru_stime), std::memory_order_relaxed);
                    if (measuring)
                    {
                        pg.disable();
                        double worst = 1.0;
                        auto vals = pg.read(worst);
                        for (size_t i = 0; i < vals.size() && i < g_cap.sums.size(); ++i)
                            g_cap.sums[i].fetch_add(vals[i], std::memory_order_relaxed);
                        if (worst < 0.99)
                            g_cap.mux_bad.fetch_add(1, std::memory_order_relaxed);
                        pg.closeAll();
                    }
                    g_cap.worker_invocations.fetch_add(1, std::memory_order_relaxed);
                }
            });
        for (auto & t : ts)
            t.join();
        if (first_exc)
            std::rethrow_exception(first_exc);
    };
}

/// A snapshot of all accumulators after a captured phase.
struct PhaseCounts
{
    std::map<std::string, double> ev;   /// "group.name" -> summed value
    double minflt = 0;
    double majflt = 0;
    double utime_us = 0;
    double stime_us = 0;
    double mux_bad = 0;
};

struct Args
{
    UInt64 build_rows = 1'000'000'000ULL;
    size_t threads = 16;
    UInt64 block_rows = 60000;
    size_t leaves = 8192;
    int passes = 1;
    int repeats = 4;       /// measured passes per (phase, group); +1 warmup
    int first_core = 0;
};

UInt64 parseU64(const char * s) { return std::strtoull(s, nullptr, 10); }

PartitionPlan makeFixedPlan(size_t leaves, int passes)
{
    PartitionPlan plan;
    leaves = ceilPowerOfTwo(std::max<size_t>(leaves, 1));
    plan.num_leaves = leaves;
    plan.total_bits = static_cast<UInt32>(std::countr_zero(leaves));
    plan.leaf_shift = PartitionPlan::ROUTE_BITS - plan.total_bits;
    UInt32 num_passes = 1;
    if (plan.total_bits > 0)
        num_passes = std::clamp<UInt32>(static_cast<UInt32>(std::max(passes, 1)), 1, plan.total_bits);
    const UInt32 base = plan.total_bits / num_passes;
    const UInt32 rem = plan.total_bits % num_passes;
    plan.pass_bits.assign(num_passes, base);
    for (UInt32 i = 0; i < rem; ++i)
        plan.pass_bits[i] += 1;
    return plan;
}

Block makeBlock(UInt64 begin, UInt64 count)
{
    Block block;
    auto col = ColumnUInt64::create();
    auto & data = col->getData();
    data.resize(count);
    for (UInt64 r = 0; r < count; ++r)
        data[r] = buildKey(begin + r);
    block.insert(ColumnWithTypeAndName(std::move(col), std::make_shared<DataTypeUInt64>(), "k0"));
    return block;
}

}

int main(int argc, char ** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--build") args.build_rows = parseU64(next());
        else if (a == "--threads") args.threads = static_cast<size_t>(parseU64(next()));
        else if (a == "--block-rows") args.block_rows = parseU64(next());
        else if (a == "--leaves") args.leaves = static_cast<size_t>(parseU64(next()));
        else if (a == "--passes") args.passes = static_cast<int>(parseU64(next()));
        else if (a == "--repeats") args.repeats = static_cast<int>(parseU64(next()));
        else if (a == "--first-core") args.first_core = static_cast<int>(parseU64(next()));
        else { fmt::print(stderr, "unknown arg: {}\n", a); return 2; }
    }

    DB::MainThreadStatus::getInstance();

    const size_t num_workers = std::max<size_t>(args.threads, 1);
    const ParallelFor parallel_for = makeInstrumentedParallelFor(num_workers, args.first_core);
    PartitionPlan plan = makeFixedPlan(args.leaves, args.passes);

    const UInt64 num_blocks = (args.build_rows + args.block_rows - 1) / args.block_rows;

    fmt::print(stderr, "build={} threads={} block_rows={} leaves={} passes={} repeats={}\n",
               args.build_rows, args.threads, args.block_rows, plan.num_leaves, plan.pass_bits.size(), args.repeats);

    /// ── Build once (capture off). ────────────────────────────────────────────────────────────────────
    std::vector<Block> blocks(num_blocks);
    parallel_for(num_blocks, [&](size_t b, size_t) {
        const UInt64 begin = b * args.block_rows;
        const UInt64 count = std::min<UInt64>(args.block_rows, args.build_rows - begin);
        blocks[b] = makeBlock(begin, count);
    });

    BuildSide build_side(plan, {0}, {KEY_WIDTH}, num_workers);
    parallel_for(num_blocks, [&](size_t b, size_t worker) { build_side.add(blocks[b], worker); });
    build_side.finishBuild();

    const UInt64 total_rows = build_side.totalRows();
    const UInt64 key_ref_volume = total_rows * (KEY_WIDTH + sizeof(BuildRef));
    fmt::print(stderr, "total_rows={} key_ref_volume={} ({:.2f} GiB)\n",
               total_rows, key_ref_volume, static_cast<double>(key_ref_volume) / (1024.0 * 1024.0 * 1024.0));

    /// ── Phase runners. ─────────────────────────────────────────────────────────────────────────────
    std::vector<BuildRef> ref_scratch(args.block_rows, BuildRef(0, 0));

    auto scatter_once = [&]() -> UInt64 {
        LeafArrays leaves = build_side.scatterToLeaves(parallel_for, num_workers, /*estimate_distinct_keys=*/false);
        return leaves.bytes_scattered;
    };

    auto memcpy_once = [&]() -> UInt64 {
        const auto & bl = build_side.blocks();
        const size_t nb = bl.size();
        const size_t record_width = KEY_WIDTH + sizeof(BuildRef);
        RadixJoin::Arena arena;
        std::vector<char *> bases(nb, nullptr);
        parallel_for(nb, [&](size_t b, size_t) {
            const size_t n = bl[b].rows();
            if (n == 0) return;
            const size_t record_bytes = roundUpToLine(n * record_width);
            bases[b] = static_cast<char *>(arena.allocate(record_bytes, LINE_BYTES));
        });
        parallel_for(nb, [&](size_t b, size_t) {
            const size_t n = bl[b].rows();
            if (n == 0) return;
            char * base = bases[b];
            const char * key_src = bl[b].getByPosition(0).column->getRawData().data();
            for (size_t row = 0; row < n; ++row)
            {
                char * rec = base + row * record_width;
                std::memcpy(rec, &ref_scratch[row], sizeof(BuildRef));
                std::memcpy(rec + sizeof(BuildRef), key_src + row * KEY_WIDTH, KEY_WIDTH);
            }
        });
        return key_ref_volume;
    };

    /// ── Timing-only sanity (capture off): confirm we reproduce bench_build_bandwidth. ────────────────
    auto time_phase = [&](const std::function<UInt64()> & once) -> std::pair<double,double> {
        once(); /// warmup
        std::vector<double> ms;
        for (int r = 0; r < args.repeats; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            once();
            ms.push_back(nowMs(t0, std::chrono::steady_clock::now()));
        }
        return {minOf(ms), medianOf(ms)};
    };
    const auto [scat_min, scat_med] = time_phase(scatter_once);
    const auto [memc_min, memc_med] = time_phase(memcpy_once);
    fmt::print(stderr, "[timing] T_scatter min/med = {:.1f}/{:.1f} ms ({:.1f} GiB/s) ; T_memcpy min/med = {:.1f}/{:.1f} ms ({:.1f} GiB/s) ; ratio {:.2f}x\n",
               scat_min, scat_med, gibPerSec(key_ref_volume, scat_min), memc_min, memc_med, gibPerSec(key_ref_volume, memc_min),
               memc_min > 0 ? scat_min / memc_min : 0.0);

    /// ── PMU capture: each group, each phase, median of `repeats` measured passes (1 warmup). ─────────
    const std::vector<std::string> groups = {"td", "lfb", "off2", "stall", "loc", "ltlb", "stlb", "mem", "rfo"};

    auto run_group_phase = [&](const std::string & gname, const std::function<UInt64()> & once) -> PhaseCounts {
        const auto g = perfGroup(gname);
        std::vector<std::vector<double>> per_ev(g.size());
        std::vector<double> minflt;
        std::vector<double> majflt;
        std::vector<double> utime;
        std::vector<double> stime;
        std::vector<double> muxbad;
        once(); /// warmup (also warms destination pages / allocator)
        for (int r = 0; r < args.repeats; ++r)
        {
            g_cap.begin(g);
            once();
            g_cap.end();
            for (size_t i = 0; i < g.size(); ++i)
                per_ev[i].push_back(static_cast<double>(g_cap.sums[i].load()));
            minflt.push_back(static_cast<double>(g_cap.minflt.load()));
            majflt.push_back(static_cast<double>(g_cap.majflt.load()));
            utime.push_back(static_cast<double>(g_cap.utime_us.load()));
            stime.push_back(static_cast<double>(g_cap.stime_us.load()));
            muxbad.push_back(static_cast<double>(g_cap.mux_bad.load()));
        }
        PhaseCounts pc;
        for (size_t i = 0; i < g.size(); ++i)
            pc.ev[fmt::format("{}.{}", gname, g[i].name)] = medianOf(per_ev[i]);
        pc.minflt = medianOf(minflt);
        pc.majflt = medianOf(majflt);
        pc.utime_us = medianOf(utime);
        pc.stime_us = medianOf(stime);
        pc.mux_bad = medianOf(muxbad);
        return pc;
    };

    PhaseCounts scat;
    PhaseCounts memc;
    for (const auto & gname : groups)
    {
        fmt::print(stderr, "  measuring group {} ...\n", gname);
        const PhaseCounts s = run_group_phase(gname, scatter_once);
        const PhaseCounts m = run_group_phase(gname, memcpy_once);
        scat.ev.insert(s.ev.begin(), s.ev.end());
        memc.ev.insert(m.ev.begin(), m.ev.end());
        /// faults/time are group-independent; keep the values from the 'mem' pass (arbitrary, all similar),
        /// but record per-group for transparency under the group name prefix.
        scat.ev[fmt::format("{}.minflt", gname)] = s.minflt;
        scat.ev[fmt::format("{}.majflt", gname)] = s.majflt;
        scat.ev[fmt::format("{}.utime_us", gname)] = s.utime_us;
        scat.ev[fmt::format("{}.stime_us", gname)] = s.stime_us;
        scat.ev[fmt::format("{}.mux_bad", gname)] = s.mux_bad;
        memc.ev[fmt::format("{}.minflt", gname)] = m.minflt;
        memc.ev[fmt::format("{}.majflt", gname)] = m.majflt;
        memc.ev[fmt::format("{}.utime_us", gname)] = m.utime_us;
        memc.ev[fmt::format("{}.stime_us", gname)] = m.stime_us;
        memc.ev[fmt::format("{}.mux_bad", gname)] = m.mux_bad;
    }

    /// ── Report. ──────────────────────────────────────────────────────────────────────────────────────
    const double bytes = static_cast<double>(key_ref_volume);
    auto event_value = [](const PhaseCounts & p, const char * k) -> double
    {
        const auto it = p.ev.find(k);
        return it != p.ev.end() ? it->second : 0.0;
    };
    auto event_ratio = [&](const PhaseCounts & p, const char * a, const char * b) -> double
    {
        const double denom = event_value(p, b);
        return denom > 0 ? event_value(p, a) / denom : 0.0;
    };

    auto line = [&](const std::string & label, double s, double m, const char * fmtspec)
    {
        const double ratio = m != 0 ? s / m : 0.0;
        fmt::print("  {:<46} scatter={:>12} memcpy={:>12}  s/m={:.2f}x\n",
                   label, fmt::format(fmt::runtime(fmtspec), s), fmt::format(fmt::runtime(fmtspec), m), ratio);
    };

    fmt::print("\n==================== bench_scatter_perf REPORT ====================\n");
    fmt::print("CPU=Sapphire Rapids 8488C  build={} total_rows={} key+ref={:.2f}GiB threads={} leaves={} passes={}\n",
               args.build_rows, total_rows, bytes/(1024*1024*1024), args.threads, plan.num_leaves, plan.pass_bits.size());
    fmt::print("T_scatter min/med = {:.1f}/{:.1f} ms ({:.1f} GiB/s)   T_memcpy min/med = {:.1f}/{:.1f} ms ({:.1f} GiB/s)   ratio {:.2f}x\n",
               scat_min, scat_med, gibPerSec(key_ref_volume, scat_min), memc_min, memc_med, gibPerSec(key_ref_volume, memc_min),
               memc_min>0 ? scat_min/memc_min : 0.0);
    fmt::print("(counts are sums over all workers, median of {} measured passes; per-byte uses {:.3e} key+ref bytes)\n",
               args.repeats, bytes);

    fmt::print("\n-- EXECUTION --\n");
    line("cycles (mem grp)",            event_value(scat,"mem.cycles"),       event_value(memc,"mem.cycles"),       "{:.3e}");
    line("instructions (mem grp)",      event_value(scat,"mem.instructions"), event_value(memc,"mem.instructions"), "{:.3e}");
    line("IPC",                         event_ratio(scat,"mem.instructions","mem.cycles"), event_ratio(memc,"mem.instructions","mem.cycles"), "{:.2f}");
    line("instructions / byte",         event_value(scat,"mem.instructions")/bytes, event_value(memc,"mem.instructions")/bytes, "{:.3f}");
    line("retired loads / byte",        event_value(scat,"mem.all_loads")/bytes,  event_value(memc,"mem.all_loads")/bytes,  "{:.3f}");
    line("retired stores / byte",       event_value(scat,"mem.all_stores")/bytes, event_value(memc,"mem.all_stores")/bytes, "{:.3f}");

    fmt::print("\n-- TOP-DOWN (TMA, %% of slots) --\n");
    line("Retiring",                    100*event_ratio(scat,"td.retiring","td.slots"), 100*event_ratio(memc,"td.retiring","td.slots"), "{:.1f}");
    line("Bad-Speculation",             100*event_ratio(scat,"td.bad_spec","td.slots"), 100*event_ratio(memc,"td.bad_spec","td.slots"), "{:.1f}");
    line("Frontend-Bound",              100*event_ratio(scat,"td.fe_bound","td.slots"), 100*event_ratio(memc,"td.fe_bound","td.slots"), "{:.1f}");
    line("Backend-Bound",               100*event_ratio(scat,"td.be_bound","td.slots"), 100*event_ratio(memc,"td.be_bound","td.slots"), "{:.1f}");
    line("  Backend: Memory-Bound",     100*event_ratio(scat,"td.mem_bound","td.slots"), 100*event_ratio(memc,"td.mem_bound","td.slots"), "{:.1f}");
    line("  Backend: Core-Bound(deriv)",100*(event_ratio(scat,"td.be_bound","td.slots")-event_ratio(scat,"td.mem_bound","td.slots")),
                                         100*(event_ratio(memc,"td.be_bound","td.slots")-event_ratio(memc,"td.mem_bound","td.slots")), "{:.1f}");

    fmt::print("\n-- MEMORY-LEVEL PARALLELISM / STALLS --\n");
    line("MLP offcore data_rd/cyc_with",event_ratio(scat,"off2.data_rd","off2.cycles_with_data_rd"), event_ratio(memc,"off2.data_rd","off2.cycles_with_data_rd"), "{:.2f}");
    line("MLP LFB pending/pend_cycles", event_ratio(scat,"lfb.pending","lfb.pending_cycles"), event_ratio(memc,"lfb.pending","lfb.pending_cycles"), "{:.2f}");
    line("fb_full / cycles %",          100*event_ratio(scat,"lfb.fb_full","lfb.cycles"), 100*event_ratio(memc,"lfb.fb_full","lfb.cycles"), "{:.1f}");
    line("pending_cycles / cycles %",   100*event_ratio(scat,"lfb.pending_cycles","lfb.cycles"), 100*event_ratio(memc,"lfb.pending_cycles","lfb.cycles"), "{:.1f}");
    line("stalls_total / cycles %",     100*event_ratio(scat,"stall.stalls_total","stall.cycles"), 100*event_ratio(memc,"stall.stalls_total","stall.cycles"), "{:.1f}");
    line("stalls_l2_miss / cycles %",   100*event_ratio(scat,"stall.stalls_l2_miss","stall.cycles"), 100*event_ratio(memc,"stall.stalls_l2_miss","stall.cycles"), "{:.1f}");
    line("stalls_l3_miss / cycles %",   100*event_ratio(scat,"stall.stalls_l3_miss","stall.cycles"), 100*event_ratio(memc,"stall.stalls_l3_miss","stall.cycles"), "{:.1f}");

    fmt::print("\n-- LOAD LOCALITY (retired loads; prefetch-confounded) --\n");
    {
        auto loctot = [&](const PhaseCounts & counts)
        {
            return event_value(counts, "loc.l1_hit") + event_value(counts, "loc.l2_hit")
                + event_value(counts, "loc.l3_hit") + event_value(counts, "loc.l3_miss");
        };
        const double scatter_loc_total = loctot(scat);
        const double memcpy_loc_total = loctot(memc);
        auto locality_pct = [&](const PhaseCounts & counts, const char * key, double total)
        {
            return total > 0 ? 100 * event_value(counts, key) / total : 0.0;
        };
        line("L1 hit %", locality_pct(scat, "loc.l1_hit", scatter_loc_total), locality_pct(memc, "loc.l1_hit", memcpy_loc_total), "{:.1f}");
        line("L2 hit %", locality_pct(scat, "loc.l2_hit", scatter_loc_total), locality_pct(memc, "loc.l2_hit", memcpy_loc_total), "{:.1f}");
        line("L3 hit %", locality_pct(scat, "loc.l3_hit", scatter_loc_total), locality_pct(memc, "loc.l3_hit", memcpy_loc_total), "{:.1f}");
        line("DRAM (L3 miss) %", locality_pct(scat, "loc.l3_miss", scatter_loc_total), locality_pct(memc, "loc.l3_miss", memcpy_loc_total), "{:.1f}");
    }

    fmt::print("\n-- L2 TRAFFIC / RFO (memcpy temporal-store RFO vs scatter NT) --\n");
    line("L2 references / byte",        event_value(scat,"rfo.l2_refs")/bytes, event_value(memc,"rfo.l2_refs")/bytes, "{:.4f}");
    line("L2 miss / byte",              event_value(scat,"rfo.l2_miss")/bytes, event_value(memc,"rfo.l2_miss")/bytes, "{:.4f}");
    line("L2 RFO (all_rfo) / byte",     event_value(scat,"rfo.all_rfo")/bytes, event_value(memc,"rfo.all_rfo")/bytes, "{:.4f}");
    line("L2 RFO miss / byte",          event_value(scat,"rfo.rfo_miss")/bytes, event_value(memc,"rfo.rfo_miss")/bytes, "{:.4f}");

    fmt::print("\n-- TLB PAGE WALKS (per byte = COUNT; active%% = COST) --\n");
    line("dTLB LOAD walks / byte",      event_value(scat,"ltlb.walk_completed")/bytes, event_value(memc,"ltlb.walk_completed")/bytes, "{:.4e}");
    line("dTLB LOAD walk_active %",     100*event_ratio(scat,"ltlb.walk_active","ltlb.cycles"), 100*event_ratio(memc,"ltlb.walk_active","ltlb.cycles"), "{:.1f}");
    line("dTLB LOAD pmh occupancy",     event_ratio(scat,"ltlb.walk_pending","ltlb.cycles"), event_ratio(memc,"ltlb.walk_pending","ltlb.cycles"), "{:.3f}");
    line("dTLB STORE walks / byte",     event_value(scat,"stlb.walk_completed")/bytes, event_value(memc,"stlb.walk_completed")/bytes, "{:.4e}");
    line("dTLB STORE walk_active %",    100*event_ratio(scat,"stlb.walk_active","stlb.cycles"), 100*event_ratio(memc,"stlb.walk_active","stlb.cycles"), "{:.1f}");
    line("dTLB STORE pmh occupancy",    event_ratio(scat,"stlb.walk_pending","stlb.cycles"), event_ratio(memc,"stlb.walk_pending","stlb.cycles"), "{:.3f}");

    fmt::print("\n-- KERNEL / PAGING (RUSAGE_THREAD deltas, per measured pass, summed over workers) --\n");
    line("minor page faults / pass",    event_value(scat,"mem.minflt"), event_value(memc,"mem.minflt"), "{:.0f}");
    line("major page faults / pass",    event_value(scat,"mem.majflt"), event_value(memc,"mem.majflt"), "{:.0f}");
    line("user us / pass (sum workers)", event_value(scat,"mem.utime_us"), event_value(memc,"mem.utime_us"), "{:.0f}");
    line("sys  us / pass (sum workers)", event_value(scat,"mem.stime_us"), event_value(memc,"mem.stime_us"), "{:.0f}");
    {
        auto sys_time_fraction = [&](const PhaseCounts & counts)
        {
            const double user_us = event_value(counts, "mem.utime_us");
            const double sys_us = event_value(counts, "mem.stime_us");
            return (user_us + sys_us) > 0 ? 100 * sys_us / (user_us + sys_us) : 0.0;
        };
        line("sys time fraction %", sys_time_fraction(scat), sys_time_fraction(memc), "{:.1f}");
    }

    /// Multiplexing sanity: any group where workers saw <99% running.
    fmt::print("\n-- MULTIPLEX SANITY (workers with <99%% group running; want 0) --\n");
    for (const auto & gname : groups)
        fmt::print("  {:<6} scatter_mux_bad={:.0f}  memcpy_mux_bad={:.0f}\n",
                   gname, event_value(scat, fmt::format("{}.mux_bad", gname).c_str()), event_value(memc, fmt::format("{}.mux_bad", gname).c_str()));
    fmt::print("===================================================================\n");
    (void)std::fflush(stdout);
    return 0;
}
