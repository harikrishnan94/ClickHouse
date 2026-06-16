/// bench_rhj_vs_chj — stand-alone driver that links the real ClickHouse `dbms` and exercises the genuine
/// RadixHashJoin (`radix_hash`) and ConcurrentHashJoin (`parallel_hash`) build/probe code paths directly
/// through the `IJoin` API, for the reference query
///
///     SELECT count() FROM <probe> INNER JOIN <build> USING (k0)
///
/// with a single non-nullable UInt64 key, distinct build keys, and a clean all-match INNER join.
///
/// This is a research/reproduction harness only: it constructs and drives the production join classes
/// unchanged (no proxy, no copy of the data structures). The classes invoked, and the entry points used,
/// are (all paths relative to the repo root, verified against the current source):
///
///   RHJ  = DB::RadixHashJoin           src/Interpreters/RadixHashJoin/RadixHashJoin.h
///   CHJ  = DB::ConcurrentHashJoin      src/Interpreters/ConcurrentHashJoin.h  (wraps DB::HashJoin slots)
///   API  = DB::IJoin                   src/Interpreters/IJoin.h
///          addBlockToJoin(block, num_rows, check_limits, build_lane)   IJoin.h:115 / RadixHashJoin.h:62
///          onBuildPhaseFinish()                                        IJoin.h:190 / RadixHashJoin.h:80
///          joinBlock(block, lane) -> IJoinResult::next()               IJoin.h:130-134, 55-64
///
/// Engine construction mirrors the planner (src/Planner/PlannerJoins.cpp:1235 for CHJ, :1270 for RHJ).
///
/// The driver replaces the server's FillingRightJoinSideTransform (parallel build) and JoiningTransform
/// (parallel probe) with its own thread pool of pinned workers, so the join compute is measured in
/// isolation without running a full ClickHouse server. See tmp/bench/realbench/README.md for the full
/// citation table and documented divergences.

#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <Core/Joins.h>
#include <Core/Settings.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/ConcurrentHashJoin.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/RadixHashJoin/RadixHashJoin.h>
#include <Interpreters/TableJoin.h>
#include <Parsers/ASTTablesInSelectQuery.h>
#include <Common/HashTable/Hash.h>
#include <Common/ThreadStatus.h>

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>
#include <unistd.h>

/// ── Additive, driver-only PMU phase instrumentation ────────────────────────────────────────────────
/// These headers and the PerfGroup helper below are used ONLY by the benchmark driver to attribute
/// hardware counters to the probe phase. They do not touch the production join classes in any way.
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>

#include <cstdint>
#include <cstring>

namespace
{

using namespace DB;

/// The server in the reference setup is pinned to physical cores 0-47; pin every worker into that window.
constexpr int PHYS_CORES = 48;
/// One generated block carries this many rows (close to the server's default max_block_size).
constexpr UInt64 BLOCK_ROWS = 65536;

void pinToCore(int core)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

/// Run `fn(tid)` on `threads` workers, worker `t` pinned to core `first_core + (t % PHYS_CORES)`.
void runParallel(int threads, int first_core, const std::function<void(int)> & fn)
{
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([t, first_core, &fn]
        {
            pinToCore(first_core + (t % PHYS_CORES));
            fn(t);
        });
    for (auto & th : pool)
        th.join();
}

/// ── PMU phase counter (driver-only; production join path untouched) ──────────────────────────────────
/// A thread-local group of perf events opened on the CALLING thread (pid=0, cpu=-1, so it follows the
/// pinned worker), enabled/disabled around exactly the probe loop so the counts attribute to the probe
/// phase only. Raw event configs (PERF_TYPE_RAW) were obtained on this CPU via `perf stat -vv` and are
/// cited next to each entry; cycles/instructions use PERF_TYPE_HARDWARE (fixed counters). exclude_kernel
/// isolates the join's own user-mode loads from page-fault / scheduler noise. Additive counts are summed
/// across workers by the caller; ratio estimators (MLP = pending/pending_cycles, etc.) are computed in
/// post-processing from the raw sums.
struct EvDef
{
    const char * name;
    uint32_t type;
    uint64_t config;
};

/// Event groups, each small enough to avoid PMU counter multiplexing (verified 100% enabled).
/// "lfb"  -> LFB-level MLP + fill-buffer saturation; "off" -> offcore/DRAM-level MLP; "loc" -> per-load
/// cache-hierarchy hit distribution.
inline std::vector<EvDef> perfGroup(const std::string & g)
{
    constexpr uint32_t raw = PERF_TYPE_RAW;       /// == 4 on this kernel ("cpu" core PMU)
    constexpr uint32_t hw = PERF_TYPE_HARDWARE;
    const EvDef cyc{"cycles", hw, PERF_COUNT_HW_CPU_CYCLES};
    const EvDef ins{"instructions", hw, PERF_COUNT_HW_INSTRUCTIONS};
    if (g == "lfb")
        return {{"pending", raw, 0x148}, {"pending_cycles", raw, 0x1000148}, {"fb_full", raw, 0x248}, cyc, ins};
    if (g == "off")
        return {{"demand_data_rd", raw, 0x120}, {"cycles_with_demand_data_rd", raw, 0x1000120}, cyc, ins};
    /// Prefetch-INCLUSIVE offcore read occupancy (data_rd counts demand + L1D prefetch, unlike the
    /// demand-only "off" group). For software-prefetch probes this is the true offcore/DRAM-level MLP.
    if (g == "off2")
        return {{"data_rd", raw, 0x820}, {"cycles_with_data_rd", raw, 0x1000820}, cyc, ins};
    /// L2 locality including the engine's own software prefetches. swpf_hit/swpf_miss reveal where the
    /// probe's prefetches actually land (L2-resident leaf vs a line that must be fetched from L3/DRAM) —
    /// the locality estimator that is NOT confounded by prefetch (mem_load_retired is).
    if (g == "l2")
        return {{"references", raw, 0xff24}, {"miss", raw, 0x3f24}, {"swpf_hit", raw, 0xc824},
                {"swpf_miss", raw, 0x2824}, cyc};
    /// Execution-stall attribution: fraction of cycles stalled with an outstanding miss at each level.
    /// Separates "memory-latency-bound" (stalls_l3_miss high) from "core/throughput-bound".
    if (g == "stall")
        return {{"stalls_total", raw, 0x40004a3}, {"stalls_l1d_miss", raw, 0xc000ca3},
                {"stalls_l2_miss", raw, 0x50005a3}, {"stalls_l3_miss", raw, 0x60006a3}, cyc, ins};
    /// mem_load_retired.* (event 0xd1) are restricted to 4 GP counters on this CPU and .fb_hit is
    /// unsupported here, so `loc` carries the 4 hierarchy buckets (l1/l2/l3 hit + l3 miss) plus the two
    /// fixed counters. Their sum approximates total retired loads; per-bucket fractions give locality.
    if (g == "loc")
        return {{"l1_hit", raw, 0x1d1}, {"l2_hit", raw, 0x2d1}, {"l3_hit", raw, 0x4d1},
                {"l3_miss", raw, 0x20d1}, cyc, ins};
    /// dTLB load page-walk activity (DTLB_LOAD_MISSES, event 0x12). walk_completed (umask 0x0E) counts
    /// finished walks; walk_pending (umask 0x10) is the per-cycle PMH occupancy (avg outstanding walks);
    /// walk_active (umask 0x10 + cmask 1) is cycles with >=1 PMH walk in flight. Raw configs verified on
    /// this CPU via `perf stat -vv` (walk_active=0x1001012, walk_pending=0x1012, walk_completed=0xe12).
    /// Separates "page-walk cost" (walk_active% / PMH occupancy up) from "more TLB misses" (walks/row up).
    if (g == "tlb")
        return {{"walk_pending", raw, 0x1012}, {"walk_active", raw, 0x1001012}, {"walk_completed", raw, 0xe12}, cyc, ins};
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

    /// Group read layout: nr, time_enabled, time_running, value[0..nr). Scale by enabled/running to
    /// correct for any multiplexing (a no-op when the group fits, i.e. running == enabled).
    std::vector<uint64_t> read() const
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

/// Data model (identical to the SQL reference): build_key[i] = intHash64(i) for i in [0, B) is a bijection,
/// so the build side holds B DISTINCT pseudo-random UInt64 keys; probe_key[j] = intHash64(j % B), so every
/// probe row matches exactly one build key, fan-out is exactly P/B, and count() == P. `intHash64` is the
/// Murmur finalizer in src/Common/HashTable/Hash.h.
UInt64 buildKey(UInt64 i)
{
    return intHash64(i);
}
UInt64 probeKey(UInt64 j, UInt64 build_rows)
{
    return intHash64(j % build_rows);
}

/// Build a one-column Block {k0 : UInt64} for rows [begin, begin + count): build-side keys when
/// `build_rows == 0`, probe-side keys (folded modulo `build_rows`) otherwise.
Block makeKeyBlock(UInt64 begin, UInt64 count, UInt64 build_rows)
{
    auto col = ColumnUInt64::create();
    auto & data = col->getData();
    data.resize(count);
    if (build_rows == 0)
        for (UInt64 r = 0; r < count; ++r)
            data[r] = buildKey(begin + r);
    else
        for (UInt64 r = 0; r < count; ++r)
            data[r] = probeKey(begin + r, build_rows);

    Block block;
    block.insert(ColumnWithTypeAndName(std::move(col), std::make_shared<DataTypeUInt64>(), "k0"));
    return block;
}

/// Construct the `TableJoin` describing `... INNER JOIN ... USING (k0)` with one non-nullable UInt64 key.
/// The `Settings`-based constructor (src/Interpreters/TableJoin.h:265) populates every block-sizing knob
/// from defaults and sets `enable_analyzer = true`, matching a real query; we then set kind/strictness,
/// the single key clause, and the column lists. This satisfies the RadixHashJoin key gate
/// (src/Planner/PlannerJoins.cpp:1174 radixHashJoinApplicable): one disjunct, Inner, ALL, fixed-width
/// non-nullable key of packed width 8 (a multiple of 4 in [4, 64]).
std::shared_ptr<TableJoin> makeTableJoin(bool join_prefetch = true)
{
    Settings settings;
    /// Toggle the real join-probe software-prefetch setting (CHJ/HashJoin honors it; RHJ does not).
    if (!join_prefetch)
        settings.set("enable_software_prefetch_in_join", false);
    auto table_join = std::make_shared<TableJoin>(settings, /*tmp_volume_=*/ nullptr, /*tmp_data_=*/ nullptr);

    table_join->getTableJoin().kind = JoinKind::Inner;
    table_join->getTableJoin().strictness = JoinStrictness::All;

    /// `getOnlyClause()` asserts the clause is already non-empty, so populate the freshly added clause via
    /// `getClauses()` (the same field `StorageJoin`'s ctor writes directly).
    table_join->addDisjunct();
    auto & clause = table_join->getClauses()[0];
    clause.key_names_left = {"k0"};
    clause.key_names_right = {"k0"};

    const NamesAndTypesList key_cols{{"k0", std::make_shared<DataTypeUInt64>()}};
    /// right joined columns, left-table column NameSet, right-table prefix, left-table columns.
    table_join->setColumnsFromJoinedTable(key_cols, {"k0"}, "", key_cols);

    return table_join;
}

struct Args
{
    std::string engine = "rhj"; /// rhj | chj
    UInt64 build_rows = 100'000'000ULL;
    UInt64 probe_rows = 0; /// 0 -> fanout * build_rows
    double fanout = 4.0;
    int threads = 48;
    int repeats = 1;
    int first_core = 0;
    /// `radix_hash_join_size_tables_by_distinct_estimate`: current production default is true
    /// (src/Core/Settings.cpp). -1 -> use that default; 0/1 -> force off/on. RHJ only.
    int distinct_estimate = -1;
    bool header = false;
    /// Driver-only PMU instrumentation. When `perf_group != "none"`, the join is built ONCE and then the
    /// probe is run `probe_repeats` times, each measured with per-worker perf counters attributed to the
    /// probe phase only (groups: lfb | off | off2 | loc | l2 | stall). Does not affect the engines.
    std::string perf_group = "none";
    /// <0 means "auto": resolved in main() to 3 for --report, 5 for a single --perf-group run.
    int probe_repeats = -1;
    /// One-shot validation mode: build once, then measure EVERY counter group and print a single
    /// consolidated report (all the MLP / locality / stall / IPC signals used to diagnose the engines).
    /// Each group is a separate probe pass so nothing is multiplexed. Intended for quickly checking
    /// whether a performance experiment moved the metrics in the expected direction.
    bool report = false;
    /// Build with a different worker count than the probe (instrumented mode only). The leaf layout /
    /// two-level map structure is independent of the worker count, so building in parallel while probing
    /// single-threaded yields a faithful per-core probe measurement without a slow single-threaded build.
    /// -1 -> same as --threads.
    int build_threads = -1;
    /// Disable the join's software prefetch via the real `enable_software_prefetch_in_join` setting.
    /// Only CHJ (HashJoin) consults it; RHJ's AMAC ring prefetch is independent of this setting.
    bool join_prefetch = true;
};

UInt64 parseU64(const char * s)
{
    return std::strtoull(s, nullptr, 10);
}

double nowMs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

/// One build+probe of the reference query against the real engine. Returns matched row count;
/// fills build_ms (addBlockToJoin across workers + onBuildPhaseFinish) and probe_ms (joinBlock across
/// workers). Mirrors the proxy's build_ms/probe_ms split.
UInt64 runOnce(const Args & args, JoinPtr join, double & build_ms, double & probe_ms)
{
    const UInt64 build_blocks = (args.build_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    const UInt64 probe_blocks = (args.probe_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    /// ---- Build phase: parallel addBlockToJoin, one build lane per worker (lock-free in RHJ). ----
    auto build_start = std::chrono::steady_clock::now();
    {
        std::atomic<UInt64> next_block{0};
        runParallel(args.threads, args.first_core, [&](int tid)
        {
            for (;;)
            {
                const UInt64 b = next_block.fetch_add(1, std::memory_order_relaxed);
                if (b >= build_blocks)
                    break;
                const UInt64 begin = b * BLOCK_ROWS;
                const UInt64 count = std::min<UInt64>(BLOCK_ROWS, args.build_rows - begin);
                Block block = makeKeyBlock(begin, count, /*build_rows=*/ 0);
                join->addBlockToJoin(block, count, /*check_limits=*/ false, /*build_lane=*/ static_cast<size_t>(tid));
            }
        });
    }
    /// The eager post-build (scatter + leaf-table build for RHJ; map merge for CHJ) is build-phase work.
    join->onBuildPhaseFinish();
    auto build_end = std::chrono::steady_clock::now();
    build_ms = nowMs(build_start, build_end);

    /// ---- Probe phase: parallel joinBlock, drain IJoinResult::next(), count matched rows. ----
    std::vector<UInt64> partial(args.threads, 0);
    auto probe_start = std::chrono::steady_clock::now();
    {
        std::atomic<UInt64> next_block{0};
        runParallel(args.threads, args.first_core, [&](int tid)
        {
            UInt64 count = 0;
            for (;;)
            {
                const UInt64 b = next_block.fetch_add(1, std::memory_order_relaxed);
                if (b >= probe_blocks)
                    break;
                const UInt64 begin = b * BLOCK_ROWS;
                const UInt64 n = std::min<UInt64>(BLOCK_ROWS, args.probe_rows - begin);
                Block block = makeKeyBlock(begin, n, /*build_rows=*/ args.build_rows);
                JoinResultPtr res = join->joinBlock(std::move(block), static_cast<size_t>(tid));
                for (;;)
                {
                    IJoinResult::JoinResultBlock jb = res->next();
                    count += jb.block.rows();
                    if (jb.is_last)
                        break;
                }
            }
            partial[tid] = count;
        });
    }
    auto probe_end = std::chrono::steady_clock::now();
    probe_ms = nowMs(probe_start, probe_end);

    UInt64 matches = 0;
    for (UInt64 c : partial)
        matches += c;
    return matches;
}

JoinPtr makeJoin(const Args & args, const std::shared_ptr<TableJoin> & table_join, const SharedHeader & right_header)
{
    /// Size the engine for the larger of build/probe worker counts (lane space); the leaf layout (RHJ)
    /// and merged two-level map (CHJ) are independent of this, so it does not change probe behavior.
    const int build_threads = args.build_threads > 0 ? args.build_threads : args.threads;
    const size_t lanes = static_cast<size_t>(std::max(args.threads, build_threads));
    if (args.engine == "rhj")
    {
        const bool size_by_distinct = args.distinct_estimate < 0 ? true : (args.distinct_estimate != 0);
        return std::make_shared<RadixHashJoin>(
            table_join,
            right_header,
            lanes,
            /*rhs_size_estimation_=*/ std::optional<UInt64>(args.build_rows),
            /*max_partitions_per_pass_=*/ 8192,
            /*size_tables_by_distinct_estimate_=*/ size_by_distinct,
            StatsCollectingParams{});
    }
    return std::make_shared<ConcurrentHashJoin>(
        table_join, lanes, right_header, StatsCollectingParams{});
}

/// Build the join ONCE (same `IJoin` calls as `runOnce`; `build_threads` may differ from the probe worker
/// count — the leaf layout (RHJ) and merged two-level map (CHJ) are independent of it). Returns the built
/// join and fills `build_ms`.
JoinPtr buildJoinOnce(const Args & args, const SharedHeader & right_header, int build_threads, double & build_ms)
{
    auto table_join = makeTableJoin(args.join_prefetch);
    JoinPtr join = makeJoin(args, table_join, right_header);
    const UInt64 build_blocks = (args.build_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    auto build_start = std::chrono::steady_clock::now();
    {
        std::atomic<UInt64> next_block{0};
        runParallel(build_threads, args.first_core, [&](int tid)
        {
            for (;;)
            {
                const UInt64 b = next_block.fetch_add(1, std::memory_order_relaxed);
                if (b >= build_blocks)
                    break;
                const UInt64 begin = b * BLOCK_ROWS;
                const UInt64 count = std::min<UInt64>(BLOCK_ROWS, args.build_rows - begin);
                Block block = makeKeyBlock(begin, count, /*build_rows=*/ 0);
                join->addBlockToJoin(block, count, /*check_limits=*/ false, /*build_lane=*/ static_cast<size_t>(tid));
            }
        });
    }
    join->onBuildPhaseFinish();
    build_ms = nowMs(build_start, std::chrono::steady_clock::now());
    return join;
}

/// One parallel probe pass with `group` counted per-worker (attributed to the probe only) and the additive
/// counts summed across workers. The engine is driven exactly as in `runOnce`.
struct ProbeSample
{
    double ms = 0.0;
    UInt64 matches = 0;
    std::vector<uint64_t> counts; /// summed across workers, in `group` order
    int perf_threads = 0;
};

ProbeSample probePass(const Args & args, const JoinPtr & join, const std::vector<EvDef> & group, UInt64 probe_blocks)
{
    std::vector<UInt64> partial(args.threads, 0);
    std::vector<std::vector<uint64_t>> per_thread(args.threads);
    std::vector<char> perf_ok(args.threads, 0);

    auto probe_start = std::chrono::steady_clock::now();
    {
        std::atomic<UInt64> next_block{0};
        runParallel(args.threads, args.first_core, [&](int tid)
        {
            PerfGroup pg;
            if (!group.empty())
                pg.open(group);
            if (pg.ok)
            {
                pg.reset();
                pg.enable();
            }
            UInt64 count = 0;
            for (;;)
            {
                const UInt64 b = next_block.fetch_add(1, std::memory_order_relaxed);
                if (b >= probe_blocks)
                    break;
                const UInt64 begin = b * BLOCK_ROWS;
                const UInt64 n = std::min<UInt64>(BLOCK_ROWS, args.probe_rows - begin);
                Block block = makeKeyBlock(begin, n, /*build_rows=*/ args.build_rows);
                JoinResultPtr res = join->joinBlock(std::move(block), static_cast<size_t>(tid));
                for (;;)
                {
                    IJoinResult::JoinResultBlock jb = res->next();
                    count += jb.block.rows();
                    if (jb.is_last)
                        break;
                }
            }
            if (pg.ok)
            {
                pg.disable();
                per_thread[tid] = pg.read();
                perf_ok[tid] = 1;
                pg.closeAll();
            }
            partial[tid] = count;
        });
    }

    ProbeSample s;
    s.ms = nowMs(probe_start, std::chrono::steady_clock::now());
    for (UInt64 c : partial)
        s.matches += c;
    s.counts.assign(group.size(), 0);
    for (int t = 0; t < args.threads; ++t)
        if (perf_ok[t] && per_thread[t].size() == group.size())
        {
            for (size_t i = 0; i < group.size(); ++i)
                s.counts[i] += per_thread[t][i];
            ++s.perf_threads;
        }
    return s;
}

double medianOf(std::vector<double> v)
{
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/// PMU-instrumented run for ONE group: build once, probe `probe_repeats` times, emit one CSV row per
/// repeat with the raw summed counts; MLP/ratios are derived downstream.
void runInstrumented(const Args & args, const SharedHeader & right_header)
{
    const std::vector<EvDef> group = perfGroup(args.perf_group);
    const int build_threads = args.build_threads > 0 ? args.build_threads : args.threads;
    const UInt64 probe_blocks = (args.probe_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    double build_ms = 0.0;
    JoinPtr join = buildJoinOnce(args, right_header, build_threads, build_ms);
    fmt::print(
        stderr, "[{}] built build={} in {:.1f} ms (build_threads={}); perf-group={} probe-repeats={} probe_threads={}\n",
        args.engine, args.build_rows, build_ms, build_threads, args.perf_group, args.probe_repeats, args.threads);

    fmt::print("engine,group,build_rows,probe_rows,threads,probe_rep,probe_ms,matches,ok,counts\n");

    for (int pr = 0; pr < args.probe_repeats; ++pr)
    {
        const ProbeSample s = probePass(args, join, group, probe_blocks);
        std::string counts;
        for (size_t i = 0; i < group.size(); ++i)
        {
            if (i)
                counts += "|";
            counts += fmt::format("{}:{}", group[i].name, s.counts[i]);
        }
        counts += fmt::format("|perf_threads:{}", s.perf_threads);

        const bool ok = (s.matches == args.probe_rows);
        fmt::print(
            "{},{},{},{},{},{},{:.3f},{},{},{}\n",
            args.engine, args.perf_group, args.build_rows, args.probe_rows, args.threads, pr, s.ms,
            s.matches, ok ? 1 : 0, counts);
        (void)std::fflush(stdout);
        fmt::print(stderr, "  probe_rep {}: {:.1f} ms matches={} {} [{}]\n", pr, s.ms, s.matches, ok ? "OK" : "MISMATCH", counts);
    }
}

/// One-shot validation report: build once, then measure EVERY counter group (each a separate probe pass so
/// nothing is multiplexed) and print one consolidated block with all the MLP / locality / stall / IPC
/// signals used to diagnose the engines, plus a single machine-parseable `SUMMARY` line for diffing runs.
void runReport(const Args & args, const SharedHeader & right_header)
{
    const int build_threads = args.build_threads > 0 ? args.build_threads : args.threads;
    const UInt64 probe_blocks = (args.probe_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    double build_ms = 0.0;
    JoinPtr join = buildJoinOnce(args, right_header, build_threads, build_ms);
    fmt::print(stderr, "[{}] built build={} in {:.1f} ms (build_threads={}); running report...\n",
               args.engine, args.build_rows, build_ms, build_threads);

    /// Median raw counts, namespaced by "group.name" (so each ratio uses cycles from its own pass).
    std::map<std::string, double> med_counts;
    std::vector<double> probe_ms_all;
    UInt64 matches = 0;
    bool perf_ok = true;

    for (const char * gname : {"lfb", "off2", "off", "l2", "stall", "loc", "tlb"})
    {
        const auto group = perfGroup(gname);
        std::vector<std::vector<double>> per_count(group.size());
        std::vector<double> ms_list;
        for (int r = 0; r < args.probe_repeats; ++r)
        {
            const ProbeSample s = probePass(args, join, group, probe_blocks);
            matches = s.matches;
            if (s.perf_threads < args.threads)
                perf_ok = false;
            if (r == 0 && args.probe_repeats > 1)
                continue; /// warmup
            for (size_t i = 0; i < group.size(); ++i)
                per_count[i].push_back(static_cast<double>(s.counts[i]));
            ms_list.push_back(s.ms);
        }
        for (size_t i = 0; i < group.size(); ++i)
            med_counts[fmt::format("{}.{}", gname, group[i].name)] = medianOf(per_count[i]);
        probe_ms_all.push_back(medianOf(ms_list));
    }

    auto val = [&](const char * k) -> double { auto it = med_counts.find(k); return it != med_counts.end() ? it->second : 0.0; };
    auto rat = [&](const char * a, const char * b) -> double { const double d = val(b); return d > 0 ? val(a) / d : 0.0; };

    const double probe_ms = medianOf(probe_ms_all);
    const double rows = static_cast<double>(args.probe_rows);
    const double ns_per_row = probe_ms * 1e6 / rows * args.threads; /// per-core ns/row
    const double mlp_off_all = rat("off2.data_rd", "off2.cycles_with_data_rd");
    const double mlp_off_dmd = rat("off.demand_data_rd", "off.cycles_with_demand_data_rd");
    const double mlp_lfb = rat("lfb.pending", "lfb.pending_cycles");
    const double fb_full = rat("lfb.fb_full", "lfb.cycles");
    const double pend_frac = rat("lfb.pending_cycles", "lfb.cycles");
    const double ipc = rat("lfb.instructions", "lfb.cycles");
    const double inst_per_row = val("lfb.instructions") / rows;
    const double l2_miss = rat("l2.miss", "l2.references");
    const double swpf_tot = val("l2.swpf_hit") + val("l2.swpf_miss");
    const double swpf_hit = swpf_tot > 0 ? val("l2.swpf_hit") / swpf_tot : 0.0;
    const double swpf_per_row = swpf_tot / rows;
    const double st_total = rat("stall.stalls_total", "stall.cycles");
    const double st_l2 = rat("stall.stalls_l2_miss", "stall.cycles");
    const double st_l3 = rat("stall.stalls_l3_miss", "stall.cycles");
    const double loc_tot = val("loc.l1_hit") + val("loc.l2_hit") + val("loc.l3_hit") + val("loc.l3_miss");
    auto pct = [&](const char * k) { return loc_tot > 0 ? 100.0 * val(k) / loc_tot : 0.0; };
    /// Page-walk activity (dTLB load walks). walks_per_row isolates COUNT; walk_active%/pmh_occupancy isolate COST.
    const double walks_per_row = val("tlb.walk_completed") / rows;
    const double walk_active = rat("tlb.walk_active", "tlb.cycles");
    const double pmh_occupancy = rat("tlb.walk_pending", "tlb.cycles");
    const bool ok = (matches == args.probe_rows);
    const char * de = args.engine == "rhj"
        ? (args.distinct_estimate < 0 ? "default(on)" : (args.distinct_estimate ? "on" : "off")) : "n/a";

    fmt::print("\n==================== bench_rhj_vs_chj PERF REPORT ====================\n");
    fmt::print("engine={} build={} probe={} threads={} build_threads={} join_prefetch={} distinct_estimate={}\n",
               args.engine, args.build_rows, args.probe_rows, args.threads, build_threads, args.join_prefetch ? 1 : 0, de);
    fmt::print("build_ms={:.1f}  probe_ms(median)={:.1f}  ns/row/core={:.2f}  matches={} ({})\n",
               build_ms, probe_ms, ns_per_row, matches, ok ? "OK" : "MISMATCH");
    fmt::print("verified per-core ceilings: L1 fill buffers (LFB/MSHR)=16 (sustained ~12)   L2 superqueue=48\n");
    if (!perf_ok)
        fmt::print("WARNING: some perf groups failed to open on all workers (counts may be partial)\n");

    fmt::print("\nMEMORY-LEVEL PARALLELISM (avg outstanding requests while busy)\n");
    fmt::print("  MLP offcore  true (demand+prefetch) data_rd/cyc_with        = {:6.2f}   [ceiling: L2 superqueue 48]\n", mlp_off_all);
    fmt::print("  MLP offcore  demand only            demand_data_rd/cyc_with = {:6.2f}\n", mlp_off_dmd);
    fmt::print("  MLP LFB      demand L1-miss          pending/pending_cycles  = {:6.2f}   [ceiling: LFB 16, sust ~12]\n", mlp_lfb);
    fmt::print("  fill-buffer-full stall fraction      fb_full/cycles          = {:5.1f}%\n", 100 * fb_full);
    fmt::print("  cycles with >=1 L1 miss pending      pending_cycles/cycles   = {:5.1f}%\n", 100 * pend_frac);

    fmt::print("\nLOCALITY (includes the engine's OWN software prefetches)\n");
    fmt::print("  L2 miss rate                         l2_rqsts.miss/refs      = {:5.1f}%\n", 100 * l2_miss);
    fmt::print("  SW-prefetch L2 hit rate              swpf_hit/(hit+miss)     = {:5.1f}%  (low => prefetch fetches from L3/DRAM)\n", 100 * swpf_hit);
    fmt::print("  SW prefetches per probe row                                  = {:6.3f}\n", swpf_per_row);
    fmt::print("  retired-load view (prefetch-CONFOUNDED) L1/L2/L3/DRAM        = {:.1f}/{:.1f}/{:.1f}/{:.1f}%\n",
               pct("loc.l1_hit"), pct("loc.l2_hit"), pct("loc.l3_hit"), pct("loc.l3_miss"));

    fmt::print("\nEXECUTION / STALLS\n");
    fmt::print("  IPC                                  instructions/cycles     = {:6.2f}\n", ipc);
    fmt::print("  instructions per probe row                                   = {:6.1f}\n", inst_per_row);
    fmt::print("  stalls total                         stalls_total/cycles     = {:5.1f}%\n", 100 * st_total);
    fmt::print("  stalls with >=1 L2 miss outstanding  stalls_l2_miss/cycles   = {:5.1f}%\n", 100 * st_l2);
    fmt::print("  stalls with >=1 L3 miss outstanding  stalls_l3_miss/cycles   = {:5.1f}%  (~0 => prefetch hides DRAM)\n", 100 * st_l3);

    fmt::print("\nPAGE WALKS (dTLB load misses; isolates walk COST from walk COUNT)\n");
    fmt::print("  dTLB page walks per probe row        walk_completed/row      = {:6.3f}  (COUNT; ~flat => not more misses)\n", walks_per_row);
    fmt::print("  cycles with >=1 PMH walk active      walk_active/cycles      = {:5.1f}%  (COST; up => walks longer/denser)\n", 100 * walk_active);
    fmt::print("  avg page-walks outstanding per cycle walk_pending/cycles     = {:6.3f}  (PMH occupancy stealing MLP)\n", pmh_occupancy);

    fmt::print(
        "\nSUMMARY engine={} build={} probe={} threads={} ns_per_row={:.2f} mlp_off={:.2f} mlp_off_dmd={:.2f} "
        "mlp_lfb={:.2f} fb_full={:.3f} l2_miss={:.3f} swpf_hit={:.3f} swpf_per_row={:.3f} ipc={:.2f} "
        "inst_per_row={:.1f} stalls_total={:.3f} stalls_l2={:.3f} stalls_l3={:.3f} "
        "walks_per_row={:.3f} walk_active={:.3f} pmh_occ={:.3f} ok={}\n",
        args.engine, args.build_rows, args.probe_rows, args.threads, ns_per_row, mlp_off_all, mlp_off_dmd,
        mlp_lfb, fb_full, l2_miss, swpf_hit, swpf_per_row, ipc, inst_per_row, st_total, st_l2, st_l3,
        walks_per_row, walk_active, pmh_occupancy, ok ? 1 : 0);
    fmt::print("======================================================================\n");
    (void)std::fflush(stdout);
}

}

int main(int argc, char ** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--engine") args.engine = next();
        else if (a == "--build") args.build_rows = parseU64(next());
        else if (a == "--probe") args.probe_rows = parseU64(next());
        else if (a == "--fanout") args.fanout = std::strtod(next(), nullptr);
        else if (a == "--threads") args.threads = static_cast<int>(parseU64(next()));
        else if (a == "--repeats") args.repeats = static_cast<int>(parseU64(next()));
        else if (a == "--first-core") args.first_core = static_cast<int>(parseU64(next()));
        else if (a == "--distinct-estimate") args.distinct_estimate = static_cast<int>(parseU64(next()));
        else if (a == "--perf-group") args.perf_group = next();
        else if (a == "--probe-repeats") args.probe_repeats = static_cast<int>(parseU64(next()));
        else if (a == "--build-threads") args.build_threads = static_cast<int>(parseU64(next()));
        else if (a == "--no-join-prefetch") args.join_prefetch = false;
        else if (a == "--report") args.report = true;
        else if (a == "--header") args.header = true;
        else if (a == "--help")
        {
            fmt::print(
                "Usage: bench_rhj_vs_chj --engine rhj|chj [--build N] [--probe M | --fanout F]\n"
                "       [--threads T] [--repeats R] [--first-core C] [--distinct-estimate 0|1] [--header]\n"
                "       [--perf-group lfb|off|off2|loc|l2|stall] [--probe-repeats N] [--build-threads B]\n"
                "       [--no-join-prefetch] [--report]\n"
                "\n"
                "  --report   Build once, then measure EVERY PMU group and print one consolidated report of\n"
                "             all MLP/locality/stall/IPC signals (each group is a separate, non-multiplexed\n"
                "             probe pass) plus a machine-parseable SUMMARY line. Use it to validate whether a\n"
                "             performance experiment moved the metrics. Quick example (per-core, ~seconds):\n"
                "               bench_rhj_vs_chj --engine rhj --build 100000000 --threads 1 --report\n");
            return 0;
        }
        else
        {
            fmt::print(stderr, "unknown arg: {}\n", a);
            return 2;
        }
    }

    /// Probe sizing: default to fanout*build, but cap the auto-default in --report mode so a bare
    /// `--report` runs in seconds (an explicit --probe always wins). Random distinct keys make a smaller
    /// probe an unbiased sample for the per-row counters.
    if (args.probe_rows == 0)
    {
        const UInt64 full = static_cast<UInt64>(args.fanout * static_cast<double>(args.build_rows));
        args.probe_rows = args.report ? std::min<UInt64>(full, 50'000'000ULL) : full;
    }
    args.threads = std::max(args.threads, 1);
    /// Resolve "auto" probe repeats: 3 for the consolidated report, 5 for a single-group run.
    if (args.probe_repeats < 0)
        args.probe_repeats = args.report ? 3 : 5;
    /// In --report mode, build in parallel by default even when probing single-threaded (fast build; the
    /// leaf layout is independent of the worker count). An explicit --build-threads still wins.
    if (args.report && args.build_threads < 0)
        args.build_threads = PHYS_CORES;
    if (args.engine != "rhj" && args.engine != "chj")
    {
        fmt::print(stderr, "engine must be rhj or chj\n");
        return 2;
    }

    /// Give the main thread a ThreadStatus so the engines' internal ThreadPools (which capture the current
    /// thread group via ThreadGroupSwitcher) behave; a null query group is tolerated by both engines.
    DB::MainThreadStatus::getInstance();

    const double fanout = static_cast<double>(args.probe_rows) / static_cast<double>(args.build_rows);

    if (args.header)
        fmt::print("engine,build_rows,probe_rows,fanout,threads,repeat,build_ms,probe_ms,total_ms,matches,ok\n");

    const char * de = args.engine == "rhj"
        ? (args.distinct_estimate < 0 ? "default(on)" : (args.distinct_estimate ? "on" : "off"))
        : "n/a";
    fmt::print(
        stderr,
        "[{}] build={} probe={} fanout={:.2f} threads={} repeats={} distinct_estimate={}\n",
        args.engine, args.build_rows, args.probe_rows, fanout, args.threads, args.repeats, de);

    const SharedHeader right_header = std::make_shared<const DB::Block>(makeKeyBlock(0, 0, 0).cloneEmpty());

    /// Consolidated validation report (driver-only): build once, measure every group, print one report.
    if (args.report)
    {
        runReport(args, right_header);
        return 0;
    }

    /// PMU-instrumented mode (driver-only): build once, probe many, with per-worker phase counters.
    if (args.perf_group != "none")
    {
        static const std::vector<std::string> valid{"lfb", "off", "off2", "loc", "l2", "stall", "tlb"};
        if (std::find(valid.begin(), valid.end(), args.perf_group) == valid.end())
        {
            fmt::print(stderr, "--perf-group must be one of: lfb | off | off2 | loc | l2 | stall | tlb\n");
            return 2;
        }
        runInstrumented(args, right_header);
        return 0;
    }

    for (int rep = 0; rep < args.repeats; ++rep)
    {
        auto table_join = makeTableJoin(args.join_prefetch);
        JoinPtr join = makeJoin(args, table_join, right_header);

        double build_ms = 0.0;
        double probe_ms = 0.0;
        const UInt64 matches = runOnce(args, join, build_ms, probe_ms);
        const double total_ms = build_ms + probe_ms;
        const bool ok = (matches == args.probe_rows);

        fmt::print(
            "{},{},{},{:.4f},{},{},{:.3f},{:.3f},{:.3f},{},{}\n",
            args.engine, args.build_rows, args.probe_rows, fanout, args.threads, rep,
            build_ms, probe_ms, total_ms, matches, ok ? 1 : 0);
        (void)std::fflush(stdout); /// stream CSV rows promptly during long matrix runs

        fmt::print(
            stderr,
            "  rep {}: build {:.1f} ms, probe {:.1f} ms, total {:.1f} ms, matches={} {}\n",
            rep, build_ms, probe_ms, total_ms, matches, ok ? "OK" : "MISMATCH!");
    }
    return 0;
}
