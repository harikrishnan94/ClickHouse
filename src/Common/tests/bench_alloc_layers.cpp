/// bench_alloc_layers.cpp
///
/// Isolates the cost of each layer in the IColumn allocation stack and prints
/// per-call latency under multi-threaded load.  Each "mode" tests exactly one
/// thing so that the difference between two modes pins the cost to a single
/// layer.
///
/// Modes:
///   aligned_alloc    — `std::aligned_alloc(align, size)` + `free`
///   malloc           — `malloc(size)` + `free`
///   allocator        — `Allocator<false,false>::alloc(size, 64)` + `free`
///                      (this includes CurrentMemoryTracker accounting)
///   podarray         — empty PaddedPODArray<UInt64> + reserve(n) (+ implicit free)
///   clone_empty      — only `ColumnVector<UInt64>::create()` (the COW `new`)
///   clone_reserve    — `cloneEmpty + getData().reserve(n)`  (no resize)
///   clone_reserve_resize — `cloneEmpty + getData().reserve(n) + resize_assume_reserved(n)`
///   full_touch       — full pattern + `memset(buf, 1, n*8)` to force page commit
///
/// Output: ns/call (min, mean), total wall time, allocations/sec/thread,
/// page-faults (minor/major) before/after via getrusage(RUSAGE_THREAD).
///
/// Usage:
///   bench_alloc_layers --mode <name> --n 4096 --calls 50000 --T 8 [--keep|--free]

#include <Columns/ColumnVector.h>
#include <Core/Types.h>
#include <Common/Allocator.h>
#include <Common/CurrentThread.h>
#include <Common/MemoryTrackerBlockerInThread.h>
#include <Common/PODArray.h>
#include <Common/ThreadPool.h>
#include <Common/ThreadStatus.h>

#include <jemalloc/jemalloc.h>

#include <fmt/format.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/resource.h>

namespace
{

using namespace DB;
using Clk = std::chrono::steady_clock;

struct Config
{
    std::string mode;
    size_t n = 4096; // elements (UInt64) per allocation
    size_t calls = 50000; // calls per thread per rep
    int T = 1;
    int reps = 3;
    bool keep = false; // keep all allocations alive in a vector (forces no reuse)
    bool touch = false; // memset each newly allocated buffer to force commit
    bool thread_status = false; // install DB::ThreadStatus so per-thread untracked memory fast path kicks in
    bool block_tracker = false; // install MemoryTrackerBlockerInThread to skip tracker accounting
    std::string thread_kind = "std"; // "std" (std::thread, no auto ThreadStatus) or "ch" (ThreadFromGlobalPool, auto ThreadStatus)
};

struct Result
{
    double wall_ns = 0; // wall time for this thread's calls
    long minor_faults = 0;
    long major_faults = 0;
    uint64_t sink = 0; // anti-DCE
};

std::optional<Config> parseCLI(int argc, char ** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto need = [&]() -> std::string
        {
            if (i + 1 >= argc)
            {
                fmt::print(stderr, "missing arg for {}\n", a);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--mode")
            cfg.mode = need();
        else if (a == "--n")
            cfg.n = std::stoull(need());
        else if (a == "--calls")
            cfg.calls = std::stoull(need());
        else if (a == "--T")
            cfg.T = std::stoi(need());
        else if (a == "--reps")
            cfg.reps = std::stoi(need());
        else if (a == "--keep")
            cfg.keep = true;
        else if (a == "--free")
            cfg.keep = false;
        else if (a == "--touch")
            cfg.touch = true;
        else if (a == "--thread-status")
            cfg.thread_status = true;
        else if (a == "--block-tracker")
            cfg.block_tracker = true;
        else if (a == "--thread-kind")
            cfg.thread_kind = need();
        else
        {
            fmt::print(stderr, "unknown arg: {}\n", a);
            return std::nullopt;
        }
    }
    if (cfg.mode.empty())
    {
        fmt::print(stderr, "--mode required\n");
        return std::nullopt;
    }
    return cfg;
}

void pin(int t)
{
    const unsigned n = std::thread::hardware_concurrency();
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(static_cast<unsigned>(t) % n, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

rusage getThreadUsage()
{
    rusage ru{};
    getrusage(RUSAGE_THREAD, &ru);
    return ru;
}

[[gnu::noinline]]
void touch_buf(void * p, size_t bytes)
{
    std::memset(p, 1, bytes);
    asm volatile("" : : "r"(p) : "memory");
}

// ===== Per-mode worker bodies =====

void runAlignedAlloc(const Config & cfg, Result & out)
{
    const size_t bytes = cfg.n * sizeof(UInt64);
    std::vector<void *> kept;
    if (cfg.keep)
        kept.reserve(cfg.calls);

    for (size_t i = 0; i < cfg.calls; ++i)
    {
        void * p = std::aligned_alloc(64, bytes);
        if (cfg.touch)
            touch_buf(p, bytes);
        out.sink ^= reinterpret_cast<uintptr_t>(p);
        if (cfg.keep)
            kept.push_back(p);
        else
            std::free(p);
    }
    for (void * p : kept)
        std::free(p);
}

void runMalloc(const Config & cfg, Result & out)
{
    const size_t bytes = cfg.n * sizeof(UInt64);
    std::vector<void *> kept;
    if (cfg.keep)
        kept.reserve(cfg.calls);

    for (size_t i = 0; i < cfg.calls; ++i)
    {
        void * p = std::malloc(bytes);
        if (cfg.touch)
            touch_buf(p, bytes);
        out.sink ^= reinterpret_cast<uintptr_t>(p);
        if (cfg.keep)
            kept.push_back(p);
        else
            std::free(p);
    }
    for (void * p : kept)
        std::free(p);
}

/// Direct jemalloc, bypassing ClickHouse's malloc.cpp wrappers (and thus
/// bypassing CurrentMemoryTracker entirely).  Only jemalloc work is timed.
void runRawJemalloc(const Config & cfg, Result & out)
{
    const size_t bytes = cfg.n * sizeof(UInt64);
    std::vector<void *> kept;
    if (cfg.keep)
        kept.reserve(cfg.calls);

    for (size_t i = 0; i < cfg.calls; ++i)
    {
        void * p = je_malloc(bytes);
        if (cfg.touch)
            touch_buf(p, bytes);
        out.sink ^= reinterpret_cast<uintptr_t>(p);
        if (cfg.keep)
            kept.push_back(p);
        else
            je_free(p);
    }
    for (void * p : kept)
        je_free(p);
}

/// Same as runRawJemalloc but aligned.
void runRawJemallocAligned(const Config & cfg, Result & out)
{
    const size_t bytes = cfg.n * sizeof(UInt64);
    std::vector<void *> kept;
    if (cfg.keep)
        kept.reserve(cfg.calls);

    for (size_t i = 0; i < cfg.calls; ++i)
    {
        void * p = je_aligned_alloc(64, bytes);
        if (cfg.touch)
            touch_buf(p, bytes);
        out.sink ^= reinterpret_cast<uintptr_t>(p);
        if (cfg.keep)
            kept.push_back(p);
        else
            je_free(p);
    }
    for (void * p : kept)
        je_free(p);
}

void runAllocator(const Config & cfg, Result & out)
{
    using A = Allocator<false, false>;
    A a;
    const size_t bytes = cfg.n * sizeof(UInt64);
    std::vector<void *> kept;
    if (cfg.keep)
        kept.reserve(cfg.calls);

    for (size_t i = 0; i < cfg.calls; ++i)
    {
        void * p = a.alloc(bytes, 64);
        if (cfg.touch)
            touch_buf(p, bytes);
        out.sink ^= reinterpret_cast<uintptr_t>(p);
        if (cfg.keep)
            kept.push_back(p);
        else
            a.free(p, bytes, 64);
    }
    for (void * p : kept)
        a.free(p, bytes, 64);
}

void runPODArray(const Config & cfg, Result & out)
{
    std::vector<PaddedPODArray<UInt64>> kept;
    if (cfg.keep)
        kept.reserve(cfg.calls);

    for (size_t i = 0; i < cfg.calls; ++i)
    {
        PaddedPODArray<UInt64> arr;
        arr.reserve_exact(cfg.n);
        arr.resize_assume_reserved(cfg.n);
        if (cfg.touch)
            touch_buf(arr.data(), cfg.n * sizeof(UInt64));
        out.sink ^= arr.size();
        if (cfg.keep)
            kept.push_back(std::move(arr));
        // else: destructor frees
    }
}

void runCloneEmpty(const Config & cfg, Result & out)
{
    std::vector<DB::ColumnVector<UInt64>::MutablePtr> kept;
    if (cfg.keep)
        kept.reserve(cfg.calls);

    for (size_t i = 0; i < cfg.calls; ++i)
    {
        auto p = DB::ColumnVector<UInt64>::create();
        out.sink ^= reinterpret_cast<uintptr_t>(p.get());
        if (cfg.keep)
            kept.push_back(std::move(p));
    }
}

void runCloneReserve(const Config & cfg, Result & out)
{
    std::vector<DB::ColumnVector<UInt64>::MutablePtr> kept;
    if (cfg.keep)
        kept.reserve(cfg.calls);

    for (size_t i = 0; i < cfg.calls; ++i)
    {
        auto p = DB::ColumnVector<UInt64>::create();
        p->getData().reserve_exact(cfg.n);
        out.sink ^= p->getData().capacity();
        if (cfg.keep)
            kept.push_back(std::move(p));
    }
}

void runCloneReserveResize(const Config & cfg, Result & out)
{
    std::vector<DB::ColumnVector<UInt64>::MutablePtr> kept;
    if (cfg.keep)
        kept.reserve(cfg.calls);

    for (size_t i = 0; i < cfg.calls; ++i)
    {
        auto p = DB::ColumnVector<UInt64>::create();
        p->getData().reserve_exact(cfg.n);
        p->getData().resize_assume_reserved(cfg.n);
        if (cfg.touch)
            touch_buf(p->getData().data(), cfg.n * sizeof(UInt64));
        out.sink ^= p->getData().size();
        if (cfg.keep)
            kept.push_back(std::move(p));
    }
}

void runFullTouch(const Config & cfg, Result & out)
{
    std::vector<DB::ColumnVector<UInt64>::MutablePtr> kept;
    if (cfg.keep)
        kept.reserve(cfg.calls);

    for (size_t i = 0; i < cfg.calls; ++i)
    {
        auto p = DB::ColumnVector<UInt64>::create();
        p->getData().reserve_exact(cfg.n);
        p->getData().resize_assume_reserved(cfg.n);
        touch_buf(p->getData().data(), cfg.n * sizeof(UInt64));
        out.sink ^= p->getData().size();
        if (cfg.keep)
            kept.push_back(std::move(p));
    }
}

void runWorker(const Config & cfg, int tid, Result & out)
{
    pin(tid);

    // Optionally install ThreadStatus so per-thread untracked-memory fast path kicks in.
    std::optional<DB::ThreadStatus> ts;
    if (cfg.thread_status)
        ts.emplace();

    std::optional<MemoryTrackerBlockerInThread> blocker;
    if (cfg.block_tracker)
        blocker.emplace(VariableContext::Global);

    rusage ru0 = getThreadUsage();
    const auto t0 = Clk::now();

    if (cfg.mode == "aligned_alloc")
        runAlignedAlloc(cfg, out);
    else if (cfg.mode == "malloc")
        runMalloc(cfg, out);
    else if (cfg.mode == "raw_jemalloc")
        runRawJemalloc(cfg, out);
    else if (cfg.mode == "raw_jemalloc_aligned")
        runRawJemallocAligned(cfg, out);
    else if (cfg.mode == "allocator")
        runAllocator(cfg, out);
    else if (cfg.mode == "podarray")
        runPODArray(cfg, out);
    else if (cfg.mode == "clone_empty")
        runCloneEmpty(cfg, out);
    else if (cfg.mode == "clone_reserve")
        runCloneReserve(cfg, out);
    else if (cfg.mode == "clone_reserve_resize")
        runCloneReserveResize(cfg, out);
    else if (cfg.mode == "full_touch")
        runFullTouch(cfg, out);
    else
    {
        fmt::print(stderr, "unknown mode: {}\n", cfg.mode);
        std::exit(1);
    }

    out.wall_ns = std::chrono::duration<double, std::nano>(Clk::now() - t0).count();
    rusage ru1 = getThreadUsage();
    out.minor_faults = ru1.ru_minflt - ru0.ru_minflt;
    out.major_faults = ru1.ru_majflt - ru0.ru_majflt;
}

} // namespace

int main(int argc, char ** argv)
{
    const auto cfg_opt = parseCLI(argc, argv);
    if (!cfg_opt)
        return 1;
    const Config & cfg = *cfg_opt;

    fmt::print(
        "bench_alloc_layers  mode={} n={} calls={} T={} reps={} keep={} touch={} thread_status={} block_tracker={} thread_kind={}\n",
        cfg.mode,
        cfg.n,
        cfg.calls,
        cfg.T,
        cfg.reps,
        cfg.keep,
        cfg.touch,
        cfg.thread_status,
        cfg.block_tracker,
        cfg.thread_kind);

    // Make sure GlobalThreadPool has enough capacity for `--thread-kind ch`.
    if (cfg.thread_kind == "ch")
    {
        GlobalThreadPool::initialize(
            /* max_threads = */ static_cast<size_t>(cfg.T) * 2,
            /* max_free_threads = */ static_cast<size_t>(cfg.T),
            /* queue_size = */ static_cast<size_t>(cfg.T) * 4);
    }

    std::vector<double> per_call_ns;
    per_call_ns.reserve(cfg.reps);

    for (int rep = 0; rep < cfg.reps; ++rep)
    {
        std::vector<Result> results(cfg.T);

        const auto wall0 = Clk::now();
        if (cfg.thread_kind == "ch")
        {
            std::vector<ThreadFromGlobalPool> threads;
            threads.reserve(cfg.T);
            for (int t = 0; t < cfg.T; ++t)
                threads.emplace_back([&, t]() { runWorker(cfg, t, results[t]); });
            for (auto & th : threads)
                th.join();
        }
        else
        {
            std::vector<std::thread> threads;
            threads.reserve(cfg.T);
            for (int t = 0; t < cfg.T; ++t)
                threads.emplace_back([&, t]() { runWorker(cfg, t, results[t]); });
            for (auto & th : threads)
                th.join();
        }
        const double wall = std::chrono::duration<double, std::nano>(Clk::now() - wall0).count();

        // Sum/avg per-thread ns/call
        double sum_ns_per_call = 0;
        double sum_wall = 0;
        long sum_minor = 0, sum_major = 0;
        uint64_t sink = 0;
        for (const auto & r : results)
        {
            sum_ns_per_call += r.wall_ns / static_cast<double>(cfg.calls);
            sum_wall += r.wall_ns;
            sum_minor += r.minor_faults;
            sum_major += r.major_faults;
            sink ^= r.sink;
        }
        const double avg_per_call_ns = sum_ns_per_call / cfg.T;
        per_call_ns.push_back(avg_per_call_ns);

        const size_t total_calls = static_cast<size_t>(cfg.T) * cfg.calls;
        const size_t total_bytes = total_calls * cfg.n * sizeof(UInt64);
        const double minor_per_call = static_cast<double>(sum_minor) / static_cast<double>(total_calls);

        fmt::print(
            "  rep {:2d}: wall={:.3f} ms  ns/call(avg-thread)={:.1f}  total_calls={}  "
            "thr={:.2f} M/s/thread  bytes={:.2f} GiB  minor_faults={}  major_faults={}  "
            "minor/call={:.3f}  minor/page={:.3f}\n",
            rep,
            wall / 1e6,
            avg_per_call_ns,
            total_calls,
            static_cast<double>(cfg.calls) / (sum_wall / cfg.T / 1e3) /* M/s = calls / (us per thread) */,
            static_cast<double>(total_bytes) / (1ULL << 30),
            sum_minor,
            sum_major,
            minor_per_call,
            static_cast<double>(sum_minor) / static_cast<double>(total_bytes / 4096));
        // ensure sink isn't dead-code-eliminated
        if (sink == 0xDEADBEEFDEADBEEFULL)
            fmt::print("  (anti-DCE)\n");
    }

    const double pmin = *std::min_element(per_call_ns.begin(), per_call_ns.end());
    const double pavg = std::accumulate(per_call_ns.begin(), per_call_ns.end(), 0.0) / cfg.reps;
    fmt::print("\nSUMMARY mode={} n={} T={}  ns/call: pmin={:.1f}  pavg={:.1f}\n", cfg.mode, cfg.n, cfg.T, pmin, pavg);
    return 0;
}
