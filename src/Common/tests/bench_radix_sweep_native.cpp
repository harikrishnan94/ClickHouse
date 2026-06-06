// bench_radix_sweep_native.cpp
//
// Calibrates the radix hash join SWWC/NT key+ref scatter primitive (DB::RadixShuffle, the focused
// fixed-width column-major key+ref path) under a *realistic* cost model: each repetition allocates a
// FRESH, unfaulted output buffer inside the timed region, so the figure includes allocation and
// first-touch page faults — the per-pass cost the deferred build scatter actually pays (it allocates
// exact-sized per-leaf outputs from the THP arena and first-touches them while scattering). Outputs
// are never reused across reps. This supersedes the earlier "pre-faulted, reused output" harness,
// whose steady-state figure understated the per-pass cost and hid the cost of *more passes*.
//
// Layout is column-major: the key column and the 8 B BuildRef column are scattered into separate
// per-partition arrays (scatterKeyRefTwoColumn), never a fused row cell. Key width is swept to cover
// every kernel path: tiled SWWC (W | 64: 4,8,16,32), byte-stream SWWC (W <= 64 non-divisor), and
// multi-line NT streaming (W a multiple of 64: 64,128).
//
//   Section 1 — P sweep (BITS_PER_PASS recalibration), key W=8, rows=100 M.
//               P in {64,256,1024,2048,4096,8192,16384}, T in {1,16}, swwc vs direct.
//   Section 2 — width sweep (kernel-path generality), P=2048, rows bounded to ~1 GiB key output.
//               W in {4,8,16,32,64,128}, T in {1,16}, swwc vs direct.
//
// "nt/bt" = direct/swwc (> 1 means write-combining + NT still wins after alloc+fault is counted).
//
// NOTE: SWWC exists only when NT stores are available. This build is `x86-64-v3`
// (`ENABLE_MULTITARGET_CODE=1`), so NT is active: `scatterColumn(use_swwc=true)` runs the genuine NT
// path (`vmovntdq`/`vmovntps`) and the "swwc" and "direct" columns differ, with SWWC+NT winning at
// high fanout (P >= 2048). In a `x86-64-v2` build (`ENABLE_MULTITARGET_CODE=0`) NT is dormant, so
// `scatterColumn(use_swwc=true)` falls back to the DIRECT path and the two columns are identical. See
// analysis/radix_hash_p2_analysis.md. (Key widths swept here are all multiples of 4, the supported set.)

#include <pthread.h>
#include <sys/mman.h>
#include <fmt/format.h>
#include <Common/RadixShuffle/Scatter.h>
#include <Common/ThreadPool.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>


namespace
{

using namespace DB;
using namespace DB::RadixShuffle;
using Clk = std::chrono::steady_clock;

constexpr size_t kTotalRows = 100'000'000ULL;
constexpr size_t kKeyBufBytes = static_cast<size_t>(1536) << 20; /// 1.5 GiB scratch input key bytes (reused for any width)
constexpr size_t kKeyBudget = static_cast<size_t>(1) << 30; /// <= 1 GiB key output per width (bounds Section-2 memory)
constexpr UInt32 kMaxBits = 14; /// pid drawn uniformly in [0, 16384); window = pid & (P-1) for any P <= 16384.
constexpr int kReps = 5;
constexpr size_t HUGE_PAGE = static_cast<size_t>(2) << 20; /// 2 MiB

size_t roundUp(size_t x, size_t a)
{
    return (x + a - 1) & ~(a - 1);
}


/// One fresh per-partition output arena: a single mmap'd, 2 MiB-aligned, MADV_HUGEPAGE region carved
/// into exact-sized, 64 B-aligned per-partition bases. `layout` (offsets) is setup; `allocFresh` is
/// run *inside the timed region* (fresh mmap, NOT pre-faulted — the scatter's NT writes first-touch
/// the pages); `reset` munmaps (untimed). Using mmap (not the recycling jemalloc arena) guarantees a
/// genuine first-touch fault every repetition.
struct FreshArena
{
    void * region = nullptr;
    size_t region_bytes = 0;
    char * base_aligned = nullptr;
    size_t usable_bytes = 0;
    std::vector<size_t> off;

    void layout(const std::vector<size_t> & hist, size_t elem_size)
    {
        off.assign(hist.size(), 0);
        size_t o = 0;
        for (size_t p = 0; p < hist.size(); ++p)
        {
            off[p] = o;
            o += roundUpTo64(hist[p] * elem_size);
        }
        usable_bytes = std::max<size_t>(o, 64);
    }

    void allocFresh()
    {
        region_bytes = usable_bytes + HUGE_PAGE; /// slack to align the usable base up to 2 MiB
        region = mmap(nullptr, region_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region == MAP_FAILED)
            throw std::bad_alloc{};
        const uintptr_t a = (reinterpret_cast<uintptr_t>(region) + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1);
        base_aligned = reinterpret_cast<char *>(a);
        /// Request THP for the usable range; fail-open (ignored if the kernel declines).
        madvise(base_aligned, roundUp(usable_bytes, HUGE_PAGE), MADV_HUGEPAGE);
    }

    void reset()
    {
        if (region && region != MAP_FAILED)
            munmap(region, region_bytes);
        region = nullptr;
        base_aligned = nullptr;
    }

    ~FreshArena() { reset(); }
    FreshArena() = default;
    FreshArena(const FreshArena &) = delete;
    FreshArena & operator=(const FreshArena &) = delete;

    char * base(size_t p) const { return base_aligned + off[p]; }
};


void pinThread(int t)
{
    const unsigned n = std::thread::hardware_concurrency();
    if (!n)
        return;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(static_cast<unsigned>(t) % n, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}


/// Per-thread state for one (W, P, variant) measurement. Histogram + scratch + arena layout are setup
/// (reused across reps); the arenas' backing memory is (re)allocated fresh inside each timed rep.
struct ThreadState
{
    std::vector<size_t> hist;
    FreshArena key_arena;
    FreshArena ref_arena;
    std::vector<void *> kbase;
    std::vector<BuildRef *> rbase;
    std::unique_ptr<ScatterScratch> scratch;
};


double measure(
    size_t W,
    int P,
    int T,
    size_t rows_per_thread,
    const std::vector<UInt32> & pid,
    const std::vector<char> & keybuf,
    const std::vector<BuildRef> & refs,
    bool swwc)
{
    const UInt32 shift = 0;
    const UInt32 mask = static_cast<UInt32>(P) - 1;
    const size_t total_rows = static_cast<size_t>(T) * rows_per_thread;

    std::vector<ThreadState> ts(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t)
    {
        auto & st = ts[static_cast<size_t>(t)];
        const size_t off = static_cast<size_t>(t) * rows_per_thread;
        st.hist.assign(static_cast<size_t>(P), 0);
        for (size_t i = 0; i < rows_per_thread; ++i)
            ++st.hist[(static_cast<UInt32>(pid[off + i]) >> shift) & mask];
        st.scratch = std::make_unique<ScatterScratch>(static_cast<size_t>(P));
        st.key_arena.layout(st.hist, W);
        st.ref_arena.layout(st.hist, sizeof(BuildRef));
        st.kbase.resize(static_cast<size_t>(P));
        st.rbase.resize(static_cast<size_t>(P));
    }

    double best = 1e30;
    for (int rep = 0; rep < kReps; ++rep)
    {
        const auto t0 = Clk::now();
        std::vector<ThreadFromGlobalPool> threads;
        threads.reserve(static_cast<size_t>(T));
        for (int t = 0; t < T; ++t)
        {
            threads.emplace_back(
                [&, t]()
                {
                    pinThread(t);
                    auto & st = ts[static_cast<size_t>(t)];
                    const size_t off = static_cast<size_t>(t) * rows_per_thread;

                    /// Allocation + first-touch faults are INSIDE the timed region (fresh every rep).
                    st.key_arena.allocFresh();
                    st.ref_arena.allocFresh();
                    for (int p = 0; p < P; ++p)
                    {
                        st.kbase[static_cast<size_t>(p)] = st.key_arena.base(static_cast<size_t>(p));
                        st.rbase[static_cast<size_t>(p)] = reinterpret_cast<BuildRef *>(st.ref_arena.base(static_cast<size_t>(p)));
                    }

                    scatterKeyRefTwoColumn(
                        pid.data() + off, shift, mask, rows_per_thread, keybuf.data() + off * W, W, refs.data() + off,
                        static_cast<size_t>(P), st.kbase.data(), st.rbase.data(), *st.scratch, swwc);
                });
        }
        for (auto & th : threads)
            th.join();
        const double wall = std::chrono::duration<double>(Clk::now() - t0).count();
        best = std::min(best, wall * 1e9 / static_cast<double>(total_rows));

        for (auto & st : ts)
        {
            st.key_arena.reset(); /// untimed
            st.ref_arena.reset();
        }
    }
    return best;
}

}


int main(int /*argc*/, char ** /*argv*/)
{
    fmt::print("bench_radix_sweep_native (RadixShuffle column-major key+ref SWWC/NT scatter)\n");
    fmt::print("REALISTIC model: fresh output mmap + first-touch faults INSIDE the timed region, per rep\n");
    fmt::print("ns/row/pass (best-of-{}); nt/bt = direct/swwc (>1 means write-combining + NT wins)\n\n", kReps);

    GlobalThreadPool::initialize(64, 32, 128);

    fmt::print("Generating {} rows (pid 14-bit + {} GiB key bytes + 8 B ref)...\n", kTotalRows, kKeyBufBytes >> 30);
    const auto tg0 = Clk::now();
    std::vector<UInt32> pid(kTotalRows);
    std::vector<char> keybuf(kKeyBufBytes);
    std::vector<BuildRef> refs(kTotalRows);
    {
        std::mt19937_64 rng(0x9E3779B9ULL); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed) -- deterministic bench input
        const UInt32 pid_mask = (1u << kMaxBits) - 1;
        for (size_t i = 0; i < kTotalRows; ++i)
        {
            const UInt64 k = rng();
            pid[i] = static_cast<UInt32>(k & pid_mask); /// uniform in [0, 16384)
            refs[i] = BuildRef{0, static_cast<UInt32>(i)};
        }
        /// Key content is irrelevant to throughput; fill the reusable key buffer with a cheap pattern.
        for (size_t i = 0; i < kKeyBufBytes; i += sizeof(UInt64))
        {
            const UInt64 v = rng();
            std::memcpy(keybuf.data() + i, &v, std::min(sizeof(UInt64), kKeyBufBytes - i));
        }
    }
    fmt::print("  {:.2f} s\n\n", std::chrono::duration<double>(Clk::now() - tg0).count());

    /// ---- Section 1: P sweep (BITS_PER_PASS recalibration), key W=8, rows=100 M ----
    fmt::print("== Section 1: P sweep (key W=8, ref 8 B, rows={} M) ==\n", kTotalRows / 1'000'000);
    fmt::print("{:>6}  {:>3}  {:>9}  {:>9}  {:>9}\n", "P", "T", "swwc", "direct", "nt/bt");
    fmt::print("{}\n", std::string(44, '-'));
    for (const int partitions : {64, 256, 1024, 2048, 4096, 8192, 16384})
    {
        for (const int threads : {1, 16})
        {
            const size_t rpt = kTotalRows / static_cast<size_t>(threads);
            const double sw = measure(8, partitions, threads, rpt, pid, keybuf, refs, /*swwc=*/true);
            const double di = measure(8, partitions, threads, rpt, pid, keybuf, refs, /*swwc=*/false);
            fmt::print("{:>6}  {:>3}  {:>9.3f}  {:>9.3f}  {:>8.2f}x\n", partitions, threads, sw, di, di / sw);
            fmt::print("RESULT sec=1 W=8 P={} T={} swwc={:.3f} direct={:.3f}\n", partitions, threads, sw, di);
        }
    }

    /// ---- Section 2: width sweep (kernel-path generality), P=2048 ----
    fmt::print("\n== Section 2: width sweep (P=2048, rows bounded to ~{} GiB key output) ==\n", kKeyBudget >> 30);
    fmt::print("{:>4}  {:>9}  {:>3}  {:>9}  {:>9}  {:>9}\n", "W", "rows(M)", "T", "swwc", "direct", "nt/bt");
    fmt::print("{}\n", std::string(52, '-'));
    constexpr int p2 = 2048;
    for (const size_t width : {size_t{4}, size_t{8}, size_t{16}, size_t{32}, size_t{64}, size_t{128}})
    {
        const size_t rows = std::min(kTotalRows, kKeyBudget / width);
        for (const int threads : {1, 16})
        {
            const size_t rpt = rows / static_cast<size_t>(threads);
            const double sw = measure(width, p2, threads, rpt, pid, keybuf, refs, /*swwc=*/true);
            const double di = measure(width, p2, threads, rpt, pid, keybuf, refs, /*swwc=*/false);
            fmt::print(
                "{:>4}  {:>9.1f}  {:>3}  {:>9.3f}  {:>9.3f}  {:>8.2f}x\n",
                width, static_cast<double>(rpt * static_cast<size_t>(threads)) / 1e6, threads, sw, di, di / sw);
            fmt::print("RESULT sec=2 W={} P={} T={} swwc={:.3f} direct={:.3f}\n", width, p2, threads, sw, di);
        }
    }

    fmt::print("\n");
    return 0;
}
