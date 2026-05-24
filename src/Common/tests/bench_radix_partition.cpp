// bench_radix_partition.cpp
//
// Benchmark: plain memcpy vs. single-pass radix partitioning.
// Both functions receive the same K-column uint64 table delivered in
// fixed-size row blocks.  Measures ns/row and GB/s (read + write).
//
// Ported from radix_part_vs_memcpy.cpp; the radix implementation lives in
// src/Common/RadixShuffle/ so it shares all logic with production code.
//
// Usage:
//   bench_radix_partition [OPTIONS]
//   --partitions P   power-of-2 in [1,32768]   (default 64)
//   --columns    K   uint64 columns per row     (default  4, max 8)
//   --rows       N   total rows                 (default  100 000 000)
//   --block-rows B   rows per input block       (default  16 384)
//   --threads    T   worker threads             (default  16)
//   --reps       R   timed repetitions          (default   5)

#include <Common/RadixShuffle/BumpArena.h>
#include <Common/RadixShuffle/NumericScatterColumn.h>
#include <Common/RadixShuffle/OutBlock.h>
#include <Common/RadixShuffle/RadixPartitionOperator.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>


namespace
{

using namespace DB::RadixShuffle;

using Clk = std::chrono::steady_clock;
using InputBlockU64 = InputBlock<uint64_t>;

static constexpr int kMaxKArg = kMaxK;
static constexpr int kMaxP = 32768;


// ── input ─────────────────────────────────────────────────────────────────────

static uint64_t * alloc64(size_t n)
{
    void * p = nullptr;
    if (posix_memalign(&p, 64, n * sizeof(uint64_t)) != 0)
        std::abort();
    return static_cast<uint64_t *>(p);
}


static std::vector<InputBlockU64> genBlocks(size_t total, size_t block_rows, int K, uint64_t seed)
{
    std::vector<InputBlockU64> out;
    std::mt19937_64 rng(seed);
    for (size_t done = 0; done < total;)
    {
        const size_t bs = std::min(block_rows, total - done);
        InputBlockU64 b;
        b.rows = bs;
        for (int k = 0; k < K; ++k)
        {
            b.cols[k] = alloc64(bs);
            for (size_t i = 0; i < bs; ++i)
                b.cols[k][i] = rng();
        }
        out.push_back(b);
        done += bs;
    }
    return out;
}


static void freeBlocks(std::vector<InputBlockU64> & v, int K)
{
    for (auto & b : v)
        for (int k = 0; k < K; ++k)
            std::free(b.cols[k]);
    v.clear();
}


// ── memcpy baseline ───────────────────────────────────────────────────────────
// Uses the same OutBlock / PartState / growPart machinery as the radix variant
// but writes all rows into a single partition (no hashing).  Allocation
// pattern, block sizes, and arena usage therefore mirror the radix variant.

static void runMemcpy(
    const std::vector<InputBlockU64> & blocks,
    int K,
    PartState & out,
    BumpArena & arena,
    size_t init_cap = kOutCapMin,
    size_t max_cap = kOutCapMax)
{
    out.next_cap = init_cap;
    for (const auto & blk : blocks)
    {
        size_t i = 0;
        while (i < blk.rows)
        {
            if (!out.cur || out.cur->filled >= out.cur->capacity)
                growPart(out, arena, K, sizeof(uint64_t), max_cap);
            const size_t n = std::min(out.cur->capacity - out.cur->filled, blk.rows - i);
            const size_t f = out.cur->filled;
            for (int k = 0; k < K; ++k)
                std::memcpy(
                    static_cast<uint64_t *>(out.cur->cols[k]) + f, blk.cols[k] + i, n * 8);
            out.cur->filled = f + n;
            i += n;
        }
    }
}


// ── radix variant ─────────────────────────────────────────────────────────────

static void runSmartRadix(
    const std::vector<InputBlockU64> & blocks,
    int K,
    int P,
    std::vector<PartState> & parts,
    BumpArena & arena,
    size_t init_cap = kOutCapMin,
    size_t max_cap = kOutCapMax)
{
    std::vector<std::unique_ptr<NumericScatterColumn<uint64_t>>> owned;
    std::vector<IScatterColumn *> ptrs;
    for (int k = 0; k < K; ++k)
    {
        owned.push_back(std::make_unique<NumericScatterColumn<uint64_t>>(static_cast<size_t>(P)));
        ptrs.push_back(owned.back().get());
    }
    const bool use_swwc = RadixPartitionOperator<uint64_t>::should_use_swwc(K, P);
    RadixPartitionOperator<uint64_t> op(P, K, std::move(ptrs), arena, use_swwc, init_cap, max_cap);
    op.process(blocks);
    parts = std::move(op.parts());
}


// ── thread pinning ────────────────────────────────────────────────────────────

static void pinThread(int t)
{
    const unsigned n = std::thread::hardware_concurrency();
    if (!n)
        return;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(static_cast<unsigned>(t) % n, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}


// ── statistics ────────────────────────────────────────────────────────────────

struct Stats
{
    double mean, stddev, p50, pmin, pmax;
};

static Stats computeStats(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v)
        sum += x;
    const double m = sum / static_cast<double>(v.size());
    double var = 0;
    for (double x : v)
        var += (x - m) * (x - m);
    return {
        m,
        v.size() > 1 ? std::sqrt(var / static_cast<double>(v.size() - 1)) : 0.0,
        v[v.size() / 2],
        v.front(),
        v.back()};
}

} // namespace


int main(int argc, char ** argv)
{
    int P = 64;
    int K = 4;
    size_t N = 100'000'000ULL;
    size_t B = 16384;
    int T = 16;
    int R = 5;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--partitions" && i + 1 < argc)
            P = std::stoi(argv[++i]);
        else if (a == "--columns" && i + 1 < argc)
            K = std::stoi(argv[++i]);
        else if (a == "--rows" && i + 1 < argc)
            N = std::stoull(argv[++i]);
        else if (a == "--block-rows" && i + 1 < argc)
            B = std::stoull(argv[++i]);
        else if (a == "--threads" && i + 1 < argc)
            T = std::stoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc)
            R = std::stoi(argv[++i]);
        else
        {
            fmt::print(stderr, "unknown arg: {}\n", argv[i]);
            return 1;
        }
    }

    if (P < 1 || P > kMaxP || (P & (P - 1)))
    {
        fmt::print(stderr, "P must be power-of-2 in [1,{}]\n", kMaxP);
        return 1;
    }
    if (K < 1 || K > kMaxKArg)
    {
        fmt::print(stderr, "K must be in [1,{}]\n", kMaxKArg);
        return 1;
    }
    if (T < 1 || R < 1 || B < 8u)
    {
        fmt::print(stderr, "T,R >= 1  B >= 8\n");
        return 1;
    }

    const int batch = std::max(1024, std::min(RadixPartitionOperator<uint64_t>::kSmartMaxBatch, P * RadixPartitionOperator<uint64_t>::kBatchFactor));
    const size_t rpt = (N + static_cast<size_t>(T) - 1) / static_cast<size_t>(T);
    const size_t total = rpt * static_cast<size_t>(T);

    const auto cap_pair = adaptiveCaps(rpt, static_cast<size_t>(P));
    const size_t cap_init = cap_pair.first;
    const size_t cap_max = cap_pair.second;

    fmt::print("bench_radix_partition\n");
    fmt::print("  partitions={:<6}  columns={:<2}  rows={:<12}\n", P, K, N);
    fmt::print("  block-rows={:<6}  threads={:<3}  reps={}\n", B, T, R);
    fmt::print("  rows/thread={}  total={}\n", rpt, total);
    fmt::print(
        "  data/thread = {:.1f} MiB  ({} cols \xc3\x97 {} rows \xc3\x97 8 B)\n",
        static_cast<double>(K) * static_cast<double>(rpt) * 8.0 / (1 << 20),
        K,
        rpt);
    fmt::print("  batch_size  = {}\n", batch);
    fmt::print(
        "  OutBlock cap: init={}  max={}  (avg rows/part\xe2\x89\x88{})\n",
        cap_init,
        cap_max,
        rpt / static_cast<size_t>(P));
    fmt::print(
        "  radix mode  = {}\n",
        RadixPartitionOperator<uint64_t>::should_use_swwc(K, P) ? "SWWC (NT stores)" : "direct");
    if (B < static_cast<size_t>(batch))
        fmt::print("  [warn] block-rows {} < batch_size {} -> scalar path only\n", B, batch);
    fmt::print("\n");

    // ── generate input streams ────────────────────────────────────────────────
    fmt::print("Generating {} streams \xc3\x97 {} blocks each...\n", T, (rpt + B - 1) / B);
    const auto tg0 = Clk::now();
    std::vector<std::vector<InputBlockU64>> streams(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t)
        streams[static_cast<size_t>(t)] = genBlocks(rpt, B, K, 42ULL + static_cast<uint64_t>(t));
    fmt::print("  {:.2f} s\n\n", std::chrono::duration<double>(Clk::now() - tg0).count());

    // ── arenas + output state — reset() between reps for warm-page reuse ─────
    // 64 MiB initial slab: at most a handful of allocations in rep 0.
    // From rep 1 onward reset() rewinds to the same warm physical pages so
    // both variants pay identical allocation and page-fault cost.
    std::vector<BumpArena> mc_arenas;
    std::vector<BumpArena> rd_arenas;
    mc_arenas.reserve(static_cast<size_t>(T));
    rd_arenas.reserve(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t)
    {
        mc_arenas.emplace_back(64ULL << 20);
        rd_arenas.emplace_back(64ULL << 20);
    }

    std::vector<PartState> mc_parts(static_cast<size_t>(T));
    std::vector<std::vector<PartState>> parts(static_cast<size_t>(T));

    // GB/s = gbs_k / ns_per_row  (factor of 2 for read + write)
    const double gbs_k = 2.0 * static_cast<double>(K) * 8.0;

    std::vector<double> mc_ns(static_cast<size_t>(R));
    std::vector<double> rd_ns(static_cast<size_t>(R));

    // ── benchmark ─────────────────────────────────────────────────────────────
    fmt::print("{:<4}  {:>12}  {:>12}  {:>6}\n", "rep", "memcpy ns/row", "radix ns/row", "ratio");
    fmt::print("----  ------------  ------------  ------\n");

    for (int rep = 0; rep < R; ++rep)
    {
        // ── memcpy ────────────────────────────────────────────────────────────
        {
            for (int t = 0; t < T; ++t)
            {
                mc_parts[static_cast<size_t>(t)] = {};
                mc_arenas[static_cast<size_t>(t)].reset();
            }
            const auto t0 = Clk::now();
            std::vector<std::thread> ths;
            ths.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
                ths.emplace_back(
                    [&, t]()
                    {
                        pinThread(t);
                        runMemcpy(
                            streams[static_cast<size_t>(t)],
                            K,
                            mc_parts[static_cast<size_t>(t)],
                            mc_arenas[static_cast<size_t>(t)],
                            cap_init,
                            cap_max);
                    });
            for (auto & th : ths)
                th.join();
            mc_ns[static_cast<size_t>(rep)]
                = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9
                / static_cast<double>(total);
        }

        // ── radix ─────────────────────────────────────────────────────────────
        {
            for (int t = 0; t < T; ++t)
            {
                parts[static_cast<size_t>(t)].clear();
                rd_arenas[static_cast<size_t>(t)].reset();
            }
            const auto t0 = Clk::now();
            std::vector<std::thread> ths;
            ths.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
                ths.emplace_back(
                    [&, t]()
                    {
                        pinThread(t);
                        runSmartRadix(
                            streams[static_cast<size_t>(t)],
                            K,
                            P,
                            parts[static_cast<size_t>(t)],
                            rd_arenas[static_cast<size_t>(t)],
                            cap_init,
                            cap_max);
                    });
            for (auto & th : ths)
                th.join();
            rd_ns[static_cast<size_t>(rep)]
                = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9
                / static_cast<double>(total);
        }

        fmt::print(
            "{:<4}  {:>12.3f}  {:>12.3f}  {:>5.2f}x\n",
            rep,
            mc_ns[static_cast<size_t>(rep)],
            rd_ns[static_cast<size_t>(rep)],
            rd_ns[static_cast<size_t>(rep)] / mc_ns[static_cast<size_t>(rep)]);
        (void)std::fflush(stdout);
    }

    // ── arena sanity check ────────────────────────────────────────────────────
    const size_t expected_data = rpt * static_cast<size_t>(K) * 8;
    fmt::print("\nArena usage after last rep (thread 0):\n");
    fmt::print(
        "  expected data  = {:>7.1f} MiB  ({} rows \xc3\x97 {} cols \xc3\x97 8 B)\n",
        static_cast<double>(expected_data) / 1048576.0,
        rpt,
        K);
    fmt::print(
        "  memcpy  used   = {:>7.1f} MiB  alloc = {:.1f} MiB\n",
        static_cast<double>(mc_arenas[0].usedBytes()) / 1048576.0,
        static_cast<double>(mc_arenas[0].allocatedBytes()) / 1048576.0);
    fmt::print(
        "  radix   used   = {:>7.1f} MiB  alloc = {:.1f} MiB\n",
        static_cast<double>(rd_arenas[0].usedBytes()) / 1048576.0,
        static_cast<double>(rd_arenas[0].allocatedBytes()) / 1048576.0);
    fmt::print("  (used > expected by OutBlock headers and partial-block waste)\n");

    // ── summary ───────────────────────────────────────────────────────────────
    const auto mc = computeStats(mc_ns);
    const auto rd = computeStats(rd_ns);

    fmt::print("\nSummary (agg = wall_ns/total_rows;  per-thr = agg\xc3\x97T;  GB/s = R+W)\n");
    fmt::print(
        "{:<8}  {:>8}  {:>8}  {:>8}  {:>5}  {:>8}  {:>8}\n",
        "variant",
        "agg-min",
        "agg-p50",
        "agg-mean",
        "cv%",
        "GB/s",
        "per-thr");
    fmt::print("--------  --------  --------  --------  -----  --------  --------\n");

    auto print_row = [&](const char * label, const Stats & s)
    {
        fmt::print(
            "{:<8}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>4.1f}%  {:>8.1f}  {:>8.1f}\n",
            label,
            s.pmin,
            s.p50,
            s.mean,
            s.mean > 0 ? 100.0 * s.stddev / s.mean : 0.0,
            gbs_k / s.pmin,
            s.pmin * static_cast<double>(T));
    };
    print_row("memcpy", mc);
    print_row("radix", rd);

    fmt::print(
        "\nOverhead (radix / memcpy):  best={:.3f}x  mean={:.3f}x"
        "  (per-thread: {:.1f} vs {:.1f} ns/row)\n",
        rd.pmin / mc.pmin,
        rd.mean / mc.mean,
        rd.pmin * static_cast<double>(T),
        mc.pmin * static_cast<double>(T));

    // ── cleanup ───────────────────────────────────────────────────────────────
    for (int t = 0; t < T; ++t)
        freeBlocks(streams[static_cast<size_t>(t)], K);
    return 0;
}
