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

#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnVector.h>
#include <Columns/IColumn_fwd.h>
#include <Common/RadixShuffle/BumpArena.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>
#include <Common/RadixShuffle/OutBlock.h>
#include <Common/RadixShuffle/RadixPartitionOperator.h>
#include <Common/assert_cast.h>
#include <base/Decimal.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <pthread.h>

#include <fmt/format.h>


namespace
{

using namespace DB::RadixShuffle;

using Clk = std::chrono::steady_clock;
using BlockStream = std::vector<DB::Columns>;

constexpr int kMaxKArg = kMaxK;
constexpr int kMaxP = 32768;
constexpr size_t kArenaSlabBytes = 64ULL << 20;


struct BenchConfig
{
    int partitions = 64;
    int columns = 4;
    size_t rows = 100'000'000ULL;
    size_t block_rows = 16'384;
    int threads = 16;
    int reps = 5;
};


struct Stats
{
    double mean = 0.0;
    double stddev = 0.0;
    double p50 = 0.0;
    double pmin = 0.0;
    double pmax = 0.0;
};


// ── input ─────────────────────────────────────────────────────────────────────

BlockStream genBlocks(size_t total, size_t block_rows, int K, uint64_t seed)
{
    BlockStream stream;
    std::mt19937_64 rng(seed);
    for (size_t done = 0; done < total;)
    {
        const size_t bs = std::min(block_rows, total - done);
        DB::Columns block;
        for (int k = 0; k < K; ++k)
        {
            auto col = DB::ColumnVector<UInt64>::create();
            col->getData().resize(bs);
            for (size_t i = 0; i < bs; ++i)
                col->getData()[i] = rng();
            block.push_back(std::move(col));
        }
        stream.push_back(std::move(block));
        done += bs;
    }
    return stream;
}


// ── typed input generators ────────────────────────────────────────────────────

/// Decimal64 blocks: K columns of ColumnDecimal<Decimal64>, 8 bytes/row each.
BlockStream genBlocksDecimal64(size_t total, size_t block_rows, int K, uint64_t seed)
{
    BlockStream stream;
    std::mt19937_64 rng(seed);
    for (size_t done = 0; done < total;)
    {
        const size_t bs = std::min(block_rows, total - done);
        DB::Columns block;
        for (int k = 0; k < K; ++k)
        {
            auto col = DB::ColumnDecimal<DB::Decimal64>::create(0, 0);
            col->getData().resize(bs);
            for (size_t i = 0; i < bs; ++i)
                col->getData()[i] = DB::Decimal64{static_cast<Int64>(rng())};
            block.push_back(std::move(col));
        }
        stream.push_back(std::move(block));
        done += bs;
    }
    return stream;
}

/// Decimal32 blocks: K columns of ColumnDecimal<Decimal32>, 4 bytes/row each.
BlockStream genBlocksDecimal32(size_t total, size_t block_rows, int K, uint64_t seed)
{
    BlockStream stream;
    std::mt19937_64 rng(seed);
    for (size_t done = 0; done < total;)
    {
        const size_t bs = std::min(block_rows, total - done);
        DB::Columns block;
        for (int k = 0; k < K; ++k)
        {
            auto col = DB::ColumnDecimal<DB::Decimal32>::create(0, 0);
            col->getData().resize(bs);
            for (size_t i = 0; i < bs; ++i)
                col->getData()[i] = DB::Decimal32{static_cast<Int32>(rng())};
            block.push_back(std::move(col));
        }
        stream.push_back(std::move(block));
        done += bs;
    }
    return stream;
}

/// FixedString(n) blocks: K columns of ColumnFixedString(n), n bytes/row each.
BlockStream genBlocksFixedStr(size_t total, size_t block_rows, int K, size_t n, uint64_t seed)
{
    BlockStream stream;
    std::mt19937_64 rng(seed);
    for (size_t done = 0; done < total;)
    {
        const size_t bs = std::min(block_rows, total - done);
        DB::Columns block;
        for (int k = 0; k < K; ++k)
        {
            auto col = DB::ColumnFixedString::create(n);
            col->getChars().resize(bs * n);
            for (size_t i = 0; i < bs; ++i)
            {
                uint64_t v = rng();
                for (size_t b = 0; b < n; ++b)
                    col->getChars()[i * n + b] = static_cast<uint8_t>(v >> (b * 8));
            }
            block.push_back(std::move(col));
        }
        stream.push_back(std::move(block));
        done += bs;
    }
    return stream;
}


/// Generic typed radix runner.  `prims_proto` is copied K times.
/// `elem_size` is sizeof(T) for the payload element — used for OutBlock sizing.
template <typename TKey>
void runTypedRadix(
    const BlockStream & blocks,
    int K,
    int P,
    std::vector<PartState> & parts,
    BumpArena & arena,
    const ColumnPrimitives & prim_proto,
    size_t init_cap = kOutCapMin,
    size_t max_cap = kOutCapMax)
{
    std::vector<ColumnPrimitives> prims(static_cast<size_t>(K), prim_proto);
    const bool use_swwc = RadixPartitionOperator<TKey>::should_use_swwc(K, P);
    RadixPartitionOperator<TKey> op(P, K, std::move(prims), arena, use_swwc, init_cap, max_cap);
    for (const auto & block : blocks)
        op.process(block);
    op.finish();
    parts = std::move(op.parts());
}


// ── memcpy baseline ───────────────────────────────────────────────────────────
// Uses the same OutBlock / PartState / growPart machinery as the radix variant
// but writes all rows into a single partition (no hashing).  Allocation
// pattern, block sizes, and arena usage therefore mirror the radix variant.

void runMemcpy(
    const BlockStream & blocks, int K, PartState & out, BumpArena & arena, size_t init_cap = kOutCapMin, size_t max_cap = kOutCapMax)
{
    out.next_cap = init_cap;
    for (const auto & blk : blocks)
    {
        const size_t blk_rows = blk[0]->size();
        size_t i = 0;
        while (i < blk_rows)
        {
            if (!out.cur || out.cur->filled >= out.cur->capacity)
                growPart(out, arena, K, sizeof(uint64_t), max_cap);
            const size_t n = std::min(out.cur->capacity - out.cur->filled, blk_rows - i);
            const size_t f = out.cur->filled;
            for (int k = 0; k < K; ++k)
            {
                const auto * src = assert_cast<const DB::ColumnVector<UInt64> &>(*blk[static_cast<size_t>(k)]).getData().data();
                std::memcpy(static_cast<uint64_t *>(out.cur->cols[k]) + f, src + i, n * 8);
            }
            out.cur->filled = f + n;
            i += n;
        }
    }
}


// ── radix variant ─────────────────────────────────────────────────────────────

void runSmartRadix(
    const BlockStream & blocks,
    int K,
    int P,
    std::vector<PartState> & parts,
    BumpArena & arena,
    size_t init_cap = kOutCapMin,
    size_t max_cap = kOutCapMax)
{
    std::vector<ColumnPrimitives> prims(static_cast<size_t>(K), makeFixedWidth<UInt64>());
    const bool use_swwc = RadixPartitionOperator<uint64_t>::should_use_swwc(K, P);
    RadixPartitionOperator<uint64_t> op(P, K, std::move(prims), arena, use_swwc, init_cap, max_cap);
    for (const auto & block : blocks)
        op.process(block);
    op.finish();
    parts = std::move(op.parts());
}


// ── thread pinning ────────────────────────────────────────────────────────────

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


// ── statistics ────────────────────────────────────────────────────────────────

Stats computeStats(std::vector<double> values)
{
    std::ranges::sort(values);
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const double mean = sum / static_cast<double>(values.size());
    double var = 0.0;
    for (const double x : values)
        var += (x - mean) * (x - mean);
    return Stats{
        .mean = mean,
        .stddev = values.size() > 1 ? std::sqrt(var / static_cast<double>(values.size() - 1)) : 0.0,
        .p50 = values[values.size() / 2],
        .pmin = values.front(),
        .pmax = values.back(),
    };
}


std::optional<BenchConfig> parseCLI(std::span<char * const> args)
{
    BenchConfig cfg;
    for (size_t i = 0; i < args.size(); ++i)
    {
        const std::string arg = args[i];
        if (arg == "--partitions" && i + 1 < args.size())
        {
            cfg.partitions = std::stoi(args[++i]);
        }
        else if (arg == "--columns" && i + 1 < args.size())
        {
            cfg.columns = std::stoi(args[++i]);
        }
        else if (arg == "--rows" && i + 1 < args.size())
        {
            cfg.rows = std::stoull(args[++i]);
        }
        else if (arg == "--block-rows" && i + 1 < args.size())
        {
            cfg.block_rows = std::stoull(args[++i]);
        }
        else if (arg == "--threads" && i + 1 < args.size())
        {
            cfg.threads = std::stoi(args[++i]);
        }
        else if (arg == "--reps" && i + 1 < args.size())
        {
            cfg.reps = std::stoi(args[++i]);
        }
        else
        {
            fmt::print(stderr, "unknown arg: {}\n", args[i]);
            return std::nullopt;
        }
    }
    return cfg;
}


bool validateConfig(const BenchConfig & cfg)
{
    if (cfg.partitions < 1 || cfg.partitions > kMaxP || !std::has_single_bit(static_cast<unsigned>(cfg.partitions)))
    {
        fmt::print(stderr, "P must be power-of-2 in [1,{}]\n", kMaxP);
        return false;
    }
    if (cfg.columns < 1 || cfg.columns > kMaxKArg)
    {
        fmt::print(stderr, "K must be in [1,{}]\n", kMaxKArg);
        return false;
    }
    if (cfg.threads < 1 || cfg.reps < 1 || cfg.block_rows < 8u)
    {
        fmt::print(stderr, "T,R >= 1  B >= 8\n");
        return false;
    }
    return true;
}

} // namespace


int main(int argc, char ** argv)
{
    const auto cfg_opt = parseCLI({argv + 1, static_cast<size_t>(argc - 1)});
    if (!cfg_opt)
        return 1;
    const BenchConfig & cfg = *cfg_opt;
    if (!validateConfig(cfg))
        return 1;

    const int P = cfg.partitions;
    const int K = cfg.columns;
    const size_t N = cfg.rows;
    const size_t B = cfg.block_rows;
    const int T = cfg.threads;
    const int R = cfg.reps;

    const int batch
        = std::max(1024, std::min(RadixPartitionOperator<uint64_t>::kSmartMaxBatch, P * RadixPartitionOperator<uint64_t>::kBatchFactor));
    const size_t rpt = (N + static_cast<size_t>(T) - 1) / static_cast<size_t>(T);
    const size_t total = rpt * static_cast<size_t>(T);

    const auto [cap_init, cap_max] = adaptiveCaps(rpt, static_cast<size_t>(P));

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
    fmt::print("  OutBlock cap: init={}  max={}  (avg rows/part\xe2\x89\x88{})\n", cap_init, cap_max, rpt / static_cast<size_t>(P));
    fmt::print("  radix mode  = {}\n", RadixPartitionOperator<uint64_t>::should_use_swwc(K, P) ? "SWWC (NT stores)" : "direct");
    if (B < static_cast<size_t>(batch))
        fmt::print("  [warn] block-rows {} < batch_size {} -> scalar path only\n", B, batch);
    fmt::print("\n");

    // ── generate input streams ────────────────────────────────────────────────
    fmt::print("Generating {} streams \xc3\x97 {} blocks each...\n", T, (rpt + B - 1) / B);
    const auto tg0 = Clk::now();
    std::vector<BlockStream> streams(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t)
        streams[static_cast<size_t>(t)] = genBlocks(rpt, B, K, 42ULL + static_cast<uint64_t>(t));
    fmt::print("  {:.2f} s\n\n", std::chrono::duration<double>(Clk::now() - tg0).count());

    // ── fresh arenas allocated every rep — no reuse between reps ─────────────
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
        // ── memcpy — fresh BumpArena each rep ────────────────────────────────
        {
            std::vector<BumpArena> mc_arenas;
            mc_arenas.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
                mc_arenas.emplace_back(kArenaSlabBytes);

            for (int t = 0; t < T; ++t)
                mc_parts[static_cast<size_t>(t)] = {};
            const auto t0 = Clk::now();
            std::vector<std::thread> ths;
            ths.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
            {
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
            }
            for (auto & th : ths)
                th.join();
            mc_ns[static_cast<size_t>(rep)] = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9 / static_cast<double>(total);
        }

        // ── radix — fresh BumpArena each rep ─────────────────────────────────
        {
            std::vector<BumpArena> rd_arenas;
            rd_arenas.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
                rd_arenas.emplace_back(kArenaSlabBytes);

            for (int t = 0; t < T; ++t)
                parts[static_cast<size_t>(t)].clear();
            const auto t0 = Clk::now();
            std::vector<std::thread> ths;
            ths.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
            {
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
            }
            for (auto & th : ths)
                th.join();
            rd_ns[static_cast<size_t>(rep)] = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9 / static_cast<double>(total);
        }

        fmt::print(
            "{:<4}  {:>12.3f}  {:>12.3f}  {:>5.2f}x\n",
            rep,
            mc_ns[static_cast<size_t>(rep)],
            rd_ns[static_cast<size_t>(rep)],
            rd_ns[static_cast<size_t>(rep)] / mc_ns[static_cast<size_t>(rep)]);
        (void)std::fflush(stdout);
    }

    // ── arena sanity check (last rep, from parts chain) ──────────────────────
    const size_t expected_data = rpt * static_cast<size_t>(K) * 8;
    size_t radix_used_rows = 0;
    for (const auto & sv : parts[0])
    {
        for (const OutBlock * b = sv.head; b != nullptr; b = b->next)
            radix_used_rows += b->filled;
    }
    const size_t radix_used_bytes = radix_used_rows * static_cast<size_t>(K) * sizeof(uint64_t);
    fmt::print("\nArena usage after last rep (thread 0):\n");
    fmt::print(
        "  expected data  = {:>7.1f} MiB  ({} rows \xc3\x97 {} cols \xc3\x97 8 B)\n",
        static_cast<double>(expected_data) / 1048576.0,
        rpt,
        K);
    fmt::print(
        "  radix   rows   = {:>7.1f} MiB  ({} rows scattered)\n", static_cast<double>(radix_used_bytes) / 1048576.0, radix_used_rows);

    // ── summary ───────────────────────────────────────────────────────────────
    const auto mc = computeStats(std::move(mc_ns));
    const auto rd = computeStats(std::move(rd_ns));

    fmt::print("\nSummary (agg = wall_ns/total_rows;  per-thr = agg\xc3\x97T;  GB/s = R+W)\n");
    fmt::print("{:<8}  {:>8}  {:>8}  {:>8}  {:>5}  {:>8}  {:>8}\n", "variant", "agg-min", "agg-p50", "agg-mean", "cv%", "GB/s", "per-thr");
    fmt::print("--------  --------  --------  --------  -----  --------  --------\n");

    const auto print_row = [&](const char * label, const Stats & s)
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

    // ── Type-sweep: same P/K/N, compare column types at equal bytes-per-row ──
    // Reference: UInt64 (already benchmarked above as `radix`).
    // Each variant uses ColumnPrimitives dispatch; TKey selects OutBlock elem size.
    //
    //  Type          elem  total B/row (K=4)
    //  -----------   ----  -----------------
    //  uint64           8           32  (reference)
    //  decimal64        8           32  (same bytes/row)
    //  decimal32        4           16  (half bytes/row)
    //  fixedstr8        8           32  (same bytes/row)
    fmt::print("\nType-sweep (P={} K={} T={} R={}):\n", P, K, T, R);
    fmt::print(
        "{:<12}  {:>6}  {:>7}  {:>8}  {:>8}\n",
        "type", "B/elem", "B/row", "ns/row", "GB/s(R+W)");
    fmt::print("------------  ------  -------  --------  --------\n");

    // Helper: run one typed variant and print a row.
    auto run_typed_variant = [&](const char * label,
                                 size_t elem_size,
                                 auto gen_fn,
                                 auto run_fn)
    {
        // Generate streams for this type.
        std::vector<BlockStream> typed_streams(static_cast<size_t>(T));
        for (int t = 0; t < T; ++t)
            typed_streams[static_cast<size_t>(t)] = gen_fn(rpt, B, K, 42ULL + static_cast<uint64_t>(t) + 1000ULL);

        std::vector<double> typed_ns(static_cast<size_t>(R));
        std::vector<std::vector<PartState>> typed_parts(static_cast<size_t>(T));
        const double bytes_per_row = static_cast<double>(K) * static_cast<double>(elem_size);
        const double gbs_typed = 2.0 * bytes_per_row;

        for (int rep = 0; rep < R; ++rep)
        {
            std::vector<BumpArena> typed_arenas;
            typed_arenas.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
                typed_arenas.emplace_back(kArenaSlabBytes);
            for (int t = 0; t < T; ++t)
                typed_parts[static_cast<size_t>(t)].clear();

            const auto t0 = Clk::now();
            std::vector<std::thread> ths;
            ths.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
            {
                ths.emplace_back([&, t]()
                {
                    pinThread(t);
                    run_fn(
                        typed_streams[static_cast<size_t>(t)],
                        K, P,
                        typed_parts[static_cast<size_t>(t)],
                        typed_arenas[static_cast<size_t>(t)],
                        cap_init, cap_max);
                });
            }
            for (auto & th : ths)
                th.join();
            typed_ns[static_cast<size_t>(rep)]
                = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9
                  / static_cast<double>(total);
        }
        const auto s = computeStats(std::move(typed_ns));
        fmt::print(
            "{:<12}  {:>6}  {:>7}  {:>8.3f}  {:>8.1f}\n",
            label,
            elem_size,
            K * elem_size,
            s.pmin,
            gbs_typed / s.pmin);
    };

    // UInt64 reference (re-run, fresh streams).
    run_typed_variant(
        "uint64", 8,
        [](size_t tot, size_t br, int k, uint64_t seed) { return genBlocks(tot, br, k, seed); },
        [](const BlockStream & blk, int k, int p, std::vector<PartState> & pts, BumpArena & ar, size_t ic, size_t mc_)
        { runTypedRadix<uint64_t>(blk, k, p, pts, ar, makeFixedWidth<UInt64>(), ic, mc_); });

    // Decimal64 (8 bytes/row — same as UInt64).
    run_typed_variant(
        "decimal64", 8,
        [](size_t tot, size_t br, int k, uint64_t seed) { return genBlocksDecimal64(tot, br, k, seed); },
        [](const BlockStream & blk, int k, int p, std::vector<PartState> & pts, BumpArena & ar, size_t ic, size_t mc_)
        { runTypedRadix<uint64_t>(blk, k, p, pts, ar, makeDecimal<DB::Decimal64>(), ic, mc_); });

    // Decimal32 (4 bytes/row — half of UInt64).
    run_typed_variant(
        "decimal32", 4,
        [](size_t tot, size_t br, int k, uint64_t seed) { return genBlocksDecimal32(tot, br, k, seed); },
        [](const BlockStream & blk, int k, int p, std::vector<PartState> & pts, BumpArena & ar, size_t ic, size_t mc_)
        { runTypedRadix<uint32_t>(blk, k, p, pts, ar, makeDecimal<DB::Decimal32>(), ic, mc_); });

    // FixedString(8) (8 bytes/row — same as UInt64).
    run_typed_variant(
        "fixedstr8", 8,
        [](size_t tot, size_t br, int k, uint64_t seed) { return genBlocksFixedStr(tot, br, k, 8, seed); },
        [](const BlockStream & blk, int k, int p, std::vector<PartState> & pts, BumpArena & ar, size_t ic, size_t mc_)
        { runTypedRadix<uint64_t>(blk, k, p, pts, ar, makeFixedString(8), ic, mc_); });

    return 0;
}
