// Benchmark: plain memcpy vs. single-pass radix partitioning via production
// `RadixShuffler` in `src/Common/RadixShuffle/`.
//
// Workload:
//   - Input is a stream of `DB::Columns` blocks (default: K × ColumnVector<UInt64>).
//   - Each thread owns one input stream; rows are split evenly across threads.
//   - Memcpy baseline appends all rows into a single `PartState` (no hash).
//   - Radix path hashes the first column (`compute_pids`), histograms, and
//     scatters all K columns into P partition chains using `ColumnPrimitives`
//     (`scatter_raw` or `scatter_raw_swwc` when `shouldUseSwwc(K,P)` is true).
//   - Both paths use the same `BumpArena`, `OutBlock`, and `growPart` helpers.
//
// Metrics: wall-clock ns/row (aggregate over all threads) and GB/s assuming
// read+write of `K × elem_size` bytes per row (elem_size = 8 for UInt64).
//
// After the UInt64 comparison, a type-sweep re-runs radix with other column
// types (Decimal, FixedString, Nullable) at matched or stated bytes/row.
//
// Usage:
//   bench_radix_partition [OPTIONS]
//   --partitions P   power-of-2 in [1,32768]   (default 64)
//   --columns    K   columns per row            (default  4, max 8)
//   --rows       N   total rows                 (default  100 000 000)
//   --block-rows B   rows per input block       (default  16 384)
//   --threads    T   worker threads             (default  16)
//   --reps       R   timed repetitions          (default   5)
//   --batch-max-blocks N  batched flush block limit (0 = P, default 0)
//   --batch-max-bytes  N  batched flush byte limit  (0 = 32 MiB, default 0)

#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnVector.h>
#include <Columns/IColumn_fwd.h>
#include <base/Decimal.h>
#include <Common/RadixShuffle/BatchedRadixShuffler.h>
#include <Common/RadixShuffle/BumpArena.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>
#include <Common/RadixShuffle/ColumnPrimitives/Nullable.h>
#include <Common/RadixShuffle/ColumnPrimitivesDispatch.h>
#include <Common/RadixShuffle/OutBlock.h>
#include <Common/RadixShuffle/RadixShuffler.h>
#include <Common/ThreadPool.h>
#include <Common/assert_cast.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

using namespace DB;

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
    size_t batch_max_blocks = 0;
    size_t batch_max_bytes = 0;
};


struct Stats
{
    double mean = 0.0;
    double stddev = 0.0;
    double p50 = 0.0;
    double pmin = 0.0;
    double pmax = 0.0;
};


// ── input (UInt64 reference) ───────────────────────────────────────────────────

/// K columns of `ColumnVector<UInt64>`, `block_rows` rows per block (last block may be smaller).
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


/// Nullable(UInt64) blocks: K columns of ColumnNullable(ColumnVector<UInt64>),
/// 9 bytes/row each (1 null byte + 8 value bytes).  ~50% of rows are NULL.
BlockStream genBlocksNullableUInt64(size_t total, size_t block_rows, int K, uint64_t seed)
{
    BlockStream stream;
    std::mt19937_64 rng(seed);
    for (size_t done = 0; done < total;)
    {
        const size_t bs = std::min(block_rows, total - done);
        DB::Columns block;
        for (int k = 0; k < K; ++k)
        {
            auto nested = DB::ColumnVector<UInt64>::create();
            nested->getData().resize(bs);
            auto null_map = DB::ColumnUInt8::create();
            null_map->getData().resize(bs);
            for (size_t i = 0; i < bs; ++i)
            {
                const uint64_t v = rng();
                nested->getData()[i] = v;
                null_map->getData()[i] = static_cast<uint8_t>(v & 1); // ~50% nulls
            }
            block.push_back(DB::ColumnNullable::create(std::move(nested), std::move(null_map)));
        }
        stream.push_back(std::move(block));
        done += bs;
    }
    return stream;
}


/// Radix partition via `RadixShuffler` with K copies of `prim_proto`.
/// OutBlock memory is sized per physical column using each primitive's
/// `raw_elem_size` (and Nullable expansion into null-map + nested columns).
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
    const bool use_swwc = RadixShuffler::shouldUseSwwc(K, P);
    RadixShuffler op(P, K, std::move(prims), arena, use_swwc, init_cap, max_cap);
    for (const auto & block : blocks)
        op.process(block);
    op.finish();
    parts = std::move(op.parts());
}


// ── memcpy baseline ───────────────────────────────────────────────────────────
// Same `BumpArena` / `OutBlock` / `growPart` path as radix, but no hash: every
// row is appended to one `PartState` with uniform `sizeof(uint64_t)` columns.
// Serves as a lower bound on memory traffic for the same read+write volume.

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


// ── radix variant (UInt64) ────────────────────────────────────────────────────
// `makeFixedWidth<UInt64>()` on every column; partition id from column 0 only.

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
    const bool use_swwc = RadixShuffler::shouldUseSwwc(K, P);
    RadixShuffler op(P, K, std::move(prims), arena, use_swwc, init_cap, max_cap);
    for (const auto & block : blocks)
        op.process(block);
    op.finish();
    parts = std::move(op.parts());
}


using BatchedOutput = std::vector<std::vector<DB::Columns>>; // [partition][flush_cycle]

void runBatchedRadix(
    const BlockStream & blocks, int K, int P, BatchedOutput & output, size_t batch_max_blocks = 0, size_t batch_max_bytes = 0)
{
    std::vector<ColumnPrimitives> prims(static_cast<size_t>(K), makeFixedWidth<UInt64>());
    const bool use_swwc = BatchedRadixShuffler::shouldUseSwwc(K, P);
    BatchedRadixShuffler op(P, K, std::move(prims), use_swwc, batch_max_blocks, batch_max_bytes);
    for (const auto & block : blocks)
        op.process(block);
    op.finish();
    output = std::move(op.output());
}


// ── thread pinning (optional, best-effort) ─────────────────────────────────────

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
        else if (arg == "--batch-max-blocks" && i + 1 < args.size())
        {
            cfg.batch_max_blocks = std::stoull(args[++i]);
        }
        else if (arg == "--batch-max-bytes" && i + 1 < args.size())
        {
            cfg.batch_max_bytes = std::stoull(args[++i]);
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

    // ThreadFromGlobalPool auto-installs DB::ThreadStatus per job so the
    // MemoryTracker thread-local fast path is active — avoids the
    // total_memory_tracker.amount cacheline ping-pong that cripples raw
    // std::thread under high allocation pressure.
    // See tmp/icolumn_alloc_root_cause.md.
    GlobalThreadPool::initialize(
        /* max_threads = */ static_cast<size_t>(cfg.threads) * 2,
        /* max_free_threads = */ static_cast<size_t>(cfg.threads),
        /* queue_size = */ static_cast<size_t>(cfg.threads) * 4);

    const int partitions = cfg.partitions;
    const int columns = cfg.columns;
    const size_t num_rows = cfg.rows;
    const size_t block_rows = cfg.block_rows;
    const int threads = cfg.threads;
    const int reps = cfg.reps;
    const size_t batch_max_blocks = cfg.batch_max_blocks;
    const size_t batch_max_bytes = cfg.batch_max_bytes;

    const int batch = std::max(1024, std::min(RadixShuffler::kSmartMaxBatch, partitions * RadixShuffler::kBatchFactor));
    const size_t rpt = (num_rows + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads);
    const size_t total = rpt * static_cast<size_t>(threads);

    const auto [cap_init, cap_max] = adaptiveCaps(rpt, static_cast<size_t>(partitions));

    fmt::print("bench_radix_partition\n");
    fmt::print("  partitions={:<6}  columns={:<2}  rows={:<12}\n", partitions, columns, num_rows);
    fmt::print("  block-rows={:<6}  threads={:<3}  reps={}\n", block_rows, threads, reps);
    fmt::print("  rows/thread={}  total={}\n", rpt, total);
    fmt::print(
        "  data/thread = {:.1f} MiB  ({} cols \xc3\x97 {} rows \xc3\x97 8 B)\n",
        static_cast<double>(columns) * static_cast<double>(rpt) * 8.0 / (1 << 20),
        columns,
        rpt);
    fmt::print("  batch_size  = {}\n", batch);
    fmt::print(
        "  OutBlock cap: init={}  max={}  (avg rows/part\xe2\x89\x88{})\n", cap_init, cap_max, rpt / static_cast<size_t>(partitions));
    fmt::print("  radix mode  = {}\n", RadixShuffler::shouldUseSwwc(columns, partitions) ? "SWWC (NT stores)" : "direct");
    fmt::print(
        "  batched flush: max_blocks={}  max_bytes={} MiB\n",
        batch_max_blocks ? batch_max_blocks : static_cast<size_t>(partitions),
        static_cast<double>(batch_max_bytes ? batch_max_bytes : BatchedRadixShuffler::kDefaultMemBound) / (1 << 20));
    // Internal batch is max(1024, min(32768, P×16)); smaller block-rows still work
    // but process() may run multiple batches per input block.
    if (block_rows < static_cast<size_t>(batch))
        fmt::print("  [warn] block-rows {} < operator batch {} (multiple batches/block)\n", block_rows, batch);
    fmt::print("\n");

    // ── generate input streams ────────────────────────────────────────────────
    fmt::print("Generating {} streams \xc3\x97 {} blocks each...\n", threads, (rpt + block_rows - 1) / block_rows);
    const auto tg0 = Clk::now();
    std::vector<BlockStream> streams(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t)
        streams[static_cast<size_t>(t)] = genBlocks(rpt, block_rows, columns, 42ULL + static_cast<uint64_t>(t));
    fmt::print("  {:.2f} s\n\n", std::chrono::duration<double>(Clk::now() - tg0).count());

    // ── timed reps: fresh BumpArena per rep (no cross-rep reuse) ─────────────
    std::vector<PartState> mc_parts(static_cast<size_t>(threads));
    std::vector<std::vector<PartState>> radix_parts(static_cast<size_t>(threads));
    std::vector<BatchedOutput> bt_out(static_cast<size_t>(threads));

    // GB/s = gbs_k / ns_per_row; gbs_k counts read + write of K × 8 bytes/row.
    const double gbs_k = 2.0 * static_cast<double>(columns) * 8.0;

    std::vector<double> mc_ns(static_cast<size_t>(reps));
    std::vector<double> rd_ns(static_cast<size_t>(reps));
    std::vector<double> bt_ns(static_cast<size_t>(reps));

    // ── benchmark ─────────────────────────────────────────────────────────────
    fmt::print("{:<4}  {:>12}  {:>12}  {:>12}  {:>6}\n", "rep", "memcpy ns/row", "radix ns/row", "batched ns/row", "ratio");
    fmt::print("----  ------------  ------------  ------------  ------\n");

    for (int rep = 0; rep < reps; ++rep)
    {
        // ── memcpy — fresh BumpArena each rep ────────────────────────────────
        {
            std::vector<BumpArena> mc_arenas;
            mc_arenas.reserve(static_cast<size_t>(threads));
            for (int t = 0; t < threads; ++t)
                mc_arenas.emplace_back(kArenaSlabBytes);

            for (auto & part : mc_parts)
                part = {};
            const auto t0 = Clk::now();
            std::vector<ThreadFromGlobalPool> ths;
            ths.reserve(static_cast<size_t>(threads));
            for (int t = 0; t < threads; ++t)
            {
                ths.emplace_back(
                    [&, t]()
                    {
                        pinThread(t);
                        runMemcpy(
                            streams[static_cast<size_t>(t)],
                            columns,
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
            rd_arenas.reserve(static_cast<size_t>(threads));
            for (int t = 0; t < threads; ++t)
                rd_arenas.emplace_back(kArenaSlabBytes);

            for (int t = 0; t < threads; ++t)
                radix_parts[static_cast<size_t>(t)].clear();
            const auto t0 = Clk::now();
            std::vector<ThreadFromGlobalPool> ths;
            ths.reserve(static_cast<size_t>(threads));
            for (int t = 0; t < threads; ++t)
            {
                ths.emplace_back(
                    [&, t]()
                    {
                        pinThread(t);
                        runSmartRadix(
                            streams[static_cast<size_t>(t)],
                            columns,
                            partitions,
                            radix_parts[static_cast<size_t>(t)],
                            rd_arenas[static_cast<size_t>(t)],
                            cap_init,
                            cap_max);
                    });
            }
            for (auto & th : ths)
                th.join();
            rd_ns[static_cast<size_t>(rep)] = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9 / static_cast<double>(total);
        }

        // ── batched radix ─────────────────────────────────────────────────────
        {
            const auto t0 = Clk::now();
            std::vector<ThreadFromGlobalPool> ths;
            ths.reserve(static_cast<size_t>(threads));
            for (int t = 0; t < threads; ++t)
            {
                ths.emplace_back(
                    [&, t]()
                    {
                        pinThread(t);
                        runBatchedRadix(
                            streams[static_cast<size_t>(t)],
                            columns,
                            partitions,
                            bt_out[static_cast<size_t>(t)],
                            batch_max_blocks,
                            batch_max_bytes);
                    });
            }
            for (auto & th : ths)
                th.join();
            bt_ns[static_cast<size_t>(rep)] = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9 / static_cast<double>(total);
        }

        fmt::print(
            "{:<4}  {:>12.3f}  {:>12.3f}  {:>12.3f}  {:>5.2f}x\n",
            rep,
            mc_ns[static_cast<size_t>(rep)],
            rd_ns[static_cast<size_t>(rep)],
            bt_ns[static_cast<size_t>(rep)],
            rd_ns[static_cast<size_t>(rep)] / mc_ns[static_cast<size_t>(rep)]);
    }

    // ── sanity: scattered row count on thread 0 after last batched rep ───────
    const size_t expected_data = rpt * static_cast<size_t>(columns) * 8;
    size_t batched_used_rows = 0;
    for (const auto & per_partition : bt_out[0])
        for (const auto & flush_block : per_partition)
            if (!flush_block.empty())
                batched_used_rows += flush_block[0]->size();
    const size_t batched_used_bytes = batched_used_rows * static_cast<size_t>(columns) * sizeof(uint64_t);
    fmt::print("\nOutput after last rep (thread 0):\n");
    fmt::print(
        "  expected data  = {:>7.1f} MiB  ({} rows \xc3\x97 {} cols \xc3\x97 8 B)\n",
        static_cast<double>(expected_data) / 1048576.0,
        rpt,
        columns);
    fmt::print(
        "  batched rows   = {:>7.1f} MiB  ({} rows scattered)\n", static_cast<double>(batched_used_bytes) / 1048576.0, batched_used_rows);
    if (batched_used_rows != rpt)
        fmt::print("  [ERROR] batched row count {} != expected {}\n", batched_used_rows, rpt);

    // ── summary ───────────────────────────────────────────────────────────────
    const auto mc = computeStats(std::move(mc_ns));
    const auto rd = computeStats(std::move(rd_ns));
    const auto bt = computeStats(std::move(bt_ns));

    fmt::print("\nSummary (agg = wall_ns/total_rows;  per-thr = agg\xc3\x97threads;  GB/s = R+W)\n");
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
            s.pmin * static_cast<double>(threads));
    };
    print_row("memcpy", mc);
    print_row("radix", rd);
    print_row("batched", bt);

    fmt::print(
        "\nOverhead (radix / memcpy):  best={:.3f}x  mean={:.3f}x"
        "  (per-thread: {:.1f} vs {:.1f} ns/row)\n",
        rd.pmin / mc.pmin,
        rd.mean / mc.mean,
        rd.pmin * static_cast<double>(threads),
        mc.pmin * static_cast<double>(threads));
    fmt::print(
        "Overhead (batched / memcpy):  best={:.3f}x  mean={:.3f}x"
        "  (batched vs radix: {:.3f}x)\n",
        bt.pmin / mc.pmin,
        bt.mean / mc.mean,
        bt.pmin / rd.pmin);

    // ── Type-sweep: same P/K/threads/reps, other ColumnPrimitives factories ───
    // UInt64 numbers above are the reference; below re-runs radix only with
    // fresh streams.  `elem_size` in the table is the logical bytes/row per
    // column used for GB/s; OutBlock layout follows each primitive's
    // `raw_elem_size` (Nullable uses 1-byte map + nested width).
    //
    //  Type          B/elem  B/row (K=4)   notes
    //  -----------   -----  -----------   -----
    //  uint64           8          32     reference
    //  decimal64        8          32     same width as UInt64
    //  decimal32        4          16     half width
    //  fixedstr8        8          32     direct scatter (no SWWC)
    //  null_uint64      9          36     ~50% nulls; 1+8 bytes logical/row
    fmt::print("\nType-sweep (partitions={} columns={} threads={} reps={}):\n", partitions, columns, threads, reps);
    fmt::print("{:<12}  {:>6}  {:>7}  {:>8}  {:>8}\n", "type", "B/elem", "B/row", "ns/row", "GB/s(R+W)");
    fmt::print("------------  ------  -------  --------  --------\n");

    // Run one typed variant (regenerate streams, fresh arena each rep).
    auto run_typed_variant = [&](const char * label, size_t elem_size, auto gen_fn, auto run_fn)
    {
        // Generate streams for this type.
        std::vector<BlockStream> typed_streams(static_cast<size_t>(threads));
        for (int t = 0; t < threads; ++t)
            typed_streams[static_cast<size_t>(t)] = gen_fn(rpt, block_rows, columns, 42ULL + static_cast<uint64_t>(t) + 1000ULL);

        std::vector<double> typed_ns(static_cast<size_t>(reps));
        std::vector<std::vector<PartState>> typed_parts(static_cast<size_t>(threads));
        const double bytes_per_row = static_cast<double>(columns) * static_cast<double>(elem_size);
        const double gbs_typed = 2.0 * bytes_per_row;

        for (int rep = 0; rep < reps; ++rep)
        {
            std::vector<BumpArena> typed_arenas;
            typed_arenas.reserve(static_cast<size_t>(threads));
            for (int t = 0; t < threads; ++t)
                typed_arenas.emplace_back(kArenaSlabBytes);
            for (int t = 0; t < threads; ++t)
                typed_parts[static_cast<size_t>(t)].clear();

            const auto t0 = Clk::now();
            std::vector<ThreadFromGlobalPool> ths;
            ths.reserve(static_cast<size_t>(threads));
            for (int t = 0; t < threads; ++t)
            {
                ths.emplace_back(
                    [&, t]()
                    {
                        pinThread(t);
                        run_fn(
                            typed_streams[static_cast<size_t>(t)],
                            columns,
                            partitions,
                            typed_parts[static_cast<size_t>(t)],
                            typed_arenas[static_cast<size_t>(t)],
                            cap_init,
                            cap_max);
                    });
            }
            for (auto & th : ths)
                th.join();
            typed_ns[static_cast<size_t>(rep)] = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9 / static_cast<double>(total);
        }
        const auto s = computeStats(std::move(typed_ns));
        fmt::print("{:<12}  {:>6}  {:>7}  {:>8.3f}  {:>8.1f}\n", label, elem_size, columns * elem_size, s.pmin, gbs_typed / s.pmin);
    };

    // UInt64 (re-run with type-sweep streams; should match main radix).
    run_typed_variant(
        "uint64",
        8,
        [](size_t tot, size_t br, int k, uint64_t seed) { return genBlocks(tot, br, k, seed); },
        [](const BlockStream & blk, int k, int p, std::vector<PartState> & pts, BumpArena & ar, size_t ic, size_t mc_)
        { runTypedRadix(blk, k, p, pts, ar, makeFixedWidth<UInt64>(), ic, mc_); });

    // Decimal64 (8 bytes/row — same as UInt64).
    run_typed_variant(
        "decimal64",
        8,
        [](size_t tot, size_t br, int k, uint64_t seed) { return genBlocksDecimal64(tot, br, k, seed); },
        [](const BlockStream & blk, int k, int p, std::vector<PartState> & pts, BumpArena & ar, size_t ic, size_t mc_)
        { runTypedRadix(blk, k, p, pts, ar, makeDecimal<DB::Decimal64>(), ic, mc_); });

    // Decimal32 (4 bytes/row — half of UInt64).
    run_typed_variant(
        "decimal32",
        4,
        [](size_t tot, size_t br, int k, uint64_t seed) { return genBlocksDecimal32(tot, br, k, seed); },
        [](const BlockStream & blk, int k, int p, std::vector<PartState> & pts, BumpArena & ar, size_t ic, size_t mc_)
        { runTypedRadix(blk, k, p, pts, ar, makeDecimal<DB::Decimal32>(), ic, mc_); });

    // FixedString(8) (8 bytes/row — same as UInt64).
    run_typed_variant(
        "fixedstr8",
        8,
        [](size_t tot, size_t br, int k, uint64_t seed) { return genBlocksFixedStr(tot, br, k, 8, seed); },
        [](const BlockStream & blk, int k, int p, std::vector<PartState> & pts, BumpArena & ar, size_t ic, size_t mc_)
        { runTypedRadix(blk, k, p, pts, ar, makeFixedString(8), ic, mc_); });


    // Nullable(UInt64): logical 9 B/row; operator expands to null-map + values.
    run_typed_variant(
        "null_uint64",
        9,
        [](size_t tot, size_t br, int k, uint64_t seed) { return genBlocksNullableUInt64(tot, br, k, seed); },
        [](const BlockStream & blk, int k, int p, std::vector<PartState> & pts, BumpArena & ar, size_t ic, size_t mc_)
        { runTypedRadix(blk, k, p, pts, ar, makeNullable(makeFixedWidth<UInt64>()), ic, mc_); });

    return 0;
}
