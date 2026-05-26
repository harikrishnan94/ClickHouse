// bench_radix_sweep.cpp
//
// Sweeps K×P×T configurations for memcpy, RadixShuffler, and BatchedRadixShuffler.
// Data is generated once per K and reused across all (P,T) configs for that K.
// Hard-coded sweep: K∈{1,2,4,8}, P∈{16,64,256,1024}, T∈{8,16,32,48},
//   block_rows=4096, rows=60M, reps=3.
//
// Usage:
//   bench_radix_sweep [--batch-max-blocks N] [--batch-max-bytes N]
//   0 means use defaults (P blocks, 32 MiB bytes).

#include <pthread.h>
#include <Columns/ColumnVector.h>
#include <Columns/IColumn_fwd.h>
#include <fmt/format.h>
#include <Common/RadixShuffle/BatchedRadixShuffler.h>
#include <Common/RadixShuffle/BumpArena.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>
#include <Common/RadixShuffle/OutBlock.h>
#include <Common/RadixShuffle/RadixShuffler.h>
#include <Common/ThreadPool.h>
#include <Common/assert_cast.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>


namespace
{

using namespace DB;
using Clk = std::chrono::steady_clock;
using BlockStream = std::vector<DB::Columns>;

constexpr size_t kArenaSlabBytes = 64ULL << 20;

constexpr std::array<int, 4> kKVals = {1, 2, 4, 8};
constexpr std::array<int, 4> kPVals = {16, 64, 256, 1024};
constexpr std::array<int, 4> kTVals = {8, 16, 32, 48};
constexpr size_t kBlockRows = 4096;
constexpr size_t kTotalRows = 60'000'000ULL;
constexpr int kReps = 3;


struct SweepConfig
{
    size_t batch_max_blocks = 0;
    size_t batch_max_bytes = 0;
};


std::optional<SweepConfig> parseCLI(std::span<char * const> args)
{
    SweepConfig cfg;
    for (size_t i = 0; i < args.size(); ++i)
    {
        const std::string arg = args[i];
        if (arg == "--batch-max-blocks" && i + 1 < args.size())
            cfg.batch_max_blocks = std::stoull(args[++i]);
        else if (arg == "--batch-max-bytes" && i + 1 < args.size())
            cfg.batch_max_bytes = std::stoull(args[++i]);
        else
        {
            fmt::print(stderr, "unknown arg: {}\n", args[i]);
            return std::nullopt;
        }
    }
    return cfg;
}


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


void runMemcpy(std::span<const DB::Columns> blocks, int K, PartState & out, BumpArena & arena, size_t init_cap, size_t max_cap)
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


void runSmartRadix(
    std::span<const DB::Columns> blocks, int K, int P, std::vector<PartState> & parts, BumpArena & arena, size_t init_cap, size_t max_cap)
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
    std::span<const DB::Columns> blocks, int K, int P, BatchedOutput & output, size_t batch_max_blocks = 0, size_t batch_max_bytes = 0)
{
    std::vector<ColumnPrimitives> prims(static_cast<size_t>(K), makeFixedWidth<UInt64>());
    const bool use_swwc = BatchedRadixShuffler::shouldUseSwwc(K, P);
    BatchedRadixShuffler op(P, K, std::move(prims), use_swwc, batch_max_blocks, batch_max_bytes);
    for (const auto & block : blocks)
        op.process(block);
    op.finish();
    output = std::move(op.output());
}

} // namespace


int main(int argc, char ** argv)
{
    const auto cfg_opt = parseCLI({argv + 1, static_cast<size_t>(argc - 1)});
    if (!cfg_opt)
        return 1;
    const SweepConfig & cfg = *cfg_opt;

    fmt::print("bench_radix_sweep\n");
    fmt::print("  batched flush: max_blocks={} (0=P)  max_bytes={} (0=32 MiB)\n\n", cfg.batch_max_blocks, cfg.batch_max_bytes);

    // ThreadFromGlobalPool auto-installs DB::ThreadStatus per job so the
    // MemoryTracker thread-local fast path is active — avoids the
    // total_memory_tracker.amount cacheline ping-pong that cripples raw
    // std::thread under high allocation pressure.
    // See tmp/icolumn_alloc_root_cause.md.
    const size_t max_T = *std::max_element(kTVals.begin(), kTVals.end());
    GlobalThreadPool::initialize(
        /* max_threads = */ max_T * 2,
        /* max_free_threads = */ max_T,
        /* queue_size = */ max_T * 4);

    const size_t total_blocks = kTotalRows / kBlockRows;

    for (const int K : kKVals)
    {
        fmt::print("=== K={} ===\n", K);
        fmt::print("Generating master stream ({} blocks x {} rows x {} cols)...\n", total_blocks, kBlockRows, K);
        const auto tg0 = Clk::now();
        const BlockStream master = genBlocks(total_blocks * kBlockRows, kBlockRows, K, 42ULL + static_cast<uint64_t>(K));
        fmt::print("  {:.2f} s\n\n", std::chrono::duration<double>(Clk::now() - tg0).count());

        fmt::print("K   P     T    memcpy   radix    batched  rd/mc   bt/mc   bt/rd\n");
        fmt::print("---  -----  ----  -------  -------  -------  ------  ------  ------\n");

        for (const int P : kPVals)
        {
            for (const int T : kTVals)
            {
                const size_t blocks_per_thread = total_blocks / static_cast<size_t>(T);
                const size_t rows_per_thread = blocks_per_thread * kBlockRows;
                const size_t actual_total = static_cast<size_t>(T) * rows_per_thread;

                const auto [cap_init, cap_max] = adaptiveCaps(rows_per_thread, static_cast<size_t>(P));

                std::vector<double> mc_ns(kReps);
                std::vector<double> rd_ns(kReps);
                std::vector<double> bt_ns(kReps);

                std::vector<PartState> mc_parts(static_cast<size_t>(T));
                std::vector<std::vector<PartState>> rd_parts(static_cast<size_t>(T));
                std::vector<BatchedOutput> bt_out(static_cast<size_t>(T));

                for (int rep = 0; rep < kReps; ++rep)
                {
                    // ── memcpy ────────────────────────────────────────────────
                    {
                        std::vector<BumpArena> arenas;
                        arenas.reserve(static_cast<size_t>(T));
                        for (int t = 0; t < T; ++t)
                            arenas.emplace_back(kArenaSlabBytes);
                        for (auto & part : mc_parts)
                            part = {};

                        const auto t0 = Clk::now();
                        std::vector<ThreadFromGlobalPool> ths;
                        ths.reserve(static_cast<size_t>(T));
                        for (int t = 0; t < T; ++t)
                        {
                            ths.emplace_back(
                                [&, t]()
                                {
                                    pinThread(t);
                                    const size_t off = static_cast<size_t>(t) * blocks_per_thread;
                                    const std::span<const DB::Columns> slice(master.data() + off, blocks_per_thread);
                                    runMemcpy(
                                        slice, K, mc_parts[static_cast<size_t>(t)], arenas[static_cast<size_t>(t)], cap_init, cap_max);
                                });
                        }
                        for (auto & th : ths)
                            th.join();
                        mc_ns[static_cast<size_t>(rep)]
                            = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9 / static_cast<double>(actual_total);
                    }

                    // ── radix ─────────────────────────────────────────────────
                    {
                        std::vector<BumpArena> arenas;
                        arenas.reserve(static_cast<size_t>(T));
                        for (int t = 0; t < T; ++t)
                            arenas.emplace_back(kArenaSlabBytes);
                        for (int t = 0; t < T; ++t)
                            rd_parts[static_cast<size_t>(t)].clear();

                        const auto t0 = Clk::now();
                        std::vector<ThreadFromGlobalPool> ths;
                        ths.reserve(static_cast<size_t>(T));
                        for (int t = 0; t < T; ++t)
                        {
                            ths.emplace_back(
                                [&, t]()
                                {
                                    pinThread(t);
                                    const size_t off = static_cast<size_t>(t) * blocks_per_thread;
                                    const std::span<const DB::Columns> slice(master.data() + off, blocks_per_thread);
                                    runSmartRadix(
                                        slice, K, P, rd_parts[static_cast<size_t>(t)], arenas[static_cast<size_t>(t)], cap_init, cap_max);
                                });
                        }
                        for (auto & th : ths)
                            th.join();
                        rd_ns[static_cast<size_t>(rep)]
                            = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9 / static_cast<double>(actual_total);
                    }

                    // ── batched radix ─────────────────────────────────────────
                    {
                        const auto t0 = Clk::now();
                        std::vector<ThreadFromGlobalPool> ths;
                        ths.reserve(static_cast<size_t>(T));
                        for (int t = 0; t < T; ++t)
                        {
                            ths.emplace_back(
                                [&, t]()
                                {
                                    pinThread(t);
                                    const size_t off = static_cast<size_t>(t) * blocks_per_thread;
                                    const std::span<const DB::Columns> slice(master.data() + off, blocks_per_thread);
                                    runBatchedRadix(slice, K, P, bt_out[static_cast<size_t>(t)], cfg.batch_max_blocks, cfg.batch_max_bytes);
                                });
                        }
                        for (auto & th : ths)
                            th.join();
                        bt_ns[static_cast<size_t>(rep)]
                            = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9 / static_cast<double>(actual_total);
                    }
                } // reps

                // Sanity check: batched row count (thread 0, partition-summed, after last rep)
                size_t batched_rows = 0;
                for (const auto & per_partition : bt_out[0])
                    for (const auto & flush_block : per_partition)
                        if (!flush_block.empty())
                            batched_rows += flush_block[0]->size();
                if (batched_rows != rows_per_thread)
                    fmt::print("[ERROR] K={} P={} T={} batched rows {} != expected {}\n", K, P, T, batched_rows, rows_per_thread);

                const double mc_min = *std::min_element(mc_ns.begin(), mc_ns.end());
                const double rd_min = *std::min_element(rd_ns.begin(), rd_ns.end());
                const double bt_min = *std::min_element(bt_ns.begin(), bt_ns.end());

                fmt::print(
                    "{}   {:<5} {:<4} {:>7.3f}  {:>7.3f}  {:>7.3f}  {:>5.2f}x  {:>5.2f}x  {:>5.2f}x\n",
                    K,
                    P,
                    T,
                    mc_min,
                    rd_min,
                    bt_min,
                    rd_min / mc_min,
                    bt_min / mc_min,
                    bt_min / rd_min);
                fmt::print("RESULT K={} P={} T={} memcpy={:.3f} radix={:.3f} batched={:.3f}\n", K, P, T, mc_min, rd_min, bt_min);
            }
        }
        fmt::print("\n");
    }

    return 0;
}
