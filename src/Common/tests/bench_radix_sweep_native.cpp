// bench_radix_sweep_native.cpp
//
// Sweeps K × P × T configurations for six variants:
//   radix    — RadixShuffler (BumpArena + OutBlock)
//   batched  — BatchedRadixShuffler (default 32 MiB flush)
//   bt_2MiB  — BatchedRadixShuffler (2 MiB flush)
//   bt_4MiB  — BatchedRadixShuffler (4 MiB flush)
//   bt_pblk  — BatchedRadixShuffler (flush every block, max_buffered_blocks=1)
//   native   — NativeRadixShuffler  (IColumn::scatter per block)
//
// Sweep parameters:
//   K ∈ {1, 4, 8}
//   P ∈ {4, 16, 32, 64, 256}
//   T ∈ {4, 8, 16, 32, 64}
//   B = 16384 (block rows, fixed)
//   rows = 100 M, reps = 5

#include <pthread.h>
#include <Columns/ColumnVector.h>
#include <Columns/IColumn_fwd.h>
#include <fmt/format.h>
#include <Common/RadixShuffle/BatchedRadixShuffler.h>
#include <Common/RadixShuffle/BumpArena.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>
#include <Common/RadixShuffle/NativeRadixShuffler.h>
#include <Common/RadixShuffle/OutBlock.h>
#include <Common/RadixShuffle/RadixShuffler.h>
#include <Common/ThreadPool.h>
#include <Common/assert_cast.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <thread>
#include <utility>
#include <vector>


namespace
{

using namespace DB;
using Clk = std::chrono::steady_clock;
using BlockStream = std::vector<DB::Columns>;
using BatchedOutput = std::vector<std::vector<DB::Columns>>;

constexpr size_t kArenaSlabBytes = 64ULL << 20;
constexpr size_t k2MiB = 2ULL << 20;
constexpr size_t k4MiB = 4ULL << 20;

constexpr std::array<int, 1> kKVals = {8};
constexpr std::array<int, 5> kPVals = {4, 16, 32, 64, 256};
constexpr std::array<int, 5> kTVals = {4, 8, 16, 32, 64};
constexpr size_t kBlockRows = 16384;
constexpr size_t kTotalRows = 100'000'000ULL;
constexpr int kReps = 5;


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


void runRadix(
    std::span<const DB::Columns> blocks, int K, int P, std::vector<PartState> & parts, BumpArena & arena, size_t init_cap, size_t max_cap)
{
    std::vector<ColumnPrimitives> prims(static_cast<size_t>(K), makeFixedWidth<UInt64>());
    RadixShuffler op(P, K, std::move(prims), arena, RadixShuffler::shouldUseSwwc(K, P), init_cap, max_cap);
    for (const auto & blk : blocks)
        op.process(blk);
    op.finish();
    parts = std::move(op.parts());
}


void runBatched(
    std::span<const DB::Columns> blocks,
    int K,
    int P,
    BatchedOutput & output,
    size_t max_buffered_blocks = 0,
    size_t max_buffered_bytes = 0)
{
    std::vector<ColumnPrimitives> prims(static_cast<size_t>(K), makeFixedWidth<UInt64>());
    BatchedRadixShuffler op(P, K, std::move(prims), BatchedRadixShuffler::shouldUseSwwc(K, P), max_buffered_blocks, max_buffered_bytes);
    for (const auto & blk : blocks)
        op.process(blk);
    op.finish();
    output = std::move(op.output());
}


void runNative(std::span<const DB::Columns> blocks, int K, int P, BatchedOutput & output)
{
    NativeRadixShuffler op(P, K);
    for (const auto & blk : blocks)
        op.process(blk);
    op.finish();
    output = std::move(op.output());
}


size_t countRows(const BatchedOutput & out)
{
    size_t rows = 0;
    for (const auto & p : out)
        for (const auto & blk : p)
            if (!blk.empty())
                rows += blk[0]->size();
    return rows;
}


template <typename Fn>
double timeThreaded(int T, size_t blocks_per_thread, const BlockStream & master, Fn fn)
{
    const auto t0 = Clk::now();
    std::vector<ThreadFromGlobalPool> ths;
    ths.reserve(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t)
    {
        ths.emplace_back(
            [&fn, &master, t, blocks_per_thread]()
            {
                pinThread(t);
                const size_t off = static_cast<size_t>(t) * blocks_per_thread;
                fn(t, std::span<const DB::Columns>(master.data() + off, blocks_per_thread));
            });
    }
    for (auto & th : ths)
        th.join();
    return std::chrono::duration<double>(Clk::now() - t0).count();
}

} // namespace


int main(int /*argc*/, char ** /*argv*/)
{
    fmt::print("bench_radix_sweep_native\n");
    fmt::print("K=8  P∈{{4,16,32,64,256}}  T∈{{4,8,16,32,64}}  B={}  rows={}M  reps={}\n\n", kBlockRows, kTotalRows / 1'000'000, kReps);

    const size_t max_T = *std::max_element(kTVals.begin(), kTVals.end());
    GlobalThreadPool::initialize(max_T * 2, max_T, max_T * 4);

    for (const int K : kKVals)
    {
        fmt::print("=== K={} ===\n", K);
        fmt::print("  Generating {} blocks × {} rows × {} cols...\n", kTotalRows / kBlockRows, kBlockRows, K);
        const auto tg0 = Clk::now();
        const BlockStream master = genBlocks(kTotalRows, kBlockRows, K, 42ULL + static_cast<uint64_t>(K) * 1000);
        fmt::print("  {:.2f} s\n\n", std::chrono::duration<double>(Clk::now() - tg0).count());

        const size_t total_blocks = master.size();

        fmt::print(
            "{:<3}  {:<5}  {:<4}"
            "  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}"
            "  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}\n",
            "K",
            "P",
            "T",
            "radix",
            "batched",
            "bt_2MiB",
            "bt_4MiB",
            "bt_pblk",
            "native",
            "nt/rd",
            "nt/bt",
            "nt/bt2m",
            "nt/bt4m",
            "nt/btpb");
        fmt::print("{}\n", std::string(124, '-'));

        for (const int P : kPVals)
        {
            for (const int T : kTVals)
            {
                const size_t blocks_per_thread = total_blocks / static_cast<size_t>(T);
                if (blocks_per_thread == 0)
                {
                    fmt::print("{:<3}  {:<5}  {:<4}  (skip)\n", K, P, T);
                    continue;
                }
                const size_t rows_per_thread = blocks_per_thread * kBlockRows;
                const size_t actual_total = static_cast<size_t>(T) * rows_per_thread;

                const auto [cap_init, cap_max] = adaptiveCaps(rows_per_thread, static_cast<size_t>(P));

                std::vector<double> rd_ns(kReps), bt_ns(kReps), bt2m_ns(kReps), bt4m_ns(kReps), btpb_ns(kReps), nt_ns(kReps);

                std::vector<std::vector<PartState>> rd_parts(static_cast<size_t>(T));
                std::vector<BatchedOutput> bt_out(static_cast<size_t>(T));
                std::vector<BatchedOutput> bt2m_out(static_cast<size_t>(T));
                std::vector<BatchedOutput> bt4m_out(static_cast<size_t>(T));
                std::vector<BatchedOutput> btpb_out(static_cast<size_t>(T));
                std::vector<BatchedOutput> nt_out(static_cast<size_t>(T));

                for (int rep = 0; rep < kReps; ++rep)
                {
                    // ── RadixShuffler ─────────────────────────────────────────
                    {
                        std::vector<BumpArena> arenas;
                        arenas.reserve(static_cast<size_t>(T));
                        for (int t = 0; t < T; ++t)
                            arenas.emplace_back(kArenaSlabBytes);
                        for (auto & p : rd_parts)
                            p.clear();

                        const double wall = timeThreaded(
                            T,
                            blocks_per_thread,
                            master,
                            [&](int t, std::span<const DB::Columns> slice)
                            {
                                runRadix(slice, K, P, rd_parts[static_cast<size_t>(t)], arenas[static_cast<size_t>(t)], cap_init, cap_max);
                            });
                        rd_ns[static_cast<size_t>(rep)] = wall * 1e9 / static_cast<double>(actual_total);
                    }

                    // ── BatchedRadixShuffler (32 MiB) ─────────────────────────
                    {
                        const double wall = timeThreaded(
                            T,
                            blocks_per_thread,
                            master,
                            [&](int t, std::span<const DB::Columns> slice) { runBatched(slice, K, P, bt_out[static_cast<size_t>(t)]); });
                        bt_ns[static_cast<size_t>(rep)] = wall * 1e9 / static_cast<double>(actual_total);
                    }

                    // ── BatchedRadixShuffler (2 MiB) ──────────────────────────
                    {
                        const double wall = timeThreaded(
                            T,
                            blocks_per_thread,
                            master,
                            [&](int t, std::span<const DB::Columns> slice)
                            { runBatched(slice, K, P, bt2m_out[static_cast<size_t>(t)], 0, k2MiB); });
                        bt2m_ns[static_cast<size_t>(rep)] = wall * 1e9 / static_cast<double>(actual_total);
                    }

                    // ── BatchedRadixShuffler (4 MiB) ──────────────────────────
                    {
                        const double wall = timeThreaded(
                            T,
                            blocks_per_thread,
                            master,
                            [&](int t, std::span<const DB::Columns> slice)
                            { runBatched(slice, K, P, bt4m_out[static_cast<size_t>(t)], 0, k4MiB); });
                        bt4m_ns[static_cast<size_t>(rep)] = wall * 1e9 / static_cast<double>(actual_total);
                    }

                    // ── BatchedRadixShuffler (per block) ──────────────────────
                    {
                        const double wall = timeThreaded(
                            T,
                            blocks_per_thread,
                            master,
                            [&](int t, std::span<const DB::Columns> slice)
                            { runBatched(slice, K, P, btpb_out[static_cast<size_t>(t)], /*max_blocks=*/1); });
                        btpb_ns[static_cast<size_t>(rep)] = wall * 1e9 / static_cast<double>(actual_total);
                    }

                    // ── NativeRadixShuffler ───────────────────────────────────
                    {
                        const double wall = timeThreaded(
                            T,
                            blocks_per_thread,
                            master,
                            [&](int t, std::span<const DB::Columns> slice) { runNative(slice, K, P, nt_out[static_cast<size_t>(t)]); });
                        nt_ns[static_cast<size_t>(rep)] = wall * 1e9 / static_cast<double>(actual_total);
                    }
                } // reps

                // Sanity checks (thread 0, last rep)
                auto check = [&](const BatchedOutput & out, const char * name)
                {
                    const size_t got = countRows(out);
                    if (got != rows_per_thread)
                        fmt::print("[ERROR] {} K={} P={} T={}: rows {} != {}\n", name, K, P, T, got, rows_per_thread);
                };
                check(bt_out[0], "batched");
                check(bt2m_out[0], "bt_2MiB");
                check(bt4m_out[0], "bt_4MiB");
                check(btpb_out[0], "bt_pblk");
                check(nt_out[0], "native");

                const double rd = *std::min_element(rd_ns.begin(), rd_ns.end());
                const double bt = *std::min_element(bt_ns.begin(), bt_ns.end());
                const double bt2m = *std::min_element(bt2m_ns.begin(), bt2m_ns.end());
                const double bt4m = *std::min_element(bt4m_ns.begin(), bt4m_ns.end());
                const double btpb = *std::min_element(btpb_ns.begin(), btpb_ns.end());
                const double nt = *std::min_element(nt_ns.begin(), nt_ns.end());

                fmt::print(
                    "{:<3}  {:<5}  {:<4}"
                    "  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}"
                    "  {:>7.2f}x  {:>7.2f}x  {:>7.2f}x  {:>7.2f}x  {:>7.2f}x\n",
                    K,
                    P,
                    T,
                    rd,
                    bt,
                    bt2m,
                    bt4m,
                    btpb,
                    nt,
                    nt / rd,
                    nt / bt,
                    nt / bt2m,
                    nt / bt4m,
                    nt / btpb);

                fmt::print(
                    "RESULT K={} P={} T={}"
                    " radix={:.3f} batched={:.3f} bt_2MiB={:.3f} bt_4MiB={:.3f} bt_pblk={:.3f} native={:.3f}\n",
                    K,
                    P,
                    T,
                    rd,
                    bt,
                    bt2m,
                    bt4m,
                    btpb,
                    nt);
            }
        }
        fmt::print("\n");
    }

    return 0;
}
