// bench_radix_strategic.cpp
//
// Runs 11 reps on 15 hand-picked (K, P, T) configs chosen to cover:
//   • native's win cells         (K=1/P=64/T=32, K=4/P=64/T=32, K=8/P=32/T=64)
//   • native's worst cells       (K=1/P=16/T=4,  K=1/P=32/T=4)
//   • near-ties                  (K=8/P=4/T=4,   K=4/P=32/T=16)
//   • bt_2MiB anomaly            (K=4/P=16/T=8,  K=8/P=16/T=8)
//   • high-P interesting zone    (K=8/P=256/T=4, K=8/P=256/T=32)
//   • sweep corners              (small P+high T, large P+low T)
//
// Variants: radix, batched(32M), bt_2MiB, bt_4MiB, bt_pblk, native
// B = 16384, rows = 100M

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
#include <chrono>
#include <cstdint>
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
constexpr size_t kBlockRows = 16384;
constexpr size_t kTotalRows = 100'000'000ULL;
constexpr int kReps = 11;

struct Config
{
    int K, P, T;
    const char * label;
};

// 15 strategic configs
constexpr Config kConfigs[] = {
    // ── native wins ─────────────────────────────────────────────────────────
    {1, 64, 32, "native-win K1"},
    {4, 64, 32, "native-win K4"},
    {8, 32, 64, "native-win K8"},

    // ── near ties ───────────────────────────────────────────────────────────
    {8, 4, 4, "near-tie  K8"},
    {4, 32, 16, "near-tie  K4"},

    // ── native worst (low T, medium P) ──────────────────────────────────────
    {1, 16, 4, "worst-nt  K1P16T4"},
    {1, 32, 4, "worst-nt  K1P32T4"},
    {4, 16, 4, "worst-nt  K4P16T4"},

    // ── bt_2MiB anomaly ─────────────────────────────────────────────────────
    {4, 16, 8, "bt2m-anom K4"},
    {8, 16, 8, "bt2m-anom K8"},

    // ── high-P zone ─────────────────────────────────────────────────────────
    {4, 256, 4, "highP     K4T4"},
    {8, 256, 4, "highP     K8T4"},
    {8, 256, 32, "highP     K8T32"},

    // ── small P, high T ─────────────────────────────────────────────────────
    {4, 4, 64, "smlP-hiT  K4"},
    {8, 4, 64, "smlP-hiT  K8"},
};
constexpr int kNConfigs = static_cast<int>(sizeof(kConfigs) / sizeof(kConfigs[0]));


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


void runBatched(std::span<const DB::Columns> blocks, int K, int P, BatchedOutput & output, size_t max_blocks = 0, size_t max_bytes = 0)
{
    std::vector<ColumnPrimitives> prims(static_cast<size_t>(K), makeFixedWidth<UInt64>());
    BatchedRadixShuffler op(P, K, std::move(prims), BatchedRadixShuffler::shouldUseSwwc(K, P), max_blocks, max_bytes);
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
double timeThreaded(int T, size_t blocks_per_thread, const BlockStream & master, size_t actual_total, Fn fn)
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
    return std::chrono::duration<double>(Clk::now() - t0).count() * 1e9 / static_cast<double>(actual_total);
}

} // namespace


int main(int /*argc*/, char ** /*argv*/)
{
    fmt::print("bench_radix_strategic\n");
    fmt::print("B={}  rows={}M  reps={}\n\n", kBlockRows, kTotalRows / 1'000'000, kReps);

    const size_t max_T = 64;
    GlobalThreadPool::initialize(max_T * 2, max_T, max_T * 4);

    // Group configs by K to generate data once per K.
    for (int K : {1, 4, 8})
    {
        bool any = false;
        for (int ci = 0; ci < kNConfigs; ++ci)
            if (kConfigs[ci].K == K)
            {
                any = true;
                break;
            }
        if (!any)
            continue;

        fmt::print("=== K={} — generating {} blocks × {} rows × {} cols... ", K, kTotalRows / kBlockRows, kBlockRows, K);
        const auto tg0 = Clk::now();
        const BlockStream master = genBlocks(kTotalRows, kBlockRows, K, 42ULL + static_cast<uint64_t>(K) * 1000);
        fmt::print("{:.2f} s ===\n\n", std::chrono::duration<double>(Clk::now() - tg0).count());

        fmt::print(
            "{:<22}  {:>3} {:>4} {:>4}"
            "  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
            "  {:>7} {:>7} {:>7} {:>7} {:>7}\n",
            "label",
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
            "nt/bt2",
            "nt/bt4",
            "nt/btp");
        fmt::print("{}\n", std::string(120, '-'));

        for (int ci = 0; ci < kNConfigs; ++ci)
        {
            const Config & cfg = kConfigs[ci];
            if (cfg.K != K)
                continue;

            const int P = cfg.P, T = cfg.T;
            const size_t total_blocks = master.size();
            const size_t blocks_per_thread = total_blocks / static_cast<size_t>(T);
            if (blocks_per_thread == 0)
            {
                fmt::print("{:<22}  (skip)\n", cfg.label);
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
                for (auto & p : rd_parts)
                    p.clear();
                rd_ns[static_cast<size_t>(rep)] = timeThreaded(
                    T,
                    blocks_per_thread,
                    master,
                    actual_total,
                    [&](int t, std::span<const DB::Columns> sl)
                    {
                        BumpArena arena(kArenaSlabBytes);
                        runRadix(sl, K, P, rd_parts[static_cast<size_t>(t)], arena, cap_init, cap_max);
                    });

                bt_ns[static_cast<size_t>(rep)] = timeThreaded(
                    T,
                    blocks_per_thread,
                    master,
                    actual_total,
                    [&](int t, std::span<const DB::Columns> sl) { runBatched(sl, K, P, bt_out[static_cast<size_t>(t)]); });

                bt2m_ns[static_cast<size_t>(rep)] = timeThreaded(
                    T,
                    blocks_per_thread,
                    master,
                    actual_total,
                    [&](int t, std::span<const DB::Columns> sl) { runBatched(sl, K, P, bt2m_out[static_cast<size_t>(t)], 0, k2MiB); });

                bt4m_ns[static_cast<size_t>(rep)] = timeThreaded(
                    T,
                    blocks_per_thread,
                    master,
                    actual_total,
                    [&](int t, std::span<const DB::Columns> sl) { runBatched(sl, K, P, bt4m_out[static_cast<size_t>(t)], 0, k4MiB); });

                btpb_ns[static_cast<size_t>(rep)] = timeThreaded(
                    T,
                    blocks_per_thread,
                    master,
                    actual_total,
                    [&](int t, std::span<const DB::Columns> sl) { runBatched(sl, K, P, btpb_out[static_cast<size_t>(t)], 1); });

                nt_ns[static_cast<size_t>(rep)] = timeThreaded(
                    T,
                    blocks_per_thread,
                    master,
                    actual_total,
                    [&](int t, std::span<const DB::Columns> sl) { runNative(sl, K, P, nt_out[static_cast<size_t>(t)]); });
            }

            // Sanity
            auto check = [&](const BatchedOutput & out, const char * name)
            {
                const size_t got = countRows(out);
                if (got != rows_per_thread)
                    fmt::print("[ERROR] {} rows {} != {}\n", name, got, rows_per_thread);
            };
            check(bt_out[0], "batched");
            check(bt2m_out[0], "bt2m");
            check(bt4m_out[0], "bt4m");
            check(btpb_out[0], "btpb");
            check(nt_out[0], "native");

            // Report all 11 reps + min
            const double rd = *std::min_element(rd_ns.begin(), rd_ns.end());
            const double bt = *std::min_element(bt_ns.begin(), bt_ns.end());
            const double bt2m = *std::min_element(bt2m_ns.begin(), bt2m_ns.end());
            const double bt4m = *std::min_element(bt4m_ns.begin(), bt4m_ns.end());
            const double btpb = *std::min_element(btpb_ns.begin(), btpb_ns.end());
            const double nt = *std::min_element(nt_ns.begin(), nt_ns.end());

            // Print all 11 reps
            fmt::print("\n{} (K={} P={} T={}):\n", cfg.label, K, P, T);
            fmt::print("  rep  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}\n", "radix", "batched", "bt_2MiB", "bt_4MiB", "bt_pblk", "native");
            for (int r = 0; r < kReps; ++r)
            {
                fmt::print(
                    "  {:>3}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}\n",
                    r + 1,
                    rd_ns[static_cast<size_t>(r)],
                    bt_ns[static_cast<size_t>(r)],
                    bt2m_ns[static_cast<size_t>(r)],
                    bt4m_ns[static_cast<size_t>(r)],
                    btpb_ns[static_cast<size_t>(r)],
                    nt_ns[static_cast<size_t>(r)]);
            }
            fmt::print(
                "  min  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}"
                "   nt/rd={:.2f}x nt/bt={:.2f}x nt/bt2={:.2f}x nt/bt4={:.2f}x nt/btp={:.2f}x\n",
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
                "RESULT label=\"{}\" K={} P={} T={}"
                " radix={:.3f} batched={:.3f} bt_2MiB={:.3f} bt_4MiB={:.3f} bt_pblk={:.3f} native={:.3f}\n",
                cfg.label,
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
        fmt::print("\n");
    }

    return 0;
}
