// bench_radix_mem_sweep.cpp
//
// Sweep benchmark: measures peak output-buffer (OutBlock arena) memory and
// arena wastage for the radix partition operator across all combinations of
// T, P, K.
//
// Sweep grid (defaults):
//   T (threads)    : 4, 8, 16, 32, 48
//   P (partitions) : 32, 64, 128, 256, 512, 1024
//   K (columns)    : 1, 2, 4, 8
//   N (total rows) : 100 000 000
//   R (reps)       : 1  (memory allocation is deterministic; one rep is enough)
//
// Waste decomposition (per thread, then summed across T):
//   allocated = arena.allocatedBytes()  — OS-committed bytes (slab totals)
//   used      = arena.usedBytes()       — bytes dispensed to callers (no align pad,
//                                         since every OutBlock alloc is a multiple of 64 B)
//   slab_slack  = allocated − used      — slab tail bytes never touched
//   hdr_bytes   = n_blocks × 128        — OutBlock header overhead (alignas(64) → 128 B each)
//   overhang    = (total_cap − filled) × K × 8   — empty rows in last block per partition
//   useful_data = filled × K × 8        — actual row payload bytes
//
//   Invariant: allocated = useful_data + hdr_bytes + overhang + slab_slack
//
// Stdout : CSV  T,P,K,rep,allocated,used,useful,slab_slack,hdr_bytes,overhang,waste_pct
// Stderr : progress messages

#include <Columns/ColumnVector.h>
#include <Columns/IColumn_fwd.h>
#include <Common/RadixShuffle/BumpArena.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>
#include <Common/RadixShuffle/OutBlock.h>
#include <Common/RadixShuffle/RadixPartitionOperator.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>

#include <fmt/format.h>


namespace
{

using namespace DB::RadixShuffle;
using BlockStream = std::vector<DB::Columns>;
using Clk = std::chrono::steady_clock;

/// Initial BumpArena slab: same as the existing bench_radix_partition.
constexpr size_t kArenaInitBytes = 64ULL << 20;


/// Generate `total` random uint64 rows in blocks of `block_rows` with K columns.
BlockStream genBlocks(size_t total, size_t block_rows, int K, uint64_t seed)
{
    BlockStream stream;
    stream.reserve((total + block_rows - 1) / block_rows);
    std::mt19937_64 rng(seed);
    for (size_t done = 0; done < total;)
    {
        const size_t bs = std::min(block_rows, total - done);
        DB::Columns block;
        block.reserve(static_cast<size_t>(K));
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


/// Per-thread result: all counters needed for waste analysis.
struct ThreadResult
{
    size_t allocated = 0; ///< arena.allocatedBytes() — slab totals (OS-committed)
    size_t used = 0; ///< arena.usedBytes()      — bytes actually dispensed (no align pad)
    size_t n_blocks = 0; ///< total OutBlock count across all P partitions
    size_t cap_rows = 0; ///< sum of OutBlock::capacity across all blocks
    size_t filled_rows = 0; ///< sum of OutBlock::filled across all blocks (== rpt)
};


/// Run radix partition on one thread's stream; collect arena + OutBlock chain stats.
ThreadResult runRadix(const BlockStream & blocks, int K, int P, size_t init_cap, size_t max_cap)
{
    BumpArena arena(kArenaInitBytes);

    std::vector<ColumnPrimitives> prims(static_cast<size_t>(K), makeFixedWidth<UInt64>());
    const bool use_swwc = RadixPartitionOperator<uint64_t>::should_use_swwc(K, P);
    RadixPartitionOperator<uint64_t> op(P, K, std::move(prims), arena, use_swwc, init_cap, max_cap);
    for (const auto & blk : blocks)
        op.process(blk);
    op.finish();

    ThreadResult r;
    r.allocated = arena.allocatedBytes();
    r.used = arena.usedBytes();

    for (const auto & ps : op.parts())
    {
        for (const OutBlock * b = ps.head; b != nullptr; b = b->next)
        {
            r.n_blocks++;
            r.cap_rows += b->capacity;
            r.filled_rows += b->filled;
        }
    }
    return r;
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

} // namespace


int main(int argc, char ** argv)
{
    size_t N = 100'000'000ULL;
    size_t B = 16384;
    int R = 1;

    for (int i = 1; i + 1 < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--rows")
            N = std::stoull(argv[++i]);
        else if (arg == "--reps")
            R = std::stoi(argv[++i]);
        else if (arg == "--block-rows")
            B = std::stoull(argv[++i]);
    }

    const std::vector<int> T_vals = {4, 8, 16, 32, 48};
    const std::vector<int> P_vals = {32, 64, 128, 256, 512, 1024};
    const std::vector<int> K_vals = {1, 2, 4, 8};

    fmt::print(stderr, "bench_radix_mem_sweep\n");
    fmt::print(stderr, "  N={} rows  B={} block_rows  R={} reps\n", N, B, R);
    fmt::print(stderr, "  T = 4, 8, 16, 32, 48\n");
    fmt::print(stderr, "  P = 32, 64, 128, 256, 512, 1024\n");
    fmt::print(stderr, "  K = 1, 2, 4, 8\n\n");

    // CSV header on stdout.
    // waste_pct = (allocated - useful_data) / allocated × 100
    fmt::print("T,P,K,rep,allocated,used,useful,slab_slack,hdr_bytes,overhang,waste_pct\n");
    fflush(stdout);

    for (int K : K_vals)
    {
        for (int T : T_vals)
        {
            const size_t rpt = (N + static_cast<size_t>(T) - 1) / static_cast<size_t>(T);

            fmt::print(stderr, "[K={} T={:2}]  generating {} streams × {:9} rows... ", K, T, T, rpt);
            fflush(stderr);

            const auto tgen = Clk::now();
            std::vector<BlockStream> streams(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
                streams[static_cast<size_t>(t)] = genBlocks(rpt, B, K, 42ULL + static_cast<uint64_t>(t));
            const double gen_s = std::chrono::duration<double>(Clk::now() - tgen).count();

            fmt::print(stderr, "{:.2f} s\n", gen_s);

            for (int P : P_vals)
            {
                const auto [cap_init, cap_max] = adaptiveCaps(rpt, static_cast<size_t>(P));
                const bool swwc = RadixPartitionOperator<uint64_t>::should_use_swwc(K, P);

                for (int rep = 0; rep < R; ++rep)
                {
                    std::vector<ThreadResult> results(static_cast<size_t>(T));

                    std::vector<std::thread> ths;
                    ths.reserve(static_cast<size_t>(T));
                    for (int t = 0; t < T; ++t)
                    {
                        ths.emplace_back(
                            [&, t]()
                            {
                                pinThread(t);
                                results[static_cast<size_t>(t)] = runRadix(streams[static_cast<size_t>(t)], K, P, cap_init, cap_max);
                            });
                    }
                    for (auto & th : ths)
                        th.join();

                    // Sum across threads.
                    size_t allocated = 0, used = 0, n_blocks = 0, cap_rows = 0, filled_rows = 0;
                    for (const auto & r : results)
                    {
                        allocated += r.allocated;
                        used += r.used;
                        n_blocks += r.n_blocks;
                        cap_rows += r.cap_rows;
                        filled_rows += r.filled_rows;
                    }

                    const size_t elem = sizeof(uint64_t);
                    const size_t useful = filled_rows * static_cast<size_t>(K) * elem;
                    // OutBlock header: alignas(64) struct with 88 B of fields → sizeof = 128 B.
                    constexpr size_t kHdrBytes = 128;
                    const size_t hdr_bytes = n_blocks * kHdrBytes;
                    const size_t overhang = (cap_rows - filled_rows) * static_cast<size_t>(K) * elem;
                    const size_t slab_slack = allocated - used;
                    const double waste_pct = 100.0 * static_cast<double>(allocated - useful) / static_cast<double>(allocated);

                    fmt::print(
                        "{},{},{},{},{},{},{},{},{},{},{:.2f}\n",
                        T,
                        P,
                        K,
                        rep,
                        allocated,
                        used,
                        useful,
                        slab_slack,
                        hdr_bytes,
                        overhang,
                        waste_pct);
                    fflush(stdout);
                }

                fmt::print(stderr, "        P={:5}  cap_init={:6}  cap_max={:6}  {}\n", P, cap_init, cap_max, swwc ? "SWWC" : "direct");
            }

            fmt::print(stderr, "\n");
        }
    }

    return 0;
}
