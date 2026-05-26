// bench_isolated.cpp
//
// Runs a SINGLE variant (radix or batched) for one fixed K/P/T configuration.
// Designed for clean perf-stat / perf-record profiling without cross-variant noise.
//
// Usage:
//   bench_isolated --variant radix|batched --K N --P N --T N --rows N
//                  --block-rows N --reps N [--batch-max-blocks N]
//                  [--batch-max-bytes N]

#include <cstddef>
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
#include <chrono>
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
using BatchedOutput = std::vector<std::vector<DB::Columns>>; // [partition][flush_cycle]


struct Config
{
    std::string variant; // "radix" or "batched"
    int K = 4;
    int P = 64;
    int T = 16;
    size_t rows = 10'000'000;
    size_t block_rows = 4096;
    int reps = 5;
    size_t batch_max_blocks = 0;
    size_t batch_max_bytes = 0;
    std::string alloc_backend = "icolumn"; // "icolumn" or "aligned_alloc"
};


std::optional<Config> parseCLI(std::span<char * const> args)
{
    Config cfg;
    for (size_t i = 0; i < args.size(); ++i)
    {
        const std::string arg = args[i];
        if (arg == "--variant" && i + 1 < args.size())
            cfg.variant = args[++i];
        else if (arg == "--K" && i + 1 < args.size())
            cfg.K = std::stoi(args[++i]);
        else if (arg == "--P" && i + 1 < args.size())
            cfg.P = std::stoi(args[++i]);
        else if (arg == "--T" && i + 1 < args.size())
            cfg.T = std::stoi(args[++i]);
        else if (arg == "--rows" && i + 1 < args.size())
            cfg.rows = std::stoull(args[++i]);
        else if (arg == "--block-rows" && i + 1 < args.size())
            cfg.block_rows = std::stoull(args[++i]);
        else if (arg == "--reps" && i + 1 < args.size())
            cfg.reps = std::stoi(args[++i]);
        else if (arg == "--batch-max-blocks" && i + 1 < args.size())
            cfg.batch_max_blocks = std::stoull(args[++i]);
        else if (arg == "--batch-max-bytes" && i + 1 < args.size())
            cfg.batch_max_bytes = std::stoull(args[++i]);
        else if (arg == "--alloc-backend" && i + 1 < args.size())
            cfg.alloc_backend = args[++i];
        else
        {
            fmt::print(stderr, "unknown arg: {}\n", args[i]);
            return std::nullopt;
        }
    }
    if (cfg.variant != "radix" && cfg.variant != "batched")
    {
        fmt::print(stderr, "variant must be 'radix' or 'batched'\n");
        return std::nullopt;
    }
    if (cfg.alloc_backend != "icolumn" && cfg.alloc_backend != "aligned_alloc")
    {
        fmt::print(stderr, "--alloc-backend must be 'icolumn' or 'aligned_alloc'\n");
        return std::nullopt;
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


BatchedTimings runBatchedRadix(
    std::span<const DB::Columns> blocks,
    int K,
    int P,
    BatchedOutput & output,
    size_t batch_max_blocks = 0,
    size_t batch_max_bytes = 0,
    bool use_aligned_alloc = false)
{
    std::vector<ColumnPrimitives> prims(static_cast<size_t>(K), makeFixedWidth<UInt64>());
    const bool use_swwc = BatchedRadixShuffler::shouldUseSwwc(K, P);
    BatchedRadixShuffler op(P, K, std::move(prims), use_swwc, batch_max_blocks, batch_max_bytes, use_aligned_alloc);
    for (const auto & block : blocks)
        op.process(block);
    op.finish();
    output = std::move(op.output());
    return op.timings();
}

} // namespace


int main(int argc, char ** argv)
{
    const auto cfg_opt = parseCLI({argv + 1, static_cast<size_t>(argc - 1)});
    if (!cfg_opt)
        return 1;
    const Config & cfg = *cfg_opt;

    // ThreadFromGlobalPool auto-installs DB::ThreadStatus per job so the
    // MemoryTracker thread-local fast path is active — avoids the
    // total_memory_tracker.amount cacheline ping-pong that cripples raw
    // std::thread under high allocation pressure.
    // See tmp/icolumn_alloc_root_cause.md.
    GlobalThreadPool::initialize(
        /* max_threads = */ static_cast<size_t>(cfg.T) * 2,
        /* max_free_threads = */ static_cast<size_t>(cfg.T),
        /* queue_size = */ static_cast<size_t>(cfg.T) * 4);

    const size_t total_blocks = cfg.rows / cfg.block_rows;
    const size_t blocks_per_thread = total_blocks / static_cast<size_t>(cfg.T);
    const size_t rows_per_thread = blocks_per_thread * cfg.block_rows;
    const size_t actual_total = static_cast<size_t>(cfg.T) * rows_per_thread;

    const auto [cap_init, cap_max] = adaptiveCaps(rows_per_thread, static_cast<size_t>(cfg.P));

    // Compute and print derived parameters for reference
    const size_t bytes_per_row = static_cast<size_t>(cfg.K) * 8;
    const size_t effective_max_bytes = cfg.batch_max_bytes ? cfg.batch_max_bytes : static_cast<size_t>(32ULL << 20);
    const size_t buffer_blocks = cfg.batch_max_blocks
        ? cfg.batch_max_blocks
        : std::min(static_cast<size_t>(cfg.P), effective_max_bytes / (cfg.block_rows * bytes_per_row));
    const size_t buffer_bytes = buffer_blocks * cfg.block_rows * bytes_per_row;
    const int batch_size = std::max(1024, std::min(32768, cfg.P * 16));
    const bool use_swwc = (cfg.K == 1) ? (cfg.P >= 512) : (cfg.P >= 32);

    fmt::print("bench_isolated\n");
    fmt::print(
        "  variant={} K={} P={} T={} rows={} block_rows={} reps={}\n",
        cfg.variant,
        cfg.K,
        cfg.P,
        cfg.T,
        cfg.rows,
        cfg.block_rows,
        cfg.reps);
    fmt::print("  batch_max_blocks={} batch_max_bytes={} alloc_backend={}\n", cfg.batch_max_blocks, cfg.batch_max_bytes, cfg.alloc_backend);
    fmt::print(
        "  derived: use_swwc={} batch_size(radix)={} buffer_blocks(batched)={} buffer_bytes(batched)={}  ({:.1f} MiB)\n",
        use_swwc,
        batch_size,
        buffer_blocks,
        buffer_bytes,
        static_cast<double>(buffer_bytes) / 1048576.0);
    fmt::print("  cap_init={} cap_max={}\n\n", cap_init, cap_max);

    fmt::print("Generating stream ({} blocks x {} rows x {} cols)...\n", total_blocks, cfg.block_rows, cfg.K);
    const auto tg0 = Clk::now();
    const BlockStream master = genBlocks(total_blocks * cfg.block_rows, cfg.block_rows, cfg.K, 42ULL + static_cast<uint64_t>(cfg.K));
    fmt::print("  {:.2f} s\n\n", std::chrono::duration<double>(Clk::now() - tg0).count());

    std::vector<double> ns_vals(static_cast<size_t>(cfg.reps));
    // radix path keeps PartState output; batched path uses IColumn output
    std::vector<std::vector<PartState>> radix_parts(static_cast<size_t>(cfg.T));
    std::vector<BatchedOutput> batched_out(static_cast<size_t>(cfg.T));
    // Per-thread timings from the last rep (batched only).
    std::vector<BatchedTimings> last_rep_timings(static_cast<size_t>(cfg.T));

    for (int rep = 0; rep < cfg.reps; ++rep)
    {
        std::vector<BumpArena> arenas;
        arenas.reserve(static_cast<size_t>(cfg.T));
        for (int t = 0; t < cfg.T; ++t)
            arenas.emplace_back(kArenaSlabBytes);
        for (int t = 0; t < cfg.T; ++t)
            radix_parts[static_cast<size_t>(t)].clear();

        const auto t0 = Clk::now();
        std::vector<ThreadFromGlobalPool> ths;
        ths.reserve(static_cast<size_t>(cfg.T));
        for (int t = 0; t < cfg.T; ++t)
        {
            ths.emplace_back(
                [&, t]()
                {
                    pinThread(t);
                    const size_t off = static_cast<size_t>(t) * blocks_per_thread;
                    const std::span<const DB::Columns> slice(master.data() + off, blocks_per_thread);
                    if (cfg.variant == "radix")
                        runSmartRadix(
                            slice, cfg.K, cfg.P, radix_parts[static_cast<size_t>(t)], arenas[static_cast<size_t>(t)], cap_init, cap_max);
                    else
                        last_rep_timings[static_cast<size_t>(t)] = runBatchedRadix(
                            slice,
                            cfg.K,
                            cfg.P,
                            batched_out[static_cast<size_t>(t)],
                            cfg.batch_max_blocks,
                            cfg.batch_max_bytes,
                            cfg.alloc_backend == "aligned_alloc");
                });
        }
        for (auto & th : ths)
            th.join();

        ns_vals[static_cast<size_t>(rep)]
            = std::chrono::duration<double>(Clk::now() - t0).count() * 1e9 / static_cast<double>(actual_total);
        fmt::print("  rep {:2d}: {:.3f} ns/row\n", rep, ns_vals[static_cast<size_t>(rep)]);
    }

    const double pmin = *std::min_element(ns_vals.begin(), ns_vals.end());
    const double pavg = std::accumulate(ns_vals.begin(), ns_vals.end(), 0.0) / static_cast<double>(cfg.reps);

    fmt::print(
        "\nSUMMARY variant={} K={} P={} T={}: pmin={:.3f} ns/row  pavg={:.3f} ns/row\n", cfg.variant, cfg.K, cfg.P, cfg.T, pmin, pavg);

    if (cfg.variant == "batched")
    {
        BatchedTimings agg;
        for (const auto & t : last_rep_timings)
        {
            agg.pid_compute_ns += t.pid_compute_ns;
            agg.histogram_ns += t.histogram_ns;
            agg.buffer_push_ns += t.buffer_push_ns;
            agg.clone_empty_ns += t.clone_empty_ns;
            agg.reserve_resize_ns += t.reserve_resize_ns;
            agg.on_grow_ns += t.on_grow_ns;
            agg.move_into_pending_ns += t.move_into_pending_ns;
            agg.scatter_ns += t.scatter_ns;
            agg.fence_drain_ns += t.fence_drain_ns;
            agg.commit_ns += t.commit_ns;
            agg.reset_ns += t.reset_ns;
            agg.total_process_ns += t.total_process_ns;
            agg.total_flush_ns += t.total_flush_ns;
            agg.flush_count += t.flush_count;
            agg.rows_processed += t.rows_processed;
            agg.alloc_count += t.alloc_count;
        }
        const auto nt = static_cast<double>(cfg.T);
        const double rows_pt = static_cast<double>(agg.rows_processed) / nt;
        const auto nspr = [&](uint64_t ns) { return static_cast<double>(ns) / nt / rows_pt; };
        const double proc_t = nspr(agg.total_process_ns);
        const double flush_t = nspr(agg.total_flush_ns);
        const double tot = proc_t + flush_t;
        const auto pct = [&](double v) { return tot > 0 ? 100.0 * v / tot : 0.0; };

        fmt::print("\nOperation breakdown (last rep, avg across {} threads):\n", cfg.T);
        fmt::print(
            "  Flushes/thread: {:.1f}  rows/thread: {}  alloc_calls/thread: {:.0f}\n",
            static_cast<double>(agg.flush_count) / nt,
            static_cast<size_t>(rows_pt),
            static_cast<double>(agg.alloc_count) / nt);
        fmt::print("  {:<26}  {:>9}  {:>7}\n", "Operation", "ns/row", "% total");
        fmt::print("  {:-<26}  {:->9}  {:->7}\n", "", "", "");
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "process: pid_compute", nspr(agg.pid_compute_ns), pct(nspr(agg.pid_compute_ns)));
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "process: histogram", nspr(agg.histogram_ns), pct(nspr(agg.histogram_ns)));
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "process: buffer_push", nspr(agg.buffer_push_ns), pct(nspr(agg.buffer_push_ns)));
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "flush: clone_empty", nspr(agg.clone_empty_ns), pct(nspr(agg.clone_empty_ns)));
        fmt::print(
            "  {:<26}  {:>9.4f}  {:>6.1f}%\n", "flush: reserve_resize", nspr(agg.reserve_resize_ns), pct(nspr(agg.reserve_resize_ns)));
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "flush: on_grow_raw", nspr(agg.on_grow_ns), pct(nspr(agg.on_grow_ns)));
        fmt::print(
            "  {:<26}  {:>9.4f}  {:>6.1f}%\n",
            "flush: move_into_pending",
            nspr(agg.move_into_pending_ns),
            pct(nspr(agg.move_into_pending_ns)));
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "flush: scatter", nspr(agg.scatter_ns), pct(nspr(agg.scatter_ns)));
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "flush: fence+drain", nspr(agg.fence_drain_ns), pct(nspr(agg.fence_drain_ns)));
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "flush: commit (move to out)", nspr(agg.commit_ns), pct(nspr(agg.commit_ns)));
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "flush: reset", nspr(agg.reset_ns), pct(nspr(agg.reset_ns)));
        fmt::print("  {:-<26}  {:->9}  {:->7}\n", "", "", "");
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "process() total", proc_t, pct(proc_t));
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "flush() total", flush_t, pct(flush_t));
        fmt::print("  {:<26}  {:>9.4f}  {:>6.1f}%\n", "INSTRUMENTED TOTAL", tot, 100.0);
        fmt::print("  wall pmin={:.4f} ns/row (instrumentation overhead ~{:.1f}%)\n", pmin, pmin > 0 ? 100.0 * (tot - pmin) / pmin : 0.0);
    }

    // Sanity-check output row count (thread 0, last rep)
    size_t out_rows = 0;
    if (cfg.variant == "radix")
    {
        for (const auto & sv : radix_parts[0])
            for (const OutBlock * b = sv.head; b != nullptr; b = b->next)
                out_rows += b->filled;
    }
    else
    {
        for (const auto & per_partition : batched_out[0])
            for (const auto & flush_block : per_partition)
                if (!flush_block.empty())
                    out_rows += flush_block[0]->size();
    }
    // Skip output-row sanity check when aligned_alloc backend is used: it
    // intentionally produces no IColumn output (buffers freed by destructor).
    const bool skip_sanity = cfg.variant == "batched" && cfg.alloc_backend == "aligned_alloc";
    if (!skip_sanity && out_rows != rows_per_thread)
        fmt::print("[ERROR] output rows {} != expected {}\n", out_rows, rows_per_thread);

    return 0;
}
