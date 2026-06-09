/// Build/scatter benchmarks for bench_radix_hash_join. Every bench drives the production path
/// (`BuildStore::add` / `finishBuild` / `scatterToLeaves` over the real `GrowingArena` / `PartitionConfig`),
/// timed with `Stopwatch`. Registered into the driver's registry via `scatterBenches()`.

#include <Interpreters/tests/bench_radix_hash_common.h>

#include <atomic>
#include <random>
#include <span>
#include <thread>
#include <vector>

namespace
{

using namespace RHJBench;

/// 100M-row single-pass scatter, single UInt64 key: wall ns/row/pass for the deferred build scatter.
void scatterNsPerRow()
{
    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, 8192);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::mt19937_64 rng(0xBE0C); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blocks.push_back(makeBlock1<UInt64>(keys, 0, b + 1));
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();
    store.finishBuild();

    CoopPool coord;
    LeafArrays leaves;
    Stopwatch sw;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });
    const double wall_ns = static_cast<double>(sw.elapsedNanoseconds());

    const double ns_per_row = wall_ns / static_cast<double>(n);
    fmt::print(
        "P3 build scatter: n={} leaves={} passes={} threads={} wall={:.1f}ms ns/row/pass(wall)={:.3f}\n",
        n, cfg.num_leaves, cfg.pass_bits.size(), num_threads, wall_ns / 1e6,
        ns_per_row / static_cast<double>(cfg.pass_bits.size()));

    checkEq(leaves.leaf_rows.size(), cfg.num_leaves, "leaf count mismatch");
}

/// 100M-row work-stolen `add()` (build-select) path: wall ns/row.
void addNsPerRow()
{
    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, 8192);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::mt19937_64 rng(0xBE0C); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blocks.push_back(makeBlock1<UInt64>(keys, 0, b + 1));
    }

    std::atomic<size_t> next{0};
    Stopwatch sw;
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();
    const double add_ns = static_cast<double>(sw.elapsedNanoseconds());

    fmt::print(
        "P3 add (build-select): n={} leaves={} threads={} wall={:.1f}ms ns/row(wall)={:.3f}\n",
        n, cfg.num_leaves, num_threads, add_ns / 1e6, add_ns / static_cast<double>(n));

    /// `numBlocks()` is only populated by `finishBuild()` (the timed add() loop above is done).
    store.finishBuild();
    checkEq(store.numBlocks(), num_blocks, "block count mismatch");
}

/// Forced two-pass {6,5} scatter (cap=64), 100M rows, single UInt64 key.
void scatterTwoPass()
{
    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, /*max_partitions_per_pass=*/64);
    checkEq(cfg.pass_bits.size(), 2u, "config must force a two-pass scatter");
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::mt19937_64 rng(0xBE1D); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blocks.push_back(makeBlock1<UInt64>(keys, 0, b + 1));
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();
    store.finishBuild();

    CoopPool coord;
    LeafArrays leaves;
    Stopwatch sw;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });
    const double wall_ns = static_cast<double>(sw.elapsedNanoseconds());

    const double ns_per_row_per_pass = wall_ns / static_cast<double>(n) / static_cast<double>(cfg.pass_bits.size());
    fmt::print(
        "P3 scatter (2-pass): n={} leaves={} passes={} threads={} wall={:.1f}ms ns/row/pass(wall)={:.3f}\n",
        n, cfg.num_leaves, cfg.pass_bits.size(), num_threads, wall_ns / 1e6, ns_per_row_per_pass);

    checkEq(leaves.leaf_rows.size(), cfg.num_leaves, "leaf count mismatch");
}

/// Wide composite-key (4x UInt64 = 32 B packed), 100M rows, single-pass.
void scatterWideKey()
{
    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, 8192);
    checkEq(cfg.pass_bits.size(), 1u, "config must be single-pass");
    const std::vector<size_t> kpos{0, 1, 2, 3};
    const std::vector<size_t> kw_arr{8, 8, 8, 8};
    BuildStore store(cfg, kpos, kw_arr, num_threads);

    std::mt19937_64 rng(0xFACE); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<std::vector<UInt64>> keys(4, std::vector<UInt64>(rows));
        for (size_t c = 0; c < 4; ++c)
            for (size_t i = 0; i < rows; ++i)
                keys[c][i] = rng();
        blocks.push_back(makeBlock<UInt64>(keys, 0, b));
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();
    store.finishBuild();

    CoopPool coord;
    LeafArrays leaves;
    Stopwatch sw;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });
    const double wall_ns = static_cast<double>(sw.elapsedNanoseconds());

    fmt::print(
        "P3 scatter wide-key (4x8B=32B): n={} leaves={} passes={} threads={} wall={:.1f}ms ns/row/pass(wall)={:.3f}\n",
        n, cfg.num_leaves, cfg.pass_bits.size(), num_threads, wall_ns / 1e6,
        wall_ns / static_cast<double>(n) / static_cast<double>(cfg.pass_bits.size()));

    checkEq(leaves.leaf_rows.size(), cfg.num_leaves, "leaf count mismatch");
}

/// Forced 3-pass {4,4,3} depth-first scatter (cap=16), 100M rows, single UInt64 key.
void scatterThreePass()
{
    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, /*max_partitions_per_pass=*/16);
    checkEq(cfg.pass_bits.size(), 3u, "config must force a three-pass scatter");
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::mt19937_64 rng(0xF3E7); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blocks.push_back(makeBlock1<UInt64>(keys, 0, b + 1));
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();
    store.finishBuild();

    CoopPool coord;
    LeafArrays leaves;
    Stopwatch sw;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });
    const double wall_ns = static_cast<double>(sw.elapsedNanoseconds());

    fmt::print(
        "P3 scatter 3-pass: n={} leaves={} passes={} threads={} wall={:.1f}ms ns/row/pass(wall)={:.3f}\n",
        n, cfg.num_leaves, cfg.pass_bits.size(), num_threads, wall_ns / 1e6,
        wall_ns / static_cast<double>(n) / static_cast<double>(cfg.pass_bits.size()));

    checkEq(leaves.leaf_rows.size(), cfg.num_leaves, "leaf count mismatch");
}

/// End-to-end build: for each of four mixed-width key configs (K = 1,2,4,8; packed 8/16/32/64) at 100M
/// rows on 16 threads, time work-stolen add(), finishBuild() and scatterToLeaves() separately.
void endToEndBuild()
{
    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, 8192); /// 2048 leaves, 1 pass {11}
    checkEq(cfg.pass_bits.size(), 1u, "config must be single-pass");

    const std::vector<std::vector<size_t>> configs = {
        {8},
        {4, 12},
        {4, 8, 16, 4},
        {4, 8, 12, 16, 4, 8, 8, 4},
    };

    for (const auto & widths : configs)
    {
        const std::vector<size_t> kpos = keyPositions(widths.size());
        const std::vector<size_t> & kw = widths;

        /// (1) Parallel block generation — setup, NOT timed.
        std::vector<Block> blocks = generateFixedStringBlocksParallel(widths, n, block_rows, num_threads, 0xE2E0000ull);

        BuildStore store(cfg, kpos, kw, num_threads);

        /// (2) Work-stolen add() across T threads — timed.
        std::atomic<size_t> next{0};
        Stopwatch sw_add;
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (size_t t = 0; t < num_threads; ++t)
            threads.emplace_back([&]
            {
                for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                    store.add(blocks[b]);
            });
        for (auto & th : threads)
            th.join();
        const double add_ns = static_cast<double>(sw_add.elapsedNanoseconds());

        /// (3) finishBuild() — timed.
        Stopwatch sw_finish;
        store.finishBuild();
        const double finish_ns = static_cast<double>(sw_finish.elapsedNanoseconds());

        /// (4) scatterToLeaves(coord, T) — timed. CoopPool construction is setup, outside the timer.
        CoopPool coord;
        Stopwatch sw_scatter;
        LeafArrays leaves;
        coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });
        const double scatter_ns = static_cast<double>(sw_scatter.elapsedNanoseconds());

        const double total_ns = add_ns + finish_ns + scatter_ns;

        const auto report = [&](const char * step, double step_ns)
        {
            fmt::print(
                "P3 e2e build: K={} key_bytes={} n={} leaves={} passes={} threads={} step={} wall={:.1f}ms ns/row={:.3f}\n",
                widths.size(), store.packedKeyWidth(), n, cfg.num_leaves, cfg.pass_bits.size(), num_threads,
                step, step_ns / 1e6, step_ns / static_cast<double>(n));
        };
        report("add", add_ns);
        report("finish", finish_ns);
        report("scatter", scatter_ns);
        report("total", total_ns);

        checkEq(leaves.leaf_rows.size(), cfg.num_leaves, "leaf count mismatch");
        checkEq(leaves.key_width, store.packedKeyWidth(), "packed key width mismatch");
    }
}

}

namespace RHJBench
{

std::span<const Bench> scatterBenches()
{
    static const std::vector<Bench> benches = {
        {"scatter", "100M single-pass build scatter ns/row/pass", noArgs(scatterNsPerRow)},
        {"add", "100M work-stolen add() (build-select) ns/row", noArgs(addNsPerRow)},
        {"scatter_two_pass", "forced two-pass {6,5} scatter, 100M rows", noArgs(scatterTwoPass)},
        {"scatter_wide_key", "wide composite key (4x8B=32B) single-pass scatter", noArgs(scatterWideKey)},
        {"scatter_three_pass", "forced three-pass {4,4,3} scatter, 100M rows", noArgs(scatterThreePass)},
        {"end_to_end", "end-to-end build (add/finish/scatter) over mixed-width keys", noArgs(endToEndBuild)},
    };
    return benches;
}

}
