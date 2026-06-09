/// bench_radix_hash_join — standalone RadixHashJoin benchmark driver.
///
/// Selects one benchmark by `argv[1]` (run with no args or `--list` to see them). Every benchmark goes
/// through the production path (`BuildStore`/`scatterToLeaves`/`buildLeafHashTables`/`collectMatches`).
/// This file holds the leaf-HT build + probe benches and `main`; the build/scatter benches live in
/// bench_radix_hash_scatter.cpp and are concatenated into the registry via `scatterBenches()`.

#include <Interpreters/tests/bench_radix_hash_common.h>

#include <atomic>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>

namespace
{

using namespace RHJBench;

/// Cell-conservation: build the full leaf-HT set and walk every occupied cell's chain (honoring the
/// singleton marker) — total visited must equal N (every build row reachable exactly once).
/// Flags: --rows (100M) --threads (16) --block-rows (65536) --max-parts (8192) --l2-bytes.
void cellConservation100M(std::span<char * const> args)
{
    const Flags flags(args);
    const size_t n = flags.size("rows", 100'000'000);
    const size_t num_threads = flags.size("threads", 16);
    const size_t block_rows = flags.size("block-rows", 65536);
    const size_t max_parts = flags.size("max-parts", 8192);
    const size_t l2 = flags.size("l2-bytes", l2_bytes);

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2, max_parts);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::vector<Block> blocks = makeRandomU64Blocks(n, block_rows, 0xCE11);
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
    LeafHashTables hts;
    Stopwatch sw;
    coopRun(coord, num_threads, [&]
    {
        leaves = store.scatterToLeaves(coord);
        hts = buildLeafHashTables(leaves, store.blockBase(), store.totalRows(), sizeof(UInt64), coord);
    });
    const double ht_ms = static_cast<double>(sw.elapsedNanoseconds()) / 1e6;

    /// Conservation: walk every occupied cell's chain across all leaves; total visited == N.
    /// next_chain may be nullptr for all-unique builds (hts.next_chain == nullptr <-> all-unique).
    UInt64 visited = 0;
    for (const LeafHT & ht : hts.leaves)
    {
        for (UInt64 b = 0; b < ht.num_buckets; ++b)
        {
            BuildRef cur = *reinterpret_cast<const BuildRef *>(ht.cells + b * leafCellBytes(sizeof(UInt64))); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            if (cur.row_no == RadixShuffle::INVALID_ROW)
                continue; /// empty cell
            if (leafIsSingleton(cur))
            {
                ++visited; /// single-row key: the head IS the only row, no next_chain (marker honored)
                continue;
            }
            /// Multi-row key: the head and every next_chain entry are flag-free.
            while (cur.row_no != RadixShuffle::INVALID_ROW)
            {
                ++visited;
                cur = ht.next_chain[leafFlat(cur, store.blockBase().data())];
            }
        }
    }
    fmt::print(
        "P4 cell conservation: n={} leaves={} ht_build={:.1f}ms visited={}\n", n, cfg.num_leaves, ht_ms, visited);
    checkEq(visited, UInt64(n), "every build row must be reachable exactly once via the chains");
}

/// Leaf-HT build wall time (random inserts are fault/TLB-bound; the jemalloc-backed arena faults its own
/// leaf arrays + HT cells per run).
/// Flags: --rows (100M) --threads (16) --block-rows (65536) --max-parts (8192) --l2-bytes.
void leafHtBuildTime(std::span<char * const> args)
{
    const Flags flags(args);
    const size_t n = flags.size("rows", 100'000'000);
    const size_t num_threads = flags.size("threads", 16);
    const size_t block_rows = flags.size("block-rows", 65536);
    const size_t max_parts = flags.size("max-parts", 8192);
    const size_t l2 = flags.size("l2-bytes", l2_bytes);

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2, max_parts);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::vector<Block> blocks = makeRandomU64Blocks(n, block_rows, 0x7777);
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
    LeafArrays la;
    LeafHashTables hts;
    Stopwatch sw;
    coopRun(coord, num_threads, [&]
    {
        la = store.scatterToLeaves(coord);
        hts = buildLeafHashTables(la, store.blockBase(), store.totalRows(), sizeof(UInt64), coord);
    });
    const double ms = static_cast<double>(sw.elapsedNanoseconds()) / 1e6;
    fmt::print(
        "P4 leaf-HT build jemalloc-parallel: n={} leaves={} wall={:.1f}ms ns/row={:.3f}\n",
        n, cfg.num_leaves, ms, ms * 1e6 / static_cast<double>(n));
    checkEq(hts.leaves.size(), cfg.num_leaves, "leaf count mismatch");
}

/// Singleton-marker probe/build micro-bench. Measures the leaf-HT build (`fillLeafT`) or the probe
/// (`collectMatches`) in isolation and single-threaded, so `perf stat` + `taskset -c <one core>` attribute
/// counters cleanly to one kernel; the measured region is wrapped by a `Stopwatch` so ns/row is exact.
///
/// Flags (all optional): --workload=U|M|D (M) --phase=build|probe (probe) --rows (30M)
/// --iters (3 build / 10 probe) --threads (1, single-thread for clean per-kernel counters)
/// --block-rows (65536) --max-parts (8192) --l2-bytes.
void probeBench(std::span<char * const> args)
{
    const Flags flags(args);
    const std::string_view workload = flags.str("workload", "M");
    const std::string_view phase = flags.str("phase", "probe");
    const size_t n = flags.size("rows", 30'000'000);
    const size_t iters = flags.size("iters", phase == "build" ? 3 : 10);
    const size_t num_threads = flags.size("threads", 1);
    const size_t block_rows = flags.size("block-rows", 65536);
    const size_t max_parts = flags.size("max-parts", 8192);
    const size_t l2 = flags.size("l2-bytes", l2_bytes);

    const std::vector<UInt64> keys = makeWorkloadKeys(workload, n, 0xB1A5ED);

    /// Distinct probe set (every distinct key once): for M this keeps the singleton branch the common
    /// case; for D it is the ~1000 long-chain heads. For U every key is already distinct.
    std::vector<UInt64> probe_keys;
    if (workload == "U")
        probe_keys = keys;
    else
    {
        std::unordered_set<UInt64> s(keys.begin(), keys.end());
        probe_keys.assign(s.begin(), s.end());
    }
    const size_t pn = probe_keys.size();

    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2, max_parts);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads); /// default --threads=1 for clean per-kernel counters
    addBlocksSerial(store, keys, num_blocks);
    store.finishBuild();

    CoopPool coord;
    LeafArrays leaves;

    if (phase == "build")
    {
        double total_ns = 0;
        size_t sink = 0;
        coopRun(coord, 1, [&]
        {
            leaves = store.scatterToLeaves(coord);
            for (size_t it = 0; it < iters; ++it)
            {
                Stopwatch sw;
                LeafHashTables hts = buildLeafHashTables(leaves, store.blockBase(), store.totalRows(), sizeof(UInt64), coord);
                total_ns += static_cast<double>(sw.elapsedNanoseconds());
                sink += hts.leaves.size() + (hts.next_chain != nullptr ? 1u : 0u);
            }
        });
        fmt::print(
            "RHJ_PERF phase=build workload={} build_rows={} iters={} total_ms={:.2f} ns_per_build_row={:.3f} sink={}\n",
            workload, n, iters, total_ns / 1e6, total_ns / static_cast<double>(n * iters), sink);
        return;
    }

    LeafHashTables hts;
    coopRun(coord, 1, [&]
    {
        leaves = store.scatterToLeaves(coord);
        hts = buildLeafHashTables(leaves, store.blockBase(), store.totalRows(), sizeof(UInt64), coord);
    });

    Block probe = makeU64Block(probe_keys);
    const std::vector<UInt64> hashes = computeHashes(probe, pn);
    const void * packed = probe.getByPosition(0).column->getRawData().data();

    std::vector<UInt32> left_rows;
    std::vector<BuildRef> refs;
    left_rows.reserve(n + pn);
    refs.reserve(n + pn);

    double total_ns = 0;
    size_t sink = 0;
    for (size_t it = 0; it < iters; ++it)
    {
        left_rows.clear();
        refs.clear();
        Stopwatch sw;
        collectMatches(
            sizeof(UInt64), hts.next_chain != nullptr,
            hts.leaves.data(), cfg.shift, cfg.total_bits, store.blockBase().data(),
            hashes.data(), packed, pn, left_rows, refs);
        total_ns += static_cast<double>(sw.elapsedNanoseconds());
        sink += refs.size();
    }
    const size_t out_rows = sink / iters;
    fmt::print(
        "RHJ_PERF phase=probe workload={} build_rows={} probe_rows={} out_rows={} has_chain={} iters={} "
        "total_ms={:.2f} ns_per_probe_row={:.3f} ns_per_out_row={:.3f}\n",
        workload, n, pn, out_rows, hts.next_chain != nullptr, iters, total_ns / 1e6,
        total_ns / static_cast<double>(pn * iters), total_ns / static_cast<double>(std::max<size_t>(1, sink)));
}

const std::vector<Bench> & leafBenches()
{
    static const std::vector<Bench> benches = {
        {"cell_conservation_100m", "leaf-HT build + chain conservation check (default 100M rows)", cellConservation100M},
        {"leaf_ht_build_time", "leaf-HT build wall time / ns-per-row (default 100M rows)", leafHtBuildTime},
        {"probe", "probe/build micro-bench (flags: --workload --phase --rows --iters)", probeBench},
    };
    return benches;
}

}

int main(int argc, char ** argv)
{
    std::vector<RHJBench::Bench> all = leafBenches();
    for (const auto & b : RHJBench::scatterBenches())
        all.push_back(b);
    return RHJBench::runBenchMain(std::span<char * const>(argv, static_cast<size_t>(argc)), all);
}
