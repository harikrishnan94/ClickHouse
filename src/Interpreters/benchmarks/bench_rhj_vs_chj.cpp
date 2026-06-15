/// bench_rhj_vs_chj — stand-alone driver that links the real ClickHouse `dbms` and exercises the genuine
/// RadixHashJoin (`radix_hash`) and ConcurrentHashJoin (`parallel_hash`) build/probe code paths directly
/// through the `IJoin` API, for the reference query
///
///     SELECT count() FROM <probe> INNER JOIN <build> USING (k0)
///
/// with a single non-nullable UInt64 key, distinct build keys, and a clean all-match INNER join.
///
/// This is a research/reproduction harness only: it constructs and drives the production join classes
/// unchanged (no proxy, no copy of the data structures). The classes invoked, and the entry points used,
/// are (all paths relative to the repo root, verified against the current source):
///
///   RHJ  = DB::RadixHashJoin           src/Interpreters/RadixHashJoin/RadixHashJoin.h
///   CHJ  = DB::ConcurrentHashJoin      src/Interpreters/ConcurrentHashJoin.h  (wraps DB::HashJoin slots)
///   API  = DB::IJoin                   src/Interpreters/IJoin.h
///          addBlockToJoin(block, num_rows, check_limits, build_lane)   IJoin.h:115 / RadixHashJoin.h:62
///          onBuildPhaseFinish()                                        IJoin.h:190 / RadixHashJoin.h:80
///          joinBlock(block, lane) -> IJoinResult::next()               IJoin.h:130-134, 55-64
///
/// Engine construction mirrors the planner (src/Planner/PlannerJoins.cpp:1235 for CHJ, :1270 for RHJ).
///
/// The driver replaces the server's FillingRightJoinSideTransform (parallel build) and JoiningTransform
/// (parallel probe) with its own thread pool of pinned workers, so the join compute is measured in
/// isolation without running a full ClickHouse server. See tmp/bench/realbench/README.md for the full
/// citation table and documented divergences.

#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <Core/Joins.h>
#include <Core/Settings.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/ConcurrentHashJoin.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/RadixHashJoin/RadixHashJoin.h>
#include <Interpreters/TableJoin.h>
#include <Parsers/ASTTablesInSelectQuery.h>
#include <Common/HashTable/Hash.h>
#include <Common/ThreadStatus.h>

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>
#include <unistd.h>

namespace
{

using namespace DB;

/// The server in the reference setup is pinned to physical cores 0-47; pin every worker into that window.
constexpr int PHYS_CORES = 48;
/// One generated block carries this many rows (close to the server's default max_block_size).
constexpr UInt64 BLOCK_ROWS = 65536;

void pinToCore(int core)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

/// Run `fn(tid)` on `threads` workers, worker `t` pinned to core `first_core + (t % PHYS_CORES)`.
void runParallel(int threads, int first_core, const std::function<void(int)> & fn)
{
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([t, first_core, &fn]
        {
            pinToCore(first_core + (t % PHYS_CORES));
            fn(t);
        });
    for (auto & th : pool)
        th.join();
}

/// Data model (identical to the SQL reference): build_key[i] = intHash64(i) for i in [0, B) is a bijection,
/// so the build side holds B DISTINCT pseudo-random UInt64 keys; probe_key[j] = intHash64(j % B), so every
/// probe row matches exactly one build key, fan-out is exactly P/B, and count() == P. `intHash64` is the
/// Murmur finalizer in src/Common/HashTable/Hash.h.
UInt64 buildKey(UInt64 i)
{
    return intHash64(i);
}
UInt64 probeKey(UInt64 j, UInt64 build_rows)
{
    return intHash64(j % build_rows);
}

/// Build a one-column Block {k0 : UInt64} for rows [begin, begin + count): build-side keys when
/// `build_rows == 0`, probe-side keys (folded modulo `build_rows`) otherwise.
Block makeKeyBlock(UInt64 begin, UInt64 count, UInt64 build_rows)
{
    auto col = ColumnUInt64::create();
    auto & data = col->getData();
    data.resize(count);
    if (build_rows == 0)
        for (UInt64 r = 0; r < count; ++r)
            data[r] = buildKey(begin + r);
    else
        for (UInt64 r = 0; r < count; ++r)
            data[r] = probeKey(begin + r, build_rows);

    Block block;
    block.insert(ColumnWithTypeAndName(std::move(col), std::make_shared<DataTypeUInt64>(), "k0"));
    return block;
}

/// Construct the `TableJoin` describing `... INNER JOIN ... USING (k0)` with one non-nullable UInt64 key.
/// The `Settings`-based constructor (src/Interpreters/TableJoin.h:265) populates every block-sizing knob
/// from defaults and sets `enable_analyzer = true`, matching a real query; we then set kind/strictness,
/// the single key clause, and the column lists. This satisfies the RadixHashJoin key gate
/// (src/Planner/PlannerJoins.cpp:1174 radixHashJoinApplicable): one disjunct, Inner, ALL, fixed-width
/// non-nullable key of packed width 8 (a multiple of 4 in [4, 64]).
std::shared_ptr<TableJoin> makeTableJoin()
{
    Settings settings;
    auto table_join = std::make_shared<TableJoin>(settings, /*tmp_volume_=*/ nullptr, /*tmp_data_=*/ nullptr);

    table_join->getTableJoin().kind = JoinKind::Inner;
    table_join->getTableJoin().strictness = JoinStrictness::All;

    /// `getOnlyClause()` asserts the clause is already non-empty, so populate the freshly added clause via
    /// `getClauses()` (the same field `StorageJoin`'s ctor writes directly).
    table_join->addDisjunct();
    auto & clause = table_join->getClauses()[0];
    clause.key_names_left = {"k0"};
    clause.key_names_right = {"k0"};

    const NamesAndTypesList key_cols{{"k0", std::make_shared<DataTypeUInt64>()}};
    /// right joined columns, left-table column NameSet, right-table prefix, left-table columns.
    table_join->setColumnsFromJoinedTable(key_cols, {"k0"}, "", key_cols);

    return table_join;
}

struct Args
{
    std::string engine = "rhj"; /// rhj | chj
    UInt64 build_rows = 100'000'000ULL;
    UInt64 probe_rows = 0; /// 0 -> fanout * build_rows
    double fanout = 4.0;
    int threads = 48;
    int repeats = 1;
    int first_core = 0;
    /// `radix_hash_join_size_tables_by_distinct_estimate`: current production default is true
    /// (src/Core/Settings.cpp). -1 -> use that default; 0/1 -> force off/on. RHJ only.
    int distinct_estimate = -1;
    bool header = false;
};

UInt64 parseU64(const char * s)
{
    return std::strtoull(s, nullptr, 10);
}

double nowMs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

/// One build+probe of the reference query against the real engine. Returns matched row count;
/// fills build_ms (addBlockToJoin across workers + onBuildPhaseFinish) and probe_ms (joinBlock across
/// workers). Mirrors the proxy's build_ms/probe_ms split.
UInt64 runOnce(const Args & args, JoinPtr join, double & build_ms, double & probe_ms)
{
    const UInt64 build_blocks = (args.build_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    const UInt64 probe_blocks = (args.probe_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    /// ---- Build phase: parallel addBlockToJoin, one build lane per worker (lock-free in RHJ). ----
    auto build_start = std::chrono::steady_clock::now();
    {
        std::atomic<UInt64> next_block{0};
        runParallel(args.threads, args.first_core, [&](int tid)
        {
            for (;;)
            {
                const UInt64 b = next_block.fetch_add(1, std::memory_order_relaxed);
                if (b >= build_blocks)
                    break;
                const UInt64 begin = b * BLOCK_ROWS;
                const UInt64 count = std::min<UInt64>(BLOCK_ROWS, args.build_rows - begin);
                Block block = makeKeyBlock(begin, count, /*build_rows=*/ 0);
                join->addBlockToJoin(block, count, /*check_limits=*/ false, /*build_lane=*/ static_cast<size_t>(tid));
            }
        });
    }
    /// The eager post-build (scatter + leaf-table build for RHJ; map merge for CHJ) is build-phase work.
    join->onBuildPhaseFinish();
    auto build_end = std::chrono::steady_clock::now();
    build_ms = nowMs(build_start, build_end);

    /// ---- Probe phase: parallel joinBlock, drain IJoinResult::next(), count matched rows. ----
    std::vector<UInt64> partial(args.threads, 0);
    auto probe_start = std::chrono::steady_clock::now();
    {
        std::atomic<UInt64> next_block{0};
        runParallel(args.threads, args.first_core, [&](int tid)
        {
            UInt64 count = 0;
            for (;;)
            {
                const UInt64 b = next_block.fetch_add(1, std::memory_order_relaxed);
                if (b >= probe_blocks)
                    break;
                const UInt64 begin = b * BLOCK_ROWS;
                const UInt64 n = std::min<UInt64>(BLOCK_ROWS, args.probe_rows - begin);
                Block block = makeKeyBlock(begin, n, /*build_rows=*/ args.build_rows);
                JoinResultPtr res = join->joinBlock(std::move(block), static_cast<size_t>(tid));
                for (;;)
                {
                    IJoinResult::JoinResultBlock jb = res->next();
                    count += jb.block.rows();
                    if (jb.is_last)
                        break;
                }
            }
            partial[tid] = count;
        });
    }
    auto probe_end = std::chrono::steady_clock::now();
    probe_ms = nowMs(probe_start, probe_end);

    UInt64 matches = 0;
    for (UInt64 c : partial)
        matches += c;
    return matches;
}

JoinPtr makeJoin(const Args & args, const std::shared_ptr<TableJoin> & table_join, const SharedHeader & right_header)
{
    if (args.engine == "rhj")
    {
        const bool size_by_distinct = args.distinct_estimate < 0 ? true : (args.distinct_estimate != 0);
        return std::make_shared<RadixHashJoin>(
            table_join,
            right_header,
            static_cast<size_t>(args.threads),
            /*rhs_size_estimation_=*/ std::optional<UInt64>(args.build_rows),
            /*max_partitions_per_pass_=*/ 8192,
            /*size_tables_by_distinct_estimate_=*/ size_by_distinct,
            StatsCollectingParams{});
    }
    return std::make_shared<ConcurrentHashJoin>(
        table_join, static_cast<size_t>(args.threads), right_header, StatsCollectingParams{});
}

}

int main(int argc, char ** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--engine") args.engine = next();
        else if (a == "--build") args.build_rows = parseU64(next());
        else if (a == "--probe") args.probe_rows = parseU64(next());
        else if (a == "--fanout") args.fanout = std::strtod(next(), nullptr);
        else if (a == "--threads") args.threads = static_cast<int>(parseU64(next()));
        else if (a == "--repeats") args.repeats = static_cast<int>(parseU64(next()));
        else if (a == "--first-core") args.first_core = static_cast<int>(parseU64(next()));
        else if (a == "--distinct-estimate") args.distinct_estimate = static_cast<int>(parseU64(next()));
        else if (a == "--header") args.header = true;
        else if (a == "--help")
        {
            fmt::print(
                "Usage: bench_rhj_vs_chj --engine rhj|chj [--build N] [--probe M | --fanout F]\n"
                "       [--threads T] [--repeats R] [--first-core C] [--distinct-estimate 0|1] [--header]\n");
            return 0;
        }
        else
        {
            fmt::print(stderr, "unknown arg: {}\n", a);
            return 2;
        }
    }

    if (args.probe_rows == 0)
        args.probe_rows = static_cast<UInt64>(args.fanout * static_cast<double>(args.build_rows));
    args.threads = std::max(args.threads, 1);
    if (args.engine != "rhj" && args.engine != "chj")
    {
        fmt::print(stderr, "engine must be rhj or chj\n");
        return 2;
    }

    /// Give the main thread a ThreadStatus so the engines' internal ThreadPools (which capture the current
    /// thread group via ThreadGroupSwitcher) behave; a null query group is tolerated by both engines.
    DB::MainThreadStatus::getInstance();

    const double fanout = static_cast<double>(args.probe_rows) / static_cast<double>(args.build_rows);

    if (args.header)
        fmt::print("engine,build_rows,probe_rows,fanout,threads,repeat,build_ms,probe_ms,total_ms,matches,ok\n");

    const char * de = args.engine == "rhj"
        ? (args.distinct_estimate < 0 ? "default(on)" : (args.distinct_estimate ? "on" : "off"))
        : "n/a";
    fmt::print(
        stderr,
        "[{}] build={} probe={} fanout={:.2f} threads={} repeats={} distinct_estimate={}\n",
        args.engine, args.build_rows, args.probe_rows, fanout, args.threads, args.repeats, de);

    const SharedHeader right_header = std::make_shared<const DB::Block>(makeKeyBlock(0, 0, 0).cloneEmpty());

    for (int rep = 0; rep < args.repeats; ++rep)
    {
        auto table_join = makeTableJoin();
        JoinPtr join = makeJoin(args, table_join, right_header);

        double build_ms = 0.0;
        double probe_ms = 0.0;
        const UInt64 matches = runOnce(args, join, build_ms, probe_ms);
        const double total_ms = build_ms + probe_ms;
        const bool ok = (matches == args.probe_rows);

        fmt::print(
            "{},{},{},{:.4f},{},{},{:.3f},{:.3f},{:.3f},{},{}\n",
            args.engine, args.build_rows, args.probe_rows, fanout, args.threads, rep,
            build_ms, probe_ms, total_ms, matches, ok ? 1 : 0);
        (void)std::fflush(stdout); /// stream CSV rows promptly during long matrix runs

        fmt::print(
            stderr,
            "  rep {}: build {:.1f} ms, probe {:.1f} ms, total {:.1f} ms, matches={} {}\n",
            rep, build_ms, probe_ms, total_ms, matches, ok ? "OK" : "MISMATCH!");
    }
    return 0;
}
