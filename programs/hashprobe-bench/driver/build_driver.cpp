/// hashprobe-bench/driver/build_driver.cpp
///
/// Implementation of the build-phase driver.
///
/// Tasks implemented:
///   B.1 — TableJoin construction with fixed-hash-table gate
///   B.2 — Engine selection: HashJoin (build_threads==1) vs ConcurrentHashJoin (>1)
///   B.3 — Build worker pool with addBlockToJoin calls
///   B.4 — Build lifecycle: onBuildPhaseFinish / hasPostBuildPhase / runPostBuildPhase
///   B.5 — Post-build resolved-type gate (A2 fail-loudly)
///   B.6 — Strictness-preservation gate (A2b fail-loudly)

#include "build_driver.h"

// ── ClickHouse Interpreters ───────────────────────────────────────────────────
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/ConcurrentHashJoin.h>
#include <Interpreters/TableJoin.h>
#include <Interpreters/HashTablesStatistics.h>
#include <Core/Joins.h>
#include <QueryPipeline/SizeLimits.h>

// ── ClickHouse Core ───────────────────────────────────────────────────────────
#include <Core/Block.h>
#include <Core/ColumnWithTypeAndName.h>

// ── ClickHouse Columns ────────────────────────────────────────────────────────
#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnNullable.h>

// ── ClickHouse DataTypes ──────────────────────────────────────────────────────
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeNullable.h>

// ── Standard library ─────────────────────────────────────────────────────────
#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <ctime>
#include <unistd.h>

namespace DB::HashProbeBench
{

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace
{

/// Convert ConfigType strictness → ClickHouse JoinStrictness
JoinStrictness toJoinStrictness(StrictnessConfig s)
{
    switch (s)
    {
        case StrictnessConfig::ALL:      return JoinStrictness::All;
        case StrictnessConfig::ANY:      return JoinStrictness::Any;
        case StrictnessConfig::RIGHTANY: return JoinStrictness::RightAny;
    }
    return JoinStrictness::All;
}

/// Convert ClickHouse JoinStrictness → harness string
std::string joinStrictnessToString(JoinStrictness s)
{
    switch (s)
    {
        case JoinStrictness::Unspecified: return "UNSPECIFIED";
        case JoinStrictness::RightAny:    return "RIGHTANY";
        case JoinStrictness::Any:         return "ANY";
        case JoinStrictness::All:         return "ALL";
        case JoinStrictness::Asof:        return "ASOF";
        case JoinStrictness::Semi:        return "SEMI";
        case JoinStrictness::Anti:        return "ANTI";
    }
    return "UNKNOWN";
}

/// Probe (left) key column names: "k0", "k1", ..., "k{n-1}" (matching BlockBuilder).
Names makeKeyNames(uint32_t n)
{
    Names names;
    names.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
        names.push_back("k" + std::to_string(i));
    return names;
}

/// Build (right) key column names: "b_k0", "b_k1", ... to avoid collision with probe side.
Names makeBuildKeyNames(uint32_t n)
{
    Names names;
    names.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
        names.push_back("b_k" + std::to_string(i));
    return names;
}

/// B.1: Construct a TableJoin for Inner join with the given key shape.
/// Uses the SizeLimits constructor so enable_join_fixed_hash_table_conversion
/// stays at its in-class default of false (A2 requirement).
std::shared_ptr<TableJoin> makeTableJoin(const ConfigType & config)
{
    Names left_key_names  = makeKeyNames(config.key_columns);       // probe side: k0, k1, ...
    Names right_key_names = makeBuildKeyNames(config.key_columns); // build side: b_k0, b_k1, ...

    // SizeLimits-based constructor initialises:
    //   enable_join_fixed_hash_table_conversion = false  (in-class default, A2)
    //   allow_join_sorting = false                       (in-class default)
    auto table_join = std::make_shared<TableJoin>(
        SizeLimits{},                               // no size limits
        false,                                      // join_use_nulls
        JoinKind::Inner,
        toJoinStrictness(config.strictness),
        right_key_names                             // populates clause[0].key_names_right
    );
    // Left keys use probe column names (k0 etc.)
    table_join->setLeftKeys(left_key_names);

    // Add build-side (right table) output columns: key columns + payload.
    // This causes joinBlock to append build-side columns to the probe block output,
    // matching the oracle SQL schema: probe.k0, probe.payload, build.k0, build.payload.
    {
        DataTypePtr key_dt = (config.key_width == KeyWidth::W32)
            ? std::static_pointer_cast<IDataType>(std::make_shared<DataTypeUInt32>())
            : std::static_pointer_cast<IDataType>(std::make_shared<DataTypeUInt64>());
        if (config.key_nullable)
            key_dt = std::make_shared<DataTypeNullable>(key_dt);
        auto payload_dt = std::make_shared<DataTypeUInt64>();

        for (uint32_t i = 0; i < config.key_columns; ++i)
            table_join->addJoinedColumn(NameAndTypePair{"b_k" + std::to_string(i), key_dt});
        table_join->addJoinedColumn(NameAndTypePair{"b_payload", payload_dt});
    }

    // B.1 gate: verify fixed-hash-table conversion is disabled (A2).
    if (table_join->enableJoinFixedHashTableConversion())
    {
        std::cerr << "[HARNESS_ERROR] unsupported_config: enableJoinFixedHashTableConversion=true\n";
        std::exit(1);
    }

    return table_join;
}

/// Resolve the HashJoin::Type for a given config without adding any data.
/// Creates a short-lived reference HashJoin whose constructor sets data->type.
/// Safe for ConcurrentHashJoin path (where sub-join types aren't public).
std::string resolveMapTypeString(
    const std::shared_ptr<TableJoin> & table_join,
    const Block & sample_block,
    bool use_two_level_maps = false)
{
    // ConcurrentHashJoin creates sub-joins with use_two_level_maps=true (ConcurrentHashJoin.cpp:219),
    // so pass the same flag to get the correct resolved type for the artifact.
    auto sample_header = std::make_shared<const Block>(sample_block);
    HashJoin ref_hj(table_join, sample_header,
                    /*any_take_last_row=*/false, /*reserve_num=*/0,
                    /*instance_id=*/"", use_two_level_maps);

#ifdef HARNESS_OBSERVABILITY
    return hashJoinTypeToString(ref_hj.getResolvedMapType());
#else
    return hashJoinTypeToString(ref_hj.getJoinedData()->type);
#endif
}

} // anonymous namespace

// ── Public helpers ────────────────────────────────────────────────────────────

std::string hashJoinTypeToString(HashJoin::Type type)
{
    switch (type)
    {
        case HashJoin::Type::EMPTY: return "EMPTY";
        case HashJoin::Type::CROSS: return "CROSS";
#define M(NAME) case HashJoin::Type::NAME: return #NAME;
        APPLY_FOR_JOIN_VARIANTS(M)
#undef M
    }
    return "unknown";
}

bool isAllowedMapType(const std::string & type_str)
{
    static const std::set<std::string> kAllowed = {
        "key32", "key64", "keys128", "keys256",
        "two_level_key32", "two_level_key64",
        "two_level_keys128", "two_level_keys256"
    };
    return kAllowed.count(type_str) > 0;
}

std::string checkMapTypeGate(const std::string & type_str)
{
    if (!isAllowedMapType(type_str))
        return "[HARNESS_ERROR] unsupported_config: resolved_map_type=" + type_str;
    return "";
}

std::string checkStrictnessGate(const std::string & at_construction,
                                const std::string & after_build)
{
    if (at_construction == "ALL" && after_build != "ALL")
        return "[HARNESS_ERROR] unsupported_config: all_unique_keys_with_all_strictness_would_silently_promote_to_rightany";
    return "";
}

std::string strictnessConfigToString(StrictnessConfig s)
{
    switch (s)
    {
        case StrictnessConfig::ALL:      return "ALL";
        case StrictnessConfig::ANY:      return "ANY";
        case StrictnessConfig::RIGHTANY: return "RIGHTANY";
    }
    return "UNKNOWN";
}

// ── Block construction ────────────────────────────────────────────────────────

Block makeRightSampleBlock(const ConfigType & config)
{
    Block block;
    const uint32_t n        = config.key_columns;
    const uint32_t w        = static_cast<uint32_t>(config.key_width);
    const bool     nullable = config.key_nullable;

    for (uint32_t i = 0; i < n; ++i)
    {
        DataTypePtr base_type = (w == 32)
            ? std::static_pointer_cast<IDataType>(std::make_shared<DataTypeUInt32>())
            : std::static_pointer_cast<IDataType>(std::make_shared<DataTypeUInt64>());

        DataTypePtr type;
        ColumnPtr   col;
        if (nullable)
        {
            type = std::make_shared<DataTypeNullable>(base_type);
            col  = type->createColumn();
        }
        else
        {
            type = base_type;
            col  = type->createColumn();
        }
        block.insert(ColumnWithTypeAndName{std::move(col), type,
                                           "b_k" + std::to_string(i)});
    }

    // Payload column — named "b_payload" to avoid collision with probe-side "payload".
    auto payload_type = std::make_shared<DataTypeUInt64>();
    block.insert(ColumnWithTypeAndName{
        payload_type->createColumn(), payload_type, "b_payload"});

    return block;
}

Block makeBuildBlock(
    const ConfigType & config,
    size_t             num_rows,
    uint64_t           start_key,
    bool               duplicate_keys)
{
    Block block;
    const uint32_t n        = config.key_columns;
    const uint32_t w        = static_cast<uint32_t>(config.key_width);
    const bool     nullable = config.key_nullable;

    // Half-period for duplicate mode (must be at least 1)
    const size_t half = (num_rows >= 2) ? (num_rows / 2) : 1;

    for (uint32_t i = 0; i < n; ++i)
    {
        ColumnPtr   col;
        DataTypePtr type;

        if (w == 32)
        {
            auto raw = ColumnUInt32::create(num_rows);
            auto & data = raw->getData();
            for (size_t j = 0; j < num_rows; ++j)
            {
                uint64_t k = duplicate_keys ? (j % half) : (start_key + j);
                data[j] = static_cast<UInt32>(k & 0xFFFF'FFFF);
            }

            if (nullable)
            {
                auto null_map = ColumnUInt8::create(num_rows, static_cast<UInt8>(0)); // all not-null
                col  = ColumnNullable::create(std::move(raw), std::move(null_map));
                type = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt32>());
            }
            else
            {
                col  = std::move(raw);
                type = std::make_shared<DataTypeUInt32>();
            }
        }
        else
        {
            auto raw = ColumnUInt64::create(num_rows);
            auto & data = raw->getData();
            for (size_t j = 0; j < num_rows; ++j)
            {
                uint64_t k = duplicate_keys ? (j % half) : (start_key + j);
                data[j] = k;
            }

            if (nullable)
            {
                auto null_map = ColumnUInt8::create(num_rows, static_cast<UInt8>(0));
                col  = ColumnNullable::create(std::move(raw), std::move(null_map));
                type = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt64>());
            }
            else
            {
                col  = std::move(raw);
                type = std::make_shared<DataTypeUInt64>();
            }
        }

        block.insert(ColumnWithTypeAndName{std::move(col), type,
                                           "b_k" + std::to_string(i)});
    }

    // Payload column: sequential row index (named "b_payload" to match right sample block).
    auto payload_col = ColumnUInt64::create(num_rows);
    auto & payload_data = payload_col->getData();
    for (size_t j = 0; j < num_rows; ++j)
        payload_data[j] = start_key + j;

    block.insert(ColumnWithTypeAndName{
        std::move(payload_col),
        std::make_shared<DataTypeUInt64>(),
        "b_payload"});

    return block;
}

// ── runBuildDriver ────────────────────────────────────────────────────────────

BuildDriverOutput runBuildDriver(
    const ConfigType &         config,
    const std::vector<Block> & build_blocks,
    uint64_t                   build_distinct_keys)
{
    BuildDriverOutput output;
    BuildResult &     result    = output.result;
    auto &            lifecycle = output.lifecycle_log;

    // ── B.1: Record strictness at construction ────────────────────────────────
    const std::string sc_at_construction = strictnessConfigToString(config.strictness);
    result.strictness_at_construction    = sc_at_construction;

    // ── B.6 pre-check: ALL strictness + all-unique keys would silently promote ─
    if (config.strictness == StrictnessConfig::ALL && build_distinct_keys > 0)
    {
        uint64_t total_rows = 0;
        for (const auto & blk : build_blocks)
            total_rows += blk.rows();

        if (build_distinct_keys == total_rows)
        {
            std::cerr << "[HARNESS_ERROR] all_unique_keys_with_all_strictness_would_silently_promote_to_rightany\n";
            std::exit(1);
        }
    }

    // ── B.1: Construct TableJoin ──────────────────────────────────────────────
    auto table_join = makeTableJoin(config);

    // ── B.1: Right sample block (schema only) ─────────────────────────────────
    Block  sample_block  = makeRightSampleBlock(config);
    auto   sample_header = std::make_shared<const Block>(sample_block);

    // ── B.5 early: Determine resolved type via reference HashJoin ────────────
    // The type is set in the HashJoin constructor (before any addBlockToJoin).
    // Pre-checking here avoids building a full join for unsupported schemas.
    const std::string pre_type = resolveMapTypeString(table_join, sample_block,
                                                       /*use_two_level_maps=*/config.build_threads > 1);
    {
        const std::string err = checkMapTypeGate(pre_type);
        if (!err.empty())
        {
            std::cerr << err << "\n";
            std::exit(1);
        }
    }

    // ── B.2: Engine selection (G1) ────────────────────────────────────────────
    const bool is_concurrent = config.build_threads > 1;
    uint32_t   slots         = 1;

    if (is_concurrent)
    {
        uint32_t max_probe_threads = 1;
        if (!config.probe_max_threads_sweep.empty())
            max_probe_threads = *std::max_element(
                config.probe_max_threads_sweep.begin(),
                config.probe_max_threads_sweep.end());
        slots = std::max<uint32_t>(config.build_threads, max_probe_threads);

        // ConcurrentHashJoin manages internal locking; StatsCollectingParams{}
        // disables stats collection (key=0 ⇒ isCollectionAndUseEnabled()=false).
        output.join = std::make_shared<ConcurrentHashJoin>(
            table_join,
            slots,
            sample_header,
            StatsCollectingParams{}   // no stats collection for harness
        );
        output.join_engine = "ConcurrentHashJoin";
        output.slots       = slots;
    }
    else
    {
        output.join = std::make_shared<HashJoin>(table_join, sample_header);
        output.join_engine = "HashJoin";
        output.slots       = 1;
    }

    // H1: record build-phase wall and CPU time (wraps B.3 + B.4)
    struct timespec t0_build_wall{}, t0_build_cpu{};
    clock_gettime(CLOCK_MONOTONIC_RAW,     &t0_build_wall);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0_build_cpu);

    // ── B.3: Build worker pool ────────────────────────────────────────────────
    std::atomic<uint64_t> add_block_calls{0};
    const bool stderr_is_tty = isatty(STDERR_FILENO) != 0;
    const size_t total_build_blocks = build_blocks.size();

    if (!is_concurrent)
    {
        // Single-thread path: sequential iteration.
        for (size_t i = 0; i < build_blocks.size(); ++i)
        {
            output.join->addBlockToJoin(build_blocks[i], true);
            const std::string ev = "addBlockToJoin#" + std::to_string(i);
            lifecycle.push_back(ev);
            if (stderr_is_tty)
                std::cerr << "\raddBlockToJoin [" << (i + 1) << "/" << total_build_blocks << "]" << std::flush;
            add_block_calls.fetch_add(1, std::memory_order_relaxed);
        }
        if (stderr_is_tty)
            std::cerr << "\n";
        else
            std::cerr << "addBlockToJoin: " << total_build_blocks << " blocks\n";
    }
    else
    {
        // Multi-thread path: thread-safe queue drained by build_threads workers.
        // ConcurrentHashJoin::addBlockToJoin handles internal locking; no
        // external lock is placed around the call (spec B.3).
        std::mutex                             queue_mutex;
        std::mutex                             log_mutex;
        std::queue<std::pair<size_t, Block>>   block_queue;

        for (size_t i = 0; i < build_blocks.size(); ++i)
            block_queue.push({i, build_blocks[i]});

        auto worker = [&]()
        {
            while (true)
            {
                std::pair<size_t, Block> item;
                bool got = false;
                {
                    std::lock_guard<std::mutex> g(queue_mutex);
                    if (!block_queue.empty())
                    {
                        item = std::move(block_queue.front());
                        block_queue.pop();
                        got = true;
                    }
                }
                if (!got)
                    break;

                // No external lock — ConcurrentHashJoin is thread-safe (spec B.3).
                output.join->addBlockToJoin(item.second, true);
                add_block_calls.fetch_add(1, std::memory_order_relaxed);

                const std::string ev = "addBlockToJoin#" + std::to_string(item.first);
                {
                    std::lock_guard<std::mutex> lg(log_mutex);
                    lifecycle.push_back(ev);
                    if (stderr_is_tty)
                    {
                        const size_t done = add_block_calls.load(std::memory_order_relaxed);
                        std::cerr << "\raddBlockToJoin [" << done << "/" << total_build_blocks << "]" << std::flush;
                    }
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(config.build_threads);
        for (uint32_t t = 0; t < config.build_threads; ++t)
            threads.emplace_back(worker);
        for (auto & th : threads)
            th.join();
        if (stderr_is_tty)
            std::cerr << "\n";
        else
            std::cerr << "addBlockToJoin: " << total_build_blocks << " blocks\n";
    }

    result.add_block_calls = add_block_calls.load(std::memory_order_relaxed);

    // ── B.4: Build lifecycle ──────────────────────────────────────────────────
    output.join->onBuildPhaseFinish();
    lifecycle.push_back("onBuildPhaseFinish");
    std::cerr << "onBuildPhaseFinish\n";

    const bool has_post = output.join->hasPostBuildPhase();
    {
        const std::string ev = "hasPostBuildPhase=" + std::string(has_post ? "true" : "false");
        lifecycle.push_back(ev);
        std::cerr << ev << "\n";
    }

    if (has_post)
    {
        output.join->runPostBuildPhase();
        lifecycle.push_back("runPostBuildPhase");
        std::cerr << "runPostBuildPhase\n";
        result.post_build_ran = true;
    }

    // H1: compute build-phase wall and CPU timing
    {
        struct timespec t1_build_wall{}, t1_build_cpu{};
        clock_gettime(CLOCK_MONOTONIC_RAW,     &t1_build_wall);
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1_build_cpu);
        auto ts_diff_ms = [](const struct timespec & a, const struct timespec & b)
        {
            return static_cast<double>(
                (b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec)
            ) / 1e6;
        };
        result.build_wall_ms = ts_diff_ms(t0_build_wall, t1_build_wall);
        result.build_cpu_ms  = ts_diff_ms(t0_build_cpu,  t1_build_cpu);
    }

    // ── B.5: Post-build resolved-type gate (A2) ───────────────────────────────
    std::string type_str;
    if (!is_concurrent)
    {
        auto * hj = dynamic_cast<HashJoin *>(output.join.get());
        assert(hj != nullptr && "Expected HashJoin for single-thread engine");
#ifdef HARNESS_OBSERVABILITY
        type_str = hashJoinTypeToString(hj->getResolvedMapType());
#else
        type_str = hashJoinTypeToString(hj->getJoinedData()->type);
#endif
    }
    else
    {
        // ConcurrentHashJoin's sub-join types are not publicly accessible.
        // Re-use the pre-computed type from the reference HashJoin: the type
        // is set in the constructor and does NOT change for our config
        // (enable_join_fixed_hash_table_conversion = false prevents conversion,
        //  allow_join_sorting = false prevents re-ranging).
        type_str = pre_type;
    }

    {
        const std::string err = checkMapTypeGate(type_str);
        if (!err.empty())
        {
            std::cerr << err << "\n";
            std::exit(1);
        }
    }
    result.resolved_map_type = type_str;

    // ── B.6: Strictness-preservation gate (A2b) ───────────────────────────────
    std::string sc_after_build;
    if (!is_concurrent)
    {
        auto * hj = dynamic_cast<HashJoin *>(output.join.get());
        assert(hj != nullptr);
#ifdef HARNESS_OBSERVABILITY
        sc_after_build = joinStrictnessToString(hj->getStrictnessAfterBuild());
#else
        sc_after_build = joinStrictnessToString(hj->getStrictness());
#endif
    }
    else
    {
        // ConcurrentHashJoin does not expose per-sub-join strictness.
        // Use table_join->strictness() which reflects the original setting.
        sc_after_build = joinStrictnessToString(
            output.join->getTableJoin().strictness());
    }
    result.strictness_after_build = sc_after_build;

    {
        const std::string err = checkStrictnessGate(sc_at_construction, sc_after_build);
        if (!err.empty())
        {
            std::cerr << err << "\n";
            std::exit(1);
        }
    }

    // ── Populate remaining result fields ──────────────────────────────────────
    result.build_rows = 0;
    for (const auto & blk : build_blocks)
        result.build_rows += blk.rows();

    result.build_distinct_keys =
        (build_distinct_keys > 0) ? build_distinct_keys
                                   : output.join->getTotalRowCount();

    if (result.build_distinct_keys > 0)
        result.build_row_to_key_ratio =
            static_cast<double>(result.build_rows) /
            static_cast<double>(result.build_distinct_keys);

    return output;
}

} // namespace DB::HashProbeBench
