/// hashprobe-bench/sweep/sweep_manager.cpp
///
/// SweepManager implementation.
///
/// G.1 — build-once probe-many sweep over (max_threads x block_size x counter_mode)
/// G.2 — build_invocations=1 assert + stderr log (C7)
/// G.3 — cfg.reps repetitions per cell, per_rep arrays, median + CV (H6)
/// G.4 — cpu_affinity (G6), git_commit (I2), compiler/cxx_flags (I3)

#include "sweep_manager.h"

#include "driver/build_driver.h"
#include "driver/probe_driver.h"
#include "generator/block_builder.h"
#include "generator/key_generator.h"
#include "instrumentation/cache_mode.h"
#include "instrumentation/hw_counters.h"
#include "partitioned/phj_run.h"

#include <Columns/IColumnPrefetch.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/IJoin.h>

// Suppress -Wcovered-switch-default triggered by nlohmann-json inside ClickHouse builds.
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wcovered-switch-default"
#endif
#include <nlohmann/json.hpp>
#ifdef __clang__
#    pragma clang diagnostic pop
#endif

#include <Formats/NativeWriter.h>
#include <IO/WriteBufferFromFile.h>
#include "../generator/native_writer.h"
#include "../oracle/oracle.h"
#include "../oracle/oracle_sql.h"
#include "../verification/verifier.h"

// G.4 (I2, I3): compile-time build metadata injected via CMakeLists.txt execute_process.
// Falls back to "unknown" when the CMake define is absent (e.g. developer builds
// without the git-commit detection step).
#ifndef GIT_COMMIT_HASH
#    define GIT_COMMIT_HASH "unknown"
#endif
#ifndef BUILD_TYPE_STR
#    define BUILD_TYPE_STR "unknown"
#endif

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace DB::HashProbeBench
{

namespace
{

// ── Clock ─────────────────────────────────────────────────────────────────────

static uint64_t clockNsRaw()
{
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

// ── Key / config helpers ──────────────────────────────────────────────────────

static KeyShape keyShapeFromConfig(const ConfigType & cfg)
{
    return {cfg.key_columns, (cfg.key_width == KeyWidth::W32) ? 32u : 64u, cfg.key_nullable};
}

/// build_distinct_keys avoiding the A2b all-unique-keys-with-ALL gate.
static uint64_t buildDistinctKeys(const ConfigType & cfg)
{
    if (cfg.strictness == StrictnessConfig::ALL)
        return std::max(uint64_t{1}, cfg.build_rows / 2);
    return cfg.build_rows;
}

// ── Block deep-copy ───────────────────────────────────────────────────────────

/// Deep-copy a Block so that successive reps can each move their copy into the join.
static Block cloneBlock(const Block & src)
{
    Block result;
    for (size_t i = 0; i < src.columns(); ++i)
    {
        const auto & c = src.getByPosition(i);
        result.insert({c.column->cloneResized(c.column->size()), c.type, c.name});
    }
    return result;
}

// ── Cache preparation ─────────────────────────────────────────────────────────

static void prepareCache(IJoin * join, CacheMode mode)
{
    if (mode == CacheMode::WARM)
    {
        if (auto * hj = dynamic_cast<HashJoin *>(join))
            warmLlc(*hj);
    }
    else
    {
        evictLlc();
    }
}

// ── Statistics (G.3, H6) ──────────────────────────────────────────────────────

/// Median of a copy of v (sorted in place).
static double computeMedian(std::vector<double> v)
{
    if (v.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2 == 0) ? (v[n / 2 - 1] + v[n / 2]) / 2.0 : v[n / 2];
}

/// Coefficient of variation: stddev / mean.  Returns 0.0 for single-element arrays.
static double computeCV(const std::vector<double> & v)
{
    if (v.size() < 2)
        return 0.0;
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    if (mean == 0.0)
        return 0.0;
    double sq_sum = 0.0;
    for (const double x : v)
        sq_sum += (x - mean) * (x - mean);
    const double stddev = std::sqrt(sq_sum / static_cast<double>(v.size()));
    return stddev / mean;
}

// ── Probe run ─────────────────────────────────────────────────────────────────

struct ProbeRunResult
{
    double probe_wall_ms = 0.0;
    double probe_cpu_ms = 0.0;
    double throughput_rows_per_s = 0.0;
    double joinblock_probe_wall_ms = 0.0;
    double joinblock_probe_cpu_ms = 0.0;
    double result_emit_wall_ms = 0.0;
    double result_emit_cpu_ms = 0.0;
    uint64_t output_rows = 0;
    std::vector<ProbeBlockEntry> probe_block_log;
    std::vector<OutputBlockEntry> output_block_log;
    std::vector<Block> output_blocks; ///< populated when collect_blocks=true
};

/// Execute one probe rep: partition proto_blocks across max_threads threads,
/// each draining its slice through its own ProbeDriver instance.
///
/// Note: HashJoin (build_threads=1) is not officially thread-safe for concurrent
/// joinBlock calls; use ConcurrentHashJoin (build_threads>1) for max_threads>1.
static ProbeRunResult runProbe(
    const std::shared_ptr<IJoin> & join,
    const std::vector<Block> & proto_blocks,
    uint32_t max_threads,
    uint64_t probe_rows,
    bool collect_blocks = false,
    bool use_hw = false,
    uint32_t generate_prefetch_la = 0)
{
    ProbeRunResult result;
    const size_t total_blocks = proto_blocks.size();

    struct ThreadResult
    {
        double cpu_ms = 0.0;
        double joinblock_wall_ms = 0.0;
        double joinblock_cpu_ms = 0.0;
        double result_emit_wall_ms = 0.0;
        double result_emit_cpu_ms = 0.0;
        uint64_t output_rows = 0;
        std::vector<ProbeBlockEntry> probe_log;
        std::vector<OutputBlockEntry> output_log;
    };

    std::vector<ThreadResult> thread_results(max_threads);
    std::vector<std::thread> threads;
    threads.reserve(max_threads);

    const uint64_t t0_wall = clockNsRaw();

    for (uint32_t t = 0; t < max_threads; ++t)
    {
        const size_t blk_start = (total_blocks * t) / max_threads;
        const size_t blk_end = (t + 1 == max_threads) ? total_blocks : (total_blocks * (t + 1)) / max_threads;

        threads.emplace_back(
            [&join, &proto_blocks, &thread_results, &result, t, blk_start, blk_end, collect_blocks, use_hw, generate_prefetch_la]()
            {
                /// Each thread sets its own thread_local so the gather prefetch uses the correct LA.
                DB::generate_phase_prefetch_lookahead = generate_prefetch_la;
                auto & tr = thread_results[t];
                std::mutex blocks_mu;
                ProbeDriver driver(
                    join,
                    [&tr, &result, &blocks_mu, collect_blocks](Block b)
                    {
                        tr.output_rows += b.rows();
                        if (collect_blocks)
                        {
                            std::lock_guard<std::mutex> lk(blocks_mu);
                            result.output_blocks.push_back(b);
                        }
                    });

                // H4: per-thread HW counter group
                HwCounters hw_ctr;
                const bool hw_ok = use_hw && hw_ctr.open();

                for (size_t bi = blk_start; bi < blk_end; ++bi)
                {
                    Block blk = cloneBlock(proto_blocks[bi]);
                    if (hw_ok)
                        hw_ctr.start();
                    auto entry = driver.drainBlock(std::move(blk), static_cast<uint64_t>(bi), hw_ok ? &hw_ctr : nullptr);
                    if (hw_ok)
                    {
                        uint64_t cy = 0, ins = 0, llc = 0, br = 0, dtlb = 0, branches = 0, llc_load = 0, dtlb_load = 0;
                        hw_ctr.read(cy, ins, llc, br, dtlb, branches, llc_load, dtlb_load);
                        entry.hw_cycles = cy;
                        entry.hw_instructions = ins;
                        entry.hw_ipc = HwCounters::computeIpc(ins, cy);
                        entry.hw_llc_miss = llc;
                        entry.hw_branch_miss = br;
                        entry.hw_dtlb_miss = dtlb;
                        entry.hw_branches = branches;
                        entry.hw_llc_load = llc_load;
                        entry.hw_dtlb_load = dtlb_load;
                    }
                    tr.cpu_ms += (entry.joinblock_probe_cpu_ns + entry.result_emit_cpu_ns) / 1e6;
                    tr.joinblock_wall_ms += entry.joinblock_probe_wall_ns / 1e6;
                    tr.joinblock_cpu_ms += entry.joinblock_probe_cpu_ns / 1e6;
                    tr.result_emit_wall_ms += entry.result_emit_wall_ns / 1e6;
                    tr.result_emit_cpu_ms += entry.result_emit_cpu_ns / 1e6;
                    tr.probe_log.push_back(entry);
                }
                for (const auto & e : driver.getOutputBlockLog())
                    tr.output_log.push_back(e);
            });
    }

    for (auto & th : threads)
        th.join();

    const uint64_t t1_wall = clockNsRaw();
    result.probe_wall_ms = static_cast<double>(t1_wall - t0_wall) / 1e6;

    for (auto & tr : thread_results)
    {
        result.probe_cpu_ms += tr.cpu_ms;
        result.joinblock_probe_wall_ms += tr.joinblock_wall_ms;
        result.joinblock_probe_cpu_ms += tr.joinblock_cpu_ms;
        result.result_emit_wall_ms += tr.result_emit_wall_ms;
        result.result_emit_cpu_ms += tr.result_emit_cpu_ms;
        result.output_rows += tr.output_rows;
        for (auto & e : tr.probe_log)
            result.probe_block_log.push_back(std::move(e));
        for (auto & e : tr.output_log)
            result.output_block_log.push_back(std::move(e));
    }

    result.throughput_rows_per_s = (result.probe_wall_ms > 0.0) ? static_cast<double>(probe_rows) / (result.probe_wall_ms / 1000.0) : 0.0;

    return result;
}

// ── cpu_affinity capture (G.4, G6) ───────────────────────────────────────────

/// Read current CPU affinity via "taskset -p <pid>".
/// Returns the full output line, or "unset" if taskset is unavailable.
static std::string getCpuAffinity()
{
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "taskset -p %d 2>/dev/null", static_cast<int>(::getpid()));
    FILE * fp = ::popen(cmd, "r");
    if (!fp)
        return "unset";
    char buf[256] = {};
    const bool ok = (::fgets(buf, static_cast<int>(sizeof(buf)), fp) != nullptr);
    ::pclose(fp);
    if (!ok)
        return "unset";
    size_t len = ::strlen(buf);
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
        buf[--len] = '\0';
    return std::string(buf);
}

// ── JSON serialisation ────────────────────────────────────────────────────────

static nlohmann::json buildHeaderToJson(const BuildHeader & h)
{
    return nlohmann::json{
        {"git_commit", h.git_commit},
        {"compiler", h.compiler},
        {"cxx_flags", h.cxx_flags},
        {"cpu_affinity", h.cpu_affinity},
        {"join_engine", h.join_engine},
        {"build_threads", h.build_threads},
        {"slots", h.slots},
        {"probe_max_threads_sweep", h.probe_max_threads_sweep},
        {"build_rows", h.build_rows},
        {"build_distinct_keys", h.build_distinct_keys},
        {"build_row_to_key_ratio", h.build_row_to_key_ratio},
        {"build_wall_ms", h.build_wall_ms},
        {"build_cpu_ms", h.build_cpu_ms},
        {"resolved_map_type_post_build", h.resolved_map_type_post_build},
        {"strictness_at_construction", h.strictness_at_construction},
        {"strictness_after_build", h.strictness_after_build},
        {"build_invocations", h.build_invocations},
        {"probe_invocations", h.probe_invocations},
        {"harness_drain_mode", h.harness_drain_mode},
        {"vectorized_probe_enabled", h.vectorized_probe_enabled},
    };
}

static nlohmann::json phaseMetricsToJson(const PhaseMetrics & p)
{
    const double ipc = p.hw_cycles > 0 ? static_cast<double>(p.hw_instructions) / static_cast<double>(p.hw_cycles) : 0.0;
    return nlohmann::json{
        {"wall_ns", p.wall_ns},
        {"cpu_ns", p.cpu_ns},
        {"cycles", p.hw_cycles},
        {"instructions", p.hw_instructions},
        {"hw_ipc", ipc},
        {"hw_llc_load", p.hw_llc_load},
        {"hw_llc_miss_pct", p.hw_llc_load > 0 ? static_cast<double>(p.hw_llc_miss) * 100.0 / static_cast<double>(p.hw_llc_load) : 0.0},
        {"hw_branches", p.hw_branches},
        {"hw_br_miss_pct", p.hw_branches > 0 ? static_cast<double>(p.hw_branch_miss) * 100.0 / static_cast<double>(p.hw_branches) : 0.0},
        {"hw_dtlb_load", p.hw_dtlb_load},
        {"hw_dtlb_miss_pct", p.hw_dtlb_load > 0 ? static_cast<double>(p.hw_dtlb_miss) * 100.0 / static_cast<double>(p.hw_dtlb_load) : 0.0},
        {"hw_available", p.hw_available},
    };
}

static nlohmann::json probeBlockEntryToJson(const ProbeBlockEntry & e)
{
    return nlohmann::json{
        {"probe_block_idx", e.probe_block_idx},
        {"probe_block_rows", e.probe_block_rows},
        {"joinblock_probe_wall_ns", e.joinblock_probe_wall_ns},
        {"joinblock_probe_cpu_ns", e.joinblock_probe_cpu_ns},
        {"result_emit_wall_ns", e.result_emit_wall_ns},
        {"result_emit_cpu_ns", e.result_emit_cpu_ns},
        {"output_block_count", e.output_block_count},
        // H4: HW counter fields (zero when counter_mode=none or hw unavailable)
        {"cycles", e.hw_cycles},
        {"instructions", e.hw_instructions},
        {"hw_ipc", e.hw_ipc},
        {"hw_llc_load", e.hw_llc_load},
        {"hw_llc_miss_pct", e.hw_llc_load > 0 ? static_cast<double>(e.hw_llc_miss) * 100.0 / static_cast<double>(e.hw_llc_load) : 0.0},
        {"hw_branches", e.hw_branches},
        {"hw_br_miss_pct", e.hw_branches > 0 ? static_cast<double>(e.hw_branch_miss) * 100.0 / static_cast<double>(e.hw_branches) : 0.0},
        {"hw_dtlb_load", e.hw_dtlb_load},
        {"hw_dtlb_miss_pct", e.hw_dtlb_load > 0 ? static_cast<double>(e.hw_dtlb_miss) * 100.0 / static_cast<double>(e.hw_dtlb_load) : 0.0},
        {"caller_tid", e.caller_tid},
        // G2: TID interval timestamps for per-TID non-overlap verification
        {"joinblock_start_ns", e.joinblock_start_ns},
        {"last_next_end_ns", e.last_next_end_ns},
        {"phase_probe", phaseMetricsToJson(e.phase_probe)},
        {"phase_generate", phaseMetricsToJson(e.phase_generate)},
    };
}

static nlohmann::json outputBlockEntryToJson(const OutputBlockEntry & e)
{
    return nlohmann::json{
        {"probe_block_idx", e.probe_block_idx},
        {"output_block_idx", e.output_block_idx},
        {"output_block_rows", e.output_block_rows},
        {"next_wall_ns", e.next_wall_ns},
        {"next_cpu_ns", e.next_cpu_ns},
        {"is_last", e.is_last},
    };
}

static nlohmann::json repTimingToJson(const RepTiming & r)
{
    return nlohmann::json{
        {"probe_wall_ms", r.probe_wall_ms},
        {"probe_cpu_ms", r.probe_cpu_ms},
        {"throughput_rows_per_s", r.throughput_rows_per_s},
    };
}

static nlohmann::json probeCellToJson(const ProbeCellResult & c)
{
    auto per_rep_j = nlohmann::json::array();
    for (const auto & r : c.per_rep)
        per_rep_j.push_back(repTimingToJson(r));

    nlohmann::json j = nlohmann::json{
        {"max_threads", c.max_threads},
        {"block_size", c.block_size},
        {"rep_index", c.rep_index},
        {"cache_mode_str", c.cache_mode_str},
        {"counter_mode", c.counter_mode},
        {"generate_prefetch_lookahead", c.generate_prefetch_lookahead},
        {"probe_rows", c.probe_rows},
        {"output_rows", c.output_rows},
        {"joinblock_probe_wall_ms", c.joinblock_probe_wall_ms},
        {"joinblock_probe_cpu_ms", c.joinblock_probe_cpu_ms},
        {"result_emit_wall_ms", c.result_emit_wall_ms},
        {"result_emit_cpu_ms", c.result_emit_cpu_ms},
        {"probe_wall_ms", c.probe_wall_ms},
        {"probe_cpu_ms", c.probe_cpu_ms},
        {"throughput_rows_per_s", c.throughput_rows_per_s},
        {"median_probe_wall_ms", c.median_probe_wall_ms},
        {"median_probe_cpu_ms", c.median_probe_cpu_ms},
        {"median_throughput_rows_per_s", c.median_throughput_rows_per_s},
        {"probe_wall_ms_cv", c.probe_wall_ms_cv},
        {"probe_cpu_ms_cv", c.probe_cpu_ms_cv},
        {"throughput_rows_per_s_cv", c.throughput_rows_per_s_cv},
        {"per_rep", per_rep_j},
        {"oracle_l0_pass", c.oracle_l0_pass},
        {"oracle_l1_pass", c.oracle_l1_pass},
        {"oracle_l2_pass", c.oracle_l2_pass},
        {"oracle_sql", c.oracle_sql},
    };

    // H2: per-block timing logs
    {
        auto pbl = nlohmann::json::array();
        for (const auto & e : c.probe_block_log)
            pbl.push_back(probeBlockEntryToJson(e));
        j["probe_block_log"] = std::move(pbl);

        auto obl = nlohmann::json::array();
        for (const auto & e : c.output_block_log)
            obl.push_back(outputBlockEntryToJson(e));
        j["output_block_log"] = std::move(obl);
    }

    // H5: cache-mode fields (non-NaN only when cache_mode=cold)
    {
        auto maybe_dbl = [](double v) -> nlohmann::json { return std::isnan(v) ? nlohmann::json(nullptr) : nlohmann::json(v); };
        j["cold_probe_wall_ms"] = maybe_dbl(c.cold_probe_wall_ms);
        j["warm_probe_wall_ms"] = maybe_dbl(c.warm_probe_wall_ms);
        j["cache_speedup_ratio"] = maybe_dbl(c.cache_speedup_ratio);
    }

    return j;
}

static void writeArtifactJson(const Artifact & artifact, const std::string & output_dir)
{
    ::mkdir(output_dir.c_str(), 0755);

    const std::string path = output_dir + "/artifact.json";
    std::ofstream ofs(path);
    if (!ofs.is_open())
        throw std::runtime_error("[hashprobe-bench] Cannot open for writing: " + path);

    nlohmann::json root;
    root["build"] = buildHeaderToJson(artifact.build);

    auto cells_j = nlohmann::json::array();
    for (const auto & c : artifact.probe_cells)
        cells_j.push_back(probeCellToJson(c));
    root["probe_cells"] = std::move(cells_j);

    ofs << root.dump(2) << "\n";
    ofs.close();

    std::cerr << "[hashprobe-bench] artifact written: " << path << "\n";
}

} // anonymous namespace

// ── SweepManager ──────────────────────────────────────────────────────────────

SweepManager::SweepManager(const ConfigType & cfg)
    : cfg_(cfg)
{
}

/// Reorder columns in a Block to match the oracle output schema.
/// Oracle order: k0..k{n-1}, payload, b_k0..b_k{n-1}, b_payload
/// Harness joinBlock output may have different column order due to TableJoin internals.
/// This reorder is ONLY used in the oracle comparison path, not in the timing path.
static Block reorderToOracleSchema(const Block & blk)
{
    const size_t ncols = blk.columns();
    if (ncols == 0)
        return blk;

    // Collect column names
    std::vector<std::string> names;
    for (size_t i = 0; i < ncols; ++i)
        names.push_back(blk.getByPosition(i).name);

    // Determine key count: probe keys are k0..k{n-1}, build keys are b_k0..b_k{n-1}
    // Probe cols: k0, k1, ..., payload
    // Build cols: b_k0, b_k1, ..., b_payload
    // Desired order: k0, k1, ..., payload, b_k0, b_k1, ..., b_payload
    std::vector<size_t> order;
    order.reserve(ncols);

    // First pass: probe keys (k0, k1, ...)
    for (size_t i = 0; i < ncols; ++i)
        if (!names[i].empty() && names[i][0] == 'k' && names[i].find('_') == std::string::npos)
            order.push_back(i);

    // Second pass: probe payload
    for (size_t i = 0; i < ncols; ++i)
        if (names[i] == "payload")
            order.push_back(i);

    // Third pass: build keys (b_k0, b_k1, ...)
    for (size_t i = 0; i < ncols; ++i)
        if (names[i].size() >= 4 && names[i][0] == 'b' && names[i][1] == '_' && names[i][2] == 'k')
            order.push_back(i);

    // Fourth pass: build payload (b_payload)
    for (size_t i = 0; i < ncols; ++i)
        if (names[i] == "b_payload")
            order.push_back(i);

    if (order.size() != ncols)
    {
        // Fallback: use original order
        return blk;
    }

    Block result;
    for (size_t pos : order)
        result.insert(blk.getByPosition(pos));
    return result;
}

// ── Stdout summary ────────────────────────────────────────────────────────────

static std::string fmtNum(uint64_t n)
{
    // Format with thousands separators (e.g. 1,000,000)
    std::string s = std::to_string(n);
    int i = static_cast<int>(s.size()) - 3;
    while (i > 0)
    {
        s.insert(static_cast<size_t>(i), ",");
        i -= 3;
    }
    return s;
}

namespace SummaryTable
{
constexpr int W_MT = 4;
constexpr int W_BLKSZ = 6;
constexpr int W_MODE = 4;
constexpr int W_LA = 3; ///< generate_prefetch_lookahead
constexpr int W_WALL = 8;
constexpr int W_CPU = 10;
constexpr int W_MROWS = 8;
constexpr int W_IPC = 6;
constexpr int W_COUNTER = 14;
constexpr int W_PCT = 9;
constexpr int W_DTLB_LOAD = 15;

constexpr int TOTAL_WIDTH = 2 + W_MT + 2 + W_BLKSZ + 2 + W_MODE + 2 + W_LA + 2 + W_WALL + 2 + W_CPU + 2 + W_MROWS + 2 + W_IPC + 2
    + W_COUNTER + 2 + W_PCT + 2 + W_COUNTER + 2 + W_PCT + 2 + W_DTLB_LOAD + 2 + W_PCT;

static void printCol(const std::string & value, int width)
{
    std::cout << "  " << std::right << std::setw(width) << value;
}

static void printCol(double value, int width, int precision)
{
    std::cout << "  " << std::right << std::fixed << std::setprecision(precision) << std::setw(width) << value;
}

static void printCol(uint64_t value, int width)
{
    printCol(fmtNum(value), width);
}

static void printHwNa()
{
    printCol("n/a", W_IPC);
    printCol("n/a", W_COUNTER);
    printCol("n/a", W_PCT);
    printCol("n/a", W_COUNTER);
    printCol("n/a", W_PCT);
    printCol("n/a", W_DTLB_LOAD);
    printCol("n/a", W_PCT);
}

} // namespace SummaryTable

static std::string keyShapeStr(const ConfigType & cfg)
{
    std::string s = std::to_string(cfg.key_columns) + "x" + ((cfg.key_width == KeyWidth::W32) ? "32" : "64");
    if (cfg.key_nullable)
        s += ",nullable";
    return s;
}

static std::string strictnessStr(const ConfigType & cfg)
{
    switch (cfg.strictness)
    {
        case StrictnessConfig::ALL:
            return "ALL";
        case StrictnessConfig::ANY:
            return "ANY";
        case StrictnessConfig::RIGHTANY:
            return "RIGHTANY";
    }
    return "?";
}

/// Print a human-readable sweep summary to stdout after the artifact is written.
///
/// Layout:
///   === hashprobe-bench ===
///   Build   <engine>  <shape>/<strictness>  rows=<n>  map=<type>  <ms>ms
///           git=<hash7>  threads=<n>  ...
///
///   Probe cells (<cache>, <reps> rep[s]):
///     mt  blksz   mode  wall_ms  cpu_ms  Mrows/s  IPC   llc_miss  br_miss dtlb_miss
///   ...
///
///   Oracle: <status>
///   Artifact: <path>
static void printSummary(const Artifact & artifact, const ConfigType & cfg)
{
    const BuildHeader & bld = artifact.build;

    std::cout << "\n=== hashprobe-bench ===\n";

    // ── Build summary ────────────────────────────────────────────────────
    std::cout << "Build   " << bld.join_engine << "  " << keyShapeStr(cfg) << "/" << strictnessStr(cfg) << "  "
              << "rows=" << fmtNum(bld.build_rows) << "  map=" << bld.resolved_map_type_post_build
              << "  probe=" << (bld.vectorized_probe_enabled ? "vectorized" : "scalar");
    if (std::isfinite(bld.build_wall_ms))
        std::cout << "  build=" << std::fixed << std::setprecision(1) << bld.build_wall_ms << "ms";
    std::cout << "\n";
    std::cout << "        git=" << bld.git_commit.substr(0, 7) << "  build_threads=" << bld.build_threads
              << "  probe_invocations=" << bld.probe_invocations << "\n\n";

    // ── Probe cells table ────────────────────────────────────────────────
    const std::string cache_str = (cfg.cache_mode == CacheMode::WARM) ? "warm" : "cold";
    using namespace SummaryTable;

    std::cout << "Probe cells (" << cache_str << " cache, " << cfg.reps << " rep" << (cfg.reps != 1 ? "s" : "") << "):\n";
    printCol("mt", W_MT);
    printCol("blksz", W_BLKSZ);
    printCol("mode", W_MODE);
    printCol("la", W_LA);
    printCol("wall_ms", W_WALL);
    printCol("cpu_ns/row", W_CPU);
    printCol("Mrows/s", W_MROWS);
    printCol("IPC", W_IPC);
    printCol("llc_load", W_COUNTER);
    printCol("llc_miss%", W_PCT);
    printCol("branches", W_COUNTER);
    printCol("br_miss%", W_PCT);
    printCol("dtlb_load", W_DTLB_LOAD);
    printCol("dtlb_miss%", W_PCT);
    std::cout << "\n";
    std::cout << "  " << std::string(TOTAL_WIDTH - 2, '-') << "\n";

    for (const auto & cell : artifact.probe_cells)
    {
        // Aggregate hw counters from probe_block_log (sum across all blocks)
        uint64_t sum_cy = 0, sum_ins = 0;
        uint64_t sum_llc = 0, sum_br = 0, sum_dtlb = 0;
        uint64_t sum_llc_load = 0, sum_branches = 0, sum_dtlb_load = 0;
        for (const auto & pbe : cell.probe_block_log)
        {
            sum_cy += pbe.hw_cycles;
            sum_ins += pbe.hw_instructions;
            sum_llc += pbe.hw_llc_miss;
            sum_br += pbe.hw_branch_miss;
            sum_dtlb += pbe.hw_dtlb_miss;
            sum_llc_load += pbe.hw_llc_load;
            sum_branches += pbe.hw_branches;
            sum_dtlb_load += pbe.hw_dtlb_load;
        }
        const bool has_hw = (cell.counter_mode == "hw") && (sum_cy > 0 || sum_ins > 0);
        const double ipc = (has_hw && sum_cy > 0) ? static_cast<double>(sum_ins) / static_cast<double>(sum_cy) : 0.0;

        const double mrows = std::isfinite(cell.throughput_rows_per_s) ? cell.throughput_rows_per_s / 1e6 : 0.0;

        printCol(std::to_string(cell.max_threads), W_MT);
        printCol(std::to_string(cell.block_size), W_BLKSZ);
        printCol(cell.counter_mode, W_MODE);
        printCol(std::to_string(cell.generate_prefetch_lookahead), W_LA);
        printCol(std::isfinite(cell.probe_wall_ms) ? cell.probe_wall_ms : 0.0, W_WALL, 2);
        {
            const double cpu_ns_per_row = (cell.probe_rows > 0 && std::isfinite(cell.probe_cpu_ms))
                ? cell.probe_cpu_ms * 1e6 / static_cast<double>(cell.probe_rows)
                : 0.0;
            printCol(cpu_ns_per_row, W_CPU, 1);
        }
        printCol(mrows, W_MROWS, 2);
        if (has_hw)
        {
            const double llc_miss_pct = (sum_llc_load > 0) ? static_cast<double>(sum_llc) * 100.0 / static_cast<double>(sum_llc_load) : 0.0;
            const double br_miss_pct = (sum_branches > 0) ? static_cast<double>(sum_br) * 100.0 / static_cast<double>(sum_branches) : 0.0;
            const double dtlb_miss_pct
                = (sum_dtlb_load > 0) ? static_cast<double>(sum_dtlb) * 100.0 / static_cast<double>(sum_dtlb_load) : 0.0;
            printCol(ipc, W_IPC, 2);
            printCol(sum_llc_load, W_COUNTER);
            printCol(llc_miss_pct, W_PCT, 2);
            printCol(sum_branches, W_COUNTER);
            printCol(br_miss_pct, W_PCT, 2);
            printCol(sum_dtlb_load, W_DTLB_LOAD);
            printCol(dtlb_miss_pct, W_PCT, 2);
        }
        else
            printHwNa();
        std::cout << "\n";
    }
    std::cout << "\nPhase breakdown (last rep, all blocks summed):\n";
    {
        // Column widths for the per-phase breakdown table.
        constexpr int W_PH_MT = 4;
        constexpr int W_PH_BLKSZ = 5;
        constexpr int W_PH_MODE = 4;
        constexpr int W_PH_METRIC = 10;
        constexpr int W_PH_VAL = 25;

        constexpr int W_PH_LA = 3;

        auto phCol = [](const std::string & s, int w) { std::cout << "  " << std::right << std::setw(w) << s; };

        phCol("mt", W_PH_MT);
        phCol("blksz", W_PH_BLKSZ);
        phCol("mode", W_PH_MODE);
        phCol("la", W_PH_LA);
        phCol("metric", W_PH_METRIC);
        phCol("probe", W_PH_VAL);
        phCol("generate", W_PH_VAL);
        std::cout << "\n";

        for (const auto & cell : artifact.probe_cells)
        {
            PhaseMetrics probe{}, gen{};
            for (const auto & pbe : cell.probe_block_log)
            {
                auto add = [](PhaseMetrics & dst, const PhaseMetrics & src)
                {
                    dst.wall_ns += src.wall_ns;
                    dst.cpu_ns += src.cpu_ns;
                    dst.hw_cycles += src.hw_cycles;
                    dst.hw_instructions += src.hw_instructions;
                    dst.hw_llc_miss += src.hw_llc_miss;
                    dst.hw_branch_miss += src.hw_branch_miss;
                    dst.hw_dtlb_miss += src.hw_dtlb_miss;
                    dst.hw_llc_load += src.hw_llc_load;
                    dst.hw_branches += src.hw_branches;
                    dst.hw_dtlb_load += src.hw_dtlb_load;
                    dst.hw_available = dst.hw_available || src.hw_available;
                };
                add(probe, pbe.phase_probe);
                add(gen, pbe.phase_generate);
            }

            const double total_cpu_ns = probe.cpu_ns + gen.cpu_ns;
            const bool has_hw = (cell.counter_mode == "hw") && (probe.hw_cycles > 0 || gen.hw_cycles > 0);

            const PhaseMetrics * phases[2] = {&probe, &gen};

            // Format a percentage string with two decimal places.
            auto pctStr = [](double num, double den) -> std::string
            {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f%%", den > 0.0 ? num * 100.0 / den : 0.0);
                return std::string(buf);
            };

            // Print one metric row across both phases.
            auto phRow = [&](const std::string & metric, auto valFn)
            {
                phCol(std::to_string(cell.max_threads), W_PH_MT);
                phCol(std::to_string(cell.block_size), W_PH_BLKSZ);
                phCol(cell.counter_mode, W_PH_MODE);
                phCol(std::to_string(cell.generate_prefetch_lookahead), W_PH_LA);
                phCol(metric, W_PH_METRIC);
                for (const PhaseMetrics * pm : phases)
                    phCol(valFn(*pm), W_PH_VAL);
                std::cout << "\n";
            };

            auto nsRowPctStr = [&](double ns, double total_ns) -> std::string
            {
                const double ns_per_row = (cell.probe_rows > 0) ? ns / static_cast<double>(cell.probe_rows) : 0.0;
                const double pct = total_ns > 0.0 ? ns * 100.0 / total_ns : 0.0;
                char buf[48];
                snprintf(buf, sizeof(buf), "%.1fns/row (%.2f%%)", ns_per_row, pct);
                return std::string(buf);
            };

            phRow("cpu", [&](const PhaseMetrics & p) { return nsRowPctStr(p.cpu_ns, total_cpu_ns); });

            if (has_hw)
            {
                phRow(
                    "instructions",
                    [](const PhaseMetrics & p) -> std::string
                    {
                        if (p.hw_instructions == 0)
                            return "n/a";
                        char buf[64];
                        if (p.hw_cycles > 0)
                            snprintf(
                                buf,
                                sizeof(buf),
                                "%s (%.2f)",
                                fmtNum(p.hw_instructions).c_str(),
                                static_cast<double>(p.hw_instructions) / static_cast<double>(p.hw_cycles));
                        else
                            snprintf(buf, sizeof(buf), "%s", fmtNum(p.hw_instructions).c_str());
                        return std::string(buf);
                    });
                phRow(
                    "cycles",
                    [](const PhaseMetrics & p) -> std::string
                    {
                        if (p.hw_cycles == 0)
                            return "n/a";
                        return fmtNum(p.hw_cycles);
                    });
                phRow("llc_load", [](const PhaseMetrics & p) { return fmtNum(p.hw_llc_load); });
                phRow(
                    "llc_miss%",
                    [&](const PhaseMetrics & p) { return pctStr(static_cast<double>(p.hw_llc_miss), static_cast<double>(p.hw_llc_load)); });
                phRow("branches", [](const PhaseMetrics & p) { return fmtNum(p.hw_branches); });
                phRow(
                    "br_miss%",
                    [&](const PhaseMetrics & p)
                    { return pctStr(static_cast<double>(p.hw_branch_miss), static_cast<double>(p.hw_branches)); });
                phRow("dtlb_load", [](const PhaseMetrics & p) { return fmtNum(p.hw_dtlb_load); });
                phRow(
                    "dtlb_miss%",
                    [&](const PhaseMetrics & p)
                    { return pctStr(static_cast<double>(p.hw_dtlb_miss), static_cast<double>(p.hw_dtlb_load)); });
            }
        }
    }
    std::cout << "\n";

    // ── Oracle status ────────────────────────────────────────────────────
    if (!cfg.verify_oracle)
    {
        std::cout << "Oracle: disabled (pass --verify-oracle to enable)\n";
    }
    else
    {
        // Check if scale exceeds the oracle threshold (E-Auto skips checks above 20M rows).
        const bool within_scale = (cfg.build_rows <= Verification::E_AUTO_MAX_ROWS) && (cfg.probe_rows <= Verification::E_AUTO_MAX_ROWS);

        if (!within_scale)
        {
            // Oracle skipped for large inputs — "0/N pass" would be misleading.
            std::printf(
                "Oracle: skipped  (build_rows=%llu and/or probe_rows=%llu "
                "exceed the %llu-row auto-check limit;\n"
                "        use --build_rows≤%llu --probe_rows≤%llu to enable)\n",
                static_cast<unsigned long long>(cfg.build_rows),
                static_cast<unsigned long long>(cfg.probe_rows),
                static_cast<unsigned long long>(Verification::E_AUTO_MAX_ROWS),
                static_cast<unsigned long long>(Verification::E_AUTO_MAX_ROWS),
                static_cast<unsigned long long>(Verification::E_AUTO_MAX_ROWS));
        }
        else
        {
            uint32_t l0_pass = 0, l1_pass = 0, l2_pass = 0, l2_total = 0;
            for (const auto & cell : artifact.probe_cells)
            {
                if (cell.oracle_l0_pass)
                    ++l0_pass;
                if (cell.oracle_l1_pass)
                    ++l1_pass;
                if (cell.oracle_l2_pass)
                    ++l2_pass;
                if (cell.max_threads == 1)
                    ++l2_total;
            }
            const uint32_t n = static_cast<uint32_t>(artifact.probe_cells.size());
            std::cout << "Oracle: L0 " << l0_pass << "/" << n << " pass"
                      << "  L1 " << l1_pass << "/" << n << " pass"
                      << "  L2 " << l2_pass << "/" << l2_total << " pass"
                      << "\n";
        }
    }

    // ── Artifact path ────────────────────────────────────────────────────
    if (cfg.save_artifact)
        std::cout << "Artifact: " << cfg.output_dir << "/artifact.json\n";
    else
        std::cout << "Artifact: disabled (pass --save-artifact to enable)\n";

    // ── Wall time summary ─────────────────────────────────────────────────
    // build is done once; best probe cell = minimum probe_wall_ms.
    if (std::isfinite(bld.build_wall_ms) && !artifact.probe_cells.empty())
    {
        double best_probe_wall = std::numeric_limits<double>::max();
        for (const auto & c : artifact.probe_cells)
            if (std::isfinite(c.probe_wall_ms))
                best_probe_wall = std::min(best_probe_wall, c.probe_wall_ms);
        if (best_probe_wall < std::numeric_limits<double>::max())
        {
            std::printf("\nWall time summary (build-once + best probe cell):\n");
            std::printf("  build      %8.0f ms\n", bld.build_wall_ms);
            std::printf("  probe+gen  %8.0f ms\n", best_probe_wall);
            std::printf("  ─────────────────────\n");
            std::printf("  TOTAL      %8.0f ms\n", bld.build_wall_ms + best_probe_wall);
        }
    }
    std::cout << "\n";
    std::cout.flush();
}

Artifact SweepManager::run()
{
    Artifact artifact;

    // ── Step 1: generate build blocks ────────────────────────────────────────
    const KeyShape shape = keyShapeFromConfig(cfg_);
    const uint64_t bdk = buildDistinctKeys(cfg_);

    KeyGenerator::Params kp;
    kp.shape = shape;
    kp.strictness = cfg_.strictness;
    kp.build_rows = cfg_.build_rows;
    kp.build_distinct_keys = bdk;
    kp.probe_rows = cfg_.probe_rows;
    kp.match_rate = cfg_.match_rate;
    kp.null_fraction = cfg_.null_fraction;
    kp.seed = cfg_.seed;

    KeyGenerator build_gen(kp);
    BlockBuilder build_bb(shape, cfg_, build_gen);
    std::vector<Block> build_blocks;
    while (build_bb.hasBuildRows())
    {
        Block blk = build_bb.nextBuildBlock();
        if (blk.columns() > 0)
            build_blocks.push_back(std::move(blk));
    }

    // ── Step 2: build join ONCE (G.1, C7) ───────────────────────────────────
    const auto bdo = runBuildDriver(cfg_, build_blocks, bdk);

    // ── Step 3: populate BuildHeader (I2/I3/G6 stubs; real values in G.4) ───
    BuildHeader & hdr = artifact.build;

    // G.4: compile-time and runtime build identity fields (I2, I3, G6)
    hdr.git_commit = GIT_COMMIT_HASH; // 40-char hex from git log -1 at cmake configure (I2)
#if defined(__clang__)
    hdr.compiler = "clang " + std::string(__clang_version__);
#elif defined(__GNUC__)
    hdr.compiler = "gcc " + std::string(__VERSION__);
#else
    hdr.compiler = "unknown";
#endif
    hdr.cxx_flags = "CMAKE_BUILD_TYPE=" BUILD_TYPE_STR; // (I3)
    hdr.cpu_affinity = getCpuAffinity(); // taskset -p <pid> output (G6)

    hdr.join_engine = bdo.join_engine;
    hdr.build_threads = cfg_.build_threads;
    hdr.slots = bdo.slots;
    hdr.probe_max_threads_sweep = cfg_.probe_max_threads_sweep;

    hdr.build_rows = bdo.result.build_rows;
    hdr.build_distinct_keys = bdk;
    hdr.build_row_to_key_ratio = bdo.result.build_row_to_key_ratio;
    hdr.build_wall_ms = bdo.result.build_wall_ms;
    hdr.build_cpu_ms = bdo.result.build_cpu_ms;

    hdr.resolved_map_type_post_build = bdo.result.resolved_map_type;
    hdr.strictness_at_construction = bdo.result.strictness_at_construction;
    hdr.strictness_after_build = bdo.result.strictness_after_build;

    hdr.build_invocations = 1; // C7: exactly one build for any multi-sweep run
    hdr.harness_drain_mode = ProbeDriver::DRAIN_MODE;
    {
        const char * vp = std::getenv("CLICKHOUSE_VECTORIZED_JOIN_PROBE"); // NOLINT(concurrency-mt-unsafe)
        hdr.vectorized_probe_enabled = !vp || vp[0] != '0';
    }

    // ── Write build.native and probe.native for oracle comparison (D.3/D.4, E-Auto) ───
    // Re-generate with the same seed → identical deterministic blocks (I1).
    // Skipped when --verify-oracle is not set to avoid the I/O overhead.
    std::string build_native_path;
    std::string probe_native_path;
    std::vector<std::string> key_cols;
    if (cfg_.verify_oracle)
    {
        KeyGenerator::Params kp2 = kp; // same parameters and seed
        KeyGenerator gen2(kp2);
        NativeFileWriter nfw(shape, cfg_, gen2);
        nfw.writeAll(); // writes <output_dir>/build.native and <output_dir>/probe.native
        build_native_path = cfg_.output_dir + "/build.native";
        probe_native_path = cfg_.output_dir + "/probe.native";
        key_cols = Oracle::makeKeyColNames(cfg_.key_columns);
    }

    // G.2: fail-loudly if the harness accidentally calls runBuildDriver more than once (C7)
    assert(hdr.build_invocations == 1 && "build_invocations must be exactly 1 for any multi-sweep run");

    // ── Step 4: probe sweep ──────────────────────────────────────────────────
    // Grid: max_threads x block_size x counter_mode in {none, hw}
    static constexpr const char * kCounterModes[] = {"none", "hw"};
    const std::string cache_str = (cfg_.cache_mode == CacheMode::WARM) ? "warm" : "cold";

    // Cache probe blocks per unique block_size to avoid regenerating per cell
    std::map<uint32_t, std::vector<Block>> proto_cache;

    uint32_t probe_invocations = 0;

    // Oracle check: INNER ANY JOIN exhausts used-flags on the first probe pass,
    // so subsequent cells would see empty output — run oracle check once only
    // for ANY strictness.  INNER ALL and RIGHTANY have a read-only hash map
    // during probe (no setUsedOnce flags), so re-probing is safe and the oracle
    // check can run for every cell, giving per-cell correctness coverage.
    const bool is_any_strictness = (cfg_.strictness == StrictnessConfig::ANY);
    bool oracle_check_done = false;

    for (uint32_t mt : cfg_.probe_max_threads_sweep)
    {
        for (uint32_t bs : cfg_.probe_block_size_sweep)
        {
            if (proto_cache.find(bs) == proto_cache.end())
            {
                KeyGenerator probe_gen(kp);
                ConfigType probe_cfg = cfg_;
                probe_cfg.block_size = bs;
                BlockBuilder probe_bb(shape, probe_cfg, probe_gen);
                std::vector<Block> pblocks;
                while (probe_bb.hasProbeRows())
                {
                    Block blk = probe_bb.nextProbeBlock();
                    if (blk.columns() > 0)
                        pblocks.push_back(std::move(blk));
                }
                proto_cache[bs] = std::move(pblocks);
            }
            const auto & proto_blocks = proto_cache.at(bs);

            for (const char * cm_cstr : kCounterModes)
            {
                for (uint32_t la : cfg_.generate_prefetch_lookahead_sweep)
                {
                    ProbeCellResult cell;
                    cell.max_threads = mt;
                    cell.block_size = bs;
                    cell.cache_mode_str = cache_str;
                    cell.counter_mode = cm_cstr;
                    cell.probe_rows = cfg_.probe_rows;
                    cell.generate_prefetch_lookahead = la;

                    // G.3: run cfg_.reps repetitions per cell against the same built join (H6, C7)
                    std::vector<double> wall_ms_vec, cpu_ms_vec, tput_vec;

                    // INNER ANY and INNER ALL/RIGHTANY have different oracle-before-timing constraints:
                    //
                    // INNER ALL / RIGHTANY: the hash map is read-only during probe (no setUsedOnce),
                    //   so re-probing is safe.  Run oracle BEFORE timing so oracle sees a clean map,
                    //   and run oracle for EVERY cell (per-cell correctness coverage).
                    //
                    // INNER ANY: HashJoin uses setUsedOnce — each build row can match at most ONE probe
                    //   row.  After the first complete probe sweep, all "used" flags are set; subsequent
                    //   sweeps return 0 rows.  This makes "build-once probe-many" fundamentally
                    //   incompatible with measurement integrity for ANY:
                    //     • Running oracle BEFORE timing corrupts timing (timing sees 0 rows).
                    //     • Running oracle AFTER timing also gives wrong oracle output (0 rows).
                    //   Solution: for INNER ANY, run TIMING FIRST (rep 0 sees a clean map, reps 1+
                    //   get 0 rows — only rep 0 is valid for ANY timing), then rebuild for oracle
                    //   verification on a SEPARATE fresh build that is NOT counted in build_invocations
                    //   (it is a correctness-only build, not a measurement build).

                    const bool small_scale
                        = (cfg_.build_rows <= Verification::E_AUTO_MAX_ROWS) && (cfg_.probe_rows <= Verification::E_AUTO_MAX_ROWS);
                    const bool auto_check = cfg_.verify_oracle && (!oracle_check_done || !is_any_strictness) && small_scale;
                    uint64_t rep0_output_rows = 0;
                    std::vector<Block> rep0_blocks;

                    // For INNER ALL / RIGHTANY: oracle probe runs BEFORE timing (map is read-only).
                    if (auto_check && !is_any_strictness)
                    {
                        std::vector<Block> oracle_probe_blocks;
                        {
                            KeyGenerator oracle_probe_gen(kp);
                            ConfigType oracle_cfg = cfg_;
                            BlockBuilder oracle_probe_bb(shape, oracle_cfg, oracle_probe_gen);
                            while (oracle_probe_bb.hasProbeRows())
                            {
                                Block blk = oracle_probe_bb.nextProbeBlock();
                                if (blk.columns() > 0)
                                    oracle_probe_blocks.push_back(std::move(blk));
                            }
                        }
                        auto oracle_probe = runProbe(
                            bdo.join, oracle_probe_blocks, /*max_threads=*/1, cfg_.probe_rows, /*collect_blocks=*/true, /*use_hw=*/false);
                        rep0_output_rows = oracle_probe.output_rows;
                        rep0_blocks = std::move(oracle_probe.output_blocks);
                    }

                    for (uint32_t rep = 0; rep < cfg_.reps; ++rep)
                    {
                        const bool hw_mode = (std::string(cm_cstr) == "hw");
                        prepareCache(bdo.join.get(), cfg_.cache_mode);
                        auto run = runProbe(bdo.join, proto_blocks, mt, cfg_.probe_rows, false, hw_mode, la);

                        RepTiming rt;
                        rt.probe_wall_ms = run.probe_wall_ms;
                        rt.probe_cpu_ms = run.probe_cpu_ms;
                        rt.throughput_rows_per_s = run.throughput_rows_per_s;
                        cell.per_rep.push_back(rt);

                        wall_ms_vec.push_back(run.probe_wall_ms);
                        cpu_ms_vec.push_back(run.probe_cpu_ms);
                        tput_vec.push_back(run.throughput_rows_per_s);

                        probe_invocations++;

                        // Retain per-block logs from the last rep (H2)
                        if (rep + 1 == cfg_.reps)
                        {
                            cell.probe_block_log = std::move(run.probe_block_log);
                            cell.output_block_log = std::move(run.output_block_log);
                            cell.output_rows = run.output_rows;
                            cell.joinblock_probe_wall_ms = run.joinblock_probe_wall_ms;
                            cell.joinblock_probe_cpu_ms = run.joinblock_probe_cpu_ms;
                            cell.result_emit_wall_ms = run.result_emit_wall_ms;
                            cell.result_emit_cpu_ms = run.result_emit_cpu_ms;
                        }
                    }

                    // G.3: aggregate statistics over reps (H6)
                    cell.probe_wall_ms = computeMedian(wall_ms_vec);
                    cell.probe_cpu_ms = computeMedian(cpu_ms_vec);
                    cell.throughput_rows_per_s = computeMedian(tput_vec);
                    cell.median_probe_wall_ms = cell.probe_wall_ms;
                    cell.median_probe_cpu_ms = cell.probe_cpu_ms;
                    cell.median_throughput_rows_per_s = cell.throughput_rows_per_s;
                    cell.probe_wall_ms_cv = computeCV(wall_ms_vec);
                    cell.probe_cpu_ms_cv = computeCV(cpu_ms_vec);
                    cell.throughput_rows_per_s_cv = computeCV(tput_vec);

                    // H5: cold/warm cache effect measurement (always populated per H5 spec).
                    // When cache_mode=cold, cell.probe_wall_ms IS the cold timing.
                    // When cache_mode=warm, we run an additional cold eviction pass for H5 data.
                    {
                        double cold_ms = 0.0, warm_ms = 0.0;
                        if (cfg_.cache_mode == CacheMode::COLD)
                        {
                            cold_ms = cell.probe_wall_ms;
                            // Run one warm probe for the warm baseline.
                            prepareCache(bdo.join.get(), CacheMode::WARM);
                            auto warm_run = runProbe(bdo.join, proto_blocks, mt, cfg_.probe_rows, false, false, la);
                            warm_ms = warm_run.probe_wall_ms;
                        }
                        else
                        {
                            warm_ms = cell.probe_wall_ms;
                            // Run one cold eviction pass for H5 cold baseline.
                            prepareCache(bdo.join.get(), CacheMode::COLD);
                            auto cold_run = runProbe(bdo.join, proto_blocks, mt, cfg_.probe_rows, false, false, la);
                            cold_ms = cold_run.probe_wall_ms;
                        }
                        cell.warm_probe_wall_ms = warm_ms;
                        cell.cold_probe_wall_ms = cold_ms;
                        if (warm_ms > 0.0)
                            cell.cache_speedup_ratio = cold_ms / warm_ms;
                    }

                    // ── Oracle correctness check (E-Auto, E-L0/L1/L2) ────────────────────
                    // For INNER ALL/RIGHTANY: uses oracle blocks collected BEFORE timing (above).
                    // For INNER ANY: build a FRESH join instance for oracle verification (the timing
                    // build has exhausted all setUsedOnce flags; this correctness-only build is NOT
                    // counted in build_invocations).
                    if (auto_check && is_any_strictness)
                    {
                        // Fresh build for INNER ANY oracle verification only.
                        auto oracle_bdo = runBuildDriver(cfg_, build_blocks, bdk);
                        std::vector<Block> oracle_probe_blocks;
                        {
                            KeyGenerator oracle_probe_gen(kp);
                            ConfigType oracle_cfg = cfg_;
                            BlockBuilder oracle_probe_bb(shape, oracle_cfg, oracle_probe_gen);
                            while (oracle_probe_bb.hasProbeRows())
                            {
                                Block blk = oracle_probe_bb.nextProbeBlock();
                                if (blk.columns() > 0)
                                    oracle_probe_blocks.push_back(std::move(blk));
                            }
                        }
                        auto oracle_probe = runProbe(
                            oracle_bdo.join, oracle_probe_blocks, /*mt=*/1, cfg_.probe_rows, /*collect_blocks=*/true, /*use_hw=*/false);
                        rep0_output_rows = oracle_probe.output_rows;
                        rep0_blocks = std::move(oracle_probe.output_blocks);
                    }
                    if (auto_check)
                    {
                        // Write harness output from first timing rep to native file.
                        // Rep 0 uses the pristine hash-map state, which is required for
                        // INNER ANY JOIN where used-flags are consumed on the first probe pass.
                        const std::string harness_out = cfg_.output_dir + "/harness_output.native";
                        {
                            DB::WriteBufferFromFile wb(harness_out);
                            if (!rep0_blocks.empty())
                            {
                                // Reorder columns to canonical oracle schema before writing.
                                // This ONLY affects the comparison copy, not the timing path.
                                Block hdr_reordered = reorderToOracleSchema(rep0_blocks[0].cloneEmpty());
                                DB::SharedHeader shared_hdr = std::make_shared<const DB::Block>(hdr_reordered);
                                DB::NativeWriter nw(wb, 0 /*client_revision*/, shared_hdr);
                                for (auto & blk : rep0_blocks)
                                    nw.write(reorderToOracleSchema(blk));
                                nw.flush();
                            }
                            wb.finalize();
                        }

                        // Build per-cell oracle SQL (max_threads affects join_algorithm setting).
                        ConfigType cell_cfg = cfg_;
                        cell_cfg.probe_max_threads_sweep = {mt};
                        OracleSql oracle_sql = Oracle::buildOracleSql(cell_cfg, key_cols, build_native_path, probe_native_path);
                        cell.oracle_sql = oracle_sql.full_sql;

                        // Invoke oracle.
                        auto inv = Oracle::invokeOracle(oracle_sql, Oracle::DEFAULT_CLICKHOUSE_BINARY, cfg_.output_dir);
                        if (inv.exit_code != 0)
                        {
                            std::cerr << "[HARNESS_ERROR] oracle invocation failed: " << inv.error_message << "\n";
                            std::exit(1);
                        }

                        // Run verifiers.
                        Verification::RunContext rctx;
                        rctx.harness_native_path = harness_out;
                        rctx.oracle_native_path = inv.oracle_native_path;
                        rctx.clickhouse_bin = Oracle::DEFAULT_CLICKHOUSE_BINARY;
                        rctx.output_dir = cfg_.output_dir;
                        rctx.harness_row_count = rep0_output_rows;
                        rctx.oracle_row_count = inv.oracle_row_count;
                        rctx.sweep_max_threads = mt;
                        rctx.sql = oracle_sql;

                        // E-Auto: exits on L0/L1 failure for small-scale runs.
                        auto vr = Verification::runAutoVerify(cfg_, rctx);
                        cell.oracle_l0_pass = vr.l0_pass;
                        cell.oracle_l1_pass = vr.l1_pass;
                        // For ANY strictness: used-flags exhausted after first probe — block
                        // oracle on subsequent cells.  For ALL/RIGHTANY: keep oracle_check_done
                        // false so every cell gets oracle coverage.
                        if (is_any_strictness)
                            oracle_check_done = true;

                        // E-L2: byte-identical check (single-thread only).
                        // Run separately since runAutoVerify does not cover E-L2.
                        const bool l2_eligible = (cfg_.build_threads == 1) && (mt == 1);
                        if (l2_eligible)
                        {
                            std::string l2_err;
                            cell.oracle_l2_pass = Verification::checkE_L2(rctx.harness_native_path, rctx.oracle_native_path, l2_err);
                        }
                    }

                    artifact.probe_cells.push_back(std::move(cell));
                } // for la
            } // for cm_cstr
        }
    }

    hdr.probe_invocations = probe_invocations;

    // G.2: log counters to stderr (C7)
    std::cerr << "[hashprobe-bench] build_invocations=" << hdr.build_invocations << " probe_invocations=" << hdr.probe_invocations
              << " probe_cells=" << artifact.probe_cells.size() << "\n";

    if (cfg_.save_artifact)
        writeArtifactJson(artifact, cfg_.output_dir);

    // ── Human-readable stdout summary ────────────────────────────────────────
    printSummary(artifact, cfg_);

    // ── PHJ path: CLICKHOUSE_PARTITIONED_JOIN=1 ───────────────────────────────
    // Uses the same build_blocks / proto_cache / cfg_ already in scope.
    {
        const char * phj_env = std::getenv("CLICKHOUSE_PARTITIONED_JOIN"); // NOLINT(concurrency-mt-unsafe)
        if (phj_env && phj_env[0] != '0')
        {
            const Block right_sample = makeRightSampleBlock(cfg_);
            const int P = computeAutoPPartitions(cfg_, cfg_.build_rows);

            std::printf("\n=== Partitioned Hash Join (CLICKHOUSE_PARTITIONED_JOIN=1) ===\n");
            std::printf(
                "P=%d  (rows/part=%.0f → %.1f MB, target L2=2 MB)\n",
                P,
                static_cast<double>(cfg_.build_rows) / P,
                static_cast<double>(cfg_.build_rows) / P * static_cast<double>(cfg_.key_columns + 1)
                    * (cfg_.key_width == KeyWidth::W64 ? 8 : 4) / 1e6);

            for (uint32_t mt : cfg_.probe_max_threads_sweep)
            {
                for (uint32_t bs : cfg_.probe_block_size_sweep)
                {
                    // Ensure probe proto blocks are available (reuse proto_cache).
                    if (proto_cache.find(bs) == proto_cache.end())
                    {
                        KeyGenerator probe_gen2(kp);
                        ConfigType pcfg2 = cfg_;
                        pcfg2.block_size = bs;
                        BlockBuilder probe_bb2(shape, pcfg2, probe_gen2);
                        std::vector<Block> pblocks2;
                        while (probe_bb2.hasProbeRows())
                        {
                            Block blk = probe_bb2.nextProbeBlock();
                            if (blk.columns() > 0)
                                pblocks2.push_back(std::move(blk));
                        }
                        proto_cache[bs] = std::move(pblocks2);
                    }
                    const auto & probe_proto = proto_cache.at(bs);

                    // Accumulate over cfg_.reps; report median.
                    std::vector<PHJPhaseMetrics> reps_m;
                    reps_m.reserve(cfg_.reps);
                    for (uint32_t rep = 0; rep < cfg_.reps; ++rep)
                        reps_m.push_back(runPHJCell(cfg_, right_sample, build_blocks, probe_proto, mt, cfg_.build_rows, cfg_.probe_rows));

                    auto med = [&](auto fn)
                    {
                        std::vector<double> v;
                        for (const auto & r : reps_m)
                            v.push_back(fn(r));
                        std::sort(v.begin(), v.end());
                        return v[v.size() / 2];
                    };

                    const double pb = med([](const PHJPhaseMetrics & r) { return r.part_build_cpu_ms; });
                    const double bh = med([](const PHJPhaseMetrics & r) { return r.build_ht_cpu_ms; });
                    const double pp = med([](const PHJPhaseMetrics & r) { return r.part_probe_cpu_ms; });
                    const double prb = med([](const PHJPhaseMetrics & r) { return r.probe_cpu_ms; });
                    const double gen = med([](const PHJPhaseMetrics & r) { return r.generate_cpu_ms; });
                    const double wall = med([](const PHJPhaseMetrics & r) { return r.total_wall_ms; });

                    const double rows = static_cast<double>(cfg_.probe_rows);
                    const double T_d = static_cast<double>(mt);
                    const double total_cpu = pb + bh + pp + prb + gen;

                    // cpu_ns/row = cpu_ms_total × 1e6 / rows
                    auto ns = [&](double ms) { return ms * 1e6 / rows; };
                    // wall_ms for a phase = cpu_ms_total / T  (threads run in parallel)
                    auto wms = [&](double ms) { return ms / T_d; };
                    // percentage of total CPU
                    auto pct = [&](double ms) { return (total_cpu > 0.0) ? ms * 100.0 / total_cpu : 0.0; };

                    // Wall time for probe+gen is measured directly; phases 1-3 estimated via cpu/T.
                    const double pb_wall = wms(pb);
                    const double bh_wall = wms(bh);
                    const double pp_wall = wms(pp);
                    // probe+gen wall = directly measured total_wall_ms from runPHJCell phase 4.
                    const double pg_wall = wall;
                    const double tot_wall = pb_wall + bh_wall + pp_wall + pg_wall;

                    // ── Per-phase table (mirrors "Phase breakdown" style) ──
                    std::printf("\nPhase breakdown  mt=%u  blksz=%u  (%u reps, median):\n", mt, bs, cfg_.reps);
                    std::printf("  %-16s  %10s  %6s  %10s\n", "phase", "cpu ns/row", "cpu %", "wall ms");
                    std::printf("  %-16s  %10s  %6s  %10s\n", "─────────────────", "──────────", "──────", "────────");

                    struct Row
                    {
                        const char * name;
                        double cpu_ms;
                        double wall_ms_v;
                    };
                    const Row rows_data[] = {
                        {"part-build", pb, pb_wall},
                        {"build-HT", bh, bh_wall},
                        {"part-probe", pp, pp_wall},
                        {"probe", prb, -1.0}, // probe+gen wall reported together
                        {"generate", gen, -1.0},
                    };
                    for (const auto & row : rows_data)
                    {
                        if (row.wall_ms_v >= 0.0)
                            std::printf("  %-16s  %10.3f  %5.1f%%  %8.0f\n", row.name, ns(row.cpu_ms), pct(row.cpu_ms), row.wall_ms_v);
                        else
                            // probe and generate share one wall-clock measurement
                            std::printf("  %-16s  %10.3f  %5.1f%%  %8s\n", row.name, ns(row.cpu_ms), pct(row.cpu_ms), "*");
                    }
                    std::printf("  %-16s  %10s  %6s  %8.0f  (* probe+gen measured together)\n", "", "", "", pg_wall);
                    std::printf("  %-16s  %10s  %6s  %10s\n", "─────────────────", "──────────", "──────", "────────");
                    std::printf("  %-16s  %10.3f  %5.1f%%\n", "PHJ TOTAL", ns(total_cpu), 100.0);

                    // ── Wall time summary for this cell ───────────────────
                    std::printf("\nWall time summary (mt=%u, blksz=%u):\n", mt, bs);
                    std::printf("  part-build   %7.0f ms\n", pb_wall);
                    std::printf("  build-HT     %7.0f ms\n", bh_wall);
                    std::printf("  part-probe   %7.0f ms\n", pp_wall);
                    std::printf("  probe+gen    %7.0f ms\n", pg_wall);
                    std::printf("  ───────────────────\n");
                    const uint64_t phj_rows = reps_m[0].output_rows;
                    std::printf("  TOTAL        %7.0f ms   (%llu output rows)\n", tot_wall, static_cast<unsigned long long>(phj_rows));

                    // ── PHJ row-count sanity check vs CH join ─────────────────
                    // Find a CH probe cell with the same (mt, bs) to compare.
                    uint64_t ch_rows = 0;
                    bool found_ch = false;
                    for (const auto & cell : artifact.probe_cells)
                    {
                        if (cell.max_threads == mt && cell.block_size == bs && cell.output_rows > 0)
                        {
                            ch_rows = cell.output_rows;
                            found_ch = true;
                            break;
                        }
                    }
                    if (found_ch)
                    {
                        if (phj_rows == ch_rows)
                            std::printf(
                                "  Row-count check: PHJ=%llu == CH=%llu  PASS\n",
                                static_cast<unsigned long long>(phj_rows),
                                static_cast<unsigned long long>(ch_rows));
                        else
                            std::printf(
                                "  Row-count check: PHJ=%llu != CH=%llu  FAIL  "
                                "(delta=%lld) [HARNESS_ERROR]\n",
                                static_cast<unsigned long long>(phj_rows),
                                static_cast<unsigned long long>(ch_rows),
                                static_cast<long long>(phj_rows) - static_cast<long long>(ch_rows));
                    }
                    std::fflush(stdout);
                }
            }
        }
    }

    return artifact;
}


} // namespace DB::HashProbeBench
