/// hashprobe-bench — HashJoin probe-path benchmark harness (issue #104581).
///
/// Drives HashJoin / ConcurrentHashJoin directly (no QueryPipeline) to isolate
/// and measure probe cost under sweeping configurations.
///
/// Phases 1-4 add data-generation, build/probe drivers, oracle, sweep manager,
/// and full measurement.  This file provides the CLI entry point.

#include <hashprobe_bench/artifact.h>
#include <hashprobe_bench/config.h>
#include <hashprobe_bench/types.h>
#include "sweep/sweep_manager.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace DB::HashProbeBench;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Parse a comma-separated list of uint32_t values, e.g. "1,2,4,8".
static std::vector<uint32_t> parseUInt32List(const std::string & s)
{
    std::vector<uint32_t> result;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        // Trim whitespace
        auto start = tok.find_first_not_of(" \t");
        auto end = tok.find_last_not_of(" \t");
        if (start == std::string::npos)
            continue;
        tok = tok.substr(start, end - start + 1);
        uint32_t v = 0;
        auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), v);
        if (ec != std::errc{} || ptr != tok.data() + tok.size())
            throw std::runtime_error("Cannot parse uint32 list token: '" + tok + "'");
        result.push_back(v);
    }
    if (result.empty())
        throw std::runtime_error("Empty uint32 list: '" + s + "'");
    return result;
}

/// Parse key_shape string: "<N>x{32|64}[,nullable]", e.g. "2x32,nullable".
static void parseKeyShape(const std::string & s, ConfigType & cfg)
{
    // Format: NxW[,nullable]
    std::string str = s;
    cfg.key_nullable = false;

    auto pos = str.find(",nullable");
    if (pos != std::string::npos)
    {
        cfg.key_nullable = true;
        str = str.substr(0, pos);
    }

    auto x = str.find('x');
    if (x == std::string::npos)
        throw std::runtime_error("Invalid --key_shape format (expected NxW or NxW,nullable): " + s);

    uint32_t n = 0, w = 0;
    {
        auto [ptr, ec] = std::from_chars(str.data(), str.data() + x, n);
        if (ec != std::errc{} || ptr != str.data() + x)
            throw std::runtime_error("Invalid N in --key_shape: " + s);
    }
    {
        auto [ptr, ec] = std::from_chars(str.data() + x + 1, str.data() + str.size(), w);
        if (ec != std::errc{} || ptr != str.data() + str.size())
            throw std::runtime_error("Invalid W in --key_shape: " + s);
    }

    if (n != 1 && n != 2 && n != 4)
        throw std::runtime_error("--key_shape N must be 1, 2, or 4; got: " + std::to_string(n));
    if (w != 32 && w != 64)
        throw std::runtime_error("--key_shape W must be 32 or 64; got: " + std::to_string(w));

    cfg.key_columns = n;
    cfg.key_width = (w == 32) ? KeyWidth::W32 : KeyWidth::W64;
}

/// Fail-loudly for unsupported configuration (A3, A4).
/// Emits: [HARNESS_ERROR] unsupported_config: <param>=<value>
[[noreturn]] static void failLoudly(const std::string & param, const std::string & value)
{
    std::cerr << "[HARNESS_ERROR] unsupported_config: " << param << "=" << value << "\n";
    std::exit(1);
}

/// Validate and populate cfg from program_options values_map.
/// Performs A3 (kind), A4 (strictness), and A2b pre-check for all-unique-keys+ALL.
static ConfigType buildConfig(const po::variables_map & vm)
{
    ConfigType cfg;

    // ── kind (A3) ─────────────────────────────────────────────────────────
    {
        auto kind_str = vm["kind"].as<std::string>();
        // Only "Inner" is allowed (A3).
        if (kind_str != "Inner" && kind_str != "inner" && kind_str != "INNER")
            failLoudly("kind", kind_str);
    }

    // ── algorithm ─────────────────────────────────────────────────────────
    {
        auto al = vm["algorithm"].as<std::string>();
        if (al == "hash")
            cfg.algorithm = AlgorithmConfig::HASH;
        else if (al == "partitioned_hash")
            cfg.algorithm = AlgorithmConfig::PARTITIONED_HASH;
        else
            failLoudly("algorithm", al);
    }

    // ── strictness (A4) ───────────────────────────────────────────────────
    {
        auto st = vm["strictness"].as<std::string>();
        if (st == "ALL" || st == "all")
            cfg.strictness = StrictnessConfig::ALL;
        else if (st == "ANY" || st == "any")
            cfg.strictness = StrictnessConfig::ANY;
        else if (st == "RIGHTANY" || st == "RightAny" || st == "rightany")
            cfg.strictness = StrictnessConfig::RIGHTANY;
        else
            failLoudly("strictness", st);
    }

    // ── build_threads (G1) ────────────────────────────────────────────────
    cfg.build_threads = vm["build_threads"].as<uint32_t>();
    if (cfg.build_threads < 1)
        throw std::runtime_error("--build_threads must be >= 1");

    // ── build_rows ────────────────────────────────────────────────────────
    cfg.build_rows = vm["build_rows"].as<uint64_t>();

    // ── key_shape ─────────────────────────────────────────────────────────
    parseKeyShape(vm["key_shape"].as<std::string>(), cfg);

    // ── probe sweeps ──────────────────────────────────────────────────────
    cfg.probe_max_threads_sweep = parseUInt32List(vm["probe_max_threads_sweep"].as<std::string>());
    cfg.probe_block_size_sweep = parseUInt32List(vm["probe_block_size_sweep"].as<std::string>());
    cfg.probe_rows = vm["probe_rows"].as<uint64_t>();

    // ── block / join sizing ───────────────────────────────────────────────
    cfg.block_size = vm["block_size"].as<uint32_t>();
    cfg.max_joined_block_size_rows = vm["max_joined_block_size_rows"].as<uint64_t>();

    // ── workload precision ────────────────────────────────────────────────
    cfg.match_rate = vm["match_rate"].as<double>();
    cfg.null_fraction = vm["null_fraction"].as<double>();

    if (cfg.match_rate < 0.0 || cfg.match_rate > 1.0)
        throw std::runtime_error("--match_rate must be in [0.0, 1.0]");
    if (cfg.null_fraction < 0.0 || cfg.null_fraction > 1.0)
        throw std::runtime_error("--null_fraction must be in [0.0, 1.0]");

    // ── reproducibility ───────────────────────────────────────────────────
    cfg.seed = vm["seed"].as<uint64_t>();

    // ── timing / reps ─────────────────────────────────────────────────────
    cfg.reps = vm["reps"].as<uint32_t>();
    if (cfg.reps < 1)
        throw std::runtime_error("--reps must be >= 1");

    // ── cache_mode ────────────────────────────────────────────────────────
    {
        auto cm = vm["cache_mode"].as<std::string>();
        if (cm == "warm" || cm == "WARM")
            cfg.cache_mode = CacheMode::WARM;
        else if (cm == "cold" || cm == "COLD")
            cfg.cache_mode = CacheMode::COLD;
        else
            throw std::runtime_error("--cache_mode must be warm or cold; got: " + cm);
    }

    // ── output_dir ────────────────────────────────────────────────────────
    cfg.output_dir = vm["output_dir"].as<std::string>();

    // ── verify_oracle ─────────────────────────────────────────────────────
    cfg.verify_oracle = vm["verify-oracle"].as<bool>();

    cfg.save_artifact = vm["save-artifact"].as<bool>();
    return cfg;
}

static void printVersion()
{
    std::cout << "hashprobe-bench (issue #104581) — HashJoin probe-path harness\n";
}

int main(int argc, char ** argv)
{
    try
    {
        po::options_description visible("hashprobe-bench options");
        visible.add_options()("help,h", "Show this help message and exit.")("version", "Print version and exit.")

            // ── Algorithm selection ───────────────────────────────────────
            ("algorithm",
             po::value<std::string>()->default_value("hash"),
             "Join algorithm: hash | partitioned_hash.  "
             "hash → HashJoin (build_threads==1) or ConcurrentHashJoin (>1);  "
             "partitioned_hash → PartitionedHashJoin (build_threads controls ingest parallelism).")

            // ── Build-side ────────────────────────────────────────────────
            ("build_threads",
             po::value<uint32_t>()->default_value(1),
             "Number of build threads.  1 → HashJoin directly (G1);  "
             ">1 → ConcurrentHashJoin with slots=max(build_threads, "
             "max(probe_max_threads_sweep)) (G1).")(
                "build_rows", po::value<uint64_t>()->default_value(1'000'000), "Total rows on the build side.")

            // ── Key shape ─────────────────────────────────────────────────
            ("key_shape",
             po::value<std::string>()->default_value("1x64"),
             "Key shape: <N>x{32|64}[,nullable].  "
             "E.g. \"1x64\", \"2x32\", \"4x64,nullable\".  "
             "N ∈ {1,2,4}; nullable applies symmetrically to build+probe (J2, F6).")

            // ── Join semantics (A3, A4) ───────────────────────────────────
            ("strictness",
             po::value<std::string>()->default_value("ALL"),
             "Join strictness: ALL | ANY | RIGHTANY.  "
             "ASOF/SEMI/ANTI/UNSPECIFIED cause immediate fail-loudly exit (A4).")(
                "kind",
                po::value<std::string>()->default_value("Inner"),
                "Join kind — must be Inner.  "
                "Left/Right/Full/Cross/Comma/Paste cause fail-loudly exit (A3).")

            // ── Probe sweep ───────────────────────────────────────────────
            ("probe_max_threads_sweep",
             po::value<std::string>()->default_value("1"),
             "Comma-separated max_threads values for the probe sweep (G1, G2).  "
             "E.g. \"1,2,4,8\".  ConcurrentHashJoin slots = "
             "max(build_threads, max(probe_max_threads_sweep)).")(
                "probe_rows", po::value<uint64_t>()->default_value(1'000'000), "Total rows on the probe side.")(
                "probe_block_size_sweep",
                po::value<std::string>()->default_value("65536"),
                "Comma-separated block sizes for the probe sweep (C3/C4).  "
                "E.g. \"4096,65536\".")

            // ── Block / join sizing ───────────────────────────────────────
            ("block_size",
             po::value<uint32_t>()->default_value(65536),
             "Default rows per block for both build and probe sides "
             "(last block may be shorter, C3/C4).")(
                "max_joined_block_size_rows",
                po::value<uint64_t>()->default_value(65536),
                "max_joined_block_size_rows setting passed to HashJoin "
                "(controls the output-block splitter, C5).")

            // ── Workload precision ────────────────────────────────────────
            ("match_rate",
             po::value<double>()->default_value(0.5),
             "Fraction of probe rows that match at least one build key (F1).  "
             "|measured − r| ≤ 0.01 over ≥10^7 rows.")(
                "null_fraction",
                po::value<double>()->default_value(0.0),
                "Null fraction per key column, applied symmetrically (F2).  "
                "|measured − q| ≤ 0.01 per column over ≥10^7 rows.")

            // ── Reproducibility ───────────────────────────────────────────
            ("seed",
             po::value<uint64_t>()->default_value(42),
             "pcg64 RNG seed for key and null generation (I1).  "
             "Same seed → identical block SHA-256 checksums across runs.")

            // ── Timing / reps (H6, C7) ────────────────────────────────────
            ("reps",
             po::value<uint32_t>()->default_value(1),
             "Probe-phase repetitions per sweep cell (H6).  "
             "The same HashJoin/ConcurrentHashJoin instance is reused — "
             "no rebuild between reps (C7).")(
                "cache_mode",
                po::value<std::string>()->default_value("cold"),
                "Cache-warming mode before probe timing: "
                "warm (walk hash table once, H5 def.) | "
                "cold (allocate+write 2×LLC buffer, H5 def.).")

            // ── Output ────────────────────────────────────────────────────
            ("output_dir",
             po::value<std::string>()->default_value("."),
             "Directory for: result artifact (JSON), per-block timing log (CSV/Parquet), "
             "build.native, probe.native, oracle output.")

            // ── Oracle verification (E-L0/L1/L2) ─────────────────────────────────
            ("verify-oracle",
             po::bool_switch()->default_value(false),
             "Enable oracle correctness verification (E-L0/L1/L2).  "
             "Off by default; enables clickhouse-local invocations, "
             "build.native/probe.native writes, and per-cell oracle checks.")(
                "save-artifact",
                po::bool_switch()->default_value(false),
                "Write artifact.json to output_dir after the sweep.  "
                "Off by default.");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, visible), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << "Usage: hashprobe-bench [options]\n\n" << visible << "\n";
            return 0;
        }
        if (vm.count("version"))
        {
            printVersion();
            return 0;
        }

        // Parse and validate configuration (fail-loudly on A3/A4 violations).
        ConfigType cfg = buildConfig(vm);

        // Phase 2, Track G: SweepManager drives the full build-once probe-many sweep.
        DB::HashProbeBench::SweepManager(cfg).run();
        return 0;
    }
    catch (const std::exception & e)
    {
        std::cerr << "[HARNESS_ERROR] " << e.what() << "\n";
        return 1;
    }
}
