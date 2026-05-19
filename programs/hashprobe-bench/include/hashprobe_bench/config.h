#pragma once

/// hashprobe-bench/include/config.h
///
/// ConfigType: full description of one harness invocation parsed from the CLI.
/// All fields map 1:1 to CLI flags documented in main.cpp.

#include <cstdint>
#include <string>
#include <vector>

namespace DB::HashProbeBench
{

/// CLI-level strictness selector.  Maps to JoinStrictness at HashJoin construction.
/// Only {ALL, ANY, RIGHTANY} are accepted; others cause fail-loudly exit (A4).
enum class StrictnessConfig
{
    ALL,
    ANY,
    RIGHTANY,
};

/// Per-key-column bit width.
enum class KeyWidth
{
    W32 = 32,
    W64 = 64,
};

/// Cache-warming mode applied before each probe-phase timing run (H5 definitions).
enum class CacheMode
{
    WARM,  ///< Walk hash table once sequentially to populate LLC.
    COLD,  ///< Allocate 2×LLC and write it twice to evict all LLC contents.
};

/// Full configuration for one harness invocation, populated from the CLI (T0.6).
/// Fields are grouped by the spec section they serve.
struct ConfigType
{
    // ── Build side (C1, C3, C6, G1) ──────────────────────────────────────
    uint32_t build_threads   = 1;        ///< 1 → HashJoin; >1 → ConcurrentHashJoin (G1)
    uint64_t build_rows      = 1'000'000;

    // ── Key shape (A2, F6) ────────────────────────────────────────────────
    uint32_t key_columns     = 1;        ///< N ∈ {1, 2, 4}
    KeyWidth key_width       = KeyWidth::W64;
    bool     key_nullable    = false;    ///< Nullable applied symmetrically to build+probe (J2)

    // ── Join semantics (A3, A4, D1-D4) ───────────────────────────────────
    StrictnessConfig strictness = StrictnessConfig::ALL;
    // kind is always Inner; anything else is rejected at parse time (A3).

    // ── Probe sweep (C7, G1, G2) ──────────────────────────────────────────
    std::vector<uint32_t> probe_max_threads_sweep = {1};
    std::vector<uint32_t> probe_block_size_sweep  = {65536};
    uint64_t probe_rows = 1'000'000;

    // ── Block / join sizing (C3, C4, C5) ──────────────────────────────────
    uint32_t block_size                  = 65536;  ///< Rows per block (last may be shorter)
    uint64_t max_joined_block_size_rows  = 65536;  ///< HashJoin output splitter limit

    // ── Workload precision (F1, F2, A2b) ──────────────────────────────────
    double match_rate    = 0.5;   ///< Fraction of probe rows matching at least one build key
    double null_fraction = 0.0;   ///< Null fraction per key column (both sides, F2)

    // ── Reproducibility (I1) ──────────────────────────────────────────────
    uint64_t seed = 42;

    // ── Timing / reps (H6, C7) ────────────────────────────────────────────
    uint32_t reps = 1;  ///< Probe-phase repetitions per sweep cell; no rebuild between reps

    // ── Cache mode (H5) ───────────────────────────────────────────────────
    CacheMode cache_mode = CacheMode::COLD;

    // ── Output ────────────────────────────────────────────────────────────
    std::string output_dir = ".";  ///< Directory for artifact, logs, .native files

    // ── Oracle verification (E-L0/L1/L2) ──────────────────────────────────
    /// When false (default) the oracle path is completely bypassed: no
    /// clickhouse-local invocations, no .native writes, no E-L0/L1/L2 checks,
    /// and oracle_* fields in the artifact are left null/false.
    /// Set to true (--verify-oracle) to restore full correctness coverage.
    bool verify_oracle = false;


    // Write artifact.json to output_dir after the sweep.  Off by default;
    // enable with --save-artifact.
    bool save_artifact = false;
};

} // namespace DB::HashProbeBench
