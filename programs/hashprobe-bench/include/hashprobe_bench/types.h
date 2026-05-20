#pragma once

/// hashprobe-bench/include/hashprobe_bench/types.h
///
/// Common structural types shared across harness modules.
/// Lightweight value types; no ClickHouse includes here so harness unit tests
/// can link these without pulling in the full DB library.

#include <cstdint>
#include <string>
#include <vector>

namespace DB::HashProbeBench
{

struct PhaseMetrics
{
    double wall_ns = 0.0;
    double cpu_ns = 0.0;
    uint64_t hw_cycles = 0;
    uint64_t hw_instructions = 0;
    uint64_t hw_llc_miss = 0;
    uint64_t hw_branch_miss = 0;
    uint64_t hw_dtlb_miss = 0;
    uint64_t hw_llc_load = 0;
    uint64_t hw_branches = 0;
    uint64_t hw_dtlb_load = 0;
    bool hw_available = false;
};

// ── Key shape (F6, A2, J2) ────────────────────────────────────────────────────

/// Describes the shape of the key columns on one join side.
/// The same shape is applied symmetrically to both build and probe sides (J2).
struct KeyShape
{
    uint32_t n;        ///< Number of key columns: 1, 2, or 4 (F6)
    uint32_t width;    ///< Per-column width in bits: 32 or 64 (F6)
    bool nullable;     ///< Wrap each key column in ColumnNullable (J2, F6)
                       ///< NOTE: nullable=true does NOT change the resolved HashJoin::Type;
                       ///< nullability is handled via per-column null masks (A2, finding 3).

    bool operator==(const KeyShape & o) const noexcept
    {
        return n == o.n && width == o.width && nullable == o.nullable;
    }
};

// ── Join configuration (A3, A4, D1-D4, F6) ───────────────────────────────────

/// Runtime join semantics handed to HashJoin construction.
/// kind is always Inner; other kinds are rejected at parse time (A3).
struct JoinConfig
{
    // kind == JoinKind::Inner always — enforced at CLI parse time (A3).
    // We use an explicit int32_t here rather than JoinKind to avoid pulling
    // src/Core/Joins.h into harness unit tests.
    int32_t kind       = 1;   ///< 1 == JoinKind::Inner
    int32_t strictness = 2;   ///< JoinStrictness::All=2, Any=3, RightAny=1 (src/Core/Joins.h:45-54)
    KeyShape key_shape;
};

// ── Build-phase results (F5, H1, A2, A2b) ────────────────────────────────────

/// Summary of the completed build phase, populated by the build driver.
struct BuildResult
{
    // --- Row counts (F5) ---
    uint64_t build_rows          = 0;
    uint64_t build_distinct_keys = 0;
    double   build_row_to_key_ratio = 0.0;  ///< build_rows / build_distinct_keys

    // --- Timing (H1) ---
    double   build_wall_ms       = 0.0;
    double   build_cpu_ms        = 0.0;
    PhaseMetrics phase_build_ht;

    // --- Post-build type (A2) ---
    std::string resolved_map_type;       ///< "key32" | "key64" | "keys128" | etc.

    // --- Strictness preservation check (A2b) ---
    std::string strictness_at_construction;  ///< "ALL" | "ANY" | "RIGHTANY"
    std::string strictness_after_build;      ///< Must equal strictness_at_construction

    // --- Lifecycle counters (C1, C6) ---
    uint64_t add_block_calls     = 0;    ///< Must equal build_blocks generated
    bool     post_build_ran      = false;///< True if hasPostBuildPhase→runPostBuildPhase called
};

// ── Probe-phase results (H1, C2, G2) ─────────────────────────────────────────

/// Summary of one probe sweep cell, populated by the probe driver.
struct ProbeResult
{
    uint32_t max_threads   = 0;
    uint32_t block_size    = 0;
    uint64_t probe_rows    = 0;
    uint64_t output_rows   = 0;

    // --- Timing breakdown (H1) ---
    double   joinblock_probe_wall_ms = 0.0;
    double   joinblock_probe_cpu_ms  = 0.0;
    double   result_emit_wall_ms     = 0.0;
    double   result_emit_cpu_ms      = 0.0;
    double   probe_wall_ms           = 0.0;  ///< joinblock_probe + result_emit
    double   probe_cpu_ms            = 0.0;
    double   throughput_rows_per_s   = 0.0;

    // --- Call counters (C2, C1) ---
    uint64_t join_block_calls  = 0;  ///< Must equal probe_blocks generated
    uint64_t next_calls        = 0;  ///< Total next() calls; must >= join_block_calls
    uint64_t output_blocks     = 0;  ///< Must equal next_calls

    // --- Match-rate accounting (F1) ---
    uint64_t probe_rows_with_match = 0;
    uint64_t total_probe_rows      = 0;
    double   measured_match_rate   = 0.0;  ///< probe_rows_with_match / total_probe_rows
};

// ── Oracle SQL (oracle definition, E-Auto, I4) ───────────────────────────────

/// SQL document emitted for one oracle comparison.
struct OracleSql
{
    std::string settings_prelude;   ///< SET statements (join_algorithm, max_threads, etc.)
    std::string create_tables;      ///< CREATE TABLE build_t/probe_t FROM file(...)
    std::string join_query;         ///< SELECT ... FROM probe_t [ANY] JOIN build_t ON ...
    std::string order_clause;       ///< ORDER BY all columns ASC NULLS FIRST (E-L1 sort)
    std::string full_sql;           ///< settings_prelude || create_tables || join_query

    /// Return the SQL as a single multiquery string suitable for clickhouse-local --multiquery.
    std::string toMultiquery() const { return full_sql; }
};

// ── Probe-block assignment for reproducibility checks (G5) ───────────────────

struct ThreadBlockAssignment
{
    uint32_t thread_idx  = 0;
    uint64_t block_start = 0;  ///< First probe block index assigned to this thread
    uint64_t block_end   = 0;  ///< One-past-last probe block index for this thread
};

} // namespace DB::HashProbeBench
