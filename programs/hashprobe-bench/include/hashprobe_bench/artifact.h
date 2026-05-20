#pragma once

/// hashprobe-bench/include/hashprobe_bench/artifact.h
///
/// ArtifactSchema: complete result document emitted per harness invocation.
///
/// Format: JSON (serialized via nlohmann-json in Phase 2 Track G).
/// One document per invocation: one BuildHeader + N ProbeCell records.
///
/// Field coverage (spec requirement → field(s)):
///   C7   → build_invocations, probe_invocations
///   C8   → harness_drain_mode ("tight_loop")
///   F5   → build_rows, build_distinct_keys, build_row_to_key_ratio
///   G1   → join_engine, build_threads, slots, probe_max_threads_sweep
///   G6   → cpu_affinity
///   H1   → build_wall_ms, build_cpu_ms,
///            joinblock_probe_wall_ms, joinblock_probe_cpu_ms,
///            result_emit_wall_ms, result_emit_cpu_ms,
///            probe_wall_ms, probe_cpu_ms
///   H5   → cold_probe_wall_ms, warm_probe_wall_ms, cache_speedup_ratio
///   H6   → per_rep_probe_wall_ms[], median_probe_wall_ms, probe_wall_ms_cv,
///            per_rep_probe_cpu_ms[], median_probe_cpu_ms, probe_cpu_ms_cv,
///            per_rep_throughput_rows_per_s[], median_throughput_rows_per_s,
///            throughput_rows_per_s_cv
///   I2   → git_commit (40-char hex, captured at build time)
///   I3   → compiler, cxx_flags
///   I4   → oracle_settings_dump_path
///   A2   → resolved_map_type_post_build
///   A2b  → strictness_at_construction, strictness_after_build
///   E-Auto → oracle_sql (verbatim SQL emitted to oracle, per oracle def.)

#include <cmath>
#include <cstdint>
#include <hashprobe_bench/types.h>
#include <string>
#include <vector>

namespace DB::HashProbeBench
{

/// Per-rep timing for one probe sweep cell (H6).
struct RepTiming
{
    double probe_wall_ms = std::numeric_limits<double>::quiet_NaN();
    double probe_cpu_ms = std::numeric_limits<double>::quiet_NaN();
    double throughput_rows_per_s = std::numeric_limits<double>::quiet_NaN();
};

/// Per-block-level timing entry (H2, per-probe-block view).
struct ProbeBlockEntry
{
    uint64_t probe_block_idx = 0;
    uint64_t probe_block_rows = 0;
    double joinblock_probe_wall_ns = 0.0; ///< Time inside HashJoin::joinBlock
    double joinblock_probe_cpu_ns = 0.0;
    double result_emit_wall_ns = 0.0; ///< Sum of next() call intervals
    double result_emit_cpu_ns = 0.0;
    uint32_t output_block_count = 0; ///< Number of next() calls until is_last
    // HW counters (H4) — valid only when counter_mode != none
    uint64_t hw_cycles = 0;
    uint64_t hw_instructions = 0;
    double hw_ipc = 0.0;
    // Raw miss counts — internal only; not emitted in JSON/CSV.
    // Used to compute hw_llc_miss_pct, hw_br_miss_pct, hw_dtlb_miss_pct.
    uint64_t hw_llc_miss = 0;
    uint64_t hw_branch_miss = 0;
    uint64_t hw_dtlb_miss = 0;
    // Total-access denominators for miss-rate percentages.
    uint64_t hw_llc_load = 0; ///< PERF_COUNT_HW_CACHE_REFERENCES (see hw_counters.h)
    uint64_t hw_branches = 0; ///< PERF_COUNT_HW_BRANCH_INSTRUCTIONS
    uint64_t hw_dtlb_load = 0; ///< PERF_TYPE_HW_CACHE / DTLB/READ/ACCESS (see hw_counters.h)
    // TID (G2)
    uint64_t caller_tid = 0;
    double joinblock_start_ns = 0.0;
    double last_next_end_ns = 0.0;

    /// On the scalar probe path, phase_probe captures the whole probe loop
    /// (probe_loop_start -> probe_loop_end). Time spent inside
    /// HashJoinResult::next() but outside generateBlock is not attributed to
    /// either phase (small in practice, < 1%).
    PhaseMetrics phase_probe;
    PhaseMetrics phase_generate;
};

/// Per-partition timing entry for PartitionedHashJoin (PHJ-specific, analogous to ProbeBlockEntry).
/// One entry is emitted per partition processed by DelayedBlocks::nextImpl().
/// The three sub-phases correspond to the PHJ_PHASE_POINT hooks in DelayedBlocks.cpp.
struct PhjPartitionEntry
{
    uint32_t partition_idx = 0; ///< Partition index p ∈ [0, P)
    uint64_t build_rows = 0; ///< Rows in the per-partition build side
    uint64_t probe_rows = 0; ///< Rows in the per-partition probe side (across all probe slots)
    uint64_t output_rows = 0; ///< Rows emitted from this partition's probe + gen phases

    /// Build-HT phase: constructing the per-partition mini-HashJoin.
    /// Wall/CPU/HW counters wrap phj_build_ht_start → phj_build_ht_end.
    PhaseMetrics phase_build_ht;

    /// Probe phase: all joinBlock() calls on the per-partition HashJoin.
    /// Spans phj_probe_start → phj_probe_end.
    /// The inner HashJoin fires its own probe_loop_start/end hooks inside
    /// joinRightColumns, which are accumulated into phase_probe.hw_* when
    /// the harness registers a ProbePointCallback.
    PhaseMetrics phase_probe;

    /// Gen phase: all next() drain calls on the per-partition HashJoin.
    /// Spans phj_gen_start → phj_gen_end.
    /// The inner HashJoin fires generate_block_start/end inside HashJoinResult.cpp,
    /// accumulated into phase_gen.hw_*.
    PhaseMetrics phase_gen;
};

/// Per-output-block timing entry (H2, per-output-block view).
struct OutputBlockEntry
{
    uint64_t probe_block_idx = 0;
    uint32_t output_block_idx = 0;
    uint64_t output_block_rows = 0;
    double next_wall_ns = 0.0;
    double next_cpu_ns = 0.0;
    bool is_last = false;
};

/// Build-phase result header.  One per harness invocation.
struct BuildHeader
{
    // ── Identifiers / reproducibility (I2, I3) ──────────────────────────
    std::string git_commit; ///< 40-char hex from git log -1 --format=%H at build time (I2)
    std::string compiler; ///< Path + version string of the C++ compiler (I3)
    std::string cxx_flags; ///< -march / -O / LTO flags as a single string (I3)

    // ── CPU affinity (G6) ───────────────────────────────────────────────
    std::string cpu_affinity; ///< "taskset -p $$" output, "numactl --hardware", or "unset"

    // ── Engine selection (G1) ───────────────────────────────────────────
    std::string join_engine; ///< "HashJoin" | "ConcurrentHashJoin"
    uint32_t build_threads = 0;
    uint32_t slots = 0; ///< ConcurrentHashJoin slot count (G1)
    std::vector<uint32_t> probe_max_threads_sweep;

    // ── Build stats (F5) ────────────────────────────────────────────────
    uint64_t build_rows = 0;
    uint64_t build_distinct_keys = 0;
    double build_row_to_key_ratio = std::numeric_limits<double>::quiet_NaN();

    // ── Build timing (H1) ───────────────────────────────────────────────
    double build_wall_ms = std::numeric_limits<double>::quiet_NaN();
    double build_cpu_ms = std::numeric_limits<double>::quiet_NaN();
    PhaseMetrics phase_build_ht;

    // ── Post-build type gates (A2, A2b) ─────────────────────────────────
    std::string resolved_map_type_post_build; ///< E.g. "key64", "two_level_keys128"
    std::string strictness_at_construction; ///< "ALL" | "ANY" | "RIGHTANY"
    std::string strictness_after_build; ///< Must equal strictness_at_construction (A2b)

    // ── Build-once / probe-many counters (C7) ───────────────────────────
    uint32_t build_invocations = 0; ///< Must be 1 for any multi-sweep run
    uint32_t probe_invocations = 0; ///< Total probe cells × reps executed

    // ── Drain mode (C8) ─────────────────────────────────────────────────
    std::string harness_drain_mode; ///< Always "tight_loop" (C8 structural deviation doc)

    // ── PHJ-specific shuffle timing ───────────────────────────────────────
    /// For algorithm=partitioned_hash only; NaN for hash algorithms.
    double shuffle_probe_wall_ms = std::numeric_limits<double>::quiet_NaN();
    double shuffle_probe_cpu_ms = std::numeric_limits<double>::quiet_NaN();
};

/// One probe sweep cell result.  N per invocation (N = sweep_grid_size × reps).
struct ProbeCellResult
{
    // ── Cell identity ───────────────────────────────────────────────────
    uint32_t max_threads = 0;
    uint32_t block_size = 0;
    uint32_t rep_index = 0; ///< 0-based rep within this sweep cell (H6)
    std::string cache_mode_str; ///< "warm" | "cold"
    std::string counter_mode; ///< "none" | "perf" | "hw"

    // ── Row counts ──────────────────────────────────────────────────────
    uint64_t probe_rows = 0;
    uint64_t output_rows = 0;

    // ── Timing (H1) — probe path broken into two sub-phases ─────────────
    double joinblock_probe_wall_ms = std::numeric_limits<double>::quiet_NaN();
    double joinblock_probe_cpu_ms = std::numeric_limits<double>::quiet_NaN();
    double result_emit_wall_ms = std::numeric_limits<double>::quiet_NaN();
    double result_emit_cpu_ms = std::numeric_limits<double>::quiet_NaN();
    double probe_wall_ms = std::numeric_limits<double>::quiet_NaN();
    double probe_cpu_ms = std::numeric_limits<double>::quiet_NaN();
    double throughput_rows_per_s = std::numeric_limits<double>::quiet_NaN();

    // ── Cache-effect data (H5, reported as data, not as a gate) ─────────
    double cold_probe_wall_ms = std::numeric_limits<double>::quiet_NaN();
    double warm_probe_wall_ms = std::numeric_limits<double>::quiet_NaN();
    double cache_speedup_ratio = std::numeric_limits<double>::quiet_NaN();

    // ── Per-rep arrays (H6) — one entry per rep for this sweep cell ──────
    std::vector<RepTiming> per_rep;
    double median_probe_wall_ms = std::numeric_limits<double>::quiet_NaN();
    double median_probe_cpu_ms = std::numeric_limits<double>::quiet_NaN();
    double median_throughput_rows_per_s = std::numeric_limits<double>::quiet_NaN();
    double probe_wall_ms_cv = std::numeric_limits<double>::quiet_NaN();
    double probe_cpu_ms_cv = std::numeric_limits<double>::quiet_NaN();
    double throughput_rows_per_s_cv = std::numeric_limits<double>::quiet_NaN();

    // ── Per-block timing logs (H2) ───────────────────────────────────────
    std::vector<ProbeBlockEntry> probe_block_log;
    std::vector<OutputBlockEntry> output_block_log;

    // ── PHJ per-partition log (PHJ-only; empty for hash algorithm) ────────
    std::vector<PhjPartitionEntry> phj_partition_log;

    // ── Oracle correctness (E-Auto, oracle def.) ─────────────────────────
    std::string oracle_sql; ///< Full SQL emitted for this cell (I4, oracle def.)
    std::string oracle_settings_dump_path; ///< Path to settings_dump.tsv (I4)
    bool oracle_l0_pass = false; ///< E-L0: row count matches oracle
    bool oracle_l1_pass = false; ///< E-L1: sorted SHA-256 matches oracle
    bool oracle_l2_pass = false; ///< E-L2: byte-identical (build_threads==1 AND max_threads==1)
};

/// Complete result document for one harness invocation.
struct Artifact
{
    BuildHeader build;
    std::vector<ProbeCellResult> probe_cells;
};

} // namespace DB::HashProbeBench
