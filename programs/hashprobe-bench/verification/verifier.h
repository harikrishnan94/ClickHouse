#pragma once

/// hashprobe-bench/verification/verifier.h
///
/// Track H: Verification wrappers (E-L0, E-L1, E-L2, E-Auto).
///
/// Thin wrappers over Track D oracle functions that add:
///   - structured error messages ([HARNESS_ERROR] oracle_mismatch: ...)
///   - artifact field population (oracle_l0_pass, oracle_l1_pass, oracle_l2_pass)
///   - E-Auto: exit-non-zero on L0/L1 mismatch for small-scale invocations
///
/// Spec requirements: E-L0, E-L1, E-L2, E-Auto.

#include "../oracle/oracle.h"
#include <hashprobe_bench/config.h>
#include <hashprobe_bench/types.h>

#include <cstdint>
#include <string>

namespace DB::HashProbeBench::Verification
{

// OracleSql is defined in DB::HashProbeBench (types.h); accessible unqualified
// here because Verification is a nested namespace of DB::HashProbeBench.

// ── Result type ──────────────────────────────────────────────────────────────

/// Populated by runVerifiers() after all applicable checks complete.
/// Fields map directly to ProbeCellResult::oracle_l{0,1,2}_pass artifact fields.
struct VerifierResult
{
    bool l0_pass     = false;   ///< E-L0: row count check
    bool l1_pass     = false;   ///< E-L1: sorted SHA-256 check
    bool l2_pass     = false;   ///< E-L2: byte-identical check (only when l2_eligible)
    bool l2_eligible = false;   ///< true when build_threads==1 && max_threads==1
    std::string error_msg;      ///< non-empty when any check fails; last failure wins
};

// ── Per-invocation context ────────────────────────────────────────────────────

/// Runtime paths and counts for one probe sweep cell.
/// Passed into runVerifiers() by the sweep manager (Track G).
struct RunContext
{
    std::string harness_native_path;    ///< Path to the harness output .native file
    std::string oracle_native_path;     ///< Path to the oracle output .native file
    std::string clickhouse_bin         = Oracle::DEFAULT_CLICKHOUSE_BINARY;
    std::string output_dir             = ".";
    uint64_t    harness_row_count      = 0;
    uint64_t    oracle_row_count       = 0;
    uint32_t    sweep_max_threads      = 1;
    OracleSql   sql;                    ///< For order_clause (E-L1 canonical sort)
};

// ── E-L0 ─────────────────────────────────────────────────────────────────────

/// Compare total row counts.
/// On mismatch sets err_msg to:
///   "[HARNESS_ERROR] oracle_mismatch: E-L0 row count mismatch: harness=N oracle=M"
/// and returns false.  On match returns true.
/// Corresponds to artifact field oracle_l0_pass.
bool checkE_L0(uint64_t harness_row_count, uint64_t oracle_row_count, std::string & err_msg);

// ── E-L1 ─────────────────────────────────────────────────────────────────────

/// Sort both harness and oracle outputs via clickhouse-local using the canonical
/// ORDER BY clause from OracleSql::order_clause, then compare SHA-256 hashes.
/// On hash mismatch sets err_msg to:
///   "[HARNESS_ERROR] oracle_mismatch: E-L1 sorted-hash mismatch"
/// and returns false.
/// Corresponds to artifact field oracle_l1_pass.
bool checkE_L1(
    const std::string  & harness_native_path,
    const std::string  & oracle_native_path,
    const OracleSql    & sql,
    const std::string  & clickhouse_bin,
    const std::string  & output_dir,
    std::string        & err_msg);

// ── E-L2 ─────────────────────────────────────────────────────────────────────

/// Binary stream compare harness_native_path against oracle_native_path byte-by-byte.
/// Active only when build_threads==1 AND max_threads==1.
/// Caller must verify the activation condition before calling (VerifierResult::l2_eligible).
/// On mismatch sets err_msg to:
///   "[HARNESS_ERROR] oracle_mismatch: E-L2 byte-diff mismatch"
/// and returns false.
/// Corresponds to artifact field oracle_l2_pass.
bool checkE_L2(
    const std::string & harness_native_path,
    const std::string & oracle_native_path,
    std::string       & err_msg);

// ── E-Auto ────────────────────────────────────────────────────────────────────

/// Activation threshold: E-Auto only runs when both row counts are within this limit.
static constexpr uint64_t E_AUTO_MAX_ROWS = 20'000'000ULL;

/// Wrap E-L0 + E-L1 into an automatic check.
/// Active only when cfg.build_rows <= E_AUTO_MAX_ROWS AND cfg.probe_rows <= E_AUTO_MAX_ROWS.
///
/// Behaviour on failure:
///   - Emits "[HARNESS_ERROR] oracle_mismatch: <L0|L1> check failed" to stderr
///   - Calls std::exit(1)
///
/// Called from the sweep manager (Track G) on every small-scale invocation.
VerifierResult runAutoVerify(
    const ConfigType & cfg,
    const RunContext & ctx);

// ── Full verifier pipeline ────────────────────────────────────────────────────

/// Run all applicable checks (E-L0, E-L1, E-L2) for one sweep cell.
/// Does NOT exit on failure — returns VerifierResult for the caller to handle.
/// E-L2 is only run when:
///   cfg.build_threads==1 AND ctx.sweep_max_threads==1.
VerifierResult runVerifiers(
    const RunContext & ctx,
    const ConfigType & cfg);

} // namespace DB::HashProbeBench::Verification
