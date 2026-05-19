#pragma once

/// hashprobe-bench/oracle/oracle.h
///
/// D.4  clickhouse-local invocation, settings dump, row-count helper, SHA-256.
/// D.5  E-L1 sort+SHA256 comparison.
/// D.6  E-L2 byte-diff: binary compare oracle.native vs harness.native.
///       Active only when build_threads==1 AND sweep_max_threads==1.

#include "oracle_sql.h"
#include <hashprobe_bench/config.h>
#include <hashprobe_bench/types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace DB::HashProbeBench::Oracle
{

inline constexpr const char * DEFAULT_CLICKHOUSE_BINARY
    = "/home/ubuntu/ClickHouse/build_relwithdebinfo/programs/clickhouse";

// ── D.4 ─────────────────────────────────────────────────────────────────────

struct OracleInvokeResult
{
    int         exit_code         = -1;
    std::string oracle_native_path;
    std::string settings_dump_path;
    std::string error_message;
    uint64_t    oracle_row_count  = 0;
};

OracleInvokeResult invokeOracle(
    const OracleSql   & sql,
    const std::string & clickhouse_binary,
    const std::string & output_dir);

int64_t countNativeRows(
    const std::string & native_path,
    const std::string & clickhouse_binary);

std::string sha256File(const std::string & path);

// ── D.5 ─────────────────────────────────────────────────────────────────────

bool checkL1SortHash(
    const OracleSql   & oracle_sql,
    const std::string & oracle_native_path,
    const std::string & harness_native_path,
    const std::string & clickhouse_binary,
    const std::string & output_dir);

// ── D.6 E-L2 byte-diff ────────────────────────────────────────────────────────

/// Binary diff oracle_native_path against harness_native_path.
/// Returns true iff the files are byte-identical.
/// Caller must verify the E-L2 activation condition before calling:
///   cfg.build_threads == 1 AND sweep_max_threads == 1.
bool checkL2ByteDiff(
    const std::string & oracle_native_path,
    const std::string & harness_native_path);

// ── Full pipeline ─────────────────────────────────────────────────────────────

struct OracleCheckResult
{
    OracleInvokeResult invoke;
    bool l0_pass = false;
    bool l1_pass = false;
    bool l2_pass = false;
};

/// Run full pipeline: D.4 invocation + E-L0 + D.5 L1 + D.6 L2.
/// Pass harness_native_path="" to skip L0/L1/L2.
OracleCheckResult runOracleChecks(
    const ConfigType               & cfg,
    const std::vector<std::string> & key_cols,
    const std::string              & build_native_path,
    const std::string              & probe_native_path,
    const std::string              & harness_native_path,
    const std::string              & clickhouse_binary,
    uint32_t                         sweep_max_threads,
    const std::string              & output_dir);

} // namespace DB::HashProbeBench::Oracle
