/// hashprobe-bench/oracle/oracle_sql.cpp
///
/// D.1 SQL emitter: join query generation.
/// D.2 Settings prelude: 9 required SET statements.
/// D.3 Native input injection: CREATE TABLE build_t/probe_t DDL + ORDER BY clause.
///
/// Column naming convention (matching harness build_driver and BlockBuilder):
///   Probe (left) side:  k0, k1, ..., payload
///   Build (right) side: b_k0, b_k1, ..., b_payload
///
/// Output schema for both oracle and harness: k0, payload, b_k0, b_payload
/// (one key group for N>1: k0, k1, ..., payload, b_k0, b_k1, ..., b_payload)
/// E-L1 uses ORDER BY ALL NULLS FIRST (schema-agnostic, works with any column names).

#include "oracle_sql.h"

#include <sstream>

namespace DB::HashProbeBench::Oracle
{

std::vector<std::string> makeKeyColNames(uint32_t n)
{
    // Probe-side key column names: "k0", "k1", ..., "k{n-1}" — matches BlockBuilder.
    std::vector<std::string> cols;
    cols.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
        cols.push_back("k" + std::to_string(i));
    return cols;
}

// ── D.2 Settings prelude ─────────────────────────────────────────────────────

std::string buildSettingsPrelude(const ConfigType & cfg)
{
    std::ostringstream ss;
    ss << "SET join_algorithm = '"
       << (cfg.build_threads == 1 ? "hash" : "parallel_hash") << "';\n";
    ss << "SET max_threads = " << cfg.build_threads << ";\n";
    ss << "SET max_block_size = " << cfg.block_size << ";\n";
    ss << "SET max_joined_block_size_rows = " << cfg.max_joined_block_size_rows << ";\n";
    ss << "SET max_joined_block_size_bytes = 0;\n";
    ss << "SET compile_expressions = 0;\n";
    ss << "SET compile_aggregate_expressions = 0;\n";
    ss << "SET enable_analyzer = 0;\n";
    ss << "SET any_join_distinct_right_table_keys = 0;\n";
    return ss.str();
}

// ── D.3 CREATE TABLE DDL and ORDER BY ────────────────────────────────────────
//
// NOTE: buildCreateTablesSql is NOT called in buildOracleSql.
// The spec (oracle definition) describes loading files into Memory tables first,
// then joining probe_t to build_t.  This implementation instead uses file()
// table functions directly in the JOIN query (see buildJoinQuery).
//
// Rationale: ClickHouse's Memory table engine may return blocks in sizes that
// differ from the on-disk block layout.  Joining file() directly causes
// clickhouse-local to process the native files in the same max_block_size chunks
// as the harness feeds to joinBlock, making E-L2 byte-identical comparison valid.
//
// This function is retained so the oracle SQL structure matches the spec intent
// (create tables → join); in practice the harness emits file() expressions inline.
// See the top-of-file comment for a full explanation.
std::string buildCreateTablesSql(
    const std::string & /*build_native_path*/,
    const std::string & /*probe_native_path*/)
{
    // Intentionally returns empty string: the join query uses file() directly.
    return "";
}

std::string buildOrderByClause(const std::vector<std::string> & /*key_cols*/)
{
    // Use schema-agnostic ORDER BY ALL NULLS FIRST.
    // Both harness and oracle output have identical column order:
    //   k0 [, k1, ...], payload, b_k0 [, b_k1, ...], b_payload
    // ORDER BY ALL sorts by column position, which is consistent across both.
    return "ORDER BY ALL NULLS FIRST";
}

// ── D.1 Join query emitter ────────────────────────────────────────────────────
//
// Output column schema to match hashJoin output: k0..k{n-1}, payload, b_k0..b_k{n-1}, b_payload
// Build side uses "b_k{i}" and "b_payload" names to avoid collision with probe side.

std::string buildJoinQuery(
    const ConfigType & cfg,
    const std::vector<std::string> & key_cols,
    const std::string & build_native_path,
    const std::string & probe_native_path)
{
    // Probe table ref: file() directly — preserves exact block sizes for E-L2.
    const std::string probe_ref = "file('" + probe_native_path + "', 'Native')";
    const std::string build_ref = "file('" + build_native_path + "', 'Native')";

    std::ostringstream ss;

    // SELECT: probe keys (k0..), probe payload, then build keys (b_k0..), build payload
    ss << "SELECT ";
    for (const auto & kc : key_cols)
        ss << "probe_t." << kc << " AS " << kc << ", ";
    ss << "probe_t.payload AS payload, ";

    if (cfg.strictness == StrictnessConfig::RIGHTANY)
    {
        // RIGHTANY: dedup build side with GROUP BY, use any(b_payload).
        // Build side in build.native has columns: b_k0, b_k1, ..., b_payload.
        for (const auto & kc : key_cols)
            ss << "build_dedup.b_" << kc << " AS b_" << kc << ", ";
        ss << "build_dedup.b_payload AS b_payload\n";

        ss << "FROM " << probe_ref << " AS probe_t\n";
        ss << "INNER JOIN (\n";
        ss << "    SELECT ";
        for (size_t i = 0; i < key_cols.size(); ++i)
        {
            if (i > 0) ss << ", ";
            ss << "b_" << key_cols[i];
        }
        ss << ", any(b_payload) AS b_payload\n";
        ss << "    FROM " << build_ref << " AS build_t\n";
        ss << "    GROUP BY ";
        for (size_t i = 0; i < key_cols.size(); ++i)
        {
            if (i > 0) ss << ", ";
            ss << "b_" << key_cols[i];
        }
        ss << "\n) AS build_dedup\n";
        ss << "ON ";
        for (size_t i = 0; i < key_cols.size(); ++i)
        {
            if (i > 0) ss << " AND ";
            ss << "probe_t." << key_cols[i] << " = build_dedup.b_" << key_cols[i];
        }
    }
    else
    {
        // ALL / ANY: join directly against build_t.
        // Build side columns: b_k0, b_k1, ..., b_payload.
        for (const auto & kc : key_cols)
            ss << "build_t.b_" << kc << " AS b_" << kc << ", ";
        ss << "build_t.b_payload AS b_payload\n";

        ss << "FROM " << probe_ref << " AS probe_t\n";
        if (cfg.strictness == StrictnessConfig::ANY)
            ss << "INNER ANY JOIN " << build_ref << " AS build_t\n";
        else
            ss << "INNER JOIN " << build_ref << " AS build_t\n";

        ss << "ON ";
        for (size_t i = 0; i < key_cols.size(); ++i)
        {
            if (i > 0) ss << " AND ";
            ss << "probe_t." << key_cols[i] << " = build_t.b_" << key_cols[i];
        }
    }
    // NOTE: No ORDER BY here — oracle.native is unsorted for E-L2 byte comparison.
    // E-L1 applies ORDER BY ALL NULLS FIRST at comparison time via checkL1SortHash.
    ss << ";\n";

    return ss.str();
}

// ── D.1 OracleSql assembly ────────────────────────────────────────────────────

OracleSql buildOracleSql(
    const ConfigType & cfg,
    const std::vector<std::string> & key_cols,
    const std::string & build_native_path,
    const std::string & probe_native_path)
{
    OracleSql result;
    result.settings_prelude = buildSettingsPrelude(cfg);
    result.create_tables    = "";  // see buildCreateTablesSql — join uses file() directly
    result.join_query       = buildJoinQuery(cfg, key_cols, build_native_path, probe_native_path);
    result.order_clause     = buildOrderByClause(key_cols);
    // full_sql: settings prelude + join query (no CREATE TABLE block; file() used inline)
    result.full_sql         = result.settings_prelude
                            + result.join_query;
    return result;
}

} // namespace DB::HashProbeBench::Oracle
