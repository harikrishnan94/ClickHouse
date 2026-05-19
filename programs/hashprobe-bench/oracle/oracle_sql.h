#pragma once

/// hashprobe-bench/oracle/oracle_sql.h
///
/// D.1 SQL emitter: join query for ALL / ANY / RIGHTANY.
/// D.2 Settings prelude: 9 required SET statements.
/// D.3 Native input injection DDL: CREATE TABLE build_t / probe_t from .native files.

#include <hashprobe_bench/config.h>
#include <hashprobe_bench/types.h>

#include <string>
#include <vector>

namespace DB::HashProbeBench::Oracle
{

/// Return {"k0", "k1", ..., "k{n-1}"} for n key columns.
std::vector<std::string> makeKeyColNames(uint32_t n);

/// D.2: Generate the SET statements block (9 required settings).
std::string buildSettingsPrelude(const ConfigType & cfg);

/// D.3: Generate CREATE TABLE build_t / probe_t AS SELECT * FROM file(...).
/// Paths must be absolute.
std::string buildCreateTablesSql(
    const std::string & build_native_path,
    const std::string & probe_native_path);

/// Return the ORDER BY clause for all output columns (E-L1 canonical sort).
/// Order: p_k0..p_k{n-1}, b_k0..b_k{n-1}, p_payload, b_payload, all ASC NULLS FIRST.
std::string buildOrderByClause(const std::vector<std::string> & key_cols);

/// D.1: Generate the full join SELECT query including ORDER BY.
std::string buildJoinQuery(
    const ConfigType & cfg,
    const std::vector<std::string> & key_cols,
    const std::string & build_native_path,
    const std::string & probe_native_path);

/// D.1: Assemble a complete OracleSql struct.
/// full_sql = settings_prelude + create_tables + join_query.
OracleSql buildOracleSql(
    const ConfigType & cfg,
    const std::vector<std::string> & key_cols,
    const std::string & build_native_path,
    const std::string & probe_native_path);

} // namespace DB::HashProbeBench::Oracle
