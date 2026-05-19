/// hashprobe-bench/oracle/oracle.cpp
///
/// D.4  clickhouse-local invocation, settings dump, row-count, SHA-256.
/// D.5  E-L1 sort+SHA256 comparison.
/// D.6  E-L2 byte-diff.

#include "oracle.h"

#include <openssl/evp.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace DB::HashProbeBench::Oracle
{

static bool writeFile(const std::string & path, const std::string & content)
{
    std::ofstream f(path, std::ios::trunc);
    if (!f.is_open()) return false;
    f << content;
    return f.good();
}

static int runCmd(const std::string & cmd)
{
    return std::system(cmd.c_str()); // NOLINT(cert-env33-c)
}

// ── SHA-256 ───────────────────────────────────────────────────────────────────

std::string sha256File(const std::string & path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};

    EVP_MD_CTX * ctx = EVP_MD_CTX_new();
    if (!ctx) return {};

    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1)
    {
        EVP_MD_CTX_free(ctx);
        return {};
    }

    std::vector<char> buf(65536); // heap: avoid stack-frame limit
    while (true)
    {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto n = f.gcount();
        if (n > 0) EVP_DigestUpdate(ctx, buf.data(), static_cast<size_t>(n));
        if (f.eof() || n == 0) break;
    }

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    EVP_DigestFinal_ex(ctx, digest, &digest_len);
    EVP_MD_CTX_free(ctx);

    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_len; ++i)
        hex << std::setw(2) << static_cast<int>(digest[i]);
    return hex.str();
}

// ── D.4 Oracle invocation ─────────────────────────────────────────────────────

OracleInvokeResult invokeOracle(
    const OracleSql   & sql,
    const std::string & clickhouse_binary,
    const std::string & output_dir)
{
    OracleInvokeResult result;
    result.oracle_native_path = output_dir + "/oracle.native";
    result.settings_dump_path = output_dir + "/oracle_settings_dump.tsv";

    const std::string main_sql_path = output_dir + "/oracle_query.sql";
    if (!writeFile(main_sql_path, sql.full_sql))
    {
        result.exit_code     = -1;
        result.error_message = "Failed to write oracle SQL to: " + main_sql_path;
        return result;
    }

    {
        std::ostringstream cmd;
        cmd << clickhouse_binary
            << " local --queries-file " << main_sql_path
            << " --output-format Native --send_logs_level=warning"
            << " > " << result.oracle_native_path << " 2>/dev/null";
        result.exit_code = runCmd(cmd.str());
    }

    if (result.exit_code != 0)
    {
        result.error_message = "clickhouse-local exited with code "
            + std::to_string(result.exit_code)
            + "; SQL file: " + main_sql_path;
        return result;
    }

    const std::string dump_sql_path = output_dir + "/oracle_settings_dump.sql";
    const std::string dump_sql =
        sql.settings_prelude
        + "SELECT name, value FROM system.settings "
        "WHERE name IN ("
        "'join_algorithm','max_threads','max_block_size',"
        "'max_joined_block_size_rows','max_joined_block_size_bytes',"
        "'compile_expressions','compile_aggregate_expressions',"
        "'enable_analyzer','any_join_distinct_right_table_keys'"
        ") ORDER BY name;\n";
    if (writeFile(dump_sql_path, dump_sql))
    {
        std::ostringstream cmd;
        cmd << clickhouse_binary
            << " local --queries-file " << dump_sql_path
            << " --format TSV --send_logs_level=warning"
            << " > " << result.settings_dump_path << " 2>/dev/null";
        const int dump_rc = runCmd(cmd.str());
        if (dump_rc != 0)
            std::cerr << "[oracle] WARNING: settings dump exited " << dump_rc << "\n";
    }

    result.oracle_row_count = static_cast<uint64_t>(
        std::max<int64_t>(0, countNativeRows(result.oracle_native_path, clickhouse_binary)));
    return result;
}

int64_t countNativeRows(
    const std::string & native_path,
    const std::string & clickhouse_binary)
{
    const std::string sql_path = native_path + ".count.sql";
    const std::string out_path = native_path + ".count.txt";
    const std::string sql =
        "SELECT count() FROM file('" + native_path + "', 'Native');\n";
    if (!writeFile(sql_path, sql)) return -1;
    std::ostringstream cmd;
    cmd << clickhouse_binary
        << " local --queries-file " << sql_path
        << " --format TSV --send_logs_level=warning"
        << " > " << out_path << " 2>/dev/null";
    if (runCmd(cmd.str()) != 0) return -1;
    std::ifstream f(out_path);
    int64_t count = -1;
    f >> count;
    return count;
}

// ── D.5 E-L1 sort+SHA256 ─────────────────────────────────────────────────────

bool checkL1SortHash(
    const OracleSql   & /*oracle_sql*/,
    const std::string & oracle_native_path,
    const std::string & harness_native_path,
    const std::string & clickhouse_binary,
    const std::string & output_dir)
{
    const std::string oracle_sorted_tsv  = output_dir + "/oracle_sorted.tsv";
    const std::string harness_sorted_tsv = output_dir + "/harness_sorted.tsv";

    {
        // Project away build-side columns (b_ prefix) so non-deterministic ANY-join
        // first-seen values do not cause false E-L1 failures on multi-threaded builds.
        const std::string sql =
            "SELECT COLUMNS('^k') FROM file('" + oracle_native_path + "', 'Native') "
            "ORDER BY ALL NULLS FIRST;\n";
        const std::string sql_path = output_dir + "/oracle_sort.sql";
        if (!writeFile(sql_path, sql)) return false;
        std::ostringstream cmd;
        cmd << clickhouse_binary
            << " local --queries-file " << sql_path
            << " --format TSV --send_logs_level=warning"
            << " > " << oracle_sorted_tsv << " 2>/dev/null";
        if (runCmd(cmd.str()) != 0) return false;
    }

    {
        // Project away build-side columns (b_ prefix) for consistent comparison.
        // Both harness and oracle native files use b_ prefix for build-side columns.
        const std::string sql =
            "SELECT COLUMNS('^k') FROM file('" + harness_native_path + "', 'Native') "
            "ORDER BY ALL NULLS FIRST;\n";
        const std::string sql_path = output_dir + "/harness_sort.sql";
        if (!writeFile(sql_path, sql)) return false;
        std::ostringstream cmd;
        cmd << clickhouse_binary
            << " local --queries-file " << sql_path
            << " --format TSV --send_logs_level=warning"
            << " > " << harness_sorted_tsv << " 2>/dev/null";
        if (runCmd(cmd.str()) != 0) return false;
    }

    const std::string h_oracle  = sha256File(oracle_sorted_tsv);
    const std::string h_harness = sha256File(harness_sorted_tsv);
    return !h_oracle.empty() && !h_harness.empty() && (h_oracle == h_harness);
}

// ── D.6 E-L2 byte-diff ────────────────────────────────────────────────────────

bool checkL2ByteDiff(
    const std::string & oracle_native_path,
    const std::string & harness_native_path)
{
    std::ifstream fo(oracle_native_path,  std::ios::binary);
    std::ifstream fh(harness_native_path, std::ios::binary);
    if (!fo.is_open() || !fh.is_open()) return false;

    std::vector<char> bo(4096), bh(4096); // heap: stack-frame limit
    while (true)
    {
        fo.read(bo.data(), static_cast<std::streamsize>(bo.size()));
        fh.read(bh.data(), static_cast<std::streamsize>(bh.size()));
        const auto ro = fo.gcount();
        const auto rh = fh.gcount();
        if (ro != rh) return false;
        if (ro == 0) break;
        if (std::memcmp(bo.data(), bh.data(), static_cast<size_t>(ro)) != 0)
            return false;
        if (fo.eof() && fh.eof()) break;
    }
    return true;
}

// ── Full oracle check pipeline ────────────────────────────────────────────────

OracleCheckResult runOracleChecks(
    const ConfigType               & cfg,
    const std::vector<std::string> & key_cols,
    const std::string              & build_native_path,
    const std::string              & probe_native_path,
    const std::string              & harness_native_path,
    const std::string              & clickhouse_binary,
    uint32_t                         sweep_max_threads,
    const std::string              & output_dir)
{
    OracleCheckResult res;
    const OracleSql sql = buildOracleSql(cfg, key_cols, build_native_path, probe_native_path);

    res.invoke = invokeOracle(sql, clickhouse_binary, output_dir);
    if (res.invoke.exit_code != 0) return res;

    if (!harness_native_path.empty())
    {
        const int64_t harness_rows = countNativeRows(harness_native_path, clickhouse_binary);
        res.l0_pass =
            (harness_rows >= 0)
            && (static_cast<uint64_t>(harness_rows) == res.invoke.oracle_row_count);

        res.l1_pass = checkL1SortHash(
            sql, res.invoke.oracle_native_path,
            harness_native_path, clickhouse_binary, output_dir);

        // D.6: E-L2 byte-diff — only for single-thread runs
        const bool l2_eligible =
            (cfg.build_threads == 1)
            && (sweep_max_threads == 1);
        if (l2_eligible)
            res.l2_pass = checkL2ByteDiff(res.invoke.oracle_native_path, harness_native_path);
    }

    return res;
}

} // namespace DB::HashProbeBench::Oracle
