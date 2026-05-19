/// hashprobe-bench/verification/e_l1.cpp
///
/// H.2  E-L1: sorted SHA-256 multiset check.
///
/// Sort both harness and oracle outputs via clickhouse-local using the canonical
/// ORDER BY clause (OracleSql::order_clause), then compare SHA-256 hashes of
/// the sorted TSV outputs.
///
/// Thin wrapper over Oracle::checkL1SortHash that adds structured error message.
/// Maps to artifact field oracle_l1_pass.

#include "verifier.h"
#include "../oracle/oracle.h"

#include <string>

namespace DB::HashProbeBench::Verification
{

bool checkE_L1(
    const std::string & harness_native_path,
    const std::string & oracle_native_path,
    const OracleSql   & sql,
    const std::string & clickhouse_bin,
    const std::string & output_dir,
    std::string       & err_msg)
{
    const bool pass = Oracle::checkL1SortHash(
        sql,
        oracle_native_path,
        harness_native_path,
        clickhouse_bin,
        output_dir);

    if (!pass)
    {
        err_msg = "[HARNESS_ERROR] oracle_mismatch: E-L1 sorted-hash mismatch";
        return false;
    }
    return true;
}

} // namespace DB::HashProbeBench::Verification
