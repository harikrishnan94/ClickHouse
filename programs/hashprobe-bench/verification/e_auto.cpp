/// hashprobe-bench/verification/e_auto.cpp
///
/// H.4  E-Auto: automatic oracle check on small-scale invocations.
///
/// Wraps E-L0 and E-L1 into an automatic check activated whenever:
///   cfg.build_rows <= E_AUTO_MAX_ROWS AND cfg.probe_rows <= E_AUTO_MAX_ROWS
///
/// On failure:
///   - Emits "[HARNESS_ERROR] oracle_mismatch: <L0|L1> check failed" to stderr
///   - Calls std::exit(1)
///
/// Called from the sweep manager (Track G) on every small-scale invocation.

#include "verifier.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace DB::HashProbeBench::Verification
{

VerifierResult runAutoVerify(
    const ConfigType & cfg,
    const RunContext & ctx)
{
    VerifierResult result;

    // E-Auto activates only for small-scale invocations.
    const bool active =
        (cfg.build_rows <= E_AUTO_MAX_ROWS)
        && (cfg.probe_rows <= E_AUTO_MAX_ROWS);

    if (!active)
        return result;

    // E-L0: row count check.
    result.l0_pass = checkE_L0(ctx.harness_row_count, ctx.oracle_row_count, result.error_msg);
    if (!result.l0_pass)
    {
        std::cerr << "[HARNESS_ERROR] oracle_mismatch: E-L0 check failed\n";
        std::exit(1);
    }

    // E-L1: sorted SHA-256 multiset check.
    result.l1_pass = checkE_L1(
        ctx.harness_native_path,
        ctx.oracle_native_path,
        ctx.sql,
        ctx.clickhouse_bin,
        ctx.output_dir,
        result.error_msg);
    if (!result.l1_pass)
    {
        std::cerr << "[HARNESS_ERROR] oracle_mismatch: E-L1 check failed\n";
        std::exit(1);
    }

    return result;
}

} // namespace DB::HashProbeBench::Verification
