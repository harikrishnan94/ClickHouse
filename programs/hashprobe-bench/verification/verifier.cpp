/// hashprobe-bench/verification/verifier.cpp
///
/// H.4  runVerifiers: full verifier pipeline (E-L0, E-L1, E-L2).
///
/// Runs all applicable checks for one probe sweep cell and returns VerifierResult
/// without exiting.  E-L2 is only run when:
///   cfg.build_threads==1 AND ctx.sweep_max_threads==1.

#include "verifier.h"

namespace DB::HashProbeBench::Verification
{

VerifierResult runVerifiers(
    const RunContext & ctx,
    const ConfigType & cfg)
{
    VerifierResult result;

    // E-L0: row count.
    result.l0_pass = checkE_L0(ctx.harness_row_count, ctx.oracle_row_count, result.error_msg);

    // E-L1: sorted SHA-256.  Run unconditionally (regardless of L0 outcome) to
    // capture independent diagnostic signal.
    if (!ctx.harness_native_path.empty() && !ctx.oracle_native_path.empty())
    {
        result.l1_pass = checkE_L1(
            ctx.harness_native_path,
            ctx.oracle_native_path,
            ctx.sql,
            ctx.clickhouse_bin,
            ctx.output_dir,
            result.error_msg);

        // E-L2: byte-identical.  Only active for single-thread runs.
        const bool l2_eligible =
            (cfg.build_threads == 1)
            && (ctx.sweep_max_threads == 1);
        result.l2_eligible = l2_eligible;

        if (l2_eligible)
            result.l2_pass = checkE_L2(
                ctx.harness_native_path,
                ctx.oracle_native_path,
                result.error_msg);
    }

    return result;
}

} // namespace DB::HashProbeBench::Verification
