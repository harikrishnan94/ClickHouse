/// hashprobe-bench/verification/e_l0.cpp
///
/// H.1  E-L0: row count check.
///
/// Compares total output row count from the harness against the oracle.
/// On mismatch sets err_msg to:
///   "[HARNESS_ERROR] oracle_mismatch: E-L0 row count mismatch: harness=N oracle=M"
/// Maps to artifact field oracle_l0_pass.

#include "verifier.h"

#include <string>

namespace DB::HashProbeBench::Verification
{

bool checkE_L0(uint64_t harness_row_count, uint64_t oracle_row_count, std::string & err_msg)
{
    if (harness_row_count == oracle_row_count)
        return true;

    err_msg = "[HARNESS_ERROR] oracle_mismatch: E-L0 row count mismatch: harness="
            + std::to_string(harness_row_count)
            + " oracle="
            + std::to_string(oracle_row_count);
    return false;
}

} // namespace DB::HashProbeBench::Verification
