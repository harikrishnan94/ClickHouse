/// hashprobe-bench/verification/e_l2.cpp
///
/// H.3  E-L2: byte-diff verifier (single-thread only).
///
/// Binary stream compare harness_native_path against oracle_native_path byte-by-byte.
/// Active only when build_threads==1 AND max_threads==1.
/// Caller must verify the E-L2 activation condition before calling.
///
/// Thin wrapper over Oracle::checkL2ByteDiff that adds structured error message.
/// Maps to artifact field oracle_l2_pass.

#include "verifier.h"
#include "../oracle/oracle.h"

#include <string>

namespace DB::HashProbeBench::Verification
{

bool checkE_L2(
    const std::string & harness_native_path,
    const std::string & oracle_native_path,
    std::string       & err_msg)
{
    const bool pass = Oracle::checkL2ByteDiff(oracle_native_path, harness_native_path);

    if (!pass)
    {
        err_msg = "[HARNESS_ERROR] oracle_mismatch: E-L2 byte-diff mismatch";
        return false;
    }
    return true;
}

} // namespace DB::HashProbeBench::Verification
