#pragma once

#include <base/types.h>

#include <cstddef>

namespace DB
{

/// Map each 32-bit hash uniformly to [0, range_size) using the multiply-and-shift method:
///
///     result[i] = (uint64_t{hashes[i]} * range_size) >> 32
///
/// This is strictly more uniform than `h % n` (no bias for non-power-of-2 ranges)
/// and avoids the division instruction.
///
/// SIMD-multi-versioned (x86_64_v4 / x86_64_v3 baseline) via
/// MULTITARGET_FUNCTION_X86_V4.  The inner multiply is expressed as a
/// uint32 × uint32 → uint64 widening product so the compiler emits
/// vpmuludq (1 µop, 0.5c throughput) instead of vpmullq (3 µops).
///
/// range_size must fit in uint32_t (any realistic partition count does).
///
/// UInt32 output — for use with ColumnsScatter::scatter.
/// Halves selector bandwidth in the hot scatter inner loop.
void mapToRange(const UInt32 * hashes, size_t n, UInt32 range_size, UInt32 * result);

} // namespace DB
