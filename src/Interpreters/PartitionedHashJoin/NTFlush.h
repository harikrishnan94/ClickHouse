#pragma once

/// Non-temporal store flush helpers for the SWWC scatter path.
///
/// flushStagingNT<T>(staging_base, out_ptrs, p):
///   - Reads 8 elements of T from staging_base[p*8 .. p*8+7]
///   - NT-stores them to *out_ptrs[p], advances out_ptrs[p] by 8.
///
/// The runtime-dispatched wrapper getNTFlushFn<T>() returns a function pointer
/// to the best available implementation, selected once at column construction.

#include <Common/TargetSpecific.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#if USE_MULTITARGET_CODE
#    include <immintrin.h>
#endif

namespace DB
{

// ── Scalar fallback (works for any T, any target) ─────────────────────────────
template <typename T>
inline void flushStagingScalar(const T * staging_base, T ** out_ptrs, uint32_t p) noexcept
{
    const T * src = staging_base + (static_cast<size_t>(p) * 8);
    T * dst = out_ptrs[p];
    for (size_t s = 0; s < 8; ++s)
        dst[s] = src[s];
    out_ptrs[p] = dst + 8;
}

// ── Per-T function pointer type ───────────────────────────────────────────────
template <typename T>
using NTFlushFn = void (*)(const T * staging_base, T ** out_ptrs, uint32_t p) noexcept;

// ── Get the best flush function for T at runtime ──────────────────────────────
// Defined in NTFlush.cpp (explicit instantiations for T ∈ {u8, u16, u32, u64}).
template <typename T>
NTFlushFn<T> getNTFlushFn() noexcept;

}
