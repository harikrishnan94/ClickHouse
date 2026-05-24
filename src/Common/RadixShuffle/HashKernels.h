#pragma once

#include <Common/RadixShuffle/HashCombiner.h>
#include <Common/TargetSpecific.h>

#include <cstddef>
#include <cstdint>
#include <cstring>


namespace DB::RadixShuffle
{

/// MurmurHash3 32-bit finalizer.  Analogue of the 64-bit fmix64 used in
/// the old `intHash64Local`.  Applied to per-row scalar payloads after
/// loading into a uint32_t (or as a fold step for wider types).
[[gnu::always_inline]] inline uint32_t fmix32(uint32_t x) noexcept
{
    x ^= x >> 16;
    x *= 0x85ebca6bU;
    x ^= x >> 13;
    x *= 0xc2b2ae35U;
    x ^= x >> 16;
    return x;
}


/// Hash one fixed-width value of type T into a uint32_t.
///
/// For sizeof(T) <= 4 the value is loaded into a uint32_t and passed
/// through fmix32.  For wider types the bytes are folded through fmix32
/// in 4-byte chunks accumulating into a uint32_t state.
template <typename T>
[[gnu::always_inline]] inline uint32_t hashOne32(const T & v) noexcept
{
    if constexpr (sizeof(T) <= sizeof(uint32_t))
    {
        uint32_t buf = 0;
        std::memcpy(&buf, &v, sizeof(T));
        return fmix32(buf);
    }
    else
    {
        const auto * bytes = reinterpret_cast<const unsigned char *>(&v);
        uint32_t acc = 0;
        for (size_t i = 0; i < sizeof(T); i += sizeof(uint32_t))
        {
            uint32_t word = 0;
            const size_t chunk = (i + sizeof(uint32_t) <= sizeof(T)) ? sizeof(uint32_t) : (sizeof(T) - i);
            std::memcpy(&word, bytes + i, chunk);
            acc ^= fmix32(word + acc + 0x9e3779b9U);
        }
        return acc;
    }
}


/// Hash a byte range into a uint32_t.  Analogous to the old `hashBytes`
/// but produces a 32-bit result.
[[gnu::always_inline]] inline uint32_t hashBytes32(const unsigned char * data, size_t n) noexcept
{
    uint32_t acc = 0x811c9dc5U ^ (static_cast<uint32_t>(n) * 0x9e3779b9U);
    size_t i = 0;
    while (i + sizeof(uint32_t) <= n)
    {
        uint32_t word = 0;
        std::memcpy(&word, data + i, sizeof(uint32_t));
        acc = fmix32(word + acc);
        i += sizeof(uint32_t);
    }
    if (i < n)
    {
        uint32_t tail = 0;
        std::memcpy(&tail, data + i, n - i);
        acc = fmix32(tail + acc);
    }
    return acc;
}


// ── Batch hash (SIMD multi-versioned) ────────────────────────────────────────

/// Inner loop body for the batch hash — force-inlined so each ISA wrapper
/// inherits the caller's target attribute and the vectoriser can use the
/// full register width (512-bit ZMM, 256-bit YMM, 128-bit XMM).
///
/// Three code paths, chosen at compile time via if-constexpr:
///   sizeof(T) <= 4  — zero-extend into uint32_t, apply fmix32; fully
///                     independent across iterations, ideal for widening SIMD.
///   sizeof(T) == 8  — split each 64-bit value into lo32/hi32 and apply two
///                     fmix32 rounds (acc depends on lo only, so the outer
///                     loop remains independent across rows).
///   sizeof(T)  > 8  — fall back to per-element hashOne32 (uncommon path).
template <typename T>
[[gnu::always_inline]] inline void hashBatch32Body(const T * __restrict__ keys, int n, uint32_t mask, uint32_t * __restrict__ pids) noexcept
{
    if constexpr (sizeof(T) <= sizeof(uint32_t))
    {
        for (int j = 0; j < n; ++j)
        {
            uint32_t x = 0;
            std::memcpy(&x, &keys[j], sizeof(T));
            x ^= x >> 16;
            x *= 0x85ebca6bU;
            x ^= x >> 13;
            x *= 0xc2b2ae35U;
            x ^= x >> 16;
            pids[j] = x & mask;
        }
    }
    else if constexpr (sizeof(T) == sizeof(uint64_t))
    {
        for (int j = 0; j < n; ++j)
        {
            uint64_t raw;
            std::memcpy(&raw, &keys[j], sizeof(uint64_t));
            const uint32_t lo = static_cast<uint32_t>(raw);
            const uint32_t hi = static_cast<uint32_t>(raw >> 32);

            // First fold: acc = fmix32(lo + MAGIC)  [no dependency on hi]
            uint32_t acc = lo + 0x9e3779b9U;
            acc ^= acc >> 16;
            acc *= 0x85ebca6bU;
            acc ^= acc >> 13;
            acc *= 0xc2b2ae35U;
            acc ^= acc >> 16;

            // Second fold: h2 = fmix32(hi + acc + MAGIC)  [depends on acc only]
            uint32_t h2 = hi + acc + 0x9e3779b9U;
            h2 ^= h2 >> 16;
            h2 *= 0x85ebca6bU;
            h2 ^= h2 >> 13;
            h2 *= 0xc2b2ae35U;
            h2 ^= h2 >> 16;

            pids[j] = (acc ^ h2) & mask;
        }
    }
    else
    {
        // Wide types: no standard ClickHouse column type is wider than 8 bytes
        // in this context, but keep the fallback for completeness.
        for (int j = 0; j < n; ++j)
            pids[j] = hashOne32(keys[j]) & mask;
    }
}

/// SIMD multi-versioned wrappers: v4 (AVX-512), v3 (AVX2), Default (scalar
/// or whatever the project baseline enables).  hashBatch32Body is force-inlined
/// so it gets compiled under each wrapper's target attribute, allowing the
/// auto-vectoriser to use the full available register width.
MULTITARGET_FUNCTION_X86_V4_V3(
    MULTITARGET_FUNCTION_HEADER(template <typename T> inline void),
    hashBatch32Impl,
    MULTITARGET_FUNCTION_BODY((const T * __restrict__ keys, int n, uint32_t mask, uint32_t * __restrict__ pids) noexcept {
        hashBatch32Body(keys, n, mask, pids);
    }))

/// Runtime-dispatch entry point: selects the best ISA variant available on
/// the current CPU (checked once via `isArchSupported` with a static cache).
///
/// A v2 (SSE4.2) wrapper was considered but is unnecessary: this project's
/// baseline is `-march=x86-64-v2`, so the Default `hashBatch32Impl` is already
/// compiled with SSE4.1 `PMULLD` and generates equivalent 128-bit SIMD code.
template <typename T>
inline void hashBatch32(const T * __restrict__ keys, int n, uint32_t mask, uint32_t * __restrict__ pids) noexcept
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
        return hashBatch32Impl_x86_64_v4(keys, n, mask, pids);
    if (isArchSupported(TargetArch::x86_64_v3))
        return hashBatch32Impl_x86_64_v3(keys, n, mask, pids);
#endif
    hashBatch32Impl(keys, n, mask, pids);
}


// ── hashBatch32Acc: hash + optional hashCombine accumulation ─────────────────

/// Unified body for SIMD hash accumulation.
///
/// When `Initial` is true:  out[j] = hash32(keys[j])           (direct overwrite)
/// When `Initial` is false: out[j] = hashCombine(out[j], hash) (accumulate)
///
/// The compile-time `Initial` parameter eliminates the runtime branch and lets
/// the vectoriser generate two independent specialisations.  All rows are
/// independent across j — the loop auto-vectorises under AVX-512, AVX2, SSE4.1.
template <typename T, bool Initial>
[[gnu::always_inline]] inline void hashBatch32AccBody(const T * __restrict__ keys, int n, uint32_t * __restrict__ out) noexcept
{
    if constexpr (sizeof(T) <= sizeof(uint32_t))
    {
        for (int j = 0; j < n; ++j)
        {
            uint32_t x = 0;
            std::memcpy(&x, &keys[j], sizeof(T));
            x ^= x >> 16;
            x *= 0x85ebca6bU;
            x ^= x >> 13;
            x *= 0xc2b2ae35U;
            x ^= x >> 16;
            if constexpr (Initial)
                out[j] = x;
            else
                out[j] = hashCombine(out[j], x);
        }
    }
    else if constexpr (sizeof(T) == sizeof(uint64_t))
    {
        for (int j = 0; j < n; ++j)
        {
            uint64_t raw;
            std::memcpy(&raw, &keys[j], sizeof(uint64_t));
            const uint32_t lo = static_cast<uint32_t>(raw);
            const uint32_t hi = static_cast<uint32_t>(raw >> 32);

            uint32_t acc = lo + 0x9e3779b9U;
            acc ^= acc >> 16;
            acc *= 0x85ebca6bU;
            acc ^= acc >> 13;
            acc *= 0xc2b2ae35U;
            acc ^= acc >> 16;

            uint32_t h2 = hi + acc + 0x9e3779b9U;
            h2 ^= h2 >> 16;
            h2 *= 0x85ebca6bU;
            h2 ^= h2 >> 13;
            h2 *= 0xc2b2ae35U;
            h2 ^= h2 >> 16;

            if constexpr (Initial)
                out[j] = acc ^ h2;
            else
                out[j] = hashCombine(out[j], acc ^ h2);
        }
    }
    else
    {
        for (int j = 0; j < n; ++j)
        {
            const uint32_t h = hashOne32(keys[j]);
            if constexpr (Initial)
                out[j] = h;
            else
                out[j] = hashCombine(out[j], h);
        }
    }
}

MULTITARGET_FUNCTION_X86_V4_V3(
    MULTITARGET_FUNCTION_HEADER(template <typename T, bool Initial> inline void),
    hashBatch32AccImpl,
    MULTITARGET_FUNCTION_BODY((const T * __restrict__ keys, int n, uint32_t * __restrict__ out) noexcept {
        hashBatch32AccBody<T, Initial>(keys, n, out);
    }))

/// Accumulate: out[j] = hashCombine(out[j], hash32(keys[j])).
/// Use when hashing into an existing partial hash (e.g., multi-column composite).
template <typename T>
inline void hashBatch32Combine(const T * __restrict__ keys, int n, uint32_t * __restrict__ out) noexcept
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
        return hashBatch32AccImpl_x86_64_v4<T, false>(keys, n, out);
    if (isArchSupported(TargetArch::x86_64_v3))
        return hashBatch32AccImpl_x86_64_v3<T, false>(keys, n, out);
#endif
    hashBatch32AccImpl<T, false>(keys, n, out);
}

/// Direct: out[j] = hash32(keys[j]).
/// Use for the first (or only) key column; no prior value, no hashCombine overhead.
template <typename T>
inline void hashBatch32Direct(const T * __restrict__ keys, int n, uint32_t * __restrict__ out) noexcept
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
        return hashBatch32AccImpl_x86_64_v4<T, true>(keys, n, out);
    if (isArchSupported(TargetArch::x86_64_v3))
        return hashBatch32AccImpl_x86_64_v3<T, true>(keys, n, out);
#endif
    hashBatch32AccImpl<T, true>(keys, n, out);
}

}
