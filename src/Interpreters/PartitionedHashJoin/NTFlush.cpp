/// Non-temporal store flush — runtime-dispatched per element type.
/// Each concrete getNTFlushFn<T>() checks isArchSupported once and returns
/// the best available function pointer.

#include <Interpreters/PartitionedHashJoin/NTFlush.h>

#include <Common/TargetSpecific.h>

#if USE_MULTITARGET_CODE
#    include <immintrin.h>
#endif

namespace DB
{

// ─── V4 (AVX-512) NT flush implementations ────────────────────────────────────
#if USE_MULTITARGET_CODE

// clang-format off: keep one function per line inside the multi-target macro body
DECLARE_X86_64_V4_SPECIFIC_CODE(

static void flushU64NT(const uint64_t * s, uint64_t ** p, uint32_t idx) noexcept
{
    const uint64_t * src = s + (static_cast<size_t>(idx) * 8);
    _mm512_stream_si512(reinterpret_cast<__m512i *>(p[idx]),
                        _mm512_load_si512(reinterpret_cast<const __m512i *>(src)));
    p[idx] += 8;
}
static void flushU32NT(const uint32_t * s, uint32_t ** p, uint32_t idx) noexcept
{
    const uint32_t * src = s + (static_cast<size_t>(idx) * 8);
    _mm256_stream_si256(reinterpret_cast<__m256i *>(p[idx]),
                        _mm256_load_si256(reinterpret_cast<const __m256i *>(src)));
    p[idx] += 8;
}
static void flushU16NT(const uint16_t * s, uint16_t ** p, uint32_t idx) noexcept
{
    const uint16_t * src = s + (static_cast<size_t>(idx) * 8);
    _mm_stream_si128(reinterpret_cast<__m128i *>(p[idx]),
                     _mm_load_si128(reinterpret_cast<const __m128i *>(src)));
    p[idx] += 8;
}
static void flushU8NT(const uint8_t * s, uint8_t ** p, uint32_t idx) noexcept
{
    const uint8_t * src = s + (static_cast<size_t>(idx) * 8);
    _mm_stream_si64(reinterpret_cast<long long *>(p[idx]),
                    *reinterpret_cast<const long long *>(src));
    p[idx] += 8;
}

) // DECLARE_X86_64_V4_SPECIFIC_CODE
// clang-format on

// clang-format off
DECLARE_X86_64_V3_SPECIFIC_CODE(

static void flushU64NT(const uint64_t * s, uint64_t ** p, uint32_t idx) noexcept
{
    const uint64_t * src = s + (static_cast<size_t>(idx) * 8);
    _mm256_stream_si256(reinterpret_cast<__m256i *>(p[idx]),
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src)));
    _mm256_stream_si256(reinterpret_cast<__m256i *>(p[idx] + 4),
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + 4)));
    p[idx] += 8;
}
static void flushU32NT(const uint32_t * s, uint32_t ** p, uint32_t idx) noexcept
{
    const uint32_t * src = s + (static_cast<size_t>(idx) * 8);
    _mm256_stream_si256(reinterpret_cast<__m256i *>(p[idx]),
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src)));
    p[idx] += 8;
}
static void flushU16NT(const uint16_t * s, uint16_t ** p, uint32_t idx) noexcept
{
    const uint16_t * src = s + (static_cast<size_t>(idx) * 8);
    _mm_stream_si128(reinterpret_cast<__m128i *>(p[idx]),
                     _mm_loadu_si128(reinterpret_cast<const __m128i *>(src)));
    p[idx] += 8;
}
static void flushU8NT(const uint8_t * s, uint8_t ** p, uint32_t idx) noexcept
{
    flushStagingScalar(s, p, idx);
}

) // DECLARE_X86_64_V3_SPECIFIC_CODE
// clang-format on

#endif // USE_MULTITARGET_CODE

// ── Scalar fallback implementations ───────────────────────────────────────────
static void flushU64Scalar(const uint64_t * s, uint64_t ** p, uint32_t idx) noexcept
{
    flushStagingScalar(s, p, idx);
}
static void flushU32Scalar(const uint32_t * s, uint32_t ** p, uint32_t idx) noexcept
{
    flushStagingScalar(s, p, idx);
}
static void flushU16Scalar(const uint16_t * s, uint16_t ** p, uint32_t idx) noexcept
{
    flushStagingScalar(s, p, idx);
}
static void flushU8Scalar(const uint8_t * s, uint8_t ** p, uint32_t idx) noexcept
{
    flushStagingScalar(s, p, idx);
}

// ── Public getNTFlushFn<T>() — explicit instantiations ────────────────────────

template <>
NTFlushFn<uint64_t> getNTFlushFn<uint64_t>() noexcept
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
        return TargetSpecific::x86_64_v4::flushU64NT;
    if (isArchSupported(TargetArch::x86_64_v3))
        return TargetSpecific::x86_64_v3::flushU64NT;
#endif
    return flushU64Scalar;
}

template <>
NTFlushFn<uint32_t> getNTFlushFn<uint32_t>() noexcept
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
        return TargetSpecific::x86_64_v4::flushU32NT;
    if (isArchSupported(TargetArch::x86_64_v3))
        return TargetSpecific::x86_64_v3::flushU32NT;
#endif
    return flushU32Scalar;
}

template <>
NTFlushFn<uint16_t> getNTFlushFn<uint16_t>() noexcept
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
        return TargetSpecific::x86_64_v4::flushU16NT;
    if (isArchSupported(TargetArch::x86_64_v3))
        return TargetSpecific::x86_64_v3::flushU16NT;
#endif
    return flushU16Scalar;
}

template <>
NTFlushFn<uint8_t> getNTFlushFn<uint8_t>() noexcept
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
        return TargetSpecific::x86_64_v4::flushU8NT;
    if (isArchSupported(TargetArch::x86_64_v3))
        return TargetSpecific::x86_64_v3::flushU8NT;
#endif
    return flushU8Scalar;
}

}
