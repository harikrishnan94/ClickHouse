/// Radix shuffle hash — Phase 1 of the batched 4-phase partition algorithm.
///
/// hashOneKeyIntoIds: hashes one key column (raw data pointer, element size,
/// row count) into pids[]. Multi-key combine: call once per key column;
/// first=true initialises, first=false XOR-combines.
///
/// Three compiled variants, runtime-dispatched:
///   x86_64-v4 (AVX-512DQ) — 8 keys per ZMM, VPMULLQ
///   x86_64-v3 (AVX2)      — 4 keys per YMM, lane-by-lane 64-bit mul emulation
///   Default (scalar)      — one row at a time, plain splitmix64
///
/// Mirrors the multi-target dispatch pattern in:
///   src/Storages/MergeTree/MergeTreeRangeReader.cpp:747-826

#include <Interpreters/PartitionedHashJoin/RadixShuffleHash.h>

#include <Common/TargetSpecific.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#if USE_MULTITARGET_CODE
#    include <immintrin.h>
#endif

namespace DB
{

// ── Scalar hash (splitmix64 finalizer) ────────────────────────────────────────
static inline uint64_t mix64(uint64_t x) noexcept
{
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

/// Scalar: load up to 8 bytes via memcpy → hash → combine with pids[].
/// pids is uint16_t (P ≤ 1024 fits in 10 bits, so mask & 0xFFFF is sufficient).
static void hashScalarImpl(const void * data, size_t elem_sz, size_t rows, uint64_t mask, uint16_t * pids, bool first)
{
    const uint8_t * p = static_cast<const uint8_t *>(data);
    const uint16_t mask16 = static_cast<uint16_t>(mask);
    for (size_t i = 0; i < rows; ++i)
    {
        uint64_t kv = 0;
        std::memcpy(&kv, p + i * elem_sz, elem_sz);
        const uint16_t h = static_cast<uint16_t>(mix64(kv)) & mask16;
        pids[i] = first ? h : static_cast<uint16_t>((pids[i] ^ h) & mask16);
    }
}


// ── x86_64-v4 (AVX-512DQ) specific implementation ────────────────────────────
DECLARE_X86_64_V4_SPECIFIC_CODE(

    /// Process 8 uint64 keys per ZMM register, store 8 uint16_t pids per __m128i.
    /// Truncation to uint16 is via VPMOVQW (_mm512_cvtepi64_epi16) — single µop on
    /// modern AVX-512 hardware. Halves the store traffic vs the previous uint32_t path.
    static void hashU64x8Impl(const uint64_t * __restrict__ data, size_t rows, uint64_t mask, uint16_t * __restrict__ pids, bool first) {
        const __m512i vmask64 = _mm512_set1_epi64(static_cast<int64_t>(mask));
        const __m128i vmask16 = _mm_set1_epi16(static_cast<int16_t>(mask));
        const __m512i M1 = _mm512_set1_epi64(static_cast<int64_t>(UINT64_C(0xbf58476d1ce4e5b9)));
        const __m512i M2 = _mm512_set1_epi64(static_cast<int64_t>(UINT64_C(0x94d049bb133111eb)));

        size_t i = 0;
        for (; i + 8 <= rows; i += 8)
        {
            __m512i k = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(data + i));
            k = _mm512_xor_epi64(k, _mm512_srli_epi64(k, 30));
            k = _mm512_mullo_epi64(k, M1); // VPMULLQ (AVX-512DQ)
            k = _mm512_xor_epi64(k, _mm512_srli_epi64(k, 27));
            k = _mm512_mullo_epi64(k, M2);
            k = _mm512_xor_epi64(k, _mm512_srli_epi64(k, 31));
            /// 8x u64 → 8x u16 truncate (low 16 bits of each lane).
            const __m128i h16 = _mm512_cvtepi64_epi16(_mm512_and_epi64(k, vmask64));
            if (first)
            {
                _mm_storeu_si128(reinterpret_cast<__m128i *>(pids + i), h16);
            }
            else
            {
                const __m128i cur = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pids + i));
                const __m128i xrd = _mm_and_si128(_mm_xor_si128(cur, h16), vmask16);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(pids + i), xrd);
            }
        }
        /// scalar tail for remaining rows
        const uint16_t mask16 = static_cast<uint16_t>(mask);
        for (; i < rows; ++i)
        {
            const uint16_t h = static_cast<uint16_t>(mix64(data[i])) & mask16;
            pids[i] = first ? h : static_cast<uint16_t>((pids[i] ^ h) & mask16);
        }
    }

    void hashOneKeyIntoIdsImpl(const void * data, size_t elem_sz, size_t n, uint64_t mask, uint16_t * pids, bool first) {
        if (elem_sz == 8)
            hashU64x8Impl(static_cast<const uint64_t *>(data), n, mask, pids, first);
        else
            hashScalarImpl(data, elem_sz, n, mask, pids, first);
    }

    ) // DECLARE_X86_64_V4_SPECIFIC_CODE


// ── x86_64-v3 (AVX2) specific implementation ─────────────────────────────────
DECLARE_X86_64_V3_SPECIFIC_CODE(

    /// 64-bit multiply via AVX2 (VPMULLQ is AVX-512DQ only).
    /// Identity: a*b = lo_lo + (lo_hi + hi_lo) << 32
    static inline __m256i mul64_avx2(__m256i a, __m256i b) {
        const __m256i lo_lo = _mm256_mul_epu32(a, b);
        const __m256i a_hi = _mm256_srli_epi64(a, 32);
        const __m256i lo_hi = _mm256_mul_epu32(a_hi, b);
        const __m256i b_hi = _mm256_srli_epi64(b, 32);
        const __m256i hi_lo = _mm256_mul_epu32(a, b_hi);
        const __m256i cross = _mm256_slli_epi64(_mm256_add_epi64(lo_hi, hi_lo), 32);
        return _mm256_add_epi64(lo_lo, cross);
    }

    /// Pack 4× uint64 from __m256i (each value ≤ 0xFFFF) into 4× uint16 (8 bytes).
    /// Strategy: byte-shuffle within each 128-bit lane to gather bytes {0,1} of u64-lane-0
    /// and bytes {8,9} of u64-lane-1, producing two __m128i halves each holding 2× u16,
    /// then interleave the low 32 bits of each half into a single 64-bit result.
    static inline __m128i pack_u64x4_to_u16x4(__m256i k) {
        const __m128i shuf = _mm_setr_epi8(0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i lo = _mm_shuffle_epi8(_mm256_castsi256_si128(k), shuf); // [u16_0, u16_1, 0, ..., 0]
        const __m128i hi = _mm_shuffle_epi8(_mm256_extracti128_si256(k, 1), shuf); // [u16_2, u16_3, 0, ..., 0]
        /// Interleave low 32 bits: result low-8-bytes = [u16_0, u16_1, u16_2, u16_3].
        return _mm_unpacklo_epi32(lo, hi);
    }

    static void hashU64x4Impl(const uint64_t * __restrict__ data, size_t rows, uint64_t mask, uint16_t * __restrict__ pids, bool first) {
        const __m256i vmask64 = _mm256_set1_epi64x(static_cast<int64_t>(mask));
        const __m128i vmask16 = _mm_set1_epi16(static_cast<int16_t>(mask));
        const __m256i M1 = _mm256_set1_epi64x(static_cast<int64_t>(UINT64_C(0xbf58476d1ce4e5b9)));
        const __m256i M2 = _mm256_set1_epi64x(static_cast<int64_t>(UINT64_C(0x94d049bb133111eb)));

        size_t i = 0;
        for (; i + 4 <= rows; i += 4)
        {
            __m256i k = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i));
            k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 30));
            k = mul64_avx2(k, M1);
            k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 27));
            k = mul64_avx2(k, M2);
            k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 31));
            k = _mm256_and_si256(k, vmask64);
            const __m128i h16 = pack_u64x4_to_u16x4(k);
            if (first)
            {
                _mm_storel_epi64(reinterpret_cast<__m128i *>(pids + i), h16);
            }
            else
            {
                /// Load 4× u16 (8 bytes) without aliasing risk.
                int64_t cur_bits;
                std::memcpy(&cur_bits, pids + i, sizeof(cur_bits));
                const __m128i cur = _mm_cvtsi64_si128(cur_bits);
                const __m128i xrd = _mm_and_si128(_mm_xor_si128(cur, h16), vmask16);
                _mm_storel_epi64(reinterpret_cast<__m128i *>(pids + i), xrd);
            }
        }
        const uint16_t mask16 = static_cast<uint16_t>(mask);
        for (; i < rows; ++i)
        {
            const uint16_t h = static_cast<uint16_t>(mix64(data[i])) & mask16;
            pids[i] = first ? h : static_cast<uint16_t>((pids[i] ^ h) & mask16);
        }
    }

    void hashOneKeyIntoIdsImpl(const void * data, size_t elem_sz, size_t n, uint64_t mask, uint16_t * pids, bool first) {
        if (elem_sz == 8)
            hashU64x4Impl(static_cast<const uint64_t *>(data), n, mask, pids, first);
        else
            hashScalarImpl(data, elem_sz, n, mask, pids, first);
    }

    ) // DECLARE_X86_64_V3_SPECIFIC_CODE


// ── Default (scalar) implementation ───────────────────────────────────────────
DECLARE_DEFAULT_CODE(

    void hashOneKeyIntoIdsImpl(const void * data, size_t elem_sz, size_t n, uint64_t mask, uint16_t * pids, bool first) {
        hashScalarImpl(data, elem_sz, n, mask, pids, first);
    }

    ) // DECLARE_DEFAULT_CODE


// ── Runtime-dispatch wrapper ───────────────────────────────────────────────────
void hashOneKeyIntoIds(const void * data, size_t elem_sz, size_t n, uint64_t mask, uint16_t * pids, bool first)
{
    if (n == 0)
        return;

#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
    {
        TargetSpecific::x86_64_v4::hashOneKeyIntoIdsImpl(data, elem_sz, n, mask, pids, first);
        return;
    }
    if (isArchSupported(TargetArch::x86_64_v3))
    {
        TargetSpecific::x86_64_v3::hashOneKeyIntoIdsImpl(data, elem_sz, n, mask, pids, first);
        return;
    }
#endif
    TargetSpecific::Default::hashOneKeyIntoIdsImpl(data, elem_sz, n, mask, pids, first);
}

}
