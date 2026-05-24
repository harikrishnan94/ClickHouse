#include <Common/RadixShuffle/RadixPartitionOperator.h>

#include <Columns/ColumnVector.h>
#include <Common/TargetSpecific.h>
#include <Common/assert_cast.h>

#if defined(__x86_64__)
#    include <immintrin.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>


namespace DB::RadixShuffle
{

// ── portable scalar hash (mirrors reference `mix`) ───────────────────────────

[[gnu::always_inline]] inline uint64_t mix64(uint64_t x) noexcept
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}


/// Load TKey as a uint64_t for hashing.  For sizeof(T) ≤ 8 the low bytes are
/// used (zero-padded); for wider types bytes are folded by XOR.
template <typename TKey>
[[gnu::always_inline]] inline uint64_t toU64ForHash(const TKey & v) noexcept
{
    if constexpr (sizeof(TKey) <= sizeof(uint64_t))
    {
        uint64_t u = 0;
        std::memcpy(&u, &v, sizeof(TKey));
        return u;
    }
    else
    {
        // Fold 64-bit chunks via XOR.
        uint64_t acc = 0;
        const auto * bytes = reinterpret_cast<const unsigned char *>(&v);
        for (size_t i = 0; i + 8 <= sizeof(TKey); i += 8)
        {
            uint64_t chunk = 0;
            std::memcpy(&chunk, bytes + i, 8);
            acc ^= chunk;
        }
        if constexpr (sizeof(TKey) % 8 != 0)
        {
            constexpr size_t tail_off = (sizeof(TKey) / 8) * 8;
            uint64_t chunk = 0;
            std::memcpy(&chunk, bytes + tail_off, sizeof(TKey) % 8);
            acc ^= chunk;
        }
        return acc;
    }
}


// ── SIMD hash kernels (sizeof(TKey)==8 only) ───────────────────────────────────
// Placed inside DB::RadixShuffle so they are callable as
// TargetSpecific::<arch>::hashBatch<T>(...) from within this namespace.

#if USE_MULTITARGET_CODE

DECLARE_X86_64_V2_SPECIFIC_CODE(

    /// 64-bit low-half multiply via SSE2 `PMULUDQ` (no native `mullo_epi64`).
    [[gnu::always_inline]] inline __m128i mullo_epi64_sse(__m128i a, __m128i b) noexcept {
        const __m128i bswap = _mm_shuffle_epi32(b, _MM_SHUFFLE(2, 3, 0, 1));
        __m128i prod02 = _mm_mul_epu32(a, b);
        const __m128i prod13 = _mm_mul_epu32(a, bswap);
        prod02 = _mm_slli_epi64(prod02, 32);
        return _mm_blend_epi16(prod02, prod13, 0xCC);
    }

    [[gnu::always_inline]] inline __m128i simd_mix(__m128i x) noexcept {
        const __m128i m1 = _mm_set1_epi64x(static_cast<int64_t>(0xbf58476d1ce4e5b9ULL));
        const __m128i m2 = _mm_set1_epi64x(static_cast<int64_t>(0x94d049bb133111ebULL));
        x = _mm_xor_si128(x, _mm_srli_epi64(x, 30));
        x = mullo_epi64_sse(x, m1);
        x = _mm_xor_si128(x, _mm_srli_epi64(x, 27));
        x = mullo_epi64_sse(x, m2);
        x = _mm_xor_si128(x, _mm_srli_epi64(x, 31));
        return x;
    }

    /// Hash `n` keys; SSE2 path processes 2 × 64-bit values per iteration.
    template <typename TKey>
    void hashBatch(const TKey * src, size_t start, int n, uint64_t mask, uint32_t * pids) {
        if constexpr (sizeof(TKey) == 8)
        {
            const __m128i vmask = _mm_set1_epi64x(static_cast<int64_t>(mask));
            const int pairs = n / 2;
            for (int g = 0; g < pairs; ++g)
            {
                const __m128i k = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + start + static_cast<size_t>(g) * 2));
                const __m128i hashed = _mm_and_si128(simd_mix(k), vmask);
                const __m128i packed = _mm_shuffle_epi32(hashed, _MM_SHUFFLE(2, 2, 0, 0));
                _mm_storel_epi64(reinterpret_cast<__m128i *>(pids + g * 2), packed);
            }
            for (int j = pairs * 2; j < n; ++j)
                pids[j] = static_cast<uint32_t>(mix64(toU64ForHash(src[start + j])) & mask);
        }
        else
        {
            for (int j = 0; j < n; ++j)
                pids[j] = static_cast<uint32_t>(mix64(toU64ForHash(src[start + j])) & mask);
        }
    }

    ) // DECLARE_X86_64_V2_SPECIFIC_CODE

DECLARE_X86_64_V3_SPECIFIC_CODE(

    /// 64-bit low-half multiply via AVX2 `PMULUDQ` (no native `mullo_epi64`).
    [[gnu::always_inline]] inline __m256i mullo_epi64_avx2(__m256i a, __m256i b) noexcept {
        const __m256i bswap = _mm256_shuffle_epi32(b, 0xB1);
        __m256i prod02 = _mm256_mul_epu32(a, b);
        const __m256i prod13 = _mm256_mul_epu32(a, bswap);
        prod02 = _mm256_slli_epi64(prod02, 32);
        return _mm256_blend_epi32(prod02, prod13, 0xAA);
    }

    [[gnu::always_inline]] inline __m256i simd_mix(__m256i x) noexcept {
        const __m256i m1 = _mm256_set1_epi64x(static_cast<int64_t>(0xbf58476d1ce4e5b9ULL));
        const __m256i m2 = _mm256_set1_epi64x(static_cast<int64_t>(0x94d049bb133111ebULL));
        x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 30));
        x = mullo_epi64_avx2(x, m1);
        x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 27));
        x = mullo_epi64_avx2(x, m2);
        x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 31));
        return x;
    }

    /// Hash `n` keys; AVX2 path processes 4 × 64-bit values per iteration.
    template <typename TKey>
    void hashBatch(const TKey * src, size_t start, int n, uint64_t mask, uint32_t * pids) {
        if constexpr (sizeof(TKey) == 8)
        {
            const __m256i vmask = _mm256_set1_epi64x(static_cast<int64_t>(mask));
            const __m256i pack_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
            const int quads = n / 4;
            for (int g = 0; g < quads; ++g)
            {
                const __m256i k = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + start + static_cast<size_t>(g) * 4));
                const __m256i hashed = _mm256_and_si256(simd_mix(k), vmask);
                const __m128i packed = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(hashed, pack_idx));
                _mm_storeu_si128(reinterpret_cast<__m128i *>(pids + g * 4), packed);
            }
            for (int j = quads * 4; j < n; ++j)
                pids[j] = static_cast<uint32_t>(mix64(toU64ForHash(src[start + j])) & mask);
        }
        else
        {
            for (int j = 0; j < n; ++j)
                pids[j] = static_cast<uint32_t>(mix64(toU64ForHash(src[start + j])) & mask);
        }
    }

    ) // DECLARE_X86_64_V3_SPECIFIC_CODE

DECLARE_X86_64_V4_SPECIFIC_CODE(

    [[gnu::always_inline]] inline __m512i simd_mix(__m512i x) noexcept {
        const __m512i m1 = _mm512_set1_epi64(static_cast<int64_t>(0xbf58476d1ce4e5b9ULL));
        const __m512i m2 = _mm512_set1_epi64(static_cast<int64_t>(0x94d049bb133111ebULL));
        x = _mm512_xor_epi64(x, _mm512_srli_epi64(x, 30));
        x = _mm512_mullo_epi64(x, m1); // VPMULLQ — requires AVX-512DQ
        x = _mm512_xor_epi64(x, _mm512_srli_epi64(x, 27));
        x = _mm512_mullo_epi64(x, m2);
        x = _mm512_xor_epi64(x, _mm512_srli_epi64(x, 31));
        return x;
    }

    /// Hash `n` keys; AVX-512 v4 path processes 8 × 64-bit values per iteration.
    template <typename TKey>
    void hashBatch(const TKey * src, size_t start, int n, uint64_t mask, uint32_t * pids) {
        if constexpr (sizeof(TKey) == 8)
        {
            const __m512i vmask = _mm512_set1_epi64(static_cast<int64_t>(mask));
            const int octets = n / 8;
            for (int g = 0; g < octets; ++g)
            {
                const __m512i k = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + start + static_cast<size_t>(g) * 8));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(pids + g * 8), _mm512_cvtepi64_epi32(_mm512_and_epi64(simd_mix(k), vmask)));
            }
            for (int j = octets * 8; j < n; ++j)
                pids[j] = static_cast<uint32_t>(mix64(toU64ForHash(src[start + j])) & mask);
        }
        else
        {
            for (int j = 0; j < n; ++j)
                pids[j] = static_cast<uint32_t>(mix64(toU64ForHash(src[start + j])) & mask);
        }
    }

    ) // DECLARE_X86_64_V4_SPECIFIC_CODE

#endif // USE_MULTITARGET_CODE


/// Scalar fallback for all architectures.
template <typename TKey>
void hashBatchScalar(const TKey * src, size_t start, int n, uint64_t mask, uint32_t * pids)
{
    for (int j = 0; j < n; ++j)
        pids[j] = static_cast<uint32_t>(mix64(toU64ForHash(src[start + j])) & mask);
}


/// Dispatch: select the best hash implementation at runtime.
template <typename TKey>
[[gnu::always_inline]] inline void hashBatch(const TKey * src, size_t start, int n, uint64_t mask, uint32_t * pids)
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
    {
        TargetSpecific::x86_64_v4::hashBatch(src, start, n, mask, pids);
        return;
    }
    if (isArchSupported(TargetArch::x86_64_v3))
    {
        TargetSpecific::x86_64_v3::hashBatch(src, start, n, mask, pids);
        return;
    }
    if (isArchSupported(TargetArch::x86_64_v2))
    {
        TargetSpecific::x86_64_v2::hashBatch(src, start, n, mask, pids);
        return;
    }
#endif
    hashBatchScalar(src, start, n, mask, pids);
}


// ── RadixPartitionOperator implementation ────────────────────────────────────

template <typename TKey>
RadixPartitionOperator<TKey>::RadixPartitionOperator(
    int P, int K, std::vector<IScatterColumn *> cols, BumpArena & arena, bool use_swwc, size_t init_cap, size_t max_cap)
    : P_(P)
    , K_(K)
    , use_swwc_(use_swwc)
    , batch_(std::max(1024, std::min(kSmartMaxBatch, P * kBatchFactor)))
    , mask_(static_cast<uint64_t>(P) - 1)
    , max_cap_(max_cap)
    , cols_(std::move(cols))
    , arena_(arena)
    , pids_(static_cast<size_t>(batch_))
    , hist_(static_cast<size_t>(P), 0)
    , pos_(static_cast<size_t>(batch_))
    , cnt_(static_cast<size_t>(P), 0)
{
    parts_.assign(static_cast<size_t>(P), {});
    for (auto & ps : parts_)
        ps.next_cap = init_cap;
}


template <typename TKey>
void RadixPartitionOperator<TKey>::process(const DB::Columns & columns)
{
    if (columns.empty() || columns[0]->size() == 0)
        return;
    const size_t n_total = columns[0]->size();
    for (size_t i = 0; i < n_total;)
    {
        const int n = static_cast<int>(std::min(static_cast<size_t>(batch_), n_total - i));
        runBatch(columns, i, n);
        i += static_cast<size_t>(n);
    }
}


template <typename TKey>
void RadixPartitionOperator<TKey>::runBatch(const DB::Columns & columns, size_t start, int n)
{
    uint32_t * pids = pids_.data();
    uint32_t * hist = hist_.data();

    // ── Phase 1: hash key column → partition IDs ──────────────────────────
    const TKey * key_data = assert_cast<const ColumnVector<TKey> &>(*columns[0]).getData().data();
    hashBatch(key_data, start, n, mask_, pids);

    // ── Phase 2: histogram ────────────────────────────────────────────────
    std::memset(hist, 0, static_cast<size_t>(P_) * sizeof(uint32_t));
    for (int j = 0; j < n; ++j)
        hist[pids[j]]++;

    // ── Phase 3: pre-grow + notify columns + pre-commit ───────────────────
    for (int p = 0; p < P_; ++p)
    {
        if (!hist[p])
            continue;
        auto & ps = parts_[static_cast<size_t>(p)];
        if (!ps.cur || ps.cur->filled + hist[p] > ps.cur->capacity)
        {
            // Drain any staged rows into the current block before growing.
            if (use_swwc_ && ps.cur && cnt_[static_cast<size_t>(p)])
            {
                for (auto * c : cols_)
                    c->drain_one(static_cast<size_t>(p), cnt_[static_cast<size_t>(p)]);
                cnt_[static_cast<size_t>(p)] = 0;
            }
            growPart(ps, arena_, K_, sizeof(TKey), max_cap_);
            for (int k = 0; k < K_; ++k)
                cols_[static_cast<size_t>(k)]->on_grow(static_cast<size_t>(p), ps.cur->cols[k]);
        }
        ps.cur->filled += hist[p]; // pre-commit (thread-private, safe)
    }

    if (use_swwc_)
    {
        // ── Phase 4a: compute staging slots (shared across columns) ───────
        uint32_t * pos = pos_.data();
        uint8_t * cnt = cnt_.data();
        for (int j = 0; j < n; ++j)
        {
            const uint32_t p = pids[j];
            const uint32_t slot = cnt[p];
            pos[j] = slot;
            cnt[p] = static_cast<uint8_t>((slot + 1) & 7);
        }
        // ── Phase 4b: SWWC scatter per column ─────────────────────────────
        for (int k = 0; k < K_; ++k)
        {
            const TKey * col_data = assert_cast<const ColumnVector<TKey> &>(*columns[static_cast<size_t>(k)]).getData().data();
            cols_[static_cast<size_t>(k)]->scatter_staged(pids, pos, col_data + start, n);
        }
    }
    else
    {
        // ── Phase 4b: direct scatter per column ───────────────────────────
        for (int k = 0; k < K_; ++k)
        {
            const TKey * col_data = assert_cast<const ColumnVector<TKey> &>(*columns[static_cast<size_t>(k)]).getData().data();
            cols_[static_cast<size_t>(k)]->scatter_direct(pids, col_data + start, n);
        }
    }
}


template <typename TKey>
void RadixPartitionOperator<TKey>::finish()
{
    if (!use_swwc_)
        return;

#if defined(__x86_64__)
    _mm_sfence();
#endif

    for (int p = 0; p < P_; ++p)
    {
        if (!cnt_[static_cast<size_t>(p)])
            continue;
        for (auto * c : cols_)
            c->drain_one(static_cast<size_t>(p), cnt_[static_cast<size_t>(p)]);
        cnt_[static_cast<size_t>(p)] = 0;
    }
}


// ── explicit instantiations ───────────────────────────────────────────────────

template class RadixPartitionOperator<uint8_t>;
template class RadixPartitionOperator<uint16_t>;
template class RadixPartitionOperator<uint32_t>;
template class RadixPartitionOperator<uint64_t>;
template class RadixPartitionOperator<int8_t>;
template class RadixPartitionOperator<int16_t>;
template class RadixPartitionOperator<int32_t>;
template class RadixPartitionOperator<int64_t>;
template class RadixPartitionOperator<float>;
template class RadixPartitionOperator<double>;

} // namespace DB::RadixShuffle
