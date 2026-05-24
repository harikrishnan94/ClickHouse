#include <Common/RadixShuffle/NumericScatterColumn.h>

#include <Common/TargetSpecific.h>

#if defined(__x86_64__)
#    include <immintrin.h>
#endif

#include <cstdlib>
#include <cstring>
#include <new>


namespace DB::RadixShuffle
{

// ── AVX-512 v4 flush helper ───────────────────────────────────────────────────
// Defined inside DB::RadixShuffle so that, from within this namespace, it is
// reachable as TargetSpecific::x86_64_v4::flushStagedNT<T>(staging_p, out_p).
//
// The function is compiled with "arch=x86-64-v4" target attribute (via the
// DECLARE macro's push/pop pragma), making _mm512_stream_si512 and
// _mm512_load_si512 available regardless of the TU-level -march flag.

#if USE_MULTITARGET_CODE

DECLARE_X86_64_V4_SPECIFIC_CODE(

template <typename T>
inline void flushStagedNT(const T * staging_p, T *& out_p) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
{
    if constexpr (sizeof(T) == 8)
    {
        _mm512_stream_si512(
            reinterpret_cast<__m512i *>(out_p),
            _mm512_load_si512(reinterpret_cast<const __m512i *>(staging_p)));
    }
    else
    {
        for (int i = 0; i < 8; ++i)
            out_p[i] = staging_p[i];
    }
    out_p += 8;
}

) // DECLARE_X86_64_V4_SPECIFIC_CODE

#endif // USE_MULTITARGET_CODE


// ── scalar flush (all architectures) ─────────────────────────────────────────

template <typename T>
[[gnu::always_inline]] inline void flushStagedScalar(const T * staging_p, T *& out_p)
{
    for (int i = 0; i < 8; ++i)
        out_p[i] = staging_p[i];
    out_p += 8;
}


// ── NumericScatterColumn implementation ──────────────────────────────────────

template <typename T>
NumericScatterColumn<T>::NumericScatterColumn(size_t P) : P_(P)
{
    if (posix_memalign(reinterpret_cast<void **>(&staging_), 64, P * 8 * sizeof(T)) != 0)
        throw std::bad_alloc{};
    std::memset(staging_, 0, P * 8 * sizeof(T));
    out_ = new T *[P]();
}


template <typename T>
NumericScatterColumn<T>::~NumericScatterColumn()
{
    std::free(staging_);
    delete[] out_;
}


template <typename T>
void NumericScatterColumn<T>::on_grow(size_t p, void * col_base)
{
    out_[p] = static_cast<T *>(col_base);
}


template <typename T>
void NumericScatterColumn<T>::drain_one(size_t p, uint32_t cnt)
{
    const T * s = staging_ + p * 8;
    T * dst = out_[p];
    for (uint32_t i = 0; i < cnt; ++i)
        dst[i] = s[i];
    out_[p] = dst + cnt;
}


template <typename T>
void NumericScatterColumn<T>::scatter_direct(const uint32_t * pids, const void * src, int n)
{
    const T * s = static_cast<const T *>(src);
    T ** ptrs = out_;
    for (int j = 0; j < n; ++j)
        *ptrs[pids[j]]++ = s[j];
}


template <typename T>
void NumericScatterColumn<T>::scatter_staged(
    const uint32_t * pids, const uint32_t * positions, const void * src, int n)
{
    const T * s = static_cast<const T *>(src);
    for (int j = 0; j < n; ++j)
    {
        const uint32_t p = pids[j];
        const uint32_t slot = positions[j];
        staging_[static_cast<size_t>(p) * 8 + slot] = s[j];
        if (slot == 7)
        {
#if USE_MULTITARGET_CODE
            if (isArchSupported(TargetArch::x86_64_v4))
                TargetSpecific::x86_64_v4::flushStagedNT(staging_ + static_cast<size_t>(p) * 8, out_[p]);
            else
                flushStagedScalar(staging_ + static_cast<size_t>(p) * 8, out_[p]);
#else
            flushStagedScalar(staging_ + static_cast<size_t>(p) * 8, out_[p]);
#endif
        }
    }
}


// ── explicit instantiations ───────────────────────────────────────────────────

template class NumericScatterColumn<uint8_t>;
template class NumericScatterColumn<uint16_t>;
template class NumericScatterColumn<uint32_t>;
template class NumericScatterColumn<uint64_t>;
template class NumericScatterColumn<int8_t>;
template class NumericScatterColumn<int16_t>;
template class NumericScatterColumn<int32_t>;
template class NumericScatterColumn<int64_t>;
template class NumericScatterColumn<float>;
template class NumericScatterColumn<double>;

} // namespace DB::RadixShuffle
