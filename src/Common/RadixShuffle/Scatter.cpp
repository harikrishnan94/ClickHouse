#include <Common/RadixShuffle/Scatter.h>

#include <Common/Exception.h>
#include <Common/TargetSpecific.h>

#if USE_MULTITARGET_CODE
#    include <immintrin.h> /// __m512i / __m256i and the NT-store intrinsics for the v4/v3 flush helpers
#endif

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace DB
{
namespace ErrorCodes
{
extern const int CANNOT_ALLOCATE_MEMORY;
}
}

namespace DB::RadixShuffle
{

namespace
{

constexpr UInt32 LINE_BYTES = 64;

/// Flush one full 64 B SWWC write-combining line to the partition's write cursor with a non-temporal
/// store, then advance the cursor by 64 B. The store bypasses the cache (no read-for-ownership, no
/// pollution), so the streamed outputs never evict the staging / pid / hash-table working set. The
/// `staging_line` is 64 B-aligned (scratch staging is 64 B-aligned, each line at a 64 B offset) and
/// `out` is 64 B-aligned (the per-partition output base is 64 B-aligned and only advances by 64 B).
using FlushFn = void (*)(const void * staging_line, void *& out);

/// Stream `lines` 64 B lines straight from a (possibly unaligned) source element to the 64 B-aligned
/// output cursor — used for keys whose width is a whole multiple of 64 B (no staging line needed,
/// each element is already a whole number of NT lines).
using StreamFn = void (*)(const void * src, void *& out, size_t lines);

#if USE_MULTITARGET_CODE

DECLARE_X86_64_V4_SPECIFIC_CODE(
    inline void flushLine64(const void * staging_line, void *& out) /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        _mm512_stream_si512(
            reinterpret_cast<__m512i *>(out), _mm512_load_si512(reinterpret_cast<const __m512i *>(staging_line)));
        out = reinterpret_cast<char *>(out) + 64;
    }

    inline void streamLines(const void * src, void *& out, size_t lines) /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        const auto * s = reinterpret_cast<const char *>(src);
        auto * d = reinterpret_cast<char *>(out);
        for (size_t k = 0; k < lines; ++k)
            _mm512_stream_si512(
                reinterpret_cast<__m512i *>(d + k * 64), _mm512_loadu_si512(reinterpret_cast<const void *>(s + k * 64)));
        out = d + lines * 64;
    }
) /// DECLARE_X86_64_V4_SPECIFIC_CODE

DECLARE_X86_64_V3_SPECIFIC_CODE(
    inline void flushLine64(const void * staging_line, void *& out) /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        const auto * s = reinterpret_cast<const __m256i *>(staging_line);
        auto * d = reinterpret_cast<__m256i *>(out);
        _mm256_stream_si256(d, _mm256_load_si256(s));
        _mm256_stream_si256(d + 1, _mm256_load_si256(s + 1));
        out = reinterpret_cast<char *>(out) + 64;
    }

    inline void streamLines(const void * src, void *& out, size_t lines) /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        const auto * s = reinterpret_cast<const char *>(src);
        auto * d = reinterpret_cast<char *>(out);
        for (size_t k = 0; k < lines; ++k)
        {
            const auto * sp = reinterpret_cast<const __m256i *>(s + k * 64);
            auto * dp = reinterpret_cast<__m256i *>(d + k * 64);
            _mm256_stream_si256(dp, _mm256_loadu_si256(sp));
            _mm256_stream_si256(dp + 1, _mm256_loadu_si256(sp + 1));
        }
        out = d + lines * 64;
    }
) /// DECLARE_X86_64_V3_SPECIFIC_CODE

#endif

/// Scalar fallback (no AVX-512 / AVX2 or multitarget disabled): plain copies. Correct, just without
/// the cache-bypassing benefit.
void flushLine64Scalar(const void * staging_line, void *& out)
{
    __builtin_memcpy_inline(out, staging_line, 64); /// constant size -> inlined vector copy, never a memcpy call
    out = reinterpret_cast<char *>(out) + 64; /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

void streamLinesScalar(const void * src, void *& out, size_t lines)
{
    std::memcpy(out, src, lines * 64);
    out = reinterpret_cast<char *>(out) + lines * 64; /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

/// Pick the best NT variants once per scatter (not per row), so the hot loop has no ISA branch.
FlushFn selectFlush() noexcept
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
        return &TargetSpecific::x86_64_v4::flushLine64;
    if (isArchSupported(TargetArch::x86_64_v3))
        return &TargetSpecific::x86_64_v3::flushLine64;
#endif
    return &flushLine64Scalar;
}

StreamFn selectStream() noexcept
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
        return &TargetSpecific::x86_64_v4::streamLines;
    if (isArchSupported(TargetArch::x86_64_v3))
        return &TargetSpecific::x86_64_v3::streamLines;
#endif
    return &streamLinesScalar;
}

/// Make the NT stores globally visible before the caller reads the outputs (NT stores are weakly
/// ordered). On x86-64 a seq_cst fence lowers to `mfence`, which serialises non-temporal stores.
void streamingFence() noexcept
{
    std::atomic_thread_fence(std::memory_order::seq_cst);
}

/// Total bytes flushed via NT stores so far = how far each cursor advanced past its (unmodified) base.
/// Must be read after the main loop but before the scalar residual drain (which advances no cursor).
size_t ntBytesFromCursors(void * const * cursors, void * const * base, size_t partitions) noexcept
{
    size_t nt = 0;
    for (size_t p = 0; p < partitions; ++p)
        nt += static_cast<const char *>(cursors[p]) - static_cast<const char *>(base[p]); /// NOLINT
    return nt;
}

inline UInt32 route(UInt16 pid, UInt32 shift, UInt32 mask) noexcept
{
    return (static_cast<UInt32>(pid) >> shift) & mask;
}

/// SWWC scatter for a `width` that divides 64 (`{1,2,4,8,16,32}`): one 64 B staging line per partition
/// holding `64 / width` slots; a full line is NT-flushed. `width` is a compile-time constant so the
/// per-row staging write lowers to a single typed store (the hot path the §9.2 gate is measured on).
template <UInt32 width>
ScatterStats scatterColumnTiled(
    const UInt16 * pid, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t partitions, void * const * out,
    ScatterScratch & scratch)
{
    static_assert(width >= 1 && width <= 32 && 64 % width == 0, "tiled SWWC width must divide 64 and be <= 32 (64 streams directly)");

    char * staging = scratch.staging();
    void ** cur = scratch.cursors();
    UInt32 * fill = scratch.fill();
    for (size_t p = 0; p < partitions; ++p)
    {
        chassert(reinterpret_cast<uintptr_t>(out[p]) % 64 == 0); /// NOLINT -- NT stores need 64 B-aligned dst
        cur[p] = out[p];
        fill[p] = 0;
    }

    const FlushFn flush = selectFlush();
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(pid[j], shift, mask);
        char * line = staging + static_cast<size_t>(p) * LINE_BYTES;
        UInt32 f = fill[p];
        __builtin_memcpy_inline(line + f, src + j * width, width); /// constant width -> single typed store, never a memcpy call
        f += width;
        if (f == LINE_BYTES)
        {
            flush(line, cur[p]);
            f = 0;
        }
        fill[p] = f;
    }

    streamingFence();
    const size_t nt = ntBytesFromCursors(cur, out, partitions);

    /// Drain the residual (< one line) of each partition with ordinary stores.
    for (size_t p = 0; p < partitions; ++p)
        if (const UInt32 f = fill[p])
            std::memcpy(cur[p], staging + p * LINE_BYTES, f);

    return ScatterStats{nt};
}

/// SWWC scatter for a width `1..64` that does NOT divide 64: a per-partition 64 B write-combining line
/// is filled as a byte stream; whenever it fills, it is NT-flushed and the element's remainder carries
/// over. Because `W <= 64`, each element straddles at most one line boundary.
ScatterStats scatterColumnBytes(
    const UInt16 * pid, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t width, size_t partitions,
    void * const * out, ScatterScratch & scratch)
{
    chassert(width <= 64);
    const UInt32 w = static_cast<UInt32>(width);

    char * staging = scratch.staging();
    void ** cur = scratch.cursors();
    UInt32 * fill = scratch.fill();
    for (size_t p = 0; p < partitions; ++p)
    {
        chassert(reinterpret_cast<uintptr_t>(out[p]) % 64 == 0); /// NOLINT
        cur[p] = out[p];
        fill[p] = 0;
    }

    const FlushFn flush = selectFlush();
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(pid[j], shift, mask);
        const char * e = src + j * width;
        char * line = staging + static_cast<size_t>(p) * LINE_BYTES;
        UInt32 f = fill[p];
        if (f + w <= LINE_BYTES)
        {
            std::memcpy(line + f, e, w);
            f += w;
            if (f == LINE_BYTES)
            {
                flush(line, cur[p]);
                f = 0;
            }
        }
        else
        {
            const UInt32 first = LINE_BYTES - f;
            std::memcpy(line + f, e, first);
            flush(line, cur[p]);
            std::memcpy(line, e + first, w - first);
            f = w - first;
        }
        fill[p] = f;
    }

    streamingFence();
    const size_t nt = ntBytesFromCursors(cur, out, partitions);

    for (size_t p = 0; p < partitions; ++p)
        if (const UInt32 f = fill[p])
            std::memcpy(cur[p], staging + p * LINE_BYTES, f);

    return ScatterStats{nt};
}

/// SWWC scatter for a width that is a whole multiple of 64 B: each element is `width / 64` NT lines,
/// streamed straight to the output cursor — no staging line and no residual drain.
ScatterStats scatterColumnStream(
    const UInt16 * pid, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t width, size_t partitions,
    void * const * out, ScatterScratch & scratch)
{
    chassert(width >= 64 && width % 64 == 0);
    const size_t lines = width / 64;

    void ** cur = scratch.cursors();
    for (size_t p = 0; p < partitions; ++p)
    {
        chassert(reinterpret_cast<uintptr_t>(out[p]) % 64 == 0); /// NOLINT
        cur[p] = out[p];
    }

    const StreamFn stream = selectStream();
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(pid[j], shift, mask);
        stream(src + j * width, cur[p], lines);
    }

    streamingFence();
    return ScatterStats{ntBytesFromCursors(cur, out, partitions)};
}

/// Non-SWWC batched fallback for a `width` that divides 64 (compile-time): plain typed per-partition
/// write pointers, no staging, no NT stores. This is the only scatter fallback (`ColumnsScatter` is
/// never used) and the `nt/bt` baseline the calibration compares SWWC against.
template <UInt32 width>
ScatterStats scatterColumnDirectTiled(
    const UInt16 * pid, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t partitions, void * const * out,
    ScatterScratch & scratch)
{
    void ** cur = scratch.cursors();
    for (size_t p = 0; p < partitions; ++p)
        cur[p] = out[p];

    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(pid[j], shift, mask);
        char * d = static_cast<char *>(cur[p]);
        __builtin_memcpy_inline(d, src + j * width, width); /// constant width -> single typed store
        cur[p] = d + width;
    }
    return {};
}

/// Non-SWWC batched fallback for an arbitrary width.
ScatterStats scatterColumnDirectRuntime(
    const UInt16 * pid, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t width, size_t partitions,
    void * const * out, ScatterScratch & scratch)
{
    void ** cur = scratch.cursors();
    for (size_t p = 0; p < partitions; ++p)
        cur[p] = out[p];

    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(pid[j], shift, mask);
        char * d = static_cast<char *>(cur[p]);
        std::memcpy(d, src + j * width, width);
        cur[p] = d + width;
    }
    return {};
}

}

bool ntStoresAvailable() noexcept
{
#if USE_MULTITARGET_CODE
    return isArchSupported(TargetArch::x86_64_v4) || isArchSupported(TargetArch::x86_64_v3);
#else
    return false;
#endif
}

bool shouldUseSwwc([[maybe_unused]] int num_columns, int partitions) noexcept
{
    /// Recalibrated in P2 under the realistic alloc+fault model (`bench_radix_sweep_native`, 16
    /// threads, fresh output + first-touch faults per rep). SWWC + NT beats the direct batched
    /// scatter only at high fanout (`P >= 2048`: ~+10% across key widths) and only when NT stores are
    /// actually emitted; at `P <= 1024` the per-partition outputs stay cache-resident and the direct
    /// scatter wins, and with NT unavailable (non-multitarget build) the staging copy makes SWWC
    /// strictly slower than direct. So engage SWWC iff NT is available and `partitions >= 2048`.
    return ntStoresAvailable() && partitions >= 2048;
}

ScatterScratch::ScatterScratch(size_t max_partitions)
    : capacity(max_partitions)
    , cursor_ptrs(max_partitions, nullptr)
    , line_fill(max_partitions, 0)
{
    /// One 64 B write-combining line per partition, 64 B-aligned for the NT stores.
    const size_t bytes = capacity * LINE_BYTES;
    if (posix_memalign(reinterpret_cast<void **>(&staging_buf), 64, bytes) != 0 || staging_buf == nullptr)
        throw Exception(ErrorCodes::CANNOT_ALLOCATE_MEMORY, "RadixShuffle::ScatterScratch failed to allocate {} bytes", bytes);
}

void ScatterScratch::freeStaging() noexcept
{
    std::free(staging_buf); /// NOLINT(cppcoreguidelines-no-malloc) -- paired with posix_memalign
    staging_buf = nullptr;
}

ScatterScratch::~ScatterScratch()
{
    freeStaging();
}

ScatterScratch::ScatterScratch(ScatterScratch && other) noexcept
    : capacity(other.capacity)
    , staging_buf(other.staging_buf)
    , cursor_ptrs(std::move(other.cursor_ptrs))
    , line_fill(std::move(other.line_fill))
{
    other.staging_buf = nullptr;
    other.capacity = 0;
}

ScatterScratch & ScatterScratch::operator=(ScatterScratch && other) noexcept
{
    if (this != &other)
    {
        freeStaging();
        capacity = other.capacity;
        staging_buf = other.staging_buf;
        cursor_ptrs = std::move(other.cursor_ptrs);
        line_fill = std::move(other.line_fill);
        other.staging_buf = nullptr;
        other.capacity = 0;
    }
    return *this;
}

ScatterStats scatterColumn(
    const UInt16 * pid,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const void * src,
    size_t elem_width,
    size_t partitions,
    void * const * out,
    ScatterScratch & scratch,
    bool use_swwc)
{
    chassert(scratch.maxPartitions() >= partitions);
    chassert(mask == static_cast<UInt32>(partitions) - 1); /// window must select exactly [0, partitions)
    chassert(elem_width >= 1);

    const auto * bytes = static_cast<const char *>(src);

    if (!use_swwc)
    {
        switch (elem_width)
        {
            case 1: return scatterColumnDirectTiled<1>(pid, shift, mask, n, bytes, partitions, out, scratch);
            case 2: return scatterColumnDirectTiled<2>(pid, shift, mask, n, bytes, partitions, out, scratch);
            case 4: return scatterColumnDirectTiled<4>(pid, shift, mask, n, bytes, partitions, out, scratch);
            case 8: return scatterColumnDirectTiled<8>(pid, shift, mask, n, bytes, partitions, out, scratch);
            case 16: return scatterColumnDirectTiled<16>(pid, shift, mask, n, bytes, partitions, out, scratch);
            case 32: return scatterColumnDirectTiled<32>(pid, shift, mask, n, bytes, partitions, out, scratch);
            default: return scatterColumnDirectRuntime(pid, shift, mask, n, bytes, elem_width, partitions, out, scratch);
        }
    }

    /// Widths that are a whole multiple of 64 B stream directly (no staging line).
    if (elem_width % 64 == 0)
        return scatterColumnStream(pid, shift, mask, n, bytes, elem_width, partitions, out, scratch);

    /// Width <= 64: tiled (divides 64) or byte-stream write-combining (does not).
    switch (elem_width)
    {
        case 1: return scatterColumnTiled<1>(pid, shift, mask, n, bytes, partitions, out, scratch);
        case 2: return scatterColumnTiled<2>(pid, shift, mask, n, bytes, partitions, out, scratch);
        case 4: return scatterColumnTiled<4>(pid, shift, mask, n, bytes, partitions, out, scratch);
        case 8: return scatterColumnTiled<8>(pid, shift, mask, n, bytes, partitions, out, scratch);
        case 16: return scatterColumnTiled<16>(pid, shift, mask, n, bytes, partitions, out, scratch);
        case 32: return scatterColumnTiled<32>(pid, shift, mask, n, bytes, partitions, out, scratch);
        default:
            chassert(elem_width <= 64 && "SWWC supports widths 1..64 or any multiple of 64");
            return scatterColumnBytes(pid, shift, mask, n, bytes, elem_width, partitions, out, scratch);
    }
}

ScatterStats scatterKeyRefTwoColumn(
    const UInt16 * pid,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const void * keys,
    size_t key_width,
    const BuildRef * refs,
    size_t partitions,
    void * const * key_out,
    BuildRef * const * ref_out,
    ScatterScratch & scratch,
    bool use_swwc)
{
    /// Column-major: scatter the key column and the BuildRef column separately, both routed by `pid`
    /// (reusing the same scratch sequentially; each call re-initialises its cursors / fills).
    const ScatterStats key_stats = scatterColumn(pid, shift, mask, n, keys, key_width, partitions, key_out, scratch, use_swwc);
    const ScatterStats ref_stats = scatterColumn(
        pid, shift, mask, n, refs, sizeof(BuildRef), partitions,
        reinterpret_cast<void * const *>(ref_out), /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        scratch, use_swwc);
    return ScatterStats{key_stats.nt_store_bytes + ref_stats.nt_store_bytes};
}

}
