#include <Common/RadixShuffle/Scatter.h>

#include <Common/Exception.h>
#include <Common/TargetSpecific.h>

#if USE_MULTITARGET_CODE
#    include <immintrin.h> /// __m512i / __m256i and the NT-store intrinsics for the v4/v3 flush helpers
#endif

#include <cstdlib>
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

inline UInt32 route(UInt16 pid, UInt32 shift, UInt32 mask) noexcept
{
    return (static_cast<UInt32>(pid) >> shift) & mask;
}

/// ---- Direct (non-SWWC) batched scatter -------------------------------------------------------
/// Plain typed per-partition write pointers, no staging, no NT stores. This is the only scatter
/// fallback (`ColumnsScatter::scatter` is never used) and the path the join uses whenever NT stores
/// are unavailable or below the SWWC fanout threshold. Every copy is `__builtin_memcpy_inline` of a
/// compile-time size, so it lowers to direct typed stores — there is no `memcpy` call anywhere.

/// Compile-time width (a multiple of 4): the per-element copy is a single inlined typed store.
template <UInt32 width>
ScatterStats scatterColumnDirectTiled(
    const UInt16 * pid, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t partitions, void * const * out,
    ScatterScratch & scratch)
{
    static_assert(width >= 4 && width % 4 == 0, "scatter width must be a multiple of 4");
    void ** cur = scratch.cursors();
    for (size_t p = 0; p < partitions; ++p)
        cur[p] = out[p];

    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(pid[j], shift, mask);
        char * d = static_cast<char *>(cur[p]);
        __builtin_memcpy_inline(d, src + j * width, width); /// constant width -> direct typed store(s)
        cur[p] = d + width;
    }
    return {};
}

/// Runtime width that is a multiple of 4 (the uncommon FixedString widths and large multiples of 64):
/// copied in `4 B` units with inlined `movl`s — a typed-store loop, still no `memcpy` call.
ScatterStats scatterColumnDirectGeneric(
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
        const char * s = src + j * width;
        for (size_t b = 0; b < width; b += 4)
            __builtin_memcpy_inline(d + b, s + b, 4); /// 4 B typed stores; width is a multiple of 4
        cur[p] = d + width;
    }
    return {};
}

/// Grand dispatch on the runtime width: the common fixed widths are width-templated (fully unrolled
/// typed stores); any other multiple of 4 falls to the generic `4 B`-stride copy. No `memcpy` call.
ScatterStats scatterColumnDirectDispatch(
    const UInt16 * pid, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t elem_width, size_t partitions,
    void * const * out, ScatterScratch & scratch)
{
    switch (elem_width)
    {
        case 4: return scatterColumnDirectTiled<4>(pid, shift, mask, n, src, partitions, out, scratch);
        case 8: return scatterColumnDirectTiled<8>(pid, shift, mask, n, src, partitions, out, scratch);
        case 16: return scatterColumnDirectTiled<16>(pid, shift, mask, n, src, partitions, out, scratch);
        case 32: return scatterColumnDirectTiled<32>(pid, shift, mask, n, src, partitions, out, scratch);
        case 64: return scatterColumnDirectTiled<64>(pid, shift, mask, n, src, partitions, out, scratch);
        case 128: return scatterColumnDirectTiled<128>(pid, shift, mask, n, src, partitions, out, scratch);
        default:
            chassert(elem_width % 4 == 0 && "RadixShuffle scatter supports element widths that are multiples of 4");
            return scatterColumnDirectGeneric(pid, shift, mask, n, src, elem_width, partitions, out, scratch);
    }
}

#if USE_MULTITARGET_CODE

constexpr UInt32 LINE_BYTES = 64;

/// Flush one full 64 B SWWC write-combining line to the partition's write cursor with a non-temporal
/// store, then advance the cursor by 64 B. The store bypasses the cache (no read-for-ownership, no
/// pollution), so the streamed outputs never evict the staging / pid / hash-table working set. Both
/// the `staging_line` and `out` are 64 B-aligned (staging is 64 B-aligned with each line at a 64 B
/// offset; the per-partition output base is 64 B-aligned and only advances by whole 64 B lines).
using FlushFn = void (*)(const void * staging_line, void *& out);

/// Stream `lines` 64 B lines straight from a (possibly unaligned) source element to the 64 B-aligned
/// output cursor — for keys whose width is a whole multiple of 64 B (each element is already a whole
/// number of NT lines, so no staging line is needed).
using StreamFn = void (*)(const void * src, void *& out, size_t lines);

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

/// Pick the best NT variant once per scatter (not per row), so the hot loop has no ISA branch. Only
/// ever called when `ntStoresAvailable()` (the SWWC path is gated on it), so a scalar fallback is
/// neither needed nor offered — SWWC exists only when NT stores do.
FlushFn selectFlush() noexcept
{
    if (isArchSupported(TargetArch::x86_64_v4))
        return &TargetSpecific::x86_64_v4::flushLine64;
    return &TargetSpecific::x86_64_v3::flushLine64;
}

StreamFn selectStream() noexcept
{
    if (isArchSupported(TargetArch::x86_64_v4))
        return &TargetSpecific::x86_64_v4::streamLines;
    return &TargetSpecific::x86_64_v3::streamLines;
}

/// Make the NT stores globally visible before the caller reads the outputs (NT stores are weakly
/// ordered). On x86-64 a seq_cst fence lowers to `mfence`, which serialises non-temporal stores.
void streamingFence() noexcept
{
    std::atomic_thread_fence(std::memory_order::seq_cst);
}

/// Total bytes flushed via NT stores so far = how far each cursor advanced past its (unmodified) base.
/// Read after the main loop but before the scalar residual drain (which advances no cursor).
size_t ntBytesFromCursors(void * const * cursors, void * const * base, size_t partitions) noexcept
{
    size_t nt = 0;
    for (size_t p = 0; p < partitions; ++p)
        nt += static_cast<const char *>(cursors[p]) - static_cast<const char *>(base[p]); /// NOLINT
    return nt;
}

/// SWWC scatter for a `width` that divides 64 (`{4,8,16,32}`): one 64 B staging line per partition
/// holding `64 / width` slots; a full line is NT-flushed. `width` is a compile-time constant so the
/// per-row staging write and the residual drain are single typed stores — no `memcpy` call.
template <UInt32 width>
ScatterStats scatterColumnTiledSwwc(
    const UInt16 * pid, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t partitions, void * const * out,
    ScatterScratch & scratch)
{
    static_assert(width >= 4 && width <= 32 && 64 % width == 0, "tiled SWWC width must divide 64 and be in [4, 32]");

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
        __builtin_memcpy_inline(line + f, src + j * width, width); /// constant width -> single typed store
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

    /// Drain the residual (< one line) of each partition: whole `width`-elements, typed-store each.
    for (size_t p = 0; p < partitions; ++p)
    {
        const UInt32 f = fill[p];
        char * d = static_cast<char *>(cur[p]);
        const char * s = staging + static_cast<size_t>(p) * LINE_BYTES;
        for (UInt32 b = 0; b < f; b += width)
            __builtin_memcpy_inline(d + b, s + b, width);
    }

    return ScatterStats{nt};
}

/// SWWC scatter for a width that is a whole multiple of 64 B: each element is `width / 64` NT lines,
/// streamed straight to the output cursor — no staging line and no residual drain.
ScatterStats scatterColumnStreamSwwc(
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

/// SWWC dispatch (NT-only; reached only when `ntStoresAvailable()`). The 64 B staging line tiles
/// without a straddle only for widths that divide 64 (`{4,8,16,32}`); multiples of 64 stream directly.
/// Any other multiple of 4 (`12,20,24,…`) cannot tile a 64 B line cleanly without a much larger
/// `lcm(width,64)` staging buffer, so it uses the direct path instead (which handles any multiple of 4).
ScatterStats scatterColumnSwwcDispatch(
    const UInt16 * pid, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t elem_width, size_t partitions,
    void * const * out, ScatterScratch & scratch)
{
    switch (elem_width)
    {
        case 4: return scatterColumnTiledSwwc<4>(pid, shift, mask, n, src, partitions, out, scratch);
        case 8: return scatterColumnTiledSwwc<8>(pid, shift, mask, n, src, partitions, out, scratch);
        case 16: return scatterColumnTiledSwwc<16>(pid, shift, mask, n, src, partitions, out, scratch);
        case 32: return scatterColumnTiledSwwc<32>(pid, shift, mask, n, src, partitions, out, scratch);
        default:
            if (elem_width % 64 == 0)
                return scatterColumnStreamSwwc(pid, shift, mask, n, src, elem_width, partitions, out, scratch);
            return scatterColumnDirectDispatch(pid, shift, mask, n, src, elem_width, partitions, out, scratch);
    }
}

#endif

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
    /// threads): SWWC + NT beats the direct batched scatter only at high fanout (`P >= 2048`, ~+10%)
    /// and only when NT stores are actually emitted; at lower fanout, or in a build without NT, the
    /// direct scatter wins. So engage SWWC iff NT is available and `partitions >= 2048`.
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
    [[maybe_unused]] bool use_swwc)
{
    chassert(scratch.maxPartitions() >= partitions);
    chassert(mask == static_cast<UInt32>(partitions) - 1); /// window must select exactly [0, partitions)
    chassert(elem_width >= 4 && elem_width % 4 == 0 && "RadixShuffle scatter supports element widths that are multiples of 4");

    const auto * bytes = static_cast<const char *>(src);

#if USE_MULTITARGET_CODE
    /// SWWC exists only when NT stores are available (otherwise it would be a scalar write-combine that
    /// is strictly slower than the direct scatter). When NT is unavailable the request runs direct.
    if (use_swwc && ntStoresAvailable())
        return scatterColumnSwwcDispatch(pid, shift, mask, n, bytes, elem_width, partitions, out, scratch);
#endif

    return scatterColumnDirectDispatch(pid, shift, mask, n, bytes, elem_width, partitions, out, scratch);
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
