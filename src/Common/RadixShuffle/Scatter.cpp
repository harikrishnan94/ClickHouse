#include <Common/RadixShuffle/Scatter.h>

#include <Common/Exception.h>
#include <Common/TargetSpecific.h>


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

/// Route row `j` to its partition for this pass: `part = (hash >> shift) & mask`.
inline UInt32 route(UInt32 hash, UInt32 shift, UInt32 mask) noexcept
{
    return (hash >> shift) & mask;
}

/// ---- Direct (non-SWWC) batched scatter -------------------------------------------------------
/// Plain typed per-partition write pointers, no staging, no NT stores. This is the only scatter
/// fallback (`ColumnsScatter::scatter` is never used) and the path the join uses whenever NT stores
/// are unavailable or below the SWWC fanout threshold. Every copy is `__builtin_memcpy_inline` of a
/// compile-time size, so it lowers to direct typed stores — there is no `memcpy` call anywhere.

/// All direct kernels operate on a caller-supplied live cursor array `cur` (one write pointer per
/// partition): they append rows and advance `cur[p]` in place — they never reset to a base. The
/// monolithic `scatterColumn` seeds `cur` from the per-partition bases (via the scratch) once per
/// call; the incremental `scatterColumnInto` passes the caller's persistent cursors straight through,
/// so successive chunks keep appending into the same single per-partition allocation.

/// Compile-time width (a multiple of 4): the per-element copy is a single inlined typed store.
template <UInt32 width>
void scatterColumnDirectTiled(const UInt32 * hash, UInt32 shift, UInt32 mask, size_t n, const char * src, void ** cur)
{
    static_assert(width >= 4 && width % 4 == 0, "scatter width must be a multiple of 4");
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(hash[j], shift, mask);
        char * d = static_cast<char *>(cur[p]);
        __builtin_memcpy_inline(d, src + j * width, width); /// constant width -> direct typed store(s)
        cur[p] = d + width;
    }
}

/// Runtime width that is a multiple of 4 (the uncommon FixedString widths and large multiples of 64):
/// copied in `4 B` units with inlined `movl`s — a typed-store loop, still no `memcpy` call.
void scatterColumnDirectGeneric(const UInt32 * hash, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t width, void ** cur)
{
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(hash[j], shift, mask);
        char * d = static_cast<char *>(cur[p]);
        const char * s = src + j * width;
        for (size_t b = 0; b < width; b += 4)
            __builtin_memcpy_inline(d + b, s + b, 4); /// 4 B typed stores; width is a multiple of 4
        cur[p] = d + width;
    }
}

/// Grand dispatch on the runtime width: the common fixed widths are width-templated (fully unrolled
/// typed stores); any other multiple of 4 falls to the generic `4 B`-stride copy. No `memcpy` call.
void scatterColumnDirectDispatch(const UInt32 * hash, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t elem_width, void ** cur)
{
    switch (elem_width)
    {
        case 4: scatterColumnDirectTiled<4>(hash, shift, mask, n, src, cur); return;
        case 8: scatterColumnDirectTiled<8>(hash, shift, mask, n, src, cur); return;
        case 16: scatterColumnDirectTiled<16>(hash, shift, mask, n, src, cur); return;
        case 32: scatterColumnDirectTiled<32>(hash, shift, mask, n, src, cur); return;
        case 64: scatterColumnDirectTiled<64>(hash, shift, mask, n, src, cur); return; /// element width 64 B, not LINE_BYTES
        case 128: scatterColumnDirectTiled<128>(hash, shift, mask, n, src, cur); return;
        default:
            chassert(elem_width % 4 == 0 && "RadixShuffle scatter supports element widths that are multiples of 4");
            scatterColumnDirectGeneric(hash, shift, mask, n, src, elem_width, cur);
            return;
    }
}

/// Seed the scratch cursors from the per-partition bases, then run the direct dispatch. Used by the
/// monolithic entry points (and the SWWC fallback) so `out[]` is left untouched.
ScatterStats scatterColumnDirectFromBases(
    const UInt32 * hash,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const char * src,
    size_t elem_width,
    size_t partitions,
    void * const * out,
    ScatterScratch & scratch)
{
    void ** cur = scratch.cursors();
    for (size_t p = 0; p < partitions; ++p)
        cur[p] = out[p];
    scatterColumnDirectDispatch(hash, shift, mask, n, src, elem_width, cur);
    return {};
}

#if USE_MULTITARGET_CODE

/// `LINE_BYTES`-wide vector for `__builtin_nontemporal_store` (one `vmovntps` on AVX-512, two on AVX2).
using VecNTLine = char __attribute__((vector_size(LINE_BYTES)));

/// Flush one full SWWC write-combining line (`LINE_BYTES`) to the partition's write cursor with a
/// non-temporal store, then advance the cursor by `LINE_BYTES`. The store bypasses the cache (no
/// read-for-ownership, no pollution), so the streamed outputs never evict the staging / hash /
/// hash-table working set. Both `staging_line` and `out` are `LINE_BYTES`-aligned (staging is
/// `LINE_BYTES`-aligned with each line at a `LINE_BYTES` offset; the per-partition output base is
/// `LINE_BYTES`-aligned and only advances by whole lines).
using FlushFn = void (*)(const void * staging_line, void *& out);

/// Stream `lines` `LINE_BYTES` lines straight from a (possibly unaligned) source element to the
/// `LINE_BYTES`-aligned output cursor — for keys whose width is a whole multiple of `LINE_BYTES`
/// (each element is already a whole number of NT lines, so no staging line is needed).
using StreamFn = void (*)(const void * src, void *& out, size_t lines);

DECLARE_MULTITARGET_CODE(
    [[maybe_unused]] inline void flushLine64(const void * staging_line, void *& out) /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        const auto * s = reinterpret_cast<const VecNTLine *>(staging_line);
        auto * d = reinterpret_cast<VecNTLine *>(out);
        __builtin_nontemporal_store(*s, d);
        out = reinterpret_cast<char *>(out) + LINE_BYTES;
    }

    [[maybe_unused]] inline void streamLines(
        const void * src, void *& out, size_t lines) /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        const auto * s = reinterpret_cast<const char *>(src);
        auto * d = reinterpret_cast<char *>(out);
        for (size_t k = 0; k < lines; ++k)
        {
            VecNTLine v;
            __builtin_memcpy_inline(&v, s + k * LINE_BYTES, LINE_BYTES);
            __builtin_nontemporal_store(v, reinterpret_cast<VecNTLine *>(d + k * LINE_BYTES));
        }
        out = d + lines * LINE_BYTES;
    }) /// DECLARE_MULTITARGET_CODE

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

/// SWWC scatter for a `width` that divides `LINE_BYTES` (`{4,8,16,32}`): one staging line per
/// partition holding `LINE_BYTES / width` slots; a full line is NT-flushed. `width` is a compile-time
/// constant so the per-row staging write and the residual drain are single typed stores — no
/// `memcpy` call.
template <UInt32 width>
ScatterStats scatterColumnTiledSwwc(
    const UInt32 * hash,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const char * src,
    size_t partitions,
    void * const * out,
    ScatterScratch & scratch)
{
    static_assert(
        width >= 4 && width <= 32 && LINE_BYTES % width == 0, "tiled SWWC width must divide LINE_BYTES and be in [4, 32]");

    char * staging = scratch.staging();
    void ** cur = scratch.cursors();
    UInt32 * fill = scratch.fill();
    for (size_t p = 0; p < partitions; ++p)
    {
        chassert(reinterpret_cast<uintptr_t>(out[p]) % LINE_BYTES == 0); /// NOLINT -- NT stores need LINE_BYTES-aligned dst
        cur[p] = out[p];
        fill[p] = 0;
    }

    const FlushFn flush = selectFlush();
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(hash[j], shift, mask);
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
        const char * s = staging + p * LINE_BYTES;
        for (UInt32 b = 0; b < f; b += width)
            __builtin_memcpy_inline(d + b, s + b, width);
    }

    return ScatterStats{nt};
}

/// SWWC scatter for a width that is a whole multiple of `LINE_BYTES`: each element is
/// `width / LINE_BYTES` NT lines, streamed straight to the output cursor — no staging line and no
/// residual drain.
ScatterStats scatterColumnStreamSwwc(
    const UInt32 * hash,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const char * src,
    size_t width,
    size_t partitions,
    void * const * out,
    ScatterScratch & scratch)
{
    chassert(width >= LINE_BYTES && width % LINE_BYTES == 0);
    const size_t lines = width / LINE_BYTES;

    void ** cur = scratch.cursors();
    for (size_t p = 0; p < partitions; ++p)
    {
        chassert(reinterpret_cast<uintptr_t>(out[p]) % LINE_BYTES == 0); /// NOLINT
        cur[p] = out[p];
    }

    const StreamFn stream = selectStream();
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(hash[j], shift, mask);
        stream(src + j * width, cur[p], lines);
    }

    streamingFence();
    return ScatterStats{ntBytesFromCursors(cur, out, partitions)};
}

/// SWWC dispatch (NT-only; reached only when `ntStoresAvailable()`). The staging line tiles without a
/// straddle only for widths that divide `LINE_BYTES` (`{4,8,16,32}`); multiples of `LINE_BYTES`
/// stream directly. Any other multiple of 4 (`12,20,24,…`) cannot tile a line cleanly without a much
/// larger `lcm(width, LINE_BYTES)` staging buffer, so it uses the direct path instead (which handles
/// any multiple of 4).
ScatterStats scatterColumnSwwcDispatch(
    const UInt32 * hash,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const char * src,
    size_t elem_width,
    size_t partitions,
    void * const * out,
    ScatterScratch & scratch)
{
    switch (elem_width)
    {
        case 4: return scatterColumnTiledSwwc<4>(hash, shift, mask, n, src, partitions, out, scratch);
        case 8: return scatterColumnTiledSwwc<8>(hash, shift, mask, n, src, partitions, out, scratch);
        case 16: return scatterColumnTiledSwwc<16>(hash, shift, mask, n, src, partitions, out, scratch);
        case 32: return scatterColumnTiledSwwc<32>(hash, shift, mask, n, src, partitions, out, scratch);
        default:
            if (elem_width % LINE_BYTES == 0)
                return scatterColumnStreamSwwc(hash, shift, mask, n, src, elem_width, partitions, out, scratch);
            return scatterColumnDirectFromBases(hash, shift, mask, n, src, elem_width, partitions, out, scratch);
    }
}

/// ---- Incremental SWWC (persistent cursors + staging across chunked calls) --------------------
/// The SWWC/NT analogue of the direct `scatterColumnInto`: the per-partition write cursor, staging
/// line and line fill live in a caller-owned `ScatterScratch` so a large input can be scattered in
/// successive small chunks into the SAME single per-partition allocation. The caller seeds
/// `scratch.cursors()[p]` to this `(thread, partition)` start (a fresh scratch already zeroes the
/// fills) before the first chunk, and calls `scatterColumnDrainSwwc` once after the last chunk.
///
/// HEAD-PEELING: a worker's start `base + w_off * width` is generally NOT `LINE_BYTES`-aligned (only
/// the partition base is), but NT stores need a `LINE_BYTES`-aligned destination. So while the cursor
/// is unaligned the rows are written DIRECTLY (head peel); once it reaches a `LINE_BYTES` boundary the
/// rows are staged and full lines are NT-flushed. Output stays contiguous (no gaps). Because `width`
/// divides `LINE_BYTES` and the start is a multiple of `width`, the peel lands exactly on a boundary.

/// Tiled incremental SWWC for a `width` that divides `LINE_BYTES` (`{4,8,16,32}`); `width` is a
/// compile-time constant so the per-row staging / head-peel copies are single typed stores.
template <UInt32 width>
void scatterColumnTiledIntoSwwc(const UInt32 * hash, UInt32 shift, UInt32 mask, size_t n, const char * src, ScatterScratch & scratch)
{
    static_assert(
        width >= 4 && width <= 32 && LINE_BYTES % width == 0, "tiled SWWC width must divide LINE_BYTES and be in [4, 32]");

    char * staging = scratch.staging();
    void ** cur = scratch.cursors();
    UInt32 * fill = scratch.fill();
    const FlushFn flush = selectFlush();
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(hash[j], shift, mask);
        char * c = static_cast<char *>(cur[p]);
        if ((reinterpret_cast<uintptr_t>(c) & (LINE_BYTES - 1)) != 0) /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        {
            /// Head peel: cursor not yet LINE_BYTES-aligned — write the row directly and advance.
            __builtin_memcpy_inline(c, src + j * width, width); /// constant width -> single typed store
            cur[p] = c + width;
        }
        else
        {
            /// Aligned: stage into the per-partition write-combining line; NT-flush a full line.
            char * line = staging + static_cast<size_t>(p) * LINE_BYTES;
            UInt32 f = fill[p];
            __builtin_memcpy_inline(line + f, src + j * width, width); /// constant width -> single typed store
            f += width;
            if (f == LINE_BYTES)
            {
                flush(line, cur[p]); /// cur[p] is LINE_BYTES-aligned here (only advances by whole lines)
                f = 0;
            }
            fill[p] = f;
        }
    }
}

/// Streaming incremental SWWC for a width that is a whole multiple of `LINE_BYTES`: the cursor is
/// always `LINE_BYTES`-aligned (aligned base + `w_off * width`, a multiple of `LINE_BYTES`), so there
/// is never a head peel and no staging — each element is NT-streamed straight to the cursor.
void scatterColumnStreamIntoSwwc(
    const UInt32 * hash, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t width, ScatterScratch & scratch)
{
    chassert(width >= LINE_BYTES && width % LINE_BYTES == 0);
    const size_t lines = width / LINE_BYTES;
    void ** cur = scratch.cursors();
    const StreamFn stream = selectStream();
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = route(hash[j], shift, mask);
        stream(src + j * width, cur[p], lines);
    }
}

/// Incremental SWWC dispatch on the runtime `elem_width`, mirroring `scatterColumnSwwcDispatch`.
void scatterColumnIntoSwwcDispatch(
    const UInt32 * hash, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t elem_width, ScatterScratch & scratch)
{
    switch (elem_width)
    {
        case 4: scatterColumnTiledIntoSwwc<4>(hash, shift, mask, n, src, scratch); return;
        case 8: scatterColumnTiledIntoSwwc<8>(hash, shift, mask, n, src, scratch); return;
        case 16: scatterColumnTiledIntoSwwc<16>(hash, shift, mask, n, src, scratch); return;
        case 32: scatterColumnTiledIntoSwwc<32>(hash, shift, mask, n, src, scratch); return;
        default:
            if (elem_width % LINE_BYTES == 0)
            {
                scatterColumnStreamIntoSwwc(hash, shift, mask, n, src, elem_width, scratch);
                return;
            }
            /// Any other multiple of 4 (12,20,24,…): no clean SWWC tiling — append directly into the
            /// persistent cursors (no NT; fill stays 0 so the drain is a no-op for these partitions).
            scatterColumnDirectDispatch(hash, shift, mask, n, src, elem_width, scratch.cursors());
            return;
    }
}

/// Drain residual partial staging lines, then fence the NT stores. Shared by the public wrapper.
void scatterColumnDrainSwwcImpl(size_t partitions, ScatterScratch & scratch)
{
    char * staging = scratch.staging();
    void ** cur = scratch.cursors();
    UInt32 * fill = scratch.fill();
    for (size_t p = 0; p < partitions; ++p)
    {
        const UInt32 f = fill[p];
        if (f == 0)
            continue;
        char * d = static_cast<char *>(cur[p]);
        /// `d` is the partition's LINE_BYTES-aligned write cursor and the array is roundUpTo64-sized,
        /// so the (< one line) residual write is in-bounds.
        chassert((reinterpret_cast<uintptr_t>(d) & (LINE_BYTES - 1)) == 0); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        __builtin_memcpy(d, staging + p * LINE_BYTES, f); /// runtime size -> memcpy
        cur[p] = d + f;
        fill[p] = 0;
    }
    streamingFence();
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
    /// Re-measured under the realistic alloc+fault model (`bench_radix_sweep_native`, 16 threads) on
    /// the `x86-64-v3` multitarget build, where the NT path is genuinely emitted: SWWC + NT beats the
    /// direct batched scatter from `P ~= 256` upward (once the fanout reaches a few hundred the
    /// per-partition outputs no longer stay cache-resident and the write-combining + cache-bypass
    /// wins); below that the direct scatter wins. In a build without NT the direct path is always used.
    /// So engage SWWC iff NT is available and `partitions >= 256`.
    return ntStoresAvailable() && partitions >= 256;
}

ScatterScratch::ScatterScratch(size_t max_partitions)
    : capacity(max_partitions)
    , cursor_ptrs(max_partitions, nullptr)
    , line_fill(max_partitions, 0)
{
    /// One write-combining line per partition, LINE_BYTES-aligned for the NT stores.
    const size_t bytes = capacity * LINE_BYTES;
    if (posix_memalign(reinterpret_cast<void **>(&staging_buf), LINE_BYTES, bytes) != 0 || staging_buf == nullptr)
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
    const UInt32 * hash,
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
        return scatterColumnSwwcDispatch(hash, shift, mask, n, bytes, elem_width, partitions, out, scratch);
#endif

    return scatterColumnDirectFromBases(hash, shift, mask, n, bytes, elem_width, partitions, out, scratch);
}

ScatterStats scatterKeyRefTwoColumn(
    const UInt32 * hash,
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
    /// Column-major: scatter the key column and the BuildRef column separately, both routed by `hash`
    /// (reusing the same scratch sequentially; each call re-initialises its cursors / fills).
    const ScatterStats key_stats = scatterColumn(hash, shift, mask, n, keys, key_width, partitions, key_out, scratch, use_swwc);
    const ScatterStats ref_stats = scatterColumn(
        hash,
        shift,
        mask,
        n,
        refs,
        sizeof(BuildRef),
        partitions,
        reinterpret_cast<void * const *>(ref_out), /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        scratch,
        use_swwc);
    return ScatterStats{key_stats.nt_store_bytes + ref_stats.nt_store_bytes};
}

size_t scatterColumnInto(
    const UInt32 * hash,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const void * src,
    size_t elem_width,
    [[maybe_unused]] size_t partitions,
    void ** cursors)
{
    chassert(mask == static_cast<UInt32>(partitions) - 1); /// window must select exactly [0, partitions)
    chassert(elem_width >= 4 && elem_width % 4 == 0 && "RadixShuffle scatter supports element widths that are multiples of 4");
    scatterColumnDirectDispatch(hash, shift, mask, n, static_cast<const char *>(src), elem_width, cursors);
    return n * elem_width;
}

size_t scatterKeyRefInto(
    const UInt32 * hash,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const void * keys,
    size_t key_width,
    const BuildRef * refs,
    size_t partitions,
    void ** key_cursors,
    BuildRef ** ref_cursors)
{
    const size_t key_bytes = scatterColumnInto(hash, shift, mask, n, keys, key_width, partitions, key_cursors);
    const size_t ref_bytes = scatterColumnInto(
        hash,
        shift,
        mask,
        n,
        refs,
        sizeof(BuildRef),
        partitions,
        reinterpret_cast<void **>(ref_cursors)); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    return key_bytes + ref_bytes;
}

size_t scatterColumnIntoSwwc(
    const UInt32 * hash,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const void * src,
    size_t elem_width,
    [[maybe_unused]] size_t partitions,
    ScatterScratch & scratch)
{
    chassert(mask == static_cast<UInt32>(partitions) - 1); /// window must select exactly [0, partitions)
    chassert(elem_width >= 4 && elem_width % 4 == 0 && "RadixShuffle scatter supports element widths that are multiples of 4");

#if USE_MULTITARGET_CODE
    scatterColumnIntoSwwcDispatch(hash, shift, mask, n, static_cast<const char *>(src), elem_width, scratch);
#else
    /// NT is unavailable in this build, so callers gate on `ntStoresAvailable()` and never reach here;
    /// keep the symbol linkable and correct by appending directly into the persistent cursors.
    scatterColumnInto(hash, shift, mask, n, src, elem_width, partitions, scratch.cursors());
#endif

    /// Bytes scattered for this call — the same accounting as `scatterColumnInto` (not only the
    /// NT-flushed subset), so the join's byte-scattered total is independent of the scatter path.
    return n * elem_width;
}

void scatterColumnDrainSwwc([[maybe_unused]] size_t partitions, [[maybe_unused]] ScatterScratch & scratch)
{
#if USE_MULTITARGET_CODE
    scatterColumnDrainSwwcImpl(partitions, scratch);
#endif
    /// Without NT (`#else`) nothing was staged — the direct path wrote every row in place.
}

}
