#include <Interpreters/RadixHashJoin/KeyRefScatter.h>

#include <Common/Exception.h>
#include <Common/TargetSpecific.h>

#include <atomic>
#include <cstdlib>
#include <utility>

#include <algorithm>

namespace DB
{
namespace ErrorCodes
{
extern const int CANNOT_ALLOCATE_MEMORY;
}
}

namespace DB::RadixJoin
{

namespace
{

inline UInt32 routeOf(UInt32 route, UInt32 shift, UInt32 mask) noexcept
{
    return (route >> shift) & mask;
}

/// ---- DIRECT path -----------------------------------------------------------------------------
/// Compile-time width: each element copy lowers to inlined typed stores (no memcpy call).
template <UInt32 width>
void appendDirectFixed(const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const char * src, void ** cursors)
{
    static_assert(width >= 4 && width % 4 == 0);
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = routeOf(route[j], shift, mask);
        char * d = static_cast<char *>(cursors[p]);
        __builtin_memcpy_inline(d, src + j * width, width);
        cursors[p] = d + width;
    }
}

/// Any other multiple of 4: 4-byte-lane inlined copies, still no memcpy call.
void appendDirectLanes(const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t width, void ** cursors)
{
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = routeOf(route[j], shift, mask);
        char * d = static_cast<char *>(cursors[p]);
        const char * s = src + j * width;
        for (size_t b = 0; b < width; b += 4)
            __builtin_memcpy_inline(d + b, s + b, 4);
        cursors[p] = d + width;
    }
}

void appendDirectDispatch(const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t width, void ** cursors)
{
    switch (width)
    {
        case 4: appendDirectFixed<4>(route, shift, mask, n, src, cursors); return;
        case 8: appendDirectFixed<8>(route, shift, mask, n, src, cursors); return;
        case 16: appendDirectFixed<16>(route, shift, mask, n, src, cursors); return;
        case 32: appendDirectFixed<32>(route, shift, mask, n, src, cursors); return;
        case 64: appendDirectFixed<64>(route, shift, mask, n, src, cursors); return;
        default: appendDirectLanes(route, shift, mask, n, src, width, cursors); return;
    }
}

#if USE_MULTITARGET_CODE

using NtLine = char __attribute__((vector_size(LINE_BYTES)));

DECLARE_MULTITARGET_CODE(

[[maybe_unused]] inline void flushOneLine(const void * line, void *& cursor) /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
{
    const auto * s = reinterpret_cast<const NtLine *>(line);
    auto * d = reinterpret_cast<NtLine *>(cursor);
    __builtin_nontemporal_store(*s, d);
    cursor = reinterpret_cast<char *>(cursor) + LINE_BYTES;
}

/// Incremental SWWC for a width that divides the line ({4,8,16,32}): one staging line per partition
/// holding LINE_BYTES/width slots. `width` is a compile-time constant -> single typed stores.
template <UInt32 width>
void appendTiledSwwc(const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const char * src, ScatterScratch & scratch)
{
    static_assert(width >= 4 && width <= 32 && LINE_BYTES % width == 0);
    char * staging = scratch.staging();
    void ** cursors = scratch.cursors();
    UInt32 * fill = scratch.fill();
    UInt32 * peel = scratch.peel();

    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = routeOf(route[j], shift, mask);
        if (peel[p])
        {
            char * c = static_cast<char *>(cursors[p]);
            __builtin_memcpy_inline(c, src + j * width, width);
            cursors[p] = c + width;
            --peel[p];
            continue;
        }

        char * line = staging + static_cast<size_t>(p) * LINE_BYTES;
        UInt32 f = fill[p];
        __builtin_memcpy_inline(line + f, src + j * width, width);
        f += width;
        if (f == LINE_BYTES)
        {
            flushOneLine(line, cursors[p]);
            f = 0;
        }
        fill[p] = f;
    }
}

/// Incremental SWWC for a width that is a whole multiple of the line: every element is already a whole
/// number of lines and the cursor is always line-aligned, so stream straight through with no staging.
void appendStreamSwwc(const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t width, ScatterScratch & scratch)
{
    const size_t lines = width / LINE_BYTES;
    void ** cursors = scratch.cursors();
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = routeOf(route[j], shift, mask);
        const char * s = src + j * width;
        for (size_t k = 0; k < lines; ++k)
            flushOneLine(s + k * LINE_BYTES, cursors[p]);
    }
}

/// Any multiple of 4: 4-byte-lane inlined copies (no memcpy call).
void memcpyLanes4(char * d, const char * s, size_t nbytes)
{
    for (size_t b = 0; b < nbytes; b += 4)
        __builtin_memcpy_inline(d + b, s + b, 4);
}

/// Generic incremental SWWC for any multiple-of-4 width that does not tile a line cleanly: copy each
/// element into the staging line, NT-flush whole lines as they fill, and straddle the 64-byte boundary
/// when the record is wider than the remaining space in the current line. If the partition cursor is not
/// line-aligned (record widths that do not divide 64 cannot align via whole-element peel), byte-copy into
/// the cursor until it is aligned before staging.
void appendGenericSwwc(
    const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t elem_width, ScatterScratch & scratch)
{
    char * staging = scratch.staging();
    void ** cursors = scratch.cursors();
    UInt32 * fill = scratch.fill();
    UInt32 * peel = scratch.peel();

    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = routeOf(route[j], shift, mask);
        const char * s = src + j * elem_width;
        size_t src_off = 0;
        size_t remaining = elem_width;

        if (peel[p])
        {
            char * c = static_cast<char *>(cursors[p]);
            memcpyLanes4(c, s, elem_width);
            cursors[p] = c + elem_width;
            --peel[p];
            continue;
        }

        char * c = static_cast<char *>(cursors[p]);
        uintptr_t mis = reinterpret_cast<uintptr_t>(c) & (LINE_BYTES - 1); /// NOLINT
        if (mis != 0)
        {
            const size_t align_bytes = LINE_BYTES - mis;
            const size_t head = std::min(remaining, align_bytes);
            memcpyLanes4(c, s, head);
            c += head;
            src_off += head;
            remaining -= head;
            cursors[p] = c;
            mis = reinterpret_cast<uintptr_t>(c) & (LINE_BYTES - 1); /// NOLINT
            chassert(mis == 0);
        }

        char * line = staging + static_cast<size_t>(p) * LINE_BYTES;
        while (remaining > 0)
        {
            UInt32 f = fill[p];
            const size_t space = LINE_BYTES - f;
            const size_t copy_n = std::min(remaining, space);
            memcpyLanes4(line + f, s + src_off, copy_n);
            f += static_cast<UInt32>(copy_n);
            src_off += copy_n;
            remaining -= copy_n;
            if (f == LINE_BYTES)
            {
                flushOneLine(line, cursors[p]);
                f = 0;
            }
            fill[p] = f;
        }
    }
}

[[maybe_unused]] void appendSwwcDispatch(
    const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const char * bytes, size_t elem_width, ScatterScratch & scratch)
{
    switch (elem_width)
    {
        case 4: appendTiledSwwc<4>(route, shift, mask, n, bytes, scratch); break;
        case 8: appendTiledSwwc<8>(route, shift, mask, n, bytes, scratch); break;
        case 16: appendTiledSwwc<16>(route, shift, mask, n, bytes, scratch); break;
        case 32: appendTiledSwwc<32>(route, shift, mask, n, bytes, scratch); break;
        default:
            if (elem_width % LINE_BYTES == 0)
                appendStreamSwwc(route, shift, mask, n, bytes, elem_width, scratch);
            else
                appendGenericSwwc(route, shift, mask, n, bytes, elem_width, scratch);
            break;
    }
}

) /// DECLARE_MULTITARGET_CODE

void streamingFence() noexcept
{
    /// NT stores are weakly ordered; make them visible before the outputs are read. Lowers to mfence.
    std::atomic_thread_fence(std::memory_order::seq_cst);
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

bool shouldUseSwwc(int partitions) noexcept
{
    return ntStoresAvailable() && partitions >= 256;
}

ScatterScratch::ScatterScratch(size_t max_partitions)
    : capacity(max_partitions)
    , cursor_ptrs(max_partitions, nullptr)
    , line_fill(max_partitions, 0)
    , line_peel(max_partitions, 0)
{
    const size_t bytes = capacity * LINE_BYTES;
    if (posix_memalign(reinterpret_cast<void **>(&staging_buf), LINE_BYTES, bytes) != 0 || staging_buf == nullptr)
        throw Exception(ErrorCodes::CANNOT_ALLOCATE_MEMORY, "RadixJoin::ScatterScratch failed to allocate {} bytes", bytes);
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
    , line_peel(std::move(other.line_peel))
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
        line_peel = std::move(other.line_peel);
        other.staging_buf = nullptr;
        other.capacity = 0;
    }
    return *this;
}

void ScatterScratch::resetFills(size_t partitions) noexcept
{
    for (size_t p = 0; p < partitions; ++p)
    {
        cursor_ptrs[p] = nullptr;
        line_fill[p] = 0;
        line_peel[p] = 0;
    }
}

size_t appendColumnDirect(
    const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const void * src, size_t elem_width, void ** cursors)
{
    chassert(elem_width >= 4 && elem_width % 4 == 0);
    appendDirectDispatch(route, shift, mask, n, static_cast<const char *>(src), elem_width, cursors);
    return n * elem_width;
}

size_t appendColumnSwwc(
    const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const void * src, size_t elem_width, ScatterScratch & scratch)
{
    chassert(elem_width >= 4 && elem_width % 4 == 0);
#if USE_MULTITARGET_CODE
    const auto * bytes = static_cast<const char *>(src);
    if (isArchSupported(TargetArch::x86_64_v4))
        TargetSpecific::x86_64_v4::appendSwwcDispatch(route, shift, mask, n, bytes, elem_width, scratch);
    else
        TargetSpecific::x86_64_v3::appendSwwcDispatch(route, shift, mask, n, bytes, elem_width, scratch);
#else
    appendDirectDispatch(route, shift, mask, n, static_cast<const char *>(src), elem_width, scratch.cursors());
#endif
    return n * elem_width;
}

void drainColumnSwwc([[maybe_unused]] size_t partitions, [[maybe_unused]] ScatterScratch & scratch)
{
#if USE_MULTITARGET_CODE
    char * staging = scratch.staging();
    void ** cursors = scratch.cursors();
    UInt32 * fill = scratch.fill();
    for (size_t p = 0; p < partitions; ++p)
    {
        const UInt32 f = fill[p];
        if (f == 0)
            continue;
        char * d = static_cast<char *>(cursors[p]);
        /// `d` is line-aligned and the per-partition array is line-padded (roundUpToLine), so the
        /// residual (< one line) copy is in-bounds.
        chassert((reinterpret_cast<uintptr_t>(d) & (LINE_BYTES - 1)) == 0); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        __builtin_memcpy(d, staging + p * LINE_BYTES, f);
        cursors[p] = d + f;
        fill[p] = 0;
    }
    streamingFence();
#endif
}

}
