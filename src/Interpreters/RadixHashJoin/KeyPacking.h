#pragma once

#include <base/types.h>

#include <cstddef>

namespace DB::RadixHash
{

/// Row-major key packing kernel: copy column `src` (raw fixed-width data, `width` bytes/row) for rows
/// `[row_begin, row_begin + rows)` into the packed buffer `dst` at byte `dst_offset` within each
/// `stride`-byte packed row. Shared by the build histogram/scatter (`BuildStore`) and the probe selector
/// (`RadixHashJoin::joinBlock`) so both pack composite keys to the identical layout. The width dispatch is
/// hoisted to construction time via `chooseKeyPacker` (a function-pointer table), so the hot loop runs a
/// compile-time-width `__builtin_memcpy_inline` (common widths) or a 4-byte-lane copy (any other multiple
/// of 4) with no runtime `memcpy`.
using PackKeyColumnFn = void (*)(const char *, size_t, size_t, char *, size_t, size_t, size_t);

template <size_t width>
inline void packKeyColumnT(const char * src, size_t row_begin, size_t rows, char * dst, size_t stride, size_t dst_offset, size_t)
{
    static_assert(width >= 4 && width % 4 == 0);
    for (size_t r = 0; r < rows; ++r)
        __builtin_memcpy_inline(dst + r * stride + dst_offset, src + (row_begin + r) * width, width);
}

inline void packKeyColumnGeneric(
    const char * src, size_t row_begin, size_t rows, char * dst, size_t stride, size_t dst_offset, size_t width)
{
    for (size_t r = 0; r < rows; ++r)
    {
        const char * s = src + (row_begin + r) * width;
        char * d = dst + r * stride + dst_offset;
        for (size_t b = 0; b < width; b += 4)
            __builtin_memcpy_inline(d + b, s + b, 4);
    }
}

inline PackKeyColumnFn chooseKeyPacker(size_t width)
{
    switch (width)
    {
        case 4:  return &packKeyColumnT<4>;
        case 8:  return &packKeyColumnT<8>;
        case 16: return &packKeyColumnT<16>;
        case 32: return &packKeyColumnT<32>;
        case 64: return &packKeyColumnT<64>;
        default: return &packKeyColumnGeneric;
    }
}

}
