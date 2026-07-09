#pragma once

#include <base/types.h>

#include <cstddef>

namespace DB::RadixJoin
{

/** Row-major key packing.
  *
  * A composite join key is materialised as a fixed-stride byte row: column `c`'s `width_c` raw
  * bytes are written at a fixed offset inside each `stride`-byte slot, so the concatenation of the
  * key columns of one row is contiguous and can be hashed / compared as a single span. A
  * single-column key needs no packing at all — the column's own raw data already IS the packed key,
  * and callers special-case that to avoid a copy.
  *
  * The width dispatch is resolved once (at construction) into a function pointer per key column via
  * `chooseColumnPacker`, so the inner copy loop runs a compile-time-width `memcpy_inline` for the
  * common widths (or a 4-byte-lane copy for any other multiple of 4) — never a runtime-length
  * `memcpy` call. The identical packer table is shared by the build and the probe so both sides
  * produce byte-identical packed keys (and therefore identical hashes).
  *
  * Signature: (src_raw, row_begin, rows, dst, stride, dst_offset, width).
  */
using ColumnPackFn = void (*)(const char *, size_t, size_t, char *, size_t, size_t, size_t);

template <size_t width>
inline void packColumnFixed(const char * src, size_t row_begin, size_t rows, char * dst, size_t stride, size_t dst_offset, size_t)
{
    static_assert(width >= 4 && width % 4 == 0, "fixed-width packer needs a multiple-of-4 width");
    for (size_t r = 0; r < rows; ++r)
        __builtin_memcpy_inline(dst + r * stride + dst_offset, src + (row_begin + r) * width, width);
}

/// Any other multiple of 4 (12, 20, 24, ...): copy in 4-byte lanes, still no runtime-length memcpy.
inline void packColumnLanes(const char * src, size_t row_begin, size_t rows, char * dst, size_t stride, size_t dst_offset, size_t width)
{
    for (size_t r = 0; r < rows; ++r)
    {
        const char * s = src + (row_begin + r) * width;
        char * d = dst + r * stride + dst_offset;
        for (size_t b = 0; b < width; b += 4)
            __builtin_memcpy_inline(d + b, s + b, 4);
    }
}

inline ColumnPackFn chooseColumnPacker(size_t width)
{
    switch (width)
    {
        case 4:  return &packColumnFixed<4>;
        case 8:  return &packColumnFixed<8>;
        case 16: return &packColumnFixed<16>;
        case 32: return &packColumnFixed<32>;
        case 64: return &packColumnFixed<64>;
        default: return &packColumnLanes;
    }
}

}
