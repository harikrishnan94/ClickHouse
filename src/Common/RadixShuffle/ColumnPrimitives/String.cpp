#include <Common/RadixShuffle/ColumnPrimitives/String.h>

#include <Columns/ColumnString.h>
#include <Common/RadixShuffle/HashCombiner.h>
#include <Common/assert_cast.h>

#include <cstring>


namespace DB::RadixShuffle
{

namespace
{

/// Per-row finalizer (same as the FixedWidth column primitives').
[[gnu::always_inline]] inline uint64_t intHash64Local(uint64_t x) noexcept
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}


/// Hash a byte range using 64-bit chunks; uses the same mixer as the
/// fixed-width column primitives (xxHash-style block accumulation) so the round-trip
/// "same column type -> same combiner" property of §3.4 holds across the
/// scatter/reconstruct cycle (since reconstruct only re-lays the bytes,
/// the same bytes produce the same hash).
[[gnu::always_inline]] inline uint64_t hashBytes(const unsigned char * data, size_t n) noexcept
{
    uint64_t acc = 0xcbf29ce484222325ULL ^ (static_cast<uint64_t>(n) * 0x9e3779b97f4a7c15ULL);
    size_t i = 0;
    while (i + sizeof(uint64_t) <= n)
    {
        uint64_t word = 0;
        std::memcpy(&word, data + i, sizeof(uint64_t));
        acc = intHash64Local(word + acc);
        i += sizeof(uint64_t);
    }
    if (i < n)
    {
        uint64_t tail = 0;
        std::memcpy(&tail, data + i, n - i);
        acc = intHash64Local(tail + acc);
    }
    return acc;
}


constexpr size_t MAX_PARTITIONS = 1024;


[[gnu::hot]] void scatterString(
    const ColumnPrimitives & /*self*/, const IColumn & src_, const uint32_t * pids, size_t n, size_t partitions, Reservation * dst)
{
    const auto & col = assert_cast<const ColumnString &>(src_);
    const auto & offsets_src = col.getOffsets();
    const auto & chars_src = col.getChars();

    /// For each partition, write pointers into:
    ///   - the offsets array within the slot (`Chunk::offsets + begin_row`)
    ///   - the chars buffer within the slot (`Chunk::primary + begin_byte`)
    /// `abs_byte` is the running write position relative to the chunk's
    /// chars buffer; the offsets written are chunk-global (i.e., the
    /// CUMULATIVE byte end-position within `chunk->primary`), which lets
    /// multiple slots share one chunk's offsets array and reconstruct
    /// decode rows uniformly via the standard ColumnString recipe.
    chassert(partitions <= MAX_PARTITIONS);
    uint64_t * off_ptrs[MAX_PARTITIONS];
    unsigned char * char_ptrs[MAX_PARTITIONS];
    size_t abs_byte[MAX_PARTITIONS];
    for (size_t p = 0; p < partitions; ++p)
    {
        if (dst[p].chunk != nullptr)
        {
            off_ptrs[p] = dst[p].chunk->offsets + dst[p].begin_row;
            char_ptrs[p] = static_cast<unsigned char *>(dst[p].chunk->primary) + dst[p].begin_byte;
            abs_byte[p] = dst[p].begin_byte;
        }
        else
        {
            off_ptrs[p] = nullptr;
            char_ptrs[p] = nullptr;
            abs_byte[p] = 0;
        }
    }

    /// Row loop. We compute the row's byte slice in the source column via
    /// the standard `offsets[i] - offsets[i-1]` recipe (with implicit -1
    /// equal to 0), then memcpy into the slot's chars buffer and write the
    /// cumulative offset. The "5 µops/row" budget cannot be matched for
    /// variable-length rows since each row entails a memcpy whose
    /// throughput depends on the average string length; the branch-free
    /// property is preserved (no per-row conditionals beyond the
    /// indirection through pids[j]).
    const auto * chars_src_bytes = reinterpret_cast<const unsigned char *>(chars_src.data());
    UInt64 prev = 0;
    for (size_t j = 0; j < n; ++j)
    {
        const UInt64 end = offsets_src[j];
        const size_t len = end - prev;
        const uint32_t p = pids[j];
        std::memcpy(char_ptrs[p], chars_src_bytes + prev, len);
        char_ptrs[p] += len;
        abs_byte[p] += len;
        *off_ptrs[p]++ = abs_byte[p];
        prev = end;
    }
}


ResumePosition
reconstructString(const ColumnPrimitives & /*self*/, const ChunkRangeView * views, size_t n_views, ResumePosition start, IColumn & target)
{
    auto & col = assert_cast<ColumnString &>(target);
    auto & out_chars = col.getChars();
    auto & out_offsets = col.getOffsets();
    const size_t rows_cap = out_offsets.capacity();
    const size_t chars_cap = out_chars.capacity();
    size_t cur_rows = out_offsets.size();
    size_t cur_chars = out_chars.size();

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;

    auto * out_chars_bytes = reinterpret_cast<unsigned char *>(out_chars.data());
    while (vi < n_views)
    {
        const ChunkRangeView & v = views[vi];
        const size_t view_rows = v.end - v.begin;

        /// Walk view rows one-by-one — we have to stop when EITHER the
        /// row capacity OR the byte capacity is exhausted. Offsets are
        /// chunk-global (the cumulative byte end-position within
        /// `chunk->primary`). The byte length of row k is
        /// `chunk->offsets[k] - chunk->offsets[k - 1]`, with implicit -1
        /// equal to 0.
        const uint64_t * chunk_offsets = v.chunk->offsets;
        const auto * chunk_chars = static_cast<const unsigned char *>(v.chunk->primary);

        UInt64 row_prev = (v.begin + in_view == 0) ? 0 : chunk_offsets[v.begin + in_view - 1];

        size_t rows_taken = 0;
        for (size_t i = in_view; i < view_rows; ++i)
        {
            const UInt64 cur_off = chunk_offsets[v.begin + i];
            const UInt64 len = cur_off - row_prev;

            if (cur_rows + 1 > rows_cap || cur_chars + len > chars_cap)
                break;

            std::memcpy(out_chars_bytes + cur_chars, chunk_chars + row_prev, len);
            cur_chars += len;
            out_offsets.resize_assume_reserved(cur_rows + 1);
            out_offsets[cur_rows] = cur_chars;
            ++cur_rows;
            ++rows_taken;
            row_prev = cur_off;
        }

        out_chars.resize_assume_reserved(cur_chars);

        in_view += rows_taken;
        if (in_view == view_rows)
        {
            ++vi;
            in_view = 0;
        }
        else
        {
            break;
        }
    }

    return ResumePosition{vi, in_view};
}


void hashString(const ColumnPrimitives & /*self*/, const IColumn & src_, size_t n, uint64_t * out)
{
    const auto & col = assert_cast<const ColumnString &>(src_);
    const auto & offsets_src = col.getOffsets();
    const auto & chars_src = col.getChars();
    const auto * chars_src_bytes = reinterpret_cast<const unsigned char *>(chars_src.data());
    UInt64 prev = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const UInt64 end = offsets_src[i];
        const size_t len = end - prev;
        const uint64_t h = hashBytes(chars_src_bytes + prev, len);
        out[i] = hashCombine(out[i], h);
        prev = end;
    }
}

}


ColumnPrimitives makeString()
{
    ColumnPrimitives column_primitives;
    column_primitives.scatter = &scatterString;
    column_primitives.reconstruct = &reconstructString;
    column_primitives.hash = &hashString;
    column_primitives.column_desc.element_size = 0;
    column_primitives.column_desc.alignment = 1;
    column_primitives.column_desc.has_offsets = true;
    column_primitives.column_desc.has_null_map = false;
    column_primitives.column_desc.variable_length = true;
    return column_primitives;
}

}
