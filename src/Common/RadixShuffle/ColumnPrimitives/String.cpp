#include <Common/RadixShuffle/ColumnPrimitives/String.h>

#include <Columns/ColumnString.h>
#include <Common/RadixShuffle/HashCombiner.h>
#include <Common/RadixShuffle/HashKernels.h>
#include <Common/assert_cast.h>

#include <cstring>


namespace DB::RadixShuffle
{

namespace
{

constexpr size_t MAX_PARTITIONS = 1024;


/// Scatter ColumnString into per-partition fixed (offsets) + data (chars).
///
/// Offsets slot (fixed_slot_indices[0]): per-row cumulative byte end-position
/// within the DataChunk.  Multiple reservations within the same DataChunk
/// share the same byte stream, so offsets are chunk-global (absolute within
/// DataChunk::bytes).
[[gnu::hot]] void scatterString(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    const uint16_t * pids,
    size_t n,
    size_t partitions,
    PartReservation * dst)
{
    const auto & col = assert_cast<const ColumnString &>(src_);
    const auto & offsets_src = col.getOffsets();
    const auto & chars_src = col.getChars();

    const size_t offsets_slot_idx = self.fixed_slot_indices[0];

    chassert(partitions <= MAX_PARTITIONS);
    uint64_t * off_ptrs[MAX_PARTITIONS];
    unsigned char * char_ptrs[MAX_PARTITIONS];
    size_t abs_byte[MAX_PARTITIONS];

    for (size_t p = 0; p < partitions; ++p)
    {
        if (dst[p].fixed != nullptr)
        {
            const size_t slot_off = dst[p].fixed->slot_byte_offsets[offsets_slot_idx];
            off_ptrs[p] = reinterpret_cast<uint64_t *>(
                              static_cast<char *>(dst[p].fixed->data) + slot_off)
                + dst[p].begin_row;
            char_ptrs[p] = (dst[p].data != nullptr)
                ? dst[p].data->bytes + dst[p].begin_byte
                : nullptr;
            abs_byte[p] = dst[p].begin_byte;
        }
        else
        {
            off_ptrs[p] = nullptr;
            char_ptrs[p] = nullptr;
            abs_byte[p] = 0;
        }
    }

    const auto * chars_src_bytes = reinterpret_cast<const unsigned char *>(chars_src.data());
    UInt64 prev = 0;
    for (size_t j = 0; j < n; ++j)
    {
        const UInt64 end = offsets_src[j];
        const size_t len = end - prev;
        const uint16_t p = pids[j];
        if (len > 0)
            std::memcpy(char_ptrs[p], chars_src_bytes + prev, len);
        char_ptrs[p] += len;
        abs_byte[p] += len;
        *off_ptrs[p]++ = abs_byte[p];
        prev = end;
    }
}


ResumePosition reconstructString(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const PartReservationView * views,
    size_t n_views,
    ResumePosition start,
    IColumn & target)
{
    auto & col = assert_cast<ColumnString &>(target);
    auto & out_chars = col.getChars();
    auto & out_offsets = col.getOffsets();
    const size_t rows_cap = out_offsets.capacity();
    const size_t chars_cap = out_chars.capacity();
    size_t cur_rows = out_offsets.size();
    size_t cur_chars = out_chars.size();

    const size_t offsets_slot_idx = self.fixed_slot_indices[0];

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;

    auto * out_chars_bytes = reinterpret_cast<unsigned char *>(out_chars.data());
    while (vi < n_views)
    {
        const PartReservationView & v = views[vi];
        const size_t view_rows = v.row_end - v.row_begin;

        const size_t slot_off = v.fixed->slot_byte_offsets[offsets_slot_idx];
        const uint64_t * chunk_offsets = reinterpret_cast<const uint64_t *>(
            static_cast<const char *>(v.fixed->data) + slot_off);

        /// DataChunk base; offset values in chunk_offsets are absolute
        /// positions within it.
        const unsigned char * chunk_chars = v.data->bytes;

        const size_t abs_start = v.row_begin + in_view;
        UInt64 row_prev = (abs_start == 0) ? 0 : chunk_offsets[abs_start - 1];

        size_t rows_taken = 0;
        for (size_t i = in_view; i < view_rows; ++i)
        {
            const UInt64 cur_off = chunk_offsets[v.row_begin + i];
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


void hashString(
    const ColumnPrimitives & /*self*/,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    size_t n,
    uint32_t * out)
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
        out[i] = hashCombine(out[i], hashBytes32(chars_src_bytes + prev, len));
        prev = end;
    }
}

} // namespace


ColumnPrimitives makeString()
{
    ColumnPrimitives cp;
    cp.scatter = &scatterString;
    cp.reconstruct = &reconstructString;
    cp.hash = &hashString;
    cp.writes_varlen = true;
    return cp;
}

}
