#include <Common/RadixShuffle/ColumnPrimitives/Nullable.h>

#include <Columns/ColumnNullable.h>
#include <Common/RadixShuffle/HashCombiner.h>
#include <Common/RadixShuffle/HashKernels.h>
#include <Common/assert_cast.h>

#include <algorithm>
#include <cstring>
#include <utility>


namespace DB::RadixShuffle
{

namespace
{

constexpr size_t MAX_PARTITIONS = 1024;


/// Nullable scatter: write the NullMap slot then delegate to the nested
/// primitive for the remaining slots (and optional data chunk).
///
/// self.fixed_slot_indices[0] is always the NullMap slot index.
/// The nested primitive owns its own fixed_slot_indices.
[[gnu::hot]] void scatterNullable(
    const ColumnPrimitives & self,
    const PartSchema & schema,
    const IColumn & src_,
    const uint16_t * pids,
    size_t n,
    size_t partitions,
    PartReservation * dst)
{
    const auto & col = assert_cast<const ColumnNullable &>(src_);
    const IColumn & nested_col = col.getNestedColumn();
    const auto & null_map = col.getNullMapData();

    const size_t null_slot_idx = self.fixed_slot_indices[0];

    chassert(partitions <= MAX_PARTITIONS);
    uint8_t * null_ptrs[MAX_PARTITIONS];
    for (size_t p = 0; p < partitions; ++p)
    {
        if (dst[p].fixed != nullptr)
        {
            const size_t slot_off = dst[p].fixed->slot_byte_offsets[null_slot_idx];
            null_ptrs[p] = static_cast<uint8_t *>(dst[p].fixed->data) + slot_off
                + dst[p].begin_row;
        }
        else
        {
            null_ptrs[p] = nullptr;
        }
    }

    for (size_t j = 0; j < n; ++j)
        *null_ptrs[pids[j]]++ = null_map[j];

    self.nested->scatter(*self.nested, schema, nested_col, pids, n, partitions, dst);
}


ResumePosition reconstructNullable(
    const ColumnPrimitives & self,
    const PartSchema & schema,
    const PartReservationView * views,
    size_t n_views,
    ResumePosition start,
    IColumn & target)
{
    auto & col = assert_cast<ColumnNullable &>(target);
    auto & null_map = col.getNullMapData();
    auto & nested_col = col.getNestedColumn();

    const size_t rows_before = nested_col.size();
    const ResumePosition end_pos
        = self.nested->reconstruct(*self.nested, schema, views, n_views, start, nested_col);
    const size_t rows_after = nested_col.size();
    const size_t rows_added = rows_after - rows_before;

    chassert(null_map.capacity() >= rows_after);

    const size_t null_slot_idx = self.fixed_slot_indices[0];

    size_t rows_remaining = rows_added;
    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    null_map.resize_assume_reserved(rows_after);
    auto * null_dst = null_map.data() + rows_before;

    while (rows_remaining > 0 && vi < n_views)
    {
        const PartReservationView & v = views[vi];
        const size_t view_rows = v.row_end - v.row_begin;
        const size_t available = view_rows - in_view;
        const size_t take = std::min(available, rows_remaining);

        const size_t slot_off = v.fixed->slot_byte_offsets[null_slot_idx];
        const uint8_t * chunk_null
            = static_cast<const uint8_t *>(v.fixed->data) + slot_off + v.row_begin + in_view;
        std::memcpy(null_dst, chunk_null, take);
        null_dst += take;
        rows_remaining -= take;

        in_view += take;
        if (in_view == view_rows)
        {
            ++vi;
            in_view = 0;
        }
    }

    chassert(rows_remaining == 0);
    chassert(end_pos.view_index == vi && end_pos.rows_consumed_in_view == in_view);
    return end_pos;
}


void hashNullable(
    const ColumnPrimitives & self,
    const PartSchema & schema,
    const IColumn & src_,
    size_t n,
    uint32_t * out)
{
    const auto & col = assert_cast<const ColumnNullable &>(src_);
    const auto & null_map = col.getNullMapData();

    for (size_t i = 0; i < n; ++i)
        out[i] = hashCombine(out[i], fmix32(static_cast<uint32_t>(null_map[i])));

    self.nested->hash(*self.nested, schema, col.getNestedColumn(), n, out);
}

} // namespace


ColumnPrimitives makeNullable(ColumnPrimitives nested)
{
    ColumnPrimitives cp;
    cp.scatter = &scatterNullable;
    cp.reconstruct = &reconstructNullable;
    cp.hash = &hashNullable;
    cp.writes_varlen = nested.writes_varlen;
    cp.nested = std::make_shared<const ColumnPrimitives>(std::move(nested));
    return cp;
}

}
