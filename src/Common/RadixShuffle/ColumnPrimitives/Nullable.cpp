#include <Common/RadixShuffle/ColumnPrimitives/Nullable.h>

#include <Columns/ColumnNullable.h>
#include <Common/RadixShuffle/HashCombiner.h>
#include <Common/assert_cast.h>

#include <algorithm>
#include <cstring>
#include <utility>


namespace DB::RadixShuffle
{

namespace
{

constexpr size_t MAX_PARTITIONS = 1024;


[[gnu::hot]] void
scatterNullable(const ColumnPrimitives & self, const IColumn & src_, const uint32_t * pids, size_t n, size_t partitions, Reservation * dst)
{
    const auto & col = assert_cast<const ColumnNullable &>(src_);
    const IColumn & nested = col.getNestedColumn();
    const auto & null_map = col.getNullMapData();

    /// Per-partition null-map write pointers. The nested scatter touches
    /// `chunk->primary` (and `chunk->offsets` for variable-length) but
    /// not `chunk->null_map`; we handle that here.
    chassert(partitions <= MAX_PARTITIONS);
    uint8_t * null_ptrs[MAX_PARTITIONS];
    for (size_t p = 0; p < partitions; ++p)
    {
        if (dst[p].chunk != nullptr)
            null_ptrs[p] = dst[p].chunk->null_map + dst[p].begin_row;
        else
            null_ptrs[p] = nullptr;
    }

    /// Branch-free null-map scatter.
    for (size_t j = 0; j < n; ++j)
        *null_ptrs[pids[j]]++ = null_map[j];

    /// Delegate to the nested column primitives for the primary buffer. The nested
    /// column primitives write into the same chunk's `primary` (and, for
    /// variable-length, `offsets`) — the chunk was sized at allocator
    /// construction with both the nested's element/byte requirements and
    /// a null map, so writes do not alias.
    self.nested->scatter(*self.nested, nested, pids, n, partitions, dst);
}


ResumePosition
reconstructNullable(const ColumnPrimitives & self, const ChunkRangeView * views, size_t n_views, ResumePosition start, IColumn & target)
{
    auto & col = assert_cast<ColumnNullable &>(target);
    auto & null_map = col.getNullMapData();
    auto & nested_col = col.getNestedColumn();

    /// Sizing contract for ColumnNullable: the caller has reserved the
    /// nested column's row capacity (and, for variable-length, byte
    /// capacity) AND the null map's row capacity. To consume from the
    /// views, we pump the nested first; whatever the nested decided to
    /// consume, we replay the null-map bytes for those exact rows.
    const size_t rows_before = nested_col.size();
    const ResumePosition end_pos = self.nested->reconstruct(*self.nested, views, n_views, start, nested_col);
    const size_t rows_after = nested_col.size();
    const size_t rows_added = rows_after - rows_before;

    /// Walk the views in lockstep with the nested reconstruct's row
    /// consumption to append the same null-map bytes. The spec (§3.3
    /// pre-conditions) requires the caller to pre-reserve the null map's
    /// row capacity; we never grow it here, so a hard `resize_assume_reserved`
    /// is contractually safe (and keeps the no-allocation invariant of §4.2).
    chassert(null_map.capacity() >= rows_after);
    size_t rows_remaining = rows_added;
    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    null_map.resize_assume_reserved(rows_after);
    auto * null_dst = null_map.data() + rows_before;

    while (rows_remaining > 0 && vi < n_views)
    {
        const ChunkRangeView & v = views[vi];
        const size_t view_rows = v.end - v.begin;
        const size_t available = view_rows - in_view;
        const size_t take = std::min(available, rows_remaining);

        const uint8_t * chunk_null = v.chunk->null_map + v.begin + in_view;
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

    /// `end_pos` is authoritative — it reflects the nested's stopping
    /// point. Sanity-check it matches our null-map traversal.
    chassert(rows_remaining == 0);
    chassert(end_pos.view_index == vi && end_pos.rows_consumed_in_view == in_view);
    return end_pos;
}


void hashNullable(const ColumnPrimitives & self, const IColumn & src_, size_t n, uint64_t * out)
{
    const auto & col = assert_cast<const ColumnNullable &>(src_);
    const auto & null_map = col.getNullMapData();

    /// The null-map's contribution to the per-row hash: a one-byte value
    /// mixed through the combiner. We feed `(0 or 1)` through `intHash64`
    /// (mixer used by the fixed-width column primitives) to keep the avalanche
    /// property; the combiner then unifies it with the nested column's
    /// per-row hash.
    for (size_t i = 0; i < n; ++i)
    {
        uint64_t bits = null_map[i] != 0 ? 0xff51afd7ed558ccdULL : 0xc4ceb9fe1a85ec53ULL;
        out[i] = hashCombine(out[i], bits);
    }

    /// Nested column's contribution. We hash the nested rows regardless
    /// of nullness — this matches the round-trip contract (nested bytes
    /// at null positions are preserved, §3.5) and means the hash of a
    /// ColumnNullable depends on the nested bytes verbatim.
    self.nested->hash(*self.nested, col.getNestedColumn(), n, out);
}

}


ColumnPrimitives makeNullable(ColumnPrimitives nested)
{
    auto nested_ptr = std::make_shared<ColumnPrimitives>(std::move(nested));
    ColumnPrimitives column_primitives;
    column_primitives.scatter = &scatterNullable;
    column_primitives.reconstruct = &reconstructNullable;
    column_primitives.hash = &hashNullable;
    column_primitives.column_desc = nested_ptr->column_desc;
    column_primitives.column_desc.has_null_map = true;
    column_primitives.nested = std::move(nested_ptr);
    return column_primitives;
}

}
