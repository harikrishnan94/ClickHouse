#include <Common/RadixShuffle/ColumnPrimitives/Nullable.h>

#include <Columns/ColumnNullable.h>
#include <Common/RadixShuffle/HashCombiner.h>
#include <Common/RadixShuffle/HashKernels.h>
#include <Common/assert_cast.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>


namespace DB::RadixShuffle
{

namespace
{

constexpr size_t SCATTER_STACK_PTRS = 1024;


[[gnu::hot]] void scatterNullable(
    const ColumnPrimitives & self,
    const PartSchema & schema,
    const IColumn & src_,
    const uint16_t * pids,
    size_t n,
    size_t partitions,
    const PartReservation * dst,
    ScatterState & state,
    const uint64_t * stale_fixed_bitset)
{
    /// Lazy nested state init.
    if (!state.nested && self.nested)
        state.nested = std::make_unique<ScatterState>(partitions);

    const auto & col = assert_cast<const ColumnNullable &>(src_);
    const IColumn & nested_col = col.getNestedColumn();
    const auto & null_map = col.getNullMapData();

    const size_t null_slot_idx = self.fixed_slot_indices[0];

    /// Refresh stale or uninitialised NullMap write pointers (fixed slot).
    if (!state.initialized)
    {
        for (size_t p = 0; p < partitions; ++p)
        {
            if (dst[p].fixed != nullptr)
            {
                const size_t slot_off = dst[p].fixed->slot_byte_offsets[null_slot_idx];
                state.fixed_ptrs[p] = static_cast<char *>(dst[p].fixed->data) + slot_off + dst[p].begin_row;
            }
            else
            {
                state.fixed_ptrs[p] = nullptr;
            }
        }
        state.initialized = true;
    }
    else
    {
        const size_t words = (partitions + 63) / 64;
        for (size_t word = 0; word < words; ++word)
        {
            uint64_t bits = stale_fixed_bitset[word];
            while (bits)
            {
                const size_t bit = static_cast<size_t>(__builtin_ctzll(bits));
                const size_t p = word * 64 + bit;
                if (p < partitions)
                {
                    if (dst[p].fixed != nullptr)
                    {
                        const size_t slot_off = dst[p].fixed->slot_byte_offsets[null_slot_idx];
                        state.fixed_ptrs[p] = static_cast<char *>(dst[p].fixed->data) + slot_off + dst[p].begin_row;
                    }
                    else
                    {
                        state.fixed_ptrs[p] = nullptr;
                    }
                }
                bits &= bits - 1;
            }
        }
    }

    /// Scatter null map bytes.
    if (partitions <= SCATTER_STACK_PTRS)
    {
        uint8_t * null_ptrs[SCATTER_STACK_PTRS];
        for (size_t p = 0; p < partitions; ++p)
            null_ptrs[p] = static_cast<uint8_t *>(state.fixed_ptrs[p]);
        for (size_t j = 0; j < n; ++j)
            *null_ptrs[pids[j]]++ = null_map[j];
        for (size_t p = 0; p < partitions; ++p)
            state.fixed_ptrs[p] = null_ptrs[p];
    }
    else
    {
        for (size_t j = 0; j < n; ++j)
        {
            *static_cast<uint8_t *>(state.fixed_ptrs[pids[j]]) = null_map[j];
            state.fixed_ptrs[pids[j]] = static_cast<char *>(state.fixed_ptrs[pids[j]]) + sizeof(uint8_t);
        }
    }

    /// Delegate to nested scatter; it handles its own pointer cache and
    /// uses the same stale bitset (same FixedChunk per partition).
    self.nested->scatter(*self.nested, schema, nested_col, pids, n, partitions, dst, *state.nested, stale_fixed_bitset);
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
    const ResumePosition end_pos = self.nested->reconstruct(*self.nested, schema, views, n_views, start, nested_col);
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
        const uint8_t * chunk_null = static_cast<const uint8_t *>(v.fixed->data) + slot_off + v.row_begin + in_view;
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
    const ColumnPrimitives & self, const PartSchema & schema, const IColumn & src_, size_t offset, size_t n, bool initial, uint32_t * out)
{
    const auto & col = assert_cast<const ColumnNullable &>(src_);
    const auto & null_map = col.getNullMapData();

    // Hash the null map into out[] using the caller-specified initial mode.
    for (size_t i = 0; i < n; ++i)
    {
        const uint32_t h = fmix32(static_cast<uint32_t>(null_map[offset + i]));
        out[i] = initial ? h : hashCombine(out[i], h);
    }

    // Always combine nested into out[] — null map already written.
    self.nested->hash(*self.nested, schema, col.getNestedColumn(), offset, n, /*initial=*/false, out);
}


void computePidsNullable(const ColumnPrimitives & self, const IColumn & src_, size_t offset, int n, uint32_t mask, uint32_t * pids)
{
    const auto & col = assert_cast<const ColumnNullable &>(src_);
    const auto & null_map = col.getNullMapData();
    const IColumn & nested_col = col.getNestedColumn();

    if (self.nested && self.nested->compute_pids)
    {
        // Let the nested column compute the initial pids, then mix in the null flag.
        self.nested->compute_pids(*self.nested, nested_col, offset, n, mask, pids);
        for (int j = 0; j < n; ++j)
        {
            const uint32_t null_h = fmix32(static_cast<uint32_t>(null_map[offset + j]));
            pids[j] = (pids[j] ^ (null_h + 0x9e3779b9U + (pids[j] << 6) + (pids[j] >> 2))) & mask;
        }
    }
    else
    {
        // Fallback: hash null map only.
        for (int j = 0; j < n; ++j)
            pids[j] = fmix32(static_cast<uint32_t>(null_map[offset + j])) & mask;
    }
}

} // namespace


ColumnPrimitives makeNullable(ColumnPrimitives nested)
{
    ColumnPrimitives cp;
    cp.scatter = &scatterNullable;
    cp.reconstruct = &reconstructNullable;
    cp.hash = &hashNullable;
    cp.compute_pids = &computePidsNullable;
    cp.writes_varlen = nested.writes_varlen;
    // raw_elem_size is set so RadixPartitionOperator can detect that this is
    // a Nullable with raw scatter support and expand it into two physical
    // primitives (makeFixedWidth<uint8_t> for null_map + nested for values).
    // scatter_raw* are intentionally NOT registered here — the operator handles
    // decomposition directly at the leaf level, eliminating any need for a
    // composite Nullable scatter function.
    if (nested.raw_elem_size > 0)
        cp.raw_elem_size = sizeof(uint8_t) + nested.raw_elem_size;
    cp.nested = std::make_shared<const ColumnPrimitives>(std::move(nested));
    return cp;
}

}
