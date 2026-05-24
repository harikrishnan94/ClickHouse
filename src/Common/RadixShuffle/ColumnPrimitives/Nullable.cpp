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

// ── Raw-output-pointer scatter for Nullable (OutBlock model) ─────────────────
//
// Layout in OutBlock for column k of type Nullable(T):
//   cols[k] points to a buffer of raw_elem_size * capacity bytes, split:
//     [0,       capacity)            — null_map  (uint8_t[capacity])
//     [capacity, capacity*(1+szT))   — values    (T[capacity])
//
// state.raw_write_ptrs[p] → current null_map write pointer for partition p.
// state.nested            → ScatterState for the nested leaf primitive.
// state.raw_prim          → &self (set by on_grow_raw; used by scatter_raw /
//                           scatter_raw_swwc which lack a self parameter).


/// Direct scatter: null_map written via raw_write_ptrs; nested value column
/// delegated to the leaf primitive through state.raw_prim.
[[gnu::hot]] void scatterRawNullable(const IColumn & src_, size_t offset, const uint32_t * pids, int n, ScatterState & state)
{
    const auto & col = assert_cast<const ColumnNullable &>(src_);
    const auto & null_map = col.getNullMapData();

    uint8_t ** null_wp = reinterpret_cast<uint8_t **>(state.raw_write_ptrs); // NOLINT
    for (int j = 0; j < n; ++j)
        *null_wp[pids[j]]++ = null_map[offset + j];

    if (state.raw_prim && state.raw_prim->nested && state.raw_prim->nested->scatter_raw && state.nested)
        state.raw_prim->nested->scatter_raw(col.getNestedColumn(), offset, pids, n, *state.nested);
}


/// SWWC scatter: null_map uses direct scatter (1 byte/row; staging not needed),
/// nested value column uses SWWC (or direct if nested has no SWWC support).
[[gnu::hot]] void scatterRawSwwcNullable(
    const IColumn & src_, size_t offset, const uint32_t * pids, const uint32_t * positions, int n, ScatterState & state)
{
    const auto & col = assert_cast<const ColumnNullable &>(src_);
    const auto & null_map = col.getNullMapData();

    // Direct scatter for null map — 1 byte/row is cheap; staging adds no benefit.
    uint8_t ** null_wp = reinterpret_cast<uint8_t **>(state.raw_write_ptrs); // NOLINT
    for (int j = 0; j < n; ++j)
        *null_wp[pids[j]]++ = null_map[offset + j];

    if (!state.raw_prim || !state.raw_prim->nested || !state.nested)
        return;

    const ColumnPrimitives & nested_prim = *state.raw_prim->nested;
    if (nested_prim.scatter_raw_swwc)
        nested_prim.scatter_raw_swwc(col.getNestedColumn(), offset, pids, positions, n, *state.nested);
    else if (nested_prim.scatter_raw)
        nested_prim.scatter_raw(col.getNestedColumn(), offset, pids, n, *state.nested);
}


/// Drain: null_map uses direct scatter so has no staged residual.
/// Delegate entirely to the nested primitive's drain.
void drainRawNullable(const ColumnPrimitives & self, size_t p, uint32_t cnt, ScatterState & state)
{
    if (!self.nested || !self.nested->drain_raw || !state.nested)
        return;
    self.nested->drain_raw(*self.nested, p, cnt, *state.nested);
}


/// On-grow: split the column buffer into null_map and values regions,
/// update null_map write pointer, and delegate values pointer to the nested
/// primitive's on_grow_raw.
void onGrowRawNullable(const ColumnPrimitives & self, size_t p, void * col_base, size_t capacity, ScatterState & state)
{
    if (state.raw_write_ptrs == nullptr)
    {
        const size_t num_parts = state.fixed_ptrs.size();
        state.raw_write_ptrs = static_cast<void **>(std::calloc(num_parts, sizeof(void *)));
        if (!state.raw_write_ptrs)
            throw std::bad_alloc{};
        if (!state.nested)
            state.nested = std::make_unique<ScatterState>(num_parts);
        state.raw_prim = &self;
    }
    // Null map occupies bytes [0, capacity) within the column buffer.
    state.raw_write_ptrs[p] = col_base;
    // Values start immediately after the null map region.
    if (self.nested && self.nested->on_grow_raw && state.nested)
    {
        char * values_base = static_cast<char *>(col_base) + capacity;
        self.nested->on_grow_raw(*self.nested, p, values_base, capacity, *state.nested);
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
    // Raw scatter is only supported when the nested type supports it.
    if (nested.raw_elem_size > 0)
    {
        cp.scatter_raw      = &scatterRawNullable;
        cp.scatter_raw_swwc = &scatterRawSwwcNullable;
        cp.drain_raw        = &drainRawNullable;
        cp.on_grow_raw      = &onGrowRawNullable;
        cp.raw_elem_size    = sizeof(uint8_t) + nested.raw_elem_size;
    }
    cp.nested = std::make_shared<const ColumnPrimitives>(std::move(nested));
    return cp;
}

}
