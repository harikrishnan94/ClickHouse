#include <Common/RadixShuffle/PerBlockArenaShuffler.h>

#include <Columns/ColumnNullable.h>
#include <Columns/IColumn.h>
#include <base/types.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>

#include <algorithm>
#include <chrono>
#include <cstring>


namespace
{

inline uint64_t elapsedNs(std::chrono::steady_clock::time_point t0) noexcept
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

/// Allocator alignment.  Chosen to match `OutBlock`'s 64-byte alignment so
/// the *chunk base* is 64-byte aligned; individual partition slices may still
/// land on arbitrary byte offsets, but the direct-scatter path doesn't care.
constexpr size_t kChunkAlign = 64;

} // namespace


namespace DB
{

PerBlockArenaShuffler::PerBlockArenaShuffler(int P, int K, std::vector<ColumnPrimitives> prims, bool /*use_swwc*/)
    : num_partitions_(P)
    , num_columns_(K)
    , mask_(static_cast<uint32_t>(P) - 1)
    , col_prims_(std::move(prims))
    , scratch_hist_(static_cast<size_t>(P), 0)
    , scratch_prefix_(static_cast<size_t>(P), 0)
    , part_slices_(static_cast<size_t>(P))
    , part_total_rows_(static_cast<size_t>(P), 0)
    , output_(static_cast<size_t>(P))
{
    for (size_t k = 0; k < static_cast<size_t>(num_columns_); ++k)
    {
        const ColumnPrimitives & prim = col_prims_[k];
        const bool expandable = prim.nested != nullptr && prim.nested->scatter_raw != nullptr;
        if (expandable)
        {
            phys_prims_.push_back(makeFixedWidth<UInt8>());
            phys_col_info_.push_back({k, true, false});

            phys_prims_.push_back(*prim.nested);
            phys_col_info_.push_back({k, false, true});
        }
        else
        {
            phys_prims_.push_back(prim);
            phys_col_info_.push_back({k, false, false});
        }
    }
    num_physical_columns_ = static_cast<int>(phys_prims_.size());

    phys_elem_size_.resize(static_cast<size_t>(num_physical_columns_));
    for (int k = 0; k < num_physical_columns_; ++k)
        phys_elem_size_[static_cast<size_t>(k)] = phys_prims_[static_cast<size_t>(k)].raw_elem_size;

    scatter_states_.reserve(static_cast<size_t>(num_physical_columns_));
    for (int k = 0; k < num_physical_columns_; ++k)
        scatter_states_.emplace_back(static_cast<size_t>(num_partitions_));
}


PerBlockArenaShuffler::~PerBlockArenaShuffler()
{
    for (auto & ch : chunks_)
    {
        if (ch.data)
            allocator_.free(ch.data, ch.bytes, ch.alignment);
    }
}


static const IColumn & extractPhysCol(const DB::Columns & columns, const PerBlockPhysColInfo & info)
{
    const IColumn & col = *columns[info.logical_k];
    if (info.use_null_map)
        return assert_cast<const ColumnNullable &>(col).getNullMapColumn();
    if (info.use_nested)
        return assert_cast<const ColumnNullable &>(col).getNestedColumn();
    return col;
}


void PerBlockArenaShuffler::process(const DB::Columns & columns)
{
    if (columns.empty() || columns[0]->size() == 0)
        return;

    const auto proc_t0 = std::chrono::steady_clock::now();
    const size_t n = columns[0]->size();

    // Capture prototype columns from the first block — used in finish() to
    // call cloneEmpty() for output column allocation.  We need them in the
    // physical shape (i.e. after Nullable expansion), so we just keep the
    // original logical columns and re-derive in finish().
    if (proto_columns_.empty())
        proto_columns_ = columns;

    if (n > max_input_block_rows_)
        max_input_block_rows_ = n;

    // ── Phase 1: compute partition IDs for this block ─────────────────────
    if (scratch_pids_.size() < n)
        scratch_pids_.resize(n);
    {
        const auto t = std::chrono::steady_clock::now();
        col_prims_[0].compute_pids(col_prims_[0], *columns[0], 0, static_cast<int>(n), mask_, scratch_pids_.data());
        timings_.pid_compute_ns += elapsedNs(t);
    }

    // ── Phase 2: histogram + exclusive prefix sum ─────────────────────────
    {
        const auto t = std::chrono::steady_clock::now();
        std::memset(scratch_hist_.data(), 0, static_cast<size_t>(num_partitions_) * sizeof(uint32_t));
        const uint32_t * pids = scratch_pids_.data();
        for (size_t j = 0; j < n; ++j)
            ++scratch_hist_[pids[j]];

        uint32_t running = 0;
        for (int p = 0; p < num_partitions_; ++p)
        {
            scratch_prefix_[static_cast<size_t>(p)] = running;
            running += scratch_hist_[static_cast<size_t>(p)];
        }
        timings_.histogram_ns += elapsedNs(t);
    }

    // ── Phase 3: allocate one chunk for this block ────────────────────────
    PerBlockChunk chunk{};
    chunk.input_rows = static_cast<uint32_t>(n);
    {
        const auto t = std::chrono::steady_clock::now();
        size_t chunk_bytes = 0;
        for (int k = 0; k < num_physical_columns_; ++k)
            chunk_bytes += n * phys_elem_size_[static_cast<size_t>(k)];

        // Round to 64-byte boundary; jemalloc returns the same size-class
        // bin for slightly padded sizes, which helps reuse across iterations.
        const size_t bytes_rounded = (chunk_bytes + (kChunkAlign - 1)) & ~(kChunkAlign - 1);
        chunk.data = allocator_.alloc(bytes_rounded, kChunkAlign);
        chunk.bytes = bytes_rounded;
        chunk.alignment = kChunkAlign;
        ++timings_.chunks_allocated;
        timings_.bytes_allocated += bytes_rounded;
        timings_.alloc_ns += elapsedNs(t);
    }
    chunks_.push_back(chunk);
    const uint32_t chunk_idx = static_cast<uint32_t>(chunks_.size() - 1);

    // ── Phase 4: register per-(partition) slice + set write pointers ──────
    {
        const auto t = std::chrono::steady_clock::now();
        char * const base = static_cast<char *>(chunk.data);

        // Compute per-column base offsets within the chunk.
        size_t col_offset = 0;
        for (int k = 0; k < num_physical_columns_; ++k)
        {
            const size_t es = phys_elem_size_[static_cast<size_t>(k)];
            char * col_base = base + col_offset;

            ScatterState & st = scatter_states_[static_cast<size_t>(k)];
            // For each partition, point the write pointer at this block's
            // partition slice.  on_grow_raw also lazily allocates the
            // raw_write_ptrs array on the first call.
            for (int p = 0; p < num_partitions_; ++p)
            {
                const uint32_t off_rows = scratch_prefix_[static_cast<size_t>(p)];
                void * wp = col_base + off_rows * es;
                phys_prims_[static_cast<size_t>(k)].on_grow_raw(
                    phys_prims_[static_cast<size_t>(k)], static_cast<size_t>(p), wp, static_cast<size_t>(scratch_hist_[static_cast<size_t>(p)]), st);
            }
            col_offset += n * es;
        }

        // Record per-partition slice (one entry per block, even for empty).
        for (int p = 0; p < num_partitions_; ++p)
        {
            const uint32_t cnt = scratch_hist_[static_cast<size_t>(p)];
            if (!cnt)
                continue;
            PartSlice s{};
            s.chunk_index = chunk_idx;
            s.row_offset = scratch_prefix_[static_cast<size_t>(p)];
            s.row_count = cnt;
            part_slices_[static_cast<size_t>(p)].push_back(s);
            part_total_rows_[static_cast<size_t>(p)] += cnt;
        }
        timings_.on_grow_ns += elapsedNs(t);
    }

    // ── Phase 5: scatter ──────────────────────────────────────────────────
    {
        const auto t = std::chrono::steady_clock::now();
        const uint32_t * pids = scratch_pids_.data();
        for (int k = 0; k < num_physical_columns_; ++k)
        {
            const IColumn & sub_col = extractPhysCol(columns, phys_col_info_[static_cast<size_t>(k)]);
            phys_prims_[static_cast<size_t>(k)].scatter_raw(
                sub_col, 0, pids, static_cast<int>(n), scatter_states_[static_cast<size_t>(k)]);
        }
        timings_.scatter_ns += elapsedNs(t);
    }

    timings_.total_process_ns += elapsedNs(proc_t0);
    timings_.rows_processed += n;
    ++timings_.blocks_processed;
}


void PerBlockArenaShuffler::finish()
{
    if (chunks_.empty() || max_input_block_rows_ == 0)
        return;

    const auto fin_t0 = std::chrono::steady_clock::now();

    // For each partition, gather its slices and pack into IColumn output blocks
    // of size max_input_block_rows_ (the final block may be shorter).
    for (int p = 0; p < num_partitions_; ++p)
    {
        const auto & slices = part_slices_[static_cast<size_t>(p)];
        if (slices.empty())
            continue;

        const size_t total_rows = part_total_rows_[static_cast<size_t>(p)];
        const size_t block_rows = max_input_block_rows_;
        const size_t num_out_blocks = (total_rows + block_rows - 1) / block_rows;

        // Walk slices with a global cursor; emit one output block at a time.
        size_t slice_idx = 0;
        size_t slice_consumed = 0;

        for (size_t ob = 0; ob < num_out_blocks; ++ob)
        {
            const size_t rows_this_block = std::min(block_rows, total_rows - ob * block_rows);

            // Allocate output IColumns (one per physical column).
            const auto t_alloc = std::chrono::steady_clock::now();
            MutableColumns mcols;
            mcols.resize(static_cast<size_t>(num_physical_columns_));
            std::vector<void *> out_ptrs(static_cast<size_t>(num_physical_columns_));
            for (int k = 0; k < num_physical_columns_; ++k)
            {
                const IColumn & src_proto = extractPhysCol(proto_columns_, phys_col_info_[static_cast<size_t>(k)]);
                auto col = src_proto.cloneEmpty();
                out_ptrs[static_cast<size_t>(k)] = phys_prims_[static_cast<size_t>(k)].resize_for_scatter(*col, rows_this_block);
                mcols[static_cast<size_t>(k)] = std::move(col);
            }
            timings_.finish_alloc_ns += elapsedNs(t_alloc);

            // Copy from slices into this output block.
            const auto t_copy = std::chrono::steady_clock::now();
            size_t dst_filled = 0;
            while (dst_filled < rows_this_block)
            {
                const PartSlice & sl = slices[slice_idx];
                const size_t avail = static_cast<size_t>(sl.row_count) - slice_consumed;
                const size_t take = std::min(avail, rows_this_block - dst_filled);
                const PerBlockChunk & chunk = chunks_[sl.chunk_index];

                size_t col_offset = 0;
                for (int k = 0; k < num_physical_columns_; ++k)
                {
                    const size_t es = phys_elem_size_[static_cast<size_t>(k)];
                    const char * src_col_base = static_cast<const char *>(chunk.data) + col_offset;
                    const char * src = src_col_base + (sl.row_offset + slice_consumed) * es;
                    char * dst = static_cast<char *>(out_ptrs[static_cast<size_t>(k)]) + dst_filled * es;
                    std::memcpy(dst, src, take * es);
                    col_offset += static_cast<size_t>(chunk.input_rows) * es;
                }

                slice_consumed += take;
                dst_filled += take;
                if (slice_consumed == sl.row_count)
                {
                    ++slice_idx;
                    slice_consumed = 0;
                }
            }
            timings_.finish_copy_ns += elapsedNs(t_copy);

            DB::Columns frozen;
            frozen.reserve(mcols.size());
            for (auto & mc : mcols)
                frozen.push_back(std::move(mc));
            output_[static_cast<size_t>(p)].push_back(std::move(frozen));
            ++timings_.output_blocks_emitted;
        }
    }

    timings_.total_finish_ns += elapsedNs(fin_t0);
}

} // namespace DB
