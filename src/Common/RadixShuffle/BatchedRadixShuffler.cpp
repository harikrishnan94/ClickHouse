#include <Common/RadixShuffle/BatchedRadixShuffler.h>

#include <Columns/ColumnNullable.h>
#include <Columns/IColumn.h>
#include <base/types.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>

namespace
{
inline uint64_t elapsedNs(std::chrono::steady_clock::time_point t0) noexcept
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}
}


namespace DB
{

BatchedRadixShuffler::BatchedRadixShuffler(
    int P,
    int K,
    std::vector<ColumnPrimitives> prims,
    bool use_swwc,
    size_t max_buffered_blocks,
    size_t max_buffered_bytes,
    bool use_aligned_alloc)
    : num_partitions_(P)
    , num_columns_(K)
    , use_swwc_(use_swwc)
    , use_aligned_alloc_(use_aligned_alloc)
    , mask_(static_cast<uint32_t>(P) - 1)
    , max_buffered_blocks_(max_buffered_blocks ? max_buffered_blocks : static_cast<size_t>(P))
    , max_buffered_bytes_(max_buffered_bytes ? max_buffered_bytes : kDefaultMemBound)
    , col_prims_(std::move(prims))
    , output_(static_cast<size_t>(P))
    , accum_hist_(static_cast<size_t>(P), 0)
    , cnt_(static_cast<size_t>(P), 0)
    , pending_cols_(static_cast<size_t>(P))
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

    scatter_states_.reserve(static_cast<size_t>(num_physical_columns_));
    for (int k = 0; k < num_physical_columns_; ++k)
        scatter_states_.emplace_back(static_cast<size_t>(num_partitions_));

    bytes_per_row_ = 0;
    for (int k = 0; k < num_physical_columns_; ++k)
        bytes_per_row_ += phys_prims_[static_cast<size_t>(k)].raw_elem_size;
}


BatchedRadixShuffler::~BatchedRadixShuffler()
{
    for (void * p : aligned_allocs_)
        std::free(p);
}


static const IColumn & extractPhysCol(const DB::Columns & columns, const BatchedPhysColInfo & info)
{
    const IColumn & col = *columns[info.logical_k];
    if (info.use_null_map)
        return assert_cast<const ColumnNullable &>(col).getNullMapColumn();
    if (info.use_nested)
        return assert_cast<const ColumnNullable &>(col).getNestedColumn();
    return col;
}


void BatchedRadixShuffler::process(const DB::Columns & columns)
{
    if (columns.empty() || columns[0]->size() == 0)
        return;

    const auto proc_t0 = std::chrono::steady_clock::now();
    const size_t n = columns[0]->size();

    {
        const auto t = std::chrono::steady_clock::now();
        const size_t pid_offset = buffered_pids_.size();
        buffered_pids_.resize(pid_offset + n);
        col_prims_[0].compute_pids(col_prims_[0], *columns[0], 0, static_cast<int>(n), mask_, buffered_pids_.data() + pid_offset);
        timings_.pid_compute_ns += elapsedNs(t);

        const auto th = std::chrono::steady_clock::now();
        for (size_t j = 0; j < n; ++j)
            ++accum_hist_[buffered_pids_[pid_offset + j]];
        timings_.histogram_ns += elapsedNs(th);
    }

    {
        const auto t = std::chrono::steady_clock::now();
        buffered_blocks_.push_back(columns);
        total_buffered_bytes_ += n * bytes_per_row_;
        timings_.buffer_push_ns += elapsedNs(t);
    }

    timings_.total_process_ns += elapsedNs(proc_t0);
    timings_.rows_processed += n;

    if (buffered_blocks_.size() >= max_buffered_blocks_ || total_buffered_bytes_ >= max_buffered_bytes_)
        flush();
}


void BatchedRadixShuffler::flush()
{
    if (buffered_blocks_.empty())
        return;

    const auto flush_t0 = std::chrono::steady_clock::now();
    ++timings_.flush_count;

    // ── Phase 1: allocate per-partition output buffers ────────────────────
    //
    // Two backends produce the SAME on-the-wire write pointers for scatter:
    //   • use_aligned_alloc_ == false: cloneEmpty + reserve_resize per (p,k).
    //     Produces a typed IColumn pinned in pending_cols_[p][k].
    //   • use_aligned_alloc_ == true : std::aligned_alloc(64, cap*elem_size)
    //     per (p,k).  Skips IColumn entirely; output() stays empty.  Used
    //     to isolate IColumn cost from scatter / page-fault cost.
    for (int p = 0; p < num_partitions_; ++p)
    {
        const uint32_t cnt = accum_hist_[static_cast<size_t>(p)];
        if (!cnt)
            continue;

        MutableColumns & mcols = pending_cols_[static_cast<size_t>(p)];
        if (!use_aligned_alloc_)
            mcols.resize(static_cast<size_t>(num_physical_columns_));

        for (int k = 0; k < num_physical_columns_; ++k)
        {
            void * ptr;
            if (use_aligned_alloc_)
            {
                const auto t_clone = std::chrono::steady_clock::now();
                const size_t bytes = static_cast<size_t>(cnt) * phys_prims_[static_cast<size_t>(k)].raw_elem_size;
                // aligned_alloc requires size to be a multiple of alignment.
                const size_t bytes_rounded = (bytes + 63) & ~size_t{63};
                ptr = std::aligned_alloc(64, bytes_rounded);
                if (!ptr)
                    throw std::bad_alloc{};
                aligned_allocs_.push_back(ptr);
                timings_.clone_empty_ns += elapsedNs(t_clone);
                ++timings_.alloc_count;
                // reserve_resize / move steps are degenerate in this backend.
            }
            else
            {
                const auto t_clone = std::chrono::steady_clock::now();
                const IColumn & src = extractPhysCol(buffered_blocks_[0], phys_col_info_[static_cast<size_t>(k)]);
                auto col = src.cloneEmpty();
                timings_.clone_empty_ns += elapsedNs(t_clone);
                ++timings_.alloc_count;

                const auto t_reserve = std::chrono::steady_clock::now();
                ptr = phys_prims_[static_cast<size_t>(k)].resize_for_scatter(*col, static_cast<size_t>(cnt));
                timings_.reserve_resize_ns += elapsedNs(t_reserve);

                const auto t_move = std::chrono::steady_clock::now();
                mcols[static_cast<size_t>(k)] = std::move(col);
                timings_.move_into_pending_ns += elapsedNs(t_move);
            }

            const auto t_on_grow = std::chrono::steady_clock::now();
            phys_prims_[static_cast<size_t>(k)].on_grow_raw(
                phys_prims_[static_cast<size_t>(k)],
                static_cast<size_t>(p),
                ptr,
                static_cast<size_t>(cnt),
                scatter_states_[static_cast<size_t>(k)]);
            timings_.on_grow_ns += elapsedNs(t_on_grow);
        }
    }

    // ── Phase 2: block-major scatter ──────────────────────────────────────
    const auto t_scatter = std::chrono::steady_clock::now();
    if (use_swwc_)
    {
        size_t pid_offset = 0;
        for (size_t bi = 0; bi < buffered_blocks_.size(); ++bi)
        {
            const DB::Columns & block = buffered_blocks_[bi];
            const int n = static_cast<int>(block[0]->size());
            const uint32_t * pids = buffered_pids_.data() + pid_offset;
            pid_offset += static_cast<size_t>(n);

            if (pos_.size() < static_cast<size_t>(n))
                pos_.resize(static_cast<size_t>(n));

            for (int j = 0; j < n; ++j)
            {
                const uint32_t p = pids[static_cast<size_t>(j)];
                pos_[static_cast<size_t>(j)] = cnt_[static_cast<size_t>(p)];
                ++cnt_[static_cast<size_t>(p)];
            }

            for (int k = 0; k < num_physical_columns_; ++k)
            {
                const IColumn & sub_col = extractPhysCol(block, phys_col_info_[static_cast<size_t>(k)]);
                auto & prim = phys_prims_[static_cast<size_t>(k)];
                if (prim.scatter_raw_swwc)
                    prim.scatter_raw_swwc(sub_col, 0, pids, pos_.data(), n, scatter_states_[static_cast<size_t>(k)]);
                else
                    prim.scatter_raw(sub_col, 0, pids, n, scatter_states_[static_cast<size_t>(k)]);
            }
        }

        timings_.scatter_ns += elapsedNs(t_scatter);
        const auto t_drain = std::chrono::steady_clock::now();
        std::atomic_thread_fence(std::memory_order_seq_cst);

        for (int p = 0; p < num_partitions_; ++p)
        {
            if (!cnt_[static_cast<size_t>(p)])
                continue;
            for (int k = 0; k < num_physical_columns_; ++k)
                if (phys_prims_[static_cast<size_t>(k)].drain_raw)
                    phys_prims_[static_cast<size_t>(k)].drain_raw(
                        phys_prims_[static_cast<size_t>(k)],
                        static_cast<size_t>(p),
                        cnt_[static_cast<size_t>(p)],
                        scatter_states_[static_cast<size_t>(k)]);
            cnt_[static_cast<size_t>(p)] = 0;
        }
        timings_.fence_drain_ns += elapsedNs(t_drain);
    }
    else
    {
        size_t pid_offset = 0;
        for (size_t bi = 0; bi < buffered_blocks_.size(); ++bi)
        {
            const DB::Columns & block = buffered_blocks_[bi];
            const int n = static_cast<int>(block[0]->size());
            const uint32_t * pids = buffered_pids_.data() + pid_offset;
            pid_offset += static_cast<size_t>(n);

            for (int k = 0; k < num_physical_columns_; ++k)
            {
                const IColumn & sub_col = extractPhysCol(block, phys_col_info_[static_cast<size_t>(k)]);
                phys_prims_[static_cast<size_t>(k)].scatter_raw(sub_col, 0, pids, n, scatter_states_[static_cast<size_t>(k)]);
            }
        }
        timings_.scatter_ns += elapsedNs(t_scatter);
    }

    // ── Phase 3: commit pending columns into output_ ──────────────────────
    // Skipped for aligned_alloc backend (no IColumn to commit).
    if (!use_aligned_alloc_)
    {
        const auto t = std::chrono::steady_clock::now();
        for (int p = 0; p < num_partitions_; ++p)
        {
            if (!accum_hist_[static_cast<size_t>(p)])
                continue;

            MutableColumns & mcols = pending_cols_[static_cast<size_t>(p)];
            DB::Columns frozen;
            frozen.reserve(mcols.size());
            for (auto & mc : mcols)
                frozen.push_back(std::move(mc));
            mcols.clear();
            output_[static_cast<size_t>(p)].push_back(std::move(frozen));
        }
        timings_.commit_ns += elapsedNs(t);
    }

    {
        const auto t = std::chrono::steady_clock::now();
        buffered_blocks_.clear();
        buffered_pids_.clear();
        total_buffered_bytes_ = 0;
        std::fill(accum_hist_.begin(), accum_hist_.end(), 0);
        timings_.reset_ns += elapsedNs(t);
    }

    timings_.total_flush_ns += elapsedNs(flush_t0);
}


void BatchedRadixShuffler::finish()
{
    flush();
}

} // namespace DB
