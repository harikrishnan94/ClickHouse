#include <Common/RadixShuffle/BatchedRadixShuffler.h>

#include <Columns/ColumnNullable.h>
#include <Columns/IColumn.h>
#include <base/types.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>


namespace DB
{

BatchedRadixShuffler::BatchedRadixShuffler(
    int P,
    int K,
    std::vector<ColumnPrimitives> prims,
    BumpArena & arena,
    bool use_swwc,
    size_t init_cap,
    size_t /*max_cap*/,
    size_t max_buffered_blocks,
    size_t max_buffered_bytes)
    : num_partitions_(P)
    , num_columns_(K)
    , use_swwc_(use_swwc)
    , mask_(static_cast<uint32_t>(P) - 1)
    , init_cap_(init_cap)
    , max_buffered_blocks_(max_buffered_blocks ? max_buffered_blocks : static_cast<size_t>(P))
    , max_buffered_bytes_(max_buffered_bytes ? max_buffered_bytes : kDefaultMemBound)
    , col_prims_(std::move(prims))
    , arena_(arena)
    , accum_hist_(static_cast<size_t>(P), 0)
    , cnt_(static_cast<size_t>(P), 0)
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

    elem_sizes_.resize(static_cast<size_t>(num_physical_columns_));
    bytes_per_row_ = 0;
    for (int k = 0; k < num_physical_columns_; ++k)
    {
        elem_sizes_[static_cast<size_t>(k)] = phys_prims_[static_cast<size_t>(k)].raw_elem_size;
        bytes_per_row_ += elem_sizes_[static_cast<size_t>(k)];
    }

    parts_.assign(static_cast<size_t>(P), {});
    for (auto & ps : parts_)
        ps.next_cap = init_cap_;
}


BatchedRadixShuffler::~BatchedRadixShuffler()
{
    for (void * p : col_allocs_)
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


/// Allocate an `OutBlock` whose column data buffers are independent
/// 64-byte-aligned heap allocations (one per physical column).
///
/// The `OutBlock` header itself comes from `arena` (a small bump allocation).
/// Each column buffer is appended to `col_allocs` so the caller can free them
/// later.  `cap` must already be a multiple of 64 (use `round64`) because
/// `std::aligned_alloc` requires size to be a multiple of alignment.
static OutBlock * allocOutBlockSeparateCols(
    BumpArena & arena, int num_physical_columns, const size_t * elem_sizes, size_t cap, std::vector<void *> & col_allocs)
{
    constexpr size_t kHdrSize = (sizeof(OutBlock) + 63) & ~size_t{63};

    char * raw = arena.alignedAlloc(kHdrSize, 64);
    auto * nb = reinterpret_cast<OutBlock *>(raw);
    nb->next = nullptr;
    nb->filled = 0;
    nb->capacity = cap;

    for (int k = 0; k < num_physical_columns; ++k)
    {
        void * col_mem = std::aligned_alloc(64, cap * elem_sizes[static_cast<size_t>(k)]);
        if (!col_mem)
            throw std::bad_alloc{};
        nb->cols[k] = col_mem;
        col_allocs.push_back(col_mem);
    }
    for (int k = num_physical_columns; k < kMaxK; ++k)
        nb->cols[k] = nullptr;

    return nb;
}


void BatchedRadixShuffler::process(const DB::Columns & columns)
{
    if (columns.empty() || columns[0]->size() == 0)
        return;

    const size_t n = columns[0]->size();
    const size_t pid_offset = buffered_pids_.size();
    buffered_pids_.resize(pid_offset + n);

    col_prims_[0].compute_pids(col_prims_[0], *columns[0], 0, static_cast<int>(n), mask_, buffered_pids_.data() + pid_offset);

    for (size_t j = 0; j < n; ++j)
        ++accum_hist_[buffered_pids_[pid_offset + j]];

    buffered_blocks_.push_back(columns);
    total_buffered_bytes_ += n * bytes_per_row_;

    if (buffered_blocks_.size() >= max_buffered_blocks_ || total_buffered_bytes_ >= max_buffered_bytes_)
        flush();
}


void BatchedRadixShuffler::flush()
{
    if (buffered_blocks_.empty())
        return;

    // ── Phase 1: exact-size OutBlocks per active partition ────────────────
    for (int p = 0; p < num_partitions_; ++p)
    {
        const uint32_t cnt = accum_hist_[static_cast<size_t>(p)];
        if (!cnt)
            continue;

        auto & ps = parts_[static_cast<size_t>(p)];
        const size_t cap = round64(cnt); // multiple of 64 — satisfies aligned_alloc requirement

        OutBlock * nb = allocOutBlockSeparateCols(arena_, num_physical_columns_, elem_sizes_.data(), cap, col_allocs_);
        nb->next = ps.head;
        ps.head = ps.cur = nb;
        nb->filled = cnt;

        for (int k = 0; k < num_physical_columns_; ++k)
            phys_prims_[static_cast<size_t>(k)].on_grow_raw(
                phys_prims_[static_cast<size_t>(k)],
                static_cast<size_t>(p),
                ps.cur->cols[k],
                ps.cur->capacity,
                scatter_states_[static_cast<size_t>(k)]);
    }

    // ── Phase 2: block-major scatter ──────────────────────────────────────
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
    }

    // Reset buffers and histogram.
    buffered_blocks_.clear();
    buffered_pids_.clear();
    total_buffered_bytes_ = 0;
    std::fill(accum_hist_.begin(), accum_hist_.end(), 0);
}


void BatchedRadixShuffler::finish()
{
    flush();
}

} // namespace DB
