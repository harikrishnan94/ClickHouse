#pragma once

#include <Columns/IColumn.h>
#include <Columns/IColumn_fwd.h>
#include <Common/Allocator.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/PartitionTypes.h>

#include <cstddef>
#include <cstdint>
#include <vector>


namespace DB
{

/// Describes how to extract a physical sub-column from a logical column.
/// Mirrors `BatchedPhysColInfo` in `BatchedRadixShuffler` — duplicated to
/// keep this header self-contained.
struct PerBlockPhysColInfo
{
    size_t logical_k;
    bool use_null_map;
    bool use_nested;
};


/// Per-operation timings accumulated over the lifetime of one
/// `PerBlockArenaShuffler` instance.  All values are nanoseconds.
struct PerBlockTimings
{
    // process()
    uint64_t pid_compute_ns = 0;
    uint64_t histogram_ns = 0;
    uint64_t alloc_ns = 0;
    uint64_t on_grow_ns = 0;
    uint64_t scatter_ns = 0;

    // finish() — gather and copy into IColumns
    uint64_t finish_alloc_ns = 0;
    uint64_t finish_copy_ns = 0;

    // totals
    uint64_t total_process_ns = 0;
    uint64_t total_finish_ns = 0;

    // counters
    size_t blocks_processed = 0;
    size_t rows_processed = 0;
    size_t chunks_allocated = 0;
    size_t bytes_allocated = 0;
    size_t output_blocks_emitted = 0; ///< total across all partitions
};


/// Radix partition operator that allocates a single chunk per input block.
///
/// Design:
///   • One CH `Allocator` chunk per `process(columns)` call, sized exactly to
///     `N × Σ raw_elem_size[k]`, where N is the input block size.  No
///     geometric growth, no per-partition fragmentation.  Pages stay warm
///     across blocks if jemalloc returns the same arena slab for same-sized
///     allocations (typical for steady-state streams).
///   • Within the chunk: column-major, then partition-major.
///         col_0 [part_0[h0]] [part_1[h1]] ... [part_{P-1}[h_{P-1}]]
///         col_1 [part_0[h0]] [part_1[h1]] ... [part_{P-1}[h_{P-1}]]
///         ...
///     Each (partition,column) slice starts at offset
///         col_k_base + prefix_offset[p] × elem_size[k]
///     where prefix_offset is the exclusive prefix sum of the per-block
///     histogram.
///   • Direct (non-SWWC) scatter: `*wp[pids[j]]++ = src[j]` per column.  SWWC
///     would require padding each partition slice up to a 64-byte boundary,
///     which negates the "exact-size, no waste" property of the design.
///   • Output IColumns are allocated lazily in `finish()` by gathering the
///     per-block slices and packing them into IColumn output blocks sized to
///     the largest input block observed (`max_input_block_rows_`).
///
/// API contract: identical to `BatchedRadixShuffler` except the constructor
/// omits the `use_aligned_alloc` parameter (this operator never uses
/// `aligned_alloc`).
class PerBlockArenaShuffler
{
public:
    static bool shouldUseSwwc(int K, int P) noexcept { return (K == 1) ? (P >= 512) : (P >= 32); }

    /// `prims`    — K logical `ColumnPrimitives` objects.  Nullable primitives
    ///              are automatically decomposed into two physical leaf primitives:
    ///              makeFixedWidth<UInt8>() for the null map and the nested
    ///              leaf primitive for the values.
    /// `use_swwc` — accepted for API parity with `BatchedRadixShuffler`;
    ///              this operator currently always uses the direct-scatter
    ///              path because per-block exact-sized partition slices are
    ///              not 64-byte aligned in general.
    PerBlockArenaShuffler(int P, int K, std::vector<ColumnPrimitives> prims, bool use_swwc);

    ~PerBlockArenaShuffler();

    PerBlockArenaShuffler(const PerBlockArenaShuffler &) = delete;
    PerBlockArenaShuffler & operator=(const PerBlockArenaShuffler &) = delete;
    PerBlockArenaShuffler(PerBlockArenaShuffler &&) = delete;
    PerBlockArenaShuffler & operator=(PerBlockArenaShuffler &&) = delete;

    void process(const DB::Columns & columns);
    void finish();

    /// Per-partition output.  `output()[p]` is a list of `Columns` blocks, one
    /// per output cycle; each block holds `num_physical_columns_` columns.
    /// Block size is `max_input_block_rows_` rows (the final block per
    /// partition may be smaller).
    [[nodiscard]] std::vector<std::vector<DB::Columns>> & output() noexcept { return output_; }
    [[nodiscard]] const std::vector<std::vector<DB::Columns>> & output() const noexcept { return output_; }

    /// Accumulated per-operation timings (valid after finish()).
    [[nodiscard]] const PerBlockTimings & timings() const noexcept { return timings_; }

private:
    /// One per-input-block arena chunk.
    /// Layout: planar column-major within the chunk; each column's data is
    /// `raw_elem_size[k] × input_rows` bytes, partition-major inside.
    struct PerBlockChunk
    {
        void * data = nullptr;          ///< Allocator-owned base pointer.
        size_t bytes = 0;               ///< Allocation size for Allocator::free.
        size_t alignment = 0;           ///< Alignment used for the allocation.
        uint32_t input_rows = 0;        ///< Total rows = sum of histogram.
    };

    /// One per-(partition,block) slice descriptor.  Carries the chunk index
    /// plus the partition's row offset and count inside that block.
    struct PartSlice
    {
        uint32_t chunk_index;
        uint32_t row_offset;            ///< Partition's prefix-sum offset.
        uint32_t row_count;              ///< Histogram entry for this partition / block.
    };

    int num_partitions_;
    int num_columns_;
    int num_physical_columns_;
    uint32_t mask_;

    std::vector<ColumnPrimitives> col_prims_;
    std::vector<ColumnPrimitives> phys_prims_;
    std::vector<PerBlockPhysColInfo> phys_col_info_;
    std::vector<size_t> phys_elem_size_;     ///< raw_elem_size per physical column.

    std::vector<ScatterState> scatter_states_;

    /// Scratch reused across `process()` calls.
    std::vector<uint32_t> scratch_pids_;
    std::vector<uint32_t> scratch_hist_;
    std::vector<uint32_t> scratch_prefix_;

    /// Per-block arenas (chunk pool).
    std::vector<PerBlockChunk> chunks_;

    /// Per-partition list of (chunk_index, row_offset, row_count).
    /// `part_slices_[p][bi]` corresponds to chunk `bi` (one slice per block).
    std::vector<std::vector<PartSlice>> part_slices_;

    /// Aggregate row count per partition (sum of slice row_count).
    std::vector<size_t> part_total_rows_;

    /// Track the largest input block we saw.  All output IColumns are sized
    /// to this many rows (except the last per partition).
    size_t max_input_block_rows_ = 0;

    /// Source IColumns from the FIRST processed block — used as templates
    /// for `cloneEmpty()` in `finish()`.  We hold a refcount via SharedPtr.
    DB::Columns proto_columns_;

    /// Output: `output_[p]` is a list of `Columns` blocks for partition p.
    std::vector<std::vector<DB::Columns>> output_;

    PerBlockTimings timings_;

    Allocator<false, false> allocator_;
};

} // namespace DB
