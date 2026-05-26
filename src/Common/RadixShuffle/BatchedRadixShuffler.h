#pragma once

#include <Columns/IColumn.h>
#include <Columns/IColumn_fwd.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/PartitionTypes.h>

#include <cstddef>
#include <cstdint>
#include <vector>


namespace DB
{

/// Describes how to extract a physical sub-column from a logical column.
struct BatchedPhysColInfo
{
    size_t logical_k;
    bool use_null_map;
    bool use_nested;
};

/// Per-operation timing counters accumulated over the lifetime of one
/// `BatchedRadixShuffler` instance.  All values are nanoseconds.
struct BatchedTimings
{
    // process()
    uint64_t pid_compute_ns = 0;
    uint64_t histogram_ns = 0;
    uint64_t buffer_push_ns = 0;

    // flush() Phase 1 — IColumn allocation, broken down
    uint64_t clone_empty_ns = 0;
    uint64_t reserve_resize_ns = 0;
    uint64_t on_grow_ns = 0;
    uint64_t move_into_pending_ns = 0;

    // flush() Phase 2 — scatter
    uint64_t scatter_ns = 0;
    uint64_t fence_drain_ns = 0;

    // flush() Phase 3 — commit + reset
    uint64_t commit_ns = 0;
    uint64_t reset_ns = 0;

    // totals + counters
    uint64_t total_process_ns = 0;
    uint64_t total_flush_ns = 0;
    size_t flush_count = 0;
    size_t rows_processed = 0;
    size_t alloc_count = 0; // total cloneEmpty calls across all flushes
};


/// Batched radix partition operator.
///
/// Like `RadixShuffler`, but buffers input blocks until either
/// `max_buffered_blocks` blocks accumulate or `max_buffered_bytes` are buffered,
/// then allocates exact-size per-partition `IColumn` output columns and scatters
/// all buffered rows in one sweep.
///
/// Output is `output_[p]` — a vector of `Columns` blocks for partition `p`,
/// one entry per flush cycle.  Each entry holds `num_physical_columns_` columns
/// of exactly `accum_hist_[p]` rows.
class BatchedRadixShuffler
{
public:
    static constexpr size_t kDefaultMemBound = 32ULL << 20; ///< 32 MiB

    static bool shouldUseSwwc(int K, int P) noexcept { return (K == 1) ? (P >= 512) : (P >= 32); }

    /// `max_buffered_blocks` — flush when this many blocks are buffered.
    ///   Pass 0 to use `P` (number of partitions).
    /// `max_buffered_bytes` — flush when buffered byte volume reaches this limit.
    ///   Pass 0 to use `kDefaultMemBound`.
    /// `use_aligned_alloc` — when true, output buffers are allocated via
    ///   `std::aligned_alloc(64, …)` instead of `cloneEmpty + reserve_resize`.
    ///   No `IColumn` wrapping is produced; `output()` stays empty.  Used for
    ///   apples-to-apples comparison against the historical OutBlock variant.
    BatchedRadixShuffler(
        int P,
        int K,
        std::vector<ColumnPrimitives> prims,
        bool use_swwc,
        size_t max_buffered_blocks = 0,
        size_t max_buffered_bytes = 0,
        bool use_aligned_alloc = false);

    ~BatchedRadixShuffler();

    void process(const DB::Columns & columns);
    void finish();

    /// Per-partition output.  `output()[p]` is a list of `Columns` blocks, one
    /// per flush cycle; each block holds `num_physical_columns_` columns.
    [[nodiscard]] std::vector<std::vector<DB::Columns>> & output() noexcept { return output_; }
    [[nodiscard]] const std::vector<std::vector<DB::Columns>> & output() const noexcept { return output_; }

    [[nodiscard]] size_t maxBufferedBlocks() const noexcept { return max_buffered_blocks_; }
    [[nodiscard]] size_t maxBufferedBytes() const noexcept { return max_buffered_bytes_; }

    /// Accumulated per-operation timings (valid after finish()).
    [[nodiscard]] const BatchedTimings & timings() const noexcept { return timings_; }

private:
    void flush();

    int num_partitions_;
    int num_columns_;
    int num_physical_columns_;
    bool use_swwc_;
    bool use_aligned_alloc_;
    uint32_t mask_;
    size_t max_buffered_blocks_;
    size_t max_buffered_bytes_;
    size_t bytes_per_row_;

    std::vector<ColumnPrimitives> col_prims_;
    std::vector<ColumnPrimitives> phys_prims_;
    std::vector<BatchedPhysColInfo> phys_col_info_;
    std::vector<ScatterState> scatter_states_;

    /// output_[p] accumulates one Columns block per flush cycle.
    std::vector<std::vector<DB::Columns>> output_;

    std::vector<uint32_t> accum_hist_;

    std::vector<DB::Columns> buffered_blocks_;
    /// Flat array of partition IDs for all buffered blocks, laid out contiguously.
    /// Cleared (not freed) after each flush so capacity is reused across flushes.
    std::vector<uint32_t> buffered_pids_;
    size_t total_buffered_bytes_ = 0;

    std::vector<uint32_t> pos_;
    std::vector<uint8_t> cnt_;

    /// Scratch: per-partition MutableColumn arrays built in flush() Phase 1,
    /// consumed in Phase 2, then moved into output_ at the end of flush().
    std::vector<MutableColumns> pending_cols_;

    /// When `use_aligned_alloc_` is true, all `std::aligned_alloc`'d output
    /// buffers go here so the destructor can free them.
    std::vector<void *> aligned_allocs_;

    BatchedTimings timings_;
};

} // namespace DB
