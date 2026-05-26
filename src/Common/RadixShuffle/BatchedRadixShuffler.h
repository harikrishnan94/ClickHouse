#pragma once

#include <Columns/IColumn_fwd.h>
#include <Common/RadixShuffle/BumpArena.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/OutBlock.h>
#include <Common/RadixShuffle/PartitionTypes.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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


/// Batched radix partition operator.
///
/// Like `RadixShuffler`, but buffers input blocks until either
/// `max_buffered_blocks` blocks accumulate or `max_buffered_bytes` are buffered,
/// then allocates exact-size `OutBlock`s per partition and scatters all buffered
/// rows in one sweep.
///
/// Column data buffers are allocated independently via `std::aligned_alloc`
/// (64-byte aligned) so each column of each partition occupies its own
/// allocation — no inter-column or inter-partition sharing of cache lines or
/// pages.  The `OutBlock` header itself comes from the `BumpArena`.
/// All column allocations are owned by this object and freed in the destructor.
class BatchedRadixShuffler
{
public:
    static constexpr size_t kDefaultMemBound = 32ULL << 20; ///< 32 MiB

    static bool shouldUseSwwc(int K, int P) noexcept { return (K == 1) ? (P >= 512) : (P >= 32); }

    /// `max_buffered_blocks` — flush when this many blocks are buffered.
    ///   Pass 0 to use `P` (number of partitions).
    /// `max_buffered_bytes` — flush when buffered byte volume reaches this limit.
    ///   Pass 0 to use `kDefaultMemBound`.
    BatchedRadixShuffler(
        int P,
        int K,
        std::vector<ColumnPrimitives> prims,
        BumpArena & arena,
        bool use_swwc,
        size_t init_cap = kOutCapMin,
        size_t max_cap = kOutCapMax,
        size_t max_buffered_blocks = 0,
        size_t max_buffered_bytes = 0);

    /// Frees all per-column buffers allocated via `std::aligned_alloc` in flush().
    ~BatchedRadixShuffler();

    void process(const DB::Columns & columns);
    void finish();

    [[nodiscard]] std::vector<PartState> & parts() noexcept { return parts_; }
    [[nodiscard]] const std::vector<PartState> & parts() const noexcept { return parts_; }

    [[nodiscard]] size_t maxBufferedBlocks() const noexcept { return max_buffered_blocks_; }
    [[nodiscard]] size_t maxBufferedBytes() const noexcept { return max_buffered_bytes_; }

private:
    void flush();

    int num_partitions_;
    int num_columns_;
    int num_physical_columns_;
    bool use_swwc_;
    uint32_t mask_;
    size_t init_cap_;
    size_t max_buffered_blocks_;
    size_t max_buffered_bytes_;
    size_t bytes_per_row_;

    std::vector<ColumnPrimitives> col_prims_;
    std::vector<ColumnPrimitives> phys_prims_;
    std::vector<BatchedPhysColInfo> phys_col_info_;
    std::vector<ScatterState> scatter_states_;
    std::vector<PartState> parts_;
    BumpArena & arena_;

    std::vector<size_t> elem_sizes_;

    std::vector<uint32_t> accum_hist_;

    std::vector<DB::Columns> buffered_blocks_;
    /// Flat array of partition IDs for all buffered blocks, laid out contiguously.
    /// Cleared (not freed) after each flush so capacity is reused across flushes.
    std::vector<uint32_t> buffered_pids_;
    size_t total_buffered_bytes_ = 0;

    std::vector<uint32_t> pos_;
    std::vector<uint8_t> cnt_;

    /// Per-column buffers allocated via `std::aligned_alloc` during flush().
    /// Each entry corresponds to one `OutBlock::cols[k]` pointer and is freed
    /// by the destructor.
    ///
    /// Lifetime caveat: once `parts()` is moved out of this object, the
    /// `OutBlock::cols[k]` pointers inside remain valid until `~BatchedRadixShuffler`
    /// fires.  Callers must not access column data after the operator is destroyed.
    std::vector<void *> col_allocs_;
};

} // namespace DB
