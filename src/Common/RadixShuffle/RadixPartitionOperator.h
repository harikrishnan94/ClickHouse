#pragma once

#include <Columns/IColumn_fwd.h>
#include <Common/RadixShuffle/Allocator.h>
#include <Common/RadixShuffle/IScatterColumn.h>
#include <Common/RadixShuffle/PartSchema.h>
#include <Common/RadixShuffle/PartitionTypes.h>

#include <cstddef>
#include <cstdint>
#include <vector>


namespace DB::RadixShuffle
{

/// Single-pass radix partition operator, templated over the key element type.
///
/// Port of `PartitionOperatorV` from `radix_part_vs_memcpy.cpp`, generalized
/// from `uint64_t` to any fixed-width POD `TKey`.
///
/// The operator auto-selects between two scatter strategies:
///   - direct   — live-pointer per-partition scatter; best for small P.
///   - SWWC     — 8-slot staging buffer, flushed via NT stores on AVX-512 v4;
///                best for large P where output is cold.
///
/// Crossover: `K==1 → SWWC when P≥512`; `K≥2 → SWWC when P≥32`.
///
/// Lifecycle:
///   1. Construct with P, K, columns, SWWC flag.
///   2. Call `process(columns)` once per input block.
///   3. Call `finish()` after all blocks to flush SWWC buffers.
///   4. Read allocator statistics through `getAllocator()`.
template <typename TKey>
class RadixPartitionOperator
{
public:
    /// Batch size: max(1024, min(32768, P × 16)) — matches the reference.
    static constexpr int kBatchFactor = 16;
    static constexpr int kSmartMaxBatch = 32768;
    static constexpr int kSimdWidth = 8; ///< uint64 lanes per AVX-512 ZMM register.

    /// True iff SWWC NT-store scatter is preferred for (K, P).
    static bool should_use_swwc(int K, int P) noexcept { return (K == 1) ? (P >= 512) : (P >= 32); }

    /// `cols`     — K column objects; ownership not transferred.
    /// `use_swwc` — select SWWC scatter path; use `should_use_swwc(K,P)` as hint.
    RadixPartitionOperator(int P, int K, std::vector<IScatterColumn *> cols, bool use_swwc);
    ~RadixPartitionOperator();

    /// Process one input block.  `columns[k]` must be a `ColumnVector<TKey>`.
    /// Call repeatedly for streaming input, then call `finish()`.
    void process(const DB::Columns & columns);

    /// Flush SWWC staging buffers and issue `sfence`.
    /// Must be called once after all `process()` calls before reading `parts()`.
    void finish();

    [[nodiscard]] const Allocator & getAllocator() const noexcept { return allocator_; }
    [[nodiscard]] int batchSize() const noexcept { return batch_; }

private:
    void runBatch(const DB::Columns & columns, size_t start, int n);

    int P_;
    int K_;
    bool use_swwc_;
    int batch_;
    uint32_t mask_; ///< P − 1 (P must be a power of two).
    Allocator allocator_;
    Handle * handle_ = nullptr;

    std::vector<IScatterColumn *> cols_;

    /// Per-batch scratch arrays (size == batch_).
    std::vector<uint32_t> pids_;
    std::vector<uint32_t> hist_;
    std::vector<size_t> size_hist_;
    std::vector<size_t> varlen_zeros_;
    std::vector<PartReserveGrant> grants_;
    std::vector<uint64_t> stale_bitset_;
    std::vector<uint32_t> pos_; ///< SWWC staging slot per row.
    std::vector<uint8_t> cnt_; ///< SWWC staging slot counter per partition (0..7).

    uint64_t debug_rows_ = 0;
    uint64_t debug_batches_ = 0;
    uint64_t debug_hash_ns_ = 0;
    uint64_t debug_hist_ns_ = 0;
    uint64_t debug_reserve_ns_ = 0;
    uint64_t debug_stale_ns_ = 0;
    uint64_t debug_scatter_ns_ = 0;
    uint64_t debug_stale_partitions_ = 0;
    uint64_t debug_drain_events_ = 0;
};

/// Explicit instantiation declarations.
extern template class RadixPartitionOperator<uint8_t>;
extern template class RadixPartitionOperator<uint16_t>;
extern template class RadixPartitionOperator<uint32_t>;
extern template class RadixPartitionOperator<uint64_t>;
extern template class RadixPartitionOperator<int8_t>;
extern template class RadixPartitionOperator<int16_t>;
extern template class RadixPartitionOperator<int32_t>;
extern template class RadixPartitionOperator<int64_t>;
extern template class RadixPartitionOperator<float>;
extern template class RadixPartitionOperator<double>;

} // namespace DB::RadixShuffle
