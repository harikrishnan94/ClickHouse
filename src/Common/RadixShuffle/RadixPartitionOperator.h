#pragma once

#include <Columns/IColumn_fwd.h>
#include <Common/RadixShuffle/BumpArena.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/OutBlock.h>
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
///   1. Construct with P, K, columns, arena, SWWC flag, capacity hints.
///   2. Call `process(columns)` once per input block.
///   3. Call `finish()` after all blocks to flush SWWC buffers.
///   4. Read `parts()` for per-partition output state.
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

    /// `prims`    — K `ColumnPrimitives` objects (must have `scatter_raw`,
    ///              `scatter_raw_swwc`, `drain_raw`, and `on_grow_raw` set).
    /// `arena`    — bump allocator for OutBlock storage; must outlive this object.
    /// `use_swwc` — select SWWC scatter path; use `should_use_swwc(K,P)` as hint.
    /// `init_cap` — initial OutBlock row capacity (must be a multiple of 8).
    /// `max_cap`  — maximum OutBlock row capacity (must be a multiple of 8).
    RadixPartitionOperator(
        int P,
        int K,
        std::vector<ColumnPrimitives> prims,
        BumpArena & arena,
        bool use_swwc,
        size_t init_cap = kOutCapMin,
        size_t max_cap = kOutCapMax);

    /// Process one input block.  `columns[k]` must be a `ColumnVector<TKey>`.
    /// Call repeatedly for streaming input, then call `finish()`.
    void process(const DB::Columns & columns);

    /// Flush SWWC staging buffers and issue `sfence`.
    /// Must be called once after all `process()` calls before reading `parts()`.
    void finish();

    /// Access per-partition output state (call after `finish()`).
    [[nodiscard]] std::vector<PartState> & parts() noexcept { return parts_; }
    [[nodiscard]] const std::vector<PartState> & parts() const noexcept { return parts_; }

    [[nodiscard]] int batchSize() const noexcept { return batch_; }

private:
    void runBatch(const DB::Columns & columns, size_t start, int n);

    int P_;
    int K_;
    bool use_swwc_;
    int batch_;
    uint32_t mask_; ///< P − 1 (P must be a power of two).
    size_t max_cap_;

    std::vector<ColumnPrimitives> col_prims_;
    std::vector<ScatterState> scatter_states_;
    std::vector<PartState> parts_;
    BumpArena & arena_;

    /// Per-column raw element sizes for OutBlock allocation.
    /// elem_sizes_[k] = col_prims_[k].raw_elem_size; set in constructor.
    std::vector<size_t> elem_sizes_;

    /// Per-batch scratch arrays (size == batch_).
    std::vector<uint32_t> pids_;
    std::vector<uint32_t> hist_;
    std::vector<uint32_t> pos_; ///< SWWC staging slot per row.
    std::vector<uint8_t> cnt_; ///< SWWC staging slot counter per partition (0..7).
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
