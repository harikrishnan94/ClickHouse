#pragma once

#include <Common/RadixShuffle/BumpArena.h>
#include <Common/RadixShuffle/IScatterColumn.h>
#include <Common/RadixShuffle/OutBlock.h>

#include <cstddef>
#include <cstdint>
#include <vector>


namespace DB::RadixShuffle
{

/// One fixed-size input block.  `cols[k]` is the k-th column array of `rows`
/// elements.  The key column used for partitioning is always `cols[0]`.
template <typename T>
struct InputBlock
{
    T * cols[kMaxK] = {};
    size_t rows = 0;
};


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
///   2. Call `process(blocks)`.
///   3. Read `parts()` for per-partition output state.
template <typename TKey>
class RadixPartitionOperator
{
public:
    /// Batch size: max(1024, min(32768, P × 16)) — matches the reference.
    static constexpr int kBatchFactor = 16;
    static constexpr int kSmartMaxBatch = 32768;
    static constexpr int kSimdWidth = 8; ///< uint64 lanes per AVX-512 ZMM register.

    /// True iff SWWC NT-store scatter is preferred for (K, P).
    static bool should_use_swwc(int K, int P) noexcept
    {
        return (K == 1) ? (P >= 512) : (P >= 32);
    }

    /// `cols`     — K column objects; ownership not transferred.
    /// `arena`    — bump allocator for OutBlock storage; must outlive this object.
    /// `use_swwc` — select SWWC scatter path; use `should_use_swwc(K,P)` as hint.
    /// `init_cap` — initial OutBlock row capacity (must be a multiple of 8).
    /// `max_cap`  — maximum OutBlock row capacity (must be a multiple of 8).
    RadixPartitionOperator(
        int P,
        int K,
        std::vector<IScatterColumn *> cols,
        BumpArena & arena,
        bool use_swwc,
        size_t init_cap = kOutCapMin,
        size_t max_cap = kOutCapMax);

    /// Process all input blocks.  Batch-slices them internally.
    void process(const std::vector<InputBlock<TKey>> & blocks);

    /// Access per-partition output state (call after `process`).
    [[nodiscard]] std::vector<PartState> & parts() noexcept { return parts_; }
    [[nodiscard]] const std::vector<PartState> & parts() const noexcept { return parts_; }

    [[nodiscard]] int batchSize() const noexcept { return batch_; }

private:
    void runBatch(const InputBlock<TKey> & blk, size_t start, int n);
    void finish();

    int P_;
    int K_;
    bool use_swwc_;
    int batch_;
    uint64_t mask_; ///< P − 1 (P must be a power of two).
    size_t elem_size_;
    size_t max_cap_;

    std::vector<IScatterColumn *> cols_;
    std::vector<PartState> parts_;
    BumpArena & arena_;

    /// Per-batch scratch arrays (size == batch_).
    std::vector<uint32_t> pids_;
    std::vector<uint32_t> hist_;
    std::vector<uint32_t> pos_; ///< SWWC staging slot per row.
    std::vector<uint8_t> cnt_;  ///< SWWC staging slot counter per partition (0..7).
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
