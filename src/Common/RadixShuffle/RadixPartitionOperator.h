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

/// Describes how to extract a physical sub-column from a logical column.
/// Used by `RadixPartitionOperator` when expanding Nullable columns into
/// separate null-map and value primitives.
struct PhysColInfo
{
    size_t logical_k;
    bool use_null_map; ///< Extract getNullMapColumn() from ColumnNullable
    bool use_nested; ///< Extract getNestedColumn() from ColumnNullable
};

/// Single-pass radix partition operator.
///
/// Port of `PartitionOperatorV` from `radix_part_vs_memcpy.cpp`.
/// Column types are fully described by the `ColumnPrimitives` objects passed
/// at construction — the operator itself is not templated on any element type.
///
/// The operator auto-selects between two scatter strategies:
///   - direct   — live-pointer per-partition scatter; best for small P.
///   - SWWC     — 8-slot staging buffer, flushed via NT stores on AVX-512 v4;
///                best for large P where output is cold.
///
/// Crossover: `K==1 → SWWC when P≥512`; `K≥2 → SWWC when P≥32`.
///
/// Nullable primitives are automatically decomposed into two physical leaf
/// primitives (null-map UInt8 + values), each using the standard SWWC path.
///
/// Lifecycle:
///   1. Construct with P, K, columns, arena, SWWC flag, capacity hints.
///   2. Call `process(columns)` once per input block.
///   3. Call `finish()` after all blocks to flush SWWC buffers.
///   4. Read `parts()` for per-partition output state.
class RadixPartitionOperator
{
public:
    /// Batch size: max(1024, min(32768, P × 16)) — matches the reference.
    static constexpr int kBatchFactor = 16;
    static constexpr int kSmartMaxBatch = 32768;
    static constexpr int kSimdWidth = 8; ///< uint64 lanes per AVX-512 ZMM register.

    /// True iff SWWC NT-store scatter is preferred for (K, P).
    static bool should_use_swwc(int K, int P) noexcept { return (K == 1) ? (P >= 512) : (P >= 32); }

    /// `prims`    — K logical `ColumnPrimitives` objects.  Nullable primitives
    ///              are automatically decomposed into two physical primitives:
    ///              makeFixedWidth<UInt8>() for the null map and the nested
    ///              leaf primitive for the values.
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

    /// Process one input block.
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
    int K_; ///< Logical column count (as provided by the caller).
    int K_phys_; ///< Physical column count; ≥ K_ when Nullable columns are expanded.
    bool use_swwc_;
    int batch_;
    uint32_t mask_; ///< P − 1 (P must be a power of two).
    size_t max_cap_;

    std::vector<ColumnPrimitives> col_prims_; ///< K_ logical prims (Phase 1: compute_pids only).
    std::vector<ColumnPrimitives> phys_prims_; ///< K_phys_ physical prims (Phase 3-4: scatter).
    std::vector<PhysColInfo> phys_col_info_; ///< K_phys_ sub-column extraction descriptors.
    std::vector<ScatterState> scatter_states_; ///< K_phys_ ScatterState objects.
    std::vector<PartState> parts_;
    BumpArena & arena_;

    /// Per-physical-column element sizes for OutBlock allocation.
    std::vector<size_t> elem_sizes_;

    /// Per-batch scratch arrays (size == batch_).
    std::vector<uint32_t> pids_;
    std::vector<uint32_t> hist_;
    std::vector<uint32_t> pos_; ///< Raw per-partition row counter snapshot per row.
    std::vector<uint8_t> cnt_; ///< Raw per-partition row counter; wraps at 256 (uint8_t natural).
};

} // namespace DB::RadixShuffle
