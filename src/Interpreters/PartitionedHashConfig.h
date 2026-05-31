#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <base/types.h>

namespace DB
{

/** Partition configuration and routing for PartitionedHashJoin (RFC #106023, spec rev 3 §4.5/§5).
  *
  * Routing is power-of-2 bit-slicing of a single 32-bit `IColumn::computeHashInto` value, top-bits-first.
  * Each pass consumes a disjoint contiguous bit window of the one hash:
  *
  *     shift_i = HASH_BITS - sum_{k<=i} pass_bits[k]
  *     pid_i   = (h >> shift_i) & ((1 << pass_bits[i]) - 1)
  *     leaf    = h >> (HASH_BITS - total_bits)          // == the top total_bits bits
  *
  * Build and probe MUST use the identical hash and pass schedule (spec invariant #6), otherwise matches
  * are silently lost. Nothing is carried between passes: refinement re-derives `h` from the buffered key.
  *
  * All functions here are pure (the only impurity is the cached L2 size read); they are unit-tested in
  * tests/gtest_partitioned_hash_config.cpp without any join existing.
  */
struct PartitionConfig
{
    /// Hash width produced by IColumn::computeHashInto.
    static constexpr UInt32 HASH_BITS = 32;

    /// Bits consumed per pass; each <= bits_per_pass cap (6 for max_partitions_per_pass=64).
    /// Empty (or all-zero) means a single leaf (no partitioning).
    std::vector<UInt8> pass_bits;
    /// == 2 ^ totalBits(). Power of two.
    size_t total_leaves = 1;

    [[nodiscard]] UInt8 totalBits() const noexcept
    {
        UInt8 sum = 0;
        for (auto b : pass_bits)
            sum = static_cast<UInt8>(sum + b);
        return sum;
    }

    [[nodiscard]] size_t numPasses() const noexcept { return pass_bits.size(); }

    /// Low-bit shift for `pass_index` (top-bits-first): number of low bits to skip before the window.
    [[nodiscard]] UInt32 shiftForPass(size_t pass_index) const noexcept
    {
        UInt8 consumed = 0;
        for (size_t i = 0; i <= pass_index && i < pass_bits.size(); ++i)
            consumed = static_cast<UInt8>(consumed + pass_bits[i]);
        return HASH_BITS - consumed;
    }

    /// Partition id selected by `pass_index` from hash `h`.
    [[nodiscard]] UInt32 pidForPass(UInt32 h, size_t pass_index) const noexcept
    {
        const UInt8 bits = pass_bits[pass_index];
        if (bits == 0)
            return 0;
        return (h >> shiftForPass(pass_index)) & ((UInt32{1} << bits) - 1);
    }

    /// Final leaf index in [0, total_leaves) for hash `h` (== the top totalBits() bits).
    [[nodiscard]] size_t leafForHash(UInt32 h) const noexcept
    {
        const UInt8 bits = totalBits();
        if (bits == 0)
            return 0;
        return static_cast<size_t>(h >> (HASH_BITS - bits));
    }
};

/// Inputs to leaf-count derivation. Byte widths are per row.
struct PartitionConfigInputs
{
    /// Right-side ROW-count estimate from the planner (`right_rows_estimation` / `hint->source_rows`,
    /// the same value compared against `parallel_hash_join_threshold`). When absent, the default leaf
    /// count is used (spec §5.2).
    std::optional<UInt64> rhs_rows_estimation;
    /// Per-row width of the join key column(s).
    size_t key_bytes = 8;
    /// Per-row width of the SELECTED right payload columns (post right_pre_join_actions).
    size_t payload_bytes = 0;
    /// Per-row leaf-HT cell overhead = sizeof(key)+sizeof(RowRef) at load factor (~48 B for UInt64).
    size_t cell_bytes = 48;
    /// Private per-core L2 bytes (0 => discover at runtime / fall back to 256 KiB).
    size_t l2_bytes = 0;
    /// Per-pass fanout cap (max_partitions_per_pass setting; 64 default).
    size_t max_partitions_per_pass = 64;
};

/// Default leaf count when the right-side estimate is absent (spec §5.2/§7.3 -> factorises to 4,4).
inline constexpr size_t PHJ_DEFAULT_LEAVES = 256;
/// Cap on leaves (Sum bits <= 16 for v1); guards an over-estimated rhs_size_estimation.
inline constexpr size_t PHJ_MAX_LEAVES = 1 << 16;
/// L2 headroom factor for leaf sizing (spec §7.3).
inline constexpr double PHJ_L2_HEADROOM = 0.8;
/// Fallback L2 size when discovery is unavailable (mirrors getMinBytesForPrefetchInJoin).
inline constexpr size_t PHJ_L2_FALLBACK_BYTES = 256 * 1024;

/// Round up to the next power of two (>= 1).
size_t roundUpPow2(size_t x) noexcept;

/// Private per-core L2 size in bytes (sysconf(_SC_LEVEL2_CACHE_SIZE)); cached. Falls back to 256 KiB.
size_t getPrivateL2Bytes();

/// Factorise a power-of-two leaf count into an even <= bits_per_pass bit schedule (spec §5.3).
/// E.g. 8192 (13 bits), cap 6 -> {5,4,4}; 4096 (12 bits) -> {6,6}; 256 (8 bits) -> {4,4}.
std::vector<UInt8> factorisePassBits(size_t total_leaves, size_t max_partitions_per_pass) noexcept;

/// Derive the full PartitionConfig (leaf count + pass schedule) from real build bytes and private L2
/// (spec §5.2/§5.3). When rhs_size_estimation_bytes is absent, uses PHJ_DEFAULT_LEAVES.
PartitionConfig derivePartitionConfig(const PartitionConfigInputs & inputs);

}
