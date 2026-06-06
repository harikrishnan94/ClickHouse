#pragma once

#include <base/types.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace DB::RadixHash
{

/** Partition configuration for the radix hash join (spec sections 5.2, 5.3, 4.5).
  *
  * `num_leaves` is the number of per-leaf hash tables, sized so each `16 B`-cell table fits
  * `0.8 x L2`. `total_bits = log2(num_leaves)` are the top hash bits used to route a row to its
  * leaf (`leaf_id = hash >> shift`, where `shift = HASH_BITS - total_bits`). The total bits are
  * split into the minimum number of scatter passes that respect the SWWC per-pass fanout cap
  * (`bits_per_pass`, derived from `max_partitions_per_pass`), spread as evenly as possible
  * (`max(pass_bits) - min(pass_bits) <= 1`).
  */
struct PartitionConfig
{
    static constexpr UInt32 HASH_BITS = 32;              /// IColumn::computeHashInto width (spec D2)
    static constexpr size_t MAX_LEAVES = 1048576;        /// 2^20 leaf upper bound, power of two (spec section 4.5)
    static constexpr size_t CELL_BYTES = 16;             /// key 8 + BuildRef 8 (spec section 5.4)
    static constexpr double LOAD_FACTOR = 0.5;           /// open addressing target -> ~32 B/row
    static constexpr double L2_HEADROOM = 0.8;           /// leaf table sized to 0.8 x L2
    static constexpr size_t DEFAULT_LEAVES = 256;        /// when the row estimate is absent (spec section 5.2)
    static constexpr size_t L2_FALLBACK_BYTES = 256 * 1024; /// mirrors getMinBytesForPrefetchInJoin

    size_t num_leaves = DEFAULT_LEAVES;
    UInt32 total_bits = 8;                               /// log2(num_leaves)
    UInt32 shift = HASH_BITS - 8;                        /// leaf_id = (hash >> shift) & (num_leaves - 1)
    std::vector<UInt32> pass_bits = {8};                 /// sums to total_bits; max - min <= 1

    /// `l2_bytes`: private per-core L2 size (sysconf); 0 -> use L2_FALLBACK_BYTES.
    /// `max_partitions_per_pass`: SWWC fanout cap; bits_per_pass = floor(log2(cap)).
    static PartitionConfig make(
        std::optional<UInt64> rhs_rows_estimation,
        size_t l2_bytes,
        UInt64 max_partitions_per_pass);
};

/// Smallest power of two >= x (x >= 1). Returns 1 for x == 0.
size_t roundUpToPowerOfTwo(size_t x);

}
