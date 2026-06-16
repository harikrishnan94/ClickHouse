#pragma once

#include <base/types.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace DB::RadixJoin
{

/** How the build is partitioned into leaves and how many radix passes the scatter takes.
  *
  * Sizing rationale (the whole point of the algorithm): the probe is a random lookup into a
  * per-leaf open-addressing table. If that table fits in the core-private L2 it stays hot and every
  * lookup is an L2 hit; once it exceeds LLC the lookup is a cold DRAM miss (this is exactly where a
  * single shared hash table — `parallel_hash` — loses). So we pick the smallest number of leaves
  * (a power of two) such that one leaf's table is at most `L2_HEADROOM x L2`. The table footprint is
  * estimated as `rhs_rows x CELL_BYTES / LOAD_FACTOR` (~32 B/row) and is INDEPENDENT of the selected
  * payload width, because payload is never stored in the leaf — only the key and an 8-byte build
  * reference are. Hence a 1-column and an 8-column projection of the same build side get the same
  * leaf count.
  *
  * `total_bits = log2(num_leaves)` top bits of the route word select the leaf. They are produced in
  * the minimum number of scatter passes whose per-pass fanout respects `max_partitions_per_pass`
  * (the SWWC staging budget), with the bits spread as evenly as possible so no pass is a tiny,
  * write-combining-wasting 2-way scatter.
  */
struct PartitionPlan
{
    /// Width of the route word (the full 32-bit packed-key `HashT`). The top `total_bits` of it select the leaf.
    static constexpr UInt32 ROUTE_BITS = 32;
    /// 2^20 leaf ceiling: routing only ever needs the high bits, and the leaf vector stays bounded.
    static constexpr size_t MAX_LEAVES = 1u << 20;
    /// Leaf-cell accounting used purely for sizing: an 8-byte key reference + ~8-byte key worth of
    /// open-addressing overhead, i.e. ~16 B occupied and ~32 B reserved at LOAD_FACTOR.
    static constexpr size_t CELL_BYTES = 16;
    static constexpr double LOAD_FACTOR = 0.5;
    static constexpr double L2_HEADROOM = 0.8;
    static constexpr size_t DEFAULT_LEAVES = 256; /// when the right-side row estimate is unavailable
    static constexpr size_t L2_FALLBACK_BYTES = 256 * 1024; /// mirrors getMinBytesForPrefetchInJoin

    size_t num_leaves = DEFAULT_LEAVES;
    UInt32 total_bits = 8;                  /// log2(num_leaves)
    UInt32 leaf_shift = ROUTE_BITS - 8;     /// leaf = (route >> leaf_shift) & (num_leaves - 1)
    std::vector<UInt32> pass_bits = {8};    /// Σ == total_bits; max - min <= 1

    /// `l2_bytes` is the private per-core L2 size (sysconf); 0 falls back to L2_FALLBACK_BYTES.
    /// `max_partitions_per_pass` caps each pass's fanout; per-pass bits = floor(log2(cap)).
    static PartitionPlan choose(
        std::optional<UInt64> rhs_rows_estimation,
        size_t l2_bytes,
        UInt64 max_partitions_per_pass);
};

/// Smallest power of two >= x (1 for x <= 1).
size_t ceilPowerOfTwo(size_t x);

}
