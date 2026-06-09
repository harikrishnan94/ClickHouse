#include <Interpreters/RadixHashJoin/PartitionPlan.h>

#include <base/defines.h>

#include <algorithm>
#include <bit>
#include <cmath>

namespace DB::RadixJoin
{

size_t ceilPowerOfTwo(size_t x)
{
    if (x <= 1)
        return 1;
    return std::bit_ceil(x);
}

PartitionPlan PartitionPlan::choose(
    std::optional<UInt64> rhs_rows_estimation,
    size_t l2_bytes,
    UInt64 max_partitions_per_pass)
{
    PartitionPlan plan;

    size_t num_leaves = DEFAULT_LEAVES;
    if (rhs_rows_estimation.has_value())
    {
        /// ~32 B reserved per build row, independent of payload width (payload is not in the leaf).
        const double table_bytes = static_cast<double>(*rhs_rows_estimation) * (static_cast<double>(CELL_BYTES) / LOAD_FACTOR);
        const size_t l2 = l2_bytes != 0 ? l2_bytes : L2_FALLBACK_BYTES;
        const double usable_l2 = L2_HEADROOM * static_cast<double>(l2);

        const double leaves_needed = std::max(1.0, std::ceil(table_bytes / usable_l2));
        num_leaves = ceilPowerOfTwo(static_cast<size_t>(leaves_needed));
        num_leaves = std::clamp<size_t>(num_leaves, 1, MAX_LEAVES);
    }

    plan.num_leaves = num_leaves;
    chassert(std::has_single_bit(num_leaves));
    plan.total_bits = static_cast<UInt32>(std::countr_zero(num_leaves));
    plan.leaf_shift = ROUTE_BITS - plan.total_bits;

    /// bits_per_pass = floor(log2(max_partitions_per_pass)), clamped so a single pass can never
    /// exceed 16 bits (the route word is 32 bits and total_bits <= 20).
    UInt32 bits_per_pass = 10;
    if (max_partitions_per_pass >= 2)
        bits_per_pass = static_cast<UInt32>(std::bit_width(max_partitions_per_pass) - 1);
    bits_per_pass = std::clamp<UInt32>(bits_per_pass, 1, 16);

    UInt32 num_passes = 1;
    if (plan.total_bits > 0)
        num_passes = (plan.total_bits + bits_per_pass - 1) / bits_per_pass;

    /// Spread the bits evenly and put the remainder on the first passes, so the per-pass fanout is
    /// balanced (max - min <= 1) and there is no degenerate trailing low-fanout pass.
    const UInt32 base = plan.total_bits / num_passes;
    const UInt32 rem = plan.total_bits % num_passes;
    plan.pass_bits.assign(num_passes, base);
    for (UInt32 i = 0; i < rem; ++i)
        plan.pass_bits[i] += 1;

    return plan;
}

}
