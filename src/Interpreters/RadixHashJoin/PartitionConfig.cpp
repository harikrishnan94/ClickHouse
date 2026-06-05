#include <Interpreters/RadixHashJoin/PartitionConfig.h>

#include <algorithm>
#include <bit>
#include <cmath>

namespace DB::RadixHash
{

size_t roundUpToPowerOfTwo(size_t x)
{
    if (x <= 1)
        return 1;
    return std::bit_ceil(x);
}

PartitionConfig PartitionConfig::make(
    std::optional<UInt64> rhs_rows_estimation,
    size_t l2_bytes,
    UInt64 max_partitions_per_pass)
{
    PartitionConfig cfg;

    size_t num_leaves = DEFAULT_LEAVES;
    if (rhs_rows_estimation.has_value())
    {
        /// table_bytes ~= rhs_rows x (cell_bytes / load_factor) = rhs_rows x 32, independent of payload width.
        const double table_bytes = static_cast<double>(*rhs_rows_estimation) * (static_cast<double>(CELL_BYTES) / LOAD_FACTOR);
        const size_t l2 = l2_bytes != 0 ? l2_bytes : L2_FALLBACK_BYTES;
        const double usable_l2 = L2_HEADROOM * static_cast<double>(l2);

        double n = std::ceil(table_bytes / usable_l2);
        if (!(n >= 1.0))
            n = 1.0;
        num_leaves = roundUpToPowerOfTwo(static_cast<size_t>(n));
        num_leaves = std::clamp<size_t>(num_leaves, 1, MAX_LEAVES);
    }

    cfg.num_leaves = num_leaves;
    cfg.total_bits = static_cast<UInt32>(std::countr_zero(num_leaves)); /// num_leaves is a power of two
    cfg.shift = HASH_BITS - cfg.total_bits;

    /// bits_per_pass = floor(log2(max_partitions_per_pass)), clamped to a sane range.
    UInt32 bits_per_pass = 10;
    if (max_partitions_per_pass >= 2)
        bits_per_pass = static_cast<UInt32>(std::bit_width(max_partitions_per_pass) - 1);
    bits_per_pass = std::clamp<UInt32>(bits_per_pass, 1, 16);

    /// Minimum number of passes, bits spread evenly with the remainder on the first passes
    /// (so max(pass_bits) - min(pass_bits) <= 1 and no tiny trailing pass), spec section 5.3.
    UInt32 num_passes = 1;
    if (cfg.total_bits > 0)
        num_passes = (cfg.total_bits + bits_per_pass - 1) / bits_per_pass;

    const UInt32 base = cfg.total_bits / num_passes;
    const UInt32 rem = cfg.total_bits % num_passes;

    cfg.pass_bits.assign(num_passes, base);
    for (UInt32 i = 0; i < rem; ++i)
        cfg.pass_bits[i] += 1;

    return cfg;
}

}
