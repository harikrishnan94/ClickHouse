#include <Interpreters/PartitionedHashConfig.h>

#include <algorithm>
#include <bit>
#include <cmath>

#if defined(OS_LINUX)
#include <unistd.h>
#endif

namespace DB
{

size_t roundUpPow2(size_t x) noexcept
{
    if (x <= 1)
        return 1;
    return std::bit_ceil(x);
}

size_t getPrivateL2Bytes()
{
    static const size_t result = []
    {
        size_t l2_size = 0;
#if defined(OS_LINUX) && defined(_SC_LEVEL2_CACHE_SIZE)
        if (auto ret = sysconf(_SC_LEVEL2_CACHE_SIZE); ret != -1)
            l2_size = static_cast<size_t>(ret);
#endif
        return std::max<size_t>(l2_size, PHJ_L2_FALLBACK_BYTES);
    }();
    return result;
}

std::vector<UInt8> factorisePassBits(size_t total_leaves, size_t max_partitions_per_pass) noexcept
{
    /// total_leaves is expected to be a power of two; total_bits = log2(total_leaves).
    const size_t tl = total_leaves < 1 ? 1 : total_leaves;
    const auto total_bits = static_cast<unsigned>(std::countr_zero(std::bit_ceil(tl)));
    if (total_bits == 0)
        return {};

    /// Per-pass cap in bits (floor(log2(max_partitions_per_pass))), clamped to [1, 6].
    size_t cap_parts = max_partitions_per_pass < 2 ? 2 : max_partitions_per_pass;
    auto bits_per_pass = static_cast<unsigned>(std::countr_zero(std::bit_floor(cap_parts)));
    bits_per_pass = std::clamp<unsigned>(bits_per_pass, 1, 6);

    const unsigned num_passes = (total_bits + bits_per_pass - 1) / bits_per_pass;
    const unsigned base = total_bits / num_passes;
    const unsigned rem = total_bits % num_passes;

    std::vector<UInt8> pass_bits(num_passes);
    for (unsigned i = 0; i < num_passes; ++i)
        pass_bits[i] = static_cast<UInt8>(base + (i < rem ? 1 : 0));
    return pass_bits;
}

PartitionConfig derivePartitionConfig(const PartitionConfigInputs & inputs)
{
    size_t total_leaves;

    if (!inputs.rhs_rows_estimation.has_value())
    {
        total_leaves = PHJ_DEFAULT_LEAVES;
    }
    else
    {
        const size_t l2_bytes = inputs.l2_bytes != 0 ? inputs.l2_bytes : getPrivateL2Bytes();

        /// rhs_rows_estimation is the right-side ROW count; size the leaf working set from the real
        /// per-row bytes incl. HT cell overhead (spec §5.2).
        const auto rhs_rows = static_cast<double>(*inputs.rhs_rows_estimation);

        const double real_build_bytes
            = rhs_rows * static_cast<double>(inputs.key_bytes + inputs.cell_bytes + inputs.payload_bytes);
        const double usable_l2 = PHJ_L2_HEADROOM * static_cast<double>(l2_bytes);

        auto num_leaves = static_cast<size_t>(std::ceil(real_build_bytes / std::max(usable_l2, 1.0)));
        num_leaves = std::max<size_t>(num_leaves, 1);
        num_leaves = roundUpPow2(num_leaves);
        num_leaves = std::clamp<size_t>(num_leaves, 1, PHJ_MAX_LEAVES);
        total_leaves = num_leaves;
    }

    PartitionConfig config;
    config.pass_bits = factorisePassBits(total_leaves, inputs.max_partitions_per_pass);
    config.total_leaves = size_t{1} << config.totalBits();
    return config;
}

}
