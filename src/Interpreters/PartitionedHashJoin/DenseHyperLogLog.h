#pragma once

#include <base/types.h>

#include <array>
#include <bit>
#include <cmath>

namespace DB
{

/** Dense-only HyperLogLog sketch used to size the leaf hash tables of `PartitionedHashJoin`.
  *
  * One sketch per fill lane, fed the 32-bit route word of every non-null build key and merged
  * once at the build barrier. The top `precision` bits of the hash pick a register, the
  * remaining bit field gives the rank (leftmost-set-bit position + 1; an all-zero field maps
  * to the maximum rank). The estimate is the classic bias-corrected harmonic mean with the
  * linear-counting correction for small cardinalities.
  *
  * At `precision` = 13 the sketch is 8 KiB and the standard error is about 1.15%, which the
  * partition plan covers with its reserve safety factor.
  */
struct DenseHyperLogLog
{
    static constexpr UInt32 precision = 13;
    static constexpr UInt32 register_count = 1u << precision;

    std::array<UInt8, register_count> registers{};

    /// The route words of wide/composite/variable keys come from a multiply-shift fold whose
    /// middle bits (the ones the rank looks at) are not avalanche-quality for structured keys,
    /// so the sketch finalizes its input itself. The finalizer (murmur-style fmix32) is a
    /// bijection: distinct inputs stay distinct, only the bit mixing changes.
    static ALWAYS_INLINE UInt32 finalize(UInt32 hash)
    {
        hash ^= hash >> 16;
        hash *= 0x85ebca6bu;
        hash ^= hash >> 13;
        hash *= 0xc2b2ae35u;
        hash ^= hash >> 16;
        return hash;
    }

    ALWAYS_INLINE void add(UInt32 hash)
    {
        const UInt32 mixed = finalize(hash);
        const UInt32 index = mixed >> (32 - precision);
        const UInt32 field = mixed & ((1u << (32 - precision)) - 1);
        const UInt8 rank = field ? static_cast<UInt8>(std::countl_zero(field) - precision + 1) : static_cast<UInt8>(32 - precision + 1);
        registers[index] = std::max(registers[index], rank);
    }

    void merge(const DenseHyperLogLog & other)
    {
        for (size_t i = 0; i < register_count; ++i)
            registers[i] = std::max(registers[i], other.registers[i]);
    }

    double estimate() const
    {
        static const std::array<double, 33> inverse_powers = []
        {
            std::array<double, 33> result{};
            for (size_t rank = 0; rank < result.size(); ++rank)
                result[rank] = std::ldexp(1.0, -static_cast<int>(rank));
            return result;
        }();

        constexpr double m = register_count;
        constexpr double alpha = 0.7213 / (1.0 + 1.079 / m);
        double inverse_sum = 0;
        size_t zeros = 0;
        for (const UInt8 rank : registers)
        {
            inverse_sum += inverse_powers[rank];
            zeros += rank == 0;
        }
        const double raw = alpha * m * m / inverse_sum;
        if (raw <= 2.5 * m && zeros > 0)
            return m * std::log(m / static_cast<double>(zeros));
        return raw;
    }
};

}
