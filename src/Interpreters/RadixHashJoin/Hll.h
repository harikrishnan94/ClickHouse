#pragma once

#include <base/types.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace DB::RadixJoin
{

/** A compact, dense-only HyperLogLog used to estimate the number of distinct keys in each radix leaf so the
  * leaf's open-addressing table can be sized by distinct keys rather than by raw row count. Modeled on the
  * dense path of the reference HLL (P. Flajolet et al., AOFA 2007): `m = 2^precision` one-byte registers,
  * a register index + leading-one rank extracted from the hash, register-wise max accumulation/merge, and a
  * harmonic-mean estimate with bias correction and a linear-counting small-range correction.
  *
  * Only the slice this use case needs is kept — no sparse format, no SIMD batch path, no C API. Storage is
  * NOT owned here: registers live in a caller-provided byte span (a flat per-worker × per-leaf array), so a
  * sketch is just `1 << precision` bytes with no per-object overhead — at the default precision one cache
  * line. Registers store the rank directly and start at 0 (an empty register contributes `2^0 = 1` to the
  * harmonic mean).
  *
  * The input is a 32-bit hash: the LOW 32 bits of the 64-bit packed-key hash. Those bits are statistically
  * independent of the high 32 bits that select the leaf (`routeBits`), so register selection within a leaf
  * is unbiased. The top `precision` bits index the register; the remaining `32 - precision` bits give the
  * rank (leftmost-set-bit position within that field, +1).
  */
namespace Hll
{
    /// Practical precision range. The maximum (64 registers) is one cache line per sketch; coarse accuracy
    /// is fine because the estimate only picks a power-of-two bucket count.
    inline constexpr UInt8 MIN_PRECISION = 4;
    inline constexpr UInt8 MAX_PRECISION = 6;

    /// We sketch the low 32 bits of the packed-key hash.
    inline constexpr UInt32 INPUT_BITS = 32;

    /// Transient sketch memory (num_workers × num_leaves × 2^p) is bounded by this; precision shrinks toward
    /// MIN_PRECISION as the leaf count grows so the partial sketches never blow past it.
    inline constexpr size_t MEMORY_BUDGET_BYTES = 32ULL << 20;

    /// Registers in a sketch of the given precision.
    constexpr UInt32 numRegisters(UInt8 precision) noexcept { return UInt32{1} << precision; }

    /// Bytes one sketch occupies (one byte per register).
    constexpr size_t sketchBytes(UInt8 precision) noexcept { return numRegisters(precision); }

    /// Accumulate one 32-bit hash into a sketch (register array of length `1 << precision`).
    inline void add(UInt8 * registers, UInt8 precision, UInt32 hash) noexcept
    {
        const UInt32 idx = hash >> (INPUT_BITS - precision); /// top `precision` bits select the register
        const UInt32 w = hash & ((UInt32{1} << (INPUT_BITS - precision)) - 1); /// low bits = the rank field
        /// Rank = position of the leftmost set bit within the `(32 - precision)`-bit field, +1; an all-zero
        /// field yields the maximum rank `(32 - precision) + 1`. `countl_zero(w) - precision` is the number
        /// of leading zeros inside the field (countl_zero counts over the full 32-bit width).
        const UInt8 rank = w == 0
            ? static_cast<UInt8>(INPUT_BITS - precision + 1)
            : static_cast<UInt8>(std::countl_zero(w) - precision + 1);
        registers[idx] = std::max(rank, registers[idx]);
    }

    /// Register-wise max merge of `src` into `dst` (both length `1 << precision`).
    inline void merge(UInt8 * dst, const UInt8 * src, UInt8 precision) noexcept
    {
        const UInt32 m = numRegisters(precision);
        for (UInt32 i = 0; i < m; ++i)
            dst[i] = std::max(dst[i], src[i]);
    }

    /// Cardinality estimate: bias-corrected harmonic mean, with linear counting in the small range.
    inline UInt64 estimate(const UInt8 * registers, UInt8 precision) noexcept
    {
        const UInt32 m = numRegisters(precision);
        /// Bias-correction constant α_m (the m == 16 value cannot be derived from the formula).
        const double alpha_m = m == 16 ? 0.673 : 0.7213 / (1.0 + 1.079 / m);
        double harmonic = 0.0;
        UInt32 zeros = 0;
        for (UInt32 i = 0; i < m; ++i)
        {
            harmonic += std::ldexp(1.0, -static_cast<int>(registers[i])); /// 2^(-register)
            zeros += registers[i] == 0;
        }
        const double raw = alpha_m * m * m / harmonic;
        /// Small-range correction: with empty registers in the small range, linear counting is more accurate.
        if (raw <= 2.5 * m && zeros != 0)
            return static_cast<UInt64>(m * std::log(static_cast<double>(m) / zeros));
        return static_cast<UInt64>(raw);
    }

    /// Largest precision in [MIN, MAX] that keeps `num_workers × num_leaves × 2^p` within the memory budget,
    /// flooring at MIN (HLL is always computed when enabled; precision just degrades for huge leaf counts).
    inline UInt8 choosePrecision(size_t num_workers, size_t num_leaves) noexcept
    {
        const size_t base = std::max<size_t>(num_workers, 1) * std::max<size_t>(num_leaves, 1);
        for (UInt8 p = MAX_PRECISION; p > MIN_PRECISION; --p)
            if (base * (size_t{1} << p) <= MEMORY_BUDGET_BYTES)
                return p;
        return MIN_PRECISION;
    }
}

}
