#pragma once

#include <cstdint>


namespace DB::RadixShuffle
{

/// Documented hash combiner used by every hash column primitive.  This is
/// the canonical 32-bit `boost::hash_combine` form:
///
///     combine(prior, h) = prior ^ (h + 0x9e3779b9 + (prior << 6) + (prior >> 2));
///
/// The constant 0x9e3779b9 is the 32-bit fractional part of phi (golden
/// ratio).  The (prior << 6) + (prior >> 2) shift pair injects bits from
/// across the prior word into the XOR, mitigating cancellation when
/// prior == h.
///
/// The combiner is chain-friendly: applying it across columns in any fixed
/// order, with the same per-column hash function, produces a deterministic
/// per-row composite hash.  Reordering columns changes the composite hash
/// by the well-defined combiner rule.
[[gnu::always_inline]] inline uint32_t hashCombine(uint32_t prior, uint32_t h) noexcept
{
    return prior ^ (h + 0x9e3779b9U + (prior << 6) + (prior >> 2));
}

}
