#pragma once

#include <cstdint>


namespace DB::RadixShuffle
{

/// Documented hash combiner used by every hash column primitive (§3.4). It is the
/// `boost::hash_combine` mixer expressed for 64-bit words:
///
///     combine(prior, h) = prior ^ (h + 0x9e3779b97f4a7c15ULL + (prior << 12) + (prior >> 4));
///
/// The constant is the 64-bit fractional part of phi (golden ratio).
/// `(prior << 12) + (prior >> 4)` injects bits from across the prior word
/// into the xor, mitigating cancellation when `prior == h`.
///
/// The combiner is associative-style chain-friendly: applying it across
/// columns in any fixed order, with the same per-column hash function,
/// produces a deterministic per-row composite hash. Reordering columns
/// changes the composite hash by the well-defined combiner rule (i.e.,
/// `combine(combine(a, hb), hc) != combine(combine(a, hc), hb)` in
/// general), which is exactly the testable property in §5.
[[gnu::always_inline]] inline uint64_t hashCombine(uint64_t prior, uint64_t h) noexcept
{
    return prior ^ (h + 0x9e3779b97f4a7c15ULL + (prior << 12) + (prior >> 4));
}

}
