#pragma once

#include <cstdint>


namespace DB
{

/// 32-bit hash combiner used by every hash column primitive.
///
///     combine(prior, h) = prior ^ (h + 0x9e3779b9 + (prior << 6) + (prior >> 2));
///
/// 0x9e3779b9 is the 32-bit fractional part of the golden ratio — a well-mixed
/// constant that prevents cancellation when prior == h.  The shift pair
/// (prior << 6) + (prior >> 2) injects bits from across the prior word into the XOR.
///
/// Applying this combiner across columns in a fixed order produces a deterministic
/// per-row composite hash.
[[gnu::always_inline]] inline uint32_t hashCombine(uint32_t prior, uint32_t h) noexcept
{
    return prior ^ (h + 0x9e3779b9U + (prior << 6) + (prior >> 2));
}

}
