#pragma once

#include <base/types.h>

#include <cstddef>
#include <cstring>

namespace DB::RadixJoin
{

/** One 64-bit hash of a fixed-width *packed* join key, used for BOTH halves of the radix join:
  *
  *   - the high 32 bits select the leaf (the top `total_bits` of the high word are the radix
  *     route, see PartitionPlan), and
  *   - the low 32 bits select the slot within that leaf's open-addressing table.
  *
  * Because routing consumes the high bits and the leaf bucket consumes the low bits, the two
  * halves must be statistically independent — otherwise every key inside one leaf would also
  * share low bits and collapse into a few buckets. We therefore run the multiply-fold mixer
  * (below) for at least two rounds, which fully avalanches every input bit into both 32-bit
  * halves of the result. This is what lets a single hash drive routing AND bucketing with no
  * separate finalizer, and it is why nothing per-row hash is ever stored: the build insert, the
  * scatter route, and the probe lookup all recompute this same function from the key bytes, so a
  * key always lands in the same leaf and bucket on both sides (the join's core invariant).
  *
  * The key is the row-major concatenation of the join-key columns (KeyLayout), of total width a
  * multiple of 4 in [4, 64]. The function is defined purely on the key bytes, so build and probe
  * agree by construction.
  */

/// wyhash-style "mum": multiply two 64-bit words and xor the two halves of the 128-bit product.
/// A single call already mixes every bit of both inputs into every output bit region; we chain it.
inline UInt64 mulFold(UInt64 a, UInt64 b) noexcept
{
    const __uint128_t product = static_cast<__uint128_t>(a) * b;
    return static_cast<UInt64>(product) ^ static_cast<UInt64>(product >> 64);
}

/// Distinct odd mixing constants (high-entropy bit patterns from the SplitMix64 / fractional-sqrt
/// families). Odd multipliers keep the multiply-fold a bijection on the low bits.
inline constexpr UInt64 KEY_HASH_C0 = 0x9E3779B97F4A7C15ULL;
inline constexpr UInt64 KEY_HASH_C1 = 0xBF58476D1CE4E5B9ULL;
inline constexpr UInt64 KEY_HASH_C2 = 0x94D049BB133111EBULL;
inline constexpr UInt64 KEY_HASH_C3 = 0xD6E8FEB86659FD93ULL;

/// Compile-time-width hash. The width is a multiple of 4 in [4, 64]; the loop and the tail are
/// fully unrolled, so each call is a short straight-line sequence of `memcpy_inline` loads and
/// multiply-folds with no runtime branch on the width. Hot sites (the per-row build/probe loops)
/// dispatch the runtime width to this via a `switch` once per block.
template <size_t width>
inline UInt64 hashPackedKey(const void * key) noexcept
{
    static_assert(width >= 4 && width % 4 == 0 && width <= 64, "packed key width must be a multiple of 4 in [4, 64]");
    const char * p = static_cast<const char *>(key);

    UInt64 acc;
    if constexpr (width == 4)
    {
        UInt32 v;
        __builtin_memcpy_inline(&v, p, 4);
        acc = mulFold(static_cast<UInt64>(v) ^ KEY_HASH_C1, KEY_HASH_C2);
    }
    else if constexpr (width == 8)
    {
        UInt64 v;
        __builtin_memcpy_inline(&v, p, 8);
        acc = mulFold(v ^ KEY_HASH_C1, KEY_HASH_C2);
    }
    else
    {
        /// Fold each 8-byte lane, then the trailing 4-byte lane if the width is not a multiple of 8.
        acc = KEY_HASH_C0 ^ width;
        size_t i = 0;
        for (; i + 8 <= width; i += 8)
        {
            UInt64 v;
            __builtin_memcpy_inline(&v, p + i, 8);
            acc = mulFold(acc ^ v, KEY_HASH_C1);
        }
        if constexpr (width % 8 != 0)
        {
            UInt32 v;
            __builtin_memcpy_inline(&v, p + i, 4);
            acc = mulFold(acc ^ static_cast<UInt64>(v), KEY_HASH_C2);
        }
    }

    /// Final avalanche round: guarantees the high and low 32-bit halves are independent.
    return mulFold(acc ^ KEY_HASH_C3, KEY_HASH_C0);
}

/// Runtime-width entry, used by the few sites that carry the width as a value (the scatter route
/// recompute). Dispatches to the unrolled compile-time kernel; never a runtime-length loop.
inline UInt64 hashPackedKey(const void * key, size_t width) noexcept
{
    switch (width)
    {
        case 4:  return hashPackedKey<4>(key);
        case 8:  return hashPackedKey<8>(key);
        case 12: return hashPackedKey<12>(key);
        case 16: return hashPackedKey<16>(key);
        case 20: return hashPackedKey<20>(key);
        case 24: return hashPackedKey<24>(key);
        case 28: return hashPackedKey<28>(key);
        case 32: return hashPackedKey<32>(key);
        case 36: return hashPackedKey<36>(key);
        case 40: return hashPackedKey<40>(key);
        case 44: return hashPackedKey<44>(key);
        case 48: return hashPackedKey<48>(key);
        case 52: return hashPackedKey<52>(key);
        case 56: return hashPackedKey<56>(key);
        case 60: return hashPackedKey<60>(key);
        case 64: return hashPackedKey<64>(key);
        default: return 0; /// unreachable: the planner gate restricts widths to a multiple of 4 in [4, 64]
    }
}

/// The high word of the hash holds the radix route; the low 32 bits hold the leaf bucket.
inline UInt32 routeBits(UInt64 hash) noexcept { return static_cast<UInt32>(hash >> 32); }
inline UInt32 bucketBits(UInt64 hash) noexcept { return static_cast<UInt32>(hash); }

}
