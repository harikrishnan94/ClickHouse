#pragma once

#include <Common/HashTable/Hash.h>

#include <base/types.h>

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace DB::RadixJoin
{

/** One 32-bit CRC32C hash (`HashT`) of a fixed-width *packed* join key, used for BOTH routing and
  * bucketing in the radix join:
  *
  *   - the top `total_bits` of the hash select the leaf (the radix route, see PartitionPlan), and
  *   - the low bits (masked by `num_buckets - 1`) select the slot within that leaf's open-addressing
  *     table.
  *
  * Leaf routing and within-leaf bucketing therefore share one 32-bit word; the partition plan must
  * keep `total_bits + log2(buckets) <= 32` so the two slices do not exhaust the hash. Nothing
  * per-row hash is ever stored: the build insert, the scatter route, and the probe lookup all
  * recompute this same function from the key bytes, so a key always lands in the same leaf and
  * bucket on both sides (the join's core invariant).
  *
  * The key is the row-major concatenation of the join-key columns (KeyLayout). For packed widths
  * that match a standard integer type (1, 2, 4, 8, 16, 32 bytes) the hash is `HashCRC32<T>` on the
  * loaded value. Composite widths with no matching integer type (12, 20, …, 64) fall back to
  * `updateWeakHash32` over the raw byte span (same CRC32C family). RHJ v1 admits widths that are
  * multiples of 4 in [4, 64]; widths 1 and 2 are included for a consistent width→type table.
  */

using HashT = UInt32;

template <typename T>
inline HashT hashPackedKeyTyped(const void * key) noexcept
{
    static_assert(std::has_unique_object_representations_v<T>);
    T v{};
    __builtin_memcpy_inline(&v, key, sizeof(T));
    return static_cast<HashT>(HashCRC32<T>{}(v));
}

/// Maps a packed-key byte width to the integer type hashed via `HashCRC32<T>`, when one exists.
template <size_t width>
struct PackedKeyType;

template <> struct PackedKeyType<1> { using Type = UInt8; };
template <> struct PackedKeyType<2> { using Type = UInt16; };
template <> struct PackedKeyType<4> { using Type = UInt32; };
template <> struct PackedKeyType<8> { using Type = UInt64; };
template <> struct PackedKeyType<16> { using Type = UInt128; };
template <> struct PackedKeyType<32> { using Type = UInt256; };

template <size_t width>
inline constexpr bool hasPackedKeyType = (width == 1 || width == 2 || width == 4 || width == 8 || width == 16 || width == 32);

/// Byte-buffer CRC32C for composite packed widths with no matching integer type.
inline HashT hashPackedKeyBytes(const void * key, size_t width) noexcept
{
    return updateWeakHash32(static_cast<const UInt8 *>(key), width, static_cast<UInt32>(-1));
}

/// Compile-time-width hash. Hot sites dispatch the runtime width via a `switch` once per block.
template <size_t width>
inline HashT hashPackedKey(const void * key) noexcept
{
    static_assert(width >= 1 && width <= 64, "packed key width must be in [1, 64]");
    if constexpr (hasPackedKeyType<width>)
        return hashPackedKeyTyped<typename PackedKeyType<width>::Type>(key);
    else
    {
        static_assert(width % 4 == 0, "composite packed key width must be a multiple of 4");
        return hashPackedKeyBytes(key, width);
    }
}

/// Runtime-width entry, used by the few sites that carry the width as a value (the scatter route
/// recompute). Dispatches to the compile-time kernel; never a runtime-length loop.
inline HashT hashPackedKey(const void * key, size_t width) noexcept
{
    switch (width)
    {
        case 1:  return hashPackedKey<1>(key);
        case 2:  return hashPackedKey<2>(key);
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

/// The full 32-bit hash is the radix route word; bucketing masks its low bits.
inline UInt32 routeBits(HashT hash) noexcept { return hash; }
inline UInt32 bucketBits(HashT hash) noexcept { return hash; }

}
