#pragma once

#include <Common/HashTable/Hash.h>

#include <base/types.h>
#include <base/unaligned.h>

#include <cstddef>

namespace DB::RadixHash
{

/// Leaf-routing hash of the PACKED key (`key_width` a multiple of 4 in [4, 64]).
///
/// A chained hardware CRC32C over the key's 8-byte words plus an optional 4-byte tail, seeded with
/// `WEAK_HASH32_INITIAL_VALUE` so that for a single 8-byte key it reproduces `ColumnVector<UInt64>`'s
/// `computeHashInto` byte-for-byte (`_mm_crc32_u64(~0u, word)`). The leaf id is the top `total_bits` of
/// this hash (`leaf = routeHash >> shift`).
///
/// The SAME function runs at every routing site — the `add` histogram, the pass-0 scatter, every refine
/// pass (which has only the scattered packed key, not the typed key columns), and the probe — all over
/// the IDENTICAL packed-key bytes. So a key always routes to the same leaf on build and probe
/// (consistency = correctness) and nothing needs to carry the hash through the multi-pass cascade.
///
/// INDEPENDENT of `bucketHash`: `routeHash` is CRC32C-class (linear over GF(2)) while `bucketHash` is a
/// Murmur-style multiply-fold (non-linear) — different function families, so within a leaf the bucket
/// keeps full 32-bit entropy regardless of how many top bits the leaf routing consumed (the saturation
/// fix). `routeHash` is not a transform of `bucketHash`, nor the reverse.
inline UInt32 routeHash(const char * key, size_t key_width) noexcept
{
    UInt64 h = WEAK_HASH32_INITIAL_VALUE;
    size_t i = 0;
    for (; i + 8 <= key_width; i += 8)
        h = intHashCRC32(unalignedLoad<UInt64>(key + i), h);
    if (i < key_width) /// key_width is a multiple of 4, so the only remainder is a single 4-byte word
        h = intHashCRC32(static_cast<UInt64>(unalignedLoad<UInt32>(key + i)), h);
    return static_cast<UInt32>(h);
}

}
