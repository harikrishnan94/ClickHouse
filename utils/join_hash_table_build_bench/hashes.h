#pragma once

#include "platform.h"

#include <array>
#include <cstdint>
#include <random>

/// Paper Figure 8, 64-bit keys. Seeds are not specified there; these two 32-bit constants are
/// distinct and documented so a rerun can match. Mixing constant is from the paper.
inline constexpr uint32_t kUmbraSeed1 = 0x9E3779B9u;
inline constexpr uint32_t kUmbraSeed2 = 0x85EBCA77u;
inline constexpr uint64_t kUmbraMix64 = 0x2545F4914F6CDD1Dull;

inline uint64_t umbra_hash64(uint64_t key)
{
    const uint32_t crc1 = crc32c_u64(kUmbraSeed1, key);
    const uint32_t crc2 = crc32c_u64(kUmbraSeed2, key);
    const uint64_t combined = crc1 | (static_cast<uint64_t>(crc2) << 32);
    return combined * kUmbraMix64;
}

/// ClickHouse HashCRC32<UInt64> / intHashCRC32: CRC32C with initial value all-ones.
inline uint64_t ch_hash64(uint64_t key)
{
    return crc32c_u64(0xFFFFFFFFu, key);
}

/// TwoLevelHashTable::getBucketFromHash with BITS_FOR_BUCKET = 8.
inline uint32_t ch_bucket(uint64_t hash, uint32_t parts_mask)
{
    return static_cast<uint32_t>((hash >> (32u - 8u)) & parts_mask);
}

/// 4-bit tags in a 16-bit word, C(16,4)=1820 patterns padded to 2048. Index is the high 11 bits of
/// the low 32 bits of the hash, as in the paper's Figure 6.
struct BloomTags
{
    std::array<uint16_t, 2048> tags{};

    BloomTags()
    {
        uint16_t combos[1820];
        size_t n = 0;
        for (int a = 0; a < 16; ++a)
            for (int b = a + 1; b < 16; ++b)
                for (int c = b + 1; c < 16; ++c)
                    for (int d = c + 1; d < 16; ++d)
                        combos[n++] = static_cast<uint16_t>((1u << a) | (1u << b) | (1u << c) | (1u << d));

        for (size_t i = 0; i < 1820; ++i)
            tags[i] = combos[i];

        /// Pad with replacement samples from the 1820, seed documented.
        std::mt19937 rng(0xC0FFEE01u);
        std::uniform_int_distribution<int> dist(0, 1819);
        for (size_t i = 1820; i < 2048; ++i)
            tags[i] = combos[static_cast<size_t>(dist(rng))];
    }

    uint16_t tag(uint64_t hash) const
    {
        const uint16_t slot = static_cast<uint16_t>(static_cast<uint32_t>(hash) >> (32u - 11u));
        return tags[slot];
    }
};

inline const BloomTags & bloom_tags()
{
    static const BloomTags t;
    return t;
}

inline uint16_t umbra_tag(uint64_t hash)
{
    return bloom_tags().tag(hash);
}
