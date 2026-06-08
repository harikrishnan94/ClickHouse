#pragma once

#include <Common/Exception.h>

#include <base/types.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rapidhash
{

inline constexpr std::array<uint64_t, 8> default_secret{
    0x2d358dccaa6c78a5ull,
    0x8bb84b93962eacc9ull,
    0x4b33a62ed433d4a3ull,
    0x4d5a2da51de1aa47ull,
    0xa0761d6478bd642full,
    0xe7037ed1a0b428dbull,
    0x90ed1765281c388cull,
    0xaaaaaaaaaaaaaaaaull};

constexpr uint64_t byte_swap64(uint64_t value) noexcept
{
    value = ((value & 0x00ff00ff00ff00ffull) << 8) | ((value >> 8) & 0x00ff00ff00ff00ffull);
    value = ((value & 0x0000ffff0000ffffull) << 16) | ((value >> 16) & 0x0000ffff0000ffffull);
    return (value << 32) | (value >> 32);
}

constexpr uint32_t byte_swap32(uint32_t value) noexcept
{
    value = ((value & 0x00ff00ffu) << 8) | ((value >> 8) & 0x00ff00ffu);
    return (value << 16) | (value >> 16);
}

inline uint64_t read64(const std::byte * data) noexcept
{
    uint64_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    if constexpr (std::endian::native == std::endian::little)
        return value;
    else
        return byte_swap64(value);
}

inline uint64_t read32(const std::byte * data) noexcept
{
    uint32_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    if constexpr (std::endian::native == std::endian::little)
        return value;
    else
        return byte_swap32(value);
}

inline void mum(uint64_t & a, uint64_t & b) noexcept
{
#if defined(__SIZEOF_INT128__)
    const auto result = static_cast<__uint128_t>(a) * b;
    a = static_cast<uint64_t>(result);
    b = static_cast<uint64_t>(result >> 64);
#else
#    error "rapidhash requires compiler support for unsigned __int128"
#endif
}

inline uint64_t mix(uint64_t a, uint64_t b) noexcept
{
    mum(a, b);
    return a ^ b;
}

template <size_t len>
inline uint64_t hash_with_seed(const void * key, uint64_t seed) noexcept
{
    static_assert(len <= 64, "rapidhash supports fixed widths up to 64 bytes");

    const auto * p = static_cast<const std::byte *>(key);
    const auto & secret = default_secret;

    seed ^= mix(seed ^ secret[2], secret[1]);

    uint64_t a = 0;
    uint64_t b = 0;

    if constexpr (len <= 16)
    {
        if constexpr (len >= 4)
        {
            seed ^= len;
            if constexpr (len >= 8)
            {
                a = read64(p);
                b = read64(p + len - 8);
            }
            else
            {
                a = read32(p);
                b = read32(p + len - 4);
            }
        }
        else if constexpr (len > 0)
        {
            a = (std::to_integer<uint64_t>(p[0]) << 45) | std::to_integer<uint64_t>(p[len - 1]);
            b = std::to_integer<uint64_t>(p[len >> 1]);
        }
    }
    else
    {
        if constexpr (len > 16)
            seed = mix(read64(p) ^ secret[2], read64(p + 8) ^ seed);
        if constexpr (len > 32)
            seed = mix(read64(p + 16) ^ secret[2], read64(p + 24) ^ seed);
        if constexpr (len > 48)
            seed = mix(read64(p + 32) ^ secret[1], read64(p + 40) ^ seed);

        a = read64(p + len - 16) ^ len;
        b = read64(p + len - 8);
    }

    a ^= secret[1];
    b ^= seed;
    mum(a, b);
    return mix(a ^ secret[7], b ^ secret[1] ^ len);
}

template <size_t len>
inline uint64_t hash(const void * key) noexcept
{
    return hash_with_seed<len>(key, 0);
}

}

namespace DB
{
namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}
}

namespace DB::RadixHash
{

/// Runtime-width entry to the compile-time-width `rapidhash::hash` (`key_width` a multiple of 4 in
/// [4, 64]). One 64-bit RapidHash of the PACKED row-major key serves BOTH routing and the within-leaf
/// bucket: the build routes the scatter / leaf by the TOP bits (`hash >> 32`, fed into the 32-bit
/// scatter kernels) and the leaf HT buckets by the LOW bits (`hash & (num_buckets - 1)`). A single
/// well-mixed 64-bit hash has effectively independent halves, so the leaf id and the bucket no longer
/// share bits — there is no `total_bits + log2(num_buckets) <= 32` saturation. The identical function
/// runs at every site (`add` histogram, every scatter/refine pass, the leaf-HT fill, and the probe), so
/// a key always routes to the same leaf and bucket on build and probe (consistency = correctness).
///
/// The width is dispatched to the fully-unrolled `rapidhash::hash<W>` (no runtime loop). Hot sites that
/// already know the width at compile time (`fillLeafT<W>`, `collectMatchesT<W>`, the probe selector's
/// per-width loop) call `rapidhash::hash<W>` directly instead.
inline UInt64 rapidHashKey(const char * key, size_t key_width) noexcept(false)
{
    switch (key_width)
    {
        case 4:  return rapidhash::hash<4>(key);
        case 8:  return rapidhash::hash<8>(key);
        case 12: return rapidhash::hash<12>(key);
        case 16: return rapidhash::hash<16>(key);
        case 20: return rapidhash::hash<20>(key);
        case 24: return rapidhash::hash<24>(key);
        case 28: return rapidhash::hash<28>(key);
        case 32: return rapidhash::hash<32>(key);
        case 36: return rapidhash::hash<36>(key);
        case 40: return rapidhash::hash<40>(key);
        case 44: return rapidhash::hash<44>(key);
        case 48: return rapidhash::hash<48>(key);
        case 52: return rapidhash::hash<52>(key);
        case 56: return rapidhash::hash<56>(key);
        case 60: return rapidhash::hash<60>(key);
        case 64: return rapidhash::hash<64>(key);
        default:
            throw Exception(
                ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: unsupported key width {} (multiple of 4 in [4, 64])", key_width);
    }
}

}
