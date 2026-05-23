#pragma once

// TODO: SIMD multi-versioning via Common/TargetSpecific.h (MULTITARGET_FUNCTION_X86_V3).
//       v1 ships scalar only.  See
//       docs/en/development/radix-shuffle-column-primitives-implementation.md
//       §"SIMD multi-versioning" for the intended future implementation.

#include <cstddef>
#include <cstdint>
#include <cstring>


namespace DB::RadixShuffle
{

/// MurmurHash3 32-bit finalizer.  Analogue of the 64-bit fmix64 used in
/// the old `intHash64Local`.  Applied to per-row scalar payloads after
/// loading into a uint32_t (or as a fold step for wider types).
[[gnu::always_inline]] inline uint32_t fmix32(uint32_t x) noexcept
{
    x ^= x >> 16;
    x *= 0x85ebca6bU;
    x ^= x >> 13;
    x *= 0xc2b2ae35U;
    x ^= x >> 16;
    return x;
}


/// Hash one fixed-width value of type T into a uint32_t.
///
/// For sizeof(T) <= 4 the value is loaded into a uint32_t and passed
/// through fmix32.  For wider types the bytes are folded through fmix32
/// in 4-byte chunks accumulating into a uint32_t state.
template <typename T>
[[gnu::always_inline]] inline uint32_t hashOne32(const T & v) noexcept
{
    if constexpr (sizeof(T) <= sizeof(uint32_t))
    {
        uint32_t buf = 0;
        std::memcpy(&buf, &v, sizeof(T));
        return fmix32(buf);
    }
    else
    {
        const auto * bytes = reinterpret_cast<const unsigned char *>(&v);
        uint32_t acc = 0;
        for (size_t i = 0; i < sizeof(T); i += sizeof(uint32_t))
        {
            uint32_t word = 0;
            const size_t chunk = (i + sizeof(uint32_t) <= sizeof(T)) ? sizeof(uint32_t) : (sizeof(T) - i);
            std::memcpy(&word, bytes + i, chunk);
            acc ^= fmix32(word + acc + 0x9e3779b9U);
        }
        return acc;
    }
}


/// Hash a byte range into a uint32_t.  Analogous to the old `hashBytes`
/// but produces a 32-bit result.
[[gnu::always_inline]] inline uint32_t hashBytes32(const unsigned char * data, size_t n) noexcept
{
    uint32_t acc = 0x811c9dc5U ^ (static_cast<uint32_t>(n) * 0x9e3779b9U);
    size_t i = 0;
    while (i + sizeof(uint32_t) <= n)
    {
        uint32_t word = 0;
        std::memcpy(&word, data + i, sizeof(uint32_t));
        acc = fmix32(word + acc);
        i += sizeof(uint32_t);
    }
    if (i < n)
    {
        uint32_t tail = 0;
        std::memcpy(&tail, data + i, n - i);
        acc = fmix32(tail + acc);
    }
    return acc;
}

}
