#pragma once

#include <cstddef>
#include <cstdint>

namespace DB
{

/// Maximum number of columns per OutBlock (keys + payloads).
static constexpr size_t kOutBlockMaxCols = 16;

/// Minimum/maximum rows per OutBlock slab. Capacity doubles on each grow,
/// capped at kOutCapMax. Chosen to balance allocation count vs cache residency.
static constexpr size_t kOutCapMin = 4096;
static constexpr size_t kOutCapMax = 60000;

inline size_t nextOutBlockCap(size_t prev)
{
    const size_t n = prev * 2;
    return (n < kOutCapMax) ? n : kOutCapMax;
}

/// One slab in a partition's output chain.
/// The header and all K column buffers are co-located in a single BumpArena bump.
/// Layout: [OutBlock header (64-aligned)] [col[0] buffer] [col[1] buffer] ...
struct alignas(64) OutBlock
{
    OutBlock * next = nullptr;
    size_t filled = 0;
    size_t capacity = 0;
    uint8_t num_cols = 0;
    void * cols[kOutBlockMaxCols] = {};
};

}
