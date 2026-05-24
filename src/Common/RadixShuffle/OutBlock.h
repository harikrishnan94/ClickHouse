#pragma once

#include <Common/RadixShuffle/BumpArena.h>

#include <algorithm>
#include <cstddef>
#include <utility>


namespace DB::RadixShuffle
{

/// Maximum number of data columns supported per row.
static constexpr int kMaxK = 8;

/// Minimum and default OutBlock capacities (in rows).
static constexpr size_t kOutCapMin = 4096;
static constexpr size_t kOutCapMax = 65536;


/// Round `n` up to the nearest multiple of 8.
/// Required because SWWC NT stores need 64-byte-aligned destinations
/// (8 × uint64_t), so every OutBlock capacity must be a multiple of 8.
[[nodiscard]] inline size_t round8(size_t n) noexcept
{
    return (n + 7) & ~size_t(7);
}


/// Compute adaptive initial/maximum OutBlock row capacities.
/// For large P the per-partition average row count is small; shrinking the
/// initial block avoids wasting arena memory.
[[nodiscard]] inline std::pair<size_t, size_t> adaptiveCaps(size_t rows_per_thread, size_t P) noexcept
{
    const size_t avg = (rows_per_thread + P - 1) / P;
    const size_t init = round8(std::max(size_t(64), std::min(kOutCapMin, avg + avg / 2 + 1)));
    const size_t maxc = round8(std::max(init, std::min(kOutCapMax, init * 4)));
    return {init, maxc};
}


/// One output block in a per-partition singly-linked chain.
/// `cols[k]` holds the start of the k-th column's array for this block.
/// The layout is column-major: each column's data is contiguous within a block.
///
/// Cache-line-aligned so the header does not share a cache line with payload.
struct alignas(64) OutBlock
{
    OutBlock * next = nullptr;
    size_t filled = 0;
    size_t capacity = 0;
    void * cols[kMaxK] = {};
};


/// Per-partition mutable output state maintained by the operator.
struct PartState
{
    OutBlock * head = nullptr;
    OutBlock * cur = nullptr;
    size_t next_cap = kOutCapMin;
};


/// Allocate a new OutBlock from `arena`.
/// `K`        — number of data columns.
/// `elem_size` — byte size of one element (e.g., sizeof(uint64_t)).
/// `cap`      — row capacity of the new block (must be a multiple of 8 for SWWC).
///
/// Layout: [OutBlock header | col_0[cap] | col_1[cap] | ... | col_{K-1}[cap]]
/// all within one contiguous 64-byte-aligned allocation from `arena`.
[[nodiscard]] inline OutBlock * newOutBlock(BumpArena & arena, int K, size_t elem_size, size_t cap)
{
    constexpr size_t hdr = (sizeof(OutBlock) + 63) & ~size_t(63); // round header to 64 B
    char * raw = arena.alignedAlloc(hdr + static_cast<size_t>(K) * cap * elem_size, 64);
    auto * b = reinterpret_cast<OutBlock *>(raw);
    b->next = nullptr;
    b->filled = 0;
    b->capacity = cap;
    for (int k = 0; k < K; ++k)
        b->cols[k] = raw + hdr + static_cast<size_t>(k) * cap * elem_size;
    for (int k = K; k < kMaxK; ++k)
        b->cols[k] = nullptr;
    return b;
}


/// Prepend a new OutBlock to partition `ps`, doubling `next_cap` up to `max_cap`.
inline void growPart(PartState & ps, BumpArena & arena, int K, size_t elem_size, size_t max_cap = kOutCapMax)
{
    OutBlock * nb = newOutBlock(arena, K, elem_size, ps.next_cap);
    nb->next = ps.head;
    ps.head = ps.cur = nb;
    ps.next_cap = std::min(ps.next_cap * 2, max_cap);
}


/// Allocate a new OutBlock with per-column element sizes.
/// `elem_sizes[k]` is the byte size of one element in column k
/// (e.g., sizeof(uint64_t) for UInt64, 9 = 1+8 for Nullable(UInt64)).
///
/// Layout: [OutBlock header | col_0[cap * elem_sizes[0]] | col_1[cap * elem_sizes[1]] | ...]
[[nodiscard]] inline OutBlock * newOutBlock(BumpArena & arena, int K, const size_t * elem_sizes, size_t cap)
{
    constexpr size_t hdr = (sizeof(OutBlock) + 63) & ~size_t(63);
    size_t total = 0;
    for (int k = 0; k < K; ++k)
        total += elem_sizes[static_cast<size_t>(k)] * cap;
    char * raw = arena.alignedAlloc(hdr + total, 64);
    auto * b = reinterpret_cast<OutBlock *>(raw);
    b->next = nullptr;
    b->filled = 0;
    b->capacity = cap;
    size_t off = 0;
    for (int k = 0; k < K; ++k)
    {
        b->cols[k] = raw + hdr + off;
        off += elem_sizes[static_cast<size_t>(k)] * cap;
    }
    for (int k = K; k < kMaxK; ++k)
        b->cols[k] = nullptr;
    return b;
}


/// Prepend a new OutBlock with per-column element sizes.
inline void growPart(PartState & ps, BumpArena & arena, int K, const size_t * elem_sizes, size_t max_cap = kOutCapMax)
{
    OutBlock * nb = newOutBlock(arena, K, elem_sizes, ps.next_cap);
    nb->next = ps.head;
    ps.head = ps.cur = nb;
    ps.next_cap = std::min(ps.next_cap * 2, max_cap);
}

} // namespace DB::RadixShuffle
