#pragma once

#include <base/types.h>
#include <Common/Allocator.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace DB::RadixJoin
{

/** A bag of individually-sized aligned allocations that are all released together (or one at a time).
  *
  * Backing choice — deliberate, and a key part of the performance story. Every block is a single
  * exact-sized aligned allocation from the default ClickHouse allocator (`Allocator<false, false>`
  * == plain jemalloc, NO zero-fill, NO `MADV_POPULATE_WRITE`, NO `mmap`/THP). Two consequences:
  *
  *   1. No per-query page-table churn. jemalloc hands back warm, already-faulted pages it retained
  *      from earlier queries, so there is no first-touch fault storm on allocation and no `munmap`
  *      teardown on free — which matters because the post-build allocates one array per leaf and one
  *      per leaf hash table every query. (An earlier design `madvise(MADV_HUGEPAGE)`'d its own 2 MiB
  *      slabs; measurement showed jemalloc's warm-page reuse was as good or better here and far
  *      simpler, so explicit THP was dropped.)
  *   2. The memory is NOT zeroed. Callers that need zeroes (the leaf-HT cells and the shared chain
  *      array) `memset` exactly the ranges they need, in parallel across the build workers; the
  *      scatter output needs no zeroing because the scatter overwrites every byte.
  *
  * Thread-safe: the `alloc`/`free` themselves run lock-free (jemalloc is thread-safe); only the
  * small block list is mutex-guarded, so many post-build workers carve their leaf arrays in parallel.
  * Returned pointers are stable until the arena is destroyed or the specific block is `release`d.
  */
class Arena
{
public:
    Arena();
    ~Arena();

    Arena(const Arena &) = delete;
    Arena & operator=(const Arena &) = delete;
    Arena(Arena && other) noexcept;
    Arena & operator=(Arena && other) noexcept;

    /// Aligned, exact-sized, NOT zero-filled. `alignment` must be a power of two. Thread-safe.
    void * allocate(size_t bytes, size_t alignment);

    template <typename T>
    T * allocateArray(size_t n)
    {
        return static_cast<T *>(allocate(n * sizeof(T), alignof(T)));
    }

    /// Return one block (a pointer previously returned by `allocate`) to the allocator immediately.
    /// Used to drop a consumed intermediate scatter partition during multi-pass refinement so peak
    /// memory tracks the live working set, not the sum of all passes. No-op for nullptr. Thread-safe.
    void release(void * base) noexcept;

    size_t blockCount() const;
    size_t bytesReserved() const;

private:
    struct Block
    {
        char * base = nullptr;
        size_t size = 0;
    };

    std::vector<Block> blocks;
    /// unique_ptr keeps the arena movable while still owning a stable mutex.
    std::unique_ptr<std::mutex> mutex;

    void freeAll() noexcept;

    [[no_unique_address]] Allocator<false, false> allocator;
};

}
