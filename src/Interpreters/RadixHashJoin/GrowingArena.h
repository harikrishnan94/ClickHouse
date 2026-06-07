#pragma once

#include <base/types.h>
#include <Common/Allocator.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace DB::RadixHash
{

/** GrowingArena — owns a set of jemalloc allocations and frees them on destruction. It backs the
  * selector `pid` arrays, the deferred-scatter output (per-leaf key/ref/hash arrays packed into one
  * contiguous block per leaf), and the leaf hash tables.
  *
  * Backing: every `alloc()` is a single, exact-sized, aligned allocation from the standard ClickHouse
  * allocator `Allocator<false, false>` — plain jemalloc `malloc`/`posix_memalign`, with **no zero-fill
  * and no `MADV_POPULATE_WRITE`**, and **no `mmap`/THP anywhere**. There is no one-shot bump block;
  * jemalloc reuses warm, already-faulted pages across queries (no first-touch fault storm, no per-query
  * `munmap` page-table teardown).
  *
  * Because the memory is NOT zeroed, callers that need zeroed memory (the leaf-HT cells and the shared
  * `next_chain`) `memset` it themselves — and they do so **in parallel across the build workers**.
  * The scatter output needs no zeroing at all (it is fully overwritten by the scatter).
  *
  * `alloc()` is thread-safe: the `malloc` runs lock-free (jemalloc is thread-safe) and only the block
  * bookkeeping is mutex-guarded, so RadixHashJoin's post-build can allocate the per-leaf / per-partition
  * arrays from many workers in parallel. `freeBlock()` releases one consumed allocation immediately (used
  * to free intermediate scatter partitions as the refine consumes them, keeping peak memory low).
  * Pointers returned by `alloc` are stable until the arena is freed (or the block is `freeBlock`-ed).
  */
class GrowingArena
{
public:
    /// Retained only for source/API compatibility with call sites that pass a cap; the value is ignored
    /// (each allocation is exact-sized).
    static constexpr size_t DEFAULT_MAX_BLOCK = 8 * 1024 * 1024;

    explicit GrowingArena(size_t max_block_bytes = DEFAULT_MAX_BLOCK);
    ~GrowingArena();

    GrowingArena(const GrowingArena &) = delete;
    GrowingArena & operator=(const GrowingArena &) = delete;
    GrowingArena(GrowingArena && other) noexcept;
    GrowingArena & operator=(GrowingArena && other) noexcept;

    /// Returns a pointer aligned to `alignment` (a power of two). The allocation is contiguous and is
    /// NOT zero-filled. Thread-safe (may be called concurrently from multiple workers on one arena).
    void * alloc(size_t bytes, size_t alignment);

    template <typename T>
    T * allocArray(size_t n)
    {
        return static_cast<T *>(alloc(n * sizeof(T), alignof(T)));
    }

    /// Free the single allocation whose base pointer is `base` (returned by a previous `alloc`),
    /// returning it to the allocator immediately. Thread-safe (different blocks may be freed
    /// concurrently). No-op for `nullptr`.
    void freeBlock(void * base) noexcept;

    /// Free every allocation (the arena stays usable for further allocs).
    void clear() noexcept { freeAll(); }

    size_t blockCount() const { return blocks.size(); }
    size_t bytesReserved() const;
    size_t bytesUsed() const { return bytesReserved(); }

private:
    struct Block
    {
        char * base = nullptr;
        size_t size = 0;
    };

    std::vector<Block> blocks;
    /// Guards `blocks` (the malloc itself runs outside it). unique_ptr so the arena stays movable.
    std::unique_ptr<std::mutex> blocks_mutex;

    void freeAll() noexcept;

    /// Non-clearing, non-populating default allocator: plain `malloc`/`posix_memalign`, no zero-fill, no
    /// `MADV_POPULATE_WRITE`, no `mmap`/THP. The RadixHashJoin build path zeroes — in parallel — only the
    /// ranges that need it (leaf cells, next_chain); the scatter output is fully overwritten. One stateless
    /// instance is enough. Frees go through the same allocator.
    [[no_unique_address]] Allocator<false, false> allocator;
};

}
