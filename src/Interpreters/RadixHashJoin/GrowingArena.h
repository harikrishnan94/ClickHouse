#pragma once

#include <base/types.h>

#include <cstddef>
#include <vector>

namespace DB::RadixHash
{

/** GrowingArena — a bump arena backed by anonymous `mmap` blocks (spec section 4.4). It backs the
  * selector `pid` arrays, the deferred-scatter output (per-leaf key/ref/hash arrays packed into one
  * contiguous block per leaf), and the leaf hash tables.
  *
  * Blocks grow geometrically: the first block is `INITIAL_BLOCK`, each new block doubles up to a
  * configurable cap `max_block` (default 8 MiB), after which all further blocks are cap-sized. A single
  * allocation larger than the cap gets its own exact (page-rounded) dedicated block, so **every
  * allocation is contiguous within one block** (the deferred scatter relies on each per-leaf combined
  * key+ref+hash block being contiguous).
  *
  * `trim()` releases the page-aligned unused tail of every block back to the OS with
  * `madvise(MADV_DONTNEED)` (wholly-unused trailing blocks are released in full); the mappings stay,
  * so the arena remains usable and any retouched page is re-faulted as zero. Use it after the
  * transient `pid` arena is consumed, and on the output arena's reserved-but-unused tail.
  *
  * Pointers returned by `alloc` are stable for the arena's lifetime; all memory is `munmap`-ed on
  * destruction. One arena is owned per worker (pid) or per result (output), so it needs no locking.
  *
  * Transparent huge pages (spec section 4.4): with `use_thp = true` every block is `2 MiB`-rounded
  * and `madvise(MADV_HUGEPAGE)`-ed (fail-open — on error the block still works on `4 KiB` pages).
  * THP is used for both the scatter output arena and the leaf hash table arena to reduce TLB pressure
  * during the random-write scatter and random-access HT lookups (benchmarked at 1.7× scatter speedup
  * at 100M rows). The `pid`/hash arenas use the default (`false`) — they are short-lived or
  * streaming-access and do not benefit. Each `madvise` success/failure is counted into
  * `RadixHashHugePagesUsed` / `RadixHashHugePagesFailed`.
  */
class GrowingArena
{
public:
    static constexpr size_t DEFAULT_MAX_BLOCK = 8 * 1024 * 1024; /// 8 MiB cap (configurable)
    static constexpr size_t INITIAL_BLOCK = 64 * 1024; /// first block size, doubles up to the cap
    static constexpr size_t HUGE_PAGE = 2 * 1024 * 1024; /// x86 2 MiB THP unit (block size/alignment when use_thp)

    explicit GrowingArena(size_t max_block_bytes = DEFAULT_MAX_BLOCK, bool use_thp = false);
    ~GrowingArena();

    GrowingArena(const GrowingArena &) = delete;
    GrowingArena & operator=(const GrowingArena &) = delete;
    GrowingArena(GrowingArena && other) noexcept;
    GrowingArena & operator=(GrowingArena && other) noexcept;

    /// Returns a pointer aligned to `alignment` (a power of two). The allocation is contiguous.
    void * alloc(size_t bytes, size_t alignment);

    template <typename T>
    T * allocArray(size_t n)
    {
        return static_cast<T *>(alloc(n * sizeof(T), alignof(T)));
    }

    /// madvise(MADV_DONTNEED) the page-aligned unused tail of every block (releases trailing blocks /
    /// over-reserved tails to the OS). The arena stays usable; retouched pages re-fault as zero.
    void trim() noexcept;

    /// Release a previously-allocated range back to the OS using madvise(MADV_DONTNEED). Only the
    /// page-aligned *interior* of `[range_start, range_start+bytes)` is released (start rounded UP,
    /// end rounded DOWN to page boundaries) so adjacent live allocations in the same page are never
    /// disturbed. The arena mapping stays; retouched pages re-fault as zero. Safe to call from
    /// multiple threads simultaneously as long as their ranges do not overlap.
    void releaseRange(const void * range_start, size_t bytes) noexcept;

    size_t blockCount() const { return blocks.size(); }
    size_t bytesReserved() const;
    size_t bytesUsed() const;

private:
    struct Block
    {
        char * base = nullptr;
        size_t size = 0;
        size_t used = 0;
    };

    std::vector<Block> blocks;
    size_t max_block;
    size_t next_block_size;
    bool thp = false;

    void addBlock(size_t min_bytes);
    void freeAll() noexcept;
};

}
