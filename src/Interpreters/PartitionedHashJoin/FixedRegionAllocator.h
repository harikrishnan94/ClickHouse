#pragma once

#include <Common/HashTable/HashTableAllocator.h>

#include <atomic>
#include <cstring>

namespace DB
{

/** Hash-table allocator that carves a map's buffer out of a pre-sized region of the single
  * contiguous slab backing all leaf hash tables of one `PartitionedHashJoin` build (requirement:
  * ONE allocation per build, performed right before the leaf builds).
  *
  * A leaf-build worker arms the calling thread with the leaf's region and constructs the map;
  * the map's single buffer allocation adopts the region, zeroes exactly the requested bytes
  * (the slab itself is allocated unzeroed), and the arming is cleared. The partition plan
  * predicts the buffer size through the map's own grower math, so the requested size equals the
  * region size; a mismatch or any later growth (a distinct-estimate shortfall) falls back to the
  * heap - correct, counted through the region's `heap_fallbacks` counter, never silent.
  *
  * A map constructed without an armed region (e.g. the fixed-size `key8`/`key16` maps) behaves
  * exactly like one using the plain `HashTableAllocator`.
  */
class FixedRegionAllocator
{
public:
    struct Region
    {
        char * begin = nullptr;
        size_t bytes = 0;
        /// Build-lifetime counters (owned by the join) recording carve/fallback behavior.
        std::atomic<UInt64> * carves = nullptr;
        std::atomic<UInt64> * heap_fallbacks = nullptr;
    };

    /// Arm the calling thread: the next buffer allocation on this thread adopts the region.
    static void armRegion(const Region & region) { pending_region = region; }

    void * alloc(size_t size)
    {
        if (pending_region.begin)
        {
            const Region region = pending_region;
            pending_region = {};
            fallback_counter = region.heap_fallbacks;
            if (size <= region.bytes)
            {
                region_begin = region.begin;
                /// Zero exactly this leaf's buffer, right before the map fills it: the slab is
                /// allocated unzeroed and pages are first-touched here, cache-hot for the fill.
                memset(region_begin, 0, size);
                if (region.carves)
                    region.carves->fetch_add(1, std::memory_order_relaxed);
                return region_begin;
            }
            /// The predicted buffer size does not cover the request: fail over to the heap.
            if (fallback_counter)
                fallback_counter->fetch_add(1, std::memory_order_relaxed);
        }
        return heap.alloc(size);
    }

    void free(void * buf, size_t size)
    {
        /// A buffer carved from the slab is released with the slab as a whole.
        if (buf == region_begin)
            return;
        heap.free(buf, size);
    }

    void * realloc(void * buf, size_t old_size, size_t new_size)
    {
        if (buf != region_begin)
            return heap.realloc(buf, old_size, new_size);

        /// Growth out of the region (the map resized past its reserve): move to the heap.
        /// `heap.alloc` zero-fills, so the bytes past `old_size` satisfy the hash table's
        /// cleared-memory expectation. The region's memory is released with the slab.
        if (fallback_counter)
            fallback_counter->fetch_add(1, std::memory_order_relaxed);
        void * new_buf = heap.alloc(new_size);
        memcpy(new_buf, buf, old_size);
        return new_buf;
    }

private:
    static thread_local Region pending_region;

    /// The adopted region's buffer (nullptr when this map's buffer lives on the heap).
    char * region_begin = nullptr;
    std::atomic<UInt64> * fallback_counter = nullptr;

    HashTableAllocator heap;
};

inline thread_local FixedRegionAllocator::Region FixedRegionAllocator::pending_region;

}
