#pragma once

#include <cstddef>
#include <vector>

namespace DB::RadixHash
{

/** Bump arena backed by 2 MiB-aligned slabs that are explicitly madvise(MADV_HUGEPAGE)-ed at
  * allocation (spec section 4.4). Backs the selector `pid` arrays and, in later phases, the
  * build-side scatter output (per-leaf key/ref arrays and the 16 B-cell leaf hash tables).
  *
  * Fail-open: if madvise fails (or THP is unsupported) the arena still works on 4 KiB pages with
  * correct results, only paying more TLB misses; the fail-open count is tracked. Many spans are
  * packed per slab to keep the allocation count low (anti-churn). One arena is owned per worker
  * thread, so it needs no internal synchronisation.
  *
  * Pointers returned by `alloc` are stable for the lifetime of the arena; all memory is released
  * when the arena is destroyed.
  */
class HugeArena
{
public:
    static constexpr size_t SLAB = 2 * 1024 * 1024; /// 2 MiB == x86 LG_HUGEPAGE, smallest THP unit

    HugeArena() = default;
    ~HugeArena();

    HugeArena(const HugeArena &) = delete;
    HugeArena & operator=(const HugeArena &) = delete;
    HugeArena(HugeArena && other) noexcept;
    HugeArena & operator=(HugeArena && other) noexcept;

    /// Returns a pointer aligned to `alignment` (a power of two, <= SLAB). The slab base is
    /// 2 MiB-aligned, so any such alignment is satisfiable.
    void * alloc(size_t bytes, size_t alignment);

    template <typename T>
    T * allocArray(size_t n)
    {
        return static_cast<T *>(alloc(n * sizeof(T), alignof(T)));
    }

    size_t hugePagesUsed() const { return huge_pages_used; }
    size_t hugePagesFailed() const { return huge_pages_failed; }
    size_t slabCount() const { return slabs.size(); }
    size_t bytesReserved() const;

private:
    struct Slab
    {
        void * base = nullptr;
        size_t size = 0;
        size_t used = 0;
    };

    std::vector<Slab> slabs;
    size_t huge_pages_used = 0;
    size_t huge_pages_failed = 0;

    void addSlab(size_t min_bytes);
    void freeAll() noexcept;
};

}
