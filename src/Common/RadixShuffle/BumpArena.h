#pragma once

#include <cstddef>
#include <vector>


namespace DB::RadixShuffle
{

/// Bump-pointer arena whose semantics match the reference benchmark's inline
/// `Arena` class in `radix_part_vs_memcpy.cpp`.
///
/// Key differences from `DB::Arena`:
///   - No pad-right SIMD padding guard-region.
///   - Exposes `reset()` which rewinds all slab cursors to their starts
///     without releasing memory.  This lets benchmark repetitions reuse the
///     same warm physical pages so allocation + page-fault overhead is paid
///     only on the first repetition, exactly as in the reference.
///
/// Key differences from `DB::RadixShuffle::Allocator/Handle`:
///   - Single-threaded, no sharded counters, no per-partition chunk chains.
///   - General-purpose bump allocator; callers decide layout.
///
/// Thread-safety: none.  Each benchmark worker thread owns its own `BumpArena`.
class BumpArena
{
public:
    static constexpr size_t kDefaultInitialBytes = 4096;
    static constexpr size_t kDefaultGrowthFactor = 2;
    static constexpr size_t kDefaultLinearThresholdBytes = 128ULL << 20; // 128 MiB

    explicit BumpArena(
        size_t initial_size = kDefaultInitialBytes,
        size_t growth_factor = kDefaultGrowthFactor,
        size_t linear_growth_threshold = kDefaultLinearThresholdBytes);

    ~BumpArena();

    BumpArena(const BumpArena &) = delete;
    BumpArena & operator=(const BumpArena &) = delete;
    BumpArena(BumpArena && other) noexcept;
    BumpArena & operator=(BumpArena &&) = delete;

    /// Allocate `size` bytes with no alignment constraint.
    /// Matches `DB::Arena::alloc`.
    [[nodiscard]] char * alloc(size_t size);

    /// Allocate `size` bytes with the given power-of-two `alignment`.
    /// Matches `DB::Arena::alignedAlloc`.
    [[nodiscard]] char * alignedAlloc(size_t size, size_t alignment);

    /// Typed aligned allocation.  Equivalent to `alignedAlloc(sizeof(T), alignof(T))`.
    template <typename T>
    [[nodiscard]] T * alloc()
    {
        return reinterpret_cast<T *>(alignedAlloc(sizeof(T), alignof(T)));
    }

    /// Rewind all slabs to their starts without releasing memory.
    /// All previously returned pointers are invalidated.
    /// Subsequent allocations reuse the same physical pages — no page faults.
    void reset();

    [[nodiscard]] size_t allocatedBytes() const noexcept { return allocated_; }
    [[nodiscard]] size_t usedBytes() const noexcept { return used_; }

private:
    size_t avail() const noexcept { return end_ ? static_cast<size_t>(end_ - cur_) : 0; }

    /// Advance to the next slab.  Reuses a pre-existing slab when available
    /// (after `reset()`); allocates a new one otherwise.
    void nextSlab(size_t min_size);

    struct Slab
    {
        char * ptr;
        size_t sz;
    };

    static constexpr size_t kNoSlab = static_cast<size_t>(-1);

    size_t initial_;
    size_t growth_;
    size_t linear_;
    size_t idx_ = kNoSlab; ///< Index of the currently active slab.
    char * cur_ = nullptr;
    char * end_ = nullptr;
    size_t allocated_ = 0;
    size_t used_ = 0;
    std::vector<Slab> slabs_;
};

} // namespace DB::RadixShuffle
