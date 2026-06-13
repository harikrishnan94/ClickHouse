#pragma once

#include <Interpreters/RadixHashJoin/Arena.h>
#include <Interpreters/RadixHashJoin/BuildSide.h>
#include <Interpreters/RadixHashJoin/KeyRefScatter.h>

#include <Common/Arena.h>

#include <atomic>
#include <base/types.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace DB::RadixJoin
{

/** One open-addressing, linear-probe hash table per leaf. A cell is laid out as
  *
  *     [ BuildRefList word (8 B) | packed key (key_width B) ]
  *
  * so the cell head is a `DB::BuildRefList` (see Interpreters/RowRefs.h): the unified ALL-join row-ref
  * value. A unique key holds the build row inline (the word IS the encoded singleton `BuildRef`); the
  * first duplicate of a key allocates a `BuildRefList::Batch` node from the owning build worker's arena
  * and the word becomes a count-tagged pointer to it. A probe match reads the cell word and either
  * emits the one inline ref directly (the common single-row fast path, NO extra load) or iterates the
  * list.
  *
  * Empty sentinel: the cell word is 0 (`BuildRefList::word == 0`). The cell array is carved from a
  * (non-zeroed) Arena and `memset` to 0 by the worker that owns the leaf, so every cell starts empty
  * with no separate init pass. (A valid inline ref always has bit 63 set — the singleton flag — so it
  * can never collide with the all-zero empty word.)
  *
  * Singleton fast path: a unique key's cell word is the encoded singleton `BuildRef` (bit 63 set), so a
  * probe emits it with NO node access — saving a likely LLC/DRAM miss for the common single-row keys
  * even in a build that also contains duplicates. The Batch node is allocated only when the first
  * duplicate of a key arrives.
  *
  * Bucket = low 32 bits of the key's hash; leaf = high bits. A well-mixed 64-bit hash has independent
  * halves, so the within-leaf bucket shares no bits with the leaf id and there is no
  * `total_bits + log2(buckets) <= 32` saturation. Build and probe recompute the identical hash, so the
  * bucket matches on both sides; the leaves carry key + ref only (no stored hash).
  */
struct LeafHT
{
    char * cells = nullptr;          /// cell array; stride = key_width + 8; memset to 0
    UInt64 num_buckets = 0;          /// power of two (0 == empty leaf); mask = num_buckets - 1
};

/// Cell stride for a key of `key_width` bytes (BuildRefList head word + key).
constexpr size_t leafCellBytes(size_t key_width) noexcept
{
    return sizeof(DB::BuildRefList) + key_width;
}

/// Bucket index in [0, num_buckets): the low bits of the hash's low word. `num_buckets` is a non-zero
/// power of two, so this is a single AND.
inline UInt64 leafBucket(UInt32 low_hash, UInt64 num_buckets) noexcept
{
    return static_cast<UInt64>(low_hash) & (num_buckets - 1);
}

/// Insert one build row (`key` / `ref`, bucketed by `low_hash`) into leaf `ht`. Returns whether the key
/// was ALREADY present (a duplicate); `false` means this was the first row of the key. The cell head is
/// a `DB::BuildRefList`: a first occurrence claims the cell and stores the ref inline (singleton); a
/// duplicate appends the ref to the list, allocating a Batch node from `arena` on the first duplicate.
/// `arena` is the calling build worker's own arena (single-writer per leaf, so no locking). Width-
/// templated so the key copy/compare are compile-time-sized.
template <size_t key_width>
inline bool leafInsert(LeafHT & ht, UInt32 low_hash, const void * key, BuildRef ref, DB::Arena & arena) noexcept
{
    static_assert(key_width >= 4 && key_width % 4 == 0 && key_width <= 64);
    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 mask = ht.num_buckets - 1;
    UInt64 pos = leafBucket(low_hash, ht.num_buckets) & mask;
    while (true)
    {
        char * cell = ht.cells + pos * stride;
        auto * list = reinterpret_cast<DB::BuildRefList *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (list->word == 0)
        {
            /// First occurrence: claim the cell, store the key, and insert the singleton ref.
            __builtin_memcpy_inline(cell + sizeof(DB::BuildRefList), key, key_width);
            list->insert(ref.word(), arena);
            return false;
        }
        if (__builtin_memcmp(cell + sizeof(DB::BuildRefList), key, key_width) == 0)
        {
            /// Duplicate key: append the ref to the list (allocates a Batch node on the first duplicate).
            list->insert(ref.word(), arena);
            return true;
        }
        pos = (pos + 1) & mask;
    }
}

/// All the built leaf tables; owns the arena backing the cells plus the per-worker arenas backing the
/// BuildRefList Batch nodes (all read-only for the whole probe).
struct LeafTables
{
    std::vector<LeafHT> leaves;      /// indexed by leaf id — the probe-side O(1) lookup vector
    /// Per-build-worker arenas holding the BuildRefList Batch nodes. One per worker (single-writer
    /// during build); their stable addresses must outlive the probe, hence they live here.
    std::vector<std::unique_ptr<DB::Arena>> build_arenas;
    /// Set during build the first time any key gets a duplicate row. Selects the grouped probe path.
    std::atomic<bool> any_duplicates{false};
    UInt64 num_rows = 0;
    Arena arena;                     /// owns the cells (read-only for the whole probe)

    LeafTables() = default;
    LeafTables(const LeafTables &) = delete;
    LeafTables & operator=(const LeafTables &) = delete;
    /// Move-only: `buildLeafTables` returns by value into `State::leaf_tables`. `std::atomic` is neither
    /// copyable nor movable, so the moves are spelled out (relaxed load/store; the build barrier already
    /// orders the concurrent build-time writes against the post-build read).
    LeafTables(LeafTables && other) noexcept
        : leaves(std::move(other.leaves))
        , build_arenas(std::move(other.build_arenas))
        , any_duplicates(other.any_duplicates.load(std::memory_order_relaxed))
        , num_rows(other.num_rows)
        , arena(std::move(other.arena))
    {
    }
    LeafTables & operator=(LeafTables && other) noexcept
    {
        leaves = std::move(other.leaves);
        build_arenas = std::move(other.build_arenas);
        any_duplicates.store(other.any_duplicates.load(std::memory_order_relaxed), std::memory_order_relaxed);
        num_rows = other.num_rows;
        arena = std::move(other.arena);
        return *this;
    }
};

/// Build every leaf table from a finished `LeafArrays`. Cell arrays are allocated, 0-initialised and
/// filled in parallel across the CoopPool workers; the BuildRefList Batch nodes are allocated from the
/// per-worker `build_arenas` (`num_workers` of them, one per CoopPool participant). Must run as the
/// CoopPool leader's body.
LeafTables buildLeafTables(
    const LeafArrays & leaf_arrays,
    UInt64 num_rows,
    size_t key_width,
    size_t num_workers,
    CoopPool & coord);

/// Probe `n` left rows (their full 64-bit hashes + packed keys) against the leaf tables, appending one
/// `(left_row, BuildRef)` per match to the output buffers (singleton keys emit one ref; multi-row keys
/// iterate the whole BuildRefList).
void collectMatches(
    size_t key_width,
    const LeafHT * leaves,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const UInt64 * hashes,
    const void * packed_keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs);

}
