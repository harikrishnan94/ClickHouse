#pragma once

#include <Interpreters/RadixHashJoin/Arena.h>
#include <Interpreters/RadixHashJoin/BuildSide.h>
#include <Interpreters/RadixHashJoin/KeyRefScatter.h>

#include <base/types.h>

#include <cstddef>
#include <vector>

namespace DB::RadixJoin
{

/** One open-addressing, linear-probe hash table per leaf. A cell is laid out as
  *
  *     [ BuildRef head (8 B) | packed key (key_width B) ]
  *
  * so the cell stores the HEAD build reference of the key's match chain directly (not a leaf-local
  * index): a probe `find` returns a `BuildRef` that resolves the payload with no extra indirection.
  *
  * Empty sentinel: `head.row_no == INVALID_ROW`. The cell array is carved from a (non-zeroed) Arena
  * and `memset` to 0xFF by the worker that owns the leaf, so every cell starts as the all-0xFF empty
  * reference with no separate init pass.
  *
  * Singleton fast path: a cell HEAD steals the MSB of `block_no` (`SINGLETON_FLAG`) to mark "this key
  * has exactly one build row". The first insert sets it; the first duplicate clears it. On probe a
  * flagged head is emitted directly with NO chain access — saving a likely LLC/DRAM miss for the
  * common single-row keys even in a build that also contains duplicates. The marker lives ONLY on cell
  * heads; `leafFind` returns the head verbatim and the caller clears it before emitting, so every ref
  * outside the cells is flag-free.
  *
  * Many-to-many (JOIN ALL): `next_chain` is one flat `BuildRef` array shared by all leaves, indexed by
  * the build row's flat index (`block_base[block_no] + row_no`). `next_chain[flat(ref)]` is the NEXT
  * ref in that key's chain, or INVALID_ROW at the tail. It is allocated LAZILY on the first duplicate
  * anywhere — an all-unique build never allocates it and the probe never touches it.
  *
  * Bucket = low 32 bits of the key's hash; leaf = high bits. A well-mixed 64-bit hash has independent
  * halves, so the within-leaf bucket shares no bits with the leaf id and there is no
  * `total_bits + log2(buckets) <= 32` saturation. Build and probe recompute the identical hash, so the
  * bucket matches on both sides; the leaves carry key + ref only (no stored hash).
  */
struct LeafHT
{
    char * cells = nullptr;          /// cell array; stride = key_width + 8; memset to 0xFF
    UInt64 num_buckets = 0;          /// power of two (0 == empty leaf); mask = num_buckets - 1
    BuildRef * next_chain = nullptr; /// shared flat chain array (null until the first duplicate)
};

/// Cell stride for a key of `key_width` bytes (head BuildRef + key).
constexpr size_t leafCellBytes(size_t key_width) noexcept
{
    return sizeof(BuildRef) + key_width;
}

/// Bucket index in [0, num_buckets): the low bits of the hash's low word. `num_buckets` is a non-zero
/// power of two, so this is a single AND.
inline UInt64 leafBucket(UInt32 low_hash, UInt64 num_buckets) noexcept
{
    return static_cast<UInt64>(low_hash) & (num_buckets - 1);
}

/// The empty-cell / chain-tail sentinel ref: `row_no == INVALID_ROW`, flag-free. All-ones word, which
/// matches the 0xFF-initialised cells and chain slots. (`DB::BuildRef` has a user-provided constructor
/// and cannot be aggregate-initialised, so the sentinel is built from its word.)
inline BuildRef invalidRef() noexcept
{
    return BuildRef::fromWord(~UInt64(0));
}

/// Flat chain index of a build row: block_base[block_no] + row_no. Masks the singleton flag.
inline UInt64 leafFlat(BuildRef ref, const UInt64 * block_base) noexcept
{
    return block_base[ref.blockNo()] + ref.rowNo();
}

inline BuildRef markSingleton(BuildRef ref) noexcept
{
    ref.block_no |= SINGLETON_FLAG;
    return ref;
}
inline BuildRef clearSingleton(BuildRef ref) noexcept
{
    ref.block_no &= BLOCK_NO_MASK;
    return ref;
}
inline bool isSingleton(BuildRef ref) noexcept
{
    return (ref.block_no & SINGLETON_FLAG) != 0;
}

/// Insert one build row (`key` / `ref`, bucketed by `low_hash`) into leaf `ht`. Returns the previous
/// head ref (flag cleared) when the key already exists (caller threads it through `next_chain`), or the
/// INVALID_ROW sentinel on a first occurrence. Width-templated so the key copy/compare are
/// compile-time-sized.
template <size_t key_width>
inline BuildRef leafInsert(LeafHT & ht, UInt32 low_hash, const void * key, BuildRef ref) noexcept
{
    static_assert(key_width >= 4 && key_width % 4 == 0 && key_width <= 64);
    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 mask = ht.num_buckets - 1;
    UInt64 pos = leafBucket(low_hash, ht.num_buckets) & mask;
    while (true)
    {
        char * cell = ht.cells + pos * stride;
        auto * head = reinterpret_cast<BuildRef *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (head->row_no == INVALID_ROW)
        {
            /// First occurrence: store the key and mark the head as a singleton (one build row so far).
            __builtin_memcpy_inline(cell + sizeof(BuildRef), key, key_width);
            *head = markSingleton(ref);
            return invalidRef();
        }
        if (__builtin_memcmp(cell + sizeof(BuildRef), key, key_width) == 0)
        {
            /// Duplicate key: the new head is flag-free (the key is now multi-row); return the old head
            /// cleared of the marker so the caller links a flag-free ref into next_chain.
            const BuildRef old = clearSingleton(*head);
            *head = clearSingleton(ref);
            return old;
        }
        pos = (pos + 1) & mask;
    }
}

/// Find the head ref for `key` (bucketed by `low_hash`) in leaf `ht`, or INVALID_ROW on miss. The
/// returned head may carry the singleton marker; the caller checks it. Width-templated.
template <size_t key_width>
inline BuildRef leafFind(const LeafHT & ht, UInt32 low_hash, const void * key) noexcept
{
    static_assert(key_width >= 4 && key_width % 4 == 0 && key_width <= 64);
    if (ht.num_buckets == 0)
        return invalidRef();
    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 mask = ht.num_buckets - 1;
    UInt64 pos = leafBucket(low_hash, ht.num_buckets) & mask;
    while (true)
    {
        const char * cell = ht.cells + pos * stride;
        const BuildRef head = *reinterpret_cast<const BuildRef *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (head.row_no == INVALID_ROW)
            return invalidRef();
        if (__builtin_memcmp(cell + sizeof(BuildRef), key, key_width) == 0)
            return head;
        pos = (pos + 1) & mask;
    }
}

/// All the built leaf tables plus the shared chain array; owns the arena backing both.
struct LeafTables
{
    std::vector<LeafHT> leaves;      /// indexed by leaf id — the probe-side O(1) lookup vector
    BuildRef * next_chain = nullptr; /// null for an all-unique build
    UInt64 num_rows = 0;
    Arena arena;                     /// owns the cells + next_chain (read-only for the whole probe)
};

/// Build every leaf table and the shared chain array from a finished `LeafArrays`. Cell arrays are
/// allocated, 0xFF-initialised and filled in parallel across the CoopPool workers; the chain array is
/// allocated lazily on the first duplicate. Must run as the CoopPool leader's body.
LeafTables buildLeafTables(
    const LeafArrays & leaf_arrays,
    const std::vector<UInt64> & block_base,
    UInt64 num_rows,
    size_t key_width,
    CoopPool & coord);

/// Probe `n` left rows (their full 64-bit hashes + packed keys) against the leaf tables, appending one
/// `(left_row, BuildRef)` per match (chain order) to the output buffers. `has_duplicates` selects a
/// simplified loop (no chain access at all) when the build had no duplicate keys.
void collectMatches(
    size_t key_width,
    bool has_duplicates,
    const LeafHT * leaves,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const UInt64 * block_base,
    const UInt64 * hashes,
    const void * packed_keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs);

}
