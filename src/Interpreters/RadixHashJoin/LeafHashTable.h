#pragma once

#include <Interpreters/RadixHashJoin/BuildStore.h>
#include <Interpreters/RadixHashJoin/GrowingArena.h>
#include <Common/RadixShuffle/Scatter.h>
#include <Common/ThreadPool_fwd.h>

#include <base/types.h>

#include <bit>
#include <cstddef>
#include <vector>

namespace DB::RadixHash
{

/** Leaf hash table for the radix hash join (spec sections 5.4, 5.6; phase P4).
  *
  * One open-addressing, linear-probe table per leaf. A **cell** is laid out as
  *
  *     offset 0:           BuildRef ref;      // 8 B — the HEAD BuildRef of this key's match chain
  *     offset 8:           key bytes;         // key_width bytes — the packed key, stored verbatim
  *
  * so a cell is `key_width + 8` bytes. The cell stores the head BuildRef **directly** (not a leaf-local
  * index), so a probe `find` returns a BuildRef that resolves the payload with no extra indirection
  * (`global_blocks[ref.block_no][ref.row_no - 1]`).
  *
  * **Empty sentinel = the all-zero cell** (`ref.row_no == 0`; `row_no` is 1-based so a real entry is
  * never 0). The cell array is carved from a `GrowingArena` (`mmap(MAP_ANONYMOUS)`) whose pages are
  * zero on first touch, so a freshly carved table is already a fully initialised empty table — **no
  * memset / init pass** (spec section 5.6, invariant 9). The key bytes are read only after a slot is
  * found non-empty.
  *
  * **Many-to-many (JOIN ALL) via `next_chain`.** `next_chain` is a single 1D flat array of `BuildRef`,
  * one slot per build row (shared by every leaf). `next_chain[flat(ref)]` holds the NEXT BuildRef in
  * that key's chain, or `BuildRef{0,0}` for the tail. `flat(ref) = block_base[ref.block_no] +
  * (ref.row_no - 1)` — the per-block prefix-summed row offset (`BuildStore::blockBase()`).
  *   build insert:  if key present -> next_chain[flat(ref)] = old_head; cell.ref = ref;  (prepend)
  *                  else (first)   -> cell.ref = ref;  next_chain[flat(ref)] stays {0,0} (zero-init)
  *   probe:         cur = find(h, key); while (cur.row_no) { emit(cur); cur = next_chain[flat(cur)]; }
  *
  * **Bucket = Fibonacci-mix of the 32-bit `IColumn::computeHashInto` hash** (the same hash used for
  * leaf routing; spec section 5.6, tree fact D2). The identical function runs on build and probe.
  */
struct LeafHT
{
    char * cells = nullptr; /// cell array; stride = key_width + 8; carved from GrowingArena (zero-init)
    UInt64 num_buckets = 0; /// power of two (0 for an empty leaf); mask = num_buckets - 1
    RadixShuffle::BuildRef * next_chain = nullptr; /// shared 1D flat BuildRef array (one slot per build row)
};
static_assert(sizeof(LeafHT) == 24, "LeafHT must be exactly 24 bytes (spec section 5.6)");

/// Cell stride for a key of `key_width` bytes (head BuildRef + key).
constexpr size_t leafCellBytes(size_t key_width) noexcept
{
    return sizeof(RadixShuffle::BuildRef) + key_width;
}

/// Bucket index in [0, num_buckets): Fibonacci-mix the 32-bit hash, take the top log2(num_buckets)
/// bits (spec section 5.6). `num_buckets` is a non-zero power of two. The multiply is a 32-bit IMUL.
inline UInt64 leafBucket(UInt32 h, UInt64 num_buckets) noexcept
{
    const UInt32 mixed = h * 0x9E3779B9u; /// 2^32 * (golden ratio); spreads entropy into the high bits
    const unsigned log2_buckets = std::countr_zero(num_buckets);
    const unsigned shift = log2_buckets >= 32 ? 0u : 32u - log2_buckets;
    return static_cast<UInt64>(mixed >> shift);
}

/// Flat next_chain slot of a build row: block_base[block_no] + row_no - 1 (row_no is 1-based).
inline UInt64 leafFlat(RadixShuffle::BuildRef ref, const UInt64 * block_base) noexcept
{
    return block_base[ref.block_no] + (ref.row_no - 1);
}

/// Insert one build row (`key`/`ref`, keyed on its 32-bit hash `h`) into leaf `ht`. Linear probing
/// from the bucket; on an empty slot write the key + head; on a key match prepend `ref` as the new
/// head and thread the old head through `next_chain`. Width-templated so the key copy/compare are
/// `__builtin_memcpy_inline` / `__builtin_memcmp` of a compile-time size (no runtime memcpy).
template <size_t key_width>
inline void leafInsert(
    LeafHT & ht, UInt32 h, const void * key, RadixShuffle::BuildRef ref, const UInt64 * block_base) noexcept
{
    static_assert(key_width >= 4 && key_width % 4 == 0 && key_width <= 64);
    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 mask = ht.num_buckets - 1;
    UInt64 pos = leafBucket(h, ht.num_buckets) & mask;
    while (true)
    {
        char * cell = ht.cells + pos * stride;
        auto * head = reinterpret_cast<RadixShuffle::BuildRef *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (head->row_no == 0)
        {
            /// First occurrence: store key + head. next_chain[flat(ref)] is already {0,0} (zero-init tail).
            __builtin_memcpy_inline(cell + sizeof(RadixShuffle::BuildRef), key, key_width);
            *head = ref;
            return;
        }
        if (__builtin_memcmp(cell + sizeof(RadixShuffle::BuildRef), key, key_width) == 0)
        {
            /// Duplicate key: prepend ref as the new head, old head becomes ref's chain successor.
            ht.next_chain[leafFlat(ref, block_base)] = *head;
            *head = ref;
            return;
        }
        pos = (pos + 1) & mask;
    }
}

/// Find the head BuildRef for `key` (keyed on its 32-bit hash `h`) in leaf `ht`, or `{0,0}` on miss.
/// Width-templated (compile-time key compare). The caller walks `next_chain` from the returned head.
template <size_t key_width>
inline RadixShuffle::BuildRef leafFind(const LeafHT & ht, UInt32 h, const void * key) noexcept
{
    static_assert(key_width >= 4 && key_width % 4 == 0 && key_width <= 64);
    if (ht.num_buckets == 0)
        return RadixShuffle::BuildRef{0, 0};
    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 mask = ht.num_buckets - 1;
    UInt64 pos = leafBucket(h, ht.num_buckets) & mask;
    while (true)
    {
        const char * cell = ht.cells + pos * stride;
        const RadixShuffle::BuildRef head = *reinterpret_cast<const RadixShuffle::BuildRef *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (head.row_no == 0)
            return RadixShuffle::BuildRef{0, 0};
        if (__builtin_memcmp(cell + sizeof(RadixShuffle::BuildRef), key, key_width) == 0)
            return head;
        pos = (pos + 1) & mask;
    }
}

/// The built leaf hash tables + the shared next_chain, owning the arena that backs both.
struct LeafHashTables
{
    std::vector<LeafHT> leaves; /// one per leaf (indexed by leaf id) — the probe-side O(1) lookup vector
    RadixShuffle::BuildRef * next_chain = nullptr; /// flat array of `num_rows` BuildRefs (shared by all leaves)
    UInt64 num_rows = 0; /// total build rows (== next_chain length)
    GrowingArena arena; /// owns the cell arrays + next_chain (read-only for the whole probe phase)
};

/// Build all leaf hash tables and the shared next_chain from a finished `LeafArrays` (which must have
/// been produced with `with_leaf_hash = true`, so `la.hash_base` is populated) plus the per-block
/// `block_base` prefix sum. Cell arrays and next_chain are carved from one `GrowingArena` (THP-backed
/// when `use_thp`); the carve is O(num_leaves) single-threaded (NC gate), the fill is work-stolen
/// across `num_threads` workers drawn from `pool` (PB gate). `key_width` selects the templated insert
/// path (a multiple of 4 in [4, 64]).
LeafHashTables buildLeafHashTables(
    const LeafArrays & la,
    const std::vector<UInt64> & block_base,
    UInt64 num_rows,
    size_t key_width,
    ThreadPool & pool,
    size_t num_threads,
    bool use_thp);

/// Probe `n` left rows against the leaf tables, collecting every match as a (left_row, BuildRef) pair
/// in `out_left_rows` / `out_refs` (grouped by left row, chain order). For row j: leaf =
/// total_bits ? (hashes[j] >> leaf_shift) : 0; head = find(hashes[j], packed_keys + j*key_width);
/// then walk next_chain to the tail. Dispatched on `key_width` to the templated `leafFind`.
void collectMatches(
    size_t key_width,
    const LeafHT * leaves,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const UInt64 * block_base,
    const UInt32 * hashes,
    const void * packed_keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<RadixShuffle::BuildRef> & out_refs);

}
