#pragma once

#include <Interpreters/RadixHashJoin/BuildStore.h>
#include <Interpreters/RadixHashJoin/GrowingArena.h>
#include <Common/RadixShuffle/Scatter.h>

#include <base/types.h>
#include <base/unaligned.h>

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
  * (`global_blocks[ref.block_no][ref.row_no]`, `row_no` 0-based).
  *
  * **Empty sentinel = `ref.row_no == INVALID_ROW` (`0xFFFFFFFF`).** The cell array is carved from a
  * jemalloc-backed `GrowingArena` (no zero-fill), so `buildLeafHashTables` `memset`s each leaf's cells to
  * `0xFF` — in parallel, on the worker that owns the leaf — before filling it, leaving every cell as the
  * all-`0xFF` empty ref (spec section 5.6, invariant 9). The key bytes are read only after a slot is
  * found non-empty.
  *
  * **Many-to-many (JOIN ALL) via `next_chain`.** `next_chain` is a single 1D flat array of `BuildRef`,
  * one slot per build row (shared by every leaf). `next_chain[flat(ref)]` holds the NEXT BuildRef in
  * that key's chain, or the `INVALID_ROW` sentinel for the tail. `flat(ref) = block_base[ref.block_no] +
  * ref.row_no` — the per-block prefix-summed row offset (`BuildStore::blockBase()`).
  *   build insert:  if key present -> next_chain[flat(ref)] = old_head; cell.ref = ref;  (prepend)
  *                  else (first)   -> cell.ref = ref;  next_chain[flat(ref)] stays the 0xFF-init tail
  *   probe:         cur = find(h, key); while (cur.row_no != INVALID_ROW) { emit(cur); cur = next_chain[flat(cur)]; }
  *
  * **Bucket = `bucketHash` of the packed key** — a cheap, non-linear, CRC-independent hash with
  * well-mixed low bits, computed from the key bytes directly (NOT the CRC32C routing hash). Leaf routing
  * still uses the CRC32C `computeHashInto` top bits; the bucket uses a separate full-32-bit-entropy hash
  * so the within-leaf bucket no longer shares bits with the leaf id (this lifts the ~2^31-row saturation
  * the single shared hash imposed). The identical `bucketHash` runs on build (fill) and probe.
  */
struct LeafHT
{
    char * cells = nullptr; /// cell array; stride = key_width + 8; carved from GrowingArena, memset to 0xFF
    UInt64 num_buckets = 0; /// power of two (0 for an empty leaf); mask = num_buckets - 1
    RadixShuffle::BuildRef * next_chain = nullptr; /// shared 1D flat BuildRef array (one slot per build row)
};
static_assert(sizeof(LeafHT) == 24, "LeafHT must be exactly 24 bytes (spec section 5.6)");

/// Cell stride for a key of `key_width` bytes (head BuildRef + key).
constexpr size_t leafCellBytes(size_t key_width) noexcept
{
    return sizeof(RadixShuffle::BuildRef) + key_width;
}

/// Independent within-leaf bucket hash of the packed key (`key_width` a multiple of 4 in [4, 64]).
///
/// A cheap Murmur-style multiply-fold over the key, NON-LINEAR and CRC32C-independent, with the
/// avalanche finalizer guaranteeing well-mixed LOW bits — which is exactly what `leafBucket` consumes
/// (`h & mask`). Leaf routing keeps using the CRC32C `computeHashInto` top bits; the bucket consumes
/// this disjoint full-32-bit-entropy hash instead of low CRC bits, so the bucket no longer shares any
/// hash bits with the leaf id. That removes the `total_bits + log2(num_buckets) <= 32` constraint the
/// single shared hash imposed (saturation at ~2^31 rows), because the bucket now has the full 32 bits of
/// within-leaf entropy regardless of how many top bits the leaf routing consumed. CRC32C is linear, so a
/// reseeded CRC cannot give an independent hash; this multiply-fold can. The identical function runs on
/// build (fill) and probe, so the bucket is the same on both sides. Width-templated so the fold fully
/// unrolls (no loop, no tail branch at runtime) for the compile-time key width.
template <size_t key_width>
inline UInt32 bucketHash(const char * key) noexcept
{
    static_assert(key_width >= 4 && key_width % 4 == 0 && key_width <= 64);
    UInt64 acc = 0x9E3779B97F4A7C15ULL;
    size_t i = 0;
    for (; i + 8 <= key_width; i += 8)
    {
        UInt64 w = unalignedLoad<UInt64>(key + i);
        w *= 0xFF51AFD7ED558CCDULL;
        w ^= w >> 33;
        acc ^= w;
        acc *= 0xC4CEB9FE1A85EC53ULL;
    }
    if constexpr (key_width % 8 != 0)
    {
        /// 4-byte tail (key_width is a multiple of 4, so the only remainder is a single 4-byte word).
        UInt64 w = unalignedLoad<UInt32>(key + i);
        w *= 0xFF51AFD7ED558CCDULL;
        w ^= w >> 33;
        acc ^= w;
        acc *= 0xC4CEB9FE1A85EC53ULL;
    }
    acc ^= acc >> 33;
    return static_cast<UInt32>(acc); /// low 32, fully mixed
}

/// Bucket index in [0, num_buckets) (spec section 5.6). `h` is the `bucketHash` of the key (full 32-bit
/// entropy, well-mixed low bits), so the bucket is indexed directly with its low bits. `num_buckets` is
/// a non-zero power of two, so this is a single AND.
inline UInt64 leafBucket(UInt32 h, UInt64 num_buckets) noexcept
{
    return static_cast<UInt64>(h) & (num_buckets - 1);
}

/// Flat next_chain slot of a build row: block_base[block_no] + row_no (row_no is 0-based).
inline UInt64 leafFlat(RadixShuffle::BuildRef ref, const UInt64 * block_base) noexcept
{
    return block_base[ref.block_no] + ref.row_no;
}

/// Insert one build row (`key`/`ref`, keyed on its 32-bit hash `h`) into leaf `ht`. Returns the old
/// head `BuildRef` when a duplicate key is found (caller threads it through `next_chain`), or the
/// `INVALID_ROW` sentinel `BuildRef` for a first occurrence (`next_chain` not touched).
///
/// Returning `BuildRef` (8 B) instead of a bool+out-parameter lets the compiler keep the result in
/// a register and avoids zero-initialising a stack slot on every call.  Width-templated so the key
/// copy/compare are `__builtin_memcpy_inline` / `__builtin_memcmp` of a compile-time size.
template <size_t key_width>
inline RadixShuffle::BuildRef leafInsert(
    LeafHT & ht, UInt32 h, const void * key, RadixShuffle::BuildRef ref) noexcept
{
    static_assert(key_width >= 4 && key_width % 4 == 0 && key_width <= 64);
    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 mask = ht.num_buckets - 1;
    UInt64 pos = leafBucket(h, ht.num_buckets) & mask;
    while (true)
    {
        char * cell = ht.cells + pos * stride;
        auto * head = reinterpret_cast<RadixShuffle::BuildRef *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (head->row_no == RadixShuffle::INVALID_ROW)
        {
            /// First occurrence: store key and head directly.
            __builtin_memcpy_inline(cell + sizeof(RadixShuffle::BuildRef), key, key_width);
            *head = ref;
            return RadixShuffle::BuildRef{RadixShuffle::INVALID_ROW, RadixShuffle::INVALID_ROW};
        }
        if (__builtin_memcmp(cell + sizeof(RadixShuffle::BuildRef), key, key_width) == 0)
        {
            /// Duplicate key: return the old head so the caller can thread it through next_chain.
            const RadixShuffle::BuildRef old = *head;
            *head = ref;
            return old;
        }
        pos = (pos + 1) & mask;
    }
}

/// Find the head BuildRef for `key` (keyed on its 32-bit hash `h`) in leaf `ht`, or the INVALID_ROW
/// sentinel on miss. Width-templated (compile-time key compare). The caller walks `next_chain` from it.
template <size_t key_width>
inline RadixShuffle::BuildRef leafFind(const LeafHT & ht, UInt32 h, const void * key) noexcept
{
    static_assert(key_width >= 4 && key_width % 4 == 0 && key_width <= 64);
    if (ht.num_buckets == 0)
        return RadixShuffle::BuildRef{RadixShuffle::INVALID_ROW, RadixShuffle::INVALID_ROW};
    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 mask = ht.num_buckets - 1;
    UInt64 pos = leafBucket(h, ht.num_buckets) & mask;
    while (true)
    {
        const char * cell = ht.cells + pos * stride;
        const RadixShuffle::BuildRef head = *reinterpret_cast<const RadixShuffle::BuildRef *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (head.row_no == RadixShuffle::INVALID_ROW)
            return RadixShuffle::BuildRef{RadixShuffle::INVALID_ROW, RadixShuffle::INVALID_ROW};
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

/// Build all leaf hash tables and the shared next_chain from a finished `LeafArrays` plus the per-block
/// `block_base` prefix sum. The within-leaf bucket is `bucketHash` of each key, recomputed from
/// `la.key_base` directly (no per-row hash is read). Cell arrays and next_chain are carved from one jemalloc-backed
/// `GrowingArena`; every per-leaf cell array is allocated, zeroed (`memset`), and filled by the worker
/// that owns that leaf (via `coord.parallelFor`), and `next_chain` is zeroed in parallel — so all the
/// allocation and zeroing is spread across the build threads. `key_width` selects the templated insert
/// path (a multiple of 4 in [4, 64]).
/// Must be called only by the leader inside a CoopPool::run body.
LeafHashTables buildLeafHashTables(
    const LeafArrays & la,
    const std::vector<UInt64> & block_base,
    UInt64 num_rows,
    size_t key_width,
    CoopPool & coord);

/// Probe `n` left rows against the leaf tables, collecting every match as a (left_row, BuildRef) pair
/// in `out_left_rows` / `out_refs` (grouped by left row, chain order). For row j: leaf =
/// total_bits ? (hashes[j] >> leaf_shift) : 0 (the CRC32C routing hash); the bucket within that leaf is
/// `bucketHash(packed_keys + j*key_width)` (the independent key hash), so head = find(bucketHash(key),
/// key); then, when `has_duplicates` is true, walk next_chain to the tail. `hashes` carries ONLY the
/// routing hash now (the bucket is recomputed from the key on both build and probe).
///
/// `has_duplicates` — whether the build contained any duplicate keys (i.e. `next_chain != nullptr`
/// in `LeafHashTables`). When false, `collectMatches` uses a simplified inner loop that emits
/// exactly one ref per hit with no `next_chain` access at all, saving the LLC/DRAM miss per row.
/// Dispatched on both `key_width` and `has_duplicates` to the templated `collectMatchesT`.
void collectMatches(
    size_t key_width,
    bool has_duplicates,
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
