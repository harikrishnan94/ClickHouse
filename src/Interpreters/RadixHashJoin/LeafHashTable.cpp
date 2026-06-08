#include <Interpreters/RadixHashJoin/LeafHashTable.h>

#include <Common/Exception.h>
#include <Common/ProfileEvents.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <mutex>

namespace ProfileEvents
{
extern const Event RadixHashBuildHTMicroseconds;
}

namespace DB
{
namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}
}

namespace DB::RadixHash
{

using RadixShuffle::BuildRef;

namespace
{

/// Shared state passed to fillLeafT for lazy next_chain allocation.
///
/// On the first duplicate found by any leaf-fill worker, that worker allocates and 0xFF-initialises
/// all `num_rows` BuildRef slots (so every chain tail reads INVALID_ROW without explicit writes),
/// stores the pointer via `nc` (release), and the other workers blocking on `call_once` then acquire
/// it.  All subsequent duplicate workers take the fast path (non-null `nc` load).
///
/// For all-unique builds `acquire` is never called → next_chain stays nullptr.
struct LazyChainState
{
    std::once_flag nc_once;
    std::atomic<BuildRef *> nc{nullptr};
    std::function<BuildRef *(UInt64)> alloc_fn; /// allocates AND 0xFF-initialises num_rows BuildRef slots
    UInt64 num_rows = 0;

    BuildRef * acquire()
    {
        BuildRef * p = nc.load(std::memory_order_acquire);
        if (p != nullptr) [[likely]]
            return p;
        std::call_once(nc_once, [this]
        {
            BuildRef * fresh = alloc_fn(num_rows);
            nc.store(fresh, std::memory_order_release);
        });
        return nc.load(std::memory_order_acquire);
    }
};

/// Fill one leaf table by inserting all of its scattered (key, ref) rows. The within-leaf bucket is
/// `bucketHash` of the key, recomputed here from `la.key_base[leaf]` directly (the routing hash is no
/// longer consulted at fill). Software-pipelined write-prefetch of the next row's bucket cell (spec
/// section 5.6, build: __builtin_prefetch RW=1).
///
/// On a duplicate key `leafInsert` returns the old head. The worker acquires `next_chain` lazily
/// (first duplicate triggers `lcs.acquire()` which allocates and 0xFF-initialises the full array so
/// every chain tail is automatically INVALID_ROW), then threads: nc[flat(new_ref)] = old_head.
template <size_t key_width>
void fillLeafT(LeafHT & ht, const LeafArrays & la, size_t leaf, const UInt64 * block_base,
    LazyChainState & lcs)
{
    const UInt64 rows = la.leaf_rows[leaf];
    if (rows == 0)
        return;

    const auto * keys = static_cast<const char *>(la.key_base[leaf]);
    const BuildRef * refs = la.ref_base[leaf];

    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 num_buckets = ht.num_buckets;
    const UInt64 mask = num_buckets - 1;

    /// Software-pipelined GROUP prefetch (machine-generic). Each step (1) computes the home buckets of
    /// the NEXT group of rows in a tight, auto-vectorizable loop, (2) issues that group's home-cell L1
    /// (locality 3) prefetches back-to-back, then (3) inserts the CURRENT group — whose cells were
    /// prefetched a full group earlier. The back-to-back burst exposes group-wide memory-level
    /// parallelism and gives each insert a full group of prefetch lead, so the random home-cell load
    /// hits L1. The win comes from this structure rather than a hardware-tuned scalar prefetch distance:
    /// `group` over-provisions the memory parallelism (a core only services up to its line-fill-buffer
    /// count, ~10-16, and drops the rest — so smaller cores degrade gracefully), and `group * cell-size`
    /// (≈ 1 KiB here) stays well within L1 on any machine. Always prefetches at least as far as, and to a
    /// closer cache level than, the previous distance-16 / L2 plan, so it cannot regress below it.
    constexpr size_t group = 64;
    UInt64 pf_pos[group];

    auto compute_buckets = [&](UInt64 base, size_t count)
    {
        for (size_t i = 0; i < count; ++i) /// tight, auto-vectorizable
            pf_pos[i] = leafBucket(bucketHash<key_width>(keys + (base + i) * key_width), num_buckets) & mask;
    };
    auto prefetch_burst = [&](size_t count)
    {
        for (size_t i = 0; i < count; ++i)
            __builtin_prefetch(ht.cells + pf_pos[i] * stride, /*rw=*/1, /*locality=*/3);
    };
    auto insert_row = [&](UInt64 r)
    {
        const BuildRef old_head = leafInsert<key_width>(ht, bucketHash<key_width>(keys + r * key_width), keys + r * key_width, refs[r]);
        if (old_head.row_no != RadixShuffle::INVALID_ROW)
        {
            /// Duplicate key: lazily acquire (or reuse already-acquired) next_chain, then thread.
            BuildRef * nc = lcs.acquire();
            ht.next_chain = nc; /// make the pointer visible to this leaf's probe path
            nc[leafFlat(refs[r], block_base)] = old_head;
        }
    };

    /// Prime: prefetch the first group's home cells.
    {
        const size_t prime = static_cast<size_t>(std::min<UInt64>(group, rows));
        compute_buckets(0, prime);
        prefetch_burst(prime);
    }

    UInt64 row = 0;
    for (; row + group <= rows; row += group)
    {
        const UInt64 next = row + group;
        const size_t to_prefetch = next < rows ? static_cast<size_t>(std::min<UInt64>(group, rows - next)) : 0;
        compute_buckets(next, to_prefetch);
        prefetch_burst(to_prefetch);
        for (size_t i = 0; i < group; ++i)
            insert_row(row + i);
    }
    /// Tail (already prefetched by the previous iteration's burst).
    for (; row < rows; ++row)
        insert_row(row);
}

void fillLeafDispatch(size_t key_width, LeafHT & ht, const LeafArrays & la, size_t leaf,
    const UInt64 * block_base, LazyChainState & lcs)
{
    switch (key_width)
    {
        case 4:  fillLeafT<4>(ht, la, leaf, block_base, lcs);  return;
        case 8:  fillLeafT<8>(ht, la, leaf, block_base, lcs);  return;
        case 12: fillLeafT<12>(ht, la, leaf, block_base, lcs); return;
        case 16: fillLeafT<16>(ht, la, leaf, block_base, lcs); return;
        case 20: fillLeafT<20>(ht, la, leaf, block_base, lcs); return;
        case 24: fillLeafT<24>(ht, la, leaf, block_base, lcs); return;
        case 28: fillLeafT<28>(ht, la, leaf, block_base, lcs); return;
        case 32: fillLeafT<32>(ht, la, leaf, block_base, lcs); return;
        case 36: fillLeafT<36>(ht, la, leaf, block_base, lcs); return;
        case 40: fillLeafT<40>(ht, la, leaf, block_base, lcs); return;
        case 44: fillLeafT<44>(ht, la, leaf, block_base, lcs); return;
        case 48: fillLeafT<48>(ht, la, leaf, block_base, lcs); return;
        case 52: fillLeafT<52>(ht, la, leaf, block_base, lcs); return;
        case 56: fillLeafT<56>(ht, la, leaf, block_base, lcs); return;
        case 60: fillLeafT<60>(ht, la, leaf, block_base, lcs); return;
        case 64: fillLeafT<64>(ht, la, leaf, block_base, lcs); return;
        default:
            throw Exception(
                ErrorCodes::LOGICAL_ERROR, "RadixHashJoin leaf HT: unsupported key width {} (multiple of 4 in [4,64])", key_width);
    }
}

/// Collect every (left_row, BuildRef) match for `n` probe rows against the leaf tables.
///
/// `has_chain` — compile-time flag: false for all-unique builds (no `next_chain` access at all,
/// one emit per hit), true for builds with duplicates (standard chain walk via `next_chain`).
/// Dispatched from `collectMatches` which picks the right instantiation based on `has_duplicates`.
template <size_t key_width, bool has_chain>
void collectMatchesT(
    const LeafHT * leaves,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const UInt64 * block_base,
    const UInt32 * hashes,
    const char * packed_keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
    constexpr size_t stride = leafCellBytes(key_width);
    /// Prefetch distance 16: without the next_chain DRAM stall the probe loop runs at ~5-10 ns per
    /// iteration; distance 8 (~40-80 ns lead) no longer covers DRAM latency for the cells array.
    constexpr size_t prefetch_distance = 16;

    for (size_t row = 0; row < n; ++row)
    {
        if (row + prefetch_distance < n)
        {
            /// Leaf from the CRC32C routing hash (top bits); bucket from the independent key hash.
            const size_t prefetch_leaf = total_bits ? (hashes[row + prefetch_distance] >> leaf_shift) : 0;
            const LeafHT & prefetch_ht = leaves[prefetch_leaf];
            if (prefetch_ht.num_buckets != 0)
            {
                const UInt32 prefetch_bucket_hash = bucketHash<key_width>(packed_keys + (row + prefetch_distance) * key_width);
                const UInt64 prefetch_pos = leafBucket(prefetch_bucket_hash, prefetch_ht.num_buckets) & (prefetch_ht.num_buckets - 1);
                __builtin_prefetch(prefetch_ht.cells + prefetch_pos * stride, /*rw=*/0, /*locality=*/1);
            }
        }

        /// Leaf from the CRC32C routing hash (top bits); bucket from the independent key hash.
        const size_t leaf = total_bits ? (hashes[row] >> leaf_shift) : 0;
        const LeafHT & ht = leaves[leaf];
        const UInt32 bucket_hash = bucketHash<key_width>(packed_keys + row * key_width);
        BuildRef cur = leafFind<key_width>(ht, bucket_hash, packed_keys + row * key_width);
        if (cur.row_no == RadixShuffle::INVALID_ROW)
            continue;

        if constexpr (!has_chain)
        {
            /// All-unique build: exactly one build row per key, no next_chain access needed.
            out_left_rows.push_back(static_cast<UInt32>(row));
            out_refs.push_back(cur);
        }
        else
        {
            /// Chain walk: follow next_chain from the head to the INVALID_ROW tail.
            chassert(ht.next_chain != nullptr);
            while (cur.row_no != RadixShuffle::INVALID_ROW)
            {
                out_left_rows.push_back(static_cast<UInt32>(row));
                out_refs.push_back(cur);
                cur = ht.next_chain[leafFlat(cur, block_base)];
            }
        }
    }
}

} /// anonymous namespace


LeafHashTables buildLeafHashTables(
    const LeafArrays & la,
    const std::vector<UInt64> & block_base,
    UInt64 num_rows,
    size_t key_width,
    CoopPool & coord)
{
    Stopwatch sw;

    LeafHashTables out;
    out.num_rows = num_rows;
    out.arena = GrowingArena(GrowingArena::DEFAULT_MAX_BLOCK);

    const size_t num_leaves = la.num_leaves;
    out.leaves.assign(num_leaves, LeafHT{});

    /// Per-leaf sizing (O(num_leaves) integer math on the leader — no allocation, no page touch).
    for (size_t leaf = 0; leaf < num_leaves; ++leaf)
    {
        /// next_chain starts nullptr; fillLeafT populates it lazily on the first duplicate.
        const UInt64 rows = la.leaf_rows[leaf];
        if (rows == 0)
            continue;
        out.leaves[leaf].num_buckets = std::bit_ceil(rows * 2); /// exact-reserve, ~50% load factor
    }

    /// Lazy next_chain: allocated on the first duplicate encountered by any fill worker. For all-unique
    /// builds next_chain is never allocated — mirrors HashJoin::all_values_unique. `alloc_fn` also
    /// 0xFF-initialises the array so every chain tail reads INVALID_ROW without explicit tail writes.
    LazyChainState lcs;
    lcs.num_rows = num_rows;
    GrowingArena * arena_ptr = &out.arena;
    lcs.alloc_fn = [arena_ptr](UInt64 n) -> BuildRef *
    {
        BuildRef * p = arena_ptr->allocArray<BuildRef>(n);
        std::memset(p, 0xFF, n * sizeof(BuildRef));
        return p;
    };

    /// Parallel: each worker ALLOCATES its leaf's cell array (thread-safe jemalloc arena), sets it to
    /// the empty sentinel (`memset` to 0xFF), and fills it. Allocation + init + fill are spread across
    /// the build threads — no single-threaded leader carve.
    const UInt64 * block_base_ptr = block_base.data();
    coord.parallelFor(num_leaves, [&](size_t leaf)
    {
        if (la.leaf_rows[leaf] == 0)
            return;
        const size_t cell_bytes = static_cast<size_t>(out.leaves[leaf].num_buckets) * leafCellBytes(key_width);
        char * cells = static_cast<char *>(out.arena.alloc(cell_bytes, RadixShuffle::LINE_BYTES));
        std::memset(cells, 0xFF, cell_bytes);
        out.leaves[leaf].cells = cells;
        fillLeafDispatch(key_width, out.leaves[leaf], la, leaf, block_base_ptr, lcs);
    });

    /// Publish the final next_chain pointer: workers that found duplicates already set their own
    /// leaf's next_chain pointer during the fill. Set it for the remaining leaves and the table.
    out.next_chain = lcs.nc.load(std::memory_order_acquire);
    for (size_t leaf = 0; leaf < num_leaves; ++leaf)
    {
        if (out.leaves[leaf].next_chain == nullptr && out.next_chain != nullptr)
            out.leaves[leaf].next_chain = out.next_chain;
    }

    ProfileEvents::increment(ProfileEvents::RadixHashBuildHTMicroseconds, sw.elapsedMicroseconds());
    return out;
}


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
    std::vector<BuildRef> & out_refs)
{
    const auto * keys = static_cast<const char *>(packed_keys);
    /// Dispatch on both key_width (compile-time) and has_duplicates (runtime → compile-time branch).
#define DISPATCH(W) \
    case W: collectMatchesT<W, false>(leaves, leaf_shift, total_bits, block_base, hashes, keys, n, out_left_rows, out_refs); return;
    if (!has_duplicates)
    {
        switch (key_width)
        {
            DISPATCH(4)  DISPATCH(8)  DISPATCH(12) DISPATCH(16)
            DISPATCH(20) DISPATCH(24) DISPATCH(28) DISPATCH(32)
            DISPATCH(36) DISPATCH(40) DISPATCH(44) DISPATCH(48)
            DISPATCH(52) DISPATCH(56) DISPATCH(60) DISPATCH(64)
            default:
                throw Exception(
                    ErrorCodes::LOGICAL_ERROR, "RadixHashJoin leaf HT: unsupported key width {} (multiple of 4 in [4,64])", key_width);
        }
    }
#undef DISPATCH
#define DISPATCH(W) \
    case W: collectMatchesT<W, true>(leaves, leaf_shift, total_bits, block_base, hashes, keys, n, out_left_rows, out_refs); return;
    switch (key_width)
    {
        DISPATCH(4)  DISPATCH(8)  DISPATCH(12) DISPATCH(16)
        DISPATCH(20) DISPATCH(24) DISPATCH(28) DISPATCH(32)
        DISPATCH(36) DISPATCH(40) DISPATCH(44) DISPATCH(48)
        DISPATCH(52) DISPATCH(56) DISPATCH(60) DISPATCH(64)
        default:
            throw Exception(
                ErrorCodes::LOGICAL_ERROR, "RadixHashJoin leaf HT: unsupported key width {} (multiple of 4 in [4,64])", key_width);
    }
#undef DISPATCH
}

}
