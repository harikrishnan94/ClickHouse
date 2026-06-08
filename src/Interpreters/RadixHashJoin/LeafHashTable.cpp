#include <Interpreters/RadixHashJoin/LeafHashTable.h>

#include <Common/Exception.h>
#include <Common/ProfileEvents.h>

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

/// Fill one leaf table by inserting all of its scattered (key, ref, hash) rows. Software-pipelined
/// write-prefetch of the next row's bucket cell (spec section 5.6, build: __builtin_prefetch RW=1).
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
    const auto * hashes = static_cast<const UInt32 *>(la.hash_base[leaf]);

    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 mask = ht.num_buckets - 1;
    /// Prefetch distance 16: without the next_chain DRAM stall the build loop runs at ~5-10 ns per
    /// iteration, so distance 8 (~40-80 ns lead) no longer covers DRAM latency (~80-100 ns).
    constexpr UInt64 prefetch_distance = 16;

    for (UInt64 row = 0; row < rows; ++row)
    {
        if (row + prefetch_distance < rows)
        {
            const UInt64 prefetch_pos = leafBucket(hashes[row + prefetch_distance], ht.num_buckets) & mask;
            __builtin_prefetch(ht.cells + prefetch_pos * stride, /*rw=*/1, /*locality=*/1);
        }
        const BuildRef old_head = leafInsert<key_width>(ht, hashes[row], keys + row * key_width, refs[row]);
        if (old_head.row_no != RadixShuffle::INVALID_ROW)
        {
            /// Duplicate key: lazily acquire (or reuse already-acquired) next_chain, then thread.
            BuildRef * nc = lcs.acquire();
            ht.next_chain = nc; /// make the pointer visible to this leaf's probe path
            nc[leafFlat(refs[row], block_base)] = old_head;
        }
    }
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
            const UInt32 prefetch_hash = hashes[row + prefetch_distance];
            const size_t prefetch_leaf = total_bits ? (prefetch_hash >> leaf_shift) : 0;
            const LeafHT & prefetch_ht = leaves[prefetch_leaf];
            if (prefetch_ht.num_buckets != 0)
            {
                const UInt64 prefetch_pos = leafBucket(prefetch_hash, prefetch_ht.num_buckets) & (prefetch_ht.num_buckets - 1);
                __builtin_prefetch(prefetch_ht.cells + prefetch_pos * stride, /*rw=*/0, /*locality=*/1);
            }
        }

        const UInt32 h = hashes[row];
        const size_t leaf = total_bits ? (h >> leaf_shift) : 0;
        const LeafHT & ht = leaves[leaf];
        BuildRef cur = leafFind<key_width>(ht, h, packed_keys + row * key_width);
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
