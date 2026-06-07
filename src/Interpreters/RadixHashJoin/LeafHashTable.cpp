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
/// On the first duplicate found by any leaf-fill worker, that worker:
///  1. Allocates `num_rows` BuildRef slots via `nc_once` / `alloc_fn`.
///  2. Stores the pointer in `nc` (release).
///  3. All other workers that subsequently find a duplicate acquire `nc`.
///
/// The allocation does NOT memset.  Instead, at the k=2 transition (old head is a singleton),
/// the worker writes the explicit tail sentinel: `nc[flat(old_head)] = INVALID`.  This ensures
/// every slot that will ever be read by the probe is initialised before reading.
struct LazyChainState
{
    std::once_flag nc_once;
    std::atomic<BuildRef *> nc{nullptr};
    std::function<BuildRef *(UInt64)> alloc_fn; /// allocates num_rows BuildRef slots (no zero-fill)
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
/// On a duplicate key the old head is returned by `leafInsert` (with `BUILDREF_SINGLETON_BIT` when
/// this is the k=2 transition).  The worker:
///   - acquires the shared `next_chain` lazily (first duplicate triggers the alloc via `lcs`).
///   - strips the singleton bit from the old head before storing it as a chain entry.
///   - writes the explicit tail sentinel when stripping the singleton bit (k=2: old singleton
///     becomes the tail; its slot is written `INVALID` so the probe terminates correctly).
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
    /// Prefetch distance 16: without the old next_chain DRAM stall the build loop runs at ~5-10 ns
    /// per iteration, so distance 8 (~40-80 ns lead) no longer covers DRAM latency (~80-100 ns).
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
            /// Duplicate key found: get (or lazily allocate) next_chain.
            BuildRef * nc = lcs.acquire();
            ht.next_chain = nc; /// make the pointer visible to this leaf's probe

            const BuildRef old_clean{old_head.block_no & ~RadixShuffle::BUILDREF_SINGLETON_BIT, old_head.row_no};
            if (old_head.block_no & RadixShuffle::BUILDREF_SINGLETON_BIT)
            {
                /// k=2 transition: old singleton becomes the chain tail.  Write the explicit INVALID
                /// sentinel so the probe terminates correctly (no blanket memset to rely on).
                nc[leafFlat(old_clean, block_base)] = BuildRef{RadixShuffle::INVALID_ROW, RadixShuffle::INVALID_ROW};
            }
            /// Thread new ref → old head (now clean, no singleton bit).
            nc[leafFlat(refs[row], block_base)] = old_clean;
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

/// Collect every (left_row, head-chain BuildRef) match for `n` probe rows against the leaf tables.
template <size_t key_width>
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
    /// Prefetch distance 16: the singleton fast path removes the old next_chain DRAM stall (~80 ns),
    /// so each probe iteration now completes in ~5-10 ns. Distance 8 would give only ~40-80 ns lead
    /// time — not enough to cover DRAM latency for the cells array. Distance 16 gives ~80-160 ns.
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
        if (cur.block_no & RadixShuffle::BUILDREF_SINGLETON_BIT)
        {
            /// Singleton head: exactly one build row, no next_chain load needed.
            out_left_rows.push_back(static_cast<UInt32>(row));
            out_refs.push_back(BuildRef{cur.block_no & ~RadixShuffle::BUILDREF_SINGLETON_BIT, cur.row_no});
            continue;
        }
        /// Chain of length >= 2: walk next_chain to the INVALID_ROW tail.
        chassert(ht.next_chain != nullptr);
        while (cur.row_no != RadixShuffle::INVALID_ROW)
        {
            out_left_rows.push_back(static_cast<UInt32>(row));
            out_refs.push_back(cur);
            cur = ht.next_chain[leafFlat(cur, block_base)];
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

    /// Fail-close: block_no values (0..numBlocks-1) must never reach the MSB reserved for
    /// BUILDREF_SINGLETON_BIT. block_base.size()-1 == numBlocks() (see BuildStore::blockBase()).
    chassert(block_base.size() == 0
        || (block_base.size() - 1) < static_cast<size_t>(RadixShuffle::BUILDREF_SINGLETON_BIT));

    /// Per-leaf sizing (O(num_leaves) integer math on the leader — no allocation, no page touch).
    for (size_t leaf = 0; leaf < num_leaves; ++leaf)
    {
        /// next_chain starts nullptr; fillLeafT populates it lazily on the first duplicate.
        const UInt64 rows = la.leaf_rows[leaf];
        if (rows == 0)
            continue;
        out.leaves[leaf].num_buckets = std::bit_ceil(rows * 2); /// exact-reserve, ~50% load factor
    }

    /// Lazy next_chain: allocated on the first duplicate encountered by any fill worker (no blanket
    /// memset). For all-unique builds next_chain is never allocated (mirrors HashJoin::all_values_unique).
    /// For duplicate builds the detecting worker allocates (no zero-fill) and writes explicit tail
    /// sentinels only at the k=2 transition — every slot the probe can read is written before being
    /// read, so the blanket memset is not needed.
    LazyChainState lcs;
    lcs.num_rows = num_rows;
    GrowingArena * arena_ptr = &out.arena;
    lcs.alloc_fn = [arena_ptr](UInt64 n) -> BuildRef *
    {
        return arena_ptr->allocArray<BuildRef>(n);
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
    switch (key_width)
    {
#define DISPATCH(W) \
    case W: collectMatchesT<W>(leaves, leaf_shift, total_bits, block_base, hashes, keys, n, out_left_rows, out_refs); return;
        DISPATCH(4)
        DISPATCH(8)
        DISPATCH(12)
        DISPATCH(16)
        DISPATCH(20)
        DISPATCH(24)
        DISPATCH(28)
        DISPATCH(32)
        DISPATCH(36)
        DISPATCH(40)
        DISPATCH(44)
        DISPATCH(48)
        DISPATCH(52)
        DISPATCH(56)
        DISPATCH(60)
        DISPATCH(64)
#undef DISPATCH
        default:
            throw Exception(
                ErrorCodes::LOGICAL_ERROR, "RadixHashJoin leaf HT: unsupported key width {} (multiple of 4 in [4,64])", key_width);
    }
}

}
