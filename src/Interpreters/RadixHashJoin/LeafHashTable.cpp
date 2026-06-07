#include <Interpreters/RadixHashJoin/LeafHashTable.h>

#include <Common/Exception.h>
#include <Common/ProfileEvents.h>

#include <algorithm>
#include <bit>
#include <cstring>

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

/// Fill one leaf table by inserting all of its scattered (key, ref, hash) rows. Software-pipelined
/// write-prefetch of the next row's bucket cell (spec section 5.6, build: __builtin_prefetch RW=1).
///
/// When `leafInsert` reports a duplicate, the old head (which may carry `BUILDREF_SINGLETON_BIT` if
/// this is the k=2 transition) is stripped and threaded through `ht.next_chain`.
template <size_t key_width>
void fillLeafT(LeafHT & ht, const LeafArrays & la, size_t leaf, const UInt64 * block_base)
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
            /// Duplicate: thread old_head through next_chain.  Strip the singleton bit before using
            /// it as an index — chain entries must never carry the bit.
            const BuildRef old_clean{old_head.block_no & ~RadixShuffle::BUILDREF_SINGLETON_BIT, old_head.row_no};
            ht.next_chain[leafFlat(refs[row], block_base)] = old_clean;
        }
    }
}

void fillLeafDispatch(size_t key_width, LeafHT & ht, const LeafArrays & la, size_t leaf, const UInt64 * block_base)
{
    switch (key_width)
    {
        case 4:  fillLeafT<4>(ht, la, leaf, block_base);  return;
        case 8:  fillLeafT<8>(ht, la, leaf, block_base);  return;
        case 12: fillLeafT<12>(ht, la, leaf, block_base); return;
        case 16: fillLeafT<16>(ht, la, leaf, block_base); return;
        case 20: fillLeafT<20>(ht, la, leaf, block_base); return;
        case 24: fillLeafT<24>(ht, la, leaf, block_base); return;
        case 28: fillLeafT<28>(ht, la, leaf, block_base); return;
        case 32: fillLeafT<32>(ht, la, leaf, block_base); return;
        case 36: fillLeafT<36>(ht, la, leaf, block_base); return;
        case 40: fillLeafT<40>(ht, la, leaf, block_base); return;
        case 44: fillLeafT<44>(ht, la, leaf, block_base); return;
        case 48: fillLeafT<48>(ht, la, leaf, block_base); return;
        case 52: fillLeafT<52>(ht, la, leaf, block_base); return;
        case 56: fillLeafT<56>(ht, la, leaf, block_base); return;
        case 60: fillLeafT<60>(ht, la, leaf, block_base); return;
        case 64: fillLeafT<64>(ht, la, leaf, block_base); return;
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

    /// next_chain: one BuildRef slot per build row, the INVALID_ROW tail until an insert prepends to it.
    /// It must hold the 0xFF sentinel before the fill (for a unique key the slot is never written).
    /// jemalloc memory is not initialised, so allocate it, then memset to 0xFF in parallel across workers.
    if (num_rows > 0)
    {
        out.next_chain = out.arena.allocArray<BuildRef>(num_rows);
        BuildRef * nc = out.next_chain;
        constexpr UInt64 chunk = 1u << 20; /// 1 Mi refs (~8 MiB) per work unit
        const size_t units = static_cast<size_t>((num_rows + chunk - 1) / chunk);
        coord.parallelFor(units, [nc, num_rows](size_t u)
        {
            const UInt64 lo = static_cast<UInt64>(u) * chunk;
            const UInt64 hi = std::min<UInt64>(lo + chunk, num_rows);
            std::memset(nc + lo, 0xFF, static_cast<size_t>(hi - lo) * sizeof(BuildRef));
        });
    }

    /// Per-leaf sizing (O(num_leaves) integer math on the leader — no allocation, no page touch).
    for (size_t leaf = 0; leaf < num_leaves; ++leaf)
    {
        out.leaves[leaf].next_chain = out.next_chain;
        const UInt64 rows = la.leaf_rows[leaf];
        if (rows == 0)
            continue;
        out.leaves[leaf].num_buckets = std::bit_ceil(rows * 2); /// exact-reserve, ~50% load factor
    }

    /// Parallel: each worker ALLOCATES its leaf's cell array (thread-safe jemalloc arena), sets it to the
    /// empty sentinel (`memset` to 0xFF), and fills it. So allocation + init + fill are all spread across
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
        fillLeafDispatch(key_width, out.leaves[leaf], la, leaf, block_base_ptr);
    });

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
