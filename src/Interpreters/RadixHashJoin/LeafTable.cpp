#include <Interpreters/RadixHashJoin/LeafTable.h>

#include <Interpreters/RadixHashJoin/PackedKeyHash.h>

#include <Common/Exception.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <functional>
#include <mutex>

namespace DB
{
namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}
}

namespace DB::RadixJoin
{

namespace
{

/// Shared lazy allocator of the chain array. The first fill worker to hit a duplicate allocates and
/// 0xFF-initialises all `num_rows` slots (so every tail reads INVALID_ROW with no explicit writes),
/// publishes the pointer, and the others pick it up. An all-unique build never calls `acquire`.
struct LazyChain
{
    std::once_flag once;
    std::atomic<BuildRef *> ptr{nullptr};
    std::function<BuildRef *(UInt64)> alloc;
    UInt64 num_rows = 0;

    BuildRef * acquire()
    {
        if (BuildRef * p = ptr.load(std::memory_order_acquire))
            return p;
        std::call_once(once, [this] { ptr.store(alloc(num_rows), std::memory_order_release); });
        return ptr.load(std::memory_order_acquire);
    }
};

/// Fill one leaf by inserting all its scattered (key, ref) rows. The bucket is the low 32 bits of the
/// key's hash, recomputed here from the key (no hash is stored on the leaves).
///
/// Group-pipelined prefetch: for each group of rows we (1) compute their home buckets in a tight
/// vectorizable loop, (2) issue the whole group's cell prefetches back-to-back (exposing group-wide
/// memory-level parallelism), then (3) insert the PREVIOUS group, whose cells were prefetched a full
/// group earlier. The group size over-provisions the line-fill buffers so it degrades gracefully on
/// smaller cores, and group*cell-bytes stays within L1. The chain link on a duplicate is, after the
/// one-time lazy allocation, a single store with no atomic on the steady-state path.
template <size_t key_width>
void fillLeaf(LeafHT & ht, const LeafArrays & la, size_t leaf, const UInt64 * block_base, LazyChain & chain)
{
    const UInt64 rows = la.leaf_rows[leaf];
    if (rows == 0)
        return;

    const auto * keys = static_cast<const char *>(la.key_base[leaf]);
    const BuildRef * refs = la.ref_base[leaf];
    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 num_buckets = ht.num_buckets;
    const UInt64 mask = num_buckets - 1;

    constexpr size_t group = 64;
    UInt64 pf_pos[group];

    auto compute_buckets = [&](UInt64 base, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
            pf_pos[i] = leafBucket(bucketBits(hashPackedKey<key_width>(keys + (base + i) * key_width)), num_buckets) & mask;
    };
    auto prefetch_burst = [&](size_t count)
    {
        for (size_t i = 0; i < count; ++i)
            __builtin_prefetch(ht.cells + pf_pos[i] * stride, /*rw=*/1, /*locality=*/3);
    };

    /// Keep the chain pointer local so the steady-state duplicate handler does not touch the atomic.
    BuildRef * chain_ptr = chain.ptr.load(std::memory_order_acquire);
    bool have_chain = chain_ptr != nullptr;
    if (have_chain)
        ht.next_chain = chain_ptr;

    auto insert_row = [&](UInt64 r)
    {
        const BuildRef old_head = leafInsert<key_width>(
            ht, bucketBits(hashPackedKey<key_width>(keys + r * key_width)), keys + r * key_width, refs[r]);
        if (old_head.row_no != INVALID_ROW)
        {
            if (!have_chain)
            {
                chain_ptr = chain.acquire();
                ht.next_chain = chain_ptr;
                have_chain = true;
            }
            chain_ptr[leafFlat(refs[r], block_base)] = old_head;
        }
    };

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
    for (; row < rows; ++row)
        insert_row(row);
}

void fillLeafDispatch(size_t key_width, LeafHT & ht, const LeafArrays & la, size_t leaf, const UInt64 * block_base, LazyChain & chain)
{
    switch (key_width)
    {
        case 4:  fillLeaf<4>(ht, la, leaf, block_base, chain);  return;
        case 8:  fillLeaf<8>(ht, la, leaf, block_base, chain);  return;
        case 12: fillLeaf<12>(ht, la, leaf, block_base, chain); return;
        case 16: fillLeaf<16>(ht, la, leaf, block_base, chain); return;
        case 20: fillLeaf<20>(ht, la, leaf, block_base, chain); return;
        case 24: fillLeaf<24>(ht, la, leaf, block_base, chain); return;
        case 28: fillLeaf<28>(ht, la, leaf, block_base, chain); return;
        case 32: fillLeaf<32>(ht, la, leaf, block_base, chain); return;
        case 36: fillLeaf<36>(ht, la, leaf, block_base, chain); return;
        case 40: fillLeaf<40>(ht, la, leaf, block_base, chain); return;
        case 44: fillLeaf<44>(ht, la, leaf, block_base, chain); return;
        case 48: fillLeaf<48>(ht, la, leaf, block_base, chain); return;
        case 52: fillLeaf<52>(ht, la, leaf, block_base, chain); return;
        case 56: fillLeaf<56>(ht, la, leaf, block_base, chain); return;
        case 60: fillLeaf<60>(ht, la, leaf, block_base, chain); return;
        case 64: fillLeaf<64>(ht, la, leaf, block_base, chain); return;
        default:
            throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin leaf table: unsupported key width {}", key_width);
    }
}

template <size_t key_width, bool has_chain>
void collectMatchesImpl(
    const LeafHT * leaves,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const UInt64 * block_base,
    const UInt64 * hashes,
    const char * packed_keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
    constexpr size_t stride = leafCellBytes(key_width);
    constexpr size_t prefetch_distance = 16;

    for (size_t row = 0; row < n; ++row)
    {
        if (row + prefetch_distance < n)
        {
            const UInt64 h = hashes[row + prefetch_distance];
            const size_t pleaf = total_bits ? (routeBits(h) >> leaf_shift) : 0;
            const LeafHT & pht = leaves[pleaf];
            if (pht.num_buckets != 0)
            {
                const UInt64 ppos = leafBucket(bucketBits(h), pht.num_buckets) & (pht.num_buckets - 1);
                __builtin_prefetch(pht.cells + ppos * stride, /*rw=*/0, /*locality=*/1);
            }
        }

        const UInt64 h = hashes[row];
        const size_t leaf = total_bits ? (routeBits(h) >> leaf_shift) : 0;
        const LeafHT & ht = leaves[leaf];
        BuildRef cur = leafFind<key_width>(ht, bucketBits(h), packed_keys + row * key_width);
        if (cur.row_no == INVALID_ROW)
            continue;

        if constexpr (!has_chain)
        {
            out_left_rows.push_back(static_cast<UInt32>(row));
            out_refs.push_back(clearSingleton(cur));
        }
        else if (isSingleton(cur))
        {
            out_left_rows.push_back(static_cast<UInt32>(row));
            out_refs.push_back(clearSingleton(cur));
        }
        else
        {
            chassert(ht.next_chain != nullptr);
            while (cur.row_no != INVALID_ROW)
            {
                out_left_rows.push_back(static_cast<UInt32>(row));
                out_refs.push_back(cur);
                cur = ht.next_chain[leafFlat(cur, block_base)];
            }
        }
    }
}

}

LeafTables buildLeafTables(
    const LeafArrays & leaf_arrays,
    const std::vector<UInt64> & block_base,
    UInt64 num_rows,
    size_t key_width,
    CoopPool & coord)
{
    LeafTables out;
    out.num_rows = num_rows;
    const size_t num_leaves = leaf_arrays.num_leaves;
    out.leaves.assign(num_leaves, LeafHT{});

    /// Per-leaf sizing (integer math only; no allocation, no page touch).
    for (size_t leaf = 0; leaf < num_leaves; ++leaf)
    {
        const UInt64 rows = leaf_arrays.leaf_rows[leaf];
        if (rows == 0)
            continue;
        out.leaves[leaf].num_buckets = std::bit_ceil(rows * 2); /// exact-reserve, ~0.5 load factor
    }

    LazyChain chain;
    chain.num_rows = num_rows;
    Arena * arena_ptr = &out.arena;
    chain.alloc = [arena_ptr](UInt64 n) -> BuildRef *
    {
        BuildRef * p = arena_ptr->allocateArray<BuildRef>(n);
        std::memset(p, 0xFF, n * sizeof(BuildRef));
        return p;
    };

    /// Each worker allocates, 0xFF-inits, and fills its own leaf's cell array — allocation, zeroing and
    /// fill all spread across the build threads, no single-threaded carve.
    const UInt64 * block_base_ptr = block_base.data();
    coord.parallelFor(num_leaves, [&](size_t leaf)
    {
        if (leaf_arrays.leaf_rows[leaf] == 0)
            return;
        const size_t cell_bytes = static_cast<size_t>(out.leaves[leaf].num_buckets) * leafCellBytes(key_width);
        char * cells = static_cast<char *>(out.arena.allocate(cell_bytes, LINE_BYTES));
        std::memset(cells, 0xFF, cell_bytes);
        out.leaves[leaf].cells = cells;
        fillLeafDispatch(key_width, out.leaves[leaf], leaf_arrays, leaf, block_base_ptr, chain);
    });

    /// Publish the final chain pointer to the leaves that did not see a duplicate themselves.
    out.next_chain = chain.ptr.load(std::memory_order_acquire);
    if (out.next_chain != nullptr)
        for (size_t leaf = 0; leaf < num_leaves; ++leaf)
            if (out.leaves[leaf].next_chain == nullptr)
                out.leaves[leaf].next_chain = out.next_chain;

    return out;
}

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
    std::vector<BuildRef> & out_refs)
{
    const auto * keys = static_cast<const char *>(packed_keys);

#define RHJ_DISPATCH(W) \
    case W: \
        if (has_duplicates) \
            collectMatchesImpl<W, true>(leaves, leaf_shift, total_bits, block_base, hashes, keys, n, out_left_rows, out_refs); \
        else \
            collectMatchesImpl<W, false>(leaves, leaf_shift, total_bits, block_base, hashes, keys, n, out_left_rows, out_refs); \
        return;
    switch (key_width)
    {
        RHJ_DISPATCH(4)  RHJ_DISPATCH(8)  RHJ_DISPATCH(12) RHJ_DISPATCH(16)
        RHJ_DISPATCH(20) RHJ_DISPATCH(24) RHJ_DISPATCH(28) RHJ_DISPATCH(32)
        RHJ_DISPATCH(36) RHJ_DISPATCH(40) RHJ_DISPATCH(44) RHJ_DISPATCH(48)
        RHJ_DISPATCH(52) RHJ_DISPATCH(56) RHJ_DISPATCH(60) RHJ_DISPATCH(64)
        default:
            throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin leaf table: unsupported key width {}", key_width);
    }
#undef RHJ_DISPATCH
}

}
