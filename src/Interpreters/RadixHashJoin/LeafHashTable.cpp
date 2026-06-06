#include <Interpreters/RadixHashJoin/LeafHashTable.h>

#include <Common/Exception.h>
#include <Common/ProfileEvents.h>
#include <Common/ThreadPool.h>

#include <atomic>
#include <future>

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

/// Run `fn(worker_id)` on `num_threads` workers drawn from `pool`. Worker 0 runs inline; workers
/// 1..T-1 are scheduled; all are joined before returning and the first worker exception is rethrown.
/// (Same shape as BuildStore::runOnPool — kept local so LeafHashTable has no cross-TU coupling.)
template <typename Fn>
void runOnPool(ThreadPool & pool, size_t num_threads, Fn && fn)
{
    if (num_threads <= 1)
    {
        fn(size_t{0});
        return;
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads - 1);
    for (size_t t = 1; t < num_threads; ++t)
    {
        const size_t id = t;
        auto task = std::make_shared<std::packaged_task<void()>>([&fn, id] { fn(id); });
        futures.push_back(task->get_future());
        pool.scheduleOrThrowOnError([pt = std::move(task)] { (*pt)(); });
    }

    fn(size_t{0});

    std::exception_ptr first_exc;
    for (auto & f : futures)
    {
        try
        {
            f.get();
        }
        catch (...)
        {
            if (!first_exc)
                first_exc = std::current_exception();
        }
    }
    if (first_exc)
        std::rethrow_exception(first_exc);
}

/// Fill one leaf table by inserting all of its scattered (key, ref, hash) rows. Software-pipelined
/// write-prefetch of the next row's bucket cell (spec section 5.6, build: __builtin_prefetch RW=1).
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
    constexpr UInt64 PD = 8; /// prefetch distance

    for (UInt64 i = 0; i < rows; ++i)
    {
        if (i + PD < rows)
        {
            const UInt64 ppos = leafBucket(hashes[i + PD], ht.num_buckets) & mask;
            __builtin_prefetch(ht.cells + ppos * stride, /*rw=*/1, /*locality=*/1);
        }
        leafInsert<key_width>(ht, hashes[i], keys + i * key_width, refs[i], block_base);
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
    constexpr size_t PD = 8; /// read-prefetch distance (spec section 5.6, probe: __builtin_prefetch RW=0)

    for (size_t j = 0; j < n; ++j)
    {
        if (j + PD < n)
        {
            const UInt32 ph = hashes[j + PD];
            const size_t pleaf = total_bits ? (ph >> leaf_shift) : 0;
            const LeafHT & pht = leaves[pleaf];
            if (pht.num_buckets != 0)
            {
                const UInt64 ppos = leafBucket(ph, pht.num_buckets) & (pht.num_buckets - 1);
                __builtin_prefetch(pht.cells + ppos * stride, /*rw=*/0, /*locality=*/1);
            }
        }

        const UInt32 h = hashes[j];
        const size_t leaf = total_bits ? (h >> leaf_shift) : 0;
        const LeafHT & ht = leaves[leaf];
        BuildRef cur = leafFind<key_width>(ht, h, packed_keys + j * key_width);
        while (cur.row_no != 0)
        {
            out_left_rows.push_back(static_cast<UInt32>(j));
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
    ThreadPool & pool,
    size_t num_threads,
    bool use_thp)
{
    Stopwatch sw;

    LeafHashTables out;
    out.num_rows = num_rows;
    out.arena = GrowingArena(GrowingArena::DEFAULT_MAX_BLOCK, use_thp);

    const size_t num_leaves = la.num_leaves;
    out.leaves.assign(num_leaves, LeafHT{});

    /// next_chain: one BuildRef slot per build row, zero-initialised by the anonymous mmap (every slot
    /// is the {0,0} tail until an insert prepends to it). Shared by all leaves.
    if (num_rows > 0)
        out.next_chain = out.arena.allocArray<BuildRef>(num_rows);

    /// (1) Single-threaded carve: O(num_leaves) allocations (NC gate). alloc only bumps a cursor and
    /// returns a pointer — it does NOT touch the pages, so the cells stay zero (empty) until inserted.
    for (size_t leaf = 0; leaf < num_leaves; ++leaf)
    {
        out.leaves[leaf].next_chain = out.next_chain;
        const UInt64 rows = la.leaf_rows[leaf];
        if (rows == 0)
            continue;
        const UInt64 num_buckets = std::bit_ceil(rows * 2); /// exact-reserve, ~50% load factor
        out.leaves[leaf].num_buckets = num_buckets;
        out.leaves[leaf].cells
            = static_cast<char *>(out.arena.alloc(num_buckets * leafCellBytes(key_width), RadixShuffle::LINE_BYTES));
    }

    /// (2) Parallel fill: work-steal leaves (disjoint per leaf -> next_chain writes are disjoint too).
    const UInt64 * bb = block_base.data();
    std::atomic<size_t> next_leaf{0};
    runOnPool(pool, num_threads, [&]([[maybe_unused]] size_t worker_id)
    {
        for (size_t leaf = next_leaf.fetch_add(1, std::memory_order_relaxed); leaf < num_leaves;
             leaf = next_leaf.fetch_add(1, std::memory_order_relaxed))
        {
            if (la.leaf_rows[leaf] == 0)
                continue;
            fillLeafDispatch(key_width, out.leaves[leaf], la, leaf, bb);
        }
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
