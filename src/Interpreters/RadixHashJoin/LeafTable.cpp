#include <Interpreters/RadixHashJoin/LeafTable.h>

#include <Interpreters/RadixHashJoin/PackedKeyHash.h>

#include <Common/Exception.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>

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

/// Fill one leaf by inserting all its scattered (key, ref) rows. The bucket is the low 32 bits of the
/// key's hash, recomputed here from the key (no hash is stored on the leaves).
///
/// Group-pipelined prefetch: for each group of rows we (1) compute their home buckets in a tight
/// vectorizable loop (one hash per key), retaining them in a ping-ponged buffer, (2) issue the whole
/// group's cell prefetches back-to-back (exposing group-wide memory-level parallelism), then (3) insert
/// the PREVIOUS group, whose cells were prefetched a full group earlier, reusing its retained home
/// buckets — so the per-key hash is computed exactly once. The group size over-provisions the line-fill
/// buffers so it degrades gracefully on smaller cores, and group*cell-bytes stays within L1. A duplicate
/// appends to the cell's BuildRefList (the first duplicate of a key allocates one Batch node from this
/// worker's `arena`).
template <size_t key_width>
void fillLeaf(LeafHT & ht, const LeafArrays & la, size_t leaf, DB::Arena & arena, std::atomic<bool> & any_duplicates)
{
    const UInt64 rows = la.leaf_rows[leaf];
    if (rows == 0)
        return;

    const auto * keys = static_cast<const char *>(la.key_base[leaf]);
    const BuildRef * refs = la.ref_base[leaf];
    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 num_buckets = ht.numBuckets();
    const UInt64 mask = num_buckets - 1;
    char * const cells = ht.cells();

    constexpr size_t group = 64;
    /// Two home-bucket buffers ping-ponged across groups: while the NEXT group's buckets are computed
    /// (and prefetched) into one buffer, the CURRENT group is inserted from the other, reusing the bucket
    /// each key's single hash already produced. `cur` always points at the buffer whose cells are now
    /// resident; `nxt` receives the lookahead group.
    UInt64 pos_buf[2][group];
    UInt64 * cur = pos_buf[0];
    UInt64 * nxt = pos_buf[1];

    auto compute_buckets = [&](UInt64 * dst, UInt64 base, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
            dst[i] = leafBucket(bucketBits(hashPackedKey<key_width>(keys + (base + i) * key_width)), num_buckets) & mask;
    };
    auto prefetch_burst = [&](const UInt64 * src, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
            __builtin_prefetch(cells + src[i] * stride, /*rw=*/1, /*locality=*/3);
    };

    bool saw_dup = false;
    auto insert_row = [&](UInt64 r, UInt64 pos)
    {
        const bool is_dup = leafInsertAt<key_width>(ht, pos, keys + r * key_width, refs[r], arena);
        saw_dup |= is_dup;
    };

    {
        const size_t prime = static_cast<size_t>(std::min<UInt64>(group, rows));
        compute_buckets(cur, 0, prime);
        prefetch_burst(cur, prime);
    }

    UInt64 row = 0;
    for (; row + group <= rows; row += group)
    {
        const UInt64 next = row + group;
        const size_t to_prefetch = next < rows ? static_cast<size_t>(std::min<UInt64>(group, rows - next)) : 0;
        compute_buckets(nxt, next, to_prefetch);
        prefetch_burst(nxt, to_prefetch);
        for (size_t i = 0; i < group; ++i)
            insert_row(row + i, cur[i]);
        std::swap(cur, nxt);
    }
    for (size_t i = 0; row < rows; ++row, ++i)
        insert_row(row, cur[i]);

    /// Relaxed + idempotent: only ever flips false->true, mirrors the old first-duplicate trigger.
    if (saw_dup)
        any_duplicates.store(true, std::memory_order_relaxed);
}

void fillLeafDispatch(
    size_t key_width, LeafHT & ht, const LeafArrays & la, size_t leaf, DB::Arena & arena, std::atomic<bool> & any_duplicates)
{
    switch (key_width)
    {
        case 4:  fillLeaf<4>(ht, la, leaf, arena, any_duplicates);  return;
        case 8:  fillLeaf<8>(ht, la, leaf, arena, any_duplicates);  return;
        case 12: fillLeaf<12>(ht, la, leaf, arena, any_duplicates); return;
        case 16: fillLeaf<16>(ht, la, leaf, arena, any_duplicates); return;
        case 20: fillLeaf<20>(ht, la, leaf, arena, any_duplicates); return;
        case 24: fillLeaf<24>(ht, la, leaf, arena, any_duplicates); return;
        case 28: fillLeaf<28>(ht, la, leaf, arena, any_duplicates); return;
        case 32: fillLeaf<32>(ht, la, leaf, arena, any_duplicates); return;
        case 36: fillLeaf<36>(ht, la, leaf, arena, any_duplicates); return;
        case 40: fillLeaf<40>(ht, la, leaf, arena, any_duplicates); return;
        case 44: fillLeaf<44>(ht, la, leaf, arena, any_duplicates); return;
        case 48: fillLeaf<48>(ht, la, leaf, arena, any_duplicates); return;
        case 52: fillLeaf<52>(ht, la, leaf, arena, any_duplicates); return;
        case 56: fillLeaf<56>(ht, la, leaf, arena, any_duplicates); return;
        case 60: fillLeaf<60>(ht, la, leaf, arena, any_duplicates); return;
        case 64: fillLeaf<64>(ht, la, leaf, arena, any_duplicates); return;
        default:
            throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin leaf table: unsupported key width {}", key_width);
    }
}

/// AMAC (Asynchronous Memory Access Chaining) probe: instead of finding matches one probe row at a
/// time (so each data-dependent miss — the home cell, every linear-probe collision step — stalls the
/// core for a full DRAM latency), we keep RING_SIZE independent probe rows in flight. Each ring slot is
/// a tiny state machine that performs exactly ONE memory-dependent step per visit and software-
/// prefetches the address it will dereference on its NEXT visit. By the time the round-robin returns to
/// that slot the line is resident, so the misses overlap instead of serialising. This beats a fixed
/// prefetch distance because the open-addressing walk length is data-dependent and unknown ahead of
/// time — the slot prefetches exactly the next address it needs.
///
/// On a key match the cell word is a `DB::BuildRefList`: a singleton (the common case) emits its one
/// inline ref with no further loads; a multi-row key iterates its BuildRefList (the rare duplicate
/// path). The set of emitted (row, ref) pairs is identical to the sequential find; only the
/// interleaving (and therefore the output order) differs, which is irrelevant for an unordered join.
template <size_t key_width>
void collectMatchesImpl(
    const LeafHT * leaves,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const char * packed_keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
    constexpr size_t stride = leafCellBytes(key_width);

    /// Number of in-flight probe rows. ~16 keeps that many independent DRAM accesses outstanding,
    /// matching the line-fill-buffer / outstanding-miss budget of a typical big core, while the ring
    /// state stays tiny and register/L1-resident.
    constexpr size_t ring_size = 16;

    struct Slot
    {
        UInt32 row = 0;            /// probe row index
        /// Activeness is encoded by `cells` itself: nullptr == Inactive (no work assigned); a non-null
        /// leaf cell array == Scan (a linear-probe step is pending on `cells + pos * stride`). An assigned
        /// slot always gets a non-null `ht.cells()` (empty leaves are skipped), so no separate flag is
        /// needed and the activeness test folds into the `cells` load the step already performs.
        const char * cells = nullptr;   /// owning leaf's cell array; nullptr == inactive slot
        const char * key = nullptr;     /// packed_keys + row * key_width
        UInt64 mask = 0;           /// num_buckets - 1
        UInt64 pos = 0;            /// current linear-probe slot
    };

    Slot ring[ring_size];

    size_t next_row = 0;
    size_t active = 0;

    /// Assign the next unprocessed probe row to `s`. Hashes the packed key here (instead of reading a
    /// precomputed hash array), then computes the leaf, home bucket, and issues the prefetch for the home
    /// cell. Doing the hash here is the point: its multiply-fold latency overlaps with the outstanding
    /// cell-miss of the other in-flight ring slots, hiding it behind memory access. Empty leaves yield no
    /// match — we skip straight to the next row. Leaves `s` Inactive once every row has been pulled.
    /// `active` counts slots currently in Scan; a slot recycled here was active for its previous row, so
    /// we adjust by the net change (decrement here, re-increment only when a new row is actually
    /// assigned). `num_buckets`/`mask` are recovered once per probe row here (a shift + an and), never in
    /// the inner round-robin step loop.
    auto pull_next = [&](Slot & s)
    {
        if (s.cells != nullptr)
            --active; /// release the slot's previous row before (maybe) taking a new one
        while (next_row < n)
        {
            const size_t row = next_row++;
            const char * key = packed_keys + row * key_width;
            const UInt64 h = hashPackedKey<key_width>(key);
            const size_t leaf = total_bits ? (routeBits(h) >> leaf_shift) : 0;
            const LeafHT & ht = leaves[leaf];
            if (ht.empty())
                continue; /// empty leaf: no match, pull another row

            const UInt64 num_buckets = ht.numBuckets();
            s.row = static_cast<UInt32>(row);
            s.cells = ht.cells(); /// non-null: marks the slot active (empty leaves were skipped above)
            s.key = key;
            s.mask = num_buckets - 1;
            s.pos = leafBucket(bucketBits(h), num_buckets) & s.mask;
            __builtin_prefetch(s.cells + s.pos * stride, /*rw=*/0, /*locality=*/1);
            ++active;
            return;
        }
        s.cells = nullptr; /// no row left: mark the slot inactive
    };

    /// Prologue: fill the ring with the first up-to-RING_SIZE rows, each computing its initial state and
    /// issuing its home-cell prefetch. Slots start Inactive, so the prologue only ever increments.
    for (Slot & s : ring)
        pull_next(s);

    /// Pipeline: round-robin over the ring, one memory-dependent step per visit, until every row has been
    /// pulled (next_row == n) AND no slot still has work (active == 0). `active` is mutated through the
    /// `pull_next` lambda's by-reference capture, which clang-tidy's local analysis cannot see.
    size_t i = 0;
    // NOLINTNEXTLINE(bugprone-infinite-loop)
    while (active != 0)
    {
        Slot & s = ring[i];
        i = (i + 1) & (ring_size - 1); /// ring_size is a power of two, so this wraps without a branch

        if (s.cells == nullptr)
            continue;

        /// The home/probe cell prefetched on the previous visit is now resident.
        const char * cell = s.cells + s.pos * stride;
        const UInt64 word = *reinterpret_cast<const UInt64 *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (word == 0)
        {
            /// Empty cell: this probe row has no match. Recycle the slot.
            pull_next(s);
        }
        else if (__builtin_memcmp(cell + sizeof(DB::BuildRefList), s.key, key_width) == 0)
        {
            /// Key match. Singleton (the common case): the word IS the encoded ref — emit it directly.
            if (refWordIsInline(word))
            {
                out_left_rows.push_back(s.row);
                out_refs.push_back(BuildRef::fromWord(word));
            }
            else
            {
                /// Multi-row key (rare): iterate the whole BuildRefList, emitting one ref per row.
                const auto & list = *reinterpret_cast<const DB::BuildRefList *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                for (auto it = list.begin(); it.ok(); ++it)
                {
                    out_left_rows.push_back(s.row);
                    out_refs.push_back(BuildRef::fromWord(*it));
                }
            }
            pull_next(s);
        }
        else
        {
            /// Collision: advance one slot, prefetch the next cell, stay in Scan.
            s.pos = (s.pos + 1) & s.mask;
            __builtin_prefetch(s.cells + s.pos * stride, /*rw=*/0, /*locality=*/1);
        }
    }
}

}

LeafTables buildLeafTables(
    const LeafArrays & leaf_arrays,
    UInt64 num_rows,
    size_t key_width,
    size_t num_workers,
    const ParallelFor & parallel_for)
{
    LeafTables out;
    out.num_rows = num_rows;
    const size_t num_leaves = leaf_arrays.num_leaves;
    out.leaves.assign(num_leaves, LeafHT{}); /// every leaf starts empty (word == 0)

    /// One arena per build worker for the BuildRefList Batch nodes. Each worker only ever allocates from
    /// its own arena (single-writer, no locking); the arenas live in `out` so the nodes outlive the
    /// probe. Constructed up front so the worker index maps to a fixed, stable slot.
    out.build_arenas.resize(num_workers);
    for (auto & a : out.build_arenas)
        a = std::make_unique<DB::Arena>();

    /// Each worker sizes, allocates, 0-inits (empty cell == BuildRefList word 0), and fills its own
    /// leaf's cell array — sizing, allocation, zeroing and fill all spread across the build threads, no
    /// single-threaded carve. The leaf descriptor (cells + bucket count) is packed and published in one
    /// store. Batch nodes for that leaf's duplicate keys come from the worker's own arena.
    parallel_for(num_leaves, [&](size_t leaf, size_t worker)
    {
        const UInt64 rows = leaf_arrays.leaf_rows[leaf];
        if (rows == 0)
            return;
        chassert(worker < out.build_arenas.size());
        const UInt64 num_buckets = std::bit_ceil(rows * 2); /// exact-reserve, ~0.5 load factor
        const size_t cell_bytes = static_cast<size_t>(num_buckets) * leafCellBytes(key_width);
        char * cells = static_cast<char *>(out.arena.allocate(cell_bytes, LINE_BYTES));
        std::memset(cells, 0, cell_bytes);
        out.leaves[leaf] = LeafHT(cells, num_buckets);
        fillLeafDispatch(key_width, out.leaves[leaf], leaf_arrays, leaf, *out.build_arenas[worker], out.any_duplicates);
    });

    return out;
}

void collectMatches(
    size_t key_width,
    const LeafHT * leaves,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const void * packed_keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
    const auto * keys = static_cast<const char *>(packed_keys);

#define RHJ_DISPATCH(W) \
    case W: \
        collectMatchesImpl<W>(leaves, leaf_shift, total_bits, keys, n, out_left_rows, out_refs); \
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
