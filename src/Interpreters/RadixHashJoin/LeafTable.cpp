#include <Interpreters/RadixHashJoin/LeafTable.h>

#include <Interpreters/RadixHashJoin/PackedKeyHash.h>

#include <Common/Exception.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <type_traits>

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

/// ── Shared adaptive AMAC ring driver ─────────────────────────────────────────────────────────────────
///
/// Both the build insert (`fillLeaf`) and the probe (`collectMatches`) are software-prefetch pipelines
/// over the same open-addressing leaves: each keeps `ring_size` independent rows in flight, and every
/// round-robin visit performs exactly ONE memory-dependent step and software-prefetches the address it
/// will dereference on its NEXT visit. By the time the round-robin returns to a slot the line is
/// resident, so the data-dependent misses (the home cell, every linear-probe collision step) overlap
/// instead of serialising. The walk length is data-dependent and unknown ahead of time, so this adapts
/// exactly — it prefetches the next address each row actually needs, with no fixed look-ahead distance.
///
/// `amacRing` owns only the mechanism that is identical for both: the power-of-two round-robin, the
/// `next_row`/`active` accounting, and the pull/recycle skeleton. The `Policy` supplies what differs:
///   - `Slot`                          per-in-flight-row state (default-constructed == INACTIVE).
///   - `bool isActive(const Slot &)` / `void markInactive(Slot &)`   the active sentinel.
///   - `bool startRow(Slot &, row)`    seed the slot for `row` and issue its home-cell prefetch; return
///                                     false to skip the row entirely (e.g. an empty leaf on the probe).
///   - `bool step(Slot &)`             perform ONE fused fresh-read→act; return true when the row is DONE
///                                     (recycle the slot) or false to CONTINUE (a collision step that has
///                                     already advanced `pos` and prefetched the next cell).
///
/// CORRECTNESS — the read and the act MUST be one indivisible `step` call. For the read-only probe this is
/// immaterial, but the build is read-modify-write: were `step` to read a batch of cells in one pass and
/// mutate them in a later pass, two in-flight rows with the same key (or two distinct keys colliding to
/// one cell) could both observe an empty cell and both claim it, silently dropping a build row. With the
/// fresh read fused to the act and one slot mutating per step, the in-flight rows are equivalent to a
/// sequential insert with the row order interleaved, which leaves the (unordered) join result unchanged.
template <size_t ring_size, typename Policy>
void amacRing(Policy & policy, size_t n)
{
    static_assert(ring_size >= 1 && (ring_size & (ring_size - 1)) == 0, "ring_size must be a power of two");
    using Slot = typename Policy::Slot;

    Slot ring[ring_size]; /// every slot default-constructs to INACTIVE

    size_t next_row = 0;
    size_t active = 0;

    /// Recycle `s`: release its previous row (if any) and assign the next row that has work, issuing that
    /// row's home prefetch via the policy. `active` counts slots in flight; a slot recycled here was active
    /// for its previous row, so we decrement first and re-increment only when a new row is actually taken.
    auto pull = [&](Slot & s)
    {
        if (policy.isActive(s))
            --active;
        while (next_row < n)
        {
            if (policy.startRow(s, next_row++))
            {
                ++active;
                return;
            }
        }
        policy.markInactive(s);
    };

    /// Prologue: fill the ring with the first up-to-`ring_size` rows that have work.
    for (Slot & s : ring)
        pull(s);

    /// Pipeline: round-robin one fused step per visit until every row is pulled and no slot still has work.
    size_t i = 0;
    // NOLINTNEXTLINE(bugprone-infinite-loop)
    while (active != 0)
    {
        Slot & s = ring[i];
        i = (i + 1) & (ring_size - 1); /// ring_size is a power of two, so this wraps without a branch

        if (!policy.isActive(s))
            continue;

        if (policy.step(s)) /// row finished -> recycle the slot with the next pending row
            pull(s);
        /// else: a collision step already advanced `pos` and prefetched the next cell; revisit later.
    }
}

/// Build-insert policy for `amacRing`: insert one leaf's scattered (key, ref) rows into the leaf's
/// open-addressing table. Because `fillLeaf` fills exactly ONE leaf, the cell base, mask and stride are
/// loop-invariant and hoisted out of the slot — so a slot carries only what varies per in-flight row: its
/// current linear-probe bucket `pos` and the leaf-row index `row` (8 bytes for the practical
/// `PosT == UInt32`). `row == kInactive` is the inactive sentinel, so no separate active flag is needed.
/// The per-key hash is computed in `startRow` (its latency overlaps the other slots' outstanding cell
/// misses); `step` is the fused read→act: claim an empty cell, append to a same-key cell, or advance the
/// linear probe on a collision. Templated on `PosT` (the bucket-index type) for the same correctness reason
/// as the probe: a leaf with >2^31 buckets needs a 64-bit index.
template <size_t key_width, typename PosT>
struct BuildPolicy
{
    static constexpr size_t stride = leafCellBytes(key_width);
    static constexpr UInt32 kInactive = std::numeric_limits<UInt32>::max();

    char * cells{};          /// the single leaf's cell array (hoisted; the same for every slot)
    PosT mask;             /// num_buckets - 1 (hoisted)
    const char * keys{};     /// la.key_base[leaf]: dense per-leaf packed keys
    const BuildRef * refs{}; /// la.ref_base[leaf]: dense per-leaf row refs
    DB::Arena & arena;     /// this build worker's arena for BuildRefList Batch nodes
    bool saw_dup = false;

    struct Slot
    {
        PosT pos = 0;           /// current linear-probe bucket index, in [0, num_buckets)
        UInt32 row = kInactive; /// leaf row being inserted; kInactive marks the slot inactive
    };
    static_assert(sizeof(Slot) == (std::is_same_v<PosT, UInt32> ? 8 : 16), "unexpected build Slot size");

    bool isActive(const Slot & s) const noexcept { return s.row != kInactive; }
    void markInactive(Slot & s) const noexcept { s.row = kInactive; }

    bool startRow(Slot & s, size_t row) noexcept
    {
        const UInt64 h = hashPackedKey<key_width>(keys + row * key_width);
        s.row = static_cast<UInt32>(row);
        s.pos = static_cast<PosT>(bucketBits(h)) & mask; /// == leafBucket(bucketBits(h), num_buckets)
        __builtin_prefetch(cells + static_cast<size_t>(s.pos) * stride, /*rw=*/1, /*locality=*/3);
        return true; /// a non-empty leaf is filled here, so every row has work
    }

    bool step(Slot & s) noexcept
    {
        char * cell = cells + static_cast<size_t>(s.pos) * stride;
        auto * list = reinterpret_cast<DB::BuildRefList *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (list->word == 0) /// fresh read: empty cell -> claim it for this key (first occurrence)
        {
            __builtin_memcpy_inline(cell + sizeof(DB::BuildRefList), keys + static_cast<size_t>(s.row) * key_width, key_width);
            list->insert(refs[s.row].word(), arena);
            return true; /// done
        }
        if (__builtin_memcmp(cell + sizeof(DB::BuildRefList), keys + static_cast<size_t>(s.row) * key_width, key_width) == 0)
        {
            list->insert(refs[s.row].word(), arena); /// duplicate key: append (allocates a Batch on the first dup)
            saw_dup = true;
            return true; /// done
        }
        /// Collision: advance one slot, prefetch the next cell, revisit later (stay active).
        s.pos = (s.pos + 1) & mask;
        __builtin_prefetch(cells + static_cast<size_t>(s.pos) * stride, /*rw=*/1, /*locality=*/3);
        return false; /// continue
    }
};

/// Fill one non-empty leaf by inserting all its scattered (key, ref) rows via the shared adaptive AMAC
/// ring. The bucket is the low 32 bits of the key's hash, recomputed here from the key (no hash is stored
/// on the leaves). A duplicate appends to the cell's BuildRefList (the first duplicate of a key allocates
/// one Batch node from this worker's `arena`).
template <size_t key_width, size_t ring_size, typename PosT>
void fillLeaf(LeafHT & ht, const LeafArrays & la, size_t leaf, DB::Arena & arena, std::atomic<bool> & any_duplicates)
{
    const UInt64 rows = la.leaf_rows[leaf];
    if (rows == 0)
        return;
    chassert((rows < BuildPolicy<key_width, PosT>::kInactive)); /// leaf row index must fit a UInt32 (sentinel reserved)

    BuildPolicy<key_width, PosT> policy{
        ht.cells(),
        static_cast<PosT>(ht.numBuckets() - 1),
        static_cast<const char *>(la.key_base[leaf]),
        la.ref_base[leaf],
        arena,
    };
    amacRing<ring_size>(policy, rows);

    /// Relaxed + idempotent: only ever flips false->true, mirrors the old first-duplicate trigger.
    if (policy.saw_dup)
        any_duplicates.store(true, std::memory_order_relaxed);
}

/// Width dispatch for a chosen `PosT` and the build ring depth (mirrors the probe's `collectMatchesPos`).
template <typename PosT>
void fillLeafDispatchPos(
    size_t key_width, LeafHT & ht, const LeafArrays & la, size_t leaf, DB::Arena & arena, std::atomic<bool> & any_duplicates)
{
    constexpr size_t ring_size = 32;
#define RHJ_FILL_DISPATCH(W) \
    case W: \
        fillLeaf<W, ring_size, PosT>(ht, la, leaf, arena, any_duplicates); \
        return;
    switch (key_width)
    {
        RHJ_FILL_DISPATCH(4)  RHJ_FILL_DISPATCH(8)  RHJ_FILL_DISPATCH(12) RHJ_FILL_DISPATCH(16)
        RHJ_FILL_DISPATCH(20) RHJ_FILL_DISPATCH(24) RHJ_FILL_DISPATCH(28) RHJ_FILL_DISPATCH(32)
        RHJ_FILL_DISPATCH(36) RHJ_FILL_DISPATCH(40) RHJ_FILL_DISPATCH(44) RHJ_FILL_DISPATCH(48)
        RHJ_FILL_DISPATCH(52) RHJ_FILL_DISPATCH(56) RHJ_FILL_DISPATCH(60) RHJ_FILL_DISPATCH(64)
        default:
            throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin leaf table: unsupported key width {}", key_width);
    }
#undef RHJ_FILL_DISPATCH
}

void fillLeafDispatch(
    size_t key_width, LeafHT & ht, const LeafArrays & la, size_t leaf, DB::Arena & arena, std::atomic<bool> & any_duplicates)
{
    /// One leaf at a time, so its bucket count is known here: a UInt32 bucket index (8-byte slot) suffices
    /// unless this single leaf has >2^31 buckets (~1e9 rows); the UInt64 fallback keeps that case correct.
    if (std::countr_zero(ht.numBuckets()) <= 31)
        fillLeafDispatchPos<UInt32>(key_width, ht, la, leaf, arena, any_duplicates);
    else
        fillLeafDispatchPos<UInt64>(key_width, ht, la, leaf, arena, any_duplicates);
}

/// Probe policy for `amacRing`: find every build match for a batch of probe rows, read-only, emitting the
/// matched (row, ref) pairs. Each in-flight slot may interleave a DIFFERENT leaf (the probe routes each row
/// to its leaf by hash), so — unlike the single-leaf build — the per-slot state must carry that leaf's
/// `cells` and `bits`. The slot is a dense 16 bytes for the practical `PosT == UInt32`:
/// {cells (8), pos (4), row (2), bits (1)}; `mask` is recomputed from `bits` rather than stored, and `row`
/// is a UInt16 because a probe batch is capped at `PROBE_BATCH_ROWS <= 65536`. A non-null `cells` is the
/// active sentinel (empty leaves are skipped in `startRow`). Hashing in `startRow` overlaps the multiply-
/// fold latency with the other slots' outstanding cell misses. On a key match the cell word is a
/// `DB::BuildRefList`: a singleton (the common case) emits its one inline ref with no further load, while a
/// multi-row key iterates its BuildRefList (the rare duplicate path). The set of emitted pairs equals the
/// sequential find; only the order differs (irrelevant for an unordered join). For the impractical
/// >2^31-bucket leaf, `PosT == UInt64` widens the slot to 24 bytes and stays correct.
template <size_t key_width, typename PosT>
struct ProbePolicy
{
    static_assert(std::is_same_v<PosT, UInt32> || std::is_same_v<PosT, UInt64>, "PosT must be UInt32 or UInt64");
    static constexpr size_t stride = leafCellBytes(key_width);

    const LeafHT * leaves;
    UInt32 leaf_shift;
    UInt32 total_bits;
    const char * packed_keys;
    std::vector<UInt32> & out_left_rows;
    std::vector<BuildRef> & out_refs;

    struct Slot
    {
        const char * cells = nullptr; /// owning leaf's cells; nullptr == inactive (empty leaves are skipped)
        PosT pos = 0;                 /// current linear-probe bucket index, in [0, num_buckets)
        UInt16 row = 0;               /// probe row within the batch (< PROBE_BATCH_ROWS <= 65536)
        UInt8 bits = 0;               /// log2(num_buckets); the probe mask is (PosT{1} << bits) - 1
    };
    static_assert(sizeof(Slot) == (std::is_same_v<PosT, UInt32> ? 16 : 24), "unexpected AMAC Slot size");

    bool isActive(const Slot & s) const noexcept { return s.cells != nullptr; }
    void markInactive(Slot & s) const noexcept { s.cells = nullptr; }

    bool startRow(Slot & s, size_t row) noexcept
    {
        const UInt64 h = hashPackedKey<key_width>(packed_keys + row * key_width);
        const size_t leaf = total_bits ? (routeBits(h) >> leaf_shift) : 0;
        const LeafHT & ht = leaves[leaf];
        if (ht.empty())
            return false; /// empty leaf: no match, the driver pulls another row
        const UInt64 num_buckets = ht.numBuckets();
        chassert(row <= 0xFFFF); /// guaranteed by PROBE_BATCH_ROWS <= 65536
        s.row = static_cast<UInt16>(row);
        s.cells = ht.cells(); /// non-null: marks the slot active (empty leaves were skipped above)
        s.bits = static_cast<UInt8>(std::countr_zero(num_buckets));
        s.pos = static_cast<PosT>(leafBucket(bucketBits(h), num_buckets)); /// already in [0, num_buckets)
        __builtin_prefetch(s.cells + static_cast<size_t>(s.pos) * stride, /*rw=*/0, /*locality=*/1);
        return true;
    }

    bool step(Slot & s) noexcept
    {
        /// The home/probe cell prefetched on the previous visit is now resident.
        const char * cell = s.cells + static_cast<size_t>(s.pos) * stride;
        const UInt64 word = *reinterpret_cast<const UInt64 *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (word == 0)
            return true; /// empty cell: this probe row has no match -> done
        if (__builtin_memcmp(cell + sizeof(DB::BuildRefList), packed_keys + s.row * key_width, key_width) == 0)
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
            return true; /// done
        }
        /// Collision: advance one slot, prefetch the next cell, stay active. The mask is recomputed from
        /// `bits` (num_buckets is a power of two, bits <= 31 for the UInt32 path so the shift is in range).
        const PosT mask = (static_cast<PosT>(1) << s.bits) - 1;
        s.pos = (s.pos + 1) & mask;
        __builtin_prefetch(s.cells + static_cast<size_t>(s.pos) * stride, /*rw=*/0, /*locality=*/1);
        return false; /// continue
    }
};

/// AMAC (Asynchronous Memory Access Chaining) probe over the shared `amacRing`: see `ProbePolicy` and the
/// `amacRing` header for the pipeline and its correctness contract. Templated on the ring depth `ring_size`
/// and on `PosT`, the bucket-index type chosen by the caller from the built leaves' max bucket count.
template <size_t key_width, size_t ring_size, typename PosT>
void collectMatchesImpl(
    const LeafHT * leaves,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const char * packed_keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
    ProbePolicy<key_width, PosT> policy{leaves, leaf_shift, total_bits, packed_keys, out_left_rows, out_refs};
    amacRing<ring_size>(policy, n);
}

/// Width dispatch for a chosen `PosT` (UInt32 for the 16-byte slot, UInt64 fallback). The production ring
/// depth is the single tuning constant; both PosT instantiations share it.
template <typename PosT>
void collectMatchesPos(
    size_t key_width,
    const LeafHT * leaves,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const char * keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
    constexpr size_t ring_size = 32;
#define RHJ_DISPATCH(W) \
    case W: \
        collectMatchesImpl<W, ring_size, PosT>(leaves, leaf_shift, total_bits, keys, n, out_left_rows, out_refs); \
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

    /// Probe slot's `pos`/`mask` are UInt32 (a 16-byte slot) iff every leaf's bucket count fits in 32 bits.
    /// Track the max log2(num_buckets) once here so the probe never has to rescan the leaves. (Empty leaves
    /// report 0 via numBuckets()==1, so they never raise the max.)
    UInt8 max_bits = 0;
    for (const LeafHT & ht : out.leaves)
        max_bits = std::max(max_bits, static_cast<UInt8>(std::countr_zero(ht.numBuckets())));
    out.max_bucket_bits = max_bits;

    return out;
}

void collectMatches(
    size_t key_width,
    const LeafHT * leaves,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const void * packed_keys,
    size_t n,
    bool pos_fits_u32,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
    const auto * keys = static_cast<const char *>(packed_keys);

    /// Pick the 16-byte (UInt32 bucket-index) slot when every leaf fits in 32 bits — the practical case;
    /// the UInt64 fallback keeps a >2^31-bucket leaf correct. `pos_fits_u32` is constant for the whole
    /// probe phase (derived from the built leaves), so this branch predicts perfectly.
    if (pos_fits_u32)
        collectMatchesPos<UInt32>(key_width, leaves, leaf_shift, total_bits, keys, n, out_left_rows, out_refs);
    else
        collectMatchesPos<UInt64>(key_width, leaves, leaf_shift, total_bits, keys, n, out_left_rows, out_refs);
}

}
