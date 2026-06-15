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
    using Slot = Policy::Slot;

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
    /// First-occurrence claims placed so far, and whether this leaf was undersized. The distinct-key
    /// ESTIMATE that sized this leaf can under-count (it estimates distinct low-32 hashes, but two distinct
    /// full keys may share those bits, and HLL itself has variance), so the table could be too small to
    /// hold every distinct key with a spare empty cell. We never let the table fill its LAST cell: a claim
    /// that would leave zero empty cells is refused and `overflowed` is set, so an empty cell ALWAYS remains
    /// — which is what makes every linear-probe walk (this build's collisions and every later probe miss)
    /// terminate. `fillLeaf` then rebuilds the flagged group with safe row-count sizing.
    UInt64 claimed = 0;
    bool overflowed = false;

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
        if (overflowed) /// undersized leaf: stop placing keys and let the ring drain; the group is rebuilt.
            return true;
        char * cell = cells + static_cast<size_t>(s.pos) * stride;
        auto * list = reinterpret_cast<DB::BuildRefList *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (list->word == 0) /// fresh read: empty cell -> claim it for this key (first occurrence)
        {
            /// `mask == num_buckets - 1`: `claimed == mask` means only one empty cell is left. Claiming it
            /// would make the table 100% full, leaving no sentinel to terminate a probe miss -> refuse and
            /// flag the group for a safe rebuild instead.
            if (claimed == mask)
            {
                overflowed = true;
                return true;
            }
            __builtin_memcpy_inline(cell + sizeof(DB::BuildRefList), keys + static_cast<size_t>(s.row) * key_width, key_width);
            list->insert(refs[s.row].word(), arena);
            ++claimed;
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
/// Returns whether the leaf overflowed — i.e. its distinct-estimate sizing was too small to hold every
/// distinct key with a spare empty cell. `buildLeafTables` rebuilds such a group with safe row-count sizing.
template <size_t key_width, size_t ring_size, typename PosT>
bool fillLeaf(char * cells, UInt64 num_buckets, const LeafArrays & la, size_t leaf, DB::Arena & arena, std::atomic<bool> & any_duplicates) /// NOLINT(readability-non-const-parameter)
{
    const UInt64 rows = la.leaf_rows[leaf];
    if (rows == 0)
        return false;
    chassert((rows < BuildPolicy<key_width, PosT>::kInactive)); /// leaf row index must fit a UInt32 (sentinel reserved)

    BuildPolicy<key_width, PosT> policy{
        cells,
        static_cast<PosT>(num_buckets - 1),
        static_cast<const char *>(la.key_base[leaf]),
        la.ref_base[leaf],
        arena,
    };
    amacRing<ring_size>(policy, rows);

    /// Relaxed + idempotent: only ever flips false->true, mirrors the old first-duplicate trigger.
    if (policy.saw_dup)
        any_duplicates.store(true, std::memory_order_relaxed);
    return policy.overflowed;
}

/// Width dispatch for a chosen `PosT` and the build ring depth (mirrors the probe's `collectMatchesPos`).
/// Returns whether the leaf overflowed its distinct-estimate sizing.
template <typename PosT>
bool fillLeafDispatchPos(
    size_t key_width, char * cells, UInt64 num_buckets, const LeafArrays & la, size_t leaf, DB::Arena & arena, std::atomic<bool> & any_duplicates) /// NOLINT(readability-non-const-parameter)
{
    constexpr size_t ring_size = 32;
#define RHJ_FILL_DISPATCH(W) \
    case W: \
        return fillLeaf<W, ring_size, PosT>(cells, num_buckets, la, leaf, arena, any_duplicates);
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

bool fillLeafDispatch(
    size_t key_width, char * cells, UInt64 num_buckets, const LeafArrays & la, size_t leaf, DB::Arena & arena, std::atomic<bool> & any_duplicates) /// NOLINT(readability-non-const-parameter)
{
    /// One leaf at a time, so its bucket count is known here: a UInt32 bucket index (8-byte slot) suffices
    /// unless this single leaf has >2^31 buckets (~1e9 rows); the UInt64 fallback keeps that case correct.
    if (std::countr_zero(num_buckets) <= 31)
        return fillLeafDispatchPos<UInt32>(key_width, cells, num_buckets, la, leaf, arena, any_duplicates);
    return fillLeafDispatchPos<UInt64>(key_width, cells, num_buckets, la, leaf, arena, any_duplicates);
}

/// Probe policy for `amacRing`: find every build match for a batch of probe rows, read-only, emitting the
/// matched (row, ref) pairs. Each in-flight slot may interleave a DIFFERENT leaf (the probe routes each row
/// to its leaf by hash), so — unlike the single-leaf build — the per-slot state must carry that leaf's
/// `cells` and `bits`. The slot is a dense 16 bytes for the practical `PosT == UInt32`:
/// {cells (8), pos (4), row (2), bits (1)}; `mask` is recomputed from `bits` rather than stored, and `row`
/// is a UInt16 because a probe batch is capped at `PROBE_BATCH_ROWS <= 65536`. A non-null `cells` is the
/// active sentinel (empty groups are skipped in `startRow`). Hashing in `startRow` overlaps the multiply-
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

    const LeafHT * groups;
    UInt32 group_bits;
    UInt32 local_shift;
    UInt32 leaf_shift;
    UInt32 total_bits;
    const char * packed_keys;
    std::vector<UInt32> & out_left_rows;
    std::vector<BuildRef> & out_refs;

    struct Slot
    {
        const char * cells = nullptr; /// owning leaf's cells; nullptr == inactive (empty groups are skipped)
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
        const UInt32 route = routeBits(h);
        const UInt32 g = total_bits ? (route >> (32 - group_bits)) : 0;
        const LeafHT gw = groups[g];
        if (gw.empty())
            return false;
        const UInt64 nb = gw.numBuckets();
        const size_t leaf_stride = roundUpToLine(static_cast<size_t>(nb) * stride);
        const size_t local = total_bits ? ((route >> leaf_shift) & ((size_t{1} << local_shift) - 1)) : 0;
        chassert(row <= 0xFFFF);
        s.row = static_cast<UInt16>(row);
        s.cells = gw.cells() + local * leaf_stride;
        s.bits = gw.bits();
        s.pos = static_cast<PosT>(leafBucket(bucketBits(h), nb));
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
/// and on `PosT`, the bucket-index type chosen by the caller from the built groups' max bucket count.
template <size_t key_width, size_t ring_size, typename PosT>
void collectMatchesImpl(
    const GroupedLeaves & grouped,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const char * packed_keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
    ProbePolicy<key_width, PosT> policy{
        grouped.groups.data(),
        grouped.group_bits,
        grouped.local_shift,
        leaf_shift,
        total_bits,
        packed_keys,
        out_left_rows,
        out_refs,
    };
    amacRing<ring_size>(policy, n);
}

/// Width dispatch for a chosen `PosT` (UInt32 for the 16-byte slot, UInt64 fallback). The production ring
/// depth is the single tuning constant; both PosT instantiations share it.
template <typename PosT>
void collectMatchesPos(
    size_t key_width,
    const GroupedLeaves & grouped,
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
        collectMatchesImpl<W, ring_size, PosT>(grouped, leaf_shift, total_bits, keys, n, out_left_rows, out_refs); \
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

void fillGroupLeaves(
    size_t gpos,
    size_t group_size,
    const LeafArrays & leaf_arrays,
    size_t key_width,
    size_t stride,
    const std::vector<UInt8> & group_bucket_bits,
    const std::vector<size_t> & leaf_stride,
    const std::vector<char *> & group_base,
    LeafTables & out,
    const ParallelFor & parallel_for)
{
    const UInt64 nb = UInt64{1} << group_bucket_bits[gpos];
    const size_t leaf_bytes = static_cast<size_t>(nb) * stride;
    parallel_for(group_size, [&](size_t local, size_t worker)
    {
        char * cells = group_base[gpos] + local * leaf_stride[gpos];
        std::memset(cells, 0, leaf_bytes);
        const size_t leaf = gpos * group_size + local;
        const UInt64 rows = leaf_arrays.leaf_rows[leaf];
        if (rows == 0)
            return;
        chassert(worker < out.build_arenas.size());
        const bool overflowed = fillLeafDispatch(
            key_width, cells, nb, leaf_arrays, leaf, *out.build_arenas[worker], out.any_duplicates);
        chassert(!overflowed);
    });
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
    const UInt32 total_bits = num_leaves <= 1 ? 0u : static_cast<UInt32>(std::countr_zero(num_leaves));

    /// One arena per build worker for the BuildRefList Batch nodes. Each worker only ever allocates from
    /// its own arena (single-writer, no locking); the arenas live in `out` so the nodes outlive the
    /// probe. Constructed up front so the worker index maps to a fixed, stable slot.
    out.build_arenas.resize(num_workers);
    for (auto & worker_arena : out.build_arenas)
        worker_arena = std::make_unique<DB::Arena>();

    const size_t stride = leafCellBytes(key_width);

    const UInt32 group_bits = std::min<UInt32>(total_bits, MAX_GROUP_BITS);
    const size_t num_groups = size_t{1} << group_bits;
    const size_t group_size = size_t{1} << (total_bits - group_bits);

    std::vector<UInt8> group_bucket_bits(num_groups, 0);
    std::vector<size_t> leaf_stride(num_groups, 0);
    std::vector<char *> group_base(num_groups, nullptr);

    /// Per-group sizing: snap to the LARGEST member so every leaf in the group fits.
    for (size_t gpos = 0; gpos < num_groups; ++gpos)
    {
        UInt64 max_sizing = 0;
        for (size_t l = 0; l < group_size; ++l)
        {
            const size_t leaf = gpos * group_size + l;
            const UInt64 rows = leaf_arrays.leaf_rows[leaf];
            if (rows == 0)
                continue;
            const UInt64 sizing = leaf_arrays.distinct_key_estimates.empty()
                ? rows
                : leaf_arrays.distinct_key_estimates[leaf];
            max_sizing = std::max(max_sizing, sizing);
        }
        if (max_sizing == 0)
            continue;
        const UInt64 nb = std::bit_ceil(max_sizing * 2);
        group_bucket_bits[gpos] = static_cast<UInt8>(std::countr_zero(nb));
        leaf_stride[gpos] = roundUpToLine(static_cast<size_t>(nb) * stride);
    }

    std::atomic<UInt64> cell_alloc_count{0};

    /// One allocation per non-empty group (<= 256 mallocs total); parallel over groups.
    parallel_for(num_groups, [&](size_t gpos, size_t /*worker*/)
    {
        if (group_bucket_bits[gpos] == 0)
            return;
        const size_t block_bytes = group_size * leaf_stride[gpos];
        char * block = static_cast<char *>(out.arena.allocate(block_bytes, LINE_BYTES));
        cell_alloc_count.fetch_add(1, std::memory_order_relaxed);
        group_base[gpos] = block;
    });

    std::vector<UInt8> leaf_overflowed(num_leaves, 0);

    /// Fill: parallel over leaves (dynamic, handles row skew; keeps build_arenas[worker]).
    parallel_for(num_leaves, [&](size_t leaf, size_t worker)
    {
        const size_t gpos = leaf >> (total_bits - group_bits);
        if (group_bucket_bits[gpos] == 0)
            return;
        const size_t local = leaf & (group_size - 1);
        const UInt64 nb = UInt64{1} << group_bucket_bits[gpos];
        char * cells = group_base[gpos] + local * leaf_stride[gpos];
        std::memset(cells, 0, static_cast<size_t>(nb) * stride);
        const UInt64 rows = leaf_arrays.leaf_rows[leaf];
        if (rows == 0)
            return;
        chassert(worker < out.build_arenas.size());
        const bool overflowed = fillLeafDispatch(
            key_width, cells, nb, leaf_arrays, leaf, *out.build_arenas[worker], out.any_duplicates);
        if (overflowed)
            leaf_overflowed[leaf] = 1;
    });

    /// Group-level rebuild for any group where a leaf overflowed its distinct-estimate sizing.
    for (size_t gpos = 0; gpos < num_groups; ++gpos)
    {
        if (group_bucket_bits[gpos] == 0)
            continue;

        bool needs_rebuild = false;
        UInt64 max_rows = 0;
        for (size_t l = 0; l < group_size; ++l)
        {
            const size_t leaf = gpos * group_size + l;
            if (leaf_overflowed[leaf])
                needs_rebuild = true;
            max_rows = std::max(max_rows, leaf_arrays.leaf_rows[leaf]);
        }
        if (!needs_rebuild)
            continue;

        const UInt64 safe_nb = std::bit_ceil(max_rows * 2);
        group_bucket_bits[gpos] = static_cast<UInt8>(std::countr_zero(safe_nb));
        leaf_stride[gpos] = roundUpToLine(static_cast<size_t>(safe_nb) * stride);

        const size_t block_bytes = group_size * leaf_stride[gpos];
        char * block = static_cast<char *>(out.arena.allocate(block_bytes, LINE_BYTES));
        cell_alloc_count.fetch_add(1, std::memory_order_relaxed);
        group_base[gpos] = block;

        fillGroupLeaves(gpos, group_size, leaf_arrays, key_width, stride, group_bucket_bits, leaf_stride, group_base, out, parallel_for);
    }

    out.grouped.group_bits = group_bits;
    out.grouped.local_shift = total_bits - group_bits;
    out.grouped.groups.assign(num_groups, LeafHT{});
    for (size_t gpos = 0; gpos < num_groups; ++gpos)
        if (group_bucket_bits[gpos] != 0)
            out.grouped.groups[gpos] = LeafHT(group_base[gpos], UInt64{1} << group_bucket_bits[gpos]);

    out.cell_alloc_count = cell_alloc_count.load(std::memory_order_relaxed);

    /// Probe slot's `pos`/`mask` are UInt32 (a 16-byte slot) iff every group's bucket count fits in 32 bits.
    UInt8 max_bits = 0;
    for (UInt8 bits : group_bucket_bits)
        max_bits = std::max(max_bits, bits);
    out.max_bucket_bits = max_bits;

    return out;
}

void collectMatches(
    size_t key_width,
    const GroupedLeaves & grouped,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const void * packed_keys,
    size_t n,
    bool pos_fits_u32,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
    const auto * keys = static_cast<const char *>(packed_keys);

    /// Pick the 16-byte (UInt32 bucket-index) slot when every group fits in 32 bits — the practical case;
    /// the UInt64 fallback keeps a >2^31-bucket group correct. `pos_fits_u32` is constant for the whole
    /// probe phase (derived from the built groups), so this branch predicts perfectly.
    if (pos_fits_u32)
        collectMatchesPos<UInt32>(key_width, grouped, leaf_shift, total_bits, keys, n, out_left_rows, out_refs);
    else
        collectMatchesPos<UInt64>(key_width, grouped, leaf_shift, total_bits, keys, n, out_left_rows, out_refs);
}

}
