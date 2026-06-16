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

/// ── Shared adaptive AMAC ring driver (build-insert only) ─────────────────────────────────────────────
///
/// The build insert (`fillLeaf`) is a software-prefetch pipeline over the open-addressing leaves: it keeps
/// `ring_size` independent rows in flight, and every round-robin visit performs exactly ONE memory-dependent
/// step and software-prefetches the address it will dereference on its NEXT visit. By the time the
/// round-robin returns to a slot the line is resident, so the data-dependent misses (the home cell, every
/// linear-probe collision step) overlap instead of serialising. The walk length is data-dependent and
/// unknown ahead of time, so this adapts exactly — it prefetches the next address each row actually needs,
/// with no fixed look-ahead distance. (The probe uses `collectMatchesPipelined` below, not this generic
/// driver; only the build still uses it via `BuildPolicy`.)
///
/// `amacRing` owns only the mechanism the policy reuses: the power-of-two round-robin, the
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
        const HashT h = hashPackedKey<key_width>(keys + row * key_width);
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

/// Width dispatch for a chosen `PosT` and the build ring depth (mirrors the probe's width dispatch).
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

/// ── AMAC probe (unified path for every key_width / PosT / dup-ness) ───────────────────────────────
///
/// Hash (`hashPackedKey`), leaf route, and home-cell address decode run inline in `admit` when a probe
/// row enters the ring — the same work `generateSeeds` used to batch ahead of the AMAC loop. With CRC32
/// `HashT` that pre-pass is no longer worth the seed-buffer traffic.
///
/// Decode a packed LeafHT word to the leaf cell-array base (the low 6 bits carry `log2(num_buckets)`).
inline const char * seedCells(UInt64 w) noexcept
{
    return reinterpret_cast<const char *>(w & ~LeafHT::EXP_MASK); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast, performance-no-int-to-ptr)
}

/// Per probe row: hash, route to a leaf group, decode the home cell-array base and home bucket index.
/// Returns the packed LeafHT word (0 == empty group).
template <size_t key_width, typename PosT>
inline UInt64 probeHomeCell(
    const LeafHT * groups,
    UInt32 leaf_shift,
    UInt32 local_shift,
    UInt32 total_bits,
    const char * keys,
    size_t row,
    PosT & out_pos) noexcept
{
    constexpr size_t stride = leafCellBytes(key_width);
    const UInt64 local_mask = (UInt64{1} << local_shift) - 1;
    const HashT h = hashPackedKey<key_width>(keys + row * key_width);
    const UInt64 leaf = total_bits ? (static_cast<UInt64>(routeBits(h)) >> leaf_shift) : 0;
    const size_t g = static_cast<size_t>(leaf >> local_shift);
    const UInt64 local = leaf & local_mask;
    const UInt64 w = groups[g].word;
    const UInt64 base = w & ~LeafHT::EXP_MASK;
    const UInt32 bits = static_cast<UInt32>(w & LeafHT::EXP_MASK);
    if (!base)
    {
        out_pos = 0;
        return 0;
    }
    const UInt64 nb = UInt64{1} << bits;
    const size_t leaf_stride = roundUpToLine(static_cast<size_t>(nb) * stride);
    out_pos = static_cast<PosT>(leafBucket(bucketBits(h), nb));
    return (base + local * leaf_stride) | UInt64{bits};
}

/// Probe match-stream write cursors. A duplicate-free build emits at most one match per probe row, so the
/// buffers are reserved to +n once and written through these plain pointers (no per-match size/capacity
/// bookkeeping). They are passed to the cold helpers BY VALUE and returned, so their address is never taken
/// in the hot ring loop — keeping `row_cur`/`ref_cur` in registers (an in-memory cursor or an outlined
/// `step` would re-add per-visit overhead and undo the hoist).
struct OutPtrs
{
    UInt32 * row_cur;
    UInt32 * row_end;
    BuildRef * ref_cur;
};

/// Cold: a multi-row (duplicate) key overflowed the n-match reservation -> grow the buffers ~2x and re-fetch
/// the cursors. Only reachable in a build that has duplicates.
[[gnu::noinline]] inline OutPtrs growOutPtrs(std::vector<UInt32> & rows, std::vector<BuildRef> & refs, size_t begin, OutPtrs p)
{
    const size_t used = static_cast<size_t>(p.row_cur - rows.data());
    const size_t new_cap = used + (used - begin) + 64;
    rows.resize(new_cap);
    refs.resize(new_cap);
    return {rows.data() + used, rows.data() + new_cap, refs.data() + used};
}

/// Cold: a multi-row key (duplicate build) -> iterate its whole BuildRefList, emitting one ref per build row
/// (growing the buffers as needed). Out of line so the ring's `step` stays small enough to inline.
[[gnu::noinline]] inline OutPtrs
emitMatchListCold(std::vector<UInt32> & rows, std::vector<BuildRef> & refs, size_t begin, OutPtrs p, UInt32 row, const char * cell)
{
    const auto & list = *reinterpret_cast<const DB::BuildRefList *>(cell); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    for (auto it = list.begin(); it.ok(); ++it)
    {
        if (p.row_cur == p.row_end)
            p = growOutPtrs(rows, refs, begin, p);
        *p.row_cur++ = row;
        *p.ref_cur++ = BuildRef::fromWord(*it);
    }
    return p;
}

/// Tile sizing removed: hash+route run inline in `admit` (no seed pre-pass).
/// AMAC ring depth: independent in-flight probe rows. The literature finds ~8-10 saturates a core's L1-D
/// MSHRs and >32 risks TLB thrashing on low-TLB-locality data (Kocberber et al., PVLDB 2015); 32 is the
/// production default carried from the generic `amacRing`, tunable here.
constexpr size_t PROBE_RING_SLOTS = 32;
static_assert((PROBE_RING_SLOTS & (PROBE_RING_SLOTS - 1)) == 0, "PROBE_RING_SLOTS must be a power of two");

/// AMAC ring probe: the SINGLE probe path for every key width, both bucket-index widths (`PosT`), and both
/// duplicate-free and duplicate builds. Each in-flight slot holds one probe row's open-addressing state;
/// `admit` hashes and routes the row, then every round-robin visit performs ONE dependent cell read and
/// software-prefetches the cell its NEXT visit will read, so the random cell misses overlap instead of
/// serialising. The body is trimmed by: a steady/drain split (no per-visit active check while rows remain),
/// a fixed power-of-two ring swept with constant offsets (no modulo in the hot path), a decoded collision
/// `mask` carried in the slot, and a growable raw cursor for match emission.
template <size_t key_width, typename PosT>
void collectMatchesPipelined(
    const GroupedLeaves & grouped,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const char * keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
    static_assert(std::is_same_v<PosT, UInt32> || std::is_same_v<PosT, UInt64>, "PosT must be UInt32 or UInt64");
    static constexpr size_t cell_stride = leafCellBytes(key_width);
    constexpr size_t ring_size = PROBE_RING_SLOTS;

    const LeafHT * groups = grouped.groups.data();
    const UInt32 local_shift = grouped.local_shift;

    /// Growable raw-cursor emit: reserve the singleton lower bound (<= n matches) once and write through
    /// plain pointers; the capacity guard only ever fires for a multi-row (duplicate) key, whose match set
    /// can exceed n. Shrink to the real count at the end (capacity retained for reuse across blocks).
    const size_t out_begin = out_left_rows.size();
    out_left_rows.resize(out_begin + n);
    out_refs.resize(out_begin + n);
    UInt32 * row_cur = out_left_rows.data() + out_begin;
    UInt32 * row_end = out_left_rows.data() + out_begin + n;
    BuildRef * ref_cur = out_refs.data() + out_begin;

    /// Hot singleton emit on register locals; the (cold) grow goes through the by-value helper so these
    /// pointers stay in registers across the ring sweep.
    auto emit_one = [&](UInt32 row, UInt64 ref_word) noexcept
    {
        if (row_cur == row_end) [[unlikely]]
        {
            const OutPtrs p = growOutPtrs(out_left_rows, out_refs, out_begin, {row_cur, row_end, ref_cur});
            row_cur = p.row_cur;
            row_end = p.row_end;
            ref_cur = p.ref_cur;
        }
        *row_cur++ = row;
        *ref_cur++ = BuildRef::fromWord(ref_word);
    };

    struct PipelineSlot
    {
        const char * cells = nullptr; /// leaf cell-array base; nullptr == inactive (empty group / drained)
        PosT pos = 0;                 /// current linear-probe bucket index
        PosT mask = 0;                /// num_buckets - 1 (decoded once in admit)
        UInt32 row = 0;               /// probe row within this call (< n, the block row count)
    };

    size_t next = 0;
    size_t active = 0;
    PipelineSlot ring[ring_size];

    /// Hash, route, and issue the home-cell prefetch for probe row `row`; false for an empty group.
    auto admit = [&](PipelineSlot & s, size_t row) noexcept -> bool
    {
        const UInt64 w = probeHomeCell<key_width, PosT>(groups, leaf_shift, local_shift, total_bits, keys, row, s.pos);
        s.cells = seedCells(w);
        s.mask = (static_cast<PosT>(1) << (w & LeafHT::EXP_MASK)) - 1; /// (1 << bits) - 1; empty -> 0
        s.row = static_cast<UInt32>(row);
        if (s.cells)
            __builtin_prefetch(s.cells + static_cast<size_t>(s.pos) * cell_stride, /*rw=*/0, /*locality=*/1);
        return s.cells != nullptr;
    };

    /// Assign the next pending row with work to `s` (skipping empty groups); false if none remain.
    auto pull = [&](PipelineSlot & s) noexcept -> bool
    {
        while (next < n)
            if (admit(s, next++))
                return true;
        s.cells = nullptr;
        return false;
    };

    /// One fused fresh-read -> act. Returns true when the row is DONE (recycle the slot), false on a
    /// collision step that has already advanced `pos` and prefetched the next cell (revisit later).
    auto step = [&](PipelineSlot & s) noexcept -> bool
    {
        const char * cell = s.cells + static_cast<size_t>(s.pos) * cell_stride;
        UInt64 word = 0;
        __builtin_memcpy(&word, cell, sizeof(UInt64)); /// the dependent miss, prefetched on the prev visit
        if (word == 0)
            return true; /// empty cell: this probe row has no match
        if (__builtin_memcmp(cell + sizeof(DB::BuildRefList), keys + static_cast<size_t>(s.row) * key_width, key_width) == 0)
        {
            /// Key match. Singleton (the common case): the word IS the encoded ref — emit with no 2nd
            /// load. Multi-row key (rare, duplicate build): iterate the whole BuildRefList via the out-of-
            /// line cold helper (keeps `step` small enough to inline; pointers passed/returned by value).
            if (refWordIsInline(word))
            {
                emit_one(s.row, word);
            }
            else
            {
                const OutPtrs p = emitMatchListCold(out_left_rows, out_refs, out_begin, {row_cur, row_end, ref_cur}, s.row, cell);
                row_cur = p.row_cur;
                row_end = p.row_end;
                ref_cur = p.ref_cur;
            }
            return true;
        }
        s.pos = (s.pos + 1) & s.mask;
        __builtin_prefetch(s.cells + static_cast<size_t>(s.pos) * cell_stride, /*rw=*/0, /*locality=*/1);
        return false; /// collision: revisit later
    };

    /// Prologue: fill the ring. Reaching the steady loop below (next < n) implies EVERY slot is
    /// active, because `pull` only fails once `next == n` and then stays failed.
    for (PipelineSlot & slot : ring)
        if (pull(slot))
            ++active;

    /// Steady phase: rows remain AND all slots active, so sweep the fixed-size ring with NO per-visit
    /// active check (constant offsets, no modulo). On the first exhausted refill, hand off to the drain.
    bool exhausted = false;
    while (!exhausted && next < n)
    {
        for (PipelineSlot & slot : ring)
            if (step(slot))
                if (!pull(slot))
                {
                    --active; /// this slot just went inactive (no more rows)
                    exhausted = true;
                    break;
                }
    }

    /// Drain phase: no rows left (`next == n`), so a finished slot just retires (no refill). Uses the
    /// active check; `ring_size` is a power of two so the wrap is a mask, not a modulo.
    size_t i = 0;
    while (active != 0)
    {
        PipelineSlot & s = ring[i];
        i = (i + 1) & (ring_size - 1);
        if (!s.cells)
            continue;
        if (step(s))
        {
            s.cells = nullptr;
            --active;
        }
    }

    /// Shrink to the actual match count (capacity is retained for reuse across blocks).
    out_left_rows.resize(static_cast<size_t>(row_cur - out_left_rows.data()));
    out_refs.resize(static_cast<size_t>(ref_cur - out_refs.data()));
}

/// Width dispatch for a chosen `PosT`: route the runtime `key_width` to `collectMatchesPipelined<W, PosT>`.
template <typename PosT>
void collectMatchesPipelinedDispatch(
    size_t key_width,
    const GroupedLeaves & grouped,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const char * keys,
    size_t n,
    std::vector<UInt32> & out_left_rows,
    std::vector<BuildRef> & out_refs)
{
#define RHJ_PIPE_DISPATCH(W) \
    case W: \
        collectMatchesPipelined<W, PosT>(grouped, leaf_shift, total_bits, keys, n, out_left_rows, out_refs); \
        return;
    switch (key_width)
    {
        RHJ_PIPE_DISPATCH(4)  RHJ_PIPE_DISPATCH(8)  RHJ_PIPE_DISPATCH(12) RHJ_PIPE_DISPATCH(16)
        RHJ_PIPE_DISPATCH(20) RHJ_PIPE_DISPATCH(24) RHJ_PIPE_DISPATCH(28) RHJ_PIPE_DISPATCH(32)
        RHJ_PIPE_DISPATCH(36) RHJ_PIPE_DISPATCH(40) RHJ_PIPE_DISPATCH(44) RHJ_PIPE_DISPATCH(48)
        RHJ_PIPE_DISPATCH(52) RHJ_PIPE_DISPATCH(56) RHJ_PIPE_DISPATCH(60) RHJ_PIPE_DISPATCH(64)
        default:
            throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin leaf table: unsupported key width {}", key_width);
    }
#undef RHJ_PIPE_DISPATCH
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

    /// AMAC probe pipeline for all key widths and dup-ness. Pick the UInt32 bucket-index slot when every
    /// group fits in 32 bits — the practical case; the UInt64 fallback keeps a >2^31-bucket group correct.
    /// `pos_fits_u32` is constant for the whole probe phase (derived from the built groups), so this branch
    /// predicts perfectly.
    if (pos_fits_u32)
        collectMatchesPipelinedDispatch<UInt32>(key_width, grouped, leaf_shift, total_bits, keys, n, out_left_rows, out_refs);
    else
        collectMatchesPipelinedDispatch<UInt64>(key_width, grouped, leaf_shift, total_bits, keys, n, out_left_rows, out_refs);
}

}
