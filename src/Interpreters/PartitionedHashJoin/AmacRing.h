#pragma once

#include <Interpreters/HashJoin/KeyGetter.h>
#include <base/defines.h>
#include <Common/ColumnsHashing.h>
#include <Common/HashTable/HashMap.h>

#include <array>
#include <bit>
#include <limits>

namespace DB
{

/** Generic AMAC (asynchronous memory access chaining) machinery for the partitioned hash join:
  * a power-of-two ring of in-flight rows where every visit performs exactly ONE memory-dependent
  * step and software-prefetches the address its next visit will dereference, so data-dependent
  * cache misses (the home cell, each collision step) overlap instead of serializing.
  *
  * The pieces:
  *  - `ResumableHashMap` - the leaf hash-table type with the monolithic emplace/find decomposed
  *    into a resumable cursor: seed (`hash` + `cursorPlace` + prefetch) and step (inspect ONE
  *    cell: `cursorCellIsEmpty` / `cursorKeyEquals` -> done, or `cursorNext` + prefetch).
  *  - `AmacRingSlot` - the per-row ring state, kept minimal (a fat slot spills the ring state to
  *    the stack and kills the memory-level parallelism the ring exists for).
  *  - `amacRun` - the policy driver with the steady/drain split and the growth cancellation
  *    point (`amacDrainAndGrow`).
  *
  * The load-bearing correctness invariant of a build policy's `step`: the cell read and the
  * resulting mutation MUST be one indivisible visit (a fused read -> act). A batched
  * read-then-mutate scheme would let two in-flight rows with the same key (or two keys colliding
  * on one cell) both observe an empty cell and both claim it, silently dropping a build row.
  * Fused, the in-flight rows are equivalent to a sequential insert with rows reordered - safe
  * for an unordered join. Rows the policy handles synchronously in `start` (zero-sentinel keys,
  * skipped rows) never enter the ring, so re-seeding after growth can never fail.
  */

/** The standard join hash map with the resumable-cursor API on top. Deriving (instead of
  * wrapping) keeps the map's cells, hash, grower and public interface exactly the standard ones -
  * `KeyGetterForType`, the probe machinery and the non-joined iteration see an unchanged map -
  * while the cursor methods get the protected internals (`buf`, `grower`, `m_size`) the
  * decomposed emplace/find needs.
  */
template <typename Base>
struct ResumableHashMap : public Base
{
    using Base::Base;
    using Cell = Base::cell_type;
    using Key = Base::key_type;

    /// Keys equal to the zero sentinel live in the map's dedicated zero-value cell, not in the
    /// buffer; a policy handles them synchronously through the standard `emplace`/`find`.
    bool isZeroKey(const Key & key) const { return Cell::isZero(key, *this); }

    /// Seed: the home cell of a hash value.
    size_t cursorPlace(size_t hash_value) const { return this->grower.place(hash_value); }

    /// Step: the next cell of the collision resolution chain.
    size_t cursorNext(size_t place) const { return this->grower.next(place); }

    /// The home mask of the grower's power-of-two region. With a tail-padded grower this is
    /// NOT `getBufferSizeInCells() - 1`; the flat leaf descriptors must carry this one.
    size_t cursorMask() const { return this->grower.mask(); }

    Cell * cursorCell(size_t place) { return &this->buf[place]; }
    const Cell * cursorCell(size_t place) const { return &this->buf[place]; }

    /// The cell buffer base, for policies that cache it in a field instead of re-resolving it
    /// through the map per visit. Invalidated by `cursorGrow`.
    Cell * cursorCells() { return this->buf; }

    bool cursorCellIsEmpty(const Cell * cell) const { return cell->isZero(*this); }

    bool cursorKeyEquals(const Cell * cell, const Key & key, size_t hash_value) const { return cell->keyEquals(key, hash_value, *this); }

    /** Claim an empty cell for a new key: exactly what `emplaceNonZeroImpl` does up to the
      * mapped-value write, which the caller performs in the same fused step. The caller passes
      * the cell pointer it already computed for the empty check. Returns whether the insert
      * overflowed the grower - the caller must then drain its ring and call `cursorGrow`
      * (the standard path resizes at the same point).
      */
    template <typename KeyHolder>
    ALWAYS_INLINE bool cursorClaim(Cell * cell, KeyHolder && key_holder, size_t hash_value)
    {
        keyHolderPersistKey(key_holder);
        const auto & key = keyHolderGetKey(key_holder);
        new (cell) Cell(key, *this);
        cell->setHash(hash_value);
        ++this->m_size;
        return this->grower.overflow(this->m_size);
    }

    /// Growth is a ring cancellation point: in-flight positions index the old buffer, so the
    /// driver drains the ring first, resizes here, and re-seeds the collected rows.
    void cursorGrow() { this->resize(); }
};

/// Cells that keep a saved hash (the string-key cells) both use it as a `keyEquals` prefilter
/// and would recompute it expensively per visit, so those rings carry the hash in the slot; the
/// cheap-key getters (single/multi-column numeric) recompute it from the reloaded key instead,
/// keeping the slot at 16 bytes.
template <typename Cell>
constexpr bool cell_stores_hash = requires(const Cell & cell) { cell.saved_hash; };

/** A ring slot: the resumable cursor position and the row it belongs to. Everything recomputable
  * from the row index (the key, the selector index) is recomputed per visit, and per-section
  * invariants (map, locators, key getter) live in the policy - measured slot minimalism. A policy
  * may extend its slot with address material resolved once at admit (the routed probe carries
  * the leaf's resolved cell pointer) when the steady step would otherwise re-resolve it per
  * visit. `PosT` is the cell-index width: a policy whose buffer provably stays within 2^32
  * cells for the whole run (growths included) narrows it to `UInt32` for the 8-byte slot.
  */
template <bool store_hash, typename PosT = size_t>
struct AmacRingSlot
{
    static constexpr UInt32 inactive_row = std::numeric_limits<UInt32>::max();

    PosT pos = 0;
    UInt32 row = inactive_row;

    bool isActive() const { return row != inactive_row; }
    void deactivate() { row = inactive_row; }
};

template <typename PosT>
struct AmacRingSlot<true, PosT> : public AmacRingSlot<false, PosT>
{
    size_t hash = 0;
};

static_assert(sizeof(AmacRingSlot<false>) == 16);
static_assert(sizeof(AmacRingSlot<true>) == 24);
static_assert(sizeof(AmacRingSlot<false, UInt32>) == 8);
static_assert(sizeof(AmacRingSlot<true, UInt32>) == 16);

/// ~8-10 in-flight rows saturate a core's L1-D miss handling; beyond 32 slots the ring risks
/// TLB thrashing (Kocberber et al., PVLDB 2015). Power of two for the branch-free wrap.
constexpr size_t amac_ring_size = 32;

/// Below this many rows per section the ring's prime/drain overhead dominates; the plain
/// sequential loop runs instead (degeneration, G6).
constexpr size_t amac_min_rows = 256;

enum class AmacStepResult : UInt8
{
    Advance, /// collision: the cursor advanced and prefetched the next cell; revisit later
    Done, /// the row completed; the slot can be recycled
    DoneNeedsGrow /// the row completed by a claim that overflowed the grower; drain + grow
};

/// The compile-time gate of the AMAC path. Excluded getters keep the plain loop: the
/// LowCardinality getter deduplicates lookups per dictionary index through its own cache (a ring
/// would fight it - the same reason it disables the look-ahead prefetch), and the `hashed`
/// fallback recomputes a 128-bit serialized-key hash on every key-holder fetch, far too
/// expensive per visit. `FixedHashMap` types (`key8`/`key16`) have no collision chain to
/// pipeline and no cursor API (they force a single-leaf plan anyway).
template <typename Map>
concept AmacResumableMap = requires(std::remove_const_t<Map> & map, const std::remove_const_t<Map> & const_map, size_t place) {
    { const_map.cursorPlace(place) } -> std::same_as<size_t>;
    { const_map.cursorNext(place) } -> std::same_as<size_t>;
    { map.cursorCell(place) };
    { map.cursorGrow() };
};

template <typename T>
inline constexpr bool is_low_cardinality_join_key_getter = false;
template <typename BaseMethod, typename Mapped>
inline constexpr bool is_low_cardinality_join_key_getter<LowCardinalityKeyGetterForJoin<BaseMethod, Mapped>> = true;

template <typename T>
inline constexpr bool is_hashed_join_key_getter = false;
template <typename Value, typename Mapped, bool use_cache, bool need_offset>
inline constexpr bool is_hashed_join_key_getter<ColumnsHashing::HashMethodHashed<Value, Mapped, use_cache, need_offset>> = true;

template <typename KeyGetter, typename Map>
constexpr bool amac_join_supported
    = AmacResumableMap<Map> && !is_low_cardinality_join_key_getter<KeyGetter> && !is_hashed_join_key_getter<KeyGetter>;

/** Growth cancellation: collect the other in-flight rows, deactivate them, let the policy grow
  * the map (the just-claimed row of slot `skip` is fully inserted and gets rehashed by the
  * resize), then re-seed the collected rows in slot order - preserving their relative stepping
  * order, so same-key rows still act in row order. Force-inlined because it is called from
  * inside the steady loop: as an out-of-line call it would capture the policy and ring
  * addresses and force conservative per-visit reloads of the policy invariants there.
  */
template <size_t ring_size, typename Policy>
ALWAYS_INLINE void amacDrainAndGrow(Policy & policy, std::array<typename Policy::Slot, ring_size> & ring, size_t skip)
{
    std::array<UInt32, ring_size> pending_rows{};
    size_t pending_count = 0;
    for (size_t j = 0; j < ring_size; ++j)
    {
        if (j == skip || !ring[j].isActive())
            continue;
        pending_rows[pending_count++] = ring[j].row;
        ring[j].deactivate();
    }

    policy.grow();

    size_t k = 0;
    for (size_t j = 0; j < ring_size && k < pending_count; ++j)
    {
        if (j == skip)
            continue;
        [[maybe_unused]] const bool restarted = policy.start(ring[j], pending_rows[k++]);
        chassert(restarted); /// rows handled synchronously by `start` never entered the ring
    }
}

/** The ring driver. The policy provides `Slot`, `may_grow`, `start(slot, row) -> bool` (seed +
  * home prefetch; false = the row was handled synchronously and the slot stays free) and
  * `step(slot) -> AmacStepResult` (ONE fused read -> act).
  *
  * Steady/drain split: while rows remain and every refill succeeded, every slot is provably
  * active, so the steady phase sweeps the ring with a plain array `for` - no per-visit active
  * check, no modulo. The first failed refill (or row exhaustion) drops to the drain loop with
  * the active check.
  */
template <typename Policy, size_t ring_size = amac_ring_size>
void amacRun(Policy & policy_arg, size_t rows)
{
    static_assert(std::has_single_bit(ring_size));
    chassert(rows < AmacRingSlot<false>::inactive_row);

    /// A policy whose fields are per-run invariants opts in to running on a frame-local copy:
    /// the copy's address never escapes (every policy call inlines), so its fields become SSA
    /// values that stores through the policy's result arrays cannot alias - behind the caller's
    /// reference they would be conservatively reloaded per visit. A policy with per-run mutable
    /// aggregates may still opt in by providing `writeBackTo`, which the driver invokes on the
    /// original after the drain; on an exception mid-run the write-back is skipped, matching the
    /// by-reference semantics where the aggregates are only consumed after a successful run.
    static constexpr bool run_on_copy = requires { requires Policy::copy_into_frame; };
    std::conditional_t<run_on_copy, Policy, Policy &> policy = policy_arg;

    std::array<typename Policy::Slot, ring_size> ring{};
    size_t next = 0;
    size_t active = 0;

    /// Pull pending rows into a slot until one of them enters the ring (rows handled
    /// synchronously by `start` leave the slot free) or the rows are exhausted.
    auto refill = [&](Policy::Slot & slot)
    {
        while (next < rows)
        {
            const size_t row = next;
            ++next;
            if (policy.start(slot, row))
                break;
        }
    };

    /// Prime the ring: after this loop either every slot is active or the rows are exhausted.
    for (size_t s = 0; s < ring_size; ++s)
    {
        refill(ring[s]);
        active += ring[s].isActive();
    }

    if (active == ring_size)
    {
        bool full = true;
        while (full && next < rows)
        {
            for (size_t s = 0; s < ring_size; ++s)
            {
                const AmacStepResult result = policy.step(ring[s]);
                if (result == AmacStepResult::Advance)
                    continue;
                if constexpr (Policy::may_grow)
                {
                    if (result == AmacStepResult::DoneNeedsGrow)
                        amacDrainAndGrow<ring_size>(policy, ring, s);
                }
                ring[s].deactivate();
                refill(ring[s]);
                if (!ring[s].isActive())
                {
                    --active;
                    full = false;
                }
            }
        }
    }

    /// Drain: no refills left; finish the in-flight rows.
    while (active > 0)
    {
        for (size_t s = 0; s < ring_size; ++s)
        {
            if (!ring[s].isActive())
                continue;
            const AmacStepResult result = policy.step(ring[s]);
            if (result == AmacStepResult::Advance)
                continue;
            if constexpr (Policy::may_grow)
            {
                if (result == AmacStepResult::DoneNeedsGrow)
                    amacDrainAndGrow<ring_size>(policy, ring, s);
            }
            ring[s].deactivate();
            --active;
        }
    }

    if constexpr (run_on_copy)
    {
        if constexpr (requires { policy.writeBackTo(policy_arg); })
            policy.writeBackTo(policy_arg);
    }
}

}
