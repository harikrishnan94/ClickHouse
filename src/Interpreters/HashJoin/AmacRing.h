#pragma once

#include <Interpreters/HashJoin/KeyGetter.h>
#include <base/defines.h>
#include <Common/ColumnsHashing.h>

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

namespace DB
{

/** Generic AMAC (asynchronous memory access chaining) machinery for the join hash maps:
  * a power-of-two ring of in-flight rows where every visit performs exactly ONE memory-dependent
  * step and software-prefetches the address its next visit will dereference, so data-dependent
  * cache misses (the home cell, each collision step) overlap instead of serializing.
  * Ported (as ideas) from the partitioned hash join prototype (branch `ahj` of this fork,
  * `src/Interpreters/PartitionedHashJoin/AmacRing.h`), adapted from that design's leaf/partition
  * wording to the `parallel_hash` per-slot maps.
  *
  * The pieces:
  *  - `ResumableHashMap` (see `ResumableHashMap.h`) - the slot hash-map type with the monolithic
  *    emplace/find decomposed into a resumable cursor: seed (`hash` + `cursorPlace` + prefetch)
  *    and step (inspect ONE cell: `cursorCellIsEmpty` / `cursorKeyEquals` -> done, or
  *    `cursorNext` + prefetch).
  *  - the policy's `Ring` - the per-row ring state as parallel arrays (struct-of-arrays), kept
  *    minimal (fat ring state spills to the stack and kills the memory-level parallelism the
  *    ring exists for).
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

/// The inactive-row sentinel of a build ring's row array; also the driver's row-count bound.
constexpr UInt32 amac_inactive_row = std::numeric_limits<UInt32>::max();

/// ~8-10 in-flight rows saturate a core's L1-D miss handling; beyond 32 slots the ring risks
/// TLB thrashing (Kocberber et al., PVLDB 2015). Power of two for the branch-free wrap.
constexpr size_t amac_ring_size = 32;

/// Below this many rows per section the ring's prime/drain overhead dominates; the plain
/// sequential loop runs instead.
constexpr size_t amac_min_rows = 256;

enum class AmacStepResult : UInt8
{
    Advance, /// collision: the cursor advanced and prefetched the next cell; revisit later
    Done, /// the row completed; the ring slot can be recycled
    DoneNeedsGrow /// the row completed by a claim that overflowed the grower; drain + grow
};

/// The compile-time gate of the AMAC path. The LowCardinality getter keeps the plain loop: it
/// deduplicates lookups per dictionary index through its own cache, which a ring would bypass
/// rather than accelerate (the same reason it disables the look-ahead prefetch). The `hashed`
/// getter's 128-bit key is expensive to compute but the ring packs it ONCE per row at admit and
/// re-reads it per visit, so the cost is paid on the plain loop too and is no reason to opt out.
/// `FixedHashMap` types (`key8`/`key16` and the range maps) have no collision chain to pipeline
/// and no cursor API.
template <typename Map>
concept AmacResumableMap = requires(std::remove_const_t<Map> & map, const std::remove_const_t<Map> & const_map, size_t place)
{
    { const_map.cursorPlace(place) } -> std::same_as<size_t>;
    { const_map.cursorNext(place) } -> std::same_as<size_t>;
    { map.cursorCell(place) };
    { map.cursorGrow() };
};

template <typename T>
inline constexpr bool is_low_cardinality_join_key_getter = false;
template <typename BaseMethod, typename Mapped>
inline constexpr bool is_low_cardinality_join_key_getter<LowCardinalityKeyGetterForJoin<BaseMethod, Mapped>> = true;

template <typename KeyGetter, typename Map>
constexpr bool amac_join_supported = AmacResumableMap<Map> && !is_low_cardinality_join_key_getter<KeyGetter>;

/** Growth cancellation: collect the other in-flight rows, deactivate them, let the policy grow
  * the map (the just-claimed row of slot `skip` is fully inserted and gets rehashed by the
  * resize), then re-admit the collected rows through the policy's `reseed` (a `start` without
  * the skip/zero-key handling: rows handled synchronously never entered the ring, so re-seeding
  * cannot fail). Two ordering rules are load-bearing:
  *
  *  - The rows go back into EXACTLY the slots they were collected from. Filling "the first
  *    pending-count free slots" instead would MOVE the active set whenever some slot is already
  *    inactive - possible in the steady phase, where a failed refill drops `full` but the sweep
  *    still finishes - and the remainder of that sweep would then step a slot the re-seed just
  *    emptied, dereferencing the inactive-row sentinel.
  *  - The rows are re-seeded in ROW order, not in collection order: a mid-sweep refill can put
  *    a later row into a lower slot than an earlier in-flight row, and since a sweep acts on
  *    lower slots first (and the growth erased the earlier row's visit-count lead), re-seeding
  *    in collection order could let a later duplicate claim a cell before an earlier one -
  *    observable through first-wins `RowRef` maps. The ring rows are source row indexes and
  *    every engaged caller's selector is monotonic (the `parallel_hash` scatter emits per-slot
  *    indexes in ascending source order), so sorting them restores the sequential loop's insert
  *    order across the ascending collected slots.
  *
  * Force-inlined because it is called from inside the steady loop: as an out-of-line call it
  * would capture the policy and ring addresses and force conservative per-visit reloads of the
  * policy invariants there.
  */
template <size_t ring_size, typename Policy, typename Ring>
ALWAYS_INLINE void amacDrainAndGrow(Policy & policy, Ring & ring, size_t skip)
{
    static_assert(ring_size <= std::numeric_limits<UInt8>::max() + 1, "slot indexes are collected as bytes");
    std::array<UInt32, ring_size> pending_rows{};
    std::array<UInt8, ring_size> pending_slots{};
    size_t pending_count = 0;
    for (size_t j = 0; j < ring_size; ++j)
    {
        if (j == skip || !ring.isActive(j))
            continue;
        pending_rows[pending_count] = ring.rowAt(j);
        pending_slots[pending_count] = static_cast<UInt8>(j);
        ++pending_count;
        ring.deactivate(j);
    }

    policy.grow();

    std::sort(pending_rows.begin(), pending_rows.begin() + pending_count);

    for (size_t k = 0; k < pending_count; ++k)
        policy.reseed(ring, pending_slots[k], pending_rows[k]);
}

/** The ring driver. The policy provides `Ring<ring_size>` - the whole ring state as a struct of
  * PARALLEL ARRAYS, one per per-row field (the resumable cursor position, the row, the saved
  * hash), value-initialized to all-inactive, with `isActive(s)` / `deactivate(s)` / `rowAt(s)` -
  * plus `may_grow`, `start(ring, s, row) -> bool` (seed slot `s` + home prefetch; false = the
  * row was handled synchronously and the slot stays free) and `step(ring, s) -> AmacStepResult`
  * (ONE fused read -> act). Struct-of-arrays rather than an array of slot structs so a wide
  * field cannot misalign every other field against cache lines, and each field array stays
  * densely packed. The rings carry only what a visit consumes (the cursor, the row, the
  * packed key, a saved hash where the cell compares one), and per-section invariants (map,
  * key getter, skip bytes) live in the policy - measured ring-state minimalism.
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
    chassert(rows < amac_inactive_row);

    /// A policy whose fields are per-run invariants opts in to running on a frame-local copy:
    /// the copy's address never escapes (every policy call inlines), so its fields become SSA
    /// values that stores through the policy's result arrays cannot alias - behind the caller's
    /// reference they would be conservatively reloaded per visit. A policy with per-run mutable
    /// aggregates may still opt in by providing `writeBackTo`, which the driver invokes on the
    /// original after the drain; on an exception mid-run the write-back is skipped, matching the
    /// by-reference semantics where the aggregates are only consumed after a successful run.
    static constexpr bool run_on_copy = requires { requires Policy::copy_into_frame; };
    std::conditional_t<run_on_copy, Policy, Policy &> policy = policy_arg;

    typename Policy::template Ring<ring_size> ring{};
    size_t next = 0;
    size_t active = 0;

    /// Pull pending rows into a slot until one of them enters the ring (rows handled
    /// synchronously by `start` leave the slot free) or the rows are exhausted.
    /// Force-inlined: as an out-of-line call (measured on the `ahj` prototype: clang leaves it
    /// outlined for the multi-column fixed-key policies) it escapes the frame-local policy
    /// copy's address, which defeats the SSA promotion described above and reintroduces
    /// per-visit stack reloads of every policy invariant in the steady loop - plus one full
    /// call per completed row.
    auto refill = [&](size_t s) ALWAYS_INLINE
    {
        while (next < rows)
        {
            const size_t row = next;
            ++next;
            if (policy.start(ring, s, row))
                break;
        }
    };

    /// Prime the ring: after this loop either every slot is active or the rows are exhausted.
    for (size_t s = 0; s < ring_size; ++s)
    {
        refill(s);
        active += ring.isActive(s);
    }

    if (active == ring_size)
    {
        bool full = true;
        while (full && next < rows)
        {
            for (size_t s = 0; s < ring_size; ++s)
            {
                const AmacStepResult result = policy.step(ring, s);
                if (result == AmacStepResult::Advance)
                    continue;
                if constexpr (Policy::may_grow)
                {
                    if (result == AmacStepResult::DoneNeedsGrow)
                        amacDrainAndGrow<ring_size>(policy, ring, s);
                }
                ring.deactivate(s);
                refill(s);
                if (!ring.isActive(s))
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
            if (!ring.isActive(s))
                continue;
            const AmacStepResult result = policy.step(ring, s);
            if (result == AmacStepResult::Advance)
                continue;
            if constexpr (Policy::may_grow)
            {
                if (result == AmacStepResult::DoneNeedsGrow)
                    amacDrainAndGrow<ring_size>(policy, ring, s);
            }
            ring.deactivate(s);
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
