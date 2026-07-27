#pragma once

#include <base/defines.h>
#include <Common/CacheLine.h>
#include <Common/HashTable/HashMap.h>
#include <Common/HashTable/HashTableKeyHolder.h>

#include <cmath>

namespace DB
{

/** The grower of the rebound join hash maps (shared by `hash` and `parallel_hash`; only the
  * `parallel_hash` AMAC rings drive the cursor API layered on top): `HashTableGrowerWithPrecalculation`
  * plus a fixed tail pad of always-present cells past the power-of-two region. The home cell and the
  * load/growth rules are exactly the standard ones (the pad does not count as capacity), but a
  * collision chain reaching the end of the power-of-two region continues into the pad instead of
  * wrapping; the walk wraps only at the END OF THE PAD — reachable only through a chain longer
  * than the pad, which a load factor of 0.5 makes astronomically rare (though not impossible:
  * adversarially colliding keys can build such a chain, so the wrap MUST stay for correctness —
  * dropping it entirely would turn hash flooding into unbounded walks past the allocation).
  * A probe-side walk that verified post-build that no chain reached the pad's last cell drops
  * its per-step wrap handling entirely: every lookup then terminates at an empty cell at or
  * before the buffer's last cell — `++pos` with no mask and no bound check.
  */
template <size_t initial_size_degree = 8>
class alignas(DB::CH_CACHE_LINE_SIZE) TailPaddedHashTableGrower
{
    UInt8 size_degree = initial_size_degree;
    size_t precalculated_mask = (1ULL << initial_size_degree) - 1;
    size_t precalculated_max_fill = 1ULL << (initial_size_degree - 1);
    size_t precalculated_buf_size = (1ULL << initial_size_degree) + tail_pad;
    static constexpr size_t max_size_degree = 23;

    void recalculate()
    {
        precalculated_mask = (1ULL << size_degree) - 1;
        precalculated_max_fill = 1ULL << (size_degree - 1);
        precalculated_buf_size = (1ULL << size_degree) + tail_pad;
    }

public:
    /// Chain length past which the walk wraps (see the class comment); 64 cells cost 1-3 KB per map.
    static constexpr size_t tail_pad = 64;

    static constexpr auto initial_count = 1ULL << initial_size_degree;

    /// If collision resolution chains are contiguous, we can implement erase operation by moving the elements.
    static constexpr auto performs_linear_probing_with_single_step = true;

    /// The size of the hash table in the cells (the power-of-two region plus the tail pad).
    size_t bufSize() const { return precalculated_buf_size; }

    /// The home mask of the power-of-two region (NOT `bufSize() - 1`).
    size_t mask() const { return precalculated_mask; }

    /// From the hash value, get the cell number in the hash table.
    size_t place(size_t x) const { return x & precalculated_mask; }

    /// The next cell in the collision resolution chain: straight into the tail pad, wrapping
    /// only at the pad's end.
    size_t next(size_t pos) const
    {
        ++pos;
        return pos == precalculated_buf_size ? 0 : pos;
    }

    /// Whether the hash table is sufficiently full (on the power-of-two capacity; the pad is
    /// overflow room, not capacity).
    bool overflow(size_t elems) const { return elems > precalculated_max_fill; }

    void increaseSize()
    {
        size_degree += size_degree >= max_size_degree ? 1 : 2;
        recalculate();
    }

    /// Set the buffer size by the number of elements in the hash table. Used when deserializing a hash table.
    void set(size_t num_elems)
    {
        if (num_elems <= 1)
            size_degree = initial_size_degree;
        else if (initial_size_degree > static_cast<size_t>(log2(num_elems - 1)) + 2)
            size_degree = initial_size_degree;
        else
            size_degree = static_cast<UInt8>(log2(num_elems - 1)) + 2;
        recalculate();
    }

    /// Not a round-trip identity on a padded grower (fed its own `bufSize`, 2^d + pad, it yields
    /// degree d + 1); unreachable for the join maps today.
    void setBufSize(size_t buf_size_)
    {
        size_degree = static_cast<UInt8>(log2(buf_size_ - 1) + 1);
        recalculate();
    }
};

/** The standard join hash map with a resumable-cursor API on top. Deriving (instead of
  * wrapping) keeps the map's cells, hash, grower and public interface exactly the standard ones —
  * `KeyGetterForType`, the probe machinery and the non-joined iteration see an unchanged map —
  * while the cursor methods get the protected internals (`buf`, `grower`, `m_size`) the
  * decomposed emplace/find needs. The AMAC rings drive this API: seed (`hash` + `cursorPlace` +
  * prefetch) and step (inspect ONE cell: `cursorCellIsEmpty` / `cursorKeyEquals` -> done, or
  * `cursorNext` + prefetch).
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
    /// NOT `getBufferSizeInCells() - 1`; callers that cache per-map geometry must carry this one.
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
      * overflowed the grower — the caller must then drain its ring and call `cursorGrow`
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

/** Rebind a standard join map type to its cursor-capable form: the grower becomes
  * `TailPaddedHashTableGrower` (same home/load/growth math, tail-padded buffer) and the
  * resumable-cursor API is added via `ResumableHashMap`. Deriving the rebound types from the
  * standard declarations (instead of mirroring them) means a change of a cell type or hash
  * function in `HashJoin::MapsTemplate` propagates here automatically, and an incompatible
  * restructuring breaks the build instead of silently diverging. The allocator is deliberately
  * NOT rebound: the maps stay on `HashTableAllocator` for both `hash` and `parallel_hash`.
  */
template <typename Map>
struct WithJoinCursor;

template <typename Key, typename Cell, typename Hash, typename Grower, typename Alloc>
struct WithJoinCursor<HashMapTable<Key, Cell, Hash, Grower, Alloc>>
{
    using Type = ResumableHashMap<HashMapTable<Key, Cell, Hash, TailPaddedHashTableGrower<>, Alloc>>;
};

}
