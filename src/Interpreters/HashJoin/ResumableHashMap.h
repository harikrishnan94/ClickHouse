#pragma once

#include <base/defines.h>
#include <Common/HashTable/HashMap.h>
#include <Common/HashTable/HashTableKeyHolder.h>

namespace DB
{

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

    Cell * cursorCell(size_t place) { return &this->buf[place]; }
    const Cell * cursorCell(size_t place) const { return &this->buf[place]; }

    /// The cell buffer base, for policies that cache it in a field instead of re-resolving it
    /// through the map per visit. Invalidated by `cursorGrow`.
    Cell * cursorCells() { return this->buf; }
    const Cell * cursorCells() const { return this->buf; }

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

/** Add the resumable-cursor API to a standard join map. Deriving the rebound types from the
  * standard declarations (instead of mirroring them) means a change of a cell type, hash,
  * grower, or allocator in `HashJoin::MapsTemplate` propagates here automatically, and an
  * incompatible restructuring breaks the build instead of silently diverging.
  */
template <typename Map>
struct WithJoinCursor;

template <typename Key, typename Cell, typename Hash, typename Grower, typename Alloc>
struct WithJoinCursor<HashMapTable<Key, Cell, Hash, Grower, Alloc>>
{
    using Type = ResumableHashMap<HashMapTable<Key, Cell, Hash, Grower, Alloc>>;
};

}
