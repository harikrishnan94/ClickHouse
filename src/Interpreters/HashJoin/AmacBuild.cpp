#include <Interpreters/HashJoin/AmacBuild.h>
#include <Interpreters/HashJoin/AmacRing.h>
#include <Interpreters/HashJoin/HashJoinMethods.h>

namespace DB
{

namespace
{

/** The AMAC build-insert policy: `start` computes the map hash - its latency overlaps the other
  * slots' outstanding cell misses - and issues the home-cell prefetch with write intent, high
  * locality; `step` is ONE fused read -> act: claim an empty cell (insert the key and the row
  * ref), append a duplicate to the arena list, or advance on a collision and prefetch the next
  * cell. The fused step is the load-bearing invariant - see `AmacRing.h`. Rows whose key is the
  * zero sentinel are handled synchronously through the standard `emplace` (the zero cell has no
  * memory-dependent walk to overlap) and never occupy a ring slot. The key is packed ONCE per
  * row at admit and re-read per visit, like the find ring (see `AmacProbeImpl.h`): re-fetching
  * it through `getKeyHolder` re-packs the wide fixed keys (keys128/keys256) from the column
  * pointers - and recomputes the `hashed` getter's 128-bit digest - on EVERY visit. Only
  * `cursorClaim` persists a key - exactly once, at the claim, through a holder rebuilt from
  * the stored key (see `claimHolderOf`).
  * Ported (as ideas) from `AmacBuildInsertPolicy` of the `ahj` prototype
  * (`src/Interpreters/PartitionedHashJoin/PartitionedHashJoinBuild.cpp`).
  */
template <typename KeyGetter, typename Map, bool selector_is_range, typename PosT>
struct AmacBuildInsertPolicy
{
    using Cell = Map::cell_type;
    using Mapped = Map::mapped_type;
    static constexpr bool store_hash = cell_stores_hash<Cell>;
    static constexpr bool may_grow = true;
    /// The frame-copy opt-in (aggregates return through `writeBackTo`) requires copying the key
    /// getter; the `KeysFixed` getter is not copyable (prepared-keys buffer, shuffle masks) and
    /// keeps the by-reference run.
    static constexpr bool copy_into_frame = std::is_copy_constructible_v<KeyGetter>;

    /// The key exactly as the map compares it (`keyHolderGetKey` of an lvalue holder): the
    /// fixed keys and the `hashed` getter's digest by value, the string keys as a view into
    /// the source column - trivially copyable across the whole AMAC getter set. The HOLDER
    /// (`ArenaKeyHolder` for the string getters) is not storable - it carries an arena
    /// reference - so the ring stores the key and `claimHolderOf` rebuilds the holder.
    using KeyHolder = std::remove_reference_t<decltype(std::declval<KeyGetter &>().getKeyHolder(0uz, std::declval<Arena &>()))>;
    using StoredKey = std::decay_t<decltype(keyHolderGetKey(std::declval<KeyHolder &>()))>;
    static_assert(std::is_trivially_copyable_v<StoredKey>);

    /// The insert-ring state, one parallel array per per-row field (see `amacRun`): the cursor
    /// position (`PosT = UInt32` at the caller's dispatch when the buffer index provably fits
    /// 32 bits for the whole run, growths included - the common case, halving the position
    /// array) and the SOURCE row index, plus the packed key and the saved hash for the cells
    /// that store one. The inactive sentinel lives in the row array, so it must be filled at
    /// construction.
    template <size_t ring_size>
    struct RingBase
    {
        std::array<PosT, ring_size> pos{};
        std::array<UInt32, ring_size> row; /// `amac_inactive_row` == inactive
        alignas(64) std::array<StoredKey, ring_size> key{};

        RingBase() { row.fill(amac_inactive_row); }
        bool isActive(size_t s) const { return row[s] != amac_inactive_row; }
        void deactivate(size_t s) { row[s] = amac_inactive_row; }
        UInt32 rowAt(size_t s) const { return row[s]; }
    };
    template <size_t ring_size>
    struct RingWithHash : public RingBase<ring_size>
    {
        std::array<size_t, ring_size> hash{};
    };
    template <size_t ring_size>
    using Ring = std::conditional_t<store_hash, RingWithHash<ring_size>, RingBase<ring_size>>;

    Map & map;
    /// Cached cell buffer base, so the per-visit cell address is one add off a register instead
    /// of a load chain through the map. Refreshed by `grow` (the resize reallocates the buffer);
    /// the zero-sentinel `emplace` in `start` never resizes, so it cannot invalidate the cache.
    Cell * cells;
    /// By value where the getter is a cheap pointer bundle: the frame-owned copy keeps the
    /// key-column bases register-resident in the steady loop.
    std::conditional_t<copy_into_frame, KeyGetter, KeyGetter &> key_getter;
    /// The selector of the section, in the two `ScatteredBlock::Selector` shapes; only the one
    /// matching `selector_is_range` is read (see `sourceRowAt`).
    size_t range_first = 0;
    const UInt64 * selector_indexes = nullptr;
    const UInt8 * skip_bytes = nullptr;
    UInt32 block_no = 0;
    bool any_take_last_row = false;
    Arena & pool;
    bool any_inserted = false;
    bool all_unique = true;
    UInt64 growths = 0;

    /// The source row index of section position `i` - the same translation the sequential
    /// loop's `selectorIndexAt` performs.
    ALWAYS_INLINE size_t sourceRowAt(size_t i) const
    {
        if constexpr (selector_is_range)
            return range_first + i;
        else
            return selector_indexes[i];
    }

    /// The one mapped-value write per row, shared with the sequential `Inserter` through
    /// `applyBuildRowToMapped`, accumulating the per-row signal exactly as the sequential loop
    /// does (`is_inserted |=` for `RowRef` maps, `all_values_unique &=` for `RowRefList`).
    ALWAYS_INLINE void applyRow(Mapped & mapped, bool inserted, size_t row)
    {
        const bool row_kept = applyBuildRowToMapped(mapped, inserted, block_no, row, pool, any_take_last_row);
        if constexpr (std::is_same_v<Mapped, RowRef>)
            any_inserted |= row_kept;
        else
            all_unique &= row_kept;
    }

    /// The holder `cursorClaim` persists, rebuilt from the ring's stored key: for the string
    /// getters an `ArenaKeyHolder` over the stored view (still pointing into the source
    /// column, exactly what `getKeyHolder` returned at admit - `keyHolderPersistKey` then
    /// copies the bytes into the map's arena); for the by-value keys the key itself (persist
    /// is a no-op). Skipping the holder would leave string cells dangling into the block.
    ALWAYS_INLINE auto claimHolderOf(const StoredKey & key) const
    {
        if constexpr (std::is_same_v<KeyHolder, ArenaKeyHolder>)
            return ArenaKeyHolder{key, pool};
        else
            return key;
    }

    template <typename RingT>
    ALWAYS_INLINE void seed(RingT & ring, size_t s, UInt32 row, const StoredKey & key)
    {
        const size_t hash = map.hash(key);
        const size_t pos = map.cursorPlace(hash);
        ring.pos[s] = static_cast<PosT>(pos);
        ring.row[s] = row;
        ring.key[s] = key;
        if constexpr (store_hash)
            ring.hash[s] = hash;
        /// Write intent, high locality: the claim/append of a later visit mutates this line.
        __builtin_prefetch(cells + pos, 1, 3);
    }

    template <typename RingT>
    ALWAYS_INLINE bool start(RingT & ring, size_t s, size_t section_pos)
    {
        if (skip_bytes && skip_bytes[section_pos])
            return false;
        const size_t row = sourceRowAt(section_pos);
        chassert(row < amac_inactive_row);
        auto && key_holder = key_getter.getKeyHolder(row, pool);
        const auto & key = keyHolderGetKey(key_holder);
        if (unlikely(map.isZeroKey(key)))
        {
            typename Map::LookupResult it;
            bool inserted = false;
            map.emplace(key_holder, it, inserted);
            applyRow(it->getMapped(), inserted, row);
            return false;
        }
        seed(ring, s, static_cast<UInt32>(row), key);
        return true;
    }

    /// Re-admit an in-flight row after a growth (`amacDrainAndGrow`). No skip or zero-key
    /// handling: rows handled synchronously by `start` never entered the ring, and the ring
    /// carries source row indexes, so the skip bytes (indexed by section position) do not even
    /// apply here.
    template <typename RingT>
    ALWAYS_INLINE void reseed(RingT & ring, size_t s, UInt32 row)
    {
        auto && key_holder = key_getter.getKeyHolder(row, pool);
        seed(ring, s, row, keyHolderGetKey(key_holder));
    }

    template <typename RingT>
    ALWAYS_INLINE AmacStepResult step(RingT & ring, size_t s)
    {
        const size_t row = ring.row[s];
        const StoredKey & key = ring.key[s];
        /// Only the saved-hash cells (the string keys) consume a hash per visit - as the
        /// `cursorKeyEquals` prefilter and in `cursorClaim`'s `setHash`; every other cell
        /// ignores both arguments, so those rings pass a literal zero instead of recomputing
        /// (or storing) a value nothing reads.
        size_t hash = 0;
        if constexpr (store_hash)
            hash = ring.hash[s];
        Cell * cell = cells + ring.pos[s];
        if (map.cursorCellIsEmpty(cell))
        {
            /// Claim the empty cell and write the mapped value in the SAME visit (fused): a
            /// later same-key or colliding row can never also observe this cell empty.
            auto claim_holder = claimHolderOf(key);
            const bool needs_grow = map.cursorClaim(cell, claim_holder, hash);
            applyRow(cell->getMapped(), /*inserted=*/true, row);
            return needs_grow ? AmacStepResult::DoneNeedsGrow : AmacStepResult::Done;
        }
        if (map.cursorKeyEquals(cell, key, hash))
        {
            applyRow(cell->getMapped(), /*inserted=*/false, row);
            return AmacStepResult::Done;
        }
        const size_t next_pos = map.cursorNext(ring.pos[s]);
        ring.pos[s] = static_cast<PosT>(next_pos);
        __builtin_prefetch(cells + next_pos, 1, 3);
        return AmacStepResult::Advance;
    }

    void grow()
    {
        ++growths;
        map.cursorGrow();
        cells = map.cursorCells();
    }

    /// The driver runs on a frame-local copy (`copy_into_frame`); these are the only fields the
    /// caller reads after the run.
    void writeBackTo(AmacBuildInsertPolicy & original) const
    {
        original.any_inserted = any_inserted;
        original.all_unique = all_unique;
        original.growths = growths;
    }
};

}

template <typename KeyGetter, typename Map, bool selector_is_range>
AmacBuildInsertResult amacBuildInsert(
    Map & map,
    KeyGetter & key_getter,
    size_t rows,
    size_t range_first,
    const UInt64 * selector_indexes,
    const UInt8 * skip_bytes,
    UInt32 stored_block_no,
    bool any_take_last_row,
    Arena & pool)
{
    static_assert(amac_join_supported<KeyGetter, Map>);

    AmacBuildInsertResult result;
    auto run_ring = [&]<typename PosT>()
    {
        AmacBuildInsertPolicy<KeyGetter, Map, selector_is_range, PosT> policy{
            .map = map,
            .cells = map.cursorCells(),
            .key_getter = key_getter,
            .range_first = range_first,
            .selector_indexes = selector_indexes,
            .skip_bytes = skip_bytes,
            .block_no = stored_block_no,
            .any_take_last_row = any_take_last_row,
            .pool = pool};
        amacRun(policy, rows);
        result.growths = policy.growths;
        result.any_inserted = policy.any_inserted;
        result.all_unique = policy.all_unique;
    };
    /// The 8-byte narrow ring slot when the cell index provably fits 32 bits for the whole run,
    /// growths included: the buffer is within 2^32 cells now and the final fill stays within
    /// 2^30 keys, which growth doubling cannot take past 2^31 cells.
    if (map.getBufferSizeInCells() <= (1uz << 32) && map.size() + rows <= (1uz << 30))
        run_ring.template operator()<UInt32>();
    else
        run_ring.template operator()<size_t>();
    return result;
}

#define M(TYPE) AMAC_BUILD_INSERT_INSTANTIATIONS(, TYPE)
APPLY_FOR_AMAC_BUILD_JOIN_VARIANTS(M)
#undef M

}
