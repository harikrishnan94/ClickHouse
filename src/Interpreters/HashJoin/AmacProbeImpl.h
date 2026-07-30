#pragma once

#include <Interpreters/HashJoin/AmacProbe.h>

#include <Common/HashTable/HashTable.h>
#include <Common/ProfileEvents.h>

namespace ProfileEvents
{
extern const Event ConcurrentHashJoinAmacProbeRows;
}

namespace DB
{

/// The find-pass template body lives here (not in `AmacProbe.cpp`) so tests can instantiate it
/// over adversarial map types - the wrapped-chain walk cannot be reached deterministically
/// through SQL or the production hash functions.
namespace AmacProbeDetail
{

/** The AMAC find policy of the two-phase routed probe (phase A): out-of-order lookups that only
  * fill the per-row result arrays - the matched cell's recorded word (the mapped value by
  * value, or its address for ASOF; see `amac_mapped_fits_word` and `amac_mapped_by_pointer` in
  * `AmacProbe.h`) into `found_word` (0 = no match) and, for the flagged shapes only, its
  * slot-local used-flags offset and route slot. Nothing is emitted here; phase B consumes the
  * results in left-row order - the flagless word-mapped lazy shapes through the dispatch-free
  * `word_loop`, the rest through the standard `processMatch` (see
  * `HashJoinRoutedMethodsImpl.h`).
  *
  * One ring serves MANY maps - each row's route slot's, derived at admit from the map hash the
  * seed computes anyway (`joinHashRouteSlot`): the hash addresses the bucket AND routes, so no
  * per-row slot-ids pass precedes the probe. The slot's address material is resolved
  * once at admit from the flat descriptor array and carried in the ring slot as the RESOLVED
  * CELL POINTER, so a steady visit dereferences nothing but the cell itself and the stored
  * probe key: the map headers, scattered across as many heap objects as there are slots, would
  * otherwise sit on the address chain of every visit. The collision walk advances linearly
  * and wraps at the standard power-of-two buffer boundary. The key is packed ONCE at admit
  * and read per visit: re-fetching it through
  * `getKeyHolder` re-packs the wide fixed keys (keys128/keys256) from the column pointers on
  * EVERY visit - measured on the `ahj` prototype as the dominant per-visit cost of the
  * wide-key ring. Ported (as ideas) from `RoutedAmacFindPolicy` of the `ahj` prototype
  * (`src/Interpreters/PartitionedHashJoin/PartitionedHashJoinProbeImpl.h`).
  */
template <typename KeyGetter, typename Map, bool need_flags, bool selector_is_range>
struct AmacFindPolicy
{
    using Cell = Map::cell_type;
    static constexpr bool store_hash = cell_stores_hash<Cell>;
    static constexpr bool may_grow = false;
    static constexpr bool copy_into_frame = true; /// results live in the arrays; no state survives the run

    /// The slot-register walk below relies on linear probing and stateless cells, whose
    /// zero-check and key-compare read nothing through the map object.
    static_assert(Map::grower_type::performs_linear_probing_with_single_step);
    static_assert(std::is_same_v<typename Cell::State, HashTableNoState>);
    static constexpr HashTableNoState no_state{};

    /// The key exactly as the map compares it (`keyHolderGetKey` of an lvalue holder, matching
    /// the call sites): the fixed keys by value, the `hashed` getter's 128-bit digest by value,
    /// the string keys as a view into the probe column - trivially copyable across the whole
    /// AMAC getter set (the arena-backed string holder persists nothing on the find path).
    using KeyHolder = std::remove_reference_t<decltype(std::declval<KeyGetter &>().getKeyHolder(0uz, std::declval<Arena &>()))>;
    using StoredKey = std::decay_t<decltype(keyHolderGetKey(std::declval<KeyHolder &>()))>;
    static_assert(std::is_trivially_copyable_v<StoredKey>);

    /** The find-ring state, one parallel array per per-row field (see `amacRun`). The resolved
      * cell pointer replaces the {buffer, mask, position} triple. `cell == nullptr` is the
      * inactive sentinel (value-initialization = all-inactive), freeing `row` for the 16-bit
      * chunk-local index. The slot id stays only for `recordHit` (recovering the slot-local
      * used-flags offset through the descriptor, once per matched row); the emit side never
      * reads it - the flagged shapes get their per-row slot through `found_slot`.
      */
    template <size_t ring_size>
    struct RingBase
    {
        std::array<const Cell *, ring_size> cell{}; /// the cell the next visit reads; nullptr == inactive
        std::array<UInt16, ring_size> row{}; /// chunk-local probe row
        std::array<UInt16, ring_size> slot{};
        alignas(64) std::array<StoredKey, ring_size> key{};

        bool isActive(size_t s) const { return cell[s] != nullptr; }
        void deactivate(size_t s) { cell[s] = nullptr; }
        UInt32 rowAt(size_t s) const { return row[s]; }
    };
    template <size_t ring_size>
    struct RingWithHash : public RingBase<ring_size>
    {
        std::array<size_t, ring_size> hash{};
    };
    /// Cells that keep a saved hash (the string keys) use it as a `keyEquals` prefilter and
    /// would recompute it expensively per visit, so those rings carry it; the cheap-key getters
    /// recompute it from the stored key instead.
    template <size_t ring_size>
    using Ring = std::conditional_t<store_hash, RingWithHash<ring_size>, RingBase<ring_size>>;

    /// The pass runs in chunks of this many rows, so the ring's row index fits 16 bits; the
    /// default probe block (65409 rows) is 8 chunks.
    static constexpr size_t chunk_rows_max = 8192;

    /// A by-value copy of a trivially copyable key getter keeps its key-column pointers plain
    /// fields of the frame-local policy instead of two dependent loads behind a reference.
    std::conditional_t<std::is_trivially_copyable_v<KeyGetter>, KeyGetter, KeyGetter &> key_getter;
    /// Hash provider and zero-key checker; reads nothing through the object (the hash functor
    /// is an empty base and the cells are stateless), so any slot's map serves.
    const Map & map0;
    const Map * const * slot_maps = nullptr; /// the zero-key sentinel path only
    const SlotMapDesc * slot_descs = nullptr;
    UInt32 route_shift = 32; /// `32 - log2(slots)`; see `joinHashRouteSlot`
    size_t selector_base = 0; /// the first row of a continuous-range selector
    const UInt64 * selector_indexes = nullptr; /// the data of an explicit-indexes selector
    const UInt8 * skip_data = nullptr; /// null on the fast path
    Arena & pool;
    UInt64 * found_word = nullptr;
    UInt64 * found_offset = nullptr; /// null unless `need_flags`
    UInt8 * found_slot = nullptr; /// null unless `need_flags`

    ALWAYS_INLINE size_t indexAt(size_t i) const
    {
        if constexpr (selector_is_range)
            return selector_base + i;
        else
            return selector_indexes[i];
    }

    /// The per-row result: the mapped value by value where it fits a word, its address
    /// otherwise (see `amac_mapped_by_pointer`). Neither is 0 for a built cell, so 0 keeps
    /// encoding a miss.
    ALWAYS_INLINE static UInt64 recordedWordOf(const Cell * cell)
    {
        if constexpr (amac_mapped_fits_word<typename Map::mapped_type>)
            return mappedWordOf(cell->getMapped());
        else
            return reinterpret_cast<UInt64>(&cell->getMapped());
    }

    /// The synchronous zero-key path of `start`: the cell (the map's dedicated zero-value cell,
    /// or null) comes from the map object, and so does its slot-local used-flags offset.
    ALWAYS_INLINE void record(size_t i, const Cell * cell, const Map & map [[maybe_unused]])
    {
        if (!cell)
        {
            found_word[i] = 0;
            return;
        }
        found_word[i] = recordedWordOf(cell);
        if constexpr (need_flags)
            found_offset[i] = map.offsetInternal(cell);
    }

    /// A ring hit: the cell is known non-zero, so its slot-local used-flags offset is its
    /// buffer position + 1 - `offsetInternal` without touching the map object. Recovering the
    /// position costs one descriptor load, on this once-per-matched-row path only (and only
    /// for the flagged shapes).
    ALWAYS_INLINE void recordHit(size_t i, size_t slot [[maybe_unused]], const Cell * cell)
    {
        found_word[i] = recordedWordOf(cell);
        if constexpr (need_flags)
        {
            const auto pos = static_cast<size_t>(cell - static_cast<const Cell *>(slot_descs[slot].buf));
            found_offset[i] = pos + 1;
        }
    }

    /// High-locality (L1) prefetch of the WHOLE cell, read intent (a probed cell is not
    /// mutated). Locality 1 compiles to `pldl3keep` on AArch64 - it stages the line in L3 only,
    /// and the visit's demand load then pays the whole L1-miss latency at use; measured on the
    /// `ahj` keys256 anchor as the ring's dominant stall. Cells wider than 24 bytes regularly
    /// straddle two lines (a 40-byte keys256 cell does on ~61% of positions); the second line
    /// would stall the limb compares the same way, so prefetch the cell's last byte's line too.
    static ALWAYS_INLINE void prefetchCell(const Cell * cell)
    {
        __builtin_prefetch(cell, 0, 3);
        if constexpr (sizeof(Cell) > 24)
            __builtin_prefetch(reinterpret_cast<const char *>(cell) + sizeof(Cell) - 1, 0, 3);
    }

    template <typename RingT>
    ALWAYS_INLINE bool start(RingT & ring, size_t s, size_t i)
    {
        const size_t ind = indexAt(i);
        if (skip_data && skip_data[ind])
        {
            found_word[i] = 0;
            return false;
        }
        auto && key_holder = key_getter.getKeyHolder(ind, pool);
        const auto & key = keyHolderGetKey(key_holder);
        /// One hash per row, admit-time: it addresses the bucket AND derives the route slot.
        const size_t hash = map0.hash(key);
        const size_t slot = joinHashRouteSlot(hash, route_shift);
        if constexpr (need_flags)
            found_slot[i] = static_cast<UInt8>(slot);
        if (map0.isZeroKey(key)) [[unlikely]]
        {
            /// The zero key lives in the dedicated zero-value cell - nothing to overlap.
            const Map & map = *slot_maps[slot];
            record(i, map.find(key), map);
            return false;
        }
        ring.key[s] = key;
        const SlotMapDesc & desc = slot_descs[slot];
        const Cell * cell = static_cast<const Cell *>(desc.buf) + (hash & desc.mask);
        ring.cell[s] = cell;
        ring.row[s] = static_cast<UInt16>(i);
        ring.slot[s] = static_cast<UInt16>(slot);
        if constexpr (store_hash)
            ring.hash[s] = hash;
        prefetchCell(cell);
        return true;
    }

    template <typename RingT>
    ALWAYS_INLINE AmacStepResult step(RingT & ring, size_t s)
    {
        const Cell * cell = ring.cell[s];
        if (cell->isZero(no_state))
        {
            found_word[ring.row[s]] = 0;
            return AmacStepResult::Done;
        }
        const StoredKey & key = ring.key[s];
        /// Only the saved-hash cells (the string keys) consume a hash in `keyEquals` - as the
        /// compare prefilter; every other cell ignores the argument, so those rings pass a
        /// literal zero instead of recomputing (or storing) a value nothing reads.
        size_t hash = 0;
        if constexpr (store_hash)
            hash = ring.hash[s];
        if (cell->keyEquals(key, hash, no_state))
        {
            recordHit(ring.row[s], ring.slot[s], cell);
            return AmacStepResult::Done;
        }
        const SlotMapDesc & desc = slot_descs[ring.slot[s]];
        const Cell * buf = static_cast<const Cell *>(desc.buf);
        if (++cell == buf + desc.mask + 1) [[unlikely]]
            cell = buf;
        ring.cell[s] = cell;
        prefetchCell(cell);
        return AmacStepResult::Advance;
    }
};

}

template <typename KeyGetter, typename Map, bool need_flags, bool selector_is_range>
void amacFindPass(
    KeyGetter & key_getter,
    const Map * const * slot_maps,
    const SlotMapDesc * slot_descs,
    UInt32 route_shift,
    size_t rows,
    size_t range_first,
    const UInt64 * selector_indexes,
    const UInt8 * skip_data,
    Arena & pool,
    /// Written through the policy's result-array fields, which the check cannot see behind the
    /// dependent `Policy` type.
    UInt64 * found_word, /// NOLINT(readability-non-const-parameter)
    UInt64 * found_offset, /// NOLINT(readability-non-const-parameter)
    UInt8 * found_slot) /// NOLINT(readability-non-const-parameter)
{
    static_assert(amac_probe_supported<KeyGetter, Map>);
    chassert(need_flags == (found_offset != nullptr));
    chassert(need_flags == (found_slot != nullptr));

    using Policy = AmacProbeDetail::AmacFindPolicy<KeyGetter, Map, need_flags, selector_is_range>;
    /// The selector view and the result arrays are re-based per chunk; the row-indexed side
    /// arrays (skip bytes) are indexed by the source row and need no re-base.
    for (size_t chunk_begin = 0; chunk_begin < rows; chunk_begin += Policy::chunk_rows_max)
    {
        const size_t chunk_rows = std::min(Policy::chunk_rows_max, rows - chunk_begin);
        Policy policy{
            .key_getter = key_getter,
            .map0 = *slot_maps[0],
            .slot_maps = slot_maps,
            .slot_descs = slot_descs,
            .route_shift = route_shift,
            .selector_base = range_first + chunk_begin,
            .selector_indexes = selector_indexes ? selector_indexes + chunk_begin : nullptr,
            .skip_data = skip_data,
            .pool = pool,
            .found_word = found_word + chunk_begin,
            .found_offset = found_offset ? found_offset + chunk_begin : nullptr,
            .found_slot = found_slot ? found_slot + chunk_begin : nullptr};
        amacRun(policy, chunk_rows);
    }

    /// Incremented ONCE per pass: per-chunk (let alone per-row) increments would put atomic
    /// traffic next to the ring.
    ProfileEvents::increment(ProfileEvents::ConcurrentHashJoinAmacProbeRows, rows);
}

}
