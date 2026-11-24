#pragma once

#include <Common/HashTable/HashTable.h>
#include <Common/HashTable/HashTableKeyHolder.h>
#include <base/arithmeticOverflow.h>
#include <base/defines.h>
#include <base/types.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace DB
{

/// What a probe reads to reach the cells of one partition: 16 bytes, so 1024 partitions take a quarter of a 64 KiB L1.
/// The end of the cells is `cells + mask + 1`.
template <typename Cell>
struct JoinPartitionDesc
{
    const Cell * cells = nullptr;
    size_t mask = 0;
};

/** The hash table of one `ON` disjunct, planned with one partition. It is the map itself, so a probe has the loads of the
  * plain map, and every method of the map is available. `freeze` adds the descriptor that the AMAC ring reads.
  */
template <typename Map>
class JoinSingleView : public Map
{
public:
    using cell_type = typename Map::cell_type;
    using Desc = JoinPartitionDesc<cell_type>;

    static constexpr bool partitioned = false;

    /// The cells and the mask of the map, set by `freeze`. Valid until the next insert.
    [[nodiscard]] const Desc & desc(size_t) const { return frozen_desc; }
    void freeze()
    {
        frozen_desc = {this->buf, this->getBufferSizeInCells() - 1};
        frozen = true;
    }
    [[nodiscard]] bool isFrozen() const { return frozen; }

private:
    Desc frozen_desc;
    bool frozen = false;
};

namespace detail
{
template <typename Key, typename Cell, typename Hash, typename Grower, typename Allocator>
Hash joinMapHashOf(const HashTable<Key, Cell, Hash, Grower, Allocator> &);
}

/// The hash function of a `HashTable` map.
template <typename Map>
using JoinMapHashOf = decltype(detail::joinMapHashOf(std::declval<const Map &>()));

/** The hash table of one `ON` disjunct as P ordinary hash tables, one per partition, P = 2^k with 1 <= k <= 15. A key goes to
  * partition `(h >> (32 - k)) & (P - 1)` of the 64-bit hash `h` that the map computes, and to a cell by the low bits of `h`.
  *
  * A probe reads only `hot[p]`, and `prefix[p]` on a hit of a join with used flags. The map objects are cold: only the zero
  * key, the scans, the growth and the destruction reach them. `freeze` fills `hot` and `prefix` and caches the bytes of all
  * maps. A map is created by the thread that fills it (`createPartition`), so a partition that no filler reached stays null
  * until `freeze`, which creates the empty ones.
  */
template <typename Map>
class JoinPartitionedView
{
public:
    using map_type = Map;
    using key_type = typename Map::key_type;
    using mapped_type = typename Map::mapped_type;
    using value_type = typename Map::value_type;
    using cell_type = typename Map::cell_type;
    using LookupResult = typename Map::LookupResult;
    using ConstLookupResult = typename Map::ConstLookupResult;
    using Desc = JoinPartitionDesc<cell_type>;

    static constexpr bool partitioned = true;
    static constexpr size_t max_partition_bits = 15;

    /// `robin_hood_allowed_` is the setting `enable_robin_hood_hash_for_joins`, for the maps that the view creates itself.
    JoinPartitionedView(size_t partition_bits, bool robin_hood_allowed_)
        : shift(32 - partition_bits)
        , pmask((1uz << partition_bits) - 1)
        , robin_hood_allowed(robin_hood_allowed_)
        , maps(1uz << partition_bits)
    {
        chassert(partition_bits >= 1 && partition_bits <= max_partition_bits);
    }

    [[nodiscard]] size_t numPartitions() const { return maps.size(); }
    [[nodiscard]] ALWAYS_INLINE size_t partitionOfHash(size_t hash_value) const { return (hash_value >> shift) & pmask; }

    /// Null until `createPartition`.
    Map * partitionOrNull(size_t p) { return maps[p].get(); }
    [[nodiscard]] const Map * partitionOrNull(size_t p) const { return maps[p].get(); }
    Map & partition(size_t p) { return *maps[p]; }
    [[nodiscard]] const Map & partition(size_t p) const { return *maps[p]; }

    /// The keys that a map reserves when the view creates it; 0 for none.
    void setReserveOnCreate(size_t keys) { reserve_on_create = keys; }
    /// Whether a map that the view creates may switch to the Robin Hood mode.
    void setRobinHoodAllowedOnCreate(bool allowed) { robin_hood_allowed = allowed; }

    /// Creates the map of partition `p` if it does not exist. With `robin_hood_allowed_` false, the map never switches to the
    /// Robin Hood mode. Returns whether it created the map. Only the filler of partition `p` may call it.
    bool createPartition(size_t p, bool robin_hood_allowed_)
    {
        chassert(!frozen, "A partition was created after `freeze`");
        if (maps[p])
            return false;
        auto map = std::make_unique<Map>();
        if constexpr (GrowerWithRobinHood<typename Map::grower_type>)
            map->grower.setRobinHoodAllowed(robin_hood_allowed_);
        map->reserve(reserve_on_create);
        maps[p] = std::move(map);
        return true;
    }
    bool createPartition(size_t p) { return createPartition(p, robin_hood_allowed); }

    [[nodiscard]] ALWAYS_INLINE size_t hash(const key_type & key) const { return hasher(key); }

    /// The lookups of the probe. They read `hot` only, and the map object only for the zero key and for maps in mixed Robin
    /// Hood modes. Valid after `freeze`.
    [[nodiscard]] ALWAYS_INLINE ConstLookupResult find(const key_type & key, size_t hash_value) const
    {
        chassert(frozen);
        const size_t p = partitionOfHash(hash_value);
        /// The zero key is never in the cells: the map keeps it aside.
        if (cell_type::isZero(key, no_state) || robin_hood_mode == RobinHoodMode::Mixed)
            return std::as_const(*maps[p]).find(key, hash_value);
        const Desc d = hot[p];
        if (robin_hood_mode == RobinHoodMode::None)
        {
            const cell_type * cell = d.cells + (hash_value & d.mask);
            while (!cell->isZero(no_state))
            {
                if (cell->keyEquals(key, hash_value, no_state))
                    return cell;
                cell = d.cells + ((cell - d.cells + 1) & d.mask);
            }
            return nullptr;
        }
        /// The miss rule of `HashTable::findCellRobinHood`, as the AMAC ring applies it. The keys of a chain are in the order
        /// of their home cells, so the key is not in the table if a cell's key is closer to its home than the key would be
        /// there. The rule does not apply in the key's home cell.
        size_t place = hash_value & d.mask;
        for (size_t distance = 0;; ++distance)
        {
            const cell_type * cell = d.cells + place;
            if (cell->isZero(no_state))
                return nullptr;
            if (cell->keyEquals(key, hash_value, no_state))
                return cell;
            /// The subtraction wraps when the home cell is after `place` in the buffer; the mask takes it modulo the table size.
            if (distance && (common::subIgnoreOverflow(place, cell->getHash(hasher)) & d.mask) < distance)
                return nullptr;
            place = (place + 1) & d.mask;
        }
    }
    [[nodiscard]] ALWAYS_INLINE ConstLookupResult find(const key_type & key) const { return find(key, hash(key)); }
    ALWAYS_INLINE LookupResult find(const key_type & key, size_t hash_value)
    {
        return const_cast<LookupResult>(std::as_const(*this).find(key, hash_value));
    }
    ALWAYS_INLINE LookupResult find(const key_type & key) { return find(key, hash(key)); }

    /// The prefetch of the probe, from the descriptors. After `freeze`.
    template <typename KeyHolder>
    ALWAYS_INLINE void prefetch(KeyHolder && key_holder) const
    {
        const auto & key = keyHolderGetKey(key_holder);
        const size_t hash_value = hash(key);
        const Desc & d = hot[partitionOfHash(hash_value)];
        __builtin_prefetch(d.cells + (hash_value & d.mask));
        keyHolderDiscardKey(key_holder);
    }

    /// The prefetch of the insert of one build stream, before `freeze`: through the map of the partition, if it exists.
    template <typename KeyHolder>
    ALWAYS_INLINE void prefetchForInsert(KeyHolder && key_holder) const
    {
        const auto & key = keyHolderGetKey(key_holder);
        const size_t hash_value = hash(key);
        if (const Map * map = maps[partitionOfHash(hash_value)].get())
            map->prefetchByHash(hash_value);
        keyHolderDiscardKey(key_holder);
    }

    /// 0 for the zero key; otherwise the cell's position over all partitions plus one. After `freeze`.
    size_t offsetInternal(ConstLookupResult cell) const
    {
        if (cell->isZero(no_state))
            return 0;
        return offsetOf(cell, partitionOfHash(cell->getHash(hasher)));
    }
    ALWAYS_INLINE size_t offsetOf(const cell_type * cell, size_t p) const { return prefix[p] + (cell - hot[p].cells) + 1; }

    /// The insert of one build stream: puts the key into the partition of its hash, and creates the map of the partition at its
    /// first key.
    template <typename KeyHolder>
    ALWAYS_INLINE void emplace(KeyHolder && key_holder, LookupResult & it, bool & inserted, size_t hash_value)
    {
        chassert(!frozen, "An insert after `freeze` would leave the descriptors and the bytes stale");
        const size_t p = partitionOfHash(hash_value);
        if (unlikely(!maps[p]))
            createPartition(p);
        maps[p]->emplace(std::forward<KeyHolder>(key_holder), it, inserted, hash_value);
    }
    template <typename KeyHolder>
    ALWAYS_INLINE void emplace(KeyHolder && key_holder, LookupResult & it, bool & inserted)
    {
        const size_t hash_value = hash(keyHolderGetKey(key_holder));
        emplace(std::forward<KeyHolder>(key_holder), it, inserted, hash_value);
    }

    /// Fills `hot` and `prefix`, the Robin Hood summary and the byte count of all maps. Creates the maps that do not exist.
    /// Call it once, after the last insert.
    void freeze()
    {
        chassert(!frozen);
        const size_t num_partitions = maps.size();
        hot.resize(num_partitions);
        prefix.resize(num_partitions);
        size_t run = 0;
        size_t robin_hood = 0;
        frozen_bytes = 0;
        /// A partition that no filler reached stays empty, so its map gets no reserve: an empty build costs almost nothing.
        reserve_on_create = 0;
        for (size_t p = 0; p < num_partitions; ++p)
        {
            createPartition(p);
            const Map & map = *maps[p];
            const size_t cells = map.getBufferSizeInCells();
            hot[p] = {map.buf, cells - 1};
            prefix[p] = run;
            run += cells;
            robin_hood += map.isRobinHood();
            frozen_bytes += map.getBufferSizeInBytes();
        }
        /// The partition bits and the cell bits must not overlap, so no table may have more than 2^shift cells.
        chassert(std::ranges::all_of(hot, [&](const Desc & d) { return d.mask < (1uz << shift); }));
        if (robin_hood == 0)
            robin_hood_mode = RobinHoodMode::None;
        else if (robin_hood == num_partitions)
            robin_hood_mode = RobinHoodMode::All;
        else
            robin_hood_mode = RobinHoodMode::Mixed;
        frozen = true;
    }
    [[nodiscard]] bool isFrozen() const { return frozen; }

    [[nodiscard]] const Desc & desc(size_t p) const { return hot[p]; }
    [[nodiscard]] const Desc * descs() const { return hot.data(); }
    [[nodiscard]] const size_t * prefixes() const { return prefix.data(); }
    [[nodiscard]] size_t prefixOf(size_t p) const { return prefix[p]; }
    [[nodiscard]] size_t partitionShift() const { return shift; }
    [[nodiscard]] size_t partitionMask() const { return pmask; }

    enum class RobinHoodMode : uint8_t
    {
        None,
        All,
        Mixed,
    };
    /// Set by `freeze`.
    [[nodiscard]] RobinHoodMode robinHoodMode() const { return robin_hood_mode; }

    /// Before `freeze`, these sum the maps that exist. After it, they read what `freeze` recorded, so that a gate of the probe
    /// does not walk every map on every call.
    [[nodiscard]] size_t getBufferSizeInBytes() const
    {
        if (frozen)
            return frozen_bytes;
        size_t res = 0;
        for (const auto & map : maps)
            if (map)
                res += map->getBufferSizeInBytes();
        return res;
    }
    [[nodiscard]] size_t getBufferSizeInCells() const
    {
        if (frozen)
            return prefix.back() + hot.back().mask + 1;
        size_t res = 0;
        for (const auto & map : maps)
            if (map)
                res += map->getBufferSizeInCells();
        return res;
    }
    [[nodiscard]] size_t size() const
    {
        size_t res = 0;
        for (const auto & map : maps)
            if (map)
                res += map->size();
        return res;
    }
    [[nodiscard]] bool empty() const { return size() == 0; }

    template <typename Func>
    void forEachMapped(Func && func)
    {
        for (auto & map : maps)
            if (map)
                map->forEachMapped(func);
    }

private:
    static constexpr HashTableNoState no_state{};

    size_t shift;
    size_t pmask;
    bool robin_hood_allowed;
    size_t reserve_on_create = 0;
    /// The hash of the maps. It is stateless, as every hash of the join's maps.
    [[no_unique_address]] JoinMapHashOf<Map> hasher;
    std::vector<std::unique_ptr<Map>> maps;
    std::vector<Desc> hot;
    std::vector<size_t> prefix;
    RobinHoodMode robin_hood_mode = RobinHoodMode::None;
    bool frozen = false;
    size_t frozen_bytes = 0;
};

/** The two forms of the table of one key type: the single form, the map planned with one partition, and the partitioned form.
  * Exactly one of them exists; a new table has the single form. For `key16`, the partitioned form is the flat table with
  * cache-line buckets.
  */
template <typename Single, typename Partitioned>
struct JoinMapForms
{
    using single_type = Single;
    using partitioned_type = Partitioned;
    using key_type = typename Single::key_type;
    using mapped_type = typename Single::mapped_type;
    using value_type = typename Single::value_type;
    using cell_type = typename Single::cell_type;

    std::unique_ptr<Single> single = std::make_unique<Single>();
    std::unique_ptr<Partitioned> partitioned;

    [[nodiscard]] bool isPartitioned() const { return partitioned != nullptr; }

    /// Calls `f` with the form that exists.
    template <typename Self, typename F>
    decltype(auto) visit(this Self && self, F && f)
    {
        if (self.partitioned)
            return std::forward<F>(f)(*self.partitioned);
        return std::forward<F>(f)(*self.single);
    }

    [[nodiscard]] size_t getBufferSizeInBytes() const
    {
        return visit([](const auto & form) { return form.getBufferSizeInBytes(); });
    }
    [[nodiscard]] size_t getBufferSizeInCells() const
    {
        return visit([](const auto & form) { return form.getBufferSizeInCells(); });
    }
    [[nodiscard]] size_t size() const
    {
        return visit([](const auto & form) { return form.size(); });
    }
    [[nodiscard]] bool empty() const { return size() == 0; }
    template <typename Func>
    void forEachMapped(Func && func)
    {
        visit([&](auto & form) { form.forEachMapped(func); });
    }
};

template <typename T>
constexpr bool is_join_partitioned_view = false;
template <typename Map>
constexpr bool is_join_partitioned_view<JoinPartitionedView<Map>> = true;

template <typename T>
constexpr bool is_join_map_forms = false;
template <typename Single, typename Partitioned>
constexpr bool is_join_map_forms<JoinMapForms<Single, Partitioned>> = true;

}
