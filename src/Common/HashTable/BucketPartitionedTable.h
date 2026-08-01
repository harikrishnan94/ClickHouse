#pragma once

#include <concepts>
#include <base/types.h>

/** The bucket protocol every JOIN map answers, whatever its storage looks like underneath.
  *
  * A bucket-partitioned table is N independent sub-tables plus a routing function, and that is all a
  * caller may assume. In particular it may NOT assume that a bucket owns a contiguous region of
  * cells, nor that its own allocation is separate from its neighbours': `RuntimeStorage` gives each
  * bucket its own hash table, while `FixedRangeStorage` addresses one flat direct-addressed buffer
  * and uses the bucket only to route. Both satisfy this concept, which is the point - a build or
  * probe loop written against it works for either, and `bucketCount() == 1` is the serial case
  * rather than a separate code path.
  *
  * The two routing calls belong together and must be used as a pair:
  *   `getBucketFromHash(bucketRoutingHash(key, hash(key)))`
  * `bucketRoutingHash` exists because a table's cell-placement hash cannot always select a bucket -
  * a direct-addressed table's hash is the identity, which would pile dense-at-zero keys into one
  * bucket. Routing a row by the cell hash alone is therefore wrong for such a table; go through
  * `bucketRoutingHash`, which returns the cell hash when the two coincide.
  */
template <typename Map>
concept BucketPartitionedTable = requires(
    Map & map,
    const Map & const_map,
    typename Map::key_type key,
    typename Map::LookupResult & lookup,
    typename Map::ConstLookupResult const_lookup,
    bool & inserted,
    size_t hash_value)
{
    typename Map::key_type;
    typename Map::mapped_type;
    typename Map::value_type;
    typename Map::cell_type;
    typename Map::LookupResult;
    typename Map::ConstLookupResult;
    typename Map::iterator;
    typename Map::const_iterator;

    /// Routing. Identical spelling for every storage kind; see the note above on using the pair.
    { const_map.hash(key) } -> std::convertible_to<size_t>;
    { const_map.bucketRoutingHash(key, hash_value) } -> std::convertible_to<size_t>;
    { const_map.getBucketFromHash(hash_value) } -> std::convertible_to<size_t>;
    { const_map.bucketCount() } -> std::convertible_to<UInt32>;

    /// Insert and lookup.
    map.emplace(key, lookup, inserted);
    map.emplace(key, lookup, inserted, hash_value);
    { map.find(key) } -> std::same_as<typename Map::LookupResult>;
    { map.find(key, hash_value) } -> std::same_as<typename Map::LookupResult>;
    { const_map.has(key) } -> std::same_as<bool>;

    /// A stable offset, unique across buckets, in `[1, getBufferSizeInCells()]` for a live cell and
    /// 0 for an empty one. `HashJoin` indexes its per-row RIGHT/FULL used flags by it.
    { const_map.offsetInternal(const_lookup) } -> std::convertible_to<size_t>;

    /// Sizing. `getBufferSizeInCells()` bounds `offsetInternal`, so the two must agree.
    { const_map.size() } -> std::convertible_to<size_t>;
    { const_map.empty() } -> std::same_as<bool>;
    { const_map.getBufferSizeInBytes() } -> std::convertible_to<size_t>;
    { const_map.getBufferSizeInCells() } -> std::convertible_to<size_t>;

    /// Iteration visits every populated cell exactly once. It is NOT required to follow bucket
    /// order, and for a routed-but-flat storage it does not.
    { map.begin() } -> std::same_as<typename Map::iterator>;
    { map.end() } -> std::same_as<typename Map::iterator>;
    { const_map.begin() } -> std::same_as<typename Map::const_iterator>;
    { const_map.end() } -> std::same_as<typename Map::const_iterator>;
};

/** The above plus mapped-value access, which the JOIN maps all have (they are maps, not sets).
  */
template <typename Map>
concept BucketPartitionedMap = BucketPartitionedTable<Map> && requires(Map & map)
{
    map.forEachMapped([](typename Map::mapped_type &) {});
};
