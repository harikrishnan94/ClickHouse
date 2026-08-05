#pragma once

#include <concepts>
#include <base/types.h>

/** Bucket-partitioned maps expose routing and iteration without exposing storage layout.
 * The routing hash must select the bucket, especially for direct-addressed storage.
 * Use `getBucketFromHash(bucketRoutingHash(key, hash(key)))`; placement and routing hashes may differ.
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

    { const_map.hash(key) } -> std::convertible_to<size_t>;
    { const_map.bucketRoutingHash(key, hash_value) } -> std::convertible_to<size_t>;
    { const_map.getBucketFromHash(hash_value) } -> std::convertible_to<size_t>;
    { const_map.bucketCount() } -> std::convertible_to<UInt32>;

    map.emplace(key, lookup, inserted);
    map.emplace(key, lookup, inserted, hash_value);
    { map.find(key) } -> std::same_as<typename Map::LookupResult>;
    { map.find(key, hash_value) } -> std::same_as<typename Map::LookupResult>;
    { const_map.has(key) } -> std::same_as<bool>;

    { const_map.offsetInternal(const_lookup) } -> std::convertible_to<size_t>;

    /// Iteration already supplies the bucket; `computeBucketPrefix()` must have run.
    { const_map.offsetInternalAtBucket(const_lookup, size_t{}) } -> std::convertible_to<size_t>;

    { const_map.size() } -> std::convertible_to<size_t>;
    { const_map.empty() } -> std::same_as<bool>;
    { const_map.getBufferSizeInBytes() } -> std::convertible_to<size_t>;
    { const_map.getBufferSizeInCells() } -> std::convertible_to<size_t>;

    { map.begin() } -> std::same_as<typename Map::iterator>;
    { map.end() } -> std::same_as<typename Map::iterator>;
    { const_map.begin() } -> std::same_as<typename Map::const_iterator>;
    { const_map.end() } -> std::same_as<typename Map::const_iterator>;
};

template <typename Map>
concept BucketPartitionedMap = BucketPartitionedTable<Map> && requires(Map & map)
{
    map.forEachMapped([](typename Map::mapped_type &) {});
};
