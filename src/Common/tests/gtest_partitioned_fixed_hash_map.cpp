#include <gtest/gtest.h>

#include <Common/HashTable/PartitionedFixedHashMap.h>

#include <bit>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>


/** Tests for `PartitionedFixedHashMap` - a direct-addressed `FixedHashMap` that answers the bucket
  * protocol so the same bucket-parallel build/probe code serves it and the open-addressing JOIN
  * maps.
  *
  * The invariant under test throughout: routing changes which lock a key belongs under, and nothing
  * else. Cells, offsets, buffer size and iteration must be indistinguishable from a plain
  * `FixedHashMap` at every bucket count.
  */

namespace
{

using Mapped = UInt64;

template <typename Key, size_t size_bits>
using Partitioned = PartitionedFixedHashMap<Key, Mapped, size_bits>;

template <typename Key, size_t size_bits>
using Plain = FixedHashMapWithSizeBits<Key, Mapped, size_bits>;

/// Bucket counts exercised by every structural test. 1 is the shipped value and the serial case.
const std::vector<size_t> bucket_counts = {1, 2, 16, 256};

template <typename Map>
void insertKeyValue(Map & map, typename Map::key_type key, UInt64 value)
{
    typename Map::LookupResult it = nullptr;
    bool inserted = false;
    map.emplace(key, it, inserted);
    if (inserted)
        new (&it->getMapped()) UInt64(value);
    else
        it->getMapped() = value;
}

/// The bucket `emplace`/`find` route `key` to. This is the pair a caller must use - the cell hash
/// alone cannot select the bucket for a direct-addressed table, whose hash is the identity.
template <typename Map>
size_t routedBucket(const Map & map, typename Map::key_type key)
{
    return map.getBucketFromHash(map.bucketRoutingHash(key, map.hash(key)));
}

template <typename Map>
constexpr size_t cellsPerLine()
{
    return std::bit_floor(std::max<size_t>(1, DB::CH_CACHE_LINE_SIZE / sizeof(typename Map::cell_type)));
}

/// Offsets of every populated cell, collected the way `NotJoinedHash::fillColumns` collects them:
/// through the iterator's raw cell pointer, which is the only access pattern the JOIN uses.
template <typename Map>
std::vector<size_t> offsetsByIteration(Map & map)
{
    std::vector<size_t> offsets;
    for (auto it = map.begin(); it != map.end(); ++it)
        offsets.push_back(map.offsetInternal(it.getPtr()));
    return offsets;
}

}


TEST(PartitionedFixedHashMap, OffsetsMatchAPlainFixedHashMap)
{
    /// `HashJoin` sizes its per-row RIGHT/FULL used flags by `getBufferSizeInCells() + 1` and indexes
    /// them by `offsetInternal`, so partitioning must not renumber a single cell.
    constexpr size_t size_bits = 16;
    constexpr UInt32 num_keys = 5000;

    Plain<UInt32, size_bits> plain;
    for (UInt32 key = 0; key < num_keys; ++key)
        insertKeyValue(plain, key, key);

    for (const size_t num_buckets : bucket_counts)
    {
        Partitioned<UInt32, size_bits> map(num_buckets);
        for (UInt32 key = 0; key < num_keys; ++key)
            insertKeyValue(map, key, key);

        ASSERT_EQ(map.size(), num_keys) << "num_buckets " << num_buckets;

        for (UInt32 key = 0; key < num_keys; ++key)
        {
            const auto * partitioned_cell = map.find(key);
            const auto * plain_cell = plain.find(key);
            ASSERT_NE(partitioned_cell, nullptr) << "key " << key << ", num_buckets " << num_buckets;
            ASSERT_NE(plain_cell, nullptr) << "key " << key;
            ASSERT_EQ(map.offsetInternal(partitioned_cell), plain.offsetInternal(plain_cell))
                << "key " << key << " got a different offset at num_buckets " << num_buckets;
        }

        ASSERT_EQ(map.find(num_keys + 1), nullptr) << "num_buckets " << num_buckets;
    }
}


TEST(PartitionedFixedHashMap, BufferSizeIsIndependentOfBucketCount)
{
    /// A bucket is a route, not an allocation: adding buckets must not add cells. If this fails the
    /// used-flags array grows with the bucket count and the memory win of one flat buffer is gone.
    constexpr size_t size_bits = 16;

    for (const size_t num_buckets : bucket_counts)
    {
        Partitioned<UInt32, size_bits> map(num_buckets);
        ASSERT_EQ(map.bucketCount(), num_buckets);
        ASSERT_EQ(map.getBufferSizeInCells(), 1ULL << size_bits) << "num_buckets " << num_buckets;
        ASSERT_EQ(map.getBufferSizeInBytes(), (1ULL << size_bits) * sizeof(Partitioned<UInt32, size_bits>::cell_type))
            << "num_buckets " << num_buckets;
        ASSERT_TRUE(map.empty());
    }
}


TEST(PartitionedFixedHashMap, IterationVisitsEveryCellExactlyOnce)
{
    /// Buckets share the cells, so iterating per bucket would walk the whole table once per bucket.
    /// The storage reports a single iteration partition to prevent that; this is the test for it.
    constexpr size_t size_bits = 16;
    constexpr UInt32 num_keys = 3000;

    for (const size_t num_buckets : bucket_counts)
    {
        Partitioned<UInt32, size_bits> map(num_buckets);
        for (UInt32 key = 0; key < num_keys; ++key)
            insertKeyValue(map, key, key * 3);

        const auto offsets = offsetsByIteration(map);
        ASSERT_EQ(offsets.size(), num_keys) << "num_buckets " << num_buckets;

        const std::unordered_set<size_t> unique(offsets.begin(), offsets.end());
        ASSERT_EQ(unique.size(), num_keys) << "iteration repeated a cell at num_buckets " << num_buckets;

        for (UInt32 key = 0; key < num_keys; ++key)
            ASSERT_TRUE(unique.contains(map.offsetInternal(map.find(key))))
                << "key " << key << " was never visited at num_buckets " << num_buckets;

        /// `tryRerangeRightTableData` rewrites mapped values through this, so it must agree.
        size_t visited = 0;
        map.forEachMapped([&](UInt64 & mapped) { ++visited; mapped += 1; });
        ASSERT_EQ(visited, num_keys) << "forEachMapped at num_buckets " << num_buckets;
        for (UInt32 key = 0; key < num_keys; ++key)
            ASSERT_EQ(map.find(key)->getMapped(), key * 3 + 1) << "key " << key;
    }
}


TEST(PartitionedFixedHashMap, EveryKeyRoutesToExactlyOneBucketAndRoutingIsStable)
{
    constexpr size_t size_bits = 16;
    constexpr UInt32 num_keys = 4000;

    for (const size_t num_buckets : bucket_counts)
    {
        Partitioned<UInt32, size_bits> map(num_buckets);

        std::vector<size_t> bucket_of_key(num_keys);
        for (UInt32 key = 0; key < num_keys; ++key)
        {
            bucket_of_key[key] = routedBucket(map, key);
            ASSERT_LT(bucket_of_key[key], num_buckets) << "key " << key;
        }

        /// Build routes and probe routes are the same function of the key; if they could drift, rows
        /// would be looked up under a lock other than the one they were inserted under.
        for (UInt32 key = 0; key < num_keys; ++key)
        {
            insertKeyValue(map, key, key);
            ASSERT_EQ(routedBucket(map, key), bucket_of_key[key]) << "routing moved for key " << key;
        }

        ASSERT_EQ(map.getBucketFromHash(map.bucketRoutingHash(UInt32(0), 0)), 0u) << "num_buckets " << num_buckets;
    }
}


TEST(PartitionedFixedHashMap, ACacheLineNeverSpansTwoBuckets)
{
    /// The reason routing shifts the key before multiplying. Without it, neighbouring cells land in
    /// different buckets and two threads holding different bucket locks write the same cache line.
    constexpr size_t size_bits = 16;
    using Map = Partitioned<UInt32, size_bits>;
    constexpr size_t cells_per_line = cellsPerLine<Map>();

    for (const size_t num_buckets : bucket_counts)
    {
        Map map(num_buckets);
        for (UInt32 line_start = 0; line_start < (1U << size_bits); line_start += cells_per_line)
        {
            const size_t expected = routedBucket(map, line_start);
            for (size_t i = 1; i < cells_per_line; ++i)
                ASSERT_EQ(routedBucket(map, static_cast<UInt32>(line_start + i)), expected)
                    << "cache line starting at " << line_start << " spans two buckets at num_buckets " << num_buckets;
        }
    }
}


TEST(PartitionedFixedHashMap, SpreadsDenseAtZeroKeys)
{
    /// The case that rules out routing by the high bits of the key. `tryConvertToFixedHashMapImpl`
    /// puts a range in the smallest power-of-two table that fits it, so a 300-key range lives in a
    /// 65536-cell table and every key shares the same high bits. High-bit routing would put all 300
    /// keys in bucket 0.
    constexpr size_t size_bits = 16;
    constexpr UInt32 num_keys = 300;
    constexpr size_t num_buckets = 16;

    Partitioned<UInt32, size_bits> map(num_buckets);
    std::vector<size_t> per_bucket(num_buckets, 0);
    for (UInt32 key = 0; key < num_keys; ++key)
    {
        insertKeyValue(map, key, key);
        ++per_bucket[routedBucket(map, key)];
    }

    size_t non_empty = 0;
    size_t largest = 0;
    for (const size_t count : per_bucket)
    {
        non_empty += count != 0;
        largest = std::max(largest, count);
    }

    ASSERT_GE(non_empty, 14u) << "dense-at-zero keys reached only " << non_empty << " of " << num_buckets << " buckets";
    ASSERT_LE(largest, num_keys / 4) << "one bucket took " << largest << " of " << num_keys << " keys";
}


TEST(PartitionedFixedHashMap, SpreadsKeysWithLowBitStructure)
{
    /// The case that rules out routing by the LOW bits of the key, which is what
    /// `ConcurrentHashJoin::hashToSelector` falls back to for a fixed map. Aligned keys - every id a
    /// multiple of 256 - all have the same low bits, so `key & (num_buckets - 1)` would put every
    /// one of them in bucket 0.
    constexpr size_t size_bits = 16;
    constexpr UInt32 stride = 256;
    constexpr UInt32 num_keys = 256;
    constexpr size_t num_buckets = 16;

    Partitioned<UInt32, size_bits> map(num_buckets);
    std::vector<size_t> per_bucket(num_buckets, 0);
    for (UInt32 i = 0; i < num_keys; ++i)
    {
        const UInt32 key = i * stride;
        insertKeyValue(map, key, key);
        ++per_bucket[routedBucket(map, key)];
    }

    size_t non_empty = 0;
    size_t largest = 0;
    for (const size_t count : per_bucket)
    {
        non_empty += count != 0;
        largest = std::max(largest, count);
    }

    ASSERT_GE(non_empty, 14u) << "aligned keys reached only " << non_empty << " of " << num_buckets << " buckets";
    ASSERT_LE(largest, num_keys / 4) << "one bucket took " << largest << " of " << num_keys << " keys";
}


TEST(PartitionedFixedHashMap, SmallKeyTypeIsFullyAddressable)
{
    /// `key8` covers its whole key space, so every key must be storable and iteration must find
    /// them all - including at more buckets than there are cache lines.
    constexpr size_t size_bits = 8;

    for (const size_t num_buckets : bucket_counts)
    {
        Partitioned<UInt8, size_bits> map(num_buckets);
        for (size_t key = 0; key < 256; ++key)
            insertKeyValue(map, static_cast<UInt8>(key), key);

        ASSERT_EQ(map.size(), 256u) << "num_buckets " << num_buckets;
        ASSERT_EQ(map.getBufferSizeInCells(), 256u) << "num_buckets " << num_buckets;

        for (size_t key = 0; key < 256; ++key)
        {
            const auto * cell = map.find(static_cast<UInt8>(key));
            ASSERT_NE(cell, nullptr) << "key " << key << ", num_buckets " << num_buckets;
            ASSERT_EQ(cell->getMapped(), key);
            ASSERT_EQ(map.offsetInternal(cell), key + 1) << "key " << key;
            ASSERT_TRUE(map.has(static_cast<UInt8>(key)));
        }

        ASSERT_EQ(offsetsByIteration(map).size(), 256u) << "num_buckets " << num_buckets;
    }
}


TEST(PartitionedFixedHashMap, ConcurrentBuildWithExternalBucketLocks)
{
    /// The point of the whole change. Workers insert into one shared table under per-bucket locks;
    /// because distinct keys are distinct cells, routed disjointness is real disjointness. Nothing
    /// synchronizes internally, exactly as for the underlying hash tables.
    constexpr size_t size_bits = 18;
    constexpr size_t num_buckets = 64;
    constexpr size_t num_threads = 16;
    constexpr UInt32 keys_per_thread = 10000;

    Partitioned<UInt32, size_bits> map(num_buckets);
    std::vector<std::mutex> bucket_mutexes(num_buckets);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&map, &bucket_mutexes, t]
        {
            const UInt32 begin = static_cast<UInt32>(t) * keys_per_thread;
            for (UInt32 key = begin; key < begin + keys_per_thread; ++key)
            {
                std::lock_guard lock(bucket_mutexes[routedBucket(map, key)]);
                insertKeyValue(map, key, key * 5);
            }
        });
    }
    for (auto & thread : threads)
        thread.join();

    constexpr UInt32 total_keys = num_threads * keys_per_thread;
    ASSERT_EQ(map.size(), total_keys);
    for (UInt32 key = 0; key < total_keys; ++key)
    {
        const auto * cell = map.find(key);
        ASSERT_NE(cell, nullptr) << "key " << key << " lost by the concurrent build";
        ASSERT_EQ(cell->getMapped(), key * 5) << "mapped value of key " << key << " was corrupted";
        ASSERT_EQ(map.offsetInternal(cell), key + 1) << "key " << key;
    }

    /// And the table must still iterate cleanly afterwards.
    ASSERT_EQ(offsetsByIteration(map).size(), total_keys);
}


TEST(PartitionedFixedHashMap, ConcurrentBuildWithContendedKeys)
{
    /// Same, but every thread inserts the same keys, so threads collide on cells inside a bucket
    /// rather than only on the bucket itself.
    constexpr size_t size_bits = 16;
    constexpr size_t num_buckets = 32;
    constexpr size_t num_threads = 16;
    constexpr UInt32 num_keys = 4000;

    Partitioned<UInt32, size_bits> map(num_buckets);
    std::vector<std::mutex> bucket_mutexes(num_buckets);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&map, &bucket_mutexes]
        {
            for (UInt32 key = 0; key < num_keys; ++key)
            {
                std::lock_guard lock(bucket_mutexes[routedBucket(map, key)]);
                insertKeyValue(map, key, key * 11);
            }
        });
    }
    for (auto & thread : threads)
        thread.join();

    ASSERT_EQ(map.size(), num_keys);
    for (UInt32 key = 0; key < num_keys; ++key)
    {
        const auto * cell = map.find(key);
        ASSERT_NE(cell, nullptr) << "key " << key;
        ASSERT_EQ(cell->getMapped(), key * 11) << "key " << key;
    }
}
