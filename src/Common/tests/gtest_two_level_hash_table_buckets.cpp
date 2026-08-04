#include <gtest/gtest.h>

#include <Common/HashTable/HashMap.h>
#include <Common/HashTable/PartitionedFixedHashMap.h>
#include <Common/HashTable/TwoLevelHashTable.h>

#include <bit>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>


/** Tests for the parts of `TwoLevelHashTable` a bucket-parallel JOIN build relies on: the one-bucket
  * bucket count (`bits_for_bucket == 0`, where routing folds away and the table must behave as a
  * single-level one), a bucket-selection hash (`BucketHash`) that may differ from the cell-placement
  * hash, the prefix sums `offsetInternal` numbers cells by, and the direct-addressed storage whose
  * buckets route into one shared buffer.
  *
  * Ordinary 256-bucket aggregation use is covered by `gtest_hash_table.cpp` through
  * `TwoLevelHashMap`, and shares the same class, so nothing here can affect it.
  */

namespace
{

using Cell = HashMapCell<UInt64, UInt64, DefaultHash<UInt64>>;
using Impl = HashMapTable<UInt64, Cell, DefaultHash<UInt64>, TwoLevelHashTableGrower<>, HashTableAllocator>;

/// A real hash for cell placement, so bucket selection can reuse it (`BucketHash = void`).
template <Int32 bits>
using MapWithBits
    = TwoLevelHashTable<UInt64, Cell, DefaultHash<UInt64>, TwoLevelHashTableGrower<>, HashTableAllocator, Impl, bits>;

/// The two bucket counts a JOIN builds with: one bucket for a serial build, 256 for a parallel one.
using SerialMap = MapWithBits<0>;
using ParallelMap = MapWithBits<8>;

/// The `FixedHashMap` shape: the table's own hash is the identity, so it is useless for bucket
/// selection and an independent route word has to select the bucket instead. Same arrangement
/// the range map types (`key8`/`key16`, `range8_key32`..`range18_key64`) need.
struct IdentityHash
{
    [[maybe_unused]] size_t operator()(UInt64 x) const { return x; }
};

/// Stand-in for the join slot-routing fold used when cell hash cannot select the bucket.
/// Matches the UInt64 `routeWord` contract: CRC-32 on AArch64, golden-ratio multiply elsewhere.
ALWAYS_INLINE UInt32 routeWord(UInt64 key)
{
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
    return __crc32d(-1U, key);
#else
    return static_cast<UInt32>((key * 0x9E3779B97F4A7C15ULL) >> 32);
#endif
}

/// Stand-in for the open-addressing families' hash-derived slot: top bits of a 32-bit word.
ALWAYS_INLINE size_t joinHashRouteSlot(size_t hash, UInt32 route_shift)
{
    return static_cast<size_t>(static_cast<UInt32>(hash)) >> route_shift;
}

struct RouteWordBucketHash
{
    [[maybe_unused]] size_t operator()(UInt64 x) const { return routeWord(x); }
};

using IdentityCell = HashMapCell<UInt64, UInt64, IdentityHash>;
using IdentityImpl = HashMapTable<UInt64, IdentityCell, IdentityHash, TwoLevelHashTableGrower<>, HashTableAllocator>;

using RoutedMap = TwoLevelHashTable<
    UInt64,
    IdentityCell,
    IdentityHash,
    TwoLevelHashTableGrower<>,
    HashTableAllocator,
    IdentityImpl,
    /*bits_for_bucket=*/8,
    /*BucketHash=*/RouteWordBucketHash>;

void insertKeyValue(auto & map, auto key, UInt64 value)
{
    typename std::decay_t<decltype(map)>::LookupResult it = nullptr;
    bool inserted = false;
    map.emplace(key, it, inserted);
    if (inserted)
        new (&it->getMapped()) UInt64(value);
    else
        it->getMapped() = value;
}

/// The bucket `emplace`/`find` would route `key` to - the same routing `insertKeyValue` uses
/// internally, exposed so concurrent tests can take the matching external per-bucket lock.
size_t routedBucket(const auto & map, auto key)
{
    return map.getBucketFromHash(map.bucketRoutingHash(key, map.hash(key)));
}

}

TEST(TwoLevelHashTableBuckets, InsertAndFindAcrossBuckets)
{
    constexpr UInt64 num_keys = 100000;

    ParallelMap map;
    ASSERT_EQ(map.bucketCount(), 256u);
    ASSERT_TRUE(map.empty());

    for (UInt64 key = 1; key <= num_keys; ++key)
        insertKeyValue(map, key, key * 3);

    ASSERT_EQ(map.size(), num_keys);
    ASSERT_FALSE(map.empty());

    for (UInt64 key = 1; key <= num_keys; ++key)
    {
        auto * it = map.find(key);
        ASSERT_NE(it, nullptr) << "key " << key << " not found";
        ASSERT_EQ(it->getMapped(), key * 3);
    }

    ASSERT_EQ(map.find(num_keys + 1), nullptr);

    /// Every bucket should have been used at this key count, and iteration should see every key.
    size_t non_empty_buckets = 0;
    for (UInt32 i = 0; i < map.bucketCount(); ++i)
        non_empty_buckets += !map.impls[i].empty();
    ASSERT_EQ(non_empty_buckets, map.bucketCount());

    size_t iterated = 0;
    for (auto it = map.begin(); it != map.end(); ++it)
        ++iterated;
    ASSERT_EQ(iterated, num_keys);
}

TEST(TwoLevelHashTableBuckets, OneBucketRoutesEverythingToItself)
{
    SerialMap map;
    ASSERT_EQ(map.bucketCount(), 1u);
    ASSERT_EQ(map.bucketShift(), 32u);

    for (UInt64 key = 1; key <= 1000; ++key)
        insertKeyValue(map, key, key);

    ASSERT_EQ(map.size(), 1000u);
    for (UInt64 key = 1; key <= 1000; ++key)
        ASSERT_NE(map.find(key), nullptr);

    /// Everything must land in the only bucket, whatever the hash says.
    ASSERT_EQ(map.impls[0].size(), 1000u);
    ASSERT_EQ(map.getBucketFromHash(0xFFFFFFFFFFFFFFFFULL), 0u);
}

TEST(TwoLevelHashTableBuckets, OneBucketNumbersCellsLikeASingleLevelTable)
{
    /// A serial JOIN uses the one-bucket table where the classic join uses a single-level `HashMap`,
    /// so it must number its cells the same way: the offset is the cell's own index within the one
    /// buffer, with no bucket prefix added and nothing to compute first.
    constexpr UInt64 num_keys = 20000;

    SerialMap map;
    for (UInt64 key = 1; key <= num_keys; ++key)
        insertKeyValue(map, key, key * 9);

    for (UInt64 key = 1; key <= num_keys; ++key)
    {
        auto * it = map.find(key);
        ASSERT_NE(it, nullptr) << "key " << key;
        /// `Impl::offsetInternal` is the single-level numbering; the two-level table must agree,
        /// both before and after the prefix sums exist.
        ASSERT_EQ(map.offsetInternal(it), map.impls[0].offsetInternal(it)) << "key " << key;
    }

    map.computeBucketPrefix();
    for (UInt64 key = 1; key <= num_keys; ++key)
    {
        auto * it = map.find(key);
        ASSERT_EQ(map.offsetInternalUnsafe(it), map.impls[0].offsetInternal(it)) << "key " << key;
        ASSERT_EQ(map.offsetInternalAtBucket(it, 0), map.impls[0].offsetInternal(it)) << "key " << key;
    }
}

TEST(TwoLevelHashTableBuckets, SizeHintReservesPerBucket)
{
    ParallelMap map(/*size_hint=*/size_t{256} * 1024);

    /// The hint is divided across buckets, so each bucket is pre-sized rather than the whole table.
    for (UInt32 i = 0; i < map.bucketCount(); ++i)
        ASSERT_GE(map.impls[i].getBufferSizeInCells(), 1024u);
}

TEST(TwoLevelHashTableBuckets, BucketHashDecorrelatesFromIdentityCellHash)
{
    constexpr UInt64 num_keys = 4096;

    RoutedMap map;
    for (UInt64 key = 1; key <= num_keys; ++key)
        insertKeyValue(map, key, key * 7);

    ASSERT_EQ(map.size(), num_keys);

    /// With the identity cell hash, bucketing by that hash would put these sequential keys into
    /// only the lowest handful of buckets; `routeWord` must spread them over (nearly) all of them.
    size_t non_empty_buckets = 0;
    for (UInt32 i = 0; i < map.bucketCount(); ++i)
        non_empty_buckets += !map.impls[i].empty();
    ASSERT_GT(non_empty_buckets, 200u) << "routeWord did not decorrelate bucket selection";

    /// And lookups must agree with insertion on the bucket, or rows visit buckets their keys were
    /// never inserted into.
    for (UInt64 key = 1; key <= num_keys; ++key)
    {
        auto * it = map.find(key);
        ASSERT_NE(it, nullptr) << "key " << key << " not found";
        ASSERT_EQ(it->getMapped(), key * 7);
    }
    ASSERT_EQ(map.find(num_keys + 1), nullptr);
}

TEST(TwoLevelHashTableBuckets, IsEmptyCellIsSoundUnderBucketHash)
{
    /// `isEmptyCell` is a correctness-bearing early-out for its caller
    /// (`ColumnsHashingImpl.h`: answering true means "no match" WITHOUT a `find()`), and a hash
    /// value alone cannot identify the bucket when bucket selection does not derive from it - so
    /// under a non-void `BucketHash` it must never claim emptiness.
    /// A 256-bucket table holds its sub-tables inline, so it is too large to keep two of them on the
    /// stack.
    auto routed = std::make_unique<RoutedMap>();
    for (UInt64 key = 1; key <= 1000; ++key)
        insertKeyValue(*routed, key, key);

    for (UInt64 key = 1; key <= 1000; ++key)
        ASSERT_FALSE(routed->isEmptyCell(routed->hash(key)));
    ASSERT_FALSE(routed->isEmptyCell(routed->hash(123456789)));

    /// With the default `BucketHash` the hash does identify the bucket, so the fast path stays.
    auto plain = std::make_unique<ParallelMap>();
    for (UInt64 key = 1; key <= 1000; ++key)
        insertKeyValue(*plain, key, key);
    for (UInt64 key = 1; key <= 1000; ++key)
        ASSERT_FALSE(plain->isEmptyCell(plain->hash(key)));
}

TEST(TwoLevelHashTableBuckets, OffsetInternalIsUniquePerCell)
{
    /// `HashJoin` indexes its per-offset RIGHT/FULL flags by `offsetInternal`, so offsets must be
    /// distinct across buckets - including when `BucketHash` decides the bucket.
    RoutedMap map;
    constexpr UInt64 num_keys = 2000;
    for (UInt64 key = 1; key <= num_keys; ++key)
        insertKeyValue(map, key, key);

    std::unordered_set<size_t> offsets;
    for (UInt64 key = 1; key <= num_keys; ++key)
    {
        auto * it = map.find(key);
        ASSERT_NE(it, nullptr);
        const size_t offset = map.offsetInternal(it);
        ASSERT_LE(offset, map.getBufferSizeInCells());
        ASSERT_TRUE(offsets.insert(offset).second) << "duplicate offset " << offset << " for key " << key;
    }
}

TEST(TwoLevelHashTableBuckets, OffsetInternalUnsafeMatchesSafeAfterComputeBucketPrefix)
{
    /// `computeBucketPrefix()` + `offsetInternalUnsafe()` is the hot-loop pattern
    /// (`Unified::HashJoin::freezeMapsForProbing` then the probe): compute the prefix sums once when
    /// the build ends, then look up offsets without paying the "already computed" check
    /// `offsetInternal()` makes on every call. The two must agree.
    RoutedMap map;
    constexpr UInt64 num_keys = 2000;
    for (UInt64 key = 1; key <= num_keys; ++key)
        insertKeyValue(map, key, key);

    map.computeBucketPrefix();

    std::unordered_set<size_t> offsets;
    for (UInt64 key = 1; key <= num_keys; ++key)
    {
        auto * it = map.find(key);
        ASSERT_NE(it, nullptr);
        const size_t safe_offset = map.offsetInternal(it);
        const size_t unsafe_offset = map.offsetInternalUnsafe(it);
        ASSERT_EQ(safe_offset, unsafe_offset) << "key " << key;
        ASSERT_TRUE(offsets.insert(unsafe_offset).second) << "duplicate offset for key " << key;
    }
}

TEST(TwoLevelHashTableBuckets, ConcurrentBuildWithExternalBucketLocks)
{
    /// The table does not synchronize internally - callers do, exactly as for the underlying hash
    /// tables. N worker threads insert into the one shared table under external per-bucket locks;
    /// growth of one bucket must not disturb another.
    constexpr size_t num_threads = 16;
    constexpr UInt64 keys_per_thread = 20000;

    ParallelMap map;
    std::vector<std::mutex> bucket_mutexes(map.bucketCount());

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&map, &bucket_mutexes, t]
        {
            const UInt64 begin = t * keys_per_thread + 1;
            for (UInt64 key = begin; key < begin + keys_per_thread; ++key)
            {
                std::lock_guard lock(bucket_mutexes[routedBucket(map, key)]);
                insertKeyValue(map, key, key * 5);
            }
        });
    }
    for (auto & thread : threads)
        thread.join();

    ASSERT_EQ(map.size(), num_threads * keys_per_thread);

    /// Offsets are published once, after the build, and must then number every cell the parallel
    /// build inserted exactly once.
    map.computeBucketPrefix();
    std::unordered_set<size_t> offsets;
    for (UInt64 key = 1; key <= num_threads * keys_per_thread; ++key)
    {
        auto * it = map.find(key);
        ASSERT_NE(it, nullptr) << "key " << key << " lost by the concurrent build";
        ASSERT_EQ(it->getMapped(), key * 5) << "mapped value of key " << key << " was corrupted";
        const size_t offset = map.offsetInternalUnsafe(it);
        ASSERT_EQ(offset, map.offsetInternal(it)) << "key " << key;
        ASSERT_TRUE(offsets.insert(offset).second) << "duplicate offset for key " << key;
    }
}

TEST(TwoLevelHashTableBuckets, BucketSelectionMatchesJoinHashRouteSlot)
{
    /// A build/probe contract: `ConcurrentHashJoin` computes a row's target partition with
    /// `joinHashRouteSlot`, and the table must pick the same partition for the same hash, or probe
    /// rows visit buckets their keys were never inserted into. The two formulas are written
    /// differently - `(UInt32)h >> (32 - b)` against `(h >> (32 - b)) & (2^b - 1)` - so pin them
    /// against each other over both bucket counts and hash shapes, including the high bits that
    /// only one of the two expressions sees.
    auto check = [](auto & map)
    {
        const auto route_shift = static_cast<UInt32>(32 - std::countr_zero(map.bucketCount()));
        ASSERT_EQ(map.bucketShift(), route_shift);

        for (const size_t hash_value : {size_t(0),
                                        size_t(1),
                                        size_t(0xFFFFFFFFULL),
                                        size_t(0x100000000ULL),
                                        size_t(0xFFFFFFFFFFFFFFFFULL),
                                        size_t(0x123456789ABCDEFULL),
                                        size_t(0xDEADBEEF00000000ULL),
                                        size_t(0x00000000DEADBEEFULL)})
        {
            ASSERT_EQ(map.getBucketFromHash(hash_value), joinHashRouteSlot(hash_value, route_shift))
                << "num_buckets " << map.bucketCount() << ", hash " << hash_value;
        }
    };

    MapWithBits<0> one_bucket;
    MapWithBits<1> two_buckets;
    MapWithBits<4> sixteen_buckets;
    auto full = std::make_unique<MapWithBits<8>>();
    check(one_bucket);
    check(two_buckets);
    check(sixteen_buckets);
    check(*full);
}

TEST(TwoLevelHashTableBuckets, ReserveSizesEveryBucket)
{
    ParallelMap map;
    map.reserve(map.bucketCount() * 2048);

    for (UInt32 i = 0; i < map.bucketCount(); ++i)
        ASSERT_GE(map.impls[i].getBufferSizeInCells(), 2048u);

    /// And the table must still work afterwards.
    for (UInt64 key = 1; key <= 10000; ++key)
        insertKeyValue(map, key, key);
    ASSERT_EQ(map.size(), 10000u);
    for (UInt64 key = 1; key <= 10000; ++key)
        ASSERT_NE(map.find(key), nullptr);
}

TEST(TwoLevelHashTableBuckets, ForEachMappedVisitsEveryBucket)
{
    ParallelMap map;
    constexpr UInt64 num_keys = 5000;
    for (UInt64 key = 1; key <= num_keys; ++key)
        insertKeyValue(map, key, key);

    /// `HashJoin`'s post-build re-ranging rewrites mapped values through `forEachMapped`, so it must
    /// reach every bucket's cells and the writes must stick.
    size_t visited = 0;
    map.forEachMapped([&](UInt64 & mapped)
    {
        ++visited;
        mapped *= 2;
    });
    ASSERT_EQ(visited, num_keys);

    for (UInt64 key = 1; key <= num_keys; ++key)
    {
        auto * it = map.find(key);
        ASSERT_NE(it, nullptr);
        ASSERT_EQ(it->getMapped(), key * 2);
    }
}

TEST(TwoLevelHashTableBuckets, OffsetsStayValidAfterRecomputingPrefixPostGrowth)
{
    /// `offsetInternal()`'s prefix-sum cache is computed once, lazily, on first use, and does NOT
    /// notice later bucket growth on its own - there is no internal tracking of buffer changes,
    /// by design (synchronization and cache freshness are the caller's responsibility, like the
    /// underlying hash tables). `StorageJoin` can insert again after offsets have already been
    /// handed out, so a caller that needs correct offsets afterward must call
    /// `computeBucketPrefix()` again itself before trusting further offsets.
    MapWithBits<4> map;
    for (UInt64 key = 1; key <= 200; ++key)
        insertKeyValue(map, key, key);

    /// Take offsets once, so the lazily-computed prefix sums are now populated.
    for (UInt64 key = 1; key <= 200; ++key)
        ASSERT_GT(map.offsetInternal(map.find(key)), 0u);

    const size_t cells_before = map.getBufferSizeInCells();
    for (UInt64 key = 201; key <= 40000; ++key)
        insertKeyValue(map, key, key);
    ASSERT_GT(map.getBufferSizeInCells(), cells_before) << "test did not actually trigger growth";

    /// The caller's responsibility after further inserts: recompute explicitly.
    map.computeBucketPrefix();

    /// Every offset must now be unique and in range against the NEW capacities.
    std::unordered_set<size_t> offsets;
    for (UInt64 key = 1; key <= 40000; ++key)
    {
        auto * it = map.find(key);
        ASSERT_NE(it, nullptr);
        const size_t offset = map.offsetInternalUnsafe(it);
        ASSERT_LE(offset, map.getBufferSizeInCells());
        ASSERT_TRUE(offsets.insert(offset).second) << "duplicate offset for key " << key;
    }
}

TEST(TwoLevelHashTableBuckets, ConcurrentBuildWithContendedKeys)
{
    /// Same, but every thread inserts the SAME key range through external bucket locks, so
    /// threads collide on cells inside a bucket instead of only on the bucket itself.
    constexpr size_t num_threads = 16;
    constexpr UInt64 num_keys = 5000;

    MapWithBits<6> map;
    std::vector<std::mutex> bucket_mutexes(map.bucketCount());

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&map, &bucket_mutexes]
        {
            for (UInt64 key = 1; key <= num_keys; ++key)
            {
                std::lock_guard lock(bucket_mutexes[routedBucket(map, key)]);
                insertKeyValue(map, key, key * 11);
            }
        });
    }
    for (auto & thread : threads)
        thread.join();

    ASSERT_EQ(map.size(), num_keys);
    for (UInt64 key = 1; key <= num_keys; ++key)
    {
        auto * it = map.find(key);
        ASSERT_NE(it, nullptr);
        ASSERT_EQ(it->getMapped(), key * 11);
    }
}

TEST(TwoLevelHashTableBuckets, DirectAddressedBucketsRouteIntoOneBuffer)
{
    /// `PartitionedFixedHashMap` is direct-addressed: `buf[key]` finds the cell, so every bucket IS
    /// the one flat table and the bucket only names a lock. Memory, addressing and offsets must
    /// therefore be those of a plain `FixedHashMap`, while routing still spreads keys over buckets
    /// so that a bucket-parallel build gets disjointness.
    using RangeMap = PartitionedFixedHashMap<UInt16, UInt64>;

    RangeMap map;
    ASSERT_EQ(map.bucketCount(), 256u);

    constexpr UInt64 num_keys = 4096;
    for (UInt64 key = 1; key <= num_keys; ++key)
        insertKeyValue(map, static_cast<UInt16>(key), key * 3);

    ASSERT_EQ(map.size(), num_keys);

    /// One buffer: every bucket reports the same one, and the whole table is not counted per bucket.
    for (UInt32 i = 1; i < map.bucketCount(); ++i)
        ASSERT_EQ(map.impls[i].getBufferSizeInBytes(), map.impls[0].getBufferSizeInBytes());
    ASSERT_EQ(map.getBufferSizeInBytes(), map.impls[0].getBufferSizeInBytes());

    /// Routing must still spread the keys, or a parallel build would serialize on one lock.
    std::unordered_set<size_t> used_buckets;
    for (UInt64 key = 1; key <= num_keys; ++key)
        used_buckets.insert(routedBucket(map, static_cast<UInt16>(key)));
    ASSERT_GT(used_buckets.size(), 200u) << "keys did not spread over the routing buckets";

    /// Offsets are already global here - there is only one buffer to be an offset into - and every
    /// populated cell must still get a distinct one.
    std::unordered_set<size_t> offsets;
    for (UInt64 key = 1; key <= num_keys; ++key)
    {
        auto * it = map.find(static_cast<UInt16>(key));
        ASSERT_NE(it, nullptr) << "key " << key;
        ASSERT_EQ(it->getMapped(), key * 3);
        const size_t offset = map.offsetInternal(it);
        ASSERT_LE(offset, map.getBufferSizeInCells());
        ASSERT_TRUE(offsets.insert(offset).second) << "duplicate offset for key " << key;
    }

    /// Iteration visits each populated cell once, not once per bucket.
    size_t iterated = 0;
    for (auto it = map.begin(); it != map.end(); ++it)
        ++iterated;
    ASSERT_EQ(iterated, num_keys);
}
