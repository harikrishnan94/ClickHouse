#include <gtest/gtest.h>

#include <Common/HashTable/HashMap.h>
#include <Common/HashTable/TwoLevelHashTable.h>

#include <bit>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>


/** Tests for `TwoLevelHashTable`'s `bits_for_bucket == -1` (runtime-sized) bucket-storage mode: a
  * bucket count fixed at construction time, and a bucket-selection hash (`BucketHash`) that may
  * differ from the cell-placement hash. Synchronization is external, as for the underlying hash
  * tables - concurrent tests below take their own per-bucket locks.
  *
  * The fixed-bucket (`bits_for_bucket >= 0`, default) mode is covered by `gtest_hash_table.cpp`
  * through `TwoLevelHashMap`, and shares the same class, so nothing here can affect it.
  */

namespace
{

using Cell = HashMapCell<UInt64, UInt64, DefaultHash<UInt64>>;
using Impl = HashMapTable<UInt64, Cell, DefaultHash<UInt64>, TwoLevelHashTableGrower<>, HashTableAllocator>;

/// A real hash for cell placement, so bucket selection can reuse it (`BucketHash = void`).
using DynamicMap = TwoLevelHashTable<
    UInt64,
    Cell,
    DefaultHash<UInt64>,
    TwoLevelHashTableGrower<>,
    HashTableAllocator,
    Impl,
    /*bits_for_bucket=*/-1>;

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

using BenchLookupResult = typename Impl::LookupResult;

using RoutedMap = TwoLevelHashTable<
    UInt64,
    IdentityCell,
    IdentityHash,
    TwoLevelHashTableGrower<>,
    HashTableAllocator,
    IdentityImpl,
    /*bits_for_bucket=*/-1,
    /*BucketHash=*/RouteWordBucketHash>;

void insertKeyValue(auto & map, UInt64 key, UInt64 value)
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
size_t routedBucket(const auto & map, UInt64 key)
{
    return map.getBucketFromHash(map.bucketRoutingHash(key, map.hash(key)));
}

}

TEST(TwoLevelHashTableDynamic, InsertAndFindAcrossBuckets)
{
    constexpr size_t num_buckets = 256;
    constexpr UInt64 num_keys = 100000;

    DynamicMap map(num_buckets);
    ASSERT_EQ(map.bucketCount(), num_buckets);
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
    ASSERT_EQ(non_empty_buckets, num_buckets);

    size_t iterated = 0;
    for (auto it = map.begin(); it != map.end(); ++it)
        ++iterated;
    ASSERT_EQ(iterated, num_keys);
}

TEST(TwoLevelHashTableDynamic, DegenerateSingleBucket)
{
    DynamicMap map(1);
    ASSERT_EQ(map.bucketCount(), 1u);

    for (UInt64 key = 1; key <= 1000; ++key)
        insertKeyValue(map, key, key);

    ASSERT_EQ(map.size(), 1000u);
    for (UInt64 key = 1; key <= 1000; ++key)
        ASSERT_NE(map.find(key), nullptr);

    /// Everything must land in the only bucket, whatever the hash says.
    ASSERT_EQ(map.impls[0].size(), 1000u);
    ASSERT_EQ(map.getBucketFromHash(0xFFFFFFFFFFFFFFFFULL), 0u);
}

TEST(TwoLevelHashTableDynamic, SingleBucketHoistsRoutingOutOfTheRowLoop)
{
    /// The bucket count is a runtime value, so a single-bucket table cannot be folded away at
    /// compile time - the routing has to be hoisted to a per-block test instead. `singleBucket()`
    /// is that test: non-null only for one bucket, and the pointer it hands back is the bucket's
    /// own `Impl`, so the selected loop calls `Impl::find` with no wrapper and no bucket
    /// indirection left to pay per row.
    constexpr UInt64 num_keys = 20000;

    DynamicMap flat_map(1);
    for (UInt64 key = 1; key <= num_keys; ++key)
        insertKeyValue(flat_map, key, key * 9);

    auto * flat = flat_map.singleBucket();
    ASSERT_NE(flat, nullptr);
    ASSERT_EQ(flat, &flat_map.impls[0]);

    /// The hoisted loop must agree with the routed one on every key, present or absent.
    for (UInt64 key = 1; key <= num_keys; ++key)
    {
        auto * hoisted = flat->find(key);
        auto * routed = flat_map.find(key);
        ASSERT_NE(hoisted, nullptr) << "key " << key;
        ASSERT_EQ(hoisted, routed) << "hoisted and routed lookups disagreed on key " << key;
        ASSERT_EQ(hoisted->getMapped(), key * 9);
    }
    ASSERT_EQ(flat->find(num_keys + 1), nullptr);
    ASSERT_EQ(flat_map.find(num_keys + 1), nullptr);

    /// Anything the hoisted loop inserts must be visible to the routed path too, and vice versa -
    /// they are the same storage, not two views that could drift.
    {
        BenchLookupResult it = nullptr;
        bool inserted = false;
        flat->emplace(num_keys + 1, it, inserted);
        ASSERT_TRUE(inserted);
        new (&it->getMapped()) UInt64(12345);
    }
    ASSERT_NE(flat_map.find(num_keys + 1), nullptr);
    ASSERT_EQ(flat_map.find(num_keys + 1)->getMapped(), 12345u);

    /// More than one bucket means there is no single bucket to hoist to, so the caller must take
    /// the routed loop. Getting this wrong would silently probe only bucket 0.
    for (size_t num_buckets = 2; num_buckets <= 256; num_buckets *= 2)
    {
        DynamicMap partitioned(num_buckets);
        ASSERT_EQ(partitioned.singleBucket(), nullptr) << "num_buckets " << num_buckets;
        const auto & const_ref = partitioned;
        ASSERT_EQ(const_ref.singleBucket(), nullptr) << "num_buckets " << num_buckets;
    }
}

TEST(TwoLevelHashTableDynamic, SizeHintReservesPerBucket)
{
    constexpr size_t num_buckets = 64;
    DynamicMap map(num_buckets, /*size_hint=*/num_buckets * 1024);

    /// The hint is divided across buckets, so each bucket is pre-sized rather than the whole table.
    for (UInt32 i = 0; i < map.bucketCount(); ++i)
        ASSERT_GE(map.impls[i].getBufferSizeInCells(), 1024u);
}

TEST(TwoLevelHashTableDynamic, BucketHashDecorrelatesFromIdentityCellHash)
{
    constexpr size_t num_buckets = 256;
    constexpr UInt64 num_keys = 4096;

    RoutedMap map(num_buckets);
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

TEST(TwoLevelHashTableDynamic, IsEmptyCellIsSoundUnderBucketHash)
{
    /// `isEmptyCell` is a correctness-bearing early-out for its caller
    /// (`ColumnsHashingImpl.h`: answering true means "no match" WITHOUT a `find()`), and a hash
    /// value alone cannot identify the bucket when bucket selection does not derive from it - so
    /// under a non-void `BucketHash` it must never claim emptiness.
    RoutedMap routed(256);
    for (UInt64 key = 1; key <= 1000; ++key)
        insertKeyValue(routed, key, key);

    for (UInt64 key = 1; key <= 1000; ++key)
        ASSERT_FALSE(routed.isEmptyCell(routed.hash(key)));
    ASSERT_FALSE(routed.isEmptyCell(routed.hash(123456789)));

    /// With the default `BucketHash` the hash does identify the bucket, so the fast path stays.
    DynamicMap plain(256);
    for (UInt64 key = 1; key <= 1000; ++key)
        insertKeyValue(plain, key, key);
    for (UInt64 key = 1; key <= 1000; ++key)
        ASSERT_FALSE(plain.isEmptyCell(plain.hash(key)));
}

TEST(TwoLevelHashTableDynamic, OffsetInternalIsUniquePerCell)
{
    /// `HashJoin` indexes its per-offset RIGHT/FULL flags by `offsetInternal`, so offsets must be
    /// distinct across buckets - including when `BucketHash` decides the bucket.
    RoutedMap map(64);
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

TEST(TwoLevelHashTableDynamic, OffsetInternalUnsafeMatchesSafeAfterComputeBucketPrefix)
{
    /// `computeBucketPrefix()` + `offsetInternalUnsafe()` is the hot-loop pattern: compute the
    /// prefix sums once ahead of a per-row loop, then look up offsets without paying the
    /// "already computed" check `offsetInternal()` makes on every call. The two must agree.
    RoutedMap map(64);
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

TEST(TwoLevelHashTableDynamic, ConcurrentBuildWithExternalBucketLocks)
{
    /// Runtime-sized storage does not synchronize internally - callers do, exactly as for the
    /// underlying hash tables. N worker threads insert directly into the one shared table under
    /// external per-bucket locks; growth of one bucket must not disturb another.
    constexpr size_t num_buckets = 256;
    constexpr size_t num_threads = 16;
    constexpr UInt64 keys_per_thread = 20000;

    DynamicMap map(num_buckets);
    std::vector<std::mutex> bucket_mutexes(num_buckets);

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
    for (UInt64 key = 1; key <= num_threads * keys_per_thread; ++key)
    {
        auto * it = map.find(key);
        ASSERT_NE(it, nullptr) << "key " << key << " lost by the concurrent build";
        ASSERT_EQ(it->getMapped(), key * 5) << "mapped value of key " << key << " was corrupted";
    }

    /// Descriptors are refreshed by `emplace` itself, so after the (externally-locked) build every
    /// one must describe its bucket's final buffer.
    const auto * descs = map.bucketDescs();
    for (UInt32 i = 0; i < map.bucketCount(); ++i)
    {
        ASSERT_NE(descs[i].buf, nullptr);
        ASSERT_EQ(descs[i].mask, map.impls[i].getBufferSizeInCells() - 1);
    }

    /// Exercise the descriptors the way a probe fast path does (`desc.buf + (hash & desc.mask)`):
    /// every cell `find()` returns must lie inside its own bucket's described buffer, or a probe
    /// reading through the descriptor would address another bucket's memory.
    for (UInt64 key = 1; key <= num_threads * keys_per_thread; ++key)
    {
        const auto * cell = map.find(key);
        ASSERT_NE(cell, nullptr);
        const auto buck = map.getBucketFromHash(map.hash(key));
        const auto * base = static_cast<const DynamicMap::cell_type *>(descs[buck].buf);
        ASSERT_GE(cell, base) << "key " << key << " resolved below bucket " << buck << "'s buffer";
        ASSERT_LT(cell, base + descs[buck].mask + 1) << "key " << key << " resolved past bucket " << buck << "'s buffer";
    }
}

TEST(TwoLevelHashTableDynamic, BucketSelectionMatchesJoinHashRouteSlot)
{
    /// A build/probe contract: `ConcurrentHashJoin` computes a row's target partition with
    /// `joinHashRouteSlot`, and the table must pick the same partition for the same hash, or probe
    /// rows visit buckets their keys were never inserted into. The two formulas are written
    /// differently - `(UInt32)h >> (32 - b)` against `(h >> (32 - b)) & (2^b - 1)` - so pin them
    /// against each other over both bucket counts and hash shapes, including the high bits that
    /// only one of the two expressions sees.
    for (size_t num_buckets = 1; num_buckets <= 256; num_buckets *= 2)
    {
        DynamicMap map(num_buckets);
        const auto route_shift = static_cast<UInt32>(32 - std::countr_zero(num_buckets));
        ASSERT_EQ(map.bucketShift(), route_shift);
        ASSERT_EQ(map.bucketCount(), num_buckets);

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
                << "num_buckets " << num_buckets << ", hash " << hash_value;
        }
    }
}

TEST(TwoLevelHashTableDynamic, ReserveSizesEveryBucket)
{
    constexpr size_t num_buckets = 128;
    DynamicMap map(num_buckets);
    map.reserve(num_buckets * 2048);

    const auto * descs = map.bucketDescs();
    for (UInt32 i = 0; i < map.bucketCount(); ++i)
    {
        ASSERT_GE(map.impls[i].getBufferSizeInCells(), 2048u);
        /// Descriptors must follow the reallocation, not keep pointing at the pre-reserve buffers.
        ASSERT_EQ(descs[i].mask, map.impls[i].getBufferSizeInCells() - 1);
    }

    /// And the table must still work afterwards.
    for (UInt64 key = 1; key <= 10000; ++key)
        insertKeyValue(map, key, key);
    ASSERT_EQ(map.size(), 10000u);
    for (UInt64 key = 1; key <= 10000; ++key)
        ASSERT_NE(map.find(key), nullptr);
}

TEST(TwoLevelHashTableDynamic, ForEachMappedVisitsEveryBucket)
{
    DynamicMap map(64);
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

TEST(TwoLevelHashTableDynamic, OffsetsStayValidAfterRecomputingPrefixPostGrowth)
{
    /// `offsetInternal()`'s prefix-sum cache is computed once, lazily, on first use, and does NOT
    /// notice later bucket growth on its own - there is no internal tracking of buffer changes,
    /// by design (synchronization and cache freshness are the caller's responsibility, like the
    /// underlying hash tables). `StorageJoin` can insert again after offsets have already been
    /// handed out, so a caller that needs correct offsets afterward must call
    /// `computeBucketPrefix()` again itself before trusting further offsets.
    DynamicMap map(16);
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

TEST(TwoLevelHashTableDynamic, ConcurrentBuildWithContendedKeys)
{
    /// Same, but every thread inserts the SAME key range through external bucket locks, so
    /// threads collide on cells inside a bucket instead of only on the bucket itself.
    constexpr size_t num_threads = 16;
    constexpr UInt64 num_keys = 5000;

    DynamicMap map(64);
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
