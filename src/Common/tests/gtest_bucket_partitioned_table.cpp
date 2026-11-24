#include <gtest/gtest.h>

#include <Common/CacheLine.h>
#include <Common/HashTable/PartitionedFixedHashTable.h>

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <vector>


namespace
{

template <size_t bits>
using FixedMap = PartitionedFixedHashMap<UInt16, UInt64, 16, bits>;

/// The buckets add no state to the flat table.
/// Neither the partitioned map nor its flat table keeps an element counter, so both are smaller than the plain map.
static_assert(sizeof(FixedMap<8>) == sizeof(FixedHashMapWithSizeBitsAndCalculatedSize<UInt16, UInt64, 16>));
static_assert(sizeof(FixedMap<8>) < sizeof(FixedHashMapWithSizeBits<UInt16, UInt64, 16>));

/// A cell with a `bool` and an 8-byte, 4-aligned payload like `RowRef` is 12 bytes.
/// Only the cell of a map with more than one bucket is padded.
struct TwoWords
{
    UInt32 a = 0;
    UInt32 b = 0;
};
static_assert(sizeof(FixedHashMapCell<UInt8, TwoWords>) == 12);
static_assert(sizeof(PartitionedFixedHashMap<UInt8, TwoWords, 8, 0>::cell_type) == 12);
static_assert(sizeof(PartitionedFixedHashMap<UInt8, TwoWords, 8, 8>::cell_type) == 16);

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

template <typename Map>
typename Map::key_type toKey(UInt64 key)
{
    return static_cast<typename Map::key_type>(key);
}

template <typename Map>
std::vector<size_t> offsetsByIteration(const Map & map)
{
    std::vector<size_t> offsets;
    for (typename Map::const_iterator it = map.begin(); it != map.end(); ++it)
        offsets.push_back(map.offsetInternal(it.getPtr()));
    return offsets;
}

template <size_t... bits, typename Fn>
void forBucketBits(Fn && fn)
{
    (fn.template operator()<bits>(), ...);
}


}

TEST(PartitionedFixedHashMap, MatchesThePlainMap)
{
    constexpr size_t size_bits = 16;
    constexpr UInt32 num_keys = 5000;
    constexpr UInt32 stride = 7;

    auto plain = std::make_unique<FixedHashMapWithSizeBits<UInt32, UInt64, size_bits>>();
    for (UInt32 i = 0; i < num_keys; ++i)
        insertKeyValue(*plain, i * stride, i);

    forBucketBits<0, 8>(
        [&]<size_t bits>()
        {
            using Map = PartitionedFixedHashMap<UInt32, UInt64, size_bits, bits>;
            auto map = std::make_unique<Map>();
            for (UInt32 i = 0; i < num_keys; ++i)
                insertKeyValue(*map, i * stride, i);

            ASSERT_EQ(map->getBufferSizeInCells(), plain->getBufferSizeInCells()) << "bits " << bits;
            for (UInt32 i = 0; i < num_keys; ++i)
            {
                const UInt32 key = i * stride;
                ASSERT_EQ(map->offsetInternal(map->find(key)), plain->offsetInternal(plain->find(key)))
                    << "key " << key << ", bits " << bits;
                ASSERT_TRUE(map->has(key)) << "key " << key << ", bits " << bits;
                ASSERT_FALSE(map->has(key + 1)) << "key " << key + 1 << ", bits " << bits;
            }
            ASSERT_EQ(offsetsByIteration(*map), offsetsByIteration(*plain)) << "bits " << bits;
        });
}


/// Two keys whose cells start on the same real cache line must share a bucket.
/// Otherwise writers under two different locks would share the line.
template <typename Map>
void assertKeysOnOneCacheLineShareABucket()
{
    constexpr size_t num_keys = size_t{1} << (8 * sizeof(typename Map::key_type));
    constexpr size_t cell_size = sizeof(typename Map::cell_type);
    constexpr uintptr_t line = DB::CH_CACHE_LINE_SIZE;

    auto map = std::make_unique<Map>();
    std::vector<uintptr_t> addresses(num_keys);
    for (size_t key = 0; key < num_keys; ++key)
    {
        const auto typed_key = toKey<Map>(key);
        typename Map::LookupResult it = nullptr;
        bool inserted = false;
        map->emplace(typed_key, it, inserted);
        addresses[key] = reinterpret_cast<uintptr_t>(map->find(typed_key));
    }

    ASSERT_EQ(addresses[0] % line, 0u) << "routing assumes the buffer starts on a cache line";
    size_t pairs_on_one_line = 0;
    for (size_t key = 0; key < num_keys; ++key)
    {
        ASSERT_LE(addresses[key] % line + cell_size, line) << "the cell of key " << key << " crosses a cache line";
        if (key == 0 || addresses[key - 1] / line != addresses[key] / line)
            continue;

        ++pairs_on_one_line;
        const auto prev = toKey<Map>(key - 1);
        const auto curr = toKey<Map>(key);
        ASSERT_EQ(Map::bucketOfKey(prev), Map::bucketOfKey(curr))
            << "keys " << key - 1 << " and " << key << " share a cache line but go to different buckets";
    }
    ASSERT_GT(pairs_on_one_line, 0u);
}

TEST(PartitionedFixedHashMap, KeysOnOneCacheLineShareABucket)
{
    /// A 12-byte cell padded to 16, a 16-byte cell, and a 1-byte set cell.
    assertKeysOnOneCacheLineShareABucket<PartitionedFixedHashMap<UInt8, TwoWords, 8, 8>>();
    assertKeysOnOneCacheLineShareABucket<PartitionedFixedHashMap<UInt16, UInt64, 16, 8>>();
    assertKeysOnOneCacheLineShareABucket<PartitionedFixedHashSet<UInt16, 16, 8>>();
}


/// At least `min_buckets` buckets receive keys, and no bucket takes more than a quarter of the keys.
template <typename Map>
void assertKeysSpread(const std::vector<typename Map::key_type> & keys, size_t min_buckets)
{
    std::vector<size_t> per_bucket(Map::NUM_BUCKETS, 0);
    for (const auto key : keys)
        ++per_bucket[Map::bucketOfKey(key)];

    const size_t non_empty = std::count_if(per_bucket.begin(), per_bucket.end(), [](size_t count) { return count != 0; });
    const size_t largest = *std::max_element(per_bucket.begin(), per_bucket.end());
    ASSERT_GE(non_empty, min_buckets) << "keys reached only " << non_empty << " of " << Map::NUM_BUCKETS << " buckets";
    ASSERT_LE(largest, keys.size() / 4) << "one bucket took " << largest << " of " << keys.size() << " keys";
}

TEST(PartitionedFixedHashMap, SpreadsKeysThatShareHighOrLowBits)
{
    using Map = PartitionedFixedHashMap<UInt32, UInt64, 16, 4>;

    /// The 300 keys at the bottom of a 65536-cell table share their high bits.
    /// Routing on those bits would put all of them into one bucket.
    std::vector<UInt32> dense_at_zero(300);
    for (UInt32 key = 0; key < dense_at_zero.size(); ++key)
        dense_at_zero[key] = key;
    assertKeysSpread<Map>(dense_at_zero, 14);

    /// Multiples of 256 share their low bits.
    std::vector<UInt32> strided(256);
    for (UInt32 i = 0; i < strided.size(); ++i)
        strided[i] = i * 256;
    assertKeysSpread<Map>(strided, 14);
}


TEST(PartitionedFixedHashMap, IterationAfterRestoringTheBounds)
{
    forBucketBits<0, 8>(
        [&]<size_t bits>()
        {
            auto map = std::make_unique<FixedMap<bits>>();
            for (const UInt16 key : std::initializer_list<UInt16>{10, 20, 40})
                insertKeyValue(*map, key, key);
            ASSERT_EQ(offsetsByIteration(*map).size(), 3u) << "bits " << bits;

            map->restoreMinMaxOptimization();
            ASSERT_EQ(offsetsByIteration(*map).size(), 3u) << "bits " << bits;

            /// Keys below the old minimum and above the old maximum.
            insertKeyValue(*map, 5, 5);
            insertKeyValue(*map, 50, 50);
            std::vector<UInt16> keys;
            for (auto it = map->begin(); it != map->end(); ++it)
                keys.push_back(it->getKey());
            ASSERT_EQ(keys, (std::vector<UInt16>{5, 10, 20, 40, 50})) << "bits " << bits;
        });
}


TEST(PartitionedFixedHashSet, RecordsPresence)
{
    /// A set cell is one byte, so one cache line holds many keys.
    /// Taking one key per line spreads the keys over the buckets.
    constexpr size_t keys_per_line = DB::CH_CACHE_LINE_SIZE / sizeof(FixedHashTableCell<UInt16>);
    constexpr UInt32 num_lines = 1000;

    forBucketBits<0, 8>(
        [&]<size_t bits>()
        {
            using Set = PartitionedFixedHashSet<UInt16, 16, bits>;
            auto set = std::make_unique<Set>();

            std::unordered_set<size_t> buckets;
            for (UInt32 line = 0; line < num_lines; ++line)
            {
                const auto key = static_cast<UInt16>(line * keys_per_line);
                typename Set::LookupResult it = nullptr;
                bool inserted = false;
                set->emplace(key, it, inserted);
                ASSERT_TRUE(inserted) << "key " << key;
                buckets.insert(Set::bucketOfKey(key));
            }
            ASSERT_EQ(set->size(), num_lines) << "bits " << bits;
            if constexpr (bits == 0)
                ASSERT_EQ(buckets.size(), 1u);
            else
                ASSERT_GT(buckets.size(), 200u) << "keys on distinct cache lines reached only " << buckets.size() << " buckets";

            for (UInt32 line = 0; line < num_lines; ++line)
                ASSERT_TRUE(set->has(static_cast<UInt16>(line * keys_per_line))) << "line " << line;
            ASSERT_FALSE(set->has(static_cast<UInt16>(1)));

            size_t iterated = 0;
            for (auto it = set->begin(); it != set->end(); ++it)
                ++iterated;
            ASSERT_EQ(iterated, num_lines) << "bits " << bits;

            /// A set has no mapped values.
            size_t visited = 0;
            set->forEachMapped([&](auto &) { ++visited; });
            ASSERT_EQ(visited, 0u);
        });
}
