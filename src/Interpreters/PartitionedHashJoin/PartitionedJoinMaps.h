#pragma once

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/PartitionedHashJoin/FixedRegionAllocator.h>
#include <Interpreters/RowRefs.h>
#include <Common/HashTable/FixedHashMap.h>
#include <Common/HashTable/Hash.h>
#include <Common/HashTable/HashMap.h>

namespace DB
{

namespace ErrorCodes
{
extern const int UNSUPPORTED_JOIN_KEYS;
}

/// The map types a `PartitionedHashJoin` build can produce: the single-level subset of
/// `HashJoin::Type` (`chooseMethod` without two-level maps; the `range*` fixed conversions are
/// post-build optimizations gated off for the partitioned join).
#define APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M) \
    M(key8) \
    M(key16) \
    M(key32) \
    M(key64) \
    M(key_string) \
    M(key_fixed_string) \
    M(keys32) \
    M(keys64) \
    M(keys128) \
    M(keys256) \
    M(hashed) \
    M(low_cardinality_key_string) \
    M(low_cardinality_key_fixed_string)

/** The leaf hash tables of one partition. Cell layouts, hash functions and growers are exactly
  * those of the corresponding `HashJoin::MapsTemplate` members, so the standard `KeyGetterForType`
  * machinery works on them unchanged; only the allocator differs - `FixedRegionAllocator` carves
  * the buffer out of the per-build contiguous slab.
  *
  * Only the `MapsAll` shape (`RowRefList` mapped values) exists: the partitioned build path is
  * gated to ALL strictness, and the post-build promotion to `RightAny` changes the probe logic,
  * not the map layout.
  */
struct PartitionedJoinMaps
{
    using Mapped = RowRefList;
    using Grower = HashTableGrowerWithPrecalculation<>;
    using StringMap = HashMapWithSavedHash<std::string_view, Mapped, DefaultHash<std::string_view>, Grower, FixedRegionAllocator>;

    template <typename Key>
    using FixedMap = FixedHashMap<
        Key,
        Mapped,
        FixedHashMapCell<Key, Mapped>,
        FixedHashTableStoredSize<FixedHashMapCell<Key, Mapped>>,
        FixedRegionAllocator>;

    /// NOLINTBEGIN(bugprone-macro-parentheses)
    std::shared_ptr<FixedMap<UInt8>> key8;
    std::shared_ptr<FixedMap<UInt16>> key16;
    std::shared_ptr<HashMap<UInt32, Mapped, HashCRC32<UInt32>, Grower, FixedRegionAllocator>> key32;
    std::shared_ptr<HashMap<UInt64, Mapped, HashCRC32<UInt64>, Grower, FixedRegionAllocator>> key64;
    std::shared_ptr<StringMap> key_string;
    std::shared_ptr<StringMap> key_fixed_string;
    std::shared_ptr<HashMap<UInt32, Mapped, HashCRC32<UInt32>, Grower, FixedRegionAllocator>> keys32;
    std::shared_ptr<HashMap<UInt64, Mapped, HashCRC32<UInt64>, Grower, FixedRegionAllocator>> keys64;
    std::shared_ptr<HashMap<UInt128, Mapped, UInt128HashCRC32, Grower, FixedRegionAllocator>> keys128;
    std::shared_ptr<HashMap<UInt256, Mapped, UInt256HashCRC32, Grower, FixedRegionAllocator>> keys256;
    std::shared_ptr<HashMap<UInt128, Mapped, UInt128TrivialHash, Grower, FixedRegionAllocator>> hashed;
    std::shared_ptr<StringMap> low_cardinality_key_string;
    std::shared_ptr<StringMap> low_cardinality_key_fixed_string;

    static bool isSupportedType(HashJoin::Type which)
    {
        switch (which)
        {
#define M(NAME) \
    case HashJoin::Type::NAME: return true;
            APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M)
#undef M
            default: return false;
        }
    }

    /// Whether the map type has a fixed buffer size independent of the build (`FixedHashMap`):
    /// such builds always degenerate to a single leaf - partitioning cannot shrink the tables.
    static bool isFixedSizeType(HashJoin::Type which) { return which == HashJoin::Type::key8 || which == HashJoin::Type::key16; }

    /// The exact buffer bytes a map created with `create(which, reserve)` will allocate: the
    /// map's own grower rounding at `reserve`, or the fixed buffer size of a `FixedHashMap`.
    /// The partition plan sizes the slab regions with this, so predicted == actual.
    template <typename Map>
    static size_t predictedBufferBytesFor(size_t reserve)
    {
        if constexpr (requires { typename Map::grower_type; })
        {
            typename Map::grower_type grower;
            grower.set(reserve);
            return grower.bufSize() * sizeof(typename Map::cell_type);
        }
        else
        {
            static_assert(std::is_same_v<Map, FixedMap<UInt8>> || std::is_same_v<Map, FixedMap<UInt16>>);
            return (1uz << (sizeof(typename Map::key_type) * 8)) * sizeof(typename Map::cell_type);
        }
    }

    static size_t predictedBufferBytes(HashJoin::Type which, size_t reserve)
    {
        switch (which)
        {
#define M(NAME) \
    case HashJoin::Type::NAME: return predictedBufferBytesFor<typename decltype(PartitionedJoinMaps::NAME)::element_type>(reserve);
            APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M)
#undef M
            default: throw Exception(ErrorCodes::UNSUPPORTED_JOIN_KEYS, "Unsupported JOIN keys for the partitioned join (type: {})", which);
        }
    }

    void create(HashJoin::Type which, size_t reserve)
    {
        switch (which)
        {
#define M(NAME) \
    case HashJoin::Type::NAME: \
        NAME = reserve ? std::make_shared<typename decltype(NAME)::element_type>(reserve) \
                       : std::make_shared<typename decltype(NAME)::element_type>(); \
        break;
            APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M)
#undef M
            default: throw Exception(ErrorCodes::UNSUPPORTED_JOIN_KEYS, "Unsupported JOIN keys for the partitioned join (type: {})", which);
        }
    }

    size_t getTotalRowCount(HashJoin::Type which) const
    {
        switch (which)
        {
#define M(NAME) \
    case HashJoin::Type::NAME: return NAME ? NAME->size() : 0;
            APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M)
#undef M
            default: return 0;
        }
    }

    size_t getBufferSizeInBytes(HashJoin::Type which) const
    {
        switch (which)
        {
#define M(NAME) \
    case HashJoin::Type::NAME: return NAME ? NAME->getBufferSizeInBytes() : 0;
            APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M)
#undef M
            default: return 0;
        }
    }
    /// NOLINTEND(bugprone-macro-parentheses)
};

}
