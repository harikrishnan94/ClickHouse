#pragma once

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/PartitionedHashJoin/AmacRing.h>
#include <Interpreters/PartitionedHashJoin/FixedRegionAllocator.h>
#include <Interpreters/RowRefs.h>
#include <Common/HashTable/FixedHashMap.h>
#include <Common/HashTable/Hash.h>
#include <Common/HashTable/HashMap.h>

#include <variant>

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

namespace PartitionedJoinMapsDetail
{

/** Rebinds a standard join hash map type to the `FixedRegionAllocator` while keeping every other
  * template argument - cell layout, hash, grower - exactly as `HashJoin::MapsTemplate` declares
  * it. Deriving the leaf map types from the standard ones (instead of mirroring their
  * declarations) means a master-side change of a cell type, hash function or grower propagates
  * here automatically, and an incompatible restructuring breaks the build instead of silently
  * diverging. The open-addressing maps additionally carry the resumable-cursor API of
  * `ResumableHashMap` (the seed/step decomposition the AMAC build and probe rings drive);
  * `FixedHashMap` has no collision chain to pipeline and keeps the plain interface.
  */
template <typename Map>
struct WithRegionAllocator;

template <typename Key, typename Cell, typename Hash, typename Grower, typename Alloc>
struct WithRegionAllocator<HashMapTable<Key, Cell, Hash, Grower, Alloc>>
{
    using Type = ResumableHashMap<HashMapTable<Key, Cell, Hash, Grower, FixedRegionAllocator>>;
};

template <typename Key, typename Mapped, typename Cell, typename Size, typename Alloc, size_t size_bits>
struct WithRegionAllocator<FixedHashMap<Key, Mapped, Cell, Size, Alloc, size_bits>>
{
    using Type = FixedHashMap<Key, Mapped, Cell, Size, FixedRegionAllocator, size_bits>;
};

}

/** The leaf hash tables of one partition, for one mapped-value shape (`RowRef`, `RowRefList` or
  * `AsofRowRefs` - the shapes of `HashJoin::MapsOne`/`MapsAll`/`MapsAsof`). Every member type is
  * the corresponding `HashJoin::MapsTemplate` member with only the allocator rebound, so the
  * standard `KeyGetterForType` machinery works on the leaf maps unchanged and the emitted cells
  * are bit-identical to what the standard insert path would produce.
  */
template <typename Mapped>
struct PartitionedJoinMapsTemplate
{
    using MappedType = Mapped;
    using StandardMaps = HashJoin::MapsTemplate<Mapped>;

    /// NOLINTBEGIN(bugprone-macro-parentheses)
#define M(NAME) \
    std::shared_ptr<typename PartitionedJoinMapsDetail::WithRegionAllocator<typename decltype(StandardMaps::NAME)::element_type>::Type> \
        NAME;
    APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M)
#undef M

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
            /// A FixedHashTable buffer always spans the full key domain, independent of the reserve.
            static_assert(sizeof(typename Map::key_type) <= 2);
            return (1uz << (sizeof(typename Map::key_type) * 8)) * sizeof(typename Map::cell_type);
        }
    }

    static size_t predictedBufferBytes(HashJoin::Type which, size_t reserve)
    {
        switch (which)
        {
#define M(NAME) \
    case HashJoin::Type::NAME: return predictedBufferBytesFor<typename decltype(PartitionedJoinMapsTemplate::NAME)::element_type>(reserve);
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

    size_t getBufferSizeInCells(HashJoin::Type which) const
    {
        switch (which)
        {
#define M(NAME) \
    case HashJoin::Type::NAME: return NAME ? NAME->getBufferSizeInCells() : 0;
            APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M)
#undef M
            default: return 0;
        }
    }
    /// NOLINTEND(bugprone-macro-parentheses)
};

using PartitionedMapsOne = PartitionedJoinMapsTemplate<RowRef>;
using PartitionedMapsAll = PartitionedJoinMapsTemplate<RowRefList>;
using PartitionedMapsAsof = PartitionedJoinMapsTemplate<AsofRowRefs>;

/// The leaf-map cells must stay bit-identical to the standard maps' cells: the probe's
/// `processMatch`, the used-flags offsets and the lazy emit all consume them through the standard
/// machinery. The rebind above guarantees this by construction; these asserts keep it explicit
/// and break the build if the member declarations ever stop being derived from the standard ones.
#define M(NAME) \
    static_assert( \
        std::is_same_v< \
            typename decltype(PartitionedMapsOne::NAME)::element_type::cell_type, \
            typename decltype(HashJoin::MapsOne::NAME)::element_type::cell_type> \
            && std::is_same_v< \
                typename decltype(PartitionedMapsAll::NAME)::element_type::cell_type, \
                typename decltype(HashJoin::MapsAll::NAME)::element_type::cell_type> \
            && std::is_same_v< \
                typename decltype(PartitionedMapsAsof::NAME)::element_type::cell_type, \
                typename decltype(HashJoin::MapsAsof::NAME)::element_type::cell_type>, \
        "partitioned leaf map cells must be identical to the standard join map cells");
APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M)
#undef M

/// Maps a standard join maps shape (the type `MapGetter`/`JoinFeatures` speak) to the
/// partitioned counterpart holding the same cells behind the region allocator.
template <typename StandardMaps>
struct PartitionedMapsFor;

template <>
struct PartitionedMapsFor<HashJoin::MapsOne>
{
    using Type = PartitionedMapsOne;
};
template <>
struct PartitionedMapsFor<HashJoin::MapsAll>
{
    using Type = PartitionedMapsAll;
};
template <>
struct PartitionedMapsFor<HashJoin::MapsAsof>
{
    using Type = PartitionedMapsAsof;
};

/** The leaf hash tables of one partition: a variant over the three mapped-value shapes, with the
  * active alternative mirroring the leaf `HashJoin`'s own `MapsVariant` (chosen at construction
  * by `joinDispatchInit` from the (kind, strictness) `MapGetter` table), so build inserts and the
  * probe agree with the standard machinery on which value shape a join shape uses.
  */
struct PartitionedJoinMaps
{
    using Variant = std::variant<PartitionedMapsOne, PartitionedMapsAll, PartitionedMapsAsof>;

    /// The alternatives must stay index-compatible with `HashJoin::MapsVariant`: the active
    /// alternative is selected by the index of the leaf join's own maps variant.
    static_assert(
        std::is_same_v<std::variant_alternative_t<0, HashJoin::MapsVariant>, HashJoin::MapsOne>
        && std::is_same_v<std::variant_alternative_t<1, HashJoin::MapsVariant>, HashJoin::MapsAll>
        && std::is_same_v<std::variant_alternative_t<2, HashJoin::MapsVariant>, HashJoin::MapsAsof>
        && std::variant_size_v<HashJoin::MapsVariant> == 3);

    Variant maps;

    explicit PartitionedJoinMaps(size_t standard_variant_index = 1)
    {
        switch (standard_variant_index)
        {
            case 0: maps.emplace<PartitionedMapsOne>(); break;
            case 1: maps.emplace<PartitionedMapsAll>(); break;
            case 2: maps.emplace<PartitionedMapsAsof>(); break;
            default: throw Exception(ErrorCodes::UNSUPPORTED_JOIN_KEYS, "Unexpected join maps variant index {}", standard_variant_index);
        }
    }

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

    static size_t predictedBufferBytes(size_t standard_variant_index, HashJoin::Type which, size_t reserve)
    {
        switch (standard_variant_index)
        {
            case 0: return PartitionedMapsOne::predictedBufferBytes(which, reserve);
            case 1: return PartitionedMapsAll::predictedBufferBytes(which, reserve);
            case 2: return PartitionedMapsAsof::predictedBufferBytes(which, reserve);
            default: throw Exception(ErrorCodes::UNSUPPORTED_JOIN_KEYS, "Unexpected join maps variant index {}", standard_variant_index);
        }
    }

    void create(HashJoin::Type which, size_t reserve)
    {
        std::visit([&](auto & shape) { shape.create(which, reserve); }, maps);
    }

    size_t getTotalRowCount(HashJoin::Type which) const
    {
        return std::visit([&](const auto & shape) { return shape.getTotalRowCount(which); }, maps);
    }

    size_t getBufferSizeInBytes(HashJoin::Type which) const
    {
        return std::visit([&](const auto & shape) { return shape.getBufferSizeInBytes(which); }, maps);
    }

    size_t getBufferSizeInCells(HashJoin::Type which) const
    {
        return std::visit([&](const auto & shape) { return shape.getBufferSizeInCells(which); }, maps);
    }
};

}
