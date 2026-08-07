#pragma once

#include <Interpreters/HashJoin/ScatteredBlock.h>
#include <Interpreters/UnifiedHashJoin/HashJoin.h>
#include <Interpreters/UnifiedHashJoin/KeyGetter.h>
#include <Columns/IColumn.h>

namespace DB
{
namespace Unified
{

/// Which `MapsTemplate` instantiation backs a clause's map. Fixed-range maps route keys in
/// cache-line-sized blocks, so their routing depends on the cell size, i.e. on the mapped type;
/// the hash-table maps route identically for every mapped type.
enum class MapsKind : uint8_t
{
    One,
    All,
    Asof,
};

struct SlotScatter
{
    std::vector<ScatteredBlock::Selector> selectors;
    std::vector<Columns> dense_keys;
};

/// Everything build-side slot scatter needs from the hash table, keyed by key type alone.
/// `RepMap` is a representative map type for the key type: any `Mapped` works for hash-table
/// maps, while fixed-range maps must be the clause's actual `MapsOne`/`MapsAll`/`MapsAsof`
/// table. Routing forwards to the map's own statics, so scatter cannot diverge from the map's
/// placement by construction.
template <HashJoin::Type TYPE, typename RepMap>
struct SlotScatterTraits
{
    using Map = RepMap;
    using KeyGetter = typename KeyGetterForType<TYPE, RepMap, false>::Type;

    template <typename K>
    static size_t hash(const K & key) { return RepMap::hash(key); }

    template <typename K>
    static size_t bucketRoutingHash(const K & key, size_t hash_value) { return RepMap::bucketRoutingHash(key, hash_value); }

    static size_t getBucketFromHash(size_t hash_value) { return RepMap::getBucketFromHash(hash_value); }
};

/// Scatters right-table rows of one clause into per-slot selectors (and dense key columns for
/// narrow fixed-size keys) so that `insertIntoSlots` can insert each slot under its own lock.
/// Instantiated once per key type in SlotScatter.cpp - deliberately NOT a member of
/// `HashJoinMethods`, where it was emitted per (kind, strictness, mapped type).
SlotScatter scatterBlockBySlot(
    HashJoin::Type type,
    MapsKind maps_kind,
    const ColumnRawPtrs & key_columns,
    const Sizes & key_sizes,
    const ScatteredBlock::Selector & selector,
    size_t num_slots,
    bool is_asof);

}
}
