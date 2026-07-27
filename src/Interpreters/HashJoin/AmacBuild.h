#pragma once

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/HashJoin/KeyGetter.h>

namespace DB
{

/** The AMAC build-insert entry point (see `AmacRing.h` for the ring machinery; ported as ideas
  * from `src/Interpreters/PartitionedHashJoin/PartitionedHashJoinBuild.cpp` on branch `ahj`,
  * with that design's scattered-locator words replaced by the plain `RowRef(block_no, row)`
  * refs of `insertFromBlockImplTypeCase`). It inserts one section of build rows into a
  * cursor-capable slot map through the ring, producing cells bit-identical to the sequential
  * `Inserter` loop (both funnel through `applyBuildRowToMapped`).
  */

/// The per-run aggregates the caller consumes after a ring run; each mirrors what the
/// sequential loop accumulates per row.
struct AmacBuildInsertResult
{
    /// Hash-table growths triggered from inside the ring (drain + resize + re-seed).
    UInt64 growths = 0;
    /// `RowRef` maps: OR of the per-row "row is referenced by the map" signal - the exact value
    /// the sequential loop feeds into `is_inserted` via `Inserter::insertOne`.
    bool any_inserted = false;
    /// `RowRefList` maps: AND of the per-row emplace outcome, i.e. false as soon as any key saw
    /// a duplicate - the exact value the sequential loop feeds into `all_values_unique`.
    bool all_unique = true;
};

/** Insert `rows` build rows into `map` through the AMAC ring. The source row index of section
  * position `i` is `range_first + i` when `selector_is_range`, `selector_indexes[i]` otherwise
  * (the two shapes of `ScatteredBlock::Selector`); the ring carries the SOURCE index, so the
  * steady loop never touches the selector. `skip_bytes` (nullable) marks section positions the
  * sequential loop would skip - null keys and ON-mask-filtered rows - merged by the caller;
  * skipped rows never enter the ring.
  */
template <typename KeyGetter, typename Map, bool selector_is_range>
AmacBuildInsertResult amacBuildInsert(
    Map & map,
    KeyGetter & key_getter,
    size_t rows,
    size_t range_first,
    const UInt64 * selector_indexes,
    const UInt8 * skip_bytes,
    UInt32 stored_block_no,
    bool any_take_last_row,
    Arena & pool);

/// The 8 cursor-capable join map families: every chained member of `HashJoin::MapsTemplate`.
/// `key8`/`key16` and the range maps (`FixedHashMap`, no cursor API), `hashed` and the
/// LowCardinality families (excluded getters) keep the sequential loop - see
/// `amac_join_supported`. ASOF maps are excluded at the call site (their mapped insert appends
/// to a per-key sorted lookup, not a one-cell fused action).
#define APPLY_FOR_AMAC_BUILD_JOIN_VARIANTS(M) \
    M(key32) \
    M(key64) \
    M(key_string) \
    M(key_fixed_string) \
    M(keys32) \
    M(keys64) \
    M(keys128) \
    M(keys256)

/// The map behind a `HashJoin::MapsTemplate` member and its build-side key getter, spelled once
/// for the explicit instantiations (mirrors how `insertFromBlockImpl` derives the getter).
#define M(TYPE) \
    template <typename Maps> \
    using AmacBuildMapFor_##TYPE = std::remove_reference_t<decltype(*std::declval<Maps &>().TYPE)>; \
    template <typename Maps> \
    using AmacBuildKeyGetterFor_##TYPE = typename KeyGetterForType<HashJoin::Type::TYPE, AmacBuildMapFor_##TYPE<Maps>>::Type;
APPLY_FOR_AMAC_BUILD_JOIN_VARIANTS(M)
#undef M

/// One `amacBuildInsert` instantiation; `EXTERN` is `extern` in this header and empty in
/// `AmacBuild.cpp`, so the declarations and the definitions cannot drift apart.
#define AMAC_BUILD_INSERT_INSTANTIATION(EXTERN, TYPE, MAPS, SELECTOR_IS_RANGE) \
    EXTERN template AmacBuildInsertResult amacBuildInsert< \
        AmacBuildKeyGetterFor_##TYPE<HashJoin::MAPS>, \
        AmacBuildMapFor_##TYPE<HashJoin::MAPS>, \
        SELECTOR_IS_RANGE>( \
        AmacBuildMapFor_##TYPE<HashJoin::MAPS> & map, \
        AmacBuildKeyGetterFor_##TYPE<HashJoin::MAPS> & key_getter, \
        size_t rows, \
        size_t range_first, \
        const UInt64 * selector_indexes, \
        const UInt8 * skip_bytes, \
        UInt32 stored_block_no, \
        bool any_take_last_row, \
        Arena & pool);

/// All 32 instantiations: 8 families x {`RowRef`, `RowRefList`} mapped x {range, indexes}
/// selector. `keys32`/`keys64` (and `key_fixed_string`) share a map type with their single-key
/// siblings but use a different key getter, so each is a distinct instantiation.
#define AMAC_BUILD_INSERT_INSTANTIATIONS(EXTERN, TYPE) \
    AMAC_BUILD_INSERT_INSTANTIATION(EXTERN, TYPE, MapsOne, true) \
    AMAC_BUILD_INSERT_INSTANTIATION(EXTERN, TYPE, MapsOne, false) \
    AMAC_BUILD_INSERT_INSTANTIATION(EXTERN, TYPE, MapsAll, true) \
    AMAC_BUILD_INSERT_INSTANTIATION(EXTERN, TYPE, MapsAll, false)

#define M(TYPE) AMAC_BUILD_INSERT_INSTANTIATIONS(extern, TYPE)
APPLY_FOR_AMAC_BUILD_JOIN_VARIANTS(M)
#undef M

}
