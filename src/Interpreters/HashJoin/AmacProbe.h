#pragma once

#include <Interpreters/HashJoin/AmacBuild.h>
#include <Interpreters/HashJoin/AmacRing.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/HashJoin/KeyGetter.h>
#include <Interpreters/HashJoin/ResumableHashMap.h>
#include <Interpreters/RowRefs.h>

namespace DB
{

/** The AMAC find pass of the routed `parallel_hash` probe (see `AmacRing.h` for the ring
  * machinery and `HashJoinRoutedMethods.h` for the host loop; ported as ideas from
  * `src/Interpreters/PartitionedHashJoin/PartitionedHashJoinProbeImpl.h` on branch `ahj`, with
  * that design's leaf descriptors renamed to slot descriptors). Phase A of a two-phase probe:
  * out-of-order lookups that only fill the per-row result arrays; phase B (the routed loop)
  * consumes them in left-row order.
  */

/// Mapped values the AMAC find pass records BY VALUE: both are 8-byte words (`RowRef` encodes to
/// its ref word, `RowRefList` IS a tagged word) that are never 0 for a built cell - a `RowRef`
/// is always constructed with `INLINE_FLAG` in bit 63, and a `RowRefList` word is an inline ref
/// (bit 63 set) or a non-null node pointer - so 0 can encode a miss. Everything the emit side
/// does with a match (`appendFromBlock`, `rows`, `refsOf`, `firstRefWord`) consumes only the
/// word, and the probe maps are immutable, so the copy is semantically the cell itself. Copying
/// the word in the same visit that reads the cell means the emit loop never touches the cell
/// again - by the time it reaches the row, the cell line has usually left the cache and
/// re-reading it through a recorded pointer would be a second random miss per row.
template <typename Mapped>
inline constexpr bool amac_mapped_fits_word = std::is_same_v<Mapped, RowRef> || std::is_same_v<Mapped, RowRefList>;

template <typename Mapped>
requires amac_mapped_fits_word<Mapped>
ALWAYS_INLINE UInt64 mappedWordOf(const Mapped & mapped)
{
    if constexpr (std::is_same_v<Mapped, RowRefList>)
        return mapped.word;
    else
        return mapped.encode();
}

template <typename Mapped>
requires amac_mapped_fits_word<Mapped>
ALWAYS_INLINE Mapped mappedFromWord(UInt64 word)
{
    if constexpr (std::is_same_v<Mapped, RowRefList>)
    {
        return RowRefList::fromWord(word);
    }
    else
    {
        /// The exact inverse of `RowRef::encode`: `block_no` (with `INLINE_FLAG` in its MSB) in
        /// the high half, `row_no` in the low half. Field-wise because the checked `RowRef`
        /// constructor takes an unflagged block number.
        RowRef ref;
        ref.block_no = static_cast<UInt32>(word >> 32);
        ref.row_no = static_cast<UInt32>(word);
        return ref;
    }
}

/// The compile-time gate of the AMAC probe find pass: the build ring's gate (cursor-capable map,
/// no excluded getter) plus the by-value mapped-word contract above. The ASOF maps carry a
/// `unique_ptr` mapped value and keep the plain routed loop.
template <typename KeyGetter, typename Map>
constexpr bool amac_probe_supported = amac_join_supported<KeyGetter, std::remove_const_t<Map>>
    /// This `typename` is required (a template argument is not a typename-optional context; the check misfires).
    && amac_mapped_fits_word<typename std::remove_const_t<Map>::mapped_type>; /// NOLINT(readability-redundant-typename)

/// Per-slot address material of the find ring, resolved once per probe call from the slot maps:
/// the cell buffer base and the home mask of the power-of-two region (NOT `bufSize - 1`; the
/// buffers are tail-padded - see `TailPaddedHashTableGrower`). At most 256 slots x 16 bytes,
/// so the whole array stays cache-resident under the ring.
struct SlotMapDesc
{
    const void * buf;
    size_t mask;
};

/// The grower contract of the descriptor-based find ring: linear probing with a power-of-two
/// home mask and the tail-padded buffer of `TailPaddedHashTableGrower` (the walk runs into the
/// pad and wraps only at its end; see the grower's comment). Every cursor-capable join map is
/// built on it; a map on any other grower must keep the map-resolved lookups.
template <typename Grower>
inline constexpr bool is_tail_padded_linear_grower = false;
template <size_t initial_size_degree>
inline constexpr bool is_tail_padded_linear_grower<TailPaddedHashTableGrower<initial_size_degree>> = true;

/** Run the AMAC find pass over `rows` probe rows: out-of-order lookups filling the per-row
  * result arrays - `found_word[i]` is the matched cell's mapped value copied BY VALUE (0 = no
  * match; skipped and zero-key rows are recorded synchronously, so every row gets a result) and,
  * for the flagged shapes only (`need_flags`), `found_offset[i]` is the matched cell's
  * slot-LOCAL used-flags offset (`(cell - buf) + 1`, matching `offsetInternal`; the flag
  * structures stay per-slot, and the emit side re-derives the slot from `slot_ids[ind]`).
  * The source row of pass position `i` is `range_first + i` when `selector_is_range`,
  * `selector_indexes[i]` otherwise. Internally chunked so the ring's row index fits 16 bits.
  * Increments `ConcurrentHashJoinAmacProbeRows` by `rows`, once per pass.
  */
template <typename KeyGetter, typename Map, bool need_flags, bool selector_is_range>
void amacFindPass(
    KeyGetter & key_getter,
    const Map * const * slot_maps,
    const SlotMapDesc * slot_descs,
    const UInt64 * slot_ids,
    size_t rows,
    size_t range_first,
    const UInt64 * selector_indexes,
    const UInt8 * skip_data,
    Arena & pool,
    UInt64 * found_word,
    UInt64 * found_offset);

/// The map behind a `HashJoin::MapsTemplate` member and its PROBE-side key getter (const mapped
/// values, unlike the build getters of `AmacBuild.h`), spelled once for the explicit
/// instantiations. The families are the build ring's 8 (`APPLY_FOR_AMAC_BUILD_JOIN_VARIANTS`).
#define M(TYPE) \
    template <typename Maps> \
    using AmacProbeMapFor_##TYPE = std::remove_reference_t<decltype(*std::declval<Maps &>().TYPE)>; \
    template <typename Maps> \
    using AmacProbeKeyGetterFor_##TYPE = typename KeyGetterForType<HashJoin::Type::TYPE, const AmacProbeMapFor_##TYPE<Maps>>::Type;
APPLY_FOR_AMAC_BUILD_JOIN_VARIANTS(M)
#undef M

/// One `amacFindPass` instantiation; `EXTERN` is `extern` in this header and empty in
/// `AmacProbe.cpp`, so the declarations and the definitions cannot drift apart.
#define AMAC_FIND_PASS_INSTANTIATION(EXTERN, TYPE, MAPS, NEED_FLAGS, SELECTOR_IS_RANGE) \
    EXTERN template void amacFindPass< \
        AmacProbeKeyGetterFor_##TYPE<HashJoin::MAPS>, \
        AmacProbeMapFor_##TYPE<HashJoin::MAPS>, \
        NEED_FLAGS, \
        SELECTOR_IS_RANGE>( \
        AmacProbeKeyGetterFor_##TYPE<HashJoin::MAPS> & key_getter, \
        const AmacProbeMapFor_##TYPE<HashJoin::MAPS> * const * slot_maps, \
        const SlotMapDesc * slot_descs, \
        const UInt64 * slot_ids, \
        size_t rows, \
        size_t range_first, \
        const UInt64 * selector_indexes, \
        const UInt8 * skip_data, \
        Arena & pool, \
        UInt64 * found_word, \
        UInt64 * found_offset);

/// All 64 instantiations: 8 families x {`RowRef`, `RowRefList`} mapped x flagged-or-not x
/// {range, indexes} selector. Both flag arms exist for every shape, so any (kind, strictness)
/// resolves to a preinstantiated symbol.
#define AMAC_FIND_PASS_INSTANTIATIONS(EXTERN, TYPE) \
    AMAC_FIND_PASS_INSTANTIATION(EXTERN, TYPE, MapsOne, false, true) \
    AMAC_FIND_PASS_INSTANTIATION(EXTERN, TYPE, MapsOne, false, false) \
    AMAC_FIND_PASS_INSTANTIATION(EXTERN, TYPE, MapsOne, true, true) \
    AMAC_FIND_PASS_INSTANTIATION(EXTERN, TYPE, MapsOne, true, false) \
    AMAC_FIND_PASS_INSTANTIATION(EXTERN, TYPE, MapsAll, false, true) \
    AMAC_FIND_PASS_INSTANTIATION(EXTERN, TYPE, MapsAll, false, false) \
    AMAC_FIND_PASS_INSTANTIATION(EXTERN, TYPE, MapsAll, true, true) \
    AMAC_FIND_PASS_INSTANTIATION(EXTERN, TYPE, MapsAll, true, false)

#define M(TYPE) AMAC_FIND_PASS_INSTANTIATIONS(extern, TYPE)
APPLY_FOR_AMAC_BUILD_JOIN_VARIANTS(M)
#undef M
}
