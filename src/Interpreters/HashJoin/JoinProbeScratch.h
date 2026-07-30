#pragma once

#include <Common/PODArray.h>
#include <base/defines.h>
#include <base/types.h>

#include <vector>

namespace DB
{

namespace JoinStuff
{
class JoinUsedFlags;
}

/** The hash-derived slot route of the open-addressing (cursor-capable) join map families: the
  * row's slot is the TOP bits of the map's own hash, so the lookup that computes the hash
  * anyway routes for free, and the build scatter derives the same slot from the same hash
  * (`computeDispatchSlotIds` - the route is a build/probe contract). The bottom bits stay the
  * cell placement (`grower.place` is `hash & mask`).
  *
  * The route space is 32-bit for EVERY family: the string hash (`CRC32Hash`) returns
  * `(res << 32) | res` - its high 32 bits are a copy of the low 32 - and the integer CRC32C
  * hashes zero-extend a 32-bit result, so only the low 32 bits can be treated as independent.
  * `route_shift` is `32 - log2(slots)`, giving `slot = word >> shift` MSB-first; the route
  * bits overlap `place` only once one slot's table exceeds 2^24 cells (~4B distinct keys over
  * 256 slots) - the same property the pre-routing two-level maps had. A single-slot plan
  * passes `route_shift = 32`; shifting the 64-bit zero-extension keeps that case well defined
  * (always slot 0).
  */
ALWAYS_INLINE inline size_t joinHashRouteSlot(size_t hash, UInt32 route_shift)
{
    return static_cast<size_t>(static_cast<UInt32>(hash)) >> route_shift;
}

/// Per-slot address material of the routed lookups: the cell buffer base and mask. At most
/// 256 slots x 16 bytes, so the whole array stays cache-resident under the find loops.
struct SlotMapDesc
{
    const void * buf;
    size_t mask;
};

/// Once-per-build address material of the routed `parallel_hash` probe, collected by
/// `ConcurrentHashJoin` when the maps are final, replacing per-probe-block O(slots) passes:
/// - `map_by_slot` - the active map object per slot, type-erased; the probe casts it back
///   under the same map-type switch that chose it;
/// - `desc_by_slot` - `SlotMapDesc` per slot for the cursor-capable map types (empty for the
///   rest);
/// - `flags_by_slot` - the per-slot used-flags structures (RIGHT/FULL shapes);
/// - `total_map_bytes` - aggregate map buffer bytes, the software-prefetch/AMAC size gate;
/// - `avg_joined_bytes_per_row` - the whole-join output-splitting estimate.
struct RoutedProbePlan
{
    std::vector<const void *> map_by_slot;
    std::vector<SlotMapDesc> desc_by_slot;
    std::vector<JoinStuff::JoinUsedFlags *> flags_by_slot;
    size_t total_map_bytes = 0;
    size_t avg_joined_bytes_per_row = 0;
};

/// Per-probe-stream scratch of the routed `parallel_hash` probe, pooled on the join and
/// reused across probe blocks so the steady state allocates nothing per block:
/// - `slot_ids` - one route slot id per source-block row, filled EAGERLY only for the map
///   families that cannot derive the slot from the lookup's own hash (`key8`/`key16` and the
///   range maps) and for the mixed ON-expression path (see `computeDispatchSlotIds`); the
///   open-addressing families route inline through `joinHashRouteSlot` and leave it empty;
/// - `found_word` - the AMAC find pass's per-row result: the matched cell's mapped value
///   copied by value (0 = no match; `RowRef`/`RowRefList` are 8-byte words that are never 0
///   for a real match, so the emit phase never dereferences the cell a second time after it
///   left the cache) or, for ASOF, the mapped value's address;
/// - `found_offset` - the used-flags offset of the match (slot-local), filled only for the
///   flagged RIGHT/FULL shapes;
/// - `found_slot` - the row's route slot as the AMAC find pass derived it, for the emit
///   side's per-slot used-flags selection; filled only for the flagged shapes, like
///   `found_offset`.
/// The arrays are resized on demand by the paths that need them; a single-slot non-AMAC probe
/// touches none of them.
struct JoinProbeScratch
{
    PaddedPODArray<UInt8> slot_ids;
    PaddedPODArray<UInt64> found_word;
    PaddedPODArray<UInt64> found_offset;
    PaddedPODArray<UInt8> found_slot;
};

}
