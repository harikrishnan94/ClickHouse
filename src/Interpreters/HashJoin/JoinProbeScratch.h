#pragma once

#include <Common/PODArray.h>
#include <base/types.h>

#include <vector>

namespace DB
{

namespace JoinStuff
{
class JoinUsedFlags;
}

/// Per-slot address material of the routed lookups: the cell buffer base and the home mask of
/// the power-of-two region (NOT `bufSize - 1`; the buffers are tail-padded - see
/// `TailPaddedHashTableGrower`). At most 256 slots x 16 bytes, so the whole array stays
/// cache-resident under the find loops.
struct SlotMapDesc
{
    const void * buf;
    size_t mask;
};

/// Once-per-build address material of the routed `parallel_hash` probe, collected by
/// `ConcurrentHashJoin` when the maps are final (and once at construction, so plan-time header
/// probes see sized arrays), replacing per-probe-block O(slots) passes:
/// - `map_by_slot` - the active map object per slot, type-erased; the probe casts it back
///   under the same map-type switch that chose it;
/// - `desc_by_slot` - `SlotMapDesc` per slot for the cursor-capable map types (empty for the
///   rest);
/// - `flags_by_slot` - the per-slot used-flags structures (RIGHT/FULL shapes);
/// - `total_map_bytes` - aggregate map buffer bytes, the software-prefetch/AMAC size gate;
/// - `avg_joined_bytes_per_row` - the whole-join output-splitting estimate;
/// - `chain_may_wrap` - some slot's collision chain reached its buffer's last pad cell, so
///   walks licensed to skip the wrap check must not run.
struct RoutedProbePlan
{
    std::vector<const void *> map_by_slot;
    std::vector<SlotMapDesc> desc_by_slot;
    std::vector<JoinStuff::JoinUsedFlags *> flags_by_slot;
    size_t total_map_bytes = 0;
    size_t avg_joined_bytes_per_row = 0;
    bool chain_may_wrap = false;
};

/// Per-probe-stream scratch of the routed `parallel_hash` probe, pooled on the join and
/// reused across probe blocks so the steady state allocates nothing per block:
/// - `slot_ids` - one route slot id per source-block row;
/// - `found_word` - the AMAC find pass's per-row result: the matched cell's mapped value
///   copied by value (0 = no match; `RowRef`/`RowRefList` are 8-byte words that are never 0
///   for a real match), so the emit phase never dereferences the cell a second time after it
///   left the cache;
/// - `found_offset` - the used-flags offset of the match (slot-local), filled only for the
///   flagged RIGHT/FULL shapes.
/// The arrays are resized on demand by the paths that need them; a single-slot non-AMAC probe
/// touches none of them.
struct JoinProbeScratch
{
    PaddedPODArray<UInt8> slot_ids;
    PaddedPODArray<UInt64> found_word;
    PaddedPODArray<UInt64> found_offset;
};

}
