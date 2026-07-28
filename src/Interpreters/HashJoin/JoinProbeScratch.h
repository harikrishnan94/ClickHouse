#pragma once

#include <Common/PODArray.h>
#include <base/types.h>

namespace DB
{

/// Per-probe-stream scratch of the routed `parallel_hash` probe, pooled on the join and
/// reused across probe blocks so the steady state allocates nothing per block:
/// - `slot_ids` - one route slot id per source-block row;
/// - `found_word` - the AMAC find pass's per-row result: the matched cell's mapped value
///   copied BY VALUE (0 = no match; `RowRef`/`RowRefList` are 8-byte words that are never 0
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
