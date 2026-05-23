#pragma once

#include <Common/RadixShuffle/PartSchema.h>
#include <Common/RadixShuffle/PartitionTypes.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>


namespace DB
{
class IColumn;
}


namespace DB::RadixShuffle
{

struct ColumnPrimitives;

/// Scatter primitive.  Reads n rows from src, routes each row to
/// partition pids[j] (uint16_t, in [0, partitions)), and writes into
/// the fixed slots and (if writes_varlen) the data portion of
/// dst[pids[j]].  The caller has pre-reserved each destination via
/// Handle::reserve.  Must not allocate.
///
/// state holds the per-partition write-pointer cache for this column.
/// On the first call (state.initialized == false) all P partitions are
/// initialised from dst.  On subsequent calls only the partitions whose
/// bit is set in stale_fixed_bitset are refreshed from dst; the rest
/// reuse their cached pointers.  For varlen columns the data pointer is
/// additionally refreshed whenever dst[p].data differs from the cached
/// DataChunk pointer.
///
/// stale_fixed_bitset is ceil(P/64) uint64_t words returned by
/// Handle::reserve; the caller passes the same array to every column's
/// scatter for a given batch.
using ScatterFn = void (*)(
    const ColumnPrimitives & self,
    const PartSchema & schema,
    const IColumn & src,
    const uint16_t * pids,
    size_t n,
    size_t partitions,
    const PartReservation * dst,
    ScatterState & state,
    const uint64_t * stale_fixed_bitset);


/// Reconstruct primitive.  Appends rows from views[start..] into target
/// up to but not exceeding the target's pre-allocated capacity.  Returns
/// the position of the first unconsumed row across the view list.  For
/// varlen types both row capacity AND byte capacity bound the content;
/// reconstruct stops at whichever boundary is reached first.  Must not
/// allocate.
using ReconstructFn = ResumePosition (*)(
    const ColumnPrimitives & self,
    const PartSchema & schema,
    const PartReservationView * views,
    size_t n_views,
    ResumePosition start,
    IColumn & target);


/// Hash primitive.  For each row i in [0, n) updates
///   out[i] = hashCombine(out[i], h(src[i]))
/// where h(.) is the column primitive's per-row hash function and the
/// combiner is uniform across all primitives resolved by
/// resolveColumnPrimitives.  out is uint32_t.  Must not allocate.
using HashFn = void (*)(const ColumnPrimitives & self, const PartSchema & schema, const IColumn & src, size_t n, uint32_t * out);


/// Column-primitive triple resolved per column type.  After
/// buildSchemaAndPrimitives fills in fixed_slot_indices and writes_varlen,
/// the struct carries everything needed for scatter, reconstruct, and hash.
struct ColumnPrimitives
{
    ScatterFn scatter = nullptr;
    ReconstructFn reconstruct = nullptr;
    HashFn hash = nullptr;

    /// Indices into PartSchema::fixed_slots that this primitive owns.
    /// For Nullable(X) the first index is always the NullMap slot; the
    /// nested primitive for X owns the remaining slot indices.
    std::vector<size_t> fixed_slot_indices;

    /// True if this primitive writes to the data (varlen) chunk.
    bool writes_varlen = false;

    /// Nested column primitives for composite types (Nullable).
    /// nullptr for non-composite.
    std::shared_ptr<const ColumnPrimitives> nested;

    /// Auxiliary scalar used by some primitives (e.g., FixedString row
    /// width n).
    size_t aux = 0;
};

}
