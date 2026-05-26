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


namespace DB
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
///   out[i] = hashCombine(out[i], h(src[offset + i]))
/// where h(.) is the column primitive's per-row hash function and the
/// combiner is uniform across all primitives resolved by
/// resolveColumnPrimitives.  out is uint32_t.  Must not allocate.
///
/// `offset` allows processing a sub-range of `src` without an `IColumn::cut`
/// allocation — useful when the caller batches a large column in-place.
/// `initial` — when true, writes `out[i] = hash(src[offset+i])` directly
/// (no prior, no `hashCombine`).  When false, accumulates
/// `out[i] = hashCombine(out[i], hash(src[offset+i]))`.
/// Use `initial=true` for the first (or only) key column to skip the `hashCombine`
/// overhead and eliminate the caller's `std::fill_n` / `memset` pre-pass.
using HashFn = void (*)(
    const ColumnPrimitives & self, const PartSchema & schema, const IColumn & src, size_t offset, size_t n, bool initial, uint32_t * out);


/// Partition-ID primitive.  Computes `pids[j] = hash(src[offset+j]) & mask`
/// in one SIMD pass — equivalent to the `HashFn` path but without an
/// intermediate hash buffer and without a second masking loop.
/// Use instead of `HashFn` when only one key column is needed.
using PidsFn = void (*)(const ColumnPrimitives & self, const IColumn & src, size_t offset, int n, uint32_t mask, uint32_t * pids);


/// Raw-output-pointer scatter primitives.  These write directly to per-partition
/// void* write pointers held in `ScatterState::fixed_ptrs`, bypassing the
/// `PartReservation` layer.  They are intended for use by callers that manage
/// their own output storage (e.g. `RadixShuffler` with `OutBlock`).
///
/// `self` and `partitions` are intentionally absent from all four signatures:
/// removing them keeps all parameters in x86-64 integer registers
/// (rdi/rsi/rdx/rcx/r8/r9), avoiding stack spills of `pids` and `positions`.
/// The callee derives `partitions` from `state.fixed_ptrs.size()` when needed.

/// Direct scatter: writes one element per row to the per-partition output pointer.
using RawScatterFn = void (*)(const IColumn & src, size_t offset, const uint32_t * pids, int n, ScatterState & state);

/// SWWC scatter: fills per-partition staging slots, flushes via NT stores.
/// 6 parameters — rdi=src, rsi=offset, rdx=pids, rcx=positions, r8=n, r9=&state.
/// All fit in registers; no stack spill for pids or positions.
using RawScatterSwwcFn
    = void (*)(const IColumn & src, size_t offset, const uint32_t * pids, const uint32_t * positions, int n, ScatterState & state);

/// Partial SWWC drain: copies residual staged elements to the output pointer.
/// `self` is provided so Nullable can delegate to its nested primitive.
using RawDrainFn = void (*)(const ColumnPrimitives & self, size_t p, uint32_t cnt, ScatterState & state);

/// Write-pointer update on new output block.
/// `self` is provided so Nullable can delegate to its nested primitive.
/// `capacity` is the OutBlock row capacity — Nullable uses it to split
/// `col_base` into the null-map region `[0, capacity)` and the values region
/// `[capacity, capacity*(1+sizeof(T)))`.
using RawOnGrowFn = void (*)(const ColumnPrimitives & self, size_t p, void * col_base, size_t capacity, ScatterState & state);


/// Column-primitive triple resolved per column type.  After
/// buildSchemaAndPrimitives fills in fixed_slot_indices and writes_varlen,
/// the struct carries everything needed for scatter, reconstruct, and hash.
struct ColumnPrimitives
{
    ScatterFn scatter = nullptr;
    ReconstructFn reconstruct = nullptr;
    HashFn hash = nullptr;

    PidsFn compute_pids = nullptr;
    RawScatterFn scatter_raw = nullptr;
    RawScatterSwwcFn scatter_raw_swwc = nullptr;
    RawDrainFn drain_raw = nullptr;
    RawOnGrowFn on_grow_raw = nullptr;

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

    /// Bytes per row in an OutBlock column buffer for the raw scatter path.
    /// RadixShuffler uses this per-column to allocate OutBlock memory
    /// instead of a uniform sizeof(TKey).
    /// 0 = raw scatter path not supported (e.g., String).
    size_t raw_elem_size = 0;
};

}
