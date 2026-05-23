#pragma once

#include <Common/RadixShuffle/PartitionTypes.h>

#include <cstddef>
#include <cstdint>
#include <memory>


namespace DB
{
class IColumn;
}


namespace DB::RadixShuffle
{

struct ColumnPrimitives;

/// Scatter primitive (§3.2). Reads `n` rows from `src`, routes each row to
/// the partition `pids[j]`, and writes to `dst[pids[j]]`. The caller has
/// pre-reserved `dst[p]` with enough rows (and, for variable-length, bytes)
/// to hold all rows with `pids[j] == p`. `partitions` is the size of the
/// `dst` array — it bounds the per-call write-pointer table.
///
/// The first argument is the resolving `ColumnPrimitives` struct itself, so
/// composite column primitives (Nullable) can reach into their nested
/// column primitives without an extra dispatch step.
using ScatterFn
    = void (*)(const ColumnPrimitives & self, const IColumn & src, const uint32_t * pids, size_t n, size_t partitions, Reservation * dst);


/// Reconstruct primitive (§3.3). Appends rows from `views[start..end)` into
/// `target` up to but not exceeding the target's pre-allocated capacity.
/// Returns the position of the first unconsumed row across the view list.
/// For variable-length types, both row capacity AND byte capacity (chars
/// reserved via the caller's prior `reserve(...)`) bound the appended
/// content; reconstruct stops at whichever boundary is reached first.
using ReconstructFn = ResumePosition (*)(
    const ColumnPrimitives & self, const ChunkRangeView * views, size_t n_views, ResumePosition start, IColumn & target);


/// Hash primitive (§3.4). For each row `i ∈ [0, n)` it updates
/// `out[i] = hashCombine(prior_out[i], h(src[i]))` where `h(.)` is the
/// column primitive's per-row hash. The combiner is the same for every
/// column primitive resolved by `resolveColumnPrimitives` (§3.4), so
/// chaining hash calls across columns yields a deterministic composite hash.
using HashFn = void (*)(const ColumnPrimitives & self, const IColumn & src, size_t n, uint64_t * out);


/// Column-primitive triple resolved per column type by
/// `resolveColumnPrimitives` (§3 last bullet). The struct also holds:
///   - The column descriptor used by the allocator at construction time
///     (`column_desc`).
///   - For composite types (`Nullable`): an owning pointer to the nested
///     column primitives (`nested`). The nested `column_desc` describes the
///     wrapped nested column.
///   - For variable-width fixed-string columns: the row's width (`aux`).
struct ColumnPrimitives
{
    ScatterFn scatter = nullptr;
    ReconstructFn reconstruct = nullptr;
    HashFn hash = nullptr;
    ColumnDesc column_desc{};

    /// Nested column primitives for composite types. `nullptr` for non-composite.
    std::shared_ptr<const ColumnPrimitives> nested;

    /// Auxiliary scalar used by some column primitives (e.g., FixedString's `n`).
    size_t aux = 0;
};

}
