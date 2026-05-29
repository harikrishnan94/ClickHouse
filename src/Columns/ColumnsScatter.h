#pragma once

#include <Columns/IColumn.h>

#include <span>

namespace DB::ColumnsScatter
{

/// Compute per-shard row-count histogram from a batch of pids spans.
///
/// `per_shard_rows` must be pre-zeroed with size == num_shards. Call once per
/// flush in BufferedShardByHashTransform before the per-column scatter loop;
/// pass the result to every `scatter` call in that loop to eliminate the
/// K − 1 redundant pids re-scans that the internal histogram path would do.
void computeHistogram(std::span<const std::span<const UInt32>> pids_per_source, std::span<UInt32> per_shard_rows);

/// Batched, type-dispatched physical scatter.
///
/// `source_columns[b]` is the source column extracted from chunk b for one
/// column-position; every element has the same concrete column type. For
/// each b, row j of `source_columns[b]` is routed to shard
/// `pids_per_source[b][j]`. Destinations are allocated and exact-sized
/// inside; the caller does not pre-reserve.
///
/// `per_shard_rows` (optional): pre-computed histogram produced by
/// `computeHistogram`.  When non-empty (size must equal `num_shards`) the
/// histogram step inside each typed kernel is skipped, saving K−1 full pids
/// re-scans per flush.  When empty the histogram is computed internally
/// (backward-compatible convenience path used by callers that process only
/// one column at a time).
///
/// PRECONDITION (asserted at dispatch entry): every element of `source_columns`
/// has the same concrete column type.  The caller groups columns by position
/// across the pending chunk queue (one `scatter` call per column-position per
/// flush).
///
/// Returns: `MutableColumns` of length `num_shards`, each of the same
/// concrete type as the source columns.  The k-th destination holds, in
/// source-column order, every row routed to shard k.
///
/// Dispatch is O(1): a static function-pointer table indexed by
/// `IColumn::getDataType()` (one virtual call + one indexed indirect call,
/// independent of the number of supported types). Transparent wrappers
/// (`ColumnConst` / `ColumnSparse` / `ColumnReplicated`), which report their
/// nested type's index, are routed to the fallback before the table lookup.
///
/// Fast paths:
///   ColumnVector<T>   for the full integer / float / UUID / IPv4 / IPv6 set
///   ColumnDecimal<T>  (reinterpret as NativeType storage; incl. DateTime64 / Time64)
///   ColumnFixedString (runtime element size = getN())
///   ColumnString      (fused chars + offsets + per-partition byte-cursor)
///   ColumnNullable(X) (null-map scatter via UInt8 path + recursive scatter on nested)
///   ColumnTuple(...)  (recursive scatter per element, wrapped with ColumnTuple::create)
///
/// Fallback: any concrete type not in the dispatch table delegates
/// per-source-column to the legacy `IColumn::scatter()` virtual and appends
/// via `insertRangeFrom` into pre-cloneEmpty'd destinations.
[[nodiscard]] MutableColumns scatter(
    std::span<const IColumn * const> source_columns,
    std::span<const std::span<const UInt32>> pids_per_source,
    size_t num_shards,
    std::span<const UInt32> per_shard_rows = {});

} // namespace DB::ColumnsScatter
