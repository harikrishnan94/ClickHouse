#pragma once

#include <Columns/IColumn.h>
#include <base/types.h>

namespace DB
{

struct DenseHyperLogLog;

/** Routing hash of `PartitionedHashJoin` (the imported `ColumnsScatter` route-word family):
  * one 32-bit route word per row over the join key columns, deliberately independent of the
  * CRC32C the leaf hash tables bucket by. The build fill saves the top 16 bits per row and the
  * probe recomputes the word per row; a partition plan of `bits` partitions routes a row to leaf
  * `word >> (32 - bits)` (MSB-first). Build and probe share one word computation, so the fold
  * must stay bit-identical across both entry points below.
  *
  * `key_columns` are the prepared key columns after null-map extraction (nested, non-nullable);
  * a live `ColumnLowCardinality` (the dictionary-aware map types) is routed by its value bytes,
  * so it produces the same words as the plain column of the same values on the other join side.
  */
void computeJoinRouteWords(const ColumnRawPtrs & key_columns, size_t rows, UInt32 * words);

/** The fill-side consumption of the same words, fused into the word loop (no 32-bit transient):
  * the top 16 bits are stored per row into `routes` for ALL rows - the scatter's bucket
  * derivation reads a skipped row's route too - and the full word feeds the lane sketch only
  * where `skip` (nullable, 1 = skip) does not filter the row.
  */
void computeJoinRoutesForFill(const ColumnRawPtrs & key_columns, size_t rows, const UInt8 * skip, UInt16 * routes, DenseHyperLogLog & hll);

/** The sketch-free fill variant, taken when a cached distinct-key count from a previous run of
  * the same query replaces the sketch estimate entirely: routes are still stored for every row
  * (the scatter's bucket derivation reads them), the per-row sketch feed is skipped. No `skip`
  * parameter - it only ever filtered the sketch feed, never the route store.
  */
void computeJoinRoutesForFill(const ColumnRawPtrs & key_columns, size_t rows, UInt16 * routes);

/** The probe-side consumption, fused the same way: the leaf id `word >> (32 - bits)` is stored
  * per row directly (no 32-bit word transient). Agrees with the build's stored-route slice
  * `route >> (16 - bits)` on the shared top bits for every plan (`0 < bits <= 16`).
  */
void computeJoinLeafIds(const ColumnRawPtrs & key_columns, size_t rows, size_t bits, UInt16 * leaf_ids);

}
