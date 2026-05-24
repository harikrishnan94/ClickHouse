#pragma once

#include <cstddef>
#include <cstdint>


namespace DB::RadixShuffle
{

/// Abstract interface for a single typed column participating in radix
/// partitioning.  Renamed from `IColumn` in the reference benchmark to avoid
/// clashing with `DB::IColumn`.
///
/// Lifecycle per partition (called by `RadixPartitionOperator`):
///   1. `on_grow(p, col_base)` — new OutBlock allocated for partition `p`;
///      `col_base` is the start of this column's array in the new block.
///   2. `scatter_direct` or `scatter_staged` — one per batch.
///   3. `drain_one(p, cnt)` — flush the SWWC staging buffer for partition `p`
///      (called before `on_grow` and at `finish`).
class IScatterColumn
{
public:
    virtual ~IScatterColumn() = default;

    /// New OutBlock allocated for partition `p`; update internal write pointer.
    /// `col_base` points to `cols[k]` of the new block (column-major layout).
    virtual void on_grow(size_t p, void * col_base) = 0;

    /// Scalar-drain `cnt` staged values for partition `p` into the output block.
    /// Called before a `grow` and at `finish()` to flush partial SWWC buffers.
    virtual void drain_one(size_t p, uint32_t cnt) = 0;

    /// Direct scatter: advance the write pointer for each `pids[j]` entry.
    /// No staging; best for small P where output fits in cache.
    virtual void scatter_direct(const uint32_t * pids, const void * src, int n) = 0;

    /// SWWC scatter: fill per-partition staging slots; NT-flush when slot reaches 7.
    /// `positions[j]` is the pre-computed staging slot for row `j`, shared
    /// across all columns so each column independently knows when to flush.
    virtual void scatter_staged(const uint32_t * pids, const uint32_t * positions, const void * src, int n) = 0;
};

} // namespace DB::RadixShuffle
