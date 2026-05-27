#pragma once

#include <Columns/IColumn.h>
#include <Columns/IColumn_fwd.h>
#include <Common/PODArray.h>
#include <Common/WeakHash.h>

#include <cstddef>
#include <cstdint>
#include <vector>


namespace DB
{

/// Per-operation timings accumulated over the lifetime of one
/// `NativeRadixShuffler` instance.  All values are nanoseconds.
struct NativeTimings
{
    uint64_t hash_ns = 0;
    uint64_t selector_build_ns = 0;
    uint64_t scatter_ns = 0;
    uint64_t total_process_ns = 0;
    size_t rows_processed = 0;
    size_t blocks_processed = 0;
};


/// Radix partition operator using `IColumn::scatter` for the scatter step,
/// with the same hash + Selector strategy as `BufferedShardByHashTransform`
/// (PR #104233):
///
///   1. Compute `WeakHash32` over ALL K key columns by calling
///      `column->getWeakHash32()` on the first column and then
///      `hash.update(column->getWeakHash32())` for each subsequent column.
///      This matches `BufferedShardByHashTransform::generateOutputChunks` exactly.
///
///   2. Build `IColumn::Selector` via Lemire fastrange:
///        selector[i] = ((intHashCRC32(hash[i]) & 0xFFFFFFFF) * P) >> 32
///      This is the same formula used in `JoinCommon::hashToSelector`.
///      Works for any P (not just powers of two).
///
///   3. For each of the K input columns call `column->scatter(P, selector)`.
///
/// No batching: each `process(columns)` call immediately scatters and emits
/// one `Columns` block per non-empty partition.  `finish()` is a no-op.
///
/// Output: `output_[p]` is a list of `Columns` blocks for partition `p`,
/// one entry per `process()` call that produced rows for partition `p`.
class NativeRadixShuffler
{
public:
    /// `num_partitions` — P (any positive integer, not required to be a power of two).
    /// `num_columns`    — K, the number of key columns passed to `process()`.
    NativeRadixShuffler(int num_partitions, int num_columns);

    /// Scatter one input block into per-partition output.
    /// All K columns must be present in `columns`.
    void process(const DB::Columns & columns);

    /// No-op — kept for API parity with `BatchedRadixShuffler`.
    void finish() { }

    /// Per-partition output.  `output()[p]` is a list of `Columns` blocks,
    /// one per `process()` call that produced rows for partition `p`.
    [[nodiscard]] std::vector<std::vector<DB::Columns>> & output() noexcept { return output_; }
    [[nodiscard]] const std::vector<std::vector<DB::Columns>> & output() const noexcept { return output_; }

    /// Accumulated per-operation timings (valid after all `process()` calls).
    [[nodiscard]] const NativeTimings & timings() const noexcept { return timings_; }

private:
    int num_partitions_;
    int num_columns_;

    std::vector<std::vector<DB::Columns>> output_; ///< [P][block]

    /// Scratch buffers reused across process() calls.
    IColumn::Selector scratch_selector_; ///< PaddedPODArray<UInt64>

    NativeTimings timings_;
};

} // namespace DB
