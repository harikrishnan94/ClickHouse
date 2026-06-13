#pragma once

#include <Interpreters/RowRefs.h>

#include <base/types.h>

#include <cstddef>
#include <vector>

namespace DB::RadixJoin
{

/// The build-row reference is the shared `DB::BuildRef` (see Interpreters/RowRefs.h): an 8-byte
/// `{ row_no, block_no }` whose `block_no` MSB is the SINGLETON_FLAG. RadixHashJoin keeps the flag on
/// cell HEADS only; every ref outside the heads (scatter output, chain links, probe output) is
/// flag-free, so callers index with `ref.blockNo()` / `ref.rowNo()` which mask the flag.
using DB::BuildRef;

/** Fixed-width, column-major radix scatter for the build side.
  *
  * The build never moves payload — it only partitions two narrow columns into per-leaf arrays: the
  * packed key (`key_width` bytes, a multiple of 4 in [4, 64]) and an 8-byte `BuildRef`. They are
  * scattered as SEPARATE columns into SEPARATE per-partition arrays (column-major), each routed by
  * the same per-row route word `part = (route >> shift) & mask`. Column-major keeps each output
  * dense and lets the key width vary independently of the 8-byte ref (a fused row cell would force a
  * single element size and waste space).
  *
  * Two write paths, chosen by fanout:
  *   - DIRECT: a plain per-partition write cursor; every element is an inlined typed store. Best when
  *     the partition count is small enough that all the live output lines stay cache-resident.
  *   - SWWC + NT: software write-combining into one 64-byte staging line per partition, flushed with
  *     a non-temporal store (`vmovnt*`) that bypasses the cache. Beyond a few hundred partitions the
  *     per-partition output lines no longer fit in cache, and the cache-bypassing NT path wins by not
  *     evicting the hot working set (staging line + route buffer + histogram). It only exists when NT
  *     stores are actually compiled and supported (`ntStoresAvailable`); otherwise DIRECT is used,
  *     because a scalar write-combine would just be a slower direct scatter.
  *
  * All entry points are INCREMENTAL: the per-partition write position lives in caller-owned cursors
  * (DIRECT) or in the `ScatterScratch` (SWWC), so a large input is scattered in successive small
  * chunks into the SAME single exact-sized per-partition allocation — no per-chunk re-basing and no
  * separate count pass (sizes come from the build histogram). This is what makes the build allocate
  * each partition exactly once (the "no allocator churn" property).
  */

/// Reserved row index marking an empty leaf cell / chain tail (cells + chain are memset to 0xFF). The
/// empty-cell / chain-tail sentinel is `row_no == INVALID_ROW`.
inline constexpr UInt32 INVALID_ROW = 0xFFFFFFFFu;
/// MSB of a cell-head block_no: "this key occurs exactly once on the build side" (probe fast path).
inline constexpr UInt32 SINGLETON_FLAG = DB::BuildRef::SINGLETON_FLAG;
/// Low 31 bits: the real block_no. The build caps the block count to this many blocks.
inline constexpr UInt32 BLOCK_NO_MASK = DB::BuildRef::BLOCK_NO_MASK;

/// Cache-line width: NT stores and the SWWC staging line are whole lines of this size.
inline constexpr size_t LINE_BYTES = 64;

/// Round up to a whole cache line (per-partition SWWC outputs are line-aligned and line-padded so a
/// final partial line can be drained with one store without overrunning the allocation).
inline size_t roundUpToLine(size_t bytes) noexcept
{
    return (bytes + LINE_BYTES - 1) & ~(LINE_BYTES - 1);
}

/// Whether non-temporal stores are compiled in and supported by the running CPU. False -> no SWWC.
bool ntStoresAvailable() noexcept;

/// Engage SWWC+NT iff NT is available and the per-pass fanout is large enough that the direct path's
/// live output lines would spill the cache. Measured crossover (realistic alloc+first-touch-fault
/// model, 16 threads, x86-64-v3): ~256 partitions.
bool shouldUseSwwc(int partitions) noexcept;

/** Per-worker reusable scratch for the SWWC path: one 64-byte write-combining staging line per
  * partition, a write cursor per partition, and the current fill of each staging line. Reused across
  * chunks/passes to avoid re-allocating the staging every call. One instance handles one column at a
  * time; key+ref need two instances (their cursors and outputs are independent).
  */
class ScatterScratch
{
public:
    explicit ScatterScratch(size_t max_partitions);
    ~ScatterScratch();

    ScatterScratch(const ScatterScratch &) = delete;
    ScatterScratch & operator=(const ScatterScratch &) = delete;
    ScatterScratch(ScatterScratch && other) noexcept;
    ScatterScratch & operator=(ScatterScratch && other) noexcept;

    size_t maxPartitions() const noexcept { return capacity; }

    char * staging() noexcept { return staging_buf; }      /// capacity * LINE_BYTES, LINE_BYTES-aligned
    void ** cursors() noexcept { return cursor_ptrs.data(); } /// per-partition write pointer
    UInt32 * fill() noexcept { return line_fill.data(); }    /// per-partition staging fill in bytes [0, LINE_BYTES)

    /// Seed every cursor to nullptr / fill to 0 — call before reusing for a fresh set of partitions
    /// whose cursors the caller then sets.
    void resetFills(size_t partitions) noexcept;

private:
    size_t capacity;
    char * staging_buf = nullptr;
    std::vector<void *> cursor_ptrs;
    std::vector<UInt32> line_fill;

    void freeStaging() noexcept;
};

/// DIRECT incremental scatter of one fixed-width column into caller-owned per-partition cursors. Each
/// row is appended to `cursors[part]` and the cursor is advanced in place. `elem_width` is a multiple
/// of 4 in [4, 64] (or any multiple of 4 for the ref's 8). Returns bytes written (`n * elem_width`).
size_t appendColumnDirect(
    const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const void * src, size_t elem_width, void ** cursors);

/// SWWC+NT incremental scatter of one fixed-width column. The per-partition write position, staging
/// line and fill live in `scratch`; the caller seeds `scratch.cursors()[p]` to each partition's start
/// (a fresh scratch already has zero fills) before the first chunk and calls `drainColumnSwwc` once
/// after the last chunk. A worker's start is generally not line-aligned, so rows are written directly
/// until the cursor reaches a line boundary (head peel), after which they are staged and NT-flushed in
/// whole lines. Only valid when `ntStoresAvailable()`. Returns bytes scattered (`n * elem_width`, the
/// same accounting as DIRECT, so totals are path-independent).
size_t appendColumnSwwc(
    const UInt32 * route, UInt32 shift, UInt32 mask, size_t n, const void * src, size_t elem_width, ScatterScratch & scratch);

/// Drain each partition's residual (< one line) staging bytes to its cursor, then make the NT stores
/// globally visible. Call once per column after all `appendColumnSwwc` chunks.
void drainColumnSwwc(size_t partitions, ScatterScratch & scratch);

}
