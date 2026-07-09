#pragma once

#include <Columns/IColumn_fwd.h>
#include <Common/ThreadPool_fwd.h>

#include <base/types.h>

#include <atomic>
#include <cstddef>
#include <functional>
#include <vector>

namespace DB::RadixJoin
{

/** Column-wise radix scatter of whole blocks (the probe-side counterpart of KeyRefScatter's
  * fused-record build scatter): every input row is routed by a caller-computed 32-bit route word
  * into one of `fanout` partitions, and each partition is materialized as one exactly-sized set
  * of destination columns.
  *
  * Routing is hash-agnostic: the kernels never look at key values, only at the route words the
  * caller derived from them (production: the packed-key hash of PackedKeyHash, so probe
  * partitions align with the build's leaf tables; the route hash must stay independent of the
  * CRC32C the leaf tables bucket with, otherwise partition assignment correlates with in-table
  * bucket placement and per-partition tables see a skewed hash space). Every pass slices a
  * disjoint bit range of the same 32-bit word, high bits first.
  *
  * The scatter follows the histogram + fused prefix-sum + exact-allocation + direct-placement
  * scheme:
  *   - the histogram phase computes each row's partition id once and stores it as 2 bytes; the
  *     ids replace all further routing uses of the route words (which lets the input's route
  *     column be dropped eagerly) and every column's scatter routes through them - a 2 B id read
  *     per row instead of a 4 B route re-read;
  *   - per-partition destination columns are allocated exactly once from the histogram, with
  *     uninitialized contents: no memset, the pages are first-touched by the scatter writes
  *     themselves. Each worker owns a contiguous disjoint partition range of the fused
  *     prefix-sum, and each worker writes only its own disjoint row range of every
  *     (partition, column) output, so the scatter inner loops need no atomics;
  *   - columns are scattered one at a time within a routing window (column-major loop order), so
  *     only `fanout` output streams and one fanout x 64 B staging set are live per worker at any
  *     instant;
  *   - at fanout >= 256 a software write-combining path stages one 64-byte line per partition
  *     and flushes it with a non-temporal store, avoiding both the cache pollution and the
  *     read-for-ownership traffic that create the high-fanout cliff of the naive scatter;
  *   - consumed input is dropped as early as possible (per chunk batch in the first pass - for
  *     pass 0 that releases the references to the caller's blocks - and per column in refine
  *     passes), keeping the scatter's resident memory near one copy of the side instead of two.
  *
  * Element widths of {1, 2, 4, 8, 16, 32, 64} bytes are supported, via the same compile-time
  * width dispatch pattern as KeyRefScatter. Accepted column types are exactly the ones the
  * output allocator handles: `ColumnVector` (integers, floats, UUID, IPv4, IPv6),
  * `ColumnDecimal` (including DateTime64 and Time64) and `ColumnFixedString`; anything else -
  * strings, Nullable, LowCardinality, ... - is rejected up front, at input validation, never
  * mid-scatter on a worker.
  *
  * Unlike the donor KeyRefScatter (whose non-temporal path is compiled for x86-64 only), these
  * kernels use the compiler's portable non-temporal store builtin directly, so the SWWC path
  * runs on every architecture, aarch64 included.
  */

/// One materialized set of scattered rows: the output of one partition. `rows == 0` means the
/// partition is empty (no columns are allocated for it).
struct ScatterChunk
{
    Columns columns;
    size_t rows = 0;
};

/// One input chunk: the fixed-width columns to scatter plus the caller-computed 32-bit route
/// word per row (a `ColumnUInt32` of the same row count). The scatter takes ownership and
/// releases both as soon as they are consumed.
struct RoutedChunk
{
    Columns columns;
    ColumnPtr routes;
    size_t rows = 0;
};

/// Hard memory-correctness ceiling of the SWWC scatter, not a runtime tuning knob: at fanout F
/// each worker's per-partition SWWC state is F * (64 B staging + 8 B cursor + 4 B fill) ~= 76 B,
/// so F = 8192 needs ~608 KiB, which only fits an L2 >= ~1 MiB. The *effective* per-pass fanout
/// a planner should use is the min of this constant and an L2-derived cap (bit_floor(L2 / 128),
/// leaving headroom over the 76 B for the histogram and cursors); this constant only bounds how
/// far a single pass may go before the SWWC state stops fitting in cache at all.
///
/// Enforced in computePassBits, which clamps the caller's f_max to this ceiling when planning
/// passes. The explicit pass_bits entry points deliberately accept up to 16 bits per pass (the
/// partition ids are UInt16, the kernels' only hard bound), so tests and benchmarks may exceed
/// the cache-fit ceiling on purpose.
constexpr size_t MAX_FANOUT_PER_PASS = 8192;

/// Which scatter write path to use. `Automatic` is the production setting; the forced values
/// exist for the SWWC-vs-DIRECT equivalence gate in the unit tests and for benchmarking.
enum class ScatterPath : UInt8
{
    Automatic, /// DIRECT below 256 partitions, SWWC + non-temporal stores at or above
    Direct,    /// plain per-partition cursor stores
    Swwc,      /// software write-combining + non-temporal stores
};

/// Splits log2(p_star) partition bits into passes of at most log2(f_max) bits each (f_max is
/// clamped to MAX_FANOUT_PER_PASS), spread evenly (pass sizes differ by at most one bit).
/// Single pass is the common case. p_star <= 1 yields an EMPTY plan by contract: no
/// partitioning is needed, and the scatter entry points (which require at least one pass) must
/// not be called with it.
std::vector<size_t> computePassBits(size_t p_star, size_t f_max);

/** Radix scatter of one whole side into 2^(sum of pass_bits) partitions (single pass up to the
  * full partition count; multiple passes only as a fallback when the partition count exceeds the
  * per-pass fanout cap). Returns one `ScatterChunk` per final partition, indexed by the top
  * `sum(pass_bits)` bits of the route word; empty partitions have `rows == 0`.
  *
  * The first pass is cooperative (per-worker histograms into disjoint slices, fused prefix-sum +
  * allocation over disjoint partition ranges, one fused all-columns scatter that walks each
  * worker's chunk stripe in batches and drops each batch's input right after its last column).
  * Refine passes pull groups with dynamic scheduling (an atomic counter, not a static stripe,
  * because group sizes diverge - the skew defense) and process each group entirely
  * worker-locally, allocating each output column just-in-time and dropping each consumed input
  * column so the freed extents are immediately reusable. Non-final passes carry the remaining
  * route bits along by scattering the route words as one more 4-byte column; the final pass
  * drops them right after its histogram.
  *
  * `pool` must be dedicated to this call and able to run `num_threads` jobs concurrently.
  * An exception on any worker aborts the scatter and is rethrown after the workers stop.
  * `cancelled` is checked at phase and batch/group boundaries and aborts the scatter with a
  * QUERY_WAS_CANCELLED exception.
  *
  * Transient scratch: the histogram phase stores one 2-byte partition id per row of the WHOLE
  * side and holds them until the first pass's scatter ends (unlike scatterWaves, whose ids only
  * ever cover one window), plus each worker's SWWC state (~76 B per partition). Callers doing
  * budget accounting must charge both.
  *
  * Preconditions (histogram/offset counters are UInt32, partition ids are UInt16): the side has
  * at most 2^32 - 1 rows, every pass has 1..16 bits, and all passes together have at most 32.
  */
std::vector<ScatterChunk> scatterColumns(
    ThreadPool & pool,
    size_t num_threads,
    std::vector<RoutedChunk> chunks,
    const std::vector<size_t> & pass_bits,
    std::atomic<bool> & cancelled,
    ScatterPath path = ScatterPath::Automatic);

/// Consumer of one non-empty partition of one wave's window. Called concurrently from all
/// workers; receives ownership of the chunk (freed on return).
using ConsumePartition = std::function<void(size_t partition, ScatterChunk chunk)>;

/** Streaming scatter + consume (the evict-all-at-budget probe shape): the side is consumed in
  * `waves` consecutive windows of whole chunks (window w covers the chunk index range
  * [n*w/waves, n*(w+1)/waves); `waves` is clamped to [1, chunks.size()]); each window is
  * radix-scattered in a single pass of `bits` bits and every non-empty partition's window chunk
  * is handed to `consume` (work-stealing over partitions) and dropped before the next window
  * starts. Each window's input chunks are released right after they are scattered, so upstream
  * blocks are recycled window by window.
  *
  * The whole wave loop runs inside ONE pool dispatch: phases (histogram, fused prefix-sum +
  * allocation, fused all-columns scatter, consume) are separated by std::barrier instead of
  * per-phase pool dispatches, and per-worker scratch (SWWC staging, histogram lanes, partition
  * ids, cursors) persists across waves. This removes the per-wave overhead that dominates small
  * windows when each wave pays its own per-phase pool round-trips (~4 dispatches/wave, measured
  * ~1.9 ms/wave at 96 threads).
  *
  * Error handling: a worker whose phase body throws (including out of `consume`) still arrives
  * at every barrier - the first exception is captured, later phases are skipped, and all
  * workers leave the barrier loop together at the wave boundary; the exception is rethrown here
  * after the dispatch joins. No worker is ever left waiting. The stop only latches at phase
  * barriers, though: `consume` may keep being invoked on other workers for the rest of the
  * current wave (they shed the remaining partitions best-effort, but promptness is not
  * guaranteed - do not assume consumption stops at the first failure). `cancelled` is checked
  * at every phase boundary and between consumed partitions and surfaces as a
  * QUERY_WAS_CANCELLED exception the same way; partitions consumed before the stop keep
  * everything `consume` was already handed.
  *
  * `pool` must be dedicated to this call and able to run `num_threads` jobs concurrently.
  * Preconditions: at most 2^32 - 1 rows and 1 <= bits <= 16.
  */
void scatterWaves(
    ThreadPool & pool,
    size_t num_threads,
    std::vector<RoutedChunk> chunks,
    size_t bits,
    size_t waves,
    const ConsumePartition & consume,
    std::atomic<bool> & cancelled);

}
