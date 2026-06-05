#pragma once

#include <base/types.h>

#include <cstddef>
#include <vector>

namespace DB::RadixShuffle
{

/** Fixed-width, column-major SWWC + NT scatter primitive for the radix hash join (spec sections 4.5,
  * 4.6).
  *
  * This is the *fixed-width key+ref path* of the SWWC/NT radix-shuffle kernels (the ones on the
  * `experiment/radix-icolumn-partition-blocks` branch under `src/Common/RadixShuffle/`), reused here
  * as a focused primitive rather than the whole generic `IColumn` shuffler: the radix hash join only
  * ever scatters the join `key` and an `8 B` `BuildRef`, never payload, so none of the generic
  * `ColumnPrimitives` / `ScatterState` / `OutBlock` machinery is needed.
  *
  * **Column-major.** Each fixed-width column is scattered *independently* (`scatterColumn`): the key
  * column and the `BuildRef` column land in *separate* per-partition arrays, never interleaved into a
  * fused row cell. `scatterKeyRefTwoColumn` is a thin convenience that scatters the two columns in
  * sequence sharing the same `pid`. Column-major keeps the per-partition output dense and lets the key
  * width vary independently of the `8 B` ref (a fused cell would need a single fixed element size).
  *
  * **Supported element widths: multiples of 4 only.** A width of `4..64` that is a multiple of 4
  * (`UInt32`, `UInt64`, `UInt128`/`UUID`, `UInt256`, `Decimal*`, `FixedString(N)` with `4 | N <= 64`),
  * or any whole multiple of `64` (`FixedString(64)`, `FixedString(128)`, …). Sub-`4 B` and non-multiple
  * -of-4 widths are out of scope (the join's key gate must reject `UInt8`/`UInt16`/`Date`/`Enum8/16`).
  * The minimum copy granularity is `4 B`, which lets a grand `switch(width)` dispatch to a
  * width-templated kernel where every copy is `__builtin_memcpy_inline` of a compile-time size — i.e.
  * **direct typed stores, no `memcpy` call anywhere** (including the residual drain). The `BuildRef`
  * column is always `8 B`.
  *
  * **Two scatter paths.**
  *   - **Direct** (always available): plain typed per-partition write pointers, no staging, no NT.
  *     Width-templated for the common widths, a `4 B`-stride typed-store loop for other multiples of 4.
  *   - **SWWC + NT** (only when `ntStoresAvailable()`): software write-combining with non-temporal
  *     stores. A row is staged into an L1/L2-resident per-partition `64 B` line and a full line is
  *     flushed with one `_mm512_stream_si512` (or two `_mm256_stream_si256` on AVX2), which bypasses
  *     the cache (no read-for-ownership, no pollution) so the streamed outputs never evict the staging
  *     / `pid` / hash-table working set. SWWC tiles only widths that divide 64 (`{4,8,16,32}`); a width
  *     that is a multiple of 64 is streamed directly (`width / 64` NT lines, no staging); any other
  *     multiple of 4 (`12,20,24,…`) cannot tile a `64 B` line without a large `lcm(width,64)` buffer,
  *     so it uses the direct path.
  *
  * **SWWC exists only when NT stores do.** There is no scalar write-combine fallback (it would be
  * strictly slower than direct). When NT is unavailable — the default `x86-64-v2` build
  * (`ENABLE_MULTITARGET_CODE=0`), or a non-`v3`/`v4` CPU — `scatterColumn(use_swwc=true)` runs the
  * direct path. Under the realistic alloc+first-touch-fault model (P2 calibration) SWWC + NT beats the
  * direct scatter only at high per-pass fanout (`P >= 2048`) — see `shouldUseSwwc`.
  *
  * Routing is the top-bit slice of the stored `uint16` leaf id: `part = (pid >> shift) & mask`
  * (spec section 4.5). There is no re-hash and no separate count pass; the per-partition output bases
  * are exact-sized from the P1 histogram and must be `64 B`-aligned (required by the NT stores).
  * `ColumnsScatter::scatter` is never used; the only fallback is the direct batched scatter.
  */

/// Build-side reference: which accumulated block, and which row within it (spec section 4.6).
/// Exactly 8 B; `EMPTY = 0xFFFFFFFF` is reserved for the leaf cell's empty sentinel (spec section 5.6).
struct BuildRef
{
    UInt32 block_no;
    UInt32 row_no;
};
static_assert(sizeof(BuildRef) == 8, "BuildRef must be exactly 8 bytes for the 16 B leaf cell");

/// Whether non-temporal (NT) stores are compiled in AND supported by the current CPU. When false there
/// is no SWWC path at all — `scatterColumn(use_swwc=true)` runs the direct batched scatter (a scalar
/// write-combine would only add a staging copy with no cache-bypass benefit, so it is not offered). NT
/// requires a multitarget build (`ENABLE_MULTITARGET_CODE=1`, which `src/CMakeLists.txt` disables at the
/// `x86-64-v2` baseline) on a `v3`/`v4`-capable CPU. (P2 finding: in the default `x86-64-v2` reldeb
/// build NT is dormant, so `shouldUseSwwc` returns false and the join uses the direct scatter; in a
/// multitarget build the NT path activates automatically.)
bool ntStoresAvailable() noexcept;

/// SWWC engagement rule, recalibrated in P2 under the realistic alloc+first-touch-fault model
/// (`bench_radix_sweep_native`, 16 threads). SWWC + NT beats the direct batched scatter only at high
/// fanout (`P >= 2048`, ~+10% across key widths) and only when NT stores are available; below `~1024`
/// the per-partition outputs stay cache-resident and the direct scatter wins. So SWWC is engaged iff
/// `ntStoresAvailable()` and `partitions >= 2048`; every other case (and the whole `x86-64-v2` build)
/// uses the direct path. `num_columns` had the same measured crossover, so a single rule is used.
bool shouldUseSwwc(int num_columns, int partitions) noexcept;

/// Round `bytes` up to a multiple of `64` (NT stores write whole `64 B` lines; per-partition output
/// bases must therefore be `64 B`-aligned and have a `64 B`-rounded capacity).
inline size_t roundUpTo64(size_t bytes) noexcept
{
    return (bytes + 63) & ~size_t{63};
}

/** Reusable, thread-local scratch for the column scatter: one `64 B`-aligned SWWC write-combining line
  * per partition, the per-partition write cursors, and the per-partition line fills. One instance is
  * owned per worker thread (the scatter is otherwise lock-free); reuse it across passes / blocks /
  * columns to avoid re-allocating the staging every call. Sized for a single column at a time — the
  * two-column wrapper scatters key then ref through the same scratch.
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

    /// 64 B-aligned write-combining lines, one per partition: `max_partitions * 64` bytes.
    char * staging() noexcept { return staging_buf; }
    /// Per-partition write cursors (type-erased so the NT flush can advance them as `void *&`).
    void ** cursors() noexcept { return cursor_ptrs.data(); }
    /// Per-partition fill of the current write-combining line, in bytes (`0..63`).
    UInt32 * fill() noexcept { return line_fill.data(); }

private:
    static constexpr size_t LINE_BYTES = 64;

    size_t capacity;
    char * staging_buf = nullptr; /// capacity * 64 B, 64 B-aligned (one write-combining line per partition).
    std::vector<void *> cursor_ptrs; /// capacity.
    std::vector<UInt32> line_fill; /// capacity.

    void freeStaging() noexcept;
};

/// Bytes written via NT stores during a SWWC scatter (feeds the `RadixHashNTStoreBytes` event, P3).
struct ScatterStats
{
    size_t nt_store_bytes = 0;
};

/** Column-major scatter of one fixed-width column, routed by `part = (pid[j] >> shift) & mask`.
  *
  * `pid`        : `n` stored `uint16` leaf ids (P1 selector output).
  * `src`        : `n` contiguous fixed-width elements of `elem_width` bytes each.
  * `elem_width` : element width in bytes — a multiple of 4 in `[4, 64]`, or any multiple of `64`.
  * `partitions` : this pass's fanout `P = 1 << pass_bits`.
  * `out`        : `partitions` per-partition output bases, each `64 B`-aligned (when `use_swwc`) with
  *                capacity `>= hist[p] * elem_width` bytes (exact-sized from the histogram).
  * `use_swwc`   : when false, the non-SWWC batched fallback is used (plain per-partition writes).
  *
  * On return each partition's output holds exactly its histogram count of elements, in arrival order.
  */
ScatterStats scatterColumn(
    const UInt16 * pid,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const void * src,
    size_t elem_width,
    size_t partitions,
    void * const * out,
    ScatterScratch & scratch,
    bool use_swwc);

/** Two-column key + `BuildRef` scatter (the production shape): scatters the key column and the ref
  * column *separately* (column-major), both routed by the same `pid`. `key_width` is the key element
  * width (see `scatterColumn`); the ref column is always `8 B`. `key_out` / `ref_out` are the
  * per-partition bases, each `64 B`-aligned, exact-sized from the histogram. Returns the combined NT
  * store bytes (key + ref).
  */
ScatterStats scatterKeyRefTwoColumn(
    const UInt16 * pid,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const void * keys,
    size_t key_width,
    const BuildRef * refs,
    size_t partitions,
    void * const * key_out,
    BuildRef * const * ref_out,
    ScatterScratch & scratch,
    bool use_swwc);

}
