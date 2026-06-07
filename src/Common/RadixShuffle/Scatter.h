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
  * sequence sharing the same `hash`. Column-major keeps the per-partition output dense and lets the key
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
  *     / `hash` / hash-table working set. SWWC tiles only widths that divide 64 (`{4,8,16,32}`); a
  *     width that is a multiple of 64 is streamed directly (`width / 64` NT lines, no staging); any
  *     other multiple of 4 (`12,20,24,…`) cannot tile a `64 B` line without a large `lcm(width,64)`
  *     buffer, so it uses the direct path.
  *
  * **SWWC exists only when NT stores do.** There is no scalar write-combine fallback (it would be
  * strictly slower than direct). When NT is unavailable — the default `x86-64-v2` build
  * (`ENABLE_MULTITARGET_CODE=0`), or a non-`v3`/`v4` CPU — `scatterColumn(use_swwc=true)` runs the
  * direct path. Under the realistic alloc+first-touch-fault model (P2 calibration) SWWC + NT beats the
  * direct scatter only at high per-pass fanout (`P >= 2048`) — see `shouldUseSwwc`.
  *
  * Routing is `part = (hash >> shift) & mask` where `hash` is the stored 32-bit row hash, `shift`
  * selects which bit-window of the hash is used for this pass, and `mask = (1 << pass_bits) - 1`
  * (spec section 4.5). There is no re-hash and no separate count pass; the per-partition output bases
  * are exact-sized from the P1 histogram and must be `64 B`-aligned (required by the NT stores).
  * `ColumnsScatter::scatter` is never used; the only fallback is the direct batched scatter.
  */

/// Build-side reference: which accumulated block, and which row within it (spec section 4.6).
/// Exactly 8 B.
///
/// `row_no` is **0-based**: build row `r` in block `b` is stored as `BuildRef{b, r}`, so payload is
/// resolved directly with `row_no` (no offset). The leaf-cell / chain-tail **empty sentinel** is
/// `row_no == INVALID_ROW` (`0xFFFFFFFF`): the leaf-HT cells and the shared `next_chain` are `memset`
/// to `0xFF`, so a freshly carved entry is the all-`0xFF` ref `{INVALID_ROW, INVALID_ROW}` and can never
/// collide with a real entry (a build block holds at most `2^32 - 1` rows — see the `BuildStore` chassert).
struct BuildRef
{
    UInt32 block_no;
    UInt32 row_no; /// 0-based; INVALID_ROW (0xFFFFFFFF) == empty sentinel
};
static_assert(sizeof(BuildRef) == 8, "BuildRef must be exactly 8 bytes for the 16 B leaf cell");

/// Reserved `row_no` value marking an empty leaf cell / chain tail (cells and `next_chain` are
/// `memset` to `0xFF`). Distinct from every valid 0-based row index.
static constexpr UInt32 INVALID_ROW = 0xFFFFFFFFu;

/// MSB of `block_no` in a head `BuildRef` stored in a leaf cell: when set the key has exactly one
/// build row (no chain). The probe path checks this bit first and skips the `next_chain` load,
/// saving one guaranteed LLC/DRAM miss per probe row for the common unique-key case.
/// `block_no` is stripped of this bit before it is used as an index (payload gather, `leafFlat`).
/// Chain entries in `next_chain` never carry the bit — it lives only in the leaf-cell head.
/// Invariant: the build side asserts `num_blocks < BUILDREF_SINGLETON_BIT` (fail-close).
static constexpr UInt32 BUILDREF_SINGLETON_BIT = 0x80000000u;

/// Whether non-temporal (NT) stores are compiled in AND supported by the current CPU. When false there
/// is no SWWC path at all — `scatterColumn(use_swwc=true)` runs the direct batched scatter (a scalar
/// write-combine would only add a staging copy with no cache-bypass benefit, so it is not offered). NT
/// requires a multitarget build (`ENABLE_MULTITARGET_CODE=1`, which `src/CMakeLists.txt` disables at the
/// `x86-64-v2` baseline) on a `v3`/`v4`-capable CPU. (P2 finding: in the default `x86-64-v2` reldeb
/// build NT is dormant, so `shouldUseSwwc` returns false and the join uses the direct scatter; in a
/// multitarget build the NT path activates automatically.)
bool ntStoresAvailable() noexcept;

/// SWWC engagement rule. Re-measured under the realistic alloc+first-touch-fault model
/// (`bench_radix_sweep_native`, 16 threads) on the `x86-64-v3` multitarget build, where the NT path is
/// genuinely emitted: SWWC + NT beats the direct batched scatter from `P ~= 256` upward (the
/// per-partition outputs no longer stay cache-resident once the fanout reaches a few hundred), so SWWC
/// is engaged iff `ntStoresAvailable()` and `partitions >= 256`; every other case (and the whole
/// `x86-64-v2` build, where NT is dormant) uses the direct path. `num_columns` had the same measured
/// crossover, so a single rule is used.
bool shouldUseSwwc(int num_columns, int partitions) noexcept;

/// x86-64 cache line size; NT stores and SWWC staging flush in whole lines of this many bytes.
constexpr size_t LINE_BYTES = 64;

/// Round `bytes` up to a multiple of `LINE_BYTES` (NT stores write whole lines; per-partition output
/// bases must therefore be `LINE_BYTES`-aligned and have a `LINE_BYTES`-rounded capacity).
inline size_t roundUpTo64(size_t bytes) noexcept
{
    return (bytes + LINE_BYTES - 1) & ~(LINE_BYTES - 1);
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
    /// Per-partition fill of the current write-combining line, in bytes (`0..LINE_BYTES - 1`).
    UInt32 * fill() noexcept { return line_fill.data(); }

private:
    size_t capacity;
    char * staging_buf = nullptr; /// capacity * LINE_BYTES, LINE_BYTES-aligned (one write-combining line per partition).
    std::vector<void *> cursor_ptrs; /// capacity.
    std::vector<UInt32> line_fill; /// capacity.

    void freeStaging() noexcept;
};

/// Bytes written via NT stores during a SWWC scatter (feeds the `RadixHashNTStoreBytes` event, P3).
struct ScatterStats
{
    size_t nt_store_bytes = 0;
};

/** Column-major scatter of one fixed-width column, routed by `part = (hash[j] >> shift) & mask`.
  *
  * `hash`       : `n` stored 32-bit row hashes (one per row; the routing key is a bit-window of it).
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
    const UInt32 * hash,
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
  * column *separately* (column-major), both routed by the same `hash`. `key_width` is the key element
  * width (see `scatterColumn`); the ref column is always `8 B`. `key_out` / `ref_out` are the
  * per-partition bases, each `64 B`-aligned, exact-sized from the histogram. Returns the combined NT
  * store bytes (key + ref).
  */
ScatterStats scatterKeyRefTwoColumn(
    const UInt32 * hash,
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

/** Incremental, direct-path column scatter (no SWWC/NT). Unlike `scatterColumn`, the per-partition
  * write positions live in the caller-owned `cursors` array: `cursors[p]` is partition `p`'s CURRENT
  * write pointer (the caller initialises it to the partition base before the first call), and this
  * call appends its `n` rows and ADVANCES `cursors[p]` in place — it is **not** reset to a base. This
  * lets a large input be scattered in successive small chunks (e.g. 1024 rows) into the same single
  * per-partition allocation without recomputing offsets or a per-chunk histogram. No scratch / staging
  * is needed (direct typed stores only). Returns the bytes written (`n * elem_width`).
  */
size_t scatterColumnInto(
    const UInt32 * hash,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const void * src,
    size_t elem_width,
    size_t partitions,
    void ** cursors);

/** Incremental key + `BuildRef` scatter (column-major, direct path): appends `n` rows to the key and
  * ref outputs whose current write positions are `key_cursors[p]` / `ref_cursors[p]`, advancing both
  * in place (see `scatterColumnInto`). Returns the bytes written (key + ref).
  */
size_t scatterKeyRefInto(
    const UInt32 * hash,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const void * keys,
    size_t key_width,
    const BuildRef * refs,
    size_t partitions,
    void ** key_cursors,
    BuildRef ** ref_cursors);

/** Incremental SWWC + NT column scatter into PERSISTENT per-partition cursors, staging lines and line
  * fills held in `scratch` — the SWWC/NT analogue of `scatterColumnInto`, for scattering a large input
  * in successive small chunks into the same single per-partition allocation. Routes row `j` to
  * `p = (hash[j] >> shift) & mask`.
  *
  * Per partition the write position lives in `scratch.cursors()[p]`. The per-`(thread, partition)`
  * start `base + w_off * elem_width` is generally NOT `LINE_BYTES`-aligned (only the partition base is),
  * but NT stores require a `LINE_BYTES`-aligned destination — so while `scratch.cursors()[p]` is
  * unaligned the rows are written DIRECTLY (head peel); once the cursor reaches a `LINE_BYTES` boundary
  * the rows are staged into `scratch.staging() + p * LINE_BYTES` and full lines are NT-flushed. Output
  * stays contiguous (no gaps), so the per-partition arrays remain densely readable. The caller seeds
  * `scratch.cursors()[p]` to this start and uses a fresh scratch (whose `fill()` is already zeroed)
  * before the first chunk, then calls `scatterColumnDrainSwwc` once after the last chunk.
  *
  * Widths that divide `LINE_BYTES` (`{4,8,16,32}`) tile a staging line; widths that are a multiple of
  * `LINE_BYTES` stream directly (always aligned, no head peel); any other multiple of 4 falls back to
  * the direct incremental path (no NT). Returns the bytes scattered for this call (`n * elem_width`) —
  * the same accounting as `scatterColumnInto` (NOT only the NT-flushed subset), so the join's
  * byte-scattered total is independent of the scatter path. Only engaged when `ntStoresAvailable()`;
  * callers must gate on `shouldUseSwwc`.
  */
size_t scatterColumnIntoSwwc(
    const UInt32 * hash,
    UInt32 shift,
    UInt32 mask,
    size_t n,
    const void * src,
    size_t elem_width,
    size_t partitions,
    ScatterScratch & scratch);

/** Drain the residual (< one `LINE_BYTES` line) staged in each partition of a `scatterColumnIntoSwwc`
  * run: copy `scratch.fill()[p]` bytes from the staging line directly to `scratch.cursors()[p]` (a
  * `LINE_BYTES`-aligned slot in the `roundUpTo64`-sized per-partition array), advance the cursor, then
  * make the run's NT stores globally visible (a streaming fence). Call ONCE per column after all
  * chunks. Only tiled widths ever leave a residual; stream / direct widths drain nothing.
  */
void scatterColumnDrainSwwc(size_t partitions, ScatterScratch & scratch);

}
