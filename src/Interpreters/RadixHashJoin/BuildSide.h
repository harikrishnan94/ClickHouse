#pragma once

#include <Core/Block.h>
#include <Interpreters/RadixHashJoin/Arena.h>
#include <Interpreters/RadixHashJoin/KeyLayout.h>
#include <Interpreters/RadixHashJoin/KeyRefScatter.h>
#include <Interpreters/RadixHashJoin/ParallelFor.h>
#include <Interpreters/RadixHashJoin/PartitionPlan.h>

#include <base/types.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

namespace DB::RadixJoin
{

/// Byte offset of the packed key within a fused scatter record `[ BuildRef | key ]` (ref-first, matching
/// the leaf cell layout `[ BuildRefList word | key ]`).
inline constexpr size_t PACKED_KEY_OFFSET_IN_RECORD = sizeof(BuildRef);

/** The per-leaf output of the deferred scatter: for each leaf, a dense fused-record array
  * (`leaf_rows[L]` elements of `record_width` bytes — ref-first `[ BuildRef | packed key ]`),
  * carved exactly once and 64-byte-aligned. `keyAt(L,i)` and `refAt(L,i)` address the two sub-fields.
  * An empty leaf has a null base. The leaf hash table (LeafTable) is built from these; afterwards the
  * arrays are dropped. Move-only (owns its arena).
  */
struct LeafArrays
{
    size_t num_leaves = 0;
    size_t key_width = 0;
    size_t record_width = 0; /// key_width + sizeof(BuildRef)

    std::vector<void *> record_base; /// num_leaves; null for an empty leaf
    std::vector<UInt64> leaf_rows;   /// num_leaves; == global histogram

    /// num_leaves; per-leaf HLL distinct-key estimate (always clamped to `leaf_rows`, so it can only
    /// shrink a leaf table). Empty when distinct-estimate sizing is disabled, in which case the leaf
    /// hash tables fall back to row-count sizing.
    std::vector<UInt64> distinct_key_estimates;

    /// Diagnostics asserted by the unit tests / gates.
    UInt64 alloc_count = 0;          /// number of per-partition output allocations (no-churn gate)
    UInt64 bytes_scattered = 0;      /// total record bytes written, summed over passes

    Arena arena;                     /// owns the fused-record memory

    const void * keyAt(size_t leaf, size_t i) const
    {
        return static_cast<const char *>(record_base[leaf]) + i * record_width + PACKED_KEY_OFFSET_IN_RECORD;
    }

    const BuildRef & refAt(size_t leaf, size_t i) const
    {
        return *reinterpret_cast<const BuildRef *>( /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            static_cast<const char *>(record_base[leaf]) + i * record_width);
    }
};

/** The build side: accumulate right blocks (zero copy), count rows per leaf, then scatter the fused
  * `[ BuildRef | packed key ]` of every row into the per-leaf arrays. Three phases:
  *
  *   add(block, lane)   per build lane, lock-free. COW-move the block into this lane's store, and
  *                      route each row by recomputing its packed-key hash (the route word = the high
  *                      32 bits) into a per-lane replicated histogram. The hash is NOT stored per
  *                      row — it is recomputed in the scatter and again in the leaf-HT build. This
  *                      trades a little compute for ~N*4 bytes of saved build memory and is why the
  *                      build copies no payload at all.
  *
  *   finishBuild()      single barrier. Concatenate the per-worker block stores (assigning final
  *                      block_no), fold the per-worker histograms into one global histogram, and
  *                      prefix-sum it. Records each worker's contiguous block range for the scatter.
  *
  *   scatterToLeaves()  allocate each leaf's fused-record array EXACTLY ONCE from `global_hist` (the
  *                      no-churn property) and scatter into them, parallelised over the caller's
  *                      `ParallelFor`. One pass when the leaf count fits the per-pass fanout cap,
  *                      otherwise a depth-first multi-pass radix that frees each intermediate as it
  *                      is consumed.
  *
  * Build lanes: each call to `add` carries a stable 0-based build-lane index (from the pipeline's
  * per-lane FillingRightJoinSideTransform) that selects this lane's LocalState slot directly — no
  * thread-local cache, no atomic counter. A lane index >= `max_threads` is a fail-close error, never
  * silent corruption.
  */
class BuildSide
{
public:
    /// Rows are scattered in chunks of this size (bounds the packed-key + route + fused-record scratch).
    static constexpr size_t SCATTER_CHUNK_ROWS = 1024;

    /// Per-leaf staging depth for the windowed multipass direct scatter (see scatterDirectFromColumnKeysFixed).
    static constexpr size_t SCATTER_WINDOW_SLOTS = 64;
    /// Above this leaf fanout the direct scatter uses a window buffer instead of per-leaf live cursors.
    static constexpr size_t SCATTER_MULTIPASS_PART_THRESHOLD = 1024;
    /// Coarse fanout for pass-1 window staging in the direct scatter (pass 2 uses full leaf fanout).
    static constexpr size_t SCATTER_MAX_PARTS_PER_PASS = 256;

    BuildSide(PartitionPlan plan_, std::vector<size_t> key_positions_, std::vector<size_t> key_widths_, size_t max_threads_);
    ~BuildSide();

    BuildSide(const BuildSide &) = delete;
    BuildSide & operator=(const BuildSide &) = delete;

    void add(const Block & block, size_t lane); /// phase 1 (per build lane, lock-free)
    void finishBuild();                     /// phase 2 (single barrier)

    /// phase 3 (parallelised post-build). `num_workers` is the dense worker-id space of `parallel_for`,
    /// used to size the per-worker HLL partial sketches. When `estimate_distinct_keys` is set, a per-worker
    /// HLL sketch of each leaf's keys is accumulated on the FINAL scatter pass (lock-free, single-writer per
    /// worker), then merged into `LeafArrays::distinct_key_estimates`; otherwise that field is left empty.
    LeafArrays scatterToLeaves(const ParallelFor & parallel_for, size_t num_workers, bool estimate_distinct_keys);

    LeafArrays scatterToLeaves2(const ParallelFor & parallel_for, size_t num_workers, bool = false);

    const PartitionPlan & plan() const { return part_plan; }
    size_t packedKeyWidth() const { return key_width; }
    size_t recordWidth() const { return record_width; }
    const std::vector<Block> & blocks() const { return all_blocks; }
    const std::vector<UInt64> & globalHistogram() const { return global_hist; }
    size_t numBlocks() const { return all_blocks.size(); }
    UInt64 totalRows() const { return total_rows; }

    /// Exclusive per-block row offset: flat index of a build row = block_base[block_no] + row_no.
    /// Used by the leaf hash table's shared chain array. Size numBlocks()+1; back() == totalRows().
    const std::vector<UInt64> & blockBase() const { return block_base; }

private:
    struct LocalState
    {
        explicit LocalState(size_t num_leaves);

        std::vector<Block> blocks;
        std::vector<UInt32> route_scratch; /// reused per-block route-word buffer (not stored per row)
        std::vector<char> pack_scratch;    /// reused multi-column packed-key chunk buffer
        std::vector<UInt32> rows_of_block;

        /// Replicated histogram: `replicas` copies of the `num_leaves` counters, round-robined per row
        /// so consecutive increments hit different counters and dodge store-to-load-forwarding stalls.
        /// Accumulated across ALL of this worker's blocks; folded into the global histogram once.
        size_t replicas = 1;
        std::vector<UInt32> rep_hist;
    };

    void packKeyChunk(const Block & block, size_t row_begin, size_t rows, char * dst) const;

    LeafArrays makeLeafArrays() const;
    LeafArrays scatterSinglePass(const ParallelFor & parallel_for);
    LeafArrays scatterMultiPass(const ParallelFor & parallel_for);

    /// The shared scatter worker body: each build-thread slot scatters its own contiguous block range
    /// into `num_parts` partitions, seeding its write cursors once from a per-(slot, partition) offset
    /// matrix (so the writes are disjoint and lock-free). Used by single-pass (leaves) and pass-0.
    /// `accumulate_hll` is set only when this call is the FINAL leaf-writing pass (single-pass scatter) and
    /// distinct-estimate sizing is on; pass-0 of a multi-pass scatter (partitions, not leaves) leaves it off.
    void scatterBlockRanges(
        const ParallelFor & parallel_for,
        size_t num_parts,
        UInt32 shift,
        UInt32 mask,
        const std::vector<UInt64> & slot_part_offset,
        void * const * record_bases,
        std::atomic<UInt64> & total_bytes,
        bool accumulate_hll);

    void scatterDirectFromColumnKeys(
        const ParallelFor & parallel_for,
        size_t num_used,
        size_t num_parts,
        UInt32 shift,
        UInt32 mask,
        const std::vector<UInt64> & slot_off,
        void * const * record_bases,
        std::atomic<UInt64> & total_bytes) const;

    template <size_t key_width, bool multi_col, bool two_pass>
    void scatterDirectFromColumnKeysFixed(
        const ParallelFor & parallel_for,
        size_t num_used,
        size_t num_parts,
        UInt32 shift,
        UInt32 mask,
        const std::vector<UInt64> & slot_off,
        void * const * record_bases,
        std::atomic<UInt64> & total_bytes) const;

    template <bool multi_col, bool two_pass>
    void dispatchScatterDirectFromColumnKeysFixed(
        const ParallelFor & parallel_for,
        size_t num_used,
        size_t num_parts,
        UInt32 shift,
        UInt32 mask,
        const std::vector<UInt64> & slot_off,
        void * const * record_bases,
        std::atomic<UInt64> & total_bytes) const;

    /// Depth-first refinement of one already-scattered partition down to its leaves (multi-pass only).
    /// `worker` is the dense worker id of the enclosing scatter unit, used to key this subtree's HLL
    /// partial sketch on the final pass (constant for the whole recursion).
    struct RefineScratch;
    void refine(
        size_t first_leaf,
        const void * in_records,
        UInt64 rows,
        size_t pass_index,
        UInt32 bits_consumed,
        LeafArrays & out,
        const std::vector<UInt64> & hist_prefix,
        RefineScratch & scratch,
        UInt64 & local_bytes,
        size_t worker);

    PartitionPlan part_plan;
    std::vector<size_t> key_positions;
    std::vector<size_t> key_widths;
    std::vector<size_t> key_offsets;        /// byte offset of each key column within a packed row
    std::vector<ColumnPackFn> key_packers;  /// one width-specialized packer per key column
    size_t key_width = 0;
    size_t record_width = 0;
    size_t max_threads = 1;

    /// Transient per-worker HLL partial sketches, owned by a local in `scatterToLeaves` and only non-null
    /// for the duration of that call; the scatter passes consult it to accumulate on the final pass.
    struct HllScatterState;
    HllScatterState * hll_scatter = nullptr;

    std::vector<std::unique_ptr<LocalState>> local;

    bool finished = false;
    std::vector<Block> all_blocks;
    std::vector<UInt32> rows_per_block;
    std::vector<UInt64> global_hist;        /// num_leaves
    UInt64 total_rows = 0;
    std::vector<UInt64> block_base;         /// numBlocks()+1, exclusive prefix sum of rows_per_block

    /// Each used slot's contiguous range in `all_blocks` (set by finishBuild) — the unit of the scatter.
    std::vector<size_t> used_slots;
    std::vector<size_t> slot_block_begin;
    std::vector<size_t> slot_block_end;
};

}
