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

/** The per-leaf output of the deferred scatter: for each leaf, a dense `key` array (`leaf_rows[L]`
  * elements of `key_width` bytes — the packed key) and a parallel `ref` array (one `BuildRef` each),
  * both carved exactly once and 64-byte-aligned. `key[L][i]` and `ref[L][i]` are the same build row.
  * An empty leaf has null bases. The leaf hash table (LeafTable) is built from these; afterwards the
  * arrays are dropped. Move-only (owns its arena).
  */
struct LeafArrays
{
    size_t num_leaves = 0;
    size_t key_width = 0;

    std::vector<void *> key_base;    /// num_leaves; null for an empty leaf
    std::vector<BuildRef *> ref_base; /// num_leaves
    std::vector<UInt64> leaf_rows;   /// num_leaves; == global histogram

    /// Diagnostics asserted by the unit tests / gates.
    UInt64 alloc_count = 0;          /// number of per-partition output allocations (no-churn gate)
    UInt64 bytes_scattered = 0;      /// total key+ref bytes written, summed over passes

    Arena arena;                     /// owns the key/ref memory

    const void * keyAt(size_t leaf, size_t i) const
    {
        return static_cast<const char *>(key_base[leaf]) + i * key_width;
    }
};

/** The build side: accumulate right blocks (zero copy), count rows per leaf, then scatter the key +
  * BuildRef of every row into the per-leaf arrays. Three phases:
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
  *   scatterToLeaves()  allocate each leaf's key/ref array EXACTLY ONCE from `global_hist` (the
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
    /// Rows are scattered in chunks of this size (bounds the packed-key + route scratch).
    static constexpr size_t SCATTER_CHUNK_ROWS = 1024;

    BuildSide(PartitionPlan plan_, std::vector<size_t> key_positions_, std::vector<size_t> key_widths_, size_t max_threads_);
    ~BuildSide();

    BuildSide(const BuildSide &) = delete;
    BuildSide & operator=(const BuildSide &) = delete;

    void add(const Block & block, size_t lane); /// phase 1 (per build lane, lock-free)
    void finishBuild();                     /// phase 2 (single barrier)
    LeafArrays scatterToLeaves(const ParallelFor & parallel_for); /// phase 3 (parallelised post-build)

    const PartitionPlan & plan() const { return part_plan; }
    size_t packedKeyWidth() const { return key_width; }
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
    void scatterBlockRanges(
        const ParallelFor & parallel_for,
        size_t num_parts,
        UInt32 shift,
        UInt32 mask,
        const std::vector<UInt64> & slot_part_offset,
        void * const * key_bases,
        BuildRef * const * ref_bases,
        std::atomic<UInt64> & total_bytes);

    /// Depth-first refinement of one already-scattered partition down to its leaves (multi-pass only).
    struct RefineScratch;
    void refine(
        size_t first_leaf,
        const void * in_keys,
        const BuildRef * in_refs,
        UInt64 rows,
        size_t pass_index,
        UInt32 bits_consumed,
        LeafArrays & out,
        const std::vector<UInt64> & hist_prefix,
        RefineScratch & scratch,
        UInt64 & local_bytes);

    PartitionPlan part_plan;
    std::vector<size_t> key_positions;
    std::vector<size_t> key_widths;
    std::vector<size_t> key_offsets;        /// byte offset of each key column within a packed row
    std::vector<ColumnPackFn> key_packers;  /// one width-specialized packer per key column
    size_t key_width = 0;
    size_t max_threads = 1;

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
