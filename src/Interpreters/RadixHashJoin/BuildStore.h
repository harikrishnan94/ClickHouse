#pragma once

#include <Core/Block.h>
#include <Interpreters/RadixHashJoin/GrowingArena.h>
#include <Interpreters/RadixHashJoin/PartitionConfig.h>
#include <Common/RadixShuffle/Scatter.h>
#include <Common/Stopwatch.h>

#include <base/types.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace DB::RadixHash
{

using PackKeyColumnFn = void (*)(const char *, size_t, size_t, char *, size_t, size_t, size_t);

/** Per-leaf dense output of the deferred build scatter (spec section 4.6 step 4), plus the gate
  * statistics the P3 performance gates assert on.
  *
  * Each leaf `L` owns a contiguous, `64 B`-aligned `key` array of `leaf_rows[L]` elements (each
  * `key_width` bytes — the packed multi-column key) and a contiguous `BuildRef` array of the same
  * length, both carved exactly once from `arena` (a non-THP `GrowingArena`). `key_base[L][i]` and
  * `ref_base[L][i]` belong to the same build row: the `BuildRef{block_no,row_no}` resolves the payload
  * back in the accumulated blocks (the leaf HT built from these arrays is phase P4). `row_no` is
  * **1-based** (see `BuildRef`); payload resolution uses `row_no - 1` as the 0-based row index.
  * An empty leaf has a `nullptr` base.
  *
  * Move-only (owns the arena).
  */
struct LeafArrays
{
    size_t num_leaves = 0;
    size_t key_width = 0; /// packed key width (sum of the key column widths)

    std::vector<void *> key_base; /// num_leaves; nullptr for an empty leaf
    std::vector<RadixShuffle::BuildRef *> ref_base; /// num_leaves
    /// Per-leaf 32-bit row-hash arrays, populated only when `scatterToLeaves(..., with_leaf_hash=true)`
    /// (phase P4): `hash_base[L][i]` is the `IColumn::computeHashInto` hash of the build row whose key
    /// is `key_base[L][i]` — the leaf-HT bucket (spec section 5.6) Fibonacci-mixes exactly this value so
    /// the bucket is identical on build and probe. Empty (size 0) when leaf hashes were not requested.
    std::vector<void *> hash_base; /// num_leaves when with_leaf_hash, else empty; UInt32* per leaf
    std::vector<UInt64> leaf_rows; /// num_leaves; == global_hist[L]

    /// Gate statistics (spec section 9.3/9.4).
    std::vector<UInt64> worker_block_counts; /// PB: build blocks scattered by each worker
    UInt64 bytes_scattered = 0; /// ZC: total bytes written by the scatter (key+ref only, summed over passes)
    UInt64 alloc_count = 0; /// NC: number of per-leaf / per-partition output allocations

    GrowingArena arena; /// owns the key/ref memory

    const void * keyAt(size_t leaf, size_t i) const
    {
        return static_cast<const char *>(key_base[leaf]) + i * key_width;
    }
    const RadixShuffle::BuildRef & refAt(size_t leaf, size_t i) const { return ref_base[leaf][i]; }
    UInt32 hashAt(size_t leaf, size_t i) const { return static_cast<const UInt32 *>(hash_base[leaf])[i]; }
};

/** Cooperative pool for the RadixHashJoin post-build phase.
  *
  * N peer threads call run() with the same leader_body lambda.  The first caller becomes the
  * leader and executes leader_body(); the others act as helpers that opportunistically drain
  * parallelFor() work units until the session ends.  Correct for any N >= 1 (N=1 is fully
  * sequential; N>1 is parallel).  No dedicated threads, no sleep.
  *
  * Protocol
  * --------
  *   every thread    -> run(body)            leader executes body(); helpers steal units.
  *   leader only     -> parallelFor(total, fn)  distributes fn over [0, total); blocks until
  *                                              all total units complete.
  *
  * Liveness: the leader never blocks mid-work(), so it always runs to completion and sets
  * session_done even if no helpers show up.  Helpers wait on the condition variable; they
  * never sleep.  Exceptions from the leader or any unit are propagated to all participants.
  */
class CoopPool
{
public:
    CoopPool() = default;
    CoopPool(const CoopPool &) = delete;
    CoopPool & operator=(const CoopPool &) = delete;

    /// Every participating thread calls this with the same body.  The first is the leader and
    /// runs body(); the rest act as helpers.  Late callers (after the session already finished)
    /// return immediately (rethrowing any leader exception).
    void run(std::function<void()> body);

    /// Distribute total work units among the leader and any present helpers.  Called only by
    /// the leader inside body().  Blocks until all total units have completed.  No-op for total == 0.
    void parallelFor(size_t total, std::function<void(size_t)> fn);

private:
    struct Job
    {
        std::function<void(size_t)> fn;
        size_t total = 0;
        std::atomic<size_t> next{0};
        std::atomic<size_t> done{0};
        std::exception_ptr exc; /// first unit exception; protected by CoopPool::mu
    };

    void drainJob(const std::shared_ptr<Job> & job);

    std::mutex mu;
    std::condition_variable cv;
    std::shared_ptr<Job> current_job;    /// non-null while parallelFor is active; guarded by mu
    bool session_done = false;           /// set by leader after body() returns; guarded by mu
    std::exception_ptr leader_exception; /// leader / unit exception to propagate; guarded by mu
    std::atomic<bool> leader_taken{false};
};


/** BuildStore — the radix hash join build path, steps 1+2+4 (spec sections 4.2, 4.6), implemented as
  * a standalone, join-independent unit (the leaf-HT build, step 5, is phase P4; the probe is P5).
  *
  *   - `add` (per build worker, lock-free): COW-move the right block into this worker's store, hash the
  *     join key (one chained `IColumn::computeHashInto` over the — possibly several — fixed-width key
  *     columns) directly into a `uint32` arena span, and accumulate the row counts into the per-thread
  *     replicated histogram (which persists across ALL of the thread's blocks). No scatter, no payload
  *     copy, no per-block histogram (spec invariants 1, 2).
  *   - `finishBuild` (single barrier): move-concat the per-worker stores (assigning final `block_no`s),
  *     fold each thread's replicated histogram into `global_hist`, and compute its exclusive prefix sum
  *     `offset`. Records the per-slot contiguous block ranges used by the scatter.
  *   - `scatterToLeaves`: allocate each leaf's key/ref array exactly once (`64 B`-aligned, exact-sized)
  *     from a `GrowingArena` (spec invariant 3), then run the deferred static-per-thread scatter
  *     of `key + BuildRef` on the **caller-provided** `ThreadPool` (the query pool, plan D5). Each
  *     build thread owns its own contiguous block range and seeds its write cursors once from a
  *     per-`(thread,partition)` offset matrix (one prefix-sum across threads) — fully lock-free.
  *     Multi-pass builds route through pass-0 partitions carrying the `uint32` row hash, with
  *     incremental `MADV_DONTNEED` release of consumed intermediate partitions. `num_threads` governs
  *     the depth-first refine work-steal parallelism (P4+).
  *
  * Thread-slot handout: each distinct thread that calls `add` is bound (once) to a `LocalBuildState`
  * slot via a thread-local cache keyed on a unique instance id + an atomic counter; more than
  * `max_threads` distinct build threads is a fail-close error (never silent corruption).
  */
class BuildStore
{
public:
    /// Rows are scattered in chunks of this many rows (bounds the temporary packed-key buffer).
    static constexpr size_t SCATTER_CHUNK_ROWS = 1024;

    /// `key_positions` / `key_widths`: the positions and byte widths of the fixed-width join key
    /// columns inside every build block (one entry for a single-column key, several for a composite
    /// key). The packed key width (their sum) must be a multiple of 4. `arena_max_block` caps the
    /// growing arena block size.
    BuildStore(
        PartitionConfig cfg_,
        std::vector<size_t> key_positions_,
        std::vector<size_t> key_widths_,
        size_t max_threads_,
        size_t arena_max_block_ = GrowingArena::DEFAULT_MAX_BLOCK);
    ~BuildStore();

    BuildStore(const BuildStore &) = delete;
    BuildStore & operator=(const BuildStore &) = delete;

    /// Step 1 (per worker, lock-free).
    void add(const Block & block);

    /// Step 2 (single barrier). Must be called once, after all `add`s, before `scatterToLeaves`.
    void finishBuild();

    /// Step 4. Deferred exact key+ref scatter into per-leaf arrays, parallelised via `coord`.
    /// When `with_leaf_hash` is true (phase P4), the per-row 32-bit routing hash is also scattered into
    /// per-leaf `hash_base` arrays (one `UInt32` per build row, aligned with key/ref), so the leaf-HT
    /// build can Fibonacci-mix the exact same hash the probe side derives (spec section 5.6, 15.9).
    /// Must be called only by the leader inside a CoopPool::run body.
    LeafArrays scatterToLeaves(CoopPool & coord, bool with_leaf_hash = false);

    const PartitionConfig & config() const { return cfg; }
    size_t packedKeyWidth() const { return key_width; }
    /// Accumulated build blocks in final `block_no` order (the payload source for P4 / tests).
    const std::vector<Block> & blocks() const { return global_blocks; }
    const std::vector<UInt64> & globalHistogram() const { return global_hist; }
    const std::vector<UInt64> & offsets() const { return offset; }
    size_t numBlocks() const { return global_blocks.size(); }
    size_t totalRows() const { return total_rows; }

    /// Exclusive per-block row offset (block_base[b] = Σ rows of blocks 0..b-1), so a build row's flat
    /// index across all blocks is `block_base[ref.block_no] + ref.row_no - 1` — the 1D mapping used by
    /// the leaf-HT `next_chain` (phase P4, spec section 5.6). Size numBlocks()+1; back() == totalRows().
    const std::vector<UInt64> & blockBase() const { return block_base; }

private:
    /// Per-build-worker state (one slot per concurrent `add` thread). Move-only; owned via unique_ptr
    /// so the hash arena pointers stored in `hash_of_block` stay stable.
    struct LocalBuildState
    {
        /// `num_leaves` is used to size the replicated histogram (see BuildStore.cpp: chooseReplicas).
        LocalBuildState(size_t num_leaves, size_t arena_max_block_);

        std::vector<Block> blocks;
        GrowingArena hash_arena;                       /// per-row 32-bit hashes for each added block
        std::vector<const UInt32 *> hash_of_block;     /// one arena span per block, n entries each
        std::vector<UInt32> rows_of_block;

        /// Replicated histogram: `replicas` copies of the `num_leaves` counters, round-robined per
        /// row to avoid store-to-load-forwarding stalls (spec section 4.2, anti-stall note).
        /// Accumulates across ALL of the thread's blocks (not reset between blocks); the replicas
        /// are folded into the global histogram once in `finishBuild()`.
        size_t replicas = 1;
        std::vector<UInt32> rep_hist; /// replicas * num_leaves counters (accumulated across all blocks)
    };

    size_t workerSlot();
    void packKeyChunk(const Block & block, size_t row_begin, size_t rows, char * dst) const;

    /// Initialise a LeafArrays ready for population (worker_block_counts sized to used_slots,
    /// key/ref/leaf_rows zeroed to num_leaves, arena attached). Setup only — not on the O(N) path.
    /// When `with_leaf_hash`, hash_base is also sized to num_leaves (per-leaf UInt32 hash arrays).
    LeafArrays makeLeafArrays(bool with_leaf_hash) const;

    /// Record the scatter ProfileEvents and trim the output arena tail.
    void finalizeScatter(LeafArrays & out, const Stopwatch & sw, std::atomic<UInt64> & total_bytes, size_t num_passes) const;

    /// Per-worker scratch for depth-first multi-pass refinement. Pre-sized once per worker (max fanout
    /// across refine passes), then reused throughout all recursive calls for that worker.
    struct RefineWorkerScratch
    {
        explicit RefineWorkerScratch(size_t max_fanout);
        RadixShuffle::ScatterScratch scratch;
        std::vector<void *> kout;
        std::vector<RadixShuffle::BuildRef *> rout;
        std::vector<void *> pout;
        UInt64 local_bytes = 0;
    };

    /// Depth-first recursive refinement of one pass-0 partition all the way to its final leaves.
    /// `global_first_leaf` is the leaf index of the first leaf in this partition's subtree. At the last
    /// pass the scattered key+ref land in the pre-allocated `out.key_base/ref_base[leaf]` directly;
    /// intermediate passes allocate a per-call RAII `GrowingArena` (freed on return -> lowest peak
    /// intermediate memory), scatter key+ref+hash into children, then recurse depth-first.
    /// `with_leaf_hash`: at the last pass, also scatter the carried row hashes into `out.hash_base`.
    void refineDepthFirst(
        size_t global_first_leaf,
        const void * in_keys,
        const RadixShuffle::BuildRef * in_refs,
        const UInt32 * in_hashes,
        UInt64 rows,
        size_t pass_index,
        UInt32 bits_consumed,
        LeafArrays & out,
        const std::vector<UInt64> & gh_prefix,
        RefineWorkerScratch & ws,
        bool with_leaf_hash);

    /// Worker kernel: scatter key + BuildRef (and, iff carry_hash, the uint32 row hash) for all build
    /// blocks into `num_parts` partitions using static per-thread ownership. Each build thread (slot)
    /// scatters its own contiguous block range into disjoint per-partition sub-regions, seeding its
    /// write cursors once from `thr_off[worker_id * num_parts + part]` (no per-block cursor reseeding).
    /// Single-pass calls this with carry_hash=false (hash scatter compiled out entirely); multi-pass
    /// pass-0 uses carry_hash=true.
    template <bool carry_hash>
    void scatterBlocksIntoPartitions(
        CoopPool & coord,
        size_t num_parts,
        UInt32 shift,
        UInt32 mask,
        const std::vector<UInt64> & thr_off, /// flat [used_slots x num_parts]: per-thread start offsets within each partition output
        void * const * key_base_arr,
        RadixShuffle::BuildRef * const * ref_base_arr,
        void * const * hash_base_arr,
        std::atomic<UInt64> & total_bytes,
        std::vector<UInt64> & worker_counts);

    LeafArrays scatterSinglePass(CoopPool & coord, bool with_leaf_hash);
    LeafArrays scatterMultiPass(CoopPool & coord, bool with_leaf_hash);

    PartitionConfig cfg;
    std::vector<size_t> key_positions;
    std::vector<size_t> key_widths;
    std::vector<size_t> key_offsets; /// prefix sums of key_widths (byte offset of each col in a packed row)
    std::vector<PackKeyColumnFn> key_packers; /// one width-specialized packer per key column
    size_t key_width; /// packed key width (sum of key_widths)
    size_t max_threads;
    size_t arena_max_block;

    /// Unique non-zero id of this instance, so the per-thread worker-slot cache cannot be confused by a
    /// later BuildStore reusing the same heap address (the raw `this` pointer is not enough).
    UInt64 instance_id;

    std::vector<std::unique_ptr<LocalBuildState>> local;
    std::atomic<size_t> next_slot{0};

    /// Filled by finishBuild().
    bool finished = false;
    std::vector<Block> global_blocks;
    std::vector<const UInt32 *> global_hash_of_block;  /// per-block row-hash arrays (in block_no order)
    std::vector<UInt32> global_rows_of_block;
    std::vector<UInt64> global_hist; /// num_leaves
    std::vector<UInt64> offset; /// num_leaves, exclusive prefix sum of global_hist
    UInt64 total_rows = 0; /// Σ rows over all build blocks (== Σ global_hist)
    std::vector<UInt64> block_base; /// numBlocks()+1, exclusive prefix sum of per-block row counts

    /// Per-slot static scatter ownership (set by finishBuild).
    /// `used_slots[w]` = index of the w-th active LocalBuildState slot in concatenation order.
    /// `thread_block_begin[w]` / `thread_block_end[w]` = contiguous range in `global_blocks` for slot w.
    std::vector<size_t> used_slots;
    std::vector<size_t> thread_block_begin;
    std::vector<size_t> thread_block_end;
};

}
