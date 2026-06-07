#include <Interpreters/RadixHashJoin/BuildStore.h>

#include <Common/Exception.h>
#include <Common/ProfileEvents.h>

#include <Columns/IColumn.h>

#include <algorithm>
#include <bit>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>

namespace ProfileEvents
{
extern const Event RadixHashBuildSelectMicroseconds;
extern const Event RadixHashBuildScatterMicroseconds;
extern const Event RadixHashScatterRows;
extern const Event RadixHashBuildBlocksMoved;
}

namespace DB
{
namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}
}

namespace DB::RadixHash
{

namespace
{

using RadixShuffle::BuildRef;
using RadixShuffle::LINE_BYTES;
using RadixShuffle::roundUpTo64;
using RadixShuffle::ScatterScratch;
using RadixShuffle::scatterColumn;
using RadixShuffle::scatterColumnDrainSwwc;
using RadixShuffle::scatterColumnInto;
using RadixShuffle::scatterColumnIntoSwwc;
using RadixShuffle::scatterKeyRefInto;
using RadixShuffle::scatterKeyRefTwoColumn;
using RadixShuffle::shouldUseSwwc;

/// Replicated-histogram replica count: round-robin per row avoids store-to-load-forwarding stalls.
/// Clamped so replicas * num_leaves * 4 B <= ~L1 (32 KiB). Power of two for a cheap mask.
size_t chooseReplicas(size_t num_leaves)
{
    const size_t fit = num_leaves != 0 ? (32 * 1024 / (num_leaves * sizeof(UInt32))) : 1;
    return std::bit_floor(std::min<size_t>(4, std::max<size_t>(1, fit)));
}

UInt64 nextInstanceId()
{
    static std::atomic<UInt64> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

/// Key packing kernels. Common widths are compile-time (direct typed stores); any other multiple of 4
/// uses 4-byte lanes. The width dispatch is hoisted to construction time (BuildStore::key_packers).
template <size_t width>
void packKeyColumnT(const char * src, size_t row_begin, size_t rows, char * dst, size_t stride, size_t dst_offset, size_t)
{
    static_assert(width >= 4 && width % 4 == 0);
    for (size_t r = 0; r < rows; ++r)
        __builtin_memcpy_inline(dst + r * stride + dst_offset, src + (row_begin + r) * width, width);
}

void packKeyColumnGeneric(
    const char * src, size_t row_begin, size_t rows, char * dst, size_t stride, size_t dst_offset, size_t width)
{
    chassert(width >= 4 && width % 4 == 0);
    for (size_t r = 0; r < rows; ++r)
    {
        const char * s = src + (row_begin + r) * width;
        char * d = dst + r * stride + dst_offset;
        for (size_t b = 0; b < width; b += 4)
            __builtin_memcpy_inline(d + b, s + b, 4);
    }
}

PackKeyColumnFn chooseKeyPacker(size_t width)
{
    switch (width)
    {
        case 4:  return &packKeyColumnT<4>;
        case 8:  return &packKeyColumnT<8>;
        case 16: return &packKeyColumnT<16>;
        case 32: return &packKeyColumnT<32>;
        case 64: return &packKeyColumnT<64>;
        default: return &packKeyColumnGeneric;
    }
}

/// Intermediate cascade level for the multi-pass scatter (key + BuildRef + carried uint32 row hash,
/// one dense array per partition). Keys and hashes are void* so they match LeafArrays and the
/// unified template worker; cast to typed pointers at the call sites.
struct CascadeLevel
{
    size_t num_parts = 0;
    GrowingArena arena;
    std::vector<void *> key;
    std::vector<BuildRef *> ref;
    std::vector<void *> hash; /// UInt32* stored as void* (cast when reading in refine pass)
    std::vector<UInt64> count;
};

/// Per-partition result type from allocExactPartitions.
struct PartitionArrays
{
    std::vector<void *> key;
    std::vector<BuildRef *> ref;
    std::vector<void *> hash; /// populated onum_leavesy when carry_hash == true
    UInt64 alloc_count = 0;
};

/// Allocate exactly-sized, 64 B-aligned key/ref/(hash) arrays for every non-empty partition from
/// `arena`. This is the single place that does per-leaf/per-partition allocation (O(num_parts) carves),
/// used by scatterSinglePass (leaves), level0, and every refine level.
///
/// All three sections are packed into one contiguous alloc per partition:
///   [ key_section | ref_section | hash_section (if carry_hash) ]
/// Each section is roundUpTo64 bytes; sub-pointers are naturally 64 B-aligned since the base is
/// LINE_BYTES-aligned and each section size is a multiple of LINE_BYTES. This halves the number
/// of arena alloc calls (1 instead of 2-3) and allows the incremental release in refine passes to
/// issue a single madvise(MADV_DONTNEED) over the whole combined block.
PartitionArrays allocExactPartitions(
    GrowingArena & arena,
    std::span<const UInt64> counts,
    size_t kw,
    bool carry_hash)
{
    const size_t num_parts = counts.size();
    PartitionArrays out;
    out.key.assign(num_parts, nullptr);
    out.ref.assign(num_parts, nullptr);
    if (carry_hash)
        out.hash.assign(num_parts, nullptr);

    for (size_t part = 0; part < num_parts; ++part)
    {
        if (counts[part] == 0)
            continue;
        const size_t key_bytes  = roundUpTo64(counts[part] * kw);
        const size_t ref_bytes  = roundUpTo64(counts[part] * sizeof(BuildRef));
        const size_t hash_bytes = carry_hash ? roundUpTo64(counts[part] * sizeof(UInt32)) : 0;
        char * base = static_cast<char *>(arena.alloc(key_bytes + ref_bytes + hash_bytes, LINE_BYTES));
        out.key[part] = base;
        out.ref[part] = reinterpret_cast<BuildRef *>(base + key_bytes);
        if (carry_hash)
            out.hash[part] = base + key_bytes + ref_bytes;
        ++out.alloc_count; /// one combined alloc per non-empty partition
    }
    return out;
}

} /// anonymous namespace


// ---- CoopPool implementation ---------------------------------------------------------------

void CoopPool::drainJob(const std::shared_ptr<Job> & job)
{
    while (true)
    {
        const size_t idx = job->next.fetch_add(1, std::memory_order_relaxed);
        if (idx >= job->total)
            break;
        try
        {
            job->fn(idx);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lk(mu);
            if (!job->exc)
                job->exc = std::current_exception();
        }
        /// Whoever increments done to total notifies the leader waiting in parallelFor.
        if (job->done.fetch_add(1, std::memory_order_acq_rel) + 1 == job->total)
            cv.notify_all();
    }
}

void CoopPool::parallelFor(size_t total, std::function<void(size_t)> fn)
{
    if (total == 0)
        return;

    auto job = std::make_shared<Job>();
    job->fn = std::move(fn);
    job->total = total;

    {
        std::lock_guard<std::mutex> lk(mu);
        current_job = job;
    }
    cv.notify_all(); /// wake helpers

    drainJob(job); /// leader also drains units

    /// Wait until all units have completed.
    {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return job->done.load(std::memory_order_acquire) >= total; });
        current_job = nullptr;
    }

    if (job->exc)
        std::rethrow_exception(job->exc);
}

void CoopPool::run(std::function<void()> body)
{
    bool expected = false;
    if (leader_taken.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    {
        /// Leader: execute body() (which issues parallelFor calls), then close the session.
        std::exception_ptr exc;
        try
        {
            body();
        }
        catch (...)
        {
            exc = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lk(mu);
            leader_exception = exc;
            session_done = true;
        }
        cv.notify_all();

        if (exc)
            std::rethrow_exception(exc);
    }
    else
    {
        /// Helper (also covers late callers after session_done is already set).
        while (true)
        {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this] { return current_job != nullptr || session_done; });
                if (session_done)
                {
                    if (leader_exception)
                        std::rethrow_exception(leader_exception);
                    return;
                }
                job = current_job;
            }
            drainJob(job);
        }
    }
}

// --------------------------------------------------------------------------------------------

BuildStore::LocalBuildState::LocalBuildState(size_t num_leaves_, size_t arena_max_block_)
    : hash_arena(arena_max_block_)
    , replicas(chooseReplicas(num_leaves_))
    , rep_hist(replicas * num_leaves_, 0)
{
}


BuildStore::BuildStore(
    PartitionConfig cfg_,
    std::vector<size_t> key_positions_,
    std::vector<size_t> key_widths_,
    size_t max_threads_,
    size_t arena_max_block_)
    : cfg(std::move(cfg_))
    , key_positions(std::move(key_positions_))
    , key_widths(std::move(key_widths_))
    , max_threads(std::max<size_t>(max_threads_, 1))
    , arena_max_block(arena_max_block_)
    , instance_id(nextInstanceId())
{
    chassert(!key_positions.empty() && key_positions.size() == key_widths.size());

    key_offsets.resize(key_widths.size());
    key_packers.resize(key_widths.size());
    size_t acc = 0;
    for (size_t col = 0; col < key_widths.size(); ++col)
    {
        key_offsets[col] = acc;
        key_packers[col] = chooseKeyPacker(key_widths[col]);
        acc += key_widths[col];
    }
    key_width = acc;

    local.reserve(max_threads);
    for (size_t slot = 0; slot < max_threads; ++slot)
        local.push_back(std::make_unique<LocalBuildState>(cfg.num_leaves, arena_max_block));
}

BuildStore::~BuildStore() = default;


size_t BuildStore::workerSlot()
{
    /// Keyed on the unique instance id (not raw `this`) so pooled threads cannot hit a stale slot
    /// from a previous BuildStore at the same address. The cache holds one entry, which suits the
    /// join (one live BuildStore per query); a thread interleaving across two live instances
    /// re-fetches a slot each switch — fail-close throw, never silent corruption.
    struct SlotCache
    {
        UInt64 owner_id = 0;
        size_t slot = 0;
    };
    thread_local SlotCache cache;

    if (cache.owner_id == instance_id)
        return cache.slot;

    const size_t s = next_slot.fetch_add(1, std::memory_order_relaxed);
    if (s >= max_threads)
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "RadixHashJoin BuildStore: more distinct build threads ({}) than max_threads ({})",
            s + 1,
            max_threads);

    cache.owner_id = instance_id;
    cache.slot = s;
    return s;
}


void BuildStore::packKeyChunk(const Block & block, size_t row_begin, size_t rows, char * dst) const
{
    /// Row-major pack: row r holds the concatenated key column values at stride `key_width`.
    /// Single-column callers skip this entirely and scatter the column's raw data directly.
    for (size_t col = 0; col < key_positions.size(); ++col)
    {
        const char * column_data = block.getByPosition(key_positions[col]).column->getRawData().data();
        key_packers[col](column_data, row_begin, rows, dst, key_width, key_offsets[col], key_widths[col]);
    }
}


void BuildStore::add(const Block & block)
{
    const size_t slot = workerSlot();
    LocalBuildState & state = *local[slot];

    /// (1) ZERO COPY: COW shared_ptr move — no column data is copied.
    Block kept = block;
    const size_t n = kept.rows();
    chassert(n <= std::numeric_limits<UInt32>::max()); /// BuildRef.row_no is 32-bit and 1-based (0 is reserved as the empty sentinel)

    Stopwatch sw;

    /// (2) Allocate the hash-arena span first, then hash all key columns directly into it —
    /// one pass, zero copies. `computeHashInto` writes its output to whatever pointer is given.
    UInt32 * row_hash = nullptr;
    if (n > 0)
    {
        row_hash = state.hash_arena.allocArray<UInt32>(n);
        for (size_t col = 0; col < key_positions.size(); ++col)
            kept.getByPosition(key_positions[col]).column->computeHashInto(0, n, row_hash, /*initial=*/col == 0);
    }

    /// (3) Accumulate this block's rows into the per-thread replicated histogram.
    /// The histogram persists across ALL of the thread's blocks; replicas are folded into
    /// `global_hist` once in `finishBuild()`. No per-block sum-and-zero needed here.
    const size_t num_leaves = cfg.num_leaves;
    const UInt32 safe_shift = cfg.total_bits > 0 ? cfg.shift : 0u; /// guard shift==32 UB when num_leaves==1
    const UInt32 leaf_mask = static_cast<UInt32>(num_leaves - 1);
    const size_t replica_mask = state.replicas - 1;
    UInt32 * rep_hist_data = state.rep_hist.data();
    for (size_t row = 0; row < n; ++row)
        ++rep_hist_data[(row & replica_mask) * num_leaves + ((row_hash[row] >> safe_shift) & leaf_mask)];

    ProfileEvents::increment(ProfileEvents::RadixHashBuildSelectMicroseconds, sw.elapsedMicroseconds());

    state.blocks.push_back(std::move(kept));
    state.hash_of_block.push_back(row_hash);
    state.rows_of_block.push_back(static_cast<UInt32>(n));

    ProfileEvents::increment(ProfileEvents::RadixHashBuildBlocksMoved);
}


void BuildStore::finishBuild()
{
    const size_t num_leaves = cfg.num_leaves;

    size_t num_blocks = 0;
    for (const auto & up : local)
        if (up)
            num_blocks += up->blocks.size();
    chassert(num_blocks <= std::numeric_limits<UInt32>::max()); /// BuildRef.block_no is 32-bit

    global_blocks.reserve(num_blocks);
    global_hash_of_block.reserve(num_blocks);
    global_rows_of_block.reserve(num_blocks);

    /// Move-concat per-worker stores in slot order (assigns final block_no).
    /// Record each used slot's contiguous block range for the static-per-thread scatter.
    for (size_t slot = 0; slot < local.size(); ++slot)
    {
        if (!local[slot] || local[slot]->blocks.empty())
            continue;
        LocalBuildState & state = *local[slot];
        thread_block_begin.push_back(global_blocks.size());
        for (size_t bi = 0; bi < state.blocks.size(); ++bi)
        {
            global_blocks.push_back(std::move(state.blocks[bi]));
            global_hash_of_block.push_back(state.hash_of_block[bi]);
            global_rows_of_block.push_back(state.rows_of_block[bi]);
        }
        thread_block_end.push_back(global_blocks.size());
        used_slots.push_back(slot);
    }

    /// Fold each used slot's replicated histogram into `global_hist`.
    /// The rep_hist accumulated across ALL blocks for that slot (no per-block reset in add()).
    global_hist.assign(num_leaves, 0);
    for (size_t slot : used_slots)
    {
        const auto & state = *local[slot];
        for (size_t rep = 0; rep < state.replicas; ++rep)
        {
            const UInt32 * rep_hist = state.rep_hist.data() + rep * num_leaves;
            for (size_t leaf = 0; leaf < num_leaves; ++leaf)
                global_hist[leaf] += rep_hist[leaf];
        }
    }

    offset.resize(num_leaves);
    std::exclusive_scan(global_hist.begin(), global_hist.end(), offset.begin(), UInt64{0});

    /// Per-block exclusive row-offset prefix sum: block_base[b] = Σ rows of blocks 0..b-1. Gives the
    /// flat next_chain index of a build row (phase P4): flat(ref) = block_base[ref.block_no] + row_no-1.
    block_base.assign(global_rows_of_block.size() + 1, 0);
    for (size_t block_idx = 0; block_idx < global_rows_of_block.size(); ++block_idx)
        block_base[block_idx + 1] = block_base[block_idx] + global_rows_of_block[block_idx];
    total_rows = block_base.empty() ? 0 : block_base.back();

    finished = true;
}


LeafArrays BuildStore::makeLeafArrays(bool with_leaf_hash) const
{
    LeafArrays out;
    out.num_leaves = cfg.num_leaves;
    out.key_width = key_width;
    out.arena = GrowingArena(arena_max_block, /*use_thp=*/true);
    out.key_base.assign(cfg.num_leaves, nullptr);
    out.ref_base.assign(cfg.num_leaves, nullptr);
    if (with_leaf_hash)
        out.hash_base.assign(cfg.num_leaves, nullptr);
    out.leaf_rows.assign(cfg.num_leaves, 0);
    /// One counter per build thread (used slot); the scatter assigns each thread's block count here.
    out.worker_block_counts.assign(used_slots.size(), 0);
    return out;
}

void BuildStore::finalizeScatter(
    LeafArrays & out,
    const Stopwatch & sw,
    std::atomic<UInt64> & total_bytes,
    size_t num_passes) const
{
    ProfileEvents::increment(ProfileEvents::RadixHashBuildScatterMicroseconds, sw.elapsedMicroseconds());
    const UInt64 scattered_rows = std::accumulate(global_hist.begin(), global_hist.end(), UInt64{0});
    ProfileEvents::increment(ProfileEvents::RadixHashScatterRows, scattered_rows * num_passes);
    out.bytes_scattered = total_bytes.load();
    out.arena.trim();
}


BuildStore::RefineWorkerScratch::RefineWorkerScratch(size_t max_fanout)
    : scratch(max_fanout), kout(max_fanout), rout(max_fanout), pout(max_fanout)
{
}

void BuildStore::refineDepthFirst(
    size_t global_first_leaf,
    const void * in_keys,
    const BuildRef * in_refs,
    const UInt32 * in_hashes,
    UInt64 rows,
    size_t pass_index,
    UInt32 bits_consumed,
    LeafArrays & out,
    const std::vector<UInt64> & gh_prefix,
    RefineWorkerScratch & ws,
    bool with_leaf_hash)
{
    const UInt32 pass_bits = cfg.pass_bits[pass_index];
    const size_t fanout = size_t{1} << pass_bits;
    const UInt32 new_bits = bits_consumed + pass_bits;
    /// leaf_fanout_shift: log2 of the number of final leaves per child partition at this level.
    const UInt32 leaf_fanout_shift = cfg.total_bits - new_bits;
    const size_t leaves_per_child = size_t{1} << leaf_fanout_shift;
    /// routing_shift: right-shift to apply to the 32-bit row hash to select this pass's bit-window.
    const UInt32 routing_shift = PartitionConfig::HASH_BITS - new_bits;
    const UInt32 mask = static_cast<UInt32>(fanout - 1);
    const bool is_last = (pass_index + 1 == cfg.pass_bits.size());
    const size_t kw = key_width;

    if (is_last)
    {
        /// Point directly at the pre-allocated final leaf arrays. When leaf hashes are requested,
        /// ws.pout points at the per-leaf hash arrays and the carried in_hashes are scattered too.
        for (size_t child = 0; child < fanout; ++child)
        {
            const size_t gidx = global_first_leaf + child * leaves_per_child;
            ws.kout[child] = out.key_base[gidx];
            ws.rout[child] = out.ref_base[gidx];
            ws.pout[child] = with_leaf_hash ? out.hash_base[gidx] : nullptr;
        }

        scatterKeyRefTwoColumn(
            in_hashes, routing_shift, mask, rows, in_keys, kw, in_refs, fanout,
            ws.kout.data(), ws.rout.data(), ws.scratch, shouldUseSwwc(2, static_cast<int>(fanout)));

        ws.local_bytes += rows * (kw + sizeof(BuildRef));

        if (with_leaf_hash)
        {
            scatterColumn(
                in_hashes, routing_shift, mask, rows, in_hashes, sizeof(UInt32),
                fanout, ws.pout.data(), ws.scratch, shouldUseSwwc(1, static_cast<int>(fanout)));
            ws.local_bytes += rows * sizeof(UInt32);
        }
    }
    else
    {
        /// Compute per-child row counts from the global leaf-prefix array.
        std::vector<UInt64> child_counts(fanout);
        for (size_t child = 0; child < fanout; ++child)
        {
            const size_t lo = global_first_leaf + child * leaves_per_child;
            const size_t hi = lo + leaves_per_child;
            child_counts[child] = gh_prefix[hi] - gh_prefix[lo];
        }

        /// Allocate the children of this partition in a RAII GrowingArena.
        /// The arena is freed when this stack frame exits — lowest peak intermediate memory.
        GrowingArena child_arena(arena_max_block);
        auto arrs = allocExactPartitions(child_arena, child_counts, kw, /*carry_hash=*/true);
        for (size_t child = 0; child < fanout; ++child)
        {
            ws.kout[child] = arrs.key[child];
            ws.rout[child] = arrs.ref[child];
            ws.pout[child] = arrs.hash[child];
        }

        scatterKeyRefTwoColumn(
            in_hashes, routing_shift, mask, rows, in_keys, kw, in_refs, fanout,
            ws.kout.data(), ws.rout.data(), ws.scratch, shouldUseSwwc(2, static_cast<int>(fanout)));

        /// Scatter the row hashes themselves so each child partition carries them for the next pass.
        scatterColumn(
            in_hashes, routing_shift, mask, rows, in_hashes, sizeof(UInt32),
            fanout, ws.pout.data(), ws.scratch, shouldUseSwwc(1, static_cast<int>(fanout)));

        ws.local_bytes += rows * (kw + sizeof(BuildRef) + sizeof(UInt32));

        /// Depth-first recursion: complete each child's subtree before the next.
        /// After each child returns, release its memory incrementally — so consumed intermediate
        /// data is returned to the OS as the final leaf output is first-touched.
        for (size_t child = 0; child < fanout; ++child)
        {
            if (child_counts[child] == 0)
                continue;
            const size_t child_first_leaf = global_first_leaf + child * leaves_per_child;
            refineDepthFirst(
                child_first_leaf,
                arrs.key[child],
                arrs.ref[child],
                static_cast<const UInt32 *>(arrs.hash[child]),
                child_counts[child],
                pass_index + 1,
                new_bits,
                out,
                gh_prefix,
                ws,
                with_leaf_hash);
            /// Child fully consumed — release its contiguous key+ref+hash block from child_arena
            /// in one madvise(MADV_DONTNEED) call (single alloc from allocExactPartitions).
            if (arrs.key[child])
            {
                const size_t total = roundUpTo64(child_counts[child] * kw)
                                   + roundUpTo64(child_counts[child] * sizeof(BuildRef))
                                   + roundUpTo64(child_counts[child] * sizeof(UInt32));
                child_arena.releaseRange(arrs.key[child], total);
            }
        }
        /// child_arena mapped memory released below (destructor calls munmap).
    }
}


template <bool carry_hash>
void BuildStore::scatterBlocksIntoPartitions(
    CoopPool & coord,
    size_t num_parts,
    UInt32 shift,
    UInt32 mask,
    const std::vector<UInt64> & thr_off,
    void * const * key_base_arr,
    BuildRef * const * ref_base_arr,
    void * const * hash_base_arr,
    std::atomic<UInt64> & total_bytes,
    std::vector<UInt64> & worker_counts)
{
    const size_t kw = key_width;
    const bool multi_col = key_positions.size() > 1;
    const size_t num_used = used_slots.size();

    /// Empty build (no accumulated blocks): nothing to scatter.
    if (num_used == 0)
        return;

    /// At high fanout the per-pass scatter routes through SWWC + NT (only emitted in a multitarget
    /// build); below the threshold, or without NT, it uses the direct incremental cursors. The hash
    /// column adds a third scattered column when carry_hash.
    const bool use_swwc = shouldUseSwwc(carry_hash ? 3 : 2, static_cast<int>(num_parts));

    coord.parallelFor(num_used, [&](size_t worker)
    {
        const UInt64 * worker_offsets = thr_off.data() + worker * num_parts;

        /// Direct path: per-partition live write cursors (used iff !use_swwc).
        std::vector<void *> kcur;
        std::vector<BuildRef *> rcur;
        std::vector<void *> hcur;

        /// SWWC path: per-worker persistent scratch (own cursors + NT staging) per scattered column.
        std::optional<ScatterScratch> key_ss;
        std::optional<ScatterScratch> ref_ss;
        std::optional<ScatterScratch> hash_ss;

        if (use_swwc)
        {
            /// A fresh ScatterScratch already zero-fills its cursors and line fills; seed the cursors
            /// ONCE from this worker's per-(thread,partition) starting offsets (no per-block reseeding).
            key_ss.emplace(num_parts);
            ref_ss.emplace(num_parts);
            if constexpr (carry_hash)
                hash_ss.emplace(num_parts);

            for (size_t part = 0; part < num_parts; ++part)
            {
                if (key_base_arr[part] != nullptr)
                {
                    key_ss->cursors()[part] = static_cast<char *>(key_base_arr[part]) + worker_offsets[part] * kw;
                    ref_ss->cursors()[part] = reinterpret_cast<char *>(ref_base_arr[part]) + worker_offsets[part] * sizeof(BuildRef);
                    if constexpr (carry_hash)
                        hash_ss->cursors()[part] = static_cast<char *>(hash_base_arr[part]) + worker_offsets[part] * sizeof(UInt32);
                }
                /// else: nullptr stays — empty partition, cursor never advanced
            }
        }
        else
        {
            kcur.assign(num_parts, nullptr);
            rcur.assign(num_parts, nullptr);
            if constexpr (carry_hash)
                hcur.assign(num_parts, nullptr);

            /// Seed cursors ONCE from this worker's per-partition starting offsets (no per-block reseeding).
            for (size_t part = 0; part < num_parts; ++part)
            {
                if (key_base_arr[part] != nullptr)
                {
                    kcur[part] = static_cast<char *>(key_base_arr[part]) + worker_offsets[part] * kw;
                    rcur[part] = ref_base_arr[part] + worker_offsets[part];
                    if constexpr (carry_hash)
                        hcur[part] = static_cast<char *>(hash_base_arr[part]) + worker_offsets[part] * sizeof(UInt32);
                }
                /// else: nullptr stays — empty partition, cursor never advanced
            }
        }

        std::vector<BuildRef> refs;
        std::vector<char> packed;
        if (multi_col)
            packed.resize(SCATTER_CHUNK_ROWS * kw);

        UInt64 local_bytes = 0;
        UInt64 local_blocks = 0;

        /// Each worker scatters its own contiguous block range — fully lock-free, disjoint regions.
        for (size_t block_idx = thread_block_begin[worker]; block_idx < thread_block_end[worker]; ++block_idx)
        {
            const size_t n = global_rows_of_block[block_idx];
            if (n == 0)
                continue;

            const UInt32 * row_hash = global_hash_of_block[block_idx];

            refs.resize(n);
            for (size_t row = 0; row < n; ++row)
                refs[row] = BuildRef{static_cast<UInt32>(block_idx), static_cast<UInt32>(row + 1)}; /// row_no is 1-based; 0 is reserved as the empty sentinel

            const char * raw_keys = multi_col
                ? nullptr
                : global_blocks[block_idx].getByPosition(key_positions[0]).column->getRawData().data();

            for (size_t row_begin = 0; row_begin < n; row_begin += SCATTER_CHUNK_ROWS)
            {
                const size_t chunk_rows = std::min(SCATTER_CHUNK_ROWS, n - row_begin);
                const void * keys_ptr = nullptr;
                if (multi_col)
                {
                    packKeyChunk(global_blocks[block_idx], row_begin, chunk_rows, packed.data());
                    keys_ptr = packed.data();
                }
                else
                {
                    keys_ptr = raw_keys + row_begin * kw;
                }

                if (use_swwc)
                {
                    local_bytes += scatterColumnIntoSwwc(
                        row_hash + row_begin, shift, mask, chunk_rows, keys_ptr, kw, num_parts, *key_ss);
                    local_bytes += scatterColumnIntoSwwc(
                        row_hash + row_begin, shift, mask, chunk_rows, refs.data() + row_begin, sizeof(BuildRef), num_parts, *ref_ss);
                    if constexpr (carry_hash)
                        local_bytes += scatterColumnIntoSwwc(
                            row_hash + row_begin, shift, mask, chunk_rows, row_hash + row_begin, sizeof(UInt32), num_parts, *hash_ss);
                }
                else
                {
                    local_bytes += scatterKeyRefInto(
                        row_hash + row_begin, shift, mask, chunk_rows, keys_ptr, kw,
                        refs.data() + row_begin, num_parts, kcur.data(), rcur.data());

                    if constexpr (carry_hash)
                    {
                        local_bytes += scatterColumnInto(
                            row_hash + row_begin, shift, mask, chunk_rows, row_hash + row_begin, sizeof(UInt32),
                            num_parts, hcur.data());
                    }
                }
            }

            ++local_blocks;
        }

        /// SWWC leaves each partition's last (< one line) residual staged; drain it once per column and
        /// fence the NT stores so the outputs are visible to the reader after the workers join.
        if (use_swwc)
        {
            scatterColumnDrainSwwc(num_parts, *key_ss);
            scatterColumnDrainSwwc(num_parts, *ref_ss);
            if constexpr (carry_hash)
                scatterColumnDrainSwwc(num_parts, *hash_ss);
        }

        worker_counts[worker] += local_blocks;
        total_bytes.fetch_add(local_bytes, std::memory_order_relaxed);
    });
}


LeafArrays BuildStore::scatterSinglePass(CoopPool & coord, bool with_leaf_hash)
{
    LeafArrays out = makeLeafArrays(with_leaf_hash);
    const size_t num_leaves = cfg.num_leaves;
    const size_t num_used = used_slots.size();

    /// Allocate each leaf's key and ref arrays (and, for P4, the per-leaf hash array) exactly once
    /// (NC gate: O(num_leaves) carves).
    auto arrs = allocExactPartitions(out.arena, global_hist, key_width, /*carry_hash=*/with_leaf_hash);
    out.key_base = std::move(arrs.key);
    out.ref_base = std::move(arrs.ref);
    if (with_leaf_hash)
        out.hash_base = std::move(arrs.hash);
    out.alloc_count = arrs.alloc_count;
    for (size_t leaf = 0; leaf < num_leaves; ++leaf)
        out.leaf_rows[leaf] = global_hist[leaf];

    const UInt32 shift = cfg.total_bits > 0 ? cfg.shift : 0u;
    const UInt32 mask = static_cast<UInt32>(num_leaves - 1);

    /// Build the per-(thread,leaf) starting offsets within each leaf array.
    /// thr_off[worker*num_leaves + leaf] = number of rows contributed by threads 0..worker-1 to leaf.
    std::vector<UInt64> thr_off(num_used * num_leaves, 0);
    {
        std::vector<UInt64> running(num_leaves, 0); /// running[leaf] = cumulative rows assigned so far
        for (size_t worker = 0; worker < num_used; ++worker)
        {
            const auto & state = *local[used_slots[worker]];
            UInt64 * worker_start = thr_off.data() + worker * num_leaves;
            for (size_t leaf = 0; leaf < num_leaves; ++leaf)
                worker_start[leaf] = running[leaf]; /// this thread starts at the current cursor
            for (size_t rep = 0; rep < state.replicas; ++rep)
            {
                const UInt32 * rep_hist = state.rep_hist.data() + rep * num_leaves;
                for (size_t leaf = 0; leaf < num_leaves; ++leaf)
                    running[leaf] += rep_hist[leaf];
            }
        }
    }

    std::atomic<UInt64> total_bytes{0};
    Stopwatch sw;
    if (with_leaf_hash)
        scatterBlocksIntoPartitions</*carry_hash=*/true>(
            coord, num_leaves, shift, mask,
            thr_off, out.key_base.data(), out.ref_base.data(), out.hash_base.data(),
            total_bytes, out.worker_block_counts);
    else
        scatterBlocksIntoPartitions</*carry_hash=*/false>(
            coord, num_leaves, shift, mask,
            thr_off, out.key_base.data(), out.ref_base.data(), /*hash_base_arr=*/nullptr,
            total_bytes, out.worker_block_counts);

    finalizeScatter(out, sw, total_bytes, /*num_passes=*/1);
    return out;
}


LeafArrays BuildStore::scatterMultiPass(CoopPool & coord, bool with_leaf_hash)
{
    const size_t num_leaves = cfg.num_leaves;
    const size_t kw = key_width;
    const size_t num_passes = cfg.pass_bits.size();

    /// gh_prefix[i+1] = sum(global_hist[0..i]); gh_prefix[hi]-gh_prefix[lo] = rows in leaves [lo,hi).
    std::vector<UInt64> gh_prefix(num_leaves + 1, 0);
    std::inclusive_scan(global_hist.begin(), global_hist.end(), gh_prefix.begin() + 1);

    /// Pre-allocate ALL final leaf key/ref arrays (and, for P4, hash arrays) exactly once
    /// (NC gate: O(num_leaves) carves).
    LeafArrays out = makeLeafArrays(with_leaf_hash);
    {
        auto arrs = allocExactPartitions(out.arena, global_hist, kw, /*carry_hash=*/with_leaf_hash);
        out.key_base = std::move(arrs.key);
        out.ref_base = std::move(arrs.ref);
        if (with_leaf_hash)
            out.hash_base = std::move(arrs.hash);
        out.alloc_count = arrs.alloc_count;
        for (size_t leaf = 0; leaf < num_leaves; ++leaf)
            out.leaf_rows[leaf] = global_hist[leaf];
    }

    std::atomic<UInt64> total_bytes{0};
    Stopwatch sw;

    // ---- Pass 0: blocks -> pass-0 partitions (carry_hash) -----------------------------------
    const UInt32 pass0_bits = cfg.pass_bits[0];
    const size_t p0 = size_t{1} << pass0_bits;
    /// shift0 selects the top `pass0_bits` of the 32-bit hash: shift0 = HASH_BITS - pass0_bits.
    const UInt32 shift0 = PartitionConfig::HASH_BITS - pass0_bits;
    const UInt32 mask0 = static_cast<UInt32>(p0 - 1);

    chassert(p0 > 0 && num_leaves % p0 == 0); /// num_leaves is 2^total_bits, p0 is 2^pass0_bits, always divides
    const size_t leaves_per_p0 = num_leaves / p0; // NOLINT(clang-analyzer-core.DivideZero)

    const size_t num_used = used_slots.size();

    /// Fold each thread's rep_hist to pass-0 granularity and compute per-(thread,partition) offsets.
    /// thr_hist0[w * p0 + part] = rows thread w contributes to pass-0 partition `part`.
    std::vector<UInt64> thr_hist0(num_used * p0, 0);
    for (size_t worker = 0; worker < num_used; ++worker)
    {
        const auto & state = *local[used_slots[worker]];
        for (size_t rep = 0; rep < state.replicas; ++rep)
        {
            const UInt32 * rep_hist = state.rep_hist.data() + rep * num_leaves;
            for (size_t part = 0; part < p0; ++part)
            {
                const size_t lo = part * leaves_per_p0;
                const size_t hi = lo + leaves_per_p0;
                UInt64 sum = 0;
                for (size_t leaf = lo; leaf < hi; ++leaf)
                    sum += rep_hist[leaf];
                thr_hist0[worker * p0 + part] += sum;
            }
        }
    }

    /// Level-0 partition sizes.
    std::vector<UInt64> level0_counts(p0, 0);
    for (size_t worker = 0; worker < num_used; ++worker)
        for (size_t part = 0; part < p0; ++part)
            level0_counts[part] += thr_hist0[worker * p0 + part];

    /// thr_off0[worker * p0 + part] = starting row index within partition `part`'s SEPARATE allocated
    /// array for that worker (starts at 0 for worker 0; each partition has its own allocation).
    std::vector<UInt64> thr_off0(num_used * p0, 0);
    {
        std::vector<UInt64> running0(p0, 0); /// offset within each partition's own array (starts at 0)
        for (size_t worker = 0; worker < num_used; ++worker)
        {
            UInt64 * worker_start = thr_off0.data() + worker * p0;
            for (size_t part = 0; part < p0; ++part)
            {
                worker_start[part] = running0[part];
                running0[part] += thr_hist0[worker * p0 + part];
            }
        }
    }

    CascadeLevel level0;
    level0.num_parts = p0;
    level0.arena = GrowingArena(arena_max_block, /*use_thp=*/true);
    {
        auto arrs = allocExactPartitions(level0.arena, level0_counts, kw, /*carry_hash=*/true);
        level0.key   = std::move(arrs.key);
        level0.ref   = std::move(arrs.ref);
        level0.hash  = std::move(arrs.hash);
        level0.count = std::move(level0_counts);
    }

    scatterBlocksIntoPartitions</*carry_hash=*/true>(
        coord, p0, shift0, mask0,
        thr_off0, level0.key.data(), level0.ref.data(), level0.hash.data(),
        total_bytes, out.worker_block_counts);

    /// Hash arenas are no longer needed after pass-0: each row's hash is now carried inside level0.
    for (auto & up : local)
        if (up)
            up->hash_arena.trim();

    // ---- Depth-first refinement: each worker owns a whole pass-0 subtree to its leaves -------
    const size_t max_refine_fanout = [&]
    {
        size_t mf = 1;
        for (size_t pass_idx = 1; pass_idx < num_passes; ++pass_idx)
            mf = std::max(mf, size_t{1} << cfg.pass_bits[pass_idx]);
        return mf;
    }();

    /// Depth-first refinement: one unit per pass-0 partition, each allocates its own scratch.
    coord.parallelFor(p0, [&](size_t partition)
    {
        if (level0.count[partition] == 0)
            return;

        RefineWorkerScratch ws(max_refine_fanout);
        refineDepthFirst(
            partition * leaves_per_p0,
            level0.key[partition],
            level0.ref[partition],
            static_cast<const UInt32 *>(level0.hash[partition]),
            level0.count[partition],
            /*pass_index=*/1,
            pass0_bits,
            out,
            gh_prefix,
            ws,
            with_leaf_hash);

        /// Partition fully consumed — release its contiguous key+ref+hash block
        /// back to the OS in one madvise(MADV_DONTNEED) call.
        if (level0.key[partition])
        {
            const size_t release_bytes = roundUpTo64(level0.count[partition] * kw)
                                       + roundUpTo64(level0.count[partition] * sizeof(BuildRef))
                                       + roundUpTo64(level0.count[partition] * sizeof(UInt32));
            level0.arena.releaseRange(level0.key[partition], release_bytes);
        }

        total_bytes.fetch_add(ws.local_bytes, std::memory_order_relaxed);
    });

    finalizeScatter(out, sw, total_bytes, num_passes);
    return out;
}


LeafArrays BuildStore::scatterToLeaves(CoopPool & coord, bool with_leaf_hash)
{
    chassert(finished);

    LeafArrays out = cfg.pass_bits.size() <= 1
        ? scatterSinglePass(coord, with_leaf_hash)
        : scatterMultiPass(coord, with_leaf_hash);

    /// For single-pass: trim hash arenas now that scatter is done.
    /// For multi-pass: already trimmed inside scatterMultiPass right after pass-0 completed.
    if (cfg.pass_bits.size() <= 1)
        for (auto & up : local)
            if (up)
                up->hash_arena.trim();

    return out;
}

} /// namespace DB::RadixHash
