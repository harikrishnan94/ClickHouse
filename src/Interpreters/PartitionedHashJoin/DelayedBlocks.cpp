#include <Interpreters/PartitionedHashJoin.h>
#include <Interpreters/PartitionedHashJoin/DelayedBlocks.h>
#include <Interpreters/PartitionedHashJoin/OutBlock.h>
#include <Interpreters/PartitionedHashJoin/ThreadSlot.h>

#include <Interpreters/HashJoin/HashJoin.h>

#include <Columns/ColumnNullable.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/IColumn.h>

#include <Core/Block.h>
#include <Interpreters/HashJoin/HashJoinProbePhaseHooks.h>
#include <Common/assert_cast.h>

#include <cstring>

// PHJ per-partition phase hook: fires the harness probe-point callback.
// The macro is a no-op unless the harness has registered a callback via
// `setProbePointCallback` — production builds see zero overhead.
#define PHJ_PHASE_POINT(partition, name) ::DB::HashProbeBench::fireProbePoint(::DB::HashProbeBench::ProbePoint::name, partition)

namespace DB
{

/// Append one OutBlock's payload into a pre-existing set of MutableColumns.
/// The columns must already be cloneEmpty'd from the right-side schema; the
/// caller controls their lifetime and the eventual cloneWithColumns step.
///
/// This is the hot inner of the build phase: each call writes `ob.filled`
/// rows into the destination columns via insertManyDefaults() + memcpy()
/// (one allocation grow + one bulk memcpy per column), and that's it. No
/// throwaway Block is materialised per OutBlock — saving a Block ctor,
/// per-column cloneEmpty, and an extra insertRangeFrom pass that the
/// previous implementation incurred for every one of the ~50K OutBlocks
/// per query.
static void appendOutBlockToColumns(const OutBlock & ob, const ShuffleSpec & spec, MutableColumns & out_cols)
{
    if (ob.filled == 0)
        return;

    const size_t n = ob.filled;
    const auto & scatter_cols = spec.scatter_cols;

    // Each scatter_cols entry maps to one ob.cols[si] buffer.
    // Nullable columns contribute TWO entries (is_nullable=true then is_nullmap=true).
    for (size_t si = 0; si < scatter_cols.size() && si < static_cast<size_t>(ob.num_cols); ++si)
    {
        const auto & sc = scatter_cols[si];
        const size_t col_idx = sc.block_pos;
        const size_t eb = sc.elem_bytes;

        if (col_idx >= out_cols.size() || ob.cols[si] == nullptr)
            continue;

        auto & dst = out_cols[col_idx];

        if (sc.is_nullmap)
        {
            // Null-map slot: write 1-byte entries into the ColumnNullable's null map.
            auto & dst_null = assert_cast<ColumnNullable &>(*dst);
            auto & nm_col = dst_null.getNullMapColumn();
            const size_t old_size = nm_col.size();
            nm_col.insertManyDefaults(n);
            std::memcpy(const_cast<char *>(nm_col.getRawData().data()) + old_size, ob.cols[si], n);
        }
        else if (sc.is_nullable)
        {
            // Inner-data slot of a nullable column.
            auto & dst_null = assert_cast<ColumnNullable &>(*dst);
            auto & inner = dst_null.getNestedColumn();
            const size_t old_size = inner.size();
            inner.insertManyDefaults(n);
            std::memcpy(const_cast<char *>(inner.getRawData().data()) + old_size * eb, ob.cols[si], n * eb);
        }
        else
        {
            // Plain non-nullable column.
            const size_t old_size = dst->size();
            dst->insertManyDefaults(n);
            std::memcpy(const_cast<char *>(dst->getRawData().data()) + old_size * eb, ob.cols[si], n * eb);
        }
    }
}

/// Legacy wrapper retained for the .h declaration; not used on the hot path
/// any more. Builds a fresh Block from one OutBlock.
Block PartitionedHashJoinDelayedBlocks::buildOutBlockToBlock(const OutBlock & ob, const ShuffleSpec & spec) const
{
    if (ob.filled == 0)
        return {};

    const auto & schema = *join_.rightSampleBlock();
    MutableColumns out_cols;
    out_cols.reserve(schema.columns());
    for (size_t ci = 0; ci < schema.columns(); ++ci)
        out_cols.push_back(schema.getByPosition(ci).column->cloneEmpty());

    appendOutBlockToColumns(ob, spec, out_cols);
    return schema.cloneWithColumns(std::move(out_cols));
}

/// State for one partition being streamed. At most one worker thread owns a
/// WorkerState at any time (enforced by the free-list discipline). When the
/// owning worker yields back to the pipeline executor between blocks, it
/// releases the state to the free-list; any worker may then pick it up to
/// resume — preserving correctness even if the pipeline migrates the
/// DelayedJoinedBlocksWorkerTransform across executor threads.
/// Target block size when re-chunking the tiny per-partition probe slices.
/// The probe shuffle produces one slice per input block per partition, which
/// for a 256-partition × ~65K-row input means ~256-row slices. Without
/// coalescing, each slice would drive one joinBlock() call that returns a
/// ~256-row result block, flooding the downstream chain with millions of
/// micro-blocks.
///
/// We coalesce to ~max_block_size (default 65K) rows: this matches the chunk
/// size HashJoin uses internally for its result iterator, so each joinBlock()
/// call produces one full-sized output block that flows efficiently through
/// the rest of the pipeline. The cost is one cheap pass of insertRangeFrom
/// per partition; the payoff is ~4× fewer joinBlock invocations (vs the
/// previous 16K target) and ~4× fewer downstream blocks.
static constexpr size_t kProbeChunkTargetRows = 65536;

struct PartitionedHashJoinDelayedBlocks::WorkerState
{
    bool active = false;
    /// Partition number that this state currently represents. Set in
    /// initStateForPartition(); used by produceFromState() to find the right
    /// probe-side slices in each ThreadSlot.
    size_t partition = 0;

    /// The per-partition mini HashJoin used for build & probe.
    std::shared_ptr<HashJoin> part_hj;
    /// Usually == part_hj. Differs only for the LEFT/FULL-with-empty-build
    /// special case, where we need a fresh HashJoin that hasn't had
    /// onBuildPhaseFinish() called on it (otherwise ALL → RightAny promotion
    /// would drop unmatched left rows).
    std::shared_ptr<HashJoin> probe_hj;
    /// Cached at init: hasNonJoinedRows() — drives whether to emit non-joined
    /// blocks after the probe phase (RIGHT/FULL).
    bool has_non_joined = false;

    /// Coalesced probe blocks for this partition. Built in initStateForPartition()
    /// by concatenating the small slices from each ThreadSlot's probe_parts[p]
    /// into ~kProbeChunkTargetRows-sized blocks. This is the iteration source
    /// for the probe phase — we no longer walk slot/slice pairs.
    std::vector<Block> probe_blocks;
    size_t probe_block_idx = 0;
    /// Result iterator of the most recent joinBlock(). When non-null, calls
    /// to produceFromState() drain a block from here before advancing.
    JoinResultPtr current_result;

    /// After probes are exhausted, the non-joined stream takes over.
    IBlocksStreamPtr non_joined_stream;
    bool non_joined_started = false;
};

PartitionedHashJoinDelayedBlocks::PartitionedHashJoinDelayedBlocks(PartitionedHashJoin & join)
    : join_(join)
{
}

PartitionedHashJoinDelayedBlocks::~PartitionedHashJoinDelayedBlocks() = default;

bool PartitionedHashJoinDelayedBlocks::initStateForPartition(WorkerState & state, size_t p) const
{
    state.partition = p;

    const ShuffleSpec & bspec = join_.buildSpec();
    /// Pipeline contract: getDelayedBlocks() is only called after all build- and
    /// probe-side ingest is done, so slots[] is stable here.  num_slots_created
    /// is updated under slot_mu by getOrAssign*Slot() and released atomically;
    /// this acquire-load is the matching pair.
    const size_t n_slots = join_.numSlots();

    size_t total_build = 0;
    size_t total_probe = 0;
    for (size_t s = 0; s < n_slots; ++s)
    {
        ThreadSlot & slot = join_.getSlot(s);
        if (slot.build_initialised)
            total_build += slot.build_parts[p].total_rows;
        if (slot.probe_initialised)
            total_probe += slot.probe_parts[p].total_rows;
    }

    // Skip only if both sides are completely empty — nothing to emit.
    if (total_build == 0 && total_probe == 0)
        return false;

    auto part_hj = std::make_shared<HashJoin>(join_.tableJoin(), join_.rightSampleBlock(), join_.anyTakeLastRow(), total_build);

    PHJ_PHASE_POINT(state.partition, phj_build_ht_start);
    bool any_build_added = false;
    for (size_t s = 0; s < n_slots; ++s)
    {
        ThreadSlot & slot = join_.getSlot(s);
        if (!slot.build_initialised)
            continue;
        for (const OutBlock * b = slot.build_parts[p].head; b; b = b->next)
        {
            const Block blk = buildOutBlockToBlock(*b, bspec);
            if (blk.rows() > 0)
            {
                part_hj->addBlockToJoin(blk, false);
                any_build_added = true;
            }
        }
    }
    part_hj->onBuildPhaseFinish();
    if (part_hj->hasPostBuildPhase())
        part_hj->runPostBuildPhase();
    PHJ_PHASE_POINT(state.partition, phj_build_ht_end);

    // LEFT/FULL with empty build: HashJoin::onBuildPhaseFinish() has promoted
    // ALL → RightAny (all_values_unique is true for an empty HT), which
    // disables add_missing and drops unmatched left rows. Use a fresh
    // HashJoin (no onBuildPhaseFinish call) for probing so add_missing stays
    // true and unmatched rows emit with NULLs.
    std::shared_ptr<HashJoin> probe_hj = part_hj;
    if (!any_build_added)
    {
        const auto k = join_.tableJoin()->kind();
        if (k == JoinKind::Left || k == JoinKind::Full)
            probe_hj = std::make_shared<HashJoin>(join_.tableJoin(), join_.rightSampleBlock(), join_.anyTakeLastRow(), 0);
    }

    // ── Coalesce probe slices ──────────────────────────────────────────
    // Each call to probeScatterBlock during probe ingest emits a slice of
    // (input_block_rows / P) rows into this partition. With default settings
    // that's ~256 rows per slice — far too small to drive joinBlock efficiently
    // (each call has fixed setup cost; tiny results flood downstream).
    //
    // We concatenate small slices into ~kProbeChunkTargetRows-sized blocks
    // using IColumn::insertRangeFrom. This is one cheap memcpy pass per
    // partition; the payoff is ~16× fewer joinBlock calls and ~16× fewer
    // downstream blocks, which is the dominant factor in PHJ wall time
    // because the executor's per-block updateNode mutex dominates otherwise.
    state.probe_blocks.clear();
    {
        MutableColumns pending_cols;
        size_t pending_rows = 0;
        Block pending_header;

        auto flush_pending = [&]()
        {
            if (pending_rows == 0)
                return;
            state.probe_blocks.push_back(pending_header.cloneWithColumns(std::move(pending_cols)));
            pending_cols.clear();
            pending_rows = 0;
        };

        for (size_t s = 0; s < n_slots; ++s)
        {
            ThreadSlot & slot = join_.getSlot(s);
            if (!slot.probe_initialised)
                continue;
            for (const Block & slice : slot.probe_parts[p].slices)
            {
                const size_t sr = slice.rows();
                if (sr == 0)
                    continue;

                // Optimization: if the slice itself is already large, push it
                // through without copying.
                if (pending_rows == 0 && sr >= kProbeChunkTargetRows)
                {
                    state.probe_blocks.push_back(slice);
                    continue;
                }

                if (pending_cols.empty())
                {
                    // Initialize accumulator from this slice (clone into mutable).
                    pending_header = slice.cloneEmpty();
                    pending_cols = pending_header.mutateColumns();
                    for (auto & c : pending_cols)
                        c->reserve(kProbeChunkTargetRows);
                }
                for (size_t ci = 0; ci < pending_cols.size(); ++ci)
                    pending_cols[ci]->insertRangeFrom(*slice.getByPosition(ci).column, 0, sr);
                pending_rows += sr;

                if (pending_rows >= kProbeChunkTargetRows)
                    flush_pending();
            }
        }
        flush_pending();
    }

    state.active = true;
    state.part_hj = std::move(part_hj);
    state.probe_hj = std::move(probe_hj);
    state.has_non_joined = state.part_hj->hasNonJoinedRows() && join_.hasNonJoinedHeaders();
    state.probe_block_idx = 0;
    state.current_result.reset();
    state.non_joined_stream.reset();
    state.non_joined_started = false;
    return true;
}

Block PartitionedHashJoinDelayedBlocks::produceFromState(WorkerState & state) const
{
    // 1) If a probe result iterator is alive, drain its next block.
    //    This is the resumption path: phj_probe_start and phj_gen_start were
    //    already fired in a previous call; fire the matching _end hooks here
    //    as soon as is_last resets current_result so the callback always sees
    //    a paired start/end regardless of how many blocks the result yields.
    while (state.current_result)
    {
        auto data = state.current_result->next();
        if (data.is_last)
        {
            state.current_result.reset();
            PHJ_PHASE_POINT(state.partition, phj_gen_end);
            PHJ_PHASE_POINT(state.partition, phj_probe_end);
        }
        if (!data.block.empty())
        {
            ::DB::HashProbeBench::setProbePointPartition(state.partition);
            return std::move(data.block);
        }
    }

    // 2) Advance through the coalesced probe blocks for this partition.
    while (state.probe_block_idx < state.probe_blocks.size())
    {
        Block & probe_blk = state.probe_blocks[state.probe_block_idx];
        ++state.probe_block_idx;
        if (probe_blk.rows() == 0)
            continue;

        PHJ_PHASE_POINT(state.partition, phj_probe_start);
        state.current_result = state.probe_hj->joinBlock(std::move(probe_blk));
        PHJ_PHASE_POINT(state.partition, phj_gen_start);
        while (state.current_result)
        {
            auto data = state.current_result->next();
            if (data.is_last)
            {
                state.current_result.reset();
                PHJ_PHASE_POINT(state.partition, phj_gen_end);
                PHJ_PHASE_POINT(state.partition, phj_probe_end);
            }
            if (!data.block.empty())
            {
                ::DB::HashProbeBench::setProbePointPartition(state.partition);
                return std::move(data.block);
            }
        }
        // current_result was exhausted without any non-empty block (empty probe
        // block or zero-match partition slice). The _end hooks already fired
        // inside the loop when is_last reset current_result.
    }

    // 3) Probes exhausted. Emit non-joined rows for RIGHT/FULL, one block per call.
    if (state.has_non_joined)
    {
        if (!state.non_joined_started)
        {
            state.non_joined_stream = state.part_hj->getNonJoinedBlocks(join_.nonJoinedLeftHeader(), join_.nonJoinedResultHeader(), 65536);
            state.non_joined_started = true;
        }
        if (state.non_joined_stream)
        {
            while (!state.non_joined_stream->isFinished())
            {
                Block b = state.non_joined_stream->next();
                if (!b.empty())
                {
                    ::DB::HashProbeBench::setProbePointPartition(state.partition);
                    return b;
                }
            }
            state.non_joined_stream.reset();
        }
    }

    // Partition fully drained.
    return {};
}

Block PartitionedHashJoinDelayedBlocks::nextImpl()
{
    const size_t P = join_.numPartitions();

    // Acquire a state up-front. Prefer ACTIVE states (work-stealing) so idle
    // workers immediately help drain in-progress partitions instead of
    // claiming a new (build-from-scratch) one.
    std::unique_ptr<WorkerState> state;
    {
        std::scoped_lock lg(states_mu_);
        if (!free_states_.empty())
        {
            // Scan backwards for an active state.
            size_t take = free_states_.size();
            for (size_t i = free_states_.size(); i-- > 0;)
            {
                if (free_states_[i]->active)
                {
                    take = i;
                    break;
                }
            }
            if (take == free_states_.size())
                take = free_states_.size() - 1; // none active — take any
            state = std::move(free_states_[take]);
            if (take + 1 != free_states_.size())
                free_states_[take] = std::move(free_states_.back());
            free_states_.pop_back();
        }
        else
        {
            state = std::make_unique<WorkerState>();
        }
    }

    // RAII guard: always return the state to the pool, even on exception.
    // If state.active=true at this point, the partition is still in-progress;
    // a subsequent caller will steal it — but peers parked on `cv_` must be
    // woken so they can do the stealing.
    struct Releaser
    {
        PartitionedHashJoinDelayedBlocks * self;
        std::unique_ptr<WorkerState> * st;
        ~Releaser()
        {
            if (!st || !*st)
                return;
            const bool was_active = (*st)->active;
            {
                std::scoped_lock lg(self->states_mu_);
                self->free_states_.push_back(std::move(*st));
            }
            // Wake parked peers if we just yielded an in-progress partition
            // back to the pool. (Inactive releases never satisfy the wait
            // predicate, so notifying for them is pure overhead.)
            if (was_active)
                self->cv_.notify_one();
        }
    } releaser{this, &state};

    while (true)
    {
        if (state->active)
        {
            Block b = produceFromState(*state);
            if (!b.empty())
                return b;
            // Partition exhausted by us. Decrement the global active count
            // under the mutex and wake any workers parked at the termination
            // gate (they may now be able to declare done if count == 0).
            {
                std::scoped_lock lg(states_mu_);
                --active_count_;
            }
            cv_.notify_all();
            state->active = false;
            state->part_hj.reset();
            state->probe_hj.reset();
            state->current_result.reset();
            state->non_joined_stream.reset();
            state->probe_blocks.clear();
            state->probe_block_idx = 0;
        }

        // Try to grab a fresh partition.
        const size_t p = join_.partitionCursor().fetch_add(1, std::memory_order_relaxed);
        if (p < P)
        {
            // Race-free: reserve a slot in active_count_ BEFORE initialising
            // the state. If we set state.active=true inside init first and
            // then incremented, a peer at the termination gate could observe
            // active_count_ == 0 (with no active state visible in the pool
            // because we hold ours) and prematurely return {}, which
            // IBlocksStream::next() would then latch into `finished=true` for
            // every other worker.
            {
                std::scoped_lock lg(states_mu_);
                ++active_count_;
            }
            if (!initStateForPartition(*state, p))
            {
                // Empty partition: undo the reservation and wake any parked
                // peers (so they re-check the predicate and either see
                // active_count_ truly drop to 0 or continue waiting).
                {
                    std::scoped_lock lg(states_mu_);
                    --active_count_;
                }
                cv_.notify_all();
            }
            continue;
        }

        // Cursor exhausted. Either steal an active state from a peer who
        // released theirs, or wait for active_count_ to hit zero (truly done).
        std::unique_lock lk(states_mu_);
        cv_.wait(
            lk,
            [this]
            {
                if (active_count_ == 0)
                    return true;
                for (const auto & s : free_states_)
                    if (s->active)
                        return true;
                return false;
            });

        if (active_count_ == 0)
        {
            // All partitions fully drained.  RAII releaser will return our
            // (inactive) state to the pool after we return.
            return {};
        }

        // Steal an active state from the pool. Swap our inactive state with
        // a peer's active one so the pool keeps its size.
        for (size_t i = free_states_.size(); i-- > 0;)
        {
            if (free_states_[i]->active)
            {
                std::swap(state, free_states_[i]);
                break;
            }
        }
        // state is now active; loop to produce a block from it.
    }
}

}
