#include <Interpreters/PartitionedHashJoin.h>
#include <Interpreters/PartitionedHashJoin/DelayedBlocks.h>
#include <Interpreters/PartitionedHashJoin/Eligibility.h>
#include <Interpreters/PartitionedHashJoin/ProbePartitions.h>
#include <Interpreters/PartitionedHashJoin/RadixShuffle.h>
#include <Interpreters/PartitionedHashJoin/RadixShuffleHash.h>

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/JoinUtils.h>

#include <Columns/ColumnNullable.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeNullable.h>

#include <Common/TargetSpecific.h>
#include <Common/assert_cast.h>

#if USE_MULTITARGET_CODE
#    include <immintrin.h>
#endif

#include <algorithm>
#include <unordered_set>

namespace DB
{
namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int SET_SIZE_LIMIT_EXCEEDED;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

static size_t nextPow2Clamped(size_t v, size_t lo, size_t hi)
{
    if (v <= lo)
        return lo;
    if (v >= hi)
        return hi;
    size_t p = 1;
    while (p < v)
        p <<= 1;
    return std::min(p, hi);
}

static bool colIsNullable(const DataTypePtr & dt)
{
    if (dt->isNullable())
        return true;
    if (dt->getTypeId() == TypeIndex::LowCardinality)
        return removeLowCardinality(dt)->isNullable();
    return false;
}

/// Push one column (and its null-map slot if nullable) into the spec.
/// Nullable columns produce TWO scatter_cols entries:
///   [i]   inner data  (is_nullable=true,  is_nullmap=false, elem_bytes=inner_size)
///   [i+1] null map    (is_nullable=false, is_nullmap=true,  elem_bytes=1)
/// Non-nullable columns produce ONE entry.
static void pushColDesc(ShuffleSpec & s, size_t block_pos, size_t eb, bool is_nullable, bool is_key)
{
    ShuffleColDesc inner;
    inner.block_pos = block_pos;
    inner.elem_bytes = eb;
    inner.is_nullable = is_nullable;
    inner.is_nullmap = false;

    if (is_key)
        s.key_cols.push_back(inner);
    else
        s.payload_cols.push_back(inner);
    s.scatter_cols.push_back(inner);
    s.col_elem_bytes.push_back(eb);

    if (is_nullable)
    {
        ShuffleColDesc nm;
        nm.block_pos = block_pos;
        nm.elem_bytes = 1;
        nm.is_nullable = false;
        nm.is_nullmap = true;
        s.scatter_cols.push_back(nm);
        s.col_elem_bytes.push_back(1);
    }
}

static ShuffleSpec makeBuildSpec(const Block & schema, const Names & key_names, size_t P, size_t max_threads)
{
    ShuffleSpec s;
    s.P = P;
    s.batch_size = std::max<size_t>(1024, std::min<size_t>(32768, P * 16));
    const std::unordered_set<std::string> key_set(key_names.begin(), key_names.end());

    for (const auto & name : key_names)
    {
        if (!schema.has(name))
            continue;
        const auto & col = schema.getByName(name);
        const size_t eb = fixedElemBytes(col.type);
        if (eb == 0)
            continue;
        pushColDesc(s, schema.getPositionByName(name), eb, colIsNullable(col.type), /*is_key=*/true);
    }

    for (size_t ci = 0; ci < schema.columns(); ++ci)
    {
        const auto & col = schema.getByPosition(ci);
        if (key_set.contains(col.name))
            continue;
        const size_t eb = fixedElemBytes(col.type);
        if (eb == 0)
            continue;
        pushColDesc(s, ci, eb, colIsNullable(col.type), /*is_key=*/false);
    }

    s.use_swwc = ShuffleSpec::shouldUseSWWC(s.scatter_cols.size(), P, max_threads);
    return s;
}

// ── PartitionedHashJoin ────────────────────────────────────────────────────────

PartitionedHashJoin::PartitionedHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader right_sample_block_,
    SharedHeader left_sample_block_,
    size_t num_partitions,
    size_t max_threads_,
    bool any_take_last_row_)
    : table_join(std::move(table_join_))
    , right_sample(std::move(right_sample_block_))
    , left_sample(std::move(left_sample_block_))
    , any_take_last_row(any_take_last_row_)
{
    const size_t P = nextPow2Clamped(num_partitions == 0 ? 256 : num_partitions, 64, 1024);
    const auto & clauses = table_join->getClauses();
    if (clauses.empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "PartitionedHashJoin: no ON clause");
    build_spec = makeBuildSpec(*right_sample, clauses.front().key_names_right, P, max_threads_);

    /// Reserve capacity so slots.push_back() never reallocates.
    /// Reallocation would make slots[s] access freed memory in another thread.
    /// Upper bound: build threads + probe threads = 2 * max_threads.
    const size_t max_slots = std::max<size_t>(2, 2 * max_threads_);
    slots.reserve(max_slots);
}

PartitionedHashJoin::~PartitionedHashJoin() = default;

ThreadSlot & PartitionedHashJoin::getOrAssignBuildSlot()
{
    const auto tid = std::this_thread::get_id();
    const std::scoped_lock lg(slot_mu);
    auto it = build_tid_to_slot.find(tid);
    if (it != build_tid_to_slot.end())
        return *slots[it->second];
    const size_t idx = slots.size();
    slots.push_back(std::make_unique<ThreadSlot>());
    slots.back()->initBuildSide(build_spec);
    build_tid_to_slot[tid] = idx;
    num_slots_created.fetch_add(1, std::memory_order_release);
    return *slots.back();
}

ThreadSlot & PartitionedHashJoin::getOrAssignProbeSlot()
{
    const auto tid = std::this_thread::get_id();
    const std::scoped_lock lg(slot_mu);
    auto it = probe_tid_to_slot.find(tid);
    if (it != probe_tid_to_slot.end())
        return *slots[it->second];
    const size_t idx = slots.size();
    slots.push_back(std::make_unique<ThreadSlot>());
    slots.back()->initProbeSide(build_spec.P);
    probe_tid_to_slot[tid] = idx;
    num_slots_created.fetch_add(1, std::memory_order_release);
    return *slots.back();
}

/// Slot resolution for the cookie path.
///
/// Each cookie owns exactly one build slot and one probe slot, allocated lazily on first
/// use. We deliberately do NOT route through the tid-keyed slot map here: that would let
/// two cookies (= two distinct processor instances) that happen to make their first call
/// on the same OS thread end up sharing a slot, which then races when their respective
/// processors later run concurrently on different OS threads (pipeline work-stealing).
///
/// The lock-free fast path is "cookie already has slot pointer cached". The slow path
/// takes slot_mu just for slots.push_back(). slots is reserved at ctor; push_back never
/// reallocates so the cached pointers stay valid for the lifetime of the join.
ThreadSlot & PartitionedHashJoin::resolveBuildSlot(IngestHandle * handle)
{
    if (!handle)
        return getOrAssignBuildSlot(); /// fallback for callers without a cookie

    auto * ck = assert_cast<IngestCookie *>(handle);
    if (ThreadSlot * cached = ck->build_slot)
        return *cached;

    const std::scoped_lock lg(slot_mu);
    if (ck->build_slot)
        return *ck->build_slot; /// raced with another path on this cookie

    auto slot = std::make_unique<ThreadSlot>();
    slot->initBuildSide(build_spec);
    ThreadSlot & ref = *slot;
    slots.push_back(std::move(slot));
    num_slots_created.fetch_add(1, std::memory_order_release);
    ck->build_slot = &ref;
    return ref;
}

ThreadSlot & PartitionedHashJoin::resolveProbeSlot(IngestHandle * handle)
{
    if (!handle)
        return getOrAssignProbeSlot();

    auto * ck = assert_cast<IngestCookie *>(handle);
    if (ThreadSlot * cached = ck->probe_slot)
        return *cached;

    const std::scoped_lock lg(slot_mu);
    if (ck->probe_slot)
        return *ck->probe_slot;

    auto slot = std::make_unique<ThreadSlot>();
    slot->initProbeSide(build_spec.P);
    ThreadSlot & ref = *slot;
    slots.push_back(std::move(slot));
    num_slots_created.fetch_add(1, std::memory_order_release);
    ck->probe_slot = &ref;
    return ref;
}

bool PartitionedHashJoin::addBlockToJoin(const Block & block, bool check_limits)
{
    return addBlockToJoin(/*handle=*/nullptr, block, check_limits);
}

bool PartitionedHashJoin::addBlockToJoin(IngestHandle * handle, const Block & block, bool check_limits)
{
    if (build_done.load(std::memory_order_acquire))
        throw Exception(ErrorCodes::LOGICAL_ERROR, "PartitionedHashJoin::addBlockToJoin called after onBuildPhaseFinish");

    const size_t rows = block.rows();
    ThreadSlot & slot = resolveBuildSlot(handle);
    // Slot resolution is lock-free after first call; shuffle is on thread-private state.
    shuffleBlockIntoPartitions(block, build_spec, slot.build_parts, slot.build_cols, slot.scratch, slot.arena);

    total_build_rows.fetch_add(rows, std::memory_order_relaxed);
    total_build_bytes.fetch_add(block.bytes(), std::memory_order_relaxed);

    if (check_limits && table_join->sizeLimits().hasLimits())
        return table_join->sizeLimits().check(
            total_build_rows.load(std::memory_order_relaxed),
            total_build_bytes.load(std::memory_order_relaxed),
            "JOIN",
            ErrorCodes::SET_SIZE_LIMIT_EXCEEDED);
    return true;
}

bool PartitionedHashJoin::addBlockToJoin(IngestHandle * handle, const Block & block, size_t /*num_rows*/, bool check_limits)
{
    /// PHJ doesn't need the num_rows override (it doesn't drop columns mid-build).
    return addBlockToJoin(handle, block, check_limits);
}

void PartitionedHashJoin::checkTypesOfKeys(const Block & block) const
{
    JoinCommon::checkTypesOfKeys(
        block, table_join->getAllNames(JoinTableSide::Left), *right_sample, table_join->getAllNames(JoinTableSide::Right));
}

JoinResultPtr PartitionedHashJoin::joinBlock(Block block)
{
    return joinBlock(/*handle=*/nullptr, std::move(block));
}

JoinResultPtr PartitionedHashJoin::joinBlock(IngestHandle * handle, Block block)
{
    const size_t n = block.rows();
    if (n == 0)
    {
        /// This is the header-transformation call from JoiningTransform::transformHeader.
        /// We must return the correct output schema (left + right columns).
        /// Delegate to a lazily-created empty HashJoin that has the right schema.
        const std::scoped_lock lg(schema_hj_mu);
        if (!schema_hj)
            schema_hj = std::make_shared<HashJoin>(table_join, right_sample, any_take_last_row);
        return schema_hj->joinBlock(std::move(block));
    }

    ThreadSlot & slot = resolveProbeSlot(handle);
    // Slot resolution is lock-free after first call.

    const size_t P = build_spec.P;
    const uint64_t mask = static_cast<uint64_t>(P - 1);

    ShuffleScratch & scratch = slot.scratch;
    uint16_t * pids = scratch.pids.data();

    /// Process the block in chunks of at most kBatchMax rows. Each chunk:
    ///   1. Computes pids[] from the LEFT-side join keys for rows [row_start, row_start+n_batch).
    ///   2. Cuts the input block to that row range to produce a sub-block.
    ///   3. Scatters the sub-block into slot.probe_parts using the chunk's pids[].
    /// Scratch arrays (uint16_t) are sized exactly for kBatchMax, so no lazy growth.
    const size_t batch_max = ShuffleScratch::kBatchMax;
    const auto & key_names_left = table_join->getClauses().front().key_names_left;
    const size_t ncols = block.columns();

    size_t row_start = 0;
    while (row_start < n)
    {
        const size_t n_batch = std::min(batch_max, n - row_start);

        /// Phase 1 — pids[] for this chunk.
        std::fill(pids, pids + n_batch, static_cast<uint16_t>(0));
        bool first = true;
        for (const auto & key_name : key_names_left)
        {
            if (!block.has(key_name))
                continue;
            const auto & col = block.getByName(key_name);

            /// Materialise ColumnConst so getRawData() works — constant-folded keys
            /// (e.g. SELECT 42 AS k) arrive as ColumnConst and are not contiguous.
            const ColumnPtr col_full = col.column->convertToFullColumnIfConst();

            /// Peel Nullable: hash the inner data. NULL rows route to partition 0.
            const IColumn * raw_col = col_full.get();
            if (const auto * nullable_col = typeid_cast<const ColumnNullable *>(raw_col))
                raw_col = &nullable_col->getNestedColumn();

            const size_t eb = fixedElemBytes(col.type);
            if (eb == 0 || eb > 8)
                continue;

            const uint8_t * key_data = reinterpret_cast<const uint8_t *>(raw_col->getRawData().data()) + (row_start * eb);
            hashOneKeyIntoIds(key_data, eb, n_batch, mask, pids, first);
            first = false;
        }

        /// Phase 2 — slice block to [row_start, row_start+n_batch) and scatter.
        /// Whole-block fast path: avoid cut() when one batch covers the entire block.
        if (n_batch == n)
        {
            probeScatterBlock(block, pids, n_batch, P, slot.probe_parts);
        }
        else
        {
            ColumnsWithTypeAndName sub_cols;
            sub_cols.reserve(ncols);
            for (size_t ci = 0; ci < ncols; ++ci)
            {
                const auto & cwt = block.getByPosition(ci);
                sub_cols.emplace_back(cwt.column->cut(row_start, n_batch), cwt.type, cwt.name);
            }
            const Block sub_block(std::move(sub_cols));
            probeScatterBlock(sub_block, pids, n_batch, P, slot.probe_parts);
        }

        row_start += n_batch;
    }

    return IJoinResult::createFromBlock(block.cloneEmpty());
}

void PartitionedHashJoin::onBuildPhaseFinish()
{
    if (build_done.load(std::memory_order_relaxed))
        return;
    build_done.store(true, std::memory_order_release);

    if (build_spec.use_swwc)
    {
        for (auto & slot_ptr : slots)
        {
            if (!slot_ptr->build_initialised)
                continue;
            const size_t P = build_spec.P;
            uint8_t * scnt = slot_ptr->scratch.swwc_cnt.data();
            for (size_t p = 0; p < P; ++p)
            {
                if (scnt[p] == 0)
                    continue;
                for (auto & col : slot_ptr->build_cols)
                    col->drain_one(p, scnt[p]);
                scnt[p] = 0;
            }
        }
#if USE_MULTITARGET_CODE
        if (isArchSupported(TargetArch::x86_64_v4) || isArchSupported(TargetArch::x86_64_v3))
            _mm_sfence();
#endif
    }
}

IBlocksStreamPtr PartitionedHashJoin::getDelayedBlocks()
{
    /// DelayedJoinedBlocksTransform calls this repeatedly, expecting nullptr as the
    /// termination signal (same pattern as GraceHashJoin's bucket iterator).
    /// We have exactly one stream covering all P partitions — return it once, then nullptr.
    bool expected = false;
    if (!delayed_blocks_given.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        return nullptr;
    return std::make_shared<PartitionedHashJoinDelayedBlocks>(*this);
}

size_t PartitionedHashJoin::getTotalRowCount() const
{
    return total_build_rows.load(std::memory_order_relaxed);
}

size_t PartitionedHashJoin::getTotalByteCount() const
{
    return total_build_bytes.load(std::memory_order_relaxed);
}

bool PartitionedHashJoin::alwaysReturnsEmptySet() const
{
    /// For INNER / RIGHT / CROSS: if build side is empty, output is always empty.
    /// For LEFT / FULL: even with empty build, there are always left (or right) rows to emit.
    const auto k = table_join->kind();
    if (k == JoinKind::Left || k == JoinKind::Full)
        return false;
    return total_build_rows.load(std::memory_order_relaxed) == 0;
}

IBlocksStreamPtr
PartitionedHashJoin::getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 /*max_block_size*/) const
{
    /// Store headers from the pipeline's non_joined_stream_builder so the
    /// delayed-blocks worker can pass them to each per-partition HashJoin.
    const std::scoped_lock lg(non_joined_headers_mu_);
    if (!non_joined_headers_set_)
    {
        non_joined_left_header_ = left_sample_block;
        non_joined_result_header_ = result_sample_block;
        non_joined_headers_set_ = true;
    }
    return nullptr; /// actual emission happens per-partition in DelayedBlocks
}

bool PartitionedHashJoin::isSupportedByColumns(const Block & right_sample, const Names & key_names, const Names & kept_payload_names)
{
    return ::DB::isSupportedByColumns(right_sample, key_names, kept_payload_names);
}

}
