#include <Interpreters/HashJoin/HashJoinBuild.h>

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/HashJoin/HashJoinPartitionedBuild.h>
#include <Interpreters/HashJoin/HashJoinTable.h>
#include <Interpreters/HashJoin/JoinUsedFlags.h>
#include <Interpreters/Context.h>
#include <Interpreters/ProcessList.h>
#include <Interpreters/TableJoin.h>
#include <Interpreters/joinDispatch.h>
#include <Common/CurrentThread.h>
#include <Common/Exception.h>
#include <Common/ProfileEvents.h>
#include <Common/Stopwatch.h>
#include <Common/formatReadable.h>
#include <Common/logger_useful.h>
#include <base/scope_guard.h>

#include <algorithm>
#include <bit>

namespace ProfileEvents
{
extern const Event HashJoinPreallocatedElementsInHashTables;
extern const Event HashJoinBuildRounds;
}

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int QUERY_WAS_CANCELLED;
extern const int SET_SIZE_LIMIT_EXCEEDED;
}

namespace
{

/// A live estimate is refreshed once the waiting rows grew by this fraction since the last refresh.
constexpr size_t REFRESH_GROWTH_DIVISOR = 16;
/// The live test: fewer than one new key in this many rows means that keys have stopped arriving.
constexpr size_t REPEATS_ONLY_RATIO = 16;
/// The backstop takes this fraction of the budget.
constexpr size_t BACKSTOP_DIVISOR = 4;
/// The lazy sketch of one build stream starts when the byte bound reaches this fraction of a budget or a byte limit.
constexpr size_t LAZY_SKETCH_DIVISOR = 4;
/// Without an estimate, an `ALL` row-list arena is charged this per waiting row: 32 bytes for two rows of one key, twice for
/// the doubling of the arena's chunks.
constexpr size_t ROW_LIST_BYTES_PER_WAITING_ROW = 64;
/// With an estimate, the rows per key above which the row lists start to cost (the live estimate lags the rows by up to 1/16).
constexpr double ROW_LIST_FREE_RATIO = 1.2;
constexpr size_t ROW_LIST_NODE_BYTES = 64;
constexpr size_t ROW_LIST_ROWS_PER_NODE = 6;
constexpr size_t ARENA_INITIAL_BYTES = 4096;

const char * waitEndName(JoinWaitEnd reason)
{
    switch (reason)
    {
        case JoinWaitEnd::NotYet: return "not yet";
        case JoinWaitEnd::BuildFinish: return "build finish";
        case JoinWaitEnd::RowLimit: return "row limit";
        case JoinWaitEnd::ByteLimit: return "byte limit";
        case JoinWaitEnd::RepeatsOnly: return "repeats only";
        case JoinWaitEnd::RememberedVerdict: return "remembered verdict";
        case JoinWaitEnd::Backstop: return "backstop";
    }
    UNREACHABLE();
}

/// A partitioned build that ends at build finish reserves this much more than its share of the estimate per partition.
constexpr size_t RESERVE_MARGIN_NUMERATOR = 6;
constexpr size_t RESERVE_MARGIN_DENOMINATOR = 5;
/// The parallel layout fed by one stream has 256 partitions, each reserved to at least this many keys.
constexpr size_t ONE_STREAM_PARTITIONS = 256;
constexpr size_t ONE_STREAM_MIN_KEYS_PER_PARTITION = 128;

size_t zeroBufferBytes(size_t, bool)
{
    return 0;
}

/// What the grower of one partition map allocates for a number of keys; zero for a key type without partition maps.
HashJoinTable::BufferBytesForReserve partitionBufferBytes(const HashJoinTable & table)
{
    const auto res = table.partitionBufferBytesForReserve();
    return res ? res : &zeroBufferBytes;
}

/// A waiting record of the build stream `stream`.
struct WaitingRef
{
    size_t stream;
    WaitingBlockList::iterator it;
};

/// Every waiting record of every build stream, in arrival order (ascending `block_no`).
std::vector<WaitingRef> waitingInArrivalOrder(HashJoinBuildState & state)
{
    std::vector<WaitingRef> res;
    for (size_t s = 0; s < state.streams.size(); ++s)
        for (auto it = state.streams[s].waiting.begin(); it != state.streams[s].waiting.end(); ++it)
            res.push_back({s, it});
    std::ranges::sort(res, {}, [](const WaitingRef & w) { return w.it->block_no; });
    return res;
}

/// The bytes of a record, of the stored block it waits with, and of the block's holders.
size_t recordAndBlockBytes(const WaitingBlockRecord & record)
{
    return record.allocated_bytes + record.nullmap_bytes + (record.block_dropped ? 0 : record.stored->allocatedBytes());
}

}

size_t WaitingBlockRecord::computeAllocatedBytes() const
{
    size_t res = 0;
    for (size_t i = 0; i < key_holders.size(); ++i)
    {
        /// A key column that the stored block also holds costs nothing but the pointer.
        const bool shared_with_stored_block = stored && i < key_holder_sources.size() && key_holder_sources[i] >= 0
            && stored->columns.size() > static_cast<size_t>(key_holder_sources[i])
            && stored->columns[key_holder_sources[i]].get() == key_holders[i].get();
        if (!shared_with_stored_block)
            res += key_holders[i]->allocatedBytes();
    }
    for (const auto & clause : clauses)
    {
        if (clause.null_map_holder)
            res += clause.null_map_holder->allocatedBytes();
        res += clause.skip.allocated_bytes() + clause.route.allocated_bytes() + clause.key128.allocated_bytes()
            + clause.hash64.allocated_bytes();
    }
    return res;
}

size_t HashJoinBuildState::waitingRows() const
{
    size_t res = 0;
    for (const auto & stream : streams)
        res += stream.rows.load(std::memory_order_relaxed);
    return res;
}

void HashJoin::decideBuildKind(const JoinBuildContext & context)
{
    auto & state = *data->build_state;
    join_checks_limits = context.joinChecksLimits();
    /// A parallel layout fed by one stream (`max_threads = 1`, totals on the right side) is built by one thread, as are the
    /// serial layout, `key8` and the joins of wrappers.
    state.partitioned = build_layout.parallel && context.getNumStreams() > 1 && keyType() != Type::key8;
    state.serialize_inserts = build_layout.parallel && context.getNumStreams() > 1 && keyType() == Type::key8;

    const bool flag_per_row = needUsedFlagsForPerRightTableRow(table_join);
    const auto maps_kind = getMapsKind();
    bool single_row_maps = false;
    dispatchJoinShape(
        kind,
        strictness,
        maps_kind,
        [&]<JoinKind, JoinStrictness, typename Maps>() { single_row_maps = std::is_same_v<Maps, MapsOne>; });
    state.drops_blocks = (maps_kind == JoinMapsKind::Set && !mustKeepRightBlocks())
        || (single_row_maps && !anyTakeLastRow() && !flag_per_row);

    if (!wait_limits_from_wrapper)
        state.wait_limits = join_checks_limits ? table_join->sizeLimits() : SizeLimits{};
    state.can_end_early = state.drops_blocks || state.wait_limits.max_rows || state.wait_limits.max_bytes;

    if (state.drops_blocks)
    {
        /// A join that can spill has a budget; one without a spill budget takes a quarter of the query's hard memory limit.
        const size_t spill_budget = isUsedByAnotherAlgorithm() ? table_join->maxBytesBeforeExternalJoin() : 0;
        state.backstop_bytes = (spill_budget ? spill_budget : table_join->getMaxMemoryUsage()) / BACKSTOP_DIVISOR;
    }

    if (stats_collecting_params.build.isCollectionAndUseEnabled())
    {
        if (const auto hint = getSizeHint(stats_collecting_params.build))
        {
            state.cached_keys.store(hint->ht_size, std::memory_order_relaxed);
            state.remembered_repeats_dominate = state.drops_blocks && hint->repeats_dominate;
        }
    }

    for (auto & clause : state.clauses)
        clause.merged = std::make_unique<JoinKeySketch>();
}

void HashJoin::throwIfBuildFailed() const
{
    const auto & state = *data->build_state;
    if (state.build_failed.load(std::memory_order_acquire))
    {
        if (state.failure)
            std::rethrow_exception(state.failure);
        throw Exception(ErrorCodes::LOGICAL_ERROR, "The build of the hash join failed earlier");
    }
}

void HashJoin::markBuildFailed() noexcept
{
    auto & state = *data->build_state;
    /// The first failure is the one rethrown; `failure` is written before the flag is published.
    std::lock_guard lock(state.failure_mutex);
    if (state.build_failed.load(std::memory_order_relaxed))
        return;
    state.failure = std::current_exception();
    state.build_failed.store(true, std::memory_order_release);
}

void HashJoin::checkBuildCancelled() const
{
    const auto query_context = CurrentThread::tryGetQueryContext();
    if (const auto query_status = query_context ? query_context->getProcessListElementSafe() : nullptr)
    {
        /// Throws on `KILL QUERY`, and on `max_execution_time` with `timeout_overflow_mode = 'throw'`. With `break` it returns
        /// false: the build stops here, and `buildWaitingBlocks` catches this exception because `cancelled` is set.
        if (!query_status->checkTimeLimit())
        {
            data->build_state->cancelled.store(true, std::memory_order_relaxed);
            throw Exception(ErrorCodes::QUERY_WAS_CANCELLED, "The build of the hash join was cancelled by the time limit");
        }
    }
    /// TODO(l5): the cancel flag of the filling transform; `IJoin` does not pass it to the join yet.
}

void HashJoin::dropStoredBlock(WaitingBlockRecord & record, size_t stream)
{
    if (record.block_dropped)
        return;
    LOG_TRACE(log, "Skipping inserting block with {} rows", record.rows);
    /// A `Join` engine table has no build state.
    if (data->build_state)
        data->build_state->freed_a_block.store(true, std::memory_order_relaxed);
    data->subBytes(data->allocated_size, record.stored->allocatedBytes());
    /// A set map references no block at all, so its blocks are dropped though their rows took part in the join.
    if (getMapsKind() != JoinMapsKind::Set)
        data->rows_to_join.fetch_sub(record.rows, std::memory_order_relaxed);
    /// Null the index entry, so that a stale reference trips the chassert in `StoredColumnsIndex::at`.
    data->stored_columns_index->clearEntry(record.block_no);
    data->build_streams[stream].columns.erase(record.stored_it);
    record.stored = nullptr;
    record.block_dropped = true;
}

void HashJoin::releaseRecord(WaitingBlockList & list, WaitingBlockList::iterator it)
{
    auto & state = *data->build_state;
    data->subBytes(data->record_bytes, it->allocated_bytes);
    state.string_key_bytes.fetch_sub(it->string_key_bytes, std::memory_order_relaxed);
    state.waiting_bytes.fetch_sub(it->counted_waiting_bytes, std::memory_order_relaxed);
    list.erase(it);
}

void HashJoin::discardWaitingBlock(WaitingBlockList::iterator it, size_t stream)
{
    auto & state = *data->build_state;
    WaitingBlockRecord & record = *it;
    auto & build_stream = data->build_streams[stream];
    if (!record.block_dropped)
    {
        /// The holders point into the block; they go first.
        std::erase_if(
            build_stream.nullmaps,
            [&](const NullMapHolder & holder)
            {
                if (holder.columns != record.stored)
                    return false;
                data->subBytes(data->nullmaps_allocated_size, holder.allocatedBytes());
                return true;
            });
        data->subBytes(data->allocated_size, record.stored->allocatedBytes());
        data->stored_columns_index->clearEntry(record.block_no);
        build_stream.columns.erase(record.stored_it);
        record.stored = nullptr;
        record.block_dropped = true;
    }
    /// Today such a block is never read, so its rows are not counted either.
    data->rows_to_join.fetch_sub(record.rows, std::memory_order_relaxed);
    state.streams[stream].rows.fetch_sub(record.rows, std::memory_order_relaxed);
    releaseRecord(state.streams[stream].waiting, it);
}

size_t HashJoin::estimatedKeysOfClause(size_t clause_idx, const char ** source) const
{
    const auto & state = *data->build_state;
    const size_t rows = state.waitingRows();
    if (const size_t e = state.clauses[clause_idx].estimate.load(std::memory_order_relaxed))
    {
        if (source)
            *source = "sketch";
        return std::min(e, rows);
    }
    /// The cached count is the keys of all disjuncts.
    if (const size_t cached = state.cached_keys.load(std::memory_order_relaxed))
    {
        if (source)
            *source = "cache";
        return std::min(cached / state.clauses.size(), rows);
    }
    if (source)
        *source = "rows";
    return rows;
}

size_t HashJoin::estimatedKeys() const
{
    size_t res = 0;
    for (size_t clause_idx = 0; clause_idx < data->build_state->clauses.size(); ++clause_idx)
        res += estimatedKeysOfClause(clause_idx);
    return res;
}

size_t HashJoin::projectedTableBytes(bool every_row_new, bool include_built, double key_scale) const
{
    const auto & state = *data->build_state;
    const size_t rows = state.waitingRows();
    const bool robin_hood = table_join->enableRobinHoodHashForJoins();
    const bool partitioned = state.partitioned.load(std::memory_order_relaxed);

    size_t res = 0;
    size_t total_keys = 0;
    bool has_estimate = !every_row_new;
    for (size_t t = 0; t < data->tables.size(); ++t)
    {
        const auto & table = *data->tables[t];
        const char * source = "rows";
        size_t keys = rows;
        if (!every_row_new)
            keys = std::min(rows, static_cast<size_t>(static_cast<double>(estimatedKeysOfClause(t, &source)) * key_scale));
        has_estimate &= std::string_view(source) != "rows";
        total_keys += keys;
        if (!include_built && table.keys())
            continue;

        /// What the grower of the map that the join really builds allocates for these keys, from its first reserve on.
        size_t bytes = 0;
        if (partitioned)
        {
            if (const auto growth = table.bufferBytesForGrowth(/*partitioned_form=*/true))
            {
                const JoinPartitionPlan plan = choosePartitionPlan(
                    keys,
                    "rows",
                    rows,
                    build_layout.num_build_streams,
                    table_join->parallelHashJoinThreshold(),
                    build_layout.num_build_streams,
                    /*early=*/false,
                    robin_hood,
                    partitionBufferBytes(table));
                const size_t partitions = size_t{1} << plan.partition_bits;
                const size_t per_partition
                    = (keys * RESERVE_MARGIN_NUMERATOR + RESERVE_MARGIN_DENOMINATOR * partitions - 1) / (RESERVE_MARGIN_DENOMINATOR * partitions);
                bytes = partitions * growth(per_partition, per_partition, robin_hood);
            }
        }
        else if (const auto growth = build_layout.parallel ? table.bufferBytesForGrowth(/*partitioned_form=*/true) : nullptr)
        {
            /// The 256 partitions of the parallel layout fed by one stream grow from their first reserve; the fullest one is
            /// taken to hold 1.2 times the average share.
            const size_t reserve = std::max(one_stream_reserve_hint / ONE_STREAM_PARTITIONS, ONE_STREAM_MIN_KEYS_PER_PARTITION);
            const size_t per_partition = (keys * RESERVE_MARGIN_NUMERATOR + RESERVE_MARGIN_DENOMINATOR * ONE_STREAM_PARTITIONS - 1)
                / (RESERVE_MARGIN_DENOMINATOR * ONE_STREAM_PARTITIONS);
            bytes = ONE_STREAM_PARTITIONS * growth(per_partition, reserve, robin_hood);
        }
        else if (const auto single_growth = table.bufferBytesForGrowth(/*partitioned_form=*/false))
            /// The single map, LowCardinality maps included, grows from the reserve that the constructor made.
            bytes = single_growth(keys, one_stream_reserve_hint, /*robin_hood_allowed=*/false);
        /// A direct-addressed map has all its cells from the constructor.

        /// Only what goes beyond the buffer that the constructor allocated: the allocated bytes hold that already.
        res += bytes > initial_table_bytes[t] ? bytes - initial_table_bytes[t] : 0;
    }

    if (getMapsKind() == JoinMapsKind::All)
    {
        if (!has_estimate)
            res += rows * ROW_LIST_BYTES_PER_WAITING_ROW + ARENA_INITIAL_BYTES * data->build_streams.size();
        else if (total_keys && static_cast<double>(rows) / static_cast<double>(total_keys) > ROW_LIST_FREE_RATIO)
        {
            const size_t rows_per_key = rows / total_keys;
            const size_t nodes_per_key = rows_per_key < 2 ? 0 : 1 + (rows_per_key > 7 ? (rows_per_key - 7) / ROW_LIST_ROWS_PER_NODE + 1 : 0);
            res += total_keys * nodes_per_key * ROW_LIST_NODE_BYTES;
        }
    }

    /// The arena copies the key bytes of every new key; without an estimate twice, because its chunks double.
    const size_t string_bytes = state.string_key_bytes.load(std::memory_order_relaxed);
    res += has_estimate ? string_bytes : 2 * string_bytes;
    return res;
}

size_t HashJoin::waitByteBound() const
{
    return getTotalByteCount() + projectedTableBytes(/*every_row_new=*/true);
}

size_t HashJoin::getProjectedByteCount() const
{
    if (!data)
        return 0;
    const size_t allocated = getTotalByteCount();
    if (!data->build_state || !isWaiting())
        return allocated;
    return allocated + projectedTableBytes(/*every_row_new=*/false);
}

void HashJoin::setWaitLimits(const SizeLimits & limits)
{
    if (!data->build_state)
        return;
    data->build_state->wait_limits = limits;
    wait_limits_from_wrapper = true;
}

bool HashJoin::isWaiting() const
{
    const auto * state = data ? data->build_state.get() : nullptr;
    if (!state || state->finished)
        return false;
    return state->waitingRows() != 0 || !state->built_early;
}

bool HashJoin::builtEarly() const
{
    const auto * state = data ? data->build_state.get() : nullptr;
    return state && state->built_early;
}

HashJoin::BuildFit HashJoin::evaluateBuildFit(size_t budget)
{
    const auto & state = *data->build_state;
    grouped_budget = 0;
    const size_t base = getTotalByteCount();
    const size_t rows = state.waitingRows();
    const size_t tables = projectedTableBytes(/*every_row_new=*/false);
    /// A table of one stream grows as today, and today's check at build finish spills at the budget.
    if (!state.partitioned)
        return base + tables < budget ? BuildFit::Fits : BuildFit::MustSpill;

    /// The scatter copy: the key and a locator of 8 bytes per row. The disjuncts build one after the other, so the largest
    /// partition count divides.
    const size_t copy_bytes_per_row = data->tables.front()->cellSize() + sizeof(UInt64);
    const size_t copy = rows * copy_bytes_per_row;
    size_t partitions = 1;
    for (size_t clause_idx = 0; clause_idx < state.clauses.size(); ++clause_idx)
    {
        const auto plan = choosePartitionPlan(
            estimatedKeysOfClause(clause_idx),
            "rows",
            rows,
            build_layout.num_build_streams,
            table_join->parallelHashJoinThreshold(),
            build_layout.num_build_streams,
            /*early=*/false,
            table_join->enableRobinHoodHashForJoins(),
            partitionBufferBytes(*data->tables[clause_idx]));
        partitions = std::max(partitions, size_t{1} << plan.partition_bits);
    }
    const size_t at_once = base + std::max(copy + tables / partitions, tables + copy / partitions);
    size_t max_block_rows = 0;
    for (const auto & stream : state.streams)
        for (const auto & record : stream.waiting)
            max_block_rows = std::max(max_block_rows, record.rows);
    const size_t in_rounds = base + tables + max_block_rows * copy_bytes_per_row;

    if (at_once <= budget)
        return BuildFit::Fits;
    if (in_rounds <= budget && !state.built_early)
    {
        grouped_budget = budget;
        return BuildFit::Grouped;
    }
    return BuildFit::MustSpill;
}

void HashJoin::startSketch(size_t stream_idx)
{
    auto & state = *data->build_state;
    auto & stream = state.streams[stream_idx];
    if (stream.sketches.empty() || stream.sketches.front().load(std::memory_order_relaxed))
        return;
    stream.sketch_storage.resize(state.clauses.size());
    for (size_t clause_idx = 0; clause_idx < state.clauses.size(); ++clause_idx)
    {
        auto sketch = std::make_unique<JoinKeySketch>();
        for (const auto & record : stream.waiting)
            JoinPartitionedBuild::sketchClause(
                *data->tables[clause_idx],
                strictness == JoinStrictness::Asof,
                record.clauses[clause_idx],
                record.stored->selector,
                *sketch);
        stream.sketch_storage[clause_idx] = std::move(sketch);
    }
    /// Published after it fed the rows the stream already holds.
    for (size_t clause_idx = 0; clause_idx < state.clauses.size(); ++clause_idx)
        stream.sketches[clause_idx].store(stream.sketch_storage[clause_idx].get(), std::memory_order_release);
}

void HashJoin::liveTest(size_t rows)
{
    auto & state = *data->build_state;
    if (!state.drops_blocks || rows < table_join->parallelHashJoinThreshold())
    {
        for (auto & clause : state.clauses)
        {
            clause.prev_rows = rows;
            clause.prev_estimate = clause.estimate.load(std::memory_order_relaxed);
        }
        return;
    }

    /// The block is dropped only when no table points into it, so keys must have stopped arriving in every table.
    bool repeats_only = true;
    bool all_baselines = true;
    for (auto & clause : state.clauses)
    {
        const size_t e = clause.merged->size();
        if (clause.baseline_taken)
        {
            /// Signed: an estimate can step down a little.
            const Int64 new_keys = static_cast<Int64>(e) - static_cast<Int64>(clause.prev_estimate);
            const Int64 new_rows = static_cast<Int64>(rows - clause.prev_rows);
            repeats_only &= new_keys * static_cast<Int64>(REPEATS_ONLY_RATIO) < new_rows;
        }
        else
            all_baselines = false;
        clause.baseline_taken = true;
        clause.prev_rows = rows;
        clause.prev_estimate = e;
    }
    if (all_baselines && repeats_only)
        state.end_wait.store(true, std::memory_order_relaxed);
}

void HashJoin::maybeRefreshEstimate()
{
    auto & state = *data->build_state;
    if (state.waitingRows() * REFRESH_GROWTH_DIVISOR < state.last_refresh_rows.load(std::memory_order_relaxed) * (REFRESH_GROWTH_DIVISOR + 1))
        return;
    /// A try-lock. The losers skip: the holder merges every sketch.
    if (state.refreshing.exchange(true, std::memory_order_acquire))
        return;
    SCOPE_EXIT({ state.refreshing.store(false, std::memory_order_release); });

    /// Read again under the flag: another refresh may have run since the check above.
    const size_t last_rows = state.last_refresh_rows.load(std::memory_order_relaxed);
    const size_t rows = state.waitingRows();
    if (rows * REFRESH_GROWTH_DIVISOR < last_rows * (REFRESH_GROWTH_DIVISOR + 1))
        return;

    for (size_t clause_idx = 0; clause_idx < state.clauses.size(); ++clause_idx)
    {
        auto & clause = state.clauses[clause_idx];
        for (auto & stream : state.streams)
            if (auto * sketch = stream.sketches[clause_idx].load(std::memory_order_acquire); sketch && sketch->consumeChange())
                clause.merged->mergeLive(*sketch);
        /// The denominator is computed here, not on every insert.
        clause.estimate.store(clause.merged->size(), std::memory_order_relaxed);
    }
    liveTest(rows);
    state.last_refresh_rows.store(rows, std::memory_order_relaxed);
}

JoinWaitEnd HashJoin::checkWaitBounds() const
{
    const auto & state = *data->build_state;
    /// The verdict is about the first wait: the batches after an early build end by the other rules.
    if (state.remembered_repeats_dominate && !state.built_early)
        return JoinWaitEnd::RememberedVerdict;
    if (state.end_wait.load(std::memory_order_relaxed))
        return JoinWaitEnd::RepeatsOnly;

    const size_t rows = state.waitingRows();
    /// Each disjunct has at most one key per row, so before this point the row limit cannot be crossed.
    if (state.wait_limits.max_rows && data->keys() + rows * state.clauses.size() >= state.wait_limits.max_rows)
        return JoinWaitEnd::RowLimit;
    /// An upper bound: every waiting row counted as a new key, whatever an estimate says.
    if (state.wait_limits.max_bytes && waitByteBound() >= state.wait_limits.max_bytes)
        return JoinWaitEnd::ByteLimit;
    /// The bytes of the current batch only, not of the blocks that earlier rounds kept.
    if (state.drops_blocks && state.backstop_bytes && state.waiting_bytes.load(std::memory_order_relaxed) >= state.backstop_bytes)
        return JoinWaitEnd::Backstop;
    return JoinWaitEnd::NotYet;
}

JoinWaitEnd HashJoin::afterBlockWaits(size_t stream_idx)
{
    auto & state = *data->build_state;
    auto & stream = state.streams[stream_idx];

    if (!state.partitioned)
    {
        /// The lazy sketch of a table of one build stream: only when something reads the estimate.
        const bool has_reader = sketch_from_first_block || state.drops_blocks || wait_limits_from_wrapper
            || state.wait_limits.max_rows || state.wait_limits.max_bytes || isUsedByAnotherAlgorithm();
        if (has_reader && !stream.sketches.front().load(std::memory_order_relaxed))
        {
            const size_t rows = state.waitingRows();
            const size_t byte_bound = waitByteBound();
            const size_t budget = table_join->maxBytesBeforeExternalJoin();
            const bool start = sketch_from_first_block || rows >= table_join->parallelHashJoinThreshold()
                || (budget && byte_bound >= budget / LAZY_SKETCH_DIVISOR)
                || (state.wait_limits.max_bytes && byte_bound >= state.wait_limits.max_bytes / LAZY_SKETCH_DIVISOR);
            if (start)
                startSketch(stream_idx);
        }
        else if (stream.sketches.front().load(std::memory_order_relaxed))
        {
            /// The newest record is not sketched yet.
            const auto & record = stream.waiting.back();
            for (size_t clause_idx = 0; clause_idx < state.clauses.size(); ++clause_idx)
                JoinPartitionedBuild::sketchClause(
                    *data->tables[clause_idx],
                    strictness == JoinStrictness::Asof,
                    record.clauses[clause_idx],
                    record.stored->selector,
                    *stream.sketch_storage[clause_idx]);
        }
    }

    maybeRefreshEstimate();
    return checkWaitBounds();
}

bool HashJoin::checkSizeLimitsAfterBuild(size_t bytes)
{
    if (!join_checks_limits || wait_limits_from_wrapper)
        return true;
    updatePeakBuildBytes(bytes);
    if (table_join->sizeLimits().check(data->keys(), bytes, "JOIN", ErrorCodes::SET_SIZE_LIMIT_EXCEEDED))
        return true;
    data->build_state->stop_reading.store(true, std::memory_order_relaxed);
    return false;
}

bool HashJoin::replayOneStream()
{
    auto & state = *data->build_state;

    /// A grace bucket is fed through several stream ids, but serializes its inserts: replay in arrival order.
    const auto order = waitingInArrivalOrder(state);
    size_t pending_bytes = 0;
    for (const auto & [_, it] : order)
        pending_bytes += recordAndBlockBytes(*it);

    if (build_layout.parallel && !state.built_early)
    {
        /// 256 partitions, each reserved at its first insert.
        const size_t hint = one_stream_reserve_hint;
        for (auto & table : data->tables)
        {
            if (!table->isPartitioned() && table->type() != Type::key8)
            {
                table->usePartitions(8);
                table->setPartitionReserveOnFirstInsert(std::max(hint / ONE_STREAM_PARTITIONS, ONE_STREAM_MIN_KEYS_PER_PARTITION));
                ProfileEvents::increment(ProfileEvents::HashJoinPreallocatedElementsInHashTables, hint);
            }
        }
    }

    for (size_t i = 0; i < order.size(); ++i)
    {
        const auto & [stream, it] = order[i];
        WaitingBlockRecord & record = *it;
        const size_t record_block_bytes = recordAndBlockBytes(record);
        const bool referenced = insertRecord(record, stream);
        if (!referenced)
            dropStoredBlock(record, stream);
        state.streams[stream].rows.fetch_sub(record.rows, std::memory_order_relaxed);
        releaseRecord(state.streams[stream].waiting, it);
        pending_bytes -= record_block_bytes;

        checkBuildCancelled();

        /// The check reads the bytes of the blocks replayed so far, not of all waiting blocks.
        if (!checkSizeLimitsAfterBuild(getTotalByteCount() - std::min(pending_bytes, getTotalByteCount())))
        {
            /// `break`: today the blocks after the one that crossed the limit are never read, so they go.
            for (size_t j = i + 1; j < order.size(); ++j)
                discardWaitingBlock(order[j].it, order[j].stream);
            return false;
        }
    }
    return true;
}

bool HashJoin::runPartitionedRound(JoinWaitEnd reason)
{
    auto & state = *data->build_state;
    const bool early = reason != JoinWaitEnd::BuildFinish;

    const auto waiting = waitingInArrivalOrder(state);
    if (waiting.empty())
        return true;
    size_t rows = 0;
    std::vector<WaitingBlockRecord *> records;
    records.reserve(waiting.size());
    for (const auto & w : waiting)
    {
        rows += w.it->rows;
        records.push_back(&*w.it);
    }

    if (state.drops_blocks)
    {
        const size_t num_blocks = data->stored_columns_index->size();
        if (state.referenced_size < num_blocks)
        {
            auto referenced = std::make_unique<std::atomic<UInt8>[]>(num_blocks);
            for (size_t i = 0; i < num_blocks; ++i)
                referenced[i].store(i < state.referenced_size ? state.referenced[i].load(std::memory_order_relaxed) : 0, std::memory_order_relaxed);
            state.referenced = std::move(referenced);
            state.referenced_size = num_blocks;
        }
    }

    /// The final merge reads every sketch, whatever its change flag says.
    for (size_t clause_idx = 0; clause_idx < state.clauses.size(); ++clause_idx)
    {
        auto & clause = state.clauses[clause_idx];
        for (auto & stream : state.streams)
            if (auto * sketch = stream.sketches[clause_idx].load(std::memory_order_acquire))
                clause.merged->mergeLive(*sketch);
        clause.estimate.store(clause.merged->size(), std::memory_order_relaxed);
    }

    /// Each disjunct's table is planned from that disjunct's estimate.
    std::vector<JoinPartitionPlan> plans;
    size_t estimate = 0;
    if (!state.built_early)
    {
        for (size_t clause_idx = 0; clause_idx < state.clauses.size(); ++clause_idx)
        {
            const char * source = "rows";
            const size_t clause_estimate = estimatedKeysOfClause(clause_idx, &source);
            estimate += clause_estimate;
            plans.push_back(choosePartitionPlan(
                clause_estimate,
                source,
                rows,
                build_layout.num_build_streams,
                table_join->parallelHashJoinThreshold(),
                build_layout.num_build_streams,
                early,
                table_join->enableRobinHoodHashForJoins(),
                partitionBufferBytes(*data->tables[clause_idx])));
            logBuild(reason, &plans.back(), rows);
        }
    }

    /// A `Grouped` fit: rounds over consecutive ranges of blocks, each as large as the free budget allows. The first round
    /// creates and reserves the tables from the plan.
    const size_t budget = !early && !plans.empty() ? std::exchange(grouped_budget, 0) : 0;
    const size_t copy_bytes_per_row = data->tables.front()->cellSize() + sizeof(UInt64);

    const auto maps_kind = getMapsKind();
    size_t begin = 0;
    size_t inserted_rows = 0;
    while (begin < records.size())
    {
        size_t end = records.size();
        if (budget)
        {
            /// The final key count: the sketch's estimate while the inserted keys stay below 1.2 times it; above that the
            /// sketch was too low, and the keys inserted so far are extrapolated over all rows.
            const size_t inserted_keys = data->keys();
            double key_scale = 1.0;
            if (inserted_rows && estimate && inserted_keys * RESERVE_MARGIN_DENOMINATOR >= estimate * RESERVE_MARGIN_NUMERATOR)
                key_scale = static_cast<double>(inserted_keys) * static_cast<double>(rows) / static_cast<double>(inserted_rows)
                    / static_cast<double>(estimate);
            const size_t tables_to_come = projectedTableBytes(/*every_row_new=*/false, /*include_built=*/true, key_scale);
            const size_t used = getTotalByteCount() + (tables_to_come > data->bucket_bytes ? tables_to_come - data->bucket_bytes : 0);
            const size_t free_budget = budget > used ? budget - used : 0;
            size_t range_bytes = records[begin]->rows * copy_bytes_per_row;
            end = begin + 1;
            while (end < records.size() && range_bytes + records[end]->rows * copy_bytes_per_row <= free_budget)
                range_bytes += records[end++]->rows * copy_bytes_per_row;
        }
        const std::span<WaitingBlockRecord * const> range(records.data() + begin, end - begin);

        for (size_t clause_idx = 0; clause_idx < state.clauses.size(); ++clause_idx)
        {
            dispatchJoinShape(
                kind,
                strictness,
                maps_kind,
                [&]<JoinKind kind_, JoinStrictness strictness_, typename Maps>()
                {
                    const JoinPartitionedBuild::RoundSettings settings{
                        .any_take_last_row = anyTakeLastRow(),
                        .drops_blocks = state.drops_blocks,
                        .asof_type = getAsofType(),
                        .asof_inequality = getAsofInequality(),
                        .plan = begin == 0 && !plans.empty() ? &plans[clause_idx] : nullptr,
                        .max_threads = build_layout.num_build_streams,
                        .enable_software_prefetch = enableSoftwarePrefetch()};
                    const BuildResult result = JoinPartitionedBuild::buildRound<Maps>(
                        *data->tables[clause_idx],
                        clause_idx,
                        range,
                        settings,
                        state,
                        [&](size_t bytes) { data->addBytes(data->bucket_bytes, bytes); },
                        [&] { checkBuildCancelled(); });
                    if (!result.all_values_unique)
                        all_values_unique.store(false, std::memory_order_relaxed);
                });
        }

        ProfileEvents::increment(ProfileEvents::HashJoinBuildRounds);

        /// After the owners: drop the blocks that nothing points at, then free the records.
        for (size_t i = begin; i < end; ++i)
        {
            const auto & [stream, it] = waiting[i];
            WaitingBlockRecord & record = *it;
            const bool referenced = record.keeps_block || record.block_dropped
                || std::ranges::any_of(record.clauses, [](const auto & clause) { return clause.skipped_null_key; })
                || !state.drops_blocks || state.referenced[record.block_no].load(std::memory_order_relaxed);
            if (!referenced)
                dropStoredBlock(record, stream);
            inserted_rows += record.rows;
            state.streams[stream].rows.fetch_sub(record.rows, std::memory_order_relaxed);
            releaseRecord(state.streams[stream].waiting, it);
        }
        begin = end;
        updatePeakBuildBytes(getTotalByteCount());
    }

    /// The live test starts again for the next batch: its baseline is taken anew.
    for (auto & clause : state.clauses)
        clause.baseline_taken = false;
    state.end_wait.store(false, std::memory_order_relaxed);
    state.last_refresh_rows.store(0, std::memory_order_relaxed);

    /// One exact check of the size limits after every round, the last one included.
    return checkSizeLimitsAfterBuild(getTotalByteCount());
}

bool HashJoin::buildWaitingBlocks(JoinWaitEnd reason)
{
    auto & state = *data->build_state;
    if (state.wait_end == JoinWaitEnd::NotYet)
        state.wait_end = reason;

    bool ok = true;
    try
    {
        ok = state.partitioned ? runPartitionedRound(reason) : replayOneStream();
    }
    catch (...)
    {
        /// `timeout_overflow_mode = 'break'`: the build stops, and the executor ends the query before any probe.
        if (state.cancelled.load(std::memory_order_relaxed))
        {
            state.stop_reading.store(true, std::memory_order_relaxed);
            return false;
        }
        markBuildFailed();
        throw;
    }

    if (!state.partitioned && !state.built_early)
        logBuild(reason, nullptr, data->rows_to_join.load(std::memory_order_relaxed));
    if (reason != JoinWaitEnd::BuildFinish)
        state.built_early = true;
    return ok;
}

void HashJoin::logBuild(JoinWaitEnd reason, const JoinPartitionPlan * plan, size_t rows) const
{
    const auto & state = *data->build_state;
    if (plan)
        LOG_TRACE(
            log,
            "{}Partitioned build: P = {}, E = {} ({}), rows = {}, reserve = {} per partition, Robin Hood = {}, the wait ended at {}{}",
            instance_log_id,
            size_t{1} << plan->partition_bits,
            plan->estimate,
            plan->estimate_source,
            rows,
            plan->reserve_per_partition,
            plan->robin_hood,
            waitEndName(reason),
            state.backstop_bytes || !state.drops_blocks ? "" : ", no backstop");
    else
        LOG_TRACE(
            log,
            "{}{} build of one stream: rows = {}, the wait ended at {}{}",
            instance_log_id,
            build_layout.parallel ? "256-partition" : "Serial",
            rows,
            waitEndName(reason),
            state.backstop_bytes || !state.drops_blocks ? "" : ", no backstop");
}

}
