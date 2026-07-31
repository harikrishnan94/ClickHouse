#include <Interpreters/SpillingHashJoin.h>

#include <Interpreters/ConcurrentHashJoin.h>
#include <Interpreters/GraceHashJoin.h>
#include <Interpreters/TableJoin.h>
#include <Common/ProfileEvents.h>
#include <Common/logger_useful.h>

namespace ProfileEvents
{
extern const Event JoinSpillingHashJoinSwitchedToGraceJoin;
}

namespace DB
{

IInMemoryHashJoin & SpillingHashJoin::collectingJoin()
{
    chassert(in_memory_hash_join);
    return *in_memory_hash_join;
}

const IInMemoryHashJoin & SpillingHashJoin::collectingJoin() const
{
    chassert(in_memory_hash_join);
    return *in_memory_hash_join;
}

SpillingHashJoin::SpillingHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader left_sample_block_,
    SharedHeader right_sample_block_,
    TemporaryDataOnDiskScopePtr tmp_data_,
    size_t initial_num_buckets_,
    size_t max_num_buckets_,
    const StatsCollectingParams & stats_collecting_params_,
    bool any_take_last_row_,
    InMemoryHashJoinKind in_memory_kind_)
    : log(getLogger("SpillingHashJoin"))
    , table_join(std::move(table_join_))
    , left_sample_block(std::move(left_sample_block_))
    , right_sample_block(right_sample_block_->cloneEmpty())
    , tmp_data(std::move(tmp_data_))
    , initial_num_buckets(initial_num_buckets_)
    , max_num_buckets(max_num_buckets_)
    , any_take_last_row(any_take_last_row_)
    , max_bytes_before_external_join(table_join->maxBytesBeforeExternalJoin())
    , in_memory_kind(in_memory_kind_)
{
    in_memory_hash_join = createInMemoryHashJoin(
        in_memory_kind,
        table_join,
        right_sample_block_,
        any_take_last_row,
        /*reserve_num_=*/0,
        /*instance_id_=*/"",
        /*use_two_level_maps_=*/false,
        stats_collecting_params_);
}

SpillingHashJoin::SpillingHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader left_sample_block_,
    SharedHeader right_sample_block_,
    TemporaryDataOnDiskScopePtr tmp_data_,
    size_t initial_num_buckets_,
    size_t max_num_buckets_,
    size_t concurrent_slots_,
    const StatsCollectingParams & stats_collecting_params_,
    bool any_take_last_row_,
    InMemoryHashJoinKind in_memory_kind_)
    : log(getLogger("SpillingHashJoin"))
    , table_join(std::move(table_join_))
    , left_sample_block(std::move(left_sample_block_))
    , right_sample_block(right_sample_block_->cloneEmpty())
    , tmp_data(std::move(tmp_data_))
    , initial_num_buckets(initial_num_buckets_)
    , max_num_buckets(max_num_buckets_)
    , any_take_last_row(any_take_last_row_)
    , max_bytes_before_external_join(table_join->maxBytesBeforeExternalJoin())
    , in_memory_kind(in_memory_kind_)
{
    concurrent_join = std::make_shared<ConcurrentHashJoin>(
        table_join,
        concurrent_slots_,
        right_sample_block_,
        stats_collecting_params_,
        any_take_last_row,
        max_bytes_before_external_join);
    supports_parallel_non_joined_blocks_processing = concurrent_join->supportParallelNonJoinedBlocksProcessing();
}

SpillingHashJoin::~SpillingHashJoin() = default;

void SpillingHashJoin::tryConvertSlots()
{
    chassert(concurrent_join);
    chassert(grace_join);

    const auto total_slots = concurrent_join->getNumSlots();

    /// Fast path: all slots already converted.
    if (next_slot_to_convert.load(std::memory_order_acquire) >= total_slots)
        return;

    while (true)
    {
        size_t slot = next_slot_to_convert.fetch_add(1);
        if (slot >= total_slots)
            break;

        auto blocks = concurrent_join->releaseSlotBlocks(slot);
        while (!blocks.empty())
        {
            grace_join->addBlockToJoin(blocks.front(), /*check_limits=*/false);
            blocks.pop_front();
        }
    }
}

std::string SpillingHashJoin::getName() const
{
    static constexpr auto name_format = "SpillingHashJoin({})";

    if (concurrent_join)
        return fmt::format(name_format, concurrent_join->getName());

    return fmt::format(name_format, in_memory_hash_join->getName());
}

bool SpillingHashJoin::addBlockToJoin(const Block & block, bool check_limits)
{
    /// Fast path: already switched to GraceHashJoin (no lock needed).
    if (state.load(std::memory_order_acquire) != State::COLLECTING)
    {
        /// Help convert one ConcurrentHashJoin slot while in GRACE_HASH_JOIN state.
        if (concurrent_join)
            tryConvertSlots();
        return chosen_join->addBlockToJoin(block, check_limits);
    }

    /// The hash table buffer grows in power-of-two steps. Doubling from X to 2X allocates the new
    /// buffer while the old one is still alive, transiently using 3X memory. We must trigger the
    /// switch BEFORE the inner `addBlockToJoin` runs (and possibly doubles the buffer); a check
    /// that runs after the call would race with the doubling and observe the OOM only as an
    /// allocator exception. Threshold is half of `max_bytes_before_external_join` so that after
    /// the switch the live buffer (already at half) plus the conversion peak still fit under the
    /// configured cap.
    if (concurrent_join)
    {
        if (concurrent_join->getTotalByteCount() * 2 >= max_bytes_before_external_join)
            switchToGraceHashJoin();
    }
    else
    {
        if (collectingJoin().getTotalByteCount() * 2 >= max_bytes_before_external_join)
            switchToGraceHashJoin();
    }

    /// Re-check: we may have just switched.
    if (state.load(std::memory_order_acquire) != State::COLLECTING)
    {
        if (concurrent_join)
            tryConvertSlots();
        return chosen_join->addBlockToJoin(block, check_limits);
    }

    if (concurrent_join)
    {
        /// Shared lock: multiple threads add to ConcurrentHashJoin concurrently.
        std::shared_lock lock(switch_mutex);

        /// Re-check: another thread may have switched while we waited for the lock.
        if (state.load(std::memory_order_acquire) != State::COLLECTING)
            return chosen_join->addBlockToJoin(block, check_limits);

        return concurrent_join->addBlockToJoin(block, check_limits);
    }

    /// Single-thread in-memory hash join path.
    return collectingJoin().addBlockToJoin(block, check_limits);
}

void SpillingHashJoin::switchToGraceHashJoin()
{
    const auto print_threshold_reached_log = [this](const JoinPtr & join, std::string_view join_name)
    {
        LOG_DEBUG(
            log,
            "Memory spill threshold reached with {} ({} bytes, {} rows), switching to GraceHashJoin",
            join_name,
            join->getTotalByteCount(),
            join->getTotalRowCount());
    };
    if (concurrent_join)
    {
        {
            /// Exclusive lock: waits for all in-flight `addBlockToJoin` (shared lock holders)
            /// to complete. After this, no thread is inside `ConcurrentHashJoin::addBlockToJoin`.
            std::unique_lock lock(switch_mutex);

            /// Re-check: another thread may have already switched.
            if (state.load(std::memory_order_relaxed) != State::COLLECTING)
                return;

            ProfileEvents::increment(ProfileEvents::JoinSpillingHashJoinSwitchedToGraceJoin);

            print_threshold_reached_log(concurrent_join, "ConcurrentHashJoin");

            /// Create GraceHashJoin.
            grace_join = std::make_shared<GraceHashJoin>(
                initial_num_buckets,
                max_num_buckets,
                table_join,
                left_sample_block,
                std::make_shared<const Block>(right_sample_block),
                tmp_data,
                any_take_last_row,
                max_bytes_before_external_join,
                in_memory_kind);
            grace_join->initialize(*left_sample_block);
            chosen_join = grace_join;

            /// Set state BEFORE releasing the lock so new `addBlockToJoin` calls
            /// see GRACE_HASH_JOIN and go directly to `grace_join`.
            state.store(State::GRACE_HASH_JOIN, std::memory_order_release);
        }
        /// Convert ConcurrentHashJoin slots into GraceHashJoin.
        /// Other build-phase threads will also help via `addBlockToJoin`.
        tryConvertSlots();
        return;
    }

    print_threshold_reached_log(in_memory_hash_join, in_memory_hash_join->getName());
    /// Single-thread path: extract from in-memory hash join, feed to GraceHashJoin.
    ProfileEvents::increment(ProfileEvents::JoinSpillingHashJoinSwitchedToGraceJoin);
    BlocksList right_blocks = in_memory_hash_join->releaseJoinedBlocks(/*restructure=*/false);

    chosen_join = std::make_shared<GraceHashJoin>(
        initial_num_buckets,
        max_num_buckets,
        table_join,
        left_sample_block,
        std::make_shared<const Block>(right_sample_block),
        tmp_data,
        any_take_last_row,
        max_bytes_before_external_join,
        in_memory_kind);

    chosen_join->initialize(*left_sample_block);

    /// Drain extracted blocks into GraceHashJoin one by one,
    /// freeing each after insertion to limit peak memory.
    while (!right_blocks.empty())
    {
        chosen_join->addBlockToJoin(right_blocks.front(), /*check_limits=*/false);
        right_blocks.pop_front();
    }

    state.store(State::GRACE_HASH_JOIN, std::memory_order_release);
}

void SpillingHashJoin::onBuildPhaseFinish()
{
    if (state.load(std::memory_order_acquire) == State::COLLECTING)
    {
        /// Safety net for the terminal block: the proactive pre-insert check in `addBlockToJoin`
        /// fires only on subsequent calls. If the very last block pushed total bytes past
        /// `max_bytes_before_external_join` without a follow-up insert to trigger the switch,
        /// promote it to `GraceHashJoin` here so the configured cap is honored.
        const size_t total_bytes = concurrent_join ? concurrent_join->getTotalByteCount() : collectingJoin().getTotalByteCount();
        if (total_bytes >= max_bytes_before_external_join)
        {
            switchToGraceHashJoin();
        }
        else if (concurrent_join)
        {
            LOG_DEBUG(
                log,
                "All blocks fit in memory ({} bytes, {} rows), promoting ConcurrentHashJoin",
                total_bytes,
                concurrent_join->getTotalRowCount());
            chosen_join = concurrent_join;
            state.store(State::IN_MEMORY_JOIN, std::memory_order_release);
        }
        else
        {
            LOG_DEBUG(
                log,
                "All blocks fit in memory ({} bytes, {} rows), promoting {}",
                total_bytes,
                collectingJoin().getTotalRowCount(),
                collectingJoin().getName());
            chosen_join = in_memory_hash_join;
            state.store(State::IN_MEMORY_JOIN, std::memory_order_release);
        }
    }

    chosen_join->onBuildPhaseFinish();
}

void SpillingHashJoin::setEnableLazyColumnsIndexing(bool value)
{
    if (in_memory_hash_join)
        in_memory_hash_join->setEnableLazyColumnsIndexing(value);
    if (concurrent_join)
        concurrent_join->setEnableLazyColumnsIndexing(value);
}

void SpillingHashJoin::checkTypesOfKeys(const Block & block) const
{
    if (concurrent_join)
        concurrent_join->checkTypesOfKeys(block);
    else
        collectingJoin().checkTypesOfKeys(block);
}

void SpillingHashJoin::initialize(const Block & sample_block)
{
    left_sample_block = std::make_shared<const Block>(sample_block.cloneEmpty());
    if (!concurrent_join)
        collectingJoin().initialize(sample_block);
}

JoinResultPtr SpillingHashJoin::joinBlock(Block block)
{
    /// During header computation (transformHeader), `joinBlock` is called with an empty block
    /// before any data is added. Delegate to the appropriate join in COLLECTING state.
    if (state.load(std::memory_order_acquire) == State::COLLECTING)
    {
        if (concurrent_join)
            return concurrent_join->joinBlock(std::move(block));
        return collectingJoin().joinBlock(std::move(block));
    }

    return chosen_join->joinBlock(std::move(block));
}

void SpillingHashJoin::setTotals(const Block & block)
{
    std::lock_guard lock(totals_mutex);
    IJoin::setTotals(block);
}

const Block & SpillingHashJoin::getTotals() const
{
    std::lock_guard lock(totals_mutex);
    return IJoin::getTotals();
}

size_t SpillingHashJoin::getTotalRowCount() const
{
    if (state.load(std::memory_order_acquire) == State::COLLECTING)
    {
        if (concurrent_join)
            return concurrent_join->getTotalRowCount();
        return collectingJoin().getTotalRowCount();
    }
    return chosen_join->getTotalRowCount();
}

size_t SpillingHashJoin::getTotalByteCount() const
{
    if (state.load(std::memory_order_acquire) == State::COLLECTING)
    {
        if (concurrent_join)
            return concurrent_join->getTotalByteCount();
        return collectingJoin().getTotalByteCount();
    }
    return chosen_join->getTotalByteCount();
}

bool SpillingHashJoin::alwaysReturnsEmptySet() const
{
    if (state.load(std::memory_order_acquire) == State::COLLECTING)
    {
        if (concurrent_join)
            return concurrent_join->alwaysReturnsEmptySet();
        return collectingJoin().alwaysReturnsEmptySet();
    }
    return chosen_join->alwaysReturnsEmptySet();
}

bool SpillingHashJoin::supportParallelNonJoinedBlocksProcessing() const
{
    return supports_parallel_non_joined_blocks_processing;
}

bool SpillingHashJoin::isParallelNonJoinedProcessingEnabled() const
{
    return state == State::IN_MEMORY_JOIN && supports_parallel_non_joined_blocks_processing
        && chosen_join->supportParallelNonJoinedBlocksProcessing();
}

IBlocksStreamPtr
SpillingHashJoin::getNonJoinedBlocks(const Block & left_sample_block_, const Block & result_sample_block, UInt64 max_block_size) const
{
    chassert(chosen_join);
    return chosen_join->getNonJoinedBlocks(left_sample_block_, result_sample_block, max_block_size);
}

IBlocksStreamPtr SpillingHashJoin::getNonJoinedBlocks(
    const Block & left_sample_block_, const Block & result_sample_block, UInt64 max_block_size, size_t stream_idx, size_t num_streams) const
{
    chassert(chosen_join);
    return chosen_join->getNonJoinedBlocks(left_sample_block_, result_sample_block, max_block_size, stream_idx, num_streams);
}

IBlocksStreamPtr SpillingHashJoin::getDelayedBlocks()
{
    chassert(chosen_join);
    return chosen_join->getDelayedBlocks();
}

}
