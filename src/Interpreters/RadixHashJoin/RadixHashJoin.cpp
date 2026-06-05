#include <Interpreters/RadixHashJoin/RadixHashJoin.h>

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/TableJoin.h>

namespace DB
{

RadixHashJoin::RadixHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader right_sample_block_,
    size_t max_threads_,
    std::optional<UInt64> rhs_size_estimation_,
    UInt64 max_partitions_per_pass_,
    bool any_take_last_row_)
    : table_join(std::move(table_join_))
    , right_sample_block(right_sample_block_)
    , max_threads(max_threads_)
    , rhs_size_estimation(rhs_size_estimation_)
    , max_partitions_per_pass(max_partitions_per_pass_)
    , hash_join(std::make_unique<HashJoin>(table_join, right_sample_block_, any_take_last_row_))
{
}

RadixHashJoin::~RadixHashJoin() = default;

const TableJoin & RadixHashJoin::getTableJoin() const
{
    return *table_join;
}

bool RadixHashJoin::addBlockToJoin(const Block & block, bool check_limits)
{
    /// P0: a single shared HashJoin is filled from parallel build threads; serialise the inserts.
    std::lock_guard lock(add_block_mutex);
    return hash_join->addBlockToJoin(block, check_limits);
}

bool RadixHashJoin::addBlockToJoin(const Block & block, size_t num_rows, bool check_limits)
{
    std::lock_guard lock(add_block_mutex);
    return hash_join->addBlockToJoin(block, num_rows, check_limits);
}

void RadixHashJoin::checkTypesOfKeys(const Block & block) const
{
    hash_join->checkTypesOfKeys(block);
}

JoinResultPtr RadixHashJoin::joinBlock(Block block)
{
    /// Probe is read-only over the built data and may run from many streams concurrently.
    return hash_join->joinBlock(std::move(block));
}

size_t RadixHashJoin::getTotalRowCount() const
{
    return hash_join->getTotalRowCount();
}

size_t RadixHashJoin::getTotalByteCount() const
{
    return hash_join->getTotalByteCount();
}

bool RadixHashJoin::alwaysReturnsEmptySet() const
{
    return hash_join->alwaysReturnsEmptySet();
}

IBlocksStreamPtr RadixHashJoin::getNonJoinedBlocks(
    const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const
{
    /// Inner join only in v1: there are no non-joined right rows.
    return hash_join->getNonJoinedBlocks(left_sample_block, result_sample_block, max_block_size);
}

void RadixHashJoin::onBuildPhaseFinish()
{
    hash_join->onBuildPhaseFinish();
}

bool RadixHashJoin::hasPostBuildPhase() const
{
    return hash_join->hasPostBuildPhase();
}

void RadixHashJoin::runPostBuildPhase()
{
    hash_join->runPostBuildPhase();
}

}
