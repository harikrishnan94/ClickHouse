#include <Interpreters/PartitionedHashJoin/PartitionedHashJoin.h>

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/TableJoin.h>
#include <Common/ElapsedTimeProfileEventIncrement.h>
#include <Common/ProfileEvents.h>
#include <Common/logger_useful.h>

namespace ProfileEvents
{
extern const Event PartitionedHashJoinBuildMicroseconds;
extern const Event PartitionedHashJoinProbeMicroseconds;
extern const Event PartitionedHashJoinPartitions;
extern const Event PartitionedHashJoinLeafRows;
}

namespace DB
{

namespace
{

/// Accumulates the time spent producing result blocks into `event`. `joinBlock` only sets up
/// a lazy result; the actual matching runs inside `IJoinResult::next`, so the probe time must
/// be accounted there.
class TimedJoinResult : public IJoinResult
{
public:
    TimedJoinResult(JoinResultPtr result_, ProfileEvents::Event event_)
        : result(std::move(result_))
        , event(event_)
    {
    }

    JoinResultBlock next() override
    {
        ProfileEventTimeIncrement<Microseconds> watch(event);
        return result->next();
    }

private:
    JoinResultPtr result;
    ProfileEvents::Event event;
};

}

PartitionedHashJoin::PartitionedHashJoin(std::shared_ptr<TableJoin> table_join_, SharedHeader right_sample_block_, bool any_take_last_row_)
    : table_join(std::move(table_join_))
    , right_sample_block(std::move(right_sample_block_))
    , any_take_last_row(any_take_last_row_)
    , leaf_join(std::make_unique<HashJoin>(table_join, right_sample_block, any_take_last_row))
    , log(getLogger("PartitionedHashJoin"))
{
}

PartitionedHashJoin::~PartitionedHashJoin() = default;

bool PartitionedHashJoin::isSupported(const TableJoin & table_join)
{
    /// The supported set is deliberately narrow while the partitioned build/probe paths are
    /// being brought up: INNER/LEFT ALL equi-joins with a single conjunction of keys and no
    /// extra ON conditions. Everything else is planned with another algorithm.
    if (!isInnerOrLeft(table_join.kind()))
        return false;
    if (table_join.strictness() != JoinStrictness::All)
        return false;
    if (!table_join.oneDisjunct())
        return false;
    if (table_join.isSpecialStorage())
        return false;
    if (table_join.getMixedJoinExpression())
        return false;

    const auto & clause = table_join.getOnlyClause();
    if (clause.on_filter_condition_left || clause.on_filter_condition_right || !clause.analyzer_left_filter_condition_column_name.empty()
        || !clause.analyzer_right_filter_condition_column_name.empty())
        return false;

    return true;
}

const TableJoin & PartitionedHashJoin::getTableJoin() const
{
    return *table_join;
}

bool PartitionedHashJoin::addBlockToJoin(const Block & block, bool check_limits)
{
    ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);
    ProfileEvents::increment(ProfileEvents::PartitionedHashJoinLeafRows, block.rows());
    return leaf_join->addBlockToJoin(block, check_limits);
}

void PartitionedHashJoin::checkTypesOfKeys(const Block & block) const
{
    leaf_join->checkTypesOfKeys(block);
}

JoinResultPtr PartitionedHashJoin::joinBlock(Block block)
{
    JoinResultPtr result;
    {
        ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinProbeMicroseconds);
        result = leaf_join->joinBlock(std::move(block));
    }
    return std::make_unique<TimedJoinResult>(std::move(result), ProfileEvents::PartitionedHashJoinProbeMicroseconds);
}

size_t PartitionedHashJoin::getTotalRowCount() const
{
    return leaf_join->getTotalRowCount();
}

size_t PartitionedHashJoin::getTotalByteCount() const
{
    return leaf_join->getTotalByteCount();
}

bool PartitionedHashJoin::alwaysReturnsEmptySet() const
{
    return leaf_join->alwaysReturnsEmptySet();
}

void PartitionedHashJoin::onBuildPhaseFinish()
{
    ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);

    /// The partition plan barrier. Only the degenerate plan exists so far: one leaf, already
    /// fully built by `addBlockToJoin`. The HLL-based bits selection will land here, moving
    /// the heavy work (scatter, hash-table allocation, leaf builds) into `runPostBuildPhase`.
    bits = 0;
    partitions = 1;
    ProfileEvents::increment(ProfileEvents::PartitionedHashJoinPartitions, partitions);

    leaf_join->onBuildPhaseFinish();

    LOG_TRACE(log, "Partition plan: bits = {}, partitions = {}, {} keys in the leaf", bits, partitions, leaf_join->getTotalRowCount());
}

void PartitionedHashJoin::runPostBuildPhase()
{
    ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);

    /// The route-driven scatter, the contiguous hash-table allocation, and the per-leaf builds
    /// run here once the partitioned build lands. The degenerate plan has nothing left to do:
    /// the single leaf was filled during `addBlockToJoin`.
    ///
    /// The leaf's own single-map post-build optimizations (`tryRerangeRightTableData`,
    /// `tryConvertToFixedHashMap`, runtime-filter publishing) are deliberately not run: they
    /// assume the data stays in one plain `HashJoin` map, which stops being true once the
    /// build is partitioned.
}

IBlocksStreamPtr
PartitionedHashJoin::getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const
{
    return leaf_join->getNonJoinedBlocks(left_sample_block, result_sample_block, max_block_size);
}

bool PartitionedHashJoin::isCloneSupported() const
{
    return getTotals().empty() && getTotalRowCount() == 0;
}

std::shared_ptr<IJoin>
PartitionedHashJoin::clone(const std::shared_ptr<TableJoin> & table_join_, SharedHeader, SharedHeader right_sample_block_) const
{
    return std::make_shared<PartitionedHashJoin>(table_join_, right_sample_block_, any_take_last_row);
}

std::shared_ptr<IJoin>
PartitionedHashJoin::cloneNoParallel(const std::shared_ptr<TableJoin> & table_join_, SharedHeader, SharedHeader right_sample_block_) const
{
    return std::make_shared<HashJoin>(table_join_, right_sample_block_, any_take_last_row);
}

void PartitionedHashJoin::setEnableLazyColumnsIndexing(bool value)
{
    leaf_join->setEnableLazyColumnsIndexing(value);
}

}
