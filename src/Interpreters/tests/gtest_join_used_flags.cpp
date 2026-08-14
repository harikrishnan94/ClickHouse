#include <Interpreters/HashJoin/JoinUsedFlags.h>
#include <Interpreters/HashJoin/ScatteredBlock.h>

#include <gtest/gtest.h>

using namespace DB;

namespace
{
/// Any (KIND, STRICTNESS, prefer_use_maps_all) with MapGetter::flagged == true works: the
/// per-row-flags path does not depend on which one is chosen.
constexpr auto KIND = JoinKind::Full;
constexpr auto STRICTNESS = JoinStrictness::All;
constexpr bool PREFER_MAPS_ALL = true;
}

TEST(JoinUsedFlags, AllOffsetFlagsSetCountsOccupiedKeys)
{
    JoinStuff::JoinUsedFlags flags;
    flags.per_offset_flags = JoinStuff::JoinUsedFlags::UsedFlagsForColumns(8);
    flags.setUnsetOffsetCount(3);

    EXPECT_FALSE(flags.allOffsetFlagsSet());

    flags.setUsed<true, false>(0, 0, 1);
    EXPECT_FALSE(flags.allOffsetFlagsSet());

    /// Duplicate mark of the same offset must not underflow the unused-key count.
    flags.setUsed<true, false>(0, 0, 1);
    EXPECT_FALSE(flags.allOffsetFlagsSet());

    flags.setUsed<true, false>(0, 0, 2);
    const bool first_once = flags.setUsedOnce<true, false>(0, 0, 4);
    EXPECT_TRUE(first_once);
    EXPECT_TRUE(flags.allOffsetFlagsSet());

    const bool second_once = flags.setUsedOnce<true, false>(0, 0, 4);
    EXPECT_FALSE(second_once);
    EXPECT_TRUE(flags.allOffsetFlagsSet());
}

TEST(JoinUsedFlags, AllOffsetFlagsSetEmptyCount)
{
    JoinStuff::JoinUsedFlags flags;
    EXPECT_TRUE(flags.allOffsetFlagsSet());
    flags.setUnsetOffsetCount(0);
    EXPECT_TRUE(flags.allOffsetFlagsSet());
}

/// Regression test for D-39: each build worker must keep its own pending per-row-flags list, so
/// concurrent `reinit` calls from different workers never mutate a shared container.
TEST(JoinUsedFlags, PendingPerRowFlagsMergeAcrossWorkers)
{
    JoinStuff::JoinUsedFlags flags;
    flags.setPendingFlagWorkers(/*num_workers=*/2);

    /// Worker 0 registers block 0 (3 rows, fully in its own shard's selector).
    flags.reinit<KIND, STRICTNESS, PREFER_MAPS_ALL>(/*worker_id=*/0, /*block_no=*/0, /*rows=*/3, ScatteredBlock::Selector(3));
    /// Worker 1 registers block 1 (2 rows, fully in its own shard's selector).
    flags.reinit<KIND, STRICTNESS, PREFER_MAPS_ALL>(/*worker_id=*/1, /*block_no=*/1, /*rows=*/2, ScatteredBlock::Selector(2));

    flags.finalizePerRowFlags(/*num_blocks=*/2);

    EXPECT_FALSE(flags.getUsedSafe(0, 0));
    EXPECT_FALSE(flags.getUsedSafe(0, 1));
    EXPECT_FALSE(flags.getUsedSafe(0, 2));
    EXPECT_FALSE(flags.getUsedSafe(1, 0));
    EXPECT_FALSE(flags.getUsedSafe(1, 1));

    /// A second finalize (e.g. if onBuildPhaseFinish ran twice) must be a no-op, not a re-throw.
    flags.finalizePerRowFlags(/*num_blocks=*/2);
    EXPECT_FALSE(flags.getUsedSafe(0, 0));
}

TEST(JoinUsedFlags, PendingPerRowFlagsMarksRowsOutsideSelectorAsUsed)
{
    JoinStuff::JoinUsedFlags flags;
    flags.setPendingFlagWorkers(/*num_workers=*/1);

    /// Rows 1 and 3 belong to this shard's selector; rows 0 and 2 belong to another shard and
    /// must come out pre-marked used so they are not emitted twice by RIGHT/FULL non-joined scan.
    auto indexes = ScatteredBlock::Selector::Indexes::create();
    indexes->insertValue(1);
    indexes->insertValue(3);
    flags.reinit<KIND, STRICTNESS, PREFER_MAPS_ALL>(
        /*worker_id=*/0, /*block_no=*/0, /*rows=*/4, ScatteredBlock::Selector(std::move(indexes)));

    flags.finalizePerRowFlags(/*num_blocks=*/1);

    EXPECT_TRUE(flags.getUsedSafe(0, 0));
    EXPECT_FALSE(flags.getUsedSafe(0, 1));
    EXPECT_TRUE(flags.getUsedSafe(0, 2));
    EXPECT_FALSE(flags.getUsedSafe(0, 3));
}

TEST(JoinUsedFlags, PendingPerRowFlagsDuplicateBlockNoAcrossWorkersThrows)
{
    JoinStuff::JoinUsedFlags flags;
    flags.setPendingFlagWorkers(/*num_workers=*/2);

    /// Two workers claiming the same block_no is a logic bug (block_no must be assigned once,
    /// by `StoredColumnsIndex::add`); finalize must detect it instead of silently overwriting.
    flags.reinit<KIND, STRICTNESS, PREFER_MAPS_ALL>(/*worker_id=*/0, /*block_no=*/5, /*rows=*/1, ScatteredBlock::Selector(1));
    flags.reinit<KIND, STRICTNESS, PREFER_MAPS_ALL>(/*worker_id=*/1, /*block_no=*/5, /*rows=*/1, ScatteredBlock::Selector(1));

    EXPECT_THROW(flags.finalizePerRowFlags(/*num_blocks=*/6), Exception);
}
