#include <Interpreters/HashJoin/JoinUsedFlags.h>

#include <gtest/gtest.h>

using namespace DB;

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
