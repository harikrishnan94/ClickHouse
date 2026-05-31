#include <thread>
#include <vector>

#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <Core/Names.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/PartitionedHashConfig.h>
#include <Interpreters/PartitionedHashJoin.h>
#include <Interpreters/TableJoin.h>

#include <gtest/gtest.h>

using namespace DB;

namespace
{

/// Right-side build header: join key `k0` + one payload column `p1` (both UInt64).
SharedHeader makeRightHeader()
{
    ColumnsWithTypeAndName cols;
    cols.emplace_back(ColumnUInt64::create(), std::make_shared<DataTypeUInt64>(), "k0");
    cols.emplace_back(ColumnUInt64::create(), std::make_shared<DataTypeUInt64>(), "p1");
    return std::make_shared<const Block>(Block(cols));
}

Block makeRightBlock(const Block & header, UInt64 begin, size_t n)
{
    auto key = ColumnUInt64::create();
    auto pay = ColumnUInt64::create();
    for (size_t i = 0; i < n; ++i)
    {
        const UInt64 k = begin + i;
        key->insertValue(k);
        pay->insertValue(k * 2654435761ULL);
    }
    Columns c;
    c.push_back(std::move(key));
    c.push_back(std::move(pay));
    return header.cloneWithColumns(c);
}

std::shared_ptr<TableJoin> makeInnerTableJoin()
{
    Names key_names_right{"k0"};
    return std::make_shared<TableJoin>(SizeLimits{}, /*use_nulls=*/false, JoinKind::Inner, JoinStrictness::All, key_names_right);
}

/// Expected per-leaf row counts computed independently from the routing function (P1), for the keys in
/// [begin, begin + n). This is the leaf-membership oracle: a key must build into the leaf its hash routes to.
std::vector<size_t> expectedLeafCounts(const PartitionConfig & cfg, UInt64 begin, size_t n)
{
    std::vector<size_t> expected(cfg.total_leaves, 0);
    auto tmp = ColumnUInt64::create();
    tmp->reserve(n);
    for (size_t i = 0; i < n; ++i)
        tmp->insertValue(begin + i);

    std::vector<UInt32> hashes(n, 0);
    tmp->computeHashInto(0, n, hashes.data(), /*initial=*/true);
    for (size_t i = 0; i < n; ++i)
        ++expected[cfg.leafForHash(hashes[i])];
    return expected;
}

}

/// Single-thread build: cell conservation (Sum leaf rows == build rows) and leaf membership (per-leaf
/// distribution equals the independent routing oracle). debug_skip_passthrough so only the leaf HTs build.
TEST(PartitionedHashBuild, CellConservationAndMembership)
{
    auto header = makeRightHeader();
    auto table_join = makeInnerTableJoin();
    const size_t total = 100000;

    PartitionedHashJoin join(
        table_join,
        header,
        /*max_threads=*/1,
        /*rhs_size_estimation=*/std::nullopt,
        /*max_partitions_per_pass=*/64,
        /*shard_by_hash_input_batch_bytes=*/0,
        /*debug_skip_passthrough=*/true);

    Block block = makeRightBlock(*header, /*begin=*/1, total);
    join.addBlockToJoin(block, /*check_limits=*/false);
    join.onBuildPhaseFinish();
    join.runPostBuildPhase();

    /// Cell conservation.
    EXPECT_EQ(join.getTotalRowCount(), total);

    /// Leaf membership: every key built into the leaf its hash routes to.
    const auto leaf_counts = join.getLeafRowCounts();
    const auto expected = expectedLeafCounts(join.getPartitionConfig(), 1, total);
    ASSERT_EQ(leaf_counts.size(), expected.size());
    EXPECT_EQ(leaf_counts, expected);

    size_t sum = 0;
    for (size_t c : leaf_counts)
        sum += c;
    EXPECT_EQ(sum, total);
}

/// Lock-free parallel build: several threads call addBlockToJoin concurrently (one slot each), then the
/// eager work-stealing leaf build runs. Conservation + membership must still hold across all slots.
TEST(PartitionedHashBuild, ParallelBuildConservationAndMembership)
{
    auto header = makeRightHeader();
    auto table_join = makeInnerTableJoin();
    const size_t num_threads = 4;
    const size_t per_thread = 50000;
    const size_t total = num_threads * per_thread;

    PartitionedHashJoin join(
        table_join,
        header,
        /*max_threads=*/num_threads,
        /*rhs_size_estimation=*/std::nullopt,
        /*max_partitions_per_pass=*/64,
        /*shard_by_hash_input_batch_bytes=*/2 * 1024 * 1024,
        /*debug_skip_passthrough=*/true);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back(
            [&, t]
            {
                Block b = makeRightBlock(*header, /*begin=*/1 + t * per_thread, per_thread);
                join.addBlockToJoin(b, /*check_limits=*/false);
            });
    for (auto & th : threads)
        th.join();

    join.onBuildPhaseFinish();
    join.runPostBuildPhase();

    EXPECT_EQ(join.getTotalRowCount(), total);

    /// The keys are the contiguous range [1, 1 + total); the per-leaf distribution must match routing.
    const auto leaf_counts = join.getLeafRowCounts();
    const auto expected = expectedLeafCounts(join.getPartitionConfig(), 1, total);
    ASSERT_EQ(leaf_counts.size(), expected.size());
    EXPECT_EQ(leaf_counts, expected);
}

/// Degenerate single-leaf config still conserves rows (no partitioning passes).
TEST(PartitionedHashBuild, SingleLeafConservation)
{
    auto header = makeRightHeader();
    auto table_join = makeInnerTableJoin();
    const size_t total = 20000;

    /// rhs_size_estimation = 1 row -> 1 leaf, 0 passes.
    PartitionedHashJoin join(
        table_join,
        header,
        /*max_threads=*/2,
        /*rhs_size_estimation=*/UInt64{1},
        /*max_partitions_per_pass=*/64,
        /*shard_by_hash_input_batch_bytes=*/0,
        /*debug_skip_passthrough=*/true);

    ASSERT_EQ(join.getPartitionConfig().total_leaves, 1u);

    Block block = makeRightBlock(*header, /*begin=*/1, total);
    join.addBlockToJoin(block, /*check_limits=*/false);
    join.onBuildPhaseFinish();
    join.runPostBuildPhase();

    EXPECT_EQ(join.getTotalRowCount(), total);
}
