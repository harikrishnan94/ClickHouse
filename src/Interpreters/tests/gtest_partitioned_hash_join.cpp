#include <gtest/gtest.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <Core/Settings.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/PartitionedHashJoin/FixedRegionAllocator.h>
#include <Interpreters/PartitionedHashJoin/PartitionedHashJoin.h>
#include <Interpreters/PartitionedHashJoin/PartitionedJoinMaps.h>
#include <Interpreters/TableJoin.h>
#include <Common/Allocator.h>
#include <Common/HashTable/Hash.h>
#include <Common/HashTable/HashMap.h>
#include <Common/assert_cast.h>

using namespace DB;

namespace
{

constexpr size_t block_rows = 65536;

/// One joined output row: (k, probe_id, rk, build_id). The multiset of these tuples over the
/// whole drain is an exact identity: a dropped, duplicated, mis-routed or cross-wired row
/// changes it. In particular, a build row inserted into the WRONG leaf can never be found by
/// the probe (which routes by value), so mis-routing shows up as missing tuples.
using JoinedRow = std::tuple<UInt64, UInt64, UInt64, UInt64>;
using JoinedRows = std::vector<JoinedRow>;

Block twoColumnBlock(const String & key_name, const String & id_name, const std::vector<UInt64> & keys, const std::vector<UInt64> & ids)
{
    auto key_column = ColumnUInt64::create();
    auto id_column = ColumnUInt64::create();
    key_column->getData().assign(keys.begin(), keys.end());
    id_column->getData().assign(ids.begin(), ids.end());
    Block block;
    block.insert({std::move(key_column), std::make_shared<DataTypeUInt64>(), key_name});
    block.insert({std::move(id_column), std::make_shared<DataTypeUInt64>(), id_name});
    return block;
}

const UInt64 * columnData(const Block & block, const String & name, ColumnPtr & holder)
{
    holder = block.getByName(name).column->convertToFullColumnIfReplicated();
    return assert_cast<const ColumnUInt64 &>(*holder).getData().data();
}

void accumulateRows(const Block & block, JoinedRows & rows)
{
    if (!block.rows())
        return;
    ColumnPtr k_holder;
    ColumnPtr probe_holder;
    ColumnPtr rk_holder;
    ColumnPtr build_holder;
    const UInt64 * k = columnData(block, "k", k_holder);
    const UInt64 * probe_id = columnData(block, "probe_id", probe_holder);
    const UInt64 * rk = columnData(block, "rk", rk_holder);
    const UInt64 * build_id = columnData(block, "build_id", build_holder);
    for (size_t i = 0; i < block.rows(); ++i)
        rows.emplace_back(k[i], probe_id[i], rk[i], build_id[i]);
}

void drainResult(IJoinResult & result, JoinedRows & rows)
{
    while (true)
    {
        auto r = result.next();
        accumulateRows(r.block, rows);
        if (r.is_last)
            return;
    }
}

std::shared_ptr<TableJoin> makeTableJoin(const Block & left_header, const Block & right_header)
{
    Settings settings;
    auto table_join = std::make_shared<TableJoin>(settings, /*tmp_volume=*/nullptr, /*tmp_data=*/nullptr);
    table_join->setKind(JoinKind::Inner);
    table_join->getTableJoin().strictness = JoinStrictness::All;
    table_join->addDisjunct();
    table_join->getClauses().back().addKey(
        left_header.getByPosition(0).name, right_header.getByPosition(0).name, /*null_safe_comparison=*/false);

    NamesAndTypesList left_columns;
    NamesAndTypesList right_columns;
    Names used_columns;
    for (const auto & col : left_header)
    {
        left_columns.emplace_back(col.name, col.type);
        used_columns.push_back(col.name);
    }
    for (const auto & col : right_header)
    {
        right_columns.emplace_back(col.name, col.type);
        used_columns.push_back(col.name);
    }
    table_join->setInputColumns(std::move(left_columns), std::move(right_columns));
    table_join->setUsedColumns(used_columns);
    return table_join;
}

struct BuiltJoin
{
    std::shared_ptr<TableJoin> table_join;
    std::shared_ptr<PartitionedHashJoin> join;
};

/// Builds a PartitionedHashJoin over `distinct_keys` keys (`key(i) = i * 2654435761 + 1`),
/// each duplicated `duplicates` times, fed in `build_block_rows`-sized blocks through the real
/// IJoin build interface (fill -> barrier -> post-build). Blocks above 65536 rows exercise the
/// wide 8-byte locator encoding; smaller ones the packed 4-byte form.
BuiltJoin buildJoin(
    size_t distinct_keys, size_t duplicates, size_t num_threads, double reserve_safety_for_tests = 0, size_t build_block_rows = block_rows)
{
    const Block left_header = twoColumnBlock("k", "probe_id", {}, {});
    const Block right_header = twoColumnBlock("rk", "build_id", {}, {});

    BuiltJoin result;
    result.table_join = makeTableJoin(left_header, right_header);
    result.join = std::make_shared<PartitionedHashJoin>(result.table_join, std::make_shared<const Block>(right_header), num_threads);
    if (reserve_safety_for_tests > 0)
        result.join->setReserveSafetyFactorForTests(reserve_safety_for_tests);

    std::vector<UInt64> keys;
    std::vector<UInt64> ids;
    keys.reserve(build_block_rows);
    ids.reserve(build_block_rows);
    UInt64 id = 0;
    for (size_t i = 0; i < distinct_keys; ++i)
    {
        for (size_t d = 0; d < duplicates; ++d)
        {
            keys.push_back(i * 2654435761ULL + 1);
            ids.push_back(id++);
            if (keys.size() == build_block_rows)
            {
                EXPECT_TRUE(result.join->addBlockToJoin(twoColumnBlock("rk", "build_id", keys, ids), /*check_limits=*/true));
                keys.clear();
                ids.clear();
            }
        }
    }
    if (!keys.empty())
        EXPECT_TRUE(result.join->addBlockToJoin(twoColumnBlock("rk", "build_id", keys, ids), /*check_limits=*/true));

    result.join->onBuildPhaseFinish();
    result.join->runPostBuildPhase();
    return result;
}

/// Probes every distinct key once (plus `misses` keys that are absent from the build) and
/// checks the exact multiset of joined rows.
void probeAndCheck(BuiltJoin & built, size_t distinct_keys, size_t duplicates, size_t misses)
{
    JoinedRows expected;
    expected.reserve(distinct_keys * duplicates);
    for (size_t i = 0; i < distinct_keys; ++i)
    {
        const UInt64 key = i * 2654435761ULL + 1;
        for (size_t d = 0; d < duplicates; ++d)
            expected.emplace_back(key, i, key, i * duplicates + d);
    }
    std::sort(expected.begin(), expected.end());

    JoinedRows actual;
    actual.reserve(expected.size());
    std::vector<UInt64> keys;
    std::vector<UInt64> ids;
    for (size_t i = 0; i < distinct_keys + misses; ++i)
    {
        /// The +2 offset cannot collide with a built key: i * K + 2 == j * K + 1 would need
        /// i - j to be the (huge) modular inverse of -K, far outside these small ranges.
        keys.push_back(i < distinct_keys ? i * 2654435761ULL + 1 : i * 2654435761ULL + 2);
        ids.push_back(i);
        if (keys.size() == block_rows || i + 1 == distinct_keys + misses)
        {
            auto result = built.join->joinBlock(twoColumnBlock("k", "probe_id", keys, ids));
            drainResult(*result, actual);
            keys.clear();
            ids.clear();
        }
    }
    std::sort(actual.begin(), actual.end());
    ASSERT_EQ(actual.size(), expected.size());
    ASSERT_TRUE(actual == expected);
}

}

TEST(PartitionedHashJoin, PartitionedBuildOneAllocationAndParity)
{
    constexpr size_t distinct_keys = 300000;
    auto built = buildJoin(distinct_keys, /*duplicates=*/1, /*num_threads=*/4);

    const auto stats = built.join->getBuildStats();
    EXPECT_GT(stats.partitions, 1u) << "a 300K-key build must partition";
    EXPECT_EQ(stats.slab_allocations, 1u) << "exactly ONE hash-table allocation per build";
    EXPECT_EQ(stats.region_carves, stats.partitions) << "every leaf buffer must be carved from the slab";
    EXPECT_EQ(stats.heap_fallbacks, 0u);
    EXPECT_TRUE(stats.predictions_exact) << "predicted bucket bytes must equal the actual map buffer bytes";
    EXPECT_EQ(stats.leaf_rows, distinct_keys);
    EXPECT_GT(stats.slab_bytes, 0u);
    EXPECT_NEAR(stats.hll_estimate, static_cast<double>(distinct_keys), 0.05 * distinct_keys) << "the distinct estimate must be within 5%";

    probeAndCheck(built, distinct_keys, /*duplicates=*/1, /*misses=*/10000);
}

TEST(PartitionedHashJoin, PartitionedBuildWithDuplicates)
{
    constexpr size_t distinct_keys = 150000;
    constexpr size_t duplicates = 4;
    auto built = buildJoin(distinct_keys, duplicates, /*num_threads=*/4);

    const auto stats = built.join->getBuildStats();
    EXPECT_GT(stats.partitions, 1u);
    EXPECT_EQ(stats.slab_allocations, 1u);
    EXPECT_EQ(stats.region_carves, stats.partitions);
    EXPECT_EQ(stats.heap_fallbacks, 0u);
    EXPECT_TRUE(stats.predictions_exact);
    EXPECT_EQ(stats.leaf_rows, distinct_keys * duplicates);

    probeAndCheck(built, distinct_keys, duplicates, /*misses=*/1000);
}

TEST(PartitionedHashJoin, DegenerateSingleLeaf)
{
    constexpr size_t distinct_keys = 1000;
    auto built = buildJoin(distinct_keys, /*duplicates=*/2, /*num_threads=*/4);

    const auto stats = built.join->getBuildStats();
    EXPECT_EQ(stats.partitions, 1u) << "a small build must degenerate to one leaf";
    EXPECT_EQ(stats.slab_allocations, 1u);
    EXPECT_EQ(stats.region_carves, 1u);
    EXPECT_EQ(stats.heap_fallbacks, 0u);
    EXPECT_TRUE(stats.predictions_exact);

    probeAndCheck(built, distinct_keys, /*duplicates=*/2, /*misses=*/100);
}

TEST(PartitionedHashJoin, WideLocatorsForLargeBlocks)
{
    /// Blocks above 65536 rows do not fit the packed 4-byte locator encoding, so the build must
    /// take the 8-byte locator path and stay exact.
    constexpr size_t distinct_keys = 300000;
    auto built = buildJoin(distinct_keys, /*duplicates=*/1, /*num_threads=*/4, /*reserve_safety_for_tests=*/0, /*build_block_rows=*/100000);

    const auto stats = built.join->getBuildStats();
    EXPECT_GT(stats.partitions, 1u);
    EXPECT_EQ(stats.slab_allocations, 1u);
    EXPECT_EQ(stats.region_carves, stats.partitions);
    EXPECT_EQ(stats.heap_fallbacks, 0u);
    EXPECT_TRUE(stats.predictions_exact);

    probeAndCheck(built, distinct_keys, /*duplicates=*/1, /*misses=*/1000);
}

TEST(PartitionedHashJoin, HeapFallbackOnUnderestimate)
{
    /// A crippled reserve safety factor forces every leaf reserve to underestimate, so the
    /// maps must grow out of their slab regions onto the heap: the result must stay correct
    /// and the fallbacks must be counted, never silent.
    constexpr size_t distinct_keys = 200000;
    auto built = buildJoin(distinct_keys, /*duplicates=*/1, /*num_threads=*/4, /*reserve_safety_for_tests=*/0.001);

    const auto stats = built.join->getBuildStats();
    EXPECT_EQ(stats.slab_allocations, 1u) << "growing out of the slab must not allocate another one";
    EXPECT_GT(stats.heap_fallbacks, 0u) << "the crippled estimate must force heap fallbacks";

    probeAndCheck(built, distinct_keys, /*duplicates=*/1, /*misses=*/1000);
}

TEST(PartitionedHashJoin, FixedRegionAllocatorCarveAndFallback)
{
    using Map = HashMap<UInt64, UInt64, HashCRC32<UInt64>, HashTableGrowerWithPrecalculation<>, FixedRegionAllocator>;

    constexpr size_t reserve = 1000;
    Map::grower_type grower;
    grower.set(reserve);
    const size_t predicted_bytes = grower.bufSize() * sizeof(Map::cell_type);

    Allocator<false, false> slab_allocator;
    void * slab = slab_allocator.alloc(predicted_bytes, 64);
    std::atomic<UInt64> carves{0};
    std::atomic<UInt64> fallbacks{0};

    {
        FixedRegionAllocator::Region region{static_cast<char *>(slab), predicted_bytes, &carves, &fallbacks};
        FixedRegionAllocator::armRegion(region);
        Map map(reserve);
        EXPECT_EQ(map.getBufferSizeInBytes(), predicted_bytes) << "the carve must be grower-exact";
        EXPECT_EQ(carves.load(), 1u);
        EXPECT_EQ(fallbacks.load(), 0u);

        /// Insert far beyond the reserve: the map must grow onto the heap (counted), keep every
        /// key, and release the heap buffer at destruction while leaving the slab alone.
        for (UInt64 i = 0; i < 20 * reserve; ++i)
            map[i * 2654435761ULL] = i;
        EXPECT_GT(fallbacks.load(), 0u);
        for (UInt64 i = 0; i < 20 * reserve; ++i)
        {
            const auto * found = map.find(i * 2654435761ULL);
            ASSERT_NE(found, nullptr);
            EXPECT_EQ(found->getMapped(), i);
        }
    }

    slab_allocator.free(slab, predicted_bytes);
}
