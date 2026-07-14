#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <set>
#include <tuple>
#include <vector>

#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnsScatter.h>
#include <Core/Block.h>
#include <Core/Settings.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/RadixHashJoin/RadixHashJoin.h>
#include <Interpreters/TableJoin.h>
#include <Common/assert_cast.h>

namespace DB::Setting
{
extern const SettingsUInt64 max_joined_block_size_rows;
}

using namespace DB;

namespace
{

/// One joined output row: (k, probe_id, rk, build_id). The multiset of these tuples over the whole
/// drain is an exact identity: any dropped, duplicated, or cross-wired row changes it.
using JoinedRow = std::tuple<UInt64, UInt64, UInt64, UInt64>;
using JoinedRows = std::multiset<JoinedRow>;

Block twoColumnBlock(const String & key_name, const String & id_name, const std::vector<UInt64> & keys, const std::vector<UInt64> & ids)
{
    auto key_column = ColumnUInt64::create();
    auto id_column = ColumnUInt64::create();
    for (size_t i = 0; i < keys.size(); ++i)
    {
        key_column->insertValue(keys[i]);
        id_column->insertValue(ids[i]);
    }
    Block block;
    block.insert({std::move(key_column), std::make_shared<DataTypeUInt64>(), key_name});
    block.insert({std::move(id_column), std::make_shared<DataTypeUInt64>(), id_name});
    return block;
}

/// Partition of a single-UInt64-key row for a 2-partition single-pass radix plan: computePassBits(2, ...)
/// yields one pass of 1 bit, consumed MSB-first, so partition = (routeWord(key) >> (32 - 1)) & 1.
/// If the routing ever changes, the two selected keys may collapse into one partition; that weakens
/// the two-worker shape but not the liveness property under test.
UInt32 partitionForKey(UInt64 key)
{
    return (ColumnsScatter::routeWord(key) >> 31) & 1;
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
        rows.emplace(k[i], probe_id[i], rk[i], build_id[i]);
}

/// Drains a result to completion, collecting every output row.
size_t drainResult(IJoinResult & result, JoinedRows & rows)
{
    size_t drained = 0;
    while (true)
    {
        auto r = result.next();
        drained += r.block.rows();
        accumulateRows(r.block, rows);
        if (r.is_last)
            return drained;
    }
}

std::shared_ptr<TableJoin> makeTableJoin(const Block & left_header, const Block & right_header)
{
    /// Constructed from query Settings like a real query, with one override: max_joined_block_size_rows = 1
    /// makes every leaf probe emit one output block per probe row (each probe row has more matches than
    /// the cap), so a wave produces enough blocks to overfill its bounded output queue.
    Settings settings;
    settings[Setting::max_joined_block_size_rows] = 1;
    auto table_join = std::make_shared<TableJoin>(settings, /*tmp_volume*/ nullptr, /*tmp_data*/ nullptr);
    table_join->setKind(JoinKind::Inner);
    table_join->getTableJoin().strictness = JoinStrictness::All;
    table_join->addDisjunct();
    table_join->getClauses().back().addKey(
        left_header.getByPosition(0).name, right_header.getByPosition(0).name, /*null_safe_comparison*/ false);

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

/// What the asynchronous lane-B call produced. On the abandoned (old-code) path the result is drained
/// and destroyed inside the lane-B thread, because there it owns a lock acquired on that thread.
struct LaneBOutcome
{
    JoinResultPtr result;
    Block first_block;
    bool first_is_last = false;
    size_t abandoned_rows = 0;
    JoinedRows abandoned_tuples;
};

}

/// Two probe lanes fill their windows back to back. Lane A triggers a wave and pops one block, leaving
/// its bounded output queue full and the wave's producers parked (8 output blocks per wave vs queue
/// capacity 2 * threads + 1 = 5). A concurrent joinBlock + next quantum on lane B must return control
/// promptly instead of waiting for lane A's wave to finish: parking every executor lane on the wave
/// admission while only the admitted result's owner can drain the queue is the deadlock under test.
TEST(RadixHashJoin, ConcurrentJoiningQuantumDoesNotWaitForPreviousWave)
{
    /// Two keys deliberately routed to different radix partitions, so both wave workers probe.
    UInt64 key_for_partition[2] = {0, 0};
    bool found[2] = {false, false};
    for (UInt64 v = 1; !(found[0] && found[1]); ++v)
    {
        const UInt32 p = partitionForKey(v);
        if (!found[p])
        {
            key_for_partition[p] = v;
            found[p] = true;
        }
    }
    const UInt64 k0 = key_for_partition[0];
    const UInt64 k1 = key_for_partition[1];

    /// Build side: each key duplicated 4 times (duplicates keep INNER ALL from promoting to RightAny),
    /// so every probe row joins to 4 output rows.
    const std::vector<UInt64> build_keys{k0, k0, k0, k0, k1, k1, k1, k1};
    const std::vector<UInt64> build_ids{100, 101, 102, 103, 104, 105, 106, 107};

    /// Probe lanes: 8 rows each (4 per key), distinct probe ids across lanes.
    const std::vector<UInt64> probe_keys{k0, k0, k0, k0, k1, k1, k1, k1};
    const std::vector<UInt64> probe_ids_a{0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<UInt64> probe_ids_b{8, 9, 10, 11, 12, 13, 14, 15};

    /// The exact expected output: every probe row of both lanes joined with every matching build row.
    JoinedRows expected;
    for (const auto * probe_ids : {&probe_ids_a, &probe_ids_b})
        for (size_t i = 0; i < probe_keys.size(); ++i)
            for (size_t j = 0; j < build_keys.size(); ++j)
                if (probe_keys[i] == build_keys[j])
                    expected.emplace(probe_keys[i], (*probe_ids)[i], build_keys[j], build_ids[j]);
    ASSERT_EQ(expected.size(), 64u);

    const Block left_header = twoColumnBlock("k", "probe_id", {}, {});
    const Block right_header = twoColumnBlock("rk", "build_id", {}, {});

    auto table_join = makeTableJoin(left_header, right_header);
    auto join = std::make_shared<RadixHashJoin>(
        table_join,
        std::make_shared<const Block>(right_header),
        /*max_threads*/ 2,
        /*rhs_size_estimation*/ std::nullopt,
        /*max_partitions_per_pass*/ 8,
        /*size_tables_by_distinct_estimate*/ false,
        /*probe_buffer_fraction*/ 0.0,
        /*probe_buffer_min_bytes*/ 1,
        /*probe_buffer_max_bytes*/ 1,
        StatsCollectingParams{});

    ASSERT_TRUE(join->addBlockToJoin(twoColumnBlock("rk", "build_id", build_keys, build_ids), /*check_limits*/ false));
    join->onBuildPhaseFinish();
    join->runPostBuildPhase();

    /// Lane A: the 1-byte window budget turns this single block into a full wave. One next() pop leaves
    /// 7 of its 8 output blocks undrained against a 5-slot queue: the wave is now mid-flight with parked
    /// producers, exactly the state every other lane must be able to pass through.
    auto result_a = join->joinBlock(twoColumnBlock("k", "probe_id", probe_keys, probe_ids_a), 0);
    ASSERT_NE(result_a, nullptr);
    JoinedRows drained_a;
    size_t rows_a = 0;
    {
        auto first = result_a->next();
        ASSERT_FALSE(first.is_last);
        rows_a += first.block.rows();
        accumulateRows(first.block, drained_a);
    }

    /// Lane B: one full JoiningTransform-style work quantum (joinBlock + immediate next) on another
    /// thread. It must give control back within the deadline whatever the state of lane A's wave.
    std::atomic<bool> abandoned{false};
    auto lane_b = std::async(
        std::launch::async,
        [&]() -> LaneBOutcome
        {
            LaneBOutcome outcome;
            auto result_b = join->joinBlock(twoColumnBlock("k", "probe_id", probe_keys, probe_ids_b), 1);
            auto r = result_b->next();
            if (abandoned.load())
            {
                /// Old-code path: this thread acquired the wave admission, so finish the result here.
                outcome.abandoned_rows += r.block.rows();
                accumulateRows(r.block, outcome.abandoned_tuples);
                if (!r.is_last)
                    outcome.abandoned_rows += drainResult(*result_b, outcome.abandoned_tuples);
                return outcome;
            }
            outcome.result = std::move(result_b);
            outcome.first_block = std::move(r.block);
            outcome.first_is_last = r.is_last;
            return outcome;
        });

    const bool lane_b_returned = lane_b.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    EXPECT_TRUE(lane_b_returned) << "lane B's joinBlock+next quantum did not return while lane A's wave was mid-flight";

    if (!lane_b_returned)
    {
        /// Controlled failure on the old code: release lane A's wave so lane B can finish, and let it
        /// drain its own result on its own thread. The test fails via the EXPECT above without hanging.
        abandoned.store(true);
        result_a.reset();
        auto outcome = lane_b.get();
        if (outcome.result)
        {
            /// Lane B finished its quantum in the gap between the deadline and the abandon flag.
            JoinedRows ignored;
            if (!outcome.first_is_last)
                drainResult(*outcome.result, ignored);
        }
        return;
    }

    /// Fixed code: drain both results concurrently and check the exact output identity.
    auto outcome = lane_b.get();
    ASSERT_NE(outcome.result, nullptr);

    auto drain_a = std::async(
        std::launch::async,
        [&]() -> size_t
        {
            return drainResult(*result_a, drained_a);
        });

    JoinedRows drained_b;
    size_t rows_b = outcome.first_block.rows();
    accumulateRows(outcome.first_block, drained_b);
    if (!outcome.first_is_last)
        rows_b += drainResult(*outcome.result, drained_b);

    rows_a += drain_a.get();
    result_a.reset();
    outcome.result.reset();

    /// Lanes drain the shared waves cooperatively, so the per-lane split is arbitrary; only the
    /// total and the exact multiset are invariant.
    EXPECT_EQ(rows_a + rows_b, 64u);

    JoinedRows drained_all = drained_a;
    drained_all.insert(drained_b.begin(), drained_b.end());
    EXPECT_EQ(drained_all.size(), 64u);
    EXPECT_TRUE(drained_all == expected) << "joined output multiset does not match the expected probe x build identity";
}
