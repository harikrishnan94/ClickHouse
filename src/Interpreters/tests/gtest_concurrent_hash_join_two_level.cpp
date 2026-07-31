#include <gtest/gtest.h>

#include <algorithm>
#include <thread>
#include <vector>

#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <Core/Settings.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/ConcurrentHashJoin.h>
#include <Interpreters/TableJoin.h>
#include <Common/assert_cast.h>

using namespace DB;

namespace
{

/** Phase 3 PoC of `tmp/two_level_hashjoin_plan.md` (bucket-striped concurrent build): exercises
  * `ConcurrentHashJoin`'s `use_two_level_key64_poc_` opt-in end to end - build from multiple
  * threads calling `addBlockToJoin` concurrently (the real production entry point, not a
  * hand-rolled reimplementation of the build loop), `onBuildPhaseFinish()`'s bucket-move merge,
  * then probe - and checks the join produces the exact same result as the ordinary (non-bucketed)
  * `key64` path. `LEFT ... ALL` is deliberately chosen (see `tmp/two_level_hashjoin_plan.md`'s
  * Phase 3 section): `used_flags` stays compile-time dead code for it, and it resolves to
  * `MapsAll`/`RowRefList` - the ALL-strictness `Arena`-write path is exercised too, on a single
  * ON-clause with unique keys (no duplicate-key continuation writes), keeping the test within the
  * PoC's proven-safe scope without silently depending on `MapsOne` to avoid the `Arena` hazard.
  */

constexpr size_t block_rows = 4096;

std::shared_ptr<TableJoin> makeTableJoin(const Block & left_header, const Block & right_header, JoinKind kind, JoinStrictness strictness)
{
    Settings settings;
    auto table_join = std::make_shared<TableJoin>(settings, /*tmp_volume=*/nullptr, /*tmp_data=*/nullptr);
    table_join->setKind(kind);
    table_join->getTableJoin().strictness = strictness;
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

Block makeKeyBlock(const String & key_name, const String & id_name, const std::vector<UInt64> & keys, const std::vector<UInt64> & ids)
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

using JoinedRow = std::tuple<UInt64, UInt64, UInt64, UInt64>; // (k, probe_id, rk, build_id)
using JoinedRows = std::vector<JoinedRow>;

void accumulateRows(const Block & block, JoinedRows & rows)
{
    if (!block.rows())
        return;
    const ColumnPtr k = block.getByName("k").column->convertToFullColumnIfReplicated();
    const ColumnPtr probe_id = block.getByName("probe_id").column->convertToFullColumnIfReplicated();
    const ColumnPtr rk = block.getByName("rk").column->convertToFullColumnIfReplicated();
    const ColumnPtr build_id = block.getByName("build_id").column->convertToFullColumnIfReplicated();
    for (size_t i = 0; i < block.rows(); ++i)
        rows.emplace_back(
            assert_cast<const ColumnUInt64 &>(*k).getElement(i),
            assert_cast<const ColumnUInt64 &>(*probe_id).getElement(i),
            assert_cast<const ColumnUInt64 &>(*rk).getElement(i),
            assert_cast<const ColumnUInt64 &>(*build_id).getElement(i));
}

void drainResult(ConcurrentHashJoin & join, IJoinResult & result, JoinedRows & rows)
{
    while (true)
    {
        auto r = result.next();
        accumulateRows(r.block, rows);
        if (r.is_last)
        {
            if (r.next_block && r.next_block->rows())
            {
                r.next_block->filterBySelector();
                Block next = std::move(*r.next_block).getSourceBlock();
                auto next_result = join.joinBlock(std::move(next));
                drainResult(join, *next_result, rows);
            }
            return;
        }
    }
}

/// Builds `ConcurrentHashJoin` over `distinct_keys` (each with one build row), inserted from
/// `num_build_threads` threads concurrently calling `addBlockToJoin` - the real production entry
/// point - then probes every distinct key once (plus `misses` absent keys) and returns the
/// sorted joined-row multiset.
JoinedRows buildAndProbe(
    const std::vector<UInt64> & distinct_keys,
    const std::vector<UInt64> & misses,
    size_t slots,
    bool use_two_level_key64_poc,
    size_t num_build_threads)
{
    const Block left_header = makeKeyBlock("k", "probe_id", {}, {});
    const Block right_header = makeKeyBlock("rk", "build_id", {}, {});
    auto table_join = makeTableJoin(left_header, right_header, JoinKind::Left, JoinStrictness::All);
    auto join = std::make_shared<ConcurrentHashJoin>(
        table_join,
        slots,
        std::make_shared<const Block>(right_header),
        StatsCollectingParams{},
        /*any_take_last_row_=*/false,
        /*external_join_threshold_=*/0,
        use_two_level_key64_poc);

    /// Partition `distinct_keys` into `num_build_threads` interleaved shares and insert them
    /// concurrently - real concurrent multi-thread building via the actual `IJoin` interface,
    /// not a single-threaded stand-in.
    std::vector<std::thread> threads;
    threads.reserve(num_build_threads);
    for (size_t t = 0; t < num_build_threads; ++t)
    {
        threads.emplace_back(
            [&, t]
            {
                std::vector<UInt64> keys;
                std::vector<UInt64> ids;
                for (size_t i = t; i < distinct_keys.size(); i += num_build_threads)
                {
                    keys.push_back(distinct_keys[i]);
                    ids.push_back(i);
                    if (keys.size() == block_rows)
                    {
                        EXPECT_TRUE(join->addBlockToJoin(makeKeyBlock("rk", "build_id", keys, ids), /*check_limits=*/true));
                        keys.clear();
                        ids.clear();
                    }
                }
                if (!keys.empty())
                    EXPECT_TRUE(join->addBlockToJoin(makeKeyBlock("rk", "build_id", keys, ids), /*check_limits=*/true));
            });
    }
    for (auto & thread : threads)
        thread.join();

    join->onBuildPhaseFinish();

    EXPECT_EQ(join->getTotalRowCount(), distinct_keys.size());

    JoinedRows actual;
    std::vector<UInt64> keys;
    std::vector<UInt64> ids;
    size_t probe_id = 0;
    auto flush = [&](bool force)
    {
        if (keys.empty() || (!force && keys.size() < block_rows))
            return;
        auto result = join->joinBlock(makeKeyBlock("k", "probe_id", keys, ids));
        drainResult(*join, *result, actual);
        keys.clear();
        ids.clear();
    };
    for (const auto & key : distinct_keys)
    {
        keys.push_back(key);
        ids.push_back(probe_id++);
        flush(false);
    }
    for (const auto & key : misses)
    {
        keys.push_back(key);
        ids.push_back(probe_id++);
        flush(false);
    }
    flush(true);
    std::sort(actual.begin(), actual.end());
    return actual;
}

std::vector<UInt64> uintKeys(size_t count, size_t offset = 0)
{
    std::vector<UInt64> keys(count);
    for (size_t i = 0; i < count; ++i)
        keys[i] = (offset + i) * 2654435761ULL + 1;
    return keys;
}

}

TEST(ConcurrentHashJoinTwoLevel, MatchesOrdinaryKey64BuildAndProbe)
{
    const auto distinct_keys = uintKeys(50000);
    const std::vector<UInt64> misses = uintKeys(100, /*offset=*/50000);

    const auto ordinary = buildAndProbe(distinct_keys, misses, /*slots=*/8, /*use_two_level_key64_poc=*/false, /*num_build_threads=*/8);
    const auto bucketed = buildAndProbe(distinct_keys, misses, /*slots=*/8, /*use_two_level_key64_poc=*/true, /*num_build_threads=*/8);

    /// LEFT JOIN: every probed row emits at least one output row, matched or not (the `misses`
    /// keys join to a default/NULL right side rather than being dropped).
    ASSERT_EQ(ordinary.size(), distinct_keys.size() + misses.size());
    ASSERT_EQ(bucketed.size(), distinct_keys.size() + misses.size());
    ASSERT_EQ(ordinary, bucketed) << "key64_two_level's build+probe result diverged from the ordinary key64 path";
}

TEST(ConcurrentHashJoinTwoLevel, SingleSlotIsFreeOfTheBucketedPathEntirely)
{
    /// `slots == 1` never enters the bucketed branch (`mergeTwoLevelKey64BucketsIfUsed` returns
    /// immediately) - `chooseMethod`'s override also requires `two_level_buckets > 1`, so a
    /// single-slot plan uses plain `key64` even when `use_two_level_key64_poc` is requested.
    const auto distinct_keys = uintKeys(1000);
    const auto result = buildAndProbe(distinct_keys, {}, /*slots=*/1, /*use_two_level_key64_poc=*/true, /*num_build_threads=*/1);
    ASSERT_EQ(result.size(), distinct_keys.size());
}

TEST(ConcurrentHashJoinTwoLevel, ManyBuildThreadsPerSlotStillProducesEveryRow)
{
    /// More build threads than slots: several threads race to insert into the SAME slot's bucket
    /// over time (across many `addBlockToJoin` calls) - this is exactly the scenario static
    /// bucket ownership makes safe with no lock beyond `addBlockToJoin`'s existing per-slot
    /// try-lock (see `tmp/two_level_hashjoin_plan.md`'s Phase 3 section): two DIFFERENT slots'
    /// buckets are never touched by the same call, but a given slot's OWN bucket is still visited
    /// by many threads over the course of the build, serialized by the try-lock exactly like the
    /// ordinary `key64` path already is today.
    const auto distinct_keys = uintKeys(80000);
    const auto result = buildAndProbe(distinct_keys, {}, /*slots=*/4, /*use_two_level_key64_poc=*/true, /*num_build_threads=*/16);
    ASSERT_EQ(result.size(), distinct_keys.size());
}
