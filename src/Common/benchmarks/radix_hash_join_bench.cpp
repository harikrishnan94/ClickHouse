#include "radix_hash_join_bench.h"

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/TableJoin.h>
#include <Common/Stopwatch.h>

namespace DB::JoinBench
{

namespace
{

Block toBlock(const Chunk & chunk, const Block & header)
{
    Block block;
    for (size_t j = 0; j < chunk.columns.size(); ++j)
    {
        const auto & sample = header.getByPosition(j);
        block.insert(ColumnWithTypeAndName(chunk.columns[j], sample.type, sample.name));
    }
    return block;
}

}


RadixHashJoinBench::RadixHashJoinBench(
    WorkerPool & pool_, const Block & left_header_, const Block & right_header_, size_t p_star_, size_t f_max_)
    : pool(pool_)
    , left_header(left_header_)
    , right_header(std::make_shared<const Block>(right_header_))
    , table_join(makeTableJoin(left_header_, right_header_))
    , pass_bits(computePassBits(p_star_, f_max_))
{
}

RadixHashJoinBench::~RadixHashJoinBench() = default;

void RadixHashJoinBench::build(const std::vector<Block> & blocks)
{
    Stopwatch scatter_watch;
    auto build_parts = scatterSide(pool, blocks, pass_bits);
    build_scatter_sec = scatter_watch.elapsedSeconds();

    partition_joins.resize(build_parts.size());
    /// Dynamic scheduling: a static stripe gives some threads 2 partitions and others 1 when
    /// the partition count is not a multiple of the thread count (e.g. p_star == bit_ceil
    /// (threads) on a non-power-of-two core count), inflating this phase by up to ~1.5x. An
    /// atomic counter removes that and is the join's defense against partition-size skew.
    std::atomic<size_t> next_partition{0};
    pool.run([&](size_t /*tid*/)
    {
        for (size_t p = next_partition.fetch_add(1, std::memory_order_relaxed); p < build_parts.size();
             p = next_partition.fetch_add(1, std::memory_order_relaxed))
        {
            /// A real radix join gets exact partition sizes for free from the scatter
            /// histogram; reserving removes all rehash growth and shrinks the table up to 2x
            /// vs the growth ladder.
            size_t part_rows = 0;
            for (const auto & chunk : build_parts[p])
                part_rows += chunk.rows;

            auto & join = partition_joins[p];
            join = std::make_unique<HashJoin>(
                table_join, right_header, /*any_take_last_row*/ false, /*reserve_num*/ part_rows,
                fmt::format("radix{}", p), /*use_two_level_maps*/ false);
            for (const auto & chunk : build_parts[p])
                join->addBlockToJoin(toBlock(chunk, *right_header), /*check_limits*/ false);
            join->onBuildPhaseFinish();
            /// Releases only the Chunk/ColumnPtr wrappers: addBlockToJoin shares the same COW
            /// columns into the join's stored blocks (HashJoin.cpp,
            /// `data->columns.emplace_back(block_to_save.getColumns(), ...)`), so the scattered
            /// build side stays resident until teardown().
            build_parts[p].clear();
        }
    });
}

size_t RadixHashJoinBench::probe(const std::vector<Block> & blocks, UInt64 * fingerprint)
{
    Stopwatch scatter_watch;
    auto probe_parts = scatterSide(pool, blocks, pass_bits);
    probe_scatter_sec = scatter_watch.elapsedSeconds();

    std::atomic<size_t> rows{0};
    std::atomic<UInt64> digest{0};
    std::atomic<size_t> next_partition{0};
    pool.run([&](size_t /*tid*/)
    {
        size_t local_rows = 0;
        UInt64 local_digest = 0;
        for (size_t p = next_partition.fetch_add(1, std::memory_order_relaxed); p < probe_parts.size();
             p = next_partition.fetch_add(1, std::memory_order_relaxed))
        {
            for (const auto & chunk : probe_parts[p])
                local_rows += drainJoinResult(
                    partition_joins[p]->joinBlock(toBlock(chunk, left_header)), fingerprint ? &local_digest : nullptr);

            /// Free the consumed scattered probe input before moving to the next partition:
            /// genuine RP probe-side work with no NP analogue. The join itself is NOT reset
            /// here - teardown() times that separately, matching a real query's pipeline
            /// destruction happening after the last output block.
            probe_parts[p].clear();
        }
        g_sink += local_rows;
        rows += local_rows;
        digest += local_digest;
    });
    if (fingerprint)
        *fingerprint += digest;
    return rows;
}

void RadixHashJoinBench::teardown()
{
    std::atomic<size_t> next{0};
    pool.run([&](size_t /*tid*/)
    {
        for (size_t p = next.fetch_add(1, std::memory_order_relaxed); p < partition_joins.size();
             p = next.fetch_add(1, std::memory_order_relaxed))
            partition_joins[p].reset();
    });
    partition_joins.clear();
}

std::string RadixHashJoinBench::phaseBreakdown() const
{
    return fmt::format("build scatter {:.2f} ms, probe scatter {:.2f} ms", build_scatter_sec * 1e3, probe_scatter_sec * 1e3);
}

}
