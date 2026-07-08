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
    const size_t threads = pool.size();

    Stopwatch scatter_watch;
    auto build_parts = scatterSide(pool, blocks, pass_bits);
    build_scatter_sec = scatter_watch.elapsedSeconds();

    partition_joins.resize(build_parts.size());
    pool.run([&](size_t tid)
    {
        for (size_t p = tid; p < build_parts.size(); p += threads)
        {
            auto & join = partition_joins[p];
            join = std::make_unique<HashJoin>(
                table_join, right_header, /*any_take_last_row*/ false, /*reserve_num*/ 0,
                fmt::format("radix{}", p), /*use_two_level_maps*/ false);
            for (const auto & chunk : build_parts[p])
                join->addBlockToJoin(toBlock(chunk, *right_header), /*check_limits*/ false);
            join->onBuildPhaseFinish();
            build_parts[p].clear();
        }
    });
}

size_t RadixHashJoinBench::probe(const std::vector<Block> & blocks, UInt64 * fingerprint)
{
    const size_t threads = pool.size();

    Stopwatch scatter_watch;
    auto probe_parts = scatterSide(pool, blocks, pass_bits);
    probe_scatter_sec = scatter_watch.elapsedSeconds();

    std::atomic<size_t> rows{0};
    std::atomic<UInt64> digest{0};
    pool.run([&](size_t tid)
    {
        size_t local_rows = 0;
        UInt64 local_digest = 0;
        for (size_t p = tid; p < probe_parts.size(); p += threads)
        {
            for (const auto & chunk : probe_parts[p])
                local_rows += drainJoinResult(
                    partition_joins[p]->joinBlock(toBlock(chunk, left_header)), fingerprint ? &local_digest : nullptr);

            /// Free the partition's data before moving to the next one.
            probe_parts[p].clear();
            partition_joins[p].reset();
        }
        g_sink += local_rows;
        rows += local_rows;
        digest += local_digest;
    });
    if (fingerprint)
        *fingerprint += digest;
    return rows;
}

std::string RadixHashJoinBench::phaseBreakdown() const
{
    return fmt::format("build scatter {:.2f} ms, probe scatter {:.2f} ms", build_scatter_sec * 1e3, probe_scatter_sec * 1e3);
}

}
