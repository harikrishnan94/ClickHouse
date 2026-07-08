#include "concurrent_hash_join_bench.h"

#include <Interpreters/ConcurrentHashJoin.h>
#include <Interpreters/HashTablesStatistics.h>
#include <Interpreters/TableJoin.h>

namespace DB::JoinBench
{

ConcurrentHashJoinBench::ConcurrentHashJoinBench(WorkerPool & pool_, const Block & left_header, const Block & right_header)
    : pool(pool_)
    , table_join(makeTableJoin(left_header, right_header))
    , join(std::make_shared<ConcurrentHashJoin>(
          table_join, pool_.size(), std::make_shared<const Block>(right_header), StatsCollectingParams{}))
{
}

ConcurrentHashJoinBench::~ConcurrentHashJoinBench() = default;

void ConcurrentHashJoinBench::build(const std::vector<Block> & blocks)
{
    const size_t threads = pool.size();
    pool.run([&](size_t tid)
    {
        for (size_t b = tid; b < blocks.size(); b += threads)
            join->addBlockToJoin(blocks[b], /*check_limits*/ false);
    });
    join->onBuildPhaseFinish();
}

size_t ConcurrentHashJoinBench::probe(const std::vector<Block> & blocks, UInt64 * fingerprint)
{
    const size_t threads = pool.size();
    std::atomic<size_t> rows{0};
    std::atomic<UInt64> digest{0};
    pool.run([&](size_t tid)
    {
        size_t local_rows = 0;
        UInt64 local_digest = 0;
        for (size_t b = tid; b < blocks.size(); b += threads)
            local_rows += drainJoinResult(join->joinBlock(blocks[b]), fingerprint ? &local_digest : nullptr);
        g_sink += local_rows;
        rows += local_rows;
        digest += local_digest;
    });
    if (fingerprint)
        *fingerprint += digest;
    return rows;
}

}
