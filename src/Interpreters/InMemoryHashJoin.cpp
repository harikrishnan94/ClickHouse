#include <Interpreters/InMemoryHashJoin.h>

#include <Interpreters/HashJoin/HashJoin.h>

namespace DB
{

InMemoryHashJoinPtr createInMemoryHashJoin(
    const std::shared_ptr<TableJoin> & table_join,
    SharedHeader right_sample_block,
    bool any_take_last_row,
    size_t reserve_num,
    const String & instance_id,
    const StatsCollectingParams & stats_collecting_params,
    size_t max_threads,
    bool use_parallel_layout)
{
    return std::make_shared<HashJoin>(
        table_join,
        right_sample_block,
        any_take_last_row,
        reserve_num,
        instance_id,
        stats_collecting_params,
        max_threads,
        use_parallel_layout);
}

}
