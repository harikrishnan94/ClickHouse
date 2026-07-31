#include <Interpreters/InMemoryHashJoin.h>

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/UnifiedHashJoin/HashJoin.h>

namespace DB
{

InMemoryHashJoinPtr createInMemoryHashJoin(
    InMemoryHashJoinKind kind,
    const std::shared_ptr<TableJoin> & table_join,
    SharedHeader right_sample_block,
    bool any_take_last_row,
    size_t reserve_num,
    const String & instance_id,
    bool use_two_level_maps,
    const StatsCollectingParams & stats_collecting_params)
{
    switch (kind)
    {
        case InMemoryHashJoinKind::Hash:
            return std::make_shared<HashJoin>(
                table_join,
                right_sample_block,
                any_take_last_row,
                reserve_num,
                instance_id,
                use_two_level_maps,
                stats_collecting_params);
        case InMemoryHashJoinKind::Unified:
            return std::make_shared<UnifiedHashJoin>(
                table_join,
                right_sample_block,
                any_take_last_row,
                reserve_num,
                instance_id,
                use_two_level_maps,
                stats_collecting_params);
    }
}

}
