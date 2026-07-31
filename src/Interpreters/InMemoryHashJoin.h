#pragma once

#include <Interpreters/IInMemoryHashJoin.h>
#include <Interpreters/HashTablesStatistics.h>
#include <Interpreters/TableJoin.h>

namespace DB
{

enum class InMemoryHashJoinKind : uint8_t
{
    Hash,
    Unified,
};

InMemoryHashJoinPtr createInMemoryHashJoin(
    InMemoryHashJoinKind kind,
    const std::shared_ptr<TableJoin> & table_join,
    SharedHeader right_sample_block,
    bool any_take_last_row,
    size_t reserve_num,
    const String & instance_id,
    bool use_two_level_maps,
    const StatsCollectingParams & stats_collecting_params);

}
