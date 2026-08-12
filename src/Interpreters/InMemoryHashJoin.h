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

/// `max_threads` controls UHJ slot parallelism; `use_parallel_layout` selects 256-bucket vs 1-bucket maps.
InMemoryHashJoinPtr createInMemoryHashJoin(
    InMemoryHashJoinKind kind,
    const std::shared_ptr<TableJoin> & table_join,
    SharedHeader right_sample_block,
    bool any_take_last_row,
    size_t reserve_num,
    const String & instance_id,
    const StatsCollectingParams & stats_collecting_params,
    size_t max_threads,
    bool use_parallel_layout = true);
}
