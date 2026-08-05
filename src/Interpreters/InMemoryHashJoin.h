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

/// `max_threads` is how many threads will feed the right side of this join concurrently. Only
/// `InMemoryHashJoinKind::Unified` uses it - it sizes the bucket count, and therefore the build's
/// lock granularity, from it. Pass 1 when the caller serializes the inserts itself.
InMemoryHashJoinPtr createInMemoryHashJoin(
    InMemoryHashJoinKind kind,
    const std::shared_ptr<TableJoin> & table_join,
    SharedHeader right_sample_block,
    bool any_take_last_row,
    size_t reserve_num,
    const String & instance_id,
    const StatsCollectingParams & stats_collecting_params,
    size_t max_threads);

}
