#pragma once

#include "hash_join_bench.h"

namespace DB
{
class ConcurrentHashJoin;
}

namespace DB::JoinBench
{

/// Non-partitioned hash join: the real ClickHouse `ConcurrentHashJoin` (`parallel_hash`),
/// used as-is through the `IJoin` interface: concurrent `addBlockToJoin` into per-slot
/// two-level maps, constant-time bucket merge in `onBuildPhaseFinish`, unpartitioned
/// shared-map probe via `joinBlock`.
class ConcurrentHashJoinBench : public IJoinBench
{
public:
    ConcurrentHashJoinBench(WorkerPool & pool_, const Block & left_header, const Block & right_header);
    ~ConcurrentHashJoinBench() override;

    std::string name() const override { return "ConcurrentHashJoin"; }
    void build(const std::vector<Block> & blocks) override;
    size_t probe(const std::vector<Block> & blocks) override;

private:
    WorkerPool & pool;
    std::shared_ptr<TableJoin> table_join;
    std::shared_ptr<ConcurrentHashJoin> join;
};

}
