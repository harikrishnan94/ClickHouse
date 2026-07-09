#pragma once

#include "hash_join_bench.h"

namespace DB
{
class HashJoin;
}

namespace DB::JoinBench
{

/// Radix partitioned hash join: multi-pass `IColumn::scatter` of both sides into p_star
/// partitions, then one real ClickHouse `HashJoin` per partition, built and probed
/// single-threaded through the same `IJoin` interface (partitions processed in parallel).
class RadixHashJoinBench : public IJoinBench
{
public:
    RadixHashJoinBench(WorkerPool & pool_, const Block & left_header_, const Block & right_header_, size_t p_star_, size_t f_max_);
    ~RadixHashJoinBench() override;

    std::string name() const override { return "RadixHashJoin"; }
    void build(const std::vector<Block> & blocks) override;
    size_t probe(const std::vector<Block> & blocks, UInt64 * fingerprint) override;
    std::string phaseBreakdown() const override;
    void teardown() override;

private:
    WorkerPool & pool;
    Block left_header;
    SharedHeader right_header;
    std::shared_ptr<TableJoin> table_join;
    std::vector<size_t> pass_bits;
    std::vector<std::unique_ptr<HashJoin>> partition_joins;
    double build_scatter_sec = 0;
    double probe_scatter_sec = 0;
};

}
