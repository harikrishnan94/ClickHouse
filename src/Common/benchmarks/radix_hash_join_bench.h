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

    /// BEP probe-budget emulation: consume the probe side in `waves` consecutive windows of
    /// blocks; each window is scattered and probed against every touched partition, then its
    /// scattered chunks are dropped - i.e. one window = one probe-buffer budget of
    /// |probe| / waves bytes, and each partition is revisited once per wave (paying the
    /// partition working-set reload) instead of once per join. `waves` == 1 is the plain
    /// radix probe. Timings land in probeScatterSec/probeJoinSec (summed over waves).
    size_t probeWaves(const std::vector<Block> & blocks, size_t waves, UInt64 * fingerprint);

    double probeScatterSec() const { return probe_scatter_sec; }
    double probeJoinSec() const { return probe_join_sec; }

private:
    WorkerPool & pool;
    Block left_header;
    SharedHeader right_header;
    std::shared_ptr<TableJoin> table_join;
    std::vector<size_t> pass_bits;
    std::vector<std::unique_ptr<HashJoin>> partition_joins;
    double build_scatter_sec = 0;
    double probe_scatter_sec = 0;
    double probe_join_sec = 0;
};

}
