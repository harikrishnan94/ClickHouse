#pragma once

/// hashprobe-bench/partitioned/phj_run.h
///
/// Partitioned Hash Join (PHJ) sweep — activated by CLICKHOUSE_PARTITIONED_JOIN=1.
///
/// Instead of one large HashJoin over all build_rows, radix-partitions both
/// build and probe sides into P buckets so each partition's hash table fits
/// in L2/L3, eliminating DRAM-latency on the probe and generate phases.
///
/// Phases reported:
///   partition-build  : radix-scatter build blocks into P buckets  (CPU ns/row)
///   build-HTs        : addBlockToJoin on P small HashJoins        (CPU ns/row)
///   partition-probe  : radix-scatter probe blocks into P buckets  (CPU ns/row)
///   probe            : key lookups (from PROBE_POINT inside joinBlock)
///   generate         : output gather (from PROBE_POINT inside joinBlock)
///
/// P is auto-selected so each partition's data fits in L2 per core (2 MB):
///   P = next_power_of_2(build_rows × bytes_per_row / L2_bytes)
///
/// Parallelism: T threads (max_threads), each handles P/T partitions per phase.

#include <hashprobe_bench/config.h>
#include <hashprobe_bench/types.h>

#include <Core/Block.h>
#include <Interpreters/HashJoin/HashJoin.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace DB::HashProbeBench
{

/// Per-phase CPU times for one PHJ cell (one (max_threads, block_size, rep) combo).
struct PHJPhaseMetrics
{
    double part_build_cpu_ms = 0.0;
    double build_ht_cpu_ms = 0.0;
    double part_probe_cpu_ms = 0.0;
    double probe_cpu_ms = 0.0; ///< from PROBE_POINT markers inside joinBlock
    double generate_cpu_ms = 0.0; ///< from PROBE_POINT markers inside joinBlock
    double total_wall_ms = 0.0;
    uint64_t output_rows = 0;
    int P = 0;
};

/// Auto-compute P: smallest power-of-2 s.t. partition data fits in L2.
int computeAutoPPartitions(const ConfigType & cfg, uint64_t build_rows);

/// Partition a single Block into P sub-blocks by hashing the key columns.
/// dest must be pre-sized to P (entries may be empty).
/// Uses mix64 on the key column(s); same hash as radixPartition in the POC.
void partitionBlock(
    const Block & src,
    int P,
    const ConfigType & cfg, ///< for key column names + shape
    bool is_build_side, ///< true → key cols are "b_k0" etc.
    std::vector<Block> & dest);

/// Run one PHJ cell: partition build + probe, build P HashJoins, probe them.
///
/// build_blocks    : all build-side blocks (re-used across reps; read-only)
/// proto_probe_blocks : all probe blocks at the given block_size (read-only)
/// max_threads     : T threads to spread P partitions across
/// build_rows      : total build rows (for normalising ns/row)
/// probe_rows      : total probe rows
PHJPhaseMetrics runPHJCell(
    const ConfigType & cfg,
    const Block & right_sample_block,
    const std::vector<Block> & build_blocks,
    const std::vector<Block> & proto_probe_blocks,
    uint32_t max_threads,
    uint64_t build_rows,
    uint64_t probe_rows);

} // namespace DB::HashProbeBench
