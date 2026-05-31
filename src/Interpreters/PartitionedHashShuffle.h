#pragma once

#include <span>
#include <vector>

#include <Columns/IColumn.h>
#include <Interpreters/PartitionedHashConfig.h>
#include <Common/PODArray.h>

namespace DB
{

/** Reusable per-slot scratch for `scatterGroupsByKeyHash`. Hoisting these transient buffers out of the
  * function turns the per-call heap churn of the build shuffle (one `hashes` array + one `pids` array per
  * source + per-shard counts + source pointers, all allocated then freed every pass/partition flush) into
  * a one-time per-slot allocation reused across every flush. Each `BuildSlot` is owned by a single build
  * thread and `scatterGroupsByKeyHash` never re-enters itself, so one scratch per slot is safe with no
  * locking. The scatter OUTPUT is written into a separate caller-owned `children` buffer (see below),
  * because the deferred cascade consumes one stage's children while deeper stages run their own scatters.
  */
struct ScatterScratch
{
    /// 32-bit key hash for the current source (overwritten per source within a call).
    PaddedPODArray<UInt32> hashes;
    /// Per-source partition ids for this pass. Grown, never shrunk, so each inner array keeps its capacity.
    std::vector<PaddedPODArray<UInt32>> pids;
    /// Views over `pids` handed to `ColumnsScatter` (rebuilt each call; trivially cheap).
    std::vector<std::span<const UInt32>> pids_spans;
    /// Per-shard row counts (`countRowsPerShard` output).
    PaddedPODArray<UInt32> rows_per_shard;
    /// Per-source column pointers for one column position (rebuilt per column).
    std::vector<const IColumn *> src_ptrs;
};

/** Primitives for the PartitionedHashJoin build-side radix shuffle (spec rev3 §4.2/§4.5), structured to
  * follow `BufferedShardByHashTransform`: each pass is a single batched multi-source
  * `DB::ColumnsScatter::scatter` over a large input. Nothing is carried — each pass re-derives the 32-bit
  * hash from the join key columns (already present among the scattered columns) and slices its window;
  * re-hashing (~0.43 ns/row) is cheaper than physically scattering a carried hash column every pass. The
  * join orchestrates the deferred cascade (pass 0 eager; trailing passes run per partition once it has
  * accumulated `shard_by_hash_input_batch_bytes`), so every scatter — at every pass — operates on a large
  * input.
  */

/// Scatter a set of source column-groups into `fanout` children by re-deriving the 32-bit hash from the
/// `key_indices` columns and slicing bits `[shift, shift + log2(fanout))`. One `countRowsPerShard` + one
/// `scatter` per column over all sources (multi-source, exact-sized). `scattered_rows` += total source rows.
///
/// `scratch` carries the reusable transient buffers (see `ScatterScratch`); pass the SAME scratch across
/// every flush of one build slot. `children` is the caller-owned output: it is reset to `fanout`
/// column-groups (each of the source's columns) and filled in place. The caller must supply a DISTINCT
/// `children` buffer per cascade stage, because a stage's children are consumed (moved out) while deeper
/// stages run their own scatters; reusing one buffer across nested stages would clobber it.
void scatterGroupsByKeyHash(
    const std::vector<Columns> & sources,
    const std::vector<size_t> & key_indices,
    UInt32 shift,
    UInt32 fanout,
    size_t & scattered_rows,
    ScatterScratch & scratch,
    std::vector<Columns> & children);

/// Full-depth single-block shuffle to leaves (correctness oracle for unit tests): scatter through every
/// pass, re-deriving the hash each pass. Returns `config.total_leaves` groups in leaf order. Row/byte
/// conserved. `scattered_rows` += rows * numPasses.
std::vector<Columns> radixShuffleBlockToLeaves(
    const Columns & input_columns, const std::vector<size_t> & key_indices, const PartitionConfig & config, size_t & scattered_rows);

}
