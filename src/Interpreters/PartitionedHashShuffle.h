#pragma once

#include <vector>

#include <Columns/IColumn.h>
#include <Interpreters/PartitionedHashConfig.h>

namespace DB
{

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
/// `scatter` per column over all sources (multi-source, exact-sized). Returns `fanout` column-groups with
/// the same columns. `scattered_rows` += total source rows.
std::vector<Columns> scatterGroupsByKeyHash(
    const std::vector<Columns> & sources, const std::vector<size_t> & key_indices, UInt32 shift, UInt32 fanout, size_t & scattered_rows);

/// Full-depth single-block shuffle to leaves (correctness oracle for unit tests): scatter through every
/// pass, re-deriving the hash each pass. Returns `config.total_leaves` groups in leaf order. Row/byte
/// conserved. `scattered_rows` += rows * numPasses.
std::vector<Columns> radixShuffleBlockToLeaves(
    const Columns & input_columns,
    const std::vector<size_t> & key_indices,
    const PartitionConfig & config,
    size_t & scattered_rows);

}
