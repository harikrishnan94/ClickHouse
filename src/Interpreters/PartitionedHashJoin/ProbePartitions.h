#pragma once

/// Probe-side per-partition storage.
/// Stores whole CH Block slices per partition using IColumn::scatter.
/// This avoids the schema-resolution complexity of typed scatter.

#include <Core/Block.h>

#include <cstddef>
#include <vector>

namespace DB
{

/// One partition's accumulated probe rows.
/// A simple list of Block slices (each slice is a subset of rows from one probe Block).
struct ProbePartition
{
    std::vector<Block> slices;
    size_t total_rows = 0;
};

/// Scatter `block` into `P` partitions using `pids` (one uint16 per row).
/// Appends one Block slice to the appropriate ProbePartition.
/// Thread-private — no locking.
void probeScatterBlock(const Block & block, const uint16_t * pids, size_t n_rows, size_t P, std::vector<ProbePartition> & probe_parts);

}
