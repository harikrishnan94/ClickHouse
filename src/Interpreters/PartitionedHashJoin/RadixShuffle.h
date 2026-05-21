#pragma once

#include <Interpreters/PartitionedHashJoin/PartitionOutput.h>
#include <Interpreters/PartitionedHashJoin/RadixShuffleColumn.h>
#include <Interpreters/PartitionedHashJoin/ShuffleScratch.h>
#include <Interpreters/PartitionedHashJoin/ShuffleSpec.h>
#include <Common/Arena.h>

#include <Core/Block.h>

#include <memory>
#include <vector>

namespace DB
{

/// Entry point for one block's radix scatter (Phases 1–4 of the algorithm).
///
/// `spec`   — spec for THIS side (build spec → build blocks, probe spec → probe blocks).
/// `parts`  — per-partition output list for this side (slot.build_parts or slot.probe_parts).
/// `cols`   — column shuffler instances for this side (slot.build_cols or slot.probe_cols).
/// `scratch`— per-thread reusable scratch arrays.
/// `arena`  — per-thread arena (see ThreadSlot::arena).
///
/// All arguments are thread-private; no locking inside.
void shuffleBlockIntoPartitions(
    const Block & block,
    const ShuffleSpec & spec,
    std::vector<PartitionOutput> & parts,
    std::vector<RadixShuffleColumnPtr> & cols,
    ShuffleScratch & scratch,
    Arena & arena);

}
