#pragma once

#include <Interpreters/PartitionedHashJoin/BumpArena.h>
#include <Interpreters/PartitionedHashJoin/PartitionOutput.h>
#include <Interpreters/PartitionedHashJoin/ProbePartitions.h>
#include <Interpreters/PartitionedHashJoin/RadixShuffleColumn.h>
#include <Interpreters/PartitionedHashJoin/ShuffleScratch.h>
#include <Interpreters/PartitionedHashJoin/ShuffleSpec.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace DB
{

struct ThreadSlot
{
    ShuffleScratch scratch;
    BumpArena arena;

    /// Build side (right table): typed scatter into typed OutBlock chains.
    std::vector<PartitionOutput> build_parts;
    std::vector<RadixShuffleColumnPtr> build_cols;
    bool build_initialised = false;

    /// Probe side (left table): whole-Block scatter using IColumn::scatter.
    /// Avoids schema-resolution issues: the Block carries its own column names.
    std::vector<ProbePartition> probe_parts;
    bool probe_initialised = false;

    ThreadSlot() = default;
    ~ThreadSlot() = default;
    ThreadSlot(const ThreadSlot &) = delete;
    ThreadSlot & operator=(const ThreadSlot &) = delete;
    ThreadSlot(ThreadSlot &&) = delete;
    ThreadSlot & operator=(ThreadSlot &&) = delete;

    void initBuildSide(const ShuffleSpec & build_spec);
    void initProbeSide(size_t P);
};

}
