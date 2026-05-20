#pragma once

#include <Interpreters/PartitionedHashJoin/BumpArena.h>
#include <Interpreters/PartitionedHashJoin/OutBlock.h>

#include <cstddef>
#include <cstdint>

namespace DB
{

/// Singly-linked list of OutBlocks for one (thread, partition) pair.
struct PartitionOutput
{
    OutBlock * head = nullptr;
    OutBlock * cur = nullptr;
    size_t next_cap = kOutCapMin;
    size_t total_rows = 0; /// running total rows committed

    [[nodiscard]] bool empty() const { return head == nullptr; }
};

/// Allocate a new OutBlock with K column buffers from `arena`, attach to the front of `po`.
/// Each column buffer holds `cap` elements of size `elem_bytes[k]`.
void growPartitionOutput(
    PartitionOutput & po,
    BumpArena & arena,
    size_t cap,
    const size_t * elem_bytes, /// array of K element sizes in bytes
    size_t num_cols);

}
