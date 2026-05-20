#pragma once

#include <Core/Block.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/PartitionedHashJoin/OutBlock.h>
#include <Interpreters/PartitionedHashJoin/ShuffleSpec.h>

#include <deque>
#include <mutex>

namespace DB
{

class PartitionedHashJoin;

/// Thread-safe: multiple DelayedJoinedBlocksWorkerTransform processors share this
/// object and call next() concurrently. A single mutex serialises all calls;
/// partition processing (small HashJoin over P/T rows) is fast.
class PartitionedHashJoinDelayedBlocks final : public IBlocksStream
{
public:
    explicit PartitionedHashJoinDelayedBlocks(PartitionedHashJoin & join);
    ~PartitionedHashJoinDelayedBlocks() override = default;

    PartitionedHashJoinDelayedBlocks(const PartitionedHashJoinDelayedBlocks &) = delete;
    PartitionedHashJoinDelayedBlocks & operator=(const PartitionedHashJoinDelayedBlocks &) = delete;
    PartitionedHashJoinDelayedBlocks(PartitionedHashJoinDelayedBlocks &&) = delete;
    PartitionedHashJoinDelayedBlocks & operator=(PartitionedHashJoinDelayedBlocks &&) = delete;

protected:
    Block nextImpl() override;

private:
    Block buildOutBlockToBlock(const OutBlock & ob, const ShuffleSpec & spec) const;

    PartitionedHashJoin & join_;
    std::mutex mutex_;
    std::deque<Block> ready_;
};

}
