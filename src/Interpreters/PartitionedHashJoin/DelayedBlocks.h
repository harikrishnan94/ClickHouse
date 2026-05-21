#pragma once

#include <Core/Block.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/PartitionedHashJoin/OutBlock.h>
#include <Interpreters/PartitionedHashJoin/ShuffleSpec.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace DB
{

class HashJoin;
class PartitionedHashJoin;

/// Streaming per-partition delayed-blocks producer for PartitionedHashJoin.
///
/// Each call to nextImpl() returns at most ONE result block. State for an
/// in-progress partition (built mini-HashJoin, probe-slot cursor, the current
/// JoinResult iterator, and any non-joined stream) lives in a `WorkerState`
/// object. States are taken from a free-list on entry and returned on exit,
/// so any worker thread can resume a partition that another worker started.
/// This preserves correctness across pipeline-executor thread migrations,
/// avoids the previous "drain-all-blocks-of-a-partition-into-shared-queue"
/// burst pattern, and lets idle workers steal in-progress states to keep
/// every executor thread busy.
///
/// Termination is coordinated via `active_count_` (total states with active
/// partitions) and `cv_`: a worker that finds no work locally (cursor
/// exhausted + own state inactive + no stealable active state) waits until
/// either a peer releases an active state back to the free-list (which we
/// then steal) or `active_count_` drops to zero (truly done — return empty).
class PartitionedHashJoinDelayedBlocks final : public IBlocksStream
{
public:
    explicit PartitionedHashJoinDelayedBlocks(PartitionedHashJoin & join);
    ~PartitionedHashJoinDelayedBlocks() override;

    PartitionedHashJoinDelayedBlocks(const PartitionedHashJoinDelayedBlocks &) = delete;
    PartitionedHashJoinDelayedBlocks & operator=(const PartitionedHashJoinDelayedBlocks &) = delete;
    PartitionedHashJoinDelayedBlocks(PartitionedHashJoinDelayedBlocks &&) = delete;
    PartitionedHashJoinDelayedBlocks & operator=(PartitionedHashJoinDelayedBlocks &&) = delete;

protected:
    Block nextImpl() override;

private:
    Block buildOutBlockToBlock(const OutBlock & ob, const ShuffleSpec & spec) const;

    /// Per-partition in-flight state, owned by at most one worker at a time.
    struct WorkerState;

    /// Build the mini-HashJoin and prime probe cursors for partition `p` into
    /// `state`. Sets state.active=true and returns true on success; returns
    /// false if the partition has no work (both sides empty), leaving state
    /// inactive so the caller can grab the next partition.
    bool initStateForPartition(WorkerState & state, size_t p) const;

    /// Produce the next result block from `state`, advancing its cursors.
    /// Returns an empty block when the partition is exhausted; the caller
    /// should then clear state.active and grab a new partition.
    Block produceFromState(WorkerState & state) const;

    PartitionedHashJoin & join_;

    /// Pool guarded by `states_mu_`. Holds states not currently being driven
    /// by a worker — including ACTIVE states (with an in-progress partition)
    /// yielded by the previous worker, which subsequent callers preferentially
    /// steal so no executor thread sits idle while work remains.
    std::mutex states_mu_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<WorkerState>> free_states_;

    /// Total number of states with `active==true` anywhere — both in
    /// `free_states_` and currently held by a worker. Termination is safe
    /// (`return {}`) only when this is zero AND the partition cursor is
    /// exhausted: otherwise a peer may still produce more blocks.
    size_t active_count_ = 0;
};

}
