#pragma once

#include <Interpreters/IJoin.h>
#include <Interpreters/PartitionedHashJoin/ShuffleSpec.h>
#include <Interpreters/PartitionedHashJoin/ThreadSlot.h>
#include <Interpreters/TableJoin.h>

#include <Core/Block.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace DB
{

class PartitionedHashJoin final : public IJoin
{
public:
    PartitionedHashJoin(
        std::shared_ptr<TableJoin> table_join_,
        SharedHeader right_sample_block_,
        SharedHeader left_sample_block_,
        size_t num_partitions,
        size_t max_threads_,
        bool any_take_last_row_ = false);

    ~PartitionedHashJoin() override;

    PartitionedHashJoin(const PartitionedHashJoin &) = delete;
    PartitionedHashJoin & operator=(const PartitionedHashJoin &) = delete;
    PartitionedHashJoin(PartitionedHashJoin &&) = delete;
    PartitionedHashJoin & operator=(PartitionedHashJoin &&) = delete;

    std::string getName() const override { return "PartitionedHashJoin"; }
    const TableJoin & getTableJoin() const override { return *table_join; }

    /// Per-stream ingest cookie. Each JoiningTransform / FillingRightJoinSideTransform
    /// instance owns one; we cache the assigned ThreadSlot inside it so subsequent
    /// addBlockToJoin / joinBlock calls bypass the slot_mu + tid-map lookup.
    struct IngestCookie final : public IJoin::IngestHandle
    {
        ThreadSlot * build_slot = nullptr;
        ThreadSlot * probe_slot = nullptr;
    };

    IJoin::IngestHandlePtr createIngestHandle() override { return std::make_unique<IngestCookie>(); }

    using IJoin::addBlockToJoin;
    using IJoin::joinBlock;
    bool addBlockToJoin(const Block & block, bool check_limits) override;
    bool addBlockToJoin(IngestHandle * handle, const Block & block, bool check_limits) override;
    bool addBlockToJoin(IngestHandle * handle, const Block & block, size_t num_rows, bool check_limits) override;

    void checkTypesOfKeys(const Block & block) const override;
    JoinResultPtr joinBlock(Block block) override;
    JoinResultPtr joinBlock(IngestHandle * handle, Block block) override;
    void onBuildPhaseFinish() override;
    bool hasPostBuildPhase() const override { return false; }

    size_t getTotalRowCount() const override;
    size_t getTotalByteCount() const override;
    bool alwaysReturnsEmptySet() const override;

    JoinPipelineType pipelineType() const override { return JoinPipelineType::FillRightFirst; }
    bool supportParallelJoin() const override { return true; }
    bool hasDelayedBlocks() const override { return true; }
    IBlocksStreamPtr getDelayedBlocks() override;

    IBlocksStreamPtr
    getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const override;

    static bool isSupportedByColumns(const Block & right_sample, const Names & key_names, const Names & kept_payload_names);

    // ── Accessors for the delayed-blocks worker ────────────────────────────────
    const ShuffleSpec & buildSpec() const { return build_spec; }
    size_t numPartitions() const { return build_spec.P; }
    size_t numSlots() const { return num_slots_created.load(std::memory_order_acquire); }
    ThreadSlot & getSlot(size_t i) { return *slots[i]; }
    std::mutex & slotMu() { return slot_mu; }
    std::atomic<size_t> & partitionCursor() { return next_partition; }
    const std::shared_ptr<TableJoin> & tableJoin() const { return table_join; }
    const SharedHeader & rightSampleBlock() const { return right_sample; }
    const SharedHeader & leftSampleBlock() const { return left_sample; }
    bool anyTakeLastRow() const { return any_take_last_row; }

    /// Headers set on first getNonJoinedBlocks() call (from pipeline's non_joined_stream_builder).
    const Block & nonJoinedLeftHeader() const { return non_joined_left_header_; }
    const Block & nonJoinedResultHeader() const { return non_joined_result_header_; }
    bool hasNonJoinedHeaders() const { return non_joined_headers_set_; }

private:
    ThreadSlot & getOrAssignBuildSlot();
    ThreadSlot & getOrAssignProbeSlot();

    /// Slot resolution: cookie fast-path → cached pointer; nullptr cookie → tid map.
    ThreadSlot & resolveBuildSlot(IngestHandle * handle);
    ThreadSlot & resolveProbeSlot(IngestHandle * handle);

    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample;
    SharedHeader left_sample;
    bool any_take_last_row;

    ShuffleSpec build_spec; /// for addBlockToJoin (right-side typed scatter)

    std::vector<std::unique_ptr<ThreadSlot>> slots;
    std::mutex slot_mu;
    std::atomic<size_t> num_slots_created{0}; /// incremented inside slot_mu; read atomically outside
    std::unordered_map<std::thread::id, size_t> build_tid_to_slot;
    std::unordered_map<std::thread::id, size_t> probe_tid_to_slot;

    std::atomic<size_t> next_partition{0};
    std::atomic<size_t> total_build_rows{0};
    std::atomic<size_t> total_build_bytes{0};

    std::atomic<bool> build_done{false};
    std::atomic<bool> delayed_blocks_given{false};

    mutable Block non_joined_left_header_;
    mutable Block non_joined_result_header_;
    mutable bool non_joined_headers_set_ = false;
    mutable std::mutex non_joined_headers_mu_;

    /// An empty HashJoin used solely to produce the correct output schema from joinBlock
    /// when called with a 0-row header block during JoiningTransform::transformHeader.
    std::shared_ptr<IJoin> schema_hj;
    std::mutex schema_hj_mu;
};

}
