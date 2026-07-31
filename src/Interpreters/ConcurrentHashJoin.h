#pragma once

#include <limits>
#include <memory>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/HashJoin/JoinProbeScratch.h>
#include <Interpreters/HashTablesStatistics.h>
#include <Interpreters/IJoin.h>
#include <base/defines.h>
#include <base/types.h>
#include <Common/ThreadPool_fwd.h>
#include <Interpreters/TableJoin.h>
#include <atomic>

namespace DB
{

/**
 * The default `HashJoin` is not thread-safe for inserting the right table's rows; thus, it is done on a single thread.
 * When the right table is large, the join process is too slow.
 *
 * `ConcurrentHashJoin` can run `addBlockToJoin()` concurrently to speed up the join process. On the test, it scales almost linearly.
 * For that, we create multiple `HashJoin` instances. In `addBlockToJoin()`, one input block is split into multiple blocks
 * corresponding to the `HashJoin` instances by hashing every row on the join keys. In particular, each `HashJoin` instance has its own hash map
 * that stores a unique set of keys. Also, `addBlockToJoin()` calls are done under mutex to guarantee
 * that every `HashJoin` instance is written only from one thread at a time.
 *
 * When matching the left table, probe blocks are NOT scattered: `joinBlock()` derives one route word per row (the same
 * hash the build scatter used, since only the instance a key was scattered to during the build phase holds that key)
 * and runs a single routed lookup pass over the original block, each row probing its slot's hash map (see
 * `RoutedHashJoinMethods`). All slots share one `StoredColumnsIndex`, so matches from every slot emit through one
 * result - in left-row order, which the scatter probe could not preserve.
 */
class ConcurrentHashJoin : public IJoin
{

public:
    /// The slot-count ceiling: the routed probe's slot ids are bytes, so at most 8 route bits.
    /// Production callers pass exactly this many slots instead of a thread-derived count:
    /// smaller per-slot maps reach their final buffer size with (almost) no rehash growth
    /// during the build, and that is what keeps the AMAC find ring's prefetches effective at
    /// probe time on huge maps - maps grown through several rehash generations defeat them
    /// (measured on `key64:probe.inner_all.S5.T96`; see the commit that introduced this
    /// constant). Tests pass explicit smaller counts to cover the single- and few-slot plans.
    static constexpr size_t max_slots = 256;

    /// `external_join_threshold_` is the auto-spill memory cap supplied by `SpillingHashJoin`
    /// when this instance is wrapped. It bounds statistics-driven preallocation so the
    /// reserve cannot blow past the wrapper's spill threshold. Pass 0 for standalone use
    /// (`join_algorithm = 'parallel_hash'`); the user-visible `max_bytes_before_external_join`
    /// setting deliberately does NOT apply to standalone instances.
    explicit ConcurrentHashJoin(
        std::shared_ptr<TableJoin> table_join_,
        size_t slots_,
        SharedHeader right_sample_block,
        const StatsCollectingParams & stats_collecting_params_,
        bool any_take_last_row_ = false,
        size_t external_join_threshold_ = 0,
        /// Phase 3 PoC of `tmp/two_level_hashjoin_plan.md` (bucket-striped concurrent build):
        /// opts eligible `key64` builds into ONE shared bucketed `key64_two_level` map instead of
        /// `slots` separate flat `key64` maps, merged by `onBuildPhaseFinish` via an O(bucket
        /// count) move (see `mergeTwoLevelKey64BucketsIfUsed`), no lock anywhere in the build -
        /// each slot's dispatched rows own their bucket exclusively by construction. Defaults to
        /// `false`: zero behavior change for every existing caller. Not yet used by any
        /// production caller - test-only until Phase 4 generalizes it to more map types and
        /// handles RIGHT/FULL/spill.
        bool use_two_level_key64_poc_ = false);

    ~ConcurrentHashJoin() override;

    std::string getName() const override { return "ConcurrentHashJoin"; }
    const TableJoin & getTableJoin() const override { return *table_join; }
    bool addBlockToJoin(const Block & right_block_, bool check_limits) override;
    void checkTypesOfKeys(const Block & block) const override;
    JoinResultPtr joinBlock(Block block) override { return joinBlock(std::move(block), invalid_lane); }
    JoinResultPtr joinBlock(Block block, size_t lane) override;
    void setTotals(const Block & block) override;
    const Block & getTotals() const override;
    size_t getTotalRowCount() const override;
    size_t getTotalByteCount() const override;
    bool alwaysReturnsEmptySet() const override;
    bool supportParallelJoin() const override { return true; }

    /// Number of internal hash join slots.
    size_t getNumSlots() const { return slots; }

    /// Extract all stored blocks from a specific slot.
    /// The slot's HashJoin data is reset afterwards.
    BlocksList releaseSlotBlocks(size_t slot_idx);

    IBlocksStreamPtr
    getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const override;

    bool supportParallelNonJoinedBlocksProcessing() const override;

    IBlocksStreamPtr getNonJoinedBlocks(
        const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size,
        size_t stream_idx, size_t num_streams) const override;

    bool isCloneSupported() const override
    {
        return getTotals().empty() && getTotalRowCount() == 0;
    }

    std::shared_ptr<IJoin> clone(const std::shared_ptr<TableJoin> & table_join_, SharedHeader, SharedHeader right_sample_block_) const override
    {
        return std::make_shared<ConcurrentHashJoin>(
            table_join_, slots, right_sample_block_, stats_collecting_params, any_take_last_row, external_join_threshold);
    }

    std::shared_ptr<IJoin> cloneNoParallel(const std::shared_ptr<TableJoin> & table_join_, SharedHeader, SharedHeader right_sample_block_) const override
    {
        return std::make_shared<HashJoin>(table_join_, right_sample_block_, any_take_last_row);
    }

    void onBuildPhaseFinish() override;

    /// Lane value used by the lane-less `joinBlock` entry points; always takes the mutexed
    /// pool path (see `probe_scratch_by_lane`).
    static constexpr size_t invalid_lane = std::numeric_limits<size_t>::max();

    std::unique_ptr<JoinProbeScratch> acquireProbeScratch(size_t lane);
    void releaseProbeScratch(std::unique_ptr<JoinProbeScratch> scratch, size_t lane);

    void setEnableLazyColumnsIndexing(bool value) override
    {
        std::ranges::for_each(hash_joins, [value](auto & hash_join) { hash_join->data->setEnableLazyColumnsIndexing(value); });
    }

    struct InternalHashJoin
    {
        std::mutex mutex;
        std::unique_ptr<HashJoin> data;
        bool space_was_preallocated = false;

        /// Snapshot of the total rows and bytes held locally by the hash join. This is updated during
        /// `addBlockToJoin` and is used to track the join state.
        size_t local_total_rows = 0;
        size_t local_total_bytes = 0;
    };

private:
    std::shared_ptr<TableJoin> table_join;
    size_t slots;
    bool any_take_last_row;
    std::unique_ptr<ThreadPool> pool;
    std::vector<std::shared_ptr<InternalHashJoin>> hash_joins;
    /// Raw per-slot `HashJoin` pointers for the routed probe, filled once at the end of the
    /// constructor. `RoutedJoinResult` holds a reference to this vector; it stays valid because
    /// the slots are only torn down in the destructor, when no probe result may be alive.
    std::vector<const HashJoin *> slot_joins;

    StatsCollectingParams stats_collecting_params;
    const size_t external_join_threshold;

    /// One parked scratch per probe lane - indexed by lane, not by hash-join slot - owned when
    /// the entry is non-null and freed by the destructor. Acquire = atomic exchange out;
    /// release = CAS back in; a lane collision or an out-of-range lane falls back to the
    /// mutexed pool, so a scratch is never lost and never double-owned. Sized once in the
    /// constructor, never resized: the lock-free fast paths index it without synchronizing
    /// against growth.
    std::vector<std::atomic<JoinProbeScratch *>> probe_scratch_by_lane;
    std::mutex probe_scratch_mutex;
    std::vector<std::unique_ptr<JoinProbeScratch>> probe_scratch_pool;

    std::mutex totals_mutex;
    Block totals;

    /// Snapshot of the total rows and bytes held globally by the concurrent hash join. This is updated during
    /// `addBlockToJoin` and is used to track the join state.
    std::atomic<size_t> global_total_rows{0};
    std::atomic<size_t> global_total_bytes{0};

    /// Once-per-build probe address material (see `RoutedProbePlan`; the collection schedule
    /// is on `collectRoutedProbePlan`). Probe results reference it like `slot_joins`.
    RoutedProbePlan routed_probe_plan;

    void collectRoutedProbePlan();

    /// Phase 3 PoC: whether the constructor opted into `key64_two_level`; set once, never changes.
    const bool use_two_level_key64_poc;
    /// Set by `mergeTwoLevelKey64BucketsIfUsed()` once the bucket-move merge has moved every
    /// slot's owned bucket into `hash_joins[0]`'s table - after this, `hash_joins[0]` alone holds
    /// every row, and probing must go through it directly (no routed dispatch, no scatter) rather
    /// than through `slot_joins`/`RoutedProbePlan`, which would find only slot 0's own original
    /// bucket for any row whose route happens to be 0 and nothing for every other slot (their
    /// buckets are empty post-move).
    bool two_level_key64_merged = false;
    /// Post-build, single-threaded: relocates each slot's exclusively-owned bucket
    /// (`impls[slot_index]`) into `hash_joins[0]`'s table via `std::move` (an O(1) transfer per
    /// bucket, not a re-insertion), and consolidates the per-slot `RightTableData` bookkeeping
    /// (`columns`/`allocated_size`/`rows_to_join`/`keys_to_join`) the same way. A no-op unless the
    /// constructor opted into `key64_two_level`. Mirrors `onBuildPhaseFinish`'s `move_buckets` in
    /// the reference worktree `ClickHouse-concurrent-hash-join-profile-events`.
    void mergeTwoLevelKey64BucketsIfUsed();

    ScatteredBlocks dispatchBlock(const Strings & key_columns_names, Block && from_block);
    std::pair<size_t, size_t> updateTotalRowsAndBytesUnlocked(std::shared_ptr<InternalHashJoin> & hash_join);
    void resetTotalRowsAndBytesUnlocked(std::shared_ptr<InternalHashJoin> & hash_join);
};

}
