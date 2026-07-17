#pragma once

#include <Columns/ColumnNullable.h>
#include <Core/Block_fwd.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/PartitionedHashJoin/DenseHyperLogLog.h>
#include <Interpreters/PartitionedHashJoin/PartitionedJoinMaps.h>
#include <Common/Allocator.h>
#include <Common/Arena.h>
#include <Common/Logger.h>
#include <Common/PODArray.h>
#include <Common/ThreadPool.h>

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace DB
{

class TableJoin;

/** Partitioned hash join (`join_algorithm = 'partitioned_hash'`).
  *
  * Key-only scatter + partitioned hash-table build, with an unpartitioned probe:
  *
  * - Fill: right-side blocks are accumulated per lane untouched; one cheap 32-bit route word is
  *   computed per row (a routing hash decorrelated from the hash the leaf tables bucket by), its
  *   top 16 bits are saved as a 2-byte route, and a per-lane HyperLogLog sketch of the route
  *   words tracks the distinct-key count. No hash-table insertion happens during the fill.
  * - Barrier: the merged sketch sizes the leaf tables and picks the partition count - the
  *   smallest power of two whose worst-case per-leaf bucket array (through the exact grower
  *   rounding of the chosen map type) fits the private L2 cache. Small builds and the fixed-size
  *   map types (`key8`/`key16`) degenerate to a single leaf with no separate code path.
  * - Post-build (parallel): a cooperative scatter of only the key columns plus an 8-byte row
  *   locator (`RowRef`-encoded) into per-partition chunks, driven by the saved routes; payload
  *   columns stay in the shared row store. Then ONE contiguous allocation backs all leaf hash
  *   tables (exact-sized, unzeroed, carved per leaf by `FixedRegionAllocator`), and workers
  *   claim leaves largest-first for sequential locator-aware inserts. Scatter transients are
  *   released as they are consumed, before the probe starts.
  * - Probe: probe blocks are never scattered or buffered; each row recomputes its route word and
  *   looks its key up in the routed leaf table through the standard `HashJoin` emit machinery.
  */
class PartitionedHashJoin : public IJoin
{
public:
    PartitionedHashJoin(
        std::shared_ptr<TableJoin> table_join_, SharedHeader right_sample_block_, size_t num_threads_, bool any_take_last_row_ = false);

    ~PartitionedHashJoin() override;

    /// Plan-time gate: whether this join shape is implemented by the partitioned algorithm.
    /// Shapes outside the predicate must be planned with another enabled algorithm instead
    /// (see `tryCreateJoin` in `Planner/PlannerJoins.cpp`) - never fail at execution time.
    static bool isSupported(const TableJoin & table_join);

    std::string getName() const override { return "PartitionedHashJoin"; }
    const TableJoin & getTableJoin() const override;

    bool addBlockToJoin(const Block & block, bool check_limits) override;
    void checkTypesOfKeys(const Block & block) const override;
    JoinResultPtr joinBlock(Block block) override;

    size_t getTotalRowCount() const override;
    size_t getTotalByteCount() const override;
    bool alwaysReturnsEmptySet() const override;

    /// The fill phase is per-lane local plus a cheap mutexed append, so right-side streams
    /// may call `addBlockToJoin` concurrently.
    bool supportParallelJoin() const override { return true; }

    void onBuildPhaseFinish() override;
    bool hasPostBuildPhase() const override { return true; }
    void runPostBuildPhase() override;

    IBlocksStreamPtr
    getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const override;

    bool isCloneSupported() const override;

    std::shared_ptr<IJoin>
    clone(const std::shared_ptr<TableJoin> & table_join_, SharedHeader left_sample_block_, SharedHeader right_sample_block_) const override;

    std::shared_ptr<IJoin> cloneNoParallel(
        const std::shared_ptr<TableJoin> & table_join_, SharedHeader left_sample_block_, SharedHeader right_sample_block_) const override;

    void setEnableLazyColumnsIndexing(bool value) override;

    /// Build introspection for tests: the one-allocation property, exact region prediction,
    /// sketch quality, and heap-fallback behavior are asserted on these.
    struct BuildStats
    {
        size_t bits = 0;
        size_t partitions = 0;
        double hll_estimate = 0;
        UInt64 slab_allocations = 0;
        size_t slab_bytes = 0;
        UInt64 region_carves = 0;
        UInt64 heap_fallbacks = 0;
        UInt64 leaf_rows = 0;
        /// Every leaf map's actual buffer bytes equaled the plan's prediction.
        bool predictions_exact = true;
    };

    BuildStats getBuildStats() const;

    /// Shrinks the reserve safety factor so that leaf reserves underestimate and the maps must
    /// grow out of their slab regions - exercises the heap-fallback path in tests.
    void setReserveSafetyFactorForTests(double factor) { reserve_safety = factor; }

private:
    /// One accumulated right-side block: the payload block in row-store form, the prepared key
    /// columns (nested, with the merged null map extracted), and the saved 2-byte routes.
    struct FillBlock
    {
        Block stored;
        Columns keys_holder;
        ColumnRawPtrs key_columns;
        ColumnPtr null_map_holder;
        ConstNullMapPtr null_map = nullptr;
        PaddedPODArray<UInt16> routes;
        size_t rows = 0;
        UInt32 block_no = 0; /// assigned at the build barrier
    };

    /// Per-fill-thread lane: blocks are appended and the sketch updated without contention.
    struct FillLane
    {
        std::vector<FillBlock> blocks;
        DenseHyperLogLog hll;
    };

    /// State shared by the post-build stages (histogram -> allocate -> scatter -> leaf builds).
    struct PostBuildContext;

    FillLane & getFillLane();
    void decidePartitionPlan();
    void storeBlocksInRowStore();

    /// Both return whether every inserted key was unique (drives the RightAny promotion).
    bool postBuildPartitioned();
    bool postBuildSingleLeaf();
    void histogramWorker(PostBuildContext & ctx, size_t worker) const;
    void allocateWorker(PostBuildContext & ctx, size_t worker) const;
    void scatterWorker(PostBuildContext & ctx, size_t worker);
    void planAndAllocateHashTables(PostBuildContext & ctx);
    void leafBuildWorker(PostBuildContext & ctx, size_t worker);
    void finishBuildPhase(bool all_values_unique);

    /// Sequential locator-aware insert of one compact section into one leaf (PartitionedHashJoinBuild.cpp).
    /// The stored row ref of row i is `locators[i]` (encoded `RowRef` word), the decoded
    /// `narrow_locators[i]` (packed 4-byte form), or `RowRef(block_no, i)` when neither is set -
    /// the single-leaf path, where `null_bytes` (when set) skips null-key rows.
    void insertLeafSection(
        PartitionedJoinMaps & maps,
        const ColumnRawPtrs & key_columns,
        size_t rows,
        const UInt64 * locators,
        const UInt32 * narrow_locators_data,
        UInt32 block_no,
        const UInt8 * null_bytes,
        Arena & pool,
        bool & all_values_unique);

    /// The routed probe (PartitionedHashJoinProbe.cpp).
    JoinResultPtr probeDispatch(Block block);

    template <JoinKind KIND, JoinStrictness STRICTNESS>
    JoinResultPtr probeImpl(Block block);

    template <JoinKind KIND, JoinStrictness STRICTNESS, typename KeyGetter, typename Map, typename AddedColumnsType>
    size_t routedJoinRightColumns(
        const std::vector<const Map *> & leaf_maps_vector, AddedColumnsType & added_columns, const ScatteredBlock & block);

    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample_block;
    const bool any_take_last_row;
    const size_t num_threads;

    /// Schema delegate and owner of everything the probe emit machinery needs: block
    /// preparation, the saved block sample, the shared row store (`StoredColumnsIndex`) and the
    /// output sample blocks. Its own map stays empty; the leaf maps below replace it.
    std::unique_ptr<HashJoin> leaf_join;

    /// Fill phase.
    std::mutex fill_mutex;
    std::deque<FillLane> lanes;
    std::unordered_map<std::thread::id, FillLane *> lane_by_thread;
    std::atomic<size_t> accumulated_rows{0};
    std::atomic<size_t> accumulated_bytes{0};

    /// Partition plan, decided at the build barrier (`onBuildPhaseFinish`).
    size_t bits = 0;
    size_t partitions = 1;
    double hll_estimate = 0;
    double reserve_safety = 1.2; /// covers the sketch error (~1.15% at precision 13) and per-leaf spread
    std::vector<FillBlock> build_blocks; /// concatenated lanes, row-store block numbers assigned
    /// When every stored block number and row number fits 16 bits, the scattered locator column
    /// uses a packed 4-byte encoding `(block_no << 16) | row_no`, decoded to the standard 8-byte
    /// `RowRef` at insert - it halves the largest scatter transient. 8-byte otherwise.
    bool narrow_locators = false;

    /// Leaf hash tables, backed by the single contiguous slab. `build_arenas` hold the string
    /// keys and duplicate-list nodes referenced by map cells, so they live as long as the maps.
    std::vector<PartitionedJoinMaps> leaf_maps;
    std::deque<Arena> build_arenas;
    char * ht_slab = nullptr;
    size_t ht_slab_bytes = 0;
    Allocator<false, false> slab_allocator; /// not zeroed, not pre-faulted

    std::unique_ptr<ThreadPool> post_build_pool;

    bool build_phase_finished = false;

    mutable std::atomic<UInt64> region_carves{0};
    mutable std::atomic<UInt64> heap_fallbacks{0};
    BuildStats stats;

    LoggerPtr log;
};

}
