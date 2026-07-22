#pragma once

#include <Columns/ColumnNullable.h>
#include <Core/Block_fwd.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/HashTablesStatistics.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/PartitionedHashJoin/DenseHyperLogLog.h>
#include <Interpreters/PartitionedHashJoin/PartitionedJoinMaps.h>
#include <Common/Arena.h>
#include <Common/Logger.h>
#include <Common/PODArray.h>
#include <Common/ThreadPool.h>

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

namespace DB
{

class TableJoin;

/// Flat lookup descriptor of one leaf's open-addressing map: the cell buffer pointer and the
/// grower mask, extracted once after the builds. 16 bytes per leaf in one contiguous array, so
/// a probe's per-row cell address is computable from one L1 load here instead of chasing
/// `leaf_map_ptrs[leaf]` and then the map header (three dependent loads); the AMAC probe ring
/// resolves a row's descriptor once at admit and carries it in the ring slot. The fixed-size map
/// types (`key8`/`key16`) keep the zero entry - their probe never takes the descriptor path.
struct LeafMapDesc
{
    const void * buf = nullptr;
    size_t mask = 0;
};

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
  *   columns stay in the shared row store. Then workers claim leaves largest-first for
  *   sequential locator-aware inserts; each claimed leaf's map is created exact-reserved on
  *   demand right before that leaf's inserts (one buffer allocation per leaf, predicted through
  *   the map's own grower math), so the allocator recycles the scatter chunks of already
  *   consumed leaves instead of holding the full table footprint alongside all transients.
  *   Scatter transients are released as they are consumed, before the probe starts.
  * - Probe: probe blocks are never scattered or buffered; each row recomputes its route word and
  *   looks its key up in the routed leaf table through the standard `HashJoin` emit machinery.
  *   Above the engagement threshold, lookups of every AMAC-capable map type run as a two-phase
  *   pass per block: an AMAC find ring (`AmacRing.h`) fills a per-row result array out of
  *   order, then an in-order pass consumes the precomputed results - a dispatch-free cursor
  *   pass on the flagless word-mapped lazy shapes, the sequential `processMatch` loop on the
  *   rest - so replication offsets, used-flags semantics and every join kind's logic stay
  *   untouched. Below the threshold (and for the AMAC-incapable getters) the sequential routed
  *   loop runs, with the look-ahead prefetcher where the key getter supports it.
  *   Right-side used flags (RIGHT/FULL/SEMI/ANTI/ANY kinds) live in one per-offset flag space
  *   spanning all leaves: leaf L's cell offsets are shifted by `flag_base[L]` (the prefix sums of
  *   the per-leaf bucket counts), so `JoinUsedFlags` and the non-joined machinery keep their
  *   single-map semantics.
  *
  * Shapes whose used flags must be keyed per right-table row instead of per hash-table cell -
  * multiple disjuncts (OR of key sets) - run the standard `HashJoin` machinery whole (fill,
  * probe, flags, non-joined) behind this interface: the per-row-flags regime is partition
  * agnostic and rare, so it is not worth a partitioned build. The partition plan is 1 there.
  */
class PartitionedHashJoin : public IJoin
{
public:
    PartitionedHashJoin(
        std::shared_ptr<TableJoin> table_join_,
        SharedHeader right_sample_block_,
        size_t num_threads_,
        bool any_take_last_row_ = false,
        const StatsCollectingParams & stats_collecting_params_ = {});

    ~PartitionedHashJoin() override;

    /// Plan-time gate: whether this join shape is implemented by the partitioned algorithm.
    /// Shapes outside the predicate must be planned with another enabled algorithm instead
    /// (see `tryCreateJoin` in `Planner/PlannerJoins.cpp`) - never fail at execution time.
    static bool isSupported(const TableJoin & table_join);

    std::string getName() const override { return "PartitionedHashJoin"; }
    const TableJoin & getTableJoin() const override;

    bool addBlockToJoin(const Block & block, bool check_limits) override;
    bool addBlockToJoin(const Block & block, size_t num_rows, bool check_limits, size_t build_lane) override;
    void checkTypesOfKeys(const Block & block) const override;
    JoinResultPtr joinBlock(Block block) override;
    JoinResultPtr joinBlock(Block block, size_t lane) override;

    /// With `supportParallelJoin` every parallel fill stream reports totals at its end-of-fill
    /// (usually an empty block; at most one stream carries the real totals), so unlike the
    /// unsynchronized base-class default this must be guarded (same as `ConcurrentHashJoin`).
    void setTotals(const Block & block) override;
    const Block & getTotals() const override;

    size_t getTotalRowCount() const override;
    size_t getTotalByteCount() const override;
    bool alwaysReturnsEmptySet() const override;

    /// The fill phase is per-lane local plus a cheap mutexed append, so right-side streams
    /// may call `addBlockToJoin` concurrently. The delegated standard path inserts into one
    /// `HashJoin`, which is not thread-safe - a single fill stream there, like `hash`.
    bool supportParallelJoin() const override { return !delegate_mode; }

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

    /// Build introspection for tests: the exact buffer-size prediction, sketch quality, and
    /// growth-past-reserve behavior are asserted on these.
    struct BuildStats
    {
        size_t bits = 0;
        size_t partitions = 0;
        /// Per-pass radix bits of the build scatter, MSB-first (one element = single-pass;
        /// the plan splits when the L2 rule wants more bits than one pass's fanout ceiling).
        std::vector<size_t> pass_bits;
        /// Final per-leaf insertable row counts (size `partitions`), filled on the partitioned
        /// path; lets tests assert per-leaf row parity between different pass plans.
        std::vector<UInt64> leaf_row_counts;
        double hll_estimate = 0;
        /// Total predicted hash-table buffer bytes across all leaves.
        size_t ht_total_bytes = 0;
        /// Leaves whose map resized during the inserts, past its create-time buffer (a
        /// distinct-estimate shortfall) - correct but unplanned, counted, never silent.
        UInt64 leaf_growths = 0;
        /// Times a leaf map grew during an AMAC insert ring (the ring was drained, the map
        /// resized, the in-flight rows re-seeded) - rare, correct, never silent.
        UInt64 amac_ring_growths = 0;
        /// Whether the leaf inserts of this build ran above the AMAC engagement threshold.
        bool amac_build_engaged = false;
        UInt64 leaf_rows = 0;
        /// Every leaf map's actual buffer bytes equaled the plan's prediction.
        bool predictions_exact = true;
        /// This build reused a cached per-partition distinct-key count from the hash-table
        /// statistics cache (a warm run) and skipped the per-row sketch feed of the fill.
        bool distinct_estimate_reused = false;
        /// Per-leaf used-flag base offsets (prefix sums of the per-leaf bucket counts + 1),
        /// size partitions + 1; empty when the join shape needs no right-side used flags.
        std::vector<UInt64> flag_base;
    };

    BuildStats getBuildStats() const;

    /// Shrinks the reserve safety factor so that leaf reserves underestimate and the maps must
    /// grow past their exact reserves - exercises the growth path in tests.
    void setReserveSafetyFactorForTests(double factor) { reserve_safety = factor; }

    /// Pins the build and probe onto the plain sequential loops regardless of the engagement
    /// threshold - lets tests cross-check the AMAC results against the sequential ones.
    void setAmacEnabledForTests(bool value) { amac_enabled = value; }

    /// Lowers the per-pass fanout ceiling so small builds plan multi-pass scatters - lets tests
    /// exercise the refine passes without a ~500M-key build.
    void setMaxFanoutPerPassForTests(size_t value) { max_fanout_per_pass = value; }

private:
    /// The non-joined-rows filler for RIGHT/FULL output over the partitioned leaf maps.
    friend class NotJoinedPartitioned;

    /// Row-store access for the non-joined filler: `HashJoin::data` is private, and the filler
    /// is a friend of this class, not of `HashJoin`.
    const HashJoin::RightTableData & storedData() const { return *leaf_join->data; }

    /// One accumulated right-side block: the payload block in row-store form, the prepared key
    /// columns (nested, with the merged null map extracted), and the saved 2-byte routes.
    struct FillBlock
    {
        Block stored;
        Columns keys_holder;
        ColumnRawPtrs key_columns;
        ColumnPtr null_map_holder;
        ConstNullMapPtr null_map = nullptr;
        /// The right-side ON-section condition of the clause, evaluated per row (`AllTrue` when
        /// the clause has none): rows it filters are not inserted, mirroring the standard build.
        JoinCommon::JoinMask join_mask;
        /// Merged build-skip bytes (key-null rows OR mask-filtered rows), materialized only when
        /// the mask actually filters; otherwise `skipData` falls back to the plain null map.
        PaddedPODArray<UInt8> skip_bytes;
        PaddedPODArray<UInt16> routes;
        size_t rows = 0;
        UInt32 block_no = 0; /// assigned at the build barrier

        const UInt8 * skipData() const
        {
            if (!skip_bytes.empty())
                return skip_bytes.data();
            return null_map ? null_map->data() : nullptr;
        }
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
    FillLane & getFillLane(size_t build_lane);
    bool addBlockToJoinImpl(const Block & source_block, bool check_limits, size_t build_lane);
    void decidePartitionPlan();
    void storeBlocksInRowStore();

    /// Both return whether every inserted key was unique (drives the RightAny promotion).
    bool postBuildPartitioned();
    bool postBuildSingleLeaf();
    void histogramWorker(PostBuildContext & ctx, size_t worker) const;
    void allocateWorker(PostBuildContext & ctx, size_t worker) const;
    void scatterWorker(PostBuildContext & ctx, size_t worker);

    /// One refine pass of a multi-pass plan: splits every current bucket into
    /// `2^refine_bits` sub-buckets by the next MSB-first slice of the saved route words
    /// (below the `bits_done` bits earlier passes consumed), group-major output. After the
    /// last pass a row's leaf index equals `route >> (16 - bits)` - the same leaf a
    /// single-pass plan of `bits` would give it, and the leaf the probe derives.
    void refinePassWave(PostBuildContext & ctx, size_t refine_bits, size_t bits_done, std::atomic<UInt64> & stage_thread_us);
    void planHashTables(PostBuildContext & ctx);
    void leafBuildWorker(PostBuildContext & ctx, size_t worker);
    void finishBuildPhase(bool all_values_unique);

    /// Locator-aware insert of one compact section into one leaf (PartitionedHashJoinBuild.cpp):
    /// an AMAC insert ring above the engagement threshold, the sequential loop below it.
    /// The stored row ref of row i is `locators[i]` (encoded `RowRef` word), the decoded
    /// `narrow_locators[i]` (packed 4-byte form), or `RowRef(block_no, i)` when neither is set -
    /// the single-leaf path, where `skip_bytes` (when set) skips null-key and mask-filtered rows.
    void insertLeafSection(
        PartitionedJoinMaps & maps,
        const ColumnRawPtrs & key_columns,
        size_t rows,
        const UInt64 * locators,
        const UInt32 * narrow_locators_data,
        UInt32 block_no,
        const UInt8 * skip_bytes,
        Arena & pool,
        bool & all_values_unique);

    /// Per-leaf used-flag base offsets from the final leaf bucket counts, and the used-flags
    /// reinit over the whole flag space; runs after the leaf builds, for flagged shapes only.
    void computeFlagBaseAndReinitUsedFlags();

    /// Fills `leaf_map_ptrs` once after the leaf builds, so the probe does not rebuild a
    /// per-block pointer table (8 bytes x partitions per `joinBlock` call).
    void collectLeafMapPointers();

    /// The AMAC engagement decision of this build, made once right after the hash tables are
    /// sized (before the leaf inserts); mirrors the software-prefetch enablement heuristics.
    void decideAmacEngagement();

    /// The routed probe (PartitionedHashJoinProbe*.cpp). `MapsShape` is the standard maps shape
    /// (`HashJoin::MapsOne`/`MapsAll`/`MapsAsof`) the (kind, strictness) pair dispatches to; the
    /// actual leaf maps are the partitioned counterpart holding identical cells.
    JoinResultPtr probeDispatch(Block block, size_t lane);

    template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsShape>
    JoinResultPtr probeImpl(Block block, size_t lane);

    template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsShape, typename KeyGetter, typename Map, typename AddedColumnsType>
    size_t routedJoinRightColumns(AddedColumnsType & added_columns, const ScatteredBlock & block, size_t lane);

    /// Per-probe-stream scratch, pooled on the join and reused across probe blocks: the per-row
    /// leaf ids, and the AMAC find pass's per-row results - the matched cell's
    /// mapped value copied BY VALUE into `found_word` (0 = no match; `RowRef`/`RowRefList` are
    /// 8-byte words that are never 0 for a real match), so phase B never dereferences the cell a
    /// second time after it left the cache. Mapped types that do not fit a word (ASOF) store the
    /// mapped pointer's bits instead. `found_offset` is the used-flags offset shifted into the
    /// shared flag space, filled only for the flagged shapes.
    struct ProbeScratch
    {
        PaddedPODArray<UInt16> leaf_ids;
        PaddedPODArray<UInt64> found_word;
        PaddedPODArray<UInt64> found_offset;
    };

    /// The pipeline-carried lane index binds a lock-free scratch slot per probe stream; lanes
    /// outside the slot table (or the lane-less legacy entry points) fall back to the mutexed
    /// pool, so lane collisions and out-of-range indices stay correct, just slower.
    static constexpr size_t invalid_lane = std::numeric_limits<size_t>::max();

    std::unique_ptr<ProbeScratch> acquireProbeScratch(size_t lane);
    void releaseProbeScratch(std::unique_ptr<ProbeScratch> scratch, size_t lane);

    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample_block;
    const bool any_take_last_row;
    const size_t num_threads;

    /// Schema delegate and owner of everything the probe emit machinery needs: block
    /// preparation, the saved block sample, the shared row store (`StoredColumnsIndex`), the
    /// used flags and the output sample blocks. Its own map stays empty; the leaf maps below
    /// replace it - except in the delegated standard path, where it runs the join whole.
    std::unique_ptr<HashJoin> leaf_join;

    /// Shapes needing the per-row used-flags regime (multiple disjuncts) run the standard
    /// `HashJoin` machinery whole; see the class comment.
    const bool delegate_mode;

    /// Index of the active `HashJoin::MapsVariant` alternative (MapsOne/MapsAll/MapsAsof) the
    /// (kind, strictness) pair dispatches to; the leaf maps mirror it.
    const size_t maps_variant_index;

    /// `IJoin::totals` is private, so the guarded overrides keep their own copy.
    std::mutex totals_mutex;
    Block totals;

    /// Fill phase. `lanes` owns the per-lane state (the barrier iterates it); the slot table
    /// gives pipeline-carried lane indices a lock-free lookup (one mutexed emplace on a lane's
    /// FIRST block, atomic loads afterwards; the table is sized once and never resized, so the
    /// fast path never races a rehash). Lane-less callers keep the thread-id map.
    std::mutex fill_mutex;
    std::deque<FillLane> lanes;
    std::unordered_map<std::thread::id, FillLane *> lane_by_thread;
    std::vector<std::atomic<FillLane *>> fill_lane_slots;
    std::atomic<size_t> accumulated_rows{0};
    std::atomic<size_t> accumulated_bytes{0};

    /// Partition plan, decided at the build barrier (`onBuildPhaseFinish`).
    size_t bits = 0;
    size_t partitions = 1;
    /// Per-pass radix bits (MSB-first slices of the route word), sum == `bits`; more than one
    /// element when the L2 rule wants a fanout above the per-pass ceiling (multi-pass scatter).
    std::vector<size_t> pass_bits;
    /// The per-pass fanout ceiling (`ColumnsScatter::MAX_FANOUT_PER_PASS`; test-overridable).
    size_t max_fanout_per_pass;
    double hll_estimate = 0;
    double reserve_safety = 1.2; /// covers the sketch error (~1.15% at precision 13) and per-leaf spread
    /// Cross-run distinct-key statistics (same cache key derivation as the other join
    /// algorithms, but a dedicated `PartitionedHashJoinEntry` cache holding the PER-PARTITION
    /// breakdown, not just a total). When a previous run of this query published its counts,
    /// the fill skips the per-row sketch feed, the barrier consumes the cached total instead of
    /// the sketch estimate, and the per-leaf hash-table sizing consumes the cached per-partition
    /// counts (folded or split to match this build's own partition count when its plan bits
    /// differ from the cached ones - see `planHashTables`) instead of the uniform rescale.
    StatsCollectingParams stats_collecting_params;
    std::optional<PartitionedHashJoinEntry> cached_stats;
    std::vector<FillBlock> build_blocks; /// concatenated lanes, row-store block numbers assigned
    /// When every stored block number and row number fits 16 bits, the scattered locator column
    /// uses a packed 4-byte encoding `(block_no << 16) | row_no`, decoded to the standard 8-byte
    /// `RowRef` at insert - it halves the largest scatter transient. 8-byte otherwise.
    bool narrow_locators = false;

    /// Leaf hash tables, each owning its exact-reserved buffer. `build_arenas` hold the string
    /// keys and duplicate-list nodes referenced by map cells, so they live as long as the maps.
    std::vector<PartitionedJoinMaps> leaf_maps;
    /// The active map of each leaf, collected once after the builds (`collectLeafMapPointers`).
    /// Type-erased: the probe dispatch casts an entry back to the concrete map type selected by
    /// the same `data->type` + maps-variant pair that stored it.
    std::vector<const void *> leaf_map_ptrs;
    /// One `LeafMapDesc` per leaf, filled by `collectLeafMapPointers`.
    std::vector<LeafMapDesc> leaf_map_descs;
    /// Whether any leaf's collision chain reached the last cell of its tail-padded buffer (see
    /// `TailPaddedHashTableGrower`) - i.e. a chain may have wrapped. Filled by
    /// `collectLeafMapPointers`; practically always false. While false, the probe walks run
    /// wrap-free (`++pos` with no bound check); a wrapped plan keeps the wrap-aware loops.
    bool any_leaf_chain_wrapped = false;
    /// Per-leaf used-flag base offsets: leaf L's flags live at `[flag_base[L], flag_base[L + 1])`
    /// of the shared per-offset flag space (`flag_base[L + 1] - flag_base[L]` = leaf bucket
    /// count + 1, the +1 covering the hash table's zero-value cell). Size partitions + 1;
    /// filled after the leaf builds, only for shapes that track right-side used flags.
    std::vector<UInt64> flag_base;
    std::deque<Arena> build_arenas;
    size_t ht_total_bytes = 0; /// total predicted hash-table bytes (drives the prefetch heuristics)

    std::unique_ptr<ThreadPool> post_build_pool;

    bool build_phase_finished = false;

    /// AMAC state: the test override, the engagement decision of this build's leaf inserts
    /// (made once, before the leaf-build wave), and the ring-growth counter.
    bool amac_enabled = true;
    bool amac_build_engaged = false;
    std::atomic<UInt64> amac_ring_growths{0};

    std::mutex probe_scratch_mutex;
    std::vector<std::unique_ptr<ProbeScratch>> probe_scratch_pool;
    /// One parked scratch per probe lane (owned when non-null; freed by the destructor).
    /// Acquire = atomic exchange out; release = CAS back in; misses go through the pool.
    std::vector<std::atomic<ProbeScratch *>> probe_scratch_slots;

    BuildStats stats;

    LoggerPtr log;
};

}
