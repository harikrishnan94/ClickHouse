#include <Columns/ColumnsScatter.h>
#include <Interpreters/HashJoin/HashJoinMethodsImpl.h>
#include <Interpreters/HashJoin/JoinUsedFlags.h>
#include <Interpreters/HashJoin/KeyGetter.h>
#include <Interpreters/PartitionedHashJoin/FixedRegionAllocator.h>
#include <Interpreters/PartitionedHashJoin/PartitionedHashJoin.h>
#include <Interpreters/TableJoin.h>
#include <Interpreters/joinDispatch.h>
#include <Common/CurrentMetrics.h>
#include <Common/CurrentThread.h>
#include <Common/ElapsedTimeProfileEventIncrement.h>
#include <Common/ProfileEvents.h>
#include <Common/ThreadGroupSwitcher.h>
#include <Common/formatReadable.h>
#include <Common/logger_useful.h>
#include <Common/setThreadName.h>

#include <algorithm>

namespace ProfileEvents
{
extern const Event PartitionedHashJoinBuildMicroseconds;
extern const Event PartitionedHashJoinLeafRows;
extern const Event PartitionedHashJoinHashTableBytes;
extern const Event PartitionedHashJoinHashTableHeapFallbacks;
}

namespace CurrentMetrics
{
extern const Metric PartitionedHashJoinPoolThreads;
extern const Metric PartitionedHashJoinPoolThreadsActive;
extern const Metric PartitionedHashJoinPoolThreadsScheduled;
}

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int UNSUPPORTED_JOIN_KEYS;
}

namespace
{

constexpr size_t locator_piece_rows = 32768; /// locator synthesis scratch stays L2-resident

/// Sequential locator-aware insert of one compact section, mirroring the semantics of
/// `insertFromBlockImplTypeCase` + the `Inserter` family for the map's value shape: the map hash
/// is computed inside `emplaceKey` (once per build row); `RowRefList` cells store the encoded
/// row ref inline and append duplicates to the arena list (`insertAll`); `RowRef` cells keep the
/// first row per key, or the last with `any_take_last_row` (`insertOne`); `AsofRowRefs` cells
/// append (asof value, row ref) to the per-key sorted lookup (`insertAsof`). The recorded ref
/// comes from the scattered locator column - the encoded 8-byte word or the packed 4-byte form -
/// or, on the single-leaf path, from `RowRef(block_no, i)` with skipped (null-key/mask-filtered)
/// rows excluded via `skip_bytes`.
template <typename KeyGetter, typename Map>
void insertSectionImpl(
    const HashJoin & join,
    Map & map,
    const ColumnRawPtrs & key_columns,
    const Sizes & key_sizes,
    size_t rows,
    const UInt64 * locators,
    const UInt32 * narrow_locators,
    UInt32 block_no,
    const UInt8 * skip_bytes,
    Arena & pool,
    bool & all_values_unique,
    bool enable_prefetch)
{
    using Mapped = Map::mapped_type;
    constexpr bool mapped_one = std::is_same_v<Mapped, RowRef>;
    constexpr bool mapped_asof = std::is_same_v<Mapped, AsofRowRefs>;

    /// The ASOF value lives at the row's own index of the trailing key column, so the sorted
    /// insert only works where the compact index IS the stored row - the single-leaf path
    /// (ASOF plans always degenerate to one leaf, see `decidePartitionPlan`).
    const IColumn * asof_column [[maybe_unused]] = nullptr;
    if constexpr (mapped_asof)
    {
        if (locators || narrow_locators)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "ASOF leaf inserts require the single-leaf build plan");
        asof_column = key_columns.back();
    }

    /// Mirrors `createKeyGetter`: the ASOF getter excludes the trailing inequality column.
    auto key_getter = [&]
    {
        if constexpr (mapped_asof)
        {
            ColumnRawPtrs equi_columns(key_columns.begin(), key_columns.end() - 1);
            Sizes equi_sizes(key_sizes.begin(), key_sizes.end() - 1);
            return KeyGetter(equi_columns, equi_sizes, nullptr);
        }
        else
        {
            return KeyGetter(key_columns, key_sizes, nullptr);
        }
    }();

    const bool any_take_last_row = join.anyTakeLastRow();

    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;
    bool use_prefetch = false;
    if constexpr (can_prefetch)
        use_prefetch = enable_prefetch && map.getBufferSizeInBytes() > getMinBytesForPrefetchInJoin();

    auto prefetcher = makeJoinPrefetcher(
        use_prefetch,
        rows,
        [&](size_t k) __attribute__((always_inline))
        {
            if constexpr (can_prefetch)
                map.prefetch(key_getter.getKeyHolder(k, pool));
        });

    bool all_unique = all_values_unique;
    for (size_t i = 0; i < rows; ++i)
    {
        if constexpr (can_prefetch)
            prefetcher.prefetchAt(i);

        if (skip_bytes && skip_bytes[i])
            continue;

        auto emplace_result = key_getter.emplaceKey(map, i, pool);

        if constexpr (mapped_asof)
        {
            Mapped * time_series_map = &emplace_result.getMapped();
            if (emplace_result.isInserted())
                time_series_map = new (time_series_map) Mapped(createAsofRowRef(*join.getAsofType(), join.getAsofInequality()));
            (*time_series_map)->insert(*asof_column, block_no, i);
        }
        else
        {
            UInt64 ref = 0;
            if (locators)
                ref = locators[i];
            else if (narrow_locators)
                ref = RowRef(narrow_locators[i] >> 16, narrow_locators[i] & 0xFFFFu).encode();
            else
                ref = RowRef(block_no, i).encode();

            if constexpr (mapped_one)
            {
                if (emplace_result.isInserted() || any_take_last_row)
                    new (&emplace_result.getMapped()) RowRef(refWordBlockNo(ref), refWordRowNo(ref));
            }
            else
            {
                if (emplace_result.isInserted())
                {
                    new (&emplace_result.getMapped()) RowRefList(RowRefList::fromWord(ref));
                }
                else
                {
                    emplace_result.getMapped().insert(ref, pool);
                    all_unique = false;
                }
            }
        }
    }
    all_values_unique = all_unique;
}

}

/// State shared by the post-build stages. The stages communicate through exact per-bucket
/// offsets: bucket `p` holds, in worker order, the rows of worker `w`'s block stripe at
/// row offsets [starts[p * workers + w], starts[p * workers + w] + worker_hist[w][p]).
/// Bucket `partitions` (the last one) collects null-key rows; it is scattered like any other
/// bucket and dropped before the leaf builds, so the leaves only ever see insertable rows.
struct PartitionedHashJoin::PostBuildContext
{
    size_t workers = 0;
    size_t fanout = 0; /// partitions + 1 (the null bucket)
    size_t num_key_columns = 0;
    bool generic_mode = false;

    PaddedPODArray<UInt64> worker_hist; /// workers x fanout
    std::vector<UInt64> bucket_rows; /// per bucket
    PaddedPODArray<UInt64> starts; /// fanout x workers

    /// Fixed mode: one exact uninitialized column per (key column, bucket), written cooperatively.
    std::vector<MutableColumns> fixed_out;
    std::vector<std::vector<char *>> fixed_base;
    std::vector<size_t> fixed_widths;

    /// Generic mode: self-contained per-(key column, worker, bucket) pieces (Layer-1 scatter).
    std::vector<std::vector<MutableColumns>> pieces;

    /// The locator column, always scattered cooperatively: encoded 8-byte `RowRef` words, or the
    /// packed 4-byte `(block_no << 16) | row_no` form when the build fits it (see `narrow_locators`).
    std::vector<PaddedPODArray<UInt64>> locators;
    std::vector<PaddedPODArray<UInt32>> locators32;

    struct WorkerState
    {
        std::vector<ColumnsScatter::ScatterScratch> key_scratch;
        ColumnsScatter::ScatterScratch locator_scratch;
        PaddedPODArray<UInt64> locator_piece;
        PaddedPODArray<UInt32> locator_piece32;
        bool all_values_unique = true;
        bool predictions_exact = true;
        UInt64 leaf_rows = 0;
    };
    std::deque<WorkerState> worker_state;

    /// Leaf plan (filled right before the leaf builds).
    std::vector<UInt64> leaf_reserve;
    std::vector<UInt64> leaf_bytes;
    std::vector<UInt64> leaf_offset;
    std::vector<UInt32> leaf_order; /// largest first
    std::atomic<UInt32> leaf_claim{0};

    std::pair<size_t, size_t> blockStripe(size_t worker, size_t num_blocks) const
    {
        return {worker * num_blocks / workers, (worker + 1) * num_blocks / workers};
    }
};

namespace
{

/// Bucket ids for one block, derived from the saved routes (MSB-first bit slice); skipped rows
/// (null keys, mask-filtered) go to the drop bucket `partitions`.
void deriveBucketIds(const PaddedPODArray<UInt16> & routes, const UInt8 * skip_bytes, size_t bits, size_t partitions, UInt16 * bucket_ids)
{
    const size_t rows = routes.size();
    const UInt32 shift = static_cast<UInt32>(16 - bits);
    if (skip_bytes)
    {
        for (size_t i = 0; i < rows; ++i)
            bucket_ids[i] = skip_bytes[i] ? static_cast<UInt16>(partitions) : static_cast<UInt16>(routes[i] >> shift);
    }
    else
    {
        for (size_t i = 0; i < rows; ++i)
            bucket_ids[i] = static_cast<UInt16>(routes[i] >> shift);
    }
}

}

void PartitionedHashJoin::insertLeafSection(
    PartitionedJoinMaps & maps,
    const ColumnRawPtrs & key_columns,
    size_t rows,
    const UInt64 * locators,
    const UInt32 * narrow_locators_data,
    UInt32 block_no,
    const UInt8 * skip_bytes,
    Arena & pool,
    bool & all_values_unique)
{
    const Sizes & key_sizes = leaf_join->key_sizes[0];
    const bool enable_prefetch = leaf_join->enableSoftwarePrefetch();

    std::visit(
        [&](auto & shape_maps)
        {
            switch (leaf_join->data->type)
            {
#define M(TYPE) \
    case HashJoin::Type::TYPE: { \
        using Map = typename decltype(shape_maps.TYPE)::element_type; \
        using KeyGetter = typename KeyGetterForType<HashJoin::Type::TYPE, Map>::Type; \
        insertSectionImpl<KeyGetter>( \
            *leaf_join, \
            *shape_maps.TYPE, \
            key_columns, \
            key_sizes, \
            rows, \
            locators, \
            narrow_locators_data, \
            block_no, \
            skip_bytes, \
            pool, \
            all_values_unique, \
            enable_prefetch); \
        break; \
    }
                APPLY_FOR_PARTITIONED_JOIN_VARIANTS(M)
#undef M
                default:
                    throw Exception(
                        ErrorCodes::UNSUPPORTED_JOIN_KEYS,
                        "Unsupported JOIN keys for the partitioned join (type: {})",
                        leaf_join->data->type);
            }
        },
        maps.maps);
}

void PartitionedHashJoin::runPostBuildPhase()
{
    chassert(!build_phase_finished);

    if (delegate_mode)
    {
        /// The standard machinery already built and finished during the fill and the barrier.
        /// Its own single-map post-build optimizations (rerange, fixed-hash-map conversion,
        /// runtime-filter publish) stay off, as for the partitioned path.
        build_phase_finished = true;
        return;
    }

    bool all_values_unique = true;
    if (bits == 0)
    {
        ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);
        all_values_unique = postBuildSingleLeaf();
    }
    else
    {
        all_values_unique = postBuildPartitioned();
    }

    ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);

    /// Free the remaining fill transients before the probe (G3): routes and prepared key columns
    /// are dropped as the scatter consumes them; the block shells and lane bookkeeping go here.
    build_blocks.clear();
    build_blocks.shrink_to_fit();
    /// The route transients are gone; from now on the byte count tracks the stored blocks.
    accumulated_bytes.store(leaf_join->data->allocated_size, std::memory_order_relaxed);

    ProfileEvents::increment(ProfileEvents::PartitionedHashJoinHashTableBytes, ht_slab_bytes);
    if (const UInt64 fallbacks = heap_fallbacks.load(std::memory_order_relaxed))
        ProfileEvents::increment(ProfileEvents::PartitionedHashJoinHashTableHeapFallbacks, fallbacks);

    finishBuildPhase(all_values_unique);

    LOG_TRACE(
        log,
        "Built {} leaf hash tables: {} keys in {} of hash tables ({} carved from one {} slab, {} heap fallbacks)",
        partitions,
        getTotalRowCount(),
        ReadableSize(getTotalByteCount()),
        region_carves.load(std::memory_order_relaxed),
        ReadableSize(ht_slab_bytes),
        heap_fallbacks.load(std::memory_order_relaxed));
}

void PartitionedHashJoin::finishBuildPhase(bool all_values_unique)
{
    /// The leaf's own barrier work: used-flags init over its (empty) map, the ALL -> RightAny
    /// promotion when every build key turned out unique (our probe dispatches on the promoted
    /// strictness), and the non-joined-rows status. The used flags are then re-sized to span
    /// all leaf maps - after the leaf builds, so the bucket counts are final.
    leaf_join->all_values_unique = all_values_unique;
    leaf_join->onBuildPhaseFinish();
    computeFlagBaseAndReinitUsedFlags();
    leaf_join->data->keys_to_join = getTotalRowCount();
    build_phase_finished = true;
}

void PartitionedHashJoin::computeFlagBaseAndReinitUsedFlags()
{
    /// One per-offset flag space spans all leaves: leaf L's flags start at `flag_base[L]`, with
    /// bucket count + 1 slots per leaf (the +1 covers the hash table's zero-value cell, exactly
    /// like the standard sizing `getBufferSizeInCells() + 1`). The probe shifts every
    /// `FindResult` offset by its leaf's base, so `JoinUsedFlags` and the non-joined iteration
    /// keep their single-map semantics.
    const HashJoin::Type type = leaf_join->data->type;
    flag_base.assign(1, 0);
    flag_base.reserve(leaf_maps.size() + 1);
    for (const auto & maps : leaf_maps)
        flag_base.push_back(flag_base.back() + maps.getBufferSizeInCells(type) + 1);

    /// `reinit` only grows and is a no-op for shapes without right-side used flags; it runs
    /// after the leaf `HashJoin`'s own barrier re-initialized the flags to its empty map's size.
    const bool prefer_use_maps_all = leaf_join->preferUseMapsAll();
    joinDispatch(
        leaf_join->getKind(),
        leaf_join->getStrictness(),
        leaf_join->data->maps.front(),
        prefer_use_maps_all,
        [&](auto kind_, auto strictness_, auto & map_)
        {
            leaf_join->used_flags->reinit<kind_, strictness_, std::is_same_v<std::decay_t<decltype(map_)>, HashJoin::MapsAll>>(
                flag_base.back());
        });

    /// Shapes that never consult right-side used flags do not pay for the vector; make that
    /// visible to tests through an empty base table.
    if (!leaf_join->used_flags->need_flags)
        flag_base.clear();
}

bool PartitionedHashJoin::postBuildSingleLeaf()
{
    const HashJoin::Type type = leaf_join->data->type;

    /// The degenerate plan (G6): one leaf over the whole build, exact-reserved from the sketch,
    /// no scatter - rows are inserted straight from the stored blocks with standard
    /// `RowRef(block_no, row)` refs, skipped rows excluded by the saved skip bytes.
    const size_t insertable_rows = accumulated_rows.load(std::memory_order_relaxed);
    const auto reserve
        = std::clamp<size_t>(static_cast<size_t>(std::ceil(hll_estimate * reserve_safety)), 1, std::max<size_t>(insertable_rows, 1));
    const size_t predicted_bytes = PartitionedJoinMaps::predictedBufferBytes(maps_variant_index, type, reserve);

    ht_slab_bytes = predicted_bytes;
    ht_slab = static_cast<char *>(slab_allocator.alloc(ht_slab_bytes, ColumnsScatter::LINE_BYTES));
    ++stats.slab_allocations;

    leaf_maps.assign(1, PartitionedJoinMaps(maps_variant_index));
    build_arenas.emplace_back();

    FixedRegionAllocator::Region region{ht_slab, predicted_bytes, &region_carves, &heap_fallbacks};
    FixedRegionAllocator::armRegion(region);
    leaf_maps[0].create(type, reserve);
    stats.predictions_exact = leaf_maps[0].getBufferSizeInBytes(type) == predicted_bytes;

    bool all_values_unique = true;
    for (auto & fill : build_blocks)
    {
        insertLeafSection(
            leaf_maps[0],
            fill.key_columns,
            fill.rows,
            /*locators=*/nullptr,
            /*narrow_locators_data=*/nullptr,
            fill.block_no,
            fill.skipData(),
            build_arenas.front(),
            all_values_unique);
        ProfileEvents::increment(ProfileEvents::PartitionedHashJoinLeafRows, fill.rows);
        stats.leaf_rows += fill.rows;

        /// Consumed: drop the prepared keys and the routes of this block.
        fill.keys_holder.clear();
        fill.key_columns.clear();
        fill.null_map_holder.reset();
        fill.null_map = nullptr;
        fill.join_mask = JoinCommon::JoinMask();
        fill.skip_bytes = {};
        fill.routes = {};
    }
    return all_values_unique;
}

bool PartitionedHashJoin::postBuildPartitioned()
{
    PostBuildContext ctx;
    ctx.workers = std::max<size_t>(1, std::min(num_threads, build_blocks.size()));
    ctx.fanout = partitions + 1;
    ctx.num_key_columns = build_blocks.front().key_columns.size();

    ctx.generic_mode = false;
    ctx.fixed_widths.resize(ctx.num_key_columns);
    for (size_t c = 0; c < ctx.num_key_columns; ++c)
    {
        const IColumn & column = *build_blocks.front().key_columns[c];
        if (column.isFixedAndContiguous())
            ctx.fixed_widths[c] = column.sizeOfValueIfFixed();
        else
            ctx.generic_mode = true;
    }

    ctx.worker_hist.resize_fill(ctx.workers * ctx.fanout, 0);
    ctx.bucket_rows.assign(ctx.fanout, 0);
    ctx.starts.resize(ctx.fanout * ctx.workers);
    if (narrow_locators)
        ctx.locators32.resize(ctx.fanout);
    else
        ctx.locators.resize(ctx.fanout);
    if (ctx.generic_mode)
    {
        ctx.pieces.resize(ctx.num_key_columns);
        for (auto & column_pieces : ctx.pieces)
            column_pieces.resize(ctx.workers);
    }
    else
    {
        ctx.fixed_out.resize(ctx.num_key_columns);
        for (auto & column_out : ctx.fixed_out)
            column_out.resize(ctx.fanout);
        ctx.fixed_base.assign(ctx.num_key_columns, std::vector<char *>(ctx.fanout, nullptr));
    }
    ctx.worker_state.resize(ctx.workers);
    for (size_t w = 0; w < ctx.workers; ++w)
        build_arenas.emplace_back();

    post_build_pool = std::make_unique<ThreadPool>(
        CurrentMetrics::PartitionedHashJoinPoolThreads,
        CurrentMetrics::PartitionedHashJoinPoolThreadsActive,
        CurrentMetrics::PartitionedHashJoinPoolThreadsScheduled,
        /*max_threads_*/ ctx.workers,
        /*max_free_threads_*/ 0,
        /*queue_size_*/ ctx.workers);

    /// Each stage runs as one wave of jobs; `post_build_pool->wait()` is the stage barrier. The build
    /// ProfileEvent accumulates per-worker THREAD time inside the jobs, keeping its unit
    /// identical to the summed thread time `parallel_hash`'s build event reports.
    auto run_wave = [&](auto && stage)
    {
        try
        {
            for (size_t w = 0; w < ctx.workers; ++w)
                post_build_pool->scheduleOrThrow(
                    [&stage, w, thread_group = CurrentThread::getGroup()]
                    {
                        ThreadGroupSwitcher switcher(thread_group, ThreadName::PARTITIONED_JOIN);
                        ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);
                        stage(w);
                    });
            post_build_pool->wait();
        }
        catch (...)
        {
            post_build_pool->wait();
            throw;
        }
    };

    run_wave([&](size_t w) { histogramWorker(ctx, w); });
    run_wave([&](size_t w) { allocateWorker(ctx, w); });
    run_wave([&](size_t w) { scatterWorker(ctx, w); });
    {
        /// The single contiguous hash-table allocation, right before the leaf builds (R1);
        /// caller-thread work, counted like any other build-thread time.
        ProfileEventTimeIncrement<Microseconds> watch(ProfileEvents::PartitionedHashJoinBuildMicroseconds);
        planAndAllocateHashTables(ctx);
    }
    run_wave([&](size_t w) { leafBuildWorker(ctx, w); });

    post_build_pool.reset();

    bool all_values_unique = true;
    for (const auto & worker : ctx.worker_state)
    {
        all_values_unique &= worker.all_values_unique;
        stats.predictions_exact = stats.predictions_exact && worker.predictions_exact;
        stats.leaf_rows += worker.leaf_rows;
    }
    return all_values_unique;
}

void PartitionedHashJoin::histogramWorker(PostBuildContext & ctx, size_t worker) const
{
    UInt64 * hist = ctx.worker_hist.data() + worker * ctx.fanout;

    PaddedPODArray<UInt64> hist_lanes_mem;
    UInt64 * hist_lanes = nullptr;
    if (ctx.fanout <= ColumnsScatter::HIST_INTERLEAVE_MAX_FANOUT)
    {
        hist_lanes_mem.resize_fill(4 * ctx.fanout, 0);
        hist_lanes = hist_lanes_mem.data();
    }

    PaddedPODArray<UInt16> bucket_ids;
    const auto [begin, end] = ctx.blockStripe(worker, build_blocks.size());
    for (size_t b = begin; b < end; ++b)
    {
        const FillBlock & fill = build_blocks[b];
        bucket_ids.resize(fill.rows);
        deriveBucketIds(fill.routes, fill.skipData(), bits, partitions, bucket_ids.data());
        ColumnsScatter::histogramPidChunk(bucket_ids.data(), fill.rows, hist, hist_lanes, ctx.fanout);
    }
    if (hist_lanes)
        ColumnsScatter::reduceHistogramLanes(hist, hist_lanes, ctx.fanout);
}

void PartitionedHashJoin::allocateWorker(PostBuildContext & ctx, size_t worker) const
{
    /// Fused parallel prefix sum over the per-worker histograms plus ONE exact uninitialized
    /// allocation per (bucket, scattered column) - pages are first-touched by the scatter
    /// writes themselves. Buckets are striped across the workers.
    const size_t buckets_begin = worker * ctx.fanout / ctx.workers;
    const size_t buckets_end = (worker + 1) * ctx.fanout / ctx.workers;
    for (size_t p = buckets_begin; p < buckets_end; ++p)
    {
        UInt64 running = 0;
        for (size_t w = 0; w < ctx.workers; ++w)
        {
            ctx.starts[p * ctx.workers + w] = running;
            running += ctx.worker_hist[w * ctx.fanout + p];
        }
        ctx.bucket_rows[p] = running;
        if (narrow_locators)
            ctx.locators32[p].resize_exact(running);
        else
            ctx.locators[p].resize_exact(running);
        if (!ctx.generic_mode)
        {
            for (size_t c = 0; c < ctx.num_key_columns; ++c)
            {
                auto [column, raw] = ColumnsScatter::allocateUninitializedFixed(*build_blocks.front().key_columns[c], running);
                ctx.fixed_out[c][p] = std::move(column);
                ctx.fixed_base[c][p] = raw.data();
            }
        }
    }
}

void PartitionedHashJoin::scatterWorker(PostBuildContext & ctx, size_t worker)
{
    auto & state = ctx.worker_state[worker];
    const auto [begin, end] = ctx.blockStripe(worker, build_blocks.size());

    const size_t locator_width = narrow_locators ? sizeof(UInt32) : sizeof(UInt64);
    const bool locator_swwc = ctx.fanout >= ColumnsScatter::SWWC_MIN_FANOUT;
    state.locator_scratch.init(ctx.fanout, locator_swwc);
    if (narrow_locators)
        state.locator_piece32.resize(locator_piece_rows);
    else
        state.locator_piece.resize(locator_piece_rows);
    for (size_t p = 0; p < ctx.fanout; ++p)
    {
        const UInt64 start = ctx.starts[p * ctx.workers + worker];
        char * cursor = narrow_locators ? reinterpret_cast<char *>(ctx.locators32[p].data() + start)
                                        : reinterpret_cast<char *>(ctx.locators[p].data() + start);
        state.locator_scratch.seed(p, cursor);
    }

    /// The locator rows must land in the same per-bucket positions the histogram assigned, so
    /// bucket ids are derived once per block and shared by every scattered column of the block.
    auto scatter_locators = [&](const FillBlock & fill, const UInt16 * bucket_ids)
    {
        for (size_t offset = 0; offset < fill.rows; offset += locator_piece_rows)
        {
            const size_t piece = std::min(locator_piece_rows, fill.rows - offset);
            const char * piece_data = nullptr;
            if (narrow_locators)
            {
                for (size_t j = 0; j < piece; ++j)
                    state.locator_piece32[j] = static_cast<UInt32>((fill.block_no << 16) | (offset + j));
                piece_data = reinterpret_cast<const char *>(state.locator_piece32.data());
            }
            else
            {
                for (size_t j = 0; j < piece; ++j)
                    state.locator_piece[j] = RowRef(fill.block_no, offset + j).encode();
                piece_data = reinterpret_cast<const char *>(state.locator_piece.data());
            }
            ColumnsScatter::scatterPidChunk(locator_width, bucket_ids + offset, piece_data, piece, locator_swwc, state.locator_scratch);
        }
    };

    auto release_block_inputs = [](FillBlock & fill)
    {
        fill.keys_holder.clear();
        fill.key_columns.clear();
        fill.null_map_holder.reset();
        fill.null_map = nullptr;
        fill.join_mask = JoinCommon::JoinMask();
        fill.skip_bytes = {};
        fill.routes = {};
    };

    if (!ctx.generic_mode)
    {
        state.key_scratch.resize(ctx.num_key_columns);
        std::vector<bool> key_swwc(ctx.num_key_columns);
        for (size_t c = 0; c < ctx.num_key_columns; ++c)
        {
            key_swwc[c] = ctx.fanout >= ColumnsScatter::SWWC_MIN_FANOUT && ColumnsScatter::widthSupportsSwwc(ctx.fixed_widths[c]);
            state.key_scratch[c].init(ctx.fanout, key_swwc[c]);
            for (size_t p = 0; p < ctx.fanout; ++p)
                state.key_scratch[c].seed(p, ctx.fixed_base[c][p] + ctx.starts[p * ctx.workers + worker] * ctx.fixed_widths[c]);
        }

        /// Fused batched scatter: whole-block batches sized by `scatterBatchRowsTarget`; the
        /// per-(column, bucket) cursors persist across batches, and a batch's input chunks are
        /// dropped as soon as their last column is scattered (memory cycling).
        const size_t batch_rows_target = ColumnsScatter::scatterBatchRowsTarget(ctx.fanout);
        std::vector<PaddedPODArray<UInt16>> batch_bucket_ids;
        size_t b = begin;
        while (b < end)
        {
            const size_t batch_begin = b;
            size_t batch_rows = 0;
            while (b < end && batch_rows < batch_rows_target)
            {
                batch_rows += build_blocks[b].rows;
                ++b;
            }
            batch_bucket_ids.resize(b - batch_begin);
            for (size_t i = batch_begin; i < b; ++i)
            {
                const FillBlock & fill = build_blocks[i];
                batch_bucket_ids[i - batch_begin].resize(fill.rows);
                deriveBucketIds(fill.routes, fill.skipData(), bits, partitions, batch_bucket_ids[i - batch_begin].data());
            }
            for (size_t c = 0; c < ctx.num_key_columns; ++c)
                for (size_t i = batch_begin; i < b; ++i)
                    /// The kernel consumes rows * width bytes; getRawData is documented to span
                    /// exactly that, the .size() of the view itself is not used.
                    ColumnsScatter::scatterPidChunk(
                        ctx.fixed_widths[c],
                        batch_bucket_ids[i - batch_begin].data(),
                        build_blocks[i].key_columns[c]->getRawData().data(), /// NOLINT(bugprone-suspicious-stringview-data-usage)
                        build_blocks[i].rows,
                        key_swwc[c],
                        state.key_scratch[c]);
            for (size_t i = batch_begin; i < b; ++i)
                scatter_locators(build_blocks[i], batch_bucket_ids[i - batch_begin].data());
            for (size_t i = batch_begin; i < b; ++i)
                release_block_inputs(build_blocks[i]);
        }

        for (auto & scratch : state.key_scratch)
            scratch.drain();
        state.locator_scratch.drain();
        return;
    }

    /// Generic mode (String / LowCardinality / exotic key columns): each worker scatters its
    /// block stripe through the type-complete Layer-1 kernels into self-contained per-bucket
    /// pieces. The overflow-tolerant per-piece allocations satisfy the String kernel's
    /// overflow-15 contract, and worker-private pieces make the parallelism safe; the leaf
    /// builds consume the pieces in worker order, which matches the cooperative locator layout.
    std::vector<PaddedPODArray<UInt16>> stripe_bucket_ids(end - begin);
    std::vector<std::span<const UInt16>> bucket_id_spans(end - begin);
    for (size_t i = begin; i < end; ++i)
    {
        const FillBlock & fill = build_blocks[i];
        stripe_bucket_ids[i - begin].resize(fill.rows);
        deriveBucketIds(fill.routes, fill.skipData(), bits, partitions, stripe_bucket_ids[i - begin].data());
        bucket_id_spans[i - begin] = {stripe_bucket_ids[i - begin].data(), fill.rows};
        scatter_locators(fill, stripe_bucket_ids[i - begin].data());
    }
    state.locator_scratch.drain();

    std::vector<const IColumn *> sources(end - begin);
    for (size_t c = 0; c < ctx.num_key_columns; ++c)
    {
        for (size_t i = begin; i < end; ++i)
            sources[i - begin] = build_blocks[i].key_columns[c];
        ctx.pieces[c][worker] = ColumnsScatter::scatter(sources, bucket_id_spans, ctx.fanout);
        /// Drop the null bucket's piece right away - those rows are never inserted.
        ctx.pieces[c][worker][partitions].reset();
    }
    for (size_t i = begin; i < end; ++i)
        release_block_inputs(build_blocks[i]);
}

void PartitionedHashJoin::planAndAllocateHashTables(PostBuildContext & ctx)
{
    /// Drop the null bucket's outputs: null-key rows are never inserted.
    if (narrow_locators)
        ctx.locators32[partitions] = {};
    else
        ctx.locators[partitions] = {};
    if (!ctx.generic_mode)
        for (size_t c = 0; c < ctx.num_key_columns; ++c)
            ctx.fixed_out[c][partitions].reset();

    const HashJoin::Type type = leaf_join->data->type;
    const auto per_leaf_estimate
        = std::max<UInt64>(1, static_cast<UInt64>(std::ceil(hll_estimate * reserve_safety / static_cast<double>(partitions))));

    ctx.leaf_reserve.resize(partitions);
    ctx.leaf_bytes.resize(partitions);
    ctx.leaf_offset.resize(partitions);
    UInt64 running = 0;
    for (size_t leaf = 0; leaf < partitions; ++leaf)
    {
        /// The sketch estimate can only shrink a leaf below its row count, never inflate it.
        ctx.leaf_reserve[leaf] = std::clamp<UInt64>(per_leaf_estimate, 1, std::max<UInt64>(ctx.bucket_rows[leaf], 1));
        ctx.leaf_bytes[leaf] = PartitionedJoinMaps::predictedBufferBytes(maps_variant_index, type, ctx.leaf_reserve[leaf]);
        running = (running + ColumnsScatter::LINE_BYTES - 1) & ~static_cast<UInt64>(ColumnsScatter::LINE_BYTES - 1);
        ctx.leaf_offset[leaf] = running;
        running += ctx.leaf_bytes[leaf];
    }

    /// ONE contiguous allocation backs all leaf hash tables: exact-sized, 64-byte aligned, NOT
    /// zeroed (each worker zeroes exactly its leaf region right before filling it).
    ht_slab_bytes = running;
    ht_slab = static_cast<char *>(slab_allocator.alloc(ht_slab_bytes, ColumnsScatter::LINE_BYTES));
    ++stats.slab_allocations;

    leaf_maps.assign(partitions, PartitionedJoinMaps(maps_variant_index));
    ctx.leaf_order.resize(partitions);
    for (size_t leaf = 0; leaf < partitions; ++leaf)
        ctx.leaf_order[leaf] = static_cast<UInt32>(leaf);
    std::sort(ctx.leaf_order.begin(), ctx.leaf_order.end(), [&](UInt32 a, UInt32 b) { return ctx.bucket_rows[a] > ctx.bucket_rows[b]; });
}

void PartitionedHashJoin::leafBuildWorker(PostBuildContext & ctx, size_t worker)
{
    const HashJoin::Type type = leaf_join->data->type;
    auto & state = ctx.worker_state[worker];
    Arena & arena = build_arenas[worker];

    ColumnRawPtrs section_columns(ctx.num_key_columns);

    /// Leaves are claimed dynamically, largest first (G6): skew can never serialize the build
    /// on a worker-partition affinity.
    while (true)
    {
        const UInt32 claim = ctx.leaf_claim.fetch_add(1, std::memory_order_relaxed);
        if (claim >= partitions)
            break;
        const UInt32 leaf = ctx.leaf_order[claim];

        FixedRegionAllocator::Region region{ht_slab + ctx.leaf_offset[leaf], ctx.leaf_bytes[leaf], &region_carves, &heap_fallbacks};
        FixedRegionAllocator::armRegion(region);
        leaf_maps[leaf].create(type, ctx.leaf_reserve[leaf]);
        state.predictions_exact = state.predictions_exact && leaf_maps[leaf].getBufferSizeInBytes(type) == ctx.leaf_bytes[leaf];

        const UInt64 leaf_rows = ctx.bucket_rows[leaf];
        if (!ctx.generic_mode)
        {
            for (size_t c = 0; c < ctx.num_key_columns; ++c)
                section_columns[c] = ctx.fixed_out[c][leaf].get();
            insertLeafSection(
                leaf_maps[leaf],
                section_columns,
                leaf_rows,
                narrow_locators ? nullptr : ctx.locators[leaf].data(),
                narrow_locators ? ctx.locators32[leaf].data() : nullptr,
                /*block_no=*/0,
                /*skip_bytes=*/nullptr,
                arena,
                state.all_values_unique);
        }
        else
        {
            /// The pieces of one leaf, in worker order, are exactly the locator layout.
            for (size_t piece_worker = 0; piece_worker < ctx.workers; ++piece_worker)
            {
                const size_t piece_rows = ctx.worker_hist[piece_worker * ctx.fanout + leaf];
                if (piece_rows == 0)
                    continue;
                for (size_t c = 0; c < ctx.num_key_columns; ++c)
                    section_columns[c] = ctx.pieces[c][piece_worker][leaf].get();
                const UInt64 piece_start = ctx.starts[leaf * ctx.workers + piece_worker];
                insertLeafSection(
                    leaf_maps[leaf],
                    section_columns,
                    piece_rows,
                    narrow_locators ? nullptr : ctx.locators[leaf].data() + piece_start,
                    narrow_locators ? ctx.locators32[leaf].data() + piece_start : nullptr,
                    /*block_no=*/0,
                    /*skip_bytes=*/nullptr,
                    arena,
                    state.all_values_unique);
            }
        }
        state.leaf_rows += leaf_rows;
        ProfileEvents::increment(ProfileEvents::PartitionedHashJoinLeafRows, leaf_rows);

        /// Release the leaf's scatter chunks as soon as they are consumed.
        if (narrow_locators)
            ctx.locators32[leaf] = {};
        else
            ctx.locators[leaf] = {};
        if (!ctx.generic_mode)
        {
            for (size_t c = 0; c < ctx.num_key_columns; ++c)
                ctx.fixed_out[c][leaf].reset();
        }
        else
        {
            for (size_t c = 0; c < ctx.num_key_columns; ++c)
                for (size_t piece_worker = 0; piece_worker < ctx.workers; ++piece_worker)
                    ctx.pieces[c][piece_worker][leaf].reset();
        }
    }
}

}
