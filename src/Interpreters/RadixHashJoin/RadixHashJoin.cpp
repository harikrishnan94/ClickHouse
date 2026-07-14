#include <Interpreters/RadixHashJoin/RadixHashJoin.h>

#include <Columns/ColumnsScatter.h>

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/HashJoin/ScatteredBlock.h>
#include <Interpreters/TableJoin.h>

#include <Common/ConcurrentBoundedQueue.h>
#include <Common/CurrentThread.h>
#include <Common/ElapsedTimeProfileEventIncrement.h>
#include <Common/Exception.h>
#include <Common/PODArray.h>
#include <Common/ProfileEvents.h>
#include <Common/SharedMutex.h>
#include <Common/Stopwatch.h>
#include <Common/ThreadGroupSwitcher.h>
#include <Common/ThreadPool.h>
#include <Common/formatReadable.h>
#include <Common/logger_useful.h>
#include <Common/setThreadName.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <list>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>

namespace ProfileEvents
{
extern const Event RadixHashJoinBuildMicroseconds;
extern const Event RadixHashJoinProbeMicroseconds;
extern const Event RadixHashJoinProbePackHashRouteMicroseconds;
extern const Event RadixHashJoinLeafGroupBuilds;
extern const Event RadixHashJoinLeafGroupBuildMicroseconds;
}

namespace CurrentMetrics
{
extern const Metric RadixHashJoinPoolThreads;
extern const Metric RadixHashJoinPoolThreadsActive;
extern const Metric RadixHashJoinPoolThreadsScheduled;
}

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int BAD_ARGUMENTS;
extern const int SET_SIZE_LIMIT_EXCEEDED;
}

namespace
{

/// ---------------------------------------------------------------------------------------------
/// The radix scatter kernels formerly defined here live in `ColumnsScatter` (semantically
/// identical extraction; the hot row loops run inside that TU and are called at chunk
/// granularity, so the call cost is per (batch, column), never per row). The join keeps its own
/// planning constants and the partition/pass orchestration below.
/// ---------------------------------------------------------------------------------------------

using ColumnsScatter::ScatterScratch;
using ColumnsScatter::scatterKeyChunk;
using ColumnsScatter::scatterPidChunk;
using ColumnsScatter::histogramKeyChunk;
using ColumnsScatter::histogramRouteChunk;
using ColumnsScatter::reduceHistogramLanes;
using ColumnsScatter::foldBytes;
using ColumnsScatter::finalizeRoute;
using ColumnsScatter::scatterBatchRowsTarget;
using ColumnsScatter::widthSupportsSwwc;
using ColumnsScatter::SWWC_MIN_FANOUT;
using ColumnsScatter::HIST_INTERLEAVE_MAX_FANOUT;
using ColumnsScatter::MAX_FANOUT_PER_PASS;

/// Partition-plan constants (5.1): the target leaf working set (~L2) and the per-entry hash-table
/// byte estimate (a cell at 0.5 load factor, matching the bench bandwidth model). The per-pass
/// fanout ceiling is the module's `MAX_FANOUT_PER_PASS`.
constexpr size_t LEAF_TARGET_BYTES = 1 << 20;
constexpr size_t HT_CELL_BYTES = 16;

/// The fixed-width layout of one side (build or probe): column widths in bytes and the key columns.
struct SideLayout
{
    size_t num_columns = 0;
    std::vector<size_t> col_widths;
    std::vector<size_t> key_positions;
    std::vector<size_t> key_widths;
    bool single_key = false;
    size_t key_pos = 0;
    size_t key_width = 0;
};

SideLayout makeSideLayout(const Block & header, const Names & key_names)
{
    SideLayout layout;
    layout.num_columns = header.columns();
    layout.col_widths.resize(layout.num_columns);
    for (size_t j = 0; j < layout.num_columns; ++j)
    {
        const auto & column = header.getByPosition(j).column;
        if (!column->isFixedAndContiguous())
            throw Exception(
                ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: column {} is not fixed-and-contiguous", header.getByPosition(j).name);
        layout.col_widths[j] = column->sizeOfValueIfFixed();
    }
    for (const auto & name : key_names)
    {
        const size_t pos = header.getPositionByName(name);
        layout.key_positions.push_back(pos);
        layout.key_widths.push_back(layout.col_widths[pos]);
    }
    layout.single_key = layout.key_positions.size() == 1;
    if (layout.single_key)
    {
        layout.key_pos = layout.key_positions[0];
        layout.key_width = layout.key_widths[0];
    }
    return layout;
}

/// Exactly-sized destination columns of one output partition, with raw write bases.
struct PartitionOutput
{
    MutableColumns columns;
    std::vector<char *> bases;
    size_t rows = 0;

    void initialize(const Block & header, size_t rows_)
    {
        rows = rows_;
        columns.resize(header.columns());
        bases.resize(header.columns(), nullptr);
    }

    /// createColumn()+insertRawUninitialized leaves POD contents uninitialized: no memset, pages are
    /// first-touched by the scatter writes themselves. Refine passes call this just-in-time so the
    /// allocator can reuse the input column released by the preceding scatter round.
    void allocateColumn(const Block & header, const std::vector<size_t> & col_widths, size_t j)
    {
        auto col = header.getByPosition(j).type->createColumn();
        auto span = col->insertRawUninitialized(rows);
        chassert(span.size() == rows * col_widths[j]);
        bases[j] = span.data();
        columns[j] = std::move(col);
    }

    void allocate(const Block & header, const std::vector<size_t> & col_widths, size_t rows_)
    {
        initialize(header, rows_);
        for (size_t j = 0; j < header.columns(); ++j)
            allocateColumn(header, col_widths, j);
    }

    Block toBlock(const Block & header)
    {
        Columns cols;
        cols.reserve(columns.size());
        for (auto & col : columns)
            cols.emplace_back(std::move(col));
        return header.cloneWithColumns(cols);
    }
};

/// Runs `fn(tid)` on `threads` pool workers and waits (rethrows the first worker exception).
void parallelRun(ThreadPool & pool, size_t threads, const ThreadGroupPtr & thread_group, const std::function<void(size_t)> & fn)
{
    try
    {
        for (size_t t = 0; t < threads; ++t)
            pool.scheduleOrThrow(
                [&fn, t, thread_group]
                {
                    ThreadGroupSwitcher switcher(thread_group, ThreadName::RADIX_JOIN);
                    fn(t);
                });
    }
    catch (...)
    {
        /// Unwinding without the wait would free `fn` and the caller's stack while the jobs that did
        /// get scheduled still reference both.
        pool.wait();
        throw;
    }
    pool.wait();
}

std::vector<size_t> computePassBits(size_t partitions, size_t max_fanout)
{
    const size_t total_bits = std::countr_zero(std::bit_ceil(partitions));
    const size_t fanout_bits = std::max<size_t>(1, std::bit_width(std::bit_floor(std::max<size_t>(2, max_fanout))) - 1);
    const size_t num_passes = (total_bits + fanout_bits - 1) / fanout_bits;
    const size_t bits_per_pass = (total_bits + num_passes - 1) / num_passes;

    std::vector<size_t> result;
    result.reserve(num_passes);
    size_t remaining = total_bits;
    while (remaining)
    {
        const size_t bits = std::min(bits_per_pass, remaining);
        result.push_back(bits);
        remaining -= bits;
    }
    return result;
}

std::vector<size_t> makeScatterOrder(const SideLayout & layout)
{
    std::vector<size_t> order;
    order.reserve(layout.num_columns);
    if (!layout.single_key)
    {
        for (size_t j = 0; j < layout.num_columns; ++j)
            order.push_back(j);
        return order;
    }

    order.push_back(layout.key_pos);
    for (size_t j = 0; j < layout.num_columns; ++j)
        if (j != layout.key_pos)
            order.push_back(j);
    return order;
}

/// Fold one key column's rows into the route accumulators with a compile-time width: `foldBytes`
/// fully unrolls into plain loads, so the row body is one load + mix per 8-byte chunk with no
/// per-row width checks or tail dispatch.
template <size_t width>
void foldKeyColumnRows(const char * base, size_t n, UInt64 * acc)
{
    for (size_t i = 0; i < n; ++i)
        acc[i] = foldBytes(acc[i], base + i * width, width);
}

/// Composite-key route materialization for one chunk of `n` rows: fold every key column's bytes
/// into per-row accumulators (`foldBytes`), then finalize each accumulator into a route word.
/// `get_key_column_base` maps a key column position to its raw data pointer. `acc` is caller-owned
/// scratch so the first pass can reuse one allocation across its whole chunk stripe. Called at
/// chunk granularity: the hot row loops live inside, never behind a per-row call. The fold row
/// loop is width-dispatched like the module's scatter kernels (the common fixed widths hit the
/// unrolled loops; anything else folds through the runtime-width loop).
template <typename GetKeyColumnBase>
void materializeCompositeRoutes(
    const SideLayout & layout, const GetKeyColumnBase & get_key_column_base, size_t n, PaddedPODArray<UInt64> & acc, UInt32 * routes)
{
    acc.resize(n);
    memset(acc.data(), 0, n * sizeof(UInt64));
    for (size_t k = 0; k < layout.key_positions.size(); ++k)
    {
        const size_t w = layout.key_widths[k];
        const char * base = get_key_column_base(layout.key_positions[k]);
        switch (w)
        {
            case 1: foldKeyColumnRows<1>(base, n, acc.data()); break;
            case 2: foldKeyColumnRows<2>(base, n, acc.data()); break;
            case 4: foldKeyColumnRows<4>(base, n, acc.data()); break;
            case 8: foldKeyColumnRows<8>(base, n, acc.data()); break;
            case 16: foldKeyColumnRows<16>(base, n, acc.data()); break;
            default:
                for (size_t i = 0; i < n; ++i)
                    acc[i] = foldBytes(acc[i], base + i * w, w);
        }
    }
    for (size_t i = 0; i < n; ++i)
        routes[i] = finalizeRoute(acc[i]);
}

/// First radix pass: all workers cooperate on the single input group in exactly three barriers
/// (histogram, fused prefix-sum + exact allocation, fused batched column-major scatter). `blocks` is
/// consumed batch-eagerly. `Counter` is selected once from the exact side row count, outside the row
/// loops: `UInt32` in the common case and `UInt64` only for sides larger than 2^32 - 1 rows.
template <typename Counter>
std::vector<PartitionOutput> scatterFirstPass(
    ThreadPool & pool,
    size_t threads,
    const ThreadGroupPtr & thread_group,
    const Block & header,
    std::vector<Block> & blocks,
    const SideLayout & layout,
    size_t bits)
{
    const size_t fanout = 1ULL << bits;
    const UInt32 route_shift = static_cast<UInt32>(32 - bits);
    const UInt32 route_mask = static_cast<UInt32>(fanout - 1);
    const size_t num_chunks = blocks.size();
    const size_t num_columns = layout.num_columns;
    const bool use_swwc_fanout = fanout >= SWWC_MIN_FANOUT;
    const bool interleave_hist = fanout <= HIST_INTERLEAVE_MAX_FANOUT;
    const bool composite = !layout.single_key;

    std::vector<size_t> chunk_rows(num_chunks);
    for (size_t c = 0; c < num_chunks; ++c)
        chunk_rows[c] = blocks[c].rows();

    /// For composite keys, the route words are materialized once (the fold genuinely reads multiple
    /// columns); the single-key path routes straight from the key column and stores nothing.
    std::vector<PaddedPODArray<UInt32>> chunk_routes;
    if (composite)
        chunk_routes.resize(num_chunks);

    /// Barrier 1: per-worker histograms into disjoint slices of one flat array.
    PaddedPODArray<Counter> hist;
    hist.resize(threads * fanout);
    parallelRun(
        pool,
        threads,
        thread_group,
        [&](size_t tid)
        {
            Counter * h = hist.data() + tid * fanout;
            memset(h, 0, fanout * sizeof(Counter));
            std::vector<Counter> lanes;
            if (interleave_hist)
                lanes.assign(4 * fanout, 0);

            PaddedPODArray<UInt64> acc; /// composite fold accumulator, reused per chunk
            for (size_t c = tid; c < num_chunks; c += threads)
            {
                const size_t n = chunk_rows[c];
                if (composite)
                {
                    chunk_routes[c].resize(n);
                    materializeCompositeRoutes(
                        layout,
                        [&](size_t pos) { return blocks[c].getByPosition(pos).column->getRawData().data(); },
                        n,
                        acc,
                        chunk_routes[c].data());
                    histogramRouteChunk(
                        chunk_routes[c].data(), n, route_shift, route_mask, h, interleave_hist ? lanes.data() : nullptr, fanout);
                }
                else
                {
                    const char * keys = blocks[c].getByPosition(layout.key_pos).column->getRawData().data();
                    histogramKeyChunk(
                        layout.key_width, keys, n, route_shift, route_mask, h, interleave_hist ? lanes.data() : nullptr, fanout);
                }
            }
            if (interleave_hist)
                reduceHistogramLanes(h, lanes.data(), fanout);
        });

    /// Barrier 2: fused prefix sum + exact one-shot allocation. Each worker owns a contiguous,
    /// disjoint range of partitions.
    PaddedPODArray<Counter> offsets; /// per (worker, partition) start row within the partition
    offsets.resize(threads * fanout);
    std::vector<size_t> totals(fanout, 0);
    std::vector<PartitionOutput> parts(fanout);
    parallelRun(
        pool,
        threads,
        thread_group,
        [&](size_t tid)
        {
            const size_t begin = fanout * tid / threads;
            const size_t end = fanout * (tid + 1) / threads;
            for (size_t p = begin; p < end; ++p)
            {
                size_t total = 0;
                for (size_t w = 0; w < threads; ++w)
                {
                    offsets[w * fanout + p] = static_cast<Counter>(total);
                    total += hist[w * fanout + p];
                }
                totals[p] = total;
                if (total)
                    parts[p].allocate(header, layout.col_widths, total);
            }
        });

    /// Barrier 3: single fused scatter run, batched. Each worker processes its chunk stripe in batches
    /// of whole chunks; the batch's input chunks are dropped after their last column is scattered.
    const size_t batch_rows_target = scatterBatchRowsTarget(fanout);

    /// Column processing order: single-key routes the key column first (emitting the pids the payload
    /// columns then consume); composite precomputes the pids from the route words up front.
    const std::vector<size_t> scatter_order = makeScatterOrder(layout);
    const bool need_pids = composite || num_columns > 1;

    parallelRun(
        pool,
        threads,
        thread_group,
        [&](size_t tid)
        {
            ScatterScratch scratch;
            scratch.init(fanout, use_swwc_fanout);

            /// Running write cursors per (column, partition), persisted across batches.
            std::vector<char *> col_cursors(num_columns * fanout, nullptr);
            for (size_t j = 0; j < num_columns; ++j)
                for (size_t p = 0; p < fanout; ++p)
                    if (totals[p])
                        col_cursors[j * fanout + p] = parts[p].bases[j] + offsets[tid * fanout + p] * layout.col_widths[j];

            PaddedPODArray<UInt16> pids;
            std::vector<size_t> batch;
            std::vector<size_t> batch_offsets;

            size_t c = tid;
            while (c < num_chunks)
            {
                batch.clear();
                batch_offsets.clear();
                size_t batch_rows = 0;
                for (; c < num_chunks && batch_rows < batch_rows_target; c += threads)
                {
                    batch.push_back(c);
                    batch_offsets.push_back(batch_rows);
                    batch_rows += chunk_rows[c];
                }

                if (need_pids)
                    pids.resize(batch_rows);

                /// Composite: derive the batch's pids from the route words before any column scatters.
                if (composite && need_pids)
                {
                    for (size_t b = 0; b < batch.size(); ++b)
                    {
                        const size_t cc = batch[b];
                        const size_t n = chunk_rows[cc];
                        UInt16 * dst = pids.data() + batch_offsets[b];
                        const UInt32 * routes = chunk_routes[cc].data();
                        for (size_t i = 0; i < n; ++i)
                            dst[i] = static_cast<UInt16>((routes[i] >> route_shift) & route_mask);
                    }
                }

                for (size_t j : scatter_order)
                {
                    const size_t w = layout.col_widths[j];
                    const bool use_swwc = use_swwc_fanout && widthSupportsSwwc(w);
                    scratch.setUseSwwc(use_swwc);
                    for (size_t p = 0; p < fanout; ++p)
                        scratch.seed(p, col_cursors[j * fanout + p]);

                    const bool key_first = !composite && j == layout.key_pos;
                    for (size_t b = 0; b < batch.size(); ++b)
                    {
                        const size_t cc = batch[b];
                        const size_t n = chunk_rows[cc];
                        if (!n)
                            continue;
                        const char * data = blocks[cc].getByPosition(j).column->getRawData().data();
                        UInt16 * pid_slice = need_pids ? pids.data() + batch_offsets[b] : nullptr;
                        if (key_first)
                        {
                            scatterKeyChunk(layout.key_width, data, n, route_shift, route_mask, pid_slice, use_swwc, scratch);
                        }
                        else
                        {
                            scatterPidChunk(w, pids.data() + batch_offsets[b], data, n, use_swwc, scratch);
                        }
                    }
                    scratch.drain();
                    for (size_t p = 0; p < fanout; ++p)
                        col_cursors[j * fanout + p] = scratch.cursors[p];
                }

                /// The batch is fully consumed: drop its input chunks before the next batch.
                for (size_t cc : batch)
                    blocks[cc].clear();
            }
        });

    return parts;
}

/// Refine one previous-pass output group. One worker owns the group for the whole operation. Output
/// columns are allocated just-in-time and each consumed input column is released before the next
/// output-column allocation, keeping resident data near one copy of the group.
template <typename Counter>
void scatterRefineGroup(
    const Block & header,
    const SideLayout & layout,
    PartitionOutput & group,
    size_t bits,
    size_t bits_done,
    std::vector<PartitionOutput> & out,
    size_t out_begin)
{
    const size_t fanout = 1ULL << bits;
    const UInt32 route_shift = static_cast<UInt32>(32 - bits_done - bits);
    const UInt32 route_mask = static_cast<UInt32>(fanout - 1);
    const bool use_swwc_fanout = fanout >= SWWC_MIN_FANOUT;
    const bool interleave_hist = fanout <= HIST_INTERLEAVE_MAX_FANOUT;
    const bool composite = !layout.single_key;
    const bool need_pids = composite || layout.num_columns > 1;

    std::vector<Counter> hist(fanout, 0);
    std::vector<Counter> lanes;
    if (interleave_hist)
        lanes.assign(4 * fanout, 0);

    PaddedPODArray<UInt16> pids;
    if (need_pids)
        pids.resize(group.rows);

    if (composite)
    {
        PaddedPODArray<UInt64> accumulators;
        PaddedPODArray<UInt32> routes;
        routes.resize(group.rows);
        materializeCompositeRoutes(
            layout,
            [&](size_t pos) { return group.columns[pos]->getRawData().data(); },
            group.rows,
            accumulators,
            routes.data());

        histogramRouteChunk(
            routes.data(), group.rows, route_shift, route_mask, hist.data(), interleave_hist ? lanes.data() : nullptr, fanout);
        for (size_t i = 0; i < group.rows; ++i)
            pids[i] = static_cast<UInt16>((routes[i] >> route_shift) & route_mask);
    }
    else
    {
        const char * keys = group.columns[layout.key_pos]->getRawData().data();
        histogramKeyChunk(
            layout.key_width, keys, group.rows, route_shift, route_mask, hist.data(), interleave_hist ? lanes.data() : nullptr, fanout);
    }
    if (interleave_hist)
        reduceHistogramLanes(hist.data(), lanes.data(), fanout);

    std::vector<PartitionOutput> parts(fanout);
    for (size_t p = 0; p < fanout; ++p)
        if (hist[p])
            parts[p].initialize(header, hist[p]);

    ScatterScratch scratch;
    scratch.init(fanout, use_swwc_fanout);
    const std::vector<size_t> scatter_order = makeScatterOrder(layout);
    for (size_t j : scatter_order)
    {
        const size_t width = layout.col_widths[j];
        const bool use_swwc = use_swwc_fanout && widthSupportsSwwc(width);
        scratch.setUseSwwc(use_swwc);
        for (size_t p = 0; p < fanout; ++p)
        {
            if (hist[p])
                parts[p].allocateColumn(header, layout.col_widths, j);
            scratch.seed(p, hist[p] ? parts[p].bases[j] : nullptr);
        }

        const char * data = group.columns[j]->getRawData().data();
        if (!composite && j == layout.key_pos)
        {
            scatterKeyChunk(
                layout.key_width, data, group.rows, route_shift, route_mask, need_pids ? pids.data() : nullptr, use_swwc, scratch);
        }
        else
        {
            scatterPidChunk(width, pids.data(), data, group.rows, use_swwc, scratch);
        }
        scratch.drain();

        group.columns[j].reset();
        group.bases[j] = nullptr;
    }

    for (size_t p = 0; p < fanout; ++p)
        if (hist[p])
            out[out_begin + p] = std::move(parts[p]);

    group.columns.clear();
    group.bases.clear();
    group.rows = 0;
}

/// Later radix passes: dynamically assign differently-sized previous-pass groups to workers. Each
/// group uses the narrowest safe histogram counters, chosen before its row loops.
std::vector<PartitionOutput> scatterRefinePass(
    ThreadPool & pool,
    size_t threads,
    const ThreadGroupPtr & thread_group,
    const Block & header,
    const SideLayout & layout,
    std::vector<PartitionOutput> & groups,
    size_t bits,
    size_t bits_done)
{
    const size_t fanout = 1ULL << bits;
    std::vector<PartitionOutput> out(groups.size() * fanout);
    std::atomic<size_t> next_group{0};

    parallelRun(
        pool,
        threads,
        thread_group,
        [&](size_t)
        {
            for (size_t g = next_group.fetch_add(1, std::memory_order_relaxed); g < groups.size();
                 g = next_group.fetch_add(1, std::memory_order_relaxed))
            {
                if (!groups[g].rows)
                    continue;
                if (groups[g].rows <= std::numeric_limits<UInt32>::max())
                    scatterRefineGroup<UInt32>(header, layout, groups[g], bits, bits_done, out, g * fanout);
                else
                    scatterRefineGroup<UInt64>(header, layout, groups[g], bits, bits_done, out, g * fanout);
            }
        });
    return out;
}

/// Radix-scatter a side according to a plan of disjoint route-word bit slices. A one-pass plan calls
/// only the cooperative three-barrier kernel above. Multi-pass plans feed its exactly-sized outputs
/// through dynamically scheduled, memory-cycling refine passes.
std::vector<PartitionOutput> scatterToPartitions(
    ThreadPool & pool,
    size_t threads,
    const ThreadGroupPtr & thread_group,
    const Block & header,
    std::vector<Block> & blocks,
    const SideLayout & layout,
    const std::vector<size_t> & pass_bits)
{
    chassert(!pass_bits.empty());
    size_t total_rows = 0;
    for (const auto & block : blocks)
        total_rows += block.rows();

    std::vector<PartitionOutput> groups;
    if (total_rows <= std::numeric_limits<UInt32>::max())
        groups = scatterFirstPass<UInt32>(pool, threads, thread_group, header, blocks, layout, pass_bits.front());
    else
        groups = scatterFirstPass<UInt64>(pool, threads, thread_group, header, blocks, layout, pass_bits.front());

    size_t bits_done = pass_bits.front();
    for (size_t pass = 1; pass < pass_bits.size(); ++pass)
    {
        groups = scatterRefinePass(pool, threads, thread_group, header, layout, groups, pass_bits[pass], bits_done);
        bits_done += pass_bits[pass];
    }
    return groups;
}

}

/// -------------------------------------------------------------------------------------------------
/// State
/// -------------------------------------------------------------------------------------------------

namespace
{

struct ActiveWave;

/// Admits one wave at a time. Every probe lane's result drains the active wave cooperatively, so a
/// lane that arrives while a wave is in flight becomes one of its consumers instead of waiting.
struct WaveCoordinator
{
    std::mutex mutex;
    std::shared_ptr<ActiveWave> active_wave;
};

}

struct RadixHashJoin::State
{
    Names left_key_names;
    Names right_key_names;

    /// Build accumulation. One slot per build lane; lanes are stable per stream but not guaranteed
    /// distinct across streams (IJoin.h contract), so each has a mutex (uncontended in practice).
    struct BuildLane
    {
        std::mutex mutex;
        std::vector<Block> blocks;
    };
    std::vector<BuildLane> build_lanes;
    std::atomic<size_t> build_rows{0};
    std::atomic<size_t> build_bytes{0};

    std::vector<Block> build_blocks; /// concatenated at the build barrier
    std::atomic<bool> post_build_done{false};

    size_t fanout = 0;
    std::vector<size_t> pass_bits;
    std::vector<std::unique_ptr<HashJoin>> partition_joins; /// size fanout, nullptr = empty partition
    size_t post_build_bytes = 0;
    size_t probe_window_budget = 0;

    /// The shared probe window (one wave = one budget's worth of input). window_mutex guards only the
    /// push_back and counters; the wave scatter/probe run outside it. Waves are sequential (the
    /// coordinator holds at most one), and no lane is ever parked behind one: whichever lanes call in
    /// while a wave runs help drain it. Left layout is resolved from the first block.
    std::mutex window_mutex;
    std::vector<Block> window_blocks;
    size_t window_bytes = 0;
    Block left_header;
    SideLayout left_layout;
    bool left_ready = false;

    WaveCoordinator wave;
    std::mutex delayed_mutex;
    bool delayed_flushed = false;

    std::unique_ptr<HashJoin> schema_join;
    /// The dedicated radix pool. Wave liveness depends on an invariant: while a wave has queued or
    /// running workers, no job that can let an exception escape may share this pool. An escaped job
    /// exception shuts the pool down and destroys still-queued jobs unrun, so a destroyed wave worker
    /// would never report its exit and the wave's consumers would wait forever. Wave workers swallow
    /// everything; scatter/build/teardown jobs run only while no wave is active.
    std::unique_ptr<ThreadPool> pool;
    bool enable_lazy_columns_indexing = true;
};

/// -------------------------------------------------------------------------------------------------
/// Probe result and delayed-blocks stream
/// -------------------------------------------------------------------------------------------------

namespace
{

/// Shared references the probe path needs; all point into State, which outlives the results.
struct ProbeShared
{
    ThreadPool & pool;
    ThreadGroupPtr thread_group;
    size_t threads;
    std::vector<std::unique_ptr<HashJoin>> & partition_joins;
    const Block & left_header;
    const SideLayout & left_layout;
    size_t fanout;
    const std::vector<size_t> & pass_bits;
};

/// Drives one leaf partition's probe, forwarding output blocks through `emit` (which returns false to
/// stop early). Handles the leaf's `max_joined_block_rows` splitting via the next_block chain.
template <typename Emit>
void probePartition(HashJoin & leaf, Block probe_block, const Emit & emit)
{
    JoinResultPtr res = leaf.joinBlock(std::move(probe_block));
    while (res)
    {
        auto r = res->next();
        Block out = std::move(r.block);
        if (r.is_last)
        {
            if (r.next_block)
            {
                r.next_block->filterBySelector();
                Block next_block = std::move(*r.next_block).getSourceBlock();
                res = next_block.rows() ? leaf.joinBlock(std::move(next_block)) : nullptr;
            }
            else
            {
                res = nullptr;
            }
        }
        if (out.rows() && !emit(std::move(out)))
            return;
    }
}

/// Probe drain order: the ids of the partitions worth probing (non-empty on both sides), largest
/// probe partition first. One worker probes one partition, so a wave's wall time is lower-bounded
/// by its largest partition; starting the largest first (LPT scheduling) keeps it off the tail
/// under imbalance, and draining the biggest buffers first also releases the most memory earliest
/// (a partition's columns are moved out when probed). Ties break by partition id for determinism.
std::vector<UInt32> probeDrainOrder(const std::vector<PartitionOutput> & parts, const std::vector<std::unique_ptr<HashJoin>> & partition_joins)
{
    std::vector<UInt32> order;
    order.reserve(parts.size());
    for (size_t p = 0; p < parts.size(); ++p)
        if (parts[p].rows && partition_joins[p])
            order.push_back(static_cast<UInt32>(p));
    std::sort(
        order.begin(),
        order.end(),
        [&](UInt32 a, UInt32 b)
        {
            if (parts[a].rows != parts[b].rows)
                return parts[a].rows > parts[b].rows;
            return a < b;
        });
    return order;
}

/// One scattered, worker-fed wave, shared by every probe lane. Workers hold the owning shared_ptr,
/// so the wave survives any consumer teardown order; the coordinator holds it while the wave is the
/// active one.
struct ActiveWave
{
    ActiveWave(ProbeShared shared_, size_t threads)
        : shared(std::move(shared_))
        , output_queue(2 * threads + 1)
    {
    }

    void worker()
    {
        /// Everything, including the thread-group switch, stays inside the try: an exception escaping
        /// a job body would poison the pool (shutdown_on_exception) and rethrow from an arbitrary
        /// later wait.
        try
        {
            ThreadGroupSwitcher switcher(shared.thread_group, ThreadName::RADIX_JOIN);
            for (size_t i = next_partition.fetch_add(1, std::memory_order_relaxed); i < drain_order.size();
                 i = next_partition.fetch_add(1, std::memory_order_relaxed))
            {
                const size_t p = drain_order[i];
                bool stop = false;
                probePartition(
                    *shared.partition_joins[p],
                    parts[p].toBlock(shared.left_header),
                    [&](Block out)
                    {
                        if (!output_queue.push(std::move(out)))
                        {
                            stop = true;
                            return false;
                        }
                        return true;
                    });
                if (stop)
                    break;
            }
        }
        catch (...)
        {
            std::lock_guard lock(exception_mutex);
            if (!exception)
                exception = std::current_exception();
        }
        onWorkerExit(1);
    }

    /// The last worker to exit finishes the queue, so consumers see finished-and-empty exactly when
    /// all output is delivered. Never called with the mutex held.
    void onWorkerExit(size_t count)
    {
        bool last = false;
        {
            std::lock_guard lock(workers_mutex);
            active_workers -= count;
            last = (active_workers == 0);
        }
        if (last)
        {
            output_queue.finish();
            workers_cv.notify_all();
        }
    }

    /// Waits for this wave's workers only — not pool-wide, so it cannot entangle with another wave's
    /// scatter or with the leaf teardown. Bounded: once the queue is finished (or cleared), workers
    /// exit on their next push.
    void joinWorkers()
    {
        std::unique_lock lock(workers_mutex);
        workers_cv.wait(lock, [this] { return active_workers == 0; });
    }

    ProbeShared shared;
    std::vector<PartitionOutput> parts;
    std::vector<UInt32> drain_order;
    ConcurrentBoundedQueue<Block> output_queue;
    std::atomic<size_t> next_partition{0};
    std::mutex workers_mutex;
    std::condition_variable workers_cv;
    size_t active_workers = 0; /// under workers_mutex
    std::exception_ptr exception; /// set under exception_mutex, read after joinWorkers
    std::mutex exception_mutex;
    std::atomic<bool> torn_down{false};
    std::atomic<size_t> consumers{0}; /// attached consumer lanes, bounded by WaveJoinResult::max_consumers
    Stopwatch watch;
};

/// One probe lane's view of the mid-stream wave machinery. The lane that finds no wave active starts
/// one from the shared window (when it has reached the budget); lanes that arrive meanwhile drain
/// the active wave's queue — cross-lane output emission is fine for this INNER join and is exactly
/// what RadixDelayedBlocks already does across the delayed-worker transforms. This is what fixes the
/// deadlock without a busy wait: the old design parked every other lane on a mutex whose release
/// needed an executor-driven drain, and a spin-instead-of-park variant starved the pool workers of
/// CPU. The file-wide liveness invariant: no executor lane ever waits on progress that only another
/// executor lane can make. Waiting on the queue is safe (producers are dedicated pool workers), and
/// waiting on the coordinator mutex is safe (its holder only scatters on the pool, then publishes).
///
/// The consumer set of one wave is bounded (max_consumers). Consumership is sticky — an attached
/// lane pulls no input until it sees no-wave-and-below-budget — so unbounded attachment converts
/// every probe lane into a queue consumer and starves the scan that must feed the next wave,
/// exposing a full window refill on the critical path once per captured generation (measured at
/// mid thread counts). A handful of consumers already saturates the drain: one pop costs a lane a
/// work quantum, an order of magnitude less than producing the block costs a pool worker, so the
/// surplus lanes serve the join better by pulling input — but only while that input is needed.
/// The governing invariant: a surplus lane scans iff the shared window is below budget. Below
/// budget, a capped-out lane reports end-of-result exactly like the below-budget path (its rows
/// are already in the shared window; a later wave or the delayed flush probes them) and goes back
/// to the input; once the window is staged to budget it attaches past the cap and stops pulling —
/// the same back-pressure as draining-only designs, bounding the window overshoot at one in-flight
/// block per lane. Over-cap consumers dissolve at the wave switch: they fail the attach on the
/// next wave and, with its window freshly swapped below budget, return to the input. No lane ever
/// waits on any of this.
class WaveJoinResult : public IJoinResult
{
public:
    WaveJoinResult(
        ProbeShared shared_,
        WaveCoordinator & coordinator_,
        std::mutex & window_mutex_,
        std::vector<Block> & shared_window_,
        size_t & shared_window_bytes_,
        size_t probe_window_budget_)
        : shared(std::move(shared_))
        , coordinator(coordinator_)
        , window_mutex(window_mutex_)
        , shared_window(shared_window_)
        , shared_window_bytes(shared_window_bytes_)
        , probe_window_budget(probe_window_budget_)
    {
    }

    /// The result owns nothing but its consumer slot: waves belong to the coordinator (and their
    /// workers), and an abandoned active wave is torn down by ~RadixHashJoin. An inert destructor is
    /// load-bearing — results are destroyed in arbitrary order at pipeline teardown, possibly while
    /// another lane's wave still has parked producers — and detach keeps it inert: a plain atomic
    /// decrement, no waiting, no teardown. Releasing the slot on destruction keeps an abandoned
    /// consumer from pinning the cap while the query is still draining other lanes.
    ~WaveJoinResult() override
    {
        detach();
    }

    JoinResultBlock next() override
    {
        while (true)
        {
            std::shared_ptr<ActiveWave> wave;
            {
                std::lock_guard lock(coordinator.mutex);
                if (!coordinator.active_wave)
                {
                    std::vector<Block> window;
                    {
                        std::lock_guard window_lock(window_mutex);
                        if (shared_window_bytes >= probe_window_budget)
                        {
                            window.swap(shared_window);
                            shared_window_bytes = 0;
                        }
                    }
                    /// Below budget means another wave already took the rows this result was created
                    /// for; whatever remains keeps accumulating for a later wave or the delayed flush.
                    if (window.empty())
                        return {Block{}, nullptr, true};
                    coordinator.active_wave = startWave(std::move(window));
                }
                wave = coordinator.active_wave;
            }

            if (attached_wave != wave)
            {
                detach();
                /// Slots only read full with max_consumers lanes attached, and an attached lane keeps
                /// draining until the wave is torn down, so a live wave always has a consumer.
                if (!tryAttach(*wave))
                {
                    /// A surplus lane scans while the next window is below budget and stops pulling
                    /// input once it is staged. Going back to the input here with a full window would
                    /// grow it by scan-rate x wave-duration — unbounded; instead attach past the cap,
                    /// which is exactly the pre-cap back-pressure (the window overshoots the budget by
                    /// at most the blocks already in flight, one per lane).
                    bool window_staged;
                    {
                        std::lock_guard window_lock(window_mutex);
                        window_staged = shared_window_bytes >= probe_window_budget;
                    }
                    if (!window_staged)
                        return {Block{}, nullptr, true};
                    wave->consumers.fetch_add(1, std::memory_order_relaxed);
                }
                attached_wave = wave;
            }

            Block block;
            if (wave->output_queue.pop(block))
                return {std::move(block), nullptr, false};

            /// Finished and empty. One lane tears the wave down; the others yield this quantum and
            /// re-check once the winner has cleared it (bounded by the winner joining already-exiting
            /// workers).
            if (wave->torn_down.exchange(true))
                return {Block{}, nullptr, false};

            wave->joinWorkers();
            {
                std::lock_guard lock(coordinator.mutex);
                if (coordinator.active_wave == wave)
                    coordinator.active_wave.reset();
            }
            if (wave->exception)
                std::rethrow_exception(wave->exception);
            ProfileEvents::increment(ProfileEvents::RadixHashJoinProbeMicroseconds, wave->watch.elapsedMicroseconds());
            /// Loop: start the next wave if a full window accumulated meanwhile, else report end.
        }
    }

private:
    /// Scatter the window and schedule the probe workers. Runs under the coordinator mutex: lanes
    /// arriving meanwhile park there, waiting only on the pool-driven scatter. On failure — scatter
    /// throw, or scheduleOrThrow stopping after k < threads workers — the worker count is corrected
    /// for the never-scheduled remainder, the queue is finished-and-cleared so the k live workers
    /// observe push() == false and exit, and the error propagates with the wave unpublished.
    std::shared_ptr<ActiveWave> startWave(std::vector<Block> window)
    {
        auto wave = std::make_shared<ActiveWave>(shared, shared.threads);
        {
            ProfileEventTimeIncrement<Microseconds> route_watch(ProfileEvents::RadixHashJoinProbePackHashRouteMicroseconds);
            wave->parts = scatterToPartitions(
                shared.pool, shared.threads, shared.thread_group, shared.left_header, window, shared.left_layout, shared.pass_bits);
        }
        wave->drain_order = probeDrainOrder(wave->parts, shared.partition_joins);

        {
            std::lock_guard lock(wave->workers_mutex);
            wave->active_workers = shared.threads;
        }
        size_t scheduled = 0;
        try
        {
            for (; scheduled < shared.threads; ++scheduled)
            {
                shared.pool.scheduleOrThrow(
                    [wave]
                    {
                        wave->worker();
                    });
            }
        }
        catch (...)
        {
            wave->output_queue.clearAndFinish();
            wave->onWorkerExit(shared.threads - scheduled);
            wave->joinWorkers();
            throw;
        }
        return wave;
    }

    bool tryAttach(ActiveWave & wave)
    {
        size_t current = wave.consumers.load(std::memory_order_relaxed);
        while (current < max_consumers)
            if (wave.consumers.compare_exchange_weak(current, current + 1, std::memory_order_relaxed))
                return true;
        return false;
    }

    void detach()
    {
        if (attached_wave)
        {
            attached_wave->consumers.fetch_sub(1, std::memory_order_relaxed);
            attached_wave.reset();
        }
    }

    ProbeShared shared;
    WaveCoordinator & coordinator;
    std::mutex & window_mutex; /// State::window_mutex, guarding the two shared-window fields below
    std::vector<Block> & shared_window;
    size_t & shared_window_bytes;
    const size_t probe_window_budget;
    /// Enough consumers to keep the drain ahead of the producers (a pop is ~10x cheaper than the
    /// probe that produced the block), few enough to leave most lanes pulling input.
    const size_t max_consumers = std::max<size_t>(2, shared.threads / 8);
    std::shared_ptr<ActiveWave> attached_wave; /// the wave this result currently consumes
};

/// The final flush: the leftover probe window is scattered once (on the pool) and then probed by the
/// executor's delayed-worker transforms, which call nextImpl() concurrently. Work-stealing over
/// partitions with the GraceHashJoin open-results pattern so multi-block leaf outputs are not lost.
class RadixDelayedBlocks : public IBlocksStream
{
public:
    RadixDelayedBlocks(ProbeShared shared_, std::vector<PartitionOutput> parts_)
        : shared(std::move(shared_))
        , parts(std::move(parts_))
        , drain_order(probeDrainOrder(parts, shared.partition_joins))
    {
    }

protected:
    Block nextImpl() override
    {
        /// Per-call leaf probe time; the whole radix probe (across all delayed workers) sums here.
        ProfileEventTimeIncrement<Microseconds> probe_watch(ProfileEvents::RadixHashJoinProbeMicroseconds);
        std::shared_lock shared_guard(eof_mutex);

        while (true)
        {
            HashJoin * leaf = nullptr;
            JoinResultPtr res;
            {
                std::lock_guard lock(pending_mutex);
                if (!pending.empty())
                {
                    leaf = pending.front().leaf;
                    res = std::move(pending.front().res);
                    pending.pop_front();
                }
            }

            if (!res)
            {
                const size_t i = next_partition.fetch_add(1, std::memory_order_relaxed);
                if (i >= drain_order.size())
                {
                    /// No new partitions. Make sure no in-flight probe is about to leave pending rows.
                    shared_guard.unlock();
                    bool more = false;
                    {
                        std::unique_lock exclusive(eof_mutex);
                        std::lock_guard lock(pending_mutex);
                        more = !pending.empty();
                    }
                    return more ? nextImpl() : Block{};
                }
                const size_t p = drain_order[i];
                leaf = shared.partition_joins[p].get();
                res = leaf->joinBlock(parts[p].toBlock(shared.left_header));
            }

            auto r = res->next();
            if (!r.is_last)
            {
                std::lock_guard lock(pending_mutex);
                pending.push_back({leaf, std::move(res)});
            }
            else if (r.next_block)
            {
                r.next_block->filterBySelector();
                Block next_block = std::move(*r.next_block).getSourceBlock();
                if (next_block.rows())
                {
                    std::lock_guard lock(pending_mutex);
                    pending.push_back({leaf, leaf->joinBlock(std::move(next_block))});
                }
            }

            if (r.block.rows())
                return std::move(r.block);
        }
    }

private:
    struct Pending
    {
        HashJoin * leaf;
        JoinResultPtr res;
    };

    ProbeShared shared;
    std::vector<PartitionOutput> parts;
    std::vector<UInt32> drain_order;
    std::atomic<size_t> next_partition{0};
    std::mutex pending_mutex;
    std::list<Pending> pending TSA_GUARDED_BY(pending_mutex);
    SharedMutex eof_mutex;
};

}

/// -------------------------------------------------------------------------------------------------
/// RadixHashJoin
/// -------------------------------------------------------------------------------------------------

RadixHashJoin::RadixHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader right_sample_block_,
    size_t max_threads_,
    std::optional<UInt64> rhs_size_estimation_,
    UInt64 max_partitions_per_pass_,
    bool size_tables_by_distinct_estimate_,
    double probe_buffer_fraction_,
    UInt64 probe_buffer_min_bytes_,
    UInt64 probe_buffer_max_bytes_,
    const StatsCollectingParams & stats_collecting_params_)
    : table_join(std::move(table_join_))
    , right_sample_block(right_sample_block_)
    , max_threads(std::max<size_t>(max_threads_, 1))
    , rhs_size_estimation(rhs_size_estimation_)
    , max_partitions_per_pass(max_partitions_per_pass_)
    , size_tables_by_distinct_estimate(size_tables_by_distinct_estimate_)
    , probe_buffer_fraction(probe_buffer_fraction_)
    , probe_buffer_min_bytes(probe_buffer_min_bytes_)
    , probe_buffer_max_bytes(probe_buffer_max_bytes_)
    , stats_collecting_params(stats_collecting_params_)
    , state(std::make_unique<State>())
{
    /// Re-check the planner-gate invariants (the planner should never let a violating shape through).
    if (!table_join->oneDisjunct())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin requires a single join disjunct");
    if (table_join->kind() != JoinKind::Inner || table_join->strictness() != JoinStrictness::All)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin supports only INNER ALL joins");
    if (table_join->isSpecialStorage())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin does not support special storage");

    if (!(probe_buffer_fraction >= 0.0 && probe_buffer_fraction <= 1.0))
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "radix_join_probe_buffer_fraction must be in [0, 1]");
    if (probe_buffer_max_bytes != 0 && probe_buffer_min_bytes > probe_buffer_max_bytes)
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "radix_join_probe_buffer_min_bytes must not exceed radix_join_probe_buffer_max_bytes");

    const auto & clause = table_join->getOnlyClause();
    state->left_key_names = clause.key_names_left;
    state->right_key_names = clause.key_names_right;

    size_t packed_key_width = 0;
    for (const auto & name : state->right_key_names)
    {
        const auto * key_column = right_sample_block->findByName(name);
        if (!key_column)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: right key column {} not found", name);
        const auto & type = key_column->type;
        if (type->isNullable() || type->lowCardinality() || !type->haveMaximumSizeOfValue())
            throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: unsupported key column {}", name);
        packed_key_width += type->getMaximumSizeOfValueInMemory();
    }
    if (!(packed_key_width % 4 == 0 && packed_key_width >= 4 && packed_key_width <= 64))
        throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: packed key width {} out of range", packed_key_width);

    state->build_lanes = std::vector<State::BuildLane>(max_threads);
    state->schema_join = std::make_unique<HashJoin>(
        table_join, right_sample_block, /*any_take_last_row*/ false, /*reserve_num*/ 0, "radix_schema", /*use_two_level_maps*/ false);
    state->schema_join->setEnableLazyColumnsIndexing(state->enable_lazy_columns_indexing);

    (void)rhs_size_estimation;
    (void)size_tables_by_distinct_estimate;
}

RadixHashJoin::~RadixHashJoin()
{
    /// A cancelled query can leave the active wave undrained, its workers parked on the full output
    /// queue. Unpark and join them (per-wave, dropping the output) before anything else needs the
    /// pool. Waves cannot exist without a pool.
    if (state->pool)
    {
        std::shared_ptr<ActiveWave> wave;
        {
            std::lock_guard lock(state->wave.mutex);
            wave.swap(state->wave.active_wave);
        }
        if (wave)
        {
            wave->output_queue.clearAndFinish();
            wave->joinWorkers();
            /// Nobody drained this wave to completion, so a stored probe error has no consumer;
            /// the query is already ending, but leave a trace.
            if (wave->exception)
                tryLogException(wave->exception, getLogger("RadixHashJoin"), "Abandoned wave had failed: ");
        }
    }

    /// Hash-table destruction can be very time-consuming; parallelise it over the pool, matching
    /// ConcurrentHashJoin's teardown.
    if (!state->pool || state->partition_joins.empty())
        return;
    try
    {
        auto thread_group = CurrentThread::getGroup();
        std::atomic<size_t> next{0};
        const size_t n = state->partition_joins.size();
        parallelRun(
            *state->pool,
            max_threads,
            thread_group,
            [&](size_t)
            {
                for (size_t p = next.fetch_add(1, std::memory_order_relaxed); p < n; p = next.fetch_add(1, std::memory_order_relaxed))
                    state->partition_joins[p].reset();
            });
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
        try
        {
            if (state->pool)
                state->pool->wait();
        }
        catch (...) /// wait() rethrows escaped job exceptions; nothing may escape a destructor
        {
            tryLogCurrentException(__PRETTY_FUNCTION__);
        }
    }
}

const TableJoin & RadixHashJoin::getTableJoin() const
{
    return *table_join;
}

bool RadixHashJoin::addBlockToJoin(const Block & block, bool check_limits)
{
    return addBlockToJoin(block, block.rows(), check_limits, 0);
}

bool RadixHashJoin::addBlockToJoin(const Block & block, size_t num_rows, bool check_limits)
{
    return addBlockToJoin(block, num_rows, check_limits, 0);
}

bool RadixHashJoin::addBlockToJoin(const Block & block, size_t num_rows, bool check_limits, size_t build_lane)
{
    if (num_rows == 0)
        return true;

    Block materialized = state->schema_join->materializeColumnsFromRightBlock(block);
    state->build_rows.fetch_add(num_rows, std::memory_order_relaxed);
    state->build_bytes.fetch_add(materialized.allocatedBytes(), std::memory_order_relaxed);

    auto & lane = state->build_lanes[build_lane % state->build_lanes.size()];
    {
        std::lock_guard lock(lane.mutex);
        lane.blocks.push_back(std::move(materialized));
    }

    if (check_limits && table_join->sizeLimits().hasLimits())
        return table_join->sizeLimits().check(
            state->build_rows.load(std::memory_order_relaxed),
            state->build_bytes.load(std::memory_order_relaxed),
            "JOIN",
            ErrorCodes::SET_SIZE_LIMIT_EXCEEDED);
    return true;
}

void RadixHashJoin::onBuildPhaseFinish()
{
    /// The cheap build barrier only: concatenate the per-lane block stores. The heavy scatter runs in
    /// runPostBuildPhase.
    size_t total = 0;
    for (auto & lane : state->build_lanes)
        total += lane.blocks.size();
    state->build_blocks.reserve(total);
    for (auto & lane : state->build_lanes)
    {
        for (auto & block : lane.blocks)
            state->build_blocks.push_back(std::move(block));
        lane.blocks.clear();
        lane.blocks.shrink_to_fit();
    }
}

void RadixHashJoin::runPostBuildPhase()
{
    Stopwatch build_watch;

    const size_t build_rows = state->build_rows.load(std::memory_order_relaxed);
    const size_t build_bytes = state->build_bytes.load(std::memory_order_relaxed);

    state->pool = std::make_unique<ThreadPool>(
        CurrentMetrics::RadixHashJoinPoolThreads,
        CurrentMetrics::RadixHashJoinPoolThreadsActive,
        CurrentMetrics::RadixHashJoinPoolThreadsScheduled,
        /*max_threads*/ max_threads,
        /*max_free_threads*/ max_threads,
        /*queue_size*/ 0);

    if (build_rows == 0 || state->build_blocks.empty())
    {
        state->post_build_done.store(true, std::memory_order_release);
        return;
    }

    /// 5.1 Partition plan. The leaf working set is the reserved hash table plus the stored build rows;
    /// pick the smallest power-of-two fanout that keeps it within an L2-sized budget (the benchmark
    /// bandwidth model's "HT + build within L2" criterion). build_rows is exact here.
    auto ht_bytes = [](size_t n) { return std::bit_ceil(std::max<size_t>(2 * n, 1)) * HT_CELL_BYTES; };
    auto leaf_bytes = [&](size_t p) { return ht_bytes(build_rows / p) + build_bytes / p; };

    constexpr size_t max_route_partitions = UInt64{1} << 32;
    const size_t lower = std::min<size_t>(std::max<size_t>(2, std::bit_ceil(max_threads)), max_route_partitions);
    size_t fanout = lower;
    while (fanout < max_route_partitions && leaf_bytes(fanout) > LEAF_TARGET_BYTES)
        fanout <<= 1;

    const size_t max_pass_fanout = std::min<size_t>(MAX_FANOUT_PER_PASS, std::bit_floor(std::max<size_t>(2, max_partitions_per_pass)));
    state->fanout = fanout;
    state->pass_bits = computePassBits(fanout, max_pass_fanout);

    /// Probe-buffer budget from the settings knobs, computed once against the built size below.
    Block build_header = state->build_blocks.front().cloneEmpty();
    SideLayout build_layout = makeSideLayout(build_header, state->right_key_names);

    auto thread_group = CurrentThread::getGroup();
    std::vector<PartitionOutput> parts
        = scatterToPartitions(*state->pool, max_threads, thread_group, build_header, state->build_blocks, build_layout, state->pass_bits);

    /// Release the (now-empty) build block shells.
    state->build_blocks.clear();
    state->build_blocks.shrink_to_fit();

    /// 5.5 Leaf builds — one exactly-reserved HashJoin per non-empty partition, built work-stealing.
    state->partition_joins.resize(fanout);
    std::atomic<size_t> next_partition{0};
    std::atomic<size_t> leaves_built{0};
    Stopwatch leaf_watch;
    parallelRun(
        *state->pool,
        max_threads,
        thread_group,
        [&](size_t)
        {
            size_t local_leaves = 0;
            for (size_t p = next_partition.fetch_add(1, std::memory_order_relaxed); p < fanout;
                 p = next_partition.fetch_add(1, std::memory_order_relaxed))
            {
                if (!parts[p].rows)
                    continue;
                auto join = std::make_unique<HashJoin>(
                    table_join,
                    right_sample_block,
                    /*any_take_last_row*/ false,
                    /*reserve_num*/ parts[p].rows,
                    fmt::format("radix{}", p),
                    /*use_two_level_maps*/ false);
                join->setMaxJoinedBlockRows(table_join->maxJoinedBlockRows());
                join->setMaxJoinedBlockBytes(table_join->maxJoinedBlockBytes());
                join->setEnableLazyColumnsIndexing(state->enable_lazy_columns_indexing);
                join->addBlockToJoin(parts[p].toBlock(build_header), /*check_limits*/ false);
                join->onBuildPhaseFinish();
                state->partition_joins[p] = std::move(join);
                ++local_leaves;
            }
            leaves_built.fetch_add(local_leaves, std::memory_order_relaxed);
        });
    ProfileEvents::increment(ProfileEvents::RadixHashJoinLeafGroupBuilds, leaves_built.load(std::memory_order_relaxed));
    ProfileEvents::increment(ProfileEvents::RadixHashJoinLeafGroupBuildMicroseconds, leaf_watch.elapsedMicroseconds());

    size_t post_build_bytes = 0;
    for (const auto & join : state->partition_joins)
        if (join)
            post_build_bytes += join->getTotalByteCount();
    state->post_build_bytes = post_build_bytes;

    double budget = probe_buffer_fraction * static_cast<double>(post_build_bytes);
    size_t window_budget = static_cast<size_t>(budget);
    window_budget = std::max(window_budget, static_cast<size_t>(probe_buffer_min_bytes));
    if (probe_buffer_max_bytes != 0)
        window_budget = std::min(window_budget, static_cast<size_t>(probe_buffer_max_bytes));
    state->probe_window_budget = std::max<size_t>(window_budget, 1);

    state->post_build_done.store(true, std::memory_order_release);

    ProfileEvents::increment(ProfileEvents::RadixHashJoinBuildMicroseconds, build_watch.elapsedMicroseconds());
    LOG_DEBUG(
        getLogger("RadixHashJoin"),
        "Built {} leaf partitions in {} radix pass(es) from {} rows ({}), probe window budget {}, in {} ms",
        fanout,
        state->pass_bits.size(),
        build_rows,
        ReadableSize(post_build_bytes),
        ReadableSize(state->probe_window_budget),
        build_watch.elapsedMilliseconds());
}

void RadixHashJoin::checkTypesOfKeys(const Block & block) const
{
    state->schema_join->checkTypesOfKeys(block);
}

void RadixHashJoin::setTotals(const Block & block)
{
    std::lock_guard lock(totals_mutex);
    IJoin::setTotals(block);
}

JoinResultPtr RadixHashJoin::joinBlock(Block block)
{
    return joinBlock(std::move(block), 0);
}

JoinResultPtr RadixHashJoin::joinBlock(Block block, size_t /*lane*/)
{
    /// Header/planning path (before the build barrier): delegate to the schema-only HashJoin, which
    /// produces the correct output header.
    if (!state->post_build_done.load(std::memory_order_acquire))
        return state->schema_join->joinBlock(std::move(block));

    if (block.rows() == 0 || state->build_rows.load(std::memory_order_relaxed) == 0)
        return state->schema_join->joinBlock(std::move(block));

    /// materializeColumnsFromLeftBlock is a no-op for INNER joins, but the scatter reads getRawData(),
    /// so normalize any Const/Sparse/LowCardinality wrappers to full fixed-width columns.
    {
        Columns columns = block.getColumns();
        bool changed = false;
        for (auto & column : columns)
        {
            auto full = column->convertToFullIfNeeded();
            if (full.get() != column.get())
            {
                column = std::move(full);
                changed = true;
            }
        }
        if (changed)
            block.setColumns(columns);
    }

    {
        std::lock_guard lock(state->window_mutex);
        if (!state->left_ready)
        {
            state->left_header = block.cloneEmpty();
            state->left_layout = makeSideLayout(state->left_header, state->left_key_names);
            state->left_ready = true;
        }
        const size_t appended_bytes = block.allocatedBytes();
        state->window_blocks.push_back(std::move(block));
        state->window_bytes += appended_bytes;
        if (state->window_bytes < state->probe_window_budget)
            return IJoinResult::createFromBlock(Block{});
    }

    ProbeShared shared{
        *state->pool,
        CurrentThread::getGroup(),
        max_threads,
        state->partition_joins,
        state->left_header,
        state->left_layout,
        state->fanout,
        state->pass_bits};

    /// The window is not swapped here: the result stays weightless until its next() either starts a
    /// wave from the shared window or joins in draining the wave someone else started. Meanwhile this
    /// lane keeps the result and pulls no further input, which bounds the shared window's overshoot.
    return std::make_unique<WaveJoinResult>(
        std::move(shared),
        state->wave,
        state->window_mutex,
        state->window_blocks,
        state->window_bytes,
        state->probe_window_budget);
}

IBlocksStreamPtr RadixHashJoin::getDelayedBlocks()
{
    std::lock_guard lock(state->delayed_mutex);
    if (state->delayed_flushed)
        return {};
    state->delayed_flushed = true;

    std::vector<Block> window;
    {
        std::lock_guard wlock(state->window_mutex);
        window.swap(state->window_blocks);
        state->window_bytes = 0;
    }
    if (window.empty() || !state->left_ready || state->build_rows.load(std::memory_order_relaxed) == 0)
        return {};

    /// A transform cannot finish while it still holds an undrained wave result, and the delayed flush
    /// starts only after every transform finished, so no wave is active here. Scattering on the shared
    /// pool while a wave still held workers parked on its full queue would wait forever; catch any
    /// future wiring change that makes this reachable.
    {
        std::lock_guard wave_lock(state->wave.mutex);
        chassert(!state->wave.active_wave);
    }

    /// The scatter is a sub-phase of the overall radix probe time.
    ProfileEventTimeIncrement<Microseconds> probe_watch(ProfileEvents::RadixHashJoinProbeMicroseconds);
    auto thread_group = CurrentThread::getGroup();
    std::vector<PartitionOutput> parts;
    {
        ProfileEventTimeIncrement<Microseconds> route_watch(ProfileEvents::RadixHashJoinProbePackHashRouteMicroseconds);
        parts = scatterToPartitions(
            *state->pool, max_threads, thread_group, state->left_header, window, state->left_layout, state->pass_bits);
    }

    ProbeShared shared{
        *state->pool,
        std::move(thread_group),
        max_threads,
        state->partition_joins,
        state->left_header,
        state->left_layout,
        state->fanout,
        state->pass_bits};

    return std::make_shared<RadixDelayedBlocks>(std::move(shared), std::move(parts));
}

size_t RadixHashJoin::getTotalRowCount() const
{
    return state->build_rows.load(std::memory_order_relaxed);
}

size_t RadixHashJoin::getTotalByteCount() const
{
    if (state->post_build_done.load(std::memory_order_acquire))
        return state->post_build_bytes;
    return state->build_bytes.load(std::memory_order_relaxed);
}

bool RadixHashJoin::alwaysReturnsEmptySet() const
{
    return state->post_build_done.load(std::memory_order_acquire) && state->build_rows.load(std::memory_order_relaxed) == 0;
}

void RadixHashJoin::setEnableLazyColumnsIndexing(bool value)
{
    state->enable_lazy_columns_indexing = value;
    if (state->schema_join)
        state->schema_join->setEnableLazyColumnsIndexing(value);
    for (auto & join : state->partition_joins)
        if (join)
            join->setEnableLazyColumnsIndexing(value);
}

IBlocksStreamPtr RadixHashJoin::getNonJoinedBlocks(
    const Block & /*left_sample_block*/, const Block & /*result_sample_block*/, UInt64 /*max_block_size*/) const
{
    /// Inner join only: no non-joined right rows.
    return {};
}

}
