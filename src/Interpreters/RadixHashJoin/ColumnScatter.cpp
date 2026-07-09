#include <Interpreters/RadixHashJoin/ColumnScatter.h>

#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnsNumber.h>

#include <Common/Exception.h>
#include <Common/PODArray.h>
#include <Common/ThreadPool.h>
#include <Common/assert_cast.h>
#include <Common/typeid_cast.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <mutex>

namespace DB
{
namespace ErrorCodes
{
extern const int BAD_ARGUMENTS;
extern const int LOGICAL_ERROR;
extern const int QUERY_WAS_CANCELLED;
}
}

namespace DB::RadixJoin
{

namespace
{

/// Cache-line width: the SWWC staging lines and non-temporal flushes are whole lines of this size.
constexpr size_t SCATTER_LINE_BYTES = 64;

/// Fanout from which the SWWC + non-temporal path wins over plain per-partition cursor stores
/// (the direct path's live output lines no longer stay cache-resident).
constexpr size_t SWWC_MIN_FANOUT = 256;

using NtLine = char __attribute__((vector_size(SCATTER_LINE_BYTES)));

/// At low fanout, consecutive rows commonly hit the same counter and the histogram's
/// load-increment-store chain serializes (measured ~1.9x slower at fanout 2 with 4 lanes vs 1).
/// 4 UInt32 lanes stay within 32 KiB even at the largest interleaved fanout.
constexpr size_t HIST_INTERLEAVE_MAX_FANOUT = 2048;

/// First-pass batch sizing: each worker scatters its chunk stripe in batches of whole chunks,
/// dropping each batch's input right after its last column is scattered. The batch must be large
/// enough that the cost of every (batch, column) boundary - the seed/save cursor sweeps plus, on
/// the SWWC path, up to one partial-line flush and one head-realignment copy per partition -
/// stays a small fraction of the lines written in between: 64 lines per partition per batch
/// bounds the boundary cost at ~1.5% (exactly, at 8-byte elements; proportionally at the other
/// widths). The row floor keeps batches at low fanout (where the SWWC boundary cost is absent)
/// big enough to amortize the boundary sweeps. The target also bounds the batch's transient
/// input (freed at batch end) at a few million rows per worker at the largest per-pass fanout.
constexpr size_t SCATTER_BATCH_MIN_ROWS = 256 << 10;
constexpr size_t SCATTER_BATCH_LINES_PER_PARTITION = 64;

size_t scatterBatchRowsTarget(size_t fanout)
{
    return std::max(SCATTER_BATCH_MIN_ROWS, fanout * SCATTER_BATCH_LINES_PER_PARTITION * (SCATTER_LINE_BYTES / sizeof(UInt64)));
}

/// Element widths with a compile-time scatter kernel. Widths up to 16 use the seeded-fill SWWC
/// kernel (see ScatterScratch); 32 and 64 use the line-straddling one.
bool isSupportedWidth(size_t width)
{
    return width == 1 || width == 2 || width == 4 || width == 8 || width == 16 || width == 32 || width == 64;
}

bool widthUsesSeededFill(size_t width)
{
    return width <= 16;
}

/** Per-worker scatter state: per-partition write cursors, and for the SWWC path one 64-byte
  * staging line per partition plus a fill counter (bytes currently staged for that partition's
  * line).
  *
  * Seeded-fill invariant (element widths <= 16): staged bytes for partition p live at
  * staging_line + [m, fill), where m = (uintptr)cursors[p] & 63. Before the first flush of a
  * seeding session the cursor has not advanced, so m equals the misalignment seeded into `fill`
  * by seed; after the first flush the cursor is line-aligned and m == 0. Output columns are
  * backed by PaddedPODArray allocations (>= 16-byte aligned), so for widths dividing 16 the
  * misalignment is always a multiple of the width and `fill` hits SCATTER_LINE_BYTES exactly.
  * This lets the kernel handle cursor misalignment once per flush (at most once per partition
  * per seeding session) instead of once per row.
  *
  * Straddle invariant (widths 32 and 64, whose misalignment need not be a multiple of the
  * width): while a cursor is line-misaligned nothing is staged (fill == 0); the first elements
  * routed to the partition head-align the cursor with direct stores, after which the cursor only
  * advances by whole flushed lines and stays aligned. `seed` therefore seeds `fill` to the
  * misalignment for the seeded-fill kernels and to 0 for the straddling ones; `drain` handles
  * both (see the f > m rule there).
  */
struct ScatterScratch
{
    size_t fanout = 0;
    bool use_swwc = false;
    PaddedPODArray<char> staging_mem;
    char * staging = nullptr;
    PaddedPODArray<char *> cursors;
    PaddedPODArray<UInt32> fill; /// bytes currently staged for the partition's line

    void init(size_t fanout_, bool use_swwc_)
    {
        fanout = fanout_;
        use_swwc = use_swwc_;
        cursors.resize(fanout);
        if (use_swwc)
        {
            staging_mem.resize(fanout * SCATTER_LINE_BYTES + SCATTER_LINE_BYTES);
            staging = reinterpret_cast<char *>( /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                (reinterpret_cast<uintptr_t>(staging_mem.data()) + SCATTER_LINE_BYTES - 1) & ~static_cast<uintptr_t>(SCATTER_LINE_BYTES - 1));
            fill.resize(fanout);
        }
    }

    void seed(size_t p, char * cursor, bool seed_fill_to_misalignment)
    {
        cursors[p] = cursor;
        if (use_swwc)
            /// nullptr -> 0, harmless: no row ever routes to an empty (never-seeded) partition.
            fill[p] = seed_fill_to_misalignment
                ? static_cast<UInt32>(reinterpret_cast<uintptr_t>(cursor) & (SCATTER_LINE_BYTES - 1))
                : 0;
    }

    /// Flush residual staged bytes of every partition and publish the NT stores.
    void drain()
    {
        if (!use_swwc)
            return;
        for (size_t p = 0; p < fanout; ++p)
        {
            const UInt32 f = fill[p];
            if (!f)
                continue;
            char * cur = cursors[p];
            const UInt32 m = static_cast<UInt32>(reinterpret_cast<uintptr_t>(cur) & (SCATTER_LINE_BYTES - 1));
            /// f == m means no rows were staged since seeding: nothing to flush. f > m covers
            /// the seeded-fill pre-first-flush case (data at [m, f)) and the post-flush /
            /// straddle case (m == 0, data at [0, f)).
            if (f > m)
            {
                memcpy(cur, staging + p * SCATTER_LINE_BYTES + m, f - m);
                cursors[p] = cur + (f - m);
            }
            fill[p] = 0;
        }
        /// NT stores are weakly ordered; make them visible before the outputs are read.
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
};

/// Histograms one chunk's rows into `hist[0..fanout)` and stores each row's partition id: the
/// ids are computed exactly once here, and every column's scatter (and nothing else) routes
/// through them. At low fanout (`lanes` non-null, a caller-owned buffer of size 4 * fanout that
/// persists across calls for the same worker/group), row i increments lane (i & 3) of its bucket
/// instead of the shared counter directly, breaking the dependency chain; the caller must reduce
/// the lanes into `hist` via reduceHistogramLanes once after all chunks are processed. At high
/// fanout collisions are rare and 4 lanes would blow the cache footprint, so `lanes` is null and
/// rows increment `hist` directly.
void histogramChunk(const UInt32 * routes, size_t n, UInt32 shift, UInt32 mask, UInt32 * hist, UInt32 * lanes, size_t fanout, UInt16 * pids)
{
    if (!lanes)
    {
        for (size_t i = 0; i < n; ++i)
        {
            const UInt32 p = (routes[i] >> shift) & mask;
            pids[i] = static_cast<UInt16>(p);
            ++hist[p];
        }
        return;
    }

    size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        const UInt32 p0 = (routes[i + 0] >> shift) & mask;
        const UInt32 p1 = (routes[i + 1] >> shift) & mask;
        const UInt32 p2 = (routes[i + 2] >> shift) & mask;
        const UInt32 p3 = (routes[i + 3] >> shift) & mask;
        pids[i + 0] = static_cast<UInt16>(p0);
        pids[i + 1] = static_cast<UInt16>(p1);
        pids[i + 2] = static_cast<UInt16>(p2);
        pids[i + 3] = static_cast<UInt16>(p3);
        ++lanes[0 * fanout + p0];
        ++lanes[1 * fanout + p1];
        ++lanes[2 * fanout + p2];
        ++lanes[3 * fanout + p3];
    }
    for (; i < n; ++i)
    {
        const UInt32 p = (routes[i] >> shift) & mask;
        pids[i] = static_cast<UInt16>(p);
        ++lanes[(i & 3) * fanout + p];
    }
}

void reduceHistogramLanes(UInt32 * hist, const UInt32 * lanes, size_t fanout)
{
    for (size_t p = 0; p < fanout; ++p)
        hist[p] += lanes[0 * fanout + p] + lanes[1 * fanout + p] + lanes[2 * fanout + p] + lanes[3 * fanout + p];
}

/// DIRECT path: a plain per-partition write cursor; every element copy lowers to inlined typed
/// stores (no memcpy call). Best when the partition count is small enough that all the live
/// output lines stay cache-resident.
template <size_t width>
void scatterChunkDirect(const UInt16 * pids, const char * src, size_t n, char ** cursors)
{
    for (size_t i = 0; i < n; ++i)
    {
        const size_t p = pids[i];
        char * d = cursors[p];
        __builtin_memcpy_inline(d, src + i * width, width);
        cursors[p] = d + width;
    }
}

/// SWWC path for widths <= 16 (which divide both 16 and the line, so a width-aligned cursor's
/// misalignment is a multiple of the width - see the ScatterScratch seeded-fill invariant).
template <size_t width>
void scatterChunkSwwcSeeded(const UInt16 * pids, const char * src, size_t n, ScatterScratch & scratch)
{
    /// Hoisted into locals: the char*/vector NT store defeats TBAA, so without this the
    /// compiler reloads scratch.cursors/fill.data() every row (measured ~1.07x on clang,
    /// ~1.65x on GCC by hoisting).
    char * const staging = scratch.staging;
    char ** const cursors = scratch.cursors.data();
    UInt32 * const fill = scratch.fill.data();

    for (size_t i = 0; i < n; ++i)
    {
        const size_t p = pids[i];
        char * line = staging + p * SCATTER_LINE_BYTES;
        UInt32 f = fill[p];
        __builtin_memcpy_inline(line + f, src + i * width, width);
        f += width;
        if (f == SCATTER_LINE_BYTES)
        {
            char * cur = cursors[p];
            const UInt32 m = static_cast<UInt32>(reinterpret_cast<uintptr_t>(cur) & (SCATTER_LINE_BYTES - 1));
            if (m) /// first flush of a misaligned stream: emit the partial head line with regular stores
            {
                __builtin_memcpy(cur, line + m, SCATTER_LINE_BYTES - m);
                cursors[p] = cur + (SCATTER_LINE_BYTES - m);
            }
            else
            {
                /// A variant reading the line as narrow loads (to dodge a store-to-load-forwarding
                /// stall on the immediately-preceding narrow stores) was measured against this
                /// wide vector load at fanouts 512 and 2048 and did not win at either point, so
                /// the wide load is kept.
                __builtin_nontemporal_store(*reinterpret_cast<const NtLine *>(line), reinterpret_cast<NtLine *>(cur));
                cursors[p] = cur + SCATTER_LINE_BYTES;
            }
            f = 0;
        }
        fill[p] = f;
    }
}

/// SWWC path for widths 32 and 64, whose cursor misalignment need not be a multiple of the
/// width: the first elements routed to a partition head-align its cursor with direct stores
/// (once per seeding session), then elements are staged with line straddles and flushed a whole
/// line at a time.
template <size_t width>
void scatterChunkSwwcStraddle(const UInt16 * pids, const char * src, size_t n, ScatterScratch & scratch)
{
    /// Hoisted for the same TBAA reason as the seeded-fill kernel.
    char * const staging = scratch.staging;
    char ** const cursors = scratch.cursors.data();
    UInt32 * const fill = scratch.fill.data();

    for (size_t i = 0; i < n; ++i)
    {
        const size_t p = pids[i];
        const char * s = src + i * width;
        size_t src_off = 0;
        size_t remaining = width;

        char * cur = cursors[p];
        const UInt32 m = static_cast<UInt32>(reinterpret_cast<uintptr_t>(cur) & (SCATTER_LINE_BYTES - 1));
        if (m)
        {
            /// Misaligned cursor => nothing staged for this partition yet (the straddle
            /// invariant): head-align with direct stores. If the element ends before the line
            /// boundary the cursor stays misaligned and the next element continues the head.
            const size_t head = std::min(remaining, SCATTER_LINE_BYTES - m);
            __builtin_memcpy(cur, s, head);
            cursors[p] = cur + head;
            src_off = head;
            remaining -= head;
            if (remaining == 0)
                continue;
        }

        char * line = staging + p * SCATTER_LINE_BYTES;
        UInt32 f = fill[p];
        while (remaining != 0)
        {
            const size_t copy_n = std::min(remaining, SCATTER_LINE_BYTES - f);
            __builtin_memcpy(line + f, s + src_off, copy_n);
            f += static_cast<UInt32>(copy_n);
            src_off += copy_n;
            remaining -= copy_n;
            if (f == SCATTER_LINE_BYTES)
            {
                char * dst = cursors[p];
                __builtin_nontemporal_store(*reinterpret_cast<const NtLine *>(line), reinterpret_cast<NtLine *>(dst));
                cursors[p] = dst + SCATTER_LINE_BYTES;
                f = 0;
            }
        }
        fill[p] = f;
    }
}

/// Scatters one chunk of one column through the stored partition ids. The width switch is
/// hoisted out of the per-row loop, so the inner loops run the compile-time kernels.
void scatterChunkColumn(const UInt16 * pids, const char * src, size_t n, size_t width, bool use_swwc, ScatterScratch & scratch)
{
    if (use_swwc)
    {
        switch (width)
        {
            case 1: scatterChunkSwwcSeeded<1>(pids, src, n, scratch); return;
            case 2: scatterChunkSwwcSeeded<2>(pids, src, n, scratch); return;
            case 4: scatterChunkSwwcSeeded<4>(pids, src, n, scratch); return;
            case 8: scatterChunkSwwcSeeded<8>(pids, src, n, scratch); return;
            case 16: scatterChunkSwwcSeeded<16>(pids, src, n, scratch); return;
            case 32: scatterChunkSwwcStraddle<32>(pids, src, n, scratch); return;
            case 64: scatterChunkSwwcStraddle<64>(pids, src, n, scratch); return;
            default: break;
        }
    }
    else
    {
        char ** cursors = scratch.cursors.data();
        switch (width)
        {
            case 1: scatterChunkDirect<1>(pids, src, n, cursors); return;
            case 2: scatterChunkDirect<2>(pids, src, n, cursors); return;
            case 4: scatterChunkDirect<4>(pids, src, n, cursors); return;
            case 8: scatterChunkDirect<8>(pids, src, n, cursors); return;
            case 16: scatterChunkDirect<16>(pids, src, n, cursors); return;
            case 32: scatterChunkDirect<32>(pids, src, n, cursors); return;
            case 64: scatterChunkDirect<64>(pids, src, n, cursors); return;
            default: break;
        }
    }
    throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix column scatter: unsupported element width {}", width);
}

const UInt32 * routeData(const RoutedChunk & chunk)
{
    return assert_cast<const ColumnUInt32 &>(*chunk.routes).getData().data();
}

/// Exactly-sized uninitialized destination column: resize_exact reserves the exact element
/// count (no power-of-two capacity growth, so the exact-allocation property and the memory
/// accounting a probe-buffer budget relies on both hold), and the PODArray resize leaves POD
/// contents untouched - no memset, the pages are first-touched by the scatter writes
/// themselves. Returns the raw write base.
template <typename T>
bool tryResizeVector(IColumn & col, size_t rows, char ** base)
{
    auto * vec = typeid_cast<ColumnVector<T> *>(&col);
    if (!vec)
        return false;
    vec->getData().resize_exact(rows);
    *base = reinterpret_cast<char *>(vec->getData().data());
    return true;
}

template <typename T>
bool tryResizeDecimal(IColumn & col, size_t rows, char ** base)
{
    auto * dec = typeid_cast<ColumnDecimal<T> *>(&col);
    if (!dec)
        return false;
    dec->getData().resize_exact(rows);
    *base = reinterpret_cast<char *>(dec->getData().data());
    return true;
}

char * resizeUninitialized(IColumn & col, size_t rows)
{
    if (auto * fixed = typeid_cast<ColumnFixedString *>(&col))
    {
        fixed->getChars().resize_exact(rows * fixed->getN());
        return reinterpret_cast<char *>(fixed->getChars().data());
    }
    char * base = nullptr;
    if (tryResizeVector<UInt8>(col, rows, &base) || tryResizeVector<UInt16>(col, rows, &base)
        || tryResizeVector<UInt32>(col, rows, &base) || tryResizeVector<UInt64>(col, rows, &base)
        || tryResizeVector<UInt128>(col, rows, &base) || tryResizeVector<UInt256>(col, rows, &base)
        || tryResizeVector<Int8>(col, rows, &base) || tryResizeVector<Int16>(col, rows, &base)
        || tryResizeVector<Int32>(col, rows, &base) || tryResizeVector<Int64>(col, rows, &base)
        || tryResizeVector<Int128>(col, rows, &base) || tryResizeVector<Int256>(col, rows, &base)
        || tryResizeVector<Float32>(col, rows, &base) || tryResizeVector<Float64>(col, rows, &base)
        || tryResizeVector<UUID>(col, rows, &base) || tryResizeVector<IPv4>(col, rows, &base)
        || tryResizeVector<IPv6>(col, rows, &base)
        || tryResizeDecimal<Decimal32>(col, rows, &base) || tryResizeDecimal<Decimal64>(col, rows, &base)
        || tryResizeDecimal<Decimal128>(col, rows, &base) || tryResizeDecimal<Decimal256>(col, rows, &base)
        || tryResizeDecimal<DateTime64>(col, rows, &base) || tryResizeDecimal<Time64>(col, rows, &base))
        return base;
    throw Exception(ErrorCodes::BAD_ARGUMENTS, "Radix column scatter: unsupported column type {}", col.getName());
}

/// Exactly-sized destination columns of one output partition, with raw write pointers.
struct PartitionOutput
{
    std::vector<MutableColumnPtr> columns;
    std::vector<char *> bases;
    size_t rows = 0;

    /// Appends one exactly-sized destination column (uninitialized contents, see
    /// resizeUninitialized). Refine passes call this just-in-time, one column per scatter
    /// round, so the allocator can serve it from the input column the previous round just
    /// dropped.
    void allocateColumn(const IColumn & sample)
    {
        MutableColumnPtr col = sample.cloneEmpty();
        bases.push_back(resizeUninitialized(*col, rows));
        columns.push_back(std::move(col));
    }

    void allocate(const Columns & samples, size_t rows_)
    {
        rows = rows_;
        columns.reserve(samples.size());
        bases.reserve(samples.size());
        for (const auto & sample : samples)
            allocateColumn(*sample);
    }

    ScatterChunk toChunk()
    {
        ScatterChunk chunk;
        chunk.rows = rows;
        chunk.columns.reserve(columns.size());
        for (auto & col : columns)
            chunk.columns.emplace_back(std::move(col));
        columns.clear();
        bases.clear();
        return chunk;
    }
};

/// The last scattered column of a non-final pass is the route words; peel it back off into the
/// `routes` slot of the next pass's input chunk.
RoutedChunk toRoutedChunk(PartitionOutput && part)
{
    ScatterChunk chunk = part.toChunk();
    RoutedChunk routed;
    routed.rows = chunk.rows;
    routed.routes = std::move(chunk.columns.back());
    chunk.columns.pop_back();
    routed.columns = std::move(chunk.columns);
    return routed;
}

/// Schema of one scatter input, validated once per entry point: column count, per-column
/// element widths, and empty sample columns for allocating the outputs.
struct ScatterSchema
{
    size_t num_columns = 0;
    std::vector<size_t> widths;
    Columns samples;
    size_t total_rows = 0;
};

ScatterSchema validateChunks(const std::vector<RoutedChunk> & chunks)
{
    ScatterSchema schema;
    if (chunks.empty())
        return schema;

    schema.num_columns = chunks.front().columns.size();
    schema.widths.reserve(schema.num_columns);
    schema.samples.reserve(schema.num_columns);
    for (const auto & col : chunks.front().columns)
    {
        if (!col || !col->isFixedAndContiguous())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Radix column scatter requires fixed-width contiguous columns, got {}",
                col ? col->getName() : "nullptr");
        const size_t width = col->sizeOfValueIfFixed();
        if (!isSupportedWidth(width))
            throw Exception(ErrorCodes::BAD_ARGUMENTS,
                "Radix column scatter: unsupported element width {} of column {} (supported: 1, 2, 4, 8, 16, 32, 64)",
                width, col->getName());
        schema.widths.push_back(width);
        /// Enforce allocability up front, on the caller's thread: every accepted column type
        /// must be one resizeUninitialized can allocate, otherwise a worker would only fail
        /// mid-scatter. The probe is free - an empty clone, zero rows.
        MutableColumnPtr sample = col->cloneEmpty();
        resizeUninitialized(*sample, 0);
        schema.samples.push_back(std::move(sample));
    }

    for (const auto & chunk : chunks)
    {
        if (chunk.columns.size() != schema.num_columns)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix column scatter: inconsistent column count across chunks ({} vs {})",
                chunk.columns.size(), schema.num_columns);
        if (!chunk.routes || !typeid_cast<const ColumnUInt32 *>(chunk.routes.get()))
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix column scatter: every chunk needs a ColumnUInt32 of route words");
        if (chunk.routes->size() != chunk.rows)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix column scatter: route word count {} does not match chunk rows {}",
                chunk.routes->size(), chunk.rows);
        for (size_t j = 0; j < schema.num_columns; ++j)
        {
            const auto & col = chunk.columns[j];
            if (!col || col->size() != chunk.rows)
                throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix column scatter: column {} size does not match chunk rows {}",
                    j, chunk.rows);
            if (!col->isFixedAndContiguous() || col->sizeOfValueIfFixed() != schema.widths[j])
                throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix column scatter: inconsistent column schema across chunks (column {})", j);
        }
        schema.total_rows += chunk.rows;
    }

    /// Histogram/offset counters are UInt32: a window with more rows would silently overflow
    /// them, so this is a hard precondition, not a soft cap.
    if (schema.total_rows > std::numeric_limits<UInt32>::max())
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Radix column scatter supports at most 2^32 - 1 rows per scattered window, got {}",
            schema.total_rows);

    return schema;
}

void validateThreads(const ThreadPool & pool, size_t num_threads)
{
    if (num_threads == 0)
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Radix column scatter needs at least one worker");
    /// All workers of one dispatch must run concurrently (they rendezvous with each other), so
    /// a pool that cannot start them all would deadlock. Fail close instead.
    if (num_threads > pool.getMaxThreads())
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Radix column scatter needs {} concurrent workers, but the pool only runs {}",
            num_threads, pool.getMaxThreads());
}

void checkCancelled(const std::atomic<bool> & cancelled)
{
    if (cancelled.load(std::memory_order_relaxed))
        throw Exception(ErrorCodes::QUERY_WAS_CANCELLED, "Radix column scatter was cancelled");
}

bool decideSwwc(ScatterPath path, size_t fanout)
{
    if (path == ScatterPath::Automatic)
        return fanout >= SWWC_MIN_FANOUT;
    return path == ScatterPath::Swwc;
}

/// Runs task(tid) for every tid in [0, num_threads) on the pool and joins. Used for the
/// per-phase dispatches of scatterColumns, where the workers never wait for each other inside a
/// phase, so a throwing worker cannot strand the others: the pool captures the first exception
/// and `wait` rethrows it here. If scheduling itself fails partway, the already-scheduled jobs
/// are joined first (they reference this frame) and the scheduling error is rethrown - never a
/// silently smaller worker set over the same data.
void runParallelPhase(ThreadPool & pool, size_t num_threads, const std::function<void(size_t)> & task)
{
    std::exception_ptr schedule_error;
    for (size_t t = 0; t < num_threads; ++t)
    {
        try
        {
            pool.scheduleOrThrowOnError([&task, t] { task(t); });
        }
        catch (...)
        {
            schedule_error = std::current_exception();
            break;
        }
    }
    pool.wait();
    if (schedule_error)
        std::rethrow_exception(schedule_error);
}

/** Shared state of one barrier team (the single-dispatch wave loop): all phase bodies run
  * through `phase`, which guarantees that a throwing worker STILL arrives at the phase barrier.
  * The first exception is captured, the stop flag is raised, and every later phase body becomes
  * a no-op, so all workers keep executing the SAME sequence of barrier arrivals and nobody is
  * ever left waiting.
  *
  * Loop-exit decisions must be based on `stopping`, never on the raw flag: the barrier's
  * completion step (which runs serially, while every worker is still blocked on the barrier)
  * latches the flag into `stop_snapshot`, so all workers observe the SAME value at the same
  * phase boundary and leave their loops together - this is what keeps the per-worker barrier
  * arrival counts equal. Reading the raw flag after the barrier instead would race with a fast
  * worker that already entered the next phase body and failed there.
  */
class PhaseTeam
{
public:
    PhaseTeam(size_t num_threads_, std::atomic<bool> & cancelled_)
        : cancelled(cancelled_)
        , barrier(static_cast<std::ptrdiff_t>(num_threads_), Completion{this})
    {
    }

    /// Runs one phase body (skipped once the team is stopping) and rendezvouses with the other
    /// workers. Cancellation is checked on entry, so every phase boundary is a cancellation
    /// point.
    template <typename F>
    void phase(F && body)
    {
        if (!stop_snapshot)
        {
            try
            {
                throwIfCancelled();
                body();
            }
            catch (...)
            {
                noteError(std::current_exception());
            }
        }
        barrier.arrive_and_wait();
    }

    /// Valid right after a phase returns; identical across all workers of that phase boundary.
    bool stopping() const { return stop_snapshot; }

    /// The raw stop flag, for best-effort work shedding INSIDE a phase body (e.g. skipping the
    /// rest of a work-stealing loop). Unlike `stopping`, concurrent workers may read different
    /// values here - never base a barrier-count decision on it.
    bool stopRequested() const { return stop.load(std::memory_order_relaxed); }

    void throwIfCancelled() const { checkCancelled(cancelled); }

    void noteError(std::exception_ptr e)
    {
        {
            std::lock_guard lock(mutex);
            if (!first_error)
                first_error = e;
        }
        stop.store(true, std::memory_order_relaxed);
    }

    void rethrowIfAny()
    {
        std::lock_guard lock(mutex);
        if (first_error)
            std::rethrow_exception(first_error);
    }

private:
    struct Completion
    {
        PhaseTeam * team;
        void operator()() const noexcept { team->stop_snapshot = team->stop.load(std::memory_order_relaxed); }
    };

    std::atomic<bool> & cancelled;
    std::atomic<bool> stop{false};
    /// Written only inside the barrier completion step (serial), read by workers after release.
    bool stop_snapshot = false;
    std::mutex mutex;
    std::exception_ptr first_error;
    std::barrier<Completion> barrier;
};

/// Launches `num_threads` team workers with ONE dispatch on the caller's pool and joins them.
/// Workers are held at a start gate until every one of them has been scheduled: if scheduling
/// fails partway, the gate is released in abort mode and the already-scheduled workers return
/// without ever touching the team barrier, so a partially-started team cannot deadlock; the
/// scheduling error is then rethrown (fail close, never a silently smaller team). The body must
/// only ever throw from inside a PhaseTeam phase - everything fallible belongs in a phase.
void runTeamWorkers(ThreadPool & pool, size_t num_threads, const std::function<void(size_t)> & body)
{
    std::promise<void> gate;
    std::shared_future<void> started = gate.get_future().share();
    std::atomic<bool> abort_workers{false};

    std::exception_ptr schedule_error;
    for (size_t t = 0; t < num_threads; ++t)
    {
        try
        {
            pool.scheduleOrThrowOnError([&body, &abort_workers, started, t]
            {
                started.wait();
                if (abort_workers.load())
                    return;
                body(t);
            });
        }
        catch (...)
        {
            schedule_error = std::current_exception();
            break;
        }
    }

    if (schedule_error)
        abort_workers.store(true);
    gate.set_value();
    pool.wait();
    if (schedule_error)
        std::rethrow_exception(schedule_error);
}

void validatePassBits(const std::vector<size_t> & pass_bits)
{
    if (pass_bits.empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix column scatter: empty pass list");
    size_t total = 0;
    for (size_t bits : pass_bits)
    {
        if (bits < 1 || bits > 16) /// partition ids are UInt16
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix column scatter: per-pass bits must be in [1, 16], got {}", bits);
        total += bits;
    }
    if (total > 32) /// all passes slice one 32-bit route word
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix column scatter: passes consume {} route bits, only 32 exist", total);
}

}

std::vector<size_t> computePassBits(size_t p_star, size_t f_max)
{
    /// MAX_FANOUT_PER_PASS is enforced here: a planned pass never exceeds the SWWC cache-fit
    /// ceiling, whatever the caller's f_max claims.
    f_max = std::min(f_max, MAX_FANOUT_PER_PASS);
    const size_t total_bits = static_cast<size_t>(std::countr_zero(std::bit_ceil(p_star)));
    const size_t f_bits = std::max<size_t>(1, static_cast<size_t>(std::bit_width(std::bit_floor(std::max<size_t>(2, f_max))) - 1));
    const size_t n_pass = (total_bits + f_bits - 1) / f_bits;
    const size_t per_pass = n_pass ? (total_bits + n_pass - 1) / n_pass : 0;

    std::vector<size_t> result;
    size_t remaining = total_bits;
    while (remaining > 0)
    {
        const size_t bits = std::min(per_pass, remaining);
        result.push_back(bits);
        remaining -= bits;
    }
    return result;
}

std::vector<ScatterChunk> scatterColumns(
    ThreadPool & pool,
    size_t num_threads,
    std::vector<RoutedChunk> chunks,
    const std::vector<size_t> & pass_bits,
    std::atomic<bool> & cancelled,
    ScatterPath path)
{
    validateThreads(pool, num_threads);
    validatePassBits(pass_bits);
    const ScatterSchema schema = validateChunks(chunks);

    size_t total_bits = 0;
    for (size_t bits : pass_bits)
        total_bits += bits;
    const size_t total_fanout = size_t{1} << total_bits;

    if (chunks.empty())
        return std::vector<ScatterChunk>(total_fanout);

    const size_t threads = num_threads;
    const size_t num_columns = schema.num_columns;

    /// The columns one pass scatters: the payload columns, plus - on every pass but the last -
    /// the route words as one more 4-byte column, so the next pass can slice its own bit range
    /// of them. The final pass consumes the route words entirely in its histogram (the stored
    /// partition ids replace them) and drops them there.
    auto pass_samples = [&](bool final_pass)
    {
        Columns samples = schema.samples;
        if (!final_pass)
            samples.push_back(ColumnUInt32::create());
        return samples;
    };
    auto pass_widths = [&](bool final_pass)
    {
        std::vector<size_t> widths = schema.widths;
        if (!final_pass)
            widths.push_back(sizeof(UInt32));
        return widths;
    };

    /// ---- First pass: all workers cooperate on the whole side in exactly 3 phases (a fused
    /// prefix-sum/allocation phase removes any single-threaded prefix-sum step, and a fused
    /// all-columns scatter phase removes any per-column dispatch).
    const size_t bits0 = pass_bits.front();
    const bool first_is_final = pass_bits.size() == 1;
    const size_t fanout0 = size_t{1} << bits0;
    const UInt32 shift0 = static_cast<UInt32>(32 - bits0);
    const UInt32 mask0 = static_cast<UInt32>(fanout0 - 1);
    const bool use_swwc0 = decideSwwc(path, fanout0);
    const bool interleave0 = fanout0 <= HIST_INTERLEAVE_MAX_FANOUT;
    const Columns samples0 = pass_samples(first_is_final);
    const std::vector<size_t> widths0 = pass_widths(first_is_final);
    const size_t scatter_columns0 = widths0.size();

    /// Phase 1: per-worker histograms into disjoint slices of one flat array, storing each
    /// row's 2-byte partition id as a by-product (the ids end all routing uses of the route
    /// words, so the final pass drops them right here).
    PaddedPODArray<UInt32> hist;
    hist.resize(threads * fanout0);
    std::vector<PaddedPODArray<UInt16>> worker_pids(threads);
    runParallelPhase(pool, threads, [&](size_t tid)
    {
        checkCancelled(cancelled);
        UInt32 * h = hist.data() + tid * fanout0;
        memset(h, 0, fanout0 * sizeof(UInt32));
        PaddedPODArray<UInt32> lanes;
        if (interleave0)
            lanes.resize_fill(4 * fanout0, 0);

        size_t stripe_rows = 0;
        for (size_t c = tid; c < chunks.size(); c += threads)
            stripe_rows += chunks[c].rows;
        auto & pids = worker_pids[tid];
        pids.resize(stripe_rows);

        size_t row = 0;
        for (size_t c = tid; c < chunks.size(); c += threads)
        {
            histogramChunk(routeData(chunks[c]), chunks[c].rows, shift0, mask0, h, interleave0 ? lanes.data() : nullptr, fanout0, pids.data() + row);
            row += chunks[c].rows;
            if (first_is_final)
                chunks[c].routes = nullptr;
        }
        if (interleave0)
            reduceHistogramLanes(h, lanes.data(), fanout0);
    });
    checkCancelled(cancelled);

    /// Phase 2: fused prefix sum + exact one-shot allocation. Each worker owns a contiguous,
    /// disjoint range of partitions, so there is no cross-worker write dependency and no
    /// separate single-threaded prefix-sum phase is needed.
    PaddedPODArray<UInt32> offsets;
    offsets.resize(threads * fanout0);
    PaddedPODArray<UInt64> totals;
    totals.resize_fill(fanout0, 0);
    std::vector<PartitionOutput> parts(fanout0);
    runParallelPhase(pool, threads, [&](size_t tid)
    {
        checkCancelled(cancelled);
        const size_t begin = fanout0 * tid / threads;
        const size_t end = fanout0 * (tid + 1) / threads;
        for (size_t p = begin; p < end; ++p)
        {
            UInt64 total = 0;
            for (size_t w = 0; w < threads; ++w)
            {
                offsets[w * fanout0 + p] = static_cast<UInt32>(total);
                total += hist[w * fanout0 + p];
            }
            totals[p] = total;
            if (total)
                parts[p].allocate(samples0, total);
        }
    });
    checkCancelled(cancelled);

    /// Phase 3: single fused scatter run, batched. Each worker processes its chunk stripe in
    /// batches of whole chunks (~scatterBatchRowsTarget rows), scattering every column of the
    /// batch through the stored ids, then drops the batch's input - each chunk belongs to
    /// exactly one worker's stripe, so the drop is worker-local and releases this side's
    /// reference to the caller's blocks (in a real pipeline the upstream source's blocks are
    /// recycled here). Each worker writes only its own [offset, offset + hist) range of every
    /// (partition, column) output buffer; those ranges are disjoint across workers and across
    /// columns, so there is no cross-worker dependency and no rendezvous between columns or
    /// batches - the phase join plus each worker's drain fences (which publish the NT stores)
    /// are enough to make every worker's writes visible before the outputs are read.
    const size_t batch_rows_target = scatterBatchRowsTarget(fanout0);
    runParallelPhase(pool, threads, [&](size_t tid)
    {
        checkCancelled(cancelled);
        ScatterScratch scratch;
        scratch.init(fanout0, use_swwc0);

        /// Running write cursors per (column, partition), persisted across batches: this
        /// worker's disjoint output ranges, advanced batch by batch. The ScatterScratch
        /// invariants handle the mid-line cursor a drain leaves behind (the next batch's first
        /// flush, or head-alignment, repairs the misaligned head).
        std::vector<char *> col_cursors(scatter_columns0 * fanout0);
        for (size_t j = 0; j < scatter_columns0; ++j)
            for (size_t p = 0; p < fanout0; ++p)
                col_cursors[j * fanout0 + p] = totals[p] ? parts[p].bases[j] + size_t{offsets[tid * fanout0 + p]} * widths0[j] : nullptr;

        const UInt16 * pids = worker_pids[tid].data();
        std::vector<size_t> batch;         /// chunk indices of the current batch
        std::vector<size_t> batch_offsets; /// each chunk's start row within this worker's ids

        size_t stripe_row = 0;
        size_t c = tid;
        while (c < chunks.size())
        {
            checkCancelled(cancelled);
            batch.clear();
            batch_offsets.clear();
            size_t batch_rows = 0;
            for (; c < chunks.size() && batch_rows < batch_rows_target; c += threads)
            {
                batch.push_back(c);
                batch_offsets.push_back(stripe_row + batch_rows);
                batch_rows += chunks[c].rows;
            }

            for (size_t j = 0; j < scatter_columns0; ++j)
            {
                const size_t width = widths0[j];
                const bool seeded = widthUsesSeededFill(width);
                for (size_t p = 0; p < fanout0; ++p)
                    scratch.seed(p, col_cursors[j * fanout0 + p], seeded);

                for (size_t b = 0; b < batch.size(); ++b)
                {
                    const RoutedChunk & chunk = chunks[batch[b]];
                    const char * data = j < num_columns
                        ? chunk.columns[j]->getRawData().data()
                        : chunk.routes->getRawData().data();
                    scatterChunkColumn(pids + batch_offsets[b], data, chunk.rows, width, use_swwc0, scratch);
                }
                scratch.drain();

                for (size_t p = 0; p < fanout0; ++p)
                    col_cursors[j * fanout0 + p] = scratch.cursors[p];
            }

            /// The batch is fully consumed (the ids replaced all routing uses of the route
            /// words): drop its input chunks before starting the next batch.
            for (size_t idx : batch)
            {
                chunks[idx].columns.clear();
                chunks[idx].routes = nullptr;
            }
            stripe_row += batch_rows;
        }
    });
    chunks.clear();
    worker_pids.clear();

    if (first_is_final)
    {
        std::vector<ScatterChunk> out(fanout0);
        for (size_t p = 0; p < fanout0; ++p)
            if (totals[p])
                out[p] = parts[p].toChunk();
        return out;
    }

    /// ---- Refine passes (multi-pass fallback): each group (one previous-pass partition) is at
    /// most 1/fanout_so_far of the side and is processed entirely worker-locally; groups are
    /// assigned to workers dynamically (an atomic counter, not a static stripe), because groups
    /// can have very different sizes and a static stripe would leave some workers idle while
    /// others are still scattering their share - the defense against per-group skew.
    std::vector<RoutedChunk> level(fanout0);
    for (size_t p = 0; p < fanout0; ++p)
        if (totals[p])
            level[p] = toRoutedChunk(std::move(parts[p]));
    parts.clear();

    size_t bits_done = bits0;
    for (size_t pass = 1; pass < pass_bits.size(); ++pass)
    {
        checkCancelled(cancelled);
        const size_t bits = pass_bits[pass];
        const bool final_pass = pass + 1 == pass_bits.size();
        const size_t fanout = size_t{1} << bits;
        const UInt32 shift = static_cast<UInt32>(32 - bits_done - bits);
        const UInt32 mask = static_cast<UInt32>(fanout - 1);
        const bool use_swwc = decideSwwc(path, fanout);
        const bool interleave = fanout <= HIST_INTERLEAVE_MAX_FANOUT;
        const Columns samples = pass_samples(final_pass);
        const std::vector<size_t> widths = pass_widths(final_pass);
        const size_t scatter_columns = widths.size();

        std::vector<RoutedChunk> next_level;
        std::vector<ScatterChunk> final_level;
        if (final_pass)
            final_level.resize(level.size() * fanout);
        else
            next_level.resize(level.size() * fanout);

        std::atomic<size_t> next_group{0};
        runParallelPhase(pool, threads, [&](size_t /*tid*/)
        {
            ScatterScratch scratch;
            scratch.init(fanout, use_swwc);
            PaddedPODArray<UInt32> hist_local;
            hist_local.resize(fanout);
            PaddedPODArray<UInt32> lanes;
            if (interleave)
                lanes.resize(4 * fanout);
            PaddedPODArray<UInt16> pids;

            for (size_t g = next_group.fetch_add(1, std::memory_order_relaxed); g < level.size();
                 g = next_group.fetch_add(1, std::memory_order_relaxed))
            {
                checkCancelled(cancelled);
                RoutedChunk & group = level[g];
                if (group.rows == 0)
                    continue;

                /// Group inputs are owned (they are the previous pass's output), so memory is
                /// cycled eagerly, all worker-local: the histogram stores the group's 2-byte
                /// partition ids, after which the route words are never read again on the
                /// final pass; each column round allocates its output columns just-in-time,
                /// scatters through the ids, and drops the consumed input column - so the
                /// freed input extents are immediately reusable for the next round's output
                /// instead of sitting dirty until allocator decay, and a group in flight holds
                /// ~(C+1)/C of its size instead of 2x.
                pids.resize(group.rows);
                std::fill(hist_local.begin(), hist_local.end(), 0);
                if (interleave)
                    std::fill(lanes.begin(), lanes.end(), 0);
                histogramChunk(routeData(group), group.rows, shift, mask, hist_local.data(), interleave ? lanes.data() : nullptr, fanout, pids.data());
                if (interleave)
                    reduceHistogramLanes(hist_local.data(), lanes.data(), fanout);
                if (final_pass)
                    group.routes = nullptr;

                std::vector<PartitionOutput> group_parts(fanout);
                for (size_t p = 0; p < fanout; ++p)
                    group_parts[p].rows = hist_local[p];

                for (size_t j = 0; j < scatter_columns; ++j)
                {
                    const size_t width = widths[j];
                    const bool seeded = widthUsesSeededFill(width);

                    /// Just-in-time exact allocation of this round's output columns: by now
                    /// the previous round's input column has been dropped, so its extents back
                    /// this.
                    for (size_t p = 0; p < fanout; ++p)
                        if (hist_local[p])
                            group_parts[p].allocateColumn(*samples[j]);

                    for (size_t p = 0; p < fanout; ++p)
                        scratch.seed(p, hist_local[p] ? group_parts[p].bases[j] : nullptr, seeded);

                    const char * data = j < num_columns
                        ? group.columns[j]->getRawData().data()
                        : group.routes->getRawData().data();
                    scatterChunkColumn(pids.data(), data, group.rows, width, use_swwc, scratch);
                    scratch.drain();

                    /// Input column j is fully consumed (the stored ids replace all further
                    /// routing uses): drop it before the next round allocates its outputs.
                    if (j < num_columns)
                        group.columns[j] = nullptr;
                    else
                        group.routes = nullptr;
                }

                for (size_t p = 0; p < fanout; ++p)
                {
                    if (!hist_local[p])
                        continue;
                    if (final_pass)
                        final_level[g * fanout + p] = group_parts[p].toChunk();
                    else
                        next_level[g * fanout + p] = toRoutedChunk(std::move(group_parts[p]));
                }

                /// Only empty column shells remain; free them before moving to the next group.
                group = RoutedChunk{};
            }
        });

        if (final_pass)
            return final_level;
        level = std::move(next_level);
        bits_done += bits;
    }

    throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix column scatter: pass loop ended without a final pass");
}

void scatterWaves(
    ThreadPool & pool,
    size_t num_threads,
    std::vector<RoutedChunk> chunks,
    size_t bits,
    size_t waves,
    const ConsumePartition & consume,
    std::atomic<bool> & cancelled)
{
    validateThreads(pool, num_threads);
    if (bits < 1 || bits > 16) /// partition ids are UInt16
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix wave scatter: bits must be in [1, 16], got {}", bits);
    if (!consume)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Radix wave scatter: no consume callback");
    const ScatterSchema schema = validateChunks(chunks);
    if (chunks.empty())
        return;

    const size_t threads = num_threads;
    const size_t fanout = size_t{1} << bits;
    const UInt32 shift = static_cast<UInt32>(32 - bits);
    const UInt32 mask = static_cast<UInt32>(fanout - 1);
    const bool use_swwc = fanout >= SWWC_MIN_FANOUT;
    const bool interleave = fanout <= HIST_INTERLEAVE_MAX_FANOUT;
    const size_t num_columns = schema.num_columns;
    const size_t num_waves = std::max<size_t>(1, std::min(waves, chunks.size()));

    /// Shared per-wave state, allocated once. Every phase writes disjoint slices per worker
    /// (histogram/offset stripes, contiguous partition ranges), so the phase barriers are the
    /// only synchronization; `next_partition` drives the consume phase's work stealing and is
    /// reset during the allocation phase (a barrier separates it from both neighboring uses).
    PaddedPODArray<UInt32> hist;
    hist.resize(threads * fanout);
    PaddedPODArray<UInt32> offsets;
    offsets.resize(threads * fanout);
    PaddedPODArray<UInt64> totals;
    totals.resize_fill(fanout, 0);
    std::vector<PartitionOutput> parts(fanout);
    std::atomic<size_t> next_partition{0};

    PhaseTeam team(threads, cancelled);

    runTeamWorkers(pool, threads, [&](size_t tid)
    {
        /// Per-worker scratch, persistent across all waves. Nothing here allocates: every
        /// fallible operation runs inside a phase, so a failure still reaches the barrier.
        ScatterScratch scratch;
        PaddedPODArray<UInt32> lanes;
        PaddedPODArray<UInt16> pids;
        std::vector<char *> col_cursors;

        for (size_t w = 0; w < num_waves; ++w)
        {
            const size_t begin = chunks.size() * w / num_waves;
            const size_t end = chunks.size() * (w + 1) / num_waves;

            /// Histogram + partition ids of this worker's chunk stripe of the window. The ids
            /// end all routing uses of the window's route words, which are dropped right here.
            team.phase([&]
            {
                if (w == 0)
                {
                    scratch.init(fanout, use_swwc);
                    if (interleave)
                        lanes.resize(4 * fanout);
                    col_cursors.resize(num_columns * fanout);
                }
                UInt32 * h = hist.data() + tid * fanout;
                memset(h, 0, fanout * sizeof(UInt32));
                if (interleave)
                    std::fill(lanes.begin(), lanes.end(), 0);
                size_t stripe_rows = 0;
                for (size_t c = begin + tid; c < end; c += threads)
                    stripe_rows += chunks[c].rows;
                pids.resize(stripe_rows);
                size_t row = 0;
                for (size_t c = begin + tid; c < end; c += threads)
                {
                    histogramChunk(routeData(chunks[c]), chunks[c].rows, shift, mask, h, interleave ? lanes.data() : nullptr, fanout, pids.data() + row);
                    row += chunks[c].rows;
                    chunks[c].routes = nullptr;
                }
                if (interleave)
                    reduceHistogramLanes(h, lanes.data(), fanout);
            });

            /// Fused prefix sum + exact allocation of this worker's partition range.
            team.phase([&]
            {
                for (size_t p = fanout * tid / threads; p < fanout * (tid + 1) / threads; ++p)
                {
                    UInt64 total = 0;
                    for (size_t worker = 0; worker < threads; ++worker)
                    {
                        offsets[worker * fanout + p] = static_cast<UInt32>(total);
                        total += hist[worker * fanout + p];
                    }
                    totals[p] = total;
                    parts[p] = PartitionOutput{};
                    if (total)
                        parts[p].allocate(schema.samples, total);
                }
                if (tid == 0)
                    next_partition.store(0, std::memory_order_relaxed);
            });

            /// Fused all-columns scatter of the stripe (same structure as the first pass of
            /// scatterColumns, without the intra-window batching: the window is the batch),
            /// then release the window's input chunks - in a real pipeline the upstream blocks
            /// are recycled here, which is what makes a byte budget on the window hold.
            team.phase([&]
            {
                for (size_t j = 0; j < num_columns; ++j)
                    for (size_t p = 0; p < fanout; ++p)
                        col_cursors[j * fanout + p]
                            = totals[p] ? parts[p].bases[j] + size_t{offsets[tid * fanout + p]} * schema.widths[j] : nullptr;
                for (size_t j = 0; j < num_columns; ++j)
                {
                    const size_t width = schema.widths[j];
                    const bool seeded = widthUsesSeededFill(width);
                    for (size_t p = 0; p < fanout; ++p)
                        scratch.seed(p, col_cursors[j * fanout + p], seeded);
                    size_t row = 0;
                    for (size_t c = begin + tid; c < end; c += threads)
                    {
                        scatterChunkColumn(pids.data() + row, chunks[c].columns[j]->getRawData().data(), chunks[c].rows, width, use_swwc, scratch);
                        row += chunks[c].rows;
                    }
                    scratch.drain();
                }
                for (size_t c = begin + tid; c < end; c += threads)
                    chunks[c].columns.clear();
            });

            /// Consume every non-empty partition of the window (work stealing), handing
            /// ownership of the window's partition chunk to the callback (freed on return).
            team.phase([&]
            {
                for (size_t p = next_partition.fetch_add(1, std::memory_order_relaxed); p < fanout;
                     p = next_partition.fetch_add(1, std::memory_order_relaxed))
                {
                    /// Best-effort shedding: once a sibling has failed there is no point in
                    /// consuming the rest of the wave. Leaving the loop early is safe - the
                    /// barrier arrival in phase is what coherence depends on, not the loop.
                    if (team.stopRequested())
                        break;
                    team.throwIfCancelled();
                    if (!totals[p])
                        continue;
                    consume(p, parts[p].toChunk());
                }
            });

            /// Uniform wave boundary: `stopping` was latched inside the consume barrier's
            /// serial completion step, so every worker reads the same value here and all of
            /// them leave the barrier loop together - nobody is left waiting.
            if (team.stopping())
                break;
        }
    });

    team.rethrowIfAny();
}

}
