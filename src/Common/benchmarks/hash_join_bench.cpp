#include "hash_join_bench.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <string_view>

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

#include <Columns/ColumnsNumber.h>
#include <Core/Defines.h>
#include <Core/Settings.h>
#include <Interpreters/TableJoin.h>
#include <QueryPipeline/SizeLimits.h>
#include <Common/CurrentMetrics.h>
#include <Common/HashTable/Hash.h>
#include <Common/PODArray.h>
#include <Common/Stopwatch.h>
#include <Common/assert_cast.h>

namespace CurrentMetrics
{
    extern const Metric LocalThread;
    extern const Metric LocalThreadActive;
    extern const Metric LocalThreadScheduled;
}

namespace DB::JoinBench
{

std::atomic<UInt64> g_sink{0};

WorkerPool::WorkerPool(size_t num_threads_)
    : num_threads(num_threads_)
    , pool(CurrentMetrics::LocalThread, CurrentMetrics::LocalThreadActive, CurrentMetrics::LocalThreadScheduled,
           num_threads_, /*max_free_threads_*/ num_threads_, /*queue_size_*/ 0)
{
}

double WorkerPool::run(const std::function<void(size_t)> & task)
{
    Stopwatch watch;
    for (size_t t = 0; t < num_threads; ++t)
        pool.scheduleOrThrowOnError([&task, t] { task(t); });
    pool.wait();
    return watch.elapsedSeconds();
}

namespace
{

/// Radix scatter after origin/phj5-real's KeyRefScatter, adapted to column-by-column output:
///   - the 32-bit route word is recomputed inline wherever needed (never stored; the route
///     hash is ~1 cycle/row, cheaper than 4 B/row of traffic); every pass slices a disjoint
///     bit range of the same word;
///   - per-partition destination columns are allocated exactly once from a histogram
///     (prefix sum + direct placement; no piece lists, no coalescing pass, no allocator churn);
///   - columns are scattered one at a time (column-major loop order), so only `fanout` output
///     streams and one fanout x 64 B staging set are live at any instant;
///   - at fanout >= 256 a software write-combining path stages one 64-byte line per partition
///     and flushes it with a non-temporal store, avoiding both the cache pollution and the
///     read-for-ownership traffic that create the high-fanout cliff of the naive scatter.

/// All passes slice disjoint bit ranges of this single 32-bit word.
///
/// The route hash must be independent of `HashCRC32` (CRC32C, the Castagnoli polynomial) that
/// the real HashJoin/parallel_hash tables use for bucketing, otherwise partition assignment
/// correlates with in-table bucket placement and per-partition tables see a skewed hash space.
///   - aarch64: the ISO-polynomial CRC32 instruction (`__crc32d`, polynomial 0x04C11DB7) is as
///     cheap as CRC32C but a different function;
///   - elsewhere: multiply-shift routing (as origin/phj5-real does on x86-64, where only the
///     CRC32C instruction exists).
inline UInt32 routeWord(UInt64 key)
{
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
    return __crc32d(-1U, key);
#else
    return static_cast<UInt32>((key * 0x9E3779B97F4A7C15ULL) >> 32);
#endif
}

constexpr size_t LINE_BYTES = 64;
constexpr size_t ELEMS_PER_LINE = LINE_BYTES / sizeof(UInt64);
/// Fanout from which the SWWC + non-temporal path wins over plain per-partition cursors
/// (the direct path's live output lines no longer stay cache-resident).
constexpr size_t SWWC_MIN_FANOUT = 256;

using NtLine = char __attribute__((vector_size(LINE_BYTES)));

/// Per-worker scatter state: write cursors, and for the SWWC path one 64-byte staging line
/// per partition plus fill/peel counters.
struct ScatterScratch
{
    size_t fanout = 0;
    bool use_swwc = false;
    PaddedPODArray<char> staging_mem;
    char * staging = nullptr;
    PaddedPODArray<UInt64 *> cursors;
    PaddedPODArray<UInt32> fill; /// bytes currently staged for the partition's line
    PaddedPODArray<UInt32> peel; /// elements to write directly until the cursor is line-aligned

    void init(size_t fanout_, bool use_swwc_)
    {
        fanout = fanout_;
        use_swwc = use_swwc_;
        cursors.resize(fanout);
        if (use_swwc)
        {
            staging_mem.resize(fanout * LINE_BYTES + LINE_BYTES);
            staging = reinterpret_cast<char *>(
                (reinterpret_cast<uintptr_t>(staging_mem.data()) + LINE_BYTES - 1) & ~static_cast<uintptr_t>(LINE_BYTES - 1));
            fill.resize(fanout);
            peel.resize(fanout);
        }
    }

    void seed(size_t p, UInt64 * cursor)
    {
        cursors[p] = cursor;
        if (use_swwc)
        {
            fill[p] = 0;
            const UInt32 misalign = static_cast<UInt32>(reinterpret_cast<uintptr_t>(cursor) & (LINE_BYTES - 1));
            peel[p] = ((LINE_BYTES - misalign) & (LINE_BYTES - 1)) / sizeof(UInt64);
        }
    }

    /// Flush residual (< one line) staged bytes of every partition and publish the NT stores.
    void drain()
    {
        if (!use_swwc)
            return;
        for (size_t p = 0; p < fanout; ++p)
        {
            const UInt32 f = fill[p];
            if (f)
            {
                memcpy(cursors[p], staging + p * LINE_BYTES, f);
                cursors[p] += f / sizeof(UInt64);
                fill[p] = 0;
            }
        }
        /// NT stores are weakly ordered; make them visible before the outputs are read.
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
};

void histogramChunk(const UInt64 * keys, size_t n, UInt32 shift, UInt32 mask, UInt64 * hist)
{
    for (size_t i = 0; i < n; ++i)
        ++hist[(routeWord(keys[i]) >> shift) & mask];
}

void scatterChunkDirect(const UInt64 * keys, const UInt64 * data, size_t n, UInt32 shift, UInt32 mask, UInt64 ** cursors)
{
    for (size_t i = 0; i < n; ++i)
    {
        const UInt32 p = (routeWord(keys[i]) >> shift) & mask;
        *cursors[p]++ = data[i];
    }
}

void scatterChunkSwwc(const UInt64 * keys, const UInt64 * data, size_t n, UInt32 shift, UInt32 mask, ScatterScratch & scratch)
{
    char * staging = scratch.staging;
    for (size_t i = 0; i < n; ++i)
    {
        const UInt32 p = (routeWord(keys[i]) >> shift) & mask;
        if (scratch.peel[p])
        {
            *scratch.cursors[p]++ = data[i];
            --scratch.peel[p];
            continue;
        }

        char * line = staging + static_cast<size_t>(p) * LINE_BYTES;
        UInt32 f = scratch.fill[p];
        *reinterpret_cast<UInt64 *>(line + f) = data[i];
        f += sizeof(UInt64);
        if (f == LINE_BYTES)
        {
            __builtin_nontemporal_store(*reinterpret_cast<const NtLine *>(line), reinterpret_cast<NtLine *>(scratch.cursors[p]));
            scratch.cursors[p] += ELEMS_PER_LINE;
            f = 0;
        }
        scratch.fill[p] = f;
    }
}

const UInt64 * keyData(const Chunk & chunk)
{
    return assert_cast<const ColumnUInt64 &>(*chunk.columns[0]).getData().data();
}

const UInt64 * columnData(const Chunk & chunk, size_t j)
{
    return assert_cast<const ColumnUInt64 &>(*chunk.columns[j]).getData().data();
}

/// Exactly-sized destination columns of one output partition, with raw write pointers.
struct PartitionOutput
{
    std::vector<MutableColumnPtr> columns;
    std::vector<UInt64 *> bases;
    size_t rows = 0;

    void allocate(size_t num_columns, size_t rows_)
    {
        rows = rows_;
        columns.reserve(num_columns);
        bases.reserve(num_columns);
        for (size_t j = 0; j < num_columns; ++j)
        {
            /// ColumnVector(n) leaves POD contents uninitialized: no memset, pages are
            /// first-touched by the scatter writes themselves.
            auto col = ColumnUInt64::create(rows_);
            bases.push_back(col->getData().data());
            columns.push_back(std::move(col));
        }
    }

    Chunk toChunk()
    {
        Chunk chunk;
        chunk.rows = rows;
        for (auto & col : columns)
            chunk.columns.emplace_back(std::move(col));
        return chunk;
    }
};

/// One radix pass: split every input group into `fanout` sub-partitions, each materialized as
/// a single exactly-sized chunk.
std::vector<ChunkList> scatterPass(WorkerPool & pool, const std::vector<ChunkList> & groups, size_t bits, size_t bits_done)
{
    const size_t threads = pool.size();
    const size_t fanout = 1ULL << bits;
    chassert(bits_done + bits <= 32);
    const UInt32 shift = static_cast<UInt32>(32 - bits_done - bits);
    const UInt32 mask = static_cast<UInt32>(fanout - 1);
    const bool use_swwc = fanout >= SWWC_MIN_FANOUT;
    std::vector<ChunkList> out(groups.size() * fanout);

    if (groups.size() == 1)
    {
        /// First pass: all threads cooperate on the single group; chunks are striped over
        /// workers identically in the histogram and scatter phases.
        const ChunkList & chunks = groups[0];
        if (chunks.empty())
            return out;
        const size_t num_columns = chunks.front().columns.size();

        /// Phase A: per-worker histograms.
        std::vector<std::vector<UInt64>> hist(threads);
        pool.run([&](size_t tid)
        {
            auto & h = hist[tid];
            h.assign(fanout, 0);
            for (size_t c = tid; c < chunks.size(); c += threads)
                histogramChunk(keyData(chunks[c]), chunks[c].rows, shift, mask, h.data());
        });

        /// Phase B: prefix sums -> per-(worker, partition) start offsets, then one exact
        /// allocation per non-empty partition.
        std::vector<std::vector<UInt64>> offsets(threads, std::vector<UInt64>(fanout));
        std::vector<UInt64> totals(fanout, 0);
        for (size_t p = 0; p < fanout; ++p)
        {
            for (size_t w = 0; w < threads; ++w)
            {
                offsets[w][p] = totals[p];
                totals[p] += hist[w][p];
            }
        }

        std::vector<PartitionOutput> parts(fanout);
        pool.run([&](size_t tid)
        {
            for (size_t p = tid; p < fanout; p += threads)
                if (totals[p])
                    parts[p].allocate(num_columns, totals[p]);
        });

        /// Phase C: column-major scatter; only `fanout` output streams (and one staging set)
        /// are live at any instant, keeping the SWWC/dTLB behavior independent of column count.
        std::vector<ScatterScratch> scratch(threads);
        for (size_t j = 0; j < num_columns; ++j)
        {
            pool.run([&, j](size_t tid)
            {
                auto & s = scratch[tid];
                if (s.fanout != fanout)
                    s.init(fanout, use_swwc);
                for (size_t p = 0; p < fanout; ++p)
                    s.seed(p, totals[p] ? parts[p].bases[j] + offsets[tid][p] : nullptr);

                for (size_t c = tid; c < chunks.size(); c += threads)
                {
                    if (use_swwc)
                        scatterChunkSwwc(keyData(chunks[c]), columnData(chunks[c], j), chunks[c].rows, shift, mask, s);
                    else
                        scatterChunkDirect(keyData(chunks[c]), columnData(chunks[c], j), chunks[c].rows, shift, mask, s.cursors.data());
                }
                s.drain();
            });
        }

        for (size_t p = 0; p < fanout; ++p)
            if (totals[p])
                out[p].push_back(parts[p].toChunk());
    }
    else
    {
        /// Refine passes (multi-pass fallback): whole groups are assigned to workers, each
        /// group scattered single-threaded with worker-local state.
        pool.run([&](size_t tid)
        {
            ScatterScratch scratch;
            scratch.init(fanout, use_swwc);
            std::vector<UInt64> hist(fanout);

            for (size_t g = tid; g < groups.size(); g += threads)
            {
                const ChunkList & chunks = groups[g];
                if (chunks.empty())
                    continue;
                const size_t num_columns = chunks.front().columns.size();

                hist.assign(fanout, 0);
                for (const auto & chunk : chunks)
                    histogramChunk(keyData(chunk), chunk.rows, shift, mask, hist.data());

                std::vector<PartitionOutput> parts(fanout);
                for (size_t p = 0; p < fanout; ++p)
                    if (hist[p])
                        parts[p].allocate(num_columns, hist[p]);

                for (size_t j = 0; j < num_columns; ++j)
                {
                    for (size_t p = 0; p < fanout; ++p)
                        scratch.seed(p, hist[p] ? parts[p].bases[j] : nullptr);

                    for (const auto & chunk : chunks)
                    {
                        if (use_swwc)
                            scatterChunkSwwc(keyData(chunk), columnData(chunk, j), chunk.rows, shift, mask, scratch);
                        else
                            scatterChunkDirect(keyData(chunk), columnData(chunk, j), chunk.rows, shift, mask, scratch.cursors.data());
                    }
                    scratch.drain();
                }

                for (size_t p = 0; p < fanout; ++p)
                    if (hist[p])
                        out[g * fanout + p].push_back(parts[p].toChunk());
            }
        });
    }

    return out;
}

}

std::vector<size_t> computePassBits(size_t p_star, size_t f_max)
{
    const size_t total_bits = static_cast<size_t>(std::countr_zero(std::bit_ceil(p_star)));
    const size_t f_bits = std::max<size_t>(1, static_cast<size_t>(std::bit_width(std::bit_floor(std::max<size_t>(2, f_max))) - 1));
    const size_t n_pass = (total_bits + f_bits - 1) / f_bits;
    const size_t per_pass = (total_bits + n_pass - 1) / n_pass;

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

std::vector<ChunkList> scatterSide(WorkerPool & pool, const std::vector<Block> & blocks, const std::vector<size_t> & pass_bits)
{
    std::vector<ChunkList> groups(1);
    groups[0].reserve(blocks.size());
    for (const auto & block : blocks)
    {
        Chunk chunk;
        chunk.rows = block.rows();
        for (size_t j = 0; j < block.columns(); ++j)
            chunk.columns.push_back(block.getByPosition(j).column);
        groups[0].push_back(std::move(chunk));
    }

    size_t bits_done = 0;
    for (size_t bits : pass_bits)
    {
        groups = scatterPass(pool, groups, bits, bits_done);
        bits_done += bits;
    }
    return groups;
}

std::shared_ptr<TableJoin> makeTableJoin(const Block & left_header, const Block & right_header)
{
    /// Construct from default query Settings so that all behavior flags match a real query —
    /// notably `enable_software_prefetch_in_join` (default true; the bare StorageJoin-style
    /// constructor leaves it false, silently disabling the join's software prefetching).
    ///
    /// INNER ALL. Note: ClickHouse ANY INNER marks right rows used-once (one output row per
    /// distinct matched right key), which does not match the model's one-match-per-probe-row
    /// assumption; benchmarks therefore use ALL with duplicate-free build keys where the output
    /// size must equal the probe side.
    static const Settings default_settings;
    auto table_join = std::make_shared<TableJoin>(default_settings, /*tmp_volume*/ nullptr, /*tmp_data*/ nullptr);
    table_join->setKind(JoinKind::Inner);
    table_join->getTableJoin().strictness = JoinStrictness::All;
    table_join->addDisjunct();
    table_join->getClauses().back().addKey(
        left_header.getByPosition(0).name, right_header.getByPosition(0).name, /*null_safe_comparison*/ false);
    chassert(table_join->enableSoftwarePrefetchInJoin());

    NamesAndTypesList left_columns;
    NamesAndTypesList right_columns;
    Names used_columns;
    for (const auto & col : left_header)
    {
        left_columns.emplace_back(col.name, col.type);
        used_columns.push_back(col.name);
    }
    for (const auto & col : right_header)
    {
        right_columns.emplace_back(col.name, col.type);
        used_columns.push_back(col.name);
    }
    table_join->setInputColumns(std::move(left_columns), std::move(right_columns));
    table_join->setUsedColumns(used_columns);
    return table_join;
}

UInt64 blockFingerprint(const Block & block)
{
    const size_t rows = block.rows();
    if (rows == 0)
        return 0;

    /// Per row: a commutative sum over columns of h(value, column name), then a non-linear
    /// finalizer so cross-column row pairing matters; per block: a commutative sum over rows.
    PaddedPODArray<UInt64> acc(rows, 0);
    for (const auto & col : block)
    {
        const UInt64 name_hash = std::hash<std::string_view>{}(col.name);
        const auto & data = assert_cast<const ColumnUInt64 &>(*col.column).getData();
        for (size_t i = 0; i < rows; ++i)
            acc[i] += intHashCRC32(data[i] ^ name_hash);
    }

    UInt64 fingerprint = 0;
    for (size_t i = 0; i < rows; ++i)
        fingerprint += intHash64(acc[i]);
    return fingerprint;
}

size_t drainJoinResult(JoinResultPtr result, UInt64 * fingerprint)
{
    size_t rows = 0;
    while (true)
    {
        auto res = result->next();
        rows += res.block.rows();
        if (fingerprint)
            *fingerprint += blockFingerprint(res.block);
        if (res.is_last)
            break;
    }
    return rows;
}

JoinStats driveJoin(IJoinBench & join, const std::vector<Block> & build_blocks, const std::vector<Block> & probe_blocks, bool verify)
{
    JoinStats stats;
    Stopwatch build_watch;
    join.build(build_blocks);
    stats.build_sec = build_watch.elapsedSeconds();

    Stopwatch probe_watch;
    stats.matches = join.probe(probe_blocks, verify ? &stats.fingerprint : nullptr);
    stats.probe_sec = probe_watch.elapsedSeconds();
    return stats;
}

}
