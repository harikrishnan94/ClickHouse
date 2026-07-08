#include "hash_join_bench.h"

#include <algorithm>
#include <bit>

#include <Columns/ColumnsNumber.h>
#include <Core/Defines.h>
#include <Interpreters/TableJoin.h>
#include <QueryPipeline/SizeLimits.h>
#include <Common/CurrentMetrics.h>
#include <Common/HashTable/Hash.h>
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

using KeyHash = DefaultHash<UInt64>;

/// One radix pass: split every input group into `fanout` sub-partitions and coalesce the
/// pieces of every sub-partition into chunks of at most DEFAULT_BLOCK_SIZE rows (partitions
/// stay lists of normally-sized blocks across passes, like grace-join buckets).
std::vector<ChunkList> scatterPass(WorkerPool & pool, const std::vector<ChunkList> & groups, size_t bits, size_t bits_done)
{
    const size_t threads = pool.size();
    const size_t fanout = 1ULL << bits;
    const size_t shift = 64 - bits_done - bits;
    const size_t num_outputs = groups.size() * fanout;
    std::vector<ChunkList> out(num_outputs);

    auto scatter_chunk = [&](const Chunk & chunk, std::vector<Chunk> * dst)
    {
        const size_t n = chunk.rows;
        const auto & keys = assert_cast<const ColumnUInt64 &>(*chunk.columns[0]).getData();
        IColumn::Selector selector(n);
        for (size_t i = 0; i < n; ++i)
            selector[i] = (KeyHash{}(keys[i]) >> shift) & (fanout - 1);

        std::vector<Chunk> pieces(fanout);
        for (const auto & col : chunk.columns)
        {
            auto parts = col->scatter(fanout, selector);
            for (size_t p = 0; p < fanout; ++p)
                pieces[p].columns.emplace_back(std::move(parts[p]));
        }
        for (size_t p = 0; p < fanout; ++p)
        {
            pieces[p].rows = pieces[p].columns[0]->size();
            if (pieces[p].rows > 0)
                dst[p].push_back(std::move(pieces[p]));
        }
    };

    /// Phase A: threads split the chunks of every group (work is balanced over chunks, not
    /// groups, so later passes with few large groups still use all threads), producing
    /// per-thread piece lists per output partition.
    std::vector<std::pair<size_t, size_t>> work; /// (group, chunk)
    for (size_t g = 0; g < groups.size(); ++g)
        for (size_t c = 0; c < groups[g].size(); ++c)
            work.emplace_back(g, c);

    std::vector<std::vector<std::vector<Chunk>>> locals(threads);
    pool.run([&](size_t tid)
    {
        locals[tid].resize(num_outputs);
        for (size_t w = tid; w < work.size(); w += threads)
        {
            const auto [g, c] = work[w];
            scatter_chunk(groups[g][c], locals[tid].data() + g * fanout);
        }
    });

    /// Collect the pieces of every output partition (pointer moves only) and compute
    /// prefix row offsets.
    struct PartitionPieces
    {
        std::vector<Chunk> pieces;
        std::vector<size_t> piece_offsets; /// prefix sums, piece_offsets[i] = rows before piece i
        size_t rows = 0;
    };
    std::vector<PartitionPieces> parts(num_outputs);
    std::vector<std::pair<size_t, size_t>> units; /// (output, unit index within output)
    for (size_t o = 0; o < num_outputs; ++o)
    {
        auto & part = parts[o];
        for (size_t t = 0; t < threads; ++t)
            for (auto & chunk : locals[t][o])
                part.pieces.push_back(std::move(chunk));
        part.piece_offsets.reserve(part.pieces.size());
        for (const auto & piece : part.pieces)
        {
            part.piece_offsets.push_back(part.rows);
            part.rows += piece.rows;
        }
        const size_t num_units = (part.rows + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
        out[o].resize(num_units);
        for (size_t u = 0; u < num_units; ++u)
            units.emplace_back(o, u);
    }

    /// Phase B: coalesce in parallel over (partition, block-sized chunk) units, so early
    /// passes with low fanout still use all threads.
    pool.run([&](size_t tid)
    {
        for (size_t w = tid; w < units.size(); w += threads)
        {
            const auto [o, u] = units[w];
            const auto & part = parts[o];
            const size_t begin = u * DEFAULT_BLOCK_SIZE;
            const size_t end = std::min<size_t>(begin + DEFAULT_BLOCK_SIZE, part.rows);
            const size_t num_columns = part.pieces.front().columns.size();

            Chunk chunk;
            chunk.rows = end - begin;
            for (size_t j = 0; j < num_columns; ++j)
            {
                auto col = part.pieces.front().columns[j]->cloneEmpty();
                col->reserve(chunk.rows);

                /// Find the first piece containing `begin` and copy ranges until `end`.
                size_t piece_idx = std::upper_bound(part.piece_offsets.begin(), part.piece_offsets.end(), begin)
                    - part.piece_offsets.begin() - 1;
                size_t pos = begin;
                while (pos < end)
                {
                    const auto & piece = part.pieces[piece_idx];
                    const size_t piece_begin = pos - part.piece_offsets[piece_idx];
                    const size_t length = std::min(end - pos, piece.rows - piece_begin);
                    col->insertRangeFrom(*piece.columns[j], piece_begin, length);
                    pos += length;
                    ++piece_idx;
                }
                chunk.columns.emplace_back(std::move(col));
            }
            out[o][u] = std::move(chunk);
        }
    });

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
    /// INNER ALL. Note: ClickHouse ANY INNER marks right rows used-once (one output row per
    /// distinct matched right key), which does not match the model's one-match-per-probe-row
    /// assumption; benchmarks therefore use ALL with duplicate-free build keys where the output
    /// size must equal the probe side.
    auto table_join = std::make_shared<TableJoin>(
        SizeLimits{}, /*use_nulls*/ false, JoinKind::Inner, JoinStrictness::All,
        Names{right_header.getByPosition(0).name});
    table_join->setLeftKeys({left_header.getByPosition(0).name});

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

size_t drainJoinResult(JoinResultPtr result)
{
    size_t rows = 0;
    while (true)
    {
        auto res = result->next();
        rows += res.block.rows();
        if (res.is_last)
            break;
    }
    return rows;
}

JoinStats driveJoin(IJoinBench & join, const std::vector<Block> & build_blocks, const std::vector<Block> & probe_blocks)
{
    JoinStats stats;
    Stopwatch build_watch;
    join.build(build_blocks);
    stats.build_sec = build_watch.elapsedSeconds();

    Stopwatch probe_watch;
    stats.matches = join.probe(probe_blocks);
    stats.probe_sec = probe_watch.elapsedSeconds();
    return stats;
}

}
