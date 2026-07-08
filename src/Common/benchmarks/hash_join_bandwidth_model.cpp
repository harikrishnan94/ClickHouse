/** hash_join_bandwidth_model: a theoretical model comparing Radix Partitioned Hash Join (RPHJ)
  * against Non-Partitioned Hash Join (NPHJ), parameterized by memory bandwidth terms measured on
  * this machine. To avoid deviations between the model and reality, every model term is measured
  * by running the SAME ClickHouse code that executes in the real joins (including its software
  * prefetching, `RowRef` machinery, stored-block handling and output materialization):
  *
  *   - memcpy bandwidth  B_cpy       : baseline sequential copy (block squashing via
  *                                     `insertRangeFrom`); reported for reference.
  *   - scatter bandwidth B_scatter(P): one single-pass call of the radix join's own partitioning
  *                                     code (scatterSide: hash -> `IColumn::Selector` ->
  *                                     `IColumn::scatter` -> coalesce into block-sized chunks),
  *                                     swept over the fanout P to expose the fanout cliff.
  *   - t_build_np(S)  : ns/row of the real `ConcurrentHashJoin` build phase (concurrent
  *                      `addBlockToJoin` with its internal hash/selector dispatch into per-slot
  *                      two-level maps, plus the `onBuildPhaseFinish` bucket merge), as a
  *                      function of total hash table byte size S.
  *   - t_build_rp(S)  : ns/row of per-thread private real `HashJoin::addBlockToJoin` — the
  *                      radix join's per-partition build — as a function of per-table size.
  *   - t_pg_np(S)     : ns/row of the real `ConcurrentHashJoin::joinBlock` from all T threads
  *                      against the shared merged map: probe, gather and output-Block
  *                      materialization fused, exactly as production runs them.
  *   - t_pg_rp(S)     : ns/row of per-thread private `HashJoin::joinBlock` — the radix join's
  *                      per-partition probe+gather.
  *   - gather         : standalone gather term (output Block built via `IColumn::insertFrom` by
  *                      RowRef + `IColumn::replicate`, dropped in the timed region), swept over
  *                      the stored-build-side working set; reported for reference, not used by
  *                      the crossover model (production fuses gather into joinBlock).
  *
  * All kernels run multi-threaded on T threads, include memory allocation cost, and reuse no
  * memory across timed iterations except the immutable input blocks. Per-row times are wall
  * seconds * 1e9 / total rows over all threads, interpolated log-linearly in S between sweep
  * points and clamped at the ends.
  *
  * Model. Build side N_b rows of width w_b = 8 * (1 + build payload columns); probe side N_p rows
  * of width w_p; D distinct build keys; S(D) = byte size of `HashMap<UInt64, UInt64>` holding D
  * keys (exact grower emulation: load factor 0.5, growth x4 up to 2^23 cells, then x2); T threads.
  *
  *   T_NP = N_b * t_build_np(S) + N_p * t_pg_np(S)
  *
  * RPHJ partitions BOTH sides by key hash into P* partitions such that a per-partition table fits
  * the private-cache budget C = L2/2 and there is enough parallelism:
  * P* = max(pow2(S/C), pow2(T)), capped by --max-partitions, executed in n_pass scatter passes
  * where each pass has fanout at most F_max (the largest measured fanout still sustaining >= 80%
  * of peak scatter bandwidth), n_pass = ceil(log2(P*) / log2(F_max)):
  *
  *   T_RP = n_pass * (N_b*w_b + N_p*w_p) / B_scatter(per-pass fanout)
  *        + N_b * t_build_rp(S/P*) + N_p * t_pg_rp(S/P*)
  *
  * Crossover condition (RPHJ wins iff):
  *
  *   N_b * [t_build_np(S) - t_build_rp(S/P*)] + N_p * [t_pg_np(S) - t_pg_rp(S/P*)]
  *   >  n_pass * (N_b*w_b + N_p*w_p) / B_scatter(f)
  *
  * which makes the regimes explicit:
  *   - input size: the left side is ~0 while S fits in cache -> NPHJ wins for small inputs;
  *   - key space:  duplicate-heavy keys (small D) keep S cache-resident regardless of N_b ->
  *     RPHJ rarely wins;
  *   - partition count: P* grows with S; once P* > F_max more scatter passes are needed,
  *     multiplying the partitioning cost and pushing the crossover out;
  *   - payload width: larger w inflates the scatter cost linearly but also the probe+gather delta.
  *
  * The program measures the terms, prints the model constants, evaluates the model over a grid
  * of (N_b, N_p/N_b, key-space regime), prints the crossover summary, and finally (unless --quick)
  * validates the model by running real multi-threaded INNER joins at points near the predicted
  * crossover. The two competitors implement a common `IJoinBench` interface driven by the driver:
  *   - NPHJ is the real ClickHouse `ConcurrentHashJoin` (`parallel_hash`), used as-is through the
  *     `IJoin` interface (concurrent `addBlockToJoin`, `onBuildPhaseFinish` bucket merge,
  *     unpartitioned shared-map probe via `joinBlock`);
  *   - RPHJ is multi-pass `IColumn::scatter` radix partitioning (the same scatterSide code the
  *     scatter kernel measures) plus one real ClickHouse `HashJoin` per partition, built and
  *     probed single-threaded per partition through the same `IJoin` interface.
  * All phases run on the ClickHouse thread pool (`ThreadPoolImpl`, threads carry `ThreadStatus`
  * for efficient memory tracking); the binary uses jemalloc via `clickhouse_new_delete`.
  */

#include "config.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <pcg_random.hpp>

#if USE_JEMALLOC
#include <jemalloc/jemalloc.h>
#endif

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <base/types.h>

#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <Core/Defines.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Common/HashTable/HashMap.h>
#include <Common/PODArray.h>
#include <Common/Stopwatch.h>
#include <Common/assert_cast.h>
#include <Common/getNumberOfCPUCoresToUse.h>

#include "concurrent_hash_join_bench.h"
#include "hash_join_bench.h"
#include "radix_hash_join_bench.h"

using namespace DB;
using namespace DB::JoinBench;

namespace
{

static_assert(sizeof(HashMapCell<UInt64, UInt64, DefaultHash<UInt64>>) == 16);

/// Bijective scrambling so that generated keys are not consecutive integers.
UInt64 permuteKey(UInt64 x)
{
    return x * 0x9E3779B97F4A7C15ULL;
}

UInt64 packRowRef(size_t block, size_t row)
{
    return (static_cast<UInt64>(block) << 32) | static_cast<UInt64>(row);
}

size_t refBlock(UInt64 ref) { return ref >> 32; }
size_t refRow(UInt64 ref) { return ref & 0xFFFFFFFFULL; }

/// Keyspaces of different generator "slots" must not overlap.
constexpr UInt64 KEY_DOMAIN_STRIDE = 1ULL << 44;


struct Config
{
    size_t threads = 0;
    size_t build_payload_columns = 1;
    size_t probe_payload_columns = 1;
    size_t tuples = 1ULL << 27;
    double hit_rate = 1.0;
    size_t max_partitions = 16384;
    size_t max_table_bytes = 256ULL << 20;
    size_t gather_bytes = 4ULL << 30;
    size_t validation_max_rows = 1ULL << 26;
    size_t runs = 3;
    bool quick = false;
    UInt64 seed = 0x8899AABBCCDDEEFFULL;

    size_t buildRowWidth() const { return 8 * (1 + build_payload_columns); }
    size_t probeRowWidth() const { return 8 * (1 + probe_payload_columns); }
};


struct CacheInfo
{
    size_t l1d = 32ULL << 10;
    size_t l2 = 1ULL << 20;
    size_t llc = 32ULL << 20;
    bool detected = false;
};

size_t parseCacheSize(std::string s)
{
    while (!s.empty() && (s.back() == '\n' || s.back() == ' '))
        s.pop_back();
    if (s.empty())
        return 0;
    size_t multiplier = 1;
    char suffix = s.back();
    if (suffix == 'K' || suffix == 'k')
        multiplier = 1ULL << 10;
    else if (suffix == 'M' || suffix == 'm')
        multiplier = 1ULL << 20;
    else if (suffix == 'G' || suffix == 'g')
        multiplier = 1ULL << 30;
    if (multiplier != 1)
        s.pop_back();
    return static_cast<size_t>(std::stoull(s)) * multiplier;
}

std::string readSysfsLine(const std::filesystem::path & path)
{
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
}

CacheInfo detectCaches()
{
    CacheInfo info;
    namespace fs = std::filesystem;

    try
    {
        int max_level = 0;
        /// (level, shared_cpu_list) -> size; used to count distinct LLC instances.
        std::map<std::pair<int, std::string>, size_t> instances;

        for (const auto & cpu_entry : fs::directory_iterator("/sys/devices/system/cpu"))
        {
            const std::string name = cpu_entry.path().filename().string();
            if (name.size() < 4 || !name.starts_with("cpu") || !isdigit(static_cast<unsigned char>(name[3])))
                continue;

            fs::path cache_dir = cpu_entry.path() / "cache";
            if (!fs::exists(cache_dir))
                continue;

            for (const auto & idx_entry : fs::directory_iterator(cache_dir))
            {
                if (!idx_entry.path().filename().string().starts_with("index"))
                    continue;

                const std::string type = readSysfsLine(idx_entry.path() / "type");
                if (type == "Instruction")
                    continue;

                const int level = std::stoi(readSysfsLine(idx_entry.path() / "level"));
                const size_t size = parseCacheSize(readSysfsLine(idx_entry.path() / "size"));
                const std::string shared = readSysfsLine(idx_entry.path() / "shared_cpu_list");
                if (size == 0)
                    continue;

                if (name == "cpu0" && level == 1)
                    info.l1d = size;
                if (name == "cpu0" && level == 2)
                    info.l2 = size;

                max_level = std::max(max_level, level);
                instances[{level, shared}] = size;
            }
        }

        if (max_level > 0)
        {
            size_t llc_total = 0;
            for (const auto & [key, size] : instances)
                if (key.first == max_level)
                    llc_total += size;
            if (llc_total > 0)
            {
                info.llc = llc_total;
                info.detected = true;
            }
        }
    }
    catch (...) /// NOLINT(bugprone-empty-catch)
    {
        /// Fall back to defaults; the values are overridable from the command line.
    }

    return info;
}


template <typename F>
double medianTime(size_t runs, F && once)
{
    once(); /// warmup
    std::vector<double> times(runs);
    for (auto & t : times)
        t = once();
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}


/// Input generation: vector of Blocks with one UInt64 key column and a number of UInt64
/// payload columns, generated in parallel. Block b is owned by thread b % threads;
/// local_row passed to the generator is the row index within the owning thread's stream.
using KeyGenerator = std::function<UInt64(size_t block_idx, size_t row_in_block, size_t local_row, pcg64_fast & rng)>;

std::vector<Block> generateBlocks(
    WorkerPool & pool,
    size_t rows,
    size_t payload_columns,
    const std::string & name_prefix,
    const KeyGenerator & keygen,
    UInt64 seed)
{
    const size_t threads = pool.size();
    const size_t num_blocks = (rows + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
    std::vector<Block> blocks(num_blocks);
    auto type = std::make_shared<DataTypeUInt64>();

    pool.run([&](size_t tid)
    {
        pcg64_fast rng(seed * 0x9E3779B1ULL + tid);
        for (size_t b = tid; b < num_blocks; b += threads)
        {
            const size_t block_rows = std::min<size_t>(DEFAULT_BLOCK_SIZE, rows - b * DEFAULT_BLOCK_SIZE);
            const size_t local_base = (b / threads) * DEFAULT_BLOCK_SIZE;

            Block block;

            auto key_col = ColumnUInt64::create(block_rows);
            auto & key_data = key_col->getData();
            for (size_t i = 0; i < block_rows; ++i)
                key_data[i] = keygen(b, i, local_base + i, rng);
            block.insert(ColumnWithTypeAndName(std::move(key_col), type, name_prefix + "key"));

            for (size_t c = 0; c < payload_columns; ++c)
            {
                auto col = ColumnUInt64::create(block_rows);
                auto & data = col->getData();
                for (size_t i = 0; i < block_rows; ++i)
                    data[i] = b * DEFAULT_BLOCK_SIZE + i;
                block.insert(ColumnWithTypeAndName(std::move(col), type, name_prefix + "p" + std::to_string(c)));
            }

            blocks[b] = std::move(block);
        }
    });

    return blocks;
}

size_t totalRows(const std::vector<Block> & blocks)
{
    size_t rows = 0;
    for (const auto & b : blocks)
        rows += b.rows();
    return rows;
}

KeyGenerator uniqueKeys()
{
    return [](size_t block_idx, size_t row_in_block, size_t /*local_row*/, pcg64_fast &)
    {
        return permuteKey(block_idx * DEFAULT_BLOCK_SIZE + row_in_block);
    };
}

/// Build-side keys: one global keyspace of `distinct` values, fully covered by the first
/// `distinct` rows, random duplicates afterwards.
KeyGenerator globalDomainKeys(size_t distinct)
{
    return [distinct](size_t block_idx, size_t row_in_block, size_t /*local_row*/, pcg64_fast & rng)
    {
        const UInt64 global_row = block_idx * DEFAULT_BLOCK_SIZE + row_in_block;
        return permuteKey(global_row < distinct ? global_row : rng() % distinct);
    };
}

/// Build-side keys: thread-private keyspace of `distinct` values, fully covered by the first
/// `distinct` rows of each thread, random duplicates afterwards.
KeyGenerator perThreadDomainKeys(size_t distinct, size_t threads)
{
    return [distinct, threads](size_t block_idx, size_t /*row_in_block*/, size_t local_row, pcg64_fast & rng)
    {
        const UInt64 offset = (block_idx % threads) * KEY_DOMAIN_STRIDE;
        const UInt64 raw = local_row < distinct ? local_row : rng() % distinct;
        return permuteKey(offset + raw);
    };
}

/// Probe-side keys against a keyspace of `distinct` values: hits with probability hit_rate,
/// misses drawn from a disjoint keyspace.
KeyGenerator probeKeys(size_t distinct, size_t threads, double hit_rate, bool per_thread_domain)
{
    const UInt64 hit_threshold = hit_rate >= 1.0
        ? std::numeric_limits<UInt64>::max()
        : static_cast<UInt64>(hit_rate * static_cast<double>(std::numeric_limits<UInt64>::max()));

    return [distinct, threads, hit_threshold, per_thread_domain](size_t block_idx, size_t /*row_in_block*/, size_t /*local_row*/, pcg64_fast & rng)
    {
        const UInt64 offset = per_thread_domain ? (block_idx % threads) * KEY_DOMAIN_STRIDE : 0;
        const bool hit = rng() <= hit_threshold;
        const UInt64 raw = rng() % distinct + (hit ? 0 : distinct);
        return permuteKey(offset + raw);
    };
}


std::string formatBytes(double bytes)
{
    const char * units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    size_t unit = 0;
    while (bytes >= 1024.0 && unit < 4)
    {
        bytes /= 1024.0;
        ++unit;
    }
    return fmt::format("{:.1f} {}", bytes, units[unit]);
}


/// Piecewise log-linear interpolation of ns/row over a size sweep.
struct Curve
{
    /// (bytes, ns_per_row), sorted by bytes.
    std::vector<std::pair<double, double>> points;

    double at(double bytes) const
    {
        chassert(!points.empty());
        if (bytes <= points.front().first)
            return points.front().second;
        if (bytes >= points.back().first)
            return points.back().second;
        for (size_t i = 1; i < points.size(); ++i)
        {
            if (bytes <= points[i].first)
            {
                const double x0 = std::log2(points[i - 1].first);
                const double x1 = std::log2(points[i].first);
                const double x = std::log2(bytes);
                const double f = (x - x0) / (x1 - x0);
                return points[i - 1].second + f * (points[i].second - points[i - 1].second);
            }
        }
        return points.back().second;
    }
};


/// ---------------------------------------------------------------------------------------------
/// Kernel 1: memcpy baseline. Each thread squashes its share of input blocks into freshly
/// allocated columns via insertRangeFrom, then the columns are dropped.
/// ---------------------------------------------------------------------------------------------
double runMemcpyKernel(const Config & cfg, WorkerPool & pool, const std::vector<Block> & blocks)
{
    const size_t threads = cfg.threads;
    const size_t rows = totalRows(blocks);
    const size_t row_width = 8 * blocks.front().columns();

    double seconds = medianTime(cfg.runs, [&]
    {
        return pool.run([&](size_t tid)
        {
            size_t my_rows = 0;
            for (size_t b = tid; b < blocks.size(); b += threads)
                my_rows += blocks[b].rows();

            MutableColumns dst;
            for (size_t j = 0; j < blocks.front().columns(); ++j)
            {
                dst.emplace_back(blocks.front().getByPosition(j).column->cloneEmpty());
                dst.back()->reserve(my_rows);
            }

            for (size_t b = tid; b < blocks.size(); b += threads)
                for (size_t j = 0; j < dst.size(); ++j)
                    dst[j]->insertRangeFrom(*blocks[b].getByPosition(j).column, 0, blocks[b].rows());

            g_sink += dst.front()->size();
            /// dst deallocated here, inside the timed region.
        });
    });

    return static_cast<double>(rows) * static_cast<double>(row_width) / seconds;
}


/// ---------------------------------------------------------------------------------------------
/// Kernel 2: scatter. One single-pass call of the radix join's own partitioning code
/// (scatterSide: hash -> Selector -> IColumn::scatter -> coalesce into block-sized chunks),
/// with all output freshly allocated and dropped inside the timed region. Sweep the fanout.
/// ---------------------------------------------------------------------------------------------
struct ScatterPoint
{
    size_t fanout;
    double bytes_per_sec;
};

std::vector<ScatterPoint> runScatterKernel(const Config & cfg, WorkerPool & pool, const std::vector<Block> & blocks)
{
    const size_t rows = totalRows(blocks);
    const size_t row_width = 8 * blocks.front().columns();

    std::vector<ScatterPoint> result;

    fmt::print("\n=== scatter bandwidth (fanout sweep, {} input, radix join's scatterSide) ===\n",
        formatBytes(static_cast<double>(rows) * static_cast<double>(row_width)));
    fmt::print("{:>10}{:>14}\n", "fanout", "GB/s");

    for (size_t fanout = 2; fanout <= cfg.max_partitions; fanout *= 4)
    {
        const size_t bits = static_cast<size_t>(std::countr_zero(fanout));

        double seconds = medianTime(cfg.runs, [&]
        {
            Stopwatch watch;
            {
                auto partitions = scatterSide(pool, blocks, {bits});
                g_sink += partitions.size();
                /// partitions deallocated here, inside the timed region.
            }
            return watch.elapsedSeconds();
        });

        double bw = static_cast<double>(rows) * static_cast<double>(row_width) / seconds;
        result.push_back({fanout, bw});
        fmt::print("{:>10}{:>14.2f}\n", fanout, bw / 1e9);
    }

    return result;
}


/// ---------------------------------------------------------------------------------------------
/// Hash table size model, shared by the sweeps and the analytical model.
/// ---------------------------------------------------------------------------------------------
/// Exact size of HashMap<UInt64, UInt64> holding `distinct` keys: 16-byte cells, load factor 0.5,
/// initial degree 8, growth by two degrees up to degree 23, then by one.
size_t htBytesForDistinct(size_t distinct)
{
    size_t degree = 8;
    while (distinct > (1ULL << (degree - 1)))
        degree += (degree >= 23 ? 1 : 2);
    return (1ULL << degree) * 16;
}

std::vector<size_t> tableSweepDistincts(const Config & cfg)
{
    std::vector<size_t> result;
    for (size_t d = 256; htBytesForDistinct(d) <= cfg.max_table_bytes; d *= 4)
        result.push_back(d);
    return result;
}

/// Empty header Block for one side: prefix + "key" column plus payload columns, all UInt64.
Block makeHeader(const std::string & prefix, size_t payload_columns)
{
    Block header;
    auto type = std::make_shared<DataTypeUInt64>();
    header.insert(ColumnWithTypeAndName(ColumnUInt64::create(), type, prefix + "key"));
    for (size_t c = 0; c < payload_columns; ++c)
        header.insert(ColumnWithTypeAndName(ColumnUInt64::create(), type, fmt::format("{}p{}", prefix, c)));
    return header;
}

/// ---------------------------------------------------------------------------------------------
/// Kernel 3a: radix per-partition build. Each thread builds a private real HashJoin (the same
/// class and code path the radix join uses per partition) over its share of the input blocks;
/// join construction, map growth and stored-block saving are inside the timed region.
/// Sweep the number of distinct keys (i.e. table size).
/// ---------------------------------------------------------------------------------------------
Curve runBuildKernelRP(const Config & cfg, WorkerPool & pool)
{
    const size_t threads = cfg.threads;
    Curve curve;

    fmt::print("\n=== HT build, radix per-partition (real HashJoin, size sweep) ===\n");
    fmt::print("{:>12}{:>14}{:>12}{:>14}\n", "distinct", "table", "ns/row", "Mrows/s");

    const Block left_header = makeHeader("p_", cfg.probe_payload_columns);
    const Block right_header = makeHeader("b_", cfg.build_payload_columns);
    auto table_join = makeTableJoin(left_header, right_header);
    auto shared_right_header = std::make_shared<const Block>(right_header);

    for (size_t distinct : tableSweepDistincts(cfg))
    {
        const size_t rows_per_thread = std::max(cfg.tuples / threads, distinct);
        const size_t rows = rows_per_thread * threads;
        auto blocks = generateBlocks(pool, rows, cfg.build_payload_columns, "b_",
                                     perThreadDomainKeys(distinct, threads), cfg.seed + distinct);
        const size_t actual_rows = totalRows(blocks);

        double seconds = medianTime(cfg.runs, [&]
        {
            std::vector<std::shared_ptr<HashJoin>> joins(threads);
            double elapsed = pool.run([&](size_t tid)
            {
                joins[tid] = std::make_shared<HashJoin>(
                    table_join, shared_right_header, /*any_take_last_row*/ false, /*reserve_num*/ 0,
                    fmt::format("bench{}", tid), /*use_two_level_maps*/ false);
                for (size_t b = tid; b < blocks.size(); b += threads)
                    joins[tid]->addBlockToJoin(blocks[b], /*check_limits*/ false);
                joins[tid]->onBuildPhaseFinish();
            });
            /// Untimed: fresh joins per iteration, destroyed in parallel after timing.
            pool.run([&](size_t tid) { joins[tid].reset(); });
            return elapsed;
        });

        const double bytes = static_cast<double>(htBytesForDistinct(distinct));
        const double ns_per_row = seconds * 1e9 / static_cast<double>(actual_rows);
        curve.points.emplace_back(bytes, ns_per_row);

        fmt::print("{:>12}{:>14}{:>12.3f}{:>14.1f}\n", distinct, formatBytes(bytes), ns_per_row, 1000.0 / ns_per_row);
    }

    return curve;
}

/// ---------------------------------------------------------------------------------------------
/// Kernel 3b: non-partitioned build. The timed region is the real ConcurrentHashJoin build
/// phase: concurrent addBlockToJoin (with its internal hash/selector dispatch) plus the
/// onBuildPhaseFinish two-level bucket merge. Sweep the number of distinct keys.
/// ---------------------------------------------------------------------------------------------
Curve runBuildKernelNP(const Config & cfg, WorkerPool & pool)
{
    Curve curve;

    fmt::print("\n=== HT build, non-partitioned (real ConcurrentHashJoin, size sweep) ===\n");
    fmt::print("{:>12}{:>14}{:>12}{:>14}\n", "distinct", "table", "ns/row", "Mrows/s");

    const Block left_header = makeHeader("p_", cfg.probe_payload_columns);
    const Block right_header = makeHeader("b_", cfg.build_payload_columns);

    /// A single shared table can be swept further than T private ones.
    for (size_t distinct = 256; htBytesForDistinct(distinct) <= cfg.max_table_bytes * 16; distinct *= 4)
    {
        const size_t rows = std::max(cfg.tuples / 4, distinct);
        auto blocks = generateBlocks(pool, rows, cfg.build_payload_columns, "b_", globalDomainKeys(distinct), cfg.seed + distinct);
        const size_t actual_rows = totalRows(blocks);

        double seconds = medianTime(cfg.runs, [&]
        {
            ConcurrentHashJoinBench bench(pool, left_header, right_header);
            Stopwatch watch;
            bench.build(blocks);
            return watch.elapsedSeconds();
            /// Untimed: the join is destroyed at scope end after the measurement.
        });

        const double bytes = static_cast<double>(htBytesForDistinct(distinct));
        const double ns_per_row = seconds * 1e9 / static_cast<double>(actual_rows);
        curve.points.emplace_back(bytes, ns_per_row);

        fmt::print("{:>12}{:>14}{:>12.3f}{:>14.1f}\n", distinct, formatBytes(bytes), ns_per_row, 1000.0 / ns_per_row);
    }

    return curve;
}

/// ---------------------------------------------------------------------------------------------
/// Kernel 4a: radix per-partition probe+gather. Per-thread private real HashJoin instances are
/// rebuilt per iteration (untimed); the timed region is joinBlock per probe block plus draining
/// the result into real output Blocks (probe, gather and output materialization fused, with
/// production prefetching), exactly what the radix join runs per partition.
/// ---------------------------------------------------------------------------------------------
Curve runProbeKernelRP(const Config & cfg, WorkerPool & pool)
{
    const size_t threads = cfg.threads;
    Curve curve;

    fmt::print("\n=== HT probe+gather, radix per-partition (real HashJoin, size sweep, hit rate {}) ===\n", cfg.hit_rate);
    fmt::print("{:>12}{:>14}{:>12}{:>14}\n", "distinct", "table", "ns/row", "Mrows/s");

    const Block left_header = makeHeader("p_", cfg.probe_payload_columns);
    const Block right_header = makeHeader("b_", cfg.build_payload_columns);
    auto table_join = makeTableJoin(left_header, right_header);
    auto shared_right_header = std::make_shared<const Block>(right_header);

    for (size_t distinct : tableSweepDistincts(cfg))
    {
        /// Build sides are duplicate-free (rows == distinct keys), so the INNER ALL output is
        /// exactly one row per matching probe row, as the model assumes.
        auto build_blocks = generateBlocks(pool, distinct * threads, cfg.build_payload_columns, "b_",
                                           perThreadDomainKeys(distinct, threads), cfg.seed + distinct);
        auto probe_blocks = generateBlocks(pool, cfg.tuples, cfg.probe_payload_columns, "p_",
                                           probeKeys(distinct, threads, cfg.hit_rate, /*per_thread_domain=*/ true),
                                           cfg.seed + distinct + 1);
        const size_t probe_rows = totalRows(probe_blocks);

        std::vector<std::shared_ptr<HashJoin>> joins(threads);

        auto rebuild = [&]
        {
            pool.run([&](size_t tid)
            {
                joins[tid] = std::make_shared<HashJoin>(
                    table_join, shared_right_header, /*any_take_last_row*/ false, /*reserve_num*/ 0,
                    fmt::format("bench{}", tid), /*use_two_level_maps*/ false);
                for (size_t b = tid; b < build_blocks.size(); b += threads)
                    joins[tid]->addBlockToJoin(build_blocks[b], /*check_limits*/ false);
                joins[tid]->onBuildPhaseFinish();
            });
        };

        double seconds = medianTime(cfg.runs, [&]
        {
            rebuild(); /// untimed: fresh joins for every iteration
            return pool.run([&](size_t tid)
            {
                size_t local_rows = 0;
                for (size_t b = tid; b < probe_blocks.size(); b += threads)
                    local_rows += drainJoinResult(joins[tid]->joinBlock(probe_blocks[b]));
                g_sink += local_rows;
                /// output Blocks deallocated here, inside the timed region.
            });
        });

        pool.run([&](size_t tid) { joins[tid].reset(); }); /// free before the next sweep point

        const double bytes = static_cast<double>(htBytesForDistinct(distinct));
        const double ns_per_row = seconds * 1e9 / static_cast<double>(probe_rows);
        curve.points.emplace_back(bytes, ns_per_row);

        fmt::print("{:>12}{:>14}{:>12.3f}{:>14.1f}\n", distinct, formatBytes(bytes), ns_per_row, 1000.0 / ns_per_row);
    }

    return curve;
}

/// ---------------------------------------------------------------------------------------------
/// Kernel 4b: non-partitioned probe+gather. The real ConcurrentHashJoin is rebuilt per
/// iteration (untimed); the timed region is the shared-map joinBlock probe from all T threads
/// with real output Blocks materialized and dropped.
/// ---------------------------------------------------------------------------------------------
Curve runProbeKernelNP(const Config & cfg, WorkerPool & pool)
{
    Curve curve;

    fmt::print("\n=== HT probe+gather, non-partitioned (real ConcurrentHashJoin, size sweep, hit rate {}) ===\n", cfg.hit_rate);
    fmt::print("{:>12}{:>14}{:>12}{:>14}\n", "distinct", "table", "ns/row", "Mrows/s");

    const Block left_header = makeHeader("p_", cfg.probe_payload_columns);
    const Block right_header = makeHeader("b_", cfg.build_payload_columns);

    for (size_t distinct = 256; htBytesForDistinct(distinct) <= cfg.max_table_bytes * 16; distinct *= 4)
    {
        /// Build side is duplicate-free (rows == distinct keys), so the INNER ALL output is
        /// exactly one row per matching probe row, as the model assumes.
        auto build_blocks = generateBlocks(pool, distinct, cfg.build_payload_columns, "b_", globalDomainKeys(distinct), cfg.seed + distinct);
        auto probe_blocks = generateBlocks(pool, cfg.tuples, cfg.probe_payload_columns, "p_",
                                           probeKeys(distinct, cfg.threads, cfg.hit_rate, /*per_thread_domain=*/ false),
                                           cfg.seed + distinct + 1);
        const size_t probe_rows = totalRows(probe_blocks);

        double seconds = medianTime(cfg.runs, [&]
        {
            /// Untimed: fresh join built for every iteration.
            ConcurrentHashJoinBench bench(pool, left_header, right_header);
            bench.build(build_blocks);

            Stopwatch watch;
            size_t rows = bench.probe(probe_blocks);
            double elapsed = watch.elapsedSeconds();
            g_sink += rows;
            return elapsed;
        });

        const double bytes = static_cast<double>(htBytesForDistinct(distinct));
        const double ns_per_row = seconds * 1e9 / static_cast<double>(probe_rows);
        curve.points.emplace_back(bytes, ns_per_row);

        fmt::print("{:>12}{:>14}{:>12.3f}{:>14.1f}\n", distinct, formatBytes(bytes), ns_per_row, 1000.0 / ns_per_row);
    }

    return curve;
}


/// ---------------------------------------------------------------------------------------------
/// Kernel 5: gather. From per-block match lists, materialize an output Block exactly like
/// production HashJoin does (LazyOutput::buildOutputFromBlocks): build-side columns appended
/// per matched RowRef via IColumn::insertFrom, probe-side columns expanded via
/// IColumn::replicate. The Block is created, filled and dropped inside the timed region.
/// Sweep the stored-build-side working set.
/// ---------------------------------------------------------------------------------------------
Curve runGatherKernel(const Config & cfg, WorkerPool & pool, const std::vector<Block> & probe_blocks)
{
    const size_t threads = cfg.threads;
    Curve curve;

    const size_t stored_block_count = std::max<size_t>(1, cfg.gather_bytes / cfg.buildRowWidth() / DEFAULT_BLOCK_SIZE);
    const size_t stored_rows = stored_block_count * DEFAULT_BLOCK_SIZE;
    auto stored_blocks = generateBlocks(pool, stored_rows, cfg.build_payload_columns, "b_", uniqueKeys(), cfg.seed + 12345);
    const size_t stored_block_bytes = DEFAULT_BLOCK_SIZE * cfg.buildRowWidth();
    const size_t build_columns = stored_blocks.front().columns();

    fmt::print("\n=== gather (stored build side sweep, output = CH Block, hit rate {}) ===\n", cfg.hit_rate);
    fmt::print("{:>16}{:>14}{:>16}\n", "working set", "ns/match", "Mmatches/s");

    struct BlockMatches
    {
        PaddedPODArray<UInt64> refs;
        IColumn::Offsets offsets;
    };

    const UInt64 hit_threshold = cfg.hit_rate >= 1.0
        ? std::numeric_limits<UInt64>::max()
        : static_cast<UInt64>(cfg.hit_rate * static_cast<double>(std::numeric_limits<UInt64>::max()));

    std::vector<size_t> sweep_blocks;
    for (size_t k = 1; k < stored_block_count; k *= 4)
        sweep_blocks.push_back(k);
    sweep_blocks.push_back(stored_block_count);

    for (size_t k : sweep_blocks)
    {
        /// Untimed prep: match lists (they are an input to the gather phase).
        std::vector<BlockMatches> matches(probe_blocks.size());
        std::atomic<size_t> total_matches{0};
        pool.run([&](size_t tid)
        {
            pcg64_fast rng(cfg.seed * 31 + tid + k);
            size_t local_matches = 0;
            for (size_t b = tid; b < probe_blocks.size(); b += threads)
            {
                const size_t n = probe_blocks[b].rows();
                auto & m = matches[b];
                m.offsets.resize(n);
                for (size_t i = 0; i < n; ++i)
                {
                    if (rng() <= hit_threshold)
                        m.refs.push_back(packRowRef(rng() % k, rng() % DEFAULT_BLOCK_SIZE));
                    m.offsets[i] = m.refs.size();
                }
                local_matches += m.refs.size();
            }
            total_matches += local_matches;
        });

        double seconds = medianTime(cfg.runs, [&]
        {
            return pool.run([&](size_t tid)
            {
                for (size_t b = tid; b < probe_blocks.size(); b += threads)
                {
                    const auto & m = matches[b];

                    Block out;
                    for (size_t j = 0; j < build_columns; ++j)
                    {
                        const auto & sample = stored_blocks.front().getByPosition(j);
                        auto dst = sample.column->cloneEmpty();
                        dst->reserve(m.refs.size());
                        for (UInt64 ref : m.refs)
                            dst->insertFrom(*stored_blocks[refBlock(ref)].getByPosition(j).column, refRow(ref));
                        out.insert(ColumnWithTypeAndName(std::move(dst), sample.type, sample.name));
                    }
                    for (size_t j = 0; j < probe_blocks[b].columns(); ++j)
                    {
                        const auto & src = probe_blocks[b].getByPosition(j);
                        out.insert(ColumnWithTypeAndName(src.column->replicate(m.offsets), src.type, src.name));
                    }

                    g_sink += out.rows();
                    /// output Block deallocated here, inside the timed region.
                }
            });
        });

        const double working_set = static_cast<double>(k) * static_cast<double>(stored_block_bytes);
        const double ns_per_match = seconds * 1e9 / static_cast<double>(total_matches.load());
        curve.points.emplace_back(working_set, ns_per_match);

        fmt::print("{:>16}{:>14.3f}{:>16.1f}\n", formatBytes(working_set), ns_per_match, 1000.0 / ns_per_match);
    }

    return curve;
}


/// ---------------------------------------------------------------------------------------------
/// Analytical model.
/// ---------------------------------------------------------------------------------------------
struct ModelInputs
{
    double memcpy_bytes_per_sec = 0;
    std::vector<ScatterPoint> scatter;
    double scatter_peak = 0;
    size_t f_max = 2;
    Curve build_np;   /// real ConcurrentHashJoin build phase, ns/row vs table size
    Curve build_rp;   /// real per-thread HashJoin build (per-partition shape), ns/row vs table size
    Curve probe_np;   /// real ConcurrentHashJoin probe+gather, ns/row vs table size
    Curve probe_rp;   /// real per-thread HashJoin probe+gather (per-partition shape), ns/row vs table size
    Curve gather;     /// standalone gather term (informational, not used by predict)
    size_t l2 = 0;
    size_t llc = 0;
    size_t w_b = 16;
    size_t w_p = 16;
    double hit_rate = 1.0;
    size_t max_partitions = 16384;
    size_t threads = 1;

    double scatterBytesPerSec(size_t fanout) const
    {
        chassert(!scatter.empty());
        if (fanout <= scatter.front().fanout)
            return scatter.front().bytes_per_sec;
        if (fanout >= scatter.back().fanout)
            return scatter.back().bytes_per_sec;
        for (size_t i = 1; i < scatter.size(); ++i)
        {
            if (fanout <= scatter[i].fanout)
            {
                const double x0 = std::log2(static_cast<double>(scatter[i - 1].fanout));
                const double x1 = std::log2(static_cast<double>(scatter[i].fanout));
                const double x = std::log2(static_cast<double>(fanout));
                const double f = (x - x0) / (x1 - x0);
                return scatter[i - 1].bytes_per_sec + f * (scatter[i].bytes_per_sec - scatter[i - 1].bytes_per_sec);
            }
        }
        return scatter.back().bytes_per_sec;
    }
};

struct Prediction
{
    double np_build_sec = 0;
    double np_probe_sec = 0; /// probe+gather (fused, as in the real joinBlock)
    double rp_scatter_sec = 0;
    double rp_build_sec = 0;
    double rp_probe_sec = 0; /// probe+gather (fused)
    size_t p_star = 1;
    size_t n_pass = 0;

    double npTotal() const { return np_build_sec + np_probe_sec; }
    double rpTotal() const { return rp_scatter_sec + rp_build_sec + rp_probe_sec; }
};

Prediction predict(const ModelInputs & m, double n_b, double n_p, double distinct)
{
    Prediction p;

    const double table_bytes = static_cast<double>(htBytesForDistinct(static_cast<size_t>(distinct)));

    /// NPHJ: both terms measured with the real ConcurrentHashJoin (build includes its internal
    /// hash/selector dispatch and the bucket merge; probe includes gather and output
    /// materialization, fused as in joinBlock).
    p.np_build_sec = n_b * m.build_np.at(table_bytes) * 1e-9;
    p.np_probe_sec = n_p * m.probe_np.at(table_bytes) * 1e-9;

    /// RPHJ: enough partitions for both cache residency and thread parallelism.
    const double budget = static_cast<double>(m.l2) / 2;
    size_t p_star = 1;
    while (table_bytes / static_cast<double>(p_star) > budget && p_star < m.max_partitions)
        p_star *= 2;
    if (p_star > 1)
        p_star = std::min(std::max(p_star, std::bit_ceil(m.threads)), std::bit_ceil(m.max_partitions));
    p.p_star = p_star;

    if (p_star == 1)
    {
        /// A radix join with one partition degenerates to the non-partitioned join.
        p.rp_build_sec = p.np_build_sec;
        p.rp_probe_sec = p.np_probe_sec;
        return p;
    }

    const size_t total_bits = static_cast<size_t>(std::countr_zero(p_star));
    const size_t f_bits = std::max<size_t>(1, static_cast<size_t>(std::bit_width(std::bit_floor(m.f_max)) - 1));
    p.n_pass = (total_bits + f_bits - 1) / f_bits;
    const size_t per_pass_bits = (total_bits + p.n_pass - 1) / p.n_pass;
    const size_t per_pass_fanout = 1ULL << per_pass_bits;

    const double scatter_bytes = n_b * static_cast<double>(m.w_b) + n_p * static_cast<double>(m.w_p);
    p.rp_scatter_sec = static_cast<double>(p.n_pass) * scatter_bytes / m.scatterBytesPerSec(per_pass_fanout);

    /// Per-partition terms measured with real per-thread HashJoin instances.
    const double part = static_cast<double>(p_star);
    p.rp_build_sec = n_b * m.build_rp.at(table_bytes / part) * 1e-9;
    p.rp_probe_sec = n_p * m.probe_rp.at(table_bytes / part) * 1e-9;

    return p;
}

struct Regime
{
    std::string name;
    std::function<size_t(size_t)> distinct_of_nb;
};

std::vector<Regime> gridRegimes()
{
    return
    {
        {"unique (D = N_b)", [](size_t n_b) { return n_b; }},
        {"dup x8 (D = N_b/8)", [](size_t n_b) { return std::max<size_t>(1, n_b / 8); }},
        {"fixed 64K (D = 65536)", [](size_t) { return size_t(65536); }},
    };
}

void printGridAndCrossover(const ModelInputs & m)
{
    const std::vector<size_t> ratios = {1, 10};

    fmt::print("\n=== model grid: predicted NPHJ vs RPHJ ===\n");
    fmt::print("{:>22}{:>9}{:>8}{:>12}{:>8}{:>7}{:>12}{:>12}{:>10}{:>10}\n",
        "regime", "N_p/N_b", "N_b", "HT size", "P*", "passes", "T_NP ms", "T_RP ms", "winner", "speedup");

    for (const auto & regime : gridRegimes())
    {
        for (size_t ratio : ratios)
        {
            for (size_t k = 16; k <= 28; k += 2)
            {
                const size_t n_b = 1ULL << k;
                const size_t n_p = n_b * ratio;
                const size_t distinct = regime.distinct_of_nb(n_b);
                auto p = predict(m, static_cast<double>(n_b), static_cast<double>(n_p), static_cast<double>(distinct));

                const bool radix_wins = p.rpTotal() < p.npTotal();
                fmt::print("{:>22}{:>9}{:>8}{:>12}{:>8}{:>7}{:>12.2f}{:>12.2f}{:>10}{:>10.2f}\n",
                    regime.name, ratio, fmt::format("2^{}", k),
                    formatBytes(static_cast<double>(htBytesForDistinct(distinct))),
                    p.p_star, p.n_pass, p.npTotal() * 1e3, p.rpTotal() * 1e3,
                    radix_wins ? "radix" : "non-part", p.npTotal() / p.rpTotal());
            }
            fmt::print("\n");
        }
    }

    fmt::print("=== crossover summary ===\n");
    for (const auto & regime : gridRegimes())
    {
        for (size_t ratio : ratios)
        {
            std::optional<size_t> crossover;
            for (size_t k = 14; k <= 30; ++k)
            {
                const size_t n_b = 1ULL << k;
                const size_t distinct = regime.distinct_of_nb(n_b);
                auto p = predict(m, static_cast<double>(n_b), static_cast<double>(n_b * ratio), static_cast<double>(distinct));
                if (p.rpTotal() < p.npTotal() * 0.999)
                {
                    crossover = n_b;
                    break;
                }
            }

            fmt::print("  regime [{}], N_p/N_b = {}: ", regime.name, ratio);
            if (!crossover)
            {
                fmt::print("radix partitioning never wins for N_b up to 2^30\n");
                continue;
            }

            const size_t n_b = *crossover;
            const size_t distinct = regime.distinct_of_nb(n_b);
            const size_t table_bytes = htBytesForDistinct(distinct);
            auto p = predict(m, static_cast<double>(n_b), static_cast<double>(n_b * ratio), static_cast<double>(distinct));
            fmt::print("radix wins from N_b >= {} (D = {}, HT = {} = {:.1f}x LLC, P* = {}, passes = {})\n",
                n_b, distinct, formatBytes(static_cast<double>(table_bytes)),
                static_cast<double>(table_bytes) / static_cast<double>(m.llc), p.p_star, p.n_pass);
        }
    }
}



/// ---------------------------------------------------------------------------------------------
/// Validation: real end-to-end INNER joins (implementations in concurrent_hash_join_bench.cpp
/// and radix_hash_join_bench.cpp) driven through the IJoinBench interface.
/// ---------------------------------------------------------------------------------------------

/// Run the two real joins at one exact (N_b, N_p) point, without measuring the model kernels.
/// Reports the measured wall time of every phase for each repetition.
void runSingleJoin(const Config & cfg, WorkerPool & pool, const CacheInfo & cache, size_t n_b, size_t n_p)
{
    const double budget = static_cast<double>(cache.l2) / 2;
    const double table_bytes = static_cast<double>(htBytesForDistinct(n_b));
    size_t p_star = 1;
    while (table_bytes / static_cast<double>(p_star) > budget && p_star < cfg.max_partitions)
        p_star *= 2;
    if (p_star > 1)
        p_star = std::min(std::max(p_star, std::bit_ceil(cfg.threads)), std::bit_ceil(cfg.max_partitions));
    p_star = std::max<size_t>(2, p_star);
    const size_t f_max = 8; /// typical measured knee of the scatter fanout curve

    fmt::print("\n=== single join: N_b = {}, N_p = {}, unique keys, hit rate {}, HT = {}, P* = {} ===\n",
        n_b, n_p, cfg.hit_rate, formatBytes(table_bytes), p_star);

    auto build_blocks = generateBlocks(pool, n_b, cfg.build_payload_columns, "b_", uniqueKeys(), cfg.seed + n_b);
    const UInt64 hit_threshold = cfg.hit_rate >= 1.0
        ? std::numeric_limits<UInt64>::max()
        : static_cast<UInt64>(cfg.hit_rate * static_cast<double>(std::numeric_limits<UInt64>::max()));
    auto probe_keygen = [n_b, hit_threshold](size_t, size_t, size_t, pcg64_fast & rng)
    {
        const bool hit = rng() <= hit_threshold;
        const UInt64 raw = rng() % n_b;
        return hit ? permuteKey(raw) : permuteKey(raw + KEY_DOMAIN_STRIDE);
    };
    auto probe_blocks = generateBlocks(pool, n_p, cfg.probe_payload_columns, "p_", probe_keygen, cfg.seed + n_b + 1);

    const Block left_header = probe_blocks.front().cloneEmpty();
    const Block right_header = build_blocks.front().cloneEmpty();

    for (size_t run = 0; run < cfg.runs; ++run)
    {
        JoinStats np;
        {
            ConcurrentHashJoinBench bench(pool, left_header, right_header);
            np = driveJoin(bench, build_blocks, probe_blocks);
        }

        JoinStats rp;
        std::string rp_detail;
        {
            RadixHashJoinBench bench(pool, left_header, right_header, p_star, f_max);
            rp = driveJoin(bench, build_blocks, probe_blocks);
            rp_detail = bench.phaseBreakdown();
        }

        fmt::print("  run {}: NPHJ total {:.2f} ms (build {:.2f} ms, probe+gather {:.2f} ms); "
                   "RPHJ total {:.2f} ms (build {:.2f} ms, probe {:.2f} ms; {}); matches {}{}\n",
            run, np.total() * 1e3, np.build_sec * 1e3, np.probe_sec * 1e3,
            rp.total() * 1e3, rp.build_sec * 1e3, rp.probe_sec * 1e3, rp_detail,
            np.matches, np.matches == rp.matches ? "" : " MISMATCH");
    }
}

void runValidation(const Config & cfg, WorkerPool & pool, const ModelInputs & model)
{
    for (size_t ratio : {size_t(1), size_t(10)})
    {
        /// Pick points around the predicted crossover in the unique-keys regime.
        std::optional<size_t> crossover;
        for (size_t k = 14; k <= 30; ++k)
        {
            const size_t n_b = 1ULL << k;
            auto p = predict(model, static_cast<double>(n_b), static_cast<double>(n_b * ratio), static_cast<double>(n_b));
            if (p.rpTotal() < p.npTotal() * 0.999)
            {
                crossover = n_b;
                break;
            }
        }

        std::vector<size_t> points;
        const size_t base = std::max(crossover.value_or(1ULL << 24), size_t(1) << 20);
        for (size_t n_b : {base / 16, base / 4, base, base * 8})
            if (n_b >= (1ULL << 16) && n_b <= cfg.validation_max_rows && n_b * ratio <= 4 * cfg.validation_max_rows
                && (points.empty() || n_b != points.back()))
                points.push_back(n_b);
        if (points.empty())
            points = {1ULL << 22, 1ULL << 24};

        fmt::print("\n=== validation: real joins vs model (unique keys, N_p = {} * N_b, hit rate {}) ===\n", ratio, cfg.hit_rate);
        if (crossover)
            fmt::print("  predicted crossover at N_b = {}\n", *crossover);
        else
            fmt::print("  no predicted crossover; validating at default points\n");

        fmt::print("{:>12}{:>8}{:>13}{:>13}{:>13}{:>13}{:>12}{:>12}{:>10}\n",
            "N_b", "P*", "NP pred ms", "NP meas ms", "RP pred ms", "RP meas ms", "pred win", "meas win", "matches");

        for (size_t n_b : points)
        {
            const size_t n_p = n_b * ratio;
            auto pred = predict(model, static_cast<double>(n_b), static_cast<double>(n_p), static_cast<double>(n_b));
            const size_t p_star = std::max<size_t>(2, pred.p_star);

            auto build_blocks = generateBlocks(pool, n_b, cfg.build_payload_columns, "b_", uniqueKeys(), cfg.seed + n_b);
            /// Probe keys drawn from the whole build keyspace (global domain).
            const UInt64 hit_threshold = cfg.hit_rate >= 1.0
                ? std::numeric_limits<UInt64>::max()
                : static_cast<UInt64>(cfg.hit_rate * static_cast<double>(std::numeric_limits<UInt64>::max()));
            auto probe_keygen = [n_b, hit_threshold](size_t, size_t, size_t, pcg64_fast & rng)
            {
                const bool hit = rng() <= hit_threshold;
                const UInt64 raw = rng() % n_b;
                return hit ? permuteKey(raw) : permuteKey(raw + KEY_DOMAIN_STRIDE);
            };
            auto probe_blocks = generateBlocks(pool, n_p, cfg.probe_payload_columns, "p_", probe_keygen, cfg.seed + n_b + 1);

            const Block left_header = probe_blocks.front().cloneEmpty();
            const Block right_header = build_blocks.front().cloneEmpty();

            JoinStats np;
            {
                ConcurrentHashJoinBench bench(pool, left_header, right_header);
                np = driveJoin(bench, build_blocks, probe_blocks);
            }

            JoinStats rp;
            std::string rp_detail;
            {
                RadixHashJoinBench bench(pool, left_header, right_header, p_star, model.f_max);
                rp = driveJoin(bench, build_blocks, probe_blocks);
                rp_detail = bench.phaseBreakdown();
            }

            const char * pred_win = pred.rpTotal() < pred.npTotal() ? "radix" : "non-part";
            const char * meas_win = rp.total() < np.total() ? "radix" : "non-part";
            const char * match_check = np.matches == rp.matches ? "ok" : "MISMATCH";

            fmt::print("{:>12}{:>8}{:>13.2f}{:>13.2f}{:>13.2f}{:>13.2f}{:>12}{:>12}{:>10}\n",
                n_b, p_star, pred.npTotal() * 1e3, np.total() * 1e3, pred.rpTotal() * 1e3, rp.total() * 1e3,
                pred_win, meas_win, match_check);

            fmt::print("      NP meas (build/probe+gather): {:.2f} / {:.2f} ms, pred: {:.2f} / {:.2f} ms;  "
                       "RP meas (build/probe): {:.2f} / {:.2f} ms ({});  "
                       "RP pred (scatter/build/probe+gather): {:.2f} / {:.2f} / {:.2f} ms\n",
                np.build_sec * 1e3, np.probe_sec * 1e3,
                pred.np_build_sec * 1e3, pred.np_probe_sec * 1e3,
                rp.build_sec * 1e3, rp.probe_sec * 1e3, rp_detail,
                pred.rp_scatter_sec * 1e3, pred.rp_build_sec * 1e3, pred.rp_probe_sec * 1e3);
        }
    }
}

}


int main(int argc, char ** argv)
{
    namespace po = boost::program_options;

    Config cfg;

    po::options_description desc("hash_join_bandwidth_model options");
    desc.add_options()
        ("help", "produce help message")
        ("threads", po::value<size_t>(), "number of worker threads (default: number of CPU cores)")
        ("build-payload-columns", po::value<size_t>(&cfg.build_payload_columns)->default_value(1), "8-byte payload columns on the build side")
        ("probe-payload-columns", po::value<size_t>(&cfg.probe_payload_columns)->default_value(1), "8-byte payload columns on the probe side")
        ("tuples", po::value<size_t>(&cfg.tuples)->default_value(1ULL << 27), "rows of work per kernel iteration (across all threads)")
        ("hit-rate", po::value<double>(&cfg.hit_rate)->default_value(1.0), "probe hit rate in [0, 1]")
        ("max-partitions", po::value<size_t>(&cfg.max_partitions)->default_value(16384), "maximum partition fanout")
        ("max-table-bytes", po::value<size_t>(&cfg.max_table_bytes)->default_value(256ULL << 20), "maximum per-thread hash table size in the sweep")
        ("gather-bytes", po::value<size_t>(&cfg.gather_bytes)->default_value(4ULL << 30), "maximum stored-build-side working set in the gather sweep")
        ("validation-max-rows", po::value<size_t>(&cfg.validation_max_rows)->default_value(1ULL << 26), "maximum N_b for validation joins")
        ("runs", po::value<size_t>(&cfg.runs)->default_value(3), "timed runs per point (median is reported)")
        ("l1", po::value<size_t>(), "override detected L1d size in bytes")
        ("l2", po::value<size_t>(), "override detected L2 size in bytes")
        ("llc", po::value<size_t>(), "override detected total LLC size in bytes")
        ("seed", po::value<UInt64>(&cfg.seed), "random seed")
        ("quick", po::bool_switch(&cfg.quick), "skip the validation joins")
        ("join-nb", po::value<size_t>()->default_value(0), "run only the real joins at this exact build-side row count (skips all kernels)")
        ("join-np", po::value<size_t>()->default_value(0), "probe-side row count for --join-nb (default: same as --join-nb)");

    po::variables_map options;
    po::store(po::parse_command_line(argc, argv, desc), options);
    po::notify(options);

    if (options.contains("help"))
    {
        fmt::print("{}\n", fmt::streamed(desc));
        return 0;
    }

    cfg.threads = options.contains("threads") ? options["threads"].as<size_t>() : getNumberOfCPUCoresToUse();
    cfg.hit_rate = std::clamp(cfg.hit_rate, 0.01, 1.0);

    CacheInfo cache = detectCaches();
    if (options.contains("l1"))
        cache.l1d = options["l1"].as<size_t>();
    if (options.contains("l2"))
        cache.l2 = options["l2"].as<size_t>();
    if (options.contains("llc"))
        cache.llc = options["llc"].as<size_t>();

    fmt::print("=== machine ===\n");
    fmt::print("  threads: {} (ClickHouse thread pool)\n", cfg.threads);
    fmt::print("  L1d: {}, L2: {}, LLC total: {}{}\n",
        formatBytes(static_cast<double>(cache.l1d)), formatBytes(static_cast<double>(cache.l2)),
        formatBytes(static_cast<double>(cache.llc)), cache.detected ? "" : " (detection failed, using defaults)");
    fmt::print("  build row width: {} B, probe row width: {} B\n", cfg.buildRowWidth(), cfg.probeRowWidth());
    fmt::print("  work rows per kernel iteration: {}\n", cfg.tuples);

#if USE_JEMALLOC
    {
        const char * jemalloc_version = nullptr;
        size_t version_size = sizeof(jemalloc_version);
        je_mallctl("version", &jemalloc_version, &version_size, nullptr, 0);
        fmt::print("  allocator: jemalloc {}\n", jemalloc_version ? jemalloc_version : "(unknown version)");
    }
#else
    fmt::print("  allocator: system malloc (jemalloc disabled in this build)\n");
#endif

    WorkerPool pool(cfg.threads);

    /// Shared immutable input blocks (the only memory reused across iterations).
    if (const size_t join_nb = options["join-nb"].as<size_t>())
    {
        const size_t join_np = options["join-np"].as<size_t>() ? options["join-np"].as<size_t>() : join_nb;
        runSingleJoin(cfg, pool, cache, join_nb, join_np);
        fmt::print("\n(check value: {})\n", g_sink.load());
        return 0;
    }

    fmt::print("\ngenerating input blocks...\n");
    auto build_work = generateBlocks(pool, cfg.tuples, cfg.build_payload_columns, "b_", uniqueKeys(), cfg.seed);
    auto probe_work = generateBlocks(pool, cfg.tuples, cfg.probe_payload_columns, "p_",
                                     probeKeys(cfg.tuples, cfg.threads, cfg.hit_rate, /*per_thread_domain=*/ false), cfg.seed + 1);

    ModelInputs model;
    model.l2 = cache.l2;
    model.llc = cache.llc;
    model.w_b = cfg.buildRowWidth();
    model.w_p = cfg.probeRowWidth();
    model.hit_rate = cfg.hit_rate;
    model.max_partitions = cfg.max_partitions;
    model.threads = cfg.threads;

    model.memcpy_bytes_per_sec = runMemcpyKernel(cfg, pool, build_work);
    fmt::print("\n=== memcpy baseline ===\n  B_cpy = {:.2f} GB/s (aggregate, block squashing via insertRangeFrom)\n",
        model.memcpy_bytes_per_sec / 1e9);

    model.scatter = runScatterKernel(cfg, pool, build_work);

    model.scatter_peak = 0;
    for (const auto & sp : model.scatter)
        model.scatter_peak = std::max(model.scatter_peak, sp.bytes_per_sec);
    model.f_max = model.scatter.front().fanout;
    for (const auto & sp : model.scatter)
        if (sp.bytes_per_sec >= 0.8 * model.scatter_peak)
            model.f_max = std::max(model.f_max, sp.fanout);

    fmt::print("  B_scatter peak = {:.2f} GB/s, F_max (>= 80% of peak) = {}\n", model.scatter_peak / 1e9, model.f_max);

    model.build_rp = runBuildKernelRP(cfg, pool);
    model.build_np = runBuildKernelNP(cfg, pool);
    model.probe_rp = runProbeKernelRP(cfg, pool);
    model.probe_np = runProbeKernelNP(cfg, pool);
    model.gather = runGatherKernel(cfg, pool, probe_work);

    const double budget = static_cast<double>(cache.l2) / 2;
    fmt::print("\n=== derived model constants (all measured with real ClickHouse join code) ===\n");
    fmt::print("  t_build (radix part): cache-resident {:.3f} ns/row, spilling {:.3f} ns/row\n",
        model.build_rp.at(budget), model.build_rp.points.back().second);
    fmt::print("  t_build (non-part):   cache-resident {:.3f} ns/row, spilling {:.3f} ns/row\n",
        model.build_np.at(budget), model.build_np.points.back().second);
    fmt::print("  t_probe+gather (radix part): cache-resident {:.3f} ns/row, spilling {:.3f} ns/row\n",
        model.probe_rp.at(budget), model.probe_rp.points.back().second);
    fmt::print("  t_probe+gather (non-part):   cache-resident {:.3f} ns/row, spilling {:.3f} ns/row\n",
        model.probe_np.at(budget), model.probe_np.points.back().second);
    fmt::print("  t_gather (standalone, IColumn::insertFrom by RowRef): cache-resident {:.3f} ns/match, spilling {:.3f} ns/match\n",
        model.gather.at(budget), model.gather.points.back().second);
    fmt::print("  per-partition table budget C = L2/2 = {}\n", formatBytes(budget));

    printGridAndCrossover(model);

    if (!cfg.quick)
        runValidation(cfg, pool, model);

    /// Prevent the compiler from optimizing the kernels away.
    fmt::print("\n(check value: {})\n", g_sink.load());
    return 0;
}
