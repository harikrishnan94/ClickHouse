/// Microbenchmark: UnifiedHashJoin lookup-driver comparison (multi-threaded).
///
/// Compares probe shapes that share `SequentialLookup` (the production driver) and differ only
/// in how matches are emitted:
///   - emit:     SequentialLookup + EmitSink (fused; production path for SEMI-LEFT / ANTI)
///   - twophase: SequentialLookup via probeTwoPhase (production path for split_can_pay kinds)
///
/// Natural home for P2 per-row-overhead checks when a new LookupDriver is added: plug it into
/// probeTwoPhase alongside SequentialLookup and keep emit fixed.
///
/// Run with:
///   ./unified_hash_join_probe_loop --verify
///   ./unified_hash_join_probe_loop --quick --benchmark_min_time=0.1s
///   ./unified_hash_join_probe_loop --benchmark_out=tmp/uhj_probe_loop.json --benchmark_out_format=json

#include <Columns/ColumnsNumber.h>
#include <Core/Defines.h>
#include <Interpreters/RowRefs.h>
#include <Interpreters/UnifiedHashJoin/HashJoin.h>
#include <Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h>
#include <Interpreters/UnifiedHashJoin/JoinFeatures.h>
#include <Interpreters/UnifiedHashJoin/JoinUsedFlags.h>
#include <Interpreters/UnifiedHashJoin/KeyGetter.h>
#include <Interpreters/UnifiedHashJoin/KnownRowsHolder.h>
#include <Interpreters/UnifiedHashJoin/ProbeLookup.h>
#include <base/defines.h>
#include <base/types.h>
#include <Common/Arena.h>
#include <Common/Exception.h>
#include <Common/PODArray.h>
#include <Common/SipHash.h>
#include <Common/Stopwatch.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(OS_LINUX)
#include <pthread.h>
#include <sched.h>
#endif

using namespace DB;
using namespace DB::Unified;

namespace
{

/// ─── CLI / config ──────────────────────────────────────────────────────────

enum class MapMode
{
    TwoLevel,
    Serial,
    Auto,
};

enum class PrefetchMode
{
    Auto,
    On,
    Off,
};

enum class BuildOrder
{
    Scattered,
    Clustered,
};

/// Probe shapes sharing SequentialLookup; extend with new LookupDrivers for P2 checks.
enum class Driver
{
    SequentialEmit,
    SequentialTwoPhase,
};

enum class Shape
{
    Inner,
    Left,
};

struct Config
{
    std::vector<size_t> cards = {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000};
    std::vector<size_t> build_mults = {1, 2, 4};
    std::vector<size_t> probe_mults = {1, 2, 4};
    std::vector<size_t> threads = {1, 8, 32, 96};
    size_t block_size = DEFAULT_BLOCK_SIZE;
    size_t batch_size = PROBE_BATCH_ROWS;
    std::vector<Shape> shapes = {Shape::Inner, Shape::Left};
    std::vector<Driver> drivers = {Driver::SequentialEmit, Driver::SequentialTwoPhase};
    MapMode map_mode = MapMode::TwoLevel;
    PrefetchMode prefetch = PrefetchMode::Auto;
    bool need_filter = false;
    double match_rate = 1.0;
    BuildOrder build_order = BuildOrder::Scattered;
    bool reserve_exact = true;
    size_t min_blocks_per_thread = 4;
    bool pin = true;
    UInt64 seed = 42;
    bool verify = false;
    bool quick = false;
};

Config g_config;

std::vector<size_t> parseSizeList(std::string_view s)
{
    std::vector<size_t> out;
    size_t start = 0;
    while (start < s.size())
    {
        size_t end = s.find(',', start);
        if (end == std::string_view::npos)
            end = s.size();
        std::string_view tok = s.substr(start, end - start);
        size_t v = 0;
        auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), v);
        if (ec != std::errc{} || ptr != tok.data() + tok.size())
            throw std::runtime_error("bad size list token: " + std::string(tok));
        out.push_back(v);
        start = end + 1;
    }
    return out;
}

bool consumeFlag(int & argc, char ** argv, const char * name, std::string & value)
{
    const size_t n = std::strlen(name);
    for (int i = 1; i < argc; ++i)
    {
        std::string_view a = argv[i];
        if (a == name)
        {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string(name) + " needs a value");
            value = argv[i + 1];
            for (int j = i; j + 2 < argc; ++j)
                argv[j] = argv[j + 2];
            argc -= 2;
            return true;
        }
        if (a.size() > n + 1 && a.substr(0, n) == name && a[n] == '=')
        {
            value = std::string(a.substr(n + 1));
            for (int j = i; j + 1 < argc; ++j)
                argv[j] = argv[j + 1];
            argc -= 1;
            return true;
        }
    }
    return false;
}

bool consumeBoolFlag(int & argc, char ** argv, const char * name)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::string_view(argv[i]) == name)
        {
            for (int j = i; j + 1 < argc; ++j)
                argv[j] = argv[j + 1];
            argc -= 1;
            return true;
        }
    }
    return false;
}

void parseConfig(int & argc, char ** argv)
{
    std::string value;
    if (consumeFlag(argc, argv, "--cards", value))
        g_config.cards = parseSizeList(value);
    if (consumeFlag(argc, argv, "--build-mult", value))
        g_config.build_mults = parseSizeList(value);
    if (consumeFlag(argc, argv, "--probe-mult", value))
        g_config.probe_mults = parseSizeList(value);
    if (consumeFlag(argc, argv, "--threads", value))
        g_config.threads = parseSizeList(value);
    if (consumeFlag(argc, argv, "--block-size", value))
        g_config.block_size = parseSizeList(value).at(0);
    if (consumeFlag(argc, argv, "--batch-size", value))
        g_config.batch_size = parseSizeList(value).at(0);
    if (consumeFlag(argc, argv, "--shapes", value))
    {
        g_config.shapes.clear();
        if (value.find("inner") != std::string::npos)
            g_config.shapes.push_back(Shape::Inner);
        if (value.find("left") != std::string::npos)
            g_config.shapes.push_back(Shape::Left);
        if (g_config.shapes.empty())
            throw std::runtime_error("--shapes must include inner and/or left");
    }
    if (consumeFlag(argc, argv, "--drivers", value) || consumeFlag(argc, argv, "--variants", value))
    {
        g_config.drivers.clear();
        if (value.find("emit") != std::string::npos)
            g_config.drivers.push_back(Driver::SequentialEmit);
        if (value.find("twophase") != std::string::npos || value.find("two-phase") != std::string::npos)
            g_config.drivers.push_back(Driver::SequentialTwoPhase);
        if (g_config.drivers.empty())
            throw std::runtime_error("--drivers must include emit and/or twophase");
    }
    if (consumeFlag(argc, argv, "--map", value))
    {
        if (value == "two-level")
            g_config.map_mode = MapMode::TwoLevel;
        else if (value == "serial")
            g_config.map_mode = MapMode::Serial;
        else if (value == "auto")
            g_config.map_mode = MapMode::Auto;
        else
            throw std::runtime_error("--map must be two-level|serial|auto");
    }
    if (consumeFlag(argc, argv, "--prefetch", value))
    {
        if (value == "auto")
            g_config.prefetch = PrefetchMode::Auto;
        else if (value == "on")
            g_config.prefetch = PrefetchMode::On;
        else if (value == "off")
            g_config.prefetch = PrefetchMode::Off;
        else
            throw std::runtime_error("--prefetch must be auto|on|off");
    }
    if (consumeFlag(argc, argv, "--need-filter", value))
        g_config.need_filter = (value != "0");
    if (consumeFlag(argc, argv, "--match-rate", value))
        g_config.match_rate = std::stod(value);
    if (consumeFlag(argc, argv, "--build-order", value))
    {
        if (value == "scattered")
            g_config.build_order = BuildOrder::Scattered;
        else if (value == "clustered")
            g_config.build_order = BuildOrder::Clustered;
        else
            throw std::runtime_error("--build-order must be scattered|clustered");
    }
    if (consumeFlag(argc, argv, "--reserve", value))
        g_config.reserve_exact = (value != "none");
    if (consumeFlag(argc, argv, "--min-blocks-per-thread", value))
        g_config.min_blocks_per_thread = parseSizeList(value).at(0);
    if (consumeFlag(argc, argv, "--pin", value))
        g_config.pin = (value != "0");
    if (consumeFlag(argc, argv, "--seed", value))
        g_config.seed = parseSizeList(value).at(0);
    if (consumeBoolFlag(argc, argv, "--verify"))
        g_config.verify = true;
    if (consumeBoolFlag(argc, argv, "--quick"))
        g_config.quick = true;

    if (g_config.quick)
    {
        g_config.cards = {10'000, 100'000};
        g_config.build_mults = {1, 2};
        g_config.probe_mults = {1, 2};
        g_config.threads = {1, 8};
    }

    if (g_config.batch_size == 0)
        throw std::runtime_error("--batch-size must be > 0");
    if (g_config.block_size == 0)
        throw std::runtime_error("--block-size must be > 0");
}

const char * shapeName(Shape s)
{
    return s == Shape::Inner ? "AllInner" : "AllLeft";
}

const char * driverName(Driver d)
{
    return d == Driver::SequentialEmit ? "seq_emit" : "seq_twophase";
}

std::string formatCard(size_t c)
{
    if (c >= 1'000'000 && c % 1'000'000 == 0)
        return "1e" + std::to_string(static_cast<int>(std::log10(static_cast<double>(c))));
    if (c >= 10'000 && c % 10'000 == 0 && c < 1'000'000)
    {
        /// 1e4 / 1e5
        return "1e" + std::to_string(static_cast<int>(std::log10(static_cast<double>(c))));
    }
    return std::to_string(c);
}

bool useTwoLevelMap(size_t threads)
{
    switch (g_config.map_mode)
    {
        case MapMode::TwoLevel: return true;
        case MapMode::Serial: return false;
        case MapMode::Auto: return threads > 1;
    }
    return true;
}

/// ─── Key generation ────────────────────────────────────────────────────────

ALWAYS_INLINE UInt64 mixSeed(UInt64 x, UInt64 seed)
{
    return x ^ (seed * 0x9E3779B97F4A7C15ULL);
}

ALWAYS_INLINE UInt64 keyOf(UInt64 j, UInt64 seed)
{
    UInt64 x = mixSeed(j, seed) + 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

ALWAYS_INLINE UInt64 permute(UInt64 i, UInt64 n, UInt32 bits, UInt64 mask, UInt64 seed)
{
    const UInt64 m1 = mixSeed(0xD6E8FEB86659FD93ULL, seed) | 1ULL;
    const UInt64 m2 = mixSeed(0xA24BAED4963EE407ULL, seed) | 1ULL;
    do
    {
        i = (i ^ (i >> (bits / 2))) * m1 & mask;
        i = (i ^ (i >> (bits / 2))) * m2 & mask;
    } while (i >= n);
    return i;
}

struct PermuteParams
{
    UInt64 n = 0;
    UInt32 bits = 0;
    UInt64 mask = 0;
};

PermuteParams makePermuteParams(UInt64 n)
{
    PermuteParams p;
    p.n = n;
    if (n <= 1)
    {
        p.bits = 1;
        p.mask = 1;
        return p;
    }
    p.bits = static_cast<UInt32>(std::bit_width(n - 1));
    p.mask = (p.bits == 64) ? ~UInt64{0} : ((UInt64{1} << p.bits) - 1);
    return p;
}

ALWAYS_INLINE UInt64 buildKeyAt(UInt64 r, size_t build_mult, BuildOrder order, UInt64 seed, const PermuteParams & pp)
{
    const UInt64 idx = (order == BuildOrder::Clustered) ? (r / build_mult) : (permute(r, pp.n, pp.bits, pp.mask, seed) / build_mult);
    return keyOf(idx, seed);
}

ALWAYS_INLINE UInt64 probeKeyAt(UInt64 r, size_t cardinality, size_t probe_mult, double match_rate, UInt64 seed, const PermuteParams & pp)
{
    const UInt64 j = permute(r, pp.n, pp.bits, pp.mask, seed) / probe_mult;
    const UInt64 matched = static_cast<UInt64>(std::llround(match_rate * static_cast<double>(cardinality)));
    if (j < matched)
        return keyOf(j, seed);
    return keyOf(cardinality + j, seed);
}

/// ─── LazySink ──────────────────────────────────────────────────────────────

struct LazySink
{
    static constexpr bool isLazy() { return true; }

    LazyOutput lazy_output;
    IColumn::Filter filter;
    IColumn::Offsets matched_rows;
    IColumn::Offsets offsets_to_replicate;
    /// Required by probeTwoPhase's outputIsProbeOutcomes / consumeFusedBatch path (AllLeft).
    bool has_columns_to_add = true;

    void appendFromBlock(UInt64 ref_word, bool) { lazy_output.addRef(ref_word); }
    void appendDefaultRow() { lazy_output.addDefault(); }
    void applyLazyDefaults() { }

    void startBlock(size_t rows, bool need_filter, bool need_replication)
    {
        lazy_output.row_refs.clear();
        lazy_output.row_count = 0;
        matched_rows.clear();
        if (need_filter)
        {
            filter.resize_fill(rows, static_cast<UInt8>(0));
            matched_rows.reserve(rows);
        }
        else
        {
            filter.clear();
        }
        if (need_replication)
            offsets_to_replicate.resize_fill(rows, static_cast<UInt64>(0));
        else
            offsets_to_replicate.clear();
    }

    UInt64 digest() const
    {
        SipHash hash;
        hash.update(lazy_output.row_count);
        hash.update(reinterpret_cast<const char *>(lazy_output.row_refs.data()), lazy_output.row_refs.size() * sizeof(UInt64));
        hash.update(reinterpret_cast<const char *>(filter.data()), filter.size() * sizeof(UInt8));
        hash.update(reinterpret_cast<const char *>(matched_rows.data()), matched_rows.size() * sizeof(UInt64));
        hash.update(reinterpret_cast<const char *>(offsets_to_replicate.data()), offsets_to_replicate.size() * sizeof(UInt64));
        return hash.get64();
    }
};

/// ─── Data caches ───────────────────────────────────────────────────────────

struct BuildColumnCache
{
    size_t cardinality = 0;
    size_t build_mult = 0;
    BuildOrder order = BuildOrder::Scattered;
    ColumnUInt64::MutablePtr column;
    PermuteParams permute;
    double load_ms = 0;
};

struct ProbeColumnCache
{
    size_t cardinality = 0;
    size_t probe_mult = 0;
    double match_rate = 0;
    ColumnUInt64::MutablePtr column;
    PermuteParams permute;
    double load_ms = 0;
};

BuildColumnCache g_build_cache;
ProbeColumnCache g_probe_cache;

void pinThread(size_t t)
{
#if defined(OS_LINUX)
    if (!g_config.pin)
        return;
    const unsigned n = std::thread::hardware_concurrency();
    if (n == 0)
        return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<int>(t % n), &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)t;
#endif
}

void fillRangeParallel(size_t threads, size_t n, const std::function<void(size_t begin, size_t end, size_t tid)> & fn)
{
    if (n == 0)
        return;
    threads = std::max<size_t>(1, std::min(threads, n));
    std::vector<std::thread> workers;
    workers.reserve(threads);
    const size_t chunk = (n + threads - 1) / threads;
    for (size_t t = 0; t < threads; ++t)
    {
        const size_t begin = t * chunk;
        if (begin >= n)
            break;
        const size_t end = std::min(n, begin + chunk);
        workers.emplace_back(
            [t, begin, end, &fn]
            {
                pinThread(t);
                fn(begin, end, t);
            });
    }
    for (auto & w : workers)
        w.join();
}

const BuildColumnCache & ensureBuildColumn(size_t cardinality, size_t build_mult, size_t threads)
{
    if (g_build_cache.column && g_build_cache.cardinality == cardinality && g_build_cache.build_mult == build_mult
        && g_build_cache.order == g_config.build_order)
        return g_build_cache;

    Stopwatch sw;
    const size_t rows = cardinality * build_mult;
    auto col = ColumnUInt64::create(rows);
    auto & data = col->getData();
    const auto pp = makePermuteParams(rows);
    const UInt64 seed = g_config.seed;
    const BuildOrder order = g_config.build_order;

    fillRangeParallel(
        threads,
        rows,
        [&](size_t begin, size_t end, size_t)
        {
            for (size_t r = begin; r < end; ++r)
                data[r] = buildKeyAt(r, build_mult, order, seed, pp);
        });

    g_build_cache = BuildColumnCache{
        .cardinality = cardinality,
        .build_mult = build_mult,
        .order = order,
        .column = std::move(col),
        .permute = pp,
        .load_ms = static_cast<double>(sw.elapsedMilliseconds()),
    };
    return g_build_cache;
}

const ProbeColumnCache & ensureProbeColumn(size_t cardinality, size_t probe_mult, size_t threads)
{
    if (g_probe_cache.column && g_probe_cache.cardinality == cardinality && g_probe_cache.probe_mult == probe_mult
        && g_probe_cache.match_rate == g_config.match_rate)
        return g_probe_cache;

    Stopwatch sw;
    const size_t rows = cardinality * probe_mult;
    auto col = ColumnUInt64::create(rows);
    auto & data = col->getData();
    const auto pp = makePermuteParams(rows);
    const UInt64 seed = g_config.seed;
    const double match_rate = g_config.match_rate;

    fillRangeParallel(
        threads,
        rows,
        [&](size_t begin, size_t end, size_t)
        {
            for (size_t r = begin; r < end; ++r)
                data[r] = probeKeyAt(r, cardinality, probe_mult, match_rate, seed, pp);
        });

    g_probe_cache = ProbeColumnCache{
        .cardinality = cardinality,
        .probe_mult = probe_mult,
        .match_rate = match_rate,
        .column = std::move(col),
        .permute = pp,
        .load_ms = static_cast<double>(sw.elapsedMilliseconds()),
    };
    return g_probe_cache;
}

/// ─── Map wrappers ──────────────────────────────────────────────────────────

using TwoLevelMap = TwoLevelJoinHashMap<UInt64, RowRefList, HashCRC32<UInt64>>;
using SerialMap = JoinHashMap<UInt64, RowRefList, HashCRC32<UInt64>>;

template <typename Map>
struct MapTraits;

template <>
struct MapTraits<TwoLevelMap>
{
    static constexpr HashJoin::Type type = HashJoin::Type::two_level_key64;
};

template <>
struct MapTraits<SerialMap>
{
    static constexpr HashJoin::Type type = HashJoin::Type::key64;
};

template <typename Map>
using BuildKeyGetter = typename KeyGetterForType<MapTraits<Map>::type, Map, /*use_offset=*/false>::Type;

template <typename Map>
using ProbeKeyGetter = typename KeyGetterForType<MapTraits<Map>::type, const Map, /*use_offset=*/false>::Type;

struct BuiltMap
{
    bool two_level = true;
    size_t cardinality = 0;
    size_t build_mult = 0;
    size_t num_slots = 0;
    bool reserve_exact = true;
    std::unique_ptr<TwoLevelMap> two_level_map;
    std::unique_ptr<SerialMap> serial_map;
    std::vector<std::unique_ptr<Arena>> pools;
    double scatter_ms = 0;
    double insert_ms = 0;
    UInt64 lock_retries = 0;
    double build_ms = 0;
    double load_ms = 0;

    size_t size() const { return two_level ? two_level_map->size() : serial_map->size(); }
    size_t bytes() const { return two_level ? two_level_map->getBufferSizeInBytes() : serial_map->getBufferSizeInBytes(); }
    size_t arenaBytes() const
    {
        size_t res = 0;
        for (const auto & p : pools)
            res += p->allocatedBytes();
        return res;
    }
};

BuiltMap g_built_map;

template <typename Map>
void insertSlotRows(
    Map & map,
    BuildKeyGetter<Map> & key_getter,
    Arena & pool,
    const PaddedPODArray<UInt32> & local_rows,
    UInt32 block_no,
    size_t block_begin)
{
    for (UInt32 local : local_rows)
    {
        const size_t key_row = block_begin + local;
        auto emplace_result = key_getter.emplaceKey(map, key_row, pool);
        if (emplace_result.isInserted())
            new (&emplace_result.getMapped()) RowRefList(block_no, local);
        else
            emplace_result.getMapped().insert(RowRef(block_no, local).encode(), pool);
    }
}

template <typename Map>
BuiltMap buildMapImpl(size_t cardinality, size_t build_mult, size_t threads)
{
    const auto & build = ensureBuildColumn(cardinality, build_mult, threads);
    const size_t rows = build.column->size();
    const size_t num_slots = slotCountForThreads(threads);
    const size_t block_size = g_config.block_size;
    const size_t num_blocks = (rows + block_size - 1) / block_size;

    BuiltMap out;
    out.two_level = std::is_same_v<Map, TwoLevelMap>;
    out.cardinality = cardinality;
    out.build_mult = build_mult;
    out.num_slots = num_slots;
    out.reserve_exact = g_config.reserve_exact;
    out.load_ms = build.load_ms;
    out.pools.resize(num_slots);
    for (size_t s = 0; s < num_slots; ++s)
        out.pools[s] = std::make_unique<Arena>();

    auto map = std::make_unique<Map>();
    if (g_config.reserve_exact)
        map->reserve(cardinality);

    std::vector<BucketLock> locks(num_slots);
    std::atomic<size_t> next_block{0};
    std::atomic<UInt64> lock_retries{0};
    std::atomic<UInt64> scatter_ns{0};
    std::atomic<UInt64> insert_ns{0};

    ColumnRawPtrs key_columns{build.column.get()};
    const Sizes key_sizes{sizeof(UInt64)};

    Stopwatch total;
    fillRangeParallel(
        threads,
        threads, /// one worker per thread; work is taken from next_block
        [&](size_t /*begin*/, size_t /*end*/, size_t tid)
        {
            pinThread(tid);
            BuildKeyGetter<Map> key_getter(key_columns, key_sizes, nullptr);
            std::vector<PaddedPODArray<UInt32>> per_slot(num_slots);
            std::vector<char> pending(num_slots, 0);

            while (true)
            {
                const size_t block_no = next_block.fetch_add(1, std::memory_order_relaxed);
                if (block_no >= num_blocks)
                    break;

                const size_t block_begin = block_no * block_size;
                const size_t block_rows = std::min(block_size, rows - block_begin);

                Stopwatch scatter_sw;
                for (size_t s = 0; s < num_slots; ++s)
                {
                    per_slot[s].clear();
                    pending[s] = 0;
                }

                const auto & data = build.column->getData();
                for (size_t i = 0; i < block_rows; ++i)
                {
                    const UInt64 key = data[block_begin + i];
                    const size_t hash_value = map->hash(key);
                    const size_t bucket = Map::getBucketFromHash(map->bucketRoutingHash(key, hash_value));
                    const size_t slot = slotForBucket(bucket, num_slots);
                    per_slot[slot].push_back(static_cast<UInt32>(i));
                }

                size_t left = 0;
                for (size_t s = 0; s < num_slots; ++s)
                {
                    if (!per_slot[s].empty())
                    {
                        pending[s] = 1;
                        ++left;
                    }
                }
                scatter_ns.fetch_add(scatter_sw.elapsedNanoseconds(), std::memory_order_relaxed);

                Stopwatch insert_sw;
                UInt64 local_retries = 0;
                const size_t first = block_no & (num_slots - 1);
                while (left)
                {
                    bool progress = false;
                    for (size_t k = 0; k < num_slots; ++k)
                    {
                        const size_t slot = (first + k) & (num_slots - 1);
                        if (!pending[slot])
                            continue;
                        std::unique_lock lock(locks[slot].mutex, std::try_to_lock);
                        if (!lock.owns_lock())
                        {
                            ++local_retries;
                            continue;
                        }
                        insertSlotRows(*map, key_getter, *out.pools[slot], per_slot[slot], static_cast<UInt32>(block_no), block_begin);
                        pending[slot] = 0;
                        --left;
                        progress = true;
                    }
                    if (!progress)
                        std::this_thread::yield();
                }
                lock_retries.fetch_add(local_retries, std::memory_order_relaxed);
                insert_ns.fetch_add(insert_sw.elapsedNanoseconds(), std::memory_order_relaxed);
            }
        });

    map->computeBucketPrefix();
    out.build_ms = static_cast<double>(total.elapsedMilliseconds());
    out.scatter_ms = static_cast<double>(scatter_ns.load()) / 1e6;
    out.insert_ms = static_cast<double>(insert_ns.load()) / 1e6;
    out.lock_retries = lock_retries.load();

    if constexpr (std::is_same_v<Map, TwoLevelMap>)
        out.two_level_map = std::move(map);
    else
        out.serial_map = std::move(map);
    return out;
}

const BuiltMap & ensureBuiltMap(size_t cardinality, size_t build_mult, size_t threads)
{
    const size_t num_slots = slotCountForThreads(threads);
    const bool two_level = useTwoLevelMap(threads);
    if (g_built_map.cardinality == cardinality && g_built_map.build_mult == build_mult && g_built_map.num_slots == num_slots
        && g_built_map.two_level == two_level && g_built_map.reserve_exact == g_config.reserve_exact
        && (two_level ? g_built_map.two_level_map != nullptr : g_built_map.serial_map != nullptr))
        return g_built_map;

    if (two_level)
        g_built_map = buildMapImpl<TwoLevelMap>(cardinality, build_mult, threads);
    else
        g_built_map = buildMapImpl<SerialMap>(cardinality, build_mult, threads);
    return g_built_map;
}

/// ─── Probe loops (production SequentialLookup drivers) ─────────────────────

bool resolveUsePrefetch(PrefetchMode mode, size_t map_bytes)
{
    switch (mode)
    {
        case PrefetchMode::On: return true;
        case PrefetchMode::Off: return false;
        case PrefetchMode::Auto: return map_bytes > getMinBytesForPrefetchInJoin();
    }
    return false;
}

/// Contiguous row range into the key column: selectorIndexAt returns first + k.
using RowRangeSelector = std::pair<size_t, size_t>;

template <JoinKind KIND, JoinStrictness STRICTNESS, bool need_filter, typename Map, typename KeyGetter>
void probeSequentialEmit(
    const Map & map,
    KeyGetter & key_getter,
    Arena & pool,
    LazySink & sink,
    JoinStuff::JoinUsedFlags & used_flags,
    size_t begin,
    size_t rows,
    bool use_prefetch)
{
    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;

    IColumn::Offset current_offset = 0;
    const RowRangeSelector selector{begin, 0};

    auto prefetcher = makeJoinPrefetcher(
        use_prefetch && can_prefetch,
        rows,
        [&](size_t k) __attribute__((always_inline))
        {
            if constexpr (can_prefetch)
                map.prefetch(key_getter.getKeyHolder(selectorIndexAt(selector, k), pool));
        });

    auto prefetch_at = [&](size_t k) __attribute__((always_inline))
    {
        if constexpr (can_prefetch)
            prefetcher.prefetchAt(k);
    };

    EmitSink<KIND, STRICTNESS, need_filter, HashJoin::MapsAll, LazySink> emit_sink{sink, used_flags, current_offset};
    SequentialLookup::run(key_getter, map, selector, /*skip_data=*/nullptr, pool, /*begin=*/0, rows, prefetch_at, emit_sink);
    sink.applyLazyDefaults();
}

template <JoinKind KIND, JoinStrictness STRICTNESS, bool need_filter, typename Map, typename KeyGetter>
void probeSequentialTwoPhase(
    const Map & map,
    KeyGetter & key_getter,
    Arena & pool,
    LazySink & sink,
    JoinStuff::JoinUsedFlags & used_flags,
    size_t begin,
    size_t rows,
    size_t batch_size,
    bool use_prefetch)
{
    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;

    IColumn::Offset current_offset = 0;
    const RowRangeSelector selector{begin, 0};
    ProbeOutcomes outcomes;

    auto prefetcher = makeJoinPrefetcher(
        use_prefetch && can_prefetch,
        rows,
        [&](size_t k) __attribute__((always_inline))
        {
            if constexpr (can_prefetch)
                map.prefetch(key_getter.getKeyHolder(selectorIndexAt(selector, k), pool));
        });

    auto prefetch_at = [&](size_t k) __attribute__((always_inline))
    {
        if constexpr (can_prefetch)
            prefetcher.prefetchAt(k);
    };

    probeTwoPhase<KIND, STRICTNESS, SequentialLookup, need_filter, HashJoin::MapsAll, Map, KeyGetter>(
        key_getter, map, sink, used_flags, selector, /*skip_data=*/nullptr, pool, prefetch_at, outcomes, rows, batch_size, current_offset);
    sink.applyLazyDefaults();
}

template <JoinKind KIND, JoinStrictness STRICTNESS, bool need_filter, typename Map, typename KeyGetter>
void probeWithDriver(
    Driver driver,
    const Map & map,
    KeyGetter & key_getter,
    Arena & pool,
    LazySink & sink,
    JoinStuff::JoinUsedFlags & used_flags,
    size_t begin,
    size_t rows,
    size_t batch_size,
    bool use_prefetch)
{
    if (driver == Driver::SequentialEmit)
        probeSequentialEmit<KIND, STRICTNESS, need_filter>(map, key_getter, pool, sink, used_flags, begin, rows, use_prefetch);
    else
        probeSequentialTwoPhase<KIND, STRICTNESS, need_filter>(
            map, key_getter, pool, sink, used_flags, begin, rows, batch_size, use_prefetch);
}

/// ─── Probe driver ──────────────────────────────────────────────────────────

struct ProbeStats
{
    UInt64 probe_rows = 0;
    UInt64 out_rows = 0;
};

struct ProbeParams
{
    size_t cardinality = 0;
    size_t build_mult = 0;
    size_t probe_mult = 0;
    size_t threads = 0;
    Shape shape = Shape::Inner;
    Driver driver = Driver::SequentialEmit;
    bool need_filter = false;
};

/// Persistent worker pool for Google Benchmark iterations.

struct ProbeSession
{
    size_t threads = 0;
    size_t blocks_per_pass = 0;
    size_t natural_blocks = 0;
    size_t probe_rows = 0;
    size_t batch_size = 0;
    size_t block_size = 0;
    bool use_prefetch = false;
    bool need_filter = false;
    Shape shape = Shape::Inner;
    Driver driver = Driver::SequentialEmit;
    bool two_level = true;

    std::unique_ptr<std::barrier<>> sync_start;
    std::unique_ptr<std::barrier<>> sync_end;
    std::atomic<bool> stop{false};
    std::atomic<size_t> next_block{0};
    std::vector<std::thread> workers;
    std::vector<ProbeStats> local_stats;

    const BuiltMap * built = nullptr;
    const ProbeColumnCache * probe = nullptr;

    template <JoinKind KIND, bool need_filter_v, typename Map>
    void startWorkers();

    void stopWorkers()
    {
        if (workers.empty())
            return;
        stop.store(true, std::memory_order_relaxed);
        sync_start->arrive_and_wait();
        for (auto & w : workers)
            w.join();
        workers.clear();
    }

    ~ProbeSession() { stopWorkers(); }

    ProbeStats runIteration()
    {
        next_block.store(0, std::memory_order_relaxed);
        for (auto & s : local_stats)
            s = {};
        sync_start->arrive_and_wait();
        sync_end->arrive_and_wait();
        ProbeStats total;
        for (const auto & s : local_stats)
        {
            total.probe_rows += s.probe_rows;
            total.out_rows += s.out_rows;
        }
        return total;
    }
};

template <JoinKind KIND, bool need_filter_v, typename Map>
void ProbeSession::startWorkers()
{
    constexpr JoinStrictness STRICTNESS = JoinStrictness::All;
    using Features = JoinFeatures<KIND, STRICTNESS, HashJoin::MapsAll>;

    const Map & map = [&]() -> const Map &
    {
        if constexpr (std::is_same_v<Map, TwoLevelMap>)
            return *built->two_level_map;
        else
            return *built->serial_map;
    }();

    ColumnRawPtrs key_columns{probe->column.get()};
    const Sizes key_sizes{sizeof(UInt64)};

    sync_start = std::make_unique<std::barrier<>>(static_cast<std::ptrdiff_t>(threads + 1));
    sync_end = std::make_unique<std::barrier<>>(static_cast<std::ptrdiff_t>(threads + 1));
    stop.store(false, std::memory_order_relaxed);
    local_stats.assign(threads, {});
    workers.clear();
    workers.reserve(threads);

    for (size_t t = 0; t < threads; ++t)
    {
        workers.emplace_back(
            [this, t, &map, key_columns, key_sizes]() mutable
            {
                pinThread(t);
                Arena pool;
                ProbeKeyGetter<Map> key_getter(key_columns, key_sizes, nullptr);
                LazySink sink;
                JoinStuff::JoinUsedFlags used_flags;

                while (true)
                {
                    sync_start->arrive_and_wait();
                    if (stop.load(std::memory_order_relaxed))
                        return;

                    ProbeStats stats;
                    for (size_t b = next_block.fetch_add(1, std::memory_order_relaxed); b < blocks_per_pass;
                         b = next_block.fetch_add(1, std::memory_order_relaxed))
                    {
                        const size_t src_block = b % natural_blocks;
                        const size_t begin = src_block * block_size;
                        const size_t rows = std::min(block_size, probe_rows - begin);

                        sink.startBlock(rows, need_filter_v || Features::need_filter, Features::need_replication);

                        probeWithDriver<KIND, STRICTNESS, need_filter_v>(
                            driver, map, key_getter, pool, sink, used_flags, begin, rows, batch_size, use_prefetch);

                        benchmark::DoNotOptimize(sink.lazy_output.row_refs.data());
                        benchmark::DoNotOptimize(sink.offsets_to_replicate.data());

                        stats.probe_rows += rows;
                        stats.out_rows += sink.lazy_output.row_count;
                    }
                    local_stats[t] = stats;
                    sync_end->arrive_and_wait();
                }
            });
    }
}

void configureSession(ProbeSession & session, const ProbeParams & p, const BuiltMap & built, const ProbeColumnCache & probe)
{
    session.stopWorkers();
    session.built = &built;
    session.probe = &probe;
    session.threads = p.threads;
    session.probe_rows = probe.column->size();
    session.block_size = g_config.block_size;
    session.batch_size = g_config.batch_size;
    session.natural_blocks = std::max<size_t>(1, (session.probe_rows + session.block_size - 1) / session.block_size);
    session.blocks_per_pass = std::max(session.natural_blocks, p.threads * g_config.min_blocks_per_thread);
    session.use_prefetch = resolveUsePrefetch(g_config.prefetch, built.bytes());
    session.need_filter = p.need_filter;
    session.shape = p.shape;
    session.driver = p.driver;
    session.two_level = built.two_level;

    if (built.two_level)
    {
        if (p.shape == Shape::Inner)
        {
            if (p.need_filter)
                session.startWorkers<JoinKind::Inner, true, TwoLevelMap>();
            else
                session.startWorkers<JoinKind::Inner, false, TwoLevelMap>();
        }
        else
        {
            if (p.need_filter)
                session.startWorkers<JoinKind::Left, true, TwoLevelMap>();
            else
                session.startWorkers<JoinKind::Left, false, TwoLevelMap>();
        }
    }
    else
    {
        if (p.shape == Shape::Inner)
        {
            if (p.need_filter)
                session.startWorkers<JoinKind::Inner, true, SerialMap>();
            else
                session.startWorkers<JoinKind::Inner, false, SerialMap>();
        }
        else
        {
            if (p.need_filter)
                session.startWorkers<JoinKind::Left, true, SerialMap>();
            else
                session.startWorkers<JoinKind::Left, false, SerialMap>();
        }
    }
}

/// ─── Verification ──────────────────────────────────────────────────────────

template <JoinKind KIND, bool need_filter, typename Map>
std::pair<std::vector<UInt64>, UInt64>
digestsDriver(const BuiltMap & built, const ProbeColumnCache & probe, Driver driver, bool include_refs)
{
    constexpr JoinStrictness STRICTNESS = JoinStrictness::All;
    constexpr JoinFeatures<KIND, STRICTNESS, HashJoin::MapsAll> features;

    const Map & map = [&]() -> const Map &
    {
        if constexpr (std::is_same_v<Map, TwoLevelMap>)
            return *built.two_level_map;
        else
            return *built.serial_map;
    }();

    const size_t probe_rows = probe.column->size();
    const size_t block_size = g_config.block_size;
    const size_t natural_blocks = (probe_rows + block_size - 1) / block_size;
    const size_t batch_size = g_config.batch_size;
    const bool use_prefetch = resolveUsePrefetch(g_config.prefetch, built.bytes());

    ColumnRawPtrs key_columns{probe.column.get()};
    const Sizes key_sizes{sizeof(UInt64)};

    Arena pool;
    ProbeKeyGetter<Map> key_getter(key_columns, key_sizes, nullptr);
    LazySink sink;
    JoinStuff::JoinUsedFlags used_flags;

    std::vector<UInt64> digests(natural_blocks);
    UInt64 total_out = 0;
    for (size_t b = 0; b < natural_blocks; ++b)
    {
        const size_t begin = b * block_size;
        const size_t rows = std::min(block_size, probe_rows - begin);
        sink.startBlock(rows, need_filter || features.need_filter, features.need_replication);

        probeWithDriver<KIND, STRICTNESS, need_filter>(
            driver, map, key_getter, pool, sink, used_flags, begin, rows, batch_size, use_prefetch);

        if (include_refs)
        {
            digests[b] = sink.digest();
        }
        else
        {
            /// Arena pointers inside RowRefList words differ across builds; compare structure only.
            SipHash hash;
            hash.update(sink.lazy_output.row_count);
            hash.update(reinterpret_cast<const char *>(sink.filter.data()), sink.filter.size() * sizeof(UInt8));
            hash.update(reinterpret_cast<const char *>(sink.matched_rows.data()), sink.matched_rows.size() * sizeof(UInt64));
            hash.update(
                reinterpret_cast<const char *>(sink.offsets_to_replicate.data()), sink.offsets_to_replicate.size() * sizeof(UInt64));
            /// Per output entry: row count of the cell word (inline=1, list=N), not the pointer bits.
            for (UInt64 word : sink.lazy_output.row_refs)
                hash.update(refWordRows(word));
            digests[b] = hash.get64();
        }
        total_out += sink.lazy_output.row_count;
    }
    return {std::move(digests), total_out};
}

template <JoinKind KIND, bool need_filter>
void verifyShape(size_t cardinality, size_t build_mult, size_t probe_mult)
{
    constexpr JoinFeatures<KIND, JoinStrictness::All, HashJoin::MapsAll> features;

    const auto & probe = ensureProbeColumn(cardinality, probe_mult, /*threads=*/8);
    auto built1
        = useTwoLevelMap(1) ? buildMapImpl<TwoLevelMap>(cardinality, build_mult, 1) : buildMapImpl<SerialMap>(cardinality, build_mult, 1);

    std::pair<std::vector<UInt64>, UInt64> emit;
    std::pair<std::vector<UInt64>, UInt64> twophase;
    if (built1.two_level)
    {
        emit = digestsDriver<KIND, need_filter, TwoLevelMap>(built1, probe, Driver::SequentialEmit, /*include_refs=*/true);
        twophase = digestsDriver<KIND, need_filter, TwoLevelMap>(built1, probe, Driver::SequentialTwoPhase, /*include_refs=*/true);
    }
    else
    {
        emit = digestsDriver<KIND, need_filter, SerialMap>(built1, probe, Driver::SequentialEmit, /*include_refs=*/true);
        twophase = digestsDriver<KIND, need_filter, SerialMap>(built1, probe, Driver::SequentialTwoPhase, /*include_refs=*/true);
    }

    if (emit.first != twophase.first)
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "verify failed: seq_emit != seq_twophase digests for {} card={} build={}x probe={}x",
            KIND == JoinKind::Inner ? "Inner" : "Left",
            cardinality,
            build_mult,
            probe_mult);

    if (emit.second != twophase.second)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "verify failed: seq_emit out_rows {} != seq_twophase {}", emit.second, twophase.second);

    /// Analytic expectation.
    const size_t probe_rows = cardinality * probe_mult;
    const UInt64 matched_keys = static_cast<UInt64>(std::llround(g_config.match_rate * static_cast<double>(cardinality)));
    const UInt64 matches = matched_keys * probe_mult;
    const UInt64 misses = probe_rows - matches;
    UInt64 expected = matches * build_mult;
    if constexpr (features.add_missing)
        expected += misses;
    if (emit.second != expected)
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "verify failed: out_rows {} != expected {} (matches={} misses={} build_mult={})",
            emit.second,
            expected,
            matches,
            misses,
            build_mult);

    /// Multi-thread build: arena pointers differ, so compare structural digests only.
    const size_t thr = std::min<size_t>(8, std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 8);
    if (thr > 1 && useTwoLevelMap(thr))
    {
        auto built_n = buildMapImpl<TwoLevelMap>(cardinality, build_mult, thr);
        auto emit_1 = digestsDriver<KIND, need_filter, TwoLevelMap>(built1, probe, Driver::SequentialEmit, /*include_refs=*/false);
        auto emit_n = digestsDriver<KIND, need_filter, TwoLevelMap>(built_n, probe, Driver::SequentialEmit, /*include_refs=*/false);
        if (emit_n.first != emit_1.first || emit_n.second != emit_1.second)
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "verify failed: multi-thread build structural digests differ from single-thread for card={}",
                cardinality);
    }

    std::fprintf(
        stderr,
        "verify OK: %s card=%zu build=%zux probe=%zux out_rows=%llu\n",
        KIND == JoinKind::Inner ? "AllInner" : "AllLeft",
        cardinality,
        build_mult,
        probe_mult,
        static_cast<unsigned long long>(emit.second));
}

void runVerify()
{
    const std::vector<size_t> cards = g_config.quick ? std::vector<size_t>{10'000} : std::vector<size_t>{10'000, 100'000};
    for (size_t card : cards)
    {
        for (size_t bm : {size_t{1}, size_t{2}})
        {
            for (size_t pm : {size_t{1}, size_t{2}})
            {
                for (Shape shape : g_config.shapes)
                {
                    if (shape == Shape::Inner)
                    {
                        if (g_config.need_filter)
                            verifyShape<JoinKind::Inner, true>(card, bm, pm);
                        else
                            verifyShape<JoinKind::Inner, false>(card, bm, pm);
                    }
                    else
                    {
                        if (g_config.need_filter)
                            verifyShape<JoinKind::Left, true>(card, bm, pm);
                        else
                            verifyShape<JoinKind::Left, false>(card, bm, pm);
                    }
                }
            }
        }
    }
}

/// ─── Benchmark registration ────────────────────────────────────────────────

void BM_Build(benchmark::State & state, size_t cardinality, size_t build_mult, size_t threads)
{
    BuiltMap last;
    for (auto _ : state)
    {
        state.PauseTiming();
        /// Drop cache so this measures a real build.
        g_built_map = BuiltMap{};
        state.ResumeTiming();
        if (useTwoLevelMap(threads))
            last = buildMapImpl<TwoLevelMap>(cardinality, build_mult, threads);
        else
            last = buildMapImpl<SerialMap>(cardinality, build_mult, threads);
        benchmark::DoNotOptimize(last.size());
    }
    /// Keep the last build for subsequent probe benches of the same key.
    g_built_map = std::move(last);

    const size_t rows = cardinality * build_mult;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * rows));
    state.counters["ns/row"]
        = benchmark::Counter(static_cast<double>(state.iterations() * rows), benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
    state.counters["keys"] = static_cast<double>(g_built_map.size());
    state.counters["map_MiB"] = static_cast<double>(g_built_map.bytes()) / (1024.0 * 1024.0);
    state.counters["arena_MiB"] = static_cast<double>(g_built_map.arenaBytes()) / (1024.0 * 1024.0);
    state.counters["scatter_ms"] = g_built_map.scatter_ms;
    state.counters["insert_ms"] = g_built_map.insert_ms;
    state.counters["lock_retries"] = static_cast<double>(g_built_map.lock_retries);
    state.counters["load_ms"] = g_built_map.load_ms;
    state.counters["threads"] = static_cast<double>(threads);
    state.counters["slots"] = static_cast<double>(g_built_map.num_slots);
}

void BM_Probe(benchmark::State & state, ProbeParams params)
{
    const auto & built = ensureBuiltMap(params.cardinality, params.build_mult, params.threads);
    const auto & probe = ensureProbeColumn(params.cardinality, params.probe_mult, params.threads);

    ProbeSession session;
    configureSession(session, params, built, probe);

    ProbeStats last;
    for (auto _ : state)
    {
        last = session.runIteration();
        benchmark::DoNotOptimize(last.out_rows);
    }

    state.SetItemsProcessed(static_cast<int64_t>(last.probe_rows * state.iterations()));
    state.counters["ns/row"] = benchmark::Counter(
        static_cast<double>(last.probe_rows * state.iterations()), benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
    /// Per-thread-normalized: wall ns/row * threads ≈ CPU ns/row if perfectly scaled.
    state.counters["ns/row/thr"] = benchmark::Counter(
        static_cast<double>(last.probe_rows * state.iterations()) / static_cast<double>(params.threads),
        benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
    state.counters["out_rows"] = static_cast<double>(last.out_rows);
    state.counters["map_MiB"] = static_cast<double>(built.bytes()) / (1024.0 * 1024.0);
    state.counters["prefetch"] = session.use_prefetch ? 1.0 : 0.0;
    state.counters["threads"] = static_cast<double>(params.threads);
    state.counters["slots"] = static_cast<double>(built.num_slots);
    state.counters["blocks"] = static_cast<double>(session.blocks_per_pass);
    state.counters["batch"] = static_cast<double>(g_config.batch_size);
    state.counters["build_ms"] = built.build_ms;
}

void registerBenchmarks()
{
    std::fprintf(
        stderr,
        "uhj_probe_loop: drivers=seq_emit|seq_twophase block_size=%zu batch_size=%zu seed=%llu map=%s prefetch=%s need_filter=%d match_rate=%.3f\n",
        g_config.block_size,
        g_config.batch_size,
        static_cast<unsigned long long>(g_config.seed),
        g_config.map_mode == MapMode::TwoLevel ? "two-level" : (g_config.map_mode == MapMode::Serial ? "serial" : "auto"),
        g_config.prefetch == PrefetchMode::Auto ? "auto" : (g_config.prefetch == PrefetchMode::On ? "on" : "off"),
        static_cast<int>(g_config.need_filter),
        g_config.match_rate);

    for (size_t card : g_config.cards)
    {
        for (size_t bm : g_config.build_mults)
        {
            for (size_t thr : g_config.threads)
            {
                const std::string build_name = "Build/card=" + formatCard(card) + "/build=" + std::to_string(bm)
                    + "x/thr=" + std::to_string(thr) + "/slots=" + std::to_string(slotCountForThreads(thr));
                benchmark::RegisterBenchmark(build_name.c_str(), [=](benchmark::State & st) { BM_Build(st, card, bm, thr); })
                    ->UseRealTime()
                    ->Iterations(1);

                for (Shape shape : g_config.shapes)
                {
                    for (Driver driver : g_config.drivers)
                    {
                        for (size_t pm : g_config.probe_mults)
                        {
                            ProbeParams params{
                                .cardinality = card,
                                .build_mult = bm,
                                .probe_mult = pm,
                                .threads = thr,
                                .shape = shape,
                                .driver = driver,
                                .need_filter = g_config.need_filter,
                            };
                            const std::string name = std::string(shapeName(shape)) + "/" + driverName(driver)
                                + "/card=" + formatCard(card) + "/build=" + std::to_string(bm) + "x/probe=" + std::to_string(pm)
                                + "x/thr=" + std::to_string(thr);
                            benchmark::RegisterBenchmark(name.c_str(), [=](benchmark::State & st) { BM_Probe(st, params); })->UseRealTime();
                        }
                    }
                }
            }
        }
    }
}

} // namespace

int main(int argc, char ** argv)
{
    try
    {
        parseConfig(argc, argv);
        if (g_config.verify)
        {
            runVerify();
            return 0;
        }
        registerBenchmarks();
        benchmark::Initialize(&argc, argv);
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        return 0;
    }
    catch (const DB::Exception & e)
    {
        std::fprintf(stderr, "Exception: %s\n", e.displayText().c_str());
        return 1;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "std::exception: %s\n", e.what());
        return 1;
    }
}
