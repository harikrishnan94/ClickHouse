#pragma once

/// Shared, gtest-free scaffolding for the RadixHashJoin benchmark executable (`bench_radix_hash_join`).
///
/// Every benchmark here drives the EXACT production path — `BuildStore::add` / `finishBuild` /
/// `scatterToLeaves` over the real `GrowingArena` / `PartitionConfig`, `buildLeafHashTables`, and
/// `collectMatches` — never a synthetic primitive harness. Modern C++23: no test macros, a `check`
/// free function instead of `EXPECT_*`/`ASSERT_*`, and a typed `Bench` registry dispatched from `main`.

#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <DataTypes/DataTypeFixedString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/RadixHashJoin/BuildStore.h>
#include <Interpreters/RadixHashJoin/LeafHashTable.h>
#include <Interpreters/RadixHashJoin/PartitionConfig.h>
#include <Interpreters/RadixHashJoin/RapidHash.h>

#include <Common/Stopwatch.h>

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace RHJBench
{

using namespace DB;
using namespace DB::RadixHash;
using RadixShuffle::BuildRef;

/// Representative per-core L2 used to derive the leaf count in `PartitionConfig::make`.
inline constexpr size_t l2_bytes = 2 * 1024 * 1024;

/// A bench fails by throwing; `main` catches and returns non-zero. No process abort, no macros.
[[noreturn]] inline void failCheck(std::string_view what, const std::source_location & loc)
{
    throw std::runtime_error(fmt::format("check failed: {} (at {}:{})", what, loc.file_name(), loc.line()));
}

inline void check(bool ok, std::string_view what, const std::source_location loc = std::source_location::current())
{
    if (!ok) [[unlikely]]
        failCheck(what, loc);
}

template <typename A, typename B>
inline void checkEq(const A & a, const B & b, std::string_view what, const std::source_location loc = std::source_location::current())
{
    if (!(a == b)) [[unlikely]]
        failCheck(fmt::format("{}: {} != {}", what, a, b), loc);
}

/// One selectable benchmark. `run` receives the positional CLI args that follow the bench name (so each
/// bench is configured on the command line, not via environment variables). It throws on failure (via
/// `check`).
struct Bench
{
    std::string_view name;
    std::string_view help;
    std::function<void(std::span<char * const>)> run;
};

/// Wrap a parameter-less bench function as a `Bench::run` that ignores any CLI args.
inline std::function<void(std::span<char * const>)> noArgs(void (*fn)())
{
    return [fn](std::span<char * const>) { fn(); };
}

/// Positional CLI arg `i` as a string view, or `dflt` if absent.
inline std::string_view argOr(std::span<char * const> args, size_t i, std::string_view dflt)
{
    return i < args.size() ? std::string_view(args[i]) : dflt;
}

/// Positional CLI arg `i` parsed as size_t, or `dflt` if absent.
inline size_t argSize(std::span<char * const> args, size_t i, size_t dflt)
{
    return i < args.size() ? static_cast<size_t>(std::strtoull(args[i], nullptr, 10)) : dflt;
}

/// Implemented in bench_radix_hash_scatter.cpp; concatenated into the registry by `main`.
std::span<const Bench> scatterBenches();

/// Dispatch `argv[1]` against `benches`, forwarding `argv[2..]` to the bench; no arg / `--list` prints the
/// registry. Returns the process code.
inline int runBenchMain(std::span<char * const> args, std::span<const Bench> benches)
{
    const auto print_list = [&]
    {
        fmt::print("usage: {} <bench> [args...]   (or --list)\n\nbenches:\n", args.empty() ? "bench_radix_hash_join" : args[0]);
        for (const auto & b : benches)
            fmt::print("  {:<26} {}\n", b.name, b.help);
    };

    if (args.size() < 2)
    {
        print_list();
        return 2;
    }
    const std::string_view want = args[1];
    if (want == "--list" || want == "-h" || want == "--help")
    {
        print_list();
        return 0;
    }

    const auto it = std::ranges::find_if(benches, [&](const Bench & b) { return b.name == want; });
    if (it == benches.end())
    {
        fmt::print(stderr, "unknown bench: {}\n", want);
        print_list();
        return 2;
    }

    try
    {
        it->run(args.subspan(2)); /// positional args after the bench name
    }
    catch (const std::exception & e)
    {
        fmt::print(stderr, "FAILED: {}\n", e.what());
        return 1;
    }
    return 0;
}

/// Drive a cooperative build with real T-thread parallelism: every thread calls `coord.run(body)`; the
/// first becomes the leader (executes body), the rest act as helpers.
inline void coopRun(CoopPool & coord, size_t threads, const std::function<void()> & body)
{
    std::vector<std::thread> th;
    th.reserve(threads);
    for (size_t t = 0; t < threads; ++t)
        th.emplace_back([&] { coord.run(body); });
    for (auto & x : th)
        x.join();
}

template <typename T>
ColumnVector<T>::MutablePtr makeColumn(const std::vector<T> & vals)
{
    auto col = ColumnVector<T>::create();
    auto & data = col->getData();
    data.resize(vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
        data[i] = vals[i];
    return col;
}

inline Block makeU64Block(const std::vector<UInt64> & keys)
{
    ColumnsWithTypeAndName cols;
    cols.emplace_back(makeColumn<UInt64>(keys), std::make_shared<DataTypeUInt64>(), "k0");
    return Block(std::move(cols));
}

/// A block of `keys.size()` fixed-width key columns ("k0"..) plus `num_payload` UInt64 payload columns.
/// The payload is never scattered (zero-copy gate); it only exercises COW-move on `add`.
template <typename Key>
Block makeBlock(const std::vector<std::vector<Key>> & keys, size_t num_payload, UInt64 payload_seed)
{
    const size_t rows = keys.empty() ? 0 : keys[0].size();
    ColumnsWithTypeAndName cols;
    for (size_t c = 0; c < keys.size(); ++c)
        cols.emplace_back(makeColumn<Key>(keys[c]), std::make_shared<DataTypeNumber<Key>>(), fmt::format("k{}", c));
    for (size_t c = 0; c < num_payload; ++c)
    {
        std::vector<UInt64> pv(rows);
        for (size_t i = 0; i < rows; ++i)
            pv[i] = payload_seed * 1000003ull + c * 7919ull + i;
        cols.emplace_back(makeColumn<UInt64>(pv), std::make_shared<DataTypeUInt64>(), fmt::format("p{}", c + 1));
    }
    return Block(std::move(cols));
}

/// Single-key-column block helper.
template <typename Key>
Block makeBlock1(const std::vector<Key> & keys, size_t num_payload, UInt64 payload_seed)
{
    return makeBlock<Key>(std::vector<std::vector<Key>>{keys}, num_payload, payload_seed);
}

/// Full 64-bit RapidHash per row for a single UInt64 key column, computed exactly like the probe
/// selector (leaf from the top bits, bucket from the low bits).
inline std::vector<UInt64> computeHashes(const Block & block, size_t n)
{
    const char * raw = block.getByPosition(0).column->getRawData().data();
    std::vector<UInt64> hash(n);
    for (size_t i = 0; i < n; ++i)
        hash[i] = rapidHashKey(raw + i * sizeof(UInt64), sizeof(UInt64));
    return hash;
}

inline std::vector<UInt64> randomKeys(size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    std::vector<UInt64> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = rng();
    return v;
}

/// Split `keys` into `num_blocks` contiguous single-UInt64-key-column blocks and feed them serially.
inline void addBlocksSerial(BuildStore & store, const std::vector<UInt64> & keys, size_t num_blocks)
{
    const size_t n = keys.size();
    const size_t per = (n + num_blocks - 1) / num_blocks;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t lo = b * per;
        if (lo >= n)
            break;
        const size_t hi = std::min(n, lo + per);
        std::vector<UInt64> slice(keys.begin() + lo, keys.begin() + hi);
        store.add(makeU64Block(slice));
    }
}

/// Key column positions {0, 1, ..., k-1} for a k-column key.
inline std::vector<size_t> keyPositions(size_t k)
{
    std::vector<size_t> pos(k);
    for (size_t i = 0; i < k; ++i)
        pos[i] = i;
    return pos;
}

/// One FixedString(width) column of `rows` rows filled with deterministic pseudo-random bytes (bulk
/// 8-byte fill; setup only, never timed).
inline ColumnFixedString::MutablePtr makeRandomFixedString(size_t width, size_t rows, std::mt19937_64 & rng)
{
    auto col = ColumnFixedString::create(width);
    col->resize(rows);
    auto & chars = col->getChars();
    const size_t total = chars.size(); /// == width * rows
    size_t i = 0;
    for (; i + 8 <= total; i += 8)
    {
        const UInt64 v = rng();
        std::memcpy(chars.data() + i, &v, 8);
    }
    if (i < total)
    {
        const UInt64 v = rng();
        std::memcpy(chars.data() + i, &v, total - i);
    }
    return col;
}

/// A build block of FixedString key columns named k0.. of the given byte widths (no payload).
inline Block makeFixedStringBlock(const std::vector<size_t> & widths, size_t rows, uint64_t seed)
{
    std::mt19937_64 rng(seed); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    ColumnsWithTypeAndName cols;
    cols.reserve(widths.size());
    for (size_t c = 0; c < widths.size(); ++c)
        cols.emplace_back(
            makeRandomFixedString(widths[c], rows, rng),
            std::make_shared<DataTypeFixedString>(widths[c]),
            fmt::format("k{}", c));
    return Block(std::move(cols));
}

/// Generate `ceil(n / block_rows)` FixedString-key blocks in parallel across `num_threads` workers
/// (work-stealing over block indices). Setup helper — NOT timed by callers.
inline std::vector<Block> generateFixedStringBlocksParallel(
    const std::vector<size_t> & widths, size_t n, size_t block_rows, size_t num_threads, uint64_t seed_base)
{
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks(num_blocks);
    std::atomic<size_t> next{0};
    std::vector<std::thread> gen;
    gen.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t)
        gen.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < num_blocks; b = next.fetch_add(1))
            {
                const size_t rows = std::min(block_rows, n - b * block_rows);
                blocks[b] = makeFixedStringBlock(widths, rows, seed_base + b);
            }
        });
    for (auto & th : gen)
        th.join();
    return blocks;
}

/// Workload key generators for the probe bench.
///   U — all-unique (100% singletons): probe hits the `has_chain == false` path.
///   M — mixed (~90% singleton keys + ~10% rows over a small domain at ~16 rows/key): the TARGET,
///       `has_chain == true` with the singleton branch HOT (most distinct keys are singletons).
///   D — heavy-dup (few distinct keys, long chains): `has_chain == true`, singleton branch never taken.
/// The seed is fixed so the build + scatter setup is byte-identical run-to-run.
inline std::vector<UInt64> makeWorkloadKeys(std::string_view workload, size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    std::vector<UInt64> keys(n);
    if (workload == "U")
    {
        for (size_t i = 0; i < n; ++i)
            keys[i] = rng(); /// draws from 2^64 — collisions negligible (matches randomKeys usage)
    }
    else if (workload == "D")
    {
        const size_t distinct = 1000;
        std::vector<UInt64> domain(distinct);
        for (auto & d : domain)
            d = rng();
        for (size_t i = 0; i < n; ++i)
            keys[i] = domain[rng() % distinct];
    }
    else /// "M"
    {
        const size_t dup_rows = n / 10;
        const size_t singleton_rows = n - dup_rows;
        const size_t dup_distinct = std::max<size_t>(1, dup_rows / 16); /// ~16 rows per duplicated key
        std::vector<UInt64> domain(dup_distinct);
        for (auto & d : domain)
            d = rng();
        for (size_t i = 0; i < singleton_rows; ++i)
            keys[i] = rng();
        for (size_t i = singleton_rows; i < n; ++i)
            keys[i] = domain[rng() % dup_distinct];
        std::shuffle(keys.begin(), keys.end(), rng); /// mix singletons and duplicates across build blocks
    }
    return keys;
}

}
