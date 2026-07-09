#include <Interpreters/RadixHashJoin/ColumnScatter.h>

#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnsNumber.h>
#include <Common/CurrentMetrics.h>
#include <Common/Exception.h>
#include <Common/ThreadPool.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace CurrentMetrics
{
    extern const Metric LocalThread;
    extern const Metric LocalThreadActive;
    extern const Metric LocalThreadScheduled;
}

namespace DB::ErrorCodes
{
    extern const int QUERY_WAS_CANCELLED;
}

using namespace DB;
using namespace DB::RadixJoin;

namespace
{

/// Deterministic 64-bit scramble (murmur finalizer): the tests derive every route word and
/// every payload byte from (row, column) through this, so a side can be regenerated instead of
/// copied (the scatter consumes its input).
UInt64 mix64(UInt64 x)
{
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33;
    x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 33;
    return x;
}

/// Caller-computed route word per row (the kernels are hash-agnostic; multiply-shift scrambling
/// like the benchmark uses on non-CRC hosts).
UInt32 routeOfRow(size_t row, UInt64 seed)
{
    return static_cast<UInt32>(mix64(seed ^ (row * 0x9E3779B97F4A7C15ULL)) >> 32);
}

size_t partitionOfRow(size_t row, UInt64 seed, size_t bits)
{
    return routeOfRow(row, seed) >> (32 - bits);
}

/// The value bytes of one (row, column) cell.
void fillValue(char * dst, size_t width, size_t row, size_t col, UInt64 seed)
{
    for (size_t b = 0; b < width; b += 8)
    {
        const UInt64 v = mix64(seed ^ (row * 0x100000001B3ULL) ^ (static_cast<UInt64>(col) << 32) ^ b);
        memcpy(dst + b, &v, std::min<size_t>(8, width - b));
    }
}

template <typename T>
MutableColumnPtr makeNumericColumn(size_t rows, size_t col, size_t first_row, UInt64 seed)
{
    auto column = ColumnVector<T>::create(rows);
    char * raw = reinterpret_cast<char *>(column->getData().data());
    for (size_t r = 0; r < rows; ++r)
        fillValue(raw + r * sizeof(T), sizeof(T), first_row + r, col, seed);
    return column;
}

MutableColumnPtr makeFixedStringColumn(size_t width, size_t rows, size_t col, size_t first_row, UInt64 seed)
{
    auto column = ColumnFixedString::create(width);
    column->getChars().resize(rows * width);
    char * raw = reinterpret_cast<char *>(column->getChars().data());
    for (size_t r = 0; r < rows; ++r)
        fillValue(raw + r * width, width, first_row + r, col, seed);
    return column;
}

/// One column per supported element width, over the two fixed-width column families the scatter
/// allocates for (ColumnVector and ColumnFixedString).
MutableColumnPtr makeColumn(size_t width, size_t rows, size_t col, size_t first_row, UInt64 seed)
{
    switch (width)
    {
        case 1: return makeNumericColumn<UInt8>(rows, col, first_row, seed);
        case 2: return makeNumericColumn<UInt16>(rows, col, first_row, seed);
        case 4: return makeNumericColumn<UInt32>(rows, col, first_row, seed);
        case 8: return makeNumericColumn<UInt64>(rows, col, first_row, seed);
        case 16: return makeNumericColumn<UInt128>(rows, col, first_row, seed);
        case 32: return makeNumericColumn<UInt256>(rows, col, first_row, seed);
        case 64: return makeFixedStringColumn(64, rows, col, first_row, seed);
    }
    ADD_FAILURE() << "unsupported test column width " << width;
    return nullptr;
}

struct SideSpec
{
    size_t rows = 0;
    size_t chunk_rows = 1000;
    std::vector<size_t> widths;
    UInt64 seed = 1;
};

std::vector<RoutedChunk> makeSide(const SideSpec & spec)
{
    std::vector<RoutedChunk> chunks;
    for (size_t begin = 0; begin < spec.rows; begin += spec.chunk_rows)
    {
        const size_t n = std::min(spec.chunk_rows, spec.rows - begin);
        RoutedChunk chunk;
        chunk.rows = n;
        for (size_t j = 0; j < spec.widths.size(); ++j)
            chunk.columns.push_back(makeColumn(spec.widths[j], n, j, begin, spec.seed));
        auto routes = ColumnUInt32::create(n);
        for (size_t r = 0; r < n; ++r)
            routes->getData()[r] = routeOfRow(begin + r, spec.seed);
        chunk.routes = std::move(routes);
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

/// A row's tuple: the concatenated value bytes of all its columns (max 1+2+4+8+16+32+64 = 127).
constexpr size_t MAX_TUPLE_BYTES = 192;

size_t tupleWidth(const std::vector<size_t> & widths)
{
    return std::accumulate(widths.begin(), widths.end(), size_t{0});
}

std::string expectedTuple(size_t row, const SideSpec & spec)
{
    std::string tuple(tupleWidth(spec.widths), '\0');
    size_t off = 0;
    for (size_t j = 0; j < spec.widths.size(); ++j)
    {
        fillValue(tuple.data() + off, spec.widths[j], row, j, spec.seed);
        off += spec.widths[j];
    }
    return tuple;
}

std::string chunkTuple(const ScatterChunk & chunk, size_t r, const std::vector<size_t> & widths)
{
    std::string tuple(tupleWidth(widths), '\0');
    size_t off = 0;
    for (size_t j = 0; j < widths.size(); ++j)
    {
        memcpy(tuple.data() + off, chunk.columns[j]->getRawData().data() + r * widths[j], widths[j]);
        off += widths[j];
    }
    return tuple;
}

std::vector<std::string> sortedChunkTuples(const ScatterChunk & chunk, const std::vector<size_t> & widths)
{
    std::vector<std::string> tuples;
    tuples.reserve(chunk.rows);
    for (size_t r = 0; r < chunk.rows; ++r)
        tuples.push_back(chunkTuple(chunk, r, widths));
    std::sort(tuples.begin(), tuples.end());
    return tuples;
}

UInt64 bytesHash(const char * data, size_t n)
{
    UInt64 h = 0xCBF29CE484222325ULL;
    for (size_t b = 0; b < n; b += 8)
    {
        UInt64 v = 0;
        memcpy(&v, data + b, std::min<size_t>(8, n - b));
        h = mix64(h ^ v);
    }
    return h;
}

/// Order-insensitive multiset fingerprint of one partition: a commutative sum over rows of the
/// mixed hash of the row's tuple bytes (row pairing across columns is preserved by hashing the
/// concatenated tuple).
UInt64 chunkFingerprint(const ScatterChunk & chunk, const std::vector<size_t> & widths)
{
    const size_t tuple_bytes = tupleWidth(widths);
    std::array<char, MAX_TUPLE_BYTES> tuple{};
    UInt64 fingerprint = 0;
    for (size_t r = 0; r < chunk.rows; ++r)
    {
        size_t off = 0;
        for (size_t j = 0; j < widths.size(); ++j)
        {
            memcpy(tuple.data() + off, chunk.columns[j]->getRawData().data() + r * widths[j], widths[j]);
            off += widths[j];
        }
        fingerprint += mix64(bytesHash(tuple.data(), tuple_bytes));
    }
    return fingerprint;
}

/// The same fingerprint, computed from the generator for the input rows [first_row, last_row)
/// that route to `partition` — the reference side of the wave tests.
UInt64 expectedFingerprint(size_t first_row, size_t last_row, size_t partition, size_t bits, const SideSpec & spec)
{
    UInt64 fingerprint = 0;
    for (size_t r = first_row; r < last_row; ++r)
    {
        if (partitionOfRow(r, spec.seed, bits) != partition)
            continue;
        const std::string tuple = expectedTuple(r, spec);
        fingerprint += mix64(bytesHash(tuple.data(), tuple.size()));
    }
    return fingerprint;
}

ThreadPool makePool(size_t threads)
{
    return ThreadPool(
        CurrentMetrics::LocalThread, CurrentMetrics::LocalThreadActive, CurrentMetrics::LocalThreadScheduled,
        threads, /*max_free_threads_*/ threads, /*queue_size_*/ 0);
}

}

TEST(ColumnScatter, ExactPartitionSizes)
{
    /// Per-partition row counts must equal the histogram totals, sum to the input row count, and
    /// every partition column must hold exactly that many rows (exact allocation, no churn).
    std::atomic<bool> cancelled{false};
    const SideSpec spec{.rows = 200000, .chunk_rows = 1009, .widths = {8, 4}, .seed = 42};
    for (size_t bits : {3, 8, 10})
    {
        const size_t fanout = size_t{1} << bits;
        std::vector<size_t> expected(fanout, 0);
        for (size_t r = 0; r < spec.rows; ++r)
            ++expected[partitionOfRow(r, spec.seed, bits)];

        ThreadPool pool = makePool(4);
        auto out = scatterColumns(pool, 4, makeSide(spec), {bits}, cancelled);
        ASSERT_EQ(out.size(), fanout);

        size_t total = 0;
        for (size_t p = 0; p < fanout; ++p)
        {
            EXPECT_EQ(out[p].rows, expected[p]) << "bits=" << bits << " partition " << p;
            if (out[p].rows)
            {
                ASSERT_EQ(out[p].columns.size(), spec.widths.size());
                for (const auto & col : out[p].columns)
                    EXPECT_EQ(col->size(), expected[p]) << "bits=" << bits << " partition " << p;
            }
            total += out[p].rows;
        }
        EXPECT_EQ(total, spec.rows);
    }
}

TEST(ColumnScatter, FingerprintStableRouting)
{
    /// The same input scattered with different worker counts must produce per-partition
    /// order-insensitive multiset fingerprints that are identical (routing is a pure function of
    /// the route words; only the row order within a partition may differ across thread counts).
    std::atomic<bool> cancelled{false};
    const SideSpec spec{.rows = 300000, .chunk_rows = 997, .widths = {8, 1, 16}, .seed = 7};
    constexpr size_t bits = 9;

    auto run = [&](size_t threads)
    {
        ThreadPool pool = makePool(threads);
        auto out = scatterColumns(pool, threads, makeSide(spec), {bits}, cancelled);
        std::vector<std::pair<size_t, UInt64>> fingerprints;
        fingerprints.reserve(out.size());
        for (const auto & chunk : out)
            fingerprints.emplace_back(chunk.rows, chunkFingerprint(chunk, spec.widths));
        return fingerprints;
    };

    const auto reference = run(1);
    EXPECT_EQ(run(4), reference);
    EXPECT_EQ(run(16), reference);
}

TEST(ColumnScatter, MultiPassEquivalence)
{
    /// One 8-bit pass and two 4-bit passes must produce multiset-equal final partitions: the
    /// passes slice disjoint bit ranges of the same route word, high bits first, so the final
    /// partition index is the same top-8-bits value either way.
    std::atomic<bool> cancelled{false};
    const SideSpec spec{.rows = 150000, .chunk_rows = 761, .widths = {8, 4}, .seed = 11};

    ThreadPool pool = makePool(4);
    auto single = scatterColumns(pool, 4, makeSide(spec), {8}, cancelled);
    auto multi = scatterColumns(pool, 4, makeSide(spec), {4, 4}, cancelled);

    ASSERT_EQ(single.size(), 256u);
    ASSERT_EQ(multi.size(), 256u);
    for (size_t p = 0; p < single.size(); ++p)
    {
        ASSERT_EQ(single[p].rows, multi[p].rows) << "partition " << p;
        EXPECT_EQ(sortedChunkTuples(single[p], spec.widths), sortedChunkTuples(multi[p], spec.widths)) << "partition " << p;
    }
}

TEST(ColumnScatter, WidthDispatchRoundTrip)
{
    /// All supported payload widths at once: the per-row tuples reassembled from the partitions
    /// must match the input exactly (every row lands in the partition its route selects, with
    /// all its column values still paired). Covers the DIRECT and the SWWC width kernels.
    std::atomic<bool> cancelled{false};
    const SideSpec spec{.rows = 60000, .chunk_rows = 613, .widths = {1, 2, 4, 8, 16, 32, 64}, .seed = 3};

    for (size_t bits : {6, 9})
    {
        const size_t fanout = size_t{1} << bits;
        std::vector<std::vector<std::string>> expected(fanout);
        for (size_t r = 0; r < spec.rows; ++r)
            expected[partitionOfRow(r, spec.seed, bits)].push_back(expectedTuple(r, spec));
        for (auto & tuples : expected)
            std::sort(tuples.begin(), tuples.end());

        ThreadPool pool = makePool(4);
        auto out = scatterColumns(pool, 4, makeSide(spec), {bits}, cancelled);
        ASSERT_EQ(out.size(), fanout);
        for (size_t p = 0; p < fanout; ++p)
        {
            ASSERT_EQ(out[p].rows, expected[p].size()) << "bits=" << bits << " partition " << p;
            EXPECT_EQ(sortedChunkTuples(out[p], spec.widths), expected[p]) << "bits=" << bits << " partition " << p;
        }
    }
}

TEST(ColumnScatter, ThrowingConsumerDoesNotDeadlock)
{
    /// One worker's consumer callback throws mid-wave. Every worker must still arrive at every
    /// barrier (the loop finishes the wave as a no-op team and exits together), and the wave
    /// loop must rethrow exactly the consumer's exception - the test completing at all is the
    /// no-hang assertion.
    std::atomic<bool> cancelled{false};
    const SideSpec spec{.rows = 80000, .chunk_rows = 641, .widths = {8}, .seed = 5};
    constexpr size_t bits = 6;
    constexpr size_t waves = 3;

    /// Non-empty partitions over all waves: the throw must cut consumption short of this.
    const size_t num_chunks = (spec.rows + spec.chunk_rows - 1) / spec.chunk_rows;
    size_t total_nonempty = 0;
    for (size_t w = 0; w < waves; ++w)
    {
        const size_t first_row = (num_chunks * w / waves) * spec.chunk_rows;
        const size_t last_row = std::min(spec.rows, (num_chunks * (w + 1) / waves) * spec.chunk_rows);
        std::vector<bool> seen(size_t{1} << bits, false);
        for (size_t r = first_row; r < last_row; ++r)
            seen[partitionOfRow(r, spec.seed, bits)] = true;
        total_nonempty += std::count(seen.begin(), seen.end(), true);
    }

    std::atomic<size_t> consumed{0};
    ThreadPool pool = makePool(4);
    try
    {
        scatterWaves(
            pool, 4, makeSide(spec), bits, waves,
            [&](size_t /*partition*/, ScatterChunk /*chunk*/)
            {
                if (consumed.fetch_add(1) == 5)
                    throw std::runtime_error("injected consumer failure");
            },
            cancelled);
        FAIL() << "expected the consumer's exception to be rethrown";
    }
    catch (const std::runtime_error & e)
    {
        EXPECT_STREQ(e.what(), "injected consumer failure");
    }
    EXPECT_GE(consumed.load(), 6u);
    EXPECT_LT(consumed.load(), total_nonempty) << "the wave loop must stop before consuming every wave";
}

TEST(ColumnScatter, CancellationStopsWaves)
{
    /// The stop flag is raised once the first wave is fully consumed: the loop must exit
    /// promptly with a QUERY_WAS_CANCELLED exception, and everything consumed from the completed
    /// wave must be intact (correct per-partition row counts and content fingerprints).
    const SideSpec spec{.rows = 120000, .chunk_rows = 500, .widths = {8, 4}, .seed = 9};
    constexpr size_t bits = 5;
    constexpr size_t fanout = size_t{1} << bits;
    constexpr size_t waves = 4;

    /// Window 0 covers whole chunks [0, n/waves) - replicate the documented split to build the
    /// per-partition reference for the first wave.
    const size_t num_chunks = (spec.rows + spec.chunk_rows - 1) / spec.chunk_rows;
    const size_t window0_rows = (num_chunks / waves) * spec.chunk_rows;
    std::vector<size_t> expected_rows(fanout, 0);
    for (size_t r = 0; r < window0_rows; ++r)
        ++expected_rows[partitionOfRow(r, spec.seed, bits)];
    size_t wave0_nonempty = 0;
    for (size_t p = 0; p < fanout; ++p)
        wave0_nonempty += (expected_rows[p] != 0);

    std::atomic<bool> cancel{false};
    std::atomic<size_t> completed{0};
    std::mutex mutex;
    std::map<size_t, std::pair<size_t, UInt64>> consumed;

    ThreadPool pool = makePool(4);
    try
    {
        scatterWaves(
            pool, 4, makeSide(spec), bits, waves,
            [&](size_t partition, ScatterChunk chunk)
            {
                const UInt64 fingerprint = chunkFingerprint(chunk, spec.widths);
                {
                    std::lock_guard lock(mutex);
                    consumed[partition] = {chunk.rows, fingerprint};
                }
                if (completed.fetch_add(1) + 1 == wave0_nonempty)
                    cancel.store(true);
            },
            cancel);
        FAIL() << "expected QUERY_WAS_CANCELLED";
    }
    catch (const Exception & e)
    {
        EXPECT_EQ(e.code(), ErrorCodes::QUERY_WAS_CANCELLED);
    }

    /// Exactly the first wave was consumed, and its partitions are intact.
    ASSERT_EQ(consumed.size(), wave0_nonempty);
    for (const auto & [partition, rows_and_fingerprint] : consumed)
    {
        EXPECT_EQ(rows_and_fingerprint.first, expected_rows[partition]) << "partition " << partition;
        EXPECT_EQ(rows_and_fingerprint.second, expectedFingerprint(0, window0_rows, partition, bits, spec)) << "partition " << partition;
    }
}

TEST(ColumnScatter, SwwcMatchesDirect)
{
    /// At fanout >= 256 the automatic path is SWWC + non-temporal stores, and on this host the
    /// kernels actually run it (the portable NT builtin compiles on every architecture). Forcing
    /// the two paths over the same input must produce bit-for-bit identical partitions: the
    /// write path only changes HOW bytes reach memory, never which bytes or where. The side is
    /// sized so every worker's stripe spans several scatter batches, exercising the mid-line
    /// cursors a drain leaves behind at batch boundaries.
    std::atomic<bool> cancelled{false};
    const SideSpec spec{.rows = 1150000, .chunk_rows = 4096, .widths = {8, 1, 16, 4}, .seed = 13};
    constexpr size_t bits = 9;

    ThreadPool pool = makePool(4);
    auto direct = scatterColumns(pool, 4, makeSide(spec), {bits}, cancelled, ScatterPath::Direct);
    auto swwc = scatterColumns(pool, 4, makeSide(spec), {bits}, cancelled, ScatterPath::Swwc);

    ASSERT_EQ(direct.size(), swwc.size());
    for (size_t p = 0; p < direct.size(); ++p)
    {
        ASSERT_EQ(direct[p].rows, swwc[p].rows) << "partition " << p;
        ASSERT_EQ(direct[p].columns.size(), swwc[p].columns.size()) << "partition " << p;
        for (size_t j = 0; j < direct[p].columns.size(); ++j)
            EXPECT_EQ(direct[p].columns[j]->getRawData(), swwc[p].columns[j]->getRawData()) << "partition " << p << " column " << j;
    }
}

TEST(ColumnScatter, WavesSwwcCompletionFingerprint)
{
    /// The wave loop through the SWWC path (fanout >= 256 engages the non-temporal kernels),
    /// run to COMPLETION: every wave's every non-empty partition must reach the consumer with
    /// exactly the window's routed rows. Waves are barrier-separated, so partition p's
    /// consumptions arrive in wave order - the per-partition consumption sequence must equal
    /// the per-wave expectation sequence of (rows, content fingerprint) from the generator.
    std::atomic<bool> cancelled{false};
    const SideSpec spec{.rows = 120000, .chunk_rows = 991, .widths = {8, 1, 16}, .seed = 17};
    constexpr size_t bits = 8;
    constexpr size_t fanout = size_t{1} << bits;
    constexpr size_t waves = 3;

    std::mutex mutex;
    std::map<size_t, std::vector<std::pair<size_t, UInt64>>> consumed;
    size_t consume_calls = 0;

    ThreadPool pool = makePool(4);
    scatterWaves(
        pool, 4, makeSide(spec), bits, waves,
        [&](size_t partition, ScatterChunk chunk)
        {
            const UInt64 fingerprint = chunkFingerprint(chunk, spec.widths);
            std::lock_guard lock(mutex);
            consumed[partition].emplace_back(chunk.rows, fingerprint);
            ++consume_calls;
        },
        cancelled);

    /// Reference: replicate the documented whole-chunk window split per wave.
    const size_t num_chunks = (spec.rows + spec.chunk_rows - 1) / spec.chunk_rows;
    size_t expected_calls = 0;
    std::map<size_t, std::vector<std::pair<size_t, UInt64>>> expected;
    for (size_t w = 0; w < waves; ++w)
    {
        const size_t first_row = (num_chunks * w / waves) * spec.chunk_rows;
        const size_t last_row = std::min(spec.rows, (num_chunks * (w + 1) / waves) * spec.chunk_rows);
        std::vector<size_t> rows_of(fanout, 0);
        for (size_t r = first_row; r < last_row; ++r)
            ++rows_of[partitionOfRow(r, spec.seed, bits)];
        for (size_t p = 0; p < fanout; ++p)
        {
            if (!rows_of[p])
                continue;
            expected[p].emplace_back(rows_of[p], expectedFingerprint(first_row, last_row, p, bits, spec));
            ++expected_calls;
        }
    }

    EXPECT_EQ(consume_calls, expected_calls);
    EXPECT_EQ(consumed, expected);
}

TEST(ColumnScatter, ComputePassBitsContract)
{
    /// p_star <= 1 means no partitioning is needed: the plan is EMPTY by contract, and the
    /// scatter entry points (which require at least one pass) must not be called with it.
    EXPECT_EQ(computePassBits(1, 64), std::vector<size_t>{});
    EXPECT_EQ(computePassBits(0, 64), std::vector<size_t>{});

    /// The common case: everything fits one pass.
    EXPECT_EQ(computePassBits(2048, 8192), std::vector<size_t>{11});

    /// f_max is clamped to MAX_FANOUT_PER_PASS (13 bits): 2^15 partitions cannot be planned as
    /// a single 15-bit pass however large the caller's f_max claims to be.
    EXPECT_EQ(computePassBits(size_t{1} << 15, size_t{1} << 16), (std::vector<size_t>{8, 7}));

    /// Passes are spread evenly: sizes differ by at most one bit.
    EXPECT_EQ(computePassBits(2048, 16), (std::vector<size_t>{4, 4, 3}));
}
