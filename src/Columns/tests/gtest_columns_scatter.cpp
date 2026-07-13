#include <Columns/ColumnsScatter.h>

#include <Columns/ColumnConst.h>
#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnReplicated.h>
#include <Columns/ColumnSparse.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <Common/Exception.h>
#include <Common/randomSeed.h>

#include <gtest/gtest.h>
#include <pcg_random.hpp>

#include <numeric>

using namespace DB;

namespace
{

/// Deterministic per-test rng (seed logged so failures are reproducible).
pcg64 & rng()
{
    static pcg64 generator = []
    {
        UInt64 seed = randomSeed();
        std::cerr << "gtest_columns_scatter seed: " << seed << '\n';
        return pcg64(seed);
    }();
    return generator;
}

template <typename Pid>
std::vector<Pid> makePids(size_t n, size_t num_shards)
{
    std::vector<Pid> pids(n);
    for (auto & pid : pids)
        pid = static_cast<Pid>(rng()() % num_shards);
    return pids;
}

/// Fill a freshly created fixed-width column with `n` rows of random bytes (valid content for
/// ColumnVector / ColumnDecimal / ColumnFixedString — the scatter contract is byte-preservation).
MutableColumnPtr fillFixedRandom(MutableColumnPtr column, size_t n)
{
    auto raw = column->insertRawUninitialized(n);
    for (auto & byte : raw)
        byte = static_cast<char>(rng()());
    return column;
}

/// Independent oracle: legacy IColumn::scatter per source + insertRangeFrom concatenation.
MutableColumns referenceScatter(std::span<const IColumn * const> sources, const std::vector<std::vector<UInt32>> & pids, size_t num_shards)
{
    MutableColumns result(num_shards);
    for (size_t s = 0; s < num_shards; ++s)
        result[s] = sources[0]->convertToFullColumnIfConst()->convertToFullColumnIfReplicated()->convertToFullColumnIfSparse()->cloneEmpty();
    for (size_t b = 0; b < sources.size(); ++b)
    {
        IColumn::Selector selector(pids[b].size());
        for (size_t j = 0; j < pids[b].size(); ++j)
            selector[j] = pids[b][j];
        auto full = sources[b]->convertToFullColumnIfConst()->convertToFullColumnIfReplicated()->convertToFullColumnIfSparse();
        auto parts = full->scatter(num_shards, selector);
        for (size_t s = 0; s < num_shards; ++s)
            if (parts[s]->size())
                result[s]->insertRangeFrom(*parts[s], 0, parts[s]->size());
    }
    return result;
}

void expectColumnsBitIdentical(const IColumn & expected, const IColumn & actual, const std::string & context)
{
    ASSERT_EQ(expected.size(), actual.size()) << context;
    ASSERT_EQ(expected.getDataType(), actual.getDataType()) << context;
    if (expected.size() == 0)
        return;
    const auto expected_raw = expected.getRawData();
    const auto actual_raw = actual.getRawData();
    ASSERT_EQ(expected_raw.size(), actual_raw.size()) << context;
    ASSERT_EQ(0, memcmp(expected_raw.data(), actual_raw.data(), expected_raw.size())) << context;
}

/// Run the module scatter (both pid widths must agree) and compare bit-exactly with the oracle.
void checkEquivalence(std::span<const IColumn * const> sources, const std::vector<std::vector<UInt32>> & pids32, size_t num_shards, bool with_precounted = false)
{
    std::vector<std::span<const UInt32>> pid_spans32;
    std::vector<std::vector<UInt16>> pids16;
    std::vector<std::span<const UInt16>> pid_spans16;
    for (const auto & p : pids32)
    {
        pid_spans32.emplace_back(p.data(), p.size());
        auto & p16 = pids16.emplace_back();
        p16.reserve(p.size());
        for (UInt32 pid : p)
            p16.push_back(static_cast<UInt16>(pid));
    }
    for (const auto & p : pids16)
        pid_spans16.emplace_back(p.data(), p.size());

    std::vector<UInt32> counts(num_shards, 0);
    ColumnsScatter::countRowsPerShard(std::span<const std::span<const UInt32>>(pid_spans32), std::span<UInt32>(counts));
    std::span<const UInt32> counts_arg;
    if (with_precounted)
        counts_arg = std::span<const UInt32>(counts);

    auto result32 = ColumnsScatter::scatter(sources, std::span<const std::span<const UInt32>>(pid_spans32), num_shards, counts_arg);
    auto result16 = ColumnsScatter::scatter(sources, std::span<const std::span<const UInt16>>(pid_spans16), num_shards, counts_arg);
    auto expected = referenceScatter(sources, pids32, num_shards);

    ASSERT_EQ(num_shards, result32.size());
    ASSERT_EQ(num_shards, result16.size());
    for (size_t s = 0; s < num_shards; ++s)
    {
        /// T3: per-shard counts equal the selector histogram, independent of contents.
        ASSERT_EQ(counts[s], result32[s]->size()) << "shard " << s;
        const std::string context = "shard " + std::to_string(s) + " of " + std::to_string(num_shards);
        expectColumnsBitIdentical(*expected[s], *result32[s], context + " (pid32)");
        expectColumnsBitIdentical(*expected[s], *result16[s], context + " (pid16)");
    }
}

void checkFixedTypeEquivalence(const IColumn & prototype)
{
    SCOPED_TRACE(prototype.getName());
    ASSERT_EQ(ColumnsScatter::ScatterKernelId::FixedWidth, ColumnsScatter::plannedKernel(prototype));

    struct Case
    {
        size_t num_sources;
        size_t rows_per_source;
        size_t num_shards;
    };
    /// Fanouts cover: trivial, direct regime, SWWC regime (>= 256), scratch heap-spill (> 256 inline).
    for (const auto & test_case : std::initializer_list<Case>{{1, 1000, 1}, {1, 1000, 8}, {3, 700, 8}, {2, 5000, 256}, {2, 3000, 512}})
    {
        std::vector<MutableColumnPtr> owned;
        std::vector<const IColumn *> sources;
        std::vector<std::vector<UInt32>> pids;
        for (size_t b = 0; b < test_case.num_sources; ++b)
        {
            owned.push_back(fillFixedRandom(prototype.cloneEmpty(), test_case.rows_per_source));
            sources.push_back(owned.back().get());
            pids.push_back(makePids<UInt32>(test_case.rows_per_source, test_case.num_shards));
        }

        ColumnsScatter::DispatchTrace trace;
        auto * previous = ColumnsScatter::exchangeDispatchTrace(&trace);
        checkEquivalence(std::span<const IColumn * const>(sources.data(), sources.size()), pids, test_case.num_shards);
        ColumnsScatter::exchangeDispatchTrace(previous);

        /// T5: both pid-width calls inside checkEquivalence hit the named kernel, never the fallback.
        ASSERT_EQ(2u, trace.entries.size());
        for (const auto & entry : trace.entries)
            ASSERT_EQ(ColumnsScatter::ScatterKernelId::FixedWidth, entry.kernel) << prototype.getName();
    }
}

}

/// T1/T2/T3/T5: every fixed-width fast-path type, batched sources, counts, dispatch.
TEST(ColumnsScatter, FixedWidthVectorTypes)
{
    checkFixedTypeEquivalence(*ColumnUInt8::create());
    checkFixedTypeEquivalence(*ColumnUInt16::create());
    checkFixedTypeEquivalence(*ColumnUInt32::create());
    checkFixedTypeEquivalence(*ColumnUInt64::create());
    checkFixedTypeEquivalence(*ColumnUInt128::create());
    checkFixedTypeEquivalence(*ColumnUInt256::create());
    checkFixedTypeEquivalence(*ColumnInt8::create());
    checkFixedTypeEquivalence(*ColumnInt16::create());
    checkFixedTypeEquivalence(*ColumnInt32::create());
    checkFixedTypeEquivalence(*ColumnInt64::create());
    checkFixedTypeEquivalence(*ColumnInt128::create());
    checkFixedTypeEquivalence(*ColumnInt256::create());
    checkFixedTypeEquivalence(*ColumnBFloat16::create());
    checkFixedTypeEquivalence(*ColumnFloat32::create());
    checkFixedTypeEquivalence(*ColumnFloat64::create());
    checkFixedTypeEquivalence(*ColumnUUID::create());
    checkFixedTypeEquivalence(*ColumnIPv4::create());
    checkFixedTypeEquivalence(*ColumnIPv6::create());
}

TEST(ColumnsScatter, FixedWidthDecimalTypes)
{
    checkFixedTypeEquivalence(*ColumnDecimal<Decimal32>::create(0, 2));
    checkFixedTypeEquivalence(*ColumnDecimal<Decimal64>::create(0, 4));
    checkFixedTypeEquivalence(*ColumnDecimal<Decimal128>::create(0, 10));
    checkFixedTypeEquivalence(*ColumnDecimal<Decimal256>::create(0, 20));
    checkFixedTypeEquivalence(*ColumnDecimal<DateTime64>::create(0, 3));
    checkFixedTypeEquivalence(*ColumnDecimal<Time64>::create(0, 6));
}

TEST(ColumnsScatter, FixedString)
{
    checkFixedTypeEquivalence(*ColumnFixedString::create(1));
    checkFixedTypeEquivalence(*ColumnFixedString::create(3));  /// generic-width kernel
    checkFixedTypeEquivalence(*ColumnFixedString::create(16)); /// SWWC-capable width
    checkFixedTypeEquivalence(*ColumnFixedString::create(32)); /// generic-width kernel
}

/// T4 edge cases.
TEST(ColumnsScatter, ZeroRowSourceAmongNonEmpty)
{
    auto a = fillFixedRandom(ColumnUInt64::create(), 100);
    auto b = ColumnUInt64::create(); /// empty
    auto c = fillFixedRandom(ColumnUInt64::create(), 50);
    std::vector<const IColumn *> sources{a.get(), b.get(), c.get()};
    std::vector<std::vector<UInt32>> pids{makePids<UInt32>(100, 4), {}, makePids<UInt32>(50, 4)};
    checkEquivalence(std::span<const IColumn * const>(sources.data(), sources.size()), pids, 4);
}

TEST(ColumnsScatter, AllRowsToOneShard)
{
    auto column = fillFixedRandom(ColumnUInt64::create(), 500);
    std::vector<const IColumn *> sources{column.get()};
    std::vector<std::vector<UInt32>> pids{std::vector<UInt32>(500, 0)};
    checkEquivalence(std::span<const IColumn * const>(sources.data(), sources.size()), pids, 8);
}

TEST(ColumnsScatter, PrecountedRowsPerShardMatchesInternalCounting)
{
    auto column = fillFixedRandom(ColumnUInt64::create(), 2000);
    std::vector<const IColumn *> sources{column.get()};
    std::vector<std::vector<UInt32>> pids{makePids<UInt32>(2000, 16)};
    checkEquivalence(std::span<const IColumn * const>(sources.data(), sources.size()), pids, 16, /*with_precounted=*/true);
    checkEquivalence(std::span<const IColumn * const>(sources.data(), sources.size()), pids, 16, /*with_precounted=*/false);
}

/// SWWC line-flush coverage: enough rows per shard that full 64-byte lines stream via NT stores.
TEST(ColumnsScatter, SwwcManyLinesPerShard)
{
    auto a = fillFixedRandom(ColumnUInt64::create(), 64 << 10);
    auto b = fillFixedRandom(ColumnUInt64::create(), 64 << 10);
    std::vector<const IColumn *> sources{a.get(), b.get()};
    std::vector<std::vector<UInt32>> pids{makePids<UInt32>(64 << 10, 256), makePids<UInt32>(64 << 10, 256)};
    checkEquivalence(std::span<const IColumn * const>(sources.data(), sources.size()), pids, 256);
}

/// T6: transparent wrappers over fixed-width nested columns.
TEST(ColumnsScatter, ConstMixedWithFull)
{
    auto full = fillFixedRandom(ColumnUInt64::create(), 300);
    auto const_column = ColumnConst::create(fillFixedRandom(ColumnUInt64::create(), 1), 200);

    for (bool const_first : {true, false})
    {
        std::vector<const IColumn *> sources;
        std::vector<std::vector<UInt32>> pids;
        if (const_first)
        {
            sources = {const_column.get(), full.get()};
            pids = {makePids<UInt32>(200, 8), makePids<UInt32>(300, 8)};
        }
        else
        {
            sources = {full.get(), const_column.get()};
            pids = {makePids<UInt32>(300, 8), makePids<UInt32>(200, 8)};
        }
        ColumnsScatter::DispatchTrace trace;
        auto * previous = ColumnsScatter::exchangeDispatchTrace(&trace);
        checkEquivalence(std::span<const IColumn * const>(sources.data(), sources.size()), pids, 8);
        ColumnsScatter::exchangeDispatchTrace(previous);
        for (const auto & entry : trace.entries)
            ASSERT_EQ(ColumnsScatter::ScatterKernelId::FixedWidth, entry.kernel);
    }
}

TEST(ColumnsScatter, TwoConstsDifferentValuesMaterialize)
{
    auto value_a = ColumnUInt64::create();
    value_a->insert(42u);
    auto value_b = ColumnUInt64::create();
    value_b->insert(43u);
    auto const_a = ColumnConst::create(std::move(value_a), 100);
    auto const_b = ColumnConst::create(std::move(value_b), 150);
    std::vector<const IColumn *> sources{const_a.get(), const_b.get()};
    std::vector<std::vector<UInt32>> pids{makePids<UInt32>(100, 4), makePids<UInt32>(150, 4)};
    checkEquivalence(std::span<const IColumn * const>(sources.data(), sources.size()), pids, 4);
}

TEST(ColumnsScatter, AllConstEqualValuesStayCompact)
{
    auto make_const = [](size_t rows)
    {
        auto value = ColumnUInt64::create();
        value->insert(7u);
        return ColumnConst::create(std::move(value), rows);
    };
    auto const_a = make_const(100);
    auto const_b = make_const(60);
    std::vector<const IColumn *> sources{const_a.get(), const_b.get()};
    std::vector<std::vector<UInt32>> pids{makePids<UInt32>(100, 4), makePids<UInt32>(60, 4)};
    std::vector<std::span<const UInt32>> pid_spans;
    for (const auto & p : pids)
        pid_spans.emplace_back(p.data(), p.size());

    ColumnsScatter::DispatchTrace trace;
    auto * previous = ColumnsScatter::exchangeDispatchTrace(&trace);
    auto result = ColumnsScatter::scatter(
        std::span<const IColumn * const>(sources.data(), sources.size()), std::span<const std::span<const UInt32>>(pid_spans), 4);
    ColumnsScatter::exchangeDispatchTrace(previous);

    ASSERT_EQ(1u, trace.entries.size());
    ASSERT_EQ(ColumnsScatter::ScatterKernelId::ConstCompact, trace.entries[0].kernel);

    size_t total = 0;
    for (const auto & shard : result)
    {
        ASSERT_TRUE(shard->isConst() || shard->empty());
        total += shard->size();
        if (shard->size())
            ASSERT_EQ(7u, (*shard)[0].safeGet<UInt64>());
    }
    ASSERT_EQ(160u, total);
}

/// Bit-exactness of the compact path: -0.0 and +0.0 compare equal by value but differ in bytes —
/// they must NOT be collapsed into one const.
TEST(ColumnsScatter, ConstBitExactNotOrderingEqual)
{
    auto value_pos = ColumnFloat64::create();
    value_pos->insert(0.0);
    auto value_neg = ColumnFloat64::create();
    double negative_zero = -0.0;
    value_neg->insertData(reinterpret_cast<const char *>(&negative_zero), sizeof(negative_zero));
    auto const_pos = ColumnConst::create(std::move(value_pos), 40);
    auto const_neg = ColumnConst::create(std::move(value_neg), 40);
    std::vector<const IColumn *> sources{const_pos.get(), const_neg.get()};
    std::vector<std::vector<UInt32>> pids{makePids<UInt32>(40, 2), makePids<UInt32>(40, 2)};
    std::vector<std::span<const UInt32>> pid_spans;
    for (const auto & p : pids)
        pid_spans.emplace_back(p.data(), p.size());

    auto result = ColumnsScatter::scatter(
        std::span<const IColumn * const>(sources.data(), sources.size()), std::span<const std::span<const UInt32>>(pid_spans), 2);

    /// Count rows whose bit pattern is -0.0 across shards: must equal the second source's rows.
    size_t negative_bits = 0;
    for (const auto & shard : result)
    {
        const auto & data = assert_cast<const ColumnFloat64 &>(*shard).getData();
        for (Float64 value : data)
        {
            UInt64 bits;
            memcpy(&bits, &value, sizeof(bits));
            negative_bits += (bits == 0x8000000000000000ULL);
        }
    }
    ASSERT_EQ(40u, negative_bits);
}

TEST(ColumnsScatter, SparseNormalizedBeforeDispatch)
{
    /// Sparse UInt64: values column row 0 is the shared default, offsets list the non-default rows.
    auto values = ColumnUInt64::create();
    values->insert(0u); /// default
    values->insert(11u);
    values->insert(22u);
    auto offsets = ColumnUInt64::create();
    offsets->insert(3u);
    offsets->insert(7u);
    auto sparse = ColumnSparse::create(std::move(values), std::move(offsets), 20);

    auto full = fillFixedRandom(ColumnUInt64::create(), 30);
    for (bool sparse_first : {true, false})
    {
        std::vector<const IColumn *> sources;
        std::vector<std::vector<UInt32>> pids;
        if (sparse_first)
        {
            sources = {sparse.get(), full.get()};
            pids = {makePids<UInt32>(20, 4), makePids<UInt32>(30, 4)};
        }
        else
        {
            sources = {full.get(), sparse.get()};
            pids = {makePids<UInt32>(30, 4), makePids<UInt32>(20, 4)};
        }
        ColumnsScatter::DispatchTrace trace;
        auto * previous = ColumnsScatter::exchangeDispatchTrace(&trace);
        checkEquivalence(std::span<const IColumn * const>(sources.data(), sources.size()), pids, 4);
        ColumnsScatter::exchangeDispatchTrace(previous);
        for (const auto & entry : trace.entries)
            ASSERT_EQ(ColumnsScatter::ScatterKernelId::FixedWidth, entry.kernel);
    }
}

/// U1 fallback coverage: String has no fast path yet; it must take the fallback and still be exact.
TEST(ColumnsScatter, StringFallsBackInU1)
{
    auto make_strings = [](size_t n)
    {
        auto column = ColumnString::create();
        for (size_t i = 0; i < n; ++i)
        {
            std::string value(rng()() % 20, 'a' + (i % 26));
            column->insertData(value.data(), value.size());
        }
        return column;
    };
    auto a = make_strings(200);
    auto b = make_strings(100);
    ASSERT_EQ(ColumnsScatter::ScatterKernelId::Fallback, ColumnsScatter::plannedKernel(*a));

    std::vector<const IColumn *> sources{a.get(), b.get()};
    std::vector<std::vector<UInt32>> pids{makePids<UInt32>(200, 8), makePids<UInt32>(100, 8)};
    std::vector<std::span<const UInt32>> pid_spans;
    for (const auto & p : pids)
        pid_spans.emplace_back(p.data(), p.size());

    ColumnsScatter::DispatchTrace trace;
    auto * previous = ColumnsScatter::exchangeDispatchTrace(&trace);
    auto result = ColumnsScatter::scatter(
        std::span<const IColumn * const>(sources.data(), sources.size()), std::span<const std::span<const UInt32>>(pid_spans), 8);
    ColumnsScatter::exchangeDispatchTrace(previous);
    ASSERT_EQ(1u, trace.entries.size());
    ASSERT_EQ(ColumnsScatter::ScatterKernelId::Fallback, trace.entries[0].kernel);

    auto expected = referenceScatter(std::span<const IColumn * const>(sources.data(), sources.size()), pids, 8);
    for (size_t s = 0; s < 8; ++s)
    {
        ASSERT_EQ(expected[s]->size(), result[s]->size());
        for (size_t i = 0; i < expected[s]->size(); ++i)
            ASSERT_EQ(expected[s]->getDataAt(i), result[s]->getDataAt(i)) << "shard " << s << " row " << i;
    }
}

/// T7 negative cases: misuse fails loudly. In debug/sanitizer builds a thrown LOGICAL_ERROR aborts
/// the process by design (also loud, but not catchable) — the throw form is asserted in release.
TEST(ColumnsScatter, NegativeMisuseThrows)
{
#ifdef DEBUG_OR_SANITIZER_BUILD
    GTEST_SKIP() << "LOGICAL_ERROR aborts under debug/sanitizer builds; the throwing contract is asserted in release builds";
#else
    auto column = fillFixedRandom(ColumnUInt64::create(), 10);
    std::vector<const IColumn *> sources{column.get()};
    auto pids = makePids<UInt32>(10, 4);
    std::vector<std::span<const UInt32>> pid_spans{{pids.data(), pids.size()}};
    std::vector<std::span<const UInt32>> empty_spans;
    std::vector<UInt32> bad_counts(3, 0);

    /// No sources.
    EXPECT_THROW(
        (void)ColumnsScatter::scatter(std::span<const IColumn * const>{}, std::span<const std::span<const UInt32>>(empty_spans), 4),
        Exception);
    /// Source/pids span count mismatch.
    EXPECT_THROW(
        (void)ColumnsScatter::scatter(
            std::span<const IColumn * const>(sources.data(), 1), std::span<const std::span<const UInt32>>(empty_spans), 4),
        Exception);
    /// Zero shards.
    EXPECT_THROW(
        (void)ColumnsScatter::scatter(
            std::span<const IColumn * const>(sources.data(), 1), std::span<const std::span<const UInt32>>(pid_spans), 0),
        Exception);
    /// rows_per_shard wrong size.
    EXPECT_THROW(
        (void)ColumnsScatter::scatter(
            std::span<const IColumn * const>(sources.data(), 1),
            std::span<const std::span<const UInt32>>(pid_spans),
            4,
            std::span<const UInt32>(bad_counts.data(), bad_counts.size())),
        Exception);
    /// Pid count != column rows.
    auto short_pids = makePids<UInt32>(5, 4);
    std::vector<std::span<const UInt32>> short_spans{{short_pids.data(), short_pids.size()}};
    EXPECT_THROW(
        (void)ColumnsScatter::scatter(
            std::span<const IColumn * const>(sources.data(), 1), std::span<const std::span<const UInt32>>(short_spans), 4),
        Exception);
    /// Mixed concrete types across sources: same TypeIndex but different value widths — a silent
    /// corruption hazard in the raw-byte kernel if unchecked.
    auto fixed_4 = fillFixedRandom(ColumnFixedString::create(4), 10);
    auto fixed_8 = fillFixedRandom(ColumnFixedString::create(8), 10);
    std::vector<const IColumn *> mixed_sources{fixed_4.get(), fixed_8.get()};
    auto pids_a = makePids<UInt32>(10, 4);
    auto pids_b = makePids<UInt32>(10, 4);
    std::vector<std::span<const UInt32>> mixed_spans{{pids_a.data(), pids_a.size()}, {pids_b.data(), pids_b.size()}};
    EXPECT_THROW(
        (void)ColumnsScatter::scatter(
            std::span<const IColumn * const>(mixed_sources.data(), 2), std::span<const std::span<const UInt32>>(mixed_spans), 4),
        Exception);
    /// Mixed TypeIndex across sources.
    auto ints = fillFixedRandom(ColumnUInt32::create(), 10);
    std::vector<const IColumn *> mixed_types{column.get(), ints.get()};
    EXPECT_THROW(
        (void)ColumnsScatter::scatter(
            std::span<const IColumn * const>(mixed_types.data(), 2), std::span<const std::span<const UInt32>>(mixed_spans), 4),
        Exception);
#endif
}

TEST(ColumnsScatter, ReplicatedNormalizedBeforeDispatch)
{
    auto nested = ColumnUInt64::create();
    for (UInt64 value : {100u, 200u, 300u, 400u, 500u})
        nested->insert(value);
    auto indexes = ColumnUInt64::create();
    for (size_t i = 0; i < 30; ++i)
        indexes->insert(rng()() % 5);
    const ColumnPtr nested_ptr = std::move(nested);
    const ColumnPtr indexes_ptr = std::move(indexes);
    auto replicated = ColumnReplicated::create(nested_ptr, indexes_ptr);

    auto full = fillFixedRandom(ColumnUInt64::create(), 40);
    for (bool replicated_first : {true, false})
    {
        std::vector<const IColumn *> sources;
        std::vector<std::vector<UInt32>> pids;
        if (replicated_first)
        {
            sources = {replicated.get(), full.get()};
            pids = {makePids<UInt32>(30, 4), makePids<UInt32>(40, 4)};
        }
        else
        {
            sources = {full.get(), replicated.get()};
            pids = {makePids<UInt32>(40, 4), makePids<UInt32>(30, 4)};
        }
        ColumnsScatter::DispatchTrace trace;
        auto * previous = ColumnsScatter::exchangeDispatchTrace(&trace);
        checkEquivalence(std::span<const IColumn * const>(sources.data(), sources.size()), pids, 4);
        ColumnsScatter::exchangeDispatchTrace(previous);
        for (const auto & entry : trace.entries)
            ASSERT_EQ(ColumnsScatter::ScatterKernelId::FixedWidth, entry.kernel);
    }
}

/// The SWWC misalignment-seeding invariant (header doc): cursors seeded mid-line (the U5 join
/// configuration, where workers seed at prefix-sum offsets) must fill the partial head line with
/// regular stores, then stream aligned NT lines, and drain the residual. Scatters all shards into
/// ONE shared buffer at prefix-sum offsets so most seeds are NOT 64-byte aligned.
TEST(ColumnsScatter, MisalignedCursorSeedingSwwc)
{
    const size_t n = 64 << 10;
    const size_t fanout = 256; /// SWWC regime
    const size_t width = 8;

    auto payload = fillFixedRandom(ColumnUInt64::create(), n);
    const char * data = payload->getRawData().data();
    auto pids = makePids<UInt16>(n, fanout);

    std::vector<size_t> counts(fanout, 0);
    for (UInt16 pid : pids)
        ++counts[pid];
    std::vector<size_t> prefix(fanout, 0);
    for (size_t p = 1; p < fanout; ++p)
        prefix[p] = prefix[p - 1] + counts[p - 1];

    PaddedPODArray<char> destination;
    destination.resize(n * width);
    size_t misaligned_seeds = 0;
    ColumnsScatter::ScatterScratch scratch;
    scratch.init(fanout, /*use_swwc=*/true);
    for (size_t p = 0; p < fanout; ++p)
    {
        char * cursor = destination.data() + prefix[p] * width;
        misaligned_seeds += (reinterpret_cast<uintptr_t>(cursor) & 63) != 0;
        scratch.seed(p, cursor);
    }
    /// The invariant coverage must be real: with random counts most prefix offsets are mid-line.
    ASSERT_GT(misaligned_seeds, fanout / 2);

    ColumnsScatter::scatterPidChunk(width, pids.data(), data, n, /*use_swwc=*/true, scratch);
    scratch.drain();

    /// Bit-exact contents at every row and exact final cursor positions.
    std::vector<size_t> cursor_rows(fanout, 0);
    for (size_t i = 0; i < n; ++i)
    {
        const size_t p = pids[i];
        UInt64 expected;
        memcpy(&expected, data + i * width, width);
        UInt64 actual;
        memcpy(&actual, destination.data() + (prefix[p] + cursor_rows[p]) * width, width);
        ASSERT_EQ(expected, actual) << "row " << i;
        ++cursor_rows[p];
    }
    for (size_t p = 0; p < fanout; ++p)
        ASSERT_EQ(destination.data() + (prefix[p] + counts[p]) * width, scratch.cursors[p]) << "shard " << p;
}

/// Layer-0 primitives: the exact composition the join will use in U5 (histogram -> exact allocation
/// -> key scatter emitting pids -> payload scatter from pids).
TEST(ColumnsScatter, Layer0KeyScatterComposition)
{
    const size_t n = 10000;
    const size_t bits = 6;
    const size_t fanout = 1ULL << bits;
    const UInt32 shift = 32 - bits;
    const UInt32 mask = static_cast<UInt32>(fanout - 1);

    auto keys = fillFixedRandom(ColumnUInt64::create(), n);
    auto payload = fillFixedRandom(ColumnUInt32::create(), n);
    const char * keys_raw = keys->getRawData().data();
    const char * payload_raw = payload->getRawData().data();

    /// Expected routing from the header-inline route hash.
    std::vector<UInt32> expected_hist(fanout, 0);
    std::vector<UInt16> expected_pids(n);
    for (size_t i = 0; i < n; ++i)
    {
        UInt64 key;
        memcpy(&key, keys_raw + i * 8, 8);
        expected_pids[i] = static_cast<UInt16>((ColumnsScatter::routeWord(key) >> shift) & mask);
        ++expected_hist[expected_pids[i]];
    }

    /// Histogram via the interleaved-lane chunk primitive.
    std::vector<UInt32> hist(fanout, 0);
    std::vector<UInt32> lanes(4 * fanout, 0);
    ColumnsScatter::histogramKeyChunk(8, keys_raw, n, shift, mask, hist.data(), lanes.data(), fanout);
    ColumnsScatter::reduceHistogramLanes(hist.data(), lanes.data(), fanout);
    ASSERT_EQ(expected_hist, hist);

    /// Exact allocation + key scatter (emitting pids) + payload scatter from those pids.
    const bool use_swwc = fanout >= ColumnsScatter::SWWC_MIN_FANOUT; /// false at 64: direct regime
    ColumnsScatter::ScatterScratch scratch;
    scratch.init(fanout, use_swwc);

    MutableColumns key_shards(fanout);
    std::vector<char *> key_bases(fanout);
    for (size_t p = 0; p < fanout; ++p)
    {
        auto [column, raw] = ColumnsScatter::allocateUninitializedFixed(*keys, hist[p]);
        key_shards[p] = std::move(column);
        key_bases[p] = raw.data();
        scratch.seed(p, raw.data());
    }
    std::vector<UInt16> emitted_pids(n);
    ColumnsScatter::scatterKeyChunk(8, keys_raw, n, shift, mask, emitted_pids.data(), use_swwc, scratch);
    scratch.drain();
    ASSERT_EQ(expected_pids, emitted_pids);

    MutableColumns payload_shards(fanout);
    for (size_t p = 0; p < fanout; ++p)
    {
        auto [column, raw] = ColumnsScatter::allocateUninitializedFixed(*payload, hist[p]);
        payload_shards[p] = std::move(column);
        scratch.seed(p, raw.data());
    }
    ColumnsScatter::scatterPidChunk(4, emitted_pids.data(), payload_raw, n, use_swwc, scratch);
    scratch.drain();

    /// Verify contents row-by-row against a scalar reference.
    std::vector<size_t> cursor(fanout, 0);
    for (size_t i = 0; i < n; ++i)
    {
        const size_t p = expected_pids[i];
        UInt64 expected_key;
        memcpy(&expected_key, keys_raw + i * 8, 8);
        UInt64 actual_key;
        memcpy(&actual_key, key_shards[p]->getRawData().data() + cursor[p] * 8, 8);
        ASSERT_EQ(expected_key, actual_key) << "row " << i;
        UInt32 expected_payload;
        memcpy(&expected_payload, payload_raw + i * 4, 4);
        UInt32 actual_payload;
        memcpy(&actual_payload, payload_shards[p]->getRawData().data() + cursor[p] * 4, 4);
        ASSERT_EQ(expected_payload, actual_payload) << "row " << i;
        ++cursor[p];
    }
}
