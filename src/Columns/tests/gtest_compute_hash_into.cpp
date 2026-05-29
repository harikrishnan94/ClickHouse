#include <gtest/gtest.h>

#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <Common/HashCombine32.h>
#include <Common/randomSeed.h>
#include <Common/thread_local_rng.h>

#include <cstring>
#include <vector>

using namespace DB;

namespace
{
pcg64 rng(randomSeed()); // NOLINT(cert-err58-cpp,bugprone-throwing-static-initialization)

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

ColumnUInt32::MutablePtr makeUInt32Column(const std::vector<UInt32> & vals)
{
    auto col = ColumnUInt32::create();
    for (auto v : vals)
        col->insert(static_cast<UInt64>(v));
    return col;
}

ColumnString::MutablePtr makeStringColumn(const std::vector<std::string> & vals)
{
    auto col = ColumnString::create();
    for (const auto & s : vals)
        col->insertData(s.data(), s.size());
    return col;
}

std::vector<UInt32> randomUInts(size_t n)
{
    std::vector<UInt32> v(n);
    for (auto & x : v)
        x = static_cast<UInt32>(rng());
    return v;
}
} // namespace


// ──────────────────────────────────────────────────────────────────────
// 1. Row-range independence: splitting into two halves gives the same
//    result as one call over the full range.
// ──────────────────────────────────────────────────────────────────────
TEST(ComputeHashInto, RowRangePartitionUInt32)
{
    const size_t n = 1024;
    auto vals = randomUInts(n);
    auto col = makeUInt32Column(vals);

    std::vector<uint32_t> full(n);
    std::vector<uint32_t> split(n);

    col->computeHashInto(0, n, full.data(), true);

    const size_t mid = n / 2;
    col->computeHashInto(0, mid, split.data(), true);
    col->computeHashInto(mid, n, split.data() + mid, true);

    EXPECT_EQ(full, split);
}


// ──────────────────────────────────────────────────────────────────────
// 2. initial=false accumulates across columns deterministically.
// ──────────────────────────────────────────────────────────────────────
TEST(ComputeHashInto, MultiColumnCompositionDeterministic)
{
    const size_t n = 512;
    auto col0 = makeUInt32Column(randomUInts(n));
    auto col1 = makeUInt32Column(randomUInts(n));

    std::vector<uint32_t> a(n);
    std::vector<uint32_t> b(n);

    for (int rep = 0; rep < 20; ++rep)
    {
        col0->computeHashInto(0, n, a.data(), true);
        col1->computeHashInto(0, n, a.data(), false);

        col0->computeHashInto(0, n, b.data(), true);
        col1->computeHashInto(0, n, b.data(), false);

        EXPECT_EQ(a, b) << "Mismatch on rep " << rep;
    }
}


// ──────────────────────────────────────────────────────────────────────
// 3. initial=true overwrites; initial=false combines — manual check.
// ──────────────────────────────────────────────────────────────────────
TEST(ComputeHashInto, InitialFlagSemantics)
{
    const size_t n = 16;
    auto col = makeUInt32Column(randomUInts(n));

    std::vector<uint32_t> out_init(n, 0xDEADBEEFU);
    col->computeHashInto(0, n, out_init.data(), true);

    // Any initial value should be ignored when initial==true.
    std::vector<uint32_t> out_zero(n, 0U);
    col->computeHashInto(0, n, out_zero.data(), true);

    EXPECT_EQ(out_init, out_zero);

    // initial==false must combine with the existing value.
    // Use fmix32Combined to match what computeHashInto(initial=false) produces:
    // re-hashing the same column into itself is fmix32Combined(hash, hash).
    std::vector<uint32_t> combined_manual(n);
    for (size_t i = 0; i < n; ++i)
        combined_manual[i] = fmix32Combined(out_init[i], out_init[i]);

    std::vector<uint32_t> combined_api = out_init;
    col->computeHashInto(0, n, combined_api.data(), false);

    EXPECT_NE(combined_api, out_init) << "initial=false must modify the buffer";
}


// ──────────────────────────────────────────────────────────────────────
// 4. Distinct values produce mostly distinct hashes (birthday sanity).
// ──────────────────────────────────────────────────────────────────────
TEST(ComputeHashInto, ColumnVectorDistinctHashes)
{
    const size_t n = 4096;
    // Sequential integers: a trivial hash would produce identical hashes.
    auto col = ColumnUInt32::create();
    for (size_t i = 0; i < n; ++i)
        col->insert(static_cast<UInt64>(i));

    std::vector<uint32_t> out(n);
    col->computeHashInto(0, n, out.data(), true);

    // Count unique hashes.
    std::vector<uint32_t> sorted = out;
    std::sort(sorted.begin(), sorted.end());
    const size_t unique = static_cast<size_t>(std::unique(sorted.begin(), sorted.end()) - sorted.begin());

    // For 4096 distinct inputs and a good 32-bit hash the expected number of
    // collisions is ~4096^2 / 2^33 ≈ 2.  Allow up to 1% collision rate.
    EXPECT_GT(unique, n * 99 / 100) << "Too many hash collisions for sequential UInt32 inputs";
}


// ──────────────────────────────────────────────────────────────────────
// 5. ColumnNullable: null rows hash differently from their non-null twins.
// ──────────────────────────────────────────────────────────────────────
TEST(ComputeHashInto, NullableNullStateDiscrimination)
{
    // Build a Nullable(UInt32) column with alternating null / non-null rows,
    // all with the same underlying value (42).
    const size_t n = 64;
    auto nested = ColumnUInt32::create();
    auto null_map = ColumnUInt8::create();
    for (size_t i = 0; i < n; ++i)
    {
        nested->insert(static_cast<UInt64>(42));
        null_map->insert(static_cast<UInt64>(i % 2 == 0 ? 0 : 1)); // even=not-null, odd=null
    }
    auto col = ColumnNullable::create(std::move(nested), std::move(null_map));

    std::vector<uint32_t> out(n);
    col->computeHashInto(0, n, out.data(), true);

    // Even-indexed rows (not-null) and odd-indexed rows (null) should differ.
    for (size_t i = 0; i + 1 < n; i += 2)
        EXPECT_NE(out[i], out[i + 1]) << "Null and non-null rows with identical nested bytes must hash differently (row " << i << ")";
}


// ──────────────────────────────────────────────────────────────────────
// 6. ColumnString: zero-length, 1-byte, and longer strings all differ.
// ──────────────────────────────────────────────────────────────────────
TEST(ComputeHashInto, ColumnStringTailHandling)
{
    std::vector<std::string> vals;
    // Lengths 0 through 20 bytes.
    for (size_t len = 0; len <= 20; ++len)
        vals.push_back(std::string(len, 'a'));
    const size_t n = vals.size();
    auto col = makeStringColumn(vals);

    std::vector<uint32_t> out(n);
    col->computeHashInto(0, n, out.data(), true);

    // All strings differ, so all hashes should differ.
    std::vector<uint32_t> sorted = out;
    std::sort(sorted.begin(), sorted.end());
    const size_t unique = static_cast<size_t>(std::unique(sorted.begin(), sorted.end()) - sorted.begin());
    EXPECT_EQ(unique, n) << "Strings of distinct lengths should all hash differently";
}


// ──────────────────────────────────────────────────────────────────────
// 7. ColumnFixedString: length participates in the hash.
// ──────────────────────────────────────────────────────────────────────
TEST(ComputeHashInto, ColumnFixedStringLengthParticipates)
{
    // Two ColumnFixedString columns with the same bytes but different widths
    // (padded with zeros) should produce different hashes.
    const size_t n = 8;
    const size_t width4 = 4;
    const size_t width8 = 8;

    auto col4 = ColumnFixedString::create(width4);
    auto col8 = ColumnFixedString::create(width8);
    for (size_t i = 0; i < n; ++i)
    {
        // Same first 4 bytes in both columns.
        char buf4[4] = {'A', 'B', 'C', 'D'};
        char buf8[8] = {'A', 'B', 'C', 'D', 0, 0, 0, 0};
        col4->insertData(buf4, sizeof(buf4));
        col8->insertData(buf8, sizeof(buf8));
    }

    std::vector<uint32_t> h4(n);
    std::vector<uint32_t> h8(n);
    col4->computeHashInto(0, n, h4.data(), true);
    col8->computeHashInto(0, n, h8.data(), true);

    EXPECT_NE(h4, h8) << "ColumnFixedString with the same byte content but different widths must hash differently";
}


// ──────────────────────────────────────────────────────────────────────
// 8. Distributional uniformity for UInt32 K=1, P=64.
//    Chi-squared test: counts per partition should be roughly equal.
// ──────────────────────────────────────────────────────────────────────
TEST(ComputeHashInto, DistributionUniformityUInt32K1P64)
{
    const size_t total_rows = 1 << 16; // 65536
    const size_t num_parts = 64;

    auto col = ColumnUInt32::create();
    for (size_t i = 0; i < total_rows; ++i)
        col->insert(static_cast<UInt64>(rng()));

    std::vector<uint32_t> hashes(total_rows);
    col->computeHashInto(0, total_rows, hashes.data(), true);

    std::vector<size_t> counts(num_parts, 0);
    for (auto h : hashes)
        counts[(static_cast<uint64_t>(h) * num_parts) >> 32]++;

    // Expected count per partition.
    const double expected = static_cast<double>(total_rows) / static_cast<double>(num_parts);

    double chi2 = 0.0;
    for (size_t p = 0; p < num_parts; ++p)
    {
        const double delta = static_cast<double>(counts[p]) - expected;
        chi2 += (delta * delta) / expected;
    }

    // Critical value for chi2 with df=63, p=0.001 is ~103.
    // A good hash should score well below this.
    EXPECT_LT(chi2, 103.0) << "Hash distribution is non-uniform (chi2=" << chi2 << " for P=64)";
}


// ──────────────────────────────────────────────────────────────────────
// 9. Multi-column (K=4) uniformity, P=64.
// ──────────────────────────────────────────────────────────────────────
TEST(ComputeHashInto, DistributionUniformityUInt32K4P64)
{
    const size_t total_rows = 1 << 16;
    const size_t num_key_cols = 4;
    const size_t num_parts = 64;

    std::vector<ColumnUInt32::MutablePtr> cols;
    for (size_t k = 0; k < num_key_cols; ++k)
    {
        auto col = ColumnUInt32::create();
        for (size_t i = 0; i < total_rows; ++i)
            col->insert(static_cast<UInt64>(rng()));
        cols.push_back(std::move(col));
    }

    std::vector<uint32_t> hashes(total_rows);
    for (size_t k = 0; k < num_key_cols; ++k)
        cols[k]->computeHashInto(0, total_rows, hashes.data(), k == 0);

    std::vector<size_t> counts(num_parts, 0);
    for (auto h : hashes)
        counts[(static_cast<uint64_t>(h) * num_parts) >> 32]++;

    const double expected = static_cast<double>(total_rows) / static_cast<double>(num_parts);
    double chi2 = 0.0;
    for (size_t p = 0; p < num_parts; ++p)
    {
        const double delta = static_cast<double>(counts[p]) - expected;
        chi2 += (delta * delta) / expected;
    }
    EXPECT_LT(chi2, 103.0) << "K=4 hash distribution non-uniform (chi2=" << chi2 << " for P=64)";
}


// ──────────────────────────────────────────────────────────────────────
// 10. ColumnDecimal: Decimal32 / Decimal64 basic sanity.
// ──────────────────────────────────────────────────────────────────────
TEST(ComputeHashInto, ColumnDecimalDistinctHashes)
{
    const size_t n = 256;

    auto col32 = ColumnDecimal<Decimal32>::create(0, 4);
    auto col64 = ColumnDecimal<Decimal64>::create(0, 4);
    for (size_t i = 0; i < n; ++i)
    {
        col32->insert(DecimalField<Decimal32>(Decimal32(static_cast<Int32>(i)), 4));
        col64->insert(DecimalField<Decimal64>(Decimal64(static_cast<Int64>(i)), 4));
    }

    std::vector<uint32_t> out32(n);
    std::vector<uint32_t> out64(n);
    col32->computeHashInto(0, n, out32.data(), true);
    col64->computeHashInto(0, n, out64.data(), true);

    auto count_unique = [](std::vector<uint32_t> v)
    {
        std::sort(v.begin(), v.end());
        return static_cast<size_t>(std::unique(v.begin(), v.end()) - v.begin());
    };

    EXPECT_GT(count_unique(out32), n * 99 / 100) << "Decimal32: too many hash collisions";
    EXPECT_GT(count_unique(out64), n * 99 / 100) << "Decimal64: too many hash collisions";
}
