#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeString.h>
#include <Interpreters/HashJoin/JoinSlotRouting.h>

using namespace DB;

namespace
{

/// Reference fold chain: the documented per-row contract (all columns in clause order,
/// value bytes through `foldBytes`, `finalizeRoute` at the end). The production code may take
/// unrolled or column-outer shortcuts; they must stay bit-identical to this chain.
UInt32 referenceWord(const std::vector<std::string_view> & column_values)
{
    UInt64 h = 0;
    for (const auto & value : column_values)
        h = JoinSlotRouting::foldBytes(h, value.data(), value.size());
    return JoinSlotRouting::finalizeRoute(h);
}

ColumnPtr makeUInt64Column(const std::vector<UInt64> & values)
{
    auto column = ColumnUInt64::create();
    for (UInt64 value : values)
        column->insertValue(value);
    return column;
}

ColumnPtr makeStringColumn(const std::vector<std::string> & values)
{
    auto column = ColumnString::create();
    for (const auto & value : values)
        column->insertData(value.data(), value.size());
    return column;
}

ColumnPtr makeLowCardinalityStringColumn(const std::vector<std::string> & values)
{
    auto type = std::make_shared<DataTypeLowCardinality>(std::make_shared<DataTypeString>());
    auto column = type->createColumn();
    for (const auto & value : values)
        column->insertData(value.data(), value.size());
    return column;
}

std::vector<UInt32> routeWordsOf(const ColumnRawPtrs & key_columns, size_t rows)
{
    std::vector<UInt32> words(rows);
    JoinSlotRouting::computeJoinRouteWords(key_columns, rows, words.data());
    return words;
}

}

TEST(JoinSlotRouting, SingleNumericMatchesRouteWord)
{
    const std::vector<UInt64> values{0, 1, 2, 42, 0xDEADBEEF, std::numeric_limits<UInt64>::max()};
    auto column = makeUInt64Column(values);
    const auto words = routeWordsOf({column.get()}, values.size());
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_EQ(words[i], JoinSlotRouting::routeWord(values[i])) << "row " << i;
}

TEST(JoinSlotRouting, NarrowNumericWidthsUseRouteWordOnWidenedValue)
{
    const std::vector<UInt64> values{0, 1, 200, 255};
    auto column = ColumnUInt8::create();
    for (UInt64 value : values)
        column->insertValue(static_cast<UInt8>(value));
    const auto words = routeWordsOf({column.get()}, values.size());
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_EQ(words[i], JoinSlotRouting::routeWord(values[i])) << "row " << i;
}

TEST(JoinSlotRouting, LowCardinalityMatchesPlainString)
{
    const std::vector<std::string> values{"", "a", "abc", std::string("em\0bedded", 9), "abc", "repeat", "repeat"};
    auto plain = makeStringColumn(values);
    auto low_cardinality = makeLowCardinalityStringColumn(values);
    const auto plain_words = routeWordsOf({plain.get()}, values.size());
    const auto lc_words = routeWordsOf({low_cardinality.get()}, values.size());
    EXPECT_EQ(plain_words, lc_words);
}

TEST(JoinSlotRouting, StringsMatchReferenceFold)
{
    const std::vector<std::string> values{"", "x", "12345678", "123456789", std::string("a\0b", 3), std::string("ab\0", 3)};
    auto column = makeStringColumn(values);
    const auto words = routeWordsOf({column.get()}, values.size());
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_EQ(words[i], referenceWord({values[i]})) << "row " << i;
    /// Embedded zeros in different positions must not collide through the zero-padded tail.
    EXPECT_NE(words[4], words[5]);
}

TEST(JoinSlotRouting, AllFixedUnrolledMatchesReference)
{
    /// 2, 3, 4 columns take the unrolled width-8 fold; 5 columns take the runtime-count arm.
    for (size_t n_columns : {2, 3, 4, 5})
    {
        constexpr size_t rows = 257;
        std::vector<ColumnPtr> holders;
        ColumnRawPtrs raw;
        std::vector<std::vector<UInt64>> data(n_columns);
        for (size_t c = 0; c < n_columns; ++c)
        {
            for (size_t i = 0; i < rows; ++i)
                data[c].push_back(i * 1000003 + c * 7 + (i << 32));
            holders.push_back(makeUInt64Column(data[c]));
            raw.push_back(holders.back().get());
        }
        const auto words = routeWordsOf(raw, rows);
        for (size_t i = 0; i < rows; ++i)
        {
            std::vector<std::string_view> row_values;
            for (size_t c = 0; c < n_columns; ++c)
                row_values.emplace_back(reinterpret_cast<const char *>(&data[c][i]), sizeof(UInt64));
            EXPECT_EQ(words[i], referenceWord(row_values)) << "columns " << n_columns << " row " << i;
        }
    }
}

TEST(JoinSlotRouting, WideNumericTakesByteFold)
{
    auto column = ColumnUInt128::create();
    std::vector<UInt128> values;
    for (size_t i = 0; i < 5; ++i)
    {
        UInt128 value = UInt128(i) << 64 | UInt128(i * 31 + 1);
        values.push_back(value);
        column->insertValue(value);
    }
    const auto words = routeWordsOf({column.get()}, values.size());
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_EQ(words[i], referenceWord({{reinterpret_cast<const char *>(&values[i]), sizeof(UInt128)}})) << "row " << i;
}

TEST(JoinSlotRouting, MixedFixedAndStringMatchesReference)
{
    const std::vector<UInt64> numbers{1, 2, 3, 0, std::numeric_limits<UInt64>::max()};
    const std::vector<std::string> strings{"", "a", "bb", "ccc", "dddd"};
    auto number_column = makeUInt64Column(numbers);
    auto string_column = makeStringColumn(strings);
    const auto words = routeWordsOf({number_column.get(), string_column.get()}, numbers.size());
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        const std::string_view number_bytes(reinterpret_cast<const char *>(&numbers[i]), sizeof(UInt64));
        EXPECT_EQ(words[i], referenceWord({number_bytes, strings[i]})) << "row " << i;
    }
}

TEST(JoinSlotRouting, SlotIdsAgreeWithWordsForEveryBitCount)
{
    constexpr size_t rows = 10007;
    std::vector<UInt64> values;
    std::mt19937_64 gen(42);
    for (size_t i = 0; i < rows; ++i)
        values.push_back(gen());
    auto column = makeUInt64Column(values);
    const auto words = routeWordsOf({column.get()}, rows);
    for (size_t bits = 1; bits <= 8; ++bits)
    {
        std::vector<UInt8> slot_ids(rows);
        JoinSlotRouting::computeJoinSlotIds({column.get()}, rows, bits, slot_ids.data());
        for (size_t i = 0; i < rows; ++i)
            ASSERT_EQ(slot_ids[i], words[i] >> (32 - bits)) << "bits " << bits << " row " << i;
    }
}

TEST(JoinSlotRouting, SlotDistributionIsBalanced)
{
    /// Require max/mean slot fill < 1.5 at 1M rows over 256 slots, for both sequential and
    /// random keys. Sequential keys are the adversarial case for a weak route (a low-bit
    /// selector such as the `hashToSelector` fallback maps them to consecutive slots).
    constexpr size_t rows = 1 << 20;
    constexpr size_t bits = 8;
    constexpr size_t slots = 1 << bits;

    auto check = [&](const std::vector<UInt64> & values, const char * label)
    {
        auto column = makeUInt64Column(values);
        std::vector<UInt8> slot_ids(rows);
        JoinSlotRouting::computeJoinSlotIds({column.get()}, rows, bits, slot_ids.data());
        std::vector<size_t> fill(slots, 0);
        for (UInt8 slot : slot_ids)
            ++fill[slot];
        const double mean = static_cast<double>(rows) / slots;
        const double max_fill = static_cast<double>(*std::ranges::max_element(fill));
        EXPECT_LT(max_fill / mean, 1.5) << label;
    };

    std::vector<UInt64> sequential(rows);
    for (size_t i = 0; i < rows; ++i)
        sequential[i] = i;
    check(sequential, "sequential");

    std::vector<UInt64> random(rows);
    std::mt19937_64 gen(7);
    for (auto & value : random)
        value = gen();
    check(random, "random");
}
