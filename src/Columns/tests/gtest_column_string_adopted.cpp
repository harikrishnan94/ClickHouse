#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Columns/ColumnString.h>
#include <Columns/IColumn.h>
#include <Columns/IColumn_fwd.h>
#include <Common/Exception.h>
#include <Common/PODArray.h>
#include <Common/assert_cast.h>
#include <Storages/SharedMemorySource/Adoption/RetainToken.h>

using namespace DB;

namespace
{

/// Backs an adopted ColumnString with caller-owned byte buffers that mimic the producer-side
/// layout contract described in ColumnString::createAdopted's docstring:
///   - chars buffer has trailing safe-read padding of PaddedPODArray<UInt8>::pad_right
///   - offsets buffer is preceded by an 8-byte zero sentinel (the offsets[-1] slot that
///     ColumnString::offsetAt(0) reads), and followed by PaddedPODArray<UInt64>::pad_right
///     bytes of trailing safe-read padding.
/// Witness counters fire from inside the retain/charge release callbacks, so they observe
/// destruction order: charge first (declared last → destroyed first), then retain.
struct AdoptedStringFixture
{
    std::vector<std::string> strings{"", "a", "bb", "ccc", "dddd"};
    std::vector<char> chars_buf;
    std::vector<char> offsets_buf;
    UInt8 * chars_ptr = nullptr;
    UInt64 * offsets_ptr = nullptr;
    int retain_witness = 0;
    int charge_witness = 0;

    AdoptedStringFixture()
    {
        const size_t rows = strings.size();
        const size_t total_chars = totalChars();

        chars_buf.assign(total_chars + PaddedPODArray<UInt8>::pad_right, 0);
        size_t off = 0;
        for (const auto & s : strings)
        {
            if (!s.empty())
                std::memcpy(chars_buf.data() + off, s.data(), s.size());
            off += s.size();
        }
        chars_ptr = reinterpret_cast<UInt8 *>(chars_buf.data());

        offsets_buf.assign(sizeof(UInt64) + rows * sizeof(UInt64) + PaddedPODArray<UInt64>::pad_right, 0);
        offsets_ptr = reinterpret_cast<UInt64 *>(offsets_buf.data() + sizeof(UInt64));
        UInt64 cum = 0;
        for (size_t i = 0; i < rows; ++i)
        {
            cum += strings[i].size();
            offsets_ptr[i] = cum;
        }
    }

    size_t totalChars() const
    {
        size_t total = 0;
        for (const auto & s : strings)
            total += s.size();
        return total;
    }

    ColumnString::MutablePtr makeColumn()
    {
        auto retain = makeRetainToken([this]() noexcept { ++retain_witness; });
        auto charge = makeRetainToken([this]() noexcept { ++charge_witness; });
        return ColumnString::createAdopted(
            chars_ptr, totalChars(),
            offsets_ptr, strings.size(),
            std::move(retain), std::move(charge));
    }
};

}

TEST(ColumnStringAdopted, PointerIdentity)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    EXPECT_EQ(col->size(), f.strings.size());
    /// Both non-const getChars() and non-const getOffsets() are guarded on adopted
    /// columns; route through a const reference for the AC3 pointer-identity read.
    const auto & cs = assert_cast<const ColumnString &>(*col);
    EXPECT_EQ(cs.getChars().data(), f.chars_ptr);
    EXPECT_EQ(cs.getOffsets().data(), f.offsets_ptr);
}

TEST(ColumnStringAdopted, ReadValues)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    for (size_t i = 0; i < f.strings.size(); ++i)
    {
        std::string_view got = col->getDataAt(i);
        EXPECT_EQ(got, std::string_view(f.strings[i])) << "row " << i;
    }
}

TEST(ColumnStringAdopted, DestructionReleasesHandles)
{
    AdoptedStringFixture f;
    {
        auto col = f.makeColumn();
        EXPECT_EQ(f.retain_witness, 0);
        EXPECT_EQ(f.charge_witness, 0);
    }
    EXPECT_EQ(f.retain_witness, 1);
    EXPECT_EQ(f.charge_witness, 1);
}

TEST(ColumnStringAdopted, FactoryRejectsNullHandles)
{
    AdoptedStringFixture f;
    auto good = makeRetainToken([]() noexcept {});
    EXPECT_THROW(
        ColumnString::createAdopted(f.chars_ptr, f.totalChars(), f.offsets_ptr, f.strings.size(),
                                    /*retain_token=*/{}, good),
        DB::Exception);
    EXPECT_THROW(
        ColumnString::createAdopted(f.chars_ptr, f.totalChars(), f.offsets_ptr, f.strings.size(),
                                    good, /*charge_handle=*/{}),
        DB::Exception);
}

TEST(ColumnStringAdopted, CowMutateMaterializes)
{
    AdoptedStringFixture f;
    ColumnPtr col = f.makeColumn();
    ColumnPtr alias = col;
    EXPECT_EQ(col->use_count(), 2u);

    auto mutated = IColumn::mutate(std::move(col));
    auto & cs_mut = assert_cast<ColumnString &>(*mutated);

    EXPECT_NE(cs_mut.getChars().data(), f.chars_ptr);
    EXPECT_NE(cs_mut.getOffsets().data(), f.offsets_ptr);
    EXPECT_EQ(cs_mut.size(), f.strings.size());
    for (size_t i = 0; i < f.strings.size(); ++i)
        EXPECT_EQ(std::string_view(cs_mut.getDataAt(i)), std::string_view(f.strings[i]));

    EXPECT_NO_THROW(cs_mut.insertData("zzzzzz", 6));
    EXPECT_EQ(cs_mut.size(), f.strings.size() + 1);
    EXPECT_EQ(std::string_view(cs_mut.getDataAt(f.strings.size())), std::string_view("zzzzzz"));

    EXPECT_EQ(f.retain_witness, 0);
    alias.reset();
    EXPECT_EQ(f.retain_witness, 1);
    EXPECT_EQ(f.charge_witness, 1);
}

TEST(ColumnStringAdopted, NonConstAccessorsThrow)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->getChars(), DB::Exception);
    EXPECT_THROW(col->getOffsets(), DB::Exception);
}

TEST(ColumnStringAdopted, ValidateAdoptedOffsetsAcceptsWellFormed)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    EXPECT_NO_THROW(col->validateAdoptedOffsets());
}

TEST(ColumnStringAdopted, ValidateAdoptedOffsetsRejectsMonotonicityViolation)
{
    AdoptedStringFixture f;
    f.offsets_ptr[2] = 0;
    auto col = f.makeColumn();
    EXPECT_THROW(col->validateAdoptedOffsets(), DB::Exception);
}

TEST(ColumnStringAdopted, ValidateAdoptedOffsetsRejectsTerminalMismatch)
{
    AdoptedStringFixture f;
    f.offsets_ptr[f.strings.size() - 1] += 1;
    auto col = f.makeColumn();
    EXPECT_THROW(col->validateAdoptedOffsets(), DB::Exception);
}

/// F3 round-2 coverage: every direct ColumnString mutator on an adopted column must
/// throw before touching producer-owned storage. Spec authority: adoption-layer
/// §Materialization-on-mutation contract, I3.

TEST(ColumnStringAdopted, InsertFieldThrows)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->insert(Field{String{"x"}}), DB::Exception);
    EXPECT_THROW(col->tryInsert(Field{String{"x"}}), DB::Exception);
}

TEST(ColumnStringAdopted, InsertFromOwnedSourceThrows)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    auto src = ColumnString::create();
    src->insertData("hello", 5);
    EXPECT_THROW(col->insertFrom(*src, 0), DB::Exception);
    EXPECT_THROW(col->insertManyFrom(*src, 0, 3), DB::Exception);
    EXPECT_THROW(col->insertRangeFrom(*src, 0, 1), DB::Exception);
}

TEST(ColumnStringAdopted, InsertDataAndDefaultsThrow)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->insertData("zzz", 3), DB::Exception);
    EXPECT_THROW(col->insertDefault(), DB::Exception);
    EXPECT_THROW(col->insertManyDefaults(4), DB::Exception);
}

TEST(ColumnStringAdopted, PopBackThrows)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->popBack(1), DB::Exception);
}

TEST(ColumnStringAdopted, InPlaceFilterThrows)
{
    /// Sibling const overload `filter(Filter, ssize_t) const` produces a new column and
    /// is intentionally not guarded.
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    IColumn::Filter mask(f.strings.size(), 1);
    EXPECT_THROW(col->filter(mask), DB::Exception);
}

TEST(ColumnStringAdopted, ExpandThrows)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    IColumn::Filter mask(f.strings.size(), 1);
    EXPECT_THROW(col->expand(mask, false), DB::Exception);
}

TEST(ColumnStringAdopted, ReserveAndShrinkThrow)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->reserve(1024), DB::Exception);
    EXPECT_THROW(col->shrinkToFit(), DB::Exception);
}

TEST(ColumnStringAdopted, PrepareForSquashingThrows)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    VectorWithMemoryTracking<ColumnPtr> srcs;
    EXPECT_THROW(col->prepareForSquashing(srcs, 1), DB::Exception);
}

TEST(ColumnStringAdopted, RollbackThrows)
{
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    auto cp = col->getCheckpoint();
    EXPECT_THROW(col->rollback(*cp), DB::Exception);
}

TEST(ColumnStringAdopted, ReadOnlyMethodsDoNotThrow)
{
    /// Regression sentinel: const accessors used by AC1 read-path hot lanes must
    /// continue to work on adopted instances. If this starts failing, a guard has been
    /// added to a method that should remain unguarded.
    AdoptedStringFixture f;
    auto col = f.makeColumn();
    EXPECT_NO_THROW((void)col->size());
    EXPECT_NO_THROW((void)col->byteSize());
    EXPECT_NO_THROW((void)col->allocatedBytes());
    EXPECT_NO_THROW((void)col->getDataAt(0));
    const auto & cs = assert_cast<const ColumnString &>(*col);
    EXPECT_NO_THROW((void)cs.getChars().data());
    EXPECT_NO_THROW((void)cs.getOffsets().data());
    IColumn::Filter mask(f.strings.size(), 1);
    EXPECT_NO_THROW((void)cs.filter(mask, -1));
}
