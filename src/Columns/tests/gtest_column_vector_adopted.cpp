#include <gtest/gtest.h>

#include <vector>

#include <Columns/ColumnVector.h>
#include <Columns/IColumn.h>
#include <Common/Exception.h>
#include <Common/PODArray.h>
#include <Common/assert_cast.h>
#include <Storages/SharedMemorySource/Adoption/RetainToken.h>

/// Tests for ColumnVector<UInt64>::createAdopted (T2.1 of the zero-copy SHM source feature).
/// Spec authority: adoption-layer spec Interfaces & contracts (Adopt entry point), I1, I3, I4;
/// system spec I5, I10; memory-tracker-integration spec I7.

using namespace DB;

namespace
{

struct AdoptedFixture
{
    static constexpr size_t N = 100;

    /// Backing store sized to satisfy PaddedPODArray<UInt64>'s safely-readable trailing
    /// padding contract (PODArrayBase::pad_right). std::vector<char>'s default allocator
    /// supplies at least alignof(std::max_align_t) alignment, which covers alignof(UInt64).
    std::vector<char> backing_buffer;
    UInt64 * data_ptr = nullptr;

    /// Witnesses incremented by the corresponding handle's release callback at final drop.
    int retain_witness = 0;
    int charge_witness = 0;

    AdoptedFixture()
        : backing_buffer(N * sizeof(UInt64) + PaddedPODArray<UInt64>::pad_right, '\0')
    {
        data_ptr = reinterpret_cast<UInt64 *>(backing_buffer.data());
        for (size_t i = 0; i < N; ++i)
            data_ptr[i] = i * 31 + 7;
    }

    auto makeColumn()
    {
        auto retain = makeRetainToken([this]() noexcept { ++retain_witness; });
        auto charge = makeRetainToken([this]() noexcept { ++charge_witness; });
        return ColumnVector<UInt64>::createAdopted(data_ptr, N, std::move(retain), std::move(charge));
    }
};

}

TEST(ColumnVectorAdopted, PointerIdentity)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    EXPECT_EQ(col->size(), AdoptedFixture::N);
    /// AC3 pointer-identity at the IColumn level: value buffer is producer memory.
    /// Route through a const reference because both non-const `getData()` and non-const
    /// `getElement()` are guarded (per F3 round-2: any path yielding a writable handle
    /// to producer memory throws); only the const overloads are AC1 read-path safe.
    const auto & cv = assert_cast<const ColumnVector<UInt64> &>(*col);
    EXPECT_EQ(cv.getData().data(), f.data_ptr);
    for (size_t i = 0; i < AdoptedFixture::N; ++i)
        EXPECT_EQ(cv.getElement(i), i * 31 + 7);
}

TEST(ColumnVectorAdopted, SizeAndConstAccessorsWork)
{
    AdoptedFixture f;
    ColumnPtr col = f.makeColumn();
    const auto & cv = assert_cast<const ColumnVector<UInt64> &>(*col);
    EXPECT_EQ(cv.size(), AdoptedFixture::N);
    EXPECT_EQ(cv.getData().size(), AdoptedFixture::N);
    EXPECT_EQ(cv.getData().data(), f.data_ptr);
    EXPECT_EQ(cv.getElement(42), 42 * 31 + 7);
}

TEST(ColumnVectorAdopted, DestructionReleasesHandles)
{
    AdoptedFixture f;
    {
        auto col = f.makeColumn();
        EXPECT_EQ(f.retain_witness, 0);
        EXPECT_EQ(f.charge_witness, 0);
    }
    /// I5 + I7: both handles release exactly once at final drop.
    EXPECT_EQ(f.retain_witness, 1);
    EXPECT_EQ(f.charge_witness, 1);
}

TEST(ColumnVectorAdopted, CowMutateMaterializes)
{
    AdoptedFixture f;
    ColumnPtr adopted_const = f.makeColumn();
    ColumnPtr alias = adopted_const; /// bump refcount so mutate triggers clone()
    EXPECT_EQ(adopted_const->use_count(), 2u);

    auto mutated = IColumn::mutate(std::move(adopted_const));
    auto & cv_mut = assert_cast<ColumnVector<UInt64> &>(*mutated);

    /// I3 + adoption-layer Materialization-on-mutation: the COW clone owns a heap buffer
    /// distinct from the producer pointer.
    EXPECT_NE(cv_mut.getData().data(), f.data_ptr);
    EXPECT_EQ(cv_mut.size(), AdoptedFixture::N);
    for (size_t i = 0; i < AdoptedFixture::N; ++i)
        EXPECT_EQ(cv_mut.getElement(i), i * 31 + 7);

    /// The owned clone is fully mutable.
    EXPECT_NO_THROW(cv_mut.getData().push_back(999));
    EXPECT_EQ(cv_mut.size(), AdoptedFixture::N + 1);
}

TEST(ColumnVectorAdopted, NonConstGetDataThrows)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    /// VC1 guard: non-const getData() on an adopted column is misuse per spec I3.
    EXPECT_THROW(col->getData(), DB::Exception);
}

TEST(ColumnVectorAdopted, DirectPodArrayMutationThrows)
{
    /// Defense in depth: even if a caller reaches past the column-level guard via const_cast,
    /// PODArray's adopted-mode mutator guards fire (T1.3).
    AdoptedFixture f;
    auto col = f.makeColumn();
    const auto & const_pod = static_cast<const ColumnVector<UInt64> &>(*col).getData();
    auto & pod = const_cast<PaddedPODArray<UInt64> &>(const_pod);
    EXPECT_THROW(pod.push_back(1), DB::Exception);
    EXPECT_THROW(pod.reserve(100), DB::Exception);
}

TEST(ColumnVectorAdopted, FactoryRejectsNullRetainToken)
{
    AdoptedFixture f;
    auto charge = makeRetainToken([]() noexcept {});
    EXPECT_THROW(
        ColumnVector<UInt64>::createAdopted(f.data_ptr, AdoptedFixture::N, nullptr, std::move(charge)),
        DB::Exception);
}

TEST(ColumnVectorAdopted, FactoryRejectsNullChargeHandle)
{
    AdoptedFixture f;
    auto retain = makeRetainToken([]() noexcept {});
    EXPECT_THROW(
        ColumnVector<UInt64>::createAdopted(f.data_ptr, AdoptedFixture::N, std::move(retain), nullptr),
        DB::Exception);
}

TEST(ColumnVectorAdopted, FactoryRejectsNonUInt64)
{
    constexpr size_t n = 4;
    std::vector<char> backing_buffer(n * sizeof(UInt32) + PaddedPODArray<UInt32>::pad_right, '\0');
    auto * data = reinterpret_cast<UInt32 *>(backing_buffer.data());
    auto retain = makeRetainToken([]() noexcept {});
    auto charge = makeRetainToken([]() noexcept {});

    EXPECT_THROW(
        ColumnVector<UInt32>::createAdopted(data, n, std::move(retain), std::move(charge)),
        DB::Exception);
}

/// F3 round-2 coverage: every direct ColumnVector mutator on an adopted column must
/// throw before touching producer-owned storage. Each test below builds a small owned
/// source column where needed and asserts the corresponding adopted mutator throws.
/// Spec authority: adoption-layer §Materialization-on-mutation contract, I3.

TEST(ColumnVectorAdopted, NonConstGetElementThrows)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->getElement(0), DB::Exception);
}

TEST(ColumnVectorAdopted, InsertFieldThrows)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->insert(Field{UInt64{42}}), DB::Exception);
}

TEST(ColumnVectorAdopted, InsertManyThrows)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->insertMany(Field{UInt64{42}}, 4), DB::Exception);
}

TEST(ColumnVectorAdopted, InsertDataThrows)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    UInt64 v = 7;
    EXPECT_THROW(col->insertData(reinterpret_cast<const char *>(&v), sizeof(v)), DB::Exception);
}

TEST(ColumnVectorAdopted, InsertDefaultsAndPopBackThrow)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->insertDefault(), DB::Exception);
    EXPECT_THROW(col->insertManyDefaults(3), DB::Exception);
    EXPECT_THROW(col->popBack(1), DB::Exception);
}

TEST(ColumnVectorAdopted, InsertFromOwnedSourceThrows)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    auto src = ColumnVector<UInt64>::create();
    src->insertValue(123);
    EXPECT_THROW(col->insertFrom(*src, 0), DB::Exception);
    EXPECT_THROW(col->insertManyFrom(*src, 0, 2), DB::Exception);
    EXPECT_THROW(col->insertRangeFrom(*src, 0, 1), DB::Exception);
}

TEST(ColumnVectorAdopted, TryInsertThrows)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->tryInsert(Field{UInt64{1}}), DB::Exception);
}

TEST(ColumnVectorAdopted, InPlaceFilterThrows)
{
    /// In-place `filter(Filter)` writes through `this->data`; the const
    /// `filter(Filter, ssize_t) const` overload that produces a NEW column is intentionally
    /// NOT guarded and continues to work (covered indirectly by other read tests).
    AdoptedFixture f;
    auto col = f.makeColumn();
    IColumn::Filter mask(AdoptedFixture::N, 1);
    EXPECT_THROW(col->filter(mask), DB::Exception);
}

TEST(ColumnVectorAdopted, ExpandAndApplyZeroMapThrow)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    IColumn::Filter mask(AdoptedFixture::N, 1);
    EXPECT_THROW(col->expand(mask, false), DB::Exception);
    EXPECT_THROW(col->applyZeroMap(mask, false), DB::Exception);
}

TEST(ColumnVectorAdopted, UpdateAtThrows)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    auto src = ColumnVector<UInt64>::create();
    src->insertValue(999);
    EXPECT_THROW(col->updateAt(*src, 0, 0), DB::Exception);
}

TEST(ColumnVectorAdopted, ReserveAndShrinkAndInsertRawUninitializedThrow)
{
    AdoptedFixture f;
    auto col = f.makeColumn();
    EXPECT_THROW(col->reserve(1024), DB::Exception);
    EXPECT_THROW(col->shrinkToFit(), DB::Exception);
    EXPECT_THROW(col->insertRawUninitialized(8), DB::Exception);
}

TEST(ColumnVectorAdopted, ReadOnlyMethodsDoNotThrow)
{
    /// Regression sentinel: const accessors used by aggregation/format hot paths must
    /// continue to work even on adopted instances. If this test starts failing, a guard
    /// has been added to a method that should remain unguarded.
    AdoptedFixture f;
    auto col = f.makeColumn();
    EXPECT_NO_THROW((void)col->size());
    EXPECT_NO_THROW((void)col->byteSize());
    EXPECT_NO_THROW((void)col->allocatedBytes());
    EXPECT_NO_THROW((void)col->getDataAt(0));
    /// const overload of getElement is the AC1 hot path — must remain unguarded.
    const auto & cv = assert_cast<const ColumnVector<UInt64> &>(*col);
    EXPECT_NO_THROW((void)cv.getElement(0));
    EXPECT_NO_THROW((void)cv.getData().data());
    /// Const filter overload is safe (produces a new owned column).
    IColumn::Filter mask(AdoptedFixture::N, 1);
    EXPECT_NO_THROW((void)cv.filter(mask, -1));
}
