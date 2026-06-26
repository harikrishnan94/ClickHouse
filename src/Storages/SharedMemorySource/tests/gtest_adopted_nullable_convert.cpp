#include <gtest/gtest.h>

#if defined(OS_LINUX)

#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <Common/Exception.h>

#include <cstdlib>
#include <cstring>
#include <memory>

using namespace DB;

namespace
{
/// A non-null, never-dereferenced token standing in for the SHM/recv RetainToken + ChargeHandle.
std::shared_ptr<void> dummyHandle() { return std::shared_ptr<void>(new int(0)); }

/// Allocate a 64-byte-aligned buffer of `n` UInt64 with >= PADDING_FOR_SIMD trailing slack and fill it.
UInt64 * makeAdoptedUInt64Buf(size_t n)
{
    const size_t cap = ((n * sizeof(UInt64) + 64 + 63) / 64) * 64;
    auto * p = static_cast<UInt64 *>(::aligned_alloc(64, cap));
    std::memset(p, 0, cap);
    for (size_t i = 0; i < n; ++i)
        p[i] = 1000 + i;
    return p;
}
}

/// D-HC-0206: an adopted ColumnVector materializes (becomes a mutable owned column) on
/// convertToFullColumnIfAdopted, and mutating the ORIGINAL adopted column throws READONLY.
TEST(AdoptedConvert, FixedWidthMaterializesAndIsMutable)
{
    constexpr size_t n = 5;
    UInt64 * buf = makeAdoptedUInt64Buf(n);
    auto retain = std::shared_ptr<void>(buf, [](void * pp) noexcept { ::free(pp); });
    auto adopted = ColumnVector<UInt64>::createAdopted(buf, n, retain, dummyHandle());

    /// Mutating the adopted column itself must throw (read-only producer/recv memory).
    EXPECT_ANY_THROW(adopted->assumeMutable()->insertDefault());

    ColumnPtr full = adopted->convertToFullColumnIfAdopted();
    ASSERT_NE(full.get(), adopted.get());   /// materialized a new column
    const auto & fv = assert_cast<const ColumnVector<UInt64> &>(*full);
    ASSERT_EQ(fv.size(), n);
    for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(fv.getData()[i], 1000 + i);
    /// The materialized column is owned + mutable (no READONLY throw).
    EXPECT_NO_THROW(full->assumeMutable()->insertDefault());
}

/// D-HC-0206: a Nullable WRAPPING an adopted nested column recurses — the default IColumn impl does
/// not, so without the override the nested would stay adopted and the mutating callers
/// (Squashing.cpp:346 / HashJoin.cpp:126) would throw READONLY. After the override it materializes.
TEST(AdoptedConvert, NullableRecursesIntoAdoptedNested)
{
    constexpr size_t n = 4;
    UInt64 * buf = makeAdoptedUInt64Buf(n);
    auto retain = std::shared_ptr<void>(buf, [](void * pp) noexcept { ::free(pp); });
    auto adopted = ColumnVector<UInt64>::createAdopted(buf, n, retain, dummyHandle());

    auto null_map = ColumnUInt8::create();
    for (int v : {0, 1, 0, 1})                   /// owned byte map (bitmap->bytemap transform)
        null_map->getData().push_back(static_cast<UInt8>(v));
    ColumnPtr nullable = ColumnNullable::create(std::move(adopted), std::move(null_map));

    ColumnPtr full = nullable->convertToFullColumnIfAdopted();
    ASSERT_NE(full.get(), nullable.get());   /// recursed + rebuilt (nested was adopted)
    const auto & fn = assert_cast<const ColumnNullable &>(*full);
    ASSERT_EQ(fn.size(), n);
    EXPECT_TRUE(fn.isNullAt(1));
    EXPECT_FALSE(fn.isNullAt(0));
    EXPECT_EQ(assert_cast<const ColumnVector<UInt64> &>(fn.getNestedColumn()).getData()[2], 1002u);
    /// The materialized Nullable is mutable end-to-end (the mutating callers' contract).
    EXPECT_NO_THROW(full->assumeMutable()->insertDefault());
}

#endif
