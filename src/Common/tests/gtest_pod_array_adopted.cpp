#include <gtest/gtest.h>

#include <Common/Exception.h>
#include <Common/PODArray.h>

#include <memory>
#include <vector>

using namespace DB;

namespace
{

/// Allocates a buffer sized for `n` UInt64 elements plus the safe-read padding required
/// by PaddedPODArray<UInt64> (see adoption-layer §Constraints and system.md glossary
/// "Adopted byte count"). All bytes are zero-initialised.
std::vector<char> makeBuffer(size_t n)
{
    return std::vector<char>(n * sizeof(UInt64) + PaddedPODArray<UInt64>::pad_right, 0);
}

UInt64 * asU64(std::vector<char> & buf) { return reinterpret_cast<UInt64 *>(buf.data()); }
UInt64 * asU64(char * p) { return reinterpret_cast<UInt64 *>(p); }

}

TEST(PaddedPODArrayAdopted, Construction)
{
    constexpr size_t n = 10;
    auto buf = makeBuffer(n);
    for (size_t i = 0; i < n; ++i)
        asU64(buf)[i] = i * 7;
    int dummy_owner = 0;

    PaddedPODArray<UInt64> arr(asU64(buf), n, &dummy_owner);

    EXPECT_EQ(arr.size(), n);
    EXPECT_EQ(arr.data(), asU64(buf));
    for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(arr[i], i * 7);
}

TEST(PaddedPODArrayAdopted, SizeEmptyCapacity)
{
    constexpr size_t n = 4;
    auto buf = makeBuffer(n);
    int dummy_owner = 0;

    PaddedPODArray<UInt64> arr(asU64(buf), n, &dummy_owner);
    EXPECT_FALSE(arr.empty());
    EXPECT_EQ(arr.size(), n);
    /// capacity() == size() in adopted mode: c_end_of_storage = c_end (the trailing bytes
    /// are reserved for safe-read padding, not for new elements).
    EXPECT_EQ(arr.capacity(), n);

    auto buf0 = makeBuffer(0);
    PaddedPODArray<UInt64> arr0(asU64(buf0), 0, &dummy_owner);
    EXPECT_TRUE(arr0.empty());
    EXPECT_EQ(arr0.size(), 0u);
    EXPECT_EQ(arr0.capacity(), 0u);
}

TEST(PaddedPODArrayAdopted, DeallocIsNoOp)
{
    /// If the adopted PODArray's destructor incorrectly tried to free this heap buffer,
    /// the subsequent write would land on freed memory and ASan/UBSan would catch it.
    constexpr size_t n = 10;
    const size_t buf_bytes = n * sizeof(UInt64) + PaddedPODArray<UInt64>::pad_right;
    auto buf = std::make_unique<char[]>(buf_bytes);
    int dummy_owner = 0;
    {
        PaddedPODArray<UInt64> arr(asU64(buf.get()), n, &dummy_owner);
        EXPECT_EQ(arr.size(), n);
    }
    asU64(buf.get())[0] = 42;
    EXPECT_EQ(asU64(buf.get())[0], 42u);
}

TEST(PaddedPODArrayAdopted, MutatorsThrow)
{
    constexpr size_t n = 4;
    auto buf = makeBuffer(n);
    int dummy_owner = 0;

    /// All mutators below leave the array's adopted state intact (the guard fires before
    /// any mutation), so we can exercise many methods on a single instance.
    PaddedPODArray<UInt64> arr(asU64(buf), n, &dummy_owner);
    PaddedPODArray<UInt64> src{1, 2, 3};
    UInt64 v = 7;

    EXPECT_THROW(arr.push_back(1), DB::Exception);
    EXPECT_THROW(arr.emplace_back(1), DB::Exception);
    EXPECT_THROW(arr.pop_back(), DB::Exception);
    EXPECT_THROW(arr.clear(), DB::Exception);
    EXPECT_THROW(arr.reserve(100), DB::Exception);
    EXPECT_THROW(arr.reserve_exact(100), DB::Exception);
    EXPECT_THROW(arr.resize(100), DB::Exception);
    EXPECT_THROW(arr.resize_exact(100), DB::Exception);
    EXPECT_THROW(arr.resize_assume_reserved(2), DB::Exception);
    EXPECT_THROW(arr.shrink_to_fit(), DB::Exception);
    EXPECT_THROW(arr.resize_fill(8), DB::Exception);
    EXPECT_THROW(arr.resize_fill(8, UInt64{1}), DB::Exception);
    EXPECT_THROW(arr.push_back_raw(&v), DB::Exception);
    EXPECT_THROW(arr.assign(size_t{2}, UInt64{1}), DB::Exception);
    EXPECT_THROW(arr.assign(src.begin(), src.end()), DB::Exception);
    EXPECT_THROW(arr.assign(src), DB::Exception);
    EXPECT_THROW(arr.insert(src.begin(), src.end()), DB::Exception);
    EXPECT_THROW(arr.insert(arr.begin(), src.begin(), src.end()), DB::Exception);
    EXPECT_THROW(arr.insert_assume_reserved(src.begin(), src.end()), DB::Exception);
    EXPECT_THROW(arr.insertPrepare(src.begin(), src.end()), DB::Exception);
    EXPECT_THROW(arr.insertByOffsets(src, 0, 1), DB::Exception);
    EXPECT_THROW(arr.insertSmallAllowReadWriteOverflow15(src.begin(), src.end()), DB::Exception);
    EXPECT_THROW(arr.insertFromItself(arr.begin(), arr.end()), DB::Exception);
    EXPECT_THROW(arr.erase(arr.begin(), arr.end()), DB::Exception);
    EXPECT_THROW(arr.erase(arr.begin()), DB::Exception);

    /// swap with an owned array on either side must throw.
    PaddedPODArray<UInt64> arr_owned{9};
    EXPECT_THROW(arr.swap(arr_owned), DB::Exception);
    EXPECT_THROW(arr_owned.swap(arr), DB::Exception);

    /// State must remain unchanged after all the failed mutations.
    EXPECT_EQ(arr.size(), n);
    EXPECT_EQ(arr.data(), asU64(buf));
}

TEST(PaddedPODArrayAdopted, CopyViaIteratorMaterializes)
{
    constexpr size_t n = 5;
    auto buf = makeBuffer(n);
    for (size_t i = 0; i < n; ++i)
        asU64(buf)[i] = i + 100;
    int dummy_owner = 0;
    PaddedPODArray<UInt64> adopted(asU64(buf), n, &dummy_owner);

    /// The iterator-range constructor must allocate fresh, owned memory and be unaffected
    /// by adopted-mode guards on the source. This is the standard COW-materialisation path
    /// (see adoption-layer §Materialization-on-mutation — "Safe materialize" row).
    PaddedPODArray<UInt64> owned(adopted.begin(), adopted.end());
    EXPECT_EQ(owned.size(), n);
    EXPECT_NE(owned.data(), adopted.data());
    for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(owned[i], i + 100);

    EXPECT_NO_THROW(owned.push_back(999));
    EXPECT_EQ(owned.back(), 999u);
    EXPECT_EQ(adopted.size(), n);
    EXPECT_EQ(adopted[0], 100u);
}

TEST(PaddedPODArrayAdopted, MoveCtorPreservesAdoptedState)
{
    /// Regression for F7: the PODArray move ctor must not delegate to swap(), because
    /// swap() throws on adopted arrays and the move ctor is noexcept (a throw there
    /// would call std::terminate). Member-wise move must transfer the adopted state
    /// (pointers + external_owner) intact and leave `src` in a valid empty,
    /// non-adopted state so that no buffer is freed when either object is destroyed.
    constexpr size_t n = 10;
    auto buf = makeBuffer(n);
    for (size_t i = 0; i < n; ++i)
        asU64(buf)[i] = i + 1000;
    int owner_marker = 0;

    PaddedPODArray<UInt64> src(asU64(buf), n, &owner_marker);
    UInt64 * orig_data = src.data();

    PaddedPODArray<UInt64> dst(std::move(src));

    EXPECT_EQ(dst.size(), n);
    EXPECT_EQ(dst.data(), orig_data);
    for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(dst[i], i + 1000);

    /// `src` must now be a valid empty PODArray (moved-from contract).
    EXPECT_TRUE(src.empty()); /// NOLINT(bugprone-use-after-move)
    EXPECT_EQ(src.size(), 0u);

    /// `src` is no longer adopted, so its destructor must not throw and the buffer
    /// must survive (ASan/UBSan would catch a stray free here).
    EXPECT_NO_THROW(src.push_back(7));
    EXPECT_EQ(src.back(), 7u);
}

TEST(PaddedPODArrayAdopted, MoveAssignAdoptedToOwnedNoTerminate)
{
    /// Regression for F7: the move-assignment operator must not delegate to swap();
    /// it must instead release any storage *this currently owns and member-wise move
    /// from `src`. Crucially, std::terminate must not fire when `src` is adopted.
    constexpr size_t n = 10;
    auto buf = makeBuffer(n);
    for (size_t i = 0; i < n; ++i)
        asU64(buf)[i] = i + 2000;
    int owner_marker = 0;
    PaddedPODArray<UInt64> src(asU64(buf), n, &owner_marker);
    UInt64 * orig_data = src.data();

    PaddedPODArray<UInt64> dst;
    dst.push_back(99);
    EXPECT_EQ(dst.size(), 1u);

    EXPECT_NO_FATAL_FAILURE(dst = std::move(src));

    EXPECT_EQ(dst.size(), n);
    EXPECT_EQ(dst.data(), orig_data);
    EXPECT_EQ(dst[0], 2000u);
    EXPECT_TRUE(src.empty()); /// NOLINT(bugprone-use-after-move)
}

TEST(PaddedPODArrayAdopted, MoveAssignOwnedToAdoptedNoTerminate)
{
    /// Move-assigning an OWNED PODArray into an ADOPTED target. The pre-existing
    /// adopted state of `dst` is discarded (its dealloc() is a no-op per round-1 F3);
    /// `dst` then becomes owned with `src`'s storage. The whole path must run without
    /// any throw, since the move-assignment operator is noexcept.
    constexpr size_t n = 5;
    auto buf = makeBuffer(n);
    int owner_marker = 0;
    PaddedPODArray<UInt64> dst(asU64(buf), n, &owner_marker);

    PaddedPODArray<UInt64> src;
    src.push_back(42);
    src.push_back(43);
    UInt64 * src_data = src.data();

    EXPECT_NO_FATAL_FAILURE(dst = std::move(src));

    EXPECT_EQ(dst.size(), 2u);
    EXPECT_EQ(dst.data(), src_data);
    EXPECT_EQ(dst[0], 42u);
    EXPECT_EQ(dst[1], 43u);
    EXPECT_TRUE(src.empty()); /// NOLINT(bugprone-use-after-move)

    /// `buf` is still alive (we own it via std::vector). Nothing freed it.
    EXPECT_EQ(asU64(buf), reinterpret_cast<UInt64 *>(buf.data()));
}

TEST(PaddedPODArrayAdopted, NormalArrayUnchanged)
{
    /// Regression: default-constructed PODArray must still behave normally now that
    /// PODArrayBase has an extra `external_owner` member.
    PaddedPODArray<UInt64> normal;
    EXPECT_EQ(normal.size(), 0u);
    EXPECT_TRUE(normal.empty());

    normal.push_back(1);
    normal.push_back(2);
    EXPECT_EQ(normal.size(), 2u);
    EXPECT_EQ(normal[0], 1u);
    EXPECT_EQ(normal[1], 2u);

    normal.reserve(64);
    EXPECT_GE(normal.capacity(), 64u);

    normal.resize(5);
    EXPECT_EQ(normal.size(), 5u);
    normal[3] = 33;
    normal[4] = 44;
    EXPECT_EQ(normal[3], 33u);
    EXPECT_EQ(normal[4], 44u);

    normal.clear();
    EXPECT_TRUE(normal.empty());

    PaddedPODArray<UInt64> other{1, 2, 3, 4};
    normal.swap(other);
    EXPECT_EQ(normal.size(), 4u);
    EXPECT_TRUE(other.empty());
}
