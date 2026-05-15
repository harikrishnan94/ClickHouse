#include <gtest/gtest.h>

#include <Storages/SharedMemorySource/Adoption/AdoptionLayer.h>
#include <Storages/SharedMemorySource/Adoption/RetainToken.h>
#include <Storages/SharedMemorySource/Tracker/ChargeHandle.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>

#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/IColumn.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Common/Exception.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>


using namespace DB;
using DB::SharedMemoryWire::ColumnDescriptor;
using DB::SharedMemoryWire::WireColumnType;
namespace SHM = DB::SharedMemoryWire;

namespace DB::ErrorCodes
{
    extern const int SHM_SCHEMA_MISMATCH;
    extern const int SHM_BUFFER_LAYOUT_INVALID;
}


namespace
{

/// One UInt64 column at offset 0, 5 rows, with PADDING_FOR_SIMD trailing safe-read padding.
/// All fixtures own the buffer so the adopted columns may safely point into it for the
/// duration of the test. The retain witness and charge witness fire from RAII releasers, so
/// gtest can assert that on the exception path both releasers run exactly once.
struct AdoptFixture
{
    static constexpr size_t ROWS = 5;
    /// Data region layout: UInt64[5] @ 0, padding [40,40+64), then ColumnString chars[6] @
    /// 128, then offsets pre-sentinel @ next_8B, then offsets[5] @ +8, then padding.
    /// Numbers are conservative; ample headroom for ASan-aware tests.
    static constexpr size_t REGION_SIZE = 4096;

    alignas(64) std::vector<char> region;
    int retain_witness = 0;
    int charge_witness_counter = 0; // counts ChargeHandle dtor invocations (release()).
    std::atomic<int64_t> tracker_counter{0};

    std::vector<UInt64> id_values{10, 20, 30, 40, 50};
    std::vector<std::string> str_values{"", "a", "bb", "ccc", "dddd"};

    /// Per-column descriptor offsets (within the data region).
    uint64_t id_value_offset = 0;
    uint64_t str_chars_offset = 0;
    uint64_t str_offsets_offset = 0;
    uint64_t str_chars_size = 0;

    AdoptFixture()
    {
        region.assign(REGION_SIZE, 0);

        /// id column — UInt64 buffer at offset 0.
        id_value_offset = 0;
        std::memcpy(region.data() + id_value_offset, id_values.data(), id_values.size() * sizeof(UInt64));

        /// String chars start at offset 128 (well past the UInt64 buffer + padding).
        str_chars_offset = 128;
        size_t cursor = str_chars_offset;
        for (const auto & s : str_values)
        {
            if (!s.empty())
                std::memcpy(region.data() + cursor, s.data(), s.size());
            cursor += s.size();
        }
        str_chars_size = cursor - str_chars_offset;

        /// Skip ahead past PADDING_FOR_SIMD, then the 8-byte pre-sentinel, then offsets.
        cursor = str_chars_offset + str_chars_size + SHM::PADDING_FOR_SIMD;
        cursor = (cursor + 7) & ~size_t{7}; // align to 8 for offsets buffer start
        /// 8-byte zero sentinel preceding offsets[0]; offsets[-1] read by ColumnString::offsetAt(0).
        std::memset(region.data() + cursor, 0, sizeof(UInt64));
        str_offsets_offset = cursor + sizeof(UInt64);

        uint64_t cum = 0;
        for (size_t i = 0; i < str_values.size(); ++i)
        {
            cum += str_values[i].size();
            auto * out = reinterpret_cast<UInt64 *>(region.data() + str_offsets_offset) + i;
            *out = cum;
        }
    }

    std::vector<std::pair<std::string, DataTypePtr>> schema() const
    {
        return {{"id", std::make_shared<DataTypeUInt64>()},
                {"s",  std::make_shared<DataTypeString>()}};
    }

    std::vector<ColumnDescriptor> descriptors() const
    {
        std::vector<ColumnDescriptor> ds(2);

        ds[0].type = static_cast<uint32_t>(WireColumnType::UInt64);
        ds[0].value_offset = id_value_offset;
        ds[0].value_count = ROWS;
        ds[0].value_padding = SHM::PADDING_FOR_SIMD;

        ds[1].type = static_cast<uint32_t>(WireColumnType::String);
        ds[1].value_offset = str_chars_offset;
        ds[1].value_count = str_chars_size;
        ds[1].value_padding = SHM::PADDING_FOR_SIMD;
        ds[1].offsets_offset = str_offsets_offset;
        ds[1].offsets_count = ROWS;
        ds[1].offsets_padding = SHM::PADDING_FOR_SIMD;

        return ds;
    }

    RetainToken makeRetain()
    {
        return makeRetainToken([this]() noexcept { ++retain_witness; });
    }

    ChargeHandle makeCharge()
    {
        /// Pre-bump the counter so the ChargeHandle's dtor decrements it back to zero.
        /// (The dtor releases against the captured tracker — nullptr here, so it falls back
        /// to CurrentMemoryTracker::free which is a no-op in a gtest binary without
        /// MainThreadStatus — then atomically fetch_subs this counter.)
        constexpr int64_t bytes = 1024;
        tracker_counter.fetch_add(bytes, std::memory_order_acq_rel);
        return ChargeHandle{static_cast<size_t>(bytes), static_cast<size_t>(bytes),
                            &tracker_counter, /*tracker_at_charge=*/nullptr};
    }
};

}


TEST(AdoptionLayer, UInt64ColumnAdoptedSuccessfully)
{
    AdoptFixture f;
    Columns cols = adopt(f.descriptors(), f.schema(),
                         f.region.data(), AdoptFixture::REGION_SIZE,
                         AdoptFixture::ROWS, f.makeRetain(), f.makeCharge());

    ASSERT_EQ(cols.size(), 2u);
    /// VC2 gotcha: pull out adopted state via the const accessor on ColumnPtr.
    const ColumnPtr & id_col = cols[0];
    const auto * cv = typeid_cast<const ColumnUInt64 *>(id_col.get());
    ASSERT_NE(cv, nullptr);
    EXPECT_EQ(cv->size(), AdoptFixture::ROWS);
    /// AC3 pointer-identity: the column's value buffer .data() equals the descriptor
    /// offset applied to the producer data region base.
    EXPECT_EQ(reinterpret_cast<const char *>(cv->getData().data()),
              f.region.data() + f.id_value_offset);
    for (size_t i = 0; i < AdoptFixture::ROWS; ++i)
        EXPECT_EQ(cv->getData()[i], f.id_values[i]) << "row " << i;
}


TEST(AdoptionLayer, StringColumnAdoptedSuccessfully)
{
    AdoptFixture f;
    Columns cols = adopt(f.descriptors(), f.schema(),
                         f.region.data(), AdoptFixture::REGION_SIZE,
                         AdoptFixture::ROWS, f.makeRetain(), f.makeCharge());

    ASSERT_EQ(cols.size(), 2u);
    const ColumnPtr & str_col = cols[1];
    const auto * cs = typeid_cast<const ColumnString *>(str_col.get());
    ASSERT_NE(cs, nullptr);
    EXPECT_EQ(cs->size(), AdoptFixture::ROWS);
    /// AC3 pointer-identity for ColumnString — chars AND offsets buffers must adopt-point.
    EXPECT_EQ(reinterpret_cast<const char *>(cs->getChars().data()),
              f.region.data() + f.str_chars_offset);
    EXPECT_EQ(reinterpret_cast<const char *>(cs->getOffsets().data()),
              f.region.data() + f.str_offsets_offset);
    for (size_t i = 0; i < AdoptFixture::ROWS; ++i)
        EXPECT_EQ(std::string(cs->getDataAt(i)), f.str_values[i]) << "row " << i;
}


TEST(AdoptionLayer, RetainSharedAcrossAllColumns)
{
    AdoptFixture f;
    Columns cols = adopt(f.descriptors(), f.schema(),
                         f.region.data(), AdoptFixture::REGION_SIZE,
                         AdoptFixture::ROWS, f.makeRetain(), f.makeCharge());

    /// Drop the local Columns to release the columns' references. The retain witness must
    /// then fire exactly once - regardless of how many columns shared the token.
    EXPECT_EQ(f.retain_witness, 0);
    cols.clear();
    EXPECT_EQ(f.retain_witness, 1);
    /// Charge counter is also released (ChargeHandle's dtor decremented tracker_counter).
    EXPECT_EQ(f.tracker_counter.load(), 0);
}


TEST(AdoptionLayer, MisalignedUInt64DescriptorRejected)
{
    AdoptFixture f;
    auto ds = f.descriptors();
    ds[0].value_offset = 1; // misaligned for UInt64 (precondition 13)

    EXPECT_EQ(f.retain_witness, 0);
    EXPECT_THROW(
        adopt(ds, f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge()),
        DB::Exception);
    /// Both handles released on the exception path (RAII rollback).
    EXPECT_EQ(f.retain_witness, 1);
    EXPECT_EQ(f.tracker_counter.load(), 0);
}


TEST(AdoptionLayer, MisalignedStringOffsetsRejected)
{
    AdoptFixture f;
    auto ds = f.descriptors();
    ds[1].offsets_offset += 1; // misaligned for UInt64 (precondition 17)

    EXPECT_THROW(
        adopt(ds, f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge()),
        DB::Exception);
    EXPECT_EQ(f.retain_witness, 1);
    EXPECT_EQ(f.tracker_counter.load(), 0);
}


TEST(AdoptionLayer, OffsetOverflowRejected)
{
    AdoptFixture f;
    auto ds = f.descriptors();
    ds[0].value_offset = AdoptFixture::REGION_SIZE; // value + bytes overflows

    EXPECT_THROW(
        adopt(ds, f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge()),
        DB::Exception);
    EXPECT_EQ(f.retain_witness, 1);
}


TEST(AdoptionLayer, PaddingShortfallRejected)
{
    AdoptFixture f;
    auto ds = f.descriptors();
    ds[0].value_padding = SHM::PADDING_FOR_SIMD - 1;

    EXPECT_THROW(
        adopt(ds, f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge()),
        DB::Exception);
    EXPECT_EQ(f.retain_witness, 1);
}


TEST(AdoptionLayer, RowCountMismatchRejected)
{
    AdoptFixture f;
    auto ds = f.descriptors();
    ds[0].value_count = AdoptFixture::ROWS + 1; // mismatch (precondition 26)

    EXPECT_THROW(
        adopt(ds, f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge()),
        DB::Exception);
    EXPECT_EQ(f.retain_witness, 1);
}


TEST(AdoptionLayer, DescriptorWireTypeMismatchRejected)
{
    AdoptFixture f;
    auto ds = f.descriptors();
    /// SQL-side says UInt64 but the descriptor claims String. Adoption layer is the
    /// late-stage cross-check (the handshake already enforces schema membership and
    /// equality, precondition 6, via `SHM_SCHEMA_MISMATCH`); once past the handshake,
    /// a per-block descriptor wire-tag inconsistency is a buffer/descriptor layout
    /// issue — precondition-13/16 territory — and must surface as
    /// `SHM_BUFFER_LAYOUT_INVALID`, NOT as `SHM_SCHEMA_MISMATCH` (F10 fix).
    ds[0].type = static_cast<uint32_t>(WireColumnType::String);

    try
    {
        adopt(ds, f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge());
        FAIL() << "expected DB::Exception with SHM_BUFFER_LAYOUT_INVALID";
    }
    catch (const DB::Exception & e)
    {
        EXPECT_EQ(e.code(), DB::ErrorCodes::SHM_BUFFER_LAYOUT_INVALID)
            << "F10: per-block descriptor wire-tag mismatch must NOT be classified as "
               "SHM_SCHEMA_MISMATCH (that class is reserved for handshake-time "
               "preconditions 4-6); got code " << e.code() << ": " << e.message();
    }
    EXPECT_EQ(f.retain_witness, 1);
}


TEST(AdoptionLayer, DescriptorStringWireTypeMismatchRejected)
{
    /// Symmetric F10 coverage for the String branch: SQL-side says String but the
    /// descriptor claims UInt64. Must surface SHM_BUFFER_LAYOUT_INVALID (per-block
    /// descriptor layout issue), NOT SHM_SCHEMA_MISMATCH.
    AdoptFixture f;
    auto ds = f.descriptors();
    ds[1].type = static_cast<uint32_t>(WireColumnType::UInt64);

    try
    {
        adopt(ds, f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge());
        FAIL() << "expected DB::Exception with SHM_BUFFER_LAYOUT_INVALID";
    }
    catch (const DB::Exception & e)
    {
        EXPECT_EQ(e.code(), DB::ErrorCodes::SHM_BUFFER_LAYOUT_INVALID)
            << "F10 String-branch: got code " << e.code() << ": " << e.message();
    }
    EXPECT_EQ(f.retain_witness, 1);
}


TEST(AdoptionLayer, UnsupportedSqlTypeStillSchemaMismatch)
{
    /// Companion to F10: the catch-all for an SQL-declared type that escapes BOTH the
    /// SQL-side gate AND the handshake cross-validation (precondition 6 escape) MUST
    /// still raise SHM_SCHEMA_MISMATCH — that case is a schema-membership failure,
    /// not a per-block descriptor layout failure. Keeping this test alongside
    /// DescriptorWireTypeMismatchRejected pins the two cases apart explicitly so a
    /// future refactor cannot silently re-collapse them.
    AdoptFixture f;
    std::vector<std::pair<std::string, DataTypePtr>> schema_bad = {
        {"id", std::make_shared<DataTypeInt32>()},
        {"s",  std::make_shared<DataTypeString>()}};

    try
    {
        adopt(f.descriptors(), schema_bad, f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge());
        FAIL() << "expected DB::Exception with SHM_SCHEMA_MISMATCH";
    }
    catch (const DB::Exception & e)
    {
        EXPECT_EQ(e.code(), DB::ErrorCodes::SHM_SCHEMA_MISMATCH)
            << "unsupported-SQL-type catch-all must remain SHM_SCHEMA_MISMATCH; "
               "got code " << e.code() << ": " << e.message();
    }
    EXPECT_EQ(f.retain_witness, 1);
}


TEST(AdoptionLayer, DescriptorCountMismatchRejected)
{
    AdoptFixture f;
    auto ds = f.descriptors();
    ds.pop_back(); // descriptors.size() != schema.size() (precondition 12)

    EXPECT_THROW(
        adopt(ds, f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge()),
        DB::Exception);
    EXPECT_EQ(f.retain_witness, 1);
}


TEST(AdoptionLayer, LazyContentValidationDetectsBadOffsets)
{
    AdoptFixture f;
    /// Make the terminal offset wrong (precondition 22). The adopt() call still succeeds
    /// (content-level check is lazy); validateAdoptedOffsets() raises before any unsafe read.
    auto * offs_in_region = reinterpret_cast<UInt64 *>(f.region.data() + f.str_offsets_offset);
    offs_in_region[AdoptFixture::ROWS - 1] += 100;

    Columns cols;
    EXPECT_NO_THROW(
        cols = adopt(f.descriptors(), f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
                     AdoptFixture::ROWS, f.makeRetain(), f.makeCharge()));

    const ColumnPtr & str_col_late = cols[1];
    const auto * cs = typeid_cast<const ColumnString *>(str_col_late.get());
    ASSERT_NE(cs, nullptr);
    EXPECT_THROW(cs->validateAdoptedOffsets(), DB::Exception);
}


TEST(AdoptionLayer, StringOffsetsOffsetTooSmallRejected)
{
    /// Finding 6: descriptor with `offsets_offset < sizeof(uint64_t)` leaves no room for
    /// the offsets[-1] zero sentinel that `ColumnString::offsetAt(0)` reads on every
    /// row-0 access. The adoption seam rejects it as SHM_BUFFER_LAYOUT_INVALID before any
    /// adopted column is constructed; on rejection, the retain and charge releasers run
    /// exactly once (RAII rollback, I10).
    AdoptFixture f;
    auto ds = f.descriptors();
    ds[1].offsets_offset = 4; // < 8; no room for the offsets[-1] sentinel

    EXPECT_THROW(
        adopt(ds, f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge()),
        DB::Exception);
    EXPECT_EQ(f.retain_witness, 1);
    EXPECT_EQ(f.tracker_counter.load(), 0);
}


TEST(AdoptionLayer, StringSentinelNonZeroRejected)
{
    /// Finding 6: even with a structurally valid `offsets_offset >= 8`, the 8 bytes
    /// immediately preceding `offsets[0]` MUST be zero — this is the offsets[-1] slot
    /// that ColumnString::offsetAt(0) reads. A non-conforming producer that fails to
    /// zero this slot is rejected with SHM_BUFFER_LAYOUT_INVALID before any unsafe row-0
    /// read can occur.
    AdoptFixture f;
    /// Corrupt the sentinel directly in the data region. The fixture leaves the 8 bytes
    /// at `str_offsets_offset - 8` set to 0 (see AdoptFixture::AdoptFixture); we overwrite
    /// them to a recognisable poison value.
    auto * sentinel = reinterpret_cast<uint64_t *>(
        f.region.data() + f.str_offsets_offset - sizeof(uint64_t));
    *sentinel = 0xDEADBEEFCAFEBABEULL;

    EXPECT_THROW(
        adopt(f.descriptors(), f.schema(), f.region.data(), AdoptFixture::REGION_SIZE,
              AdoptFixture::ROWS, f.makeRetain(), f.makeCharge()),
        DB::Exception);
    EXPECT_EQ(f.retain_witness, 1);
    EXPECT_EQ(f.tracker_counter.load(), 0);
}
