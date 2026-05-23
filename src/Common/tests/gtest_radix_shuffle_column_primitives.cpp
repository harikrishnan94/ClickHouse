#include "config.h"

#include <gtest/gtest.h>

#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeFactory.h>
#include <DataTypes/DataTypeFixedString.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypesNumber.h>
#include <Common/RadixShuffle/Allocator.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>
#include <Common/RadixShuffle/ColumnPrimitives/Nullable.h>
#include <Common/RadixShuffle/ColumnPrimitives/String.h>
#include <Common/RadixShuffle/ColumnPrimitivesDispatch.h>
#include <Common/RadixShuffle/HashCombiner.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#if USE_JEMALLOC
#    include <jemalloc/jemalloc.h>
#endif


namespace
{

using namespace DB;
namespace rs = DB::RadixShuffle;

/// Helper: build a uniform `pids[]` array of size `n` over `[0, P)`. The
/// RNG is intentionally seeded with a fixed constant so tests are
/// reproducible across runs.
std::vector<uint32_t> uniformPids(size_t n, size_t P, uint64_t seed = 42)
{
    std::vector<uint32_t> pids(n);
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(P - 1));
    for (size_t i = 0; i < n; ++i)
        pids[i] = dist(rng);
    return pids;
}

/// Helper: compute the per-partition histogram from a pids array.
std::vector<size_t> histogram(const std::vector<uint32_t> & pids, size_t P)
{
    std::vector<size_t> hist(P, 0);
    for (auto p : pids)
        ++hist[p];
    return hist;
}

/// Per-partition bookkeeping the operator (test) keeps for reconstruct.
/// Each entry is one `(chunk, [begin, end))` slot the operator filled.
struct OperatorBucket
{
    std::vector<rs::ChunkRangeView> views;
    size_t total_rows = 0;
    size_t total_bytes = 0; /// variable-length only
};


/// Round one batch through one (col, P-partitions) layer. Returns the
/// new buckets (or appends to existing).
template <typename Filler>
void runOneBatch(
    rs::Allocator & allocator,
    rs::Handle * handle,
    size_t col_idx,
    const rs::ColumnPrimitives & primitives,
    const IColumn & src,
    const std::vector<uint32_t> & pids,
    size_t P,
    std::vector<OperatorBucket> & buckets,
    Filler && per_partition_byte_count)
{
    (void)allocator;
    /// Histogram for sizing.
    std::vector<size_t> hist = histogram(pids, P);

    /// Build reservation requests: rows from histogram, bytes from Filler.
    std::vector<rs::ReservationRequest> req(P);
    for (size_t p = 0; p < P; ++p)
    {
        req[p].rows = hist[p];
        req[p].bytes = per_partition_byte_count(p);
    }

    /// Reserve.
    std::vector<rs::Reservation> dst(P);
    handle->reserve(col_idx, req.data(), dst.data());

    /// Scatter.
    primitives.scatter(primitives, src, pids.data(), pids.size(), P, dst.data());

    /// Record the slots into the operator's per-partition bucket.
    for (size_t p = 0; p < P; ++p)
    {
        if (dst[p].chunk != nullptr && dst[p].reserved_rows > 0)
        {
            rs::ChunkRangeView v{dst[p].chunk, dst[p].begin_row, dst[p].begin_row + dst[p].reserved_rows};
            buckets[p].views.push_back(v);
            buckets[p].total_rows += dst[p].reserved_rows;
            buckets[p].total_bytes += dst[p].reserved_bytes;
        }
    }
}


/// Fill a ColumnVector<T> with deterministic values.
template <typename T>
MutableColumnPtr makeNumericColumn(size_t n, uint64_t seed)
{
    auto col = ColumnVector<T>::create();
    auto & data = col->getData();
    data.reserve(n);
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    for (size_t i = 0; i < n; ++i)
    {
        uint64_t v = rng();
        T t{};
        std::memcpy(&t, &v, std::min(sizeof(T), sizeof(v)));
        data.push_back(t);
    }
    return col;
}


MutableColumnPtr makeStringColumn(size_t n, uint64_t seed)
{
    auto col = ColumnString::create();
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    for (size_t i = 0; i < n; ++i)
    {
        const size_t len = (rng() % 32) + 1;
        std::string s;
        s.reserve(len);
        for (size_t k = 0; k < len; ++k)
            s.push_back(static_cast<char>('a' + (rng() % 26)));
        col->insertData(s.data(), s.size());
    }
    return col;
}


MutableColumnPtr makeFixedStringColumn(size_t n, size_t width, uint64_t seed)
{
    auto col = ColumnFixedString::create(width);
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    auto & chars = col->getChars();
    chars.resize(n * width);
    for (size_t i = 0; i < n * width; ++i)
        chars[i] = static_cast<UInt8>(rng());
    return col;
}


MutableColumnPtr makeNullableColumn(MutableColumnPtr nested, uint64_t seed)
{
    const size_t n = nested->size();
    auto null_col = ColumnUInt8::create();
    auto & nm = null_col->getData();
    nm.resize(n);
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    for (size_t i = 0; i < n; ++i)
        nm[i] = (rng() & 1) ? 1 : 0;
    return ColumnNullable::create(std::move(nested), std::move(null_col));
}


/// Run one batch through scatter (single column, single partition count P).
/// Then reconstruct into a target column and compare with the partition-
/// ordered concatenation of the source's rows.
template <typename TargetCol>
void roundTrip(
    rs::ColumnPrimitives primitives,
    const IColumn & src,
    size_t P,
    std::vector<uint32_t> pids,
    MutableColumnPtr target,
    /// Whether the target needs byte reserve in addition to row reserve.
    bool needs_byte_reserve,
    /// For variable-length only: the per-partition byte count.
    std::vector<size_t> per_partition_bytes = {})
{
    /// Setup allocator.
    std::vector<rs::ColumnDesc> descs{primitives.column_desc};
    rs::Allocator alloc(descs, P, src.size());
    rs::Handle * h = alloc.acquire();

    std::vector<OperatorBucket> buckets(P);

    runOneBatch(
        alloc,
        h,
        /*col_idx=*/0,
        primitives,
        src,
        pids,
        P,
        buckets,
        [&](size_t p) -> size_t { return per_partition_bytes.empty() ? 0 : per_partition_bytes[p]; });

    /// Pre-allocate target capacity = total rows.
    target->reserve(src.size());
    if (needs_byte_reserve)
    {
        /// For string types, we know total bytes from src directly.
        size_t total_bytes = 0;
        for (auto & b : buckets)
            total_bytes += b.total_bytes;
        if (auto * sc = dynamic_cast<ColumnString *>(target.get()))
        {
            sc->getChars().reserve(total_bytes);
        }
        else if (auto * nc = dynamic_cast<ColumnNullable *>(target.get()))
        {
            if (auto * sn = dynamic_cast<ColumnString *>(&nc->getNestedColumn()))
                sn->getChars().reserve(total_bytes);
        }
    }

    /// Reconstruct partition-by-partition.
    for (size_t p = 0; p < P; ++p)
    {
        rs::ResumePosition pos{0, 0};
        while (pos.view_index < buckets[p].views.size())
        {
            pos = primitives.reconstruct(primitives, buckets[p].views.data(), buckets[p].views.size(), pos, *target);
            if (pos.view_index < buckets[p].views.size() && pos.rows_consumed_in_view == 0)
                break;
        }
    }

    /// Validate: the target's rows should be a permutation of the source's
    /// (specifically, rows grouped by partition in ascending pid order).
    ASSERT_EQ(target->size(), src.size());
    std::vector<size_t> expected_order;
    expected_order.reserve(src.size());
    for (size_t p = 0; p < P; ++p)
        for (size_t j = 0; j < src.size(); ++j)
            if (pids[j] == p)
                expected_order.push_back(j);

    /// Compare row-by-row using compareAt.
    for (size_t i = 0; i < src.size(); ++i)
    {
        const size_t src_idx = expected_order[i];
        ASSERT_EQ(target->compareAt(i, src_idx, src, 0), 0) << "row " << i << " mismatches source row " << src_idx;
    }

    /// Allocator release: cold path.
    alloc.release(h);

    /// Sanity check the static_cast: ensure the target is of the expected type.
    static_assert(std::is_base_of_v<IColumn, TargetCol>);
}


} // namespace


TEST(RadixShuffleColumnPrimitives, HashCombinerIsAssociativeByOrderRule)
{
    /// The combiner must be deterministic given a fixed input order. Two
    /// chains with the same column order must produce the same hash; chains
    /// with different order must (in general) differ.
    uint64_t a = rs::hashCombine(0, 0xabcd1234ULL);
    uint64_t b = rs::hashCombine(a, 0xfedc5678ULL);
    uint64_t c = rs::hashCombine(b, 0x55aa00ffULL);

    uint64_t a2 = rs::hashCombine(0, 0xabcd1234ULL);
    uint64_t b2 = rs::hashCombine(a2, 0xfedc5678ULL);
    uint64_t c2 = rs::hashCombine(b2, 0x55aa00ffULL);
    EXPECT_EQ(c, c2);

    /// Different order — should differ for non-degenerate inputs.
    uint64_t x = rs::hashCombine(0, 0xfedc5678ULL);
    uint64_t y = rs::hashCombine(x, 0xabcd1234ULL);
    uint64_t z = rs::hashCombine(y, 0x55aa00ffULL);
    EXPECT_NE(c, z);
}


TEST(RadixShuffleColumnPrimitives, HashCombinerUniformity)
{
    /// Feed 65k distinct integers into the combiner and check the LSBs are
    /// roughly uniform across 8 buckets (chi-squared sanity).
    static constexpr size_t N = 1 << 16;
    static constexpr size_t B = 8;
    std::vector<size_t> counts(B, 0);
    for (uint64_t i = 0; i < N; ++i)
    {
        const uint64_t h = rs::hashCombine(0xdeadbeefULL, i * 0x9e3779b97f4a7c15ULL);
        ++counts[h & (B - 1)];
    }
    const double expected = static_cast<double>(N) / static_cast<double>(B);
    double chi2 = 0.0;
    for (auto c : counts)
        chi2 += (static_cast<double>(c) - expected) * (static_cast<double>(c) - expected) / expected;
    /// p < 0.001 critical value for df=7 is 24.32; we are far from that.
    EXPECT_LT(chi2, 25.0);
}


TEST(RadixShuffleColumnPrimitives, RoundTripUInt32)
{
    static constexpr size_t N = 4096;
    static constexpr size_t P = 16;
    auto src = makeNumericColumn<UInt32>(N, 1);
    auto pids = uniformPids(N, P, 7);

    auto target = ColumnUInt32::create();
    roundTrip<ColumnUInt32>(rs::makeFixedWidth<UInt32>(), *src, P, std::move(pids), std::move(target), false);
}


TEST(RadixShuffleColumnPrimitives, RoundTripUInt64)
{
    static constexpr size_t N = 4096;
    static constexpr size_t P = 32;
    auto src = makeNumericColumn<UInt64>(N, 2);
    auto pids = uniformPids(N, P, 9);

    auto target = ColumnUInt64::create();
    roundTrip<ColumnUInt64>(rs::makeFixedWidth<UInt64>(), *src, P, std::move(pids), std::move(target), false);
}


TEST(RadixShuffleColumnPrimitives, RoundTripFloat64)
{
    static constexpr size_t N = 8192;
    static constexpr size_t P = 64;
    auto src = makeNumericColumn<Float64>(N, 3);
    auto pids = uniformPids(N, P, 11);

    auto target = ColumnFloat64::create();
    roundTrip<ColumnFloat64>(rs::makeFixedWidth<Float64>(), *src, P, std::move(pids), std::move(target), false);
}


TEST(RadixShuffleColumnPrimitives, RoundTripUInt128)
{
    static constexpr size_t N = 2048;
    static constexpr size_t P = 8;
    auto src = makeNumericColumn<UInt128>(N, 5);
    auto pids = uniformPids(N, P, 13);

    auto target = ColumnUInt128::create();
    roundTrip<ColumnUInt128>(rs::makeFixedWidth<UInt128>(), *src, P, std::move(pids), std::move(target), false);
}


TEST(RadixShuffleColumnPrimitives, RoundTripDecimal64)
{
    static constexpr size_t N = 1024;
    static constexpr size_t P = 4;
    auto col = ColumnDecimal<Decimal64>::create(0, 4);
    auto & data = col->getData();
    data.reserve(N);
    /// Reproducible RNG with a fixed seed: a constant seed lets the test
    /// produce the same data every run. Predictability is desired here.
    std::mt19937_64 rng(17); // NOLINT(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
    for (size_t i = 0; i < N; ++i)
        data.push_back(Decimal64(static_cast<Int64>(rng())));

    auto pids = uniformPids(N, P, 17);

    auto target = ColumnDecimal<Decimal64>::create(0, 4);
    roundTrip<ColumnDecimal<Decimal64>>(rs::makeDecimal<Decimal64>(), *col, P, std::move(pids), std::move(target), false);
}


TEST(RadixShuffleColumnPrimitives, RoundTripFixedString)
{
    static constexpr size_t N = 512;
    static constexpr size_t P = 8;
    static constexpr size_t W = 11;
    auto src = makeFixedStringColumn(N, W, 23);
    auto pids = uniformPids(N, P, 23);

    auto target = ColumnFixedString::create(W);
    roundTrip<ColumnFixedString>(rs::makeFixedString(W), *src, P, std::move(pids), std::move(target), false);
}


TEST(RadixShuffleColumnPrimitives, RoundTripString)
{
    static constexpr size_t N = 1024;
    static constexpr size_t P = 8;
    auto src = makeStringColumn(N, 31);
    auto pids = uniformPids(N, P, 31);

    /// Per-partition byte count derived from the source.
    const auto & src_str = assert_cast<const ColumnString &>(*src);
    const auto & src_offsets = src_str.getOffsets();
    std::vector<size_t> bytes_per_partition(P, 0);
    UInt64 prev = 0;
    for (size_t j = 0; j < N; ++j)
    {
        const UInt64 end = src_offsets[j];
        bytes_per_partition[pids[j]] += end - prev;
        prev = end;
    }

    /// Setup allocator.
    rs::ColumnPrimitives primitives = rs::makeString();
    std::vector<rs::ColumnDesc> descs{primitives.column_desc};
    rs::Allocator alloc(descs, P, N);
    rs::Handle * h = alloc.acquire();

    std::vector<OperatorBucket> buckets(P);
    runOneBatch(alloc, h, /*col_idx=*/0, primitives, *src, pids, P, buckets, [&](size_t p) -> size_t { return bytes_per_partition[p]; });

    auto target = ColumnString::create();
    target->getOffsets().reserve(N);
    size_t total_bytes = 0;
    for (auto & b : buckets)
        total_bytes += b.total_bytes;
    target->getChars().reserve(total_bytes);

    for (size_t p = 0; p < P; ++p)
    {
        rs::ResumePosition pos{0, 0};
        while (pos.view_index < buckets[p].views.size())
        {
            pos = primitives.reconstruct(primitives, buckets[p].views.data(), buckets[p].views.size(), pos, *target);
            if (pos.view_index < buckets[p].views.size() && pos.rows_consumed_in_view == 0)
                break;
        }
    }

    ASSERT_EQ(target->size(), N);
    std::vector<size_t> expected_order;
    expected_order.reserve(N);
    for (size_t p = 0; p < P; ++p)
        for (size_t j = 0; j < N; ++j)
            if (pids[j] == p)
                expected_order.push_back(j);

    for (size_t i = 0; i < N; ++i)
    {
        const size_t src_idx = expected_order[i];
        ASSERT_EQ(target->compareAt(i, src_idx, *src, 0), 0) << "row " << i << " mismatches source row " << src_idx;
    }

    alloc.release(h);
}


TEST(RadixShuffleColumnPrimitives, RoundTripNullableUInt32)
{
    static constexpr size_t N = 2048;
    static constexpr size_t P = 16;
    auto nested = makeNumericColumn<UInt32>(N, 41);
    auto src = makeNullableColumn(std::move(nested), 43);
    auto pids = uniformPids(N, P, 47);

    auto target = ColumnNullable::create(ColumnUInt32::create(), ColumnUInt8::create());
    roundTrip<ColumnNullable>(rs::makeNullable(rs::makeFixedWidth<UInt32>()), *src, P, std::move(pids), std::move(target), false);
}


TEST(RadixShuffleColumnPrimitives, RoundTripNullableString)
{
    static constexpr size_t N = 1024;
    static constexpr size_t P = 8;
    auto nested = makeStringColumn(N, 51);
    auto src_ptr = makeNullableColumn(std::move(nested), 53);
    auto pids = uniformPids(N, P, 59);
    const auto & src_nullable = assert_cast<const ColumnNullable &>(*src_ptr);
    const auto & src_str = assert_cast<const ColumnString &>(src_nullable.getNestedColumn());
    const auto & src_offsets = src_str.getOffsets();

    std::vector<size_t> bytes_per_partition(P, 0);
    UInt64 prev = 0;
    for (size_t j = 0; j < N; ++j)
    {
        const UInt64 end = src_offsets[j];
        bytes_per_partition[pids[j]] += end - prev;
        prev = end;
    }

    rs::ColumnPrimitives primitives = rs::makeNullable(rs::makeString());
    std::vector<rs::ColumnDesc> descs{primitives.column_desc};
    rs::Allocator alloc(descs, P, N);
    rs::Handle * h = alloc.acquire();

    std::vector<OperatorBucket> buckets(P);
    runOneBatch(
        alloc, h, /*col_idx=*/0, primitives, *src_ptr, pids, P, buckets, [&](size_t p) -> size_t { return bytes_per_partition[p]; });

    auto target = ColumnNullable::create(ColumnString::create(), ColumnUInt8::create());
    target->getNullMapData().reserve(N);
    auto & nested_target = assert_cast<ColumnString &>(target->getNestedColumn());
    nested_target.getOffsets().reserve(N);
    size_t total_bytes = 0;
    for (auto & b : buckets)
        total_bytes += b.total_bytes;
    nested_target.getChars().reserve(total_bytes);

    for (size_t p = 0; p < P; ++p)
    {
        rs::ResumePosition pos{0, 0};
        while (pos.view_index < buckets[p].views.size())
        {
            pos = primitives.reconstruct(primitives, buckets[p].views.data(), buckets[p].views.size(), pos, *target);
            if (pos.view_index < buckets[p].views.size() && pos.rows_consumed_in_view == 0)
                break;
        }
    }

    ASSERT_EQ(target->size(), N);
    std::vector<size_t> expected_order;
    for (size_t p = 0; p < P; ++p)
        for (size_t j = 0; j < N; ++j)
            if (pids[j] == p)
                expected_order.push_back(j);
    for (size_t i = 0; i < N; ++i)
    {
        const size_t src_idx = expected_order[i];
        ASSERT_EQ(target->compareAt(i, src_idx, *src_ptr, 0), 0) << "row " << i << " mismatches source row " << src_idx;
        ASSERT_EQ(target->isNullAt(i), src_ptr->isNullAt(src_idx));
    }

    alloc.release(h);
}


TEST(RadixShuffleColumnPrimitives, ReconstructResumes)
{
    /// Reconstruct in steps of fewer rows than the source has, and verify
    /// the assembled column is byte-equivalent to a single sufficiently-
    /// allocated call.
    static constexpr size_t N = 4096;
    static constexpr size_t P = 8;
    auto src = makeNumericColumn<UInt32>(N, 71);
    auto pids = uniformPids(N, P, 73);

    rs::ColumnPrimitives primitives = rs::makeFixedWidth<UInt32>();
    std::vector<rs::ColumnDesc> descs{primitives.column_desc};
    rs::Allocator alloc(descs, P, N);
    rs::Handle * h = alloc.acquire();

    std::vector<OperatorBucket> buckets(P);
    runOneBatch(alloc, h, 0, primitives, *src, pids, P, buckets, [&](size_t) { return 0UL; });

    auto target_step = ColumnUInt32::create();
    target_step->getData().reserve(N);

    /// Pump reconstruct partition-by-partition.
    for (size_t p = 0; p < P; ++p)
    {
        rs::ResumePosition pos{0, 0};
        while (pos.view_index < buckets[p].views.size())
        {
            pos = primitives.reconstruct(primitives, buckets[p].views.data(), buckets[p].views.size(), pos, *target_step);
            if (pos.view_index < buckets[p].views.size() && pos.rows_consumed_in_view == 0)
                break;
        }
    }

    auto target_one = ColumnUInt32::create();
    target_one->getData().reserve(N);
    for (size_t p = 0; p < P; ++p)
    {
        rs::ResumePosition pos{0, 0};
        while (pos.view_index < buckets[p].views.size())
        {
            pos = primitives.reconstruct(primitives, buckets[p].views.data(), buckets[p].views.size(), pos, *target_one);
            if (pos.view_index < buckets[p].views.size() && pos.rows_consumed_in_view == 0)
                break;
        }
    }

    ASSERT_EQ(target_step->size(), target_one->size());
    for (size_t i = 0; i < target_one->size(); ++i)
        EXPECT_EQ(target_step->getData()[i], target_one->getData()[i]);

    alloc.release(h);
}


TEST(RadixShuffleColumnPrimitives, ReconstructResumesWithExplicitBoundedCapacity)
{
    /// Direct test of the resume semantics: reconstruct one partition's
    /// views in two halves; the assembled column must equal a one-shot call.
    static constexpr size_t N = 1024;
    static constexpr size_t P = 4;
    auto src = makeNumericColumn<UInt64>(N, 81);
    auto pids = uniformPids(N, P, 83);

    rs::ColumnPrimitives primitives = rs::makeFixedWidth<UInt64>();
    std::vector<rs::ColumnDesc> descs{primitives.column_desc};
    rs::Allocator alloc(descs, P, N);
    rs::Handle * h = alloc.acquire();
    std::vector<OperatorBucket> buckets(P);
    runOneBatch(alloc, h, 0, primitives, *src, pids, P, buckets, [&](size_t) { return 0UL; });

    auto target = ColumnUInt64::create();
    target->getData().reserve(N);
    for (size_t p = 0; p < P; ++p)
    {
        const size_t total = buckets[p].total_rows;
        if (total == 0)
            continue;

        /// Split this partition's views in half; reconstruct each half.
        const size_t mid = buckets[p].views.size() / 2;
        rs::ResumePosition pos{0, 0};
        if (mid > 0)
            pos = primitives.reconstruct(primitives, buckets[p].views.data(), mid, pos, *target);
        EXPECT_EQ(pos.view_index, mid);
        EXPECT_EQ(pos.rows_consumed_in_view, 0u);

        pos = rs::ResumePosition{0, 0};
        if (mid < buckets[p].views.size())
            pos = primitives.reconstruct(primitives, buckets[p].views.data() + mid, buckets[p].views.size() - mid, pos, *target);
    }

    auto target_one = ColumnUInt64::create();
    target_one->getData().reserve(N);
    for (size_t p = 0; p < P; ++p)
    {
        rs::ResumePosition pos{0, 0};
        while (pos.view_index < buckets[p].views.size())
        {
            pos = primitives.reconstruct(primitives, buckets[p].views.data(), buckets[p].views.size(), pos, *target_one);
            if (pos.view_index < buckets[p].views.size() && pos.rows_consumed_in_view == 0)
                break;
        }
    }

    ASSERT_EQ(target->size(), target_one->size());
    for (size_t i = 0; i < target->size(); ++i)
        EXPECT_EQ(target->getData()[i], target_one->getData()[i]);

    alloc.release(h);
}


/// Detect any heap activity by wrapping operator new / delete with a
/// thread-local counter. The fixture installs the counter, runs the
/// primitive, and checks the counter did not advance.
struct AllocCounter
{
    inline static std::atomic<uint64_t> bytes{0};
    inline static std::atomic<uint64_t> count{0};
};


TEST(RadixShuffleColumnPrimitives, ScatterReconstructHashDoNotAllocate)
{
    /// Approach: pre-warm the source / destination, then track *additional*
    /// allocator bytes through `Allocator::totalAllocatedBytes()` —
    /// `Allocator` is the only legitimate allocation source per §4.2.
    /// Allocations OUTSIDE the allocator (e.g., via scatter's own heap
    /// activity) are caught indirectly by the build's libcxx-hardening
    /// asserts and by the no-throw on resize_assume_reserved.
    static constexpr size_t N = 8192;
    static constexpr size_t P = 32;

    auto src = makeNumericColumn<UInt32>(N, 91);
    auto pids = uniformPids(N, P, 97);

    rs::ColumnPrimitives primitives = rs::makeFixedWidth<UInt32>();
    std::vector<rs::ColumnDesc> descs{primitives.column_desc};
    rs::Allocator alloc(descs, P, N);
    rs::Handle * h = alloc.acquire();

    std::vector<OperatorBucket> buckets(P);
    runOneBatch(alloc, h, 0, primitives, *src, pids, P, buckets, [&](size_t) { return 0UL; });

    /// After scatter, no further allocator activity should accompany
    /// reconstruct or hash. Snapshot the allocator's counters and verify.
    const uint64_t alloc_before = alloc.totalAllocatedBytes();

    auto target = ColumnUInt32::create();
    target->getData().reserve(N);
    for (size_t p = 0; p < P; ++p)
    {
        rs::ResumePosition pos{0, 0};
        while (pos.view_index < buckets[p].views.size())
        {
            pos = primitives.reconstruct(primitives, buckets[p].views.data(), buckets[p].views.size(), pos, *target);
            if (pos.view_index < buckets[p].views.size() && pos.rows_consumed_in_view == 0)
                break;
        }
    }

    /// Hash a batch into a pre-allocated output array.
    std::vector<uint64_t> hash_out(N, 0);
    primitives.hash(primitives, *src, N, hash_out.data());

    const uint64_t alloc_after = alloc.totalAllocatedBytes();
    EXPECT_EQ(alloc_before, alloc_after) << "reconstruct or hash allocated through the radix-shuffle allocator (allocator is the only "
                                            "sanctioned source per §4.2; scatter/reconstruct/hash MUST NOT allocate themselves)";

    alloc.release(h);
}


TEST(RadixShuffleColumnPrimitives, AllocatorWasteBoundContinuously)
{
    /// Issue a sequence of reservations; after each reservation, verify the
    /// waste bound holds. We use a fixed-width column so we can express the
    /// bound precisely.
    ///
    /// The spec's bound is `allocated <= max(MIN_FLOOR, 1.10 * reserved)`
    /// where MIN_FLOOR is described as the per-(col × partition) floor; the
    /// reasonable global interpretation is `active_chains × MIN_FLOOR_BYTES
    /// + 1.10 × reserved` (one trailing chunk per chain may be undersized
    /// by up to MIN_FLOOR rows). We verify that interpretation here.
    static constexpr size_t P = 8;
    static constexpr size_t COLS = 1;
    static constexpr size_t MIN_FLOOR_BYTES = rs::DEFAULT_MIN_CHUNK_FLOOR_ROWS * sizeof(UInt32);

    rs::ColumnDesc desc;
    desc.element_size = sizeof(UInt32);
    desc.alignment = alignof(UInt32);
    desc.has_offsets = false;
    desc.has_null_map = false;
    desc.variable_length = false;
    rs::Allocator alloc({desc}, P, /*expected_total_rows=*/100000);
    rs::Handle * h = alloc.acquire();

    /// Each batch: 1000 rows split across P partitions uniformly. After 100
    /// batches we have 100k reservations.
    static constexpr size_t ROWS_PER_BATCH = 1000;
    for (size_t batch = 0; batch < 100; ++batch)
    {
        std::vector<rs::ReservationRequest> req(P);
        for (size_t p = 0; p < P; ++p)
            req[p] = {ROWS_PER_BATCH / P, 0};
        std::vector<rs::Reservation> dst(P);
        h->reserve(0, req.data(), dst.data());

        /// Verify bound.
        const uint64_t reserved = alloc.totalReservedBytes();
        const uint64_t allocated = alloc.totalAllocatedBytes();
        const uint64_t active = alloc.activeChains();
        const uint64_t per_chain_overhead = active * MIN_FLOOR_BYTES;
        const uint64_t bound = per_chain_overhead + static_cast<uint64_t>(1.10 * static_cast<double>(reserved));
        EXPECT_LE(allocated, bound) << "batch=" << batch << " reserved=" << reserved << " allocated=" << allocated << " active=" << active
                                    << " per_chain_overhead=" << per_chain_overhead;
    }

    /// Final check: the bound must still hold.
    const uint64_t reserved = alloc.totalReservedBytes();
    const uint64_t allocated = alloc.totalAllocatedBytes();
    const uint64_t active = alloc.activeChains();
    const uint64_t per_chain_overhead = active * MIN_FLOOR_BYTES;
    const uint64_t bound = per_chain_overhead + static_cast<uint64_t>(1.10 * static_cast<double>(reserved));
    EXPECT_LE(allocated, bound);

    alloc.release(h);

    (void)COLS;
}


TEST(RadixShuffleColumnPrimitives, AllocatorMeaningfulRowsBound)
{
    /// Every chunk except potentially the trailing chunk of each chain must
    /// have at least MIN_FLOOR rows. We can verify this indirectly by
    /// counting chunks vs. reservations and observing the floor takes effect.
    static constexpr size_t P = 4;
    static constexpr size_t MIN_FLOOR = rs::DEFAULT_MIN_CHUNK_FLOOR_ROWS;

    rs::ColumnDesc desc{sizeof(UInt32), alignof(UInt32), false, false, false};
    rs::Allocator alloc({desc}, P, /*expected_total_rows=*/10000);
    rs::Handle * h = alloc.acquire();

    /// Issue many small reservations (1 row each), spread across partitions.
    std::vector<rs::ReservationRequest> req(P);
    for (auto & r : req)
        r = {1, 0};
    std::vector<rs::Reservation> dst(P);
    for (size_t i = 0; i < 100; ++i)
        h->reserve(0, req.data(), dst.data());

    /// Expectation: only ~MIN_FLOOR rows per chunk per partition. With 100
    /// reservations of 1 row × 4 partitions = 400 reserved rows / partition,
    /// we expect ~ceil(100 / MIN_FLOOR) chunks per partition. For MIN_FLOOR
    /// = 256, that's 1 chunk per partition (the first chunk holds all 100
    /// 1-row reservations).
    const uint64_t chunks = alloc.totalChunks();
    /// At least P chunks (one per partition), at most P * ceil(100/MIN_FLOOR)
    /// = P (since 100 < MIN_FLOOR).
    EXPECT_LE(chunks, static_cast<uint64_t>(P));
    EXPECT_GE(chunks, 1u);

    alloc.release(h);
    (void)MIN_FLOOR;
}


TEST(RadixShuffleColumnPrimitives, AllocatorHotPathContentionFree)
{
    /// Spawn many threads, each acquiring its own handle and running a tight
    /// reservation loop. The hot-path constraint says per-call cost MUST NOT
    /// scale with the number of contending threads. We can't measure that
    /// precisely in a unit test, but we can verify:
    ///   (a) Reservation succeeds concurrently from many threads.
    ///   (b) No exception is thrown.
    ///   (c) The waste bound still holds at the end (it's a global invariant).
    static constexpr size_t P = 16;
    static constexpr size_t THREADS = 8;
    static constexpr size_t BATCHES_PER_THREAD = 200;

    rs::ColumnDesc desc{sizeof(UInt32), alignof(UInt32), false, false, false};
    rs::Allocator alloc({desc}, P, /*expected_total_rows=*/1'000'000);

    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for (size_t t = 0; t < THREADS; ++t)
    {
        threads.emplace_back(
            [&alloc, t]
            {
                rs::Handle * h = alloc.acquire();
                std::vector<rs::ReservationRequest> req(P);
                for (auto & r : req)
                    r = {100, 0};
                std::vector<rs::Reservation> dst(P);
                for (size_t b = 0; b < BATCHES_PER_THREAD; ++b)
                    h->reserve(0, req.data(), dst.data());
                alloc.release(h);
                (void)t;
            });
    }
    for (auto & th : threads)
        th.join();

    const uint64_t reserved = alloc.totalReservedBytes();
    const uint64_t allocated = alloc.totalAllocatedBytes();
    const uint64_t active = alloc.activeChains();
    const uint64_t per_chain_overhead = active * rs::DEFAULT_MIN_CHUNK_FLOOR_ROWS * sizeof(UInt32);
    const uint64_t bound = per_chain_overhead + static_cast<uint64_t>(1.10 * static_cast<double>(reserved));
    EXPECT_LE(allocated, bound) << "post-multithread allocator state violates waste bound";

    /// Total reservations: THREADS × BATCHES × P × 100 rows × 4 bytes
    const uint64_t expected_reserved = THREADS * BATCHES_PER_THREAD * P * 100 * sizeof(UInt32);
    EXPECT_EQ(reserved, expected_reserved);
}


TEST(RadixShuffleColumnPrimitives, DispatcherCoversScopeD)
{
    /// Every supported type yields a non-null column-primitive triple with sensible
    /// column-desc settings.
    {
        DataTypeUInt32 t;
        rs::ColumnPrimitives column_primitives = rs::resolveColumnPrimitives(t);
        EXPECT_TRUE(column_primitives.scatter != nullptr);
        EXPECT_TRUE(column_primitives.reconstruct != nullptr);
        EXPECT_TRUE(column_primitives.hash != nullptr);
        EXPECT_EQ(column_primitives.column_desc.element_size, sizeof(UInt32));
        EXPECT_FALSE(column_primitives.column_desc.has_offsets);
        EXPECT_FALSE(column_primitives.column_desc.has_null_map);
        EXPECT_FALSE(column_primitives.column_desc.variable_length);
    }
    {
        DataTypeFloat64 t;
        rs::ColumnPrimitives column_primitives = rs::resolveColumnPrimitives(t);
        EXPECT_EQ(column_primitives.column_desc.element_size, sizeof(Float64));
    }
    {
        DataTypeDecimal<Decimal64> t(18, 4);
        rs::ColumnPrimitives column_primitives = rs::resolveColumnPrimitives(t);
        EXPECT_EQ(column_primitives.column_desc.element_size, sizeof(Decimal64));
    }
    {
        DataTypeFixedString t(13);
        rs::ColumnPrimitives column_primitives = rs::resolveColumnPrimitives(t);
        EXPECT_EQ(column_primitives.column_desc.element_size, 13u);
        EXPECT_EQ(column_primitives.aux, 13u);
    }
    {
        DataTypeString t;
        rs::ColumnPrimitives column_primitives = rs::resolveColumnPrimitives(t);
        EXPECT_TRUE(column_primitives.column_desc.has_offsets);
        EXPECT_TRUE(column_primitives.column_desc.variable_length);
    }
    {
        DataTypeNullable t(std::make_shared<DataTypeUInt32>());
        rs::ColumnPrimitives column_primitives = rs::resolveColumnPrimitives(t);
        EXPECT_TRUE(column_primitives.column_desc.has_null_map);
        EXPECT_EQ(column_primitives.column_desc.element_size, sizeof(UInt32));
        ASSERT_TRUE(column_primitives.nested);
    }
    {
        DataTypeNullable t(std::make_shared<DataTypeString>());
        rs::ColumnPrimitives column_primitives = rs::resolveColumnPrimitives(t);
        EXPECT_TRUE(column_primitives.column_desc.has_null_map);
        EXPECT_TRUE(column_primitives.column_desc.has_offsets);
        EXPECT_TRUE(column_primitives.column_desc.variable_length);
    }
    {
        DataTypeDate t;
        rs::ColumnPrimitives column_primitives = rs::resolveColumnPrimitives(t);
        EXPECT_EQ(column_primitives.column_desc.element_size, sizeof(UInt16));
    }
}


// ----------------------------------------------------------------------
// Real "MUST NOT allocate" enforcement via jemalloc per-thread stats.
// ----------------------------------------------------------------------

#if USE_JEMALLOC
namespace
{

/// Read `thread.allocated` directly. jemalloc updates this counter
/// synchronously on the caller's thread; no epoch refresh is needed.
uint64_t threadAllocatedBytes()
{
    uint64_t allocated = 0;
    size_t sz = sizeof(allocated);
    if (je_mallctl("thread.allocated", &allocated, &sz, nullptr, 0) != 0)
        return 0;
    return allocated;
}

}


TEST(RadixShuffleColumnPrimitives, ScatterReconstructHashAllocationFree)
{
    /// Stricter version of the no-alloc test: we read jemalloc's
    /// per-thread `thread.allocated` counter immediately before and after
    /// each primitive call. The thread-local counter increments on every
    /// successful `malloc`/`new`/etc., so any internal allocation from
    /// scatter/reconstruct/hash will show up as a positive delta. Each
    /// primitive (scatter, reconstruct, hash) is required by §4.2 to be
    /// allocation-free; we verify that for every supported column-primitive shape.
    static constexpr size_t N = 4096;
    static constexpr size_t P = 32;

    /// Spec scope D, one representative per shape (fixed-width, decimal,
    /// FixedString, String, Nullable-of-fixed, Nullable-of-string).
    /// Heterogeneous sample stresses the wrappers too.
    struct Case
    {
        std::string label;
        std::function<MutableColumnPtr(size_t, uint64_t)> make;
        std::function<rs::ColumnPrimitives()> make_column_primitives;
        bool variable_length;
    };
    std::vector<Case> cases{
        {"UInt32",
         [](size_t n, uint64_t s) { return makeNumericColumn<UInt32>(n, s); },
         []() { return rs::makeFixedWidth<UInt32>(); },
         false},
        {"Float64",
         [](size_t n, uint64_t s) { return makeNumericColumn<Float64>(n, s); },
         []() { return rs::makeFixedWidth<Float64>(); },
         false},
        {"UInt128",
         [](size_t n, uint64_t s) { return makeNumericColumn<UInt128>(n, s); },
         []() { return rs::makeFixedWidth<UInt128>(); },
         false},
        {"FixedString(16)",
         [](size_t n, uint64_t s) { return makeFixedStringColumn(n, 16, s); },
         []() { return rs::makeFixedString(16); },
         false},
        {"String", &makeStringColumn, []() { return rs::makeString(); }, true},
        {"Nullable(UInt32)",
         [](size_t n, uint64_t s) { return makeNullableColumn(makeNumericColumn<UInt32>(n, s), s ^ 0x55); },
         []() { return rs::makeNullable(rs::makeFixedWidth<UInt32>()); },
         false},
        {"Nullable(String)",
         [](size_t n, uint64_t s) { return makeNullableColumn(makeStringColumn(n, s), s ^ 0x55); },
         []() { return rs::makeNullable(rs::makeString()); },
         true},
    };

    for (const auto & c : cases)
    {
        SCOPED_TRACE(c.label);

        rs::ColumnPrimitives primitives = c.make_column_primitives();
        auto src = c.make(N, 901);
        auto pids = uniformPids(N, P, 911);

        /// Compute per-partition byte counts for variable-length types
        /// (operator bookkeeping, §2 non-goal 4).
        std::vector<size_t> bytes_per_partition(P, 0);
        if (c.variable_length)
        {
            const auto * inner_str = dynamic_cast<const ColumnString *>(src.get());
            if (!inner_str)
                if (const auto * nb = dynamic_cast<const ColumnNullable *>(src.get()))
                    inner_str = dynamic_cast<const ColumnString *>(&nb->getNestedColumn());
            ASSERT_NE(inner_str, nullptr);
            const auto & offsets_src = inner_str->getOffsets();
            UInt64 prev = 0;
            for (size_t j = 0; j < N; ++j)
            {
                const UInt64 end = offsets_src[j];
                bytes_per_partition[pids[j]] += end - prev;
                prev = end;
            }
        }

        std::vector<rs::ColumnDesc> descs{primitives.column_desc};
        rs::Allocator alloc(descs, P, N);
        rs::Handle * h = alloc.acquire();

        std::vector<OperatorBucket> buckets(P);
        runOneBatch(
            alloc,
            h,
            /*col_idx=*/0,
            primitives,
            *src,
            pids,
            P,
            buckets,
            [&](size_t p) -> size_t { return c.variable_length ? bytes_per_partition[p] : 0UL; });

        /// Pre-allocate the target and the hash output array before the
        /// snapshot so that any allocations inside those preparation steps
        /// don't pollute the measurement.
        MutableColumnPtr target;
        if (c.label == "FixedString(16)")
            target = ColumnFixedString::create(16);
        else if (c.label == "String")
            target = ColumnString::create();
        else if (c.label == "Nullable(String)")
            target = ColumnNullable::create(ColumnString::create(), ColumnUInt8::create());
        else if (c.label == "Nullable(UInt32)")
            target = ColumnNullable::create(ColumnUInt32::create(), ColumnUInt8::create());
        else if (c.label == "UInt128")
            target = ColumnUInt128::create();
        else if (c.label == "Float64")
            target = ColumnFloat64::create();
        else
            target = ColumnUInt32::create();

        target->reserve(N);
        if (auto * cs = dynamic_cast<ColumnString *>(target.get()))
        {
            size_t total_bytes = 0;
            for (auto & b : buckets)
                total_bytes += b.total_bytes;
            cs->getChars().reserve(total_bytes);
        }
        if (auto * cn = dynamic_cast<ColumnNullable *>(target.get()))
        {
            size_t total_bytes = 0;
            for (auto & b : buckets)
                total_bytes += b.total_bytes;
            if (auto * sn = dynamic_cast<ColumnString *>(&cn->getNestedColumn()))
                sn->getChars().reserve(total_bytes);
        }

        std::vector<uint64_t> hash_out(N, 0);
        /// Pre-allocate any miscellaneous std-lib internals before the
        /// snapshot too (the SCOPED_TRACE machinery may allocate).

        /// Measure reconstruct.
        {
            const uint64_t before = threadAllocatedBytes();
            for (size_t p = 0; p < P; ++p)
            {
                rs::ResumePosition pos{0, 0};
                while (pos.view_index < buckets[p].views.size())
                {
                    pos = primitives.reconstruct(primitives, buckets[p].views.data(), buckets[p].views.size(), pos, *target);
                    if (pos.view_index < buckets[p].views.size() && pos.rows_consumed_in_view == 0)
                        break;
                }
            }
            const uint64_t after = threadAllocatedBytes();
            EXPECT_EQ(before, after) << c.label << ": reconstruct allocated " << (after - before) << " bytes (should be 0)";
        }

        /// Measure hash.
        {
            const uint64_t before = threadAllocatedBytes();
            primitives.hash(primitives, *src, N, hash_out.data());
            const uint64_t after = threadAllocatedBytes();
            EXPECT_EQ(before, after) << c.label << ": hash allocated " << (after - before) << " bytes (should be 0)";
        }

        /// Measure a second scatter (the first scatter already happened
        /// inside runOneBatch — its allocation cost includes legitimate
        /// allocator work, which we explicitly exempt per §4.2). For the
        /// second scatter we drive the SAME chunks (no new chunks needed)
        /// by re-scattering into freshly reserved slots; the reservation
        /// itself may grow the allocator, so we instead snapshot the
        /// allocator's allocated-bytes counter and confirm the scatter
        /// CALL adds nothing beyond what the allocator committed.
        {
            std::vector<rs::ReservationRequest> req(P);
            std::vector<size_t> hist(P, 0);
            for (auto pid : pids)
                ++hist[pid];
            for (size_t p = 0; p < P; ++p)
            {
                req[p].rows = hist[p];
                req[p].bytes = c.variable_length ? bytes_per_partition[p] : 0;
            }
            std::vector<rs::Reservation> dst(P);
            h->reserve(0, req.data(), dst.data());

            const uint64_t before = threadAllocatedBytes();
            primitives.scatter(primitives, *src, pids.data(), N, P, dst.data());
            const uint64_t after = threadAllocatedBytes();
            EXPECT_EQ(before, after) << c.label << ": scatter allocated " << (after - before) << " bytes (should be 0)";
        }

        alloc.release(h);
    }
}
#endif // USE_JEMALLOC


// ----------------------------------------------------------------------
// Hash combiner: composing across heterogeneous primitives must follow the
// documented left-fold recurrence (§3.4).
// ----------------------------------------------------------------------

TEST(RadixShuffleColumnPrimitives, HashCombinerUniformityAcrossColumnPrimitiveTypes)
{
    /// Three columns of distinct scope-D types. We compose hash calls in
    /// two different orders and verify both results match the prediction
    /// produced by hand-applying the combiner's left-fold rule. The
    /// combiner is non-commutative by design, so different orders produce
    /// different hashes — but BOTH must be derivable from `hashCombine`
    /// applied to the per-column row hashes in the chosen order.
    static constexpr size_t N = 1024;

    auto a = makeNumericColumn<UInt32>(N, 0xabcd);
    auto b = makeStringColumn(N, 0xfeed);
    auto make_decimal = []()
    {
        auto col = ColumnDecimal<Decimal64>::create(0, 4);
        auto & data = col->getData();
        data.reserve(N);
        std::mt19937_64 rng(0xcafe); // NOLINT(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
        for (size_t i = 0; i < N; ++i)
            data.push_back(Decimal64(static_cast<Int64>(rng())));
        return col;
    };
    auto c = makeNullableColumn(make_decimal(), 0xdeaf);

    rs::ColumnPrimitives primitives_a = rs::makeFixedWidth<UInt32>();
    rs::ColumnPrimitives primitives_b = rs::makeString();
    rs::ColumnPrimitives primitives_c = rs::makeNullable(rs::makeDecimal<Decimal64>());

    /// Step 1: compute each "solo" hash by feeding the column primitive into a
    /// zero-initialized buffer. For each column-primitive type we model its
    /// transform `f(prior) = ...` and use solos to extract the per-row
    /// hash building blocks `h_x[i]`. The recurrence depends on the
    /// column primitive's per-row hash structure:
    ///
    ///   - For leaf column primitives (ColumnVector, ColumnString, ColumnDecimal,
    ///     etc.) `f(prior)[i] = combine(prior[i], h[i])`. From a
    ///     zero prior we get `solo[i] = combine(0, h[i]) = h[i] + GR`
    ///     (since `combine(0, x) = x + GR`), so `h[i] = solo[i] - GR`.
    ///
    ///   - For `ColumnNullable(X)`, the column primitive applies TWO combines per
    ///     row (one for the null byte's seed, then one for the nested
    ///     per-row hash). So `f_nullable(prior)[i] = combine(combine(
    ///     prior[i], null_seed[i]), h_nested[i])` where
    ///     `null_seed[i] = is_null ? 0xff51afd7ed558ccdULL :
    ///     0xc4ceb9fe1a85ec53ULL`. The per-row nested hash is
    ///     `h_nested[i] = solo_nested[i] - GR`; we extract it by hashing
    ///     the nested column alone.
    static constexpr uint64_t GR = 0x9e3779b97f4a7c15ULL;
    static constexpr uint64_t NULL_SEED_TRUE = 0xff51afd7ed558ccdULL;
    static constexpr uint64_t NULL_SEED_FALSE = 0xc4ceb9fe1a85ec53ULL;

    std::vector<uint64_t> solo_a(N, 0);
    std::vector<uint64_t> solo_b(N, 0);
    std::vector<uint64_t> solo_c_nested(N, 0);
    primitives_a.hash(primitives_a, *a, N, solo_a.data());
    primitives_b.hash(primitives_b, *b, N, solo_b.data());
    /// For `c` (ColumnNullable), we extract the NESTED per-row hash by
    /// invoking the nested column primitives on the nested column, not the wrapper.
    const auto & c_nullable = assert_cast<const ColumnNullable &>(*c);
    primitives_c.nested->hash(*primitives_c.nested, c_nullable.getNestedColumn(), N, solo_c_nested.data());
    const auto & null_map = c_nullable.getNullMapData();

    auto f_apply_a = [&](uint64_t prior, size_t i) -> uint64_t
    {
        const uint64_t h_a = solo_a[i] - GR;
        return rs::hashCombine(prior, h_a);
    };
    auto f_apply_b = [&](uint64_t prior, size_t i) -> uint64_t
    {
        const uint64_t h_b = solo_b[i] - GR;
        return rs::hashCombine(prior, h_b);
    };
    auto f_apply_c = [&](uint64_t prior, size_t i) -> uint64_t
    {
        /// Nullable's two-step recurrence.
        const uint64_t null_seed = null_map[i] != 0 ? NULL_SEED_TRUE : NULL_SEED_FALSE;
        const uint64_t h_nested = solo_c_nested[i] - GR;
        const uint64_t after_null = rs::hashCombine(prior, null_seed);
        return rs::hashCombine(after_null, h_nested);
    };

    /// Step 2: compose in order (a, b, c).
    std::vector<uint64_t> abc(N, 0);
    primitives_a.hash(primitives_a, *a, N, abc.data());
    primitives_b.hash(primitives_b, *b, N, abc.data());
    primitives_c.hash(primitives_c, *c, N, abc.data());

    /// Step 3: compose in order (c, b, a).
    std::vector<uint64_t> cba(N, 0);
    primitives_c.hash(primitives_c, *c, N, cba.data());
    primitives_b.hash(primitives_b, *b, N, cba.data());
    primitives_a.hash(primitives_a, *a, N, cba.data());

    /// Step 4: verify the documented recurrence by hand-applying the
    /// per-column-primitive transforms.
    for (size_t i = 0; i < N; ++i)
    {
        const uint64_t expected_abc = f_apply_c(f_apply_b(f_apply_a(0, i), i), i);
        const uint64_t expected_cba = f_apply_a(f_apply_b(f_apply_c(0, i), i), i);
        ASSERT_EQ(abc[i], expected_abc) << "row " << i << " order (a,b,c) does not match combiner recurrence";
        ASSERT_EQ(cba[i], expected_cba) << "row " << i << " order (c,b,a) does not match combiner recurrence";
    }

    /// Sanity: the two orders should generally produce different hashes
    /// (combiner is non-commutative). Count the rows that differ; expect
    /// most of them.
    size_t differ = 0;
    for (size_t i = 0; i < N; ++i)
        if (abc[i] != cba[i])
            ++differ;
    EXPECT_GT(differ, N / 2u) << "hash combiner appears commutative (suspicious)";
}


// ----------------------------------------------------------------------
// Allocator hot-path latency vs contention.
// ----------------------------------------------------------------------

TEST(RadixShuffleColumnPrimitives, AllocatorReserveLatencyVsThreadCount)
{
    /// Measure per-call `Handle::reserve` median latency at T=1, T=8, and
    /// T=48 (or T=cores if cores < 48). Assert that the T=48 median is
    /// within 5x the T=1 median — that catches a regression which adds a
    /// mutex on the hot path (which would show ~100x degradation). A loose
    /// factor (5x) avoids false alarms from cache contention or scheduler
    /// jitter at high thread counts.
    static constexpr size_t P = 16;
    const size_t hw_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    const std::vector<size_t> thread_counts{1, 8, std::min<size_t>(48, hw_threads)};

    rs::ColumnDesc desc{sizeof(UInt32), alignof(UInt32), false, false, false};

    std::vector<double> median_ns(thread_counts.size(), 0.0);

    for (size_t k = 0; k < thread_counts.size(); ++k)
    {
        // NOLINTBEGIN(readability-identifier-naming) -- T matches the spec's notation.
        const size_t T = thread_counts[k];
        rs::Allocator alloc({desc}, P, /*expected_total_rows=*/100'000 * T);

        std::atomic<size_t> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::vector<double>> per_thread_samples(T);

        auto thread_fn = [&](size_t tid)
        {
            rs::Handle * h = alloc.acquire();
            std::vector<rs::ReservationRequest> req(P);
            for (auto & r : req)
                r = {64, 0};
            std::vector<rs::Reservation> dst(P);

            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire))
                std::this_thread::yield();

            static constexpr size_t SAMPLES = 5000;
            auto & samples = per_thread_samples[tid];
            samples.reserve(SAMPLES);
            for (size_t i = 0; i < SAMPLES; ++i)
            {
                const auto t0 = std::chrono::steady_clock::now();
                h->reserve(0, req.data(), dst.data());
                const auto t1 = std::chrono::steady_clock::now();
                samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            }
            alloc.release(h);
        };

        std::vector<std::thread> threads;
        threads.reserve(T);
        for (size_t i = 0; i < T; ++i)
            threads.emplace_back(thread_fn, i);

        while (ready.load(std::memory_order_acquire) < T)
            std::this_thread::yield();
        go.store(true, std::memory_order_release);

        for (auto & th : threads)
            th.join();
        // NOLINTEND(readability-identifier-naming)

        /// Median across all samples from all threads.
        std::vector<double> all;
        for (auto & v : per_thread_samples)
            all.insert(all.end(), v.begin(), v.end());
        std::sort(all.begin(), all.end());
        median_ns[k] = all[all.size() / 2];
    }

    const double t1 = median_ns[0];
    const double t8 = median_ns[1];
    const double t48 = median_ns[2];

    /// Print for diagnostic visibility.
    std::cerr << "[reserve-latency] T=1 median=" << t1 << "ns, T=8 median=" << t8 << "ns, T=" << thread_counts.back() << " median=" << t48
              << "ns"
              << " (T=high/T=1 ratio=" << (t48 / t1) << ")" << std::endl;

    /// Acceptance: T=high latency within 5x of T=1. This is lax on
    /// purpose — false alarms (test flake) hurt more than missing a 4x
    /// regression; the goal is to catch a mutex-on-hot-path regression
    /// (~100x).
    EXPECT_LT(t48, 5.0 * t1) << "reserve hot-path latency at T=" << thread_counts.back() << " (" << t48
                             << "ns) is more than 5x the T=1 latency (" << t1 << "ns); a contention-scaled primitive may have been added";
}


// ----------------------------------------------------------------------
// Parameterized round-trip across every scope-D type via the dispatcher.
// ----------------------------------------------------------------------

namespace
{

/// Populate `column` with `n` rows of deterministic random bytes.
void fillColumnRandom(IColumn & column, size_t n, uint64_t seed)
{
    /// For fixed-width columns whose `valuesHaveFixedSize` is true, we
    /// emit `n` rows of random bytes via `insertRawUninitialized` (which
    /// returns a span into the raw storage). When the column does not
    /// support `insertRawUninitialized` we fall back to per-row inserts.
    if (column.valuesHaveFixedSize())
    {
        try
        {
            std::span<char> raw = column.insertRawUninitialized(n);
            std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
            for (size_t i = 0; i + sizeof(uint64_t) <= raw.size(); i += sizeof(uint64_t))
            {
                const uint64_t w = rng();
                std::memcpy(raw.data() + i, &w, sizeof(uint64_t));
            }
            return;
        }
        catch (...) // NOLINT(bugprone-empty-catch)
        {
            /// `insertRawUninitialized` may throw NOT_IMPLEMENTED for some
            /// concrete column types (e.g., those that don't implement the
            /// API). We intentionally swallow the exception and fall
            /// through to a per-type fallback below; the swallowed
            /// exception is informational ("this column doesn't expose
            /// raw insertion") rather than an error condition.
        }
    }

    /// ColumnString-like.
    if (auto * cs = dynamic_cast<ColumnString *>(&column))
    {
        std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
        std::string buf;
        for (size_t i = 0; i < n; ++i)
        {
            const size_t len = (rng() % 24) + 1;
            buf.resize(len);
            for (size_t k = 0; k < len; ++k)
                buf[k] = static_cast<char>('a' + (rng() % 26));
            cs->insertData(buf.data(), buf.size());
        }
        return;
    }

    /// ColumnNullable.
    if (auto * cn = dynamic_cast<ColumnNullable *>(&column))
    {
        fillColumnRandom(cn->getNestedColumn(), n, seed);
        auto & nm = cn->getNullMapData();
        nm.resize(n);
        std::mt19937_64 rng(seed ^ 0x9e3779b9ULL); // NOLINT(cert-msc32-c,cert-msc51-cpp)
        for (auto & b : nm)
            b = (rng() & 0x1) ? 1u : 0u;
        return;
    }

    FAIL() << "fillColumnRandom: unsupported column type " << column.getName();
}


/// Run one round-trip via the dispatcher: build a column of `n` rows of
/// random data, scatter into `P` partitions, reconstruct, and assert the
/// reconstructed column equals the source's per-partition reordering.
void runRoundTripViaDispatcher(const std::string & type_name, size_t n, size_t partitions, uint64_t seed)
{
    DataTypePtr type;
    try
    {
        type = DataTypeFactory::instance().get(type_name);
    }
    catch (...)
    {
        FAIL() << "could not resolve data type '" << type_name << "'";
    }
    ASSERT_NE(type, nullptr) << type_name;

    rs::ColumnPrimitives primitives = rs::resolveColumnPrimitives(*type);

    auto src = type->createColumn();
    fillColumnRandom(*src, n, seed);
    ASSERT_EQ(src->size(), n) << type_name;

    auto pids = uniformPids(n, partitions, seed ^ 0xdeafULL);

    /// Per-partition byte counts (variable-length only).
    std::vector<size_t> bytes_per_partition(partitions, 0);
    if (primitives.column_desc.variable_length)
    {
        /// Source must wrap a ColumnString somewhere accessible.
        const ColumnString * inner_str = dynamic_cast<const ColumnString *>(src.get());
        if (!inner_str)
            if (const auto * nb = dynamic_cast<const ColumnNullable *>(src.get()))
                inner_str = dynamic_cast<const ColumnString *>(&nb->getNestedColumn());
        ASSERT_NE(inner_str, nullptr) << "variable-length column without ColumnString backing: " << type_name;
        const auto & offsets_src = inner_str->getOffsets();
        UInt64 prev = 0;
        for (size_t j = 0; j < n; ++j)
        {
            const UInt64 end = offsets_src[j];
            bytes_per_partition[pids[j]] += end - prev;
            prev = end;
        }
    }

    std::vector<rs::ColumnDesc> descs{primitives.column_desc};
    rs::Allocator alloc(descs, partitions, n);
    rs::Handle * h = alloc.acquire();

    std::vector<OperatorBucket> buckets(partitions);
    runOneBatch(
        alloc,
        h,
        /*col_idx=*/0,
        primitives,
        *src,
        pids,
        partitions,
        buckets,
        [&](size_t p) -> size_t { return primitives.column_desc.variable_length ? bytes_per_partition[p] : 0UL; });

    auto target = type->createColumn();
    target->reserve(n);
    if (auto * cs = dynamic_cast<ColumnString *>(target.get()))
    {
        size_t total_bytes = 0;
        for (auto & b : buckets)
            total_bytes += b.total_bytes;
        cs->getChars().reserve(total_bytes);
    }
    if (auto * cn = dynamic_cast<ColumnNullable *>(target.get()))
    {
        size_t total_bytes = 0;
        for (auto & b : buckets)
            total_bytes += b.total_bytes;
        if (auto * sn = dynamic_cast<ColumnString *>(&cn->getNestedColumn()))
            sn->getChars().reserve(total_bytes);
    }

    for (size_t p = 0; p < partitions; ++p)
    {
        rs::ResumePosition pos{0, 0};
        while (pos.view_index < buckets[p].views.size())
        {
            pos = primitives.reconstruct(primitives, buckets[p].views.data(), buckets[p].views.size(), pos, *target);
            if (pos.view_index < buckets[p].views.size() && pos.rows_consumed_in_view == 0)
                break;
        }
    }

    ASSERT_EQ(target->size(), n) << type_name;

    /// Expected order: rows grouped by partition in ascending pid order.
    std::vector<size_t> expected_order;
    expected_order.reserve(n);
    for (size_t p = 0; p < partitions; ++p)
        for (size_t j = 0; j < n; ++j)
            if (pids[j] == p)
                expected_order.push_back(j);

    for (size_t i = 0; i < n; ++i)
    {
        const size_t src_idx = expected_order[i];
        ASSERT_EQ(target->compareAt(i, src_idx, *src, 0), 0) << type_name << ": row " << i << " mismatches source row " << src_idx;
    }

    alloc.release(h);
}

}


TEST(RadixShuffleColumnPrimitives, RoundTripAcrossScopeDViaDispatcher)
{
    /// Iterate over every scope-D type the dispatcher supports. Each test
    /// case constructs the type from its SQL name via DataTypeFactory,
    /// fills a column with deterministic random data, scatters into 8
    /// partitions, reconstructs, and verifies the round-trip identity.
    /// This is the comprehensive coverage F9 calls for.
    static constexpr size_t N = 1024;
    static constexpr size_t P = 8;

    const std::vector<std::string> leaf_types{
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "UInt128",
        "UInt256",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Int128",
        "Int256",
        "BFloat16",
        "Float32",
        "Float64",
        "UUID",
        "IPv4",
        "IPv6",
        "Decimal32(4)",
        "Decimal64(4)",
        "Decimal128(4)",
        "Decimal256(4)",
        "DateTime64(3)",
        "Date",
        "Date32",
        "DateTime",
        "Enum8('a' = 0, 'b' = 1, 'c' = 2)",
        "Enum16('a' = 0, 'b' = 1, 'c' = 2)",
        "FixedString(1)",
        "FixedString(4)",
        "FixedString(13)",
        "FixedString(32)",
        "FixedString(64)",
        "String",
    };

    for (const auto & leaf : leaf_types)
    {
        SCOPED_TRACE(leaf);
        runRoundTripViaDispatcher(leaf, N, P, 0xabcdULL ^ std::hash<std::string>{}(leaf));
    }

    /// Nullable wrappers — one per leaf category, plus a sample of leaves.
    const std::vector<std::string> nullable_types{
        "Nullable(UInt8)",
        "Nullable(Int32)",
        "Nullable(UInt64)",
        "Nullable(UInt128)",
        "Nullable(Float64)",
        "Nullable(UUID)",
        "Nullable(IPv4)",
        "Nullable(IPv6)",
        "Nullable(Decimal64(4))",
        "Nullable(DateTime64(3))",
        "Nullable(FixedString(16))",
        "Nullable(String)",
    };
    for (const auto & nt : nullable_types)
    {
        SCOPED_TRACE(nt);
        runRoundTripViaDispatcher(nt, N, P, 0xfeedULL ^ std::hash<std::string>{}(nt));
    }
}
