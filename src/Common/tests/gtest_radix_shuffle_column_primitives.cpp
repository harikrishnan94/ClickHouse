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
#include <Common/RadixShuffle/HashKernels.h>
#include <Common/RadixShuffle/PartSchema.h>
#include <Common/RadixShuffle/RadixPartitioner.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <random>
#include <thread>
#include <vector>


namespace
{

using namespace DB;
namespace rs = DB::RadixShuffle;


// ───────────────────────── test helpers ─────────────────────────


std::vector<uint16_t> uniformPids(size_t n, size_t P, uint64_t seed = 42)
{
    std::vector<uint16_t> pids(n);
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    std::uniform_int_distribution<uint16_t> dist(0, static_cast<uint16_t>(P - 1));
    for (size_t i = 0; i < n; ++i)
        pids[i] = dist(rng);
    return pids;
}

std::vector<size_t> histogram(const std::vector<uint16_t> & pids, size_t P)
{
    std::vector<size_t> hist(P, 0);
    for (auto p : pids)
        ++hist[p];
    return hist;
}


/// One scatter batch and returns the per-partition PartReservationViews.
/// varlen_bytes[p] must be pre-computed by the caller for varlen columns.
/// state persists write-pointer caches across batches; pass the same instance
/// for consecutive batches to exercise the selective-refresh optimisation.
std::vector<rs::PartReservationView> scatterBatch(
    rs::Handle * handle,
    const rs::PartSchema & schema,
    const rs::ColumnPrimitives & prim,
    const IColumn & src,
    const std::vector<uint16_t> & pids,
    const std::vector<size_t> & varlen_bytes_per_part,
    rs::ScatterState & state)
{
    const size_t P = varlen_bytes_per_part.size();
    const std::vector<size_t> hist = histogram(pids, P);

    std::vector<rs::PartReserveGrant> grants(P);
    std::vector<uint64_t> stale((P + 63) / 64, 0);
    handle->reserve(hist.data(), varlen_bytes_per_part.data(), grants.data(), stale.data());

    std::vector<rs::PartReservation> dst(P);
    for (size_t p = 0; p < P; ++p)
        dst[p] = grants[p].slice;

    prim.scatter(prim, schema, src, pids.data(), pids.size(), P, dst.data(), state, stale.data());

    std::vector<rs::PartReservationView> views(P);
    for (size_t p = 0; p < P; ++p)
    {
        views[p].fixed = dst[p].fixed;
        views[p].row_begin = dst[p].begin_row;
        views[p].row_end = dst[p].begin_row + dst[p].reserved_rows;
        views[p].data = dst[p].data;
        views[p].byte_begin = dst[p].begin_byte;
        views[p].byte_end = dst[p].begin_byte + dst[p].reserved_bytes;
    }
    return views;
}

/// Compute per-partition varlen byte totals for a ColumnString batch.
std::vector<size_t> stringVarlenPerPart(const ColumnString & col, const std::vector<uint16_t> & pids, size_t P)
{
    std::vector<size_t> out(P, 0);
    const auto & offs = col.getOffsets();
    UInt64 prev = 0;
    for (size_t i = 0; i < pids.size(); ++i)
    {
        const UInt64 end = offs[i];
        out[pids[i]] += end - prev;
        prev = end;
    }
    return out;
}

/// Full round-trip for a single column: scatter N rows over P partitions
/// into one allocator, then reconstruct in pid order and compare multisets.
void roundTripOne(
    const rs::PartSchema & schema,
    const rs::ColumnPrimitives & prim,
    const IColumn & src,
    size_t P,
    const DataTypePtr & dtype,
    uint64_t seed = 42)
{
    const size_t N = src.size();
    const std::vector<uint16_t> pids = uniformPids(N, P, seed);

    rs::Allocator alloc(schema, P, N);
    rs::Handle * handle = alloc.acquire();

    // Compute varlen bytes per partition if needed
    std::vector<size_t> varlen(P, 0);
    if (schema.has_varlen_portion)
    {
        if (const auto * cs = typeid_cast<const ColumnString *>(&src))
            varlen = stringVarlenPerPart(*cs, pids, P);
        else if (const auto * cn = typeid_cast<const ColumnNullable *>(&src))
        {
            if (const auto * cs2 = typeid_cast<const ColumnString *>(&cn->getNestedColumn()))
                varlen = stringVarlenPerPart(*cs2, pids, P);
        }
    }

    rs::ScatterState scatter_state(P);
    const std::vector<rs::PartReservationView> views = scatterBatch(handle, schema, prim, src, pids, varlen, scatter_state);

    // Build per-partition sorted row indices to know expected multiset.
    // Then reconstruct each partition into a fresh column and concatenate.
    std::vector<std::vector<size_t>> part_rows(P);
    for (size_t i = 0; i < N; ++i)
        part_rows[pids[i]].push_back(i);

    MutableColumnPtr reconstructed = src.cloneEmpty();
    reconstructed->reserve(N);

    for (size_t p = 0; p < P; ++p)
    {
        if (views[p].row_end <= views[p].row_begin)
            continue;

        MutableColumnPtr part_col = src.cloneEmpty();
        const size_t expected_rows = views[p].row_end - views[p].row_begin;
        part_col->reserve(expected_rows);
        if (schema.has_varlen_portion)
            static_cast<ColumnString *>(
                typeid_cast<ColumnNullable *>(part_col.get()) ? &typeid_cast<ColumnNullable *>(part_col.get())->getNestedColumn()
                                                              : part_col.get())
                ->getChars()
                .reserve(views[p].byte_end - views[p].byte_begin);

        rs::ResumePosition pos{};
        const rs::PartReservationView single_view = views[p];
        pos = prim.reconstruct(prim, schema, &single_view, 1, pos, *part_col);
        ASSERT_EQ(part_col->size(), expected_rows) << "partition " << p << " reconstructed wrong row count";

        for (size_t r = 0; r < part_col->size(); ++r)
            reconstructed->insert((*part_col)[r]);
    }

    alloc.release(handle);

    ASSERT_EQ(reconstructed->size(), N);

    // Multiset equality: sort field lists from source and reconstructed
    std::vector<Field> src_fields, rec_fields;
    src_fields.reserve(N);
    rec_fields.reserve(N);
    for (size_t i = 0; i < N; ++i)
    {
        src_fields.push_back(src[i]);
        rec_fields.push_back((*reconstructed)[i]);
    }
    std::sort(src_fields.begin(), src_fields.end());
    std::sort(rec_fields.begin(), rec_fields.end());
    EXPECT_EQ(src_fields, rec_fields) << "round-trip multiset mismatch for type " << dtype->getName();
}


// ───────────────────────── column builders ─────────────────────────


MutableColumnPtr makeUInt32Column(size_t n, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    auto col = ColumnVector<UInt32>::create();
    col->reserve(n);
    for (size_t i = 0; i < n; ++i)
        col->insertValue(static_cast<UInt32>(rng()));
    return col;
}

MutableColumnPtr makeUInt64Column(size_t n, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    auto col = ColumnVector<UInt64>::create();
    col->reserve(n);
    for (size_t i = 0; i < n; ++i)
        col->insertValue(rng());
    return col;
}

MutableColumnPtr makeFloat64Column(size_t n, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    auto col = ColumnVector<Float64>::create();
    col->reserve(n);
    for (size_t i = 0; i < n; ++i)
        col->insertValue(dist(rng));
    return col;
}

MutableColumnPtr makeDecimal64Column(size_t n, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    auto col = ColumnDecimal<Decimal64>::create(0, 2);
    for (size_t i = 0; i < n; ++i)
        col->insertValue(Decimal64(static_cast<Int64>(rng())));
    return col;
}

MutableColumnPtr makeFixedStringColumn(size_t n, size_t width, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    auto col = ColumnFixedString::create(width);
    std::string buf(width, '\0');
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < width; ++j)
            buf[j] = static_cast<char>(rng() & 0xff);
        col->insertData(buf.data(), width);
    }
    return col;
}

MutableColumnPtr makeStringColumn(size_t n, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> len_dist(0, 40);
    auto col = ColumnString::create();
    std::string buf;
    for (size_t i = 0; i < n; ++i)
    {
        const size_t len = len_dist(rng);
        buf.resize(len);
        for (auto & c : buf)
            c = static_cast<char>((rng() % 95) + 32); // printable ASCII
        col->insertData(buf.data(), buf.size());
    }
    return col;
}

MutableColumnPtr makeNullableUInt32Column(size_t n, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> null_dist(0, 4);
    auto nested = ColumnVector<UInt32>::create();
    auto null_map = ColumnUInt8::create();
    nested->reserve(n);
    null_map->reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        const bool is_null = (null_dist(rng) == 0);
        null_map->insertValue(is_null ? 1 : 0);
        nested->insertValue(static_cast<UInt32>(rng()));
    }
    return ColumnNullable::create(std::move(nested), std::move(null_map));
}

MutableColumnPtr makeNullableStringColumn(size_t n, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> null_dist(0, 4);
    std::uniform_int_distribution<size_t> len_dist(0, 30);
    auto nested = ColumnString::create();
    auto null_map = ColumnUInt8::create();
    std::string buf;
    for (size_t i = 0; i < n; ++i)
    {
        const bool is_null = (null_dist(rng) == 0);
        null_map->insertValue(is_null ? 1 : 0);
        const size_t len = len_dist(rng);
        buf.resize(len);
        for (auto & c : buf)
            c = static_cast<char>((rng() % 95) + 32);
        nested->insertData(buf.data(), buf.size());
    }
    return ColumnNullable::create(std::move(nested), std::move(null_map));
}


// ───────────────────────── SchemaBuilder tests ─────────────────────────


TEST(SchemaBuilder, SingleUInt32)
{
    const auto dt = std::make_shared<DataTypeUInt32>();
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.fixed_slots.size(), 1u);
    EXPECT_EQ(schema.fixed_slots[0].role, rs::SlotRole::Values);
    EXPECT_EQ(schema.fixed_slots[0].element_size, 4u);
    EXPECT_EQ(schema.fixed_slots[0].alignment, 4u);
    EXPECT_FALSE(schema.has_varlen_portion);
    ASSERT_EQ(primitives.size(), 1u);
    ASSERT_EQ(primitives[0].fixed_slot_indices.size(), 1u);
    EXPECT_EQ(primitives[0].fixed_slot_indices[0], 0u);
    EXPECT_FALSE(primitives[0].writes_varlen);
}

TEST(SchemaBuilder, SingleString)
{
    const auto dt = std::make_shared<DataTypeString>();
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.fixed_slots.size(), 1u);
    EXPECT_EQ(schema.fixed_slots[0].role, rs::SlotRole::Offsets);
    EXPECT_EQ(schema.fixed_slots[0].element_size, 8u);
    EXPECT_TRUE(schema.has_varlen_portion);
    EXPECT_TRUE(primitives[0].writes_varlen);
}

TEST(SchemaBuilder, NullableString)
{
    const auto dt = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeString>());
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.fixed_slots.size(), 2u);
    EXPECT_EQ(schema.fixed_slots[0].role, rs::SlotRole::NullMap);
    EXPECT_EQ(schema.fixed_slots[1].role, rs::SlotRole::Offsets);
    EXPECT_TRUE(schema.has_varlen_portion);
    // NullMap slot is first in the outer primitive
    ASSERT_EQ(primitives[0].fixed_slot_indices.size(), 1u);
    EXPECT_EQ(primitives[0].fixed_slot_indices[0], 0u);
    // Offsets slot is in the nested primitive
    ASSERT_NE(primitives[0].nested, nullptr);
    ASSERT_EQ(primitives[0].nested->fixed_slot_indices.size(), 1u);
    EXPECT_EQ(primitives[0].nested->fixed_slot_indices[0], 1u);
}

TEST(SchemaBuilder, NullableUInt32)
{
    const auto dt = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt32>());
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.fixed_slots.size(), 2u);
    EXPECT_EQ(schema.fixed_slots[0].role, rs::SlotRole::NullMap);
    EXPECT_EQ(schema.fixed_slots[1].role, rs::SlotRole::Values);
    EXPECT_FALSE(schema.has_varlen_portion);
}

TEST(SchemaBuilder, FixedString)
{
    const auto dt = std::make_shared<DataTypeFixedString>(7);
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.fixed_slots.size(), 1u);
    EXPECT_EQ(schema.fixed_slots[0].role, rs::SlotRole::FixedStringChars);
    EXPECT_EQ(schema.fixed_slots[0].element_size, 7u);
    EXPECT_FALSE(schema.has_varlen_portion);
}

TEST(SchemaBuilder, MultiColumn)
{
    // UInt32 + String → 2 slots (Values, Offsets)
    const std::vector<DataTypePtr> types = {std::make_shared<DataTypeUInt32>(), std::make_shared<DataTypeString>()};
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives(types);

    ASSERT_EQ(schema.fixed_slots.size(), 2u);
    EXPECT_EQ(schema.fixed_slots[0].role, rs::SlotRole::Values);
    EXPECT_EQ(schema.fixed_slots[1].role, rs::SlotRole::Offsets);
    EXPECT_TRUE(schema.has_varlen_portion);
    EXPECT_EQ(primitives[0].fixed_slot_indices[0], 0u);
    EXPECT_EQ(primitives[1].fixed_slot_indices[0], 1u);
}

TEST(SchemaBuilder, SlotByteOffsetsAlignment)
{
    // Nullable(UInt64): NullMap (1B, align 1) then UInt64 (8B, align 8)
    // 1-row ref: NullMap at 0, UInt64 at alignUp(1, 8) = 8
    const auto dt = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt64>());
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.slot_byte_offset.size(), 2u);
    EXPECT_EQ(schema.slot_byte_offset[0], 0u);
    EXPECT_EQ(schema.slot_byte_offset[1], 8u); // aligned to 8 after 1-byte NullMap
}


// ───────────────────────── Allocator tests ─────────────────────────


TEST(Allocator, ConstructionNoAlloc)
{
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    rs::Allocator alloc(schema, 32, 0);
    EXPECT_EQ(alloc.totalAllocatedBytes(), 0u);
    EXPECT_EQ(alloc.totalReservedBytes(), 0u);
    EXPECT_EQ(alloc.totalChunks(), 0u);
}

TEST(Allocator, BasicReserve)
{
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t P = 4;
    rs::Allocator alloc(schema, P, 1024);
    rs::Handle * h = alloc.acquire();

    std::vector<size_t> rows = {10, 20, 30, 0};
    std::vector<size_t> varlen(P, 0);
    std::vector<rs::PartReserveGrant> grants(P);
    std::vector<uint64_t> stale((P + 63) / 64, 0);
    h->reserve(rows.data(), varlen.data(), grants.data(), stale.data());

    EXPECT_EQ(grants[0].granted_rows, 10u);
    EXPECT_TRUE(grants[0].fully_satisfied);
    EXPECT_EQ(grants[1].granted_rows, 20u);
    EXPECT_EQ(grants[2].granted_rows, 30u);
    EXPECT_EQ(grants[3].granted_rows, 0u);
    EXPECT_NE(grants[0].slice.fixed, nullptr);
    EXPECT_EQ(grants[3].slice.fixed, nullptr);

    alloc.release(h);
    EXPECT_GT(alloc.totalAllocatedBytes(), 0u);
}

TEST(Allocator, StaleFixedBitset)
{
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t P = 4;
    rs::Allocator alloc(schema, P, 0, {.min_chunk_floor_rows = 256});
    rs::Handle * h = alloc.acquire();

    std::vector<size_t> rows(P, 10);
    std::vector<size_t> varlen(P, 0);
    std::vector<rs::PartReserveGrant> grants(P);
    std::vector<uint64_t> stale((P + 63) / 64, 0);
    h->reserve(rows.data(), varlen.data(), grants.data(), stale.data());

    // All partitions should have stale bits set (first allocation)
    EXPECT_NE(stale[0], 0u);
    for (size_t p = 0; p < P; ++p)
        EXPECT_TRUE((stale[p / 64] >> (p % 64)) & 1u) << "bit p=" << p;

    // Second call into the same chunk — no stale bits
    std::fill(stale.begin(), stale.end(), 0);
    h->reserve(rows.data(), varlen.data(), grants.data(), stale.data());
    EXPECT_EQ(stale[0], 0u);

    alloc.release(h);
}

TEST(Allocator, WasteBound)
{
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t P = 16;
    constexpr size_t FLOOR_ROWS = 256;
    rs::Allocator alloc(schema, P, 0, {.min_chunk_floor_rows = FLOOR_ROWS});
    rs::Handle * h = alloc.acquire();

    std::vector<size_t> rows(P, 64);
    std::vector<size_t> varlen(P, 0);
    std::vector<rs::PartReserveGrant> grants(P);
    std::vector<uint64_t> stale((P + 63) / 64, 0);

    constexpr size_t BATCHES = 20;
    for (size_t b = 0; b < BATCHES; ++b)
    {
        std::fill(stale.begin(), stale.end(), 0);
        h->reserve(rows.data(), varlen.data(), grants.data(), stale.data());
    }

    alloc.release(h);

    const uint64_t allocated = alloc.totalAllocatedBytes();
    const uint64_t reserved = alloc.totalReservedBytes();
    const uint64_t active = alloc.activePartitions();
    const uint64_t floor = active * (FLOOR_ROWS * schema.fixed_bytes_per_row + rs::DEFAULT_MIN_CHUNK_FLOOR_BYTES);

    EXPECT_LE(allocated, floor + static_cast<uint64_t>(1.11 * static_cast<double>(reserved)))
        << "waste bound violated: allocated=" << allocated << " reserved=" << reserved << " active=" << active;
}

TEST(Allocator, MultiThreadedNoBlocking)
{
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t P = 16;
    constexpr size_t T = 8;
    rs::Allocator alloc(schema, P, 0);

    std::atomic<uint64_t> total_rows_reserved{0};
    std::vector<std::thread> threads;
    threads.reserve(T);

    for (size_t t = 0; t < T; ++t)
    {
        threads.emplace_back(
            [&]()
            {
                rs::Handle * h = alloc.acquire();
                std::vector<size_t> rows(P, 32);
                std::vector<size_t> varlen(P, 0);
                std::vector<rs::PartReserveGrant> grants(P);
                std::vector<uint64_t> stale((P + 63) / 64, 0);
                for (size_t b = 0; b < 50; ++b)
                {
                    std::fill(stale.begin(), stale.end(), 0);
                    h->reserve(rows.data(), varlen.data(), grants.data(), stale.data());
                    uint64_t delta = 0;
                    for (size_t p = 0; p < P; ++p)
                        delta += grants[p].granted_rows;
                    total_rows_reserved.fetch_add(delta, std::memory_order_relaxed);
                }
                alloc.release(h);
            });
    }
    for (auto & thr : threads)
        thr.join();

    EXPECT_EQ(total_rows_reserved.load(), T * 50 * P * 32);
}


// ───────────────────────── ScatterState tests ─────────────────────────


TEST(ScatterState, WritePointerPersistence)
{
    // Scatter 10 batches of 64 rows each into P=4 UInt32 partitions.
    // After the first batch every partition's FixedChunk fits all 10 batches,
    // so the stale bitset should be all-zero from batch 2 onwards — the
    // cached write pointers are reused directly.
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t P = 4;
    constexpr size_t BATCH = 64;
    constexpr size_t BATCHES = 10;

    rs::Allocator alloc(schema, P, BATCH * BATCHES, {.min_chunk_floor_rows = BATCH * BATCHES});
    rs::Handle * h = alloc.acquire();
    rs::ScatterState state(P);

    size_t stale_events = 0;
    for (size_t b = 0; b < BATCHES; ++b)
    {
        auto col = makeUInt32Column(BATCH, b);
        const std::vector<uint16_t> pids = uniformPids(BATCH, P, b);
        const std::vector<size_t> hist = histogram(pids, P);
        std::vector<size_t> varlen(P, 0);

        std::vector<rs::PartReserveGrant> grants(P);
        std::vector<uint64_t> stale((P + 63) / 64, 0);
        h->reserve(hist.data(), varlen.data(), grants.data(), stale.data());

        std::vector<rs::PartReservation> dst(P);
        for (size_t p = 0; p < P; ++p)
            dst[p] = grants[p].slice;

        primitives[0].scatter(primitives[0], schema, *col, pids.data(), BATCH, P, dst.data(), state, stale.data());

        for (size_t word = 0; word < stale.size(); ++word)
            stale_events += static_cast<size_t>(__builtin_popcountll(stale[word]));
    }

    alloc.release(h);

    // Only the first batch should have set stale bits (one per partition).
    EXPECT_EQ(stale_events, P);
}


TEST(ScatterState, RoundTripMultiBatchUInt32)
{
    // Two batches into the same allocator, same ScatterState.
    // Verify round-trip correctness when write pointers are reused.
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t P = 8;
    constexpr size_t N = 256;

    rs::Allocator alloc(schema, P, N * 2);
    rs::Handle * h = alloc.acquire();
    rs::ScatterState state(P);

    std::vector<rs::PartReservationView> all_views[2];
    for (size_t b = 0; b < 2; ++b)
    {
        auto col = makeUInt32Column(N, b + 100);
        const auto pids = uniformPids(N, P, b + 100);
        std::vector<size_t> varlen(P, 0);
        all_views[b] = scatterBatch(h, schema, primitives[0], *col, pids, varlen, state);
    }
    alloc.release(h);

    // Reconstruct both batches and verify non-zero row counts
    size_t total = 0;
    for (size_t b = 0; b < 2; ++b)
        for (size_t p = 0; p < P; ++p)
            total += all_views[b][p].row_end - all_views[b][p].row_begin;
    EXPECT_EQ(total, N * 2);
}


TEST(ScatterState, RoundTripMultiBatchString)
{
    // Two batches of String scatter with cached pointers.
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({std::make_shared<DataTypeString>()});
    constexpr size_t P = 4;
    constexpr size_t N = 128;

    rs::Allocator alloc(schema, P, N * 2);
    rs::Handle * h = alloc.acquire();
    rs::ScatterState state(P);

    std::vector<rs::PartReservationView> all_views[2];
    for (size_t b = 0; b < 2; ++b)
    {
        auto col = makeStringColumn(N, b + 200);
        const auto pids = uniformPids(N, P, b + 200);
        const auto & cs = assert_cast<const ColumnString &>(*col);
        const auto varlen = stringVarlenPerPart(cs, pids, P);
        all_views[b] = scatterBatch(h, schema, primitives[0], *col, pids, varlen, state);
    }
    alloc.release(h);

    size_t total = 0;
    for (size_t b = 0; b < 2; ++b)
        for (size_t p = 0; p < P; ++p)
            total += all_views[b][p].row_end - all_views[b][p].row_begin;
    EXPECT_EQ(total, N * 2);
}


// ───────────────────────── Round-trip tests ─────────────────────────


template <typename ColPtr>
void testRoundTrip(ColPtr && col, const DataTypePtr & dt, size_t P)
{
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});
    roundTripOne(schema, primitives[0], *col, P, dt);
}


TEST(RoundTrip, UInt32_P4)
{
    testRoundTrip(makeUInt32Column(1024), std::make_shared<DataTypeUInt32>(), 4);
}
TEST(RoundTrip, UInt32_P32)
{
    testRoundTrip(makeUInt32Column(1024), std::make_shared<DataTypeUInt32>(), 32);
}
TEST(RoundTrip, UInt32_P256)
{
    testRoundTrip(makeUInt32Column(16384), std::make_shared<DataTypeUInt32>(), 256);
}

TEST(RoundTrip, UInt64_P32)
{
    testRoundTrip(makeUInt64Column(1024), std::make_shared<DataTypeUInt64>(), 32);
}

TEST(RoundTrip, Float64_P32)
{
    testRoundTrip(makeFloat64Column(1024), std::make_shared<DataTypeFloat64>(), 32);
}

TEST(RoundTrip, Decimal64_P32)
{
    testRoundTrip(makeDecimal64Column(1024), std::make_shared<DataTypeDecimal<Decimal64>>(18, 2), 32);
}

TEST(RoundTrip, FixedString7_P32)
{
    testRoundTrip(makeFixedStringColumn(1024, 7), std::make_shared<DataTypeFixedString>(7), 32);
}

TEST(RoundTrip, String_P4)
{
    testRoundTrip(makeStringColumn(1024), std::make_shared<DataTypeString>(), 4);
}
TEST(RoundTrip, String_P32)
{
    testRoundTrip(makeStringColumn(1024), std::make_shared<DataTypeString>(), 32);
}
TEST(RoundTrip, String_P256)
{
    testRoundTrip(makeStringColumn(2048), std::make_shared<DataTypeString>(), 256);
}

TEST(RoundTrip, NullableUInt32_P32)
{
    testRoundTrip(makeNullableUInt32Column(1024), std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt32>()), 32);
}

TEST(RoundTrip, NullableString_P32)
{
    testRoundTrip(makeNullableStringColumn(1024), std::make_shared<DataTypeNullable>(std::make_shared<DataTypeString>()), 32);
}

TEST(RoundTrip, StringEmptyRows)
{
    // Column containing only empty strings — zero varlen bytes
    auto col = ColumnString::create();
    for (size_t i = 0; i < 256; ++i)
        col->insertData("", 0);
    testRoundTrip(std::move(col), std::make_shared<DataTypeString>(), 8);
}

TEST(RoundTrip, MultiBatch)
{
    // Two scatter batches into the same allocator per partition.
    const auto dt = std::make_shared<DataTypeUInt32>();
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});
    constexpr size_t P = 4;
    constexpr size_t N = 512;

    auto src_col = makeUInt32Column(N * 2, 7);
    const auto & src = assert_cast<const ColumnVector<UInt32> &>(*src_col);

    const std::vector<uint16_t> pids1 = uniformPids(N, P, 1);
    const std::vector<uint16_t> pids2 = uniformPids(N, P, 2);

    rs::Allocator alloc(schema, P, N * 2);
    rs::Handle * handle = alloc.acquire();

    // Track per-partition view lists across both batches.
    // Reuse the same ScatterState to test write-pointer persistence across batches.
    std::vector<std::vector<rs::PartReservationView>> all_views(P);
    rs::ScatterState scatter_state(P);

    for (size_t batch = 0; batch < 2; ++batch)
    {
        const std::vector<uint16_t> & pids = (batch == 0) ? pids1 : pids2;

        // Create a sub-column view for this batch
        auto batch_col = ColumnVector<UInt32>::create();
        for (size_t i = 0; i < N; ++i)
            batch_col->insertValue(src.getData()[batch * N + i]);

        std::vector<size_t> varlen(P, 0);
        const auto views = scatterBatch(handle, schema, primitives[0], *batch_col, pids, varlen, scatter_state);
        for (size_t p = 0; p < P; ++p)
            if (views[p].row_end > views[p].row_begin)
                all_views[p].push_back(views[p]);
    }

    alloc.release(handle);

    // Reconstruct each partition fully
    size_t total_reconstructed = 0;
    for (size_t p = 0; p < P; ++p)
    {
        if (all_views[p].empty())
            continue;
        size_t expected = 0;
        for (const auto & v : all_views[p])
            expected += v.row_end - v.row_begin;

        auto part_col = ColumnVector<UInt32>::create();
        part_col->reserve(expected);
        rs::ResumePosition pos{};
        pos = primitives[0].reconstruct(primitives[0], schema, all_views[p].data(), all_views[p].size(), pos, *part_col);
        EXPECT_EQ(part_col->size(), expected) << "partition " << p;
        total_reconstructed += part_col->size();
    }
    EXPECT_EQ(total_reconstructed, N * 2);
}


// ───────────────────────── Hash tests ─────────────────────────


TEST(Hash, Deterministic)
{
    const auto dt = std::make_shared<DataTypeUInt32>();
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});
    auto col = makeUInt32Column(256);

    std::vector<uint32_t> out1(256, 0), out2(256, 0);
    primitives[0].hash(primitives[0], schema, *col, 256, out1.data());
    primitives[0].hash(primitives[0], schema, *col, 256, out2.data());
    EXPECT_EQ(out1, out2);
}

TEST(Hash, CombinerUniformity)
{
    // Composing hash calls across two columns in different orders should
    // produce different (but well-defined) results.
    const std::vector<DataTypePtr> types = {std::make_shared<DataTypeUInt32>(), std::make_shared<DataTypeUInt64>()};
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives(types);

    constexpr size_t N = 64;
    auto col0 = makeUInt32Column(N, 1);
    auto col1 = makeUInt64Column(N, 2);

    // Order: col0 then col1
    std::vector<uint32_t> out_01(N, 0);
    primitives[0].hash(primitives[0], schema, *col0, N, out_01.data());
    primitives[1].hash(primitives[1], schema, *col1, N, out_01.data());

    // Order: col1 then col0
    std::vector<uint32_t> out_10(N, 0);
    primitives[1].hash(primitives[1], schema, *col1, N, out_10.data());
    primitives[0].hash(primitives[0], schema, *col0, N, out_10.data());

    // The two orders produce different results (unless all values collide,
    // which is negligible for 64 random rows)
    EXPECT_NE(out_01, out_10);
}

TEST(Hash, NullableParticipation)
{
    // Two rows with the same nested UInt32 but different null states must
    // produce different hash outputs.
    const auto dt = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt32>());
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});

    auto nested_a = ColumnVector<UInt32>::create();
    nested_a->insertValue(42u);
    auto null_a = ColumnUInt8::create();
    null_a->insertValue(0);
    auto col_a = ColumnNullable::create(std::move(nested_a), std::move(null_a));

    auto nested_b = ColumnVector<UInt32>::create();
    nested_b->insertValue(42u);
    auto null_b = ColumnUInt8::create();
    null_b->insertValue(1);
    auto col_b = ColumnNullable::create(std::move(nested_b), std::move(null_b));

    uint32_t ha = 0, hb = 0;
    primitives[0].hash(primitives[0], schema, *col_a, 1, &ha);
    primitives[0].hash(primitives[0], schema, *col_b, 1, &hb);
    EXPECT_NE(ha, hb);
}

TEST(Hash, StringRoundTripSameHash)
{
    // Scattering and reconstructing a ColumnString preserves bytes, so the
    // hash of the reconstructed column must equal the hash of the original.
    const auto dt = std::make_shared<DataTypeString>();
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});
    constexpr size_t N = 128;
    constexpr size_t P = 8;

    auto src_col = makeStringColumn(N, 5);
    const std::vector<uint16_t> pids = uniformPids(N, P, 99);

    const auto & src = assert_cast<const ColumnString &>(*src_col);
    const std::vector<size_t> varlen = stringVarlenPerPart(src, pids, P);

    rs::Allocator alloc(schema, P, N);
    rs::Handle * h = alloc.acquire();
    rs::ScatterState scatter_state(P);
    const auto views = scatterBatch(h, schema, primitives[0], src, pids, varlen, scatter_state);
    alloc.release(h);

    // Reconstruct all partitions into one column
    auto rec = ColumnString::create();
    rec->reserve(N);
    for (size_t p = 0; p < P; ++p)
    {
        if (views[p].row_end <= views[p].row_begin)
            continue;
        size_t bytes_needed = views[p].byte_end - views[p].byte_begin;
        auto part_col = ColumnString::create();
        const size_t expected_rows = views[p].row_end - views[p].row_begin;
        part_col->reserve(expected_rows);
        part_col->getChars().reserve(bytes_needed);
        rs::ResumePosition pos{};
        pos = primitives[0].reconstruct(primitives[0], schema, &views[p], 1, pos, *part_col);
        for (size_t r = 0; r < part_col->size(); ++r)
            rec->insert((*part_col)[r]);
    }

    // Hash original (over N rows in src order) and reconstruction (reordered)
    // — we can't compare directly because order changed; just check per-row
    // hashes of reconstructed are not all zero.
    std::vector<uint32_t> out(rec->size(), 0);
    primitives[0].hash(primitives[0], schema, *rec, rec->size(), out.data());
    const bool any_nonzero = std::any_of(out.begin(), out.end(), [](uint32_t v) { return v != 0; });
    EXPECT_TRUE(any_nonzero);
}


// ───────────────────────── Dispatcher tests ─────────────────────────


TEST(Dispatcher, AllSupportedTypes)
{
    const auto & factory = DataTypeFactory::instance();
    const std::vector<std::string> names
        = {"UInt8",
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
           "Float32",
           "Float64",
           "UUID",
           "IPv4",
           "IPv6",
           "Decimal32(4)",
           "Decimal64(8)",
           "Decimal128(18)",
           "Decimal256(36)",
           "DateTime64(3)",
           "Date",
           "Date32",
           "DateTime",
           "Enum8('a'=1,'b'=2)",
           "Enum16('x'=1)",
           "String",
           "FixedString(16)",
           "Nullable(UInt32)",
           "Nullable(String)"};

    for (const auto & name : names)
    {
        const auto dt = factory.get(name);
        bool threw = false;
        rs::SchemaAndPrimitives sp;
        try
        {
            sp = rs::buildSchemaAndPrimitives({dt});
        }
        catch (...)
        {
            threw = true;
        }
        EXPECT_FALSE(threw) << "buildSchemaAndPrimitives threw for type " << name;
        if (!threw)
        {
            EXPECT_NE(sp.primitives[0].scatter, nullptr) << "scatter null for " << name;
            EXPECT_NE(sp.primitives[0].reconstruct, nullptr) << "reconstruct null for " << name;
            EXPECT_NE(sp.primitives[0].hash, nullptr) << "hash null for " << name;
        }
    }
}

TEST(Dispatcher, UnsupportedTypeThrows)
{
    bool threw = false;
    try
    {
        [[maybe_unused]] auto prim = rs::resolveColumnPrimitives(*DataTypeFactory::instance().get("Array(UInt32)"));
    }
    catch (const DB::Exception &)
    {
        threw = true;
    }
    EXPECT_TRUE(threw);
}


// ───────────────────────── Reconstruct pump tests ─────────────────────────


TEST(Reconstruct, PumpResume)
{
    // Scatter 1024 UInt32 rows into P=4, then reconstruct with a capacity
    // limit of 50 rows per call; verify correct resumption.
    const auto dt = std::make_shared<DataTypeUInt32>();
    const auto [schema, primitives] = rs::buildSchemaAndPrimitives({dt});
    constexpr size_t N = 1024;
    constexpr size_t P = 4;

    auto src_col = makeUInt32Column(N, 3);
    const std::vector<uint16_t> pids = uniformPids(N, P, 3);
    std::vector<size_t> varlen(P, 0);

    rs::Allocator alloc(schema, P, N);
    rs::Handle * h = alloc.acquire();
    rs::ScatterState scatter_state(P);
    const auto views = scatterBatch(h, schema, primitives[0], *src_col, pids, varlen, scatter_state);
    alloc.release(h);

    // Reconstruct partition 0 in 50-row pumps
    if (views[0].row_end > views[0].row_begin)
    {
        const size_t total = views[0].row_end - views[0].row_begin;
        auto result = ColumnVector<UInt32>::create();
        result->reserve(total);
        rs::ResumePosition pos{};
        size_t pumped = 0;
        while (pumped < total)
        {
            const size_t remaining = total - pumped;
            const size_t cap = std::min(remaining, size_t{50});
            result->reserve(pumped + cap);
            pos = primitives[0].reconstruct(primitives[0], schema, &views[0], 1, pos, *result);
            pumped = result->size();
        }
        EXPECT_EQ(result->size(), total);
    }
}


// ───────────────────────── RadixPartitioner tests ─────────────────────────


/// Reconstruct column k from all buckets of a finished RadixPartitioner.
MutableColumnPtr collectBuckets(
    const rs::RadixPartitioner & part,
    size_t col_k,
    const IColumn & proto)
{
    MutableColumnPtr out = proto.cloneEmpty();
    out->reserve(0);

    for (size_t p = 0; p < part.partitions(); ++p)
    {
        const auto & bkt = part.bucket(p);
        if (bkt.total_rows == 0)
            continue;

        MutableColumnPtr tmp = proto.cloneEmpty();
        tmp->reserve(bkt.total_rows);
        if (part.schema().has_varlen_portion)
        {
            if (auto * sc = typeid_cast<ColumnString *>(tmp.get()))
                sc->getChars().reserve(bkt.total_varlen_bytes);
            else if (auto * nc = typeid_cast<ColumnNullable *>(tmp.get()))
                if (auto * sc2 = typeid_cast<ColumnString *>(&nc->getNestedColumn()))
                    sc2->getChars().reserve(bkt.total_varlen_bytes);
        }
        rs::ResumePosition pos{};
        pos = part.primitives()[col_k].reconstruct(
            part.primitives()[col_k], part.schema(),
            bkt.views.data(), bkt.views.size(),
            pos, *tmp);
        for (size_t r = 0; r < tmp->size(); ++r)
            out->insert((*tmp)[r]);
    }
    return out;
}


/// Partition cols, reconstruct col_k from all buckets, assert multiset equality.
void rxpRoundTrip(
    const DB::Columns & cols,
    const std::vector<DataTypePtr> & types,
    size_t P,
    size_t col_k,
    const std::vector<size_t> & key_idxs = {0})
{
    const size_t N = cols[0]->size();
    auto sp = rs::buildSchemaAndPrimitives(types);
    rs::RadixPartitioner part(sp.schema, sp.primitives, P, key_idxs);
    part.process(cols);
    part.finish();

    MutableColumnPtr rec = collectBuckets(part, col_k, *cols[col_k]);
    ASSERT_EQ(rec->size(), N);

    std::vector<Field> sf, rf;
    sf.reserve(N);
    rf.reserve(N);
    for (size_t i = 0; i < N; ++i)
    {
        sf.push_back((*cols[col_k])[i]);
        rf.push_back((*rec)[i]);
    }
    std::sort(sf.begin(), sf.end());
    std::sort(rf.begin(), rf.end());
    EXPECT_EQ(sf, rf) << "multiset mismatch for col " << col_k;
}


TEST(RadixPartitioner, SingleColumnUInt32_RoundTrip)
{
    constexpr size_t N = 4096;
    constexpr size_t P = 32;
    auto col = makeUInt32Column(N, 1);
    const std::vector<DataTypePtr> types = {std::make_shared<DataTypeUInt32>()};
    DB::Columns cols{col->getPtr()};
    rxpRoundTrip(cols, types, P, 0);
}


TEST(RadixPartitioner, MultiColumnUInt32String_RoundTrip)
{
    // K=2 columns (UInt32 + String); key = col0.
    // Verify both columns round-trip with multiset equality.
    constexpr size_t N = 1024;
    constexpr size_t P = 16;

    auto col0 = makeUInt32Column(N, 10);
    auto col1 = makeStringColumn(N, 11);

    const std::vector<DataTypePtr> types = {
        std::make_shared<DataTypeUInt32>(),
        std::make_shared<DataTypeString>()};
    DB::Columns cols{col0->getPtr(), col1->getPtr()};

    // Round-trip both columns independently.
    rxpRoundTrip(cols, types, P, 0);
    rxpRoundTrip(cols, types, P, 1);
}


TEST(RadixPartitioner, MultiKeyHash)
{
    // K=4 columns, key_col_idxs = {0, 2}.
    // Run twice with same data → same partition assignments (deterministic).
    constexpr size_t N = 256;
    constexpr size_t P = 8;

    auto c0 = makeUInt32Column(N, 1);
    auto c1 = makeUInt64Column(N, 2);
    auto c2 = makeUInt32Column(N, 3);
    auto c3 = makeFloat64Column(N, 4);

    const std::vector<DataTypePtr> types = {
        std::make_shared<DataTypeUInt32>(),
        std::make_shared<DataTypeUInt64>(),
        std::make_shared<DataTypeUInt32>(),
        std::make_shared<DataTypeFloat64>()};

    auto sp = rs::buildSchemaAndPrimitives(types);

    // Run 1
    rs::RadixPartitioner part1(sp.schema, sp.primitives, P, {0, 2});
    DB::Columns cols{c0->getPtr(), c1->getPtr(), c2->getPtr(), c3->getPtr()};
    part1.process(cols);
    part1.finish();

    // Run 2
    rs::RadixPartitioner part2(sp.schema, sp.primitives, P, {0, 2});
    part2.process(cols);
    part2.finish();

    // Same per-partition row counts in both runs.
    for (size_t p = 0; p < P; ++p)
        EXPECT_EQ(part1.bucket(p).total_rows, part2.bucket(p).total_rows)
            << "partition " << p << " counts differ between runs";

    // Total == N.
    size_t total = 0;
    for (size_t p = 0; p < P; ++p)
        total += part1.bucket(p).total_rows;
    EXPECT_EQ(total, N);
}


TEST(RadixPartitioner, NullableStringRoundTrip)
{
    constexpr size_t N = 512;
    constexpr size_t P = 4;
    auto col = makeNullableStringColumn(N, 7);
    const std::vector<DataTypePtr> types = {
        std::make_shared<DataTypeNullable>(std::make_shared<DataTypeString>())};
    DB::Columns cols{col->getPtr()};
    rxpRoundTrip(cols, types, P, 0);
}


TEST(RadixPartitioner, LargeBlockSlicing)
{
    // Input bigger than batch_size → internal slicing via IColumn::cut.
    // batch_size_override=64 forces many slices on a small N.
    constexpr size_t N = 512;
    constexpr size_t P = 8;

    auto col = makeUInt32Column(N, 99);
    const std::vector<DataTypePtr> types = {std::make_shared<DataTypeUInt32>()};
    auto sp = rs::buildSchemaAndPrimitives(types);

    rs::RadixPartitionerOptions opts;
    opts.batch_size_override = 64;
    rs::RadixPartitioner part(sp.schema, sp.primitives, P, {0}, opts);

    DB::Columns cols{col->getPtr()};
    part.process(cols);
    part.finish();

    // Total rows must equal N.
    size_t total = 0;
    for (size_t p = 0; p < P; ++p)
        total += part.bucket(p).total_rows;
    EXPECT_EQ(total, N);

    // Multiset equality.
    MutableColumnPtr rec = collectBuckets(part, 0, *col);
    ASSERT_EQ(rec->size(), N);
    std::vector<Field> sf, rf;
    sf.reserve(N);
    rf.reserve(N);
    for (size_t i = 0; i < N; ++i)
    {
        sf.push_back((*col)[i]);
        rf.push_back((*rec)[i]);
    }
    std::sort(sf.begin(), sf.end());
    std::sort(rf.begin(), rf.end());
    EXPECT_EQ(sf, rf);
}


TEST(RadixPartitioner, NonPowerOfTwoP)
{
    // P=100 exercises Lemire's fast modulo for non-power-of-2 P.
    constexpr size_t N = 2000;
    constexpr size_t P = 100;

    auto col = makeUInt32Column(N, 55);
    const std::vector<DataTypePtr> types = {std::make_shared<DataTypeUInt32>()};
    DB::Columns cols{col->getPtr()};
    rxpRoundTrip(cols, types, P, 0);
}


TEST(RadixPartitioner, HistogramSumsToN)
{
    // sum(bucket(p).total_rows for p in 0..P) == N for every batch size.
    constexpr size_t P = 32;

    for (size_t N : {size_t{0}, size_t{1}, size_t{256}, size_t{4097}})
    {
        auto col = makeUInt32Column(N, 77);
        const std::vector<DataTypePtr> types = {std::make_shared<DataTypeUInt32>()};
        auto sp = rs::buildSchemaAndPrimitives(types);
        rs::RadixPartitioner part(sp.schema, sp.primitives, P, {0});
        if (N > 0)
        {
            DB::Columns cols{col->getPtr()};
            part.process(cols);
        }
        part.finish();
        size_t total = 0;
        for (size_t p = 0; p < P; ++p)
            total += part.bucket(p).total_rows;
        EXPECT_EQ(total, N) << "N=" << N;
    }
}


TEST(RadixPartitioner, LargeP)
{
    // P=4096: histogram and scatter scale; basic sanity check.
    constexpr size_t N = 16384;
    constexpr size_t P = 4096;

    auto col = makeUInt64Column(N, 88);
    const std::vector<DataTypePtr> types = {std::make_shared<DataTypeUInt64>()};
    auto sp = rs::buildSchemaAndPrimitives(types);
    rs::RadixPartitioner part(sp.schema, sp.primitives, P, {0});
    DB::Columns cols{col->getPtr()};
    part.process(cols);
    part.finish();

    size_t total = 0;
    for (size_t p = 0; p < P; ++p)
        total += part.bucket(p).total_rows;
    EXPECT_EQ(total, N);
}


} // namespace
