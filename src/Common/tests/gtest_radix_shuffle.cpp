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
#include <Common/RadixShuffle/BumpArena.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>
#include <Common/RadixShuffle/ColumnPrimitives/Nullable.h>
#include <Common/RadixShuffle/ColumnPrimitives/String.h>
#include <Common/RadixShuffle/ColumnPrimitivesDispatch.h>
#include <Common/RadixShuffle/HashCombiner.h>
#include <Common/RadixShuffle/HashKernels.h>
#include <Common/RadixShuffle/OutBlock.h>
#include <Common/RadixShuffle/PartSchema.h>
#include <Common/RadixShuffle/RadixShuffler.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <random>
#include <thread>
#include <vector>


namespace
{

using namespace DB;


// ───────────────────────── test helpers ─────────────────────────


std::vector<uint16_t> uniformPids(size_t n, size_t num_partitions, uint64_t seed = 42)
{
    std::vector<uint16_t> pids(n);
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    std::uniform_int_distribution<uint16_t> dist(0, static_cast<uint16_t>(num_partitions - 1));
    for (size_t i = 0; i < n; ++i)
        pids[i] = dist(rng);
    return pids;
}

std::vector<size_t> histogram(const std::vector<uint16_t> & pids, size_t num_partitions)
{
    std::vector<size_t> hist(num_partitions, 0);
    for (auto p : pids)
        ++hist[p];
    return hist;
}


/// One scatter batch and returns the per-partition PartReservationViews.
/// varlen_bytes[p] must be pre-computed by the caller for varlen columns.
/// state persists write-pointer caches across batches; pass the same instance
/// for consecutive batches to exercise the selective-refresh optimisation.
std::vector<PartReservationView> scatterBatch(
    Handle * handle,
    const PartSchema & schema,
    const ColumnPrimitives & prim,
    const IColumn & src,
    const std::vector<uint16_t> & pids,
    const std::vector<size_t> & varlen_bytes_per_part,
    ScatterState & state)
{
    const size_t num_partitions = varlen_bytes_per_part.size();
    const std::vector<size_t> hist = histogram(pids, num_partitions);

    std::vector<PartReserveGrant> grants(num_partitions);
    std::vector<uint64_t> stale((num_partitions + 63) / 64, 0);
    handle->reserve(hist.data(), varlen_bytes_per_part.data(), grants.data(), stale.data());

    std::vector<PartReservation> dst(num_partitions);
    for (size_t p = 0; p < num_partitions; ++p)
        dst[p] = grants[p].slice;

    prim.scatter(prim, schema, src, pids.data(), pids.size(), num_partitions, dst.data(), state, stale.data());

    std::vector<PartReservationView> views(num_partitions);
    for (size_t p = 0; p < num_partitions; ++p)
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
std::vector<size_t> stringVarlenPerPart(const ColumnString & col, const std::vector<uint16_t> & pids, size_t num_partitions)
{
    std::vector<size_t> out(num_partitions, 0);
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
    const PartSchema & schema,
    const ColumnPrimitives & prim,
    const IColumn & src,
    size_t num_partitions,
    const DataTypePtr & dtype,
    uint64_t seed = 42)
{
    const size_t num_rows = src.size();
    const std::vector<uint16_t> pids = uniformPids(num_rows, num_partitions, seed);

    ShuffleAllocator alloc(schema, num_partitions, num_rows);
    Handle * handle = alloc.acquire();

    // Compute varlen bytes per partition if needed
    std::vector<size_t> varlen(num_partitions, 0);
    if (schema.has_varlen_portion)
    {
        if (const auto * cs = typeid_cast<const ColumnString *>(&src))
        {
            varlen = stringVarlenPerPart(*cs, pids, num_partitions);
        }
        else if (const auto * cn = typeid_cast<const ColumnNullable *>(&src))
        {
            if (const auto * cs2 = typeid_cast<const ColumnString *>(&cn->getNestedColumn()))
            {
                varlen = stringVarlenPerPart(*cs2, pids, num_partitions);
            }
        }
    }

    ScatterState scatter_state(num_partitions);
    const std::vector<PartReservationView> views = scatterBatch(handle, schema, prim, src, pids, varlen, scatter_state);

    // Build per-partition sorted row indices to know expected multiset.
    // Then reconstruct each partition into a fresh column and concatenate.
    std::vector<std::vector<size_t>> part_rows(num_partitions);
    for (size_t i = 0; i < num_rows; ++i)
        part_rows[pids[i]].push_back(i);

    MutableColumnPtr reconstructed = src.cloneEmpty();
    reconstructed->reserve(num_rows);

    for (size_t p = 0; p < num_partitions; ++p)
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

        ResumePosition pos{};
        const PartReservationView single_view = views[p];
        pos = prim.reconstruct(prim, schema, &single_view, 1, pos, *part_col);
        ASSERT_EQ(part_col->size(), expected_rows) << "partition " << p << " reconstructed wrong row count";

        for (size_t r = 0; r < part_col->size(); ++r)
            reconstructed->insert((*part_col)[r]);
    }

    alloc.release(handle);

    ASSERT_EQ(reconstructed->size(), num_rows);

    // Multiset equality: sort field lists from source and reconstructed
    std::vector<Field> src_fields;
    std::vector<Field> rec_fields;
    src_fields.reserve(num_rows);
    rec_fields.reserve(num_rows);
    for (size_t i = 0; i < num_rows; ++i)
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
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.fixed_slots.size(), 1u);
    EXPECT_EQ(schema.fixed_slots[0].role, SlotRole::Values);
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
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.fixed_slots.size(), 1u);
    EXPECT_EQ(schema.fixed_slots[0].role, SlotRole::Offsets);
    EXPECT_EQ(schema.fixed_slots[0].element_size, 8u);
    EXPECT_TRUE(schema.has_varlen_portion);
    EXPECT_TRUE(primitives[0].writes_varlen);
}

TEST(SchemaBuilder, NullableString)
{
    const auto dt = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeString>());
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.fixed_slots.size(), 2u);
    EXPECT_EQ(schema.fixed_slots[0].role, SlotRole::NullMap);
    EXPECT_EQ(schema.fixed_slots[1].role, SlotRole::Offsets);
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
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.fixed_slots.size(), 2u);
    EXPECT_EQ(schema.fixed_slots[0].role, SlotRole::NullMap);
    EXPECT_EQ(schema.fixed_slots[1].role, SlotRole::Values);
    EXPECT_FALSE(schema.has_varlen_portion);
}

TEST(SchemaBuilder, FixedString)
{
    const auto dt = std::make_shared<DataTypeFixedString>(7);
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.fixed_slots.size(), 1u);
    EXPECT_EQ(schema.fixed_slots[0].role, SlotRole::FixedStringChars);
    EXPECT_EQ(schema.fixed_slots[0].element_size, 7u);
    EXPECT_FALSE(schema.has_varlen_portion);
}

TEST(SchemaBuilder, MultiColumn)
{
    // UInt32 + String → 2 slots (Values, Offsets)
    const std::vector<DataTypePtr> types = {std::make_shared<DataTypeUInt32>(), std::make_shared<DataTypeString>()};
    const auto [schema, primitives] = buildSchemaAndPrimitives(types);

    ASSERT_EQ(schema.fixed_slots.size(), 2u);
    EXPECT_EQ(schema.fixed_slots[0].role, SlotRole::Values);
    EXPECT_EQ(schema.fixed_slots[1].role, SlotRole::Offsets);
    EXPECT_TRUE(schema.has_varlen_portion);
    EXPECT_EQ(primitives[0].fixed_slot_indices[0], 0u);
    EXPECT_EQ(primitives[1].fixed_slot_indices[0], 1u);
}

TEST(SchemaBuilder, SlotByteOffsetsAlignment)
{
    // Nullable(UInt64): NullMap (1B, align 1) then UInt64 (8B, align 8)
    // 1-row ref: NullMap at 0, UInt64 at alignUp(1, 8) = 8
    const auto dt = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt64>());
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});

    ASSERT_EQ(schema.slot_byte_offset.size(), 2u);
    EXPECT_EQ(schema.slot_byte_offset[0], 0u);
    EXPECT_EQ(schema.slot_byte_offset[1], 8u); // aligned to 8 after 1-byte NullMap
}


// ───────────────────────── ShuffleAllocator tests ─────────────────────────


TEST(ShuffleAllocator, ConstructionNoAlloc)
{
    const auto [schema, primitives] = buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    ShuffleAllocator alloc(schema, 32, 0);
    EXPECT_EQ(alloc.totalAllocatedBytes(), 0u);
    EXPECT_EQ(alloc.totalReservedBytes(), 0u);
    EXPECT_EQ(alloc.totalChunks(), 0u);
}

TEST(ShuffleAllocator, BasicReserve)
{
    const auto [schema, primitives] = buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t num_partitions = 4;
    ShuffleAllocator alloc(schema, num_partitions, 1024);
    Handle * h = alloc.acquire();

    std::vector<size_t> rows = {10, 20, 30, 0};
    std::vector<size_t> varlen(num_partitions, 0);
    std::vector<PartReserveGrant> grants(num_partitions);
    std::vector<uint64_t> stale((num_partitions + 63) / 64, 0);
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

TEST(ShuffleAllocator, StaleFixedBitset)
{
    const auto [schema, primitives] = buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t num_partitions = 4;
    ShuffleAllocator alloc(schema, num_partitions, 0, {.min_chunk_floor_rows = 256});
    Handle * h = alloc.acquire();

    std::vector<size_t> rows(num_partitions, 10);
    std::vector<size_t> varlen(num_partitions, 0);
    std::vector<PartReserveGrant> grants(num_partitions);
    std::vector<uint64_t> stale((num_partitions + 63) / 64, 0);
    h->reserve(rows.data(), varlen.data(), grants.data(), stale.data());

    // All partitions should have stale bits set (first allocation)
    EXPECT_NE(stale[0], 0u);
    for (size_t p = 0; p < num_partitions; ++p)
        EXPECT_TRUE((stale[p / 64] >> (p % 64)) & 1u) << "bit p=" << p;

    // Second call into the same chunk — no stale bits
    std::fill(stale.begin(), stale.end(), 0);
    h->reserve(rows.data(), varlen.data(), grants.data(), stale.data());
    EXPECT_EQ(stale[0], 0u);

    alloc.release(h);
}

TEST(ShuffleAllocator, WasteBound)
{
    const auto [schema, primitives] = buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t num_partitions = 16;
    constexpr size_t floor_rows = 256;
    ShuffleAllocator alloc(schema, num_partitions, 0, {.min_chunk_floor_rows = floor_rows});
    Handle * h = alloc.acquire();

    std::vector<size_t> rows(num_partitions, 64);
    std::vector<size_t> varlen(num_partitions, 0);
    std::vector<PartReserveGrant> grants(num_partitions);
    std::vector<uint64_t> stale((num_partitions + 63) / 64, 0);

    constexpr size_t num_batches = 20;
    for (size_t b = 0; b < num_batches; ++b)
    {
        std::fill(stale.begin(), stale.end(), 0);
        h->reserve(rows.data(), varlen.data(), grants.data(), stale.data());
    }

    alloc.release(h);

    const uint64_t allocated = alloc.totalAllocatedBytes();
    const uint64_t reserved = alloc.totalReservedBytes();
    const uint64_t active = alloc.activePartitions();
    const uint64_t floor = active * (floor_rows * schema.fixed_bytes_per_row + DEFAULT_MIN_CHUNK_FLOOR_BYTES);

    EXPECT_LE(allocated, floor + static_cast<uint64_t>(1.11 * static_cast<double>(reserved)))
        << "waste bound violated: allocated=" << allocated << " reserved=" << reserved << " active=" << active;
}

TEST(ShuffleAllocator, MultiThreadedNoBlocking)
{
    const auto [schema, primitives] = buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t num_partitions = 16;
    constexpr size_t num_threads = 8;
    ShuffleAllocator alloc(schema, num_partitions, 0);

    std::atomic<uint64_t> total_rows_reserved{0};
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&]()
            {
                Handle * h = alloc.acquire();
                std::vector<size_t> rows(num_partitions, 32);
                std::vector<size_t> varlen(num_partitions, 0);
                std::vector<PartReserveGrant> grants(num_partitions);
                std::vector<uint64_t> stale((num_partitions + 63) / 64, 0);
                for (size_t b = 0; b < 50; ++b)
                {
                    std::fill(stale.begin(), stale.end(), 0);
                    h->reserve(rows.data(), varlen.data(), grants.data(), stale.data());
                    uint64_t delta = 0;
                    for (size_t p = 0; p < num_partitions; ++p)
                        delta += grants[p].granted_rows;
                    total_rows_reserved.fetch_add(delta, std::memory_order_relaxed);
                }
                alloc.release(h);
            });
    }
    for (auto & thr : threads)
        thr.join();

    EXPECT_EQ(total_rows_reserved.load(), num_threads * 50 * num_partitions * 32);
}


// ───────────────────────── ScatterState tests ─────────────────────────


TEST(ScatterState, WritePointerPersistence)
{
    // Scatter 10 batches of 64 rows each into P=4 UInt32 partitions.
    // After the first batch every partition's FixedChunk fits all 10 batches,
    // so the stale bitset should be all-zero from batch 2 onwards — the
    // cached write pointers are reused directly.
    const auto [schema, primitives] = buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t num_partitions = 4;
    constexpr size_t batch_size = 64;
    constexpr size_t num_batches = 10;

    ShuffleAllocator alloc(schema, num_partitions, batch_size * num_batches, {.min_chunk_floor_rows = batch_size * num_batches});
    Handle * h = alloc.acquire();
    ScatterState state(num_partitions);

    size_t stale_events = 0;
    for (size_t b = 0; b < num_batches; ++b)
    {
        auto col = makeUInt32Column(batch_size, b);
        const std::vector<uint16_t> pids = uniformPids(batch_size, num_partitions, b);
        const std::vector<size_t> hist = histogram(pids, num_partitions);
        std::vector<size_t> varlen(num_partitions, 0);

        std::vector<PartReserveGrant> grants(num_partitions);
        std::vector<uint64_t> stale((num_partitions + 63) / 64, 0);
        h->reserve(hist.data(), varlen.data(), grants.data(), stale.data());

        std::vector<PartReservation> dst(num_partitions);
        for (size_t p = 0; p < num_partitions; ++p)
            dst[p] = grants[p].slice;

        primitives[0].scatter(primitives[0], schema, *col, pids.data(), batch_size, num_partitions, dst.data(), state, stale.data());

        for (const uint64_t word : stale)
            stale_events += static_cast<size_t>(__builtin_popcountll(word));
    }

    alloc.release(h);

    // Only the first batch should have set stale bits (one per partition).
    EXPECT_EQ(stale_events, num_partitions);
}


TEST(ScatterState, RoundTripMultiBatchUInt32)
{
    // Two batches into the same allocator, same ScatterState.
    // Verify round-trip correctness when write pointers are reused.
    const auto [schema, primitives] = buildSchemaAndPrimitives({std::make_shared<DataTypeUInt32>()});
    constexpr size_t num_partitions = 8;
    constexpr size_t num_rows = 256;

    ShuffleAllocator alloc(schema, num_partitions, num_rows * 2);
    Handle * h = alloc.acquire();
    ScatterState state(num_partitions);

    std::vector<PartReservationView> all_views[2];
    for (size_t b = 0; b < 2; ++b)
    {
        auto col = makeUInt32Column(num_rows, b + 100);
        const auto pids = uniformPids(num_rows, num_partitions, b + 100);
        std::vector<size_t> varlen(num_partitions, 0);
        all_views[b] = scatterBatch(h, schema, primitives[0], *col, pids, varlen, state);
    }
    alloc.release(h);

    // Reconstruct both batches and verify non-zero row counts
    size_t total = 0;
    for (const auto & batch_views : all_views)
        for (const auto & view : batch_views)
            total += view.row_end - view.row_begin;
    EXPECT_EQ(total, num_rows * 2);
}


TEST(ScatterState, RoundTripMultiBatchString)
{
    // Two batches of String scatter with cached pointers.
    const auto [schema, primitives] = buildSchemaAndPrimitives({std::make_shared<DataTypeString>()});
    constexpr size_t num_partitions = 4;
    constexpr size_t num_rows = 128;

    ShuffleAllocator alloc(schema, num_partitions, num_rows * 2);
    Handle * h = alloc.acquire();
    ScatterState state(num_partitions);

    std::vector<PartReservationView> all_views[2];
    for (size_t b = 0; b < 2; ++b)
    {
        auto col = makeStringColumn(num_rows, b + 200);
        const auto pids = uniformPids(num_rows, num_partitions, b + 200);
        const auto & cs = assert_cast<const ColumnString &>(*col);
        const auto varlen = stringVarlenPerPart(cs, pids, num_partitions);
        all_views[b] = scatterBatch(h, schema, primitives[0], *col, pids, varlen, state);
    }
    alloc.release(h);

    size_t total = 0;
    for (const auto & batch_views : all_views)
        for (const auto & view : batch_views)
            total += view.row_end - view.row_begin;
    EXPECT_EQ(total, num_rows * 2);
}


// ───────────────────────── Round-trip tests ─────────────────────────


template <typename ColPtr>
void testRoundTrip(ColPtr && col, const DataTypePtr & dt, size_t num_partitions)
{
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});
    roundTripOne(schema, primitives[0], *col, num_partitions, dt);
}


TEST(RoundTrip, UInt32P4)
{
    testRoundTrip(makeUInt32Column(1024), std::make_shared<DataTypeUInt32>(), 4);
}
TEST(RoundTrip, UInt32P32)
{
    testRoundTrip(makeUInt32Column(1024), std::make_shared<DataTypeUInt32>(), 32);
}
TEST(RoundTrip, UInt32P256)
{
    testRoundTrip(makeUInt32Column(16384), std::make_shared<DataTypeUInt32>(), 256);
}

TEST(RoundTrip, UInt64P32)
{
    testRoundTrip(makeUInt64Column(1024), std::make_shared<DataTypeUInt64>(), 32);
}

TEST(RoundTrip, Float64P32)
{
    testRoundTrip(makeFloat64Column(1024), std::make_shared<DataTypeFloat64>(), 32);
}

TEST(RoundTrip, Decimal64P32)
{
    testRoundTrip(makeDecimal64Column(1024), std::make_shared<DataTypeDecimal<Decimal64>>(18, 2), 32);
}

TEST(RoundTrip, FixedString7P32)
{
    testRoundTrip(makeFixedStringColumn(1024, 7), std::make_shared<DataTypeFixedString>(7), 32);
}

TEST(RoundTrip, StringP4)
{
    testRoundTrip(makeStringColumn(1024), std::make_shared<DataTypeString>(), 4);
}
TEST(RoundTrip, StringP32)
{
    testRoundTrip(makeStringColumn(1024), std::make_shared<DataTypeString>(), 32);
}
TEST(RoundTrip, StringP256)
{
    testRoundTrip(makeStringColumn(2048), std::make_shared<DataTypeString>(), 256);
}

TEST(RoundTrip, NullableUInt32P32)
{
    testRoundTrip(makeNullableUInt32Column(1024), std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt32>()), 32);
}

TEST(RoundTrip, NullableStringP32)
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
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});
    constexpr size_t num_partitions = 4;
    constexpr size_t num_rows = 512;

    auto src_col = makeUInt32Column(num_rows * 2, 7);
    const auto & src = assert_cast<const ColumnVector<UInt32> &>(*src_col);

    const std::vector<uint16_t> pids1 = uniformPids(num_rows, num_partitions, 1);
    const std::vector<uint16_t> pids2 = uniformPids(num_rows, num_partitions, 2);

    ShuffleAllocator alloc(schema, num_partitions, num_rows * 2);
    Handle * handle = alloc.acquire();

    // Track per-partition view lists across both batches.
    // Reuse the same ScatterState to test write-pointer persistence across batches.
    std::vector<std::vector<PartReservationView>> all_views(num_partitions);
    ScatterState scatter_state(num_partitions);

    for (size_t batch = 0; batch < 2; ++batch)
    {
        const std::vector<uint16_t> & pids = (batch == 0) ? pids1 : pids2;

        // Create a sub-column view for this batch
        auto batch_col = ColumnVector<UInt32>::create();
        for (size_t i = 0; i < num_rows; ++i)
            batch_col->insertValue(src.getData()[batch * num_rows + i]);

        std::vector<size_t> varlen(num_partitions, 0);
        const auto views = scatterBatch(handle, schema, primitives[0], *batch_col, pids, varlen, scatter_state);
        for (size_t p = 0; p < num_partitions; ++p)
            if (views[p].row_end > views[p].row_begin)
                all_views[p].push_back(views[p]);
    }

    alloc.release(handle);

    // Reconstruct each partition fully
    size_t total_reconstructed = 0;
    for (size_t p = 0; p < num_partitions; ++p)
    {
        if (all_views[p].empty())
            continue;
        size_t expected = 0;
        for (const auto & v : all_views[p])
            expected += v.row_end - v.row_begin;

        auto part_col = ColumnVector<UInt32>::create();
        part_col->reserve(expected);
        ResumePosition pos{};
        pos = primitives[0].reconstruct(primitives[0], schema, all_views[p].data(), all_views[p].size(), pos, *part_col);
        EXPECT_EQ(part_col->size(), expected) << "partition " << p;
        total_reconstructed += part_col->size();
    }
    EXPECT_EQ(total_reconstructed, num_rows * 2);
}


// ───────────────────────── Hash tests ─────────────────────────


TEST(Hash, Deterministic)
{
    const auto dt = std::make_shared<DataTypeUInt32>();
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});
    auto col = makeUInt32Column(256);

    std::vector<uint32_t> out1(256, 0);
    std::vector<uint32_t> out2(256, 0);
    primitives[0].hash(primitives[0], schema, *col, 0, 256, /*initial=*/true, out1.data());
    primitives[0].hash(primitives[0], schema, *col, 0, 256, /*initial=*/true, out2.data());
    EXPECT_EQ(out1, out2);
}

TEST(Hash, CombinerUniformity)
{
    // Composing hash calls across two columns in different orders should
    // produce different (but well-defined) results.
    const std::vector<DataTypePtr> types = {std::make_shared<DataTypeUInt32>(), std::make_shared<DataTypeUInt64>()};
    const auto [schema, primitives] = buildSchemaAndPrimitives(types);

    constexpr size_t num_rows = 64;
    auto col0 = makeUInt32Column(num_rows, 1);
    auto col1 = makeUInt64Column(num_rows, 2);

    // Order: col0 then col1
    std::vector<uint32_t> out_01(num_rows, 0);
    primitives[0].hash(primitives[0], schema, *col0, 0, num_rows, /*initial=*/true, out_01.data());
    primitives[1].hash(primitives[1], schema, *col1, 0, num_rows, /*initial=*/false, out_01.data());

    // Order: col1 then col0
    std::vector<uint32_t> out_10(num_rows, 0);
    primitives[1].hash(primitives[1], schema, *col1, 0, num_rows, /*initial=*/true, out_10.data());
    primitives[0].hash(primitives[0], schema, *col0, 0, num_rows, /*initial=*/false, out_10.data());

    // The two orders produce different results (unless all values collide,
    // which is negligible for 64 random rows)
    EXPECT_NE(out_01, out_10);
}

TEST(Hash, NullableParticipation)
{
    // Two rows with the same nested UInt32 but different null states must
    // produce different hash outputs.
    const auto dt = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt32>());
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});

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

    uint32_t ha = 0;
    uint32_t hb = 0;
    primitives[0].hash(primitives[0], schema, *col_a, 0, 1, /*initial=*/true, &ha);
    primitives[0].hash(primitives[0], schema, *col_b, 0, 1, /*initial=*/true, &hb);
    EXPECT_NE(ha, hb);
}

TEST(Hash, StringRoundTripSameHash)
{
    // Scattering and reconstructing a ColumnString preserves bytes, so the
    // hash of the reconstructed column must equal the hash of the original.
    const auto dt = std::make_shared<DataTypeString>();
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});
    constexpr size_t num_rows = 128;
    constexpr size_t num_partitions = 8;

    auto src_col = makeStringColumn(num_rows, 5);
    const std::vector<uint16_t> pids = uniformPids(num_rows, num_partitions, 99);

    const auto & src = assert_cast<const ColumnString &>(*src_col);
    const std::vector<size_t> varlen = stringVarlenPerPart(src, pids, num_partitions);

    ShuffleAllocator alloc(schema, num_partitions, num_rows);
    Handle * h = alloc.acquire();
    ScatterState scatter_state(num_partitions);
    const auto views = scatterBatch(h, schema, primitives[0], src, pids, varlen, scatter_state);
    alloc.release(h);

    // Reconstruct all partitions into one column
    auto rec = ColumnString::create();
    rec->reserve(num_rows);
    for (size_t p = 0; p < num_partitions; ++p)
    {
        if (views[p].row_end <= views[p].row_begin)
            continue;
        size_t bytes_needed = views[p].byte_end - views[p].byte_begin;
        auto part_col = ColumnString::create();
        const size_t expected_rows = views[p].row_end - views[p].row_begin;
        part_col->reserve(expected_rows);
        part_col->getChars().reserve(bytes_needed);
        ResumePosition pos{};
        pos = primitives[0].reconstruct(primitives[0], schema, &views[p], 1, pos, *part_col);
        for (size_t r = 0; r < part_col->size(); ++r)
            rec->insert((*part_col)[r]);
    }

    // Hash original (over N rows in src order) and reconstruction (reordered)
    // — we can't compare directly because order changed; just check per-row
    // hashes of reconstructed are not all zero.
    std::vector<uint32_t> out(rec->size(), 0);
    primitives[0].hash(primitives[0], schema, *rec, 0, rec->size(), /*initial=*/true, out.data());
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
        SchemaAndPrimitives sp;
        try
        {
            sp = buildSchemaAndPrimitives({dt});
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
        [[maybe_unused]] auto prim = resolveColumnPrimitives(*DataTypeFactory::instance().get("Array(UInt32)"));
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
    const auto [schema, primitives] = buildSchemaAndPrimitives({dt});
    constexpr size_t num_rows = 1024;
    constexpr size_t num_partitions = 4;

    auto src_col = makeUInt32Column(num_rows, 3);
    const std::vector<uint16_t> pids = uniformPids(num_rows, num_partitions, 3);
    std::vector<size_t> varlen(num_partitions, 0);

    ShuffleAllocator alloc(schema, num_partitions, num_rows);
    Handle * h = alloc.acquire();
    ScatterState scatter_state(num_partitions);
    const auto views = scatterBatch(h, schema, primitives[0], *src_col, pids, varlen, scatter_state);
    alloc.release(h);

    // Reconstruct partition 0 in 50-row pumps
    if (views[0].row_end > views[0].row_begin)
    {
        const size_t total = views[0].row_end - views[0].row_begin;
        auto result = ColumnVector<UInt32>::create();
        result->reserve(total);
        ResumePosition pos{};
        size_t pumped = 0;
        while (pumped < total)
        {
            const size_t remaining = total - pumped;
            const size_t cap = std::min(remaining, size_t{50});
            result->reserve(pumped + cap);
            pos = primitives[0].reconstruct(primitives[0], schema, views.data(), 1, pos, *result);
            pumped = result->size();
        }
        EXPECT_EQ(result->size(), total);
    }
}

// ─────────────────────────── RadixShuffler tests ───────────────────

// These tests exercise the OutBlock / BumpArena based scatter operator.
// Coverage goals:
//   • all code paths in RadixShuffler.cpp
//   • direct scatter (small num_partitions) and SWWC scatter (large num_partitions) for every column type
//   • Nullable decomposition into [null_map, values] physical columns
//   • drain-before-grow in Phase 3 (block overflow during SWWC)
//   • finish() both with and without staged residuals
//   • multi-block partitions, multi-batch input, empty input, P=1


// Helpers ──────────────────────────────────────────────────────────────────


/// Walk the OutBlock chain for partition p and collect raw T values from
/// physical column `col_idx`.
template <typename T>
std::vector<T> collectScalar(const std::vector<PartState> & parts, size_t col_idx = 0)
{
    std::vector<T> out;
    for (const auto & ps : parts)
        for (const OutBlock * b = ps.head; b; b = b->next)
        {
            const T * data = static_cast<const T *>(b->cols[col_idx]);
            out.insert(out.end(), data, data + b->filled);
        }
    return out;
}

/// Collect FixedString values (runtime-sized rows) from physical column `col_idx`.
std::vector<std::string> collectFixedString(const std::vector<PartState> & parts, size_t col_idx, size_t n)
{
    std::vector<std::string> out;
    for (const auto & ps : parts)
        for (const OutBlock * b = ps.head; b; b = b->next)
        {
            const char * data = static_cast<const char *>(b->cols[col_idx]);
            for (size_t r = 0; r < b->filled; ++r)
                out.emplace_back(data + r * n, n);
        }
    return out;
}

/// Collect Nullable<T> values.  Physical layout: cols[null_col] = uint8_t,
/// cols[val_col] = T.  Returns (is_null, value) pairs.
template <typename T>
struct NV
{
    bool is_null;
    T value;
    bool operator<(const NV & o) const
    {
        if (is_null != o.is_null)
            return is_null < o.is_null;
        return value < o.value;
    }
    bool operator==(const NV & o) const { return is_null == o.is_null && (is_null || value == o.value); }
};

template <typename T>
std::vector<NV<T>> collectNullable(const std::vector<PartState> & parts, size_t null_col, size_t val_col)
{
    std::vector<NV<T>> out;
    for (const auto & ps : parts)
        for (const OutBlock * b = ps.head; b; b = b->next)
        {
            const uint8_t * nulls = static_cast<const uint8_t *>(b->cols[null_col]);
            const T * vals = static_cast<const T *>(b->cols[val_col]);
            for (size_t r = 0; r < b->filled; ++r)
                out.push_back({nulls[r] != 0, vals[r]});
        }
    return out;
}

/// Total rows across all partitions.
size_t totalRows(const std::vector<PartState> & parts)
{
    size_t n = 0;
    for (const auto & ps : parts)
        for (const OutBlock * b = ps.head; b; b = b->next)
            n += b->filled;
    return n;
}


// ── Column factories for the operator tests ─────────────────────────────────

template <typename T>
auto makeVec(size_t n, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    auto col = DB::ColumnVector<T>::create();
    col->reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            std::uniform_real_distribution<T> dist(-1e6, 1e6);
            col->insertValue(dist(rng));
        }
        else
        {
            col->insertValue(static_cast<T>(rng()));
        }
    }
    return col;
}

/// Decimal factory: maps NativeType values.
template <typename DecT>
auto makeDecCol(size_t n, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    auto col = DB::ColumnDecimal<DecT>::create(0, 0);
    for (size_t i = 0; i < n; ++i)
        col->insertValue(DecT(static_cast<DecT::NativeType>(rng())));
    return col;
}

/// Nullable(T) factory: ~20% nulls.
template <typename T>
DB::MutableColumnPtr makeNullableVec(size_t n, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    auto nested = DB::ColumnVector<T>::create();
    auto nulls = DB::ColumnUInt8::create();
    nested->reserve(n);
    nulls->reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        bool is_null = (rng() % 5 == 0);
        nulls->insertValue(is_null ? 1 : 0);
        nested->insertValue(static_cast<T>(rng()));
    }
    return DB::ColumnNullable::create(std::move(nested), std::move(nulls));
}

/// FixedString factory: fills with random bytes.
DB::MutableColumnPtr makeFixedStrCol(size_t n, size_t width, uint64_t seed = 1)
{
    std::mt19937_64 rng(seed);
    auto col = DB::ColumnFixedString::create(width);
    std::string buf(width, '\0');
    for (size_t i = 0; i < n; ++i)
    {
        for (char & c : buf)
            c = static_cast<char>(rng() & 0xff);
        col->insertData(buf.data(), width);
    }
    return col;
}


// ── Core round-trip helper ───────────────────────────────────────────────────

/// Run the operator on `N` rows across `num_blocks` calls to `process()`, then
/// collect all scattered rows and verify they form a multiset-equal copy of
/// the input.  Returns the total rows collected.
template <typename T>
size_t
runNumericRoundTrip(size_t num_rows, int num_partitions, size_t blocks = 1, size_t init_cap = kOutCapMin, size_t max_cap = kOutCapMax)
{
    auto col = makeVec<T>(num_rows, 17);
    const auto & src = assert_cast<const DB::ColumnVector<T> &>(*col);

    std::vector<ColumnPrimitives> prims(1, makeFixedWidth<T>());
    BumpArena arena(64ULL << 20);
    bool use_swwc = RadixShuffler::shouldUseSwwc(1, num_partitions);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, use_swwc, init_cap, max_cap);

    const size_t rows_per_block = (num_rows + blocks - 1) / blocks;
    for (size_t b = 0; b < blocks; ++b)
    {
        const size_t start = b * rows_per_block;
        const size_t end = std::min(start + rows_per_block, num_rows);
        if (start >= end)
            break;
        DB::Columns blk = {col->cut(start, end - start)};
        op.process(blk);
    }
    op.finish();

    std::vector<T> collected = collectScalar<T>(op.parts(), 0);
    EXPECT_EQ(collected.size(), num_rows);

    std::vector<T> original(src.getData().begin(), src.getData().end());
    std::sort(original.begin(), original.end());
    std::sort(collected.begin(), collected.end());
    EXPECT_EQ(original, collected) << "type=" << typeid(T).name() << " P=" << num_partitions << " N=" << num_rows;

    return totalRows(op.parts());
}


// ─────────────────────────── Direct scatter tests ───────────────────────────
// shouldUseSwwc(1, num_partitions) == false  ↔  P < 512
// shouldUseSwwc(K, num_partitions) == false  ↔  K>=2 && P < 32

TEST(RadixShuffler, ShouldUseSwwcCrossover)
{
    // K=1: threshold at P=512
    EXPECT_FALSE(RadixShuffler::shouldUseSwwc(1, 1));
    EXPECT_FALSE(RadixShuffler::shouldUseSwwc(1, 256));
    EXPECT_FALSE(RadixShuffler::shouldUseSwwc(1, 511));
    EXPECT_TRUE(RadixShuffler::shouldUseSwwc(1, 512));
    EXPECT_TRUE(RadixShuffler::shouldUseSwwc(1, 1024));
    // K>=2: threshold at P=32
    EXPECT_FALSE(RadixShuffler::shouldUseSwwc(2, 16));
    EXPECT_FALSE(RadixShuffler::shouldUseSwwc(4, 31));
    EXPECT_TRUE(RadixShuffler::shouldUseSwwc(2, 32));
    EXPECT_TRUE(RadixShuffler::shouldUseSwwc(4, 64));
}

TEST(RadixShuffler, DirectUInt8P4)
{
    runNumericRoundTrip<UInt8>(2048, 4);
}
TEST(RadixShuffler, DirectUInt16P16)
{
    runNumericRoundTrip<UInt16>(2048, 16);
}
TEST(RadixShuffler, DirectUInt32P64)
{
    runNumericRoundTrip<UInt32>(4096, 64);
}
TEST(RadixShuffler, DirectUInt64P256)
{
    runNumericRoundTrip<UInt64>(4096, 256);
}
TEST(RadixShuffler, DirectUInt128P4)
{
    runNumericRoundTrip<UInt128>(1024, 4);
}
TEST(RadixShuffler, DirectUInt256P4)
{
    runNumericRoundTrip<UInt256>(512, 4);
}
TEST(RadixShuffler, DirectInt8P8)
{
    runNumericRoundTrip<Int8>(2048, 8);
}
TEST(RadixShuffler, DirectInt16P8)
{
    runNumericRoundTrip<Int16>(2048, 8);
}
TEST(RadixShuffler, DirectInt32P64)
{
    runNumericRoundTrip<Int32>(4096, 64);
}
TEST(RadixShuffler, DirectInt64P128)
{
    runNumericRoundTrip<Int64>(4096, 128);
}
TEST(RadixShuffler, DirectFloat32P16)
{
    runNumericRoundTrip<Float32>(2048, 16);
}
TEST(RadixShuffler, DirectFloat64P128)
{
    runNumericRoundTrip<Float64>(4096, 128);
}

TEST(RadixShuffler, DirectDecimal32P4)
{
    constexpr size_t num_rows = 1024;
    constexpr int num_partitions = 4;
    auto col = makeDecCol<Decimal32>(num_rows, 7);
    std::vector<ColumnPrimitives> prims(1, makeDecimal<Decimal32>());
    BumpArena arena(16ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, false);
    DB::Columns blk{col->getPtr()};
    op.process(blk);
    op.finish();

    using NT = Decimal32::NativeType;
    std::vector<NT> orig;

    std::vector<NT> coll;
    for (size_t i = 0; i < num_rows; ++i)
        orig.push_back(col->getData()[i].value);
    coll = collectScalar<NT>(op.parts(), 0);

    ASSERT_EQ(coll.size(), num_rows);
    std::sort(orig.begin(), orig.end());
    std::sort(coll.begin(), coll.end());
    EXPECT_EQ(orig, coll);
}

TEST(RadixShuffler, DirectDecimal64P64)
{
    constexpr size_t num_rows = 2048;
    constexpr int num_partitions = 64;
    auto col = makeDecCol<Decimal64>(num_rows, 8);
    std::vector<ColumnPrimitives> prims(1, makeDecimal<Decimal64>());
    BumpArena arena(16ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, false);
    DB::Columns blk{col->getPtr()};
    op.process(blk);
    op.finish();

    using NT = Decimal64::NativeType;
    std::vector<NT> orig;

    std::vector<NT> coll;
    for (size_t i = 0; i < num_rows; ++i)
        orig.push_back(col->getData()[i].value);
    coll = collectScalar<NT>(op.parts(), 0);

    ASSERT_EQ(coll.size(), num_rows);
    std::sort(orig.begin(), orig.end());
    std::sort(coll.begin(), coll.end());
    EXPECT_EQ(orig, coll);
}

TEST(RadixShuffler, DirectFixedString8P16)
{
    constexpr size_t num_rows = 1024;
    constexpr int num_partitions = 16;
    constexpr size_t width = 8;
    auto col = makeFixedStrCol(num_rows, width, 5);
    std::vector<ColumnPrimitives> prims(1, makeFixedString(width));
    BumpArena arena(16ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, false);
    DB::Columns blk{col->getPtr()};
    op.process(blk);
    op.finish();

    std::vector<std::string> orig;


    std::vector<std::string> coll;
    const auto & fscol = assert_cast<const DB::ColumnFixedString &>(*col);
    for (size_t i = 0; i < num_rows; ++i)
        orig.emplace_back(reinterpret_cast<const char *>(fscol.getChars().data() + i * width), width);
    coll = collectFixedString(op.parts(), 0, width);

    ASSERT_EQ(coll.size(), num_rows);
    std::sort(orig.begin(), orig.end());
    std::sort(coll.begin(), coll.end());
    EXPECT_EQ(orig, coll);
}

TEST(RadixShuffler, DirectNullableUInt32P8)
{
    constexpr size_t num_rows = 1024;
    constexpr int num_partitions = 8;
    auto col = makeNullableVec<UInt32>(num_rows, 3);
    const auto & nc = assert_cast<const DB::ColumnNullable &>(*col);
    std::vector<ColumnPrimitives> prims(1, makeNullable(makeFixedWidth<UInt32>()));
    BumpArena arena(16ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, false);
    DB::Columns blk{col->getPtr()};
    op.process(blk);
    op.finish();

    std::vector<NV<UInt32>> orig;


    std::vector<NV<UInt32>> coll;
    const auto & nulls = nc.getNullMapData();
    const auto & vals = assert_cast<const DB::ColumnVector<UInt32> &>(nc.getNestedColumn()).getData();
    for (size_t i = 0; i < num_rows; ++i)
        orig.push_back({nulls[i] != 0, vals[i]});
    coll = collectNullable<UInt32>(op.parts(), 0, 1);

    ASSERT_EQ(coll.size(), num_rows);
    std::sort(orig.begin(), orig.end());
    std::sort(coll.begin(), coll.end());
    EXPECT_EQ(orig, coll);
}

// K>=2 direct scatter (K=2, P=16 < 32)
TEST(RadixShuffler, DirectTwoColumnP16)
{
    constexpr size_t num_rows = 1024;
    constexpr int num_partitions = 16;
    auto col0 = makeVec<UInt64>(num_rows, 1);
    auto col1 = makeVec<UInt32>(num_rows, 2);

    std::vector<ColumnPrimitives> prims = {makeFixedWidth<UInt64>(), makeFixedWidth<UInt32>()};
    BumpArena arena(16ULL << 20);
    RadixShuffler op(num_partitions, 2, std::move(prims), arena, false);
    DB::Columns blk{col0->getPtr(), col1->getPtr()};
    op.process(blk);
    op.finish();

    // Verify column 0 (UInt64)
    std::vector<UInt64> orig0(col0->getData().begin(), col0->getData().end());
    std::vector<UInt64> coll0 = collectScalar<UInt64>(op.parts(), 0);
    ASSERT_EQ(coll0.size(), num_rows);
    std::sort(orig0.begin(), orig0.end());
    std::sort(coll0.begin(), coll0.end());
    EXPECT_EQ(orig0, coll0);

    // Verify column 1 (UInt32)
    std::vector<UInt32> orig1(col1->getData().begin(), col1->getData().end());
    std::vector<UInt32> coll1 = collectScalar<UInt32>(op.parts(), 1);
    ASSERT_EQ(coll1.size(), num_rows);
    std::sort(orig1.begin(), orig1.end());
    std::sort(coll1.begin(), coll1.end());
    EXPECT_EQ(orig1, coll1);
}

// K=4, P=16 < 32 → direct (exercises kMaxK boundary)
TEST(RadixShuffler, DirectFourColumnP16)
{
    constexpr size_t num_rows = 512;
    constexpr int num_partitions = 16;
    auto c0 = makeVec<UInt64>(num_rows, 10);
    auto c1 = makeVec<UInt32>(num_rows, 11);
    auto c2 = makeVec<UInt16>(num_rows, 12);
    auto c3 = makeVec<UInt8>(num_rows, 13);

    std::vector<ColumnPrimitives> prims
        = {makeFixedWidth<UInt64>(), makeFixedWidth<UInt32>(), makeFixedWidth<UInt16>(), makeFixedWidth<UInt8>()};
    BumpArena arena(16ULL << 20);
    RadixShuffler op(num_partitions, 4, std::move(prims), arena, false);
    DB::Columns blk{c0->getPtr(), c1->getPtr(), c2->getPtr(), c3->getPtr()};
    op.process(blk);
    op.finish();

    EXPECT_EQ(totalRows(op.parts()), num_rows);

    // Spot-check col 0 and col 3
    std::vector<UInt64> orig0(c0->getData().begin(), c0->getData().end());
    auto coll0 = collectScalar<UInt64>(op.parts(), 0);
    std::sort(orig0.begin(), orig0.end());
    std::sort(coll0.begin(), coll0.end());
    EXPECT_EQ(orig0, coll0);

    std::vector<UInt8> orig3(c3->getData().begin(), c3->getData().end());
    auto coll3 = collectScalar<UInt8>(op.parts(), 3);
    std::sort(orig3.begin(), orig3.end());
    std::sort(coll3.begin(), coll3.end());
    EXPECT_EQ(orig3, coll3);
}


// ─────────────────────────── SWWC scatter tests ─────────────────────────────
// K=1: P >= 512    K>=2: P >= 32

TEST(RadixShuffler, SWWCUInt64P512)
{
    runNumericRoundTrip<UInt64>(8192, 512);
}
TEST(RadixShuffler, SWWCUInt64P1024)
{
    runNumericRoundTrip<UInt64>(8192, 1024);
}
TEST(RadixShuffler, SWWCUInt32P512)
{
    runNumericRoundTrip<UInt32>(8192, 512);
}
TEST(RadixShuffler, SWWCInt64P1024)
{
    runNumericRoundTrip<Int64>(4096, 1024);
}
TEST(RadixShuffler, SWWCFloat64P512)
{
    runNumericRoundTrip<Float64>(4096, 512);
}
TEST(RadixShuffler, SWWCUInt128P512)
{
    runNumericRoundTrip<UInt128>(2048, 512);
}
TEST(RadixShuffler, SWWCUInt256P512)
{
    runNumericRoundTrip<UInt256>(1024, 512);
}

TEST(RadixShuffler, SWWCDecimal64P512)
{
    constexpr size_t num_rows = 4096;
    constexpr int num_partitions = 512;
    auto col = makeDecCol<Decimal64>(num_rows, 9);
    std::vector<ColumnPrimitives> prims(1, makeDecimal<Decimal64>());
    BumpArena arena(32ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, true);
    DB::Columns blk{col->getPtr()};
    op.process(blk);
    op.finish();

    using NT = Decimal64::NativeType;
    std::vector<NT> orig;

    std::vector<NT> coll;
    for (size_t i = 0; i < num_rows; ++i)
        orig.push_back(col->getData()[i].value);
    coll = collectScalar<NT>(op.parts(), 0);
    ASSERT_EQ(coll.size(), num_rows);
    std::sort(orig.begin(), orig.end());
    std::sort(coll.begin(), coll.end());
    EXPECT_EQ(orig, coll);
}

// FixedString in SWWC mode: scatter_raw_swwc=nullptr → falls back to scatter_raw
// This covers the `else` branch in Phase 4b and also `drain_raw=nullptr` in finish().
TEST(RadixShuffler, SWWCFixedString8P512)
{
    constexpr size_t num_rows = 4096;
    constexpr int num_partitions = 512;
    constexpr size_t width = 8;
    auto col = makeFixedStrCol(num_rows, width, 6);
    std::vector<ColumnPrimitives> prims(1, makeFixedString(width));
    BumpArena arena(32ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, true);
    DB::Columns blk{col->getPtr()};
    op.process(blk);
    op.finish();

    std::vector<std::string> orig;


    std::vector<std::string> coll;
    const auto & fscol = assert_cast<const DB::ColumnFixedString &>(*col);
    for (size_t i = 0; i < num_rows; ++i)
        orig.emplace_back(reinterpret_cast<const char *>(fscol.getChars().data() + i * width), width);
    coll = collectFixedString(op.parts(), 0, width);

    ASSERT_EQ(coll.size(), num_rows);
    std::sort(orig.begin(), orig.end());
    std::sort(coll.begin(), coll.end());
    EXPECT_EQ(orig, coll);
}

// Nullable(UInt64) in SWWC mode: null_map gets kSlotsPerFlush=64, values get 8.
TEST(RadixShuffler, SWWCNullableUInt64P512)
{
    constexpr size_t num_rows = 4096;
    constexpr int num_partitions = 512;
    auto col = makeNullableVec<UInt64>(num_rows, 4);
    const auto & nc = assert_cast<const DB::ColumnNullable &>(*col);
    std::vector<ColumnPrimitives> prims(1, makeNullable(makeFixedWidth<UInt64>()));
    BumpArena arena(32ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, true);
    DB::Columns blk{col->getPtr()};
    op.process(blk);
    op.finish();

    const auto & nulls = nc.getNullMapData();
    const auto & vals = assert_cast<const DB::ColumnVector<UInt64> &>(nc.getNestedColumn()).getData();
    std::vector<NV<UInt64>> orig;

    std::vector<NV<UInt64>> coll;
    for (size_t i = 0; i < num_rows; ++i)
        orig.push_back({nulls[i] != 0, vals[i]});
    // Physical layout: cols[0]=null_map (uint8_t), cols[1]=uint64_t values
    coll = collectNullable<UInt64>(op.parts(), 0, 1);

    ASSERT_EQ(coll.size(), num_rows);
    std::sort(orig.begin(), orig.end());
    std::sort(coll.begin(), coll.end());
    EXPECT_EQ(orig, coll);
}

// K=2, P=32 → SWWC
TEST(RadixShuffler, SWWCTwoColumnP32)
{
    constexpr size_t num_rows = 2048;
    constexpr int num_partitions = 32;
    auto c0 = makeVec<UInt64>(num_rows, 20);
    auto c1 = makeVec<UInt64>(num_rows, 21);
    std::vector<ColumnPrimitives> prims = {makeFixedWidth<UInt64>(), makeFixedWidth<UInt64>()};
    BumpArena arena(32ULL << 20);
    RadixShuffler op(num_partitions, 2, std::move(prims), arena, true);
    DB::Columns blk{c0->getPtr(), c1->getPtr()};
    op.process(blk);
    op.finish();

    for (int ci : {0, 1})
    {
        const auto & src_data = (ci == 0 ? c0->getData() : c1->getData());
        std::vector<UInt64> orig(src_data.begin(), src_data.end());
        auto coll = collectScalar<UInt64>(op.parts(), static_cast<size_t>(ci));
        ASSERT_EQ(coll.size(), num_rows);
        std::sort(orig.begin(), orig.end());
        std::sort(coll.begin(), coll.end());
        EXPECT_EQ(orig, coll) << "col=" << ci;
    }
}

// K=4, P=64 → SWWC (exercises kMaxK=8 limit since no Nullable expansion here)
TEST(RadixShuffler, SWWCFourColumnP64)
{
    constexpr size_t num_rows = 4096;
    constexpr int num_partitions = 64;
    auto c0 = makeVec<UInt64>(num_rows, 30);
    auto c1 = makeVec<UInt32>(num_rows, 31);
    auto c2 = makeVec<UInt64>(num_rows, 32);
    auto c3 = makeVec<UInt32>(num_rows, 33);
    std::vector<ColumnPrimitives> prims
        = {makeFixedWidth<UInt64>(), makeFixedWidth<UInt32>(), makeFixedWidth<UInt64>(), makeFixedWidth<UInt32>()};
    BumpArena arena(64ULL << 20);
    RadixShuffler op(num_partitions, 4, std::move(prims), arena, true);
    DB::Columns blk{c0->getPtr(), c1->getPtr(), c2->getPtr(), c3->getPtr()};
    op.process(blk);
    op.finish();

    EXPECT_EQ(totalRows(op.parts()), num_rows);

    std::vector<UInt64> orig0(c0->getData().begin(), c0->getData().end());
    auto coll0 = collectScalar<UInt64>(op.parts(), 0);
    std::sort(orig0.begin(), orig0.end());
    std::sort(coll0.begin(), coll0.end());
    EXPECT_EQ(orig0, coll0);
}


// ─────────────────────────── Edge-case tests ────────────────────────────────

TEST(RadixShuffler, EmptyInput)
{
    // process() early return when N=0
    constexpr int num_partitions = 64;
    std::vector<ColumnPrimitives> prims(1, makeFixedWidth<UInt64>());
    BumpArena arena(1ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, false);

    auto col = DB::ColumnVector<UInt64>::create(); // empty column
    DB::Columns blk{col->getPtr()};
    op.process(blk); // must not crash
    op.finish();

    EXPECT_EQ(totalRows(op.parts()), 0u);
}

TEST(RadixShuffler, EmptyColumns)
{
    // process() early return when columns is empty
    constexpr int num_partitions = 4;
    std::vector<ColumnPrimitives> prims(1, makeFixedWidth<UInt64>());
    BumpArena arena(1ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, false);

    DB::Columns blk{}; // empty columns vector
    op.process(blk); // must not crash
    op.finish();

    EXPECT_EQ(totalRows(op.parts()), 0u);
}

TEST(RadixShuffler, SinglePartitionP1)
{
    // num_partitions=1: every row goes to partition 0.  mask_ = 0, so pids[j] = hash & 0 = 0.
    constexpr size_t num_rows = 512;
    constexpr int num_partitions = 1;
    auto col = makeVec<UInt32>(num_rows, 5);
    std::vector<ColumnPrimitives> prims(1, makeFixedWidth<UInt32>());
    BumpArena arena(8ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, false);
    DB::Columns blk{col->getPtr()};
    op.process(blk);
    op.finish();

    EXPECT_EQ(totalRows(op.parts()), num_rows);
    // All rows in partition 0
    size_t p0_rows = 0;
    for (const OutBlock * b = op.parts()[0].head; b; b = b->next)
        p0_rows += b->filled;
    EXPECT_EQ(p0_rows, num_rows);
}

TEST(RadixShuffler, MultiBatchDirectScatter)
{
    // num_rows > batch_size → multiple runBatch() calls.
    // batch = max(1024, min(32768, P*16)).  With P=4, batch=1024.
    // Use N=3000 to get 3 batches.
    constexpr size_t num_rows = 3000;
    constexpr int num_partitions = 4;
    runNumericRoundTrip<UInt64>(num_rows, num_partitions, 3 /*blocks*/);
}

TEST(RadixShuffler, MultiBatchSWWC)
{
    // num_partitions=512 → SWWC.  batch = max(1024, min(32768, 512*16)) = 8192.
    // Use N=20000 to get 3 batches.
    constexpr size_t num_rows = 20000;
    constexpr int num_partitions = 512;
    runNumericRoundTrip<UInt64>(num_rows, num_partitions, 3 /*blocks*/);
}

TEST(RadixShuffler, BlockOverflowDirectScatter)
{
    // Small init_cap forces multiple OutBlocks per partition (linked chain).
    // init_cap must be >= hist[p] for any single batch (≈ N/(P*batches_approx)).
    // With P=4, N=8192, 4 process() calls (blocks=4), each of 2048 rows:
    //   hist[p] ≈ 2048/4 = 512.  Use init_cap=512, max_cap=512 so every
    //   process() call triggers a growth → 4 OutBlocks per partition.
    constexpr size_t num_rows = 8192;
    constexpr int num_partitions = 4;
    runNumericRoundTrip<UInt64>(num_rows, num_partitions, 4 /*blocks*/, 512 /*init_cap*/, 512 /*max_cap*/);
}

TEST(RadixShuffler, BlockOverflowSWWCDrainBeforeGrow)
{
    // SWWC + small init_cap: forces a grow while cnt_[p] > 0, exercising the
    // "drain before grow" path in Phase 3 (drain_raw is called with non-zero cnt).
    // num_partitions=512, N=65536, 2 process() calls of 32768 rows each.
    // hist[p] ≈ 32768/512 = 64.  Use init_cap=64 so every batch triggers a grow.
    // With cnt_[p] accumulating (SWWC), drain_raw is called before the grow.
    constexpr size_t num_rows = 65536;
    constexpr int num_partitions = 512;
    runNumericRoundTrip<UInt64>(num_rows, num_partitions, 2 /*blocks*/, 64 /*init_cap*/, 128 /*max_cap*/);
}

TEST(RadixShuffler, FinishWithSWWCZeroCnt)
{
    // finish() with all cnt_[p]=0 (nothing staged) — the fast path.
    // Achieved by choosing N that's a perfect multiple of kSlotsPerFlush<UInt64>=8
    // so all staging buffers flush completely during scatter.
    constexpr size_t num_rows = 8192; // 8192 rows, P=512 partitions, ~16 rows/part on average
    constexpr int num_partitions = 512;
    runNumericRoundTrip<UInt64>(num_rows, num_partitions);
}

TEST(RadixShuffler, FinishWithSWWCNonZeroCnt)
{
    // finish() with some cnt_[p]!=0 — exercises the drain loop.
    // num_rows=8193 is NOT a multiple of 8 so at least some partitions will have residual.
    constexpr size_t num_rows = 8193;
    constexpr int num_partitions = 512;
    runNumericRoundTrip<UInt64>(num_rows, num_partitions);
}

TEST(RadixShuffler, FinishNoopDirectMode)
{
    // finish() with use_swwc_=false → returns immediately.
    constexpr size_t num_rows = 512;
    constexpr int num_partitions = 4;
    auto col = makeVec<UInt32>(num_rows, 99);
    std::vector<ColumnPrimitives> prims(1, makeFixedWidth<UInt32>());
    BumpArena arena(8ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, /*use_swwc=*/false);
    DB::Columns blk{col->getPtr()};
    op.process(blk);
    op.finish(); // no-op
    EXPECT_EQ(totalRows(op.parts()), num_rows);
}

TEST(RadixShuffler, MultiProcessSameOperator)
{
    // Call process() multiple times on the same operator instance with different blocks.
    constexpr size_t rows_per_batch = 512;
    constexpr size_t num_blocks = 5;
    constexpr int num_partitions = 8;
    auto col = makeVec<UInt64>(rows_per_batch * num_blocks, 77);
    std::vector<ColumnPrimitives> prims(1, makeFixedWidth<UInt64>());
    BumpArena arena(64ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, false);

    for (size_t b = 0; b < num_blocks; ++b)
    {
        DB::Columns blk{col->cut(b * rows_per_batch, rows_per_batch)};
        op.process(blk);
    }
    op.finish();

    const size_t total = totalRows(op.parts());
    EXPECT_EQ(total, rows_per_batch * num_blocks);

    std::vector<UInt64> orig(col->getData().begin(), col->getData().end());
    auto coll = collectScalar<UInt64>(op.parts(), 0);
    std::sort(orig.begin(), orig.end());
    std::sort(coll.begin(), coll.end());
    EXPECT_EQ(orig, coll);
}


// ─────────────────────────── Multi-thread tests ─────────────────────────────
// Each thread owns its own BumpArena + RadixShuffler.  Threads run
// concurrently but are fully independent — tests correct thread isolation.

TEST(RadixShuffler, MultiThreadDirectP16)
{
    constexpr size_t rows_per_thread = 2048;
    constexpr int num_partitions = 16;
    constexpr size_t num_threads = 4;

    std::vector<std::thread> threads;
    std::vector<size_t> results(num_threads, 0);

    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                auto col = makeVec<UInt32>(rows_per_thread, t + 100);
                std::vector<ColumnPrimitives> prims(1, makeFixedWidth<UInt32>());
                BumpArena arena(16ULL << 20);
                RadixShuffler op(num_partitions, 1, std::move(prims), arena, false);
                DB::Columns blk{col->getPtr()};
                op.process(blk);
                op.finish();

                std::vector<UInt32> orig(col->getData().begin(), col->getData().end());
                auto coll = collectScalar<UInt32>(op.parts(), 0);
                std::sort(orig.begin(), orig.end());
                std::sort(coll.begin(), coll.end());
                if (orig == coll)
                    results[t] = coll.size();
            });
    }
    for (auto & th : threads)
        th.join();

    for (size_t t = 0; t < num_threads; ++t)
        EXPECT_EQ(results[t], rows_per_thread) << "thread " << t;
}

TEST(RadixShuffler, MultiThreadSWWCP512)
{
    constexpr size_t rows_per_thread = 4096;
    constexpr int num_partitions = 512;
    constexpr size_t num_threads = 4;

    std::vector<std::thread> threads;
    std::vector<bool> ok(num_threads, false);

    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                auto col = makeVec<UInt64>(rows_per_thread, t + 200);
                std::vector<ColumnPrimitives> prims(1, makeFixedWidth<UInt64>());
                BumpArena arena(32ULL << 20);
                RadixShuffler op(num_partitions, 1, std::move(prims), arena, true);
                DB::Columns blk{col->getPtr()};
                op.process(blk);
                op.finish();

                std::vector<UInt64> orig(col->getData().begin(), col->getData().end());
                auto coll = collectScalar<UInt64>(op.parts(), 0);
                std::sort(orig.begin(), orig.end());
                std::sort(coll.begin(), coll.end());
                ok[t] = (orig == coll && coll.size() == rows_per_thread);
            });
    }
    for (auto & th : threads)
        th.join();

    for (size_t t = 0; t < num_threads; ++t)
        EXPECT_TRUE(ok[t]) << "thread " << t;
}

TEST(RadixShuffler, MultiThreadEightThreadsMixed)
{
    constexpr size_t num_rows = 2048;
    constexpr size_t num_threads = 8;

    std::vector<std::thread> threads;
    std::atomic<size_t> passed{0};

    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                // Alternate between direct (P=16) and SWWC (P=512)
                const int num_partitions = (t % 2 == 0) ? 16 : 512;
                auto col = makeVec<UInt64>(num_rows, t + 300);
                std::vector<ColumnPrimitives> prims(1, makeFixedWidth<UInt64>());
                BumpArena arena(32ULL << 20);
                bool use_swwc = RadixShuffler::shouldUseSwwc(1, num_partitions);
                RadixShuffler op(num_partitions, 1, std::move(prims), arena, use_swwc);
                DB::Columns blk{col->getPtr()};
                op.process(blk);
                op.finish();

                std::vector<UInt64> orig(col->getData().begin(), col->getData().end());
                auto coll = collectScalar<UInt64>(op.parts(), 0);
                if (coll.size() == num_rows)
                {
                    std::sort(orig.begin(), orig.end());
                    std::sort(coll.begin(), coll.end());
                    if (orig == coll)
                        ++passed;
                }
            });
    }
    for (auto & th : threads)
        th.join();

    EXPECT_EQ(passed.load(), num_threads);
}


// ─────────────────────────── Nullable decomposition tests ───────────────────

TEST(RadixShuffler, NullableExpansionKphys)
{
    // With a Nullable column, K_phys = 2K (each Nullable → 2 physical columns).
    // Verify the K=2 Nullable(UInt32) case: K_phys=4, cols[0]=null0, cols[1]=vals0,
    // cols[2]=null1, cols[3]=vals1.
    constexpr size_t num_rows = 512;
    constexpr int num_partitions = 8;
    auto c0 = makeNullableVec<UInt32>(num_rows, 1);
    auto c1 = makeNullableVec<UInt32>(num_rows, 2);
    const auto & nc0 = assert_cast<const DB::ColumnNullable &>(*c0);
    const auto & nc1 = assert_cast<const DB::ColumnNullable &>(*c1);

    std::vector<ColumnPrimitives> prims = {makeNullable(makeFixedWidth<UInt32>()), makeNullable(makeFixedWidth<UInt32>())};
    BumpArena arena(16ULL << 20);
    RadixShuffler op(num_partitions, 2, std::move(prims), arena, false);
    DB::Columns blk{c0->getPtr(), c1->getPtr()};
    op.process(blk);
    op.finish();

    EXPECT_EQ(totalRows(op.parts()), num_rows);

    // col 0: physical cols 0 (null_map) and 1 (values)
    auto orig0 = [&]()
    {
        std::vector<NV<UInt32>> v;
        const auto & nulls = nc0.getNullMapData();
        const auto & vals = assert_cast<const DB::ColumnVector<UInt32> &>(nc0.getNestedColumn()).getData();
        for (size_t i = 0; i < num_rows; ++i)
            v.push_back({nulls[i] != 0, vals[i]});
        return v;
    }();
    auto coll0 = collectNullable<UInt32>(op.parts(), 0, 1);
    std::sort(orig0.begin(), orig0.end());
    std::sort(coll0.begin(), coll0.end());
    EXPECT_EQ(orig0, coll0) << "col 0";

    // col 1: physical cols 2 (null_map) and 3 (values)
    auto orig1 = [&]()
    {
        std::vector<NV<UInt32>> v;
        const auto & nulls = nc1.getNullMapData();
        const auto & vals = assert_cast<const DB::ColumnVector<UInt32> &>(nc1.getNestedColumn()).getData();
        for (size_t i = 0; i < num_rows; ++i)
            v.push_back({nulls[i] != 0, vals[i]});
        return v;
    }();
    auto coll1 = collectNullable<UInt32>(op.parts(), 2, 3);
    std::sort(orig1.begin(), orig1.end());
    std::sort(coll1.begin(), coll1.end());
    EXPECT_EQ(orig1, coll1) << "col 1";
}

TEST(RadixShuffler, NullableFixedStringP8)
{
    // Nullable(FixedString(4)): null_map + fixedstring.
    constexpr size_t num_rows = 512;
    constexpr int num_partitions = 8;
    constexpr size_t width = 4;

    std::mt19937_64 rng(42); // NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    auto nested = DB::ColumnFixedString::create(width);
    auto null_map = DB::ColumnUInt8::create();
    std::string buf(width, '\0');
    for (size_t i = 0; i < num_rows; ++i)
    {
        bool is_null = (rng() % 4 == 0);
        null_map->insertValue(is_null ? 1 : 0);
        for (char & c : buf)
            c = static_cast<char>(rng() & 0xff);
        nested->insertData(buf.data(), width);
    }
    auto col = DB::ColumnNullable::create(std::move(nested), std::move(null_map));
    const auto & nc = assert_cast<const DB::ColumnNullable &>(*col);

    std::vector<ColumnPrimitives> prims(1, makeNullable(makeFixedString(width)));
    BumpArena arena(16ULL << 20);
    RadixShuffler op(num_partitions, 1, std::move(prims), arena, false);
    DB::Columns blk{col->getPtr()};
    op.process(blk);
    op.finish();

    EXPECT_EQ(totalRows(op.parts()), num_rows);

    // Physical cols: 0=null_map (uint8_t), 1=fixedstring (width bytes/row)
    struct NFS
    {
        bool is_null;
        std::string value;
        bool operator<(const NFS & o) const
        {
            if (is_null != o.is_null)
            {
                return is_null < o.is_null;
            }
            return value < o.value;
        }
        bool operator==(const NFS & o) const { return is_null == o.is_null && (is_null || value == o.value); }
    };

    std::vector<NFS> orig;


    std::vector<NFS> coll;
    const auto & null_data = nc.getNullMapData();
    const auto & fs_data = assert_cast<const DB::ColumnFixedString &>(nc.getNestedColumn()).getChars();
    for (size_t i = 0; i < num_rows; ++i)
        orig.push_back({null_data[i] != 0, std::string(reinterpret_cast<const char *>(fs_data.data() + i * width), width)});

    for (const auto & ps : op.parts())
        for (const OutBlock * b = ps.head; b; b = b->next)
        {
            const uint8_t * nulls = static_cast<const uint8_t *>(b->cols[0]);
            const char * chars = static_cast<const char *>(b->cols[1]);
            for (size_t r = 0; r < b->filled; ++r)
                coll.push_back({nulls[r] != 0, std::string(chars + r * width, width)});
        }

    ASSERT_EQ(coll.size(), num_rows);
    std::sort(orig.begin(), orig.end());
    std::sort(coll.begin(), coll.end());
    EXPECT_EQ(orig, coll);
}


// ─────────────────────────── BatchSize tests ────────────────────────────────

TEST(RadixShuffler, BatchSizeFormula)
{
    // batch = max(1024, min(32768, num_partitions * 16))
    struct TC
    {
        int num_partitions;
        int expected_batch;
    };
    for (const auto & tc : std::initializer_list<TC>{{1, 1024}, {4, 1024}, {64, 1024}, {128, 2048}, {1024, 16384}, {4096, 32768}})
    {
        std::vector<ColumnPrimitives> prims(1, makeFixedWidth<UInt64>());
        BumpArena arena(1ULL << 20);
        RadixShuffler op(tc.num_partitions, 1, std::move(prims), arena, false);
        EXPECT_EQ(op.batchSize(), tc.expected_batch) << "num_partitions=" << tc.num_partitions;
    }
}


// ─────────────────────────── Adaptable capacity tests ───────────────────────

TEST(RadixShuffler, AdaptiveCaps)
{
    // round64 and adaptiveCaps helpers are exercised via the operator.
    // Verify round64 semantics.
    EXPECT_EQ(round64(0), 0u);
    EXPECT_EQ(round64(1), 64u);
    EXPECT_EQ(round64(63), 64u);
    EXPECT_EQ(round64(64), 64u);
    EXPECT_EQ(round64(65), 128u);

    // adaptiveCaps: init ≤ kOutCapMin, max ≥ init, both multiples of 64.
    for (size_t num_partitions : {size_t{4}, size_t{64}, size_t{1024}, size_t{4096}})
    {
        const auto [init, maxc] = adaptiveCaps(100000, num_partitions);
        EXPECT_LE(init, kOutCapMin) << "P=" << num_partitions;
        EXPECT_GE(maxc, init) << "P=" << num_partitions;
        EXPECT_EQ(init % 64, 0u) << "init not multiple of 64 for P=" << num_partitions;
        EXPECT_EQ(maxc % 64, 0u) << "max not multiple of 64 for P=" << num_partitions;
    }
}


} // namespace
