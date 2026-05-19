/// hashprobe-bench/generator/block_builder.cpp

#include "generator/block_builder.h"

#include <Columns/ColumnNullable.h>
#include <Columns/ColumnsNumber.h>
#include <Core/ColumnWithTypeAndName.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesNumber.h>

namespace DB::HashProbeBench
{

namespace
{

DataTypePtr makeBaseKeyType(uint32_t width)
{
    if (width == 32)
        return std::make_shared<DataTypeUInt32>();
    return std::make_shared<DataTypeUInt64>();
}

DataTypePtr makeKeyType(KeyShape shape)
{
    DataTypePtr base = makeBaseKeyType(shape.width);
    if (shape.nullable)
        return std::make_shared<DataTypeNullable>(base);
    return base;
}

} // namespace

Block BlockBuilder::makeHeader(KeyShape shape)
{
    ColumnsWithTypeAndName cols;
    DataTypePtr key_dt = makeKeyType(shape);
    for (uint32_t i = 0; i < shape.n; ++i)
        cols.push_back(ColumnWithTypeAndName(key_dt, "k" + std::to_string(i)));
    cols.push_back(ColumnWithTypeAndName(std::make_shared<DataTypeUInt64>(), "payload"));
    return Block(cols);
}

BlockBuilder::BlockBuilder(KeyShape shape, const ConfigType & config, KeyGenerator & gen)
    : shape_(shape)
    , block_size_(config.block_size)
    , build_iter_(gen.buildIterator())
    , probe_iter_(gen.probeIterator())
{
}

Block BlockBuilder::nextBuildBlock()
{
    return fillBlock(*build_iter_, build_done_, /*is_build=*/true);
}

Block BlockBuilder::nextProbeBlock()
{
    return fillBlock(*probe_iter_, probe_done_, /*is_build=*/false);
}

Block BlockBuilder::fillBlock(KeyGenerator::Iterator & iter, bool & done, bool is_build)
{
    if (done)
        return {};

    const uint32_t n        = shape_.n;
    const bool     nullable = shape_.nullable;
    const bool     is_w32   = (shape_.width == 32);

    // Allocate mutable column storage for key columns
    std::vector<MutableColumnPtr> key_nested(n);
    std::vector<MutableColumnPtr> key_nullmaps;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (is_w32)
            key_nested[i] = ColumnUInt32::create();
        else
            key_nested[i] = ColumnUInt64::create();
    }

    // J2: ColumnNullable wrapping is required whenever nullable == true,
    // REGARDLESS of whether null_fraction == 0.
    if (nullable)
    {
        key_nullmaps.resize(n);
        for (uint32_t i = 0; i < n; ++i)
            key_nullmaps[i] = ColumnUInt8::create();
    }

    auto payload_col = ColumnUInt64::create();

    // Fill rows
    uint32_t rows_filled = 0;
    KeyGenerator::KeyRow row;
    while (rows_filled < block_size_ && iter.next(row))
    {
        for (uint32_t col = 0; col < n; ++col)
        {
            const uint64_t kv = row.key_values[col];
            if (is_w32)
                static_cast<ColumnUInt32 &>(*key_nested[col]).getData().push_back(static_cast<UInt32>(kv));
            else
                static_cast<ColumnUInt64 &>(*key_nested[col]).getData().push_back(kv);

            if (nullable)
                static_cast<ColumnUInt8 &>(*key_nullmaps[col]).getData().push_back(row.null_mask[col]);
        }
        payload_col->getData().push_back(row.payload);
        ++rows_filled;
    }

    if (rows_filled == 0)
    {
        done = true;
        return {};
    }

    ColumnsWithTypeAndName cols;
    DataTypePtr key_dt = makeKeyType(shape_);

    for (uint32_t i = 0; i < n; ++i)
    {
        ColumnPtr col;
        if (nullable)
            col = ColumnNullable::create(std::move(key_nested[i]), std::move(key_nullmaps[i]));
        else
            col = std::move(key_nested[i]);
        const std::string col_name = is_build
            ? ("b_k" + std::to_string(i))
            : ("k" + std::to_string(i));
        cols.push_back(ColumnWithTypeAndName(std::move(col), key_dt, col_name));
    }
    cols.push_back(ColumnWithTypeAndName(
        std::move(payload_col), std::make_shared<DataTypeUInt64>(),
        is_build ? "b_payload" : "payload"));

    return Block(std::move(cols));
}

} // namespace DB::HashProbeBench
