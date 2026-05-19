#pragma once

/// hashprobe-bench/generator/block_builder.h
///
/// BlockBuilder: assembles ClickHouse Blocks from KeyGenerator row streams.
///
/// Each block contains shape.n key columns plus one UInt64 payload column.
/// Key columns are ColumnNullable when shape.nullable == true (J2 compliance).
/// Block size is exactly config.block_size rows; the last block may be shorter.

#include "generator/key_generator.h"

#include <hashprobe_bench/config.h>
#include <hashprobe_bench/types.h>

#include <Core/Block.h>

#include <memory>

namespace DB::HashProbeBench
{

class BlockBuilder
{
public:
    BlockBuilder(KeyShape shape, const ConfigType & config, KeyGenerator & gen);

    /// Next build-side Block; returns empty Block when all build rows are consumed.
    Block nextBuildBlock();

    /// Next probe-side Block; returns empty Block when all probe rows are consumed.
    Block nextProbeBlock();

    bool hasBuildRows() const { return !build_done_; }
    bool hasProbeRows() const { return !probe_done_; }

    /// Returns a header-only Block (columns are nullptr) matching the generated schema.
    static Block makeHeader(KeyShape shape);

private:
    Block fillBlock(KeyGenerator::Iterator & iter, bool & done, bool is_build = false);

    KeyShape   shape_;
    uint32_t   block_size_;

    std::unique_ptr<KeyGenerator::Iterator> build_iter_;
    std::unique_ptr<KeyGenerator::Iterator> probe_iter_;

    bool build_done_ = false;
    bool probe_done_ = false;
};

} // namespace DB::HashProbeBench
