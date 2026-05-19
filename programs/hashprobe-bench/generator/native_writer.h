#pragma once

/// hashprobe-bench/generator/native_writer.h
///
/// NativeFileWriter: serializes build-side and probe-side Block streams to
/// <output_dir>/build.native and <output_dir>/probe.native using ClickHouse's
/// native binary format.
///
/// Each file carries inline per-block headers (column names + types) as
/// required by the ClickHouse native format.  Blocks are written in the same
/// order and row-count as they will be fed to HashJoin.

#include "generator/block_builder.h"
#include <hashprobe_bench/config.h>
#include <hashprobe_bench/types.h>

#include <string>

namespace DB::HashProbeBench
{

class NativeFileWriter
{
public:
    NativeFileWriter(KeyShape shape, const ConfigType & config, KeyGenerator & gen);

    /// Write all build-side blocks to <output_dir>/build.native.
    void writeBuild();

    /// Write all probe-side blocks to <output_dir>/probe.native.
    void writeProbe();

    /// Convenience: write both build and probe files.
    void writeAll();

private:
    void writeStream(const std::string & path, bool is_build);

    KeyShape         shape_;
    const ConfigType & config_;
    KeyGenerator &   gen_;
};

} // namespace DB::HashProbeBench
