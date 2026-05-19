/// hashprobe-bench/generator/native_writer.cpp

#include "generator/native_writer.h"

#include <Formats/NativeWriter.h>
#include <IO/WriteBufferFromFile.h>
#include <Core/Block.h>

#include <memory>
#include <stdexcept>

namespace DB::HashProbeBench
{

NativeFileWriter::NativeFileWriter(KeyShape shape, const ConfigType & config, KeyGenerator & gen)
    : shape_(shape), config_(config), gen_(gen)
{
}

void NativeFileWriter::writeBuild()
{
    const std::string path = config_.output_dir + "/build.native";
    writeStream(path, /*is_build=*/true);
}

void NativeFileWriter::writeProbe()
{
    const std::string path = config_.output_dir + "/probe.native";
    writeStream(path, /*is_build=*/false);
}

void NativeFileWriter::writeAll()
{
    writeBuild();
    writeProbe();
}

void NativeFileWriter::writeStream(const std::string & path, bool is_build)
{
    BlockBuilder builder(shape_, config_, gen_);

    // Build a shared header for the NativeWriter
    Block hdr = BlockBuilder::makeHeader(shape_);
    SharedHeader shared_hdr = std::make_shared<const Block>(std::move(hdr));

    WriteBufferFromFile write_buf(path);
    NativeWriter writer(write_buf, /*client_revision=*/0, shared_hdr);

    if (is_build)
    {
        while (builder.hasBuildRows())
        {
            Block blk = builder.nextBuildBlock();
            if (blk.empty())
                break;
            writer.write(blk);
        }
    }
    else
    {
        while (builder.hasProbeRows())
        {
            Block blk = builder.nextProbeBlock();
            if (blk.empty())
                break;
            writer.write(blk);
        }
    }

    writer.flush();
}

} // namespace DB::HashProbeBench
