#include <Interpreters/PartitionedHashJoin/RadixShuffleColumnFixed.h>
#include <Interpreters/PartitionedHashJoin/RadixShuffleColumnFixedString.h>
#include <Interpreters/PartitionedHashJoin/ThreadSlot.h>

namespace DB
{

/// Nullable columns are expanded to TWO scatter_cols entries by makeBuildSpec() →
/// pushColDesc(). Each entry is a flat RadixShuffleColumnFixed — no composite needed.
static RadixShuffleColumnPtr makeColumnShuffler(const ShuffleColDesc & desc, size_t P)
{
    switch (desc.elem_bytes)
    {
        case 1:
            return std::make_unique<RadixShuffleColumnFixed<uint8_t>>(P);
        case 2:
            return std::make_unique<RadixShuffleColumnFixed<uint16_t>>(P);
        case 4:
            return std::make_unique<RadixShuffleColumnFixed<uint32_t>>(P);
        case 8:
            return std::make_unique<RadixShuffleColumnFixed<uint64_t>>(P);
        case 16:
            return std::make_unique<RadixShuffleColumnFixedString<16>>(P);
        default:
            return std::make_unique<RadixShuffleColumnFixed<uint8_t>>(P);
    }
}

void ThreadSlot::initBuildSide(const ShuffleSpec & build_spec)
{
    if (build_initialised)
        return;
    build_parts.resize(build_spec.P);
    build_cols.reserve(build_spec.scatter_cols.size());
    for (const auto & sc : build_spec.scatter_cols)
        build_cols.push_back(makeColumnShuffler(sc, build_spec.P));
    build_initialised = true;
}

void ThreadSlot::initProbeSide(size_t P)
{
    if (probe_initialised)
        return;
    probe_parts.resize(P);
    probe_initialised = true;
}

}
