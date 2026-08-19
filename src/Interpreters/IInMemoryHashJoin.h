#pragma once

#include <Interpreters/IJoin.h>

namespace DB
{

class MatchedRowsStats;

/// Common interface for in-memory hash joins used by SpillingHashJoin and GraceHashJoin.
class IInMemoryHashJoin : public IJoin
{
public:
    ~IInMemoryHashJoin() override = default;

    /// Number of right-side rows ingested into the build.
    virtual size_t getRightTableRowCount() const = 0;
    /// Peak bytes the build occupied.
    virtual size_t getPeakBuildBytes() const = 0;
    /// Null when the query does not track matched rows.
    virtual const MatchedRowsStats * getMatchStats() const = 0;

    virtual BlocksList releaseJoinedBlocks(bool restructure = false) = 0;

    /// Parallel spill convert: default is a single chunk wrapping `releaseJoinedBlocks(false)`.
    virtual size_t getNumReleaseChunks() const { return 1; }
    virtual BlocksList releaseJoinedBlocksChunk(size_t chunk_idx)
    {
        if (chunk_idx != 0)
            return {};
        return releaseJoinedBlocks(false);
    }

    /// Drop remaining map/arena storage after a chunked convert has drained all worker lists.
    virtual void releaseJoinSideStorage() { }

    /// Free maps and per-slot arenas while stored blocks may still be draining. Caller must
    /// guarantee no further insert or probe on this join.
    virtual void releaseJoinMaps() { }

    virtual const Block & savedBlockSample() const = 0;

    virtual size_t getAndSetRightTableKeys() const = 0;

    virtual Block prepareRightBlock(const Block & block) const = 0;
};

using InMemoryHashJoinPtr = std::shared_ptr<IInMemoryHashJoin>;

}
