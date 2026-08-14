#pragma once

#include <Interpreters/IJoin.h>

namespace DB
{

/// Common interface for in-memory hash joins used by SpillingHashJoin and GraceHashJoin.
class IInMemoryHashJoin : public IJoin
{
public:
    ~IInMemoryHashJoin() override = default;

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

    /// Drop just the hash-table maps and their per-slot arenas, before the stored right-side
    /// blocks have been drained. Safe to call as soon as the caller can guarantee no further
    /// insert or probe will touch this join's maps (e.g. right after switching to a different
    /// join implementation under an exclusive lock) — it lets the (often large) maps be freed
    /// without waiting for every worker's stored-block chunk to be converted first.
    virtual void releaseJoinMaps() { }

    virtual const Block & savedBlockSample() const = 0;

    virtual size_t getAndSetRightTableKeys() const = 0;

    virtual Block prepareRightBlock(const Block & block) const = 0;
};

using InMemoryHashJoinPtr = std::shared_ptr<IInMemoryHashJoin>;

}
