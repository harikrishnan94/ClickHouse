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

    virtual const Block & savedBlockSample() const = 0;

    virtual size_t getAndSetRightTableKeys() const = 0;

    virtual Block prepareRightBlock(const Block & block) const = 0;
};

using InMemoryHashJoinPtr = std::shared_ptr<IInMemoryHashJoin>;

}
