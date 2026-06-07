#include <Interpreters/RadixHashJoin/GrowingArena.h>

#include <Common/Allocator.h>
#include <Common/Exception.h>

#include <numeric>
#include <utility>

namespace DB::RadixHash
{

GrowingArena::GrowingArena(size_t /*max_block_bytes*/)
    : blocks_mutex(std::make_unique<std::mutex>())
{
}

GrowingArena::GrowingArena(GrowingArena && other) noexcept
    : blocks(std::move(other.blocks))
    , blocks_mutex(std::move(other.blocks_mutex))
{
    other.blocks.clear();
}

GrowingArena & GrowingArena::operator=(GrowingArena && other) noexcept
{
    if (this != &other)
    {
        freeAll();
        blocks = std::move(other.blocks);
        blocks_mutex = std::move(other.blocks_mutex);
        other.blocks.clear();
    }
    return *this;
}

GrowingArena::~GrowingArena()
{
    freeAll();
}

void GrowingArena::freeAll() noexcept
{
    for (auto & block : blocks)
        if (block.base != nullptr)
            allocator.free(block.base, block.size);
    blocks.clear();
}

void * GrowingArena::alloc(size_t bytes, size_t alignment)
{
    chassert(alignment != 0 && (alignment & (alignment - 1)) == 0);

    if (bytes == 0)
        bytes = 1;

    /// The allocation (and any first-touch faulting) runs lock-free — jemalloc is thread-safe; only the
    /// block bookkeeping is serialized, so concurrent workers allocate their leaf/partition arrays in
    /// parallel.
    void * p = allocator.alloc(bytes, alignment);
    {
        std::lock_guard lock(*blocks_mutex);
        blocks.push_back(Block{static_cast<char *>(p), bytes});
    }
    return p;
}

void GrowingArena::freeBlock(void * base) noexcept
{
    if (base == nullptr)
        return;
    size_t size = 0;
    {
        std::lock_guard lock(*blocks_mutex);
        for (auto & block : blocks)
        {
            if (block.base == base)
            {
                size = block.size;
                block = blocks.back(); /// swap-with-back + pop: O(1), order is irrelevant
                blocks.pop_back();
                break;
            }
        }
    }
    if (size != 0) /// free outside the lock — jemalloc is thread-safe
        allocator.free(base, size);
}

size_t GrowingArena::bytesReserved() const
{
    return std::accumulate(blocks.begin(), blocks.end(), size_t{0}, [](size_t sum, const Block & block) { return sum + block.size; });
}

}
