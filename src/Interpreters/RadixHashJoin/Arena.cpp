#include <Interpreters/RadixHashJoin/Arena.h>

#include <base/defines.h>

#include <numeric>

namespace DB::RadixJoin
{

Arena::Arena()
    : mutex(std::make_unique<std::mutex>())
{
}

Arena::Arena(Arena && other) noexcept
    : blocks(std::move(other.blocks))
    , mutex(std::move(other.mutex))
{
    other.blocks.clear();
}

Arena & Arena::operator=(Arena && other) noexcept
{
    if (this != &other)
    {
        freeAll();
        blocks = std::move(other.blocks);
        mutex = std::move(other.mutex);
        other.blocks.clear();
    }
    return *this;
}

Arena::~Arena()
{
    freeAll();
}

void Arena::freeAll() noexcept
{
    for (auto & block : blocks)
        if (block.base != nullptr)
            allocator.free(block.base, block.size);
    blocks.clear();
}

void * Arena::allocate(size_t bytes, size_t alignment)
{
    chassert(alignment != 0 && (alignment & (alignment - 1)) == 0);

    /// jemalloc returns nullptr for a zero-byte request; round up so callers always get a valid
    /// distinct pointer (an empty leaf still wants a non-null, releasable base).
    if (bytes == 0)
        bytes = 1;

    /// Allocate outside the lock (jemalloc is thread-safe); only the bookkeeping is serialized.
    void * p = allocator.alloc(bytes, alignment);
    {
        std::lock_guard lock(*mutex);
        blocks.push_back(Block{static_cast<char *>(p), bytes});
    }
    return p;
}

void Arena::release(void * base) noexcept
{
    if (base == nullptr)
        return;

    size_t size = 0;
    {
        std::lock_guard lock(*mutex);
        for (auto & block : blocks)
        {
            if (block.base == base)
            {
                size = block.size;
                /// Swap-with-back + pop: O(1); block order does not matter.
                block = blocks.back();
                blocks.pop_back();
                break;
            }
        }
    }
    if (size != 0)
        allocator.free(base, size);
}

size_t Arena::blockCount() const
{
    std::lock_guard lock(*mutex);
    return blocks.size();
}

size_t Arena::bytesReserved() const
{
    std::lock_guard lock(*mutex);
    return std::accumulate(
        blocks.begin(), blocks.end(), size_t{0}, [](size_t sum, const Block & block) { return sum + block.size; });
}

}
