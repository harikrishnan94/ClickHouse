#include <Interpreters/RadixHashJoin/Arena.h>

#include <base/defines.h>

#include <numeric>
#include <unordered_set>

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

void Arena::releaseMany(const std::vector<void *> & bases) noexcept
{
    if (bases.empty())
        return;

    /// The lookup set makes the whole batch one O(blocks + bases) pass instead of bases.size() scans.
    std::unordered_set<const void *> to_release;
    to_release.reserve(bases.size());
    for (const void * base : bases)
        if (base != nullptr)
            to_release.insert(base);
    if (to_release.empty())
        return;

    std::vector<Block> released;
    released.reserve(to_release.size());
    {
        std::lock_guard lock(*mutex);
        size_t kept = 0;
        for (size_t i = 0; i < blocks.size(); ++i)
        {
            if (to_release.contains(blocks[i].base))
                released.push_back(blocks[i]);
            else
                blocks[kept++] = blocks[i];
        }
        blocks.resize(kept);
    }
    /// Free outside the lock (jemalloc is thread-safe); only the bookkeeping is serialized.
    for (const Block & block : released)
        allocator.free(block.base, block.size);
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
