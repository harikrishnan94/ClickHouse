#include <Interpreters/RadixHashJoin/GrowingArena.h>

#include <Common/Exception.h>

#include <algorithm>
#include <cstdint>
#include <utility>

#include <sys/mman.h>
#include <unistd.h>

namespace DB
{
namespace ErrorCodes
{
extern const int CANNOT_ALLOCATE_MEMORY;
}
}

namespace DB::RadixHash
{

namespace
{

size_t pageSize()
{
    static const size_t page = []
    {
        const Int64 p = ::sysconf(_SC_PAGESIZE);
        return p > 0 ? static_cast<size_t>(p) : size_t{4096};
    }();
    return page;
}

size_t roundUp(size_t v, size_t a)
{
    return (v + a - 1) & ~(a - 1);
}

}

GrowingArena::GrowingArena(size_t max_block_bytes)
    : max_block(std::max<size_t>(roundUp(max_block_bytes, pageSize()), pageSize()))
    , next_block_size(std::min(roundUp(INITIAL_BLOCK, pageSize()), max_block))
{
}

GrowingArena::GrowingArena(GrowingArena && other) noexcept
    : blocks(std::move(other.blocks)), max_block(other.max_block), next_block_size(other.next_block_size)
{
    other.blocks.clear();
}

GrowingArena & GrowingArena::operator=(GrowingArena && other) noexcept
{
    if (this != &other)
    {
        freeAll();
        blocks = std::move(other.blocks);
        max_block = other.max_block;
        next_block_size = other.next_block_size;
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
            ::munmap(block.base, block.size);
    blocks.clear();
}

void GrowingArena::addBlock(size_t min_bytes)
{
    /// Geometric growth capped at `max_block`; a single allocation larger than the cap gets its own
    /// page-rounded dedicated block so the allocation stays contiguous.
    const size_t want = std::max(next_block_size, min_bytes);
    const size_t size = roundUp(want, pageSize());

    void * base = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED)
        throw Exception(ErrorCodes::CANNOT_ALLOCATE_MEMORY, "RadixHash::GrowingArena failed to mmap {} bytes", size);

    blocks.push_back(Block{static_cast<char *>(base), size, 0});

    if (next_block_size < max_block)
        next_block_size = std::min(max_block, next_block_size * 2);
}

void * GrowingArena::alloc(size_t bytes, size_t alignment)
{
    /// Alignment must be a power of two no larger than a page (mmap returns only page-aligned memory).
    chassert(alignment != 0 && (alignment & (alignment - 1)) == 0 && alignment <= pageSize());
    chassert(bytes <= SIZE_MAX - alignment - pageSize()); /// no overflow in the size rounding below

    if (bytes == 0)
        bytes = 1;

    if (blocks.empty())
        addBlock(bytes + alignment);

    size_t pos = roundUp(blocks.back().used, alignment);
    if (pos + bytes > blocks.back().size)
    {
        addBlock(bytes + alignment);
        pos = roundUp(blocks.back().used, alignment);
    }

    Block & cur = blocks.back();
    void * p = cur.base + pos;
    cur.used = pos + bytes;
    return p;
}

void GrowingArena::trim() noexcept
{
    const size_t page = pageSize();
    for (auto & block : blocks)
    {
        const size_t keep = roundUp(block.used, page);
        if (keep < block.size)
            ::madvise(block.base + keep, block.size - keep, MADV_DONTNEED);
    }
}

void GrowingArena::releaseRange(const void * range_start, size_t bytes) noexcept
{
    if (!range_start || bytes == 0)
        return;

    const size_t page = pageSize();
    const auto * s = static_cast<const char *>(range_start);
    const auto * e = s + bytes;

    /// Round start UP and end DOWN to page boundaries — never touch a page shared with neighbours.
    const auto * rel_start = reinterpret_cast<const char *>(
        (reinterpret_cast<uintptr_t>(s) + page - 1) & ~(page - 1));
    const auto * rel_end = reinterpret_cast<const char *>(
        reinterpret_cast<uintptr_t>(e) & ~(page - 1));

    if (rel_end > rel_start)
        ::madvise(
            /// NOLINT(cppcoreguidelines-pro-type-const-cast)
            const_cast<char *>(rel_start),
            static_cast<size_t>(rel_end - rel_start),
            MADV_DONTNEED);
}

size_t GrowingArena::bytesReserved() const
{
    size_t total = 0;
    for (const auto & block : blocks)
        total += block.size;
    return total;
}

size_t GrowingArena::bytesUsed() const
{
    size_t total = 0;
    for (const auto & block : blocks)
        total += block.used;
    return total;
}

}
