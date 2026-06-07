#include <Interpreters/RadixHashJoin/GrowingArena.h>

#include <Common/Exception.h>
#include <Common/ProfileEvents.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <utility>

#include <unistd.h>
#include <sys/mman.h>

#if defined(OS_FREEBSD)
#include <stdlib.h> /// getpagesizes
#endif

namespace ProfileEvents
{
extern const Event RadixHashHugePagesUsed;
extern const Event RadixHashHugePagesFailed;
}

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

#if defined(MADV_HUGEPAGE) || defined(MADV_SUPERPAGE)

#if defined(OS_LINUX)
size_t readHugePageSizeFromMeminfo()
{
    FILE * file = std::fopen("/proc/meminfo", "re");
    if (!file)
        return 0;

    size_t result = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), file))
    {
        constexpr const char * prefix = "Hugepagesize:";
        if (std::strncmp(line, prefix, std::strlen(prefix)) != 0)
            continue;

        const char * cursor = line + std::strlen(prefix);
        while (*cursor == ' ' || *cursor == '\t')
            ++cursor;

        errno = 0;
        char * end = nullptr;
        const UInt64 kb = std::strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor)
            continue;
        while (*end == ' ' || *end == '\t')
            ++end;
        if (std::strncmp(end, "kB", 2) != 0)
            continue;
        if (kb > 0)
        {
            result = static_cast<size_t>(kb) * 1024;
            break;
        }
    }
    static_cast<void>(std::fclose(file));
    return result;
}
#endif

/// Runtime huge-page unit for THP / superpage madvise. Returns 0 when unsupported or unknown.
size_t hugePageSize()
{
    static const size_t size = []() -> size_t
    {
#ifdef _SC_HUGE_PAGESIZE
        if (const Int64 hp = ::sysconf(_SC_HUGE_PAGESIZE); hp > 0)
            return static_cast<size_t>(hp);
#endif

#if defined(OS_LINUX)
        if (const size_t from_meminfo = readHugePageSizeFromMeminfo(); from_meminfo > 0)
            return from_meminfo;
#endif

#if defined(OS_FREEBSD)
        /// FreeBSD exposes superpage sizes via getpagesizes().
        const int count = getpagesizes(nullptr, 0);
        if (count > 1)
        {
            size_t sizes[16];
            const int n = count <= 16 ? count : getpagesizes(sizes, 16);
            if (n > 1)
            {
                const size_t regular = pageSize();
                size_t best = 0;
                for (int i = 0; i < n; ++i)
                    if (sizes[i] > regular && sizes[i] > best)
                        best = sizes[i];
                if (best > 0)
                    return best;
            }
        }
#endif

        return size_t{0};
    }();
    return size;
}

#else

size_t hugePageSize()
{
    return 0;
}

#endif

}

GrowingArena::GrowingArena(size_t max_block_bytes, bool use_thp)
    : max_block(pageSize())
    , next_block_size(roundUp(INITIAL_BLOCK, pageSize()))
    , thp(false)
{
    const size_t hp = hugePageSize();
    thp = use_thp && hp > 0 && hp <= max_block_bytes;

    const size_t grain = thp ? hp : pageSize();
    max_block = std::max(roundUp(max_block_bytes, grain), pageSize());
    next_block_size = std::min(roundUp(INITIAL_BLOCK, pageSize()), max_block);

    /// With THP, blocks are huge-page-rounded; start the geometric growth at one huge page.
    if (thp)
        next_block_size = std::min(hp, max_block);
}

GrowingArena::GrowingArena(GrowingArena && other) noexcept
    : blocks(std::move(other.blocks))
    , max_block(other.max_block)
    , next_block_size(other.next_block_size)
    , thp(other.thp)
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
        thp = other.thp;
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
    /// dedicated block so the allocation stays contiguous. With THP every block is huge-page-rounded.
    const size_t grain = thp ? hugePageSize() : pageSize();
    const size_t want = std::max(next_block_size, min_bytes);
    const size_t size = std::max(roundUp(want, grain), grain);

    void * base = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED)
        throw Exception(ErrorCodes::CANNOT_ALLOCATE_MEMORY, "RadixHash::GrowingArena failed to mmap {} bytes", size);

    if (thp)
    {
        /// Fail-open: on a madvise error the block is still correct, just on regular pages (slower TLB).
#if defined(MADV_HUGEPAGE)
        if (::madvise(base, size, MADV_HUGEPAGE) == 0)
            ProfileEvents::increment(ProfileEvents::RadixHashHugePagesUsed);
        else
            ProfileEvents::increment(ProfileEvents::RadixHashHugePagesFailed);
#elif defined(MADV_SUPERPAGE)
        if (::madvise(base, size, MADV_SUPERPAGE) == 0)
            ProfileEvents::increment(ProfileEvents::RadixHashHugePagesUsed);
        else
            ProfileEvents::increment(ProfileEvents::RadixHashHugePagesFailed);
#endif
    }

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
    const auto * rel_start = reinterpret_cast<const char *>((reinterpret_cast<uintptr_t>(s) + page - 1) & ~(page - 1));
    const auto * rel_end = reinterpret_cast<const char *>(reinterpret_cast<uintptr_t>(e) & ~(page - 1));

    if (rel_end > rel_start)
        ::madvise(
            /// NOLINT(cppcoreguidelines-pro-type-const-cast)
            const_cast<char *>(rel_start),
            static_cast<size_t>(rel_end - rel_start),
            MADV_DONTNEED);
}

size_t GrowingArena::bytesReserved() const
{
    return std::accumulate(blocks.begin(), blocks.end(), size_t{0}, [](size_t sum, const Block & block) { return sum + block.size; });
}

size_t GrowingArena::bytesUsed() const
{
    return std::accumulate(blocks.begin(), blocks.end(), size_t{0}, [](size_t sum, const Block & block) { return sum + block.used; });
}

}
