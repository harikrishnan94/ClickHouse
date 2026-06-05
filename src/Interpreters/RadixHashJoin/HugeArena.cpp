#include <Interpreters/RadixHashJoin/HugeArena.h>

#include <Common/Exception.h>
#include <Common/ProfileEvents.h>

#include <algorithm>
#include <cstdlib>
#include <utility>

#include <sys/mman.h>

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

HugeArena::HugeArena(HugeArena && other) noexcept
    : slabs(std::move(other.slabs))
    , huge_pages_used(other.huge_pages_used)
    , huge_pages_failed(other.huge_pages_failed)
{
    other.slabs.clear();
    other.huge_pages_used = 0;
    other.huge_pages_failed = 0;
}

HugeArena & HugeArena::operator=(HugeArena && other) noexcept
{
    if (this != &other)
    {
        freeAll();
        slabs = std::move(other.slabs);
        huge_pages_used = other.huge_pages_used;
        huge_pages_failed = other.huge_pages_failed;
        other.slabs.clear();
        other.huge_pages_used = 0;
        other.huge_pages_failed = 0;
    }
    return *this;
}

HugeArena::~HugeArena()
{
    freeAll();
}

void HugeArena::freeAll() noexcept
{
    for (auto & slab : slabs)
        free(slab.base); /// NOLINT(cppcoreguidelines-no-malloc) -- paired with posix_memalign
    slabs.clear();
}

void HugeArena::addSlab(size_t min_bytes)
{
    /// Round up to a whole number of 2 MiB slabs so the whole mapping is huge-page-eligible.
    const size_t size = ((std::max(SLAB, min_bytes) + SLAB - 1) / SLAB) * SLAB;

    void * base = nullptr;
    if (posix_memalign(&base, SLAB, size) != 0 || base == nullptr)
        throw Exception(ErrorCodes::CANNOT_ALLOCATE_MEMORY, "RadixHash::HugeArena failed to allocate {} bytes", size);

#if defined(MADV_HUGEPAGE)
    if (madvise(base, size, MADV_HUGEPAGE) == 0)
    {
        ++huge_pages_used;
        ProfileEvents::increment(ProfileEvents::RadixHashHugePagesUsed);
    }
    else
    {
        /// Fail-open: keep going on 4 KiB pages (e.g. EINVAL when THP is disabled).
        ++huge_pages_failed;
        ProfileEvents::increment(ProfileEvents::RadixHashHugePagesFailed);
    }
#else
    ++huge_pages_failed;
    ProfileEvents::increment(ProfileEvents::RadixHashHugePagesFailed);
#endif

    slabs.push_back(Slab{base, size, 0});
}

void * HugeArena::alloc(size_t bytes, size_t alignment)
{
    if (bytes == 0)
        bytes = 1;

    if (slabs.empty())
        addSlab(bytes + alignment);

    auto align_up = [](size_t v, size_t a) { return (v + a - 1) & ~(a - 1); };

    size_t pos = align_up(slabs.back().used, alignment);
    if (pos + bytes > slabs.back().size)
    {
        addSlab(bytes + alignment);
        pos = align_up(slabs.back().used, alignment);
    }

    Slab & cur = slabs.back();
    void * p = static_cast<char *>(cur.base) + pos;
    cur.used = pos + bytes;
    return p;
}

size_t HugeArena::bytesReserved() const
{
    size_t total = 0;
    for (const auto & slab : slabs)
        total += slab.size;
    return total;
}

}
