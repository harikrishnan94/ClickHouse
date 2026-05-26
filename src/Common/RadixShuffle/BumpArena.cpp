#include <Common/RadixShuffle/BumpArena.h>

#include <algorithm>
#include <cstdlib>
#include <new>


namespace DB
{

namespace
{

constexpr size_t kOsPageSize = 4096;

[[nodiscard]] inline size_t roundUpToPage(size_t n) noexcept
{
    return (n + kOsPageSize - 1) & ~(kOsPageSize - 1);
}

} // namespace


BumpArena::BumpArena(size_t initial_size, size_t growth_factor, size_t linear_growth_threshold)
    : initial_(initial_size)
    , growth_(growth_factor)
    , linear_(linear_growth_threshold)
{
}


BumpArena::~BumpArena()
{
    for (auto & s : slabs_)
        std::free(s.ptr);
}


BumpArena::BumpArena(BumpArena && other) noexcept
    : initial_(other.initial_)
    , growth_(other.growth_)
    , linear_(other.linear_)
    , idx_(other.idx_)
    , cur_(other.cur_)
    , end_(other.end_)
    , allocated_(other.allocated_)
    , used_(other.used_)
    , slabs_(std::move(other.slabs_))
{
    other.idx_ = kNoSlab;
    other.cur_ = nullptr;
    other.end_ = nullptr;
    other.allocated_ = 0;
    other.used_ = 0;
}


char * BumpArena::alloc(size_t size)
{
    used_ += size;
    if (!cur_ || avail() < size)
        nextSlab(size);
    char * r = cur_;
    cur_ += size;
    return r;
}


char * BumpArena::alignedAlloc(size_t size, size_t alignment)
{
    const size_t pad = cur_ ? ((-reinterpret_cast<uintptr_t>(cur_)) & (alignment - 1)) : 0;
    if (!cur_ || avail() < pad + size)
    {
        nextSlab(size + alignment);
        // New slab starts at 64-byte alignment (posix_memalign(64)).
        // Recompute padding in case alignment > 64.
        const size_t pad2 = (-reinterpret_cast<uintptr_t>(cur_)) & (alignment - 1);
        cur_ += pad2;
        used_ += size;
        char * r = cur_;
        cur_ += size;
        return r;
    }
    cur_ += pad;
    used_ += size;
    char * r = cur_;
    cur_ += size;
    return r;
}


void BumpArena::reset()
{
    if (slabs_.empty())
        return;
    idx_ = 0;
    cur_ = slabs_[0].ptr;
    end_ = slabs_[0].ptr + slabs_[0].sz;
    used_ = 0;
}


void BumpArena::nextSlab(size_t min_size)
{
    const size_t ni = (idx_ == kNoSlab) ? 0 : idx_ + 1;

    // Fast path: reuse a pre-warmed slab from a previous repetition.
    if (ni < slabs_.size() && slabs_[ni].sz >= min_size)
    {
        idx_ = ni;
        cur_ = slabs_[ni].ptr;
        end_ = cur_ + slabs_[ni].sz;
        return;
    }

    // Slab growth policy: double up to linear_threshold_, then fixed linear increments.
    size_t sz;
    if (slabs_.empty())
        sz = std::max(min_size, initial_);
    else if (allocated_ < linear_)
        sz = std::max(min_size, slabs_.back().sz * growth_);
    else
        sz = ((min_size + linear_ - 1) / linear_) * linear_;

    sz = roundUpToPage(sz);

    void * p = nullptr;
    if (posix_memalign(&p, 64, sz) != 0)
        throw std::bad_alloc();

    Slab s{static_cast<char *>(p), sz};

    // Insert immediately after the current slab so reset() replays in order.
    if (ni >= slabs_.size())
        slabs_.push_back(s);
    else
        slabs_.insert(slabs_.begin() + static_cast<std::ptrdiff_t>(ni), s);

    idx_ = ni;
    cur_ = s.ptr;
    end_ = cur_ + sz;
    allocated_ += sz;
}

} // namespace DB
