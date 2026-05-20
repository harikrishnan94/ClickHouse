#pragma once

#include <base/types.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace DB
{

/// Bump allocator backed by 64 MiB mmap-aligned slabs.
/// Allocations are 64-byte aligned. reset() frees all slabs to the OS.
/// Thread-unsafe: each caller owns one arena exclusively.
class BumpArena
{
public:
    static constexpr size_t kSlabSize = 64ULL << 20; /// 64 MiB

    BumpArena() = default;

    ~BumpArena() { reset(); }

    BumpArena(const BumpArena &) = delete;
    BumpArena & operator=(const BumpArena &) = delete;

    BumpArena(BumpArena && other) noexcept
        : slabs_(std::move(other.slabs_))
        , cur_(other.cur_)
        , remaining_(other.remaining_)
    {
        other.cur_ = nullptr;
        other.remaining_ = 0;
    }

    BumpArena & operator=(BumpArena && other) noexcept
    {
        if (this != &other)
        {
            reset();
            slabs_ = std::move(other.slabs_);
            cur_ = other.cur_;
            remaining_ = other.remaining_;
            other.cur_ = nullptr;
            other.remaining_ = 0;
        }
        return *this;
    }

    /// Allocate `bytes` rounded up to 64-byte alignment.
    uint8_t * alloc(size_t bytes)
    {
        bytes = (bytes + 63) & ~static_cast<size_t>(63);
        if (bytes > remaining_)
            grow(std::max(bytes, kSlabSize));
        uint8_t * p = cur_;
        cur_ += bytes;
        remaining_ -= bytes;
        return p;
    }

    /// Total bytes allocated across all slabs (upper bound, conservative).
    [[nodiscard]] size_t allocatedBytes() const { return slabs_.size() * kSlabSize; }

    /// Return all slabs to the OS and reset to empty.
    void reset()
    {
        for (auto * s : slabs_)
            std::free(s);
        slabs_.clear();
        cur_ = nullptr;
        remaining_ = 0;
    }

private:
    void grow(size_t min_size)
    {
        const size_t sz = (std::max(min_size, kSlabSize) + 4095) & ~static_cast<size_t>(4095);
        void * p = nullptr;
        if (posix_memalign(&p, 64, sz) != 0)
            throw std::bad_alloc();
        slabs_.push_back(static_cast<uint8_t *>(p));
        cur_ = static_cast<uint8_t *>(p);
        remaining_ = sz;
    }

    std::vector<uint8_t *> slabs_;
    uint8_t * cur_ = nullptr;
    size_t remaining_ = 0;
};

}
