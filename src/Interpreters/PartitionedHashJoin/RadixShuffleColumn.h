#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace DB
{

/// Abstract interface for one column's scatter operations inside the radix shuffle.
/// Matches the algo-doc §"IColumn interface" verbatim:
///   on_grow / drain_one / scatter_direct / scatter_staged
///
/// One virtual call per column per batch (~batch_size=16384 rows); overhead < 0.1%.
class RadixShuffleColumn
{
public:
    RadixShuffleColumn() = default;
    virtual ~RadixShuffleColumn() = default;

    RadixShuffleColumn(const RadixShuffleColumn &) = delete;
    RadixShuffleColumn & operator=(const RadixShuffleColumn &) = delete;
    RadixShuffleColumn(RadixShuffleColumn &&) = delete;
    RadixShuffleColumn & operator=(RadixShuffleColumn &&) = delete;

    /// Phase 3: reset the live write pointer for partition `p` to `col_base`
    /// (first byte of that partition's column buffer in a newly allocated OutBlock).
    virtual void on_grow(size_t p, void * col_base) = 0;

    /// Phase 3 (SWWC only): scalar drain `cnt` in-flight staged rows for partition `p`
    /// into the partition's current OutBlock chunk. Advances out_ptrs_[p] by cnt.
    virtual void drain_one(size_t p, uint32_t cnt) = 0;

    /// Phase 4b (direct mode): branch-free live-pointer scatter.
    ///   for j in 0..n-1: *out_ptrs_[pids[j]]++ = src[j]
    virtual void scatter_direct(const uint16_t * pids, const void * src, size_t n) = 0;

    /// Phase 4b (SWWC mode): staging fill + inline NT-store flush.
    ///   for j in 0..n-1:
    ///     staging_[pids[j]*8 + positions[j]] = src[j]
    ///     if positions[j] == 7: NT-store staging_[pids[j]*8..pids[j]*8+7]
    /// Inline flush is required because a partition can receive >8 rows per batch;
    /// deferring flushes until after the fill loop would let later rows overwrite
    /// staging slots before the original 8 are flushed.
    virtual void scatter_staged(const uint16_t * pids, const uint8_t * positions, const void * src, size_t n) = 0;
};

using RadixShuffleColumnPtr = std::unique_ptr<RadixShuffleColumn>;

}
