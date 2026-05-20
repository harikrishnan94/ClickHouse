#pragma once

#include <Interpreters/PartitionedHashJoin/NTFlush.h>
#include <Interpreters/PartitionedHashJoin/RadixShuffleColumn.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace DB
{

/// Concrete scatter column for fixed-size numeric/date/decimal types.
/// T ∈ {uint8_t, uint16_t, uint32_t, uint64_t}
template <typename T>
class RadixShuffleColumnFixed final : public RadixShuffleColumn
{
public:
    explicit RadixShuffleColumnFixed(size_t P)
        : P_(P)
        , staging_(nullptr)
        , out_ptrs_(nullptr)
        , flush_fn_(getNTFlushFn<T>()) /// dispatch once at construction
    {
        /// staging_[p*8 + slot] — partition-first layout, 64-byte aligned.
        if (posix_memalign(reinterpret_cast<void **>(&staging_), 64, P * 8 * sizeof(T)) != 0)
            throw std::bad_alloc();
        std::memset(staging_, 0, P * 8 * sizeof(T));

        out_ptrs_ = new T *[P]();
    }

    ~RadixShuffleColumnFixed() override
    {
        std::free(staging_);
        delete[] out_ptrs_;
    }

    RadixShuffleColumnFixed(const RadixShuffleColumnFixed &) = delete;
    RadixShuffleColumnFixed & operator=(const RadixShuffleColumnFixed &) = delete;
    RadixShuffleColumnFixed(RadixShuffleColumnFixed &&) = delete;
    RadixShuffleColumnFixed & operator=(RadixShuffleColumnFixed &&) = delete;

    // ── Phase 3: set write pointer for partition p to the start of a new chunk ──
    void on_grow(size_t p, void * col_base) override { out_ptrs_[p] = static_cast<T *>(col_base); }

    // ── Phase 3 (SWWC only): drain staged rows before chunk grows ─────────────
    void drain_one(size_t p, uint32_t cnt) override
    {
        const T * src = staging_ + p * 8;
        T * dst = out_ptrs_[p];
        for (uint32_t s = 0; s < cnt; ++s)
            dst[s] = src[s];
        out_ptrs_[p] = dst + cnt;
    }

    // ── Phase 4b direct: branch-free live-pointer scatter ─────────────────────
    void scatter_direct(const uint16_t * pids, const void * src, size_t n) override
    {
        const T * s = static_cast<const T *>(src);
        for (size_t j = 0; j < n; ++j)
            *out_ptrs_[pids[j]]++ = s[j];
    }

    // ── Phase 4b SWWC: staging fill + inline NT-store flush ──────────────────
    //
    // Inline flush is required for correctness: a single batch can route more
    // than 8 rows to the same partition (batch_size = max(1024, P*16), so for
    // P=64 the average is 16 rows per partition). If we deferred all flushes
    // until after the fill loop, the 9th+ rows would overwrite staging slots
    // 0..k before the original 8 were flushed, losing data.
    //
    // The branch is highly predictable (slot==7 fires only on every 8th row
    // for a given partition) and the flush is amortised over 8 inserts.
    void scatter_staged(const uint16_t * pids, const uint8_t * positions, const void * src, size_t n) override
    {
        const T * s = static_cast<const T *>(src);

        for (size_t j = 0; j < n; ++j)
        {
            const uint16_t p = pids[j];
            const uint8_t slot = positions[j];
            staging_[(static_cast<size_t>(p) * 8) + slot] = s[j];
            if (slot == 7)
                flush_fn_(staging_, out_ptrs_, p);
        }
    }

    [[nodiscard]] T * stagingBase() { return staging_; }
    [[nodiscard]] T ** outPtrs() { return out_ptrs_; }
    [[nodiscard]] size_t P() const { return P_; }

private:
    size_t P_;
    T * staging_;
    T ** out_ptrs_;
    NTFlushFn<T> flush_fn_; /// function pointer, dispatched once at ctor
};

}
