#pragma once

#include <Interpreters/PartitionedHashJoin/NTFlush.h>
#include <Interpreters/PartitionedHashJoin/RadixShuffleColumn.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace DB
{

/// Concrete scatter column for FixedString(N) where N ∈ {2, 4, 8, 16}.
/// Each "element" is N bytes; we treat it as a UInt<N*8> for scatter purposes.
///
/// The NT-flush helper is chosen based on N:
///   N=2  → flushStagingNT<uint16_t>
///   N=4  → flushStagingNT<uint32_t>
///   N=8  → flushStagingNT<uint64_t>
///   N=16 → two flushStagingNT<uint64_t> per flush (two 8-byte halves per slot)
template <size_t N>
class RadixShuffleColumnFixedString final : public RadixShuffleColumn
{
    static_assert(N == 2 || N == 4 || N == 8 || N == 16, "Only N=2,4,8,16 supported");

    /// Underlying flush type: 8 slots of N bytes each = 8*N bytes per flush.
    /// For N≤8 we use a single aligned NT store. For N=16 we use two uint64_t stores.
    using FlushT = std::conditional_t<(N <= 2), uint16_t, std::conditional_t<(N <= 4), uint32_t, uint64_t>>;

public:
    explicit RadixShuffleColumnFixedString(size_t P)
        : P_(P)
        , staging_(nullptr)
        , out_ptrs_(nullptr)
    {
        /// staging_[p*8 + slot] — each slot is N bytes; total = P*8*N, 64-aligned.
        if (posix_memalign(reinterpret_cast<void **>(&staging_), 64, P * 8 * N) != 0)
            throw std::bad_alloc();
        std::memset(staging_, 0, P * 8 * N);

        out_ptrs_ = new uint8_t *[P]();

        /// For N=16 we flush 2 uint64_t blocks per slot; reuse flushU64.
        /// For N=8, 4, 2 — reuse the corresponding typed flush.
        flush_fn_ = getNTFlushFn<FlushT>();
    }

    ~RadixShuffleColumnFixedString() override
    {
        std::free(staging_);
        delete[] out_ptrs_;
    }

    RadixShuffleColumnFixedString(const RadixShuffleColumnFixedString &) = delete;
    RadixShuffleColumnFixedString & operator=(const RadixShuffleColumnFixedString &) = delete;
    RadixShuffleColumnFixedString(RadixShuffleColumnFixedString &&) = delete;
    RadixShuffleColumnFixedString & operator=(RadixShuffleColumnFixedString &&) = delete;

    void on_grow(size_t p, void * col_base) override { out_ptrs_[p] = static_cast<uint8_t *>(col_base); }

    void drain_one(size_t p, uint32_t cnt) override
    {
        const uint8_t * src = staging_ + (p * 8 * N);
        uint8_t * dst = out_ptrs_[p];
        const size_t bytes = static_cast<size_t>(cnt) * N;
        std::memcpy(dst, src, bytes);
        out_ptrs_[p] = dst + bytes;
    }

    void scatter_direct(const uint16_t * pids, const void * src, size_t n) override
    {
        const uint8_t * s = static_cast<const uint8_t *>(src);
        for (size_t j = 0; j < n; ++j)
        {
            std::memcpy(out_ptrs_[pids[j]], s + (j * N), N);
            out_ptrs_[pids[j]] += N;
        }
    }

    /// Inline-flush variant — see RadixShuffleColumnFixed::scatter_staged for rationale.
    void scatter_staged(const uint16_t * pids, const uint8_t * positions, const void * src, size_t n) override
    {
        const uint8_t * s = static_cast<const uint8_t *>(src);

        if constexpr (N == 16)
        {
            /// Each slot is 16 bytes. Flush is a scalar 128-byte memcpy.
            for (size_t j = 0; j < n; ++j)
            {
                const uint16_t p = pids[j];
                const uint8_t slot = positions[j];
                uint8_t * dst_slot = staging_ + (static_cast<size_t>(p) * 8 * N) + (static_cast<size_t>(slot) * N);
                std::memcpy(dst_slot, s + (j * N), N);
                if (slot == 7)
                {
                    const uint8_t * src_p = staging_ + (static_cast<size_t>(p) * 8 * N);
                    std::memcpy(out_ptrs_[p], src_p, 8 * N);
                    out_ptrs_[p] += 8 * N;
                }
            }
        }
        else
        {
            auto * typed_staging = reinterpret_cast<FlushT *>(staging_);
            auto ** typed_ptrs = reinterpret_cast<FlushT **>(out_ptrs_);
            for (size_t j = 0; j < n; ++j)
            {
                const uint16_t p = pids[j];
                const uint8_t slot = positions[j];
                uint8_t * dst_slot = staging_ + (static_cast<size_t>(p) * 8 * N) + (static_cast<size_t>(slot) * N);
                std::memcpy(dst_slot, s + (j * N), N);
                if (slot == 7)
                    flush_fn_(typed_staging, typed_ptrs, p);
            }
        }
    }

    [[nodiscard]] uint8_t * stagingBase() { return staging_; }
    [[nodiscard]] uint8_t ** outPtrs() { return out_ptrs_; }
    [[nodiscard]] size_t P() const { return P_; }

private:
    size_t P_;
    uint8_t * staging_;
    uint8_t ** out_ptrs_;
    NTFlushFn<FlushT> flush_fn_;
};

}
