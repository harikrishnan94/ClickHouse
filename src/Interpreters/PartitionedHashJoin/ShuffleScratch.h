#pragma once

#include <base/types.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace DB
{

/// Per-thread mutable scratch arrays, reused across every batch call.
/// All arrays are heap-allocated at ShuffleScratch construction and never reallocated
/// during shuffle (sized for worst case: P_MAX = 1024, batch_MAX = 65535).
///
/// Type rationale (all values fit comfortably in their lane widths):
///   • pids[j]      ∈ [0, P) with P ≤ 1024 (10 bits) → uint16_t
///   • hist[p]      ∈ [0, batch_sz] with batch_sz ≤ 65535 → uint16_t (exact fit)
///   • positions[j] ∈ [0, 7]  → uint8_t
///   • swwc_cnt[p]  ∈ [0, 7]  → uint8_t
///
/// SWWC flushes happen inline inside scatter_staged() when positions[j] == 7,
/// so there is no separate flush_ps[] array.
struct ShuffleScratch
{
    static constexpr size_t kPMax = 1024;
    static constexpr size_t kBatchMax = 65535;

    std::vector<uint16_t> pids; /// partition id per row, size = kBatchMax
    std::vector<uint16_t> hist; /// histogram per partition, size = kPMax
    std::vector<uint8_t> positions; /// SWWC: staging slot per row, size = kBatchMax
    std::vector<uint8_t> swwc_cnt; /// SWWC: current staging fill per partition, size = kPMax

    explicit ShuffleScratch()
        : pids(kBatchMax, 0)
        , hist(kPMax, 0)
        , positions(kBatchMax, 0)
        , swwc_cnt(kPMax, 0)
    {
    }

    ~ShuffleScratch() = default;
    ShuffleScratch(const ShuffleScratch &) = delete;
    ShuffleScratch & operator=(const ShuffleScratch &) = delete;
    ShuffleScratch(ShuffleScratch &&) = delete;
    ShuffleScratch & operator=(ShuffleScratch &&) = delete;

    void clearHist(size_t P) { std::memset(hist.data(), 0, P * sizeof(uint16_t)); }
};

}
