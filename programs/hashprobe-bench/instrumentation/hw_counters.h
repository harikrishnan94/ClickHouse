#pragma once

/// hashprobe-bench/instrumentation/hw_counters.h
///
/// Hardware performance counter group (H4).
///
/// Opens a group of 8 PMCs:
///
///   1. CPU cycles
///   2. Instructions retired
///   3. LLC load misses       (PERF_COUNT_HW_CACHE_MISSES)
///   4. Branch misses         (PERF_COUNT_HW_BRANCH_MISSES)
///   5. DTLB load misses      (PERF_TYPE_HW_CACHE / DTLB/READ/MISS)
///   6. Branch instructions   (PERF_COUNT_HW_BRANCH_INSTRUCTIONS) — denominator for miss rate
///   7. Cache references      (PERF_COUNT_HW_CACHE_REFERENCES) — generic LLC-access denominator;
///                              on most x86 maps to LLC accesses but may include prefetch/
///                              coherency traffic; used as hw_llc_load.
///   8. DTLB load accesses    (PERF_TYPE_HW_CACHE / DTLB/READ/ACCESS) — denominator for DTLB
///                              miss rate. On Intel SKL/SPR maps to all retired loads (not
///                              first-level DTLB accesses), so the ratio approximates
///                              page-walks-per-load rather than a true DTLB miss rate; AMD
///                              differs. Used as hw_dtlb_load.
///
/// Event encoding path:
///   Uses hardcoded PERF_TYPE_HARDWARE / PERF_TYPE_HW_CACHE constants
///   (no libpfm4 dependency).
///
/// Counter read path:
///   User-space rdpmc via the leader's mmap'd perf_event_mmap_page when
///   cap_user_rdpmc is set and index != 0.  Falls back to
///   ioctl(DISABLE, GROUP) + ::read() otherwise.
///   All 8 fds are individually mmap'd so each counter's index is available.
///
/// Fail-soft contract: if perf_event_open or mmap fails (no CAP_PERFMON,
/// perf_event_paranoid too high, etc.) open() returns false and all read()
/// values are zero — never throws or calls std::exit.
///
/// Requirements:
///   /proc/sys/kernel/perf_event_paranoid <= 0, or CAP_PERFMON capability.
///
/// Typical usage:
///   HwCounters ctrs;
///   if (!ctrs.open()) { /* HW counting unavailable */ }
///   ctrs.start();
///   /* ... workload ... */
///   uint64_t cy, ins, llc, br, dtlb, branches, llc_load, dtlb_load;
///   ctrs.read(cy, ins, llc, br, dtlb, branches, llc_load, dtlb_load);
///   double ipc = HwCounters::computeIpc(ins, cy);
///   ctrs.close();

#include <cstdint>

namespace DB::HashProbeBench
{

class HwCounters
{
public:
    HwCounters() = default;
    ~HwCounters() { close(); }

    HwCounters(const HwCounters &) = delete;
    HwCounters & operator=(const HwCounters &) = delete;

    /// Open the PMC group.  Returns true on success, false if unavailable.
    /// Safe to call multiple times; re-opens if already closed.
    bool open();

    /// Enable counting (reset+enable the group via the leader fd).
    void start();

    /// Snapshot current accumulated counts.
    /// Uses user-space rdpmc via mmap'd perf_event_mmap_page when available;
    /// falls back to ioctl(DISABLE)+::read() otherwise.
    /// All outputs are set to 0 if the group was never opened successfully.
    void read(
        uint64_t & cycles,
        uint64_t & instructions,
        uint64_t & llc_miss,
        uint64_t & branch_miss,
        uint64_t & dtlb_miss,
        uint64_t & branches,
        uint64_t & llc_load,
        uint64_t & dtlb_load);

    /// Snapshot without disabling the counter group. Returns false when the
    /// low-overhead rdpmc path is unavailable; callers can still use read() for
    /// the final whole-block fallback sample.
    bool snapshot(
        uint64_t & cycles,
        uint64_t & instructions,
        uint64_t & llc_miss,
        uint64_t & branch_miss,
        uint64_t & dtlb_miss,
        uint64_t & branches,
        uint64_t & llc_load,
        uint64_t & dtlb_load) const;

    /// Disable counting and close all fds / unmap all pages.
    void close();

    /// Returns true iff the group was opened successfully.
    bool isAvailable() const { return leader_fd_ >= 0; }

    /// Convenience: compute IPC from a read() result.
    static double computeIpc(uint64_t instructions, uint64_t cycles)
    {
        return (cycles > 0) ? static_cast<double>(instructions) / static_cast<double>(cycles) : 0.0;
    }

private:
    // File descriptors: [0]=cycles, [1]=instructions, [2]=llc_miss,
    //                   [3]=branch_miss, [4]=dtlb_miss,
    //                   [5]=branches, [6]=llc_load, [7]=dtlb_load
    static constexpr int kNumCounters = 8;
    int fds_[kNumCounters] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int leader_fd_ = -1; // == fds_[0] when open

    // mmap'd perf_event_mmap_page for each counter (NULL when not mapped).
    // Used for user-space rdpmc reads.
    void * mmap_pages_[kNumCounters] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    bool rdpmc_ok_ = false; ///< true when all 8 pages are mapped
};

} // namespace DB::HashProbeBench
