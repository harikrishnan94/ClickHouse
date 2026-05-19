/// hashprobe-bench/instrumentation/hw_counters.cpp
///
/// HW counter group implementation: perf_event_open group +
/// user-space rdpmc via mmap'd perf_event_mmap_page.
///
/// Events are encoded using hardcoded PERF_TYPE_HARDWARE / PERF_TYPE_HW_CACHE
/// constants (no libpfm4 dependency).
///
/// Run-time requirements for non-zero counter reads
///   /proc/sys/kernel/perf_event_paranoid <= 0,  or  CAP_PERFMON capability.
///   (Open() returns false / fail-soft otherwise.)

#include "hw_counters.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifdef __linux__
#  include <asm/unistd.h>
#  include <linux/perf_event.h>

// perf_event_open is not always wrapped by glibc
static long perf_event_open_syscall(
    struct perf_event_attr * hw_event,
    pid_t pid,
    int   cpu,
    int   group_fd,
    unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

// ── rdpmc helpers ─────────────────────────────────────────────────────────────

#  if defined(__x86_64__) || defined(__i386__)
static inline uint64_t hw_rdpmc(uint32_t idx)
{
    uint32_t lo, hi;
    __asm__ volatile("rdpmc" : "=a"(lo), "=d"(hi) : "c"(idx));
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

/// Read one counter from its mmap page using rdpmc.
///
/// The kernel stores counter state in the mmap page as:
///   offset = virtual_count_at_last_reset - hardware_count_at_last_reset
/// so that: rdpmc() + offset  =  virtual_count_current
///
/// Because hardware counters are width-bit registers (typically 48 on x86), we
/// do all arithmetic in unsigned modular width-bit space to avoid sign overflow
/// when the hardware counter passes the 2^(width-1) boundary.  Specifically:
///
///   count = (rdpmc_raw & mask) + (offset & mask)   [mod 2^width, unsigned]
///
/// where mask = (1<<width) - 1.  This matches the kernel's own implementation
/// in tools/lib/perf and is correct regardless of where the hardware counter
/// sits in its range.
///
/// Returns the accumulated count since last reset, or 0 if rdpmc is unavailable.
/// Sets *available to false when rdpmc cannot be used for this counter.
static uint64_t read_mmap_counter(const volatile struct perf_event_mmap_page * pc,
                                  bool & available)
{
    if (!pc)
    {
        available = false;
        return 0;
    }

    uint32_t seq;
    uint64_t count = 0;

    do
    {
        seq = __atomic_load_n(&pc->lock, __ATOMIC_SEQ_CST);
        if (seq & 1u)
            continue;  // write-side lock held; spin

        __atomic_thread_fence(__ATOMIC_SEQ_CST);

        if (!pc->cap_user_rdpmc || pc->index == 0)
        {
            // rdpmc not available for this counter (not enabled, or no HW PMC assigned)
            available = false;
            return 0;
        }

        // Unsigned modular arithmetic in pmc_width-bit space.
        // This is correct even when the hardware counter crosses the 2^(width-1)
        // boundary (which would otherwise cause spurious sign-extension artifacts).
        const uint32_t idx   = pc->index - 1;
        const uint32_t width = pc->pmc_width;
        const uint64_t mask  = (width > 0 && width < 64)
                                ? ((1ULL << width) - 1ULL)
                                : UINT64_MAX;

        const uint64_t raw_masked = hw_rdpmc(idx) & mask;
        const uint64_t off_masked = static_cast<uint64_t>(pc->offset) & mask;
        count = (raw_masked + off_masked) & mask;

        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    } while (__atomic_load_n(&pc->lock, __ATOMIC_SEQ_CST) != seq);

    // Multiplexing correction: scale by time_enabled / time_running.
    // Only applied when the counter was multiplexed (time_running < time_enabled).
    if (pc->time_running && pc->time_enabled != pc->time_running)
    {
        count = static_cast<uint64_t>(
            static_cast<double>(count)
            * static_cast<double>(pc->time_enabled)
            / static_cast<double>(pc->time_running));
    }

    available = true;
    return count;
}
#  endif // x86_64 / i386

#endif // __linux__

namespace DB::HashProbeBench
{

bool HwCounters::open()
{
    close();  // ensure clean state

#ifndef __linux__
    return false;
#else

    struct EventSpec
    {
        uint32_t type;
        uint64_t config;
    };

    static const EventSpec kEvents[kNumCounters] = {
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
        // DTLB load misses via PERF_TYPE_HW_CACHE encoding
        {PERF_TYPE_HW_CACHE,
         static_cast<uint64_t>(PERF_COUNT_HW_CACHE_DTLB)
             | (static_cast<uint64_t>(PERF_COUNT_HW_CACHE_OP_READ)     << 8)
             | (static_cast<uint64_t>(PERF_COUNT_HW_CACHE_RESULT_MISS) << 16)},
        // Total branch instructions — denominator for branch miss rate
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
        // Generic cache references — used as LLC load denominator.
        // On most x86 maps to LLC accesses; may include prefetch/coherency traffic.
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
        // DTLB load accesses — denominator for DTLB miss rate.
        // On Intel SKL/SPR maps to all retired loads, not first-level DTLB accesses.
        {PERF_TYPE_HW_CACHE,
         static_cast<uint64_t>(PERF_COUNT_HW_CACHE_DTLB)
             | (static_cast<uint64_t>(PERF_COUNT_HW_CACHE_OP_READ)       << 8)
             | (static_cast<uint64_t>(PERF_COUNT_HW_CACHE_RESULT_ACCESS) << 16)},
    };

    for (int i = 0; i < kNumCounters; ++i)
    {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(pe));
        pe.type           = kEvents[i].type;
        pe.size           = sizeof(pe);
        pe.config         = kEvents[i].config;
        pe.disabled       = (i == 0) ? 1 : 0;
        pe.exclude_kernel = 1;
        pe.exclude_hv     = 1;
        pe.inherit        = 0;

        const int group = (i == 0) ? -1 : fds_[0];
        long fd = perf_event_open_syscall(&pe, 0 /*current thread*/, -1 /*any cpu*/, group, 0);
        if (fd < 0)
        {
            close();
            return false;
        }
        fds_[i] = static_cast<int>(fd);
    }

    leader_fd_ = fds_[0];

    // ── mmap each fd for user-space rdpmc reads ───────────────────────────
    //
    // A single page (PAGE_SIZE) mmap on each fd exposes its perf_event_mmap_page,
    // which contains the rdpmc index, offset, and seqlock.
    // Failure to mmap is non-fatal: we fall back to ioctl(DISABLE)+::read().
    const long page_size = sysconf(_SC_PAGESIZE);
    rdpmc_ok_ = true;
    for (int i = 0; i < kNumCounters; ++i)
    {
        void * p = mmap(nullptr, static_cast<size_t>(page_size),
                        PROT_READ, MAP_SHARED, fds_[i], 0);
        if (p == MAP_FAILED)
        {
            mmap_pages_[i] = nullptr;
            rdpmc_ok_      = false;
        }
        else
        {
            mmap_pages_[i] = p;
        }
    }

    return true;
#endif // __linux__
}

void HwCounters::start()
{
#ifdef __linux__
    if (leader_fd_ < 0)
        return;
    ioctl(leader_fd_, PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
    ioctl(leader_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
#endif
}

void HwCounters::read(
    uint64_t & cycles,
    uint64_t & instructions,
    uint64_t & llc_miss,
    uint64_t & branch_miss,
    uint64_t & dtlb_miss,
    uint64_t & branches,
    uint64_t & llc_load,
    uint64_t & dtlb_load)
{
    cycles = instructions = llc_miss = branch_miss = dtlb_miss
           = branches = llc_load = dtlb_load = 0;

#ifdef __linux__
    if (leader_fd_ < 0)
        return;

#  if defined(__x86_64__) || defined(__i386__)
    // ── User-space rdpmc path (no syscall) ───────────────────────────────
    if (rdpmc_ok_)
    {
        uint64_t vals[kNumCounters] = {};
        bool all_ok = true;
        for (int i = 0; i < kNumCounters; ++i)
        {
            bool ok = false;
            vals[i] = read_mmap_counter(
                static_cast<volatile struct perf_event_mmap_page *>(mmap_pages_[i]),
                ok);
            if (!ok)
            {
                all_ok = false;
                break;
            }
        }
        if (all_ok)
        {
            cycles       = vals[0];
            instructions = vals[1];
            llc_miss     = vals[2];
            branch_miss  = vals[3];
            dtlb_miss    = vals[4];
            branches     = vals[5];
            llc_load     = vals[6];
            dtlb_load    = vals[7];
            return;
        }
        // rdpmc unavailable at runtime: fall through to ioctl path
    }
#  endif // x86_64 / i386

    // ── Fallback: ioctl(DISABLE) + individual ::read() per fd ────────────
    ioctl(leader_fd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

    uint64_t vals[kNumCounters] = {};
    for (int i = 0; i < kNumCounters; ++i)
    {
        if (fds_[i] >= 0)
            ::read(fds_[i], &vals[i], sizeof(vals[i]));
    }

    cycles       = vals[0];
    instructions = vals[1];
    llc_miss     = vals[2];
    branch_miss  = vals[3];
    dtlb_miss    = vals[4];
    branches     = vals[5];
    llc_load     = vals[6];
    dtlb_load    = vals[7];
#endif // __linux__
}

bool HwCounters::snapshot(
    uint64_t & cycles,
    uint64_t & instructions,
    uint64_t & llc_miss,
    uint64_t & branch_miss,
    uint64_t & dtlb_miss,
    uint64_t & branches,
    uint64_t & llc_load,
    uint64_t & dtlb_load) const
{
    cycles = instructions = llc_miss = branch_miss = dtlb_miss
           = branches = llc_load = dtlb_load = 0;

#ifdef __linux__
    if (leader_fd_ < 0)
        return false;

#  if defined(__x86_64__) || defined(__i386__)
    if (!rdpmc_ok_)
        return false;

    uint64_t vals[kNumCounters] = {};
    for (int i = 0; i < kNumCounters; ++i)
    {
        bool ok = false;
        vals[i] = read_mmap_counter(
            static_cast<volatile struct perf_event_mmap_page *>(mmap_pages_[i]),
            ok);
        if (!ok)
            return false;
    }

    cycles       = vals[0];
    instructions = vals[1];
    llc_miss     = vals[2];
    branch_miss  = vals[3];
    dtlb_miss    = vals[4];
    branches     = vals[5];
    llc_load     = vals[6];
    dtlb_load    = vals[7];
    return true;
#  else
    return false;
#  endif
#else
    return false;
#endif
}

void HwCounters::close()
{
    rdpmc_ok_ = false;

#ifdef __linux__
    const long page_size = sysconf(_SC_PAGESIZE);
    for (int i = 0; i < kNumCounters; ++i)
    {
        if (mmap_pages_[i] != nullptr)
        {
            munmap(mmap_pages_[i], static_cast<size_t>(page_size));
            mmap_pages_[i] = nullptr;
        }
        if (fds_[i] >= 0)
        {
            ::close(fds_[i]);
            fds_[i] = -1;
        }
    }
#else
    for (int i = 0; i < kNumCounters; ++i)
    {
        mmap_pages_[i] = nullptr;
        fds_[i] = -1;
    }
#endif
    leader_fd_ = -1;
}

} // namespace DB::HashProbeBench
