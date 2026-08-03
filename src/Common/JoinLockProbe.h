#pragma once

/// INSTRUMENTATION -- throwaway, never shipped, reverted before delivery.
///
/// Per-site lock and atomic accounting for the hash-join lock analysis (Unit 2:
/// gates G0.2, G2.1, G2.2). Records, per site: successful acquisitions, `try_lock`
/// failures, blocking acquisitions, and a log2 histogram of hold times, so the
/// report can state a measured distribution rather than a mean or an estimate.
///
/// Design constraints this satisfies:
///   * no added synchronisation on the hot path -- all counters are thread-local,
///     summed only when the totals are dumped;
///   * a raw counter read (`cntvct_el0`) rather than `clock_gettime`, because the
///     hold times being measured are of the same order as a `clock_gettime` call;
///   * no allocation and no branching beyond an already-hot thread-local pointer.
///
/// It still perturbs timing, so no binary containing it is ever used for a timing
/// verdict. Which binary produced which number is recorded in the worklog.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace JoinLockProbe
{

enum Site : unsigned
{
    /// unified_hash: per-bucket BucketLock, taken with try_lock in the drain loop
    UNI_BUCKET_TRY = 0,
    /// unified_hash: per-bucket BucketLock, taken blocking on the no-progress path
    UNI_BUCKET_BLOCK,
    /// unified_hash: per-bucket BucketLock, the empty-block special case
    UNI_BUCKET_EMPTY,
    /// unified_hash: blocks_mutex, block registration and friends
    UNI_BLOCKS_MUTEX,
    /// parallel_hash: per-slot mutex, try_lock in the dispatch drain loop
    PAR_SLOT_TRY,
    /// StoredColumnsIndex mutex -- shared infrastructure, all three implementations
    SCI_ADD,
    SCI_RESOLVE,
    SITE_COUNT
};

inline const char * siteName(unsigned s)
{
    static const char * names[] = {
        "UNI_BUCKET_TRY", "UNI_BUCKET_BLOCK", "UNI_BUCKET_EMPTY", "UNI_BLOCKS_MUTEX",
        "PAR_SLOT_TRY", "SCI_ADD", "SCI_RESOLVE"};
    return s < SITE_COUNT ? names[s] : "?";
}

enum Counter : unsigned
{
    /// per matched row, both trees
    ATOM_SET_USED = 0,
    ATOM_SET_USED_ONCE,
    ATOM_SET_USED_ONCE_CAS_FAIL,
    /// unified_hash: bucket_bytes.fetch_add, once per locked bucket per block
    ATOM_BUCKET_BYTES,
    COUNTER_COUNT
};

inline const char * counterName(unsigned c)
{
    static const char * names[] = {
        "ATOM_SET_USED", "ATOM_SET_USED_ONCE", "ATOM_SET_USED_ONCE_CAS_FAIL",
        "ATOM_BUCKET_BYTES"};
    return c < COUNTER_COUNT ? names[c] : "?";
}

static constexpr unsigned HIST_BUCKETS = 28;   /// log2 of ticks, 0 .. 2^27

struct SiteStat
{
    uint64_t acquisitions = 0;
    uint64_t try_failures = 0;
    uint64_t ticks_total = 0;
    uint64_t hist[HIST_BUCKETS] = {};
};

struct ThreadStats
{
    SiteStat sites[SITE_COUNT];
    uint64_t counters[COUNTER_COUNT] = {};
};

/// Registry of every thread that has touched a probe. Threads outlive individual
/// joins, so stats are cumulative for the process and the harness takes differences.
struct Registry
{
    std::mutex mutex;
    std::vector<ThreadStats *> threads;

    static Registry & get()
    {
        static Registry r;
        return r;
    }
};

inline ThreadStats * makeThreadStats()
{
    auto * s = new ThreadStats();
    auto & r = Registry::get();
    std::lock_guard lock(r.mutex);
    r.threads.push_back(s);
    return s;
}

inline ThreadStats & tls()
{
    static thread_local ThreadStats * s = makeThreadStats();
    return *s;
}

inline uint64_t ticks()
{
#if defined(__aarch64__)
    uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return __builtin_readcyclecounter();
#endif
}

inline uint64_t tickHz()
{
#if defined(__aarch64__)
    uint64_t v;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
#else
    return 1;
#endif
}

inline unsigned log2Bucket(uint64_t v)
{
    if (v == 0)
        return 0;
    unsigned b = 63u - static_cast<unsigned>(__builtin_clzll(v));
    return b < HIST_BUCKETS ? b : HIST_BUCKETS - 1;
}

inline void countTryFailure(unsigned site) { ++tls().sites[site].try_failures; }

/// Record one duration sample directly. Used for blocked-wait time, which is not a
/// hold and so cannot be measured by HoldTimer.
inline void record(unsigned site, uint64_t dt)
{
    auto & s = tls().sites[site];
    ++s.acquisitions;
    s.ticks_total += dt;
    ++s.hist[log2Bucket(dt)];
}

inline void bump(unsigned counter, uint64_t n = 1) { tls().counters[counter] += n; }

/// RAII: start the clock when the lock is held, stop when it is released.
struct HoldTimer
{
    unsigned site;
    uint64_t t0;

    explicit HoldTimer(unsigned site_) : site(site_), t0(ticks()) {}

    ~HoldTimer()
    {
        const uint64_t dt = ticks() - t0;
        auto & s = tls().sites[site];
        ++s.acquisitions;
        s.ticks_total += dt;
        ++s.hist[log2Bucket(dt)];
    }
};

/// Cumulative process totals, appended as one JSON line. Called from the join
/// destructors; the harness diffs the last line before and after a query.
inline void dump(const char * tag)
{
    const char * path = std::getenv("UHJ_LOCKPROBE_OUT");
    if (!path)
        return;

    ThreadStats total;
    {
        auto & r = Registry::get();
        std::lock_guard lock(r.mutex);
        for (auto * t : r.threads)
        {
            for (unsigned i = 0; i < SITE_COUNT; ++i)
            {
                total.sites[i].acquisitions += t->sites[i].acquisitions;
                total.sites[i].try_failures += t->sites[i].try_failures;
                total.sites[i].ticks_total += t->sites[i].ticks_total;
                for (unsigned b = 0; b < HIST_BUCKETS; ++b)
                    total.sites[i].hist[b] += t->sites[i].hist[b];
            }
            for (unsigned c = 0; c < COUNTER_COUNT; ++c)
                total.counters[c] += t->counters[c];
        }
    }

    static std::mutex file_mutex;
    std::lock_guard lock(file_mutex);
    FILE * f = std::fopen(path, "a");
    if (!f)
        return;
    std::fprintf(f, "{\"tag\":\"%s\",\"tick_hz\":%llu,\"sites\":{", tag,
                 static_cast<unsigned long long>(tickHz()));
    for (unsigned i = 0; i < SITE_COUNT; ++i)
    {
        std::fprintf(f, "%s\"%s\":{\"acq\":%llu,\"tryfail\":%llu,\"ticks\":%llu,\"hist\":[",
                     i ? "," : "", siteName(i),
                     static_cast<unsigned long long>(total.sites[i].acquisitions),
                     static_cast<unsigned long long>(total.sites[i].try_failures),
                     static_cast<unsigned long long>(total.sites[i].ticks_total));
        for (unsigned b = 0; b < HIST_BUCKETS; ++b)
            std::fprintf(f, "%s%llu", b ? "," : "",
                         static_cast<unsigned long long>(total.sites[i].hist[b]));
        std::fprintf(f, "]}");
    }
    std::fprintf(f, "},\"counters\":{");
    for (unsigned c = 0; c < COUNTER_COUNT; ++c)
        std::fprintf(f, "%s\"%s\":%llu", c ? "," : "", counterName(c),
                     static_cast<unsigned long long>(total.counters[c]));
    std::fprintf(f, "}}\n");
    std::fclose(f);
}

}
