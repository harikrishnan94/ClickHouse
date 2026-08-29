#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>

#if defined(__x86_64__)
#include <nmmintrin.h>
#include <x86intrin.h>
#elif defined(__aarch64__)
#include <arm_acle.h>
#endif

inline constexpr size_t kCacheLine = 64;

inline size_t ncpus()
{
    static const size_t n = []
    {
        long v = sysconf(_SC_NPROCESSORS_ONLN);
        return v > 0 ? static_cast<size_t>(v) : 1;
    }();
    return n;
}

inline void pin_thread(size_t tid)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<int>(tid % ncpus()), &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        throw std::runtime_error("pthread_setaffinity_np failed");
}

inline uint64_t ns_now()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull + static_cast<uint64_t>(ts.tv_nsec);
}

#if defined(__aarch64__)
inline uint64_t cycles_now()
{
    uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

inline uint64_t cycles_per_sec()
{
    uint64_t v;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}
#elif defined(__x86_64__)
inline uint64_t cycles_now()
{
    unsigned aux;
    return __rdtscp(&aux);
}

inline uint64_t cycles_per_sec()
{
    static const uint64_t hz = []
    {
        uint64_t t0 = ns_now();
        uint64_t c0 = cycles_now();
        while (ns_now() - t0 < 50'000'000ull)
        {
        }
        uint64_t t1 = ns_now();
        uint64_t c1 = cycles_now();
        return static_cast<uint64_t>(static_cast<double>(c1 - c0) * 1e9 / static_cast<double>(t1 - t0));
    }();
    return hz;
}
#else
#error "Need x86-64 or aarch64"
#endif

inline uint64_t cycles_to_ns(uint64_t cyc)
{
    return static_cast<uint64_t>(static_cast<double>(cyc) * 1e9 / static_cast<double>(cycles_per_sec()));
}

inline void cpu_relax()
{
#if defined(__aarch64__)
    asm volatile("yield");
#elif defined(__x86_64__)
    _mm_pause();
#endif
}

inline uint32_t ceil_log2_u64(uint64_t x)
{
    if (x <= 1)
        return 0;
    return 64u - static_cast<uint32_t>(__builtin_clzll(x - 1));
}

inline uint64_t next_pow2(uint64_t x)
{
    if (x <= 1)
        return 1;
    return 1ull << ceil_log2_u64(x);
}

inline void * map_anon(size_t bytes)
{
    if (bytes == 0)
        return nullptr;
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t rounded = (bytes + page - 1) & ~(page - 1);
    void * p = mmap(nullptr, rounded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED)
        throw std::bad_alloc();
    madvise(p, rounded, MADV_HUGEPAGE);
    return p;
}

inline size_t map_bytes_rounded(size_t bytes)
{
    if (bytes == 0)
        return 0;
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    return (bytes + page - 1) & ~(page - 1);
}

inline void unmap_anon(void * p, size_t bytes)
{
    if (!p || bytes == 0)
        return;
    munmap(p, map_bytes_rounded(bytes));
}

/// First-touch every page with a write, so later timed reads do not hit the shared zero page.
inline void prefault_write(void * p, size_t bytes)
{
    if (!p || bytes == 0)
        return;
    auto * c = static_cast<volatile char *>(p);
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    for (size_t i = 0; i < bytes; i += page)
        c[i] = 0;
    c[bytes - 1] = 0;
}

#if defined(__x86_64__)
inline uint32_t crc32c_u64(uint32_t crc, uint64_t key)
{
    return static_cast<uint32_t>(_mm_crc32_u64(crc, key));
}
#elif defined(__aarch64__)
inline uint32_t crc32c_u64(uint32_t crc, uint64_t key)
{
    return __crc32cd(crc, key);
}
#endif

template <typename F>
void parallel_for(size_t threads, bool pin, F && fn)
{
    if (threads == 0)
        throw std::runtime_error("threads must be > 0");
    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    for (size_t i = 1; i < threads; ++i)
    {
        workers.emplace_back(
            [&fn, pin, i]
            {
                if (pin)
                    pin_thread(i);
                fn(i);
            });
    }
    if (pin)
        pin_thread(0);
    fn(0);
    for (auto & t : workers)
        t.join();
}

inline std::string host_summary()
{
#if defined(__aarch64__)
    const char * arch = "aarch64";
#elif defined(__x86_64__)
    const char * arch = "x86-64";
#endif
    return std::string(arch) + "  cpus=" + std::to_string(ncpus());
}
