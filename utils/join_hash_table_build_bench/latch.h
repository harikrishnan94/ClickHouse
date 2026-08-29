#pragma once

#include "platform.h"

#include <atomic>
#include <cstdint>
#include <mutex>

/// One-byte test-and-test-and-set. Isolated to the ClickHouse contender.
struct alignas(kCacheLine) SpinLatch
{
    bool try_lock()
    {
        if (flag.load(std::memory_order_relaxed) != 0)
            return false;
        uint8_t expected = 0;
        return flag.compare_exchange_strong(expected, 1, std::memory_order_acquire, std::memory_order_relaxed);
    }

    void lock()
    {
        for (;;)
        {
            while (flag.load(std::memory_order_relaxed) != 0)
                cpu_relax();
            uint8_t expected = 0;
            if (flag.compare_exchange_weak(expected, 1, std::memory_order_acquire, std::memory_order_relaxed))
                return;
        }
    }

    void unlock() { flag.store(0, std::memory_order_release); }

private:
    std::atomic<uint8_t> flag{0};
};

struct alignas(kCacheLine) MutexLatch
{
    bool try_lock() { return mutex.try_lock(); }
    void lock() { mutex.lock(); }
    void unlock() { mutex.unlock(); }

private:
    std::mutex mutex;
};

/// Cycle-accounted acquire. Failed try_lock counts; unlock does not.
template <typename Latch>
struct TimedLatch
{
    Latch * latch = nullptr;
    uint64_t * cycles = nullptr;

    bool try_lock()
    {
        const uint64_t t0 = cycles_now();
        const bool ok = latch->try_lock();
        *cycles += cycles_now() - t0;
        return ok;
    }

    void lock()
    {
        const uint64_t t0 = cycles_now();
        latch->lock();
        *cycles += cycles_now() - t0;
    }

    void unlock() { latch->unlock(); }
};
