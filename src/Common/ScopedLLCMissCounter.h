#pragma once

#include <Common/ProfileEvents.h>

namespace DB
{

/** Benchmarking instrumentation: counts demand last-level-cache load misses
  * (`PERF_TYPE_HW_CACHE`, `LL | READ | MISS`) incurred on the current thread over the scope's
  * lifetime and adds the delta to a `ProfileEvent` on destruction.
  *
  * A `perf_event` counter fd is opened lazily per thread and reused across scopes; if it cannot be
  * opened (non-Linux, or `perf_event_paranoid` too high to allow a hardware counter) the scope is a
  * silent no-op and the event stays at 0. Reading the counter is two `read` syscalls per scope, which
  * is negligible relative to a cache-heavy probe over a whole block — but this is deliberately a
  * profiling aid (it should be scoped to the hot stage, not wrapped around tiny regions).
  */
class ScopedLLCMissCounter
{
public:
    explicit ScopedLLCMissCounter(ProfileEvents::Event event_) noexcept;
    ~ScopedLLCMissCounter();

    ScopedLLCMissCounter(const ScopedLLCMissCounter &) = delete;
    ScopedLLCMissCounter & operator=(const ScopedLLCMissCounter &) = delete;

private:
    ProfileEvents::Event event;
    int fd;
    UInt64 start;
};

}
