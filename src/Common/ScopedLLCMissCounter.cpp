#include <Common/ScopedLLCMissCounter.h>

#if defined(OS_LINUX)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace DB
{

#if defined(OS_LINUX)

namespace
{

/// Per-thread LL read-miss counter fd. -2 == not yet attempted, -1 == unavailable, >= 0 == open.
thread_local int llc_miss_fd = -2;

int openLLCMissCounter() noexcept
{
    perf_event_attr attr{};
    attr.size = sizeof(attr);
    attr.type = PERF_TYPE_HW_CACHE;
    attr.config = PERF_COUNT_HW_CACHE_LL
        | (PERF_COUNT_HW_CACHE_OP_READ << 8)
        | (static_cast<unsigned>(PERF_COUNT_HW_CACHE_RESULT_MISS) << 16);
    /// Count this thread, in user space only (the probe is user-space compute), on any CPU it runs on.
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    return static_cast<int>(syscall(SYS_perf_event_open, &attr, /*pid=*/0, /*cpu=*/-1, /*group_fd=*/-1, /*flags=*/0UL));
}

int llcMissFd() noexcept
{
    if (llc_miss_fd == -2)
        llc_miss_fd = openLLCMissCounter();
    return llc_miss_fd;
}

UInt64 readCounter(int fd) noexcept
{
    if (fd < 0)
        return 0;
    UInt64 value = 0;
    if (::read(fd, &value, sizeof(value)) != static_cast<ssize_t>(sizeof(value)))
        return 0;
    return value;
}

}

ScopedLLCMissCounter::ScopedLLCMissCounter(ProfileEvents::Event event_) noexcept
    : event(event_), fd(llcMissFd()), start(readCounter(fd))
{
}

ScopedLLCMissCounter::~ScopedLLCMissCounter()
{
    if (fd < 0)
        return;
    const UInt64 end = readCounter(fd);
    if (end > start)
        ProfileEvents::increment(event, end - start);
}

#else

ScopedLLCMissCounter::ScopedLLCMissCounter(ProfileEvents::Event event_) noexcept : event(event_), fd(-1), start(0)
{
}

ScopedLLCMissCounter::~ScopedLLCMissCounter() = default;

#endif

}
