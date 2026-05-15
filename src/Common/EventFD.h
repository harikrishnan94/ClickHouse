#pragma once

#if defined(OS_LINUX)

#include <cstdint>


namespace DB
{

struct EventFD
{
    /// Default: non-blocking + close-on-exec. This is the spec-correct default
    /// for async/poll usage (the only intended pattern in new code: register
    /// the fd with epoll, react to readiness via the executor) and avoids
    /// fd leaks across exec. Existing call sites that genuinely need blocking
    /// `read()` (the only blocking API on this struct — `write()` blocks only
    /// when the eventfd counter is saturated, which is itself a degenerate
    /// case) must opt in explicitly via the `flags` overload below.
    EventFD();
    /// Explicit-flags overload for callers that need a specific eventfd flag
    /// set (e.g. `0` for blocking semantics expected by some pre-existing
    /// users, or just `EFD_CLOEXEC` for a blocking-but-exec-safe fd). Flags
    /// are passed through verbatim to `eventfd(2)`.
    explicit EventFD(int flags);
    ~EventFD();

    /// `read()` may return 0 with no syscall side effects on non-blocking
    /// eventfds whose counter is empty (EAGAIN/EWOULDBLOCK is mapped to a
    /// silent "no value" by the implementation). Callers that need to
    /// distinguish "drained" from "nothing to drain" must use a blocking fd
    /// (constructor flags = 0).
    uint64_t read() const;
    bool write(uint64_t increase = 1) const;

    int fd = -1;
};

}

#else

namespace DB
{

struct EventFD
{
};

}

#endif
