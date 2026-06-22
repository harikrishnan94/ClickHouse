#pragma once

/// SCM_RIGHTS fd-passing for the SHM block-stream control plane.
///
/// Per `shm-block-stream.md` §Notification contract, the readiness eventfd
/// is created by the producer and passed to the consumer over a Unix-domain
/// socket whose path is derived by convention from the SHM object name. The
/// fixed-size handshake region does not carry the path; both sides compute
/// it via `controlSocketPathForShmName`.
///
/// The consumer keeps the connection fd open after receiving the eventfd so
/// it can detect producer death via POLLHUP on that fd
/// (`pollable-shm-source.md` Producer-side preconditions enumerated #25).
///
/// Consumer-side connect/recv failures and any malformed cmsg surface as
/// `SHM_ATTACH_FAILED` per `pollable-shm-source.md` Attach-time observable
/// failures row "readiness-fd locator unresolvable". Producer-side
/// bind/listen failures raise `ErrnoException(CANNOT_OPEN_FILE)`.

#include <base/types.h>

#include <atomic>
#include <functional>

namespace DB
{

/// Convention: `/tmp/clickhouse_shm_<sanitized_name>.sock`. Leading '/' on
/// `shm_name` is stripped; any embedded '/' becomes '_'.
String controlSocketPathForShmName(const String & shm_name);

class ControlSocketClient
{
public:
    /// Connect to `socket_path`, receive exactly one eventfd via SCM_RIGHTS,
    /// return it. The live socket connection is handed back via `out_conn_fd`
    /// for POLLHUP monitoring; the caller owns and must close both fds.
    /// `out_conn_fd` is set to -1 on every throw path.
    ///
    /// The optional cancellation callback is checked during short nonblocking
    /// poll slices so attach cannot sit in an uninterruptible recvmsg wait.
    ///
    /// Throws `SHM_ATTACH_FAILED` for: connect failure (ENOENT, ECONNREFUSED,
    /// EACCES, timeout), recvmsg failure/timeout, peer-close before send,
    /// nonblocking setup failure on the received fd, or a cmsg that is
    /// non-SCM_RIGHTS or carries 0/multiple fds.
    static int connectAndReceiveEventFd(
        const String & socket_path,
        int & out_conn_fd,
        const std::function<bool()> & is_cancelled = {});
};

class ControlSocketServer
{
public:
    /// Bind + listen on `socket_path_`. Stale socket files are unlinked
    /// first so producer restart after a crash works. Throws
    /// `ErrnoException(CANNOT_OPEN_FILE)` on socket/bind/listen failure.
    explicit ControlSocketServer(const String & socket_path_);

    ~ControlSocketServer();

    ControlSocketServer(const ControlSocketServer &) = delete;
    ControlSocketServer & operator=(const ControlSocketServer &) = delete;

    /// Block until a consumer connects, return accepted fd (caller-owned).
    /// Returns -1 if `shutdown()` is or was called concurrently. Other
    /// accept errors propagate as `ErrnoException(CANNOT_OPEN_FILE)`.
    int accept();

    /// Send `eventfd_to_pass` on `conn_fd` via SCM_RIGHTS with a 1-byte
    /// dummy payload (kernel requires non-empty iov). Sent with
    /// `MSG_NOSIGNAL`. Throws `ErrnoException(CANNOT_OPEN_FILE)` on sendmsg
    /// failure.
    void sendEventFd(int conn_fd, int eventfd_to_pass);

    /// Idempotent. Sets the shut-down flag, closes the listen fd (any
    /// concurrently-blocked accept() unblocks with EBADF), unlinks the path.
    /// Also invoked from the destructor.
    void shutdown();

private:
    String socket_path;
    int listen_fd;
    std::atomic<bool> is_shutdown{false};
};

}
