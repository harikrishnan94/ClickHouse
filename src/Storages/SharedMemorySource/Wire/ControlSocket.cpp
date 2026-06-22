#if defined(OS_LINUX)

#    include <Storages/SharedMemorySource/Wire/ControlSocket.h>

#    include <Common/ErrnoException.h>
#    include <Common/Exception.h>

#    include <fcntl.h>
#    include <poll.h>
#    include <unistd.h>
#    include <sys/socket.h>
#    include <sys/time.h>
#    include <sys/un.h>

#    include <algorithm>
#    include <cerrno>
#    include <chrono>
#    include <cstdint>
#    include <cstring>

namespace DB
{

namespace ErrorCodes
{
extern const int SHM_ATTACH_FAILED;
extern const int CANNOT_OPEN_FILE;
}

namespace
{
constexpr int CONTROL_SOCKET_LISTEN_BACKLOG = 16;

/// Overall attach budget for a producer that bound the socket but never sends
/// the readiness fd. The wait is split into short nonblocking poll slices so
/// cancellation is still observed promptly.
constexpr int CONTROL_SOCKET_ATTACH_TIMEOUT_MS = 5000;
constexpr int CONTROL_SOCKET_POLL_SLICE_MS = 100;

/// > 1 so a producer that mistakenly sends 2-4 fds is detected by an
/// explicit fd-count check; extras get closed instead of leaked via the
/// MSG_CTRUNC-with-installed-fds Linux corner (`scm_detach_fds`).
constexpr size_t CONTROL_SOCKET_MAX_RECV_FDS = 4;

/// Move-only RAII close-on-scope helper used on the error paths.
struct FdGuard
{
    int fd;
    explicit FdGuard(int fd_)
        : fd(fd_)
    {
    }
    ~FdGuard()
    {
        if (fd >= 0)
            ::close(fd);
    }
    FdGuard(const FdGuard &) = delete;
    FdGuard & operator=(const FdGuard &) = delete;
    int release()
    {
        int f = fd;
        fd = -1;
        return f;
    }
};

void fillSunPath(sockaddr_un & addr, const String & socket_path)
{
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, socket_path.data(), socket_path.size());
}

void waitForSocketEvent(
    int fd,
    int16_t events,
    const String & socket_path,
    const char * operation,
    std::chrono::steady_clock::time_point deadline,
    const std::function<bool()> & is_cancelled)
{
    while (true)
    {
        if (is_cancelled && is_cancelled())
            throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "control socket '{}' attach cancelled while waiting to {}", socket_path, operation);

        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline)
            throw Exception(
                ErrorCodes::SHM_ATTACH_FAILED,
                "control socket '{}' timed out after {}ms while waiting to {}",
                socket_path,
                CONTROL_SOCKET_ATTACH_TIMEOUT_MS,
                operation);

        const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const int timeout_ms = static_cast<int>(
            std::min<int64_t>(remaining_ms, CONTROL_SOCKET_POLL_SLICE_MS));

        pollfd pfd{};
        pfd.fd = fd;
        pfd.events = events;
        const int rc = ::poll(&pfd, 1, timeout_ms);
        if (rc < 0)
        {
            if (errno == EINTR)
                continue;
            ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, errno, "poll() on control socket '{}' failed", socket_path);
        }
        if (rc == 0)
            continue;

        return;
    }
}
}

String controlSocketPathForShmName(const String & shm_name)
{
    String sanitized;
    sanitized.reserve(shm_name.size());
    size_t start = 0;
    while (start < shm_name.size() && shm_name[start] == '/')
        ++start;
    for (size_t i = start; i < shm_name.size(); ++i)
        sanitized.push_back(shm_name[i] == '/' ? '_' : shm_name[i]);
    return "/tmp/clickhouse_shm_" + sanitized + ".sock";
}

ControlSocketServer::ControlSocketServer(const String & socket_path_)
    : socket_path(socket_path_)
    , listen_fd(-1)
{
    sockaddr_un addr{};
    if (socket_path.size() >= sizeof(addr.sun_path))
        throw Exception(
            ErrorCodes::CANNOT_OPEN_FILE,
            "ControlSocket path '{}' is too long ({} bytes, max {})",
            socket_path,
            socket_path.size(),
            sizeof(addr.sun_path) - 1);

    FdGuard guard(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (guard.fd < 0)
        throw ErrnoException(ErrorCodes::CANNOT_OPEN_FILE, "ControlSocket socket() failed");

    /// Stale file from a previous crash blocks bind() with EADDRINUSE.
    if (::unlink(socket_path.c_str()) < 0 && errno != ENOENT)
        ErrnoException::throwWithErrno(ErrorCodes::CANNOT_OPEN_FILE, errno, "ControlSocket failed to unlink stale path '{}'", socket_path);

    fillSunPath(addr, socket_path);
    if (::bind(guard.fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
        ErrnoException::throwWithErrno(ErrorCodes::CANNOT_OPEN_FILE, errno, "ControlSocket bind('{}') failed", socket_path);

    if (::listen(guard.fd, CONTROL_SOCKET_LISTEN_BACKLOG) < 0)
    {
        int saved = errno;
        ::unlink(socket_path.c_str());
        ErrnoException::throwWithErrno(ErrorCodes::CANNOT_OPEN_FILE, saved, "ControlSocket listen('{}') failed", socket_path);
    }

    listen_fd = guard.release();
}

ControlSocketServer::~ControlSocketServer()
{
    shutdown();
}

int ControlSocketServer::accept()
{
    if (is_shutdown.load(std::memory_order_acquire))
        return -1;

    int conn_fd = ::accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
    if (conn_fd < 0)
    {
        int saved = errno;
        /// shutdown() from another thread closed listen_fd under us.
        if (is_shutdown.load(std::memory_order_acquire))
            return -1;
        if (saved == EBADF || saved == EINVAL)
            return -1;
        ErrnoException::throwWithErrno(ErrorCodes::CANNOT_OPEN_FILE, saved, "ControlSocket accept4() failed on '{}'", socket_path);
    }
    return conn_fd;
}

void ControlSocketServer::sendEventFd(int conn_fd, int eventfd_to_pass)
{
    char dummy_byte = '\0';
    iovec iov{};
    iov.iov_base = &dummy_byte;
    iov.iov_len = 1;

    /// Union enforces cmsghdr alignment portably.
    union
    {
        char buf[CMSG_SPACE(sizeof(int))];
        cmsghdr align;
    } cmsg_buf{};

    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.buf;
    msg.msg_controllen = CMSG_SPACE(sizeof(int));

    cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), &eventfd_to_pass, sizeof(int));

    ssize_t sent = ::sendmsg(conn_fd, &msg, MSG_NOSIGNAL);
    if (sent < 0)
        throw ErrnoException(ErrorCodes::CANNOT_OPEN_FILE, "ControlSocket sendmsg(SCM_RIGHTS) on '{}' failed", socket_path);
    if (sent != 1)
        throw Exception(
            ErrorCodes::CANNOT_OPEN_FILE, "ControlSocket sendmsg(SCM_RIGHTS) on '{}' was partial ({} of 1 byte)", socket_path, sent);
}

void ControlSocketServer::shutdown()
{
    if (is_shutdown.exchange(true, std::memory_order_acq_rel))
        return;
    if (listen_fd >= 0)
    {
        /// Closing a listening socket from another thread does not reliably interrupt a
        /// blocking accept4() on Linux because the file description can remain alive while
        /// the syscall holds a reference. shutdown() wakes the blocking accept path; close()
        /// then releases our descriptor.
        ::shutdown(listen_fd, SHUT_RDWR);
        ::close(listen_fd);
        listen_fd = -1;
    }
    if (!socket_path.empty())
        ::unlink(socket_path.c_str());
}

int ControlSocketClient::connectAndReceiveEventFd(
    const String & socket_path,
    int & out_conn_fd,
    const std::function<bool()> & is_cancelled)
{
    out_conn_fd = -1;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(CONTROL_SOCKET_ATTACH_TIMEOUT_MS);

    sockaddr_un addr{};
    if (socket_path.size() >= sizeof(addr.sun_path))
        throw Exception(
            ErrorCodes::SHM_ATTACH_FAILED,
            "control socket path '{}' is too long ({} bytes, max {})",
            socket_path,
            socket_path.size(),
            sizeof(addr.sun_path) - 1);

    FdGuard guard(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    if (guard.fd < 0)
        throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "control socket: socket() failed for '{}'", socket_path);

    fillSunPath(addr, socket_path);
    if (::connect(guard.fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
    {
        int saved = errno;
        if (saved == EINPROGRESS || saved == EAGAIN)
        {
            waitForSocketEvent(guard.fd, POLLOUT, socket_path, "connect", deadline, is_cancelled);

            int so_error = 0;
            socklen_t so_error_size = sizeof(so_error);
            if (::getsockopt(guard.fd, SOL_SOCKET, SO_ERROR, &so_error, &so_error_size) < 0)
                ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, errno, "getsockopt(SO_ERROR) on control socket '{}' failed", socket_path);
            if (so_error != 0)
            {
                if (so_error == ENOENT)
                    ErrnoException::throwWithErrno(
                        ErrorCodes::SHM_ATTACH_FAILED,
                        so_error,
                        "control socket '{}' does not exist (producer not listening?)",
                        socket_path);
                ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, so_error, "connect() to control socket '{}' failed", socket_path);
            }
        }
        else if (saved == ENOENT)
        {
            ErrnoException::throwWithErrno(
                ErrorCodes::SHM_ATTACH_FAILED, saved, "control socket '{}' does not exist (producer not listening?)", socket_path);
        }
        else
        {
            ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, saved, "connect() to control socket '{}' failed", socket_path);
        }
    }

    char dummy_byte = 0;
    iovec iov{};
    iov.iov_base = &dummy_byte;
    iov.iov_len = 1;

    union
    {
        char buf[CMSG_SPACE(CONTROL_SOCKET_MAX_RECV_FDS * sizeof(int))];
        cmsghdr align;
    } cmsg_buf{};

    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.buf;
    msg.msg_controllen = sizeof(cmsg_buf.buf);

    /// MSG_CMSG_CLOEXEC: kernel sets FD_CLOEXEC on received fds.
    ssize_t received = -1;
    while (true)
    {
        received = ::recvmsg(guard.fd, &msg, MSG_CMSG_CLOEXEC);
        if (received >= 0)
            break;

        const int saved = errno;
        if (saved == EINTR)
            continue;
        if (saved == EAGAIN || saved == EWOULDBLOCK)
        {
            waitForSocketEvent(guard.fd, POLLIN, socket_path, "receive readiness fd", deadline, is_cancelled);
            continue;
        }
        ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, saved, "recvmsg() on control socket '{}' failed", socket_path);
    }
    if (received == 0)
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "producer closed control socket '{}' before sending readiness fd", socket_path);

    if ((msg.msg_flags & MSG_CTRUNC) != 0)
    {
        /// Linux still installs fds that fit even when MSG_CTRUNC fires;
        /// walk the partial cmsg list and close them so they don't leak.
        for (cmsghdr * c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c))
        {
            if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS)
                continue;
            size_t n_fds = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
            int * rfds = reinterpret_cast<int *>(CMSG_DATA(c));
            for (size_t i = 0; i < n_fds; ++i)
                ::close(rfds[i]);
        }
        throw Exception(
            ErrorCodes::SHM_ATTACH_FAILED, "producer sent oversized SCM_RIGHTS message on '{}' (control buffer truncated)", socket_path);
    }

    cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == nullptr || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS)
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "producer did not deliver readiness fd via SCM_RIGHTS on '{}'", socket_path);

    size_t fd_count = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    if (fd_count != 1)
    {
        int * received_fds = reinterpret_cast<int *>(CMSG_DATA(cmsg));
        for (size_t i = 0; i < fd_count; ++i)
            ::close(received_fds[i]);
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "producer delivered {} fds on '{}', expected 1", fd_count, socket_path);
    }

    int received_fd = -1;
    std::memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));
    FdGuard received_guard(received_fd);

    /// Defensive non-blocking on the received fd. The producer now creates the
    /// eventfd with EFD_NONBLOCK (Common/EventFD.cpp F4 default) and SCM_RIGHTS
    /// transfers the same file description so the flag is already set — but
    /// older or out-of-tree producers may not, and the consumer's
    /// `PollableShmSource::onAsyncJobReady` MUST NOT block in the executor's
    /// async-job path (`pollable-shm-source.md` I6). If the defensive fcntl
    /// fails, attach fails rather than risking a blocking executor callback.
    /// The connection fd and received fd are still owned by FdGuard here, so
    /// every fcntl failure closes both before the SHM_ATTACH_FAILED exception
    /// reaches the caller.
    const int existing_flags = ::fcntl(received_guard.fd, F_GETFL);
    if (existing_flags < 0)
    {
        const int saved = errno;
        ErrnoException::throwWithErrno(
            ErrorCodes::SHM_ATTACH_FAILED,
            saved,
            "fcntl(F_GETFL) on received readiness fd from '{}' failed",
            socket_path);
    }
    if (::fcntl(received_guard.fd, F_SETFL, existing_flags | O_NONBLOCK) < 0)
    {
        const int saved = errno;
        ErrnoException::throwWithErrno(
            ErrorCodes::SHM_ATTACH_FAILED,
            saved,
            "fcntl(F_SETFL, O_NONBLOCK) on received readiness fd from '{}' failed",
            socket_path);
    }

    out_conn_fd = guard.release();
    return received_guard.release();
}

}

#endif
