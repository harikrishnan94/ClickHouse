#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/Wire/ControlSocket.h>

#include <Common/ErrnoException.h>
#include <Common/Exception.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

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

    /// AF_UNIX SOCK_STREAM connect() never blocks, so SO_RCVTIMEO/SO_SNDTIMEO
    /// covers every operation that could otherwise hang against a half-bound
    /// producer (per task brief, picked over O_NONBLOCK+poll for simplicity).
    constexpr int CONTROL_SOCKET_TIMEOUT_SEC = 5;

    /// > 1 so a producer that mistakenly sends 2-4 fds is detected by an
    /// explicit fd-count check; extras get closed instead of leaked via the
    /// MSG_CTRUNC-with-installed-fds Linux corner (`scm_detach_fds`).
    constexpr size_t CONTROL_SOCKET_MAX_RECV_FDS = 4;

    /// Move-only RAII close-on-scope helper used on the error paths.
    struct FdGuard
    {
        int fd;
        explicit FdGuard(int fd_) : fd(fd_) {}
        ~FdGuard() { if (fd >= 0) ::close(fd); }
        FdGuard(const FdGuard &) = delete;
        FdGuard & operator=(const FdGuard &) = delete;
        int release() { int f = fd; fd = -1; return f; }
    };

    void fillSunPath(sockaddr_un & addr, const String & socket_path)
    {
        addr.sun_family = AF_UNIX;
        std::memcpy(addr.sun_path, socket_path.data(), socket_path.size());
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
        throw Exception(ErrorCodes::CANNOT_OPEN_FILE,
            "ControlSocket path '{}' is too long ({} bytes, max {})",
            socket_path, socket_path.size(), sizeof(addr.sun_path) - 1);

    FdGuard guard(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (guard.fd < 0)
        throw ErrnoException(ErrorCodes::CANNOT_OPEN_FILE, "ControlSocket socket() failed");

    /// Stale file from a previous crash blocks bind() with EADDRINUSE.
    if (::unlink(socket_path.c_str()) < 0 && errno != ENOENT)
        ErrnoException::throwWithErrno(ErrorCodes::CANNOT_OPEN_FILE, errno,
            "ControlSocket failed to unlink stale path '{}'", socket_path);

    fillSunPath(addr, socket_path);
    if (::bind(guard.fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
        ErrnoException::throwWithErrno(ErrorCodes::CANNOT_OPEN_FILE, errno,
            "ControlSocket bind('{}') failed", socket_path);

    if (::listen(guard.fd, CONTROL_SOCKET_LISTEN_BACKLOG) < 0)
    {
        int saved = errno;
        ::unlink(socket_path.c_str());
        ErrnoException::throwWithErrno(ErrorCodes::CANNOT_OPEN_FILE, saved,
            "ControlSocket listen('{}') failed", socket_path);
    }

    listen_fd = guard.release();
}

ControlSocketServer::~ControlSocketServer() { shutdown(); }

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
        ErrnoException::throwWithErrno(ErrorCodes::CANNOT_OPEN_FILE, saved,
            "ControlSocket accept4() failed on '{}'", socket_path);
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
    union { char buf[CMSG_SPACE(sizeof(int))]; cmsghdr align; } cmsg_buf{};

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
        throw ErrnoException(ErrorCodes::CANNOT_OPEN_FILE,
            "ControlSocket sendmsg(SCM_RIGHTS) on '{}' failed", socket_path);
    if (sent != 1)
        throw Exception(ErrorCodes::CANNOT_OPEN_FILE,
            "ControlSocket sendmsg(SCM_RIGHTS) on '{}' was partial ({} of 1 byte)",
            socket_path, sent);
}

void ControlSocketServer::shutdown()
{
    if (is_shutdown.exchange(true, std::memory_order_acq_rel))
        return;
    if (listen_fd >= 0)
    {
        ::close(listen_fd);
        listen_fd = -1;
    }
    if (!socket_path.empty())
        ::unlink(socket_path.c_str());
}

int ControlSocketClient::connectAndReceiveEventFd(const String & socket_path, int & out_conn_fd)
{
    out_conn_fd = -1;

    sockaddr_un addr{};
    if (socket_path.size() >= sizeof(addr.sun_path))
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED,
            "control socket path '{}' is too long ({} bytes, max {})",
            socket_path, socket_path.size(), sizeof(addr.sun_path) - 1);

    FdGuard guard(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (guard.fd < 0)
        throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED,
            "control socket: socket() failed for '{}'", socket_path);

    timeval timeout{};
    timeout.tv_sec = CONTROL_SOCKET_TIMEOUT_SEC;
    timeout.tv_usec = 0;
    ::setsockopt(guard.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(guard.fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    fillSunPath(addr, socket_path);
    if (::connect(guard.fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
    {
        int saved = errno;
        if (saved == ENOENT)
            ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, saved,
                "control socket '{}' does not exist (producer not listening?)", socket_path);
        ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, saved,
            "connect() to control socket '{}' failed", socket_path);
    }

    char dummy_byte = 0;
    iovec iov{};
    iov.iov_base = &dummy_byte;
    iov.iov_len = 1;

    union { char buf[CMSG_SPACE(CONTROL_SOCKET_MAX_RECV_FDS * sizeof(int))]; cmsghdr align; } cmsg_buf{};

    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.buf;
    msg.msg_controllen = sizeof(cmsg_buf.buf);

    /// MSG_CMSG_CLOEXEC: kernel sets FD_CLOEXEC on received fds.
    ssize_t received = ::recvmsg(guard.fd, &msg, MSG_CMSG_CLOEXEC);
    if (received < 0)
        ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, errno,
            "recvmsg() on control socket '{}' failed", socket_path);
    if (received == 0)
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED,
            "producer closed control socket '{}' before sending readiness fd", socket_path);

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
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED,
            "producer sent oversized SCM_RIGHTS message on '{}' (control buffer truncated)",
            socket_path);
    }

    cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == nullptr || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS)
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED,
            "producer did not deliver readiness fd via SCM_RIGHTS on '{}'", socket_path);

    size_t fd_count = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    if (fd_count != 1)
    {
        int * received_fds = reinterpret_cast<int *>(CMSG_DATA(cmsg));
        for (size_t i = 0; i < fd_count; ++i)
            ::close(received_fds[i]);
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED,
            "producer delivered {} fds on '{}', expected 1", fd_count, socket_path);
    }

    int received_fd = -1;
    std::memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));

    /// Defensive non-blocking on the received fd. The producer now creates the
    /// eventfd with EFD_NONBLOCK (Common/EventFD.cpp F4 default) and SCM_RIGHTS
    /// transfers the same file description so the flag is already set — but
    /// older or out-of-tree producers may not, and the consumer's
    /// `PollableShmSource::onAsyncJobReady` MUST NOT block in the executor's
    /// async-job path (`pollable-shm-source.md` I6). A failed fcntl on an
    /// otherwise-valid fd is non-fatal: we ignore it because the worst case is
    /// the producer-created fd remained blocking, which only matters if the
    /// consumer's `read()` ever races a wake — bounded by the executor's
    /// epoll-readable signal.
    const int existing_flags = ::fcntl(received_fd, F_GETFL);
    if (existing_flags >= 0)
        (void)::fcntl(received_fd, F_SETFL, existing_flags | O_NONBLOCK);

    out_conn_fd = guard.release();
    return received_fd;
}

}

#endif
