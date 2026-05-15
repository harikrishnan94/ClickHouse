#include <gtest/gtest.h>

#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/Wire/ControlSocket.h>

#include <Common/Exception.h>

#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>

namespace DB::ErrorCodes
{
    extern const int SHM_ATTACH_FAILED;
}

namespace
{
    /// PID + test name keeps parallel runs from colliding on /tmp.
    std::string makeUniquePath(const char * test_name)
    {
        return "/tmp/gtest_control_socket_" + std::to_string(getpid()) + "_" + test_name + ".sock";
    }
}

class ControlSocketTest : public ::testing::Test
{
protected:
    std::string socket_path;

    void SetUp() override
    {
        socket_path = makeUniquePath(::testing::UnitTest::GetInstance()->current_test_info()->name());
        ::unlink(socket_path.c_str());
    }
    void TearDown() override { ::unlink(socket_path.c_str()); }
};

TEST(ControlSocketPath, DerivationMatchesConvention)
{
    using DB::controlSocketPathForShmName;
    EXPECT_EQ(controlSocketPathForShmName("foo"), "/tmp/clickhouse_shm_foo.sock");
    EXPECT_EQ(controlSocketPathForShmName("/foo"), "/tmp/clickhouse_shm_foo.sock");
    EXPECT_EQ(controlSocketPathForShmName("/foo/bar"), "/tmp/clickhouse_shm_foo_bar.sock");
    EXPECT_EQ(controlSocketPathForShmName("foo/bar/baz"), "/tmp/clickhouse_shm_foo_bar_baz.sock");
    EXPECT_EQ(controlSocketPathForShmName("//abc"), "/tmp/clickhouse_shm_abc.sock");
    EXPECT_EQ(controlSocketPathForShmName(""), "/tmp/clickhouse_shm_.sock");
}

TEST_F(ControlSocketTest, HappyPathFdRefersToSameKernelObject)
{
    DB::ControlSocketServer server(socket_path);
    int server_evfd = ::eventfd(0, EFD_CLOEXEC);
    ASSERT_GE(server_evfd, 0);

    int received_fd = -1;
    int client_conn_fd = -1;
    std::optional<DB::Exception> client_error;

    std::thread server_thread([&]
    {
        int conn = server.accept();
        if (conn < 0) return;
        server.sendEventFd(conn, server_evfd);
        /// Hold conn briefly so the consumer side sees a live POLLHUP-able fd.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ::close(conn);
    });
    std::thread client_thread([&]
    {
        try { received_fd = DB::ControlSocketClient::connectAndReceiveEventFd(socket_path, client_conn_fd); }
        catch (const DB::Exception & e) { client_error = e; }
    });
    server_thread.join();
    client_thread.join();

    ASSERT_FALSE(client_error.has_value()) << client_error->displayText();
    ASSERT_GE(received_fd, 0);
    ASSERT_GE(client_conn_fd, 0);
    EXPECT_NE(received_fd, server_evfd) << "kernel returns a distinct fd number";

    /// Write through the received fd, read from the server-side handle.
    constexpr uint64_t increment = 7;
    uint64_t write_val = increment;
    EXPECT_EQ(::write(received_fd, &write_val, sizeof(write_val)), static_cast<ssize_t>(sizeof(write_val)));
    uint64_t read_val = 0;
    EXPECT_EQ(::read(server_evfd, &read_val, sizeof(read_val)), static_cast<ssize_t>(sizeof(read_val)));
    EXPECT_EQ(read_val, increment);

    ::close(received_fd);
    ::close(client_conn_fd);
    ::close(server_evfd);
}

TEST_F(ControlSocketTest, ConnectToNonExistentSocketRaisesAttachFailed)
{
    int conn_fd = 42;
    std::optional<int> code;
    try { DB::ControlSocketClient::connectAndReceiveEventFd(socket_path, conn_fd); }
    catch (const DB::Exception & e) { code = e.code(); }
    ASSERT_TRUE(code.has_value());
    EXPECT_EQ(*code, DB::ErrorCodes::SHM_ATTACH_FAILED);
    EXPECT_EQ(conn_fd, -1) << "out_conn_fd must reset on failure";
}

TEST_F(ControlSocketTest, ServerDestroyedBeforeAcceptingRaisesAttachFailed)
{
    /// Server binds+listens but never accepts. Client connect succeeds
    /// (backlog), recvmsg blocks. Destroying the server resets the backlog
    /// and the client's recvmsg fails → SHM_ATTACH_FAILED per the
    /// spec's "readiness-fd locator unresolvable" mapping.
    auto server = std::make_unique<DB::ControlSocketServer>(socket_path);
    int client_conn_fd = -1;
    std::optional<int> code;

    std::thread client_thread([&]
    {
        try { DB::ControlSocketClient::connectAndReceiveEventFd(socket_path, client_conn_fd); }
        catch (const DB::Exception & e) { code = e.code(); }
    });
    /// Well below the 5s socket timeout so scheduler jitter cannot flake.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    server.reset();
    client_thread.join();

    ASSERT_TRUE(code.has_value());
    EXPECT_EQ(*code, DB::ErrorCodes::SHM_ATTACH_FAILED);
    EXPECT_EQ(client_conn_fd, -1);
}

TEST_F(ControlSocketTest, ShutdownUnblocksConcurrentAccept)
{
    DB::ControlSocketServer server(socket_path);
    int accept_result = 42;
    std::thread accept_thread([&] { accept_result = server.accept(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    server.shutdown();
    accept_thread.join();
    EXPECT_EQ(accept_result, -1);
}

TEST_F(ControlSocketTest, ProducerSendsTwoFdsClientRejects)
{
    /// Bypass ControlSocketServer::sendEventFd to send 2 fds in one cmsg.
    DB::ControlSocketServer server(socket_path);
    int evfd1 = ::eventfd(0, EFD_CLOEXEC);
    int evfd2 = ::eventfd(0, EFD_CLOEXEC);
    ASSERT_GE(evfd1, 0);
    ASSERT_GE(evfd2, 0);

    int client_conn_fd = -1;
    std::optional<int> code;

    std::thread server_thread([&]
    {
        int peer = server.accept();
        if (peer < 0) return;

        char dummy = 0;
        iovec iov{};
        iov.iov_base = &dummy;
        iov.iov_len = 1;
        union { char buf[CMSG_SPACE(sizeof(int) * 2)]; cmsghdr align; } cbuf{};
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cbuf.buf;
        msg.msg_controllen = sizeof(cbuf.buf);
        cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int) * 2);
        int fds[2] = {evfd1, evfd2};
        std::memcpy(CMSG_DATA(cmsg), fds, sizeof(fds));
        ::sendmsg(peer, &msg, MSG_NOSIGNAL);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ::close(peer);
    });
    std::thread client_thread([&]
    {
        try { DB::ControlSocketClient::connectAndReceiveEventFd(socket_path, client_conn_fd); }
        catch (const DB::Exception & e) { code = e.code(); }
    });
    server_thread.join();
    client_thread.join();

    ASSERT_TRUE(code.has_value());
    EXPECT_EQ(*code, DB::ErrorCodes::SHM_ATTACH_FAILED);
    EXPECT_EQ(client_conn_fd, -1);

    ::close(evfd1);
    ::close(evfd2);
}

#endif
