#include <gtest/gtest.h>

#if defined(OS_LINUX)

#    include <Storages/SharedMemorySource/Source/TcpStreamSource.h>
#    include <Storages/SharedMemorySource/Wire/Layout.h>
#    include <Storages/SharedMemorySource/Wire/TcpFrame.h>
#    include <Storages/SharedMemorySource/Wire/WireTypeMapping.h>

#    include <Columns/ColumnString.h>
#    include <Columns/ColumnsNumber.h>
#    include <Core/Block.h>
#    include <DataTypes/DataTypeString.h>
#    include <DataTypes/DataTypesNumber.h>
#    include <Processors/Chunk.h>
#    include <Processors/Executors/PullingPipelineExecutor.h>
#    include <QueryPipeline/QueryPipeline.h>
#    include <Common/Exception.h>

#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#    include <unistd.h>

#    include <atomic>
#    include <cerrno>
#    include <chrono>
#    include <cstring>
#    include <thread>
#    include <vector>


using namespace DB;
using namespace DB::SharedMemoryWire;

namespace DB::ErrorCodes
{
    extern const int SHM_PRODUCER_STALL;
    extern const int SHM_PRODUCER_DEATH_BEFORE_EOS;
}

namespace
{

size_t alignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

/// Reference frame serializer (the canonical Phase-1 layout the PG producer mirrors). Builds one
/// BLOCK frame = TcpBlockHeader + frame-relative data region: ColumnDescriptor[ncols] at offset 0,
/// then each column's buffers laid out with the SHM align / SharedMemoryWire::PADDING_FOR_SIMD / offsets[-1]-zero-
/// sentinel rules, offsets relative to the payload base. Schema here = (id UInt64, s String).
std::vector<char> serializeBlock(const std::vector<uint64_t> & ids,
                                 const std::vector<uint8_t> & chars,
                                 const std::vector<uint64_t> & offsets,
                                 bool eos)
{
    constexpr size_t ncols = 2;
    const uint64_t row_count = ids.size();

    /// Layout pass.
    ColumnDescriptor descs[ncols];
    std::memset(descs, 0, sizeof(descs));
    size_t cursor = alignUp(ncols * sizeof(ColumnDescriptor), 8);

    // col 0: id UInt64
    const size_t id_bytes = ids.size() * sizeof(uint64_t);
    const size_t id_off = alignUp(cursor, 8);
    cursor = id_off + id_bytes + SharedMemoryWire::PADDING_FOR_SIMD;
    descs[0].type = static_cast<uint32_t>(WireColumnType::UInt64);
    descs[0].value_offset = id_off;
    descs[0].value_count = row_count;
    descs[0].value_padding = SharedMemoryWire::PADDING_FOR_SIMD;

    // col 1: s String  (chars, sentinel, offsets)
    const size_t chars_off = alignUp(cursor, 8);
    cursor = chars_off + chars.size() + SharedMemoryWire::PADDING_FOR_SIMD;
    const size_t sentinel_off = alignUp(cursor, 8);
    cursor = sentinel_off + sizeof(uint64_t);
    const size_t offs_off = alignUp(cursor, 8);
    const size_t offs_bytes = offsets.size() * sizeof(uint64_t);
    cursor = offs_off + offs_bytes + SharedMemoryWire::PADDING_FOR_SIMD;
    descs[1].type = static_cast<uint32_t>(WireColumnType::String);
    descs[1].value_offset = chars_off;
    descs[1].value_count = chars.size();
    descs[1].value_padding = SharedMemoryWire::PADDING_FOR_SIMD;
    descs[1].offsets_offset = offs_off;
    descs[1].offsets_count = row_count;
    descs[1].offsets_padding = SharedMemoryWire::PADDING_FOR_SIMD;

    const size_t payload_len = cursor;

    std::vector<char> frame(sizeof(TcpBlockHeader) + payload_len, 0);  // zero-init => sentinel+pads are 0
    auto * bh = reinterpret_cast<TcpBlockHeader *>(frame.data());
    bh->payload_len = payload_len;
    bh->row_count = row_count;
    bh->descriptors_offset = 0;
    bh->eos_marker = eos ? 1 : 0;

    char * payload = frame.data() + sizeof(TcpBlockHeader);
    std::memcpy(payload, descs, sizeof(descs));
    if (id_bytes) std::memcpy(payload + id_off, ids.data(), id_bytes);
    if (!chars.empty()) std::memcpy(payload + chars_off, chars.data(), chars.size());
    if (offs_bytes) std::memcpy(payload + offs_off, offsets.data(), offs_bytes);
    return frame;
}

void sendAll(int fd, const void * buf, size_t n)
{
    const auto * p = static_cast<const char *>(buf);
    while (n)
    {
        ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) { if (errno == EINTR) continue; return; }
        p += w; n -= static_cast<size_t>(w);
    }
}

/// Send `n` bytes in <=`frag` chunks with a `delay_ms` pause between fragments. Forces the consumer's
/// non-blocking recv to hit EAGAIN mid-frame so the ASYNC source's resumable recv + wake bridge are
/// exercised (a frame straddling schedule cycles). With delay_ms==0 / frag>=n this is a plain sendAll.
void sendFragmented(int fd, const void * buf, size_t n, size_t frag, int delay_ms)
{
    const auto * p = static_cast<const char *>(buf);
    while (n)
    {
        const size_t step = std::min(frag, n);
        sendAll(fd, p, step);
        p += step; n -= step;
        if (n && delay_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
}

/// End-to-end consumer drain: a reference TCP producer thread serves the handshake + N blocks + EOS;
/// TcpStreamSource connects, reconstructs the columns via the unchanged adopt(), and emits them in
/// order. Runs through a REAL PullingPipelineExecutor, so the async source's Status::Async / schedule()
/// / onAsyncJobReady() contract is exercised by the actual executor. `async` selects the source mode;
/// `slow` fragments each frame with delays so the async resumable-recv path is forced (not just the
/// fast path where whole frames are already buffered).
void runDrainTest(bool async, bool slow)
{
    constexpr size_t n_blocks = 25;

    int listen_fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(listen_fd, 0);
    int one = 1;
    ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;  // ephemeral
    ASSERT_EQ(::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(listen_fd, 4), 0);
    socklen_t alen = sizeof(addr);
    ASSERT_EQ(::getsockname(listen_fd, reinterpret_cast<sockaddr *>(&addr), &alen), 0);
    const uint16_t port = ntohs(addr.sin_port);

    std::thread producer([&]
    {
        int conn = ::accept(listen_fd, nullptr, nullptr);
        if (conn < 0) return;

        TcpHandshakeHeader hs{};
        hs.magic = SHM_TCP_MAGIC;
        hs.abi_version = SHM_TCP_ABI_VERSION_1;
        hs.schema_count = 2;
        sendAll(conn, &hs, sizeof(hs));
        SchemaEntry se[2];
        std::memset(se, 0, sizeof(se));
        std::strcpy(se[0].name, "id");  std::strcpy(se[0].type_string, "UInt64");
        std::strcpy(se[1].name, "s");   std::strcpy(se[1].type_string, "String");
        sendAll(conn, se, sizeof(se));

        for (size_t b = 0; b < n_blocks; ++b)
        {
            const std::vector<uint64_t> ids = {b, b + 100, b + 200};
            const std::vector<uint8_t> chars = {'a', 'b', 'c'};   // 3 one-char strings
            const std::vector<uint64_t> offs = {1, 2, 3};
            auto frame = serializeBlock(ids, chars, offs, /*eos=*/false);
            if (slow)
                sendFragmented(conn, frame.data(), frame.size(), /*frag=*/17, /*delay_ms=*/2);
            else
                sendAll(conn, frame.data(), frame.size());
        }
        // EOS frame must carry payload_len==0 for the source's EOS branch.
        TcpBlockHeader eos_h{};
        eos_h.eos_marker = 1;
        sendAll(conn, &eos_h, sizeof(eos_h));
        ::close(conn);
    });

    Block b;
    b.insert({std::make_shared<DataTypeUInt64>()->createColumn(), std::make_shared<DataTypeUInt64>(), "id"});
    b.insert({std::make_shared<DataTypeString>()->createColumn(), std::make_shared<DataTypeString>(), "s"});
    auto header = std::make_shared<const Block>(std::move(b));

    auto src = std::make_shared<TcpStreamSource>(
        header, "127.0.0.1", port,
        std::vector<DataTypePtr>{std::make_shared<DataTypeUInt64>(), std::make_shared<DataTypeString>()},
        std::vector<String>{"id", "s"}, std::vector<String>{"id", "s"}, 60'000, async);

    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);
    size_t chunks = 0;
    Chunk chunk;
    while (executor.pull(chunk))
    {
        if (!chunk.hasRows()) continue;
        ASSERT_EQ(chunk.getNumColumns(), 2u);
        ASSERT_EQ(chunk.getNumRows(), 3u) << "block " << chunks;
        const auto & cols = chunk.getColumns();
        const auto * id_col = typeid_cast<const ColumnUInt64 *>(cols[0].get());
        const auto * s_col = typeid_cast<const ColumnString *>(cols[1].get());
        ASSERT_NE(id_col, nullptr);
        ASSERT_NE(s_col, nullptr);
        EXPECT_EQ(id_col->getData()[0], chunks);
        EXPECT_EQ(id_col->getData()[1], chunks + 100);
        EXPECT_EQ(id_col->getData()[2], chunks + 200);
        EXPECT_EQ(std::string(s_col->getDataAt(0)), "a");
        EXPECT_EQ(std::string(s_col->getDataAt(1)), "b");
        EXPECT_EQ(std::string(s_col->getDataAt(2)), "c");
        ++chunks;
    }
    producer.join();
    ::close(listen_fd);
    ASSERT_EQ(chunks, n_blocks);
}


/// --- Hot-Cold Phase 3 Branch C1 helpers (epoll-fd readiness) ---

/// Bind+listen an ephemeral 127.0.0.1 socket; returns the listen fd and writes the chosen port.
int bindLoopbackListener(uint16_t & port)
{
    int lfd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (lfd < 0) return -1;
    int one = 1;
    ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(lfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) { ::close(lfd); return -1; }
    if (::listen(lfd, 4) != 0) { ::close(lfd); return -1; }
    socklen_t alen = sizeof(addr);
    ::getsockname(lfd, reinterpret_cast<sockaddr *>(&addr), &alen);
    port = ntohs(addr.sin_port);
    return lfd;
}

/// Send the (id UInt64, s String) bespoke handshake the consumer cross-validates.
void sendHandshakeIdS(int conn)
{
    TcpHandshakeHeader hs{};
    hs.magic = SHM_TCP_MAGIC;
    hs.abi_version = SHM_TCP_ABI_VERSION_1;
    hs.schema_count = 2;
    sendAll(conn, &hs, sizeof(hs));
    SchemaEntry se[2];
    std::memset(se, 0, sizeof(se));
    std::strcpy(se[0].name, "id");  std::strcpy(se[0].type_string, "UInt64");
    std::strcpy(se[1].name, "s");   std::strcpy(se[1].type_string, "String");
    sendAll(conn, se, sizeof(se));
}

/// Block on the connection until the peer (consumer) closes/shuts down its write side (recv → 0), then
/// close. Lets a "stalling" producer thread exit promptly once the consumer cancels / tears down.
void drainUntilPeerClose(int conn)
{
    char buf[256];
    for (;;)
    {
        ssize_t r = ::recv(conn, buf, sizeof(buf), 0);
        if (r > 0) continue;
        if (r < 0 && errno == EINTR) continue;
        break;   /// r==0 (peer FIN) or hard error
    }
    ::close(conn);
}

std::shared_ptr<TcpStreamSource> makeIdSSource(uint16_t port, UInt64 stall_ms)
{
    Block b;
    b.insert({std::make_shared<DataTypeUInt64>()->createColumn(), std::make_shared<DataTypeUInt64>(), "id"});
    b.insert({std::make_shared<DataTypeString>()->createColumn(), std::make_shared<DataTypeString>(), "s"});
    auto header = std::make_shared<const Block>(std::move(b));
    return std::make_shared<TcpStreamSource>(
        header, "127.0.0.1", port,
        std::vector<DataTypePtr>{std::make_shared<DataTypeUInt64>(), std::make_shared<DataTypeString>()},
        std::vector<String>{"id", "s"}, std::vector<String>{"id", "s"}, stall_ms, /*async=*/true);
}

}


/// Async source (Branch-0 default), fast producer: whole frames already buffered, mostly fast-path recv.
TEST(TcpStreamSource, DrainsHandshakeAndBlocks)
{
    runDrainTest(/*async=*/true, /*slow=*/false);
}

/// Blocking source (A/B baseline, shm_tcp_source_async=0): the Phase-1 leaf source still drains correctly.
TEST(TcpStreamSource, DrainsHandshakeAndBlocksBlocking)
{
    runDrainTest(/*async=*/false, /*slow=*/false);
}

/// Async source, SLOW fragmented producer: forces the consumer's non-blocking recv to hit EAGAIN
/// mid-frame, exercising the resumable recv state machine + the async wake bridge across schedule
/// cycles. Proves a partial Arrow/bespoke frame straddling schedule cycles reassembles correctly.
TEST(TcpStreamSource, AsyncResumesAcrossPartialFrames)
{
    runDrainTest(/*async=*/true, /*slow=*/true);
}


/// Phase-1 instrument #6: isolated loopback-TCP throughput (per-byte transfer ns / GB/s) with no
/// scan/adopt/consumer noise, to fix the TCP transfer constant the overhead prediction uses. A
/// sender thread streams `total` bytes in block-sized send()s over a 127.0.0.1 socket with the same
/// 32 MiB SO_SNDBUF/SO_RCVBUF the transport uses; the main thread recvs and times it. RecordProperty
/// + stdout; not a pass/fail gate.
TEST(TcpStreamSource, LoopbackThroughputMicrobench)
{
    using clock = std::chrono::steady_clock;
    constexpr size_t total = 1024ULL * 1024 * 1024;   // 1 GiB
    constexpr size_t chunk = 2 * 1024 * 1024;         // 2 MiB "blocks"
    const int bufsz = 32 * 1024 * 1024;

    int lfd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(lfd, 0);
    int one = 1; ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); addr.sin_port = 0;
    ASSERT_EQ(::bind(lfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(lfd, 1), 0);
    socklen_t al = sizeof(addr); ::getsockname(lfd, reinterpret_cast<sockaddr *>(&addr), &al);

    int cfd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(cfd, 0);
    ::setsockopt(cfd, SOL_SOCKET, SO_SNDBUF, &bufsz, sizeof(bufsz));
    ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    ASSERT_EQ(::connect(cfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)), 0);
    int afd = ::accept(lfd, nullptr, nullptr);
    ASSERT_GE(afd, 0);
    ::setsockopt(afd, SOL_SOCKET, SO_RCVBUF, &bufsz, sizeof(bufsz));

    std::thread sender([&]
    {
        std::vector<char> buf(chunk, 'x');
        size_t left = total;
        while (left) { size_t n = std::min(chunk, left); sendAll(cfd, buf.data(), n); left -= n; }
    });

    std::vector<char> rbuf(chunk);
    size_t got = 0;
    const auto t0 = clock::now();
    while (got < total)
    {
        ssize_t r = ::recv(afd, rbuf.data(), chunk, 0);
        if (r <= 0) { if (errno == EINTR) continue; break; }
        got += static_cast<size_t>(r);
    }
    const auto t1 = clock::now();
    sender.join();
    ::close(afd); ::close(cfd); ::close(lfd);

    ASSERT_EQ(got, total);
    const double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    const double gbps = static_cast<double>(total) / ns;        // bytes/ns = GB/s
    RecordProperty("loopback_tcp_GBps", std::to_string(gbps));
    RecordProperty("loopback_tcp_ns_per_byte", std::to_string(ns / total));
    std::cout << "[microbench] loopback TCP (" << (total / (1024 * 1024)) << " MiB, " << (chunk / (1024 * 1024))
              << " MiB chunks, 32 MiB bufs): " << gbps << " GB/s, " << (ns / total) << " ns/byte\n";
}


/// --- Hot-Cold Phase 3 Branch C1: epoll-fd readiness acceptance gtests ---

/// H15 (thread elimination) + H2 (bounded work / no hot-spin). A SLOW fragmented producer forces the
/// async source to park many times across schedule cycles. Post-C1 the source must: (1) spawn ZERO
/// std::threads (the per-async wake-bridge thread is deleted) — proven by threadsSpawned()==0 while the
/// async path is genuinely exercised (asyncWaitCount()>0); (2) NOT hot-spin on the level-triggered epoll
/// fd — proven by a BOUNDED async-park count (a hot-spin would re-park without new data, exploding it).
TEST(TcpStreamSource, C1AsyncEpollNoThreadsAndBounded)
{
    constexpr size_t n_blocks = 25;
    uint16_t port = 0;
    int listen_fd = bindLoopbackListener(port);
    ASSERT_GE(listen_fd, 0);

    std::thread producer([&]
    {
        int conn = ::accept(listen_fd, nullptr, nullptr);
        if (conn < 0) return;
        sendHandshakeIdS(conn);
        for (size_t b = 0; b < n_blocks; ++b)
        {
            const std::vector<uint64_t> ids = {b, b + 100, b + 200};
            const std::vector<uint8_t> chars = {'a', 'b', 'c'};
            const std::vector<uint64_t> offs = {1, 2, 3};
            auto frame = serializeBlock(ids, chars, offs, /*eos=*/false);
            sendFragmented(conn, frame.data(), frame.size(), /*frag=*/17, /*delay_ms=*/2);
        }
        TcpBlockHeader eos_h{};
        eos_h.eos_marker = 1;
        sendAll(conn, &eos_h, sizeof(eos_h));
        ::close(conn);
    });

    TcpStreamSource::resetAsyncCounters();
    auto src = makeIdSSource(port, /*stall_ms=*/60'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);
    size_t chunks = 0;
    Chunk chunk;
    while (executor.pull(chunk))
        if (chunk.hasRows()) ++chunks;
    producer.join();
    ::close(listen_fd);

    ASSERT_EQ(chunks, n_blocks);
    const uint64_t parks = TcpStreamSource::asyncWaitCount();
    const uint64_t threads = TcpStreamSource::threadsSpawned();
    RecordProperty("c1_async_parks", std::to_string(parks));
    RecordProperty("c1_threads_spawned", std::to_string(threads));
    EXPECT_EQ(threads, 0u) << "C1: streaming path must spawn zero std::threads (was 1 per async wait)";
    EXPECT_GT(parks, 0u) << "C1: the async resumable-recv path must actually have been exercised";
    /// Bounded-work (H2): each park corresponds to a real fragment gap (~tens of fragments per frame ×
    /// n_blocks). A level-triggered hot-spin would re-park without new data, producing orders of
    /// magnitude more. 100×n_blocks is a generous ceiling that still traps a true spin (≥1e5).
    EXPECT_LT(parks, 100u * n_blocks) << "C1: async-park count too high — possible epoll hot-spin (H2)";
}

/// H3 (cancel during async-park): the producer handshakes then STALLS (no blocks, no EOS), so the source
/// parks on the epoll fd with the timerfd armed to the 60s stall budget. A canceller thread calls
/// executor.cancel() → onCancel() → ::shutdown(sock, SHUT_RDWR) → the socket becomes readable/RDHUP in
/// the epoll fd → the parked single-threaded executor wakes → pull() returns false. The drain must exit
/// FAR sooner than the 60s budget — proving shutdown() (not a readiness eventfd) is the cancel wake.
TEST(TcpStreamSource, C1CancelViaShutdownWakesParkedEpoll)
{
    using clock = std::chrono::steady_clock;
    uint16_t port = 0;
    int listen_fd = bindLoopbackListener(port);
    ASSERT_GE(listen_fd, 0);

    std::thread producer([&]
    {
        int conn = ::accept(listen_fd, nullptr, nullptr);
        if (conn < 0) return;
        sendHandshakeIdS(conn);
        drainUntilPeerClose(conn);   /// exits when the consumer shuts down / closes on cancel
    });

    auto src = makeIdSSource(port, /*stall_ms=*/60'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    std::thread canceller([&]
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        executor.cancel();
    });

    const auto t0 = clock::now();
    Chunk chunk;
    size_t rows = 0;
    while (executor.pull(chunk))
        rows += chunk.getNumRows();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();

    canceller.join();
    producer.join();
    ::close(listen_fd);

    EXPECT_EQ(rows, 0u);
    EXPECT_LT(elapsed_ms, 10'000) << "cancel via shutdown() did not promptly wake the parked epoll (hang)";
    RecordProperty("c1_cancel_wake_ms", std::to_string(elapsed_ms));
}

/// Stall-timeout acceptance: the timerfd is the SOLE wakeup in the single-threaded executor's
/// async_task_queue.wait(-1) path, so a stalled producer must still raise SHM_PRODUCER_STALL within the
/// budget. Producer handshakes then stalls; consumer stall budget = 400ms; pull() must THROW
/// SHM_PRODUCER_STALL at ~400ms (not hang, not spin).
TEST(TcpStreamSource, C1StallTimeoutFires)
{
    using clock = std::chrono::steady_clock;
    uint16_t port = 0;
    int listen_fd = bindLoopbackListener(port);
    ASSERT_GE(listen_fd, 0);

    std::thread producer([&]
    {
        int conn = ::accept(listen_fd, nullptr, nullptr);
        if (conn < 0) return;
        sendHandshakeIdS(conn);
        drainUntilPeerClose(conn);
    });

    auto src = makeIdSSource(port, /*stall_ms=*/400);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    const auto t0 = clock::now();
    bool threw_stall = false;
    try
    {
        Chunk chunk;
        while (executor.pull(chunk)) {}
    }
    catch (const Exception & e)
    {
        threw_stall = (e.code() == ErrorCodes::SHM_PRODUCER_STALL);
        if (!threw_stall) throw;
    }
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();

    producer.join();
    ::close(listen_fd);

    EXPECT_TRUE(threw_stall) << "expected SHM_PRODUCER_STALL from a stalled producer";
    EXPECT_GE(elapsed_ms, 350) << "stall fired implausibly early (budget 400ms)";
    EXPECT_LT(elapsed_ms, 5'000) << "stall fired far too late — timerfd may not be waking the wait(-1)";
    RecordProperty("c1_stall_fire_ms", std::to_string(elapsed_ms));
}

/// Half-close before EOS (H2 terminal-event handling): the producer sends a few blocks then close()s the
/// connection WITHOUT an EOS frame. EPOLLRDHUP on a half-closed peer is NOT clearable by reading, so the
/// source must drive recv to a terminal SHM_PRODUCER_DEATH_BEFORE_EOS throw — NOT return Async on the
/// RDHUP (which would permanently spin). Asserts the throw and a bounded async-park count.
TEST(TcpStreamSource, C1HalfCloseBeforeEosThrows)
{
    constexpr size_t n_blocks = 4;
    uint16_t port = 0;
    int listen_fd = bindLoopbackListener(port);
    ASSERT_GE(listen_fd, 0);

    std::thread producer([&]
    {
        int conn = ::accept(listen_fd, nullptr, nullptr);
        if (conn < 0) return;
        sendHandshakeIdS(conn);
        for (size_t b = 0; b < n_blocks; ++b)
        {
            const std::vector<uint64_t> ids = {b, b + 100, b + 200};
            const std::vector<uint8_t> chars = {'a', 'b', 'c'};
            const std::vector<uint64_t> offs = {1, 2, 3};
            auto frame = serializeBlock(ids, chars, offs, /*eos=*/false);
            sendAll(conn, frame.data(), frame.size());
        }
        ::close(conn);   /// no EOS frame — abrupt half-close
    });

    TcpStreamSource::resetAsyncCounters();
    auto src = makeIdSSource(port, /*stall_ms=*/60'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    bool threw_death = false;
    try
    {
        Chunk chunk;
        while (executor.pull(chunk)) {}
    }
    catch (const Exception & e)
    {
        threw_death = (e.code() == ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS);
        if (!threw_death) throw;
    }

    producer.join();
    ::close(listen_fd);

    EXPECT_TRUE(threw_death) << "expected SHM_PRODUCER_DEATH_BEFORE_EOS on producer half-close before EOS";
    const uint64_t parks = TcpStreamSource::asyncWaitCount();
    RecordProperty("c1_halfclose_parks", std::to_string(parks));
    EXPECT_LT(parks, 100u * (n_blocks + 1)) << "C1: park count too high on RDHUP — possible spin (H2)";
}

#endif
