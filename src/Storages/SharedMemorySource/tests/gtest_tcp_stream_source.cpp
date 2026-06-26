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

#    include <cstring>
#    include <thread>
#    include <vector>


using namespace DB;
using namespace DB::SharedMemoryWire;

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

}


/// End-to-end consumer test: a reference TCP producer thread serves the handshake + N blocks + EOS;
/// TcpStreamSource connects, reconstructs the columns via the unchanged adopt(), and emits them in
/// order. Proves the framing + the recv-buffer adopt path (the consumer half of Phase 1).
TEST(TcpStreamSource, DrainsHandshakeAndBlocks)
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
            sendAll(conn, frame.data(), frame.size());
        }
        auto eos = serializeBlock({}, {}, {}, /*eos=*/true);
        // EOS frame must carry payload_len==0 for the source's EOS branch.
        TcpBlockHeader eos_h{};
        eos_h.eos_marker = 1;
        sendAll(conn, &eos_h, sizeof(eos_h));
        (void)eos;
        ::close(conn);
    });

    Block b;
    b.insert({std::make_shared<DataTypeUInt64>()->createColumn(), std::make_shared<DataTypeUInt64>(), "id"});
    b.insert({std::make_shared<DataTypeString>()->createColumn(), std::make_shared<DataTypeString>(), "s"});
    auto header = std::make_shared<const Block>(std::move(b));

    auto src = std::make_shared<TcpStreamSource>(
        header, "127.0.0.1", port,
        std::vector<DataTypePtr>{std::make_shared<DataTypeUInt64>(), std::make_shared<DataTypeString>()},
        std::vector<String>{"id", "s"}, std::vector<String>{"id", "s"}, 60'000);

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

#endif
