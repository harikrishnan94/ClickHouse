#include <gtest/gtest.h>

#if defined(OS_LINUX)

#include "config.h"   /// USE_ARROW

#if USE_ARROW

#    include <Storages/SharedMemorySource/Source/TcpStreamSource.h>

#    include <Columns/ColumnString.h>
#    include <Columns/ColumnsNumber.h>
#    include <Core/Block.h>
#    include <DataTypes/DataTypeDate.h>
#    include <DataTypes/DataTypeString.h>
#    include <DataTypes/DataTypesNumber.h>
#    include <Formats/FormatSettings.h>
#    include <Processors/Chunk.h>
#    include <Processors/Executors/PullingPipelineExecutor.h>
#    include <Processors/Formats/Impl/ArrowColumnToCHColumn.h>
#    include <QueryPipeline/QueryPipeline.h>
#    include <Common/Exception.h>

#    include <arrow/api.h>
#    include <arrow/io/memory.h>
#    include <arrow/ipc/options.h>
#    include <arrow/ipc/writer.h>
#    include <arrow/record_batch.h>
#    include <arrow/table.h>

#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <sys/socket.h>
#    include <unistd.h>

#    include <chrono>
#    include <cstring>
#    include <thread>
#    include <vector>


using namespace DB;

namespace
{

/// The Arrow schema the PG producer emits (D-HC-0207): id UInt64, s String -> LargeBinary, d Date ->
/// raw uint16 (non-semantic; recovered to Date from the SQL schema). Built with Arrow's OWN reference
/// stream writer, so the consumer is validated against the canonical Arrow IPC encoder (independent of
/// the producer's nanoarrow encoder, which is exercised end-to-end by verify_offload.sh TRANSPORT=arrow).
std::shared_ptr<arrow::Schema> arrowSchema()
{
    return arrow::schema({
        arrow::field("id", arrow::uint64()),
        arrow::field("s", arrow::large_binary()),
        arrow::field("d", arrow::uint16()),
    });
}

std::shared_ptr<arrow::RecordBatch> makeBatch(size_t b)
{
    arrow::UInt64Builder idb;
    if (!idb.AppendValues(std::vector<uint64_t>{b, b + 100, b + 200}).ok()) return nullptr;
    std::shared_ptr<arrow::Array> id_arr;
    if (!idb.Finish(&id_arr).ok()) return nullptr;

    arrow::LargeBinaryBuilder sb;
    if (!sb.Append("a").ok() || !sb.Append("bb").ok() || !sb.Append("ccc").ok()) return nullptr;
    std::shared_ptr<arrow::Array> s_arr;
    if (!sb.Finish(&s_arr).ok()) return nullptr;

    arrow::UInt16Builder db;
    if (!db.AppendValues(std::vector<uint16_t>{static_cast<uint16_t>(b + 1),
                                               static_cast<uint16_t>(b + 2),
                                               static_cast<uint16_t>(b + 3)}).ok()) return nullptr;
    std::shared_ptr<arrow::Array> d_arr;
    if (!db.Finish(&d_arr).ok()) return nullptr;

    return arrow::RecordBatch::Make(arrowSchema(), 3, {id_arr, s_arr, d_arr});
}

/// Serialize a standard Arrow IPC STREAM: Schema message + `n_blocks` RecordBatch messages + EOS.
std::vector<uint8_t> buildArrowStream(size_t n_blocks)
{
    auto schema = arrowSchema();
    auto out_res = arrow::io::BufferOutputStream::Create();
    if (!out_res.ok()) return {};
    auto out = *out_res;
    auto writer_res = arrow::ipc::MakeStreamWriter(out.get(), schema, arrow::ipc::IpcWriteOptions::Defaults());
    if (!writer_res.ok()) return {};
    auto writer = *writer_res;
    for (size_t b = 0; b < n_blocks; ++b)
    {
        auto batch = makeBatch(b);
        if (!batch || !writer->WriteRecordBatch(*batch).ok()) return {};
    }
    if (!writer->Close().ok()) return {};
    auto buf_res = out->Finish();
    if (!buf_res.ok()) return {};
    auto buf = *buf_res;
    return std::vector<uint8_t>(buf->data(), buf->data() + buf->size());
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

/// End-to-end: an Arrow-IPC-stream producer thread (Arrow C++ reference writer) on loopback; a
/// TcpStreamSource(wire=Arrow) connects, decodes via the COPYING Branch-A path, and emits the columns.
/// Drains through a REAL PullingPipelineExecutor so the async Status::Async/schedule/onAsyncJobReady
/// contract + the resumable 3-phase Arrow recv are exercised. `slow` fragments the stream with delays.
void runArrowDrainTest(bool async, bool slow, bool zero_copy = true)
{
    constexpr size_t n_blocks = 25;
    const std::vector<uint8_t> stream = buildArrowStream(n_blocks);
    ASSERT_FALSE(stream.empty()) << "Arrow stream encode failed";

    int listen_fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(listen_fd, 0);
    int one = 1;
    ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ASSERT_EQ(::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(listen_fd, 4), 0);
    socklen_t alen = sizeof(addr);
    ASSERT_EQ(::getsockname(listen_fd, reinterpret_cast<sockaddr *>(&addr), &alen), 0);
    const uint16_t port = ntohs(addr.sin_port);

    std::thread producer([&]
    {
        int conn = ::accept(listen_fd, nullptr, nullptr);
        if (conn < 0) return;
        if (slow)
            sendFragmented(conn, stream.data(), stream.size(), /*frag=*/17, /*delay_ms=*/2);
        else
            sendAll(conn, stream.data(), stream.size());
        ::close(conn);
    });

    Block b;
    b.insert({std::make_shared<DataTypeUInt64>()->createColumn(), std::make_shared<DataTypeUInt64>(), "id"});
    b.insert({std::make_shared<DataTypeString>()->createColumn(), std::make_shared<DataTypeString>(), "s"});
    b.insert({std::make_shared<DataTypeDate>()->createColumn(), std::make_shared<DataTypeDate>(), "d"});
    auto header = std::make_shared<const Block>(std::move(b));

    auto src = std::make_shared<TcpStreamSource>(
        header, "127.0.0.1", port,
        std::vector<DataTypePtr>{std::make_shared<DataTypeUInt64>(), std::make_shared<DataTypeString>(),
                                 std::make_shared<DataTypeDate>()},
        std::vector<String>{"id", "s", "d"}, std::vector<String>{"id", "s", "d"}, 60'000, async,
        TcpStreamSource::WireFormat::Arrow, zero_copy);

    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);
    size_t chunks = 0;
    Chunk chunk;
    while (executor.pull(chunk))
    {
        if (!chunk.hasRows()) continue;
        ASSERT_EQ(chunk.getNumColumns(), 3u);
        ASSERT_EQ(chunk.getNumRows(), 3u) << "block " << chunks;
        const auto & cols = chunk.getColumns();
        const auto * id_col = typeid_cast<const ColumnUInt64 *>(cols[0].get());
        const auto * s_col = typeid_cast<const ColumnString *>(cols[1].get());
        const auto * d_col = typeid_cast<const ColumnUInt16 *>(cols[2].get());   // DataTypeDate -> ColumnUInt16
        ASSERT_NE(id_col, nullptr);
        ASSERT_NE(s_col, nullptr);
        ASSERT_NE(d_col, nullptr);
        EXPECT_EQ(id_col->getData()[0], chunks);
        EXPECT_EQ(id_col->getData()[1], chunks + 100);
        EXPECT_EQ(id_col->getData()[2], chunks + 200);
        EXPECT_EQ(std::string(s_col->getDataAt(0)), "a");
        EXPECT_EQ(std::string(s_col->getDataAt(1)), "bb");
        EXPECT_EQ(std::string(s_col->getDataAt(2)), "ccc");
        EXPECT_EQ(d_col->getData()[0], static_cast<UInt16>(chunks + 1));
        EXPECT_EQ(d_col->getData()[2], static_cast<UInt16>(chunks + 3));
        ++chunks;
    }
    producer.join();
    ::close(listen_fd);
    ASSERT_EQ(chunks, n_blocks);
}

}


/// Async source, fast producer, ZERO-COPY adoption (Branch B default): whole stream buffered.
TEST(ArrowStreamSource, DrainsSchemaAndBatches)
{
    runArrowDrainTest(/*async=*/true, /*slow=*/false, /*zero_copy=*/true);
}

/// Branch-A COPYING decode (shm_arrow_zero_copy=0) still drains correctly (the A/B baseline).
TEST(ArrowStreamSource, DrainsCopyingDecode)
{
    runArrowDrainTest(/*async=*/true, /*slow=*/false, /*zero_copy=*/false);
}

/// ZERO-COPY adoption across the SLOW fragmented stream — adopted columns alias a body buffer
/// reassembled across schedule cycles; proves the adopt path + resumable recv compose correctly.
TEST(ArrowStreamSource, ZeroCopyResumesAcrossPartialMessages)
{
    runArrowDrainTest(/*async=*/true, /*slow=*/true, /*zero_copy=*/true);
}

/// Blocking source (A/B baseline): the Arrow path drains correctly with SO_RCVTIMEO slicing too.
TEST(ArrowStreamSource, DrainsBlocking)
{
    runArrowDrainTest(/*async=*/false, /*slow=*/false);
}

/// Async source, SLOW fragmented stream: forces non-blocking recv to hit EAGAIN mid-message, so the
/// resumable 3-phase (prefix/metadata/body) Arrow recv + the wake bridge are exercised across schedule
/// cycles — proves a partial Arrow IPC message straddling cycles reassembles correctly.
TEST(ArrowStreamSource, AsyncResumesAcrossPartialMessages)
{
    runArrowDrainTest(/*async=*/true, /*slow=*/true);
}

/// Independent decode oracle: the stock ArrowColumnToCHColumn reader decodes the SAME RecordBatch the
/// custom Branch-A path decodes; both must agree (and agree with the known values). uint16 'd' maps to
/// Date via the header type hint (ArrowColumnToCHColumn.cpp UINT16 case).
TEST(ArrowStreamSource, DecodeMatchesStockArrowReader)
{
    auto batch = makeBatch(7);
    ASSERT_NE(batch, nullptr);
    auto table_res = arrow::Table::FromRecordBatches(arrowSchema(), {batch});
    ASSERT_TRUE(table_res.ok());
    auto table = *table_res;

    Block header;
    header.insert({std::make_shared<DataTypeUInt64>()->createColumn(), std::make_shared<DataTypeUInt64>(), "id"});
    header.insert({std::make_shared<DataTypeString>()->createColumn(), std::make_shared<DataTypeString>(), "s"});
    header.insert({std::make_shared<DataTypeDate>()->createColumn(), std::make_shared<DataTypeDate>(), "d"});

    FormatSettings fmt;
    ArrowColumnToCHColumn converter(
        header, "Arrow", fmt, std::nullopt, std::nullopt,
        /*allow_missing_columns=*/false, /*null_as_default=*/false,
        fmt.date_time_overflow_behavior, /*allow_geoparquet=*/false,
        /*case_insensitive=*/false, /*is_stream=*/false);
    Chunk oracle = converter.arrowTableToCHChunk(table, table->num_rows(), nullptr, nullptr);

    ASSERT_EQ(oracle.getNumColumns(), 3u);
    ASSERT_EQ(oracle.getNumRows(), 3u);
    const auto & cols = oracle.getColumns();
    const auto * id_col = typeid_cast<const ColumnUInt64 *>(cols[0].get());
    const auto * s_col = typeid_cast<const ColumnString *>(cols[1].get());
    const auto * d_col = typeid_cast<const ColumnUInt16 *>(cols[2].get());
    ASSERT_NE(id_col, nullptr);
    ASSERT_NE(s_col, nullptr);
    ASSERT_NE(d_col, nullptr);
    EXPECT_EQ(id_col->getData()[0], 7u);
    EXPECT_EQ(id_col->getData()[1], 107u);
    EXPECT_EQ(id_col->getData()[2], 207u);
    EXPECT_EQ(std::string(s_col->getDataAt(0)), "a");
    EXPECT_EQ(std::string(s_col->getDataAt(2)), "ccc");
    EXPECT_EQ(d_col->getData()[0], 8u);
    EXPECT_EQ(d_col->getData()[2], 10u);
}

#endif

#endif
