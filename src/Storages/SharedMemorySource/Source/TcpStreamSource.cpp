#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/Source/TcpStreamSource.h>

#include "config.h"   /// USE_ARROW

#include <Storages/SharedMemorySource/Adoption/AdoptionLayer.h>
#include <Storages/SharedMemorySource/Adoption/RetainToken.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>
#include <Storages/SharedMemorySource/Wire/TcpFrame.h>
#include <Storages/SharedMemorySource/Wire/WireTypeMapping.h>

#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnDecimal.h>
#include <Columns/IColumn.h>
#include <DataTypes/DataTypeFactory.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/IDataType.h>

#if USE_ARROW
#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/type.h>
#include <arrow/ipc/dictionary.h>
#include <arrow/ipc/message.h>
#include <arrow/ipc/options.h>
#include <arrow/ipc/reader.h>
/// Branch-B iteration 3 (lean extraction): the arrow-internal flatbuffer-generated headers. This is the
/// same header arrow's own ipc/reader.cc uses to walk a RecordBatch's `buffers()`/`nodes()`; it resolves
/// via the _arrow target's PUBLIC include dir (contrib/arrow/cpp/src) and pulls in generated/Message_generated.h
/// + generated/Schema_generated.h + flatbuffers. We only read the flatbuffer accessors (no arrow::Array build).
#include <arrow/ipc/metadata_internal.h>
#endif

#include <Common/Exception.h>
#include <Common/ErrnoException.h>
#include <Common/ProfileEvents.h>
#include <Common/Stopwatch.h>
#include <Common/assert_cast.h>
#include <Core/Types.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/timerfd.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <thread>


namespace ProfileEvents
{
    extern const Event ShmCopiedBytesLogical;
    extern const Event ShmCopyTimeMicroseconds;
}

namespace DB
{

namespace ErrorCodes
{
    extern const int SHM_SCHEMA_MISMATCH;
    extern const int SHM_BLOCK_FRAMING_INVALID;
    extern const int SHM_PRODUCER_STALL;
    extern const int SHM_PRODUCER_DEATH_BEFORE_EOS;
    extern const int SHM_ATTACH_FAILED;
}

using SharedMemoryWire::SchemaEntry;
using SharedMemoryWire::ColumnDescriptor;
using SharedMemoryWire::TcpHandshakeHeader;
using SharedMemoryWire::TcpBlockHeader;
using SharedMemoryWire::IMPL_MAX_ROWS_PER_BLOCK;

#if USE_ARROW
/// Resumable Arrow IPC recv + decode state (Branch A). Lives behind the pimpl so arrow headers
/// stay out of TcpStreamSource.h. One encapsulated message = an 8-byte prefix (continuation
/// 0xFFFFFFFF + int32 metadata size), then `metadata_size` flatbuffer bytes, then `body_len` body
/// bytes; a 0 metadata size after the continuation is the stream EOS marker.
struct TcpStreamSource::ArrowRecvState
{
    std::shared_ptr<arrow::Schema> schema;
    arrow::ipc::IpcReadOptions read_options = arrow::ipc::IpcReadOptions::Defaults();

    enum class Phase : uint8_t { Prefix, Metadata, Body };
    Phase phase = Phase::Prefix;
    uint8_t prefix[8] = {};
    std::vector<uint8_t> metadata;   /// metadata_size bytes (grown as needed)
    uint32_t metadata_size = 0;
    int64_t body_len = 0;
};
#else
struct TcpStreamSource::ArrowRecvState {};   /// ArrowTcp path errors before use when built w/o Arrow
#endif

namespace
{
    constexpr int RECV_SLICE_MS = 200;          /// SO_RCVTIMEO slice (blocking mode + handshake).
    constexpr int CONNECT_BUDGET_MS = 5000;     /// total connect budget while the producer comes up.
    constexpr int CONNECT_RETRY_MS = 50;

    void freeAligned(void * p) noexcept { ::free(p); }

    /// Allocate the frame-relative recv buffer: 64-byte aligned for Decimal128(16) + SIMD-safe-read
    /// padding; extra PADDING_FOR_SIMD slack so any over-read stays in-buffer.
    char * allocFrameBuffer(size_t payload_len, const String & host, UInt16 port)
    {
        const size_t cap = ((payload_len + SharedMemoryWire::PADDING_FOR_SIMD + 63) / 64) * 64;
        auto * buffer = static_cast<char *>(::aligned_alloc(64, cap));
        if (buffer == nullptr)
            throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}': aligned_alloc({}) failed", host, port, cap);
        return buffer;
    }

    /// Drain an 8-byte-counter fd (the timerfd) to clear its level-triggered readiness; idempotent
    /// (returns EAGAIN when not fired). Branch C1: replaces drainEventFd (the readiness eventfd is gone).
    void drainCounterFd(int fd) noexcept
    {
        if (fd < 0)
            return;
        uint64_t buf = 0;
        ssize_t r;
        do { r = ::read(fd, &buf, sizeof(buf)); } while (r > 0 || (r < 0 && errno == EINTR));
    }

    /// Branch C1 test-hook counters (H15): process-global; see TcpStreamSource.h for semantics.
    std::atomic<uint64_t> g_async_wait_count{0};
    std::atomic<uint64_t> g_threads_spawned{0};   /// bump at ANY std::thread ctor in this TU (post-C1: none)

#if USE_ARROW
    /// Branch-A COPYING decode of one Arrow column into a freshly-allocated, owned ClickHouse column
    /// of the SQL-declared type `type` (the authority — Date/DateTime/DateTime64/Decimal ship as raw
    /// integers / FixedSizeBinary(16), D-HC-0203/0207, so the Arrow logical type is not consulted).
    /// Fixed-width: bulk memcpy of the contiguous LE data buffer (bytes identical to CH storage).
    /// String (Arrow LargeBinary): copy the chars buffer + the N END offsets (arrow_offsets[1..N];
    /// arrow_offsets[0]==0 is CH's offsets[-1] sentinel). Branch B replaces this copy with adoption.
    ColumnPtr copyArrowColumnToCH(const arrow::Array & arr, const DataTypePtr & type,
                                  const String & host, UInt16 port)
    {
        const size_t n = static_cast<size_t>(arr.length());

        if (type->getTypeId() == TypeIndex::String)
        {
            const auto & bin = assert_cast<const arrow::LargeBinaryArray &>(arr);
            const int64_t * offs = bin.raw_value_offsets();   /// length+1 int64, already includes arr.offset()
            auto col = ColumnString::create();
            auto & chars = col->getChars();
            auto & coffs = col->getOffsets();
            const int64_t base = n ? offs[0] : 0;
            const int64_t total = n ? offs[n] - base : 0;
            chars.resize(static_cast<size_t>(total));
            if (total > 0)
                ::memcpy(chars.data(), bin.value_data()->data() + base, static_cast<size_t>(total));
            coffs.resize(n);
            for (size_t i = 0; i < n; ++i)
                coffs[i] = static_cast<UInt64>(offs[i + 1] - base);
            return col;
        }

        /// Fixed-width: the data buffer (buffer index 1) is a contiguous LE array == CH PODArray.
        const auto & data = arr.data();
        if (data->buffers.size() < 2 || !data->buffers[1])
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': Arrow column '{}' missing data buffer", host, port, type->getName());

        MutableColumnPtr mut = type->createColumn();
        const uint8_t * raw = data->buffers[1]->data();

#define PGCH_COPY_FIXED(TINDEX, COLT)                                                               \
        case TypeIndex::TINDEX:                                                                     \
        {                                                                                           \
            auto & c = assert_cast<COLT &>(*mut);                                                   \
            using Elem = std::decay_t<decltype(c.getData()[0])>;                                    \
            c.getData().resize(n);                                                                  \
            if (n) ::memcpy(c.getData().data(), raw + arr.offset() * sizeof(Elem), n * sizeof(Elem)); \
            break;                                                                                  \
        }

        switch (type->getTypeId())
        {
            PGCH_COPY_FIXED(UInt8, ColumnUInt8)
            PGCH_COPY_FIXED(UInt16, ColumnUInt16)
            PGCH_COPY_FIXED(UInt32, ColumnUInt32)
            PGCH_COPY_FIXED(UInt64, ColumnUInt64)
            PGCH_COPY_FIXED(Int8, ColumnInt8)
            PGCH_COPY_FIXED(Int16, ColumnInt16)
            PGCH_COPY_FIXED(Int32, ColumnInt32)
            PGCH_COPY_FIXED(Int64, ColumnInt64)
            PGCH_COPY_FIXED(Float32, ColumnFloat32)
            PGCH_COPY_FIXED(Float64, ColumnFloat64)
            PGCH_COPY_FIXED(Date, ColumnUInt16)
            PGCH_COPY_FIXED(DateTime, ColumnUInt32)
            PGCH_COPY_FIXED(Date32, ColumnInt32)
            PGCH_COPY_FIXED(DateTime64, ColumnDecimal<DateTime64>)
            PGCH_COPY_FIXED(Decimal32, ColumnDecimal<Decimal32>)
            PGCH_COPY_FIXED(Decimal64, ColumnDecimal<Decimal64>)
            PGCH_COPY_FIXED(Decimal128, ColumnDecimal<Decimal128>)
            default:
                throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                    "TCP stream '{}:{}': Arrow decode unsupported CH type '{}'", host, port, type->getName());
        }
#undef PGCH_COPY_FIXED
        return mut;
    }

    /// Adopt a FIXED-WIDTH column from a raw contiguous little-endian data pointer `raw` (n elements) into
    /// a CH column that ALIASES it, sharing `retain`/`charge`. Returns nullptr if not adoptable: a
    /// Decimal128 16-byte-alignment miss (the 8-byte-padded IPC body may not give it), or a CH type outside
    /// the supported fixed-width set. SHARED by the ReadRecordBatch adopt path (adoptArrowColumnToCH) and
    /// the lean direct-flatbuffer extraction (buildChunkFromArrowLean) — one source of truth for the switch.
    /// The +PADDING_FOR_SIMD over-read is memory-safe (the recv body is 64-byte slacked; a mid-body over-read
    /// lands in an adjacent in-allocation buffer).
    ColumnPtr adoptFixedRaw(const DataTypePtr & type, uint8_t * raw, size_t n,
                            const std::shared_ptr<void> & retain, const std::shared_ptr<void> & charge)
    {
        const size_t width = type->getSizeOfValueInMemory();
        if (width >= 16 && (reinterpret_cast<uintptr_t>(raw) % 16) != 0)
            return nullptr;   /// Decimal128 16-byte alignment not met by the 8-byte IPC body → copy fallback

        switch (type->getTypeId())
        {
#define PGCH_ADOPT_VEC(TI, T) case TypeIndex::TI: return ColumnVector<T>::createAdopted(reinterpret_cast<T *>(raw), n, retain, charge);
#define PGCH_ADOPT_DEC(TI, T) case TypeIndex::TI: return ColumnDecimal<T>::createAdopted(reinterpret_cast<T *>(raw), n, getDecimalScale(*type), retain, charge);
            PGCH_ADOPT_VEC(UInt8, UInt8)
            PGCH_ADOPT_VEC(UInt16, UInt16)
            PGCH_ADOPT_VEC(UInt32, UInt32)
            PGCH_ADOPT_VEC(UInt64, UInt64)
            PGCH_ADOPT_VEC(Int8, Int8)
            PGCH_ADOPT_VEC(Int16, Int16)
            PGCH_ADOPT_VEC(Int32, Int32)
            PGCH_ADOPT_VEC(Int64, Int64)
            PGCH_ADOPT_VEC(Float32, Float32)
            PGCH_ADOPT_VEC(Float64, Float64)
            PGCH_ADOPT_VEC(Date, UInt16)
            PGCH_ADOPT_VEC(DateTime, UInt32)
            PGCH_ADOPT_VEC(Date32, Int32)
            PGCH_ADOPT_DEC(DateTime64, DateTime64)
            PGCH_ADOPT_DEC(Decimal32, Decimal32)
            PGCH_ADOPT_DEC(Decimal64, Decimal64)
            PGCH_ADOPT_DEC(Decimal128, Decimal128)
#undef PGCH_ADOPT_VEC
#undef PGCH_ADOPT_DEC
            default: return nullptr;
        }
    }

    /// Adopt a STRING column from raw LargeBinary buffers: `offs` = the int64 offsets buffer (length n+1,
    /// MUST begin with the 0 sentinel that becomes CH `offsets[-1]`); `chars_data` = the values buffer (may
    /// be null only when the column is all-empty). Returns nullptr if the leading-0 sentinel is violated or
    /// chars is unexpectedly null. SHARED by adoptArrowColumnToCH and buildChunkFromArrowLean.
    ColumnPtr adoptStringRaw(const int64_t * offs, const uint8_t * chars_data, size_t n,
                             const std::shared_ptr<void> & retain, const std::shared_ptr<void> & charge)
    {
        if (offs == nullptr || offs[0] != 0)   /// leading-0 sentinel (D-HC-0201/0207) MUST hold
            return nullptr;
        const size_t chars_size = static_cast<size_t>(offs[n]);
        /// CH offsets = &arrow_offsets[1]; CH offsets[-1] reads arrow_offsets[0]==0 (validated).
        auto * ch_offsets = reinterpret_cast<UInt64 *>(const_cast<int64_t *>(offs)) + 1;
        /// chars may be null only when chars_size==0; give createAdopted a non-null in-allocation stand-in
        /// (the offsets buffer) so its pad-right slack stays within the recv body.
        UInt8 * chars = (chars_data != nullptr)
            ? reinterpret_cast<UInt8 *>(const_cast<uint8_t *>(chars_data))
            : reinterpret_cast<UInt8 *>(ch_offsets);
        if (chars_size != 0 && chars_data == nullptr)
            return nullptr;
        return ColumnString::createAdopted(chars, chars_size, ch_offsets, n, retain, charge);
    }

    /// Branch-B ZERO-COPY adoption of one Arrow column (the ReadRecordBatch path): extract the buffer
    /// pointers from `arr` and adopt via adoptFixedRaw/adoptStringRaw (the returned CH column ALIASES the
    /// recv body). Returns nullptr (→ caller copy-falls-back) for an empty/sliced array or a non-adoptable
    /// buffer (Decimal128 alignment, a violated String leading-0 sentinel).
    ColumnPtr adoptArrowColumnToCH(const arrow::Array & arr, const DataTypePtr & type,
                                   const std::shared_ptr<void> & retain, const std::shared_ptr<void> & charge)
    {
        const size_t n = static_cast<size_t>(arr.length());
        if (n == 0 || arr.offset() != 0)
            return nullptr;   /// empty (copy is trivial) or sliced (aliasing assumptions break)

        if (type->getTypeId() == TypeIndex::String)
        {
            const auto & bin = assert_cast<const arrow::LargeBinaryArray &>(arr);
            const auto & data_buf = bin.value_data();
            const uint8_t * chars_data = (data_buf && data_buf->data()) ? data_buf->data() : nullptr;
            return adoptStringRaw(bin.raw_value_offsets(), chars_data, n, retain, charge);
        }

        const auto & data = arr.data();
        if (data->buffers.size() < 2 || !data->buffers[1])
            return nullptr;
        return adoptFixedRaw(type, const_cast<uint8_t *>(data->buffers[1]->data()), n, retain, charge);
    }
#endif
}


TcpStreamSource::TcpStreamSource(
    SharedHeader header,
    String host_,
    UInt16 port_,
    std::vector<DataTypePtr> full_column_types_,
    std::vector<String> full_column_names_,
    std::vector<String> requested_column_names_,
    UInt64 stall_timeout_ms_,
    bool async_,
    WireFormat wire_,
    bool arrow_zero_copy_,
    bool arrow_lean_extract_,
    bool validate_adopted_offsets_)
    : ISource(std::move(header))
    , host(std::move(host_))
    , port(port_)
    , full_column_types(std::move(full_column_types_))
    , full_column_names(std::move(full_column_names_))
    , requested_column_names(std::move(requested_column_names_))
    , stall_timeout_ms(stall_timeout_ms_)
    , async(async_)
    , wire(wire_)
    , arrow_zero_copy(arrow_zero_copy_)
    , arrow_lean_extract(arrow_lean_extract_)
    , validate_adopted_offsets(validate_adopted_offsets_)
{
    chassert(full_column_types.size() == full_column_names.size());
    if (wire == WireFormat::Arrow)
        arrow_state = std::make_unique<ArrowRecvState>();
}

TcpStreamSource::~TcpStreamSource()
{
    if (pending_buf != nullptr)
        freeAligned(pending_buf);
    /// Branch C1: no bridge thread to join. Close the epoll fd first (auto-removes its registrations),
    /// then the timerfd, then the socket. exchange(-1) on sock_fd before close so a concurrent
    /// onCancel() shutdown() (only sock_fd is touched cross-thread) cannot double-act on the fd.
    if (epoll_fd >= 0)
        ::close(epoll_fd);
    if (timerfd >= 0)
        ::close(timerfd);
    const int fd_to_close = sock_fd.exchange(-1, std::memory_order_acq_rel);
    if (fd_to_close >= 0)
        ::close(fd_to_close);
}


void TcpStreamSource::recvAll(void * dst, size_t n)
{
    auto * p = static_cast<uint8_t *>(dst);
    size_t left = n;
    const int fd = sock_fd.load(std::memory_order_acquire);
    while (left > 0)
    {
        if (cancelled.load(std::memory_order_acquire) || isCancelled())
            throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}' recv cancelled", host, port);

        const ssize_t r = ::recv(fd, p, left, 0);
        if (r > 0)
        {
            p += r;
            left -= static_cast<size_t>(r);
            stall_timer.restart();
            continue;
        }
        if (r == 0)
        {
            if (cancelled.load(std::memory_order_acquire))
                throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}' cancelled", host, port);
            throw Exception(ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS,
                "TCP stream '{}:{}': producer closed the connection before end-of-stream "
                "(peer EOF with {} of {} bytes still owed)", host, port, left, n);
        }
        if (errno == EINTR)
            continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
            if (cancelled.load(std::memory_order_acquire) || isCancelled())
                throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}' cancelled", host, port);
            if (stall_timer.elapsedMilliseconds() > stall_timeout_ms)
                throw Exception(ErrorCodes::SHM_PRODUCER_STALL,
                    "TCP stream '{}:{}': no producer progress for {}ms (stall_timeout_ms={})",
                    host, port, stall_timer.elapsedMilliseconds(), stall_timeout_ms);
            continue;
        }
        if (cancelled.load(std::memory_order_acquire))
            throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}' cancelled", host, port);
        throw ErrnoException(ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS,
            "TCP stream '{}:{}' recv failed", host, port);
    }
}


void TcpStreamSource::ensureConnected()
{
    /// Build the socket in a local fd, then publish it to the atomic member so onCancel() can
    /// shutdown() it during connect/handshake; `fd` and `sock_fd` are the same value for this call.
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
        throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "socket() failed for TCP stream '{}:{}'", host, port);
    sock_fd.store(fd, std::memory_order_release);

    /// Size the receive buffer to several blocks so the producer can run ahead (the TCP analog of
    /// the SHM ring's K in-flight slots). Set BEFORE connect so window scaling negotiates the large
    /// window. Capped by net.core.rmem_max (see 10-REPRODUCTION).
    const int rcvbuf = 32 * 1024 * 1024;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream: bad host '{}'", host);

    Stopwatch connect_timer;
    while (true)
    {
        if (cancelled.load(std::memory_order_acquire) || isCancelled())
            throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}' cancelled during connect", host, port);
        if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0)
            break;
        const int e = errno;
        if ((e == ECONNREFUSED || e == EAGAIN || e == EINTR)
            && connect_timer.elapsedMilliseconds() < CONNECT_BUDGET_MS)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(CONNECT_RETRY_MS));
            continue;
        }
        throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "connect('{}:{}') failed", host, port);
    }

    /// Handshake reads use blocking recv with SO_RCVTIMEO slices (one-shot, small frames); the async
    /// streaming phase switches to O_NONBLOCK below.
    timeval tv{.tv_sec = RECV_SLICE_MS / 1000, .tv_usec = (RECV_SLICE_MS % 1000) * 1000};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    const int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    stall_timer.restart();

    if (wire == WireFormat::Arrow)
    {
        /// Arrow wire (D-HC-0207): no bespoke handshake — read the leading Arrow IPC Schema message,
        /// cross-validate the field count, and build the projection map.
        readArrowSchema();
    }
    else
    {
        /// Bespoke handshake: header + SchemaEntry[schema_count]; cross-validate vs the SQL schema.
        TcpHandshakeHeader hs{};
        recvAll(&hs, sizeof(hs));
        if (hs.magic != SharedMemoryWire::SHM_TCP_MAGIC)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': bad handshake magic {:#x}", host, port, hs.magic);
        if (hs.abi_version != SharedMemoryWire::SHM_TCP_ABI_VERSION_1)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': unsupported abi_version {}", host, port, hs.abi_version);
        if (hs.schema_count != full_column_names.size())
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "TCP stream '{}:{}': handshake schema_count={} but SQL columns={}",
                host, port, hs.schema_count, full_column_names.size());

        std::vector<SchemaEntry> schema(hs.schema_count);
        recvAll(schema.data(), schema.size() * sizeof(SchemaEntry));

        auto & type_factory = DataTypeFactory::instance();
        for (size_t i = 0; i < full_column_names.size(); ++i)
        {
            const size_t name_len = ::strnlen(schema[i].name, SharedMemoryWire::SCHEMA_ENTRY_STR_MAX);
            const String producer_name(schema[i].name, name_len);
            if (producer_name != full_column_names[i])
                throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                    "TCP stream '{}:{}' column {}: handshake name='{}' but SQL name='{}'",
                    host, port, i, producer_name, full_column_names[i]);

            const size_t type_len = ::strnlen(schema[i].type_string, SharedMemoryWire::SCHEMA_ENTRY_STR_MAX);
            const String producer_type_str(schema[i].type_string, type_len);
            DataTypePtr producer_type;
            try
            {
                producer_type = type_factory.get(producer_type_str);
            }
            catch (const Exception & e)
            {
                throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                    "TCP stream '{}:{}' column {}: handshake type='{}' does not parse: {}",
                    host, port, i, producer_type_str, e.message());
            }
            if (!producer_type->equals(*full_column_types[i]))
                throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                    "TCP stream '{}:{}' column {}: handshake type='{}' but SQL type='{}'",
                    host, port, i, producer_type->getName(), full_column_types[i]->getName());
            if (!SharedMemoryWire::isSupportedShmType(producer_type->getTypeId()))
                throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                    "TCP stream '{}:{}' column {}: type '{}' outside supported set {}",
                    host, port, i, producer_type->getName(), SharedMemoryWire::supportedShmTypeList());
        }

        buildProjectionIndices();
    }

    if (async)
    {
        /// Branch C1: switch to non-blocking for the resumable recv state machine, then build the
        /// source-owned epoll fd that schedule() returns. Into it register the socket
        /// (EPOLLIN|EPOLLRDHUP|EPOLLERR) and a one-shot timerfd (the stall-budget alarm). Both are
        /// LEVEL-triggered (Epoll.cpp; no EPOLLET), so each wake must drain them (H2). No eventfd / thread.
        int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
            throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}': fcntl(O_NONBLOCK) failed", host, port);

        epoll_fd = ::epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd < 0)
            throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}': epoll_create1() failed", host, port);
        timerfd = ::timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
        if (timerfd < 0)
            throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}': timerfd_create() failed", host, port);

        epoll_event sev{};
        sev.events = EPOLLIN | EPOLLRDHUP | EPOLLERR;
        sev.data.fd = fd;
        if (::epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &sev) < 0)
            throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}': epoll_ctl(ADD sock) failed", host, port);
        epoll_event tev{};
        tev.events = EPOLLIN;
        tev.data.fd = timerfd;
        if (::epoll_ctl(epoll_fd, EPOLL_CTL_ADD, timerfd, &tev) < 0)
            throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}': epoll_ctl(ADD timer) failed", host, port);
    }

    connected = true;
}


Chunk TcpStreamSource::buildChunkFromPayload(char * buffer, const TcpBlockHeader & bh)
{
    bool buffer_owned_here = true;
    try
    {
        const auto * descs = reinterpret_cast<const ColumnDescriptor *>(buffer + bh.descriptors_offset);
        std::vector<ColumnDescriptor> descs_vec(descs, descs + full_column_types.size());

        /// Charge (copied=true: TCP is a copy transport — bumps ShmCopiedBlocks/BytesCharged).
        size_t adopted_bytes = 0, logical_bytes = 0;
        for (const auto & d : descs_vec)
        {
            if (const size_t elem = SharedMemoryWire::wireFixedWidthSize(d.type))
            {
                logical_bytes += d.value_count * elem;
                adopted_bytes += d.value_count * elem + d.value_padding;
            }
            else if (d.type == static_cast<uint32_t>(SharedMemoryWire::WireColumnType::String))
            {
                logical_bytes += d.value_count + d.offsets_count * sizeof(uint64_t);
                adopted_bytes += d.value_count + d.value_padding + sizeof(uint64_t)
                              + d.offsets_count * sizeof(uint64_t) + d.offsets_padding;
            }
        }
        ChargeHandle charge_handle = charger.charge(adopted_bytes, logical_bytes, /*copied=*/true);
        ProfileEvents::increment(ProfileEvents::ShmCopiedBytesLogical, logical_bytes);

        char * buffer_capture = buffer;
        buffer_owned_here = false;
        RetainToken retain_token;
        try
        {
            retain_token = makeRetainToken([buffer_capture]() noexcept { freeAligned(buffer_capture); });
        }
        catch (...)
        {
            buffer_owned_here = true;
            throw;
        }

        std::vector<std::pair<std::string, DataTypePtr>> schema;
        schema.reserve(full_column_types.size());
        for (size_t i = 0; i < full_column_types.size(); ++i)
            schema.emplace_back(full_column_names[i], full_column_types[i]);

        Columns full_cols = adopt(descs_vec, schema, buffer, bh.payload_len, bh.row_count,
                                  std::move(retain_token), std::move(charge_handle));

        Columns emitted_cols;
        emitted_cols.reserve(projection_indices.size());
        for (size_t idx : projection_indices)
            emitted_cols.push_back(full_cols[idx]);
        full_cols.clear();

        if (validate_adopted_offsets)
            for (const auto & col : emitted_cols)
                if (const auto * cs = typeid_cast<const ColumnString *>(col.get()))
                    cs->validateAdoptedOffsets();

        if (bh.eos_marker != 0)
            eos_observed = true;

        stall_timer.restart();
        return Chunk(std::move(emitted_cols), bh.row_count);
    }
    catch (...)
    {
        if (buffer_owned_here)
            freeAligned(buffer);
        throw;
    }
}


void TcpStreamSource::buildProjectionIndices()
{
    /// requested subset → index into the full schema (emit order = request order).
    projection_indices.clear();
    projection_indices.reserve(requested_column_names.size());
    for (const auto & requested : requested_column_names)
    {
        bool found = false;
        for (size_t j = 0; j < full_column_names.size(); ++j)
            if (full_column_names[j] == requested) { projection_indices.push_back(j); found = true; break; }
        if (!found)
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "TCP stream '{}:{}': requested column '{}' not in producer schema", host, port, requested);
    }
}


void TcpStreamSource::readArrowSchema()
{
#if USE_ARROW
    /// Read the leading Arrow IPC encapsulated Schema message (blocking, one-shot): 8-byte prefix
    /// (continuation 0xFFFFFFFF + int32 metadata size), then metadata; Schema messages carry no body.
    uint8_t prefix[8];
    recvAll(prefix, sizeof(prefix));
    uint32_t cont = 0, msize = 0;
    ::memcpy(&cont, prefix, 4);
    ::memcpy(&msize, prefix + 4, 4);
    if (cont != 0xFFFFFFFFu)
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "TCP stream '{}:{}': bad Arrow continuation {:#x}", host, port, cont);
    if (msize == 0)
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "TCP stream '{}:{}': empty Arrow stream (EOS before Schema)", host, port);

    std::vector<uint8_t> meta(msize);
    recvAll(meta.data(), msize);
    auto meta_buf = std::make_shared<arrow::Buffer>(meta.data(), msize);
    auto msg_res = arrow::ipc::Message::Open(meta_buf, nullptr);
    if (!msg_res.ok())
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "TCP stream '{}:{}': Arrow Schema Message::Open failed: {}", host, port, msg_res.status().ToString());
    auto msg = std::move(msg_res).ValueOrDie();
    if (msg->type() != arrow::ipc::MessageType::SCHEMA)
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "TCP stream '{}:{}': expected Arrow Schema message, got type {}", host, port, static_cast<int>(msg->type()));
    if (msg->body_length() > 0)   /// defensive: keep the stream aligned if a body ever appears
    {
        std::vector<uint8_t> dump(static_cast<size_t>(msg->body_length()));
        recvAll(dump.data(), dump.size());
    }

    auto schema_res = arrow::ipc::ReadSchema(*msg, nullptr);
    if (!schema_res.ok())
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "TCP stream '{}:{}': Arrow ReadSchema failed: {}", host, port, schema_res.status().ToString());
    arrow_state->schema = std::move(schema_res).ValueOrDie();

    if (static_cast<size_t>(arrow_state->schema->num_fields()) != full_column_names.size())
        throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
            "TCP stream '{}:{}': Arrow schema has {} fields but SQL columns={}",
            host, port, arrow_state->schema->num_fields(), full_column_names.size());

    /// Per-field layout cross-validation (D-HC-0207). The decode trusts the SQL type (the Arrow type
    /// is intentionally non-semantic for Date/Decimal), so we must confirm the Arrow field's PHYSICAL
    /// layout matches what copyArrowColumnToCH will read: String <-> a variable binary field; every
    /// other (fixed-width) SQL type <-> a fixed-width Arrow field of the SAME byte width. This both
    /// honours the decision and forecloses a width-mismatch heap over-read in the bulk memcpy.
    for (size_t i = 0; i < full_column_types.size(); ++i)
    {
        const auto & arrow_type = arrow_state->schema->field(static_cast<int>(i))->type();
        const bool sql_is_string = full_column_types[i]->getTypeId() == TypeIndex::String;
        const bool arrow_is_binary = arrow_type->id() == arrow::Type::LARGE_BINARY
                                  || arrow_type->id() == arrow::Type::BINARY
                                  || arrow_type->id() == arrow::Type::LARGE_STRING
                                  || arrow_type->id() == arrow::Type::STRING;
        if (sql_is_string != arrow_is_binary)
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "TCP stream '{}:{}' column {} '{}': SQL type '{}' vs Arrow type '{}' "
                "(string/binary mismatch)", host, port, i, full_column_names[i],
                full_column_types[i]->getName(), arrow_type->ToString());
        if (!sql_is_string)
        {
            const auto fw = std::dynamic_pointer_cast<arrow::FixedWidthType>(arrow_type);
            const size_t sql_width = full_column_types[i]->getSizeOfValueInMemory();
            if (!fw || static_cast<size_t>(fw->bit_width()) != sql_width * 8)
                throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                    "TCP stream '{}:{}' column {} '{}': SQL type '{}' ({} bytes) vs Arrow type '{}' "
                    "(width mismatch)", host, port, i, full_column_names[i],
                    full_column_types[i]->getName(), sql_width, arrow_type->ToString());
        }
    }

    buildProjectionIndices();
#else
    throw Exception(ErrorCodes::SHM_ATTACH_FAILED,
        "TCP stream '{}:{}': arrow transport requires a ClickHouse built with Arrow", host, port);
#endif
}


TcpStreamSource::RecvResult TcpStreamSource::tryRecvArrowMessage(Chunk & out_chunk)
{
#if USE_ARROW
    auto & st = *arrow_state;

    if (st.phase == ArrowRecvState::Phase::Prefix)
    {
        const RecvInto r = tryRecvInto(st.prefix, sizeof(st.prefix), recv_filled);
        if (r == RecvInto::WouldBlock)
            return RecvResult::WouldBlock;
        if (r == RecvInto::PeerClosed)
            throw Exception(ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS,
                "TCP stream '{}:{}': producer closed before end-of-stream (peer EOF reading the Arrow "
                "message prefix, {} of 8 bytes)", host, port, recv_filled);
        recv_filled = 0;

        uint32_t cont = 0;
        ::memcpy(&cont, st.prefix, 4);
        ::memcpy(&st.metadata_size, st.prefix + 4, 4);
        if (cont != 0xFFFFFFFFu)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': bad Arrow continuation {:#x}", host, port, cont);
        if (st.metadata_size == 0)
            return RecvResult::Eos;   /// the stream EOS marker (continuation + 0 length)
        if (st.metadata_size > SharedMemoryWire::TCP_MAX_BLOCK_PAYLOAD)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': Arrow metadata size {} exceeds cap", host, port, st.metadata_size);
        st.metadata.resize(st.metadata_size);
        st.phase = ArrowRecvState::Phase::Metadata;
    }

    if (st.phase == ArrowRecvState::Phase::Metadata)
    {
        const RecvInto r = tryRecvInto(st.metadata.data(), st.metadata_size, recv_filled);
        if (r == RecvInto::WouldBlock)
            return RecvResult::WouldBlock;
        if (r == RecvInto::PeerClosed)
            throw Exception(ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS,
                "TCP stream '{}:{}': producer closed before end-of-stream (peer EOF reading Arrow "
                "metadata, {} of {} bytes)", host, port, recv_filled, st.metadata_size);
        recv_filled = 0;

        auto meta_buf = std::make_shared<arrow::Buffer>(st.metadata.data(), st.metadata_size);
        auto msg_res = arrow::ipc::Message::Open(meta_buf, nullptr);
        if (!msg_res.ok())
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': Arrow Message::Open failed: {}", host, port, msg_res.status().ToString());
        st.body_len = std::move(msg_res).ValueOrDie()->body_length();
        if (st.body_len < 0 || static_cast<UInt64>(st.body_len) > SharedMemoryWire::TCP_MAX_BLOCK_PAYLOAD)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': Arrow bodyLength {} out of range", host, port, st.body_len);

        if (st.body_len == 0)
        {
            char * empty = allocFrameBuffer(0, host, port);
            st.phase = ArrowRecvState::Phase::Prefix;
            out_chunk = buildChunkFromArrow(empty, 0);
            return RecvResult::BlockReady;
        }
        pending_payload_len = static_cast<size_t>(st.body_len);
        pending_buf = allocFrameBuffer(pending_payload_len, host, port);
        st.phase = ArrowRecvState::Phase::Body;
    }

    /// Body phase.
    const RecvInto r = tryRecvInto(pending_buf, pending_payload_len, recv_filled);
    if (r == RecvInto::WouldBlock)
        return RecvResult::WouldBlock;
    if (r == RecvInto::PeerClosed)
        throw Exception(ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS,
            "TCP stream '{}:{}': producer closed before end-of-stream (peer EOF reading Arrow body, "
            "{} of {} bytes)", host, port, recv_filled, pending_payload_len);

    char * buf = pending_buf;
    pending_buf = nullptr;
    const int64_t blen = st.body_len;
    st.phase = ArrowRecvState::Phase::Prefix;
    recv_filled = 0;
    out_chunk = buildChunkFromArrow(buf, blen);
    return RecvResult::BlockReady;
#else
    (void) out_chunk;
    throw Exception(ErrorCodes::SHM_ATTACH_FAILED,
        "TCP stream '{}:{}': arrow transport requires a ClickHouse built with Arrow", host, port);
#endif
}


Chunk TcpStreamSource::buildChunkFromArrow(char * body_buf, int64_t body_len)
{
#if USE_ARROW
    /// Branch B iteration 3: the lean direct-flatbuffer extraction handles the common all-adoptable
    /// zero-copy case and delegates anything it cannot adopt back to the ReadRecordBatch path below.
    if (arrow_zero_copy && arrow_lean_extract)
        return buildChunkFromArrowLean(body_buf, body_len);
    return buildChunkFromArrowViaReader(body_buf, body_len);
#else
    (void) body_len;
    freeAligned(body_buf);
    throw Exception(ErrorCodes::SHM_ATTACH_FAILED,
        "TCP stream '{}:{}': arrow transport requires a ClickHouse built with Arrow", host, port);
#endif
}


Chunk TcpStreamSource::buildChunkFromArrowViaReader(char * body_buf, int64_t body_len)
{
#if USE_ARROW
    bool body_owned_here = true;   /// false once a RetainToken takes ownership of body_buf (adopt path)
    try
    {
        auto & st = *arrow_state;
        auto meta_buf = std::make_shared<arrow::Buffer>(st.metadata.data(), st.metadata_size);
        auto body_buffer = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t *>(body_buf), body_len);

        auto msg_res = arrow::ipc::Message::Open(meta_buf, body_buffer);
        if (!msg_res.ok())
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': Arrow Message::Open(body) failed: {}", host, port, msg_res.status().ToString());
        auto msg = std::move(msg_res).ValueOrDie();
        if (msg->type() != arrow::ipc::MessageType::RECORD_BATCH)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': expected Arrow RecordBatch, got type {}", host, port, static_cast<int>(msg->type()));

        auto batch_res = arrow::ipc::ReadRecordBatch(*msg, st.schema, /*dictionary_memo=*/nullptr, st.read_options);
        if (!batch_res.ok())
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': Arrow ReadRecordBatch failed: {}", host, port, batch_res.status().ToString());
        auto batch = std::move(batch_res).ValueOrDie();

        const size_t n_cols = full_column_types.size();
        if (static_cast<size_t>(batch->num_columns()) != n_cols)
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "TCP stream '{}:{}': Arrow batch has {} columns but schema {}", host, port, batch->num_columns(), n_cols);
        const size_t n_rows = static_cast<size_t>(batch->num_rows());
        Columns full_cols(n_cols);

        if (!arrow_zero_copy)
        {
            /// Branch A — COPYING decode: each Arrow column → an owned CH column of the SQL type.
            size_t logical_bytes = 0;
            for (size_t i = 0; i < n_cols; ++i)
            {
                full_cols[i] = copyArrowColumnToCH(*batch->column(static_cast<int>(i)), full_column_types[i], host, port);
                logical_bytes += full_cols[i]->byteSize();
            }
            batch.reset();   /// drop the arrays that view body_buf BEFORE freeing it below
            ChargeHandle charge_handle = charger.charge(logical_bytes, logical_bytes, /*copied=*/true);
            ProfileEvents::increment(ProfileEvents::ShmCopiedBytesLogical, logical_bytes);

            Columns emitted_copy;
            emitted_copy.reserve(projection_indices.size());
            for (size_t idx : projection_indices)
                emitted_copy.push_back(full_cols[idx]);
            stall_timer.restart();
            freeAligned(body_buf);
            return Chunk(std::move(emitted_copy), n_rows);
        }

        /// Branch B — ZERO-COPY adoption: the CH columns ALIAS the Arrow buffers (slices of body_buf),
        /// held alive by a single RetainToken (deleter frees body_buf on last-drop) shared across the
        /// adopted columns + a shared ChargeHandle. body_buf is NOT freed here — the token owns it (it
        /// drops at function end iff no column adopted, i.e. all copy-fell-back, freeing body_buf then).
        /// Decimal128 / a misaligned buffer falls back to a per-column copy (it does not hold the token).
        char * body_capture = body_buf;
        body_owned_here = false;
        RetainToken retain_token = makeRetainToken([body_capture]() noexcept { freeAligned(body_capture); });
        ChargeHandle charge_handle = charger.charge(static_cast<size_t>(body_len), static_cast<size_t>(body_len), /*copied=*/true);
        ProfileEvents::increment(ProfileEvents::ShmCopiedBytesLogical, static_cast<size_t>(body_len));
        auto charge_shared = std::make_shared<ChargeHandle>(std::move(charge_handle));
        std::shared_ptr<void> charge_token = charge_shared;

        for (size_t i = 0; i < n_cols; ++i)
        {
            const auto & col = *batch->column(static_cast<int>(i));
            ColumnPtr c = adoptArrowColumnToCH(col, full_column_types[i], retain_token, charge_token);
            if (!c)   /// not adoptable at the required alignment (e.g. Decimal128) → copy fallback
                c = copyArrowColumnToCH(col, full_column_types[i], host, port);
            full_cols[i] = std::move(c);
        }
        batch.reset();   /// arrow arrays were non-owning views of body_buf; adopted CH cols alias it via retain_token

        Columns emitted_cols;
        emitted_cols.reserve(projection_indices.size());
        for (size_t idx : projection_indices)
            emitted_cols.push_back(full_cols[idx]);
        if (validate_adopted_offsets)
            for (const auto & ec : emitted_cols)
                if (const auto * cs = typeid_cast<const ColumnString *>(ec.get()))
                    cs->validateAdoptedOffsets();

        stall_timer.restart();
        return Chunk(std::move(emitted_cols), n_rows);
    }
    catch (...)
    {
        if (body_owned_here)
            freeAligned(body_buf);
        throw;
    }
#else
    (void) body_len;
    freeAligned(body_buf);
    throw Exception(ErrorCodes::SHM_ATTACH_FAILED,
        "TCP stream '{}:{}': arrow transport requires a ClickHouse built with Arrow", host, port);
#endif
}


Chunk TcpStreamSource::buildChunkFromArrowLean(char * body_buf, int64_t body_len)
{
#if USE_ARROW
    /// Branch B iteration 3 — LEAN extraction. The metadata flatbuffer (`arrow_state.metadata`) was already
    /// recv'd AND validated by the metadata-phase arrow::ipc::Message::Open (tryRecvArrowMessage), so we
    /// parse it directly here and adopt each buffer at `body_buf + flatbuf::Buffer.offset()` — NO second
    /// Message::Open, NO arrow::ipc::ReadRecordBatch, NO per-column arrow::Array/ArrayData/Buffer-slice
    /// objects. The buffer-tree walk mirrors arrow's own ipc/reader.cc ArrayLoader exactly for our FLAT,
    /// non-Nullable, primitive/LargeBinary schema: a fixed-width field consumes [validity, data] (advance 2),
    /// a LargeBinary field consumes [validity, offsets, data] (advance 3). ANY column we cannot adopt
    /// (Decimal128 alignment, a sliced/empty/sentinel-violating batch, an out-of-bounds buffer, an
    /// unexpected layout) makes the WHOLE block fall back to the ReadRecordBatch path (body_buf untouched
    /// until we commit), which keeps the per-column copy-fallback — so correctness is never lost.
    namespace flatbuf = org::apache::arrow::flatbuf;
    auto & st = *arrow_state;
    const size_t n_cols = full_column_types.size();

    const flatbuf::Message * msg = flatbuf::GetMessage(st.metadata.data());
    const flatbuf::RecordBatch * rb = msg ? msg->header_as_RecordBatch() : nullptr;
    if (rb == nullptr)
        return buildChunkFromArrowViaReader(body_buf, body_len);   /// not a RecordBatch → safe path
    const auto * fb_buffers = rb->buffers();
    const auto * fb_nodes = rb->nodes();
    if (fb_buffers == nullptr || fb_nodes == nullptr || static_cast<size_t>(fb_nodes->size()) != n_cols)
        return buildChunkFromArrowViaReader(body_buf, body_len);
    const int64_t n_rows64 = rb->length();
    if (n_rows64 <= 0)
        return buildChunkFromArrowViaReader(body_buf, body_len);   /// empty batch → reader builds empty cols
    const size_t n = static_cast<size_t>(n_rows64);
    const size_t nbuf = static_cast<size_t>(fb_buffers->size());

    /// Pass 1: compute each column's raw pointers from the flatbuffer + check adoptability + bounds.
    /// Bail to the ReadRecordBatch path on ANY non-adoptable / structurally-suspicious column — BEFORE we
    /// take ownership of body_buf, so the delegate gets an untouched buffer.
    struct ColPtrs { bool is_string; uint8_t * data; const int64_t * offs; };
    std::vector<ColPtrs> cols(n_cols);
    int bi = 0;
    for (size_t i = 0; i < n_cols; ++i)
    {
        if (static_cast<int64_t>(fb_nodes->Get(static_cast<flatbuffers::uoffset_t>(i))->length()) != n_rows64)
            return buildChunkFromArrowViaReader(body_buf, body_len);   /// per-field row count disagrees → suspicious

        const auto & type = full_column_types[i];
        if (type->getTypeId() == TypeIndex::String)
        {
            if (static_cast<size_t>(bi) + 3 > nbuf)
                return buildChunkFromArrowViaReader(body_buf, body_len);
            const flatbuf::Buffer * offs_b = fb_buffers->Get(static_cast<flatbuffers::uoffset_t>(bi + 1));
            const flatbuf::Buffer * data_b = fb_buffers->Get(static_cast<flatbuffers::uoffset_t>(bi + 2));
            bi += 3;
            const int64_t off_o = offs_b->offset(), len_o = offs_b->length();
            const int64_t off_d = data_b->offset(), len_d = data_b->length();
            if (off_o < 0 || len_o < static_cast<int64_t>((n + 1) * sizeof(int64_t)) || off_o + len_o > body_len)
                return buildChunkFromArrowViaReader(body_buf, body_len);
            const int64_t * offs = reinterpret_cast<const int64_t *>(body_buf + off_o);
            if (offs[0] != 0)   /// leading-0 sentinel (D-HC-0201/0207) MUST hold for the &offs[1] alias
                return buildChunkFromArrowViaReader(body_buf, body_len);
            const int64_t chars_size = offs[n];
            if (chars_size < 0)
                return buildChunkFromArrowViaReader(body_buf, body_len);
            uint8_t * chars = (len_d > 0) ? reinterpret_cast<uint8_t *>(body_buf + off_d) : nullptr;
            if (chars_size != 0 && (chars == nullptr || off_d < 0 || off_d + chars_size > body_len))
                return buildChunkFromArrowViaReader(body_buf, body_len);
            cols[i] = ColPtrs{true, chars, offs};
        }
        else
        {
            if (static_cast<size_t>(bi) + 2 > nbuf)
                return buildChunkFromArrowViaReader(body_buf, body_len);
            const flatbuf::Buffer * data_b = fb_buffers->Get(static_cast<flatbuffers::uoffset_t>(bi + 1));
            bi += 2;
            const int64_t off_d = data_b->offset(), len_d = data_b->length();
            const size_t width = type->getSizeOfValueInMemory();
            if (off_d < 0 || off_d + len_d > body_len || len_d < static_cast<int64_t>(n * width))
                return buildChunkFromArrowViaReader(body_buf, body_len);
            uint8_t * raw = reinterpret_cast<uint8_t *>(body_buf + off_d);
            if (width >= 16 && (reinterpret_cast<uintptr_t>(raw) % 16) != 0)
                return buildChunkFromArrowViaReader(body_buf, body_len);   /// Decimal128 misaligned → reader copy-fallback
            cols[i] = ColPtrs{false, raw, nullptr};
        }
    }

    /// Pass 2: all columns adoptable. Take ownership of body_buf via one RetainToken (deleter frees it on
    /// last-drop) shared across the adopted columns + a shared ChargeHandle, then adopt each in place.
    char * body_capture = body_buf;
    RetainToken retain_token = makeRetainToken([body_capture]() noexcept { freeAligned(body_capture); });
    ChargeHandle charge_handle = charger.charge(static_cast<size_t>(body_len), static_cast<size_t>(body_len), /*copied=*/true);
    ProfileEvents::increment(ProfileEvents::ShmCopiedBytesLogical, static_cast<size_t>(body_len));
    auto charge_shared = std::make_shared<ChargeHandle>(std::move(charge_handle));
    std::shared_ptr<void> charge_token = charge_shared;

    Columns full_cols(n_cols);
    for (size_t i = 0; i < n_cols; ++i)
    {
        ColumnPtr c = cols[i].is_string
            ? adoptStringRaw(cols[i].offs, cols[i].data, n, retain_token, charge_token)
            : adoptFixedRaw(full_column_types[i], cols[i].data, n, retain_token, charge_token);
        /// Pass 1 already proved every column adoptable, so this is unreachable in practice; defend anyway
        /// (RAII-safe: retain_token + any adopted cols unwind on throw, freeing body_buf exactly once).
        if (!c)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': lean Arrow extraction could not adopt column {} ('{}')",
                host, port, i, full_column_types[i]->getName());
        full_cols[i] = std::move(c);
    }

    Columns emitted_cols;
    emitted_cols.reserve(projection_indices.size());
    for (size_t idx : projection_indices)
        emitted_cols.push_back(full_cols[idx]);
    if (validate_adopted_offsets)
        for (const auto & ec : emitted_cols)
            if (const auto * cs = typeid_cast<const ColumnString *>(ec.get()))
                cs->validateAdoptedOffsets();

    stall_timer.restart();
    return Chunk(std::move(emitted_cols), n);
#else
    (void) body_len;
    freeAligned(body_buf);
    throw Exception(ErrorCodes::SHM_ATTACH_FAILED,
        "TCP stream '{}:{}': arrow transport requires a ClickHouse built with Arrow", host, port);
#endif
}


namespace
{
    /// Validate a just-received TcpBlockHeader; returns true if it is the EOS frame (no payload).
    bool validateBlockHeader(const TcpBlockHeader & bh, size_t n_columns, const String & host, UInt16 port)
    {
        if (bh.row_count > IMPL_MAX_ROWS_PER_BLOCK)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': row_count={} > cap {}", host, port, bh.row_count, IMPL_MAX_ROWS_PER_BLOCK);
        if (bh.payload_len > SharedMemoryWire::TCP_MAX_BLOCK_PAYLOAD)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': payload_len={} exceeds cap {}", host, port, bh.payload_len,
                SharedMemoryWire::TCP_MAX_BLOCK_PAYLOAD);

        if (bh.eos_marker != 0 && bh.payload_len == 0)
            return true;

        const size_t descs_bytes = n_columns * sizeof(ColumnDescriptor);
        if (bh.descriptors_offset > bh.payload_len
            || bh.descriptors_offset + descs_bytes > bh.payload_len
            || bh.descriptors_offset % alignof(ColumnDescriptor) != 0)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "TCP stream '{}:{}': descriptors_offset={} + {} does not fit/align in payload_len={}",
                host, port, bh.descriptors_offset, descs_bytes, bh.payload_len);
        return false;
    }
}


Chunk TcpStreamSource::recvBlockBlocking()
{
    TcpBlockHeader bh{};
    recvAll(&bh, sizeof(bh));

    if (validateBlockHeader(bh, full_column_types.size(), host, port))
    {
        eos_observed = true;
        return {};
    }

    char * buffer = allocFrameBuffer(bh.payload_len, host, port);
    Stopwatch recv_timer;
    try
    {
        recvAll(buffer, bh.payload_len);
    }
    catch (...)
    {
        freeAligned(buffer);
        throw;
    }
    ProfileEvents::increment(ProfileEvents::ShmCopyTimeMicroseconds, recv_timer.elapsedMicroseconds());
    return buildChunkFromPayload(buffer, bh);
}


TcpStreamSource::RecvInto TcpStreamSource::tryRecvInto(void * dst, size_t need, size_t & filled)
{
    auto * p = static_cast<uint8_t *>(dst);
    const int fd = sock_fd.load(std::memory_order_acquire);
    Stopwatch recv_timer;
    while (filled < need)
    {
        const ssize_t r = ::recv(fd, p + filled, need - filled, 0);
        if (r > 0)
        {
            filled += static_cast<size_t>(r);
            stall_timer.restart();
            continue;
        }
        if (r == 0)
        {
            ProfileEvents::increment(ProfileEvents::ShmCopyTimeMicroseconds, recv_timer.elapsedMicroseconds());
            return RecvInto::PeerClosed;
        }
        if (errno == EINTR)
            continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK)
        {
            ProfileEvents::increment(ProfileEvents::ShmCopyTimeMicroseconds, recv_timer.elapsedMicroseconds());
            return RecvInto::WouldBlock;
        }
        ProfileEvents::increment(ProfileEvents::ShmCopyTimeMicroseconds, recv_timer.elapsedMicroseconds());
        throw ErrnoException(ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS,
            "TCP stream '{}:{}' recv failed", host, port);
    }
    ProfileEvents::increment(ProfileEvents::ShmCopyTimeMicroseconds, recv_timer.elapsedMicroseconds());
    return RecvInto::Complete;
}


TcpStreamSource::RecvResult TcpStreamSource::tryRecvBlock(Chunk & out_chunk)
{
    if (recv_phase == RecvPhase::Header)
    {
        const RecvInto r = tryRecvInto(&cur_bh, sizeof(cur_bh), recv_filled);
        if (r == RecvInto::WouldBlock)
            return RecvResult::WouldBlock;
        if (r == RecvInto::PeerClosed)
            throw Exception(ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS,
                "TCP stream '{}:{}': producer closed the connection before end-of-stream "
                "(peer EOF with {} of {} header bytes received)", host, port, recv_filled, sizeof(cur_bh));

        /// Header complete.
        recv_filled = 0;
        if (validateBlockHeader(cur_bh, full_column_types.size(), host, port))
            return RecvResult::Eos;

        pending_payload_len = cur_bh.payload_len;
        pending_buf = allocFrameBuffer(pending_payload_len, host, port);
        recv_phase = RecvPhase::Payload;
    }

    /// Payload phase.
    const RecvInto r = tryRecvInto(pending_buf, pending_payload_len, recv_filled);
    if (r == RecvInto::WouldBlock)
        return RecvResult::WouldBlock;
    if (r == RecvInto::PeerClosed)
        throw Exception(ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS,
            "TCP stream '{}:{}': producer closed the connection before end-of-stream "
            "(peer EOF with {} of {} payload bytes received)", host, port, recv_filled, pending_payload_len);

    /// Payload complete: hand the buffer to buildChunkFromPayload (it takes ownership).
    char * buf = pending_buf;
    pending_buf = nullptr;
    const TcpBlockHeader bh = cur_bh;
    recv_phase = RecvPhase::Header;
    recv_filled = 0;
    out_chunk = buildChunkFromPayload(buf, bh);
    return RecvResult::BlockReady;
}


std::optional<Chunk> TcpStreamSource::tryGenerate()
{
    if (cancelled.load(std::memory_order_acquire) || isCancelled())
        return {};
    if (!connected)
        ensureConnected();

    /// If we were async-waiting, leave that state now — BEFORE any early return, so even the EOS/done
    /// path resets it (else prepare() below would keep returning Async forever). The single-threaded
    /// PullingPipelineExecutor runs work() directly on a ready async node and NEVER calls
    /// onAsyncJobReady (that is only the multi-threaded processAsyncTasks monitor's job), so we MUST
    /// drain the timerfd here to clear its level-triggered readiness in the epoll fd, or the epoll fd
    /// stays readable and the executor hot-spins (H2). The recv loop below then drives the socket to
    /// EAGAIN, clearing the socket's readiness too; only then may we re-arm + re-park. Idempotent with
    /// onAsyncJobReady (drainCounterFd no-ops when the timerfd has not fired).
    if (is_async_state)
    {
        drainCounterFd(timerfd);
        is_async_state = false;
    }

    if (eos_observed)
        return {};

    if (!async)
    {
        if (wire == WireFormat::Arrow)
        {
            /// Blocking Arrow: the socket has SO_RCVTIMEO slices (no O_NONBLOCK), so tryRecvInto may
            /// report WouldBlock on a timeout slice with no data — loop, honouring cancel + the stall
            /// budget, until a full message or EOS.
            Chunk c;
            for (;;)
            {
                const RecvResult r = tryRecvArrowMessage(c);
                if (r == RecvResult::BlockReady)
                    return c;
                if (r == RecvResult::Eos)
                {
                    eos_observed = true;
                    return {};
                }
                if (cancelled.load(std::memory_order_acquire) || isCancelled())
                    return {};
                if (stall_timer.elapsedMilliseconds() > stall_timeout_ms)
                    throw Exception(ErrorCodes::SHM_PRODUCER_STALL,
                        "TCP stream '{}:{}': no producer progress for {}ms (stall_timeout_ms={})",
                        host, port, stall_timer.elapsedMilliseconds(), stall_timeout_ms);
            }
        }
        Chunk c = recvBlockBlocking();
        if (!c)
            return {};   /// EOS frame (empty Chunk) → ISource marks finished
        return c;
    }

    /// Async: resumable non-blocking recv. A full block → emit it; partial → go async; EOS → finish.
    Chunk c;
    const RecvResult r = (wire == WireFormat::Arrow) ? tryRecvArrowMessage(c) : tryRecvBlock(c);
    if (r == RecvResult::BlockReady)
        return c;
    if (r == RecvResult::Eos)
    {
        eos_observed = true;
        return {};
    }

    /// WouldBlock: no full frame available yet. Enforce the stall budget here (the bridge wakes us on
    /// the stall deadline; if we still have no progress, fail). Otherwise yield async.
    if (stall_timer.elapsedMilliseconds() > stall_timeout_ms)
        throw Exception(ErrorCodes::SHM_PRODUCER_STALL,
            "TCP stream '{}:{}': no producer progress for {}ms (stall_timeout_ms={})",
            host, port, stall_timer.elapsedMilliseconds(), stall_timeout_ms);

    is_async_state = true;
    g_async_wait_count.fetch_add(1, std::memory_order_relaxed);   /// H15 test hook: an async park happened
    armStallTimer();   /// re-arm the one-shot stall alarm; socket is already EAGAIN (drained above, H2)
    return Chunk{};    /// non-EOS empty Chunk → ISource yields; prepare() will return Async
}


ISource::Status TcpStreamSource::prepare()
{
    if (cancelled.load(std::memory_order_acquire) || isCancelled())
    {
        cancelled.store(true, std::memory_order_release);
        return ISource::prepare();
    }
    /// Only park in Async while we still have work to do. Guarding with !finished && !eos_observed
    /// ensures a stale is_async_state can never mask the base Finished status (which would loop forever).
    if (async && is_async_state && !finished && !eos_observed)
        return Status::Async;
    return ISource::prepare();
}


int TcpStreamSource::schedule()
{
    /// Branch C1: the source-owned epoll fd aggregating {sock_fd, timerfd}. The executor registers it
    /// LEVEL-triggered (Epoll.cpp; scheduleForEvent default {schedule(), EPOLLIN|EPOLLERR}); it is
    /// EPOLLIN-readable whenever the socket has data/RDHUP/err OR the stall timerfd has fired. H2: the
    /// wake handler (tryGenerate early-path / onAsyncJobReady) drains BOTH before re-returning Async.
    chassert(epoll_fd >= 0);
    return epoll_fd;
}


void TcpStreamSource::onAsyncJobReady()
{
    /// Multi-threaded executor wake hook (the single-threaded PullingPipelineExecutor never calls this —
    /// tryGenerate's early-path does the equivalent). Clear the timerfd's level-triggered readiness and
    /// leave the async state so the next prepare() returns Ready → work() → tryGenerate() drives recv to
    /// EAGAIN (clearing the socket readiness). Idempotent with the tryGenerate early-path (H2).
    drainCounterFd(timerfd);
    is_async_state = false;
}


void TcpStreamSource::onCancel() noexcept
{
    cancelled.store(true, std::memory_order_release);
    /// Branch C1: cancellation is SOLELY `::shutdown(sock_fd, SHUT_RDWR)`. When async-parked the socket is
    /// registered in the epoll fd, so shutdown makes it readable/RDHUP → the epoll fd fires → the executor
    /// wakes → tryGenerate observes `cancelled` and finishes. During connect/handshake the executor is
    /// inside ensureConnected (NOT parked on the epoll fd, which does not exist yet); shutdown makes the
    /// blocking handshake recv return promptly and the connect/recv loops poll `cancelled` each slice.
    const int fd = sock_fd.load(std::memory_order_acquire);
    if (fd >= 0)
        ::shutdown(fd, SHUT_RDWR);
}


void TcpStreamSource::armStallTimer() noexcept
{
    if (timerfd < 0)
        return;
    /// One-shot alarm at the remaining stall budget. We only reach here after tryGenerate's stall check
    /// confirmed elapsed <= stall_timeout_ms, so remaining > 0; floor at 1ms so an all-zero itimerspec can
    /// never DISARM the timer — which would leave the single-threaded executor's async_task_queue.wait(-1)
    /// with no wakeup if the producer then stalls (a hang). it_interval stays 0 → one-shot.
    const uint64_t elapsed = stall_timer.elapsedMilliseconds();
    const uint64_t remaining = (elapsed >= stall_timeout_ms) ? 1 : (stall_timeout_ms - elapsed);
    itimerspec its{};
    its.it_value.tv_sec = static_cast<time_t>(remaining / 1000);
    its.it_value.tv_nsec = static_cast<long>((remaining % 1000) * 1000000L);
    (void) ::timerfd_settime(timerfd, 0, &its, nullptr);   /// failure is non-fatal: socket data still wakes us
}


uint64_t TcpStreamSource::asyncWaitCount() noexcept { return g_async_wait_count.load(std::memory_order_relaxed); }
uint64_t TcpStreamSource::threadsSpawned() noexcept { return g_threads_spawned.load(std::memory_order_relaxed); }
void TcpStreamSource::resetAsyncCounters() noexcept
{
    g_async_wait_count.store(0, std::memory_order_relaxed);
    g_threads_spawned.store(0, std::memory_order_relaxed);
}

}

#endif
