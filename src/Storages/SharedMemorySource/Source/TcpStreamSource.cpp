#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/Source/TcpStreamSource.h>

#include <Storages/SharedMemorySource/Adoption/AdoptionLayer.h>
#include <Storages/SharedMemorySource/Adoption/RetainToken.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>
#include <Storages/SharedMemorySource/Wire/TcpFrame.h>
#include <Storages/SharedMemorySource/Wire/WireTypeMapping.h>

#include <Columns/ColumnString.h>
#include <Columns/IColumn.h>
#include <DataTypes/DataTypeFactory.h>

#include <Common/Exception.h>
#include <Common/ErrnoException.h>
#include <Common/ProfileEvents.h>
#include <Common/Stopwatch.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
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

namespace
{
    constexpr int RECV_SLICE_MS = 200;          /// SO_RCVTIMEO slice; lets recv observe cancel/stall.
    constexpr int CONNECT_BUDGET_MS = 5000;     /// total connect budget while the producer comes up.
    constexpr int CONNECT_RETRY_MS = 50;

    void freeAligned(void * p) noexcept { ::free(p); }
}


TcpStreamSource::TcpStreamSource(
    SharedHeader header,
    String host_,
    UInt16 port_,
    std::vector<DataTypePtr> full_column_types_,
    std::vector<String> full_column_names_,
    std::vector<String> requested_column_names_,
    UInt64 stall_timeout_ms_)
    : ISource(std::move(header))
    , host(std::move(host_))
    , port(port_)
    , full_column_types(std::move(full_column_types_))
    , full_column_names(std::move(full_column_names_))
    , requested_column_names(std::move(requested_column_names_))
    , stall_timeout_ms(stall_timeout_ms_)
{
    chassert(full_column_types.size() == full_column_names.size());
}

TcpStreamSource::~TcpStreamSource()
{
    if (sock_fd >= 0)
        ::close(sock_fd);
}


void TcpStreamSource::recvAll(void * dst, size_t n)
{
    auto * p = static_cast<uint8_t *>(dst);
    size_t left = n;
    while (left > 0)
    {
        if (cancelled.load(std::memory_order_acquire) || isCancelled())
            throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}' recv cancelled", host, port);

        const ssize_t r = ::recv(sock_fd, p, left, 0);
        if (r > 0)
        {
            p += r;
            left -= static_cast<size_t>(r);
            stall_timer.restart();   /// progress resets the stall budget (I12 analog)
            continue;
        }
        if (r == 0)
        {
            /// Orderly peer close. Mid-frame this is producer death before EOS (or cancellation
            /// via our own shutdown()). A clean EOS arrives as a TcpBlockHeader with eos_marker=1,
            /// never as a short read, so reaching here with bytes still owed is always a fault.
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
            /// SO_RCVTIMEO slice elapsed with no data: check cancellation + the stall budget.
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
    sock_fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock_fd < 0)
        throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "socket() failed for TCP stream '{}:{}'", host, port);

    /// Size the receive buffer to several blocks so the producer can run ahead while this
    /// (often single-threaded) consumer processes a block — the TCP analog of the SHM ring's K
    /// in-flight slots. Set BEFORE connect so window scaling negotiates the large window. Capped
    /// by net.core.rmem_max (raised to match the SHM 64 MiB data region; see 10-REPRODUCTION).
    const int rcvbuf = 32 * 1024 * 1024;
    ::setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream: bad host '{}'", host);

    /// Connect, retrying ECONNREFUSED/EAGAIN while the producer's listener comes up. The PG side
    /// waits for the worker to be ready before dispatching this query, so connect normally succeeds
    /// immediately; the retry budget covers a small launch race.
    Stopwatch connect_timer;
    while (true)
    {
        if (cancelled.load(std::memory_order_acquire) || isCancelled())
            throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}' cancelled during connect", host, port);
        if (::connect(sock_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0)
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

    /// Recv-timeout slices so recvAll can poll cancellation + the stall budget; TCP_NODELAY to keep
    /// small handshake/EOS frames prompt.
    timeval tv{.tv_sec = RECV_SLICE_MS / 1000, .tv_usec = (RECV_SLICE_MS % 1000) * 1000};
    ::setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    const int one = 1;
    ::setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    stall_timer.restart();

    /// Handshake: header + SchemaEntry[schema_count]; cross-validate against the SQL-declared schema
    /// (preconditions 4-6, identical to the SHM ensureAttached path).
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

    /// Projection map: requested subset → index into the full schema (emit order = request order).
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

    connected = true;
}


Chunk TcpStreamSource::recvBlock()
{
    TcpBlockHeader bh{};
    recvAll(&bh, sizeof(bh));

    if (bh.row_count > IMPL_MAX_ROWS_PER_BLOCK)
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "TCP stream '{}:{}': row_count={} > cap {}", host, port, bh.row_count, IMPL_MAX_ROWS_PER_BLOCK);
    if (bh.payload_len > SharedMemoryWire::TCP_MAX_BLOCK_PAYLOAD)
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "TCP stream '{}:{}': payload_len={} exceeds cap {}", host, port, bh.payload_len,
            SharedMemoryWire::TCP_MAX_BLOCK_PAYLOAD);

    /// EOS frame: no payload, no rows -> stream done.
    if (bh.eos_marker != 0 && bh.payload_len == 0)
    {
        eos_observed = true;
        return {};
    }

    const size_t descs_bytes = full_column_types.size() * sizeof(ColumnDescriptor);
    if (bh.descriptors_offset > bh.payload_len
        || bh.descriptors_offset + descs_bytes > bh.payload_len
        || bh.descriptors_offset % alignof(ColumnDescriptor) != 0)
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "TCP stream '{}:{}': descriptors_offset={} + {} does not fit/align in payload_len={}",
            host, port, bh.descriptors_offset, descs_bytes, bh.payload_len);

    /// Owned recv buffer = the frame-relative data region. 64-byte aligned for Decimal128 (16) +
    /// SIMD-safe-read padding; extra PADDING_FOR_SIMD of slack so any over-read stays in-buffer.
    const size_t cap = ((bh.payload_len + SharedMemoryWire::PADDING_FOR_SIMD + 63) / 64) * 64;
    auto * buffer = static_cast<char *>(::aligned_alloc(64, cap));
    if (buffer == nullptr)
        throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}': aligned_alloc({}) failed", host, port, cap);

    bool buffer_owned_here = true;
    try
    {
        Stopwatch recv_timer;
        recvAll(buffer, bh.payload_len);
        ProfileEvents::increment(ProfileEvents::ShmCopyTimeMicroseconds, recv_timer.elapsedMicroseconds());

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

        /// RetainToken owns the recv buffer: freed on last adopted-alias drop (the TCP analog of the
        /// SHM slot release). Adopt zero-copy over the owned buffer (the recv already did the copy).
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


Chunk TcpStreamSource::generate()
{
    if (cancelled.load(std::memory_order_acquire) || isCancelled())
        return {};
    if (!connected)
        ensureConnected();
    if (eos_observed)
        return {};
    return recvBlock();
}


void TcpStreamSource::onCancel() noexcept
{
    cancelled.store(true, std::memory_order_release);
    /// Unblock any in-progress recv (and the connect retry). SHUT_RDWR makes a blocked recv return
    /// 0 / error promptly; the recvAll loop then observes `cancelled` and stops.
    if (sock_fd >= 0)
        ::shutdown(sock_fd, SHUT_RDWR);
}

}

#endif
