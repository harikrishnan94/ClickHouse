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
#include <sys/eventfd.h>
#include <poll.h>
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

    void drainEventFd(int fd) noexcept
    {
        if (fd < 0)
            return;
        uint64_t buf = 0;
        ssize_t r;
        do { r = ::read(fd, &buf, sizeof(buf)); } while (r > 0 || (r < 0 && errno == EINTR));
    }
}


TcpStreamSource::TcpStreamSource(
    SharedHeader header,
    String host_,
    UInt16 port_,
    std::vector<DataTypePtr> full_column_types_,
    std::vector<String> full_column_names_,
    std::vector<String> requested_column_names_,
    UInt64 stall_timeout_ms_,
    bool async_)
    : ISource(std::move(header))
    , host(std::move(host_))
    , port(port_)
    , full_column_types(std::move(full_column_types_))
    , full_column_names(std::move(full_column_names_))
    , requested_column_names(std::move(requested_column_names_))
    , stall_timeout_ms(stall_timeout_ms_)
    , async(async_)
{
    chassert(full_column_types.size() == full_column_names.size());
}

TcpStreamSource::~TcpStreamSource()
{
    requestAsyncWakeBridgeStop();
    joinAsyncWakeBridge();
    if (pending_buf != nullptr)
        freeAligned(pending_buf);
    if (sock_fd >= 0)
        ::close(sock_fd);
    if (ready_event_fd >= 0)
        ::close(ready_event_fd);
    if (async_wake_stop_fd >= 0)
        ::close(async_wake_stop_fd);
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
    sock_fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock_fd < 0)
        throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "socket() failed for TCP stream '{}:{}'", host, port);

    /// Size the receive buffer to several blocks so the producer can run ahead (the TCP analog of
    /// the SHM ring's K in-flight slots). Set BEFORE connect so window scaling negotiates the large
    /// window. Capped by net.core.rmem_max (see 10-REPRODUCTION).
    const int rcvbuf = 32 * 1024 * 1024;
    ::setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

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

    /// Handshake reads use blocking recv with SO_RCVTIMEO slices (one-shot, small frames); the async
    /// streaming phase switches to O_NONBLOCK below.
    timeval tv{.tv_sec = RECV_SLICE_MS / 1000, .tv_usec = (RECV_SLICE_MS % 1000) * 1000};
    ::setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    const int one = 1;
    ::setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    stall_timer.restart();

    /// Handshake: header + SchemaEntry[schema_count]; cross-validate against the SQL-declared schema.
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

    if (async)
    {
        /// Switch to non-blocking for the resumable recv state machine, and create the readiness
        /// eventfd (schedule() returns it) + the bridge stop eventfd.
        int flags = ::fcntl(sock_fd, F_GETFL, 0);
        if (flags < 0 || ::fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK) < 0)
            throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}': fcntl(O_NONBLOCK) failed", host, port);
        ready_event_fd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (ready_event_fd < 0)
            throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}': eventfd() failed", host, port);
        async_wake_stop_fd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (async_wake_stop_fd < 0)
            throw ErrnoException(ErrorCodes::SHM_ATTACH_FAILED, "TCP stream '{}:{}': stop eventfd() failed", host, port);
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
    Stopwatch recv_timer;
    while (filled < need)
    {
        const ssize_t r = ::recv(sock_fd, p + filled, need - filled, 0);
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
    /// drain the readiness eventfd + stop the one-shot bridge here, or the level-triggered fd stays
    /// readable and the executor hot-spins. Idempotent with onAsyncJobReady (the guard no-ops it).
    if (is_async_state)
    {
        drainEventFd(ready_event_fd);
        requestAsyncWakeBridgeStop();
        is_async_state = false;
    }

    if (eos_observed)
        return {};

    if (!async)
    {
        Chunk c = recvBlockBlocking();
        if (!c)
            return {};   /// EOS frame (empty Chunk) → ISource marks finished
        return c;
    }

    /// Async: resumable non-blocking recv. A full block → emit it; partial → go async; EOS → finish.
    Chunk c;
    const RecvResult r = tryRecvBlock(c);
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
    startAsyncWakeBridge();
    return Chunk{};   /// non-EOS empty Chunk → ISource yields; prepare() will return Async
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
    chassert(ready_event_fd >= 0);
    return ready_event_fd;
}


void TcpStreamSource::onAsyncJobReady()
{
    /// Drain the readiness eventfd and hand control back immediately; stop the (one-shot) bridge and
    /// leave the async state so the next prepare() returns Ready → work() → tryGenerate() resumes recv.
    drainEventFd(ready_event_fd);
    requestAsyncWakeBridgeStop();
    is_async_state = false;
}


void TcpStreamSource::onCancel() noexcept
{
    cancelled.store(true, std::memory_order_release);
    /// Wake any executor wait on the readiness fd, stop the bridge, and unblock a pending blocking
    /// handshake recv (SHUT_RDWR makes recv return promptly; the recv loops then observe `cancelled`).
    wakeReadyEvent();
    requestAsyncWakeBridgeStop();
    if (sock_fd >= 0)
        ::shutdown(sock_fd, SHUT_RDWR);
}


void TcpStreamSource::wakeReadyEvent() const noexcept
{
    if (ready_event_fd < 0)
        return;
    const uint64_t one = 1;
    ssize_t w;
    do { w = ::write(ready_event_fd, &one, sizeof(one)); } while (w < 0 && errno == EINTR);
}


void TcpStreamSource::requestAsyncWakeBridgeStop() noexcept
{
    async_wake_bridge_stop.store(true, std::memory_order_release);
    if (async_wake_stop_fd < 0)
        return;
    const uint64_t one = 1;
    ssize_t w;
    do { w = ::write(async_wake_stop_fd, &one, sizeof(one)); } while (w < 0 && errno == EINTR);
}


void TcpStreamSource::joinAsyncWakeBridge() noexcept
{
    if (async_wake_thread.joinable())
        async_wake_thread.join();
}


void TcpStreamSource::startAsyncWakeBridge()
{
    /// Stop + join any previous (one-shot, likely already-returned) bridge, then arm a fresh one.
    requestAsyncWakeBridgeStop();
    joinAsyncWakeBridge();
    drainEventFd(async_wake_stop_fd);
    async_wake_bridge_stop.store(false, std::memory_order_release);

    const uint64_t elapsed = stall_timer.elapsedMilliseconds();
    const uint64_t remaining = elapsed >= stall_timeout_ms ? 0 : stall_timeout_ms - elapsed;
    async_wake_thread = std::thread([this, remaining] { asyncWakeBridgeLoop(remaining); });
}


void TcpStreamSource::asyncWakeBridgeLoop(uint64_t initial_timeout_ms) noexcept
{
    /// One-shot: poll the socket fd (+ the stop eventfd) up to the remaining stall budget; on socket
    /// readiness/error OR the stall deadline, write ready_event_fd so the executor re-enters
    /// prepare()/work(); on stop, just return.
    const int timeout = initial_timeout_ms > static_cast<uint64_t>(INT_MAX)
        ? INT_MAX : static_cast<int>(initial_timeout_ms);

    while (!async_wake_bridge_stop.load(std::memory_order_acquire))
    {
        pollfd fds[2];
        nfds_t nfds = 0;
        int sock_idx = -1;
        int stop_idx = -1;

        if (sock_fd >= 0)
        {
            fds[nfds].fd = sock_fd;
            fds[nfds].events = POLLIN | POLLHUP | POLLERR;
            fds[nfds].revents = 0;
            sock_idx = static_cast<int>(nfds++);
        }
        if (async_wake_stop_fd >= 0)
        {
            fds[nfds].fd = async_wake_stop_fd;
            fds[nfds].events = POLLIN;
            fds[nfds].revents = 0;
            stop_idx = static_cast<int>(nfds++);
        }

        const int rc = ::poll(fds, nfds, timeout);
        if (async_wake_bridge_stop.load(std::memory_order_acquire))
            return;
        if (rc < 0)
        {
            if (errno == EINTR)
                continue;
            /// On a hard poll error, wake the executor so the main thread surfaces it via recv.
            wakeReadyEvent();
            return;
        }
        if (rc == 0)
        {
            /// Stall deadline reached with no socket activity: wake so prepare()/tryGenerate() raises
            /// SHM_PRODUCER_STALL.
            wakeReadyEvent();
            return;
        }
        if (stop_idx >= 0 && (fds[stop_idx].revents & POLLIN))
            return;
        if (sock_idx >= 0 && (fds[sock_idx].revents & (POLLIN | POLLHUP | POLLERR)))
        {
            wakeReadyEvent();
            return;
        }
    }
}

}

#endif
