#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/TestProducer/InProcessProducer.h>
#include <Storages/SharedMemorySource/Wire/ControlSocket.h>

#include <Common/EventFD.h>

#include <base/errnoToString.h>
#include <base/getPageSize.h>

#include <sys/mman.h>
#include <poll.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>


namespace DB
{

using SharedMemoryWire::HandshakeRegion;
using SharedMemoryWire::SlotEntry;
using SharedMemoryWire::SlotState;
using SharedMemoryWire::SchemaEntry;
using SharedMemoryWire::ColumnDescriptor;
using SharedMemoryWire::WireColumnType;
using SharedMemoryWire::SHM_MAGIC;
using SharedMemoryWire::SHM_ABI_VERSION_1;
using SharedMemoryWire::IMPL_MAX_K;
using SharedMemoryWire::IMPL_MAX_COLUMNS;
using SharedMemoryWire::PADDING_FOR_SIMD;
using SharedMemoryWire::SCHEMA_ENTRY_STR_MAX;

namespace
{
    /// Per-slot zero-pre-sentinel for String offsets — 8 bytes that
    /// `ColumnString::offsetAt(0)` reads via the adopted PaddedPODArray<UInt64>'s pad_left.
    constexpr size_t OFFSETS_PRE_SENTINEL_BYTES = sizeof(uint64_t);

    /// Sleep slice for both ring-full waits and AC10 retain-release waits. Small enough that
    /// tests don't visibly slow down; large enough not to spin a CPU.
    constexpr int WAIT_SLICE_USEC = 1000;
    constexpr int CONNECTED_SOCKET_PRUNE_INTERVAL_MS = 50;

    size_t alignUp(size_t value, size_t alignment) noexcept
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    std::string normalizeShmName(const std::string & n)
    {
        if (n.empty() || n.front() != '/')
            return "/" + n;
        return n;
    }

    [[noreturn]] void throwRuntime(const std::string & msg)
    {
        throw std::runtime_error(msg);
    }

    [[noreturn]] void throwErrno(const char * op, const std::string & name)
    {
        const int saved_errno = errno;
        throw std::runtime_error(std::string(op) + " failed on '" + name + "': " + errnoToString(saved_errno));
    }

    /// Map a simple CH type name to its wire tag. The test producer only ever uses the
    /// canonical names of the supported set (no parametrized types), so a direct string
    /// lookup keeps this scaffolding free of the DataTypeFactory dependency. The consumer
    /// side's authoritative TypeIndex-based map lives in Wire/WireTypeMapping.h.
    WireColumnType wireTagForTypeString(const std::string & t)
    {
        if (t == "UInt8") return WireColumnType::UInt8;
        if (t == "UInt16") return WireColumnType::UInt16;
        if (t == "UInt32") return WireColumnType::UInt32;
        if (t == "UInt64") return WireColumnType::UInt64;
        if (t == "Int8") return WireColumnType::Int8;
        if (t == "Int16") return WireColumnType::Int16;
        if (t == "Int32") return WireColumnType::Int32;
        if (t == "Int64") return WireColumnType::Int64;
        if (t == "Float32") return WireColumnType::Float32;
        if (t == "Float64") return WireColumnType::Float64;
        if (t == "Date") return WireColumnType::Date;
        if (t == "DateTime") return WireColumnType::DateTime;
        if (t == "Date32") return WireColumnType::Date32;
        if (t == "String") return WireColumnType::String;
        throw std::runtime_error("InProcessProducer: unsupported schema type '" + t + "'");
    }
}

/// =====================================================================
/// computeTotalSize: pin the on-wire layout.
///
/// Offsets (all page-aligned where it matters for mmap-friendly access):
///   0                          -> HandshakeRegion (128 B)
///   PAGE                       -> SlotEntry[K] (K * 64 B each, alignas(64))
///   slot_table_end             -> SchemaEntry[schema.size()] (128 B each, alignas(8))
///   next PAGE boundary         -> data region (cfg.data_region_size bytes)
///
/// Total is rounded up to a page so the kernel doesn't truncate the trailing region.
/// =====================================================================
size_t InProcessProducer::computeTotalSize(const Config & cfg)
{
    const size_t page = static_cast<size_t>(::getPageSize());
    size_t off = sizeof(HandshakeRegion);
    off = alignUp(off, page);
    off += static_cast<size_t>(cfg.ring_depth_k) * sizeof(SlotEntry);
    off += cfg.schema.size() * sizeof(SchemaEntry);
    off = alignUp(off, page);
    off += cfg.data_region_size;
    return alignUp(off, page);
}

HandshakeRegion * InProcessProducer::handshake() noexcept
{
    return reinterpret_cast<HandshakeRegion *>(mapping);
}

SlotEntry * InProcessProducer::slotAt(uint32_t i) noexcept
{
    auto * base = reinterpret_cast<char *>(mapping) + handshake()->slot_table_offset;
    return reinterpret_cast<SlotEntry *>(base + i * handshake()->slot_table_stride);
}

char * InProcessProducer::dataRegion() noexcept
{
    return reinterpret_cast<char *>(mapping) + handshake()->data_region_offset;
}

int InProcessProducer::eventFd() const noexcept
{
    return ready_event ? ready_event->fd : -1;
}

void InProcessProducer::populateHandshake()
{
    auto * hs = handshake();
    const size_t page = static_cast<size_t>(::getPageSize());

    hs->abi_version = SHM_ABI_VERSION_1;
    hs->ring_depth_k = config.ring_depth_k;
    hs->schema_count = static_cast<uint32_t>(config.schema.size());
    hs->reserved_pad32 = 0;
    hs->slot_table_offset = alignUp(sizeof(HandshakeRegion), page);
    hs->slot_table_stride = sizeof(SlotEntry);
    hs->schema_table_offset = hs->slot_table_offset
        + static_cast<uint64_t>(config.ring_depth_k) * sizeof(SlotEntry);
    hs->schema_table_size = static_cast<uint64_t>(config.schema.size()) * sizeof(SchemaEntry);
    hs->data_region_offset = alignUp(hs->schema_table_offset + hs->schema_table_size, page);
    hs->data_region_size = config.data_region_size;
    for (auto & r : hs->reserved64)
        r = 0;

    /// `shm-block-stream.md` §Memory ordering / §ABI version negotiation: magic is the LAST
    /// write on the handshake region; the consumer's acquire-load of magic implies acquire
    /// of every other field.
    hs->magic.store(SHM_MAGIC, std::memory_order_release);
}

void InProcessProducer::pruneConnectedSockets() noexcept
{
    std::lock_guard<std::mutex> lock(connected_sockets_mutex);
    auto it = connected_sockets.begin();
    while (it != connected_sockets.end())
    {
        pollfd pfd{};
        pfd.fd = *it;
        pfd.events = 0;
        const int rc = ::poll(&pfd, 1, 0);
        if (rc > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)))
        {
            ::close(*it);
            it = connected_sockets.erase(it);
            continue;
        }
        ++it;
    }
}

size_t InProcessProducer::connectedSocketCountForTesting() const noexcept
{
    std::lock_guard<std::mutex> lock(connected_sockets_mutex);
    return connected_sockets.size();
}

void InProcessProducer::acceptLoop()
{
    while (!shutdown_requested.load(std::memory_order_acquire))
    {
        pruneConnectedSockets();
        int conn = control_socket->accept();
        if (conn < 0)
            return;
        try
        {
            control_socket->sendEventFd(conn, ready_event->fd);
        }
        catch (...) // NOLINT(bugprone-empty-catch)
        {
            /// Best-effort: a consumer that disappeared mid-handshake should not crash the
            /// producer; the next consumer reconnects. We deliberately swallow and close the
            /// half-dead fd immediately so we don't park a no-longer-connected peer.
            ::close(conn);
            continue;
        }
        /// Park the accepted fd while the consumer is connected. Closing it now would make the
        /// consumer's matching connection fd report POLLHUP and trigger a false-positive
        /// SHM_PRODUCER_DEATH_BEFORE_EOS (precondition 25).
        pruneConnectedSockets();
        {
            std::lock_guard<std::mutex> lock(connected_sockets_mutex);
            connected_sockets.push_back(conn);
        }
    }
}

void InProcessProducer::pruneLoop() noexcept
{
    while (!shutdown_requested.load(std::memory_order_acquire))
    {
        pruneConnectedSockets();
        for (int i = 0; i < CONNECTED_SOCKET_PRUNE_INTERVAL_MS
             && !shutdown_requested.load(std::memory_order_acquire); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    pruneConnectedSockets();
}

/// =====================================================================
/// ctor
/// =====================================================================
InProcessProducer::InProcessProducer(Config cfg)
    : config(std::move(cfg))
{
    config.shm_name = normalizeShmName(config.shm_name);

    if (config.ring_depth_k == 0 || config.ring_depth_k > IMPL_MAX_K)
        throwRuntime("InProcessProducer: ring_depth_k out of range");
    if (config.schema.empty() || config.schema.size() > IMPL_MAX_COLUMNS)
        throwRuntime("InProcessProducer: schema column count out of range");
    if (config.data_region_size == 0)
        throwRuntime("InProcessProducer: data_region_size must be > 0");
    for (const auto & col : config.schema)
    {
        if (col.first.size() + 1 > SCHEMA_ENTRY_STR_MAX || col.second.size() + 1 > SCHEMA_ENTRY_STR_MAX)
            throwRuntime("InProcessProducer: schema name/type string too long");
    }

    /// Stale region from a prior crash blocks O_EXCL.
    ::shm_unlink(config.shm_name.c_str());

    shm_fd = ::shm_open(config.shm_name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    if (shm_fd < 0)
        throwErrno("shm_open", config.shm_name);

    mapping_size = computeTotalSize(config);
    if (::ftruncate(shm_fd, static_cast<off_t>(mapping_size)) != 0)
        throwErrno("ftruncate", config.shm_name);

    mapping = ::mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (mapping == MAP_FAILED)
        throwErrno("mmap", config.shm_name);

    /// ftruncate-zeroed memory makes every atomic and POD field start at 0; that matches
    /// SlotState::EMPTY, sequence=0, refcount=0, eos=0. We only need to set the non-zero
    /// metadata fields.

    /// Schema entries.
    auto * schema_base = reinterpret_cast<SchemaEntry *>(
        reinterpret_cast<char *>(mapping) + alignUp(sizeof(HandshakeRegion), ::getPageSize())
        + static_cast<size_t>(config.ring_depth_k) * sizeof(SlotEntry));
    for (size_t i = 0; i < config.schema.size(); ++i)
    {
        auto & dst = schema_base[i];
        std::memset(&dst, 0, sizeof(dst));
        std::memcpy(dst.name, config.schema[i].first.data(), config.schema[i].first.size());
        std::memcpy(dst.type_string, config.schema[i].second.data(), config.schema[i].second.size());
    }

    populateHandshake();

    /// Per-slot bookkeeping (sequence starts at 0; first publish writes sequence=1).
    next_sequence_per_slot.assign(config.ring_depth_k, 0);

    per_slot_data_capacity = config.data_region_size / config.ring_depth_k;
    if (per_slot_data_capacity < PADDING_FOR_SIMD * 4)
        throwRuntime("InProcessProducer: data_region_size too small for K slots");

    /// Per-slot region layout: descriptors live at offset 0 (sized & padded for max columns);
    /// payload begins after. This is a producer-side choice; the consumer reads via the
    /// descriptor's value_offset / offsets_offset only.
    const size_t descriptor_array_bytes = config.schema.size() * sizeof(ColumnDescriptor);
    per_slot_payload_offset = alignUp(descriptor_array_bytes, 64);

    /// Set self-identity slot_index now (precondition 9); other slot fields stay 0 until publish.
    for (uint32_t i = 0; i < config.ring_depth_k; ++i)
        slotAt(i)->slot_index = i;

    ready_event = std::make_unique<EventFD>();
    control_socket = std::make_unique<ControlSocketServer>(controlSocketPathForShmName(config.shm_name));

    try
    {
        accept_thread = std::thread([this] { acceptLoop(); });
        socket_prune_thread = std::thread([this] { pruneLoop(); });
    }
    catch (...)
    {
        shutdown_requested.store(true, std::memory_order_release);
        if (control_socket)
            control_socket->shutdown();
        if (accept_thread.joinable())
            accept_thread.join();
        if (socket_prune_thread.joinable())
            socket_prune_thread.join();
        throw;
    }

    ready.store(true, std::memory_order_release);
}

/// =====================================================================
/// dtor
/// =====================================================================
InProcessProducer::~InProcessProducer()
{
    shutdown_requested.store(true, std::memory_order_release);

    if (control_socket)
        control_socket->shutdown();
    if (accept_thread.joinable())
        accept_thread.join();
    if (socket_prune_thread.joinable())
        socket_prune_thread.join();

    /// Close every parked consumer-connection fd. Prune first so long-running tests that
    /// dropped consumers earlier do not keep already-dead peers until process exit. Closing
    /// any remaining live fd here surfaces POLLHUP on the consumer side, which is correct:
    /// at this point the producer really is going away.
    pruneConnectedSockets();
    {
        std::lock_guard<std::mutex> lock(connected_sockets_mutex);
        for (int fd : connected_sockets)
            ::close(fd);
        connected_sockets.clear();
    }

    control_socket.reset();
    ready_event.reset();

    if (mapping != nullptr && mapping != MAP_FAILED)
        ::munmap(mapping, mapping_size);
    if (shm_fd >= 0)
        ::close(shm_fd);
    if (!config.shm_name.empty())
        ::shm_unlink(config.shm_name.c_str());
}

/// =====================================================================
/// publishBlock + signalEndOfStream + publishMalformedBlock — all delegate to publishBlockImpl.
/// =====================================================================
void InProcessProducer::publishBlock(const std::vector<ColumnPayload> & payloads, size_t row_count)
{
    publishBlockImpl(payloads, row_count, /*is_eos=*/false, nullptr);
}

void InProcessProducer::signalEndOfStream()
{
    std::vector<ColumnPayload> empty(config.schema.size());
    publishBlockImpl(empty, 0, /*is_eos=*/true, nullptr);
    eos_published.store(true, std::memory_order_release);
}

void InProcessProducer::publishMalformedBlock(
    const std::vector<ColumnPayload> & payloads, size_t row_count, Malformation kind)
{
    publishBlockImpl(payloads, row_count, /*is_eos=*/false, &kind);
}

/// =====================================================================
/// publishBlockImpl: the per-block memory-ordering contract.
///
/// Order (`shm-block-stream.md` §Memory ordering / §Notification contract /
/// §Publication state machine):
///   1. Wait for slot.state == EMPTY (AC10 cooperation + ring-full backpressure). The
///      consumer drives the PUBLISHED→EMPTY transition on its last RetainToken drop, so
///      polling `state` rather than `retain_refcount` ensures we never overwrite an
///      unconsumed PUBLISHED slot whose refcount is incidentally 0 (Findings 1 + 3).
///   2. Transition EMPTY → WRITING. The consumer owns PUBLISHED→EMPTY, so we never write
///      EMPTY here; the slot enters this step already-EMPTY by step-1's wait.
///   3. Write payload bytes, descriptor array, row_count, eos_marker, sequence.
///   4. Release-store slot.state = PUBLISHED — single ordering barrier; everything in step 3
///      is now observable to a consumer that does an acquire-load on slot.state.
///   5. eventfd write — notification ordered AFTER the metadata publication.
/// =====================================================================
void InProcessProducer::publishBlockImpl(
    const std::vector<ColumnPayload> & payloads, size_t row_count,
    bool is_eos, Malformation * malformation)
{
    if (stalled.load(std::memory_order_acquire))
        throwRuntime("InProcessProducer: producer is stalled");
    if (eos_published.load(std::memory_order_acquire))
        throwRuntime("InProcessProducer: stream already ended");
    if (payloads.size() != config.schema.size())
        throwRuntime("InProcessProducer: payload count mismatch");
    if (row_count > SharedMemoryWire::IMPL_MAX_ROWS_PER_BLOCK)
        throwRuntime("InProcessProducer: row_count exceeds IMPL_MAX_ROWS_PER_BLOCK");

    const uint32_t slot_pos = next_publish_slot % config.ring_depth_k;
    auto * slot = slotAt(slot_pos);

    /// Step 1: wait for the slot to be reusable. The wire's release contract
    /// (shm-block-stream.md §Publication state machine) makes the *consumer* drive the
    /// PUBLISHED→released transition by storing EMPTY on the last retain drop. The
    /// producer therefore polls `state == EMPTY` rather than `retain_refcount == 0`: an
    /// unconsumed PUBLISHED slot (refcount still 0 because the consumer has not yet
    /// attached) would otherwise be silently overwritten.
    while (slot->state.load(std::memory_order_acquire) != static_cast<uint32_t>(SlotState::EMPTY))
    {
        if (shutdown_requested.load(std::memory_order_acquire))
            throwRuntime("InProcessProducer: shutdown during ring-full wait");
        ::usleep(WAIT_SLICE_USEC);
    }

    /// Step 2: EMPTY → WRITING. No need to drive PUBLISHED → EMPTY here anymore — that
    /// transition is the consumer's responsibility per the wire contract above.
    /// Bump `transition_counter` BEFORE the state store, with release ordering. The
    /// consumer's precondition-24 check handles the small window where the counter is
    /// visible before the state store by retrying briefly for the state to catch up.
    slot->transition_counter.fetch_add(1, std::memory_order_release);
    slot->state.store(static_cast<uint32_t>(SlotState::WRITING), std::memory_order_release);

    /// Step 3: per-slot layout (descriptors at offset 0, payload after).
    const size_t slot_data_base = static_cast<size_t>(slot_pos) * per_slot_data_capacity;
    auto * desc_array = reinterpret_cast<ColumnDescriptor *>(dataRegion() + slot_data_base);
    std::memset(desc_array, 0, config.schema.size() * sizeof(ColumnDescriptor));

    size_t cursor = slot_data_base + per_slot_payload_offset;
    const size_t slot_end = slot_data_base + per_slot_data_capacity;

    auto reserve = [&](size_t want, size_t align) -> size_t
    {
        const size_t aligned = alignUp(cursor, align);
        if (aligned + want > slot_end)
            throwRuntime("InProcessProducer: per-slot data region overflow");
        cursor = aligned + want;
        return aligned;
    };

    for (size_t i = 0; i < config.schema.size(); ++i)
    {
        const auto & p = payloads[i];
        auto & d = desc_array[i];

        /// Resolve the column's wire tag from its declared CH type name.
        const WireColumnType wire_tag = wireTagForTypeString(config.schema[i].second);

        if (wire_tag == WireColumnType::String)
        {
            d.type = static_cast<uint32_t>(WireColumnType::String);

            const size_t chars_bytes = is_eos ? 0 : p.value_count;
            const size_t chars_off = reserve(chars_bytes + PADDING_FOR_SIMD, 8);
            if (!is_eos && p.value_bytes != nullptr && chars_bytes > 0)
                std::memcpy(dataRegion() + chars_off, p.value_bytes, chars_bytes);

            /// 8 zero bytes preceding offsets[0] — the offsets[-1] sentinel
            /// `ColumnString::offsetAt(0)` reads through the adopted PaddedPODArray<UInt64>'s
            /// pad_left. The descriptor's `offsets_offset` points at offsets[0], NOT here.
            const size_t sentinel_off = reserve(OFFSETS_PRE_SENTINEL_BYTES, 8);
            std::memset(dataRegion() + sentinel_off, 0, OFFSETS_PRE_SENTINEL_BYTES);

            const size_t offs_bytes = is_eos ? 0 : p.offset_count * sizeof(uint64_t);
            const size_t offs_off = reserve(offs_bytes + PADDING_FOR_SIMD, 8);
            if (!is_eos && p.offset_bytes != nullptr && offs_bytes > 0)
                std::memcpy(dataRegion() + offs_off, p.offset_bytes, offs_bytes);

            d.value_offset = chars_off;
            d.value_count = chars_bytes;
            d.value_padding = PADDING_FOR_SIMD;
            d.offsets_offset = offs_off;
            d.offsets_count = is_eos ? 0 : p.offset_count;
            d.offsets_padding = PADDING_FOR_SIMD;
        }
        else
        {
            /// Any fixed-width type: one value buffer of `value_count` elements, each
            /// `elem_size` bytes. Offset is 8-aligned (a multiple of every supported
            /// elem_size), satisfying the consumer's precondition-13 alignment check.
            const size_t elem_size = SharedMemoryWire::wireFixedWidthSize(wire_tag);
            d.type = static_cast<uint32_t>(wire_tag);
            const size_t vbytes = is_eos ? 0 : p.value_count * elem_size;
            const size_t voff = reserve(vbytes + PADDING_FOR_SIMD, 8);
            if (!is_eos && p.value_bytes != nullptr && vbytes > 0)
                std::memcpy(dataRegion() + voff, p.value_bytes, vbytes);
            d.value_offset = voff;
            d.value_count = is_eos ? 0 : p.value_count;
            d.value_padding = PADDING_FOR_SIMD;
        }
    }

    /// Apply descriptor-level malformations after the correct descriptors are in place so
    /// the named precondition is the precise violation (other checks still pass).
    if (malformation != nullptr)
    {
        auto & d0 = desc_array[0];
        switch (*malformation)
        {
            case Malformation::MisalignedDescriptorOffset:
                break;
            case Malformation::OffsetOverflow:
                d0.value_offset = config.data_region_size;
                break;
            case Malformation::MisalignedColumn:
                d0.value_offset = d0.value_offset + 1;
                break;
            case Malformation::WrongRowCount:
                d0.value_count = row_count + 1;
                break;
            case Malformation::WrongTerminalOffset:
                if (d0.type == static_cast<uint32_t>(WireColumnType::String)
                    && row_count > 0 && d0.offsets_offset != 0)
                {
                    auto * offs = reinterpret_cast<uint64_t *>(dataRegion() + d0.offsets_offset);
                    offs[row_count - 1] = d0.value_count + 100;
                }
                break;
            case Malformation::NonMonotonicOffsets:
                if (d0.type == static_cast<uint32_t>(WireColumnType::String)
                    && row_count >= 2 && d0.offsets_offset != 0)
                {
                    auto * offs = reinterpret_cast<uint64_t *>(dataRegion() + d0.offsets_offset);
                    offs[1] = 0;
                }
                break;
            case Malformation::BadSequence:
            case Malformation::BadSlotIdentity:
                break;
        }
    }

    /// Slot metadata.
    slot->per_column_descriptors_offset = slot_data_base;
    if (malformation != nullptr && *malformation == Malformation::MisalignedDescriptorOffset)
        slot->per_column_descriptors_offset = slot_data_base + 1;
    slot->row_count = row_count;
    slot->eos_marker.store(is_eos ? 1 : 0, std::memory_order_relaxed);

    uint64_t seq = ++next_sequence_per_slot[slot_pos];
    if (malformation != nullptr && *malformation == Malformation::BadSequence && seq > 1)
        seq = seq - 2;
    slot->sequence.store(seq, std::memory_order_relaxed);

    if (malformation != nullptr && *malformation == Malformation::BadSlotIdentity)
        slot->slot_index = slot_pos + 1;

    /// Step 4: the release-store that publishes the slot. Pair the transition
    /// counter bump with this store too (W→P transition) per Layout.h's
    /// `transition_counter` protocol; both the counter bump and the state
    /// store use release ordering, and the counter happens-before the state.
    /// PollableShmSource tolerates the resulting short counter-before-state
    /// propagation window with a bounded retry before declaring
    /// precondition-24 malformed.
    slot->transition_counter.fetch_add(1, std::memory_order_release);
    slot->state.store(static_cast<uint32_t>(SlotState::PUBLISHED), std::memory_order_release);

    /// Step 5: notification, ordered AFTER metadata publication.
    if (ready_event)
        ready_event->write();
    pruneConnectedSockets();

    ++next_publish_slot;
}

void InProcessProducer::waitForRetainToRelease(uint32_t slot_index)
{
    auto * slot = slotAt(slot_index);
    while (slot->retain_refcount.load(std::memory_order_acquire) != 0)
        ::usleep(WAIT_SLICE_USEC);
}

[[noreturn]] void InProcessProducer::forceUngracefulExit()
{
    ::_exit(1);
}

void InProcessProducer::setSlotStateForTesting(uint32_t slot_index, SlotState new_state) noexcept
{
    /// Test-only escape hatch (AC6 mid-publication crash). Mirrors the
    /// precondition-24 protocol: bump `transition_counter` BEFORE the state
    /// store, both with release ordering. PollableShmSource's bounded retry
    /// handles the same counter-before-state visibility window as normal
    /// publication.
    auto * slot = slotAt(slot_index);
    slot->transition_counter.fetch_add(1, std::memory_order_release);
    slot->state.store(static_cast<uint32_t>(new_state), std::memory_order_release);
}

}

#endif
