#pragma once

#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/Wire/Layout.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <utility>


namespace DB
{
class ControlSocketServer;
struct EventFD;

/// In-process test producer for the zero-copy SHM source feature. Creates a POSIX shared memory
/// region with `shm_open(O_RDWR|O_CREAT)`, populates the AC8 wire ABI (handshake region + slot
/// table + schema table + data region), spawns a control-socket accept loop for the readiness
/// fd, and exposes a thread-safe `publishBlock` API.
///
/// Lifecycle:
///   - Construct: opens SHM, mmaps RW, writes handshake (with release-store on `magic` LAST),
///     creates eventfd, binds Unix socket, starts accept-loop thread.
///   - publishBlock / signalEndOfStream: synchronous; blocks if the ring is full (waits for
///     the consumer to drop its retain on the oldest slot).
///   - Destruct: stops accept loop, closes socket, closes eventfd, unmaps, closes shm fd,
///     unlinks shm name + socket path.
///
/// Spec authority: shm-block-stream §Per-type buffer layout, §Schema declaration and negotiation,
/// §Block framing, §Publication state machine, §Memory ordering, §End-of-stream, §Backpressure,
/// §Retain/release contract, AC8, AC10, I2, I11.
class InProcessProducer
{
public:
    using ColumnSpec = std::pair<std::string, std::string>; // (name, type_string)

    struct Config
    {
        std::string shm_name;                ///< e.g. "test_shm_42"; leading '/' added if missing.
        uint32_t ring_depth_k = 4;           ///< 1 .. SharedMemoryWire::IMPL_MAX_K
        std::vector<ColumnSpec> schema;      ///< ordered; size in [1, IMPL_MAX_COLUMNS].
        size_t data_region_size = 16 * 1024 * 1024;  ///< default 16 MiB.
    };

    explicit InProcessProducer(Config cfg);
    ~InProcessProducer();
    InProcessProducer(const InProcessProducer &) = delete;
    InProcessProducer & operator=(const InProcessProducer &) = delete;

    // ---------- T2.4a (wire setup) ----------

    /// True after the ctor populates the handshake AND the accept loop is running.
    bool isReady() const noexcept { return ready.load(std::memory_order_acquire); }

    /// Eventfd the producer created. Consumers normally receive it via SCM_RIGHTS; some
    /// tests bypass the socket and grab it directly.
    int eventFd() const noexcept;

    /// SHM object name actually used (with leading '/').
    const std::string & shmName() const noexcept { return config.shm_name; }

    // ---------- T2.4b (publish API) ----------

    /// One column's payload for one block.
    ///   - UInt64: value_bytes / value_count describe the raw value buffer (bytes == count*8).
    ///     offset_bytes / offset_count are unused.
    ///   - String: value_bytes / value_count describe the concatenated chars (UInt8 buffer,
    ///     value_count == byte size). offset_bytes / offset_count describe the offsets
    ///     buffer (UInt64 buffer, offset_count == rows).
    struct ColumnPayload
    {
        const void * value_bytes = nullptr;
        size_t value_count = 0;
        const void * offset_bytes = nullptr;
        size_t offset_count = 0;
    };

    /// Publish one block. Blocks if the ring is full until a slot becomes available (i.e. the
    /// consumer drops its retain). Throws `std::runtime_error` on layout overflow (the block
    /// does not fit in the per-slot region) or schema mismatch (`payloads.size() != schema`).
    void publishBlock(const std::vector<ColumnPayload> & payloads, size_t row_count);

    /// Publish an empty block with `eos_marker == 1`. Further `publishBlock` calls throw.
    void signalEndOfStream();

    // ---------- test-only knobs ----------

    /// Block until the slot at `slot_index` has `retain_refcount == 0`. Used to test the AC10
    /// republish-under-retain cooperation point.
    void waitForRetainToRelease(uint32_t slot_index);

    /// Mark the producer as stalled: `publishBlock` will throw immediately. Used by stall tests.
    void stallProducer() noexcept { stalled.store(true, std::memory_order_release); }

    /// `_exit(1)` the process immediately. Use only from a forked child (AC6 producer-death).
    [[noreturn]] static void forceUngracefulExit();

    /// FOR TESTS ONLY — directly poke a slot's state without going through the
    /// normal publish state machine. Used by the AC6 mid-publication crash
    /// test (gtest_pollable_shm_source.cpp) to simulate a producer that
    /// transitioned E→W on a slot and then died before W→P. Also bumps
    /// `transition_counter` to mirror the precondition-24 protocol so the
    /// consumer's monotonicity check stays valid. Not thread-safe vs.
    /// `publishBlock`.
    void setSlotStateForTesting(uint32_t slot_index, SharedMemoryWire::SlotState new_state) noexcept;

    /// Publish a single block whose slot is deliberately malformed per `kind`. AC6 tests.
    enum class Malformation
    {
        BadSequence,           ///< precondition 10: sequence not strictly greater
        BadSlotIdentity,       ///< precondition 9: slot identity != position
        OffsetOverflow,        ///< preconditions 14/18/19: declared sizes overflow data region
        MisalignedColumn,      ///< preconditions 13/16/17: descriptor offset misaligned
        WrongRowCount,         ///< precondition 26: declared count != row_count
        WrongTerminalOffset,   ///< precondition 22: String terminal offset wrong
        NonMonotonicOffsets,   ///< precondition 21: String offsets not monotonic
    };
    void publishMalformedBlock(const std::vector<ColumnPayload> & payloads, size_t row_count, Malformation kind);

private:
    Config config;

    int shm_fd = -1;
    void * mapping = nullptr;
    size_t mapping_size = 0;

    std::unique_ptr<EventFD> ready_event;

    std::unique_ptr<ControlSocketServer> control_socket;
    std::thread accept_thread;
    std::atomic<bool> shutdown_requested{false};

    /// Every accepted control-socket connection is parked here for the producer's lifetime.
    /// Closing the accept-side fd immediately after `sendEventFd` would surface POLLHUP on
    /// the consumer's connection fd and trip `PollableShmSource::checkProducerDeath` while
    /// the producer is still alive (shm-block-stream.md I11 + pollable-shm-source.md
    /// precondition 25). Only the dtor closes these fds, *after* `accept_thread` is joined
    /// — so no concurrent writer races the dtor's read.
    std::vector<int> connected_sockets;

    std::vector<uint64_t> next_sequence_per_slot;
    uint32_t next_publish_slot = 0;
    size_t per_slot_data_capacity = 0;    ///< data_region_size / K; per-slot working region.
    size_t per_slot_payload_offset = 0;   ///< bytes from start of each slot region to payload.

    std::atomic<bool> ready{false};
    std::atomic<bool> stalled{false};
    std::atomic<bool> eos_published{false};

    SharedMemoryWire::HandshakeRegion * handshake() noexcept;
    SharedMemoryWire::SlotEntry * slotAt(uint32_t i) noexcept;
    char * dataRegion() noexcept;

    static size_t computeTotalSize(const Config & cfg);
    void populateHandshake();
    void acceptLoop();

    void publishBlockImpl(const std::vector<ColumnPayload> & payloads, size_t row_count,
                          bool is_eos, Malformation * malformation);
};
}

#endif
