#pragma once

#if defined(OS_LINUX)

#include <DataTypes/IDataType.h>
#include <Processors/Chunk.h>
#include <Processors/ISource.h>
#include <Storages/SharedMemorySource/Tracker/AdoptedByteCharger.h>
#include <Storages/SharedMemorySource/Wire/TcpFrame.h>
#include <Common/Stopwatch.h>
#include <base/types.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <vector>


namespace DB
{

/// Hot-Cold Phase 1/2 — TCP transport consumer (D-HC-0101/0104; async = D-HC-0204). The
/// producer-per-stream worker listens on its own TCP port (D-HC-0102); this source connects, reads
/// the handshake (schema cross-validation, identical to the SHM handshake), then reads BLOCK frames
/// (`Wire/TcpFrame.h`). Each block's payload is the SAME frame-relative data-region bytes the SHM
/// data plane carries, so the existing `adopt()` reconstructs columns identically — the recv buffer
/// becomes the `data_region_base` and a RetainToken frees it on last-alias drop. TCP delivery is
/// inherently one kernel copy (recv), so this is adopt over an owned recv buffer.
///
/// Two source modes, selected by the `async` ctor flag (CH setting `shm_tcp_source_async`):
///   * async (Branch 0 default): an IProcessor::Status::Async source mirroring `PollableShmSource`.
///     `prepare()` returns Async while waiting for the socket; `schedule()` returns an epollable
///     readiness eventfd; a source-owned wake bridge polls the socket fd + stall deadline and writes
///     the eventfd so the executor calls back. Recv is a resumable non-blocking state machine, so a
///     partial frame straddling schedule cycles never blocks a pipeline thread — recv of block N+1
///     overlaps downstream processing of block N. This relaxes the Phase-1 blocking-source
///     deadlock invariant (`max_threads >= #blocking sources`): an async source no longer pins a
///     thread inside recv (see DEADLOCK-SAFETY note in the .cpp).
///   * blocking (A/B baseline): the Phase-1 leaf source; `generate()` blocks on the socket (with
///     SO_RCVTIMEO slices to observe cancel + the stall budget); an empty Chunk signals EOS.
class TcpStreamSource final : public ISource
{
public:
    /// Wire format on the socket. Bespoke = the SHM-derived TcpFrame.h blocks (Phase 1). Arrow =
    /// a standard Apache Arrow IPC stream (Schema message, then one RecordBatch message per block,
    /// then the EOS marker) — Phase 2 Branch A (D-HC-0205/0207). The async recv/wake/lifetime
    /// machinery is wire-agnostic; only the framing (tryRecv*) and decode (buildChunkFrom*) differ.
    enum class WireFormat : uint8_t { Bespoke, Arrow };

    TcpStreamSource(
        SharedHeader header,
        String host_,
        UInt16 port_,
        std::vector<DataTypePtr> full_column_types_,
        std::vector<String> full_column_names_,
        std::vector<String> requested_column_names_,
        UInt64 stall_timeout_ms_,
        bool async_ = true,
        WireFormat wire_ = WireFormat::Bespoke,
        bool arrow_zero_copy_ = true,
        bool arrow_lean_extract_ = true);

    ~TcpStreamSource() override;

    TcpStreamSource(const TcpStreamSource &) = delete;
    TcpStreamSource & operator=(const TcpStreamSource &) = delete;

    String getName() const override { return "TcpStreamSource"; }

    /// Async-source contract (only meaningful when `async`); harmless in blocking mode (prepare()
    /// then just defers to the base, schedule()/onAsyncJobReady() are never reached).
    Status prepare() override;
    int schedule() override;
    void onAsyncJobReady() override;

protected:
    std::optional<Chunk> tryGenerate() override;
    void onCancel() noexcept override;

private:
    /// Connect (retrying ECONNREFUSED while the producer comes up), read + cross-validate the
    /// handshake, build the projection map. In async mode the socket is switched to O_NONBLOCK
    /// after the (blocking, one-shot) handshake. Lazy on first tryGenerate().
    void ensureConnected();

    /// Build the projection map (requested subset → full-schema index, emit order = request order).
    /// Shared by the bespoke handshake and the Arrow schema path.
    void buildProjectionIndices();

    /// Build the emitted Chunk from a fully-received frame-relative payload buffer `buffer`
    /// (`bh.payload_len` bytes; descriptors at `bh.descriptors_offset`). Takes ownership of
    /// `buffer`: on success a RetainToken frees it on last-alias drop; on throw it is freed here.
    /// Sets `eos_observed` if `bh.eos_marker`. Shared by the blocking and async paths.
    Chunk buildChunkFromPayload(char * buffer, const SharedMemoryWire::TcpBlockHeader & bh);

    /// Blocking-mode block read (Phase-1 behavior): blocking recvAll of header then payload;
    /// returns an empty Chunk on the EOS frame.
    Chunk recvBlockBlocking();
    /// Blocking recv of exactly `n` bytes (handshake + blocking mode); honours cancel + the stall
    /// budget across SO_RCVTIMEO slices; peer close mid-stream throws producer-death/framing-invalid.
    void recvAll(void * dst, size_t n);

    /// Async-mode resumable recv. Returns one of: a full data block (BlockReady), EOS, or WouldBlock
    /// (partial — caller goes async). Uses the recv_* member state to resume across schedule cycles.
    enum class RecvResult : uint8_t { BlockReady, Eos, WouldBlock };
    RecvResult tryRecvBlock(Chunk & out_chunk);
    /// Non-blocking recv of up to (need - filled) bytes into dst+filled; advances `filled`. Returns
    /// Complete (filled==need), WouldBlock (EAGAIN), or PeerClosed (orderly peer EOF). Throws on a
    /// hard recv error. Resets the stall timer on any progress.
    enum class RecvInto : uint8_t { Complete, WouldBlock, PeerClosed };
    RecvInto tryRecvInto(void * dst, size_t need, size_t & filled);

    /// --- Apache Arrow IPC wire (Branch A; only when wire == WireFormat::Arrow) ---
    /// Blocking, one-shot read of the leading Schema message (replaces the bespoke handshake);
    /// builds + cross-validates the arrow::Schema and the projection map. Called by ensureConnected.
    void readArrowSchema();
    /// Async-mode resumable recv of one Arrow IPC encapsulated message (3 phases: 8-byte prefix →
    /// metadata → body). The 0-length-metadata continuation is the stream EOS. Reuses tryRecvInto.
    RecvResult tryRecvArrowMessage(Chunk & out_chunk);
    /// Decode a fully-received RecordBatch (metadata in arrow_state, `body_buf`/`body_len` the body)
    /// into the emitted Chunk. Dispatches: Branch B iteration 3 (arrow_zero_copy && arrow_lean_extract)
    /// → buildChunkFromArrowLean; otherwise → buildChunkFromArrowViaReader. Takes ownership of `body_buf`.
    Chunk buildChunkFromArrow(char * body_buf, int64_t body_len);
    /// Iteration-2 decode via arrow::ipc::ReadRecordBatch. Branch A (arrow_zero_copy=false): COPYING decode,
    /// frees `body_buf`. Branch B (arrow_zero_copy=true): ZERO-COPY adoption — columns alias `body_buf`, kept
    /// alive by a RetainToken whose deleter frees it on last-drop (Decimal128 / misaligned buffers fall back
    /// to a per-column copy). Takes ownership of `body_buf` either way.
    Chunk buildChunkFromArrowViaReader(char * body_buf, int64_t body_len);
    /// Iteration-3 LEAN zero-copy decode (arrow_zero_copy && arrow_lean_extract): parse the RecordBatch
    /// flatbuffer in arrow_state.metadata directly and adopt each buffer at `body_buf + Buffer.offset()`,
    /// skipping the per-block arrow::Array/ArrayData/Buffer-slice construction. Any non-adoptable column
    /// (Decimal128 alignment, sliced/empty/sentinel-violating batch, out-of-bounds buffer) → delegates the
    /// whole block to buildChunkFromArrowViaReader (body_buf untouched). Takes ownership of `body_buf`.
    Chunk buildChunkFromArrowLean(char * body_buf, int64_t body_len);

    /// Async wake bridge (mirrors PollableShmSource): IProcessor exposes one fd, so while async the
    /// bridge polls the socket fd (+ a stop eventfd) up to the remaining stall budget, then writes
    /// `ready_event_fd` so the executor re-enters prepare()/work(). Stopped on readiness/cancel/dtor.
    void startAsyncWakeBridge();
    void requestAsyncWakeBridgeStop() noexcept;
    void joinAsyncWakeBridge() noexcept;
    void asyncWakeBridgeLoop(uint64_t initial_timeout_ms) noexcept;
    void wakeReadyEvent() const noexcept;

    String host;
    UInt16 port;
    std::vector<DataTypePtr> full_column_types;
    std::vector<String> full_column_names;
    std::vector<String> requested_column_names;
    std::vector<size_t> projection_indices;
    UInt64 stall_timeout_ms;
    const bool async;
    const WireFormat wire;
    const bool arrow_zero_copy;   /// Branch B: adopt the Arrow buffers zero-copy (vs the Branch-A copy)
    const bool arrow_lean_extract;   /// Branch B it3: lean direct-flatbuffer extraction (vs ReadRecordBatch)

    /// Arrow IPC recv/decode state (pimpl: keeps arrow headers out of this header). Non-null iff
    /// wire == WireFormat::Arrow; holds the decoded arrow::Schema + the resumable message-recv state.
    struct ArrowRecvState;
    std::unique_ptr<ArrowRecvState> arrow_state;

    /// Atomic so onCancel() (callable from an arbitrary thread, noexcept) and the dtor cannot
    /// double-close / torn-read it; dtor exchange(-1)s before close. (Adversarial review B0 #1.)
    std::atomic<int> sock_fd{-1};
    bool connected = false;
    bool eos_observed = false;
    std::atomic<bool> cancelled{false};

    /// Async-source machinery (async mode only).
    int ready_event_fd = -1;       /// owned; schedule() returns it; bridge writes it
    int async_wake_stop_fd = -1;   /// owned; wakes the bridge out of poll() on stop
    std::atomic<bool> async_wake_bridge_stop{false};
    std::thread async_wake_thread;
    bool is_async_state = false;

    /// Resumable recv state (async mode). recv_phase tracks header vs payload; recv_filled is how
    /// many bytes of the current target have arrived. pending_buf is the to-be-adopted payload
    /// buffer, allocated when the header completes and owned here until handed to a RetainToken.
    enum class RecvPhase : uint8_t { Header, Payload };
    RecvPhase recv_phase = RecvPhase::Header;
    SharedMemoryWire::TcpBlockHeader cur_bh{};
    size_t recv_filled = 0;
    char * pending_buf = nullptr;
    size_t pending_payload_len = 0;

    AdoptedByteCharger charger;
    Stopwatch stall_timer;
};

}

#endif
