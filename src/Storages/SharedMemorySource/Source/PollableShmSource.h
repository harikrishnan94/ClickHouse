#pragma once

#if defined(OS_LINUX)

#include <Core/Block_fwd.h>
#include <DataTypes/IDataType.h>
#include <Processors/Chunk.h>
#include <Processors/ISource.h>
#include <Storages/SharedMemorySource/Tracker/AdoptedByteCharger.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>
#include <Storages/SharedMemorySource/Wire/SharedMemoryRegion.h>
#include <Common/Stopwatch.h>
#include <base/types.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>


namespace DB
{

/// Pollable IProcessor source that consumes producer-published SHM blocks via the
/// SHM-adoption ABI wire (T0.1 `Wire/Layout.h`), routes them through the adoption
/// seam (T3.1 `Adoption/AdoptionLayer.h`), and emits zero-copy `Chunk`s of adopted
/// `ColumnVector<UInt64>` / `ColumnString` columns to the downstream pipeline.
///
/// Phase-1 contract pinpoints (mirrored from `pollable-shm-source.md`):
///   - I6 (Pollable contract): `prepare()` returns `Status::Async` only when no
///     block is drainable AND not EOS; `schedule()` returns the readiness eventfd
///     received from the producer at attach; `onAsyncJobReady()` drains the
///     eventfd and returns immediately; `onCancel()` is `noexcept` and writes
///     the eventfd which unblocks any executor wait.
///   - I9 (Cancellation is bounded): the only consumer-side waits are on
///     `read(eventfd)` which the executor manages via epoll; `onCancel()` writes
///     to the eventfd to wake any pending wait, then sets the cancellation flag
///     that next `prepare()` reads. No producer cooperation is required.
///   - I12 (Stall is bounded): `Stopwatch stall_timer` is reset on every
///     observed publication progress AND on every `onAsyncJobReady()`; checked
///     at the top of `prepare()` while async; exceeds → throw `SHM_PRODUCER_STALL`.
///   - Per-block adoption call (`pollable-shm-source.md` §Interfaces & contracts):
///     the source executes the 3-step RAII sequence retain → charge → adopt,
///     with on-throw rollback of intermediate handles.
///
/// Per-precondition detection-point map (the VC3 obligation; the precondition
/// numbers refer to `pollable-shm-source.md` Producer-side preconditions
/// enumerated):
///   1, 2, 3, 7 → `SharedMemoryRegion::attach` (T1.1)
///   4, 5, 6   → `ensureAttached()` handshake cross-validation (this file)
///   8          → `findNextReadySlot()` state-load check
///   9          → `findNextReadySlot()` slot_index check
///   10         → `findNextReadySlot()` sequence check
///   11         → `drainSlot()` row_count check
///   12         → adopt() descriptor count check (T3.1)
///   13–22      → adopt() per-column descriptor checks (T3.1) + lazy
///                validateAdoptedOffsets() for 21–22
///   23         → `tryGenerate()` post-EOS publish check (rejects any PUBLISHED
///                slot observed after `eos_observed`)
///   24         → `findNextReadySlot()` `transition_counter` monotonicity check
///                PLUS strict expected-delta state-machine validation. The wire's
///                per-slot `transition_counter` (Layout.h) is incremented under
///                release ordering by whoever drives a state transition (producer
///                on E→W and W→P; consumer on P→E), so a complete publish cycle
///                produces exactly +3. The check walks the legal cycle
///                EMPTY(0)→WRITING(1)→PUBLISHED(2)→EMPTY(0)→… by `delta` steps
///                from `last_observed_state[i]` and confirms it lands on the
///                current `obs_state` (cyclic position `(prev_pos + delta) % 3`).
///                A producer that skips a state increments the counter by fewer
///                steps than the legal walk from prev to obs state — the cyclic
///                arithmetic catches that as `expected_pos != obs_pos` and the
///                source raises `SHM_BLOCK_FRAMING_INVALID`. Monotonic-regression
///                (`obs_counter < last_observed`) is rejected by the same path
///                (separate branch). The counter also doubles as a richer
///                progress signal for the I12 stall timer (any positive delta
///                evidences producer activity even when no slot reached PUBLISHED).
///   25         → `prepare()` POLLHUP check on control socket fd (T3.2b);
///                mid-publication crash (any slot in WRITING at HUP) is
///                surfaced as SHM_BLOCK_FRAMING_INVALID per `pollable-shm-source.md`
///                AC6 instead of SHM_PRODUCER_DEATH_BEFORE_EOS.
///   26         → adopt() per-column row-count cross-check (T3.1)
class PollableShmSource final : public ISource
{
public:
    /// `header` describes the columns the source emits downstream (the requested subset, in
    /// query order). `full_column_types_` and `full_column_names_` are the FULL producer
    /// schema as declared by the SQL table function — what handshake cross-validation
    /// compares against (preconditions 4–6). `requested_column_names_` is the subset of
    /// `full_column_names_` to actually emit, in downstream order; for `SELECT count()` this
    /// is empty and the source emits a zero-column Chunk with the right `row_count`
    /// (Finding 4 — the wire ABI handshake requires full-schema parity, while the executor
    /// asks the source only for the columns it consumes downstream).
    PollableShmSource(
        SharedHeader header,
        const String & shm_name_,
        std::vector<DataTypePtr> full_column_types_,
        std::vector<String> full_column_names_,
        std::vector<String> requested_column_names_,
        UInt64 stall_timeout_ms_);

    ~PollableShmSource() override;

    PollableShmSource(const PollableShmSource &) = delete;
    PollableShmSource & operator=(const PollableShmSource &) = delete;

    String getName() const override { return "PollableShmSource"; }

    Status prepare() override;
    int schedule() override;
    void onAsyncJobReady() override;

protected:
    std::optional<Chunk> tryGenerate() override;
    void onCancel() noexcept override;

private:
    /// Performs the SHM attach lazily on first prepare(): T1.1 attach (RW),
    /// then schema cross-validation (preconditions 4–6), then receives the
    /// readiness eventfd via the control socket. Done in prepare() (not the
    /// ctor) so that attach-time exceptions surface through the executor's
    /// normal error path.
    void ensureAttached();

    /// Iterates the K slots, validating preconditions 8–10 (state enumerator,
    /// slot identity, monotonic sequence) and precondition 24 (strict
    /// state-machine: counter delta walks the legal EMPTY→WRITING→PUBLISHED→
    /// EMPTY cycle exactly to the observed state — see the precondition-24
    /// note in the class-level map). Returns the PUBLISHED slot with the
    /// lowest unconsumed sequence number, or nullptr if no slot is ready.
    /// Sets `progress_observed` out-param to true when any slot's
    /// `transition_counter` increased since the prior scan (used by the I12
    /// stall timer to count producer activity broader than "reached
    /// PUBLISHED").
    SharedMemoryWire::SlotEntry * findNextReadySlot(bool & progress_observed);

    /// Per-block 3-step RAII sequence: retain → charge → adopt. Builds the
    /// returned Chunk, updates `last_consumed_sequence`, sets `eos_observed`
    /// if the slot's eos_marker is set. Throws on any precondition or limit
    /// violation; the caller (tryGenerate) propagates.
    Chunk drainSlot(SharedMemoryWire::SlotEntry * slot);

    /// (T3.2b) Returns true if POLLHUP is observed on the control socket fd.
    /// Polled non-blocking (timeout=0); used for producer-death-before-EOS
    /// detection per precondition 25.
    bool controlSocketPollHup() const noexcept;

    /// (T3.2b) If async-state and not cancelled/EOS, raise SHM_PRODUCER_STALL
    /// when the stall budget has elapsed since the last observed progress
    /// (publication or readiness wake).
    void checkStallBudget();

    /// (T3.2b) If POLLHUP observed before EOS, throw SHM_PRODUCER_DEATH_BEFORE_EOS.
    /// If POLLHUP observed after EOS, treat as clean shutdown (no throw).
    void checkProducerDeath();

    String shm_name;
    /// Full producer schema (matches handshake exactly). Always passed wholesale into
    /// adopt(); the projection happens after adopt() returns the full Columns vector.
    std::vector<DataTypePtr> full_column_types;
    std::vector<String> full_column_names;
    /// Requested subset in downstream emit order. May be empty (degenerate `SELECT count()`).
    std::vector<String> requested_column_names;
    /// Map from emit position k → index into the full schema. Computed once in
    /// ensureAttached() after handshake cross-validation has confirmed the full schema
    /// matches the producer. Same size as `requested_column_names`; empty when the query
    /// only needs `row_count` (zero-column Chunk).
    std::vector<size_t> projection_indices;
    UInt64 stall_timeout_ms;

    bool attached = false;
    /// Guards a single CurrentMetrics::ShmActiveRegions add/sub pair so a cancel-before-
    /// attach path doesn't sub a metric that was never added (Finding 12).
    bool registered_metric = false;
    /// Shared so the per-RetainToken deleter closure built in `drainSlot` can capture
    /// a copy and keep the consumer's mapping address-valid for the lifetime of every
    /// adopted column, even when the column outlives the source itself (system spec I5;
    /// shm-block-stream spec I11). The source's own reference drops on destruction; the
    /// underlying SharedMemoryRegion is unmapped only once both the source and every
    /// outstanding RetainToken alias have been released.
    std::shared_ptr<SharedMemoryRegion> region;

    /// File descriptors are owned here (RAII closed in dtor). EventFD is not
    /// reused because its ctor opens a fresh eventfd; here we receive an
    /// already-open fd from the producer via SCM_RIGHTS and just store it.
    int control_socket_fd = -1;
    int ready_event_fd = -1;

    AdoptedByteCharger charger;

    /// One entry per slot in the producer's ring. Initialised to 0 in
    /// ensureAttached() once K is known from the handshake.
    std::vector<uint64_t> last_consumed_sequence;

    /// Last observed `transition_counter` for each slot. Atomic-single-variable
    /// modification-order consistency makes this monotonic in time on the wire;
    /// a regression (`obs < prev`) is precondition-24's determinable violation.
    /// A positive delta is producer-progress evidence used by the I12 stall
    /// timer (richer than "saw PUBLISHED"). Same size as `last_consumed_sequence`.
    std::vector<uint64_t> last_observed_transition_counter;

    /// Last observed `state` value for each slot (initialised to EMPTY = 0,
    /// which matches the ftruncate-zeroed state of an unpublished slot at
    /// attach time). Paired with `last_observed_transition_counter` to drive
    /// the strict precondition-24 state-machine check: the legal cycle
    /// EMPTY(0)→WRITING(1)→PUBLISHED(2)→EMPTY(0)→… advances one cycle
    /// position per legal transition AND increments the counter by exactly 1,
    /// so `(prev_state + delta) % 3 == obs_state` is the consistency
    /// invariant whose violation evidences a producer-side state skip. Same
    /// size as `last_consumed_sequence`.
    std::vector<uint32_t> last_observed_state;

    bool is_async_state = false;
    std::atomic<bool> eos_observed{false};
    std::atomic<bool> cancelled{false};

    /// (T3.2b) Stall budget timer. Reset on every observed publication progress
    /// AND on every onAsyncJobReady. Checked at the top of prepare() while
    /// async; if elapsedMilliseconds() > stall_timeout_ms, raise SHM_PRODUCER_STALL.
    Stopwatch stall_timer;
};

}

#endif
