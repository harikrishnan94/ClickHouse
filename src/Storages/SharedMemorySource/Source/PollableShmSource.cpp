#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/Source/PollableShmSource.h>

#include <Storages/SharedMemorySource/Adoption/AdoptionLayer.h>
#include <Storages/SharedMemorySource/Adoption/RetainToken.h>
#include <Storages/SharedMemorySource/Wire/ControlSocket.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>

#include <Columns/ColumnString.h>
#include <Columns/IColumn.h>
#include <DataTypes/DataTypeFactory.h>

#include <Common/CurrentMetrics.h>
#include <Common/Exception.h>

#include <poll.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>


namespace CurrentMetrics
{
    extern const Metric ShmActiveRegions;
}

namespace DB
{

namespace ErrorCodes
{
    extern const int SHM_SCHEMA_MISMATCH;
    extern const int SHM_BLOCK_FRAMING_INVALID;
    extern const int SHM_PRODUCER_STALL;
    extern const int SHM_PRODUCER_DEATH_BEFORE_EOS;
}

using SharedMemoryWire::SlotEntry;
using SharedMemoryWire::SlotState;
using SharedMemoryWire::SchemaEntry;
using SharedMemoryWire::ColumnDescriptor;
using SharedMemoryWire::IMPL_MAX_ROWS_PER_BLOCK;

namespace
{
    /// Precondition-24 strict state-machine check uses raw enum values as
    /// positions in the legal publication cycle EMPTY→WRITING→PUBLISHED→EMPTY.
    /// The cycle math `(prev_pos + delta) % 3 == obs_pos` only works if those
    /// values are exactly {0, 1, 2}; pin the assumption here so a future
    /// (ABI-breaking) reordering in Wire/Layout.h forces a compile error
    /// against this consumer.
    static_assert(static_cast<uint32_t>(SlotState::EMPTY) == 0);
    static_assert(static_cast<uint32_t>(SlotState::WRITING) == 1);
    static_assert(static_cast<uint32_t>(SlotState::PUBLISHED) == 2);

    constexpr const char * slotStateName(uint32_t s) noexcept
    {
        switch (s)
        {
            case static_cast<uint32_t>(SlotState::EMPTY):     return "EMPTY";
            case static_cast<uint32_t>(SlotState::WRITING):   return "WRITING";
            case static_cast<uint32_t>(SlotState::PUBLISHED): return "PUBLISHED";
            default:                                          return "<undefined>";
        }
    }
}


PollableShmSource::PollableShmSource(
    SharedHeader header,
    const String & shm_name_,
    std::vector<DataTypePtr> full_column_types_,
    std::vector<String> full_column_names_,
    std::vector<String> requested_column_names_,
    UInt64 stall_timeout_ms_)
    : ISource(std::move(header))
    , shm_name(shm_name_)
    , full_column_types(std::move(full_column_types_))
    , full_column_names(std::move(full_column_names_))
    , requested_column_names(std::move(requested_column_names_))
    , stall_timeout_ms(stall_timeout_ms_)
{
    chassert(full_column_types.size() == full_column_names.size());
}

PollableShmSource::~PollableShmSource()
{
    /// RAII: region's destructor unmaps + closes the SHM fd. The control
    /// socket fd and the eventfd we hold raw need explicit close (eventfd was
    /// passed in via SCM_RIGHTS so we own it).
    if (ready_event_fd >= 0)
        ::close(ready_event_fd);
    if (control_socket_fd >= 0)
        ::close(control_socket_fd);
    /// Finding 12 — the metric add lives at the end of ensureAttached(); pair it here.
    /// `registered_metric` guards the cancel-before-attach case so we never sub a metric
    /// that was never added. The source-level "attached region" gauge does not wait for
    /// downstream Chunks to release their region_capture aliases — see
    /// CurrentMetrics.cpp's ShmActiveRegions description for the gauge's intent.
    if (registered_metric)
    {
        CurrentMetrics::sub(CurrentMetrics::ShmActiveRegions);
        registered_metric = false;
    }
}


void PollableShmSource::ensureAttached()
{
    /// Step 1: attach + handshake validation (preconditions 1, 2, 3, 7).
    /// Throws SHM_ATTACH_FAILED or SHM_HANDSHAKE_INVALID; the executor surfaces.
    /// The unique_ptr returned by attach converts implicitly into the
    /// shared_ptr member; the explicit move makes the ownership transfer obvious.
    /// `attach` opens the SHM RW so the control-plane writes performed by
    /// `drainSlot` (retain_refcount++) and the RetainToken deleter
    /// (state.store(EMPTY), transition_counter++) land in the mapping.
    region = std::shared_ptr<SharedMemoryRegion>(SharedMemoryRegion::attach(shm_name));

    const auto & hs = region->handshake();

    /// Step 2: schema cross-validation against the FULL SQL-declared schema. The handshake
    /// always describes the producer's complete schema (preconditions 4–6); the requested
    /// subset is a *projection* applied after adopt(), not a different handshake. Validating
    /// against `requested_column_names` here would mis-flag every `SELECT count()` or
    /// per-column projection as SHM_SCHEMA_MISMATCH (Finding 4).

    /// Precondition 4: schema_count matches the full SQL column count.
    if (hs.schema_count != full_column_names.size())
        throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
            "SHM '{}' handshake schema_count={} but SQL-declared columns={} (precondition 4)",
            shm_name, hs.schema_count, full_column_names.size());

    const auto * schema_base = reinterpret_cast<const SchemaEntry *>(
        static_cast<const char *>(region->data()) + hs.schema_table_offset);

    auto & type_factory = DataTypeFactory::instance();
    for (size_t i = 0; i < full_column_names.size(); ++i)
    {
        const auto & entry = schema_base[i];

        /// Precondition 5: name match (NUL-terminated strings, max length already
        /// bounded by `SCHEMA_ENTRY_STR_MAX` at producer write).
        /// strnlen guards against a missing NUL terminator on a non-conforming producer.
        const size_t name_len = ::strnlen(entry.name, SharedMemoryWire::SCHEMA_ENTRY_STR_MAX);
        const String producer_name(entry.name, name_len);
        if (producer_name != full_column_names[i])
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "SHM '{}' column {}: handshake name='{}' but SQL name='{}' (precondition 5)",
                shm_name, i, producer_name, full_column_names[i]);

        /// Precondition 6 (handshake equality + parse-via-factory side):
        const size_t type_len = ::strnlen(entry.type_string, SharedMemoryWire::SCHEMA_ENTRY_STR_MAX);
        const String producer_type_str(entry.type_string, type_len);
        DataTypePtr producer_type;
        try
        {
            producer_type = type_factory.get(producer_type_str);
        }
        catch (const Exception & e)
        {
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "SHM '{}' column {}: handshake type='{}' does not parse via DataTypeFactory: {} (precondition 6)",
                shm_name, i, producer_type_str, e.message());
        }

        if (!producer_type->equals(*full_column_types[i]))
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "SHM '{}' column {}: handshake type='{}' but SQL type='{}' (precondition 6)",
                shm_name, i, producer_type->getName(), full_column_types[i]->getName());

        /// Belt-and-braces: post-handshake membership re-check on the producer's parsed
        /// type. The SQL-side gate in T3.4 already enforced this for full_column_types; a
        /// matching producer type therefore also passes — but we keep the check explicit
        /// so a future SQL gate regression doesn't silently widen the supported set.
        const auto producer_type_id = producer_type->getTypeId();
        if (producer_type_id != TypeIndex::UInt64 && producer_type_id != TypeIndex::String)
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "SHM '{}' column {}: handshake type='{}' is outside the supported set "
                "{{UInt64, String}} (precondition 6)",
                shm_name, i, producer_type->getName());
    }

    /// Step 2b: build the projection index map. With the full schema validated above we
    /// know every requested name exists in `full_column_names`. We resolve each requested
    /// name to its position in the full schema (the emit order matches the request order,
    /// not the full-schema order). Linear scan: schemas in phase 1 are bounded by
    /// IMPL_MAX_COLUMNS=64 so O(N*M) is fine.
    projection_indices.clear();
    projection_indices.reserve(requested_column_names.size());
    for (const auto & requested : requested_column_names)
    {
        bool found = false;
        for (size_t j = 0; j < full_column_names.size(); ++j)
        {
            if (full_column_names[j] == requested)
            {
                projection_indices.push_back(j);
                found = true;
                break;
            }
        }
        if (!found)
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "SHM '{}' requested column '{}' is not in the producer-declared schema "
                "(this is an orchestration bug — StorageShm should have rejected at parse time)",
                shm_name, requested);
    }

    /// Step 3: receive the readiness eventfd. The connection socket fd is also
    /// kept open for POLLHUP-based producer-death detection (precondition 25
    /// and `shm-block-stream.md` I11).
    ready_event_fd = ControlSocketClient::connectAndReceiveEventFd(
        controlSocketPathForShmName(shm_name), control_socket_fd);

    /// Step 4: per-slot bookkeeping; sequence numbers start at 0 (first publish
    /// stores sequence=1, which precondition 10 requires to be strictly greater).
    /// Pair the precondition-24 trackers: `transition_counter` starts at 0 (any
    /// positive observation counts as progress) and `last_observed_state` starts
    /// at EMPTY (= 0). EMPTY matches the ftruncate-zeroed state of an unpublished
    /// slot, so the cycle-position check `(prev_state + delta) % 3 == obs_state`
    /// validates cleanly on the consumer's very first scan against a fresh
    /// producer SHM.
    last_consumed_sequence.assign(hs.ring_depth_k, 0);
    last_observed_transition_counter.assign(hs.ring_depth_k, 0);
    last_observed_state.assign(hs.ring_depth_k, static_cast<uint32_t>(SlotState::EMPTY));

    attached = true;
    /// Finding 12 — bump the "currently-attached regions" gauge after the attach has
    /// fully succeeded (so a throwing handshake doesn't leak an increment). The dtor
    /// pairs the decrement via `registered_metric`.
    CurrentMetrics::add(CurrentMetrics::ShmActiveRegions);
    registered_metric = true;
    /// Reset the stall timer at attach so the first wait isn't counted from
    /// PollableShmSource ctor time.
    stall_timer.restart();
}


SlotEntry * PollableShmSource::findNextReadySlot(bool & progress_observed)
{
    chassert(attached);
    progress_observed = false;
    const auto & hs = region->handshake();
    auto * base = const_cast<char *>(static_cast<const char *>(region->data())) + hs.slot_table_offset;

    SlotEntry * chosen = nullptr;
    uint64_t chosen_seq = std::numeric_limits<uint64_t>::max();

    for (uint32_t i = 0; i < hs.ring_depth_k; ++i)
    {
        auto * slot = reinterpret_cast<SlotEntry *>(base + i * hs.slot_table_stride);

        /// Precondition 8: the state value must be a defined enumerator. acquire-load
        /// gives us the memory-ordering contract the wire promises (the producer
        /// release-stores `state` after writing the slot's payload and metadata).
        const uint32_t state = slot->state.load(std::memory_order_acquire);
        if (state != static_cast<uint32_t>(SlotState::EMPTY)
            && state != static_cast<uint32_t>(SlotState::WRITING)
            && state != static_cast<uint32_t>(SlotState::PUBLISHED))
        {
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "SHM '{}' slot {}: observed state={} is not a defined SlotState (precondition 8)",
                shm_name, i, state);
        }

        /// Precondition 24 (`pollable-shm-source.md` row 24, F4): transitions
        /// follow the publication state machine in order. The wire's per-slot
        /// `transition_counter` (Layout.h `SlotEntry`) is incremented under
        /// release ordering by EVERY state transition driver — producer on E→W
        /// and W→P, consumer on P→E — i.e. exactly +1 per legal transition. A
        /// full publish cycle is +3.
        ///
        /// Monotonic-regression branch (the determinable subset previously
        /// caught here): the wire is single-modification-order on the counter,
        /// so any observation below the previous one is an unambiguous wire
        /// violation. Kept as a fast-path check before the cycle-walk math.
        ///
        /// Cycle-position branch (new in F4 — strict): walk the legal cycle
        /// EMPTY(0)→WRITING(1)→PUBLISHED(2)→EMPTY(0)→… by `delta` steps from
        /// `prev_state`; the result MUST equal the observed state. A producer
        /// that skips a state (e.g. EMPTY→PUBLISHED directly) bumps the
        /// counter by fewer steps than the legal walk to the observed state,
        /// and the cyclic check `(prev_pos + delta) % 3 != obs_pos` catches
        /// it. The enum values pinned at file scope above make `state` itself
        /// the cycle position (no separate lookup table needed).
        const uint64_t obs_counter = slot->transition_counter.load(std::memory_order_acquire);
        const uint64_t prev_counter = last_observed_transition_counter[i];
        if (obs_counter < prev_counter)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "SHM '{}' slot {}: transition_counter regressed (observed={}, "
                "last={}) — precondition 24 violation (atomic single-variable "
                "consistency forbids non-monotonic observations)",
                shm_name, i, obs_counter, prev_counter);
        if (obs_counter > prev_counter)
            progress_observed = true;

        const uint32_t prev_state = last_observed_state[i];
        const uint64_t delta = obs_counter - prev_counter;
        const uint64_t expected_pos =
            (static_cast<uint64_t>(prev_state) + delta) % 3;
        uint32_t observed_state = state;
        if (expected_pos != static_cast<uint64_t>(observed_state))
        {
            /// The producer's E→W transition pairs a release counter
            /// `fetch_add` with a relaxed `state.store` (InProcessProducer
            /// E→W path), so under weak ordering we may catch the counter
            /// store visible but the state store not yet propagated. Re-read
            /// `state` with acquire ordering — if it has now caught up to
            /// the expected position the check passes (a benign in-flight
            /// transition); only a persistent mismatch indicates a real
            /// skip-the-state violation.
            observed_state = slot->state.load(std::memory_order_acquire);
            if (expected_pos != static_cast<uint64_t>(observed_state))
                throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                    "SHM '{}' slot {}: state-machine violation (precondition 24) — "
                    "prev_state={} obs_state={} counter_delta={} but the legal "
                    "EMPTY→WRITING→PUBLISHED→EMPTY walk lands at cycle position {} "
                    "after {} transition(s), not at obs_state's position {}",
                    shm_name, i, slotStateName(prev_state), slotStateName(observed_state),
                    delta, expected_pos, delta, static_cast<uint64_t>(observed_state));
        }
        last_observed_transition_counter[i] = obs_counter;
        last_observed_state[i] = observed_state;

        if (observed_state != static_cast<uint32_t>(SlotState::PUBLISHED))
            continue;

        /// Precondition 9: slot identity self-check.
        if (slot->slot_index != i)
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "SHM '{}' slot {}: slot_index={} != position {} (precondition 9)",
                shm_name, i, slot->slot_index, i);

        const uint64_t seq = slot->sequence.load(std::memory_order_acquire);

        /// Precondition 10: monotonically increasing per-slot sequence number. We split the
        /// old `seq <= last_consumed` check into two cases:
        ///   - `seq <  last_consumed[i]` — producer regression (real wire violation).
        ///   - `seq == last_consumed[i]` — this is the *same* block we already drained, still
        ///     PUBLISHED because the consumer's RetainToken alias has not yet dropped (the
        ///     chunk is in-flight / held). Per the new release contract, the consumer
        ///     transitions state→EMPTY only on last-alias drop, so re-observing
        ///     state==PUBLISHED with seq==last_consumed is legitimate and means "skip,
        ///     awaiting release". Throwing here would mis-report a live, valid retained
        ///     block as a wire violation (Finding 3).
        if (seq < last_consumed_sequence[i])
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "SHM '{}' slot {}: observed sequence={} < last_consumed={} (precondition 10)",
                shm_name, i, seq, last_consumed_sequence[i]);
        if (seq == last_consumed_sequence[i])
            continue;

        if (seq < chosen_seq)
        {
            chosen_seq = seq;
            chosen = slot;
        }
    }

    return chosen;
}


Chunk PollableShmSource::drainSlot(SlotEntry * slot)
{
    /// 3-step RAII per-block adoption sequence
    /// (`pollable-shm-source.md` §Per-block adoption call; `system.md` I5, I10).
    ///
    /// Step 1: increment retain refcount on the slot (0 -> 1 transition or
    ///         higher; producer is contracted to wait on refcount==0 before reuse).
    /// Step 2: charge adopted bytes against the MemoryTracker via AdoptedByteCharger
    ///         (may throw MEMORY_LIMIT_EXCEEDED).
    /// Step 3: construct retain_token whose deleter decrements the same refcount,
    ///         transferring ownership of the increment out of this stack frame.
    /// Step 4: call adopt() to construct the columns; on success ownership of both
    ///         retain_token and the charge handle transfer into the columns.
    ///
    /// On any throw between step 1 and step 3 we must release the retain manually
    /// (the lambda `release_retain_if_local` below); after step 3 the RetainToken's
    /// destructor handles it. After step 4's success, no rollback is needed.

    const uint64_t this_seq = slot->sequence.load(std::memory_order_acquire);
    const auto & hs = region->handshake();

    /// Precondition 11: row_count must fit the implementation cap. Acquire-load is
    /// implied by the previous acquire on `slot->state == PUBLISHED` from
    /// findNextReadySlot() (the wire pins payload-before-metadata, and `row_count`
    /// is written before the release-store on `state`).
    if (slot->row_count > IMPL_MAX_ROWS_PER_BLOCK)
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "SHM '{}' slot {}: row_count={} > IMPL_MAX_ROWS_PER_BLOCK={} (precondition 11)",
            shm_name, slot->slot_index, slot->row_count, IMPL_MAX_ROWS_PER_BLOCK);

    /// Step 1.
    slot->retain_refcount.fetch_add(1, std::memory_order_acq_rel);
    bool retain_owned_here = true;
    auto release_retain_if_local = [&]() noexcept
    {
        if (retain_owned_here)
        {
            slot->retain_refcount.fetch_sub(1, std::memory_order_acq_rel);
            retain_owned_here = false;
        }
    };

    try
    {
        /// Read per-block descriptor array. The descriptors live at
        /// data_region_base + per_column_descriptors_offset. Bounds-check the offset
        /// to be defensive (the producer writes both fields under the same release
        /// barrier, but a misbehaving producer could set per_column_descriptors_offset
        /// past data_region_size — adopt() catches this only as offset_offset bounds).
        const auto * data_region = static_cast<const char *>(region->data()) + hs.data_region_offset;
        const size_t descs_bytes = full_column_types.size() * sizeof(ColumnDescriptor);

        if (slot->per_column_descriptors_offset > hs.data_region_size
            || slot->per_column_descriptors_offset + descs_bytes > hs.data_region_size
            || slot->per_column_descriptors_offset + descs_bytes < slot->per_column_descriptors_offset)
        {
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "SHM '{}' slot {}: per_column_descriptors_offset={} + {}*{} overflows data_region_size={} "
                "(precondition 12)",
                shm_name, slot->slot_index, slot->per_column_descriptors_offset,
                full_column_types.size(), sizeof(ColumnDescriptor), hs.data_region_size);
        }

        const auto * descs = reinterpret_cast<const ColumnDescriptor *>(
            data_region + slot->per_column_descriptors_offset);
        std::vector<ColumnDescriptor> descs_vec(descs, descs + full_column_types.size());

        /// Step 2: charge adopted bytes.
        size_t adopted_bytes = 0;
        size_t logical_bytes = 0;
        for (const auto & d : descs_vec)
        {
            if (d.type == static_cast<uint32_t>(SharedMemoryWire::WireColumnType::UInt64))
            {
                logical_bytes += d.value_count * sizeof(uint64_t);
                adopted_bytes += d.value_count * sizeof(uint64_t) + d.value_padding;
            }
            else if (d.type == static_cast<uint32_t>(SharedMemoryWire::WireColumnType::String))
            {
                /// String charge components (system.md glossary "Adopted byte
                /// count" + ColumnString::createAdopted contract + PaddedPODArray
                /// pad_left convention):
                ///   - `value_count` bytes of chars + `value_padding` trailing safe-read pad.
                ///   - 8-byte zero-sentinel preceding `offsets[0]` (the producer pinned
                ///     this in the slot, the consumer reads it via `offsets[-1]` per
                ///     `ColumnString::offsetAt(0)`; AdoptionLayer's `validateStringDescriptor`
                ///     enforces `offsets_offset >= 8` and the pre-offsets byte == 0).
                ///     F8: previously omitted; the sentinel IS part of the adopted region.
                ///   - `offsets_count * 8` bytes of offsets + `offsets_padding` trailing pad.
                logical_bytes += d.value_count + d.offsets_count * sizeof(uint64_t);
                adopted_bytes += d.value_count + d.value_padding
                              + sizeof(uint64_t) /* offsets[-1] sentinel */
                              + d.offsets_count * sizeof(uint64_t) + d.offsets_padding;
            }
            /// Unknown wire types fall through; adopt() will raise SHM_SCHEMA_MISMATCH.
            /// We avoid charging for an unknown type so a malformed descriptor doesn't
            /// inflate the MemoryTracker before the error fires.
        }

        ChargeHandle charge_handle = charger.charge(adopted_bytes, logical_bytes);

        /// Step 3: build the RetainToken. From this point on, releasing the retain is
        /// the RetainToken's responsibility — so we clear the local guard flag
        /// BEFORE constructing the token. If makeRetainToken's allocation throws
        /// std::bad_alloc, the local guard is back in charge (we restore it). The
        /// alloc/construct sequence is so tight that any throw here is benign;
        /// charge_handle's local destructor still releases the charge.
        ///
        /// Lifetime correctness (system spec I5 + shm-block-stream spec I11): capture
        /// a `shared_ptr<SharedMemoryRegion>` copy into the deleter closure so the
        /// consumer's mapping stays address-valid for the entire life of every
        /// RetainToken alias — including aliases held by adopted columns that
        /// outlive the source. The deleter's `slot_capture->retain_refcount` write
        /// is into that pinned mapping, so it remains a valid memory access when the
        /// callback fires at last-alias drop.
        SlotEntry * slot_capture = slot;
        auto region_capture = region;
        retain_owned_here = false;
        RetainToken retain_token;
        try
        {
            retain_token = makeRetainToken([slot_capture, region_capture]() noexcept
            {
                /// Release contract (shm-block-stream.md §Publication state machine):
                /// the consumer drives the PUBLISHED→released transition. On the LAST
                /// retain-refcount decrement (fetch_sub returning 1) we release-store
                /// SlotState::EMPTY into the slot, which is the producer's wait-loop
                /// condition for slot reuse. The producer no longer polls retain_refcount;
                /// it polls `state == EMPTY` (Findings 1 + 3). Storing EMPTY *after* the
                /// fetch_sub guarantees the producer cannot observe EMPTY while another
                /// alias still pins the slot.
                ///
                /// Pair the P→E state store with a `transition_counter` bump per the
                /// precondition-24 protocol (Layout.h SlotEntry). The counter bump
                /// happens BEFORE the state store, both with release ordering, so the
                /// producer's subsequent acquire-load of `state == EMPTY` implies the
                /// counter update is visible to anyone (consumer side included).
                if (slot_capture->retain_refcount.fetch_sub(1, std::memory_order_acq_rel) == 1)
                {
                    slot_capture->transition_counter.fetch_add(1, std::memory_order_release);
                    slot_capture->state.store(static_cast<uint32_t>(SlotState::EMPTY),
                                              std::memory_order_release);
                }
                /// `region_capture` is destroyed when this lambda is destroyed (after
                /// the last RetainToken alias drops); its destruction may run
                /// ~SharedMemoryRegion if it was the last shared_ptr to the mapping.
            });
        }
        catch (...)
        {
            retain_owned_here = true;
            throw;
        }

        /// Step 4: adopt — descriptor-level validation + column construction. We always
        /// adopt the FULL producer schema so handshake parity is preserved; projection
        /// happens below.
        std::vector<std::pair<std::string, DataTypePtr>> schema;
        schema.reserve(full_column_types.size());
        for (size_t i = 0; i < full_column_types.size(); ++i)
            schema.emplace_back(full_column_names[i], full_column_types[i]);

        Columns full_cols = adopt(descs_vec, schema, data_region, hs.data_region_size,
                                  slot->row_count, std::move(retain_token), std::move(charge_handle));

        /// Step 5: project down to the requested subset (Finding 4). The columns we drop
        /// release their RetainToken / ChargeHandle aliases here; the remaining columns
        /// keep at least one alias of each shared_ptr, so refcounts stay positive until
        /// the emitted Chunk is dropped downstream. For the degenerate `SELECT count()`
        /// case (`projection_indices` empty), we emit a zero-column Chunk carrying just
        /// the row count — ClickHouse's convention for source-side count(). The full
        /// adopt() still ran, which is what releases the slot at chunk-drop time.
        Columns emitted_cols;
        emitted_cols.reserve(projection_indices.size());
        for (size_t idx : projection_indices)
            emitted_cols.push_back(full_cols[idx]);
        /// Drop the un-emitted aliases promptly — the columns themselves we no longer
        /// reference. (RAII would do this at scope exit; explicit clear documents intent.)
        full_cols.clear();

        /// Step 6: lazy content-level validation (preconditions 21 + 22) for any emitted
        /// ColumnString. We only validate columns we actually emit — preconditions 21/22
        /// are "lazy; performed only if a downstream read would be invalidated by
        /// violation", and we never expose the dropped columns to a downstream reader.
        for (const auto & col : emitted_cols)
        {
            if (const auto * cs = typeid_cast<const ColumnString *>(col.get()))
                cs->validateAdoptedOffsets();
        }

        /// Record this seq as consumed so the next findNextReadySlot() round
        /// doesn't re-pick this slot before the producer republishes (the
        /// `seq == last_consumed[i]` skip branch handles the in-flight case).
        last_consumed_sequence[slot->slot_index] = this_seq;

        /// EOS detection per `shm-block-stream.md` §End-of-stream. Setting
        /// `eos_observed` here arms the precondition-23 check in tryGenerate: any
        /// PUBLISHED slot subsequently observed by the consumer will throw
        /// SHM_BLOCK_FRAMING_INVALID rather than be silently drained.
        if (slot->eos_marker.load(std::memory_order_acquire) != 0)
            eos_observed.store(true, std::memory_order_release);

        /// Successful drain: progress observed → reset stall timer.
        stall_timer.restart();
        is_async_state = false;

        return Chunk(std::move(emitted_cols), slot->row_count);
    }
    catch (...)
    {
        release_retain_if_local();
        throw;
    }
}


ISource::Status PollableShmSource::prepare()
{
    /// I9: cancellation is the very first check, before any other work or wait.
    if (cancelled.load(std::memory_order_acquire) || isCancelled())
    {
        cancelled.store(true, std::memory_order_release);
        return ISource::prepare();
    }

    if (!attached)
    {
        ensureAttached();
    }

    /// (T3.2b) Stall + producer-death detection are evaluated only while we're
    /// already async-waiting; before then the source is making forward progress.
    if (is_async_state)
    {
        checkProducerDeath();
        checkStallBudget();
        return Status::Async;
    }

    return ISource::prepare();
}


std::optional<Chunk> PollableShmSource::tryGenerate()
{
    if (cancelled.load(std::memory_order_acquire) || isCancelled())
        return std::nullopt;

    /// Re-check attach state defensively (prepare() handles the first call but
    /// the contract permits an executor to call tryGenerate without an
    /// intervening prepare in some edge paths).
    if (!attached)
        ensureAttached();

    bool progress_observed = false;
    auto * slot = findNextReadySlot(progress_observed);
    /// I12 + F5: any positive `transition_counter` delta is producer-progress
    /// evidence, even when no slot reached PUBLISHED this scan. Reset the stall
    /// timer here so a producer that is actively cycling slots (E→W→P→E→W ...)
    /// against an in-flight retain alias doesn't appear stalled.
    if (progress_observed)
        stall_timer.restart();
    if (slot != nullptr)
    {
        /// Precondition 23 (`pollable-shm-source.md` §Producer-side preconditions row
        /// 23 / shm-block-stream §End-of-stream): once a block is published with the
        /// EOS marker set, the producer commits to no further publications. Any later
        /// PUBLISHED slot observed by the consumer is a wire-contract violation.
        /// The check fires only AFTER findNextReadySlot returns non-null AND
        /// `eos_observed` is already true; if EOS flips during the same drainSlot
        /// invocation (the one that observes the EOS marker), this block emits and the
        /// NEXT tryGenerate call sees `eos_observed == true` and any further published
        /// slot throws here.
        if (eos_observed.load(std::memory_order_acquire))
        {
            const uint64_t observed_seq = slot->sequence.load(std::memory_order_acquire);
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "SHM '{}': producer published slot_index={} sequence={} AFTER EOS was "
                "already signalled (precondition 23)",
                shm_name, slot->slot_index, observed_seq);
        }
        return drainSlot(slot);
    }

    /// No drainable slot:
    ///   - if EOS has been observed AND no slot is published, we're done.
    ///   - otherwise, go async: ISource::work treats an empty Chunk as "yield
    ///     but not done"; prepare() will then return Async, schedule() exposes
    ///     the eventfd, and the executor will wake us via onAsyncJobReady.
    if (eos_observed.load(std::memory_order_acquire))
        return std::nullopt;

    is_async_state = true;
    return Chunk{};
}


int PollableShmSource::schedule()
{
    /// I6: schedule must return the readiness fd; the executor polls it for
    /// EPOLLIN (eventfd readability). The fd is owned by this source for the
    /// lifetime of `attached == true`; dtor closes it.
    chassert(ready_event_fd >= 0);
    return ready_event_fd;
}


void PollableShmSource::onAsyncJobReady()
{
    /// I6 (Pollable contract): the executor invokes this method when the eventfd
    /// returned by `schedule()` is readable. Our job is to drain the counter and
    /// hand control back immediately — under NO circumstances do we block.
    ///
    /// The eventfd is non-blocking (Common/EventFD.cpp F4 default + the
    /// defensive fcntl in ControlSocketClient::connectAndReceiveEventFd), so a
    /// `read` with no pending writes returns -1/EAGAIN rather than blocking the
    /// executor thread. EAGAIN is BENIGN — the most common cause is a spurious
    /// wake or a previous drain by an interleaved scan; the next prepare() will
    /// re-evaluate state. EINTR loops (slow-syscall signal interruption); every
    /// other errno (EBADF after close, EIO, etc.) is also non-fatal here — the
    /// next prepare()/tryGenerate() will pick up cancellation, EOS, or
    /// producer-death classification.
    ///
    /// F5: do NOT reset the stall timer here. Per spec I12, publication progress
    /// is defined as a slot transitioning to PUBLISHED (with the `transition_counter`
    /// delta picking up the broader "producer cycled a slot" case in
    /// `findNextReadySlot`). A spurious wake — eventfd readability without a
    /// corresponding publication — must NOT count as progress, otherwise a
    /// pathological producer can spin the eventfd write indefinitely and the
    /// stall timer never trips.
    uint64_t buf = 0;
    ssize_t r;
    do
    {
        r = ::read(ready_event_fd, &buf, sizeof(buf));
    } while (r < 0 && errno == EINTR);
    if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    {
        /// Non-blocking eventfd was already drained — benign no-op.
    }

    is_async_state = false;
}


void PollableShmSource::onCancel() noexcept
{
    cancelled.store(true, std::memory_order_release);

    /// I9: unblock any executor wait on the readiness fd. We write rather than
    /// close because close-while-poll has a small window where the executor
    /// may not have epoll-registered yet; the eventfd write is the canonical
    /// wake. The actual fd close happens in the dtor.
    ///
    /// The fd is non-blocking after attach (Common/EventFD.cpp F4 default +
    /// the defensive fcntl in ControlSocketClient::connectAndReceiveEventFd),
    /// so a saturated eventfd counter — the counter reaches
    /// 0xFFFFFFFFFFFFFFFE, an extreme corner — returns EAGAIN rather than
    /// blocking the cancelling thread (which would itself violate I9). We
    /// silently ignore the EAGAIN: cancellation has already been latched via
    /// `cancelled.store(true)` above, and the executor's next poll will see
    /// `prepare()` short-circuit on that flag without needing a wake.
    if (ready_event_fd >= 0)
    {
        const uint64_t one = 1;
        [[maybe_unused]] ssize_t w = ::write(ready_event_fd, &one, sizeof(one));
    }
}


/// =====================================================================
/// T3.2b — Stall + producer-death detection.
/// =====================================================================

bool PollableShmSource::controlSocketPollHup() const noexcept
{
    if (control_socket_fd < 0)
        return false;

    pollfd pfd{};
    pfd.fd = control_socket_fd;
    pfd.events = 0; // we only care about POLLHUP/POLLERR/POLLNVAL revents.
    const int rc = ::poll(&pfd, 1, /*timeout_ms=*/0);
    if (rc <= 0)
        return false;
    /// POLLHUP fires when the peer closes its end of the socket — for our
    /// control socket that happens when the producer process exits (or its
    /// ControlSocketServer is torn down). POLLERR / POLLNVAL also indicate
    /// the fd is no longer valid which we treat as producer death.
    return (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0;
}


void PollableShmSource::checkStallBudget()
{
    /// I12: only check while async-waiting, not cancelled, not EOS.
    if (!is_async_state)
        return;
    if (cancelled.load(std::memory_order_acquire))
        return;
    if (eos_observed.load(std::memory_order_acquire))
        return;

    const uint64_t elapsed = stall_timer.elapsedMilliseconds();
    if (elapsed > stall_timeout_ms)
        throw Exception(ErrorCodes::SHM_PRODUCER_STALL,
            "SHM '{}': no producer publication progress for {}ms (stall_timeout_ms={})",
            shm_name, elapsed, stall_timeout_ms);
}


void PollableShmSource::checkProducerDeath()
{
    /// `shm-block-stream.md` I11 + precondition 25: producer-death detection via
    /// the control plane. POLLHUP-after-EOS is a clean shutdown (the producer
    /// is contracted to detach after EOS); POLLHUP-before-EOS surfaces per
    /// `pollable-shm-source.md` AC6 in two flavours:
    ///   - "producer dying after publishing a complete block but before
    ///     signalling end-of-stream" → SHM_PRODUCER_DEATH_BEFORE_EOS (778).
    ///   - "producer crash mid-publication (before a complete block is
    ///     published)" → SHM_BLOCK_FRAMING_INVALID (775).
    /// We discriminate by scanning the slot table: any slot stuck in WRITING
    /// when the producer is gone is exactly the mid-publication signature
    /// (the producer transitioned E→W but never reached W→P). All other
    /// post-EOS-or-EMPTY-or-PUBLISHED arrangements are clean death.
    /// The retain protocol keeps any in-flight adopted columns' mappings valid
    /// until they drop (I11 last sentence).
    if (!controlSocketPollHup())
        return;
    if (eos_observed.load(std::memory_order_acquire))
        return;
    if (cancelled.load(std::memory_order_acquire))
        return;

    /// Scan slots for the mid-publication signature. Use acquire-loads so that
    /// if we see WRITING the producer's prior transition_counter bump (E→W) is
    /// also synchronously visible (not strictly required for the decision, but
    /// keeps the read ordered against any subsequent state inspection).
    const auto & hs = region->handshake();
    auto * base = const_cast<char *>(static_cast<const char *>(region->data())) + hs.slot_table_offset;
    for (uint32_t i = 0; i < hs.ring_depth_k; ++i)
    {
        auto * slot = reinterpret_cast<SlotEntry *>(base + i * hs.slot_table_stride);
        const uint32_t s = slot->state.load(std::memory_order_acquire);
        if (s == static_cast<uint32_t>(SlotState::WRITING))
        {
            throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
                "SHM '{}': producer died mid-publication (slot {} in WRITING "
                "state with no further producer activity possible) — "
                "precondition 25 + AC6 mid-publication branch",
                shm_name, i);
        }
    }

    throw Exception(ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS,
        "SHM '{}': producer control socket closed before end-of-stream "
        "(precondition 25; producer-death-before-eos class)",
        shm_name);
}

}

#endif
