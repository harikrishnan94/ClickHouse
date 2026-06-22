#pragma once

/// On-wire byte layout for the SHM-adoption ABI (version 1).
///
/// This header is the C++ source of truth for the producer/consumer wire
/// described in the in-tree doc `docs/en/development/shm-block-stream-abi-v1.md`
/// and the spec at `streamed_table/specs/shm-block-stream.md`. The doc enumerates
/// the semantic obligations; this header pins the byte layouts that satisfy
/// them. Together they constitute the AC8 versioned ABI artifact (see
/// `shm-block-stream.md` §Acceptance criteria, AC8).
///
/// Every struct below describes a region of bytes that lives inside the SHM
/// object the producer creates. The consumer attaches read-only and reads
/// these regions through acquire-loads of the atomic fields. The producer
/// writes them in the order pinned by `shm-block-stream.md` §Memory ordering:
/// payload bytes are released BEFORE the metadata write that publishes the
/// block.
///
/// ABI compatibility: changes to the byte layout, the enum values, the magic,
/// or the `SHM_ABI_VERSION_*` constants are explicit versioned bumps per
/// `shm-block-stream.md` §I2. The static_asserts below catch silent layout
/// drift at compile time.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>


namespace DB::SharedMemoryWire
{

/// ABI identity sentinel. On little-endian targets (x86_64, ARM64) the
/// uint64_t value `0x504F44415F4D4853ULL` lays out in memory as the byte
/// sequence `53 48 4D 5F 41 44 4F 50` = ASCII "SHM_ADOP". The wire ABI is
/// little-endian per AC8; phase 1 ships LE-only platforms.
///
/// Producer release-stores this into `HandshakeRegion::magic` AFTER all other
/// handshake fields are written; consumer's acquire-load of this value implies
/// acquire of every other handshake field.
inline constexpr uint64_t SHM_MAGIC = 0x504F44415F4D4853ULL;

/// Phase-1 ABI version. Any bump to the byte layouts or semantics below
/// requires a new constant and an explicit cross-validation by the consumer
/// (`shm-block-stream.md` §ABI version negotiation, precondition 2 in
/// `pollable-shm-source.md` §Producer-side preconditions enumerated).
inline constexpr uint32_t SHM_ABI_VERSION_1 = 1;

/// Implementation-defined upper bound on `HandshakeRegion::ring_depth_K`.
/// Consumer rejects values exceeding this (precondition 3).
inline constexpr uint32_t IMPL_MAX_K = 256;

/// Implementation-defined upper bound on `SlotEntry::row_count`.
/// Consumer rejects values exceeding this (precondition 11).
inline constexpr uint32_t IMPL_MAX_ROWS_PER_BLOCK = 1u << 20;

/// Implementation-defined upper bound on `HandshakeRegion::schema_count`.
/// Bounds the schema table footprint (`schema_count * sizeof(SchemaEntry)`).
inline constexpr uint32_t IMPL_MAX_COLUMNS = 64;

/// Safe-read padding required by ClickHouse's column-storage contract
/// (`PaddedPODArray<T>` contract). This matches `PADDING_FOR_SIMD` defined in
/// `src/Core/Defines.h`. Producer descriptors that declare less padding than
/// this for any column buffer are rejected at the per-column descriptor check
/// (preconditions 15, 20).
///
/// NB: This constant is part of the wire ABI. Any change to ClickHouse's
/// internal `PADDING_FOR_SIMD` that diverges from this value requires an ABI
/// version bump.
inline constexpr size_t PADDING_FOR_SIMD = 64;

/// Maximum length of a column name or type-string in `SchemaEntry`, including
/// the terminating NUL. A producer that emits a longer string truncates and
/// the consumer detects the mismatch at handshake cross-validation
/// (precondition 5 for names, precondition 6 for type strings).
inline constexpr size_t SCHEMA_ENTRY_STR_MAX = 64;

/// Publication state machine values for `SlotEntry::state`
/// (`shm-block-stream.md` §Publication state machine, precondition 8).
/// The transition sequence is EMPTY → WRITING → PUBLISHED → EMPTY. The
/// producer drives EMPTY→WRITING→PUBLISHED; the consumer drives the
/// PUBLISHED→EMPTY transition by release-storing EMPTY when its last
/// RetainToken alias for the slot drops (the slot is now "released" — the
/// wire's named state — and EMPTY is the byte-level encoding of that state).
/// The producer's reuse wait polls `state == EMPTY` rather than the retain
/// refcount itself, so an unconsumed PUBLISHED slot (refcount still 0
/// because the consumer has not yet attached) is never silently overwritten.
/// Any observed value outside this set is a block-framing-invalid class
/// violation.
enum class SlotState : uint32_t
{
    EMPTY = 0,
    WRITING = 1,
    PUBLISHED = 2,
};

/// Per-column descriptor type tag (`adoption-layer.md` §AC2 Type coverage).
/// Phase-1 covers `ColumnVector<UInt64>` and `ColumnString` only; any other
/// value is a buffer-layout-invalid class violation.
enum class WireColumnType : uint32_t
{
    UInt64 = 1,
    String = 2,
};

/// Per-column descriptor — one entry per declared schema column in every
/// block, located at `data_region_base + SlotEntry::per_column_descriptors_offset`.
/// (`shm-block-stream.md` §Per-type buffer layout, §Block framing).
///
/// The two type variants share this struct; unused fields are zero. The
/// per-type field meanings:
///
/// - WireColumnType::UInt64:
///       value_offset  — byte offset (within data region) of the value buffer.
///                       Must satisfy `UInt64` alignment (precondition 13).
///       value_count   — element count; must equal SlotEntry::row_count
///                       (preconditions 14, 26).
///       value_padding — trailing safe-read padding bytes; must be
///                       >= PADDING_FOR_SIMD (precondition 15).
///       offsets_*     — unused; producer sets to 0.
///
/// - WireColumnType::String:
///       value_offset   — `chars` byte offset; UInt8 alignment is trivial
///                        (precondition 16).
///       value_count    — `chars` byte size (precondition 18).
///       value_padding  — `chars` trailing safe-read padding; must be
///                        >= PADDING_FOR_SIMD (precondition 20).
///       offsets_offset — `offsets` byte offset; must satisfy `UInt64`
///                        alignment (precondition 17).
///       offsets_count  — element count of the `offsets` buffer; must equal
///                        SlotEntry::row_count (preconditions 19, 26).
///       offsets_padding — `offsets` trailing safe-read padding; must be
///                         >= PADDING_FOR_SIMD (precondition 20).
///
/// Read by the consumer only after the owning slot's `state` is observed
/// PUBLISHED under acquire ordering; the producer must finish writing the
/// descriptor and the payload it points at BEFORE the release-store that
/// publishes the slot (`shm-block-stream.md` §Memory ordering).
struct alignas(8) ColumnDescriptor
{
    uint32_t type;             ///< `WireColumnType` numeric value.
    uint32_t reserved_pad32;   ///< Padding to 8-byte alignment; producer MBZ.
    uint64_t value_offset;
    uint64_t value_count;
    uint64_t value_padding;
    uint64_t offsets_offset;   ///< 0 for UInt64.
    uint64_t offsets_count;    ///< 0 for UInt64.
    uint64_t offsets_padding;  ///< 0 for UInt64.
};

static_assert(std::is_trivially_copyable_v<ColumnDescriptor>);
static_assert(std::is_standard_layout_v<ColumnDescriptor>);
static_assert(sizeof(ColumnDescriptor) == 56);
static_assert(alignof(ColumnDescriptor) == 8);

/// Fixed-width per-column schema entry — one per declared column in the
/// `schema_table` located at `data_region_base + HandshakeRegion::schema_table_offset`.
/// (`shm-block-stream.md` §Schema declaration and negotiation, preconditions 5–6).
///
/// `name` and `type_string` are NUL-terminated. `type_string` parses via
/// `DataTypeFactory::get`; both are cross-validated against the SQL-declared
/// schema at attach time before any block adoption.
struct alignas(8) SchemaEntry
{
    char name[SCHEMA_ENTRY_STR_MAX];
    char type_string[SCHEMA_ENTRY_STR_MAX];
};

static_assert(std::is_trivially_copyable_v<SchemaEntry>);
static_assert(std::is_standard_layout_v<SchemaEntry>);
static_assert(sizeof(SchemaEntry) == 128);
static_assert(alignof(SchemaEntry) == 8);

/// Per-block slot metadata — one of K, located at
/// `HandshakeRegion::slot_table_offset + i * HandshakeRegion::slot_table_stride`.
/// (`shm-block-stream.md` §Block framing, §Publication state machine,
/// §Memory ordering).
///
/// Concurrent-access fields are typed `std::atomic` and accessed under the
/// acquire/release ordering the wire mandates. Plain fields (slot_index,
/// row_count, per_column_descriptors_offset) are written by the producer
/// BEFORE the release-store to `state` that transitions the slot to
/// PUBLISHED; the consumer's acquire-load of `state == PUBLISHED` implies
/// acquire of every other field of the slot AND of the payload buffers the
/// descriptors point at.
///
/// Slot ownership for the read side is taken via atomic increment of
/// `retain_refcount` from 0 to 1, scoped by an RAII handle on the consumer.
/// Slot reuse on the producer side is gated on observing
/// `retain_refcount == 0` for the prior occupant
/// (`shm-block-stream.md` §Retain/release contract, §Region identity and
/// reuse rules, AC10).
///
/// NB: `std::atomic<T>` deletes copy/move and is therefore NOT
/// trivially-copyable; only `is_standard_layout_v` is asserted below. The
/// byte layout is still fully pinned by `sizeof` + the per-field offsets
/// implied by explicit padding members.
///
/// Sizing note: explicit fields below total 56 bytes; `alignas(64)` rounds
/// `sizeof(SlotEntry)` up to 64. Adding the new `transition_counter` field
/// therefore did NOT change `sizeof(SlotEntry)` — both the pre-T0.1 and
/// post-F2 layouts are 64 bytes. The pinned cache-line alignment is
/// preserved without bumping to 128.
struct alignas(64) SlotEntry
{
    std::atomic<uint32_t> state;             ///< `SlotState` numeric value.
    uint32_t slot_index;                     ///< Self-identity; must equal position
                                             ///  in slot table (precondition 9).
    std::atomic<uint64_t> transition_counter;///< Incremented (fetch_add(1, release))
                                             ///  by whoever drives a state transition:
                                             ///  producer on E→W and W→P; consumer on
                                             ///  P→E. The increment happens BEFORE the
                                             ///  corresponding state.store(). Consumer
                                             ///  acquire-loads validate monotonicity to
                                             ///  detect precondition-24 violations
                                             ///  (atomic single-variable consistency
                                             ///  guarantees monotonicity in time on a
                                             ///  conforming producer; a regression is
                                             ///  the determinable form of "transitions
                                             ///  out of order"). Doubles as a positive
                                             ///  progress signal for the I12 stall
                                             ///  timer when no slot has reached
                                             ///  PUBLISHED yet.
    std::atomic<uint64_t> sequence;          ///< Monotonically increasing block id
                                             ///  per slot (precondition 10).
    std::atomic<uint64_t> retain_refcount;   ///< Consumer inc on adopt, dec on drop;
                                             ///  on the LAST dec the consumer also
                                             ///  release-stores SlotState::EMPTY into
                                             ///  `state`, which is what the producer's
                                             ///  reuse wait actually polls.
    std::atomic<uint8_t> eos_marker;         ///< 1 if this block (after drain) is the
                                             ///  last; producer commits to no further
                                             ///  publications (`shm-block-stream.md`
                                             ///  §End-of-stream, precondition 23).
    uint8_t reserved_pad8[7];                ///< Padding to 8-byte alignment.
    uint64_t row_count;                      ///< Block row count; zero permitted
                                             ///  (`shm-block-stream.md` §Block
                                             ///  framing, precondition 11).
    uint64_t per_column_descriptors_offset;  ///< Byte offset (within data region)
                                             ///  of the `ColumnDescriptor[schema_count]`
                                             ///  array for this block (precondition 12).
};

static_assert(std::is_standard_layout_v<SlotEntry>);
static_assert(sizeof(SlotEntry) == 64);
static_assert(alignof(SlotEntry) == 64);
/// Field-offset pins (catch silent layout drift on layout-changing edits;
/// the ABI doc owns the wire-level guarantees, this file owns the C++ pin).
static_assert(offsetof(SlotEntry, state) == 0);
static_assert(offsetof(SlotEntry, slot_index) == 4);
static_assert(offsetof(SlotEntry, transition_counter) == 8);
static_assert(offsetof(SlotEntry, sequence) == 16);
static_assert(offsetof(SlotEntry, retain_refcount) == 24);
static_assert(offsetof(SlotEntry, eos_marker) == 32);
static_assert(offsetof(SlotEntry, row_count) == 40);
static_assert(offsetof(SlotEntry, per_column_descriptors_offset) == 48);

/// Top-of-SHM handshake region — exactly one per SHM object, located at
/// byte offset 0. (`shm-block-stream.md` §ABI version negotiation,
/// preconditions 1–7).
///
/// Producer writes every field of the handshake EXCEPT `magic` first, then
/// release-stores `magic = SHM_MAGIC`. Consumer's acquire-load of `magic` is
/// the synchronization point and is the FIRST read the consumer performs on
/// the SHM region. A read of any other field before observing the correct
/// magic is a programming error on the consumer side.
///
/// The readiness fd is NOT named here. It is conveyed out-of-band via a
/// Unix-domain socket whose path is derived from the SHM object name by
/// convention (`${tmpdir}/clickhouse_shm_<name>.sock`); the consumer
/// `connect()`s and receives the fd via `SCM_RIGHTS`
/// (`shm-block-stream.md` §Notification contract). Keeping the handshake
/// fixed-size and string-free simplifies validation.
///
/// NB: as with `SlotEntry`, the embedded `std::atomic<uint64_t> magic`
/// disables trivial copyability; only `is_standard_layout_v` is asserted.
struct alignas(64) HandshakeRegion
{
    std::atomic<uint64_t> magic;     ///< Release-stored last by producer; equals
                                     ///  SHM_MAGIC after handshake completes
                                     ///  (precondition 1).
    uint32_t abi_version;            ///< Must equal SHM_ABI_VERSION_1
                                     ///  (precondition 2).
    uint32_t ring_depth_k;           ///< In [1, IMPL_MAX_K] (precondition 3).
    uint32_t schema_count;           ///< In [1, IMPL_MAX_COLUMNS]; must equal
                                     ///  SQL-declared column count
                                     ///  (precondition 4).
    uint32_t reserved_pad32;         ///< Padding to 8-byte alignment; producer MBZ.
    uint64_t slot_table_offset;      ///< Byte offset of `SlotEntry[K]`
                                     ///  (precondition 7).
    uint64_t slot_table_stride;      ///< Bytes between consecutive `SlotEntry`s;
                                     ///  >= sizeof(SlotEntry).
    uint64_t data_region_offset;     ///< Byte offset of the data plane
                                     ///  (column buffers + descriptor arrays).
    uint64_t data_region_size;       ///< Data plane size in bytes.
    uint64_t schema_table_offset;    ///< Byte offset of `SchemaEntry[schema_count]`.
    uint64_t schema_table_size;      ///< `schema_count * sizeof(SchemaEntry)`.
    uint64_t reserved64[6];          ///< Pad to 128 bytes for future
                                     ///  same-version compat; producer MBZ.
};

static_assert(std::is_standard_layout_v<HandshakeRegion>);
static_assert(sizeof(HandshakeRegion) == 128);
static_assert(alignof(HandshakeRegion) == 64);

/// Build-time sanity: every region the consumer has to interpret has a
/// byte-pinned size and a documented alignment. Adding a new wire struct
/// here MUST come with size + alignment + layout asserts and a sibling
/// edit to the markdown doc.
static_assert(sizeof(SlotState) == 4);
static_assert(sizeof(WireColumnType) == 4);

}
