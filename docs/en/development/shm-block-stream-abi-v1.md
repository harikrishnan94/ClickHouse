---
description: 'On-wire ABI between an external SHM producer and the ClickHouse shm() table function, version 1.'
sidebar_label: 'SHM Block-Stream ABI v1'
sidebar_position: 200
slug: /development/shm-block-stream-abi-v1
title: 'SHM Block-Stream ABI (version 1)'
doc_type: 'reference'
---

# SHM Block-Stream ABI (version 1)

This document is the producer-facing reference for the wire ABI consumed by
the experimental `shm()` table function. It is the human-readable companion
to the C++ source-of-truth header
[`src/Storages/SharedMemorySource/Wire/Layout.h`](https://github.com/ClickHouse/ClickHouse/blob/master/src/Storages/SharedMemorySource/Wire/Layout.h).
Together they constitute the AC8 versioned ABI artifact named in the
`shm-block-stream` boundary spec; an external producer can be implemented
from these two artifacts alone (no consumer-code reading required).

The header pins concrete byte layouts; this document explains the semantic
obligations behind them. Where this document refers to a struct or constant
name (`HandshakeRegion`, `SlotState::PUBLISHED`, `SHM_MAGIC`, etc.) it means
the symbol of that name in `Layout.h`.

This is version `1` of the ABI. The `abi_version` field in the handshake is
`1`. Any change to the byte layouts, the magic, the enum numeric values, or
the per-section semantic obligations below is an explicit ABI version bump.

## SHM primitive

The producer creates a single POSIX shared-memory object via
`shm_open(name, O_RDWR | O_CREAT, ...)`, sizes it with `ftruncate`, and
`mmap`s it `MAP_SHARED | PROT_READ | PROT_WRITE`. The consumer attaches
the SAME mapping read-write via `shm_open(name, O_RDWR)` and
`mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)`.

The consumer requires write access to a strict subset of the SHM region -
the control-plane fields of the slot table that the retain / state-transition
protocol mutates from the consumer side:

- `SlotEntry::retain_refcount` - the consumer atomic-increments this
  field at adoption and atomic-decrements at handle release.
- `SlotEntry::state` - the consumer release-stores
  `PUBLISHED -> EMPTY` when the last retain on a published block is
  dropped (see Publication state machine).
- `SlotEntry::transition_counter` - the consumer increments this
  before the `PUBLISHED -> EMPTY` release-store (precondition 24).

The PAYLOAD data plane (every byte in `[data_region_offset,
data_region_offset + data_region_size)`, including the per-column
descriptor arrays and the column buffers) is logically read-only by the
consumer. The kernel cannot enforce a payload-vs-control-plane split
inside a single mapping, so the no-payload-write rule is enforced
ABOVE the syscall layer: ClickHouse's adopted `IColumn` /
`PaddedPODArray` adapters tag the borrowed buffers as
external-owner-only and route any mutation through copy-on-write
(`IColumn::mutate`) so the SHM bytes are never written from the
consumer's side. External producers can therefore continue to treat
the data plane as their own; the only fields a producer must tolerate
the consumer modifying are the three slot-table fields above, which
the producer itself already mutates via atomic operations under the
same release/acquire ordering rules.

The producer is expected to keep the mapping address-valid for the lifetime
of every outstanding consumer retain - no truncation, no unmap-and-replace
under live retain. Phase 1's trust model assumes a local, non-malicious,
conforming producer; the consumer does NOT rely on kernel-enforced sealing
in this version. A future ABI version bump may switch to a sealed
primitive that exposes the slot-table region as a separately-sealed,
writeable sub-mapping (so the data plane can be made `PROT_READ`-only on
the consumer side while preserving the retain protocol).

## ABI version negotiation

The first 128 bytes of the SHM object (byte offset 0) are the
`HandshakeRegion`. Producer write order:

1. Zero or otherwise initialize every field EXCEPT `magic`.
2. Write `abi_version = SHM_ABI_VERSION_1` (`= 1`), `ring_depth_k`,
   `schema_count`, the four region offsets/sizes, and the schema table.
3. Release-store `magic = SHM_MAGIC`. The wire-defined magic is the
   8-byte sequence `53 48 4D 5F 41 44 4F 50` (ASCII characters
   `S H M _ A D O P`, in that order) starting at byte offset 0 of
   the SHM region. On little-endian platforms (x86-64, ARM64; the
   only architectures supported in phase 1), this byte sequence is
   equivalent to the host-endian `uint64_t` literal
   `0x504F44415F4D4853` — note the byte reversal versus the
   ASCII-reading hex `0x53484D5F41444F50`, which is the BIG-endian
   interpretation. External producers SHOULD write the magic as the
   explicit 8-byte sequence above (e.g. via `memcpy` of a literal
   `const char magic_bytes[8] = {0x53, 0x48, 0x4D, 0x5F, 0x41, 0x44,
   0x4F, 0x50}` or its language equivalent), NOT as a host-endian
   `uint64_t`, so the on-wire bytes are unambiguous across host
   endianness. A big-endian port is an ABI version bump (see
   §Alignment, padding, byte order).

The consumer's FIRST read of the SHM region is an acquire-load of
`magic`. Observing `magic == SHM_MAGIC` makes every other handshake
field, the schema table, and the per-slot metadata table visible to the
consumer under acquire ordering.

Consumer rejection at attach time (each maps to a typed exception):

- `magic` not equal to `SHM_MAGIC` -> `handshake-invalid`
  (`SHM_HANDSHAKE_INVALID`).
- `abi_version` not in the supported set (phase 1: `{1}`) ->
  `handshake-invalid`.
- `ring_depth_k` zero or > `IMPL_MAX_K` (256) -> `handshake-invalid`.
- `slot_table_offset`, `data_region_offset`, `schema_table_offset` plus
  their declared sizes/strides do not fit in the SHM object, or any pair
  overlaps -> `handshake-invalid`.
- Readiness-fd locator (see Notification contract) unresolvable ->
  `attach-failed` (`SHM_ATTACH_FAILED`).

Layout of the handshake (see `HandshakeRegion` in `Layout.h`):

| Field | Type | Notes |
|---|---|---|
| `magic` | `std::atomic<uint64_t>` | Release-stored last; `SHM_MAGIC`. |
| `abi_version` | `uint32_t` | `= 1`. |
| `ring_depth_k` | `uint32_t` | In `[1, IMPL_MAX_K]`. |
| `schema_count` | `uint32_t` | In `[1, IMPL_MAX_COLUMNS]`. |
| `reserved_pad32` | `uint32_t` | Producer must zero. |
| `slot_table_offset` | `uint64_t` | Byte offset of `SlotEntry[K]`. |
| `slot_table_stride` | `uint64_t` | Bytes between successive slots; `>= sizeof(SlotEntry)`. |
| `data_region_offset` | `uint64_t` | Byte offset of the data plane. |
| `data_region_size` | `uint64_t` | Data plane size in bytes. |
| `schema_table_offset` | `uint64_t` | Byte offset of `SchemaEntry[schema_count]`. |
| `schema_table_size` | `uint64_t` | `schema_count * sizeof(SchemaEntry)`. |
| `reserved64[6]` | `uint64_t x 6` | Producer must zero. |

Total size: 128 bytes, 64-byte aligned.

## Schema declaration and negotiation

The schema lives in two places:

- SQL-declared: the second argument to the `shm()` table function, parsed by
  ClickHouse at query parse/resolve time. Types outside the phase-1
  supported set `{UInt64, String}` are rejected with `SHM_SCHEMA_MISMATCH`
  before any attach is attempted.
- Producer-declared: an array of `SchemaEntry[schema_count]` at byte offset
  `schema_table_offset`. Each entry is 128 bytes:

  | Field | Size | Notes |
  |---|---|---|
  | `name` | 64 B, NUL-terminated | Column name. |
  | `type_string` | 64 B, NUL-terminated | e.g. `"UInt64"`, `"String"`. |

Cross-validation at attach time, after handshake validation, before any
block adoption:

- Producer `schema_count` must equal SQL-declared column count -> otherwise
  `SHM_SCHEMA_MISMATCH`.
- For each position `i`, `name[i]` must equal the SQL-declared column name
  at position `i`.
- For each position `i`, `type_string[i]` must parse via
  `DataTypeFactory::get` and must equal the SQL-declared type at position
  `i`; both must be in `{UInt64, String}`.

A mismatch in any of count, name, type, or order surfaces
`SHM_SCHEMA_MISMATCH` before any block is read.

## Per-type buffer layout

Producer payload buffers live inside the data region
(`[data_region_offset, data_region_offset + data_region_size)`). Per-column
descriptors point at offsets within this region. The consumer validates
every descriptor before reading any payload byte; an invalid descriptor
surfaces `SHM_BUFFER_LAYOUT_INVALID`.

The descriptor struct is `ColumnDescriptor` (56 bytes, 8-aligned). The
`type` tag selects the field interpretation.

### `ColumnVector<UInt64>`

Single value buffer.

| Field | Meaning |
|---|---|
| `type` | `WireColumnType::UInt64` (`= 1`). |
| `value_offset` | Byte offset of the value buffer within the data region. Must be 8-byte aligned (precondition 13). |
| `value_count` | Element count. Must equal `SlotEntry::row_count` (preconditions 14, 26). |
| `value_padding` | Trailing safe-read padding in bytes; must be `>= PADDING_FOR_SIMD` (`= 64`) (precondition 15). |
| `offsets_offset`, `offsets_count`, `offsets_padding` | Unused; producer sets to 0. |

`value_offset + value_count * 8 + value_padding` must not exceed
`data_region_size` (precondition 14).

### `ColumnString`

Two buffers: `chars` (concatenated UTF-8 / arbitrary bytes; no separator)
and `offsets` (one `uint64_t` per row, monotonically non-decreasing).

| Field | Meaning |
|---|---|
| `type` | `WireColumnType::String` (`= 2`). |
| `value_offset` | `chars` byte offset; UInt8 alignment is trivial (precondition 16). |
| `value_count` | `chars` byte size (precondition 18). |
| `value_padding` | `chars` trailing safe-read padding; `>= PADDING_FOR_SIMD` (precondition 20). |
| `offsets_offset` | `offsets` byte offset; must be 8-byte aligned (precondition 17). |
| `offsets_count` | Element count of `offsets`; must equal `SlotEntry::row_count` (preconditions 19, 26). |
| `offsets_padding` | `offsets` trailing safe-read padding; `>= PADDING_FOR_SIMD` (precondition 20). |

`value_offset + value_count + value_padding` and
`offsets_offset + offsets_count * 8 + offsets_padding` must each be within
`data_region_size`.

Content-level rules (validated lazily, before any consumer-side read that
would otherwise observe a violation):

- Each `offsets[i]` is non-decreasing relative to `offsets[i-1]` (i.e.
  monotonic; precondition 21).
- `offsets[row_count - 1] == value_count` (the terminal offset equals the
  `chars` byte size; precondition 22).

Violations surface as `SHM_BUFFER_LAYOUT_INVALID` before unsafe reads.

#### `offsets[-1]` zero sentinel

ClickHouse's `ColumnString::offsetAt(0)` is implemented as `offsets[-1]` - a
one-element back-step into the offsets buffer's `pad_left` region. For an
owned `PaddedPODArray<UInt64>` the `pad_left` slot is zero-initialised by the
allocator, so the read is well-defined and returns 0. For an adopted
offsets buffer that lives in producer SHM, the producer MUST supply the
equivalent invariant explicitly:

- The eight bytes immediately preceding `offsets_offset` (at byte offset
  `offsets_offset - 8` within the data region) MUST be readable AND hold the
  value 0 (little-endian `uint64_t` zero).
- Consequently `offsets_offset >= 8`. Placing the offsets buffer flush with
  the start of the data region (`offsets_offset == 0`) is REJECTED.

This is part of ClickHouse's column-storage contract, not visible in the
descriptor fields above; it mirrors `PaddedPODArray<UInt64>::pad_left`. The
consumer enforces it at descriptor-validation time: a missing or non-zero
sentinel surfaces `SHM_BUFFER_LAYOUT_INVALID` before any row is read.

Typical producers satisfy this by sizing the offsets allocation to
`8 + row_count * 8 + offsets_padding`, zero-filling the first 8 bytes, and
setting `offsets_offset` to the byte AFTER the leading zero slot.

### Alignment, padding, byte order

- All multi-byte integer fields on the wire are little-endian. Phase 1 is
  Linux-only, x86-64 or aarch64; both are little-endian. A big-endian port
  is an ABI version bump.
- All alignments named above are minimums; producers MAY over-align.
- Safe-read padding bytes need not be zeroed but must be readable
  (no page fault). Producers typically achieve this by sizing the data
  region's allocations to include the padding.

## Block framing

Each block published by the producer occupies one slot in the slot table
(see Publication state machine). The slot holds the block's framing
metadata. The block's payload (per-column descriptor array + per-column
buffers) lives in the data region.

`SlotEntry` (64 bytes, 64-aligned) carries:

| Field | Meaning |
|---|---|
| `state` | `std::atomic<uint32_t>` holding `SlotState` (`EMPTY` / `WRITING` / `PUBLISHED`). |
| `slot_index` | Self-id; must equal slot's position in the slot table (precondition 9). |
| `sequence` | Monotonically increasing block id per slot; strictly greater than the previous adopted block's sequence for the same slot (precondition 10). |
| `retain_refcount` | `std::atomic<uint64_t>`; consumer inc on adoption, dec on drop. Producer waits for `0` before slot reuse. |
| `eos_marker` | `std::atomic<uint8_t>`: `1` if this is the last block (precondition 23). |
| `transition_counter` | `std::atomic<uint64_t>` incremented by the party driving each state transition (producer: `EMPTY -> WRITING` and `WRITING -> PUBLISHED`; consumer: `PUBLISHED -> EMPTY`). Monotonically non-decreasing per slot; consumers may sample deltas to detect producer-side progress for stall-timer purposes (precondition 24). See [Transition counter (precondition 24)](#transition-counter-precondition-24). |
| `row_count` | Number of rows; zero permitted; must be `<= IMPL_MAX_ROWS_PER_BLOCK` (precondition 11). |
| `per_column_descriptors_offset` | Byte offset (in data region) of `ColumnDescriptor[schema_count]` for this block (precondition 12). |

The per-column descriptor array is contiguous, exactly `schema_count`
entries long. Each descriptor's `value_count` (UInt64 columns) or
`offsets_count` (String columns) must equal `row_count` (precondition 26).

### End-of-stream

The producer signals end-of-stream by publishing a block (possibly with
`row_count = 0`) whose `eos_marker = 1`. After this transition the
producer commits to no further publications and is free to exit once all
retains on previously published blocks have been released. Producer
detachment BEFORE end-of-stream, while any retain refcount is non-zero,
surfaces `SHM_PRODUCER_DEATH_BEFORE_EOS` on the consumer side.

The consumer recognises end-of-stream by acquire-loading `state ==
PUBLISHED` and `eos_marker == 1` on the same slot. Because `eos_marker`
lives in the same slot as `state` they are in the same memory-ordering
domain.

## Size and row-count limit communication

Phase-1 implementation limits, enforced by the consumer:

- `ring_depth_k` must be `<= IMPL_MAX_K` (256). The consumer rejects
  larger values at handshake validation.
- `row_count` must be `<= IMPL_MAX_ROWS_PER_BLOCK` (`1 << 20 = 1,048,576`).
  Larger values surface `SHM_BLOCK_FRAMING_INVALID`.
- `schema_count` must be `<= IMPL_MAX_COLUMNS` (64).
- There is no aggregate per-block byte-size limit beyond what the data
  region itself can hold; descriptor checks ensure all declared offsets +
  sizes + padding stay within `data_region_size`.

These limits are part of the v1 ABI; raising them is an ABI version bump.

## Backpressure and ring-full contract

When all `K` slots are in `PUBLISHED` state with non-zero
`retain_refcount`, the producer cannot reuse a slot. The producer's
options (chosen on the producer side; not pinned by this ABI) are:

- Block the publish call until at least one slot's `retain_refcount`
  returns to zero, then transition that slot `PUBLISHED -> EMPTY ->
  WRITING -> PUBLISHED`.
- Refuse the publish and surface a producer-side error (out of band).

The consumer side is identical under either policy: it drains and
releases at its own pace. The consumer never blocks the producer
intentionally; backpressure is the natural consequence of the retain
protocol.

## Publication state machine

Slot state transitions (driven by the producer, except for the implicit
retain protocol on the consumer side):

```text
        producer pick slot          producer release-store
EMPTY ---------------------->  WRITING ---------------------->  PUBLISHED
   ^                                                                  |
   |           producer reuse                  consumer drain          |
   `------------ (retain_refcount == 0) <---- (retain_refcount == 0) <-`
```

- `EMPTY` (`0`) - slot is producer-owned and idle; the consumer must not
  read any field except `state` from this slot.
- `WRITING` (`1`) - producer is filling the slot's descriptor and payload
  buffers; consumer must not read.
- `PUBLISHED` (`2`) - slot is consumer-readable. Producer must NOT modify
  `slot_index`, `sequence`, `eos_marker`, `row_count`, or
  `per_column_descriptors_offset` while in this state; only the consumer
  modifies `retain_refcount`.

Slot reuse: once consumer-side `retain_refcount` returns to zero, the
producer may transition the slot back to `EMPTY` and re-enter `WRITING`.
The next published block in that slot must carry a `sequence` strictly
greater than the prior block's; the consumer enforces this per-slot
(precondition 10).

Any observed `state` value outside `{0, 1, 2}` is
`SHM_BLOCK_FRAMING_INVALID`.

State transitions must follow this order (no skipping); skips are
`SHM_BLOCK_FRAMING_INVALID`.

### Transition counter (precondition 24)

Each `SlotEntry` carries a `transition_counter` field that the party
driving each state transition increments before completing the
release-store that advertises the new state. Specifically:

- Producer: increment before the release-store that transitions
  `EMPTY -> WRITING`, and increment again before the release-store
  that transitions `WRITING -> PUBLISHED`.
- Consumer: increment before the release-store that transitions
  `PUBLISHED -> EMPTY` (i.e. when the last retain on a published
  block is dropped and the slot is handed back to the producer).

The counter is monotonically non-decreasing within a slot for the
lifetime of the SHM object. It is independent of `sequence`:
`sequence` advances only on producer publications, whereas
`transition_counter` advances on every state transition including
consumer-side drains.

Stall-timer use: the consumer's stall-timer machinery samples
`transition_counter` across all slots between wake-ups. If the
configured `shm_source_stall_timeout_ms` elapses with NO observed
delta across any slot AND at least one consumer-side wait is
outstanding, the consumer surfaces `SHM_PRODUCER_STALL` per
[`shm-pollable-source-spec.md` I12](./shm-pollable-source-spec.md#invariants).
Because consumer drains also bump the counter, a healthy
consumer-only drain (rare under steady-state load) does NOT mask a
stalled producer: the stall timer fires only when neither side has
made progress.

The counter is a `uint64_t` and rolls over modulo `2^64`. At the
implementation upper bound of roughly one transition per nanosecond,
the rollover horizon is on the order of centuries — well beyond any
realistic SHM-object lifetime — so wrap-around handling is out of
scope for phase 1. A future ABI version that needs longer-running
streams would either widen the field or accept wrap-around in the
delta-comparison logic; either change is an ABI version bump.

## Memory ordering

Publication uses release/acquire ordering, with the slot's `state` field as
the publication point.

Producer side, per published block:

1. Write all payload bytes (column buffers) for the block.
2. Write the `ColumnDescriptor[schema_count]` array for the block.
3. Write the slot's plain fields: `slot_index`, `sequence`, `row_count`,
   `per_column_descriptors_offset`, `eos_marker`.
4. Release-store `state = PUBLISHED`.

Consumer side, per drained block:

1. Acquire-load `state`. If `== PUBLISHED`, every preceding write is
   visible.
2. Read the slot's plain fields and the descriptor array.
3. Read payload bytes through the adopted column.
4. Acquire-load `sequence`/`eos_marker` if they need to be sampled
   independently of `state` (e.g. on a per-slot identity check).
5. Atomic-increment `retain_refcount` from 0 to 1 with acquire/release as
   needed for the read side's RAII handle; atomic-decrement to 0 at
   handle release.

The readiness-fd notification (next section) is ordered AFTER the
release-store of `state`. The consumer's executor wake on the fd implies
that at least one slot's `state` is visible as `PUBLISHED` somewhere in
the ring; the consumer must verify, because spurious wakes are admissible.

### Wire fields with atomic access (and how to portably access them)

Several fields in this document and in
[`Layout.h`](https://github.com/ClickHouse/ClickHouse/blob/master/src/Storages/SharedMemorySource/Wire/Layout.h)
participate in the publication protocol and MUST be accessed with
acquire/release semantics on BOTH sides
(`HandshakeRegion::magic`; `SlotEntry::state`, `sequence`,
`retain_refcount`, `eos_marker`, `transition_counter`).

#### The wire is defined in BYTES, not in C++ types

The wire contract is defined ENTIRELY in terms of **byte layout**: each
field is a `uint{8,32,64}_t` of explicit size and alignment, as listed in
the handshake table and the slot-entry table above. The wire spec does
NOT mention any C++ type, any C `_Atomic` qualifier, any Rust
`AtomicU*`, or any kernel `READ_ONCE`/`WRITE_ONCE`. Those are
language-specific tools for emitting the right machine instructions on
each side; they are NOT part of the on-wire representation.

In particular: the consumer's `Layout.h` happens to declare these
fields with C++ `std::atomic<uint64_t>` / `std::atomic<uint32_t>` /
`std::atomic<uint8_t>` wrappers. That is a CONSUMER-SIDE
IMPLEMENTATION DETAIL chosen because it gives the consumer
`memory_order_acquire` / `memory_order_release` operations directly
from the C++ language. It does NOT define the wire. The C++ standard
does NOT guarantee that `std::atomic<T>` is layout-compatible with `T`
for arbitrary `T` (the standard requires `T` to be trivially copyable
in `[atomics.types.generic]`, but says nothing about the
size/alignment/representation of the `std::atomic` wrapper itself; a
conforming implementation could in principle add a lock byte, padding,
or different alignment for an arbitrary `T`).

What we DO rely on, and what every supported ClickHouse platform
delivers as a matter of **ABI convention** (not standard guarantee):

- On x86-64 and aarch64 with libstdc++ or libc++,
  `sizeof(std::atomic<uintN_t>) == sizeof(uintN_t)` and
  `alignof(std::atomic<uintN_t>) == alignof(uintN_t)` for `N` in `{8,
  32, 64}`, with the object representation of the atomic equal to the
  object representation of the underlying integer. These are the only
  platforms phase 1 supports (see §Alignment, padding, byte order); the
  consumer-side `std::atomic<T>` wire-field declaration depends on this
  ABI fact and would break on any platform where it does not hold.
  A future ABI v2 will remove this implementation-defined dependency
  by switching the consumer-side header to plain `uintN_t` fields and
  using `std::atomic_ref<uintN_t>` (C++20) for atomic access (see
  below).

External producers MUST NOT rely on the consumer's choice of C++
`std::atomic<T>` wrapper. They MUST treat each wire field as a plain
`uintN_t` of the declared size and alignment, and use an
implementation-appropriate atomic-acquire / atomic-release / fence
primitive for every read or write that participates in the publication
protocol (the `magic` release-store at handshake-publish time; the
`state` release-store at block-publish time; the `retain_refcount`
increment/decrement at adopt/release time; the `transition_counter`
increment at every state transition). A producer that uses plain
non-atomic reads/writes against these fields is non-conforming: the
consumer's acquire-load semantics only deliver the documented
ordering guarantees when paired with a producer-side release-store
(or equivalent), and without that pairing the consumer may observe
torn or stale handshake/slot fields and surface
`SHM_HANDSHAKE_INVALID` or `SHM_BLOCK_FRAMING_INVALID`
non-deterministically.

#### Per-language portable atomic-access recipes

- **C11 `<stdatomic.h>`:** declare the wire fields as `_Atomic
  uint64_t` (etc.) in the producer-side struct definition; use
  `atomic_store_explicit(&hs->magic, value, memory_order_release)` /
  `atomic_load_explicit(&slot->state, memory_order_acquire)`. For
  the scalar integer types used here, `_Atomic uintN_t` has the same
  size and alignment as the underlying `uintN_t` on every C11
  implementation phase 1 targets, but this too is an ABI fact about
  the platforms (gcc/clang on x86-64 / aarch64 / Linux), not a C
  standard guarantee — a strictly portable producer that wants to
  share its struct definition with non-atomic code should prefer the
  next bullet (`__atomic_*` builtins on plain `uintN_t`) or the
  `_Atomic` cast pattern shown in the example below.
- **GCC / Clang `__atomic` builtins** (for producers whose struct
  definitions must remain plain `uintN_t`): use
  `__atomic_load_n(p, __ATOMIC_ACQUIRE)`,
  `__atomic_store_n(p, v, __ATOMIC_RELEASE)`, and
  `__atomic_fetch_add(p, 1, __ATOMIC_ACQ_REL)`. These operate on the
  underlying scalar without requiring the field to be declared
  `_Atomic`, and are the most portable choice for a producer that
  wants to share its struct layout with non-atomic code.
- **C++20 `std::atomic_ref<T>`:** declare the wire fields as plain
  `uint64_t` / `uint32_t` / `uint8_t` in the producer-side struct, and
  obtain atomic access on demand:
  `uint64_t v = std::atomic_ref<uint64_t>(handshake->magic).load(std::memory_order_acquire);`
  / `std::atomic_ref<uint64_t>(handshake->magic).store(value,
  std::memory_order_release);`. `std::atomic_ref<T>` is guaranteed by
  the C++20 standard to provide atomic operations on a plain `T`
  lvalue, with no requirement that the underlying `T` be wrapped or
  re-typed — this is the standard-blessed primitive an external C++
  producer should reach for, and is the migration target for the
  ClickHouse consumer-side header in ABI v2.
- **Linux kernel idiom** (for producers that reuse kernel-style
  code): `smp_load_acquire(p)` / `smp_store_release(p, v)`, or the
  `READ_ONCE` / `WRITE_ONCE` plus `smp_mb__before_atomic` /
  `smp_mb__after_atomic` fence pair.
- **Rust:** declare the wire fields as `AtomicU64` / `AtomicU32` /
  `AtomicU8`. Unlike C++, Rust DOES guarantee these are
  layout-compatible with `u64` / `u32` / `u8`: `AtomicUN` is
  `#[repr(C, align(N))]` (effectively transparent over `uN`) per the
  Rust reference, so casting a `&u64` in shared memory to
  `&AtomicU64` and back is a defined operation. Use
  `.store(v, Ordering::Release)` / `.load(Ordering::Acquire)`.

#### Minimum worked C11 example

A producer or consumer written in C11 acquires the handshake magic
without redeclaring the struct fields as `_Atomic` by casting the
plain `uint64_t *` to `_Atomic uint64_t *` at the point of use. This
is well-defined for the scalar integer types used here on every C11
implementation phase 1 supports:

```c
#include <stdint.h>
#include <stdatomic.h>

struct HandshakeRegion {
    uint64_t magic;          /* 8 bytes at offset 0 */
    uint32_t abi_version;
    /* ... remaining fields per the handshake layout table ... */
};

uint64_t v = atomic_load_explicit(
    (_Atomic uint64_t *)&hs->magic, memory_order_acquire);
if (v != 0x504F44415F4D4853ULL) {
    /* bytes != 'SHM_ADOP' on LE; magic not yet written or wrong
     * ABI: do not read any other handshake field. */
}
```

Equivalent C++20 form via `std::atomic_ref<T>` (this is what an
external C++ producer should use, and is also the migration target
for the ClickHouse consumer-side header in ABI v2 — at which point
`Layout.h` switches from `std::atomic<uintN_t>` wire fields to plain
`uintN_t` fields plus `std::atomic_ref<uintN_t>` at every access
site, removing the implementation-defined `std::atomic<T>` layout
dependency described above):

```cpp
#include <atomic>
#include <cstdint>

uint64_t v = std::atomic_ref<uint64_t>(hs->magic)
    .load(std::memory_order_acquire);
```

Equivalent GCC/Clang form without `<stdatomic.h>` or `<atomic>`:

```c
uint64_t v = __atomic_load_n(&hs->magic, __ATOMIC_ACQUIRE);
```

## Region identity and reuse rules

The pair `(slot_index, sequence)` uniquely identifies a block across the
stream lifetime. The consumer keeps a per-slot "last adopted sequence"
counter; any newly observed `sequence` for a slot must be strictly greater
than the prior. A non-strictly-greater value is
`SHM_BLOCK_FRAMING_INVALID`.

Region reuse is gated on the consumer's retain protocol: the producer must
NOT overwrite a slot's payload region while any `retain_refcount` on that
slot is non-zero. As long as the producer respects this contract, the
bytes the consumer reads through an adopted column remain bit-identical
for the column's lifetime; AC10 (in the boundary spec) covers this
property.

If a producer violates retain-respect (truncates, unmaps, or republishes
under live retain), behaviour is undefined per the phase-1 trust model.

## Notification contract

The readiness mechanism is an `eventfd` created by the producer
(`eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK | EFD_SEMAPHORE)` or similar).

The fd is conveyed out-of-band via a Unix-domain socket whose path is
derived from the SHM object name by convention:

```text
${TMPDIR:-/tmp}/clickhouse_shm_<name>.sock
```

(where `<name>` is the SHM object name with any leading `/` stripped; the
exact normalization is part of v1 and any change requires an ABI version
bump).

Producer responsibilities:

- Create the socket (`AF_UNIX, SOCK_STREAM`), bind, listen.
- On each consumer `connect()`, send the eventfd via `sendmsg` with
  `SCM_RIGHTS`.
- For each publication, post one count to the eventfd (level-triggered;
  the consumer drains).
- On clean exit (after end-of-stream is signalled and all retains are
  released), close the eventfd and unlink the socket.

Consumer responsibilities:

- `connect()` to the conventional socket path during attach; on failure
  surface `SHM_ATTACH_FAILED`.
- Receive the eventfd via `recvmsg` with `SCM_RIGHTS`. The consumer takes
  ownership of the fd and closes it exactly once at source destruction.
- On wake (eventfd read returns), the consumer drains ALL currently
  `PUBLISHED` slots before re-arming readiness. Spurious wakes are
  admissible: the consumer must verify against the publication state
  machine.
- `POLLHUP` on the control socket indicates the producer has detached.
  If observed before end-of-stream and any retain is still live, the
  consumer surfaces `SHM_PRODUCER_DEATH_BEFORE_EOS`.

## Retain/release contract

`SlotEntry::retain_refcount` is the consumer-driven reference count on the
slot's payload region. Semantics:

- The consumer atomically increments `retain_refcount` (0 -> 1) at
  adoption time, scoped by an RAII handle.
- Every adopted column produced from the block carries a shared reference
  to the same handle; the count returns to 0 when the last reference is
  dropped.
- The producer is contracted not to reuse, truncate, or republish a slot
  while its `retain_refcount` is non-zero.
- The consumer guarantees that the count returns to 0 in bounded time
  after the corresponding chunk's lifetime ends (including the
  cancellation path; the consumer's I9 cancellation bound is independent
  of producer state).

This is a protocol-level obligation, not a kernel-enforced one. Phase
1's SHM primitive does not seal against truncation/unmap; the consumer
trusts a conforming producer. If you need kernel-enforced sealing,
the wire would need to switch primitives in a new ABI version.

## Covered `IColumn` method surface

The consumer-side guarantees the producer relies on. Each method on an
adopted column is observationally indistinguishable from the same method
on a copy-owned column of the same bytes; the canonical list is owned by
the consumer-side adoption-layer spec (`shm-adoption-layer-spec.md` Covered
IColumn method surface). Reproduced here for producer reference:

For `ColumnVector<UInt64>`:

- `size_t size() const` - row count; must agree with `SlotEntry::row_count`.
- `const PaddedPODArray<UInt64> & getData() const` - value-buffer reader.

For `ColumnString`:

- `size_t size() const` - row count.
- `const PaddedPODArray<UInt8> & getChars() const` - byte-data reader.
- `const PaddedPODArray<UInt64> & getOffsets() const` - offset reader.

These are read-only paths. Mutation paths on an adopted column either
materialize a ClickHouse-owned copy (via the standard COW
`IColumn::mutate` entry) or raise a typed exception; in neither case is
producer memory written to.

A producer that wants to know whether a query exercises an unsupported
method must consult the ClickHouse-side adoption-layer spec. Phase 1
queries that stay within `count()`, `sum()`, `cityHash64()`, and
`length()` over the AC2 type set are guaranteed to use only the methods
above.

## Failure-class summary

Every consumer-side detection point surfaces exactly one typed exception
class, defined in `src/Common/ErrorCodes.cpp`:

| Class | ErrorCode | Trigger source |
|---|---|---|
| `attach-failed` | `SHM_ATTACH_FAILED` | `shm_open` / `mmap` / socket-connect failure; readiness-fd locator unresolvable. |
| `handshake-invalid` | `SHM_HANDSHAKE_INVALID` | Preconditions 1-3, 7. |
| `schema-mismatch` | `SHM_SCHEMA_MISMATCH` | Preconditions 4-6 (membership at SQL parse; equality at handshake cross-validation). |
| `block-framing-invalid` | `SHM_BLOCK_FRAMING_INVALID` | Preconditions 8-12, 23, 24, 26. |
| `buffer-layout-invalid` | `SHM_BUFFER_LAYOUT_INVALID` | Preconditions 13-22. |
| `producer-stall` | `SHM_PRODUCER_STALL` | `shm_source_stall_timeout_ms` elapsed with no publication progress. |
| `producer-death-before-eos` | `SHM_PRODUCER_DEATH_BEFORE_EOS` | Producer detach observed (POLLHUP on control socket) before end-of-stream with live retains. |

The `feature-gate-disabled` and `memory-limit-exceeded` classes reuse
existing ClickHouse error codes (`SUPPORT_IS_DISABLED`,
`MEMORY_LIMIT_EXCEEDED`).

Precondition numbers refer to the enumerated list in
`shm-pollable-source-spec.md` Producer-side preconditions
enumerated.
