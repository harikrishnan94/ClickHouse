---
description: 'Spec for the adoption layer component: zero-copy column construction from producer-published SHM buffers with retain semantics.'
sidebar_label: 'SHM Adoption Layer Spec'
sidebar_position: 203
slug: /streamed_table/specs/adoption-layer-spec
title: 'Adoption Layer — Zero-Copy Column Construction'
doc_type: 'reference'
---

# Adoption Layer — Zero-Copy Column Construction

This spec defines component (a) — sub-deliverable A — of the zero-copy SHM source feature: a column-construction surface that wraps externally-owned, pre-padded byte buffers as `IColumn`s with retain semantics, with no `memcpy` of column payload.

The system mission, glossary, non-goals, cross-component invariants (I5, I10), and end-to-end ACs (AC1, AC7) are owned by [system spec](./system-spec.md). The producer-facing wire that supplies the byte buffers is owned by [shm-block-stream spec](./shm-block-stream-spec.md). The executor-side source that uses this surface to assemble `Chunk`s is owned by [pollable-shm-source spec](./pollable-source-spec.md). The memory-accounting counterpart is owned by [memory-tracker-integration spec](./memory-tracker-spec.md).

## Mission

Expose a localized seam in ClickHouse's column-construction surface such that:

- a producer-published byte buffer that already satisfies ClickHouse's column-storage contract — padding, alignment, sentinel bytes — can be wrapped as an `IColumn` whose primary byte buffer points at producer memory, with no `memcpy` of column payload at construction or on the read paths exercised by [system spec — AC1](./system-spec.md#end-to-end-acceptance-criteria); mutation paths materialize per [I3](#invariants);
- the resulting adopted column carries a retain token whose lifetime keeps the producer-side region pinned for as long as ClickHouse references the column or any handle derived from it;
- on the read paths exercised by [system spec — AC1 Functional correctness](./system-spec.md#end-to-end-acceptance-criteria), the adopted column behaves indistinguishably from a copy-owned column built from the same bytes — including in downstream operators that consume it.

The adoption seam is the only consumer-side party that turns data-plane bytes into `IColumn`s. Per the system component map, the pollable source routes per-block byte ranges to this layer; this layer returns adopted columns plus the retain token; this layer carries the source-supplied charge handle through every adopted handle on the same RAII boundary as the retain — it does not itself enter the MemoryTracker.

## Non-goals

- The adoption layer does not own the on-wire byte layout, padding rules, sentinel-byte definitions, framing, or version negotiation. Those live in [shm-block-stream spec — Wire interfaces](./shm-block-stream-spec.md#wire-interfaces). This layer consumes a layout declared by the wire; it does not define one.
- The adoption layer does not own fd lifetime, executor integration, or the `streamed_table()` table-function call site (legacy alias `shm()`). Those live in [pollable-shm-source spec — Interfaces & contracts](./pollable-source-spec.md#interfaces--contracts).
- The adoption layer does not own memory-accounting policy, the limit-enforcement decision, or the slack model. It carries the charge handle supplied by the pollable source; accounting calls are issued by the source before adopt() is called, per [memory-tracker-integration spec — Interfaces & contracts](./memory-tracker-spec.md#interfaces--contracts).
- The adoption layer does not extend type coverage in phase 1; the supported set is closed at `{UInt64, String}` per [AC2 Type coverage](#acceptance-criteria) and is not grown in phase 1.

System-level non-goals — full text in [system spec — Non-goals](./system-spec.md#non-goals). N7 (no coverage of every `IColumn` kind in phase 1) and N8 (no operator changes outside the new source) are the system-level constraints that most directly bind this layer.

## Constraints

- The adoption seam must be a localized addition to ClickHouse's column-construction surface. If achieving adoption requires invasive changes that ripple into the broader column or executor surface, the contract halts — see [S1](#stop-conditions).
- Every adopted `IColumn` must satisfy ClickHouse's column-storage contract on the read paths exercised by [system spec — AC1](./system-spec.md#end-to-end-acceptance-criteria). The producer-supplied buffer is already pre-padded; the layer's job is to assert conformance, not to rewrite the buffer.
- The retain token threads through every derived handle of an adopted column. There is no observable path that releases producer memory while a live reference exists. Surface-area authority for this property is [system spec — I5 Retain correctness](./system-spec.md#cross-component-invariants); the local consequence is [I3](#invariants).
- No code path reachable from an adopted column writes to producer memory. Any path that needs mutable storage must materialize a ClickHouse-owned copy before mutating. Authority: [I3](#invariants).
- Validation of producer-declared layouts is two-tier. Descriptor-level validation (alignment, declared sizes fit the region, declared padding meets the safe-read contract — preconditions 13–20 in [pollable-shm-source spec — Producer-side preconditions enumerated](./pollable-source-spec.md#producer-side-preconditions-enumerated)) must complete before any read of the data-plane payload; failures surface as typed exceptions from adopt() before any unsafe read. Content-level validation that requires reading a declared buffer (today: `ColumnString` offsets monotonicity and terminal-offset value — preconditions 21–22) fires post-adoption, on or before the consumer-side read path that would otherwise observe the violation; failures surface as typed exceptions before that unsafe read. Buffers that violate either tier fail the query with a typed exception. Authority: [I4](#invariants).
- The set of `IColumn` methods on which an adopted column matches a copy-owned column is documented as part of [shm-block-stream spec — AC8 Producer ABI documented in-tree](./shm-block-stream-spec.md#acceptance-criteria). The list is constrained to be sufficient for [system spec — AC1](./system-spec.md#end-to-end-acceptance-criteria) but otherwise minimal.
- ABI conformance is per-version, with version negotiated at attach time via [shm-block-stream spec — Wire interfaces](./shm-block-stream-spec.md#wire-interfaces). The adoption layer is entitled to reject buffers tagged with an unsupported version through the same typed-exception path as malformed buffers.

## Interfaces & contracts

The adoption layer's observable surface is the adopt seam itself plus the retain semantics it carries. Each observable boundary below is owed *to* the named caller or owed *by* the named callee.

**Adopt entry point — owed *to* [pollable-shm-source spec — Interfaces & contracts](./pollable-source-spec.md#interfaces--contracts).** For each producer-published block, the source hands the adoption layer the per-column layout descriptor (buffer pointers, sizes, padding indicator, sentinel-byte indicator, offset-encoding tag) drawn from [shm-block-stream spec — Wire interfaces](./shm-block-stream-spec.md#wire-interfaces), a retain token acquired against that block, and a charge handle acquired from the memory-tracker-integration charge entry for that block's adopted byte count. The adopt entry returns either:

- one `IColumn` per declared column whose primary byte buffer points into producer memory, with the retain token and charge handle jointly carried as the adopted state's lifetime handles; the adopted state has passed descriptor-level validation but is not yet subject to content-level validation of any declared buffer; or
- a typed exception, raised before any read of the data-plane payload, if descriptor-level validation rejects the declared layout — see [I4](#invariants).

Content-level validation of declared buffers (today: `ColumnString` offsets monotonicity and terminal-offset value) fires post-adoption on or before the consumer-side read path that would observe a violation, and surfaces a typed exception before that unsafe read. Authority: [I4](#invariants).

The adopt entry does not modify its arguments. It does not retain anything other than the producer-supplied region's reference count; it does not require ownership transfer of fds or non-region state.

**Retain and charge handle semantics — owed *by* the adoption layer to the producer side and to [memory-tracker-integration spec — Interfaces & contracts](./memory-tracker-spec.md#interfaces--contracts).** The retain token and charge handle are both acquired by the source before the adopt() call (per [pollable-shm-source spec — Per-block adoption call](./pollable-source-spec.md#interfaces--contracts)) and handed in as adopt() arguments. Ownership of both transfers to adopted state on successful adoption: the adopted columns jointly carry them, and each is released exactly once at adopted-state final drop. "Final drop" means destruction of the last `IColumn` (or derived handle) that references the underlying region. On any failure path of adopt() — descriptor-level validation failure or any other exception before successful return — the adoption layer releases both the handed-in retain and the handed-in charge handle before propagating the exception. No path leaves either dangling: either ownership transfers to adopted state on success, or the adoption layer releases both on failure. Authority for the system-wide retain property: [system spec — I5 Retain correctness](./system-spec.md#cross-component-invariants). Authority for the charge-handle property: [memory-tracker-integration spec — I7](./memory-tracker-spec.md#invariants).

**Accounting ownership.** Memory accounting is owned by [memory-tracker-integration spec — Interfaces & contracts](./memory-tracker-spec.md#interfaces--contracts) and orchestrated at the [pollable source](./pollable-source-spec.md#interfaces--contracts). The adoption layer is the carrier of the charge handle inside adopted state; it does not itself call the memory-tracker-integration charge or release entries.

**Unsupported types — owed *to* the source pipeline.** Any type outside `{UInt64, String}` is rejected with a typed exception at the SQL parse/resolve seam (per [pollable-shm-source spec — Producer-side preconditions enumerated](./pollable-source-spec.md#producer-side-preconditions-enumerated)) and at the handshake cross-validation seam (per [shm-block-stream spec — Schema declaration and negotiation](./shm-block-stream-spec.md#schema-declaration-and-negotiation)). No fall-back path exists on the adoption side. Authority: [I4](#invariants).

**Covered `IColumn` method surface — owed *to* downstream operators that consume the emitted `Chunk`.** The set of read methods on an adopted column that the layer commits to behaving indistinguishably from a copy-owned column is fixed by [I1](#invariants) and reproduced in [shm-block-stream spec — AC8 Producer ABI documented in-tree](./shm-block-stream-spec.md#acceptance-criteria). The canonical list, derived by tracing the call graph of [system spec — AC1](./system-spec.md#end-to-end-acceptance-criteria)'s query:

For `ColumnVector<UInt64>` (covers `id`, `v1`, `v2` and the implicit row-count consumer for `count()`):

| Method | Purpose |
|---|---|
| `size_t size() const final` | Row-count consumer for every operator that sizes the chunk; must agree with the chunk's declared row count |
| `const Container & getData() const` (returns `PaddedPODArray<UInt64> &`) | Value-buffer reader used by `sum()` batch aggregation and per-row access |

For `ColumnString` (covers `s1`, `s2`):

| Method | Purpose |
|---|---|
| `size_t size() const override` | Row-count consumer used by `length()` |
| `const Chars & getChars() const` (returns `PaddedPODArray<UInt8> &`) | Byte-data reader used by `cityHash64()` and `length()` |
| `const Offsets & getOffsets() const` (returns `PaddedPODArray<UInt64> &`) | Offset reader used by `cityHash64()` and `length()` |

These methods are reads only. They return references or pointers into the adopted byte buffer; the adoption layer's job is to make those references point into producer memory and to enforce that all callers honour `const`. No method outside this list is in I1's scope under AC1; the documented surface is sufficient for AC1 and is the canonical reference for any future extension.


**Materialization-on-mutation contract — owed *to* any reachable mutating path.** If a method that would mutate the byte buffer is reached for an adopted column, the layer transparently materializes a ClickHouse-owned copy and mutates that copy. If a reachable mutating path has no safe materialization, the layer raises a typed exception before any write. Authority: [I3](#invariants).

Reachable mutation paths under [system spec — AC1](./system-spec.md#end-to-end-acceptance-criteria): none. The AC1 operators — `count()`, `sum()`, `cityHash64()`, `length()` — read the adopted columns and emit fresh result columns; none mutate inputs.

Contractual behaviour on the mutating-method surface of `ColumnVector<UInt64>` and `ColumnString`, in case a future query reaches one:

| Path | Classification |
|---|---|
| Any caller routed through the COW `IColumn::mutate` entry point | **Safe materialize.** The adoption layer ensures that a mutating caller through the standard COW mutation entry receives a freshly allocated, copy-owned column whose buffer is a copy of the adopted bytes. Mutation proceeds against the copy. Producer memory is untouched. Authority: [I3](#invariants) |
| Any call to a column-class mutator (`insertFrom`, `popBack`, `cloneResized`, `insertRangeFrom`, non-const `getData()` / `getChars()` / `getOffsets()`, or any `IColumn` base-class mutator) issued directly against a column still in adopted state — i.e. bypassing the standard COW mutation entry | **Typed exception.** The adoption layer raises a typed exception before any write to producer memory. Phase 1 does not implement transparent rewrite for direct-mutation paths. Authority: [I3](#invariants) |

Direct mutation callers that bypass the COW mutation entry are treated as misuse and fail loudly.

## Invariants

**I1. Observability for supported read paths.** For the phase-1 types and AC1's query, an adopted column is observationally indistinguishable from a copy-owned column across the read-only `IColumn` methods exercised by that query path. The implementation documents the covered method surface (see [shm-block-stream spec — AC8 Producer ABI documented in-tree](./shm-block-stream-spec.md#acceptance-criteria)).

**I3. Adopted memory is immutable from ClickHouse.** No ClickHouse code path writes to producer memory. Any path that requires mutable storage transparently materializes a ClickHouse-owned copy first and then mutates that copy. If safe materialization is not implemented for a reachable path, the query fails with a typed exception before any write — never silently corrupts producer memory. Phase 1 documents which mutation/materialization paths are reachable from AC1's query and which are intentionally unsupported.

**I4. Malformed supported buffers fail loudly.** For AC2 types, buffers that violate the SHM-adoption ABI are rejected with a typed exception. Descriptor-level violations (bad alignment, declared sizes overflow the region, bad declared padding, malformed per-column metadata) are rejected before adopt() returns successfully and before any read of the data-plane payload. Content-level violations that require reading a declared buffer (today: `ColumnString` offsets monotonicity and terminal-offset value) are rejected post-adoption and before the consumer-side read that would otherwise observe the violation. There is no silent copy fall-back from a malformed adopted buffer.

Cross-component and wire invariants — full text in the spec named in each link:

- [system spec — I5 Retain correctness](./system-spec.md#cross-component-invariants) — *the adoption layer owns the retain-token protocol carried by every adopted column*
- [system spec — I10 Exception safety](./system-spec.md#cross-component-invariants) — *the adoption layer owns producer-retain state across adoption failures*
- [shm-block-stream spec — I2 Producer satisfies the SHM-adoption ABI](./shm-block-stream-spec.md#invariants) — *adoption layer rejects any buffer that does not conform to the ABI declared by this invariant*
- [shm-block-stream spec — I11 Producer death is detected through the control plane](./shm-block-stream-spec.md#invariants) — *the wire-level guarantee that retained mappings stay address-valid is what makes adopted reads safe*

## Acceptance criteria

**AC2. Type coverage.** Adoption is supported for exactly `ColumnVector<UInt64>` and `ColumnString`; the set is closed in phase 1. Any other type is rejected with a typed exception — at SQL parse/resolve time for the table function's `columns` argument (per [pollable-shm-source spec — Producer-side preconditions enumerated](./pollable-source-spec.md#producer-side-preconditions-enumerated)) and at handshake cross-validation against the producer-declared schema (per [shm-block-stream spec — Schema declaration and negotiation](./shm-block-stream-spec.md#schema-declaration-and-negotiation)).

**AC3. Adoption proof.** The test asserts that, for the AC2 types on AC1's data, the `Chunk` emitted by the source contains columns whose payload pointers fall inside the producer SHM mapping. Specifically: `ColumnVector<UInt64>` for `id`, `v1`, `v2` has its value buffer adopted; `ColumnString` for `s1`, `s2` has both its chars buffer and its offsets buffer adopted. The pointer-identity check is paired with metadata proving the column references the byte range the producer published for that specific block — not an unrelated allocation that happens to share the same mapping. Every emitted column across the run is adopted — i.e., AC3's pointer-identity check applies to every column of every emitted chunk. Retain counters are positive while adopted columns are alive and return to zero after destruction.

End-to-end and sibling acceptance criteria — full text in the spec named in each link:

- [system spec — AC1 Functional correctness](./system-spec.md#end-to-end-acceptance-criteria) — *the joint query whose correctness the adoption layer participates in*
- [system spec — AC7 Safety / leak audit](./system-spec.md#end-to-end-acceptance-criteria) — *the leak/stability run that observes adoption-layer retains and tracker bytes returning to zero*
- [pollable-shm-source spec — AC4 Pollable wiring works](./pollable-source-spec.md#acceptance-criteria) — *the source feeds adoption from blocks; AC4's drain and cancellation rely on adoption returning cleanly*
- [memory-tracker-integration spec — AC5 MemoryTracker correctness](./memory-tracker-spec.md#acceptance-criteria) — *the charges the adoption layer issues are what AC5 observes*
- [pollable-shm-source spec — AC6 Producer-misbehaviour coverage](./pollable-source-spec.md#acceptance-criteria) — *the "malformed buffer" and "non-conforming buffer" entries in AC6 are detected by the adoption layer per I4*
- [shm-block-stream spec — AC8 Producer ABI documented in-tree](./shm-block-stream-spec.md#acceptance-criteria) — *the doc that, jointly with this spec, lets an external producer be written; the adopted-column method surface lives under I1 here*
- [shm-block-stream spec — AC10 Retain integrity under producer reuse](./shm-block-stream-spec.md#acceptance-criteria) — *the retain protocol whose consumer-side carrier is this layer*

## Stop conditions

Halt and re-open the contract — do not silently work around — if any of the following becomes true.

**S1. Invasive ClickHouse changes.** Achieving adoption requires invasive changes to ClickHouse internals beyond a localized adopt seam (i.e., the changes ripple into the broader column or executor surface).

**S6. Zero adoption on AC1 query.** The adoption proof (AC3) shows zero adoption on the test query after phase 1 lands. The core purpose has not been delivered.

**S7. Test query reaches `IColumn` behaviour the design cannot uphold.** The test query reaches an `IColumn` behaviour that the design cannot uphold within I1 + I3 without materializing every column on entry. Either the test query, the type coverage, or the contract must be revisited; the implementation must not silently degrade.
