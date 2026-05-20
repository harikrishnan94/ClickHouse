---
description: 'Wire contract between any conforming external SHM producer and the ClickHouse SHM consumer: layout, framing, ordering, notification, retain/release, and version negotiation.'
sidebar_label: 'SHM Block-Stream Wire Contract'
sidebar_position: 202
slug: /pg_auto_click/specs/shm-block-stream-spec
title: 'SHM Block-Stream Wire Contract'
doc_type: 'reference'
---

# SHM Block-Stream Wire Contract

This is the boundary contract between any conforming external producer and the ClickHouse consumer of the zero-copy SHM source feature. It is satisfied by both sides: producers conform to it; the consumer (jointly satisfied by [adoption-layer spec](./adoption-layer-spec.md) and [pollable-shm-source spec](./pollable-source-spec.md)) is entitled to assume it. It is not a sub-deliverable — there is no spec for a specific producer implementation or a test harness, just this wire.

System mission, glossary, non-goals, and cross-component invariants live in [system spec](./system-spec.md).

## Mission

Define the semantic obligations of the wire between any conforming external producer and the ClickHouse SHM consumer:

- This spec defines the wire's semantic obligations — layout semantics, framing rules, ordering contracts, notification semantics, retain/release semantics, and version-negotiation rules. Concrete encodings (exact byte layouts, field offsets, magic values, on-wire numeric values, fd plumbing choices) live in a separate versioned ABI artifact whose form (a markdown spec, a C++ header in the ClickHouse source tree, or both) is decided at implementation time. This spec and that artifact are jointly sufficient for an external producer to be implemented; neither is sufficient alone.
- The ClickHouse consumer can rely on a single, versioned, named set of producer-side guarantees that span layout, framing, ordering, notification, retain/release, and version negotiation.
- The boundary's failure modes (malformed buffers, framing errors, producer death, region reuse contention) have wire-level treatments that translate to typed exceptions on the consumer side without consumer-side signal handling.

## Scope and trust model

The wire spec covers the producer/consumer interface only. It does *not* cover:

- the consumer's executor integration (lives in [pollable-shm-source spec](./pollable-source-spec.md)),
- the consumer's column construction (lives in [adoption-layer spec](./adoption-layer-spec.md)),
- ClickHouse's internal memory accounting (lives in [memory-tracker-integration spec](./memory-tracker-spec.md)).

The phase-1 trust model is owned by the system spec's glossary: see [system spec — Glossary](./system-spec.md#glossary), entry **Trust model**. Verbatim restatement is not duplicated here. The wire-level consequences are:

- the consumer handles malformed metadata, bad layout declarations, premature stream termination, producer crash, and stall (bounded by [pollable-shm-source spec — I12 Stall is bounded](./pollable-source-spec.md#invariants)) — as typed exceptions through the control plane;
- the consumer does *not* sandbox a malicious producer with arbitrary capability to invalidate the backing object after publication; the retain protocol and the chosen SHM primitive bound that risk;
- the consumer does *not* rely on catching SIGBUS or SIGSEGV for normal error handling (see [I11](#invariants)).

Per-type coverage of the wire — i.e. which `IColumn` kinds the wire is obligated to carry in phase 1 — is set by [adoption-layer spec — AC2 Type coverage](./adoption-layer-spec.md#acceptance-criteria). The wire carries exactly `ColumnVector<UInt64>` and `ColumnString`; producer schemas containing any other type are rejected at handshake cross-validation, with the SQL side rejecting at parse/resolve time per [pollable-shm-source spec — Producer-side preconditions enumerated](./pollable-source-spec.md#producer-side-preconditions-enumerated). No copy fall-back path exists on the consumer side.

## Wire interfaces

Each subsection states the semantic obligations the producer must satisfy and what the consumer is entitled to assume. Concrete encodings (numeric values, exact message layouts, fd plumbing choices) are in the AC8 versioned ABI artifact. The obligations are pinned here.

### Per-type buffer layout

For each AC2 type: buffer count, alignment, padding, safe-read area, offset encoding, byte order, lifetime rules.

The consumer's column-storage contract — ClickHouse's existing alignment, padding, and safe-read requirements for the byte buffers backing `IColumn`s — is the constraint these layouts must satisfy. The wire ABI is *distinct from* ClickHouse's internal column-storage implementation: the producer ABI may mirror it where convenient, but the internal layout is not promoted to public-ABI status by this work (see [I2](#invariants)).

#### Semantic obligations per AC2 type

For `ColumnVector<UInt64>` ([adoption-layer spec — Covered IColumn method surface](./adoption-layer-spec.md#interfaces--contracts)):

- Exactly one value buffer per column.
- Alignment: the buffer's start offset must satisfy ClickHouse's column-storage alignment contract for `UInt64`.
- Safe-read padding: a consumer-readable trailing padding region follows the last value, matching ClickHouse's `PaddedPODArray` padding contract. Padding bytes need not be zero-filled but must be safely readable (no fault on access).
- The per-column metadata field declares the start offset (bytes into the SHM region) and the value count. The declared value count must equal the block's `row_count`.

For `ColumnString` ([adoption-layer spec — Covered IColumn method surface](./adoption-layer-spec.md#interfaces--contracts)):

- Two buffers per column: `chars` (concatenated string bytes, laid out per `ColumnString`'s representation) and `offsets` (array of unsigned past-end offsets matching `ColumnString::Offsets` convention).
- Both buffers must satisfy ClickHouse's column-storage alignment and padding contracts.
- Per-column metadata declares: `chars` start offset, `chars` byte size, `chars` padding amount; `offsets` start offset, `offsets` element count, `offsets` padding amount. The declared `offsets` element count must equal the block's `row_count`.

### Schema declaration and negotiation

The schema (ordered list of `(column-name, column-type)` entries) is declared in two places and cross-validated on attach:

1. **SQL site.** The `shm()` table function takes a schema spec as one of its arguments (exact signature owned by [pollable-shm-source spec — Interfaces & contracts](./pollable-source-spec.md#interfaces--contracts)). ClickHouse parses and resolves the schema from this spec at parse/resolve time, before executing the query. This is what fixes the chunk header structure downstream operators see.
2. **Control plane.** The producer publishes its own copy of the schema as part of the attach-time handshake (see [ABI version negotiation](#abi-version-negotiation)), preceding any block publication. The producer commits to publishing blocks whose column count, order, and types match this declaration.

On attach, the consumer cross-validates the SQL-declared schema against the producer-published schema. Any of {column-count mismatch, column-name mismatch, column-type mismatch, column-order mismatch} surfaces as a typed exception, raised by the source before any block adoption.

The producer ABI commits to publishing the schema in this declaration before publishing any block. The exact on-wire encoding is delegated to the AC8 versioned ABI artifact; the field semantics are pinned here in [ABI version negotiation](#abi-version-negotiation).

### Block framing

Row-count semantics including zero-row blocks; how end-of-stream is signalled; per-block metadata.

The wire admits a bounded number of simultaneously in-flight published-but-unreleased blocks. The bound (`K`) is declared by the producer at attach time in the handshake (see [ABI version negotiation](#abi-version-negotiation)). `K ≥ 1`; an implementation-defined upper bound caps `K` to keep producer pre-allocation bounded.

Each block carries:

- **Identity.** A slot index and a monotonically-increasing sequence number that together form a globally-unique block identity across the stream lifetime. Used by the consumer for region-identity and reuse checks (see [Region identity and reuse rules](#region-identity-and-reuse-rules)).
- **Lifecycle state.** The block's current position in the [Publication state machine](#publication-state-machine).
- **Retain refcount.** An atomic counter the consumer increments on adoption and decrements on drop (see [Retain/release contract](#retainrelease-contract)).
- **Row count.** Number of rows; zero is permitted (empty-block semantics). Every per-column descriptor in the same block must describe exactly `row_count` logical values: value count for `ColumnVector<UInt64>`, offsets element count for `ColumnString`. A mismatch is a block-framing violation surfaced as a `block-framing-invalid` class exception on the consumer side.
- **EOS marker.** A boolean; when set, this block (after consumer drain) is the last one. The producer commits to no further publications. The EOS marker is carried in the same slot metadata as the rest of publication state to keep it in the same memory-ordering domain (see [Memory ordering](#memory-ordering)).
- **Per-column descriptors.** One entry per declared schema column, carrying the per-type sub-buffer descriptors declared in [Per-type buffer layout](#per-type-buffer-layout) above.

#### End-of-stream

The producer signals end-of-stream by publishing a block (possibly with row count zero) whose EOS marker is set. No further blocks become published after the EOS block; the producer may exit. The consumer recognises EOS by observing the EOS marker under acquire ordering on its read of the block metadata.

### Size and row-count limit communication

The existence and communication of any per-block size or row-count limits the consumer enforces. Concrete numeric values are implementation detail.

### Backpressure and ring-full contract

When the in-flight published-but-unreleased block count equals `K` (the ring is full), the producer is contracted to block (or fail with a typed wire-level error surfaced through the control plane) until at least one block transitions out of published via consumer release. The producer must not overwrite a block whose retain refcount is non-zero.

The choice between blocking and failing on ring-full is producer-side policy and is not pinned by this wire. The consumer side is identical under either: it drains and releases at its own pace.

### Publication state machine

Each block progresses through a defined lifecycle. The producer drives all transitions until the block is published; the consumer drives the transition from published to retained-to-consumer (by incrementing the retain refcount) and from retained-to-consumer to released (by decrementing the retain refcount to zero).

A block is only readable from the consumer side once it has reached the published state. Transitions out of writing states must be observable to the consumer with the memory ordering specified below.

### Memory ordering

Acquire/release semantics, payload-before-metadata visibility, readiness notification ordering.

The producer's payload writes must be globally visible before the metadata write that publishes the block; the consumer's acquire on metadata implies acquire of payload. Readiness notification is ordered after both.

### Region identity and reuse rules

How the consumer distinguishes a complete block from an incomplete, stale, or aborted block; when a region is permitted to be reused. The interaction between region reuse and live consumer-side retains is the subject of [AC10](#acceptance-criteria).

### Notification contract

A level-triggered, drainable readiness fd integrates with the consumer's executor pollable-fd contract. Spurious wakes are admissible: the consumer must verify actual block availability against the publication state machine and not assume "fd readable ⇒ block ready." The producer's signal to the fd must be ordered after the release-store of the metadata that publishes the block (see [Memory ordering](#memory-ordering)).

The consumer drains all currently-published blocks before re-arming readiness, and tolerates spurious wakeups. The pollable-source-side contract for honoring this is [pollable-shm-source spec — I6 Pollable contract](./pollable-source-spec.md#invariants).

The fd is created by the producer and passed to the consumer.

### Retain/release contract

Token lifetime, refcount semantics.

A retain token is acquired by the consumer at adoption time and released exactly once at adopted-state destruction. The producer is contracted not to alter the backing object's offset coverage for a region while its retain refcount is non-zero. The consumer-side mechanism that carries this token through every adopted-column lifetime is [adoption-layer spec — Interfaces & contracts](./adoption-layer-spec.md#interfaces--contracts).

This is a *protocol-level* obligation, not a kernel-enforced one. The phase-1 SHM primitive does not provide kernel-enforced non-truncation. Phase 1's trust model ([system spec — Glossary](./system-spec.md#glossary), entry **Trust model**) makes this assumption explicit; consumer-side defences against retain-protocol violation are out of scope.

### SHM primitive

An SHM primitive whose mappings remain address-valid for the lifetime of the consumer's mapping or fd, as long as the producer conforms to the [Retain/release contract](#retainrelease-contract). The primitive is not required to kernel-enforce non-truncation in phase 1 (see Trust model above).

Out-of-scope alternatives include kernel-enforced sealing primitives; promotion to a sealed primitive would require an explicit ABI version bump per [I2](#invariants).

### ABI version negotiation

How producer and consumer agree on the ABI version at attach time.

Per [I2](#invariants), changes to this ABI are explicit versioned bumps; producer and consumer must agree on a version on attach and reject incompatible versions through the control plane.

An attach-time handshake region — written by the producer under release ordering before any block reaches the published state — establishes:

- a magic sentinel identifying the object as a conforming SHM-adoption-ABI source;
- the ABI version (phase 1 = version `1`); consumer rejects unknown versions (typed exception per [adoption-layer spec — I4](./adoption-layer-spec.md#invariants));
- the ring bound `K`; consumer rejects values exceeding its implementation-defined upper bound;
- the producer-declared schema (`column count`, ordered `(name, type_string)` pairs); consumer cross-validates against the SQL-declared schema per [Schema declaration and negotiation](#schema-declaration-and-negotiation);
- a locator sufficient for the consumer to obtain the readiness fd before the first published block;
- the location and stride of the per-block metadata table within the SHM object;
- the location and size of the data plane within the SHM object.

The consumer's first read of the handshake region is acquire-ordered and happens-before any data-plane read.

### Covered `IColumn` method surface

The read-side guarantees the consumer makes about adopted columns. The canonical statement and coverage list live in [adoption-layer spec — I1 Observability for supported read paths](./adoption-layer-spec.md#invariants). The producer ABI doc references that list so an external producer knows what consumer-side read behaviour the published bytes must support; the guarantee itself is the adoption layer's.

## Invariants

**I2. Producer satisfies the SHM-adoption ABI.** The producer publishes buffers conforming to a versioned **SHM-adoption ABI** whose semantic obligations are defined by this spec and whose concrete encodings are defined by the AC8 versioned ABI artifact; together they specify per supported type the layout, alignment, padding, safe-read area, offset encoding, byte order, and lifetime rules. The ABI may mirror ClickHouse's current in-memory layouts where convenient, but does not commit ClickHouse's internal column-storage implementation to public-ABI status. Changes to the adoption ABI are explicit versioned bumps to the artifact per [AC8](#acceptance-criteria).

**I11. Producer death is detected through the control plane, not by faulting on the data plane.** Under a producer that conforms to the [retain/release contract](#retainrelease-contract) (no truncate, unmap, or backing-object alteration of a region while its retain refcount is non-zero), the consumer's mapping is address-valid for the lifetime of every retain token. Producer death before a declared end-of-stream, framing errors, mid-stream aborts, and stalls (bounded by [pollable-shm-source spec — I12 Stall is bounded](./pollable-source-spec.md#invariants)) are detected through the control channel and surfaced as typed exceptions. Producer death *after* publishing a complete stream and signalling end-of-stream is not an error: the consumer may drain remaining retained blocks as long as their retain tokens keep the mappings valid. The implementation must not rely on catching SIGBUS or SIGSEGV for normal error handling. Non-conforming producers (e.g. truncation under live retain) are out of scope per [system spec — Glossary](./system-spec.md#glossary), entry **Trust model**; the chosen SHM primitive is not required to kernel-enforce non-truncation.

Cross-component invariants and component-local invariants — full text in the spec named in each link:

- [system spec — I5 Retain correctness](./system-spec.md#cross-component-invariants) — *the consumer-side retain protocol that this wire's retain/release contract is paired with*
- [system spec — I10 Exception safety](./system-spec.md#cross-component-invariants) — *a failed adoption must roll back wire-side retain state along with the rest*
- [adoption-layer spec — I1 Observability for supported read paths](./adoption-layer-spec.md#invariants) — *the consumer-side read surface that the producer ABI promises to support*
- [adoption-layer spec — I3 Adopted memory is immutable from ClickHouse](./adoption-layer-spec.md#invariants) — *the producer can assume the consumer never writes through the wire*
- [adoption-layer spec — I4 Malformed supported buffers fail loudly](./adoption-layer-spec.md#invariants) — *consumer-side validation of wire conformance*

## Acceptance criteria

**AC8. Producer ABI artifact.** A versioned ABI artifact checked in alongside the code that, together with this spec, is sufficient for an external producer to be implemented. The artifact's form (a markdown spec, a C++ header in the ClickHouse source tree, or both) is decided at implementation time; the artifact is versioned per I2 and covers the topics enumerated in [Wire interfaces](#wire-interfaces): per-type buffer layout; per-column row-count consistency; schema declaration / negotiation; schema-mismatch behaviour; block framing; size and row-count limit communication; backpressure / ring-full contract; publication state machine; memory ordering; region identity and reuse rules; notification contract; retain/release contract; ABI version negotiation; the covered `IColumn` method surface (the canonical list of which lives in [adoption-layer spec — I1](./adoption-layer-spec.md#invariants)).

**AC10. Retain integrity under producer reuse.** While the test holds a still-live adopted `Chunk` from a published block `B`, the producer attempts to republish region `B` with different bytes and is sequenced behind the retain release — the attempt blocks (or is rejected) by the retain protocol until the held `Chunk` is released. The bytes visible through the held `Chunk` remain bit-identical to what was published when adoption occurred. After the test releases the `Chunk`, retain counters return to zero and the producer's republish completes. The republished bytes appear on subsequent adoptions of region `B`.

Truncation, unmap, and other backing-object alterations under live retain are out of scope per [system spec — Glossary](./system-spec.md#glossary), entry **Trust model**; the chosen SHM primitive does not kernel-enforce against them, and the wire relies on producer conformance. AC10 covers only retain-protocol-conforming reuse.

End-to-end and component-level acceptance criteria — full text in the spec named in each link:

- [system spec — AC1 Functional correctness](./system-spec.md#end-to-end-acceptance-criteria) — *the joint correctness check the wire participates in*
- [system spec — AC7 Safety / leak audit](./system-spec.md#end-to-end-acceptance-criteria) — *the leak/stability check; wire-side fds and SHM segments are within its scope*
- [adoption-layer spec — AC2 Type coverage](./adoption-layer-spec.md#acceptance-criteria) — *the set of types the wire is obligated to carry in phase 1*
- [adoption-layer spec — AC3 Adoption proof](./adoption-layer-spec.md#acceptance-criteria) — *the pointer-identity check that the wire's payload actually lands as the adopted column's bytes*
- [pollable-shm-source spec — AC4 Pollable wiring works](./pollable-source-spec.md#acceptance-criteria) — *the consumer-side honoring of the notification contract*
- [pollable-shm-source spec — AC6 Producer-misbehaviour coverage](./pollable-source-spec.md#acceptance-criteria) — *the producer-failure matrix this wire is graded against (I11 last-sentence trigger)*

## Stop conditions

The following triggers halt-and-reopen on the wire side rather than silent workaround.

- **Trust-model expansion to non-conforming producers.** If a future scope change brings non-conforming producers (truncate/unmap/reuse under live retain) into scope, the current SHM primitive does not kernel-enforce address-validity and the wire must be revisited — either by switching to a sealed primitive under an ABI version bump per [I2](#invariants), or by re-opening [I11](#invariants).
- [pollable-shm-source spec — S5 Producer-side preconditions not finite/bounded](./pollable-source-spec.md#stop-conditions) — *cross-referenced because the producer-side preconditions are wire-defined; the consumer-side detection bound lives in the pollable source spec*
