---
description: 'Spec for the pollable SHM source component: async executor integration, block ingestion, table function, cancellation, and producer-misbehaviour coverage.'
sidebar_label: 'SHM Pollable Source Spec'
sidebar_position: 204
slug: /streamed_table/specs/pollable-source-spec
title: 'Pollable SHM Source — Async Executor Integration'
doc_type: 'reference'
---

# Pollable SHM Source — Async Executor Integration

This spec defines component (b) — sub-deliverable B — of the zero-copy SHM source feature: a new `IProcessor` that consumes producer-published SHM blocks via ClickHouse's async/poll path and emits `Chunk`s of adopted columns, reached from SQL via a new `shm()` table function.

System mission, glossary, non-goals, cross-component invariants (I5, I10), and end-to-end ACs (AC1, AC7) are owned by [system spec](./system-spec.md). The producer-facing wire is owned by [shm-block-stream spec](./shm-block-stream-spec.md). Column construction is owned by [adoption-layer spec](./adoption-layer-spec.md). Memory accounting is owned by [memory-tracker-integration spec](./memory-tracker-spec.md).

## Mission

Build a query-pipeline source that:

- participates in ClickHouse's standard async/poll executor path — cooperative cancellation, fd lifetime ownership, bounded-time async-completion callback, level-triggered drain — so that a query reading from a producer-driven SHM stream waits, drains, and cancels exactly like any other async source;
- consumes producer-published SHM blocks through the wire defined in [shm-block-stream spec — Wire interfaces](./shm-block-stream-spec.md#wire-interfaces), routing each block's declared byte ranges to the adoption layer and assembling the returned `IColumn`s into `Chunk`s;
- exposes itself to SQL through a new table function, `shm()`, gated by an experimental setting;
- bounds cancellation regardless of producer state — a stalled, slow, or crashed producer cannot make cancellation unbounded;
- surfaces every in-scope producer-misbehaviour case as a typed exception drawn from a named failure class, with no hang, no UB, and no reliance on signal handling. The phase-1 in-scope set and explicit exclusions are pinned in [Failure scope](#failure-scope).

The source is the only consumer-side reader of the control plane. It is the only place where readiness fds and block lifecycle state machine transitions are observed; everything else hangs off the adopted columns it emits.

## Non-goals

- The source does not own column construction. Each adopted `IColumn` is built by [adoption-layer spec — Interfaces & contracts](./adoption-layer-spec.md#interfaces--contracts). The source's relationship to the byte plane is mediated entirely by the adoption layer.
- The source does not own the wire's per-type buffer layout, padding rules, framing, or version handshake. Those live in [shm-block-stream spec — Wire interfaces](./shm-block-stream-spec.md#wire-interfaces). The source observes the control plane; it does not define it.
- The source does not own memory-accounting policy, the limit-enforcement decision, or the slack model; those live in [memory-tracker-integration spec — Interfaces & contracts](./memory-tracker-spec.md#interfaces--contracts). The source orchestrates the per-block charge sequence per [Per-block adoption call](#interfaces--contracts) — it calls the memory-tracker-integration charge entry on retain acquisition and hands the resulting charge handle into adopt() — but it does not perform the accounting itself or interpose on the tracker chain. The adoption layer only carries the charge handle through adopted state; release is RAII via charge-handle destruction at adopted-state final drop.
- The source exposes a single stream per instance in phase 1; no multi-stream parallelism inside the source (per [system spec — N11](./system-spec.md#non-goals)). Pipeline parallelism is downstream's concern.
- The source does not modify ClickHouse's `IProcessor` or executor contract. It implements against the existing contract. If it cannot, [S2](#stop-conditions) fires.

System-level non-goals — full text in [system spec — Non-goals](./system-spec.md#non-goals). N3 (no general-purpose IPC framework — one source type, one direction), N8 (no operator changes outside the new source), and N11 (no multi-stream parallelism inside the source) are the system-level constraints that most directly bind this component.

## Constraints

- The source satisfies the executor's async/poll-processor contract — cooperative cancellation, fd lifetime ownership, bounded-time async-completion callback, level-triggered drain semantics, and correct behaviour under both single-threaded and multi-threaded executors. Authority: [I6](#invariants).
- Query cancellation reclaims every SHM resource the consumer holds within a bounded time, regardless of producer state. Authority: [I9](#invariants); upper-bound failure: [S4](#stop-conditions).
- Producer stall is bounded by an observable, configurable timeout owned by the source. If no producer publication progress is observed within the configured budget while the source has no drainable block, no end-of-stream has been observed, and the query is not cancelled, the source surfaces a typed exception. "Publication progress" means the transition of any slot into the `published` state. Authority: [I12](#invariants).
- The set of producer-side preconditions whose violation must be deterministically detectable on the consumer side is finite and bounded; if it cannot be, [S5](#stop-conditions) fires. The wire-side definition of those preconditions lives in [shm-block-stream spec — Wire interfaces](./shm-block-stream-spec.md#wire-interfaces); the *detection bound* is a property this component must uphold. The enumeration is given in [Producer-side preconditions enumerated](#producer-side-preconditions-enumerated) below.
- The new `shm` table function is gated behind an explicit experimental setting, disabled by default. Authority: [AC9](#acceptance-criteria).
- Producer death after a complete stream is *not* an error; the source drains remaining retained blocks. Producer death before end-of-stream, framing errors, mid-stream aborts, and stalls *are* errors, surfaced as typed exceptions. Authority: [shm-block-stream spec — I11 Producer death is detected through the control plane](./shm-block-stream-spec.md#invariants).
- The source never relies on catching SIGBUS or SIGSEGV for normal error handling. Authority: [shm-block-stream spec — I11](./shm-block-stream-spec.md#invariants).
- fd lifetime — the readiness fd is opened, registered, drained, and closed by the source, with the close/unregister occurring exactly once across the source's lifetime even on exception paths. Authority: [I6](#invariants) and [AC4](#acceptance-criteria).
- Per-block size and row-count limits — the consumer enforces such limits as defined by [shm-block-stream spec — Wire interfaces](./shm-block-stream-spec.md#wire-interfaces). Concrete numeric values are implementation detail; their existence and effect on the control plane are pinned by the wire.

### Producer-side preconditions enumerated

The producer-side preconditions whose violation the consumer must deterministically detect, derived from [shm-block-stream spec — Wire interfaces](./shm-block-stream-spec.md#wire-interfaces). Each is checked at a single, well-defined point on the consumer side. The list is finite (26 entries); this discharges the finiteness obligation backing [S5](#stop-conditions).

| # | Precondition | Detection point | Wire-spec reference |
|---|---|---|---|
| 1 | `magic` matches the SHM-adoption-ABI sentinel | first read of handshake region after `mmap` | [ABI version negotiation](./shm-block-stream-spec.md#abi-version-negotiation) |
| 2 | `abi_version` is in the set the consumer supports (phase 1: `{1}`) | first read of handshake region | [ABI version negotiation](./shm-block-stream-spec.md#abi-version-negotiation) |
| 3 | `ring_depth_K` is in `[1, implementation_max_K]` | first read of handshake region | [Block framing](./shm-block-stream-spec.md#block-framing) |
| 4 | `schema_count` matches the SQL-declared column count | handshake cross-validation | [Schema declaration and negotiation](./shm-block-stream-spec.md#schema-declaration-and-negotiation) |
| 5 | Each producer-declared column name matches the SQL-declared name at the same position | handshake cross-validation | [Schema declaration and negotiation](./shm-block-stream-spec.md#schema-declaration-and-negotiation) |
| 6 | Each SQL-declared type and each producer-declared type is a member of the supported set `{UInt64, String}`, and each producer-declared type string parses via `DataTypeFactory::get` and equals the SQL-declared type at the same position | SQL parse/resolve of the table function's `columns` argument (SQL-side membership gate, before attach); handshake cross-validation (producer-side membership and equality) | [Schema declaration and negotiation](./shm-block-stream-spec.md#schema-declaration-and-negotiation) |
| 7 | `slot_table_offset`, `data_region_offset`, `data_region_size`, and `slot_table_stride` collectively fit within the SHM object and do not overlap each other | first read of handshake region | [ABI version negotiation](./shm-block-stream-spec.md#abi-version-negotiation) |
| 8 | Observed slot state value is a member of the publication state machine's defined state set | acquire-load of slot state | [Publication state machine](./shm-block-stream-spec.md#publication-state-machine) |
| 9 | Slot identity field equals its position in the slot table | acquire-load of slot metadata | [Block framing](./shm-block-stream-spec.md#block-framing) |
| 10 | Block sequence number is strictly greater than the previous adopted block's sequence number for the same slot | per-slot identity check | [Region identity and reuse rules](./shm-block-stream-spec.md#region-identity-and-reuse-rules) |
| 11 | `row_count ≤ implementation_max_rows_per_block` | per-block validation | [Size and row-count limit communication](./shm-block-stream-spec.md#size-and-row-count-limit-communication) |
| 12 | `per_column[]` length equals `schema_count` | per-block validation | [Block framing](./shm-block-stream-spec.md#block-framing) |
| 13 | For `ColumnVector<UInt64>`: declared value-buffer offset satisfies the alignment required by the column-storage contract | per-column descriptor check | [Per-type buffer layout](./shm-block-stream-spec.md#per-type-buffer-layout) |
| 14 | For `ColumnVector<UInt64>`: value count × element size + declared padding fits within the `data_region` capacity at the declared offset | per-column descriptor check | [Per-type buffer layout](./shm-block-stream-spec.md#per-type-buffer-layout) |
| 15 | For `ColumnVector<UInt64>`: declared padding meets the safe-read padding required by the column-storage contract | per-column descriptor check | [Per-type buffer layout](./shm-block-stream-spec.md#per-type-buffer-layout) |
| 16 | For `ColumnString`: declared `chars` offset satisfies the alignment required by the column-storage contract | per-column descriptor check | [Per-type buffer layout](./shm-block-stream-spec.md#per-type-buffer-layout) |
| 17 | For `ColumnString`: declared `offsets` offset satisfies the alignment required by the column-storage contract | per-column descriptor check | [Per-type buffer layout](./shm-block-stream-spec.md#per-type-buffer-layout) |
| 18 | For `ColumnString`: `chars` byte size + `chars` padding fits within the `data_region` capacity at the declared offset | per-column descriptor check | [Per-type buffer layout](./shm-block-stream-spec.md#per-type-buffer-layout) |
| 19 | For `ColumnString`: `offsets` byte size + `offsets` padding fits within the `data_region` capacity at the declared offset | per-column descriptor check | [Per-type buffer layout](./shm-block-stream-spec.md#per-type-buffer-layout) |
| 20 | For `ColumnString`: `chars` padding and `offsets` padding each meet the safe-read padding required by the column-storage contract | per-column descriptor check | [Per-type buffer layout](./shm-block-stream-spec.md#per-type-buffer-layout) |
| 21 | For `ColumnString`: each value in the `offsets` buffer is non-decreasing relative to the previous offset | post-adoption integrity check (lazy; performed only if a downstream read would be invalidated by violation) | [Per-type buffer layout](./shm-block-stream-spec.md#per-type-buffer-layout) |
| 22 | For `ColumnString`: the final `offsets` value equals the declared `chars` byte size | post-adoption integrity check | [Per-type buffer layout](./shm-block-stream-spec.md#per-type-buffer-layout) |
| 23 | After a block is published with the EOS marker set, no further blocks transition to published | EOS state-machine validation | [End-of-stream](./shm-block-stream-spec.md#end-of-stream) |
| 24 | Slot transitions follow the publication state machine in order (no skipping) | state-machine validation per acquire-load | [Publication state machine](./shm-block-stream-spec.md#publication-state-machine) |
| 25 | Producer does not detach (unlink the backing object, drop fds, and terminate) before signalling end-of-stream while retain refcounts on any slot are non-zero — surfaces a `producer-death-before-eos` class exception, tested by [AC6](#acceptance-criteria) | control-plane heartbeat or backing-object re-check after wake | [I11](./shm-block-stream-spec.md#invariants) |
| 26 | For each per-column descriptor: declared value count (for `ColumnVector<UInt64>`) or declared `offsets` element count (for `ColumnString`) equals the block's `row_count` | per-column descriptor check | [Block framing](./shm-block-stream-spec.md#block-framing) |

Memory-ordering and notification-ordering violations are *contracts* with no direct detection point; their consumer-observable effects are downstream corruption that the bounds/format checks above catch. They are not separately enumerated.

Consumer-side runtime asserts (e.g. that the consumer's atomic decrement of any retain refcount never produces a negative value) are by-construction discipline checks on the consumer side. They are not part of the producer-side detection enumeration above and do not count against [S5](#stop-conditions)'s finiteness obligation; their violation is a consumer bug, not a producer misbehaviour.

Retain-protocol-violation by the producer (truncation, unmap, region alteration under live retain) is out of scope per [system spec — Glossary](./system-spec.md#glossary), entry **Trust model**, and is not enumerated above.

### Failure scope

The phase-1 boundary for typed-exception coverage:

**In scope** — each surfaces a class from [Failure classes](#failure-classes):

- every entry in [Producer-side preconditions enumerated](#producer-side-preconditions-enumerated);
- producer stall, per [I12](#invariants);
- producer death before EOS, per [shm-block-stream spec — I11](./shm-block-stream-spec.md#invariants);
- SHM attach-time failures per [Attach-time observable failures](#attach-time-observable-failures);
- memory-limit failure at per-block retain-acquisition charge step, per [memory-tracker-integration spec — I8](./memory-tracker-spec.md#invariants);
- feature-gate disabled, per [AC9](#acceptance-criteria).

**Out of scope** — per [system spec — Glossary](./system-spec.md#glossary), entry **Trust model**:

- retain-protocol violation (truncation, unmap, backing-object alteration under live retain);
- memory-ordering and notification-ordering contract violations — no direct detection point; downstream effects are caught only insofar as they manifest as bounds/format violations already enumerated above.

**Illustrative excluded behaviours** (non-exhaustive):

- `ftruncate(0)` on the SHM object under live retain — consumer may fault on a data-plane read; not detected.
- Producer publishes metadata before payload writes become globally visible — surfaced (if at all) as a `buffer-layout-invalid` class exception, not as a distinct "ordering violation" class.
- Producer holds an fd indefinitely without publishing — bounded by [I12](#invariants) and surfaced as `producer-stall`.

## Interfaces & contracts

**`shm()` table function — owed *to* SQL.** The new table function exposes this source with the positional signature:

```sql
shm(name, columns)
```

- `name` (String): the SHM object name, suitable for the phase-1 SHM primitive (e.g. illustratively `'/clickhouse_shm_<id>'`). See [shm-block-stream spec — SHM primitive](./shm-block-stream-spec.md#shm-primitive).
- `columns` (String): the schema spec in ClickHouse's standard table-function column-list grammar (e.g. illustratively `'id UInt64, v1 UInt64, v2 UInt64, s1 String, s2 String'`). Parsed at SQL-resolve time per [shm-block-stream spec — Schema declaration and negotiation](./shm-block-stream-spec.md#schema-declaration-and-negotiation).

The readiness notification fd and any auxiliary fds are not surfaced through the SQL signature; they are obtained from the SHM object's control-plane header on attach, per [shm-block-stream spec — Notification contract](./shm-block-stream-spec.md#notification-contract).

The function is gated by an experimental setting per [AC9](#acceptance-criteria); the gate is checked at parse/resolve time and surfaces a `feature-gate-disabled` class exception if disabled.

**Attach-time observable failures — owed *to* SQL.** The user-visible outcomes of running `SELECT … FROM shm(name, columns)` against a bad or absent producer are enumerated below. Each surfaces a class from [Failure classes](#failure-classes); none hang, none yield UB.

| Observable outcome | Failure class | Detection point |
|---|---|---|
| Experimental setting is `false` in the session | `feature-gate-disabled` | SQL parse/resolve |
| `columns` argument contains a type outside `{UInt64, String}` | `schema-mismatch` | SQL parse/resolve (membership gate, before attach) |
| SHM object `name` does not exist | `attach-failed` | source attach (object open) |
| SHM object `name` exists but is not accessible (permissions, etc.) | `attach-failed` | source attach (open / `mmap`) |
| Object opens but the handshake region carries a wrong magic, unsupported ABI version, an out-of-range `ring_depth_K`, or out-of-bounds region offsets (including an uninitialized handshake) | `handshake-invalid` | first acquire-read of handshake region |
| Handshake declares a readiness-fd locator the consumer cannot resolve to a valid fd | `attach-failed` | locator resolution after handshake read |
| Handshake declares a schema that disagrees with the SQL-declared `columns` (count, name, type, or order) | `schema-mismatch` | handshake cross-validation |

The handshake region is the single observation point that distinguishes "no producer here" (`attach-failed`) from "wrong/stale producer here" (`handshake-invalid`). Once the handshake validates, runtime failures are surfaced through the remaining classes per [Failure scope](#failure-scope).

**`IProcessor` integration — owed *to* the executor.** The source implements ClickHouse's `IProcessor` async/poll-processor contract. The observable surface comprises at least:

- `Status::Async` is returned only when no producer block is currently drainable, and a readiness fd is provided so the executor can wait on it; tested in [AC4](#acceptance-criteria).
- All currently-available blocks are drained before the readiness fd is re-armed; tested in [AC4](#acceptance-criteria).
- Spurious readiness signals do not produce a chunk; tested in [AC4](#acceptance-criteria).
- The async-completion callback (`onAsyncJobReady()` or equivalent) does not block; tested in [AC4](#acceptance-criteria).
- The readiness fd is closed and unregistered exactly once across the source's lifetime; tested in [AC4](#acceptance-criteria).
- Cooperative cancellation, bounded by [I9](#invariants) and tested in [AC4](#acceptance-criteria), terminates the source and releases retained SHM/fds without requiring producer cooperation.

Authority for the joint property: [I6](#invariants). The contract is *implemented against*, not modified; failure mode is [S2](#stop-conditions).

**Block ingestion — owed *to* and *from* [shm-block-stream spec — Wire interfaces](./shm-block-stream-spec.md#wire-interfaces).** The source observes the publication state machine and the notification contract through the wire. It distinguishes complete blocks from incomplete, stale, or aborted blocks per the wire's region-identity rules. Per the wire's memory-ordering contract, it observes payload before metadata under acquire/release. On end-of-stream it stops requesting blocks; on producer death without end-of-stream it raises a typed exception.

**Per-block adoption call — owed *to* [adoption-layer spec — Interfaces & contracts](./adoption-layer-spec.md#interfaces--contracts) and [memory-tracker-integration spec — Interfaces & contracts](./memory-tracker-spec.md#interfaces--contracts).** For each drainable block the source executes the following RAII-guarded sequence:

1. Acquire the wire retain against the block (refcount 0 to 1).
2. Call the memory-tracker-integration charge entry with the block's adopted byte count. If the charge would exceed `max_memory_usage`, the entry raises `memory-limit-exceeded` before recording the charge; the source's RAII releases the wire retain and the exception propagates.
3. Call `adopt()` with the producer-declared layout, the retain token, and the charge handle. If adopt() fails, the source's RAII releases both the retain and the charge handle before propagating the exception.

On success, ownership of both handles transfers into the adopted columns returned by adopt(). The source receives back one `IColumn` per declared column and does not touch data-plane bytes itself.

**`Chunk` emission — owed *to* downstream pipeline.** The source assembles adopted columns into a `Chunk` and emits it. Per [system spec — N8](./system-spec.md#non-goals) and [adoption-layer spec — I1](./adoption-layer-spec.md#invariants), adopted columns are consumable by the existing pipeline without semantic differences from copy-owned columns on the read paths exercised by [system spec — AC1](./system-spec.md#end-to-end-acceptance-criteria). Rows within an emitted `Chunk` reflect the producer's buffer order for that block; the source makes no promise about the order in which chunks are emitted relative to producer publication order. Downstream operators receive blocks in drain order. Per [system spec — N12](./system-spec.md#non-goals), downstream operators must not rely on any global row order across chunks.

**Cancellation — owed *to* the executor.** On cancellation the source releases every retained SHM resource and closes the readiness fd within a bounded time `T` (the wire/adoption layers' resources are released RAII-style on retain drop and adopted-column destruction respectively). Producer cooperation is *not* required. Authority: [I9](#invariants); failure mode: [S4](#stop-conditions); test: [AC4](#acceptance-criteria) cancellation sub-test.

**Producer-failure surfacing — owed *to* the query.** Every in-scope producer misbehaviour — malformed block, non-conforming buffer, mid-publication crash, stall (no publication progress within the configured budget), post-publish-pre-EOS death — surfaces as a typed exception drawn from a named failure class on the consumer side. The contractual coverage matrix is [AC6](#acceptance-criteria). The boundary's no-signal-handling guarantee is owed by [shm-block-stream spec — I11](./shm-block-stream-spec.md#invariants); the stall budget is owed by [I12](#invariants). Retain-protocol violation — region truncation/unmap under live retain — is out of scope per [system spec — Glossary](./system-spec.md#glossary), entry **Trust model**, and per [AC6](#acceptance-criteria)'s OOS clause.

**Failure classes — owed *to* the query and to tests.** Every typed exception the source surfaces falls into exactly one of the named classes below. The class is the stable, test-assertable taxonomy. Tests and operators distinguish failure modes by class, not by exception string content.

| Class | Triggered by | Detection point |
|---|---|---|
| `feature-gate-disabled` | [AC9](#acceptance-criteria) gate is off | SQL parse/resolve of `shm()` |
| `attach-failed` | SHM object missing, inaccessible, or readiness-fd locator unresolvable | source attach |
| `handshake-invalid` | preconditions 1–3, 7 | first acquire-read of handshake region |
| `schema-mismatch` | preconditions 4–6 | SQL parse/resolve (membership) and handshake cross-validation (equality, order, count) |
| `block-framing-invalid` | preconditions 8–12, 23, 24, 26 | per-block state-machine / identity validation |
| `buffer-layout-invalid` | preconditions 13–22 | per-column descriptor check (13–20) or post-adoption content check (21–22) |
| `producer-stall` | [I12](#invariants) budget elapsed with no publication progress | source's stall timer |
| `producer-death-before-eos` | producer detach observed before EOS while retains are live (precondition 25), per [shm-block-stream spec — I11](./shm-block-stream-spec.md#invariants) | control-plane heartbeat or backing-object re-check after wake |
| `memory-limit-exceeded` | retain-acquisition charge rejected per [memory-tracker-integration spec — I8](./memory-tracker-spec.md#invariants) | source's charge step (post-retain-acquisition, before adopt()) |

Each AC6 scenario is bound to a specific class; the test asserts on the class identity.

## Invariants

**I6. Pollable contract.** The new source correctly satisfies the contract that ClickHouse's executor places on async/poll processors: cooperative cancellation, fd lifetime ownership, bounded-time async-completion callback, level-triggered drain semantics, and correct behaviour under both single-threaded and multi-threaded executors.

**I9. Cancellation is bounded.** Query cancellation reclaims every SHM resource the consumer holds within a bounded time, regardless of producer state. A stalled, slow, or crashed producer cannot make cancellation unbounded.

**I12. Stall is bounded.** A producer that ceases making publication progress without dying and without signalling end-of-stream causes the source to surface a typed exception within a configurable observable budget. The budget is finite, deterministic per build mode, and reportable via the query's exception text.

Cross-component and wire invariants — full text in the spec named in each link:

- [system spec — I5 Retain correctness](./system-spec.md#cross-component-invariants) — *the source owns block-reuse sequencing on the consumer side; together with the adoption layer's retain protocol this makes I5 hold*
- [system spec — I10 Exception safety](./system-spec.md#cross-component-invariants) — *the source owns fd lifetime across exception paths; failures during ingestion must close fds and release retains cleanly*
- [shm-block-stream spec — I2 Producer satisfies the SHM-adoption ABI](./shm-block-stream-spec.md#invariants) — *the source is entitled to assume conformance; non-conforming buffers route to the adoption layer's typed-exception path*
- [shm-block-stream spec — I11 Producer death is detected through the control plane](./shm-block-stream-spec.md#invariants) — *the source is the only consumer-side reader of the control plane; this is where producer death gets observed*

## Acceptance criteria

**AC4. Pollable wiring works.** The test exercises AC1's query under both `max_threads = 1` and `max_threads ≥ 2`, asserting that the source:

- returns `Status::Async` only when no producer block is currently drainable;
- drains all currently-available blocks before re-arming its readiness fd;
- tolerates spurious readiness signals without producing a chunk;
- does not block in `onAsyncJobReady()`;
- closes and unregisters its readiness fd exactly once across the source's lifetime.

A separate cancellation test stalls the producer after some output, issues a query cancellation, and asserts the query terminates and releases all retained SHM/fds within a bounded time `T` without requiring producer cooperation. `T` is bounded and deterministic per build mode, defined by the test harness against the executor's cancellation-check cadence (`interactive_delay`):

| Build mode | T |
|---|---|
| Release | 5 × `interactive_delay` (default 500 ms) |
| ASan / TSan / MSan / UBSan | 30 × `interactive_delay` (default 3 s) |
| Debug / slow CI | 100 × `interactive_delay` (default 10 s) |

The test treats exceeding `T` as a regression. Reaching `T` is by construction a stop trigger for [S4](#stop-conditions).

**AC6. Producer-misbehaviour coverage.** The test exercises at least:

- a malformed block → `block-framing-invalid`;
- a non-conforming buffer (violating I2) → `buffer-layout-invalid`;
- a producer crash mid-publication (before a complete block is published) → `block-framing-invalid`;
- a producer that, *after* the consumer has released all retains on a block, reuses/truncates/republishes that region — the consumer continues without incident (positive test of retain-protocol observance);
- a producer dying *after* publishing a complete block but *before* signalling end-of-stream, with the consumer still holding a live retain on that block — the consumer drains the live retain to completion (per I11) and surfaces a `producer-death-before-eos` class exception for the missing end-of-stream signal;
- a producer that publishes one or more blocks then ceases making publication progress without dying and without signalling end-of-stream → `producer-stall`, surfaced within the [I12](#invariants) budget;
- attach-time outcomes: object missing, object inaccessible, handshake invalid (wrong magic / unsupported ABI / out-of-range region offsets), schema mismatch (count/name/type/order), readiness-fd locator unresolvable — each test asserts the corresponding class from [Failure classes](#failure-classes). The `feature-gate-disabled` case is covered by [AC9](#acceptance-criteria).

Every failure path surfaces a typed exception drawn from the named class. No hang, no UB.

The test asserts on the failure class, not on string content or generic exception type. "Some exception" does not satisfy AC6.

A producer that violates the retain/release contract (e.g. truncates or unmaps a region while consumer retains against it are live) is out of scope per [system spec — Glossary](./system-spec.md#glossary), entry **Trust model**, and per [shm-block-stream spec — SHM primitive](./shm-block-stream-spec.md#shm-primitive)'s rationale; AC6 does not cover that case in phase 1.

**AC9. Feature gate.** The new `shm` table function is exposed behind an experimental setting (Bool, default `false`). The gate is checked at parse/resolve time of any `shm()` reference; if the setting is `false`, parsing fails with a typed exception. Phase 1 does not claim production hardening or multi-tenant safety.

End-to-end and sibling acceptance criteria — full text in the spec named in each link:

- [system spec — AC1 Functional correctness](./system-spec.md#end-to-end-acceptance-criteria) — *the joint query the source emits chunks for*
- [system spec — AC7 Safety / leak audit](./system-spec.md#end-to-end-acceptance-criteria) — *the stable-fd / stable-SHM-segment check observes this source's lifecycle directly*
- [adoption-layer spec — AC2 Type coverage](./adoption-layer-spec.md#acceptance-criteria) — *the set of types the source feeds through the adoption layer end-to-end*
- [adoption-layer spec — AC3 Adoption proof](./adoption-layer-spec.md#acceptance-criteria) — *the pointer-identity check on chunks this source emits*
- [memory-tracker-integration spec — AC5 MemoryTracker correctness](./memory-tracker-spec.md#acceptance-criteria) — *the limit-failure sub-test of AC5 fires during this source's adopt path*
- [shm-block-stream spec — AC8 Producer ABI documented in-tree](./shm-block-stream-spec.md#acceptance-criteria) — *the wire's notification/state-machine sub-bullets are what this source satisfies on the consumer side*
- [shm-block-stream spec — AC10 Retain integrity under producer reuse](./shm-block-stream-spec.md#acceptance-criteria) — *the source sequences block reuse on the consumer side; AC10 is observed via this source plus the adoption layer's retain*

## Stop conditions

Halt and re-open the contract — do not silently work around — if any of the following becomes true.

**S2. IProcessor / executor contract modification required.** The implementation requires modifying ClickHouse's `IProcessor` / executor contract, rather than implementing against it.

**S4. Cancellation unbounded.** Cancellation cannot reliably reclaim SHM in bounded time under any reproducible sequence (I9 is unachievable).

**S5. Producer-side preconditions not finite/bounded.** The set of producer-side preconditions whose violation must be deterministically detectable (I4, I11) cannot be made finite and bounded.
