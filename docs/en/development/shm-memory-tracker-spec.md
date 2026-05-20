---
description: 'Spec for the MemoryTracker integration component: RAII accounting of SHM adopted bytes, limit enforcement, and observability.'
sidebar_label: 'SHM MemoryTracker Integration Spec'
sidebar_position: 205
slug: /development/shm-memory-tracker-spec
title: 'MemoryTracker Integration — SHM Adopted-Byte Accounting'
doc_type: 'reference'
---

# MemoryTracker Integration — SHM Adopted-Byte Accounting

This spec defines component (c) — sub-deliverable C — of the zero-copy SHM source feature: the integration that makes the SHM bytes the consumer holds visible to ClickHouse's memory accounting and limits.

System mission, glossary, non-goals, cross-component invariants (I5, I10), and end-to-end ACs (AC1, AC7) are owned by [system spec](./shm-system-spec.md). The producer-facing wire is owned by [shm-block-stream spec](./shm-block-stream-spec.md). Column construction is owned by [adoption-layer spec](./shm-adoption-layer-spec.md). Executor integration is owned by [pollable-shm-source spec](./shm-pollable-source-spec.md).

## Mission

Make the retained SHM working set visible to ClickHouse's memory accounting and limits without double-counting and without silently overcommitting past `max_memory_usage`, so that:

- a feature-local RAII counter exactly tracks currently-adopted bytes, charged once on retain acquisition (refcount 0 to 1) and released once on retain refcount return to zero;
- those charges enter the active query-level MemoryTracker so that `max_memory_usage` observes the retained SHM working set within ClickHouse's documented tracker slack;
- a retain acquisition whose charge would exceed `max_memory_usage` fails cleanly, with rollback to the producer, and without retained-SHM leak.

This integration is invoked by the pollable source on each per-block retain acquisition; it does not interpose on the executor and does not touch the wire. It is the only consumer-side party that translates adopted bytes into MemoryTracker bookkeeping.

## Non-goals

- This component does not own column construction; it exposes the charge/release surface consumed by [pollable-shm-source spec — Interfaces & contracts](./shm-pollable-source-spec.md#interfaces--contracts). It does not decide *what* counts as adopted bytes — that is fixed by the **Adopted byte count** entry in [system spec — Glossary](./shm-system-spec.md#glossary).
- This component does not redefine, extend, or rewrite ClickHouse's `MemoryTracker` or the limit-checking semantics. It uses the existing surfaces.
- This component does not account for memory the producer already accounted for through ClickHouse's allocator surface. Double-counting is explicitly out of scope; the boundary is "bytes adopted from foreign memory that ClickHouse's allocator never saw" (see [I7](#invariants)).
- This component does not own executor integration, cancellation, or fd lifetime; those live in [pollable-shm-source spec — Interfaces & contracts](./shm-pollable-source-spec.md#interfaces--contracts). Cancellation-time release of charges happens automatically because the source destroys its RAII charge handles when adopted state is released; this component does not need a cancellation hook of its own.
- This component does not own the wire's per-type buffer layout; the byte ranges it accounts are described by [shm-block-stream spec — Wire interfaces](./shm-block-stream-spec.md#wire-interfaces) and selected per the glossary's "Adopted byte count" rule.

System-level non-goals — full text in [system spec — Non-goals](./shm-system-spec.md#non-goals). N5 (no replacement or refactor of existing copy-based ingestion paths, of which `MemoryTracker` integration is one slice) and N7 (no coverage of every `IColumn` kind) are the system-level constraints that most directly bind this component.

## Constraints

- The adopted-byte counter is exact at the feature boundary: each byte range is charged exactly once on retain acquisition and released exactly once on retain refcount return to zero, with no double-counting against producer-allocated-through-ClickHouse memory. Authority: [I7](#invariants); failure mode: [S3](#stop-conditions).
- Charges enter the active query-level MemoryTracker so that `max_memory_usage` observes the retained SHM working set, within ClickHouse's documented per-thread `max_untracked_memory` slack plus normal concurrent-allocation tolerance. Authority: [I7](#invariants); failure mode: [S3](#stop-conditions).
- A retain acquisition whose charge would exceed `max_memory_usage` fails the query cleanly, with rollback to the producer (the wire retain acquired by the source is released, and the byte range is not charged). Producer-published bytes are not silently overcommitted past `max_memory_usage`. Authority: [I8](#invariants).
- Charge/release are RAII: an exception between charge-handle acquisition and ownership transfer to adopted state must execute charge-handle destruction (reversing the charge) before propagation. This is the local consequence of [system spec — I10 Exception safety](./shm-system-spec.md#cross-component-invariants).
- The bytes reported are the **Adopted byte count** as defined in [system spec — Glossary](./shm-system-spec.md#glossary): for `ColumnVector<UInt64>` the value buffer plus its safe-read padding, for `ColumnString` the chars and offsets buffers plus their safe-read padding. The handshake region and per-slot control-plane metadata declared in [shm-block-stream spec — ABI version negotiation](./shm-block-stream-spec.md#abi-version-negotiation) and [Block framing](./shm-block-stream-spec.md#block-framing) are not charged; they are fixed-size and observed once per attach / per publication, not retained for column lifetime. The implementation reports both logical payload bytes (without padding) and charged adopted bytes (with padding) for observability.

## Interfaces & contracts

**Charge entry point — owed *to* [pollable-shm-source spec — Interfaces & contracts](./shm-pollable-source-spec.md#interfaces--contracts).** Immediately after a successful per-block wire retain acquisition (refcount 0 to 1), the source calls this entry, naming the adopted byte count for that block. The entry:

- increments the feature-local adopted-byte counter exactly once for the range;
- enters the charge into the active query-level MemoryTracker so the charge is visible to `max_memory_usage`;
- raises a typed exception, *before* recording the charge, if the charge would exceed `max_memory_usage`. The exception is rollback-safe — the wire retain acquired by the source in the prior step is released by the source's own RAII before the block is abandoned; no charged-then-rolled-back transient exists. The source raises `memory-limit-exceeded` per [pollable-shm-source spec — Failure classes](./shm-pollable-source-spec.md#interfaces--contracts).

On success, the entry returns a charge handle. Ownership of this handle is transferred to adopted state (alongside the retain token) by the subsequent `adopt()` call per [adoption-layer spec — Interfaces & contracts](./shm-adoption-layer-spec.md#interfaces--contracts).

**Release semantics — owed *to* [pollable-shm-source spec — Interfaces & contracts](./shm-pollable-source-spec.md#interfaces--contracts).** Release is effected by charge-handle destruction — the handle's destructor:

- decrements the feature-local adopted-byte counter exactly once for the range;
- reverses the entry into the active query-level MemoryTracker, within documented slack and concurrent-allocation tolerance.

Each successful charge is balanced by exactly one handle destruction. Handle destruction without a prior successful charge is a programming error.

**Observability of the feature-local counter — owed *to* the test in [AC5](#acceptance-criteria).** The feature-local adopted-byte counter is per-source-instance, test-visible, and returns to zero on source destruction, so that the test can assert the counter reaches the expected peak and returns to zero across a complete source lifecycle.

**Process-wide observability — owed *to* operators.** The implementation publishes a process-wide current-bytes gauge via `system.metrics` and a cumulative-charge counter via `ProfileEvents`, so that operators can observe SHM working-set pressure without enabling test instrumentation. Exact metric names are implementation detail; their existence (a live-bytes gauge and a cumulative-charge counter) and shape (current value readable via `system.metrics`, cumulative count readable via `ProfileEvents`) are the contract.

**Reporting of logical vs charged bytes — owed *to* operators and observability surfaces.** The implementation reports both logical payload bytes and charged adopted bytes (per the glossary's "Adopted byte count" entry) through both the per-source and process-wide surfaces. The distinction is safe-read padding: logical payload is the data bytes; charged adopted bytes additionally include the safe-read padding required by ClickHouse's column-storage contract.

**No producer-facing surface.** This component does not communicate with the producer. The only wire-side dependency is the **Adopted byte count** rule, which is defined in the glossary.

## Invariants

**I7. Adopted-byte accounting is exact at the feature boundary; query-level tracker reflects it within documented slack.** The pollable source maintains an exact RAII counter of currently-adopted bytes: each byte range is charged exactly once on retain acquisition (refcount 0 to 1) and released exactly once on retain refcount return to zero, with no double-counting against memory the producer already accounted for through ClickHouse's allocator surface. These charges enter the active query-level MemoryTracker so that `max_memory_usage` observes the retained SHM working set, subject only to ClickHouse's documented per-thread `max_untracked_memory` slack and normal concurrent-allocation tolerance.

**I8. Memory limit is enforced.** A retain acquisition whose charge would exceed `max_memory_usage` fails the query cleanly, with rollback to the producer. Producer-published bytes are not silently overcommitted past ClickHouse's configured `max_memory_usage`.

Cross-component and wire invariants — full text in the spec named in each link:

- [system spec — I5 Retain correctness](./shm-system-spec.md#cross-component-invariants) — *charge/release pair shares the same RAII scope as the retain token; a leak of charges and a leak of retains have the same root cause*
- [system spec — I10 Exception safety](./shm-system-spec.md#cross-component-invariants) — *this component is one of the three surfaces (tracker, retains, fds) that must roll back atomically on failure*

## Acceptance criteria

**AC5. MemoryTracker correctness.** Across an isolated test run: the feature-local adopted-byte counter reaches the expected peak and returns to zero; the query-level MemoryTracker peak delta is at least the peak charged adopted byte count minus the test's max-thread-count × `max_untracked_memory` slack (i.e. `peak_charged − max_threads × max_untracked_memory ≤ tracker_peak`), subject to documented tracker behaviour (relaxed atomics, concurrent-allocation tolerance); the tracker returns to within `max_threads × max_untracked_memory` of its pre-query baseline after the source is destroyed; running under a `max_memory_usage` lower than the working set fails cleanly — the typed error fires at the source's charge step (post-retain-acquisition, before `adopt()` is called), with no charged-then-rolled-back transient and no retained SHM leak, raising the `memory-limit-exceeded` class per [pollable-shm-source spec — Failure classes](./shm-pollable-source-spec.md#interfaces--contracts).
End-to-end and sibling acceptance criteria — full text in the spec named in each link:

- [system spec — AC1 Functional correctness](./shm-system-spec.md#end-to-end-acceptance-criteria) — *the joint query whose adopt path triggers charge/release pairs*
- [system spec — AC7 Safety / leak audit](./shm-system-spec.md#end-to-end-acceptance-criteria) — *the leak/stability run observes tracker bytes returning to baseline across ≥1000 iterations*
- [adoption-layer spec — AC3 Adoption proof](./shm-adoption-layer-spec.md#acceptance-criteria) — *AC3 asserts retain counters return to zero; AC5 asserts the matching tracker baseline*

## Stop conditions

Halt and re-open the contract — do not silently work around — if any of the following becomes true.

**S3. Adopted-byte exactness or query-level tracker propagation infeasible.** The feature-local adopted-byte RAII counter cannot be made exact across a single block lifecycle, or the query-level MemoryTracker charge cannot be made close enough for `max_memory_usage` to observe the retained SHM working set within documented tracker slack and concurrent-allocation tolerance.
