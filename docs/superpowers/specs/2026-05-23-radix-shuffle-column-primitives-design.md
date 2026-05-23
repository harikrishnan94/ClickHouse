# Radix Shuffle Type-Specific Column Primitives — Design Spec

**Status:** Draft (awaiting user review)
**Reference baseline:** `/home/ubuntu/phj-bench` (the performance budget we must not exceed)

## 1. Goal

Ship the **type-specific column primitives and bucket/chain seams** in `src/Common/RadixShuffle/` so that a future partitioned-hash-join (PHJ) shuffle inside ClickHouse can call them with the same per-row cost as the `phj-bench` reference. The PHJ algorithm itself (build/probe, hash table, multi-partition orchestration) is **out of scope for v1** — we are shipping the foundation it will sit on.

Concretely, in scope:

- A type-aware **allocator** that hands writable destinations to scatter (defined in §3.1). Per-partition tracking of the resulting chunks is the RadixShuffle operator's bookkeeping, not the allocator's.
- Per-column-type **scatter** primitives (defined in §3.2).
- Per-column-type **reconstruct** primitives (defined in §3.3).
- Per-column-type **hash** primitives (defined in §3.4).
- A dispatcher (`resolveColumnPrimitives`) that resolves the column-primitive triple `(scatter, reconstruct, hash)` from a `const IDataType &` into a `ColumnPrimitives` value.

## 2. Non-goals (explicit)

1. The full PHJ shuffle / build / probe driver.
2. Pid computation, hashing-to-pids, or any radix policy.
3. Histogram (row counts per partition) as part of the column-primitive seam. Histogram is trivial; the RadixShuffle layer provides it. **Not a seam.**
4. Byte-count sizing for variable-length columns as part of the column-primitive seam. The per-(column × partition × batch) primary-byte totals consumed by the allocator's reservation API (§3.1) and required for target pre-allocation by reconstruct (§3.3) are operator bookkeeping — the RadixShuffle layer tracks them however it chooses (e.g., by accumulating what scatter wrote, by a separate sizing pass over source columns, or by maintaining a running counter alongside the row histogram). **Not a seam.**
5. Column types beyond user-selected scope D: `ColumnVector<T>` / `ColumnDecimal<T>` / `ColumnFixedString` / `ColumnString` / `ColumnNullable(X)`. **No** `LowCardinality`, `Array`, `Tuple`, `Map`, `Variant`, `Dynamic`, `Object`, `AggregateFunction`.
6. SWWC (software write-combining buffers), software prefetching, non-temporal stores — explicit non-goals in `phj-bench/README.md`; we honor that.
7. NUMA awareness, multi-socket pinning, work-stealing.
8. CH `MemoryTracker` integration beyond what `Allocator<true>` already gives us.
9. Modifying `IColumn` (no new virtual methods).

## 3. Definitions and API contracts

### 3.1 Allocator

**Definition.** The allocator is owned by the RadixShuffle operator. It hands out writable chunks for scatter to write into. It is **append-only and monotonic**: it supports allocation but never per-chunk deallocation during its lifetime; all its memory is freed in a single act at allocator destruction. It is **type-aware**: it knows each column's element size, alignment, and which secondary arrays (offsets, null map) are needed, and sizes/aligns chunks accordingly.

Tracking which chunks belong to which partition (so that reconstruct can consume them per-partition) is a **separate concern**, not part of this allocator's API. The RadixShuffle operator owns that bookkeeping.

The allocator must balance three conflicting constraints:

1. **High performance, low sync.** The hot path (per-batch reservation) MUST NOT block on cross-thread synchronization, and its per-call cost MUST NOT grow with the number of other threads concurrently using the allocator. Uncontended lock-free operations, atomic counters, and per-thread state are permitted; blocking primitives (mutexes, condition variables, semaphores) and any synchronization whose latency scales with contender count are forbidden on the hot path. Synchronization on the cold path (handle acquire / release / destruction) is permitted.
2. **Low waste.** Total bytes the allocator has handed out via chunks MUST be no more than `active_chains * MIN_CHUNK_FLOOR_BYTES + 1.10 * total_reserved_bytes`, where `active_chains` is the number of (column × partition) chains that currently hold at least one chunk, and `MIN_CHUNK_FLOOR_BYTES` is the per-(column × partition) floor implied by constraint 3 — the minimum bytes the allocator allocates when it has to create a chunk at all. Since reservation is the commit, reserved bytes equal "occupied" bytes from the allocator's accounting perspective. This bound applies to bytes the allocator hands out via chunks; internal bump-allocator pages or per-chunk header bytes used by the implementation to back those chunks are NOT counted toward the bound (a bounded per-handle overhead, e.g., one arena page per active thread, is permitted). The bound must be achieved continuously through the lifetime of the allocator (i.e., without a separate compaction or trim step). The `active_chains * MIN_FLOOR` term acknowledges that for small-scale workloads (few reservations, large P, or skewed pids) the meaningful-rows floor dominates per-chain; once total reservations grow, the 10% percentage component dominates.
3. **Meaningful rows per chunk (minimum row size rule).** Each chunk the allocator allocates MUST be sized to a minimum row count to amortize downstream per-batch fixed overhead. The minimum is a tunable; the default is at least 256 rows for fixed-width columns and an equivalent byte budget for variable-length. A trailing chunk (the final chunk of a sequence) MAY be smaller. This rule applies to chunk **allocation** size, not to caller **reservation** size — small reservations slice within larger chunks; the allocator does not round individual reservations up to the floor.

**API surface (the seam):**

- **Construction.** The allocator is constructed with a description of all columns it will allocate for (per-column element size, alignment, presence of offsets, presence of null map), the partition count `P`, and an expected-total-rows hint used to size initial chunks to roughly match the average per-partition contribution. Construction MUST NOT allocate any chunks; it only computes per-column sizing parameters.
- **Handle acquisition.** A producer thread acquires a handle from the allocator. Acquisition is on the cold path; synchronization is permitted. A thread holds at most one handle. Handles are not transferable across threads.
- **Per-batch reservation (also the commit).** Given a handle, a column index, and per-partition sizing requests (rows and, for variable-length columns, primary bytes), the allocator returns writable destinations sufficient for one scatter call. Reservation is the only act needed — internal cursors advance by the requested amounts atomically with the reservation itself; there is no separate commit step. The caller is responsible for filling the reserved space; under-filling is the caller's choice (the space remains "spent" from the allocator's accounting). Reservation MAY allocate (growing a chain by appending a new chunk if the current tail does not have room). Reservation is on the hot path and is bound by the hot-path synchronization rule in constraint 1.
  **Caller responsibility for written vs reserved row counts.** The allocator counts reserved rows toward the waste bound (constraint 2). It does NOT track how many of those rows the caller actually wrote into. If the caller under-fills, the unused capacity is dead space from the allocator's perspective. Tracking per-chunk actual-written-row counts, and ensuring that downstream consumers (scatter, reconstruct, the RadixShuffle operator) see a consistent view of which rows are real, is the caller's responsibility — not the allocator's.
- **Handle release.** Releasing a handle finalizes its allocations — no further reservations occur through it, and the chunks it returned remain valid for read until allocator destruction. Release is on the cold path; synchronization is permitted.
- **Destruction.** All allocated memory is freed at allocator destruction. There is no per-chunk or per-handle deallocation before destruction.

**Behavioral contract:**

- Hot-path synchronization rule (constraint 1) holds: no blocking primitives, no contention-scaled latency on reservation. Implementation MAY use per-thread arenas, lock-free freelists, atomic counters, or similar.
- Waste bound holds continuously throughout the allocator's lifetime: `total_allocated_bytes <= active_chains * MIN_CHUNK_FLOOR_BYTES + 1.10 * total_reserved_bytes`, without any separate compaction or trim step.
- Meaningful-rows bound holds for every chunk handed out by the allocator, modulo trailing chunks.
- Each chunk's primary, offsets, and null-map regions meet the alignment required by the column type's element type.
- The allocator never deallocates a chunk during its lifetime.

### 3.2 Scatter

**Definition.** Scatter takes a source column of N rows, a per-row partition assignment `pids[]` of size N (each value in `[0, P)`), and a set of P per-partition writable destinations. It appends to each partition's destination, in the order they appear in the source, the rows of the source column for which `pids[j] == p`. The bytes written to each destination are whatever uniquely represent those rows for later reconstruction; the layout is private to the column primitive for that column type.

**Pre-conditions (caller's responsibility):**

- `pids[j] < P` for all `j ∈ [0, N)`.
- Each of the P destinations has sufficient capacity for the rows that will land in it.
- The source column's concrete type matches the type the resolved `ColumnPrimitives` was built for.

**Post-conditions (column primitive's guarantee):**

- For every partition `p`, the destination contains, contiguously appended after any prior content, every row `j` with `pids[j] == p`, in ascending `j` order.
- Total rows appended across all destinations equals `N`. The source column is unchanged.
- The destinations, taken together for one partition, are sufficient and necessary input to the reconstruct primitive (§3.3) for the same column type.

### 3.3 Reconstruct

**Definition.** Reconstruct takes an **ordered list of chunk-range views** (the "input buffer list"), a starting position within that list, and a pre-allocated target column. Each view in the list is a `(chunk, [begin, end))` pair — a reference to a chunk produced by scatter together with a half-open row range over that chunk identifying which rows contain actual written data. Reserved-but-unwritten slots are excluded by the range itself; reconstruct never sees them.

The order of views in the input list is chosen by the caller; the spec does NOT require it to match the order scatter produced the chunks. Reconstruct appends rows from the views into the target, **only up to the target's pre-allocated capacity**, and returns the position of the first unconsumed row across the input list.

The caller drives reconstruct as a pump: allocate target capacity, call reconstruct, observe the returned position, optionally allocate more target capacity and call again with the returned position as the new start.

**Pre-conditions (caller's responsibility):**

- Target column has been pre-allocated to the desired capacity. **For variable-length column types** (`ColumnString`, and `ColumnNullable(ColumnString)`), pre-allocation means reserving BOTH the row-count capacity (e.g., `offsets`) AND the byte-content capacity (e.g., `chars`) sufficient to hold the data being reconstructed. **For `ColumnNullable(X)`**, pre-allocation additionally requires that the null-map capacity be AT LEAST the nested column's row capacity (so the null map can absorb every row that reconstruct decides to append into the nested column). The caller is responsible for sizing every backing array such that this asymmetric-pre-reservation invariant holds; how the caller obtains byte totals is operator bookkeeping (§2 non-goal #4), not a column-primitive-seam concern. Reconstruct's stop decision is driven by the nested column's row capacity and (for variable-length nested columns) its byte capacity; the null map is a pure caller pre-condition, not a runtime stop dimension. The returned position reflects whichever stop boundary was reached first.
- All chunks referenced by views in the input list were produced by scatter calls using the `ColumnPrimitives` resolved for the same data type as the target.
- Each view's `[begin, end)` range covers rows that scatter actually wrote into the referenced chunk and nothing else. Constructing this range correctly (using the per-chunk written-row count the caller tracks per §3.1) is the caller's responsibility; reconstruct consumes the range verbatim and assumes it is correct.
- The starting position `(view_index, rows_consumed_in_view)` is `(0, 0)` for the first call, or the value returned by a prior reconstruct call on the same input list.

**Post-conditions (column primitive's guarantee):**

- Rows are appended into the target up to but not exceeding the target's pre-allocated capacity (rows AND, for variable-length types, bytes). The target's `size()` after the call reflects the number of rows actually appended.
- Target capacity (in rows and in bytes) is unchanged.
- Input views and the chunks they reference are unchanged. Underlying buffers remain valid for further reconstruct calls.
- For every view that is consumed in full, exactly `end - begin` rows from that view are appended to the target. For the trailing view that may be consumed only partially, the rows appended correspond to a contiguous prefix of its `[begin, end)` range starting at `begin + rows_consumed_in_view` from the starting position.
- Returned position `(next_view_index, next_rows_consumed_in_view)` identifies the first unconsumed row across the input list. When the input list is exhausted, the position is the end sentinel `(views.size(), 0)`.
- The bytes appended equal what scatter produced for the consumed rows, decoded back into the target column's native layout.

### 3.4 Hash

**Definition.** Hash takes a source column of N rows and a caller-allocated output array of at least N UInt64s. It updates the output array such that, after the call, `out[i]` reflects the column's content for row `i`, mixed with the array's prior contents using a documented combiner. The combiner is the same across all `ColumnPrimitives` resolved by the dispatcher, so a caller may chain hash calls across multiple columns to compute a composite per-row hash key.

**Pre-conditions (caller's responsibility):**

- Output array is pre-allocated with at least N entries.
- Output array is initialized to whatever the caller intends as the starting value for the documented combiner (e.g., zero or a seed for fresh hashing; the result of a prior hash call for accumulation across columns).
- Source column's concrete type matches the resolved `ColumnPrimitives`.

**Post-conditions (column primitive's guarantee):**

- For every `i ∈ [0, N)`, `out[i]` equals `combiner(prior_out[i], h(src[i]))` where `h(.)` is the column primitive's per-row hash function for this column type. The combiner is uniform across column primitives.
- Source unchanged.

### 3.5 Round-trip invariant

For any supported column type and any partition assignment `pids[]`:

`source → scatter → (for each partition: caller assembles the input buffer list in any order it chooses; reconstruct repeatedly until the input list is exhausted) → concatenate partitions in pid order`

yields a column whose multiset of rows equals the source's multiset of rows. The output's row order depends on the caller's chosen input-buffer-list order; the spec does not require it to match the source order.

**Composite types (ColumnNullable).** For `ColumnNullable(X)`, the round-trip preserves BOTH the null map AND the nested column's content byte-for-byte. Every nested-X row is scattered and reconstructed regardless of whether the null map marks the row as NULL; the nested column's data at null positions is preserved (not normalized, not stripped, not defaulted). Equivalently, scatter and reconstruct treat ColumnNullable as two parallel sub-columns each subject to the round-trip invariant independently.

## 4. Contracts

### 4.1 Performance contract

Column primitives are always called in batches. The benchmark (`bench_radix_shuffle_column_primitives`) measures, over a configurable workload, the per-row cost of each primitive.

Workload parameters, all CLI-configurable:

- **Batch size** (rows processed per column-primitive invocation) — required sweep: `{1024, 4096, 16384}`.
- **Total number of rows** — CLI-configurable; defines how many batches run for each configuration.
- **P** (partition count) — required sweep: `{4, 8, 16, 32, 64, 128, 256}`.
- **K** (total columns) — required sweep: `{1, 2, 4, 8}`.
- **T** (total threads) — required sweep: `{1, 4, 8, 16, 32, 48}`.

For every supported column type and every configuration in the workload sweep, the benchmark measures and reports the achieved performance. There is no statistical gating; the benchmark is a measurement tool, not a pass/fail check.

Reported numbers per (column type, workload configuration):

- **ns/row** for the scatter primitive (median over the configured batches).
- **Total bandwidth** across all threads, expressed as rows/second (= threads × per-thread rows/second).

`phj-bench`'s published numbers remain a reference baseline; comparison against them is a manual engineer-driven activity, not a CI gate.

For variable-length and nullable types, the structural contract still applies:

- Branch-free row loop in the scatter primitive.
- Exactly one batch-level dispatch per column (no per-row virtual calls).

### 4.2 Behavioral contract

- **Round-trip identity (§3.5)** holds for every supported column type.
- **Scatter, reconstruct, and hash primitives MUST NOT allocate.** The allocator (§3.1) is the sole source of allocation in this work; its reservation path MAY allocate (this is its purpose).
- **Hash combiner is uniform** across column primitives: chaining hash calls across multiple columns of any supported types produces a deterministic result that depends only on the column contents and the combiner.
- **Allocator waste bound** (§3.1, constraint 2) holds continuously throughout the allocator's lifetime — not just at end-of-phase — with no separate compaction or trim step: `total_allocated_bytes <= active_chains * MIN_CHUNK_FLOOR_BYTES + 1.10 * total_reserved_bytes`. The `active_chains * MIN_CHUNK_FLOOR_BYTES` term is the only carve-out and exists solely to satisfy constraint 3 (minimum chunk size per chain).
- **Allocator meaningful-rows bound** (§3.1, constraint 3) holds for every chunk except the trailing chunk of each chain.
- **Allocator hot-path synchronization rule** (§3.1, constraint 1) holds: no blocking primitives, no contention-scaled latency on reservation.

## 5. Acceptance criteria

- Benchmark runs the full workload sweep (§4.1) and reports, per supported column type and per configuration, the achieved ns/row and total bandwidth across all threads.
- Round-trip identity (§3.5) holds for every supported column type.
- Reconstruct's returned position correctly resumes when used as the start of a subsequent call on the same input list; assembling the final column across multiple bounded calls is byte-equivalent to a single sufficiently-allocated call.
- Hash combiner is uniform: composing hash calls across columns of different types in different orders yields results that differ only by the well-defined combiner.
- Scatter, reconstruct, and hash primitives do not allocate.
- Allocator waste bound holds continuously throughout the allocator's lifetime (`allocated_bytes <= active_chains * MIN_CHUNK_FLOOR_BYTES + 1.10 * reserved_bytes`, per §3.1 constraint 2). The bound is verified at multiple intermediate points during the allocator's lifetime, not only at the end.
- Allocator meaningful-rows bound holds for every chunk it hands out, except trailing chunks.
- Allocator hot-path synchronization rule (§3.1, constraint 1) holds: no blocking primitives on reservation; per-call cost does not scale with the number of contending threads.
- All `clang-tidy` warnings and errors in newly added files are fixed.
- All newly added files compile under CH's existing warning set with `-Werror`.
- Tests pass under both `-O3` (release) and `asan` builds.
- Build succeeds via standard CH build flow.
- Benchmark prints a stdout summary by default (per workload configuration and per type, the achieved ns/row and total bandwidth across threads). CSV output is supported via an opt-in CLI flag and is disabled by default.
