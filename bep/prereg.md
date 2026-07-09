# RADIX-JOIN-V1 — Pre-registration (§9.3: written BEFORE gathering acceptance evidence)

Discovery output (bep/discovery/*.md) is orientation, not acceptance evidence (§11).
Each unit's pre-registration is appended here before that unit's acceptance evidence is gathered.

## U1 — Component port
Pre-registered 2026-07-09T17:30Z, before any port work. Two commit seams.

### Seam (a): donor component library + gtest
Ported from `origin/phj5-real:src/Interpreters/RadixHashJoin/` via `git restore --source`.

Files expected to port VERBATIM (no donor-visible behavior change, at most include fixes):
- `Arena.{h,cpp}`, `Hll.h`, `KeyLayout.h`, `PackedKeyHash.h`, `ParallelFor.h`, `PartitionPlan.{h,cpp}`.

Files expected to need SURGERY (upstream renamed `BuildRef`/`BuildRefList` → `RowRef`/`RowRefList`):
- `KeyRefScatter.{h,cpp}`, `BuildSide.{h,cpp}`, `LeafTable.{h,cpp}`, `tests/gtest_radix_hash_join.cpp`.
  Surgery = mechanical rename + re-verification that HEAD's `RowRefList` API (`word`,
  `insert(UInt64 ref_word, Arena &)`, iterator `begin`/`ok`/`++`, `refWordIsInline`, `Batch` layout)
  still matches donor semantics. Predicted risk point: `Batch` chained-node layout drift.

DEFERRED (not in U1): `RadixHashJoin.{h,cpp}` (IJoin-entangled → U2), donor
`src/Interpreters/benchmarks/` (needs IJoin; decide at U6), ProfileEvents/CurrentMetrics/
ThreadName/ScopedLLCMissCounter/HashTablesStatistics wiring (RadixHashJoin.cpp deps → U2).

CMake: `add_object_library(clickhouse_interpreters_radix_hash_join Interpreters/RadixHashJoin)`
next to the existing `clickhouse_interpreters_hash_join` line; gtest auto-globbed into
`unit_tests_dbms`.

Behavior-preservation proof per component (the donor's own suite, ported with the renames):
- KeyRefScatter: `ScatterColumnRoundTripDirect`, parametrised `FusedKeyWidthFanout`
  (widths {4,8,12,16,24,32,56,64} × leaves {16,512,8192}, DIRECT + SWWC — SWWC variants expected
  to SKIP on this aarch64 host since donor NT path is x86-only).
- BuildSide: `BuildProbeUniqueKeys`, `BuildProbeManyToManyParallel`, `BuildProbeForcedMultiPass`,
  `BuildProbeMultiPassSwwc` (skips on ARM), no-churn property (`alloc_count` == non-empty leaves).
- LeafTable: `BuildProbeHeavyDuplicatesFewKeys` (AMAC same-key in-flight stress — THE test for
  RowRefList drift), `BuildProbeGroupedLeaves4096`, `GroupedLeavesEmptyLeafSlotInitialized`,
  `BuildProbeGroupedLeaves64ByteKeys`, `GroupedLeavesLoadInvariant`,
  `DistinctEstimateNeverUndersizesLeaf` (overflow-rebuild guard).
- Hll: `HllEstimateAccuracy`, `HllMergeEqualsSingleSketch`, `DistinctEstimateShrinksDuplicateHeavyBuild`.
- PackedKeyHash/PartitionPlan: `PackedKeyHashDeterministicAndSpread`, `PartitionPlanSizingAndPasses`.

Expected outcome: all ported gtests GREEN in build/reldeb; the same suite GREEN under an ASan build;
style check clean on the new files. Confirm = green runs (logged, subagent-analyzed).
Refute/finding = any test needing a semantic (non-rename) change to pass — that is a real
behavioral divergence to investigate and log, not to patch silently.

### Seam (b): productionized scatter kernels from `src/Common/benchmarks/hash_join_bench.{h,cpp}`
Addendum pre-registered 2026-07-09T17:55Z (after seam (a) reldeb green; before any seam (b) code).

New files: `src/Interpreters/RadixHashJoin/ColumnScatter.{h,cpp}` (kernels + barrier wave loop),
`src/Interpreters/RadixHashJoin/tests/gtest_column_scatter.cpp` (new test file).
The benchmark TU (`src/Common/benchmarks/`) is NOT modified — it keeps working as the evidence base.

What ports (from bench-kernels.md inventory): histogramChunk + 4-lane interleave
(HIST_INTERLEAVE_MAX_FANOUT=2048), fused prefix-sum + exact one-shot allocation with per-worker
disjoint partition ranges, DIRECT (<256 fanout) vs SWWC+NT (>=256) column scatter with
ScatterScratch staging (~76 B/partition/worker; MAX_FANOUT_PER_PASS=8192 L2 ceiling), window
batching >= max(256Ki rows, fanout*64 lines) so each window writes >=4 KiB/partition/column,
multi-pass refine, and the fused single-dispatch barrier wave loop (streamingWaveProbe shape:
histogram / alloc / scatter / work-stealing consume, std::barrier transitions, persistent
per-worker scratch).

Pre-registered DEVIATIONS from the bench (each is a finding if it breaks equivalence):
- D1: partition ids (2 B) are computed ONCE during the histogram phase from caller-provided route
  words and stored for the scatter phase, instead of the bench's fused re-route inside the key
  column scatter. Rationale: production routes come from PackedKeyHash over packed (possibly
  multi-column, up to 64 B) keys — re-deriving per phase is costlier than a 2 B pid store/load;
  byte-traffic neutral for the bench's 8 B case. Bandwidth-shape impact is NOT claimed at U1;
  it is measured at U3 against the §7.5 priors.
- D2: routing is hash-agnostic — the kernels consume route words the caller computed (production:
  PackedKeyHash top bits so probe partitions align with leaf tables; tests: bench-style ISO-CRC32).
- D3: the key column is scattered as a "packed keys" pseudo-column via pids, like any payload.
- D4: element widths generalized from UInt64-only to {1,2,4,8,16,32,64} B via the same
  compile-time dispatch-table pattern the donor KeyRefScatter uses.
- D5 (productionization, per spec): DB::Exception instead of std::runtime_error/chassert for the
  2^32-rows-per-window and fanout<=2^16 preconditions; std::barrier phases wrapped so a throwing
  worker STILL arrives (first exception captured, stop flag set, rethrown after join);
  cancellation = caller-provided std::atomic<bool> (or stop token) checked at wave and batch
  boundaries; zero globals (no g_sink); all buffers PODArray/IColumn (MemoryTracker-visible via
  Allocator).

Pre-registered tests (gtest_column_scatter.cpp) and what each proves:
1. `ExactPartitionSizes` — fanout {8, 256, 1024}: per-partition row counts == histogram totals;
   sum == input rows; every partition column has exactly that many rows (exact-allocation, no-churn).
2. `FingerprintStableRouting` — same input, thread counts {1,4,16}: per-partition order-insensitive
   multiset fingerprints identical across thread counts.
3. `MultiPassEquivalence` — pass_bits {8} vs {4,4}: final 256 partitions multiset-equal.
4. `WidthDispatchRoundTrip` — payload widths {1,2,4,8,16,32,64} B: per-row tuples reassembled from
   partitions match the input exactly (route-consistent).
5. `ThrowingConsumerDoesNotDeadlock` — one worker's consumer callback throws mid-wave: the loop
   rethrows that exception within a bounded time; no hang (E3 acceptance).
6. `CancellationStopsWaves` — stop flag set after wave 1: loop exits promptly; partitions from
   completed waves intact; exception (or clean stop) surfaced.
7. `SwwcMatchesDirect` — fanout >= 256 on this aarch64 host: SWWC path output == DIRECT path output
   bit-for-bit (this ARM host RUNS the SWWC path in the bench kernels — unlike donor KeyRefScatter).
Expected: all green in reldeb + ASan. Any semantic divergence between ported kernels and bench
behavior on the same inputs = finding to investigate (cross-check available: run the bench binary).

## U2 — Skeleton + build side
(pending)

## U3 — Streaming budgeted probe
(pending)

## U4 — End-of-input drain
(pending)

## U5 — Correctness harness
(pending)

## U6 — Performance validation (mechanism + magnitude per shape)
(pending)

## U6b — AMAC vs amortization floor
Pre-registered hypothesis (from spec, restated verbatim so it cannot drift):
H = "AMAC prefetch lowers the parity threshold r* below the bench's 4–16K rows/leaf/visit,
so budgets below 5% stay profitable."
CONFIRM = sweep points below the bench parity budget still ≥1.0x with LLC-miss counters showing
hidden reloads; REFUTE = parity at the same rows/leaf/visit as the bench.
