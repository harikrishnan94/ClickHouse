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
Pre-registered 2026-07-09T19:10Z, before any U2 implementation. Decisions D-0003..D-0007 bind.

### Scope recap (per spec §8-U2, adapted by D-0003/4/5/6/7)
`RadixHashJoin : IJoin` in src/Interpreters/RadixHashJoin/ (ported from donor, adapted:
StoredColumnsIndex/StoredBlock migration, runPostBuildPhase hook, lazy group builds, lane-scratch
sizing robust to any lane index); JoinAlgorithm::RADIX_JOIN + "radix_join"; five radix_join_*
settings + history in 26.7 + JoinSettings/QueryPlanSerializationSettings/JoinAlgorithmParams
plumbing; IJoin lane overloads + using-decls in 9 implementors; JoiningTransform stream_index +
FillingRightJoinSideTransform build_lane + QueryPipelineBuilder lane assignment; planner gate
radixHashJoinApplicable + createRadixJoinFallback (adapted to pass stats_collecting_params) +
tryCreateJoin dispatch; HashTablesStatistics RadixHashJoinEntry; ProfileEvents (Build/Probe/
CollectMatches/PackHashRoute/LeafGroupBuilds µs+count) + 3 CurrentMetrics + ThreadName::RADIX_JOIN.
Probe = immediate per-block (D-0005). Old analyzer NOT taught radix_join (documented limitation;
tests pin enable_analyzer=1).

### Commit seams (each must build green)
C1 lane plumbing (IJoin + transforms + pipeline builder, inert alone) · C2 enum/settings/plumbing
(inert-ish: radix_join alone pre-C4 throws NOT_IMPLEMENTED — acceptable intermediate) ·
C3 RadixHashJoin class + events/metrics/stats + LeafTable group-build refactor + gtests ·
C4 planner gate/fallback/dispatch + stateless SQL tests.

### Pre-registered gate acceptance/rejection table (each row exercised by SQL test; rejected rows
must fall back and produce results equal to `hash`)
ACCEPT: k UInt64 · k UInt32 · (UInt64,UInt32) · k FixedString(16) · (UInt64×8)=64 B · k Date32
  (4 B) · k UUID (16 B).
REJECT (gate): k UInt8 (w=1, %4 fail) · k String (not fixed) · k LowCardinality(String) ·
  k Nullable(UInt64) · (UInt64,UInt8)=9 B (%4 fail) · k FixedString(68) (>64) · k FixedString(3)
  (w=3) · LEFT JOIN (kind) · INNER ANY (strictness) · OR-disjunct ON (oneDisjunct fail) ·
  join with StorageJoin (special storage).
Engagement proof: EXPLAIN (description) shows Algorithm: RadixHashJoin for accepted shapes and the
fallback name for rejected; system.query_log ProfileEvents[RadixHashJoinBuildMicroseconds] > 0 on
accepted shapes only.

### Pre-registered result-equality matrix (radix_join vs hash; sorted-result fingerprint =
(count(), sum(cityHash64(all cols)), and full sorted output on <=1e5 shapes))
Axes: key widths {4,8,16,32,64} B, single + multi-column mixes; duplicates {unique, x8, zipf-skew};
hit rates {1.0, 0.5, 0.05}; build rows {1e5, 1e7}: FULL CROSS at 1e5 and 1e7.
At 1e8 rows (memory/time bound): pre-registered SUBSET {8 B unique hit1.0, 8 B x8 hit0.05,
64 B unique hit0.5, (UInt64,UInt32) zipf hit0.5}. Plus: empty build side, empty probe side,
all-miss (hit 0), one-row build. TOTALS + extremes: one WITH TOTALS and one extremes=1 query
equal vs hash. Expected: byte-equal results everywhere; any mismatch = stop-and-diagnose finding.

### Lazy leaf build demonstration (D-0004; pre-registered evidence)
(1) empty-probe query: RadixHashJoinLeafGroupBuilds == 0 while build events > 0;
(2) full probe: LeafGroupBuilds == number of non-empty groups (gtest asserts exactly-once under
    16 concurrent probe threads x 100 blocks);
(3) U1 suite still green after the LeafTable per-group refactor (D-0002 revisit re-verification).

### Pre-registered risks/unknowns to resolve with evidence during U2
R-a: max_streams vs max_threads — can lanes exceed ctor max_threads on HEAD? (donor threw
  LOGICAL_ERROR). Resolve by reading pipeline code; implement lane storage safe for any index.
R-b: StoredBlock::replicated_columns — our normalized materialized right blocks must never carry
  them (chassert + test with replicated-prone input e.g. Sparse/Const columns).
R-c: header-time joinBlock (empty block, pre-build) must emit the exact output header
  transformHeader expects (donor handled; verify against HEAD's JoiningTransform.cpp:30-39).
R-d: JoinResult single-block emission with duplicate-heavy output (no max_block_size splitting,
  donor behavior) — acceptable for U2 (documented), revisit at U3 with max_joined_block_rows.
Expected outcome: all gates green; any semantic adaptation beyond the listed ones = logged finding.

## U3 — Streaming budgeted probe
Pre-registered 2026-07-09T20:50Z, before any U3 implementation. D-0012/D-0013/D-0014 bind.
Amendments in force: sanitizer gates waived (D-0009); review deferred post-U5 (D-0011) —
the spec's "TSan-clean" MUST-HOLD is replaced by reldeb gtest stress runs.

### Scope
Budget = clamp(radix_join_probe_buffer_fraction × build_accumulated_bytes,
radix_join_probe_buffer_min_bytes, radix_join_probe_buffer_max_bytes[0=∞]), where
build_accumulated_bytes = stored build block bytes + scattered record bytes + built leaf-table
bytes at post-build end. Buffered-probe path per D-0012 (accumulate per lane, scatter at
eviction via ColumnScatter, probe partitions on the join pool, stream output through the
triggering JoinResult per D-0013). Eligibility per D-0014 (fixed-width probe output columns;
else U2 immediate path). AMAC stays on the buffered path (partition probe = the U1 leaf-table
AMAC kernel). Lazy group builds on the pool use per-worker arenas (lazy_build_mutex removed on
that path). The U2-only double-hash pre-pass disappears on the buffered path (routes computed
once at scatter). New ProfileEvents: RadixHashJoinEvictions, RadixHashJoinProbeBufferedBytesPeak,
RadixHashJoinProbeScatterMicroseconds, RadixHashJoinEvictProbeMicroseconds,
RadixHashJoinBufferedProbeBlocks (names may be refined; semantics fixed here).

### Pre-registered MUST-HOLD acceptance
1. Result equality vs hash on the FULL U2 matrix (bep/tools/u2_equality_matrix.py, all tiers)
   with budget forced tiny: fraction=0, min_bytes=1 → eviction on ~every block (hundreds+).
   Also at fraction=0, min_bytes=64MiB (mid), and defaults. Expected: byte-equal everywhere.
2. Memory bound: peak buffered probe bytes ≤ budget + one scatter window + one block, asserted
   via the peak ProfileEvent in a dedicated SQL test (tiny + mid budgets) and in the gtest.
3. Deadlock negative-test (gtest, IJoin level with fabricated TableJoin — port the donor bench's
   makeTableJoin helper): N lanes feed blocks; lane X stops feeding entirely mid-stream while
   others trip evictions repeatedly; the join must complete all evictions and the final result
   (after U4 lands, drain; within U3: all evicted output correct + no hang for 30s watchdog).
4. Concurrency stress (TSan replacement per D-0009): the same gtest at 16 lanes × hundreds of
   blocks × tiny budget, --gtest_repeat=20 in reldeb; zero failures/hangs.
5. No-harm: U2 gate/fallback tests 04508/04509 still green; immediate-path (non-buffered
   eligibility) queries still equal hash.

### Pre-registered GOAL (streaming demonstrated)
6. IJoin-level gtest: with tiny budget, some joinBlock call k returns a non-empty JoinResult
   BEFORE block k+1 is fed (first output precedes probe-input exhaustion). Plus SQL-level:
   RadixHashJoinEvictions ≥ 1 with correct results on a budget-tripping query.
7. R-d regression fix check: the U2-observed ~1 GB single-block emission shape (dup-heavy 20M
   output rows) emits in ≤ max(block size)-bounded chunks on the eviction path; peak tracked
   memory for that query drops to the same order as hash (measure before/after).

### Pre-registered risks
R-e: executor-thread condvar wait (contract step 4) interacting with executor scheduling —
     watch for stalls in the stress gtest.
R-f: output-queue backpressure vs cancellation — abort path must unblock workers (gtest kills
     a query mid-eviction... covered in U4 cancellation tests; U3 gtest covers dtor-mid-eviction).
R-g: budget floor default (512 MiB) means small joins never evict — streaming tests must force
     tiny budgets explicitly (fraction=0 semantics: 0 disables the fraction term, min_bytes rules).

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
