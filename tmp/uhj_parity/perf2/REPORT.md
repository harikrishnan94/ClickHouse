# Per-row-loop and locking cost analysis: `unified_hash` vs `hash` vs `parallel_hash`

Host: aarch64 **Neoverse-V2** (Graviton4), 96 CPUs, 1 NUMA node, L2 2 MiB/core, L3 36 MiB.
Reference binary `tmp/uhj_parity/perf2/bin/clickhouse.ref` (RelWithDebInfo).
Baselines pristine at delivery (G5.1 green). Nothing pushed; no PR.

---

## 1. Per-unit verdict

| Unit | Verdict | Notes |
| --- | --- | --- |
| **0 — enumeration + completeness** | **GREEN** | 45 loops enumerated; G0.1 green with **zero** unexplained symbols over 550k in-join samples across 132 profiled runs; G0.2 green, static and dynamic lock sets agree |
| **1 — codegen per loop** | **PARTIAL** | 31 of 45 loops carry codegen evidence (2 proved zero-delta by identical-code folding, 6 by a single shared source file, 8 by full disassembly + `llvm-mca`, 15 have no counterpart). **14 loops are covered by a cross-tree byte comparison but have no `llvm-mca` cycle estimate**, so under G3.3 they may not carry a POSITIVE or NEGATIVE verdict and are UNSETTLED |
| **2 — locks** | **GREEN** | every lock's count verified against its formula at 1/16/64 threads (ratio 1.000), every hold time **measured** as a distribution |
| **3 — verdicts** | **PARTIAL** | two MEASURED verdicts (locking; the one-thread deficit's mechanism), one refuted ablation, several BOUNDED, the composite-key magnitude UNSETTLED with a named experiment |
| **4 — necessity** | **PARTIAL** | 4 rows with falsification attempts actually run; 2 rows UNSETTLED with the attempt specified |

**No authorization-required flags. No risk-accepted blocking leads.**

### HIGH-IMPACT assumptions

1. **The benchmark matrix does not exercise long `RowRefList` chains** (2 rows per key) or
   `LowCardinality`/`key8`/`key16` keys. Inherited from the prior mission's harness and
   not fixed here. Every verdict is scoped to what the matrix covers.
2. **The non-joined scan is under-exercised at small and medium cardinality** — the
   match-rate knob controls the fraction of matching *probe* rows, not the fraction of
   unmatched *build* keys. The mission brief asked for this to be fixed; **it was not**,
   for time. Consequently loops N1-N7 carry no new measured verdict here beyond the
   prior mission's closed A7/N1 result.
3. Measurements are single-tenant on this host. The prior mission recorded that a second
   concurrent agent invalidates all timings; nothing else was running during timed runs,
   but this is asserted, not enforced.

---

## 2. Headline results

**Three findings, in order of value.**

**(i) `unified_hash`'s build-side locking is decisively better than `parallel_hash`'s, and
it is measured, not inferred.** At 64 threads on a 30M-row build, `unified_hash` takes
twice as many locks, holds each about four times more briefly, spends 26% less total time
inside critical sections, and fails a `try_lock` **0.70 times per successful acquisition
against `parallel_hash`'s 309**. `parallel_hash` burns roughly **11 million** failed
`try_lock` attempts where `unified_hash` burns fifty thousand. The mechanism is structural
and identified: `unified_hash` rotates its scan start per block and blocks when a pass
makes no progress; `parallel_hash` always scans from slot 0 and `yield()`s straight back
into another failing pass.

**(ii) A large per-partition cost exists on the composite-key build path, in code all three
share, and it is the mechanism behind the known 64-thread composite-key build-CPU deficit.**
`HashMethodKeysFixed`'s *constructor* packs the whole block's keys, sized by
`column->size()` and **not** by the partition's selector. `unified_hash` constructs one per
bucket (K = 2·bit_ceil(threads), so 129 per block at 64 threads), `parallel_hash` one per
slot, `hash` one per block. The profiler independently puts **67%** of `unified_hash`'s
in-join samples in that packing at 64 threads with composite keys, against **0%** for the
single-UInt64 key which packs nothing.

**(iii) The one-thread deficit is an instruction-count cause, and the standing
cache-footprint candidate is refuted.** On `FULL|u64|hi|t1|medium` — a used-flag kind,
where the deficit actually lives — hardware counters reproduce it at **+8.1% cycles per
probe row** and account for it completely: **+6.7% instructions at −1.3% IPC**
(1.067 / 0.987 = 1.081), with cache misses +1.4%, **dTLB misses −1.2%** and branch misses
−1.2%, all inside the run-to-run spread. Candidate **A5** — `per_offset_flags` being
wider and sparser on the partitioned map — required IPC to fall by more than 10% and is
**REFUTED**. The codegen story stands: the `Prober` does not stay in registers.

**(iv) The obvious lever for (ii) does not work, and knowing that is worth as much as the
finding.** A pre-registered ablation halving the bucket count predicted −34% and measured
**+9% slower**, because K also supplies the slack the `try_lock` drain loop needs. At
`BUCKETS_PER_THREAD = 1` there are exactly as many buckets as threads and contention wins:
the `u64` arm, where there is no packing to save, is **+90%** at 64 threads. `K` is not the
lever; the key-getter hoist is, and it is untested.

---

## 3. (A) The loop table

45 loops. Multiplicity symbols: `R` probe rows, `R'` inserted build rows, `B` blocks,
`C` ON-clauses, `K` unified buckets = 1 if threads ≤ 1 else 2·bit_ceil(threads),
`S` parallel_hash slots = min(threads, 256), `M` matched rows, `O` output rows,
`T` threads, `P` partitions.

Codegen-evidence tiers (`codegen/tiers.json`, generated by `tier.py`):
**T0** the linker folded both trees' symbols onto one address — delta exactly zero;
**T1** one shared source file, no per-tree copy — delta zero by construction;
**T2** per-tree copies, emitted symbols byte-compared;
**T3** genuinely different, full counts + `llvm-mca`;
**T4** no counterpart.

### 3.1 Probe path — where the one-thread deficit lives

| loop | sub-phase | multiplicity | present in | codegen delta | mca | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **P1** probe main per-row loop | probe lookup | `R` | all 3 | **T3.** base 123 insns / 29 ld / 5 st / 28 br / 0 spill-st / 2 spill-ld / dep-load depth **3**; unified 172 / 44 / 7 / 36 / **2 spill-st / 9 spill-ld** / depth **5**. Hot path only (matched row, first-probe hit), prefetch on: base 59, unified 85 | base **10.41** cyc/iter, IPC 5.57, RThr 10.2; unified **15.79** cyc/iter, IPC 5.32, RThr 15.2. Both dispatch-width-limited (RThroughput = uops/6 exactly), most-pressed port `V2UnitL01`. **Lower bound**: two calls dropped identically on both sides | **NEGATIVE at 1 thread on used-flag kinds, MEASURED: +8.1% cycles/row** on `FULL|u64|hi|t1|medium`, of which +6.7% is instructions at −1.3% IPC (G3.2, §9). mca predicted +5.4 cyc/row on the loop body; the whole query moved +17.0 cyc/row, so **mca's loop-body figure is 32% of the query-level delta** — consistent, since the loop is part of the query. NEUTRAL on `INNER` (+2.1% cycles, +0.5% instructions — measured, and matching the map's +0.64% median) |
| **P2** adaptive lookahead prefetch | probe lookup | `R` | all 3 | T3, fused into P1. Prefetch adds 14 insns base / 20 unified | included above | UNSETTLED — separable only by the `enable_software_prefetch_in_join` setting, not run |
| **P3** key extract + hash + find | probe lookup | `R` | all 3 | **T3, the divergence point.** Baseline reads `mask`/`buf` from `[x22,#0x48]`/`[x22,#0x20]` off a register live across the whole loop. Unified reloads `shift`/`max_bucket` (`ldp [sp,#0x10]`) and `prefix` (`ldr [sp]`) **per row**, stores `routed_prefix` (`str [sp,#0x28]`), and computes `add x21, x28, x10, lsl #7` so the sub-table pointer **depends on the hash** | this is the 3→5 dependent-load-depth change; see P1 | **NEGATIVE at 1 thread on used-flag kinds, MEASURED via P1** — and the counters say *this* mechanism rather than cache footprint: instructions moved, memory did not |
| **P4** per-matched-row `offsetInternal` | probe match | `M` | all 3 | **T3.** `Prober::offsetInternal` is 8 insns and 3 loads (one a spill reload) against the baseline's single `add x8, x9, #1`, **and is paid even on the `sole != nullptr` fast path** | +29% on the `sole` path | **NEGATIVE at 1 thread on used-flag kinds, BOUNDED** (not separately ablated). This answers the inherited open question: the `sole` short-circuit is **not** free |
| **P5** per-matched-row `setUsed` | probe match | `M` | all 3 | T3; relaxed load + relaxed store, identical logic both trees | — | **NEUTRAL**, detection bound: the counters would have shown it as raised cache misses on the flag array, and cache-misses/row moved **+1.4%** with dTLB **−1.2%** on the kind that exercises flags most (`FULL`). An effect larger than ~1.5% of memory traffic on the flag array would have been detected |
| **P6** `addFoundRowAll` list walk | probe match | `M` | all 3 | **T2. 196 insns each side, byte-identical after normalising the self-referential branch target** | n/a — identical | **NEUTRAL**, delta exactly 0 |
| **P7** `appendFromBlock` | probe match | `M`(×cols) | all 3 | **T0 ICF — one address holds both names** | n/a | **NEUTRAL**, delta exactly 0 |
| **P0/P8** per-block probe setup, `joinBlockImpl` | probe lookup | `B·C` | all 3 | **T2.** 687 insns each side; 44 aligned differences, all struct field offsets, **plus 2 `ldar` where the baseline uses `ldr`** — two fields are atomic in unified | not modelled | **UNSETTLED** — 2 acquire loads per block is almost certainly negligible but has no bound |
| **P9** additional-filter loops | probe match | `R + ΣM` | all 3 | T5 | — | **UNSETTLED** — not exercised by this matrix (no residual filter) |
| **P10** per-block `Prober` construction | probe lookup | `B·C` | unified only | **T4 no counterpart** — the baseline uses the map directly | — | **NEUTRAL** per block; its per-row consequence is P3 |

### 3.2 Result gather — the largest sample share, and provably identical

| loop | multiplicity | present in | codegen delta | mca | verdict |
| --- | --- | --- | --- | --- | --- |
| **G1** partial replicate offsets | `R'` | all 3 | T3, identical | — | **NEUTRAL**, delta 0 |
| **G2** `buildOutputFromBlocks` per output row | `O` | all 3 | **T3. 397 instructions each side, ZERO positional mismatches.** Symbol sizes equal on both sides (`0x634`); `generateBlock` `0x1cf4` on both | **6.09** cyc/iter, IPC 3.78, RThr 5.2, identical both sides (22 insns, 5 ld, 4 st, 7 br, dep-load depth 2) | **NEUTRAL**, delta exactly 0. This is the **hottest loop in the join** (43-49% of in-join samples in all three) and it is identical |
| **G3** per-column gather | `C·O` | all 3 | T3, identical | as G2 | **NEUTRAL**, delta 0 |
| **G4** lazy-default fill | `C·gaps` | all 3 | **T0 ICF** | n/a | **NEUTRAL**, delta exactly 0 |
| **G5** `filterBySelector` left-block materialisation | `O·C_left` | all 3 | **T1** — `find src -name ScatteredBlock.h` returns exactly one path, no Unified copy | n/a | **NEUTRAL**, delta 0 by construction. **This loop was missing from the enumeration until G0.1 caught it at 0.48% of samples** |

### 3.3 Build path

| loop | multiplicity | present in | codegen delta | verdict |
| --- | --- | --- | --- | --- |
| **B22** whole-block fixed-key packing in the key-getter constructor | **`R·B·P`** — per row **per partition**; `P`=1 / `S` / `K` | all 3, at wildly different multiplicities | **T3.** The two specialisations (`need_offset` true/false) are instruction-for-instruction identical, 107 insns, 644 B each. `fillFixedBatch` inner loop: 4 insns, **1.56 cyc/iter**, IPC 3.21, RThr 1.3, limit `V2UnitL01`. **The delta is not codegen — it is call count** | **NEGATIVE at 16/64 threads with fixed-width composite keys; NEUTRAL at 1 thread** (all implementations then have 1-2 partitions and the profiler shares agree: 43.1 / 41.3 / 37.3%). Magnitude **UNSETTLED** — see §6 |
| **B7/B8/B10** per-row bucket scatter (hash+count, place, reserve) | `R·B·C` | unified only | **T4 no counterpart.** `parallel_hash` scatters by copying or by selector into per-slot blocks (B9); `hash` has no scatter | **POSITIVE vs `parallel_hash`, BOUNDED**: unified scatters row indices only and never copies columns |
| **B11/B12** per-row emplace + prefetch | `R'·B·C` | all 3 | **T2/T3.** base(flat) 311, base(two-level) 325, unified 353 insns. Unified's extra is a **one-time** `bucket` range check plus a run-time bucket count; **the per-row loop is 3 instructions SHORTER** because the sub-table is hoisted out of it | **POSITIVE per row, NEUTRAL overall, BOUNDED** — the extra is per call, the saving per row |
| **B13** hash-table probe chain | `R'·B·chain` | all 3 | **T1** `HashTable.h`, one shared file | **NEUTRAL**, delta 0 |
| **B14** `RowRefList` append + arena | `(R'−distinct)·B` | all 3 | **T1** `RowRefs.h`, one shared file. Unified uses a per-bucket arena, the others one arena per (slot's) join — a *layout* difference, not codegen | **UNSETTLED** — layout not measured |
| **B15** per-block per-bucket `try_lock` drain | `≤K·B·C` | unified only | **T4** — see lock table L1 | **POSITIVE at 16/64, MEASURED** |
| **B16** per-bucket byte-delta accounting | `≤K·B·C` | unified only | **T4**, inside the critical section | **NEGATIVE, CONTINGENT** — see §5 |
| **B17** per-slot `try_lock` drain | `≤S·B` | parallel only | **T4** — see lock table L2 | **NEGATIVE for `parallel_hash`, MEASURED** |
| **B18** `blocks_mutex` registration | `B` | unified only | **T4** — see lock table L3 | **NEGATIVE at 64 threads, CONTINGENT** — see §5 |
| **B19** post-build bucket prefix pass | `K`, once | unified only | **T4** | **NEUTRAL** — O(buckets), not O(rows); never sampled |
| **B20** slot→slot0 bucket merge | `(K/S)(S−1)`, once | parallel only | **T4** — unified shares one map from the start | **POSITIVE for unified, BOUNDED** |
| **B21** owned-bucket pre-reserve | `K/S` per slot, once | parallel only | **T4** | **NEUTRAL**, never sampled |
| **B0/B1** right-block materialisation, nullable unfold | `D·B` / `R·B` | all 3 | **T1**, `JoinUtils.cpp` / `NullableUtils.cpp`, one shared file each | **NEUTRAL**, delta 0 |
| **B2/B3/B4** null-in-selector scan, `not_joined_map` mark, used-flag reinit | `R·B` | hash, unified | **T2.** Entry overload `addBlockToJoin(Block const&, bool)` **byte-identical, 142 insns**. The `Selector` overload is **not comparable**: the baseline inlines a 40-arm dispatch (3456 insns), unified outlines it into 19 helpers (2086 + 19694 insns) and takes a mutex | **UNSETTLED** — the outlining difference is real and unmeasured |
| **B5/B6/B9** per-row routing hash, hash→slot, scatter | `R·B` (×cols on the copy path) | parallel only | **T4** | **NEGATIVE for `parallel_hash`, BOUNDED** |

### 3.4 Non-joined scan

| loop | multiplicity | present in | codegen delta | verdict |
| --- | --- | --- | --- | --- |
| **N1/N3/N4/N7** per-cell scan, per-row `isUsed`, bucket skip, column fill | `Cells/T`, `R_build`, `K`, `C·O_nj` | all 3 | **T2.** `fillColumns` 660 insns base / 634 unified. Real difference: **a different bucket walker** — baseline `const_iterator::operator++` (1 call), unified `beginOfNextNonEmptyBucket` (4 calls) | **UNSETTLED here.** The prior mission's closed A7/N1 result stands (−9-11% of the non-joined phase at high match rate); this mission adds no new measurement because the matrix under-exercises the path (assumption 2) |
| **N2** flat single-level cell scan | `Cells` | hash only | **T4** | **NEUTRAL** |
| **N5** `RowRefList` collect | unmatched rows | all 3 | **T1** `RowRefs.h` | **NEUTRAL**, delta 0 |
| **N6** null-key row scan | null rows | all 3 | **T2.** 394 insns each, **6 differing positions, all `HashJoin` member offsets** | **NEUTRAL**, bound: 6 field offsets |

---

## 4. (B) The lock and atomic table

All counts and hold times from the **instrumented** binary `bin/clickhouse.instr`
(commit `79a9eee2619`, reverted). No timing verdict uses that binary.
Raw: `locks/gate_l1.txt`, `locks/locks_l1.jsonl`.

| lock | guards | impl | acquisitions: formula → measured | granularity | length (measured) | contention (measured) | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **L1** `BucketLock[b]` | bucket `b` of every clause's map **plus** bucket `b`'s arena | unified | `K·B·C` → **exact, ratio 1.000 in all 12 cells** (t1 18/18; t16 18752 = 32×586; t64 72064 = 128×563; t64 comp 97152 = 128×759) | whole per-bucket insert, arena growth, `BuildResult` merge, **plus** the byte-delta accounting that need not be there (B16) | t64 u64: **p50 11.6 µs, p99 46.3 µs**, mean 18.1, total 1.30 s. t64 comp: p50 46.3 µs, total 5.75 s. t1: 1.48 ms (uncontended — this is the whole block insert, not a cost) | **0.70 `try_lock` failures per acquisition**; 614 blocking waits of 72,064 (0.85%), 0.23 s total wait | **POSITIVE at 16/64 vs L2, MEASURED.** NEUTRAL at 1 thread (K=1, always acquired) |
| **L2** per-slot `mutex` | the slot's entire inner `HashJoin` | parallel | `B·S` → **exact, ratio 1.000** (t16 9360; t64 35840; t64 comp 48576) | one-time map preallocation, the **whole** `HashJoin::addBlockToJoin`, and a relaxed global-counter update | t64 u64: **p50 46.3 µs, p99 92.7 µs**, mean 49.2, total 1.76 s | **309 `try_lock` failures per acquisition** (11,074,336 vs 35,840). t16: 204 per acquisition | **NEGATIVE at 16/64, MEASURED** |
| **L3** `blocks_mutex` | `data->columns`, nullmaps, `allocated_size`, index registration | unified | `B` (+0-3/block) → **exact** (t64 563, t16 586) | list append, `StoredColumnsIndex::add`, byte accounting, **`assertBlocksHaveEqualStructureAllowReplicated`, `doDebugAsserts()`, `getCurrentQueryMemoryUsage()`**. Bucket inserts run **outside** it | **p50 2.90 µs**, p99 5.79 µs, total 1.83 ms at 64 threads | not separately instrumented | **NEGATIVE at 64 threads, CONTINGENT.** The part that needs the lock (`SCI_ADD`) takes **32 ns**; ~99% of the 2.9 µs is work that does not |
| **L4** `StoredColumnsIndex::mutex` | `blocks[]`, `blocks_generation`, lazy emit table | **all three** | build 1/block; probe 1/batch → exact | `add`: a `push_back` and a generation bump. `resolveEmitColumns`: optional full emit-table rebuild | `add` **p50 11-45 ns**; `resolve` **p50 91-181 ns** | none observed | **NEUTRAL** — shared infrastructure, not a source of asymmetry |
| **L5** `hash` build lock | — | hash | **zero, confirmed dynamically**: only L4 fires | — | — | **POSITIVE for `hash` at 1 thread**; irrelevant above, since `hash` does not build in parallel |
| **L6** `BucketLock[0]` empty-block case | bucket 0 | unified | `0` predicted, **0 measured** | — | — | **Unreachable in this workload.** Two attempts to fire it failed. LEAD: possibly unreachable through the pipeline entirely |
| **A1** `setUsed` flag store | `per_offset_flags` / `per_row_flags` | all 3 | `M`, **relaxed** load + relaxed store | — | not a lock | idempotent; no RMW | **NEUTRAL** — identical in both trees |
| **A2** `setUsedOnce` CAS | same flags | all 3 | ≤1 successful CAS per right key; relaxed pre-check, **`compare_exchange_strong` seq_cst** | — | — | — | **NEUTRAL** — identical in both trees. LEAD: seq_cst may be stronger than needed |
| **A3** `bucket_bytes.fetch_add` | running byte sum | unified | `K·B·C`, **relaxed** | inside L1 | — | — | **NEGATIVE (small), CONTINGENT** — see §5 |
| **A4** `global_total_rows/bytes` | parallel_hash totals | parallel | 2 per slot insert, **relaxed** | inside L2 | — | — | **NEUTRAL** |
| **A5** `std::call_once` in `BucketPrefixSums::offset` | prefix array | hash probe; unified avoids it | once per map | — | — | — | **POSITIVE for unified** — `freezeMapsForProbing` precomputes; the baseline pays a `call_once` check |

**All three probe paths take zero join-level mutexes.** That is measured, not assumed.

---

## 5. (C) The necessity analysis

| difference | invariant (checkable) | what forces it | falsification attempt — **run** | verdict |
| --- | --- | --- | --- | --- |
| **L1 exists at all** (per-bucket lock) | Two build threads may route rows to the same bucket in the same pass, and `HashMapTable::emplace` is not safe against concurrent mutation of the same sub-table | correctness, under the one-algorithm-for-both-regimes goal | **Not run.** Removing it needs the concurrent-emplace race to be *demonstrated*, which needs a stress harness this mission did not build | **UNSETTLED, not FUNDAMENTAL.** Naming what would settle it: delete the `unique_lock` in `insertIntoBuckets`, run `04658`/`04659` at 64 threads under TSan (`build/tsan` already exists) |
| **K = 2·bit_ceil(T)** (twice as many buckets as threads) | The `try_lock` drain loop needs more buckets than threads, or a thread that finds its bucket busy has nowhere else to go in the pass | throughput, not correctness | **RUN — ablation A-K1**, `BUCKETS_PER_THREAD` 2→1, pre-registered PREREG P3.1, validity-checked (`codegen/ak1_validity.txt`: 0 symbols changed outside `DB::Unified::`, ablation target confirmed changed, `fillFixedBatch` and both baselines opcode-identical). **Nothing broke functionally; it got slower**: comp t64 **+9.0%**, u64 t64 **+90.1%**, controls flat | **FUNDAMENTAL-for-throughput.** The invariant is real and the violation is demonstrably worse. The constant 2 is justified, not arbitrary |
| **B22** whole-block key packing once per partition | *Claimed*: each partition's insert needs a key getter over the block's key columns | **nothing forces the packing to be per-partition.** `prepared_keys` is read-only after construction, identical for every partition of a block, and sized by the whole block regardless of selector | **Not run** — hoisting needs `insertFromBlockImpl`'s signature changed to accept a shared key getter | **CONTINGENT.** Concrete alternative: construct the key getter once per block before the bucket loop and pass it by const reference. Cost: a signature change through `insertFromBlockImpl`; the object is already immutable after construction, so no synchronisation is needed. **This is the top item in the next-mission handoff** |
| **L3 holds `assertBlocksHaveEqualStructureAllowReplicated` + `doDebugAsserts` + `getCurrentQueryMemoryUsage`** | *Claimed*: block registration must be atomic against other build threads | only the `data->columns` append and the `StoredColumnsIndex::add` need the lock. The structure assertion reads `data->sample_block`, which is immutable after construction; the memory-usage read is a process-wide query | **Not run** — but the measurement already localises it: the part that needs the lock is **32 ns** of a **2.90 µs** critical section | **CONTINGENT — a real invariant, over-served.** Exactly the shape the mission said to hunt for. Falsification that would settle it: move the three calls above the `lock_guard` and re-measure L3's hold-time distribution |
| **B16/A3** byte accounting inside L1 | *Claimed*: the growth an insert caused must be attributed exactly once, so it is measured under the bucket's own lock (the code says so at `HashJoin.cpp:137-138`) | accuracy of a memory-accounting number, not correctness of the join | **Not run.** Moving it out races the accounting, not the data | **CONTINGENT.** The invariant is about an accounting number's exactness. Alternative: accumulate per-thread and reduce at build finish |
| **P4** `offsetInternal` on the `sole` path | *Claimed*: the probe needs a global cell offset to index `per_offset_flags` | `need_flags` kinds only — unified already templates this away for INNER/LEFT, which is a genuine advantage over the baseline's hardcoded `use_offset = true` | **Not run.** The codegen artefact shows it costs 8 instructions and 3 loads even when `num_buckets == 1`, where the answer is trivially the flat offset | **CONTINGENT.** When `sole != nullptr` the bucket prefix is 0 and the offset is the flat one; a `sole`-specialised `offsetInternal` would be a pointer subtraction, as the baseline's is |

---

## 6. REMOVABLE summary

**No difference tested in this mission turned out to be REMOVABLE.** One was tested by
violating its invariant (K = 2·bit_ceil(T)) and the violation made things demonstrably
worse, so that one is genuinely load-bearing.

**Three differences are CONTINGENT with a concrete alternative named and no falsification
run** — these are the honest near-misses, and they are where the next mission should look:

1. **B22, the per-partition key packing.** Nothing in the code requires the packing to
   happen once per partition. The strongest single opportunity found.
2. **L3's over-served critical section.** 32 ns of a 2.90 µs hold actually needs the lock.
3. **P4's `offsetInternal` on the `sole` path**, where the computed value is provably the
   flat offset.

---

## 7. Refuted hypotheses

| hypothesis | refuted by | what it turned out to be |
| --- | --- | --- |
| The result-gather loop is inlined into `generateBlock` in the baseline and left out of line in unified, on the hottest loop in the join | `codegen/P1_G2_probe_and_gather.md`: both symbols exist on both sides at `0x634`, `generateBlock` is `0x1cf4` on both, and `buildOutputFromBlocks<true>` is 397 instructions with **zero** positional mismatches | **A defect in my own profiler attribution.** `IN_JOIN_MARKERS` had a blanket `DB::Unified::` but no `DB::LazyOutput::`, so baseline samples rolled up one frame to the caller. Fixed; a symmetry audit then found a second one (`DB::processMatch`). Re-run confirmed the registered prediction: all three trees now at 44.3 / 43.3 / 49.0% on the same symbol |
| `hash` inlines the composite key getter and the others do not | `codegen/K1_composite_keygetter.md`: `hash` and `parallel_hash` call the **same** out-of-line symbol at `0x13ab1280`, and `hash` has samples in it | The difference is **call count**, not codegen. That led to B22 |
| Halving the bucket count recovers ~34% of the composite-key build cost at 64 threads | Ablation A-K1, pre-registered and validity-checked: measured **+9.0%**, model/measured ratio **−3.79** (wrong sign) | K also supplies the drain loop's slack; contention dominates. The `u64` control arm shows it cleanly at **+90.1%** |
| The prior mission's symbol-level algorithm assertion is sound | `algoassert.py recheck --tag u0a`: **9 of 132 runs mislabelled** by the inherited rule, 0 by the corrected one | Identical-code folding puts `DB::HashJoin::canRemoveColumnsFromLeftBlock` and its `DB::Unified::` twin at address `0x14289180`; `addressToSymbol` picks one arbitrarily |

---

## 8. Per-regime unexplained residual

| regime | comparator | deficit (post-fix map) | explained here | residual | BOUNDED candidates against it (listed, **not** folded in) |
| --- | --- | --- | --- | --- | --- |
| **1 thread, used-flag kinds** | `hash` | 10 of 72 cells slower, mean **+6.54%** wall | **mechanism MEASURED.** +8.1% cycles/row on `FULL|u64|hi|t1|medium`, of which +6.7% instructions at −1.3% IPC; memory counters flat. **A5 cache-footprint REFUTED** | the *split within* the +6.7% across P1/P3/P4/B2-B4 | P3 `Prober` register spill (dep-load depth 3→5, 2 spill-st / 9 spill-ld); P4 `offsetInternal` not free on the `sole` path (+29%); B2/B3/B4 dispatch outlining, unmeasured |
| **1 thread, INNER** | `hash` | +0.64% median | **MEASURED as no deficit**: +2.1% cycles/row, +0.5% instructions/row | none | — |
| **16 threads** | `parallel_hash` | **0 cells slower**, 14 faster | L1-vs-L2 locking MEASURED as a `unified_hash` advantage; B20 merge absent | n/a — no deficit to explain | — |
| **64 threads, wall** | `parallel_hash` | **0 slower, all 24 faster** | as above | n/a | — |
| **64 threads, CPU** | `parallel_hash` | 5 cells slower, **+5.9%..+18.5%**, composite keys, build CPU +21%..+38% | **mechanism identified and corroborated by three origins that fail differently** (code reading of the sizing expression, K-scaling regression, profiler share 67% vs 0%) — but the **magnitude is UNSETTLED** because the only lever tested moves contention too | magnitude | B22 key-getter hoist (untested, the named experiment); B16 byte accounting inside L1 |

---

## 9. Evidence matrix

| Criterion | Gate invocation (copy-paste re-runnable) | Result (raw) | Non-gate sources (origins) | Verdict |
| --- | --- | --- | --- | --- |
| G0.1 loop enumeration complete | `python3 tmp/uhj_parity/perf2/enumerate.py gate --tag u0c` | 243 syms mapped / 536,954 samples (98.53%), 71 excluded with reasons (1.47%), **0 unexplained** | 132 profiled runs, 3 algorithms × 4 thread/card × 3 key families | **GREEN** |
| G0.1 gate has power to fail | `python3 tmp/uhj_parity/perf2/enumerate.py power --tag u0c` | 3/3 injected controls unexplained; exclusion budget 1.47%; per-loop knockout, B22 alone 7.55% | — | **GREEN** |
| G0.1b algorithm identity sound | `python3 tmp/uhj_parity/perf2/algoassert.py recheck --tag u0a` | inherited rule **9** mismatches, corrected rule **0** | `llvm-nm` shows one address, two names | **GREEN** |
| G0.2 lock set complete | `python3 tmp/uhj_parity/perf2/lockmeas.py gate --tag l1` | sets agree for all three; one corrected prediction (`UNI_BUCKET_EMPTY` unreachable, asserted to stay 0) | static grep + reading | **GREEN** |
| G2.1 lock counts match formula | same invocation | ratio **1.000** in all 12 unified cells and all 12 parallel cells, at 1/16/64 threads | `B` measured by an independent site | **GREEN** |
| G2.2 hold times measured | same invocation | p50/p99/mean/total per site per cell from a log2 histogram | `cntvct_el0`, thread-local | **GREEN** |
| G1.1 codegen artefact per loop | `python3 tmp/uhj_parity/perf2/tier.py` | T0=2 T1=6 T2=1 T3=8 T4=15 **T5=14** | `codegen/{P1_G2,K1,X1}*.md`, `icf_census.json` | **PARTIAL** — 14 loops have a cross-tree byte comparison (`X1_crosstree.md`) but no mca |
| G3.3 mca per loop | in `codegen/P1_G2_probe_and_gather.md`, `codegen/K1_composite_keygetter.md` | P1 base 10.41 / uni 15.79 cyc/iter; G2 6.09 both; `fillFixedBatch` 1.56 | `llvm-mca -mcpu=neoverse-v2 -bottleneck-analysis` | **PARTIAL** — present for 9 loops; the 14 T5 loops are UNSETTLED as G3.3 requires |
| G3.1 no verdict without its tier | §3-§5 tables; every % traces to §9 | one MEASURED % (A-K1), one MEASURED lock comparison, rest BOUNDED/UNSETTLED | — | **GREEN** |
| G3.2 counters discriminate | `KIND=FULL TAG=full REPS=7 bash tmp/uhj_parity/perf2/perfstat.sh` then the scorer in `results/g32_full.txt` | cycles/row **+8.1%**, instructions/row **+6.7%**, IPC **−1.3%**, cache-misses **+1.4%**, dTLB **−1.2%**, branch-misses **−1.2%**; hash sd 0.3-2.9% | codegen artefact `P1_G2_probe_and_gather.md` + `llvm-mca`; two unsound routes rejected first (§10) | **GREEN** — instruction-count cause; **A5 cache-footprint candidate REFUTED** |
| A-K1 pre-registered + validity-checked | `python3 tmp/uhj_parity/perf2/symdiff.py --before .../clickhouse.ref --after .../clickhouse.ak1 --expect-changed-regex 'DB::Unified::' --expect-differ 'DB::Unified::HashJoin::HashJoin\(std' --byte-compare 'fillFixedBatch<unsigned long' --byte-compare 'DB::ConcurrentHashJoin::addBlockToJoin' --byte-compare 'DB::HashJoin::addBlockToJoin'` | added 0 / removed 0 / resized 0 outside `DB::Unified::`; target DIFFERS as required; baselines opcode-identical | PREREG P3.1 predates commit | **GREEN** |
| A-K1 result | `python3 tmp/uhj_parity/perf2/kscale.py fit --tag k1` then the scorer in `results/ak1_scored.txt` | predicted −34.1%, measured **+9.0%**; controls −1.6/−0.8/+1.7/+0.8% | 7 reps, interleaved | **GREEN (prediction refuted)** |
| G5.1 baselines pristine | `git diff 0945a745399 -- src/Interpreters/HashJoin/ src/Interpreters/ConcurrentHashJoin.cpp src/Interpreters/ConcurrentHashJoin.h` | empty | — | **GREEN** |
| G5.3 shared-header edits inert | `symdiff.py` as above; plus the reverted build reproduces BuildID `7de0c7e8…` | 0 baseline symbols perturbed | ICF de-folding handled explicitly | **GREEN** |
| G5.2 regression suites | `bash tests/queries/0_stateless/04658_unified_hash_join_equivalence.sh` (via server on 9111); `UHJ_PORT=9111 bash tmp/uhj_parity/run_04659.sh` | 04658 **matches reference**; 04659 **OK** | — | **GREEN** |

---

## 10. Gates not met, stated plainly

- **G3.2 — GREEN, but with two rejected routes and one miscalibrated threshold, all
  disclosed.** Two measurement routes are unsound on this host and produced
  plausible-looking nonsense before being caught:
  `perf stat -p <server pid>` follows only threads existing at attach time and
  **undercounted by ~2000x** (178,633 cycles for a query executing ~385 million), because
  ClickHouse spawns per-query threads; and ClickHouse's own
  `metrics_perf_events_enabled` **UInt64-underflows** its per-thread deltas, leaving 2 of
  15 reps usable. The sound route is `perf stat` *launching* the process, which follows
  all its threads; the cell was therefore reproduced in `clickhouse local` with the
  harness's own `SCATTER` constant and returns 3,600,000 output rows, identical to the
  server-side cell.
  Two further disclosures: the first run used `INNER`, **a cell with essentially no
  deficit** (+0.64% median in the map), and duly measured nothing — corrected by re-running
  on `FULL`. And the registered "instructions up >10%" threshold was **not met (+6.7%)**,
  because I set it above the size of the deficit being explained; the discriminating
  comparison — *which* counter moved — is nonetheless unambiguous.
  Residual limitation: the counters cover the whole query, not the probe in isolation, so
  attributing the +6.7% *within* the query to the probe loop still rests on the codegen
  artefact and the profiler.
- **G1.1 / G3.3 — PARTIAL.** 14 loops have a cross-tree byte comparison but no mca cycle
  estimate. Under G3.3 they may not carry a POSITIVE or NEGATIVE verdict, and none does.
- **The non-joined-scan coverage gap (assumption 2) was not fixed**, though the brief
  asked for it. Loops N1-N7 carry no new measurement.

---

## 11. Next-mission input — ranked

| # | difference | headroom | tier | necessity | risk of changing it |
| --- | --- | --- | --- | --- | --- |
| 1 | **B22 hoist the key-getter construction out of the per-bucket loop** | up to ~75% of `unified_hash`'s composite-key build phase at 64 threads is in per-partition packing (67% of in-join samples). The *recoverable* fraction is unmeasured | UNSETTLED magnitude, mechanism corroborated by 3 origins | **CONTINGENT** | **Low.** `prepared_keys` is read-only after construction and identical for every partition of a block. The change is a signature, not a semantic |
| 2 | **Isolate the probe phase within the +6.7% instruction delta** | splits the measured one-thread deficit across P1/P3/P4/B2-B4 | UNSETTLED | n/a — a measurement | none. The whole-query counters are done; what remains is per-phase attribution, e.g. `perf record` with a phase-tagged symbol filter |
| 3 | **P3/P4 keep the `Prober` in registers; specialise `offsetInternal` for `sole`** | BOUNDED +5.4 cyc/row by mca ≈ 7% of a 1-thread cell | BOUNDED | CONTINGENT | Medium — touches `TwoLevelHashTable.h`, shared with `Aggregator`; needs the G5.3 inertness proof |
| 4 | **L3 move the three non-shared calls out of `blocks_mutex`** | 2.87 µs of a 2.90 µs critical section, × B blocks, serialised across all build threads | MEASURED granularity, unmeasured benefit | CONTINGENT | Low |
| 5 | **B16/A3 accumulate byte accounting per thread** | inside L1, unmeasured | UNSETTLED | CONTINGENT | Low; costs accounting exactness |
| 6 | **`parallel_hash`'s 309 failed `try_lock` per acquisition** | not a `unified_hash` problem — recorded because it is the largest single inefficiency measured anywhere in this mission | MEASURED | n/a | n/a — out of scope |

**Do not attack:** the bucket count `K`. Ablation A-K1 tested it and it is load-bearing
for contention; halving it costs +90% at 64 threads on the arm with nothing to save.
