# Phase B pre-registrations

Every candidate's entry here is committed BEFORE its implementing change (verifier checks the ordering in git history). Base SHA for all of Phase B: `d8f6f57ee656a7fb73448fb78fdbc772232ffa54`.

## Protocol (fixed at Unit-1 start, shared by every candidate)

**Harness:** `bep/tools/join_mergetree_bench.py run` against the persistent dataset at `/mnt/data/join_bench_data` (loaded metadata verified: schema v4, D_max=524288000, bucket_width=4194304, max_multiplicity=1, max_cycles=4, 7+7 payload columns; build 524,288,000 rows, probe 2,097,152,000 rows). The harness pins settings, asserts the execution path fail-closed via ProfileEvents, asserts row counts, runs ONE `clickhouse local` process per (point, algorithm) with 1 warmup + `--runs` timed queries — so timed runs are warm-cache runs by construction (this is what makes candidate 1 measurable). Counters come from the timed run closest to the median wall.

**Grid (6 points; identical for every candidate and for the Unit-1-start baseline):**

| Point | cardinality | ratio | bp x pp | threads | role |
|-------|------------|-------|---------|---------|------|
| V | 4194304 | 1 | 1x1 | 32 | verification point — output 4.19M rows <= 10M so the sorted `FORMAT Hash` cross-check vs parallel_hash RUNS here every time |
| A | 8388608 | 4 | 1x1 | 32 | small build, probe-heavy |
| B | 524288000 | 1 | 1x1 | 32 | large build, build-heavy, narrow |
| C | 524288000 | 1 | 4x4 | 32 | large build, wide payloads |
| D | 67108864 | 4 | 0x0 | 32 | mid build, key-only |
| E | 67108864 | 4 | 1x1 | 96 | thread-scaling |

**Exact invocation** (per point; `--cardinalities/--ratios/--build-payload-columns/--probe-payload-columns/--threads` from the table):

```
python3 bep/tools/join_mergetree_bench.py run --path /mnt/data/join_bench_data \
  --binary build/reldeb/programs/clickhouse \
  --cardinalities <c> --multiplicities 1 --ratios <r> --hit-rates 1.0 \
  --build-payload-columns <bp> --probe-payload-columns <pp> \
  --threads <t> --runs 5 > <log> 2>&1
```

**Pairing:** per candidate, a FRESH baseline (base = the commit the candidate's diff applies to) is run on the full grid immediately before the candidate run; never an inherited number. After the candidate run, points V and B are re-run once on the baseline binary as a drift check — if baseline drift exceeds the band, the pair is void and both sides are re-measured.

**Noise band:** max(3%, observed run-to-run spread of the baseline at that point), where spread = (max-min)/median of the 5 timed walls. Effects inside the band are "no result".

*Amendment (documented deviation, before any candidate comparison was made):* the harness prints only the median and min of the timed walls, not all samples, so (max-min)/median is not computable from its output. The operative band is `max(3%, 2 x (median - min) / median)` of the baseline invocation — symmetric-spread proxy, conservative for the acceptance direction. Recorded in WORKLOG iteration 5 before the first baseline/candidate comparison.

**Acceptance (G3, includes the requester's tightening from GA):** a candidate is kept only if, on at least one grid point, BOTH (a) median whole-query wall of `partitioned_hash` improves beyond the band AND (b) the candidate's declared phase counter improves beyond the band — and NO grid point's wall regresses beyond the band. Phase counters come from the median-closest run's packet (one sample per invocation); if a phase delta lands between 1x and 2x the band, one repeat invocation settles direction before deciding. Anything else = `rejected-by-measurement`, revert, record numbers.

**Environment:** 96-core box; a concurrent session is running its own bench campaign (32T) as of Phase B start. Concurrent load is recorded in the worklog per G3 run (`uptime` + `ps` snapshot before/after); the drift check above is the guard. If drift repeatedly voids pairs, escalate to the requester rather than widening the band silently.

**Correctness gates per candidate:** G1 `ninja -C build/reldeb clickhouse unit_tests_dbms > build/reldeb/build_port_<id>.log 2>&1`; G2 `build/reldeb/src/unit_tests_dbms --gtest_filter='PartitionedHashJoin*:*ColumnsScatter*' > build/reldeb/test_port_<id>.log 2>&1` plus stateless `04603|04604|04605|04606` via `tests/clickhouse-test` (output to `build/reldeb/stateless_port_<id>.log`); the V point's hash verification vs parallel_hash must pass in every G3 run.

---

## Candidate 1 — warm-run-cached-distinct-estimate (approved; phj5 `3faf01c1e90`)

**Pre-registered 2026-07-21, before any implementing change (base `d8f6f57ee65`).**

**Mechanism to implement:** a cross-run distinct-key-count cache for `partitioned_hash`, in `PartitionedHashJoin` idiom:
- New entry type in `src/Interpreters/HashTablesStatistics.h` (e.g. `PartitionedHashJoinEntry { size_t distinct_keys; }`) — deliberately separate from `HashJoinEntry` so `parallel_hash` and `partitioned_hash` never clobber each other's entry under the same cache key (phj5's stated rationale).
- Plumb `StatsCollectingParams` into the `PartitionedHashJoin` constructor from `PlannerJoins.cpp` (same key derivation as the other join algorithms), honored only when `collect_hash_table_stats_during_joins` allows.
- Fill phase: on a cache HIT, skip the per-row HyperLogLog feed — the fill loop still computes and stores route words (they are load-bearing) but calls a sketch-free variant of `computeJoinRoutesForFill`. On MISS, feed the sketch exactly as today.
- Barrier: on HIT, `hll_estimate` := cached `distinct_keys` (clamped >= 1); partition plan and per-leaf reserves consume it unchanged.
- Post-build: publish the EXACT distinct count (sum of leaf map sizes — exactly known at ahj, better than phj5's estimate republish) back to the cache.
- Observability: new ProfileEvent `PartitionedHashJoinDistinctEstimateReused` (count of joins that skipped the sketch), used by the bench/stateless assertions.

**Declared phase counter (the "portion being optimized"):** `PartitionedHashJoinBuildFillMicroseconds`.

**Predictions:**
1. Warm timed runs fire `PartitionedHashJoinDistinctEstimateReused` = 1 per join; the warmup (cold) run does not.
2. `BuildFillMicroseconds` (thread-summed) drops on the large-build points B and C — expected order 10-30% of fill (the sketch add is a minority of the fused route loop; phj5 claimed ~11% of whole warm build under its architecture, cited as claim).
3. Whole-query wall improves most on B/C (build-heavy shapes); A/D/E likely inside the band.
4. `PartitionedHashJoinPartitions` may legitimately change on HIT runs (the cached exact count replaces the HLL estimate — a BETTER input to the same plan formula); `leaf_growths`/`amac_ring_growths` must NOT increase beyond baseline.
5. No grid point's wall regresses beyond the band (the cold path is unchanged; the hit path only removes work).

**What refutes the port (=> rejected-by-measurement, revert):** wall delta inside the band on ALL points, or fill delta inside the band on ALL points, or any point regressing beyond the band, or any correctness gate red after the 3-cycle ceiling.

**Round-1 result (2026-07-21): PAIR VOID by the drift check, as pre-registered.** Candidate grid showed reuse firing on all 6 points and fill -13..-47%, B wall -8.4% beyond its 3% band, but V/C/D/E walls +9.7..+21.4% — and the base-binary drift re-run moved -7.1% at B and +12.9% at V, both outside their bands, so baseline and candidate sampled different machine conditions (concurrent bench campaign's load varies). Neither acceptance nor rejection is valid on this pair.

*Amendment 2 (before any round-2 data):* G3 measurement becomes POINT-INTERLEAVED — for each grid point, the base invocation and the candidate invocation run back-to-back (base first), so both sides sample the same conditions window; the band for each point is computed from the base invocation of the SAME round. The drift check is replaced by this pairing (it was the coarse form of it). Acceptance rule unchanged.

---

## Candidate 2 — narrow-scatter-counters (approved; rbm diff-derived, origin commit 832ebbbc51f)

**Pre-registered 2026-07-21, before any implementing change. Written while candidate 1's G3 was in flight; candidate 2 implementation starts only after candidate 1 closes.**

**Mechanism to implement:** the first-pass build histogram and prefix-sum arrays in `PartitionedHashJoinBuild.cpp` (`worker_hist`, `starts`, currently `PaddedPODArray<UInt64>` at ~354-356, plus the interleaved `hist_lanes` counters they merge) switch to `UInt32` when the accumulated build rows fit `UInt32` (RBM selected the counter type per scatter; the refine path at ahj is already 32-bit). Fallback to the existing `UInt64` shape when rows exceed `UInt32` (kept as the general form, exercised by a forced-plan test if feasible; rows > 4.29B are otherwise impractical in tests).

**Declared phase counter:** `PartitionedHashJoinBuildHistogramMicroseconds`.

**Predictions:**
1. Histogram phase (thread-summed) drops on the high-fanout points B/C (32768 partitions x 32 workers: the histogram+prefix working set halves from 8 MB to 4 MB per array); baseline B histogram to be read from `c1_base2_B.log` PhaseEvents at analysis time.
2. Whole-query wall: expected small; may land inside the band (fill/scatter dominate the build; honest reject possible).
3. No behavioral change: partitions, leaf rows, verify status identical to baseline.
4. No point regresses beyond the band (the change only narrows memory traffic; same algorithm).

**What refutes the port:** wall or histogram delta inside the band on ALL points, or any point regressing beyond the band, or any gate red after 3 cycles.

---

## Candidate 5 — pipeline-lane-identity (approved; rbm 6503c7cfa9a + phj5 27872ed292a)

**Pre-registered 2026-07-21, before any implementing change (written while candidate 2's G3 was in flight; implementation starts after candidate 2 closes).**

**Mechanism to implement:** IJoin overloads `addBlockToJoin(block, num_rows, check_limits, build_lane)` and `joinBlock(block, lane)` with defaulted forwarding for all implementers; lane indices assigned as counters in `QueryPipelineBuilder::joinPipelinesRightLeft` and carried by `FillingRightJoinSideTransform` / `JoiningTransform` (source forms read via `git show 6503c7cfa9a` / `git show 27872ed292a`). `PartitionedHashJoin` binds its fill lane by index (replacing the `fill_mutex` + `unordered_map<thread::id, FillLane*>` lookup per fill block) and its probe scratch by index (replacing the `probe_scratch_mutex` pool acquire/release per probe block). The source branches' documented caveat (lane indices not guaranteed distinct across probe streams) must be handled collision-tolerantly.

**Declared phase counters:** `PartitionedHashJoinBuildFillMicroseconds` (fill-side mutex+map removal) and `PartitionedHashJoinProbeMicroseconds` (scratch pool mutex removal); acceptance requires whichever is declared per point to improve beyond band alongside the wall.

**Predictions (honest):** the removed cost is per-BLOCK (one mutex+hash lookup per fill block; one mutexed acquire/release per probe block) — at the grid's block counts this is sub-millisecond total. Best shot is E (96 threads, highest contention). REALISTIC EXPECTATION: all points inside the band on the wall ⇒ rejected-by-measurement, mirroring candidate 1's pattern. The candidate is implemented and measured because the requester approved it; the measurement is the deliverable either way.

**What refutes the port:** wall or declared-phase delta inside the band on ALL points, or any point regressing beyond the band, or any gate red after 3 cycles.

---

## Candidate 7 — parallel-hash-table-teardown (approved; rbm diff-derived, upstream-mirrored)

**Pre-registered 2026-07-21, before any implementing change (implementation starts after candidate 5 closes).**

**Mechanism to implement:** parallelize `~PartitionedHashJoin` teardown — per-leaf map destruction (and arena/stored-block release where separable) work-stealing over `post_build_pool` (alive in the destructor body; members destroy after it), mirroring upstream `ConcurrentHashJoin::~ConcurrentHashJoin`'s pool-parallel `clearAndShrink` and RBM's `partition_joins[p].reset()` parallelRun. Premise strengthened by the requester's base commit `d8f6f57ee65`: teardown now frees one exact-reserved allocation per leaf (up to 32768 at point B) plus per-worker arenas, all sequential today.

**Instrumentation prerequisite (part of this candidate):** new ProfileEvent `PartitionedHashJoinTeardownMicroseconds` timing the destructor body, appended to the harness `EVENTS` list. Prediction 0 (settles measurability): if the event appears with nonzero value in the harness's per-query packet, teardown is inside the measured query scope and the wall criterion is decidable; if it never appears (destruction deferred past the packet), the wall criterion cannot be satisfied and the candidate resolves per the GA rule on the wall evidence as measured — reported explicitly either way.

**Declared phase counter:** `PartitionedHashJoinTeardownMicroseconds` (added by this candidate; measured on BOTH sides of the pair — the baseline side gets the event via a preparatory instrumentation-only commit so the pair compares like for like).

**Predictions:** teardown is largest on B/C (32768 leaf maps + large arenas); parallelizing over ~32 workers should cut the phase severalfold IF it is on the measured path. Wall: unknown magnitude — this is the one approved candidate whose target grew with the new base; measurement decides.

**What refutes the port:** teardown phase or wall inside the band on ALL points, or any point regressing beyond the band, or any gate red after 3 cycles.

---

# Retention round (GA amendment, WORKLOG it.10)

The requester retains candidates 1, 5, 7 after their measured verdicts; the improvement requirement is waived for them, the NO-REGRESSION requirement and all correctness gates stay. Per-candidate acceptance for this round: G1+G2 green AND paired grid shows no wall regression beyond the band on any point. Measured deltas are recorded either way.

## Candidate 5 retention — re-apply the dropped diff
Re-apply `tmp/port_audit/dropped/c5_pipeline_lane_identity.diff` (22 files); the destructor hunk needs a manual merge with the teardown-timing destructor from `13900f0daca` (combined: timed scope + explicit heavy-state destruction + parked-scratch cleanup). No design change.

## Candidate 7 retention — implement the parallel destructor (step B)
Work-stealing destruction of the per-leaf maps (and per-arena release) over `post_build_pool` inside the timed scope, mirroring upstream `ConcurrentHashJoin`'s pool-parallel teardown; guards: pool exists, not delegate mode, partitions above a small floor, `try/catch` fallback to serial (destructors must not throw). Expected phase effect: `PartitionedHashJoinTeardownMicroseconds` (observable via the global counter / discriminator, NOT the per-query packet) drops severalfold at B/C; wall unaffected (off-path, previously measured).

## Candidate 1 retention v2 — per-partition distinct cache (requester's design)
Cache entry becomes `{bits, total_distinct, per_partition_distinct[2^bits]}` published post-build from the EXACT per-leaf map sizes at the built plan's bits. Warm run: fill skips the sketch (as v1); the barrier uses `total_distinct` for the partition-count decision and, when the warm plan's bits match the cached bits, sizes each leaf by its OWN cached distinct count (folded by summation when the warm plan is coarser; parent-count uniform split + row clamp when finer); every reserve stays clamped by the exact per-leaf row histogram, so a stale entry can only mis-size (growth path absorbs it, counted). Predictions: reuse event fires on warm runs; `leaf_growths`/`amac_ring_growths` not increased vs baseline; `PartitionedHashJoinHashTableBytes` on warm runs <= v1's uniform-rescale sizing on skewed-duplicate builds (exactness); wall in-band (waived); no regression beyond band. Refutation (this round): any correctness gate red after 3 cycles, or any wall regression beyond band, or leaf growths increasing on the standard grid.
