# Port audit report — `RadixHashJoin` → `PartitionedHashJoin` (`ahj`)

**Status: IN PROGRESS — Unit 1 (all four candidates) is closed; Unit 2 (final regression + independent verification) remains.**

## Retention round (GA amendment)

After the four measurement verdicts were reported, the requester amended the acceptance rule (verbatim in WORKLOG it.10): candidates 1, 5, 7 are RETAINED despite their measured in-band/off-path verdicts; candidate 2 stays rejected; candidate 1 re-scoped to a per-partition distinct cache driving leaf count and per-leaf HT sizing. For the retained candidates the improvement requirement is waived; correctness gates and the no-regression requirement stay. **All three retentions are now implemented, gated, and measured no-regression-clean (WORKLOG it.11-13); candidate 1's v2 design additionally cleared a genuine wall improvement on one grid point, unforced.**

## Verdicts at a glance

| Unit | Verdict |
|------|---------|
| Unit 0 (Phase A audit) | GREEN — G0 + G0b, report delivered, GA obtained |
| Approval checkpoint | GREEN — GA verbatim in WORKLOG it.4; predates every Phase B implementation commit |
| Unit 1 candidate 1 (warm-run-cached-distinct-estimate) | GREEN — measured `rejected-by-measurement` (global-count v1) → RETAINED, re-scoped to per-partition v2, `ported` |
| Unit 1 candidate 2 (narrow-scatter-counters) | GREEN (closed) — `rejected-by-measurement`, not retained |
| Unit 1 candidate 5 (pipeline-lane-identity) | GREEN — measured `rejected-by-measurement` → RETAINED, `ported` |
| Unit 1 candidate 7 (parallel-hash-table-teardown) | GREEN — measured `rejected-by-measurement` (prediction-0) → RETAINED, `ported` |
| Unit 2 (full regression + close-out) | in progress |

## Pinned SHAs and bases

- Audit pins: `AHJ_SHA=6143ed95a2ba782dbb0166ea6ecf2b8a756d26aa`, `RBM_SHA=ca217fc57eb8be194c95a008ec933972565a21ff`, `PHJ5_SHA=82005a0cc2600382a5699a0576bc871eafad230d`.
- Re-pin at Phase B start (branch moved): implementation base `d8f6f57ee656a7fb73448fb78fdbc772232ffa54` (requester's on-demand leaf-allocation commit; delta audited, WORKLOG it.4).
- Baseline binary for candidates 1/2/5: `tmp/port_audit/bench/bin_base_c1` (bit-frozen copy of the base build; src/ between base and each candidate untouched by campaign commits, verified per candidate).

## Dispositions (53 matrix rows, checker-enforced)

20 `already-present` (each verified by symbol at `$AHJ_SHA`, G0b) · 18 `process-artifact` · 7 `not-applicable` · 3 `ported` (candidates 1, 5, 7 — retained by the requester over their measurement verdicts) · 1 `rejected-by-measurement` (candidate 2, not retained) · 4 `deferred-by-requester` (8b-leaf-descriptor, grouped-leaves, block-sorted-gather, nt-store-gating). Zero `approved`/`port-candidate` rows remain.

## Headline Phase B findings

1. **The build's phase-local costs were not wall-relevant at the original (global-count) designs.** Candidate 1 v1 removed 21–51% of the fill phase on every point and moved no wall beyond noise; candidate 2's counter-width change didn't even move its own phase beyond noise; candidate 5's lane plumbing left both the wall and its declared phases in-band; candidate 7's teardown was proven structurally unmeasurable in the per-query packet and arithmetically bounded in-band even under a perfect parallelization. All four measurements were honest, pre-registered predictions that came true — informative rejections, not failed experiments.
2. **The requester's retention call, and candidate 1's re-scoping, paid off.** Retaining 5 and 7 cost nothing (no-regression confirmed on all points, both candidates); re-scoping candidate 1 to cache the PER-PARTITION distinct breakdown (not just one global count) let per-leaf hash-table sizing improve materially — `BuildFill` now drops 24–56% (vs v1's 21–51%) and the largest build point (B) clears a genuine wall improvement (−9.8%) beyond its own noise band, unforced (the improvement requirement was waived for this round).
3. **Measurement discipline mattered throughout:** round 1 of candidate 1 v1 was voided by the pre-registered drift check (a concurrent bench campaign shares the box); the protocol was amended (before any accepted comparison) to point-interleaved base/candidate pairs. The same contention pattern recurred during candidate 5's retention grid (A/E nominally regressed under a load spike, phase-breakdown evidence pointed to noise, a repeat settled it in-band) and during candidate 7's teardown discriminator (the wide-shape reading was inconclusive under a load spike to 77 and was treated as illustrative, not load-bearing) — in every case the response was to repeat the affected points, never to rationalize a result.
4. **A real bug was caught before landing, not after:** candidate 7's originally staged parallel-destructor design reused a `ThreadPool` member that is already reset to null by the time the destructor runs — its guard would have been permanently false, a silent no-op dressed as a feature. Caught by reading the surrounding lifecycle code before applying the staged patch, not by a failing test (the flawed version would have compiled and passed every gate).

## Evidence matrix

Every invocation below is copy-paste re-runnable from the repo root. Raw outputs in `tmp/port_audit/bench/`, gates logs in `build/reldeb/`.

| Criterion | Gate invocation | Result (raw) | Non-gate origins | Verdict |
|---|---|---|---|---|
| G0 coverage (Unit 0) | `python3 tmp/port_audit/check_matrix.py` | `OK: 78 rbm + 255 phj5 commits covered by 53 rows, all dispositions valid, all evidence non-empty` → exit 0 | 11-agent inventory (workflow `wf_2cec28ee-445`), raw JSON in `tmp/port_audit/agents/` | GREEN |
| G0b dispositions (Unit 0) | seeded-random 5-row re-derivation (seed 20260721) + every already-present row's symbols read at `$AHJ_SHA` (WORKLOG it.3) | all matched; two grep-pattern corrections, zero content mismatches | independent diff cross-check agents corroborated 3 dispositions against shard agents | GREEN |
| GA approval ordering | `git log --oneline ahj` — GA record commit `a9785dd0219` precedes every implementation | approval verbatim in WORKLOG it.4 | — | GREEN |
| Prereg ordering (c1) | `git show a9785dd0219:tmp/port_audit/PREREG.md` contains candidate-1 entry; implementation commits none (rejected) | prereg committed before any candidate-1 code existed | — | GREEN |
| G1 c1 (v1) | `ninja -C build/reldeb clickhouse unit_tests_dbms > build/reldeb/build_port_warmrun.log 2>&1` | exit 0 | — | GREEN |
| G2 c1 (v1) | gtest filter + `tmp/port_audit/run_stateless.sh` (logs `test_port_warmrun.log`, `stateless_port_warmrun2.log`) | 48 gtests PASSED; stateless 04603–04607 OK (2 test-side red-green cycles, WORKLOG it.5) | — | GREEN |
| G3 c1 (v1) | `tmp/port_audit/bench/run_paired.sh c1r2 <bin_base_c1> <bin_cand_c1>` then `python3 tmp/port_audit/bench/compare_grids.py c1r2_base c1r2_cand PartitionedHashJoinBuildFillMicroseconds` | fill −21.6..−51.2% ALL points; wall inside band ALL points (−3.1..+3.1% vs 4.7–9.7% bands); reuse event = 1 on every point; V hash-verify PASS | round-1 grids (voided by drift check, kept as evidence) | REJECTED-BY-MEASUREMENT (per GA rule) |
| G1 c1 (v2 retention) | `ninja -C build/reldeb clickhouse unit_tests_dbms > build/reldeb/build_port_c1v2_test.log 2>&1` | exit 0, all 128 targets | — | GREEN |
| G2 c1 (v2 retention) | gtest filter + `tmp/port_audit/run_stateless.sh` (logs `test_port_c1v2.log`, `stateless_port_c1v2_2.log`) | 50 gtests PASSED (incl. 2 new: warm-run reuse, fold/split with a hard `EXPECT_LT` bit-count assertion); stateless 04603–04607 OK (04607 recreated) | — | GREEN |
| G3 c1 (v2 retention) | `run_paired.sh ret_c1 <bin_ret_c7> <bin_ret_c1>` (V-E); `compare_grids.py ret_c1_base ret_c1_cand PartitionedHashJoinBuildFillMicroseconds`; growths spot-check `ret_c1_growths` (B,C only, after adding `PartitionedHashJoinHashTableGrowths` to the bench harness) | reuse=1 on every point; fill −24.4%..−56.3%; wall in-band on 5 points, **B wall-improved −9.8% vs 3.0% band**; growths=0 both sides at B,C | — | PORTED (no regression; bonus improvement at B) |
| Prereg ordering (c2) | candidate-2 entry committed in `50f5631c974` (with c1 close-out), implementation followed | — | — | GREEN |
| G1 c2 | `ninja ... > build/reldeb/build_port_narrowhist.log` | exit 0 | — | GREEN |
| G2 c2 | logs `test_port_narrowhist.log`, `stateless_port_narrowhist.log` | 48 gtests (incl. `WideCounterFallbackParity`); stateless 04603–04606 OK | — | GREEN |
| G3 c2 | `run_paired.sh c2r1 <bin_base_c1> <bin_cand_c2>`; `compare_grids.py c2r1_base c2r1_cand PartitionedHashJoinBuildHistogramMicroseconds` | histogram −0.7..−8.2%, inside band on ALL points (prereg refutation condition); no wall improvement | — | REJECTED-BY-MEASUREMENT |
| G3 c5 (retention) | `run_paired.sh ret_c5 <bin_final> <bin_ret_c5>` (V-E) then `ret_c5r2 <bin_final> <bin_ret_c5> A E` (repeat); `compare_grids.py ... PartitionedHashJoinBuildFillMicroseconds` | round 1: V/B/C/D in-band, A/E nominally regressed (+16.8%/+15.7%) but flagged contention-suspect (load 19.9→52.1 mid-grid, fill phase flat while untouched phases inflated); round 2 (A/E only, quiet box): A +0.9%/15.8% band, E +1.1%/3.3% band — both in-band, confirming contention | ret_c5/ret_c5r2 grid logs | PORTED (no regression beyond band on any of 6 points) |
| G3 c7 (retention) | teardown discriminator `c7_pred0_bp{1,4}.sql` (global counter, before=bin_ret_c5/after=bin_ret_c7) + `run_paired.sh ret_c7 <bin_ret_c5> <bin_ret_c7>` (V-E) | narrow discriminator ~2.5x drop (33.9ms→13.8ms avg, both runs consistent); wide discriminator inconclusive under a box-load spike to 77 (illustrative only); paired grid wall −2.1%..+0.9% vs 3.0-18.8% bands, all in-band | ret_c7 grid log | PORTED (no regression beyond band on any point) |
| G4 (Unit 2) | *(pending — full paired grid vs Unit-1-start baseline + checker with zero approved rows)* | | | |

## Per-candidate detail

### Candidate 1 — warm-run-cached-distinct-estimate → ported (retained by requester, re-scoped to per-partition v2)
**v1 (global count, original measurement):** mechanism ported faithfully into `PartitionedHashJoin` idiom (dedicated `PartitionedHashJoinEntry` statistics cache, sketch-free fill on hit, exact post-build publish) and PROVEN working: `PartitionedHashJoinDistinctEstimateReused=1` on all six grid points, `BuildFill` −21.6..−51.2%. Whole-query wall never left the noise band (largest build point B: +0.3% vs 4.7% band). Root cause of the null wall result: fill is ~6% of build thread-time, the sketch feed less than half of that, all overlapped across lanes. Dropped diff: `tmp/port_audit/dropped/c1_warm_run_distinct_cache.diff`.

**v2 (retention round, per-partition, the requester's own design):** the cache entry becomes `{bits, total_distinct, per_partition}` - the EXACT distinct count of every leaf, not just their sum - and `planHashTables` consumes it leaf-by-leaf instead of the v1 uniform rescale (exact copy when the warm build's own plan bits match the cache; sum-fold when the cache was finer; uniform-split when coarser - always valid because partitions are MSB-first radix ranges that nest exactly across bit counts). Every leaf's reserve stays clamped to its exact row count regardless, so a stale or differently-shaped cache entry can only mis-size, never break correctness. The added sizing precision shows up directly: `BuildFill` now drops 24–56% (vs v1's 21–51%) and the largest build point (B) clears a genuine, unforced wall improvement (−9.8% vs its 3.0% band) - the retention's improvement requirement was waived, but this design earned one anyway. `PartitionedHashJoinHashTableGrowths` (added to the bench harness this round) reads 0 on both sides at the two largest shapes. Evidence commit: see WORKLOG it.13.

### Candidate 2 — narrow-scatter-counters → rejected-by-measurement
Dual-shape UInt32/UInt64 histogram+prefix counters (UInt64 fallback intact, forced-fallback parity gtest). The declared phase itself stayed inside the band everywhere — the histogram stage is bounded by bucket-id derivation and pid-array traffic, not counter width. Dropped diff: `tmp/port_audit/dropped/c2_narrow_scatter_counters.diff`. Evidence commit `3db90315f78`.

### Candidate 5 — pipeline-lane-identity → ported (retained by requester)
Full pipeline port (IJoin lane overloads, transform/builder plumbing, lock-free lane slots with collision-tolerant fallbacks; 22 files). All correctness gates green (incl. a lane-parity gtest with out-of-range lanes). Originally measured `rejected-by-measurement` (G3 round `c5r1`: wall and both declared phases inside the band on every point). Retained per the requester's GA amendment (WORKLOG it.10) with the improvement requirement waived; re-applied from the dropped diff with the destructor hunk merged against candidate 7's teardown-timing scope. No-regression re-confirmed across two grid rounds (`ret_c5`, `ret_c5r2`) — all six points in-band on both wall and the `BuildFill` phase (WORKLOG it.11). Superseded dropped diff (historical only): `tmp/port_audit/dropped/c5_pipeline_lane_identity.diff`.

### Candidate 7 — parallel-hash-table-teardown → ported (retained by requester)
Originally settled `rejected-by-measurement` via the pre-registered prediction-0 decidability check (WORKLOG it.9): the destructor runs after the per-query ProfileEvents packet (structurally unmeasurable there) and the total serial teardown was at most 1.6-2.2% of the largest walls — below the band floor even for a perfect parallelization. Retained per the requester's GA amendment with the improvement requirement waived. Implemented: `~PartitionedHashJoin` work-steals per-leaf map destruction over a short-lived local `ThreadPool` spun up inside the destructor (guarded `!delegate_mode && leaf_maps.size() >= 64`), matching `ConcurrentHashJoin`'s teardown rationale. Deviation caught and fixed before landing: the originally staged design (`tmp/port_audit/staged_c7_parallel_teardown.patch.txt`) guarded on the `post_build_pool` member, which is already null by destructor time (reset right after the post-build phase) — that guard would have been permanently false, a silent no-op; fixed by allocating a fresh pool locally instead (WORKLOG it.12). Teardown-magnitude discriminator: narrow shape ~2.5x faster (consistent both runs); wide shape inconclusive under heavy box contention (illustrative only, not a gate). No-regression paired grid `ret_c7`: wall in-band on all 6 points.

## Deviations (all documented at decision time)

1. Band formula amended to `max(3%, 2 x (median-min)/median)` — the harness reports median+min only (PREREG amendment 1, WORKLOG it.5).
2. Point-interleaved pairing replaced the sequential-grids + drift-check protocol after round 1 of candidate 1 was voided (PREREG amendment 2, WORKLOG it.6).
3. Checker accepts `diff-derived` commitless rows (WORKLOG D2) — mechanisms found only by the full-tree diff whose commits are owned by other rows.
4. The mission's "~44 phj5 commits" was actually 255 in the pinned range (212 upstream master commits grouped per WORKLOG D1).
5. Candidate 7's originally staged parallel-destructor patch guarded on the `post_build_pool` member, which is reset to null right after the post-build phase - long before the destructor runs - so its guard would always have been false, a silent no-op. Caught before applying it; the landed destructor spins up its own short-lived local `ThreadPool` instead (WORKLOG it.12).
6. Candidate 5's retention no-regression grid (`ret_c5`) read A/E as nominally regressed on the first pass under box contention (load 19.9→52.1 mid-grid); the phase breakdown pointed to noise (the touched phase was flat, untouched phases inflated together with wall time), and a repeat of just those two points on a quiet box confirmed it (WORKLOG it.11) rather than accepting either the regression or a rationalized dismissal.
7. WORKLOG iteration 11 briefly described candidate 1 v2 as "implemented (working tree)" before any of its code actually existed - a documentation-ahead-of-implementation slip from the prior session, caught and flagged in WORKLOG (not silently corrected) before the real implementation (iteration 13).

## Environment caveats

A concurrent bench campaign (another session) ran on the box throughout Phase B; the paired protocol was adopted specifically to neutralize it. Load snapshots per grid run are in the `*_grid.log` files.
