# Port audit report — `RadixHashJoin` → `PartitionedHashJoin` (`ahj`)

**Status: IN PROGRESS — this file is updated per candidate; the final version closes Unit 2.**

## Retention round (GA amendment)

After the four measurement verdicts were reported, the requester amended the acceptance rule (verbatim in WORKLOG it.10): candidates 1, 5, 7 are RETAINED; candidate 2 stays rejected; candidate 1 re-scoped to a per-partition distinct cache driving leaf count and per-leaf HT sizing. For the retained candidates the improvement requirement is waived; correctness gates and the no-regression requirement stay. Status: candidate 5 PORTED (WORKLOG it.11), candidate 7 PORTED (WORKLOG it.12), candidate 1 v2 in progress.

## Verdicts at a glance

| Unit | Verdict |
|------|---------|
| Unit 0 (Phase A audit) | GREEN — G0 + G0b, report delivered, GA obtained |
| Approval checkpoint | GREEN — GA verbatim in WORKLOG it.4; predates every Phase B implementation commit |
| Unit 1 candidate 1 (warm-run-cached-distinct-estimate) | GREEN (closed) — `rejected-by-measurement` |
| Unit 1 candidate 2 (narrow-scatter-counters) | GREEN (closed) — `rejected-by-measurement` |
| Unit 1 candidate 5 (pipeline-lane-identity) | GREEN (retained) — `ported` |
| Unit 1 candidate 7 (parallel-hash-table-teardown) | GREEN (retained) — `ported` |
| Unit 2 (full regression + close-out) | not started |

## Pinned SHAs and bases

- Audit pins: `AHJ_SHA=6143ed95a2ba782dbb0166ea6ecf2b8a756d26aa`, `RBM_SHA=ca217fc57eb8be194c95a008ec933972565a21ff`, `PHJ5_SHA=82005a0cc2600382a5699a0576bc871eafad230d`.
- Re-pin at Phase B start (branch moved): implementation base `d8f6f57ee656a7fb73448fb78fdbc772232ffa54` (requester's on-demand leaf-allocation commit; delta audited, WORKLOG it.4).
- Baseline binary for candidates 1/2/5: `tmp/port_audit/bench/bin_base_c1` (bit-frozen copy of the base build; src/ between base and each candidate untouched by campaign commits, verified per candidate).

## Dispositions (53 matrix rows, checker-enforced)

20 `already-present` (each verified by symbol at `$AHJ_SHA`, G0b) · 18 `process-artifact` · 7 `not-applicable` · 4 approved candidates in flight/closed · 4 `deferred-by-requester` (8b-leaf-descriptor, grouped-leaves, block-sorted-gather, nt-store-gating). Zero `port-candidate` rows remain (all resolved to approved/deferred at GA).

## Headline Phase B findings so far

1. **The build's phase-local costs are not wall-relevant on this grid.** Candidate 1 removed 21–51% of the fill phase on every point and moved no wall beyond noise; candidate 2's counter-width change didn't even move its own phase beyond noise. Under the GA rule (keep only wall-movers), both were rejected and reverted — with the mechanisms proven working (reuse events firing, phase timers dropping), which makes the rejections informative, not failed experiments.
2. **Measurement discipline mattered:** round 1 of candidate 1 was voided by the pre-registered drift check (a concurrent bench campaign shares the box); the protocol was amended (before any accepted comparison) to point-interleaved base/candidate pairs, which produced clean, reproducible verdicts.

## Evidence matrix

Every invocation below is copy-paste re-runnable from the repo root. Raw outputs in `tmp/port_audit/bench/`, gates logs in `build/reldeb/`.

| Criterion | Gate invocation | Result (raw) | Non-gate origins | Verdict |
|---|---|---|---|---|
| G0 coverage (Unit 0) | `python3 tmp/port_audit/check_matrix.py` | `OK: 78 rbm + 255 phj5 commits covered by 53 rows, all dispositions valid, all evidence non-empty` → exit 0 | 11-agent inventory (workflow `wf_2cec28ee-445`), raw JSON in `tmp/port_audit/agents/` | GREEN |
| G0b dispositions (Unit 0) | seeded-random 5-row re-derivation (seed 20260721) + every already-present row's symbols read at `$AHJ_SHA` (WORKLOG it.3) | all matched; two grep-pattern corrections, zero content mismatches | independent diff cross-check agents corroborated 3 dispositions against shard agents | GREEN |
| GA approval ordering | `git log --oneline ahj` — GA record commit `a9785dd0219` precedes every implementation | approval verbatim in WORKLOG it.4 | — | GREEN |
| Prereg ordering (c1) | `git show a9785dd0219:tmp/port_audit/PREREG.md` contains candidate-1 entry; implementation commits none (rejected) | prereg committed before any candidate-1 code existed | — | GREEN |
| G1 c1 | `ninja -C build/reldeb clickhouse unit_tests_dbms > build/reldeb/build_port_warmrun.log 2>&1` | exit 0 | — | GREEN |
| G2 c1 | gtest filter + `tmp/port_audit/run_stateless.sh` (logs `test_port_warmrun.log`, `stateless_port_warmrun2.log`) | 48 gtests PASSED; stateless 04603–04607 OK (2 test-side red-green cycles, WORKLOG it.5) | — | GREEN |
| G3 c1 | `tmp/port_audit/bench/run_paired.sh c1r2 <bin_base_c1> <bin_cand_c1>` then `python3 tmp/port_audit/bench/compare_grids.py c1r2_base c1r2_cand PartitionedHashJoinBuildFillMicroseconds` | fill −21.6..−51.2% ALL points; wall inside band ALL points (−3.1..+3.1% vs 4.7–9.7% bands); reuse event = 1 on every point; V hash-verify PASS | round-1 grids (voided by drift check, kept as evidence) | REJECTED-BY-MEASUREMENT (per GA rule) |
| Prereg ordering (c2) | candidate-2 entry committed in `50f5631c974` (with c1 close-out), implementation followed | — | — | GREEN |
| G1 c2 | `ninja ... > build/reldeb/build_port_narrowhist.log` | exit 0 | — | GREEN |
| G2 c2 | logs `test_port_narrowhist.log`, `stateless_port_narrowhist.log` | 48 gtests (incl. `WideCounterFallbackParity`); stateless 04603–04606 OK | — | GREEN |
| G3 c2 | `run_paired.sh c2r1 <bin_base_c1> <bin_cand_c2>`; `compare_grids.py c2r1_base c2r1_cand PartitionedHashJoinBuildHistogramMicroseconds` | histogram −0.7..−8.2%, inside band on ALL points (prereg refutation condition); no wall improvement | — | REJECTED-BY-MEASUREMENT |
| G3 c5 (retention) | `run_paired.sh ret_c5 <bin_final> <bin_ret_c5>` (V-E) then `ret_c5r2 <bin_final> <bin_ret_c5> A E` (repeat); `compare_grids.py ... PartitionedHashJoinBuildFillMicroseconds` | round 1: V/B/C/D in-band, A/E nominally regressed (+16.8%/+15.7%) but flagged contention-suspect (load 19.9→52.1 mid-grid, fill phase flat while untouched phases inflated); round 2 (A/E only, quiet box): A +0.9%/15.8% band, E +1.1%/3.3% band — both in-band, confirming contention | ret_c5/ret_c5r2 grid logs | PORTED (no regression beyond band on any of 6 points) |
| G3 c7 (retention) | teardown discriminator `c7_pred0_bp{1,4}.sql` (global counter, before=bin_ret_c5/after=bin_ret_c7) + `run_paired.sh ret_c7 <bin_ret_c5> <bin_ret_c7>` (V-E) | narrow discriminator ~2.5x drop (33.9ms→13.8ms avg, both runs consistent); wide discriminator inconclusive under a box-load spike to 77 (illustrative only); paired grid wall −2.1%..+0.9% vs 3.0-18.8% bands, all in-band | ret_c7 grid log | PORTED (no regression beyond band on any point) |
| G4 (Unit 2) | *(pending — full paired grid vs Unit-1-start baseline + checker with zero approved rows)* | | | |

## Per-candidate detail

### Candidate 1 — warm-run-cached-distinct-estimate → rejected-by-measurement
Mechanism ported faithfully into `PartitionedHashJoin` idiom (dedicated `PartitionedHashJoinEntry` statistics cache, sketch-free fill on hit, exact post-build publish) and PROVEN working: `PartitionedHashJoinDistinctEstimateReused=1` on all six grid points, `BuildFill` −21.6..−51.2%. Whole-query wall never left the noise band (largest build point B: +0.3% vs 4.7% band). Root cause of the null wall result: fill is ~6% of build thread-time, the sketch feed less than half of that, all overlapped across lanes. Dropped diff: `tmp/port_audit/dropped/c1_warm_run_distinct_cache.diff`. Evidence commit `50f5631c974`.

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

## Environment caveats

A concurrent bench campaign (another session) ran on the box throughout Phase B; the paired protocol was adopted specifically to neutralize it. Load snapshots per grid run are in the `*_grid.log` files.
