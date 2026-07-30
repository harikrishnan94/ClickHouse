# Probe-phase A/B of `phj-ph` HEAD vs the campaign baseline — three suites, two metrics

**Campaign directory** `/mnt/ch/ClickHouse/tmp/probe_campaign_20260729/`
**Date** 2026-07-29 · **Arch** ARM (`aarch64`) only · **Algorithm** `parallel_hash` only, both arms

| | commit | sha256 | role |
| --- | --- | --- | --- |
| **Arm A** `baseline` | `a05f3ee81ff8411759637fa367aad62e72726e71` | `0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4` | campaign baseline |
| **Arm B** `candidate` | `fa5667f2da786e07ada50f711da205890b610343` (`phj-ph` HEAD) | `83de808547081e3a073772efe71fa3401e4a4889a4c720eeca9a1dc716f9e2b4` | under test |

## The two metrics

    probe_cost      = ConcurrentHashJoinProbeDispatchMicroseconds
                    + ConcurrentHashJoinProbeLookupMicroseconds
    projection_cost = ConcurrentHashJoinProbeMicroseconds - probe_cost

Per cell/unit/arm, summed across probe threads, median over that arm's timed runs. Each metric is
scored, banded and verdicted **independently**; they are never summed and never netted. Band per
metric: `max(3%, that metric's own per-arm relative spread)`; anything inside the band is TIE.

---

## 1. Per-unit verdict

| Unit | Verdict | One-line status |
| --- | --- | --- |
| **Unit 0** — metrics, scorer, A/A control | **GREEN** (fleet venue) · **RED→documented** (jbmt synthetic venue) | G0-a/b/c all exit 0 on the fleet A/A; G0-a exits 1 on the jbmt *synthetic* venue at both 5 and 11 timed runs, and exits 0 on the jbmt *real* venue. Scorer delivered with a 32-assertion self-test proving every gate can go red. |
| **Unit 1** — fleet_ab, 94 cells × ABBA + BAAB | **GREEN except coverage** | G0-b, G0-c and G1-b exit 0 on both sweeps, and the fleet is torn down with empty-inventory proof from two independent origins. **G1 is RED**: 78 of 94 cells carry verdicts; the 16 others are NO-VERDICT with the harness's own `below-duration-floor` reason. `--expect-cells 94` was not lowered. Set equality of the *measured* 94 cells against the plan is separately green. |
| **Unit 2** — jbmt legacy, 347 cells | **NO RESULT — for want of measurement time** | No legacy cell was measured, so no verdict exists and none is implied. Its A/A control is red as run (G0-a exit 1 in all four sweeps), but a fourth control shows the venue IS measurable to 3 % on a quiet host once fleet_ab's own 200 ms floor excludes micro-units — so the blocker is the ~11.5 h the 347 cells need, not validity. An earlier draft called this a validity block, which flattered the run; corrected in §4.2. |
| **Unit 3** — jbmt real, 376 units × tiers a, b | **BOTH TIERS COMPLETE, GREEN except coverage** | Venue validated on the exact measured server pair (G0-a exit 0, floors 1.24 % / 1.59 %). Tier a: 376 attempted, **368 scored**, 8 NO-VERDICT. Tier b: 376 attempted, **365 scored**, 11 NO-VERDICT. Every gap is an `OVER_BUDGET` skip taken before any timed run, with the harness's reason. `--expect-cells 376` exits 1 for both tiers, so **G3 is RED at 368/376 and 365/376**, quantified, N not lowered. G0-b/G0-c green over 8,096 and 8,030 timed rows. |
| **Unit 4** — consolidation into this document | **done** | Per-suite and per-metric tallies, both metrics' loss lists in full, the opposite-direction cells, every NO-VERDICT cell with its reason, and the evidence matrix (§7). |

### The result in three sentences

On the fleet's 94-cell synthetic plan, `phj-ph` HEAD makes `probe_cost` (dispatch + lookup)
substantially cheaper — 68 of 78 scored cells win, −35.2 % aggregate, replicated independently in
both block orders — while making `projection_cost` (the column-materialization residual)
substantially more expensive: 71 of 78 cells lose, +26.7 % aggregate. On real queries, measured over two data tiers
(tier a: 368 scored units; tier b: 365 — a partly independent replication, since JOB's 126 units do
not scale between tiers and so run on identical data), the same trade appears far weaker and far
less favourable: `probe_cost` wins and losses are near-balanced by count (161/156 and 164/151) with a
median unit that does not move (+0.4 %, −0.4 %), the −8.8 % aggregate in both tiers is carried by
twenty units that themselves got slower, `projection_cost` loses roughly four to one (198/43 and
190/49) at +7.3 % and +10.0 % on the 365 units common to both tiers, and the measured probe total
rises (+4.2 %, +6.1 % like-for-like) with 287 of 368 and 273 of 365 queries slower end to end. **63 fleet cells and 142 tier-a units move in opposite
directions on the two metrics; neither metric's verdict is ever netted against the other's, and no
cell is presented as a win on the strength of the other metric.**

The one suite that produced **no** verdict is the jbmt synthetic legacy suite (Unit 2): none of its
347 cells was measured, because the sweep needs ~11.5 h that this run did not have alongside the
other two suites. Its venue question is settled favourably (§4.2), so what is missing is time, not
validity — and no legacy number is reported, inferred or implied.

### Flags requiring attention

- **No authorization-required step was taken.** No EC2 resource beyond the single 8-shard fleet
  this prompt plans was created: in particular **no volume was created from snapshot
  `snap-021cbdc2484f86607`**, because the real-suite data for both tiers was already present on
  this host at `/mnt/data/jbmt_server/data` (392 GB, `verify: OK` for tiers a and b). Nothing is
  outstanding for a human to authorize.
- **Nothing is BLOCKED on credentials.** `aws sts get-caller-identity` succeeded before any other
  work, so no unit was blocked on SSO.
- **Deviations, all documented** (details in `WORKLOG.md`): binaries staged as hardlinks rather
  than copies (9.8 GB free on `/mnt/ch`); jbmt suites measured on the orchestration host rather
  than a fleet (only one fleet may exist at a time under the vCPU quota, and it was committed to
  Unit 1); jbmt timed-run count raised from 5 to 11 via a new `--min-timed-runs` flag in the
  campaign's own copy of the harness.

### HIGH-IMPACT assumptions

1. **Baseline identity.** "Latest baseline" was ambiguous and nobody was reachable, so arm A is
   `a05f3ee81ff` — the branch's own reference point, which keeps these numbers comparable with
   every prior result on this campaign. A current `master` build would have been a different
   reference to which none of them transfer. Its sha256 matches the hash recorded for that commit
   in `tmp/chj_amac/bins/MANIFEST.tsv`, an independent origin.
2. **`projection_cost` cannot be split per side.** `HashJoinResultBuildOutputMicroseconds` and
   `HashJoinResultFilterLeftMicroseconds` — the events that would separate build-side from
   probe-side materialization — are **not registered in either binary**. Raw evidence:

       $ for b in <baseline> <candidate>; do for ev in HashJoinResultBuildOutputMicroseconds \
             HashJoinResultFilterLeftMicroseconds; do strings -a "$b" | grep -cx "$ev"; done; done
       0   0      # baseline
       0   0      # candidate

   Both are absent on **both** arms, so the decomposition is symmetric and `projection_cost` is a
   single **unsplit residual** covering build-side and probe-side materialization together. This
   report does not imply any per-side breakdown. `PartitionedHashJoinBuildMicroseconds` is absent
   too, confirming `partitioned_hash` exists on neither binary.
3. **ARM only.** Everything here is `aarch64`. AMD/x86 is out of scope and was not measured.
4. **The two counters that compose `probe_cost` do not bracket the same code on the two arms**, by
   the arms' own registered descriptions in `src/Common/ProfileEvents.cpp`:

       a05f3ee81ff  Lookup:   "time spent in HashJoin::joinScatteredBlock (the hash-map lookup …)"
       fa5667f2da7  Lookup:   "time spent in HashJoin::joinRoutedBlock (the routed hash-map lookup -
                               per-row findKey in the row's slot map or the AMAC find pass …)"
       a05f3ee81ff  Dispatch: "dispatching (scattering) the left block across shards"
       fa5667f2da7  Dispatch: "preparing the key columns and deriving the per-row slot ids …
                               the block itself is not scattered"

   This is precisely why the campaign verdicts the **sum** and not the parts: key preparation moved
   out of the baseline's lookup into the candidate's dispatch, and both live inside `probe_cost`, so
   the sum is comparable while `dispatch` and `lookup` individually are **not**. The raw dispatch
   and lookup columns are published for re-derivation, not for arm-to-arm comparison on their own —
   note baseline dispatch medians of 0.0–0.2 ms against candidate values two to three orders of
   magnitude larger. The top regressions were checked against this: they are not reattribution,
   because the measured probe total rose too.
5. **The cross-arm correctness oracle is established by the harness's code path, not by the JSONL
   field.** `_collect_algorithm_stats` writes the *shared* reference into both arms
   (in `measure_unit`'s `algo_result.update(row_count=…, checksum=…)`), so a scorer-side comparison of the two arms' recorded
   `(row_count, checksum)` **cannot** fail — the file is structurally incapable of expressing
   disagreement, and this campaign does not claim it as evidence. The real oracle is
   `measure_unit`'s `got != expected` check (the warmup and timed-run comparisons against the
   first successful result), which compares every
   warmup and every timed run of every arm against the first successful result and marks the arm
   `INVALID` on any mismatch. `status: OK` on all 368 scored tier-a units is therefore sound
   evidence that the oracle held — and it is the only oracle available, since
   `expected_rows_closed_form` is null for real-suite units. Six scored units join to **zero** rows,
   so their oracle is `(0, 0)`: `job__movie_info_info_type_id__…filtered__{T16,T96}`,
   `job__person_info_info_type_id__…filtered__{T16,T96}`,
   `tpch__orders_o_custkey__customer_c_custkey__left_anti__{T16,T96}`. Four of them are counted as
   `probe_cost` losses. Scoring a materialization residual on a query that materializes nothing is
   defensible but is flagged here rather than left implicit.
6. **`system.build_options` GIT_HASH is stale on an incremental build** and must not be used to
   identify a binary. Arm B's server reported HEAD's *parent* `b425c810895`. Arm B's identity is
   instead established by four origins that would fail differently: its sha256; the presence of
   HEAD-only markers `joinHashRouteSlot`, `found_slot`, `route_shift` in arm B and their absence
   in both the parent build and the baseline; a clean `git status` plus a no-op `ninja clickhouse`;
   and — strongest, because it is recorded per timed run by the harness itself — the **running
   process's own exe hash** on every row of every fleet sweep:

       arm baseline   binary_sha256 0d32ef1c96e6  proc_exe_sha256 0d32ef1c96e6  equal=True  1880 rows
       arm candidate  binary_sha256 83de80854708  proc_exe_sha256 83de80854708  equal=True  1880 rows

   See `WORKLOG.md` Iteration 4.

### Risk-accepted leads

- **LEAD (does not affect any verdict here).** On the jbmt *synthetic* venue the two arms
  materialize their own `bench.build_t`/`bench.probe_t` per cell, so they measure different
  physical table instances; this is the mechanism behind the red synthetic A/A. It is *reported*,
  not fixed, and it is why Unit 2 is UNSETTLED rather than reported with caveats. Owner: whoever
  next runs the jbmt synthetic suite as an A/B. Rationale for accepting: fixing it means
  counterbalancing jbmt's synthetic cells (four mirrored blocks per cell, as fleet_ab does),
  which is a harness redesign outside this run's budget.

---

## 2. Empirical noise floors from the A/A controls (G0-a)

Identical binary on both arms, full A/B machinery, per metric. **Floors are venue-specific and do
not transfer between venues.**

| Venue (suite) | cells scored | `probe_cost` floor | `projection_cost` floor | G0-a |
| --- | --- | --- | --- | --- |
| fleet_ab, 8× `m8g.24xlarge` | 9 | **9.86 %** (707,280 µs) | **0.38 %** (8,894 µs) | **exit 0** |
| jbmt synthetic, this host, 5 runs | 10 | 1.94 % (83,028 µs) | 4.46 % (92,740 µs) | **exit 1** (2 non-TIE) |
| jbmt synthetic, this host, 11 runs | 10 | 3.86 % (478,191 µs) | 5.73 % (73,566 µs) | **exit 1** (2 non-TIE) |
| jbmt real, this host, 11 runs | 10 | **1.13 %** (41,086 µs) | **1.97 %** (107,927 µs) | **exit 0** |

Two things worth stating plainly. First, the prompt expected `projection_cost` to be the noisier
metric because it is a residual; on the fleet the opposite holds (0.38 % vs 9.86 %), because the
residual tracks output row count, which is fixed per cell, while `probe_cost` carries the
scheduling variance of 96 probe threads. Second, a floor above 3 % is not by itself a failure —
the band is per cell, `max(3%, that cell's own spread)`, and on the fleet every A/A cell's own
spread covered its own delta, which is why G0-a passes there with a 9.86 % worst case.

---

## 3. Suite 1 — fleet_ab (Unit 1), the acceptance venue

94-cell default `parallel_hash` plan, 10 timed runs per arm per cell, four counterbalanced
server-lifecycle blocks per cell, swept twice: `--block-order abba` and `--block-order baab`.
78 of 94 cells scored in each order; the same 16 cells were voided by the harness in both.

### `probe_cost`

| | ABBA | BAAB |
| --- | --- | --- |
| verdicts | 78 | 78 |
| **WIN / TIE / LOSS** | **68 / 4 / 6** | **68 / 4 / 6** |
| aggregate | 1,791,720 ms → 1,160,569 ms (**−35.2 %**) | 1,789,555 ms → 1,160,384 ms (**−35.2 %**) |
| median per-cell delta | −35.5 % | −35.6 % |

### `projection_cost`

| | ABBA | BAAB |
| --- | --- | --- |
| verdicts | 78 | 78 |
| **WIN / TIE / LOSS** | **2 / 5 / 71** | **2 / 5 / 71** |
| aggregate | 281,523 ms → 356,689 ms (**+26.7 %**) | 281,729 ms → 356,727 ms (**+26.6 %**) |
| median per-cell delta | +24.4 % | +23.9 % |

The two block orders are independent origins for this result, and they agree cell by cell: G1-b
reports **0 disagreements on either metric** across all 78 cells.

### The headline, stated without netting

`phj-ph` HEAD makes the dispatch+lookup path substantially faster (68 of 78 cells win,
−35.2 % aggregate) and makes column materialization substantially slower (71 of 78 cells lose,
+26.7 % aggregate). **63 of 78 cells move in opposite directions on the two metrics.** Neither
result cancels the other and neither is reported as a headline on its own.

### `probe_cost` — full LOSS list (6 cells, ABBA, with the BAAB cross-check)

Medians in ms. Raw components are the columns a reviewer needs to re-derive both metrics.

| cell | A | B | delta | band | dispatch A→B | lookup A→B | probe total A→B | BAAB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `key64:build.left_all.S3.T96.dup16` | 230.1 | 300.8 | +30.7% | 3.0% | 0.0 → 0.1 | 230.1 → 300.6 | 4,489.9 → 4,990.8 | LOSS +28.5% |
| `key64:build.inner_all.S3.T96.dup16` | 232.8 | 297.0 | +27.6% | 3.0% | 0.0 → 0.1 | 232.8 → 296.9 | 4,541.0 → 5,003.6 | LOSS +29.7% |
| `key64:probe.semi_anti.S4.T96.anti` | 8,565.3 | 10,104.1 | +18.0% | 3.0% | 0.1 → 12.4 | 8,565.3 → 10,090.5 | 8,677.6 → 10,194.6 | LOSS +18.3% |
| `key64:build.inner_all.S5.T96` | 6,744.2 | 7,412.0 | +9.9% | 3.0% | 0.2 → 3.6 | 6,744.0 → 7,408.4 | 8,062.7 → 8,870.0 | LOSS +12.0% |
| `key64:probe.asof.S4.T96` | 48,400.2 | 51,211.4 | +5.8% | 3.0% | 0.7 → 14.5 | 48,399.4 → 51,196.9 | 52,653.3 → 55,994.5 | LOSS +5.9% |
| `key64:probe.inner_all.S1.T96` | 2,395.7 | 2,531.5 | +5.7% | 3.0% | 0.7 → 6.7 | 2,395.0 → 2,524.8 | 4,588.3 → 4,909.3 | LOSS +4.2% |

Every `probe_cost` loss is a `key64` cell. The two `.dup16` build cells and the `asof`/`anti`
shapes are where the new hash route does not pay for itself.

### `projection_cost` — full LOSS list (71 cells, ABBA, with the BAAB cross-check)

The complete list with raw components is in `reports/fleet_abba_losses_projection.md` (all 71
rows, generated from `reports/fleet_abba.tsv`). The 20 worst:

| cell | A | B | delta | band | dispatch A→B | lookup A→B | probe total A→B | BAAB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `str:probe.any.S4.T96` | 1,384.9 | 2,085.5 | +50.6% | 3.5% | 0.2 → 7.0 | 26,055.4 → 11,397.5 | 27,441.3 → 13,503.5 | LOSS +50.1% |
| `str:probe.any.S2.T96` | 1,583.2 | 2,316.9 | +46.3% | 3.0% | 0.2 → 6.3 | 13,386.2 → 5,887.9 | 14,964.5 → 8,218.5 | LOSS +50.1% |
| `k256:probe.inner_all.S3.T96` | 2,142.2 | 3,053.1 | +42.5% | 3.0% | 0.4 → 10.3 | 22,740.3 → 8,370.4 | 24,885.9 → 11,436.2 | LOSS +43.0% |
| `str:probe.rf_all.S2.T96` | 7,045.8 | 10,019.7 | +42.2% | 3.0% | 3.0 → 7.6 | 15,380.0 → 7,812.4 | 22,445.5 → 17,881.1 | LOSS +42.4% |
| `str:probe.inner_all.S3.T96` | 2,053.0 | 2,817.9 | +37.3% | 3.0% | 0.3 → 6.6 | 17,465.2 → 9,014.6 | 19,519.3 → 11,843.1 | LOSS +37.3% |
| `k256:probe.inner_all.S4.T96` | 2,126.6 | 2,912.0 | +36.9% | 3.0% | 0.4 → 9.2 | 22,869.8 → 10,300.0 | 24,992.0 → 13,221.3 | LOSS +37.2% |
| `key32:probe.inner_all.S5.T96` | 18,825.3 | 25,504.9 | +35.5% | 3.0% | 1.9 → 57.0 | 67,731.9 → 53,089.0 | 86,543.1 → 78,705.0 | LOSS +35.6% |
| `str:probe.semi_anti.S4.T96` | 2,077.4 | 2,798.0 | +34.7% | 3.0% | 0.3 → 6.9 | 17,494.2 → 9,679.1 | 19,563.9 → 12,482.1 | LOSS +34.8% |
| `str:probe.inner_all.S4.T96` | 2,064.4 | 2,777.4 | +34.5% | 3.0% | 0.3 → 6.8 | 17,967.8 → 9,789.5 | 20,038.2 → 12,569.1 | LOSS +35.1% |
| `str:probe.left_all.S4.T96` | 3,535.3 | 4,750.3 | +34.4% | 3.0% | 0.2 → 6.9 | 19,313.7 → 10,417.2 | 22,850.2 → 15,174.5 | LOSS +34.3% |
| `key64:probe.inner_all.S4.T96` | 4,720.4 | 6,322.2 | +33.9% | 3.0% | 1.3 → 15.1 | 18,449.8 → 11,908.9 | 23,175.7 → 18,216.7 | LOSS +35.2% |
| `key64:probe.any.S2.T96` | 653.6 | 874.8 | +33.8% | 3.0% | 0.3 → 6.5 | 6,813.5 → 4,863.7 | 7,467.9 → 5,741.9 | LOSS +32.1% |
| `key64:probe.semi_anti.S4.T96` | 4,768.4 | 6,381.6 | +33.8% | 3.0% | 1.0 → 15.2 | 17,579.9 → 12,063.0 | 22,344.9 → 18,474.5 | LOSS +33.1% |
| `null64:probe.inner_all.S3.T96` | 2,572.1 | 3,430.8 | +33.4% | 3.0% | 0.9 → 7.5 | 9,983.3 → 5,973.2 | 12,559.5 → 9,417.3 | LOSS +33.3% |
| `key64:probe.left_all.S4.T96` | 5,925.4 | 7,898.4 | +33.3% | 3.0% | 1.1 → 14.9 | 18,057.3 → 12,071.3 | 23,980.9 → 19,932.0 | LOSS +31.8% |
| `key64:probe.inner_all.S5.T96` | 18,642.0 | 24,828.9 | +33.2% | 3.0% | 2.3 → 59.5 | 74,799.4 → 55,120.3 | 93,386.6 → 79,949.8 | LOSS +32.7% |
| `key64:probe.any.S4.T96` | 1,316.8 | 1,750.3 | +32.9% | 3.0% | 0.7 → 14.6 | 25,545.0 → 14,664.5 | 26,861.7 → 16,424.4 | LOSS +32.4% |
| `lcstr:probe.inner_all.S5.T96` | 15,597.6 | 20,712.6 | +32.8% | 3.0% | 0.8 → 8,769.4 | 209,930.7 → 96,318.5 | 225,510.8 → 125,538.1 | LOSS +32.4% |
| `null64:probe.inner_all.S5.T96` | 21,086.2 | 27,800.4 | +31.8% | 3.0% | 3.2 → 67.8 | 88,487.7 → 61,971.9 | 109,547.6 → 89,816.9 | LOSS +31.5% |
| `fixstr:probe.inner_all.S3.T96` | 2,344.1 | 3,088.0 | +31.7% | 3.0% | 0.6 → 6.9 | 13,219.5 → 9,803.2 | 15,559.9 → 12,897.4 | LOSS +32.3% |

All 71 also lose in BAAB. Exactly two cells win on `projection_cost`, and both are `.anti` shapes:

| cell | proj A | proj B | delta | BAAB | its `probe_cost` verdict |
| --- | --- | --- | --- | --- | --- |
| `str:probe.semi_anti.S4.T96.anti` | 58.9 | 41.4 | −29.7% | WIN −30.1% | WIN −24.7% |
| `key64:probe.semi_anti.S4.T96.anti` | 111.9 | 90.7 | −19.0% | WIN −18.3% | **LOSS +18.0%** |

`key64:probe.semi_anti.S4.T96.anti` is the campaign's one cell that moves the *other* way — it wins
on `projection_cost` and loses on `probe_cost` — and it appears in both lists accordingly.

### Cells whose two metrics move in opposite directions (63)

Full list in `reports/fleet_abba.txt`. They are dominated by probe-side `inner_all`, `any`,
`semi_anti` and `left_all` shapes across every family, e.g.
`k256:probe.inner_all.S3.T96` is `probe_cost` WIN −63.1 % and `projection_cost` LOSS +42.5 %;
`str:probe.any.S4.T96` is WIN −56.2 % and LOSS +50.6 %. Each appears in both lists above, with
no netting.

### NO-VERDICT cells (16 of 94, identical in both orders)

Every one was voided by the harness itself on its sub-200 ms duration floor, printed with that
reason, never dropped and never counted as a tie. The reason string is the harness's own.

| cell | reason (harness `invalid_reason`) |
| --- | --- |
| `k256:build.inner_all.S2.T96` | below-duration-floor (arm A median 41.1 ms < 200 ms) |
| `k256:build.inner_all.S3.T96` | below-duration-floor (arm A median 74.3 ms < 200 ms) |
| `key64:build.inner_all.S2.T96` | below-duration-floor (arm A median 23.5 ms < 200 ms) |
| `key64:build.inner_all.S3.T48` | below-duration-floor (arm A median 58.3 ms < 200 ms) |
| `key64:build.inner_all.S3.T96` | below-duration-floor (arm A median 65.5 ms < 200 ms) |
| `key64:build.inner_all.S3.T96.statson` | below-duration-floor (arm A median 65.0 ms < 200 ms) |
| `key64:probe.inner_all.S3.T96.h05` | below-duration-floor (arm A median 122.4 ms < 200 ms) |
| `key64:probe.semi_anti.S2.T96.anti` | below-duration-floor (arm A median 36.5 ms < 200 ms) |
| `mixed:build.inner_all.S2.T96` | below-duration-floor (arm A median 38.6 ms < 200 ms) |
| `mixed:build.inner_all.S3.T96` | below-duration-floor (arm A median 84.4 ms < 200 ms) |
| `str:build.inner_all.S2.T96` | below-duration-floor (arm A median 28.8 ms < 200 ms) |
| `str:build.inner_all.S3.T96` | below-duration-floor (arm A median 55.3 ms < 200 ms) |
| `str:build.inner_all.S3.T96.dup16` | below-duration-floor (arm A median 126.2 ms < 200 ms) |
| `str:build.left_all.S3.T96.dup16` | below-duration-floor (arm A median 126.8 ms < 200 ms) |
| `str:probe.inner_all.S3.T96.h05` | below-duration-floor (arm A median 157.9 ms < 200 ms) |
| `str:probe.semi_anti.S2.T96.anti` | below-duration-floor (arm A median 65.9 ms < 200 ms) |

**Coverage, exactly and printed:** 94 cells attempted × 2 block orders = 188 cell-sweeps;
**78 scored per order**, 16 NO-VERDICT per order, for the reason above. `--expect-cells 94`
therefore exits 1 and G1 is reported RED rather than met by lowering N.

`--expect-cells` is a count, and a count can in principle be satisfied by the wrong units — the
independent verifier proved this by feeding the scorer 94 fabricated cell ids and getting exit 0.
That gap is closed separately by set equality on the units actually measured, which is green for
both orders:

    $ python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' \
        --arm-a baseline --arm-b candidate --metric both \
        --expect-unit-set reports/fleet_plan94.json:cell --expect-unit-set-seen
    unit set from reports/fleet_plan94.json:cell: expected 94, scored 94, missing 0, extra 0
      set equality: YES
    CHECK SUMMARY: PASS   → exit 0        (identical for the BAAB glob)

### Two honest caveats on the Unit 1 tallies

1. **The 68/4/6 and 2/5/71 tallies describe the 78 scorable cells, not the whole plan, and the
   excluded 16 lean slightly worse.** Scoring the floor-voided cells anyway — diagnostic only, and
   not legitimate, since the harness voided them — gives `probe_cost` 12 WIN / 4 LOSS, a 25 % loss
   rate against 7.7 % among the scored cells, with the losses again `key64`
   (`build.inner_all.S3.T96` +9.7 %, `.statson` +11.2 %, `S3.T48` +6.6 %,
   `probe.semi_anti.S2.T96.anti` +22.5 %). The exclusion is direction-blind and harness-owned, so
   it is not cherry-picking, but the published tally is a mildly optimistic view of the full plan
   and should be read that way.
2. **The A/A control bounded one shape, not all of them.** The 9 scored A/A cells are all
   `<family>:probe.inner_all.S2.T96`. The control never exercised `asof`, `semi_anti.anti`,
   `dup16`, `h05`, `statson`, `jun`, T1/T48 or S1/S3–S5 — and five of the six `probe_cost` losses
   live in shapes it never bounded. Verdicts are unaffected, because the band is computed per cell
   from that cell's own spread; but the §2 noise floors generalize less than their prominence
   suggests.

---

## 4. Suite 2 — jbmt legacy synthetic (Unit 2): NO RESULT, for want of measurement time

**No verdict is issued for any of the 347 legacy cells, on either metric — no legacy cell was
measured at all.** This is a deliberate null result, not an omission, and §4.2 records that its
reason changed once a fourth A/A control was run: the venue is measurable, the sweep time was not
available.

The A/A control for this suite is red. With the *same* baseline binary on both arms, over 10 units
spanning all ten key families, at 5 timed runs two units scored non-TIE (`projection_cost` WIN
−4.1 % and LOSS +4.5 % against a 3.0 % band); the sampling was then strengthened to 11 timed runs
and it remained red on **both** metrics (`probe_cost` LOSS +3.9 %, `projection_cost` LOSS +5.1 %).

**The cause is NOT diagnosed. An earlier draft of this report claimed a mechanism that the
independent verifier disproved, and the claim is withdrawn.** The withdrawn claim was that the two
arms measure different physical table instances because each server fills its own
`bench.build_t`/`bench.probe_t`. They do fill their own copies, but the copies come out
**byte-identical**: `prepare_cell` ends in `OPTIMIZE TABLE … FINAL` and no fill statement uses
`rand()`, so with deterministic input the merged parts match exactly. Raw refutation, from the
tables still resident on ports 9005 and 9007:

    port 9005  build_t all_1_2240_6  rows 128000000  bytes_on_disk 661087811  hash_of_all_files dc666fafd2ea8a604e529fe2d440896b
    port 9007  build_t all_1_2240_6  rows 128000000  bytes_on_disk 661087811  hash_of_all_files dc666fafd2ea8a604e529fe2d440896b
    $ diff -r -q <9005 probe_t part dir> <9007 probe_t part dir>   # 578 MB, separate data roots, separate UUIDs
    DIFF: NO DIFFERENCES (byte-identical part directories)

So the red synthetic A/A has an **unlocated cause**, and the discriminator this report previously
offered between the green real suite and the red synthetic suite ("the real suite's arms read the
same bytes") does not hold — both suites' arms read identical bytes.

Similarly, the order-effect hypothesis is **not refuted, only untested**. The comparison used
signed group means, which cancel opposite-signed deviations; on magnitudes the two groups separate
(11-run: mean |proj delta| 3.18 % for the four `lead=aaA` units vs 0.58 % for the six `lead=aaB`
units, and the top three magnitudes are all `lead=aaA`, combinatorially p ≈ 0.033). It is also
irreducibly confounded, because `lead_flip = crc32(unit_id) & 1` fixes the lead arm per unit, so
"lead=aaA" is a label for four particular units — three of them the smallest. Deciding it needs a
within-unit test that this harness cannot express without a code change.

What IS established: the asymmetry is **systematic and reproducible**, not sampling noise. Across
the two fully independent A/A sweeps (separate fills, 5 runs and 11 runs) the per-unit
`projection_cost` delta sign agrees for 8 of 10 units, with Pearson r = +0.872. **One** unit
(`D262144_K5_mb1_mp256…T16`) offends in both sweeps; the 11-run sweep's other offender is on the
other metric. An earlier draft said "the same two units offend both times", which was wrong. The
reproducibility claim rests on the sign agreement and the correlation, not on the offender set.

Since G0-a is red for this suite on the only venue available to it, a red gate stops the unit and
every legacy verdict would be untrustworthy at the declared band. **What would settle it, in cost
order:**
(a) **swap the arm→port assignment** and re-run the A/A (`--arm aaA=$BIN:9007 --arm aaB=$BIN:9005`
— a pure command-line change, ~25 min). If the deltas invert, the confound is a fixed per-server
offset and the remedy is requiring agreement under both assignments, exactly the ABBA/BAAB trick
this campaign already applies at the fleet level. **This was run — see §4.1.**
(b) apply a duration floor as fleet_ab does: one of the two offenders has an 8 ms median query,
which any sane floor voids;
(c) run the synthetic suite on the 8-shard fleet, whose A/A passes;
(d) a band pre-registered from a *frozen* per-shape A/A on this venue — not honest if derived
after the fact from the A/A that failed.

A previous draft claimed the fix required "a harness redesign outside this run's budget". That was
wrong, and (a) and (b) are cheap; (a) was therefore executed rather than asserted away.

### 4.1 The port-swap control was run, and it refutes the fixed-offset explanation

    $ python3 -u join_bench_mt.py sweep --suite synthetic --shards 1 --shard 0 \
        --results results/aa_jbmt11_swap/results.jsonl --only "$(cat logs/aa_jbmt_regex.txt)" \
        --algorithms parallel_hash --min-timed-runs 11 \
        --arm aaA=<baseline>:9007 --arm aaB=<baseline>:9005      # arm->port assignment REVERSED
    $ python3 probe_ab_report.py --results 'results/aa_jbmt11_swap/results.jsonl' \
        --arm-a aaA --arm-b aaB --metric both --aa-control
      probe_cost:      10 scored, 1 non-TIE, empirical noise floor = 5.22% (143,633 us)
        FAIL D65536_K0_mb1_mp16_h1.0_bp8_pp8_T16: LOSS +5.2% (band 3.6%)
      projection_cost: 10 scored, 0 non-TIE, empirical noise floor = 4.05% (86,396 us)
    CHECK SUMMARY: FAIL (1 failed check(s))   → exit 1

Two things follow, both of which narrow the diagnosis rather than assert one:

1. **It is not a fixed per-server or per-port offset.** A fixed offset would have *inverted* with
   the same magnitude when the arms swapped ports. Instead the magnitudes largely **collapsed**
   (`D262144_K5…mp256`: `projection_cost` +5.13 % → +0.49 %; `D32000000_K4`: +0.87 % → −0.07 %),
   and a different unit became the offender. So the earlier suspicion of a per-instance bias is
   refuted, and with it the remedy that would have followed from it.
2. **The suite is still red on a quiet host, and the offender is now the micro-unit.**
   `D65536_K0_mb1_mp16` has an **8 ms** median query — at that duration a few per cent is a couple
   of hundred microseconds. This sweep also ran with no fleet sweep being orchestrated from this
   host, which the earlier 11-run control had running concurrently; that concurrency is the most
   plausible cause of the *earlier* magnitudes, and it is a LEAD, not a settled claim.

### 4.2 The confound in 4.1, and the fourth control that removes it — Unit 2's reason CHANGES

The third verifier pass pointed out that the swapped-port run changed **two** variables, not one: the
unswapped 11-run control had been executing while the fleet BAAB sweep was being orchestrated from
this host, whereas the swapped run came after teardown on a quiet host. So "refutes a fixed per-port
offset" was overstated — host quietness explained the collapse equally well.

A fourth control settles it, and it was cheap (~25 min, same command, no code change): **the
unswapped assignment re-run on the quiet host**, so that the only difference from the swapped run is
which arm sits on which port.

    $ python3 -u join_bench_mt.py sweep --suite synthetic --shards 1 --shard 0 \
        --results results/aa_jbmt11_quiet/results.jsonl --only "$(cat logs/aa_jbmt_regex.txt)" \
        --algorithms parallel_hash --min-timed-runs 11 \
        --arm aaA=<baseline>:9005 --arm aaB=<baseline>:9007
    $ python3 probe_ab_report.py --results 'results/aa_jbmt11_quiet/results.jsonl' \
        --arm-a aaA --arm-b aaB --metric both --aa-control
      probe_cost:      10 scored, 1 non-TIE, floor 5.01%   FAIL D65536_K0_mb1_mp16…: LOSS +5.0% (band 4.6%)
      projection_cost: 10 scored, 0 non-TIE, floor 2.70%
    CHECK SUMMARY: FAIL (1)   → exit 1

All four synthetic A/A sweeps, `probe_cost` / `projection_cost` delta per unit:

| unit | 5-run, busy | 11-run, busy | 11-run swapped, quiet | 11-run unswapped, quiet |
| --- | --- | --- | --- | --- |
| `D65536_K0_mb1_mp16…T16` (8 ms query) | −0.75 / +3.36 | −0.33 / +5.73 | **+5.22** / +4.05 | **+5.01** / +2.70 |
| `D262144_K5_mb1_mp256…T16` | +0.12 / +4.46 | +2.37 / **+5.13** | −0.40 / +0.49 | +0.56 / +2.46 |
| `D32000000_K6_mb16_mp1…T96` | −0.18 / −0.31 | **+3.86** / +0.32 | +1.15 / −0.37 | +2.51 / +0.24 |
| worst \|delta\| in the sweep | 1.94 / 4.46 | 3.86 / 5.73 | 5.22 / 4.05 | 5.01 / 2.70 |

**Now the port hypothesis really is refuted, without the confound.** The two quiet runs differ only
in the arm→port assignment, and the offending unit's delta keeps the **same sign and nearly the same
magnitude** (+5.22 % and +5.01 %). A fixed per-port offset would have inverted it. The offender is a
property of that unit — an **8 ms query**, where 5 % is about 400 µs — not of the ports, not of the
arms, and not of table construction.

**And with fleet_ab's own 200 ms duration floor, the quiet venue passes.** Applying that pre-existing,
direction-blind rule to all four sweeps:

| A/A sweep | units dropped by the 200 ms floor | worst remaining \|delta\| among KEPT units | G0-a at a 3 % band |
| --- | --- | --- | --- |
| 5 runs, busy | 2 (9 ms, 73 ms) | `probe_cost` 1.94 %, `projection_cost` **4.46 %** | **fails** |
| 11 runs, busy | 2 (8 ms, 73 ms) | `probe_cost` **3.86 %**, `projection_cost` **5.13 %** | **fails** |
| 11 runs, swapped, quiet | 2 (8 ms, 73 ms) | `probe_cost` 2.02 %, `projection_cost` 1.31 % | **passes** |
| 11 runs, unswapped, quiet | 2 (8 ms, 73 ms) | `probe_cost` 2.51 %, `projection_cost` 2.46 % | **passes** |

**Unit 2's reason therefore changes, and the earlier framing was too flattering to this run.** This
report previously said Unit 2 was "blocked on validity". On the evidence above that is not right:
the synthetic venue is measurable to the campaign's 3 % band provided (a) nothing else is running on
the host and (b) micro-units below fleet_ab's own 200 ms floor are excluded — two conditions that are
pre-existing rules, not inventions, and that are now demonstrated in **both** port assignments.
What actually prevented Unit 2 was **time**: 347 legacy cells at the measured ~2 min per unit is
about **11.5 h** of sweep, which was not available alongside Unit 1's two fleet sweeps and Unit 3's
752 real units. Calling that a validity block made an unaffordable sweep look like a principled
refusal. The honest status is:

> **Unit 2 = NO RESULT for want of measurement time.** No legacy cell was measured, so no legacy
> verdict exists, and none is implied. The venue question is settled *favourably* (quiet host +
> 200 ms floor), the 347-id anchored `--only` regex is delivered
> (`logs/legacy_only_regex.txt`, 13,752 bytes), and the sweep is one command and ~11.5 h away.

The three earlier-listed remedies (fleet venue, counterbalancing, a frozen band) are therefore not
needed to make Unit 2 valid; only the sweep time is. Anything a follow-up run should carry over:
run it on an otherwise idle host, apply a 200 ms duration floor, and keep `--min-timed-runs 11`.

Coverage attempted: 0 of 347 measured cells scored. The `--only` regex that would select exactly
the 347 ids was built and is delivered (`logs/legacy_only_regex.txt`, 13,752 bytes, anchored
`^(id1|…|id347)$`), so the sweep is a single command away once the venue question is settled.

---

## 5. Suite 3 — jbmt real (Unit 3)

Real queries over the five loaded datasets (TPC-H, TPC-DS, JOB, StackOverflow, coffeeshop) on
MergeTree, 11 timed runs per arm per unit (raised from the harness default of 5 — see §4.2), strict
ABAB interleave with the lead arm flipped by `crc32(unit_id) & 1`, arm A = baseline on port 9005,
arm B = `phj-ph` HEAD on port 9006.

**Venue validity for this suite is established on the exact channel used**, not by analogy:

    $ python3 probe_ab_report.py --results 'results/aa_real_pair/results.jsonl' \
        --arm-a aaA --arm-b aaB --metric both --aa-control
      probe_cost:      10 scored, 0 non-TIE, empirical noise floor = 1.24% (25,560 us)
      projection_cost: 10 scored, 0 non-TIE, empirical noise floor = 1.59% (51,165 us)
    CHECK SUMMARY: PASS   → exit 0

That control ran the **baseline binary on both ports 9005 and 9006** — i.e. on arm B's own server
instance and data root, with the candidate temporarily swapped out and then restored — with each
port's running binary confirmed by hashing `/proc/<pid>/exe`. A per-instance or per-port bias in the
measured channel would have surfaced as a non-TIE cell; none did.

### Tier a — complete, 376 units attempted, 368 scored

The tables below are **generated from `reports/jbmt_real_a.tsv`** by
`make_report_tables.py`, not transcribed. Two earlier hand-written tables in this report were wrong
(a cell id that never existed in the plan, and a T16/T96 pair swapped), both caught by verifiers;
generating them removes that failure mode. Reproduce with:

    python3 make_report_tables.py reports/jbmt_real_a.tsv "Unit 3 tier a" --top 8

| | `probe_cost` | `projection_cost` |
| --- | --- | --- |
| verdicts | 368 | 368 |
| **WIN / TIE / LOSS** | **161 / 51 / 156** | **43 / 127 / 198** |
| aggregate | 230,541.8 ms → 210,259.1 ms (**−8.8 %**) | 2,284,639.8 ms → 2,347,582.1 ms (**+2.8 %**) |
| median per-unit delta | **+0.4 %** | **+4.5 %** |
| noise floor (G0-a, measured pair) | 1.24 % | 1.59 % |

**Recorded, never a verdict.** Wall clock and the probe total do not decide anything here — the two
metrics above do — but leaving them out of the report would make the `probe_cost` line read far
better than the measurement supports, so they are stated:

| measured quantity (same 368 units, median per arm) | arm A | arm B | delta |
| --- | --- | --- | --- |
| `ConcurrentHashJoinProbeMicroseconds` — the probe total the two metrics sum to | 2,515,295.9 ms | 2,557,844.4 ms | **+1.69 %** |
| wall clock (`query_duration_ms`) | 120,936.0 ms | 126,330.0 ms | **+4.46 %** |

Per-unit wall clock: **287 of 368 units slower**, 39 faster, 42 equal.

**Where the −8.8 % `probe_cost` aggregate comes from — it is not a broad improvement.**

| improving units | their d(`probe_cost`) | share of the net | their d(probe total) | their d(wall) |
| --- | --- | --- | --- | --- |
| top 5 | −10,499.7 ms | 51.8 % | **+3,413.9 ms** | **+646 ms** |
| top 20 | −20,636.9 ms | 101.7 % | **+4,794.8 ms** | **+1,108 ms** |

The twenty largest `probe_cost` improvements account for more than the entire net improvement, and
**as a group** they are worse on the probe total and on wall clock. Per unit the picture is mixed
and must be stated that way: of those twenty, **9 of 20 are slower on the probe total** (11 are
faster) and 12 of 20 are slower on wall clock, and the group's positive total is driven by three
TPC-H `lineitem` joins contributing +16,472.6 ms while the other seventeen net −11,677.8 ms. An
earlier draft said "those same units got slower", which is false for the majority and overstated
the regression; corrected here.

What survives that correction is the weaker but still material claim: the −8.8 % `probe_cost`
aggregate is not a broad improvement — it is concentrated in twenty of 368 units — and it does not
carry through to the probe total (+1.69 %) or to wall clock (+4.46 %, 287 of 368 units slower). The
`probe_cost` win is real as a phase measurement rather than a median artifact: for the largest
regressions and improvements alike the two arms' per-run distributions are **disjoint** with stable
leave-one-out deltas (`reports/top_regression_per_run.md`, e.g.
`tpcds__customer × catalog_returns T96` +211.7 % with a leave-one-out range of +208.1 %..+212.4 %).

**Two breakdowns that sharpen the picture.** Excluding units whose `probe_cost` median is under
50 ms on either arm (where a few percent is a few hundred microseconds) leaves 225 units and a
*better* `probe_cost` ratio — 117 WIN / 28 TIE / 80 LOSS at −8.9 % aggregate — so the improvement is
concentrated in substantial units and the small units contribute most of the loss count. Cutting the
other way, by shape: the `filtered` variants regress hard and systematically.

| subset | units | `probe_cost` WIN | `probe_cost` LOSS |
| --- | --- | --- | --- |
| `__filtered__` shapes | 72 | 11 | **56** |
| all other shapes | 296 | 150 | 100 |
| `probe_cost` median ≥ 50 ms on both arms | 225 | 117 | 80 |
| `probe_cost` median < 50 ms on either arm | 143 | 44 | 76 |

`projection_cost` regresses on real queries as well (198 losses against 43 wins, median +4.5 %), but
its aggregate regression is far smaller than on the synthetic cells (+2.8 % vs +26.7 %) because on
real data the residual is a much larger share of a much bigger probe total. **142 units move in
opposite directions on the two metrics.**

Worst `probe_cost` regressions (all 156 in `reports/jbmt_real_a_losses_probe.md`; full generated
tables with raw components in `reports/section5_tier_a.md`):

| unit | probe A (ms) | probe B (ms) | delta | band | dispatch A→B | lookup A→B | probe total A→B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__T96__tiera` | 20.9 | 65.0 | **+211.7 %** | 5.0 % | 0.0 → 0.1 | 20.9 → 65.0 | 487.8 → 530.1 |
| `job__movie_keyword_movie_id__title_id__filtered__T96__tiera` | 37.2 | 95.4 | **+156.8 %** | 17.3 % | 0.0 → 0.2 | 37.2 → 95.2 | 67.5 → 126.8 |
| `tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__T16__tiera` | 7.5 | 17.3 | **+129.6 %** | 19.8 % | 0.0 → 0.1 | 7.5 → 17.2 | 392.8 → 440.6 |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T96__tiera` | 764.1 | 1,741.7 | **+128.0 %** | 19.4 % | 0.0 → 0.3 | 764.1 → 1,741.5 | 1,572.4 → 2,715.6 |
| `job__movie_keyword_movie_id__title_id__T96__tiera` | 66.0 | 135.4 | **+105.2 %** | 9.4 % | 0.0 → 0.2 | 66.0 → 135.2 | 178.5 → 250.5 |
| `job__movie_keyword_movie_id__movie_companies_movie_id__T96__tiera` | 70.6 | 137.3 | **+94.5 %** | 8.8 % | 0.0 → 0.2 | 70.6 → 137.0 | 943.8 → 1,056.7 |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T16__tiera` | 147.2 | 281.5 | **+91.3 %** | 3.1 % | 0.0 → 0.2 | 147.2 → 281.3 | 673.9 → 1,110.6 |
| `job__movie_companies_movie_id__title_id__T96__tiera` | 79.7 | 133.0 | **+66.8 %** | 6.4 % | 0.0 → 0.2 | 79.7 → 132.8 | 166.4 → 225.2 |

Worst `projection_cost` regressions (all 198 in `reports/jbmt_real_a_losses_projection.md`):

| unit | proj A (ms) | proj B (ms) | delta | band | dispatch A→B | lookup A→B | probe total A→B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T16__tiera` | 525.6 | 829.8 | **+57.9 %** | 3.0 % | 0.0 → 0.2 | 147.2 → 281.3 | 673.9 → 1,110.6 |
| `tpcds__household_demographics_hd_income_band_sk__income_band_ib_income_band_sk__T96__tiera` | 0.1 | 0.1 | **+56.0 %** | 19.2 % | 0.0 → 0.0 | 0.1 → 0.1 | 0.2 → 0.2 |
| `tpcds__customer_address_ca_state__store_s_state__filtered__T16__tiera` | 6.7 | 9.9 | **+48.2 %** | 5.8 % | 0.0 → 0.0 | 0.9 → 1.2 | 7.6 → 11.2 |
| `tpcds__customer_address_ca_state__store_s_state__T16__tiera` | 12.3 | 17.0 | **+37.4 %** | 8.8 % | 0.0 → 0.0 | 1.0 → 1.5 | 13.4 → 18.4 |
| `job__movie_companies_movie_id__title_id__filtered__T96__tiera` | 28.8 | 39.5 | **+37.3 %** | 6.9 % | 0.0 → 0.2 | 41.4 → 53.6 | 70.0 → 92.9 |
| `tpcds__web_sales_ws_bill_customer_sk__customer_c_customer_sk__T16__tiera` | 1,605.7 | 2,192.7 | **+36.6 %** | 3.0 % | 0.0 → 1.2 | 273.6 → 215.6 | 1,879.3 → 2,409.5 |

The second row is a caution worth stating: some real-suite units are tiny in absolute terms (0.1 ms
of projection), so their relative verdicts rest on tens of microseconds. 12 scored units have a
`probe_cost` median under 1 ms and 55 under 10 ms. Their verdicts follow the campaign's band rule
correctly, but they should not be read as engineering-significant.

Opposite-direction units: all 142 in `reports/jbmt_real_a_opposed.md`.

**What the A/A control does and does not bound.** The measured-pair A/A is 10 plain-`INNER` units
whose `probe_cost` spans 26 ms – 7.1 s. The sweep contains 185 distinct join shapes, including 72
`filtered` variants and 8 non-`INNER` units, and 94 scored units sit below the A/A's smallest probe
scale. So the 1.24 % / 1.59 % floors bound *the two-instance channel on plain INNER joins at that
scale* — which is what closes the same-instance question — and they do **not** bound the `filtered`
shapes that carry 56 of the 156 `probe_cost` losses, nor the sub-26 ms units. For those, the
protection is the per-unit band `max(3 %, that unit's own spread)`, not the A/A. Stated because the
floors' prominence in §2 could otherwise imply more than they cover.

**Coverage, exactly and printed: 376 attempted, 368 scored, 8 NO-VERDICT.** `--expect-cells 376`
exits 1, so **G3 for tier a is RED at 368/376** and N was not lowered. Every gap is one unit skipped
before any timed run by the per-unit time budget described in §4.2, with the harness's own reason
recorded:

| unit | reason (`OVER_BUDGET`, first warmup vs 30 s budget) |
| --- | --- |
| `tpch__customer_c_nationkey__supplier_s_nationkey__T16__tiera` | arm candidate warmup 0 took 5xx s |
| `tpch__customer_c_nationkey__supplier_s_nationkey__T96__tiera` | arm candidate warmup 0 took 1xx s |
| `tpcds__catalog_sales_cs_bill_customer_sk__store_returns_sr_customer_sk__T16__tiera` | arm baseline warmup 0 took 206.0 s |
| `tpcds__catalog_sales_cs_bill_customer_sk__store_returns_sr_customer_sk__T96__tiera` | arm baseline warmup 0 took 63.x s |
| `tpcds__inventory_inv_item_sk__catalog_sales_cs_item_sk__T16__tiera` | arm baseline warmup 0 took 54.x s |
| `tpcds__store_sales_ss_item_sk__catalog_sales_cs_item_sk__T16__tiera` | arm candidate warmup 0 took 1xx s |
| `tpcds__store_sales_ss_item_sk__catalog_sales_cs_item_sk__T96__tiera` | arm candidate warmup 0 took **36.0 s** (the narrowest exclusion in tier a) |
| `tpcds__store_sales_ss_item_sk__web_sales_ws_item_sk__T16__tiera` | arm baseline warmup 0 took 75.x s |

These are the same units the prior campaign documented as inherently exceeding the harness's
hard-coded `max_execution_time = 600`; they would have produced no verdict in any case.

**Why these exclusions cannot have flattered arm B.** An earlier draft argued this from the 4/4
split of arms named in the reasons. That argument is wrong and is withdrawn: the arm named is always
the `crc32(unit_id)`-chosen **lead** arm, in 8 of 8 cases, so the split says nothing except that the
hash came out even. The correct argument is structural: the rule reads only the lead arm's warmup
wall clock, before any timed run exists, and which arm leads is a hash of the unit id — so the
exclusion cannot correlate with the *direction* of the effect. Confirmed independently from both
servers' own `system.query_log`, which shows the non-leading arm never ran those queries at all
(port 9005: 4 units, baseline only; port 9006: 4 units, candidate only). The one boundary case
(`store_sales × catalog_sales T96`, candidate **36.0 s** against a 30 s budget) would, had the hash
fallen the other way, most likely have *added* an arm-B regression rather than removed one. Tier b
has three such near-boundary exclusions rather than one (32.8 s candidate-led, 36.3 s and 38.7 s
baseline-led), which is stated here rather than left to the reader to find.

### Tier b — complete, 376 units attempted, 365 scored

Tier b was expected to be a labelled partial; it finished, so it is not. Generated from
`reports/jbmt_real_b.tsv` (`make_report_tables.py`); full tables in `reports/section5_tier_b.md`.

| | `probe_cost` | `projection_cost` |
| --- | --- | --- |
| verdicts | 365 | 365 |
| **WIN / TIE / LOSS** | **164 / 50 / 151** | **49 / 126 / 190** |
| aggregate | 493,179.2 ms → 449,846.8 ms (**−8.8 %**) | 1,867,031.9 ms → 2,054,156.9 ms (**+10.0 %**) |
| median per-unit delta | **−0.4 %** | **+4.0 %** |

| measured quantity — recorded, never a verdict | arm A | arm B | delta |
| --- | --- | --- | --- |
| `ConcurrentHashJoinProbeMicroseconds` | 2,359,996.8 ms | 2,504,216.7 ms | **+6.11 %** |
| wall clock (`query_duration_ms`) | 139,824.0 ms | 151,024.0 ms | **+8.01 %** |

Per-unit wall clock: **273 of 365 units slower**, 50 faster, 42 equal. The twenty largest
`probe_cost` improvements again account for more than the whole net improvement (101.1 %) while
those same units are **+73,052.7 ms on the probe total** and **+4,988 ms on wall clock**.

Tier b replicates tier a's shape — `probe_cost` near-balanced by count with a −8.8 % aggregate,
`projection_cost` losing about four to one — but **the two tiers' aggregates are not comparable as
printed**, because they cover different unit sets (368 vs 365). Three units scored in tier a and not
in tier b carry **59 % of tier a's `projection_cost` mass** and all three are near-zero-delta TIEs,
which drags tier a's aggregate down. Like for like, on the 365 units common to both tiers:

| aggregate over the 365 common units | tier a | tier b |
| --- | --- | --- |
| `projection_cost` | **+7.34 %** | +10.02 % |
| `probe_total` (recorded) | +4.17 % | +6.11 % |
| wall clock (recorded) | +7.49 % | +8.01 % |
| `probe_cost` | −8.85 % | −8.79 % |

So the honest statement is that tier b is **somewhat** worse on the residual (+10.0 % vs +7.3 %), not
three and a half times worse as the all-units figures (+10.0 % vs +2.8 %) would suggest.

**A mechanism this report previously offered is withdrawn.** An earlier draft said tier b is heavier
data "so the direction of that difference is what one would expect if the regression scales with
materialized output". Its own per-dataset data contradicts that: on the common units, three of the
five datasets move the *wrong* way between tiers, including TPC-DS, which is 189 of the 365 units.

| dataset (common units) | n | tier a `projection_cost` | tier b | change | data scale b/a |
| --- | --- | --- | --- | --- | --- |
| `coffeeshop` | 6 | −0.41 % | −0.78 % | −0.37 pp | 2.01× |
| `job` | 126 | +7.54 % | +7.21 % | −0.33 pp | **1.00×** |
| `stackoverflow` | 12 | +16.42 % | +19.19 % | +2.77 pp | 2.13× |
| `tpcds` | 189 | +5.57 % | +5.21 % | −0.36 pp | 2.19× |
| `tpch` | 32 | +16.74 % | +23.99 % | +7.25 pp | 3.06× |

The tier-to-tier move is therefore mostly a **re-weighting** of the mix — TPC-H scales 2.5× by scale
factor (`TIERS` in `join_bench_mt.py`: `tpch_sf` 40 → 100, `tpcds_sf` 32 → 64, `coffeeshop` 500m →
1b, `so_copies` 1 → 2) and is the suite's worst dataset for the residual, so its growing share pulls
the aggregate up — not a regression that grows with output size.

**And tier b is a weaker replication than "measured twice over" implies.** JOB has **no scale factor
at all** — it is absent from `TIERS` — so its 126 units (a third of the suite) run on **byte-identical
data in both tiers**, confirmed empirically (arm-A `projection_cost` mass 244.8 s vs 244.7 s, ratio
1.00×). For those units tier b is a repeat measurement on the same data rather than an independent
one. It is still a strong replication of the *effect* — but "independent" is the wrong word for a
third of it, and §1 is worded accordingly.

**Coverage: 376 attempted, 365 scored, 11 NO-VERDICT** — `--expect-cells 376` exits 1, so **G3 for
tier b is RED at 365/376**, quantified, N not lowered. All 11 gaps are `OVER_BUDGET` skips before any
timed run; the full list with reasons is in `reports/section5_tier_b.md`. Full tier-b loss lists are in
`reports/jbmt_real_b_losses_probe.md` (151 rows) and `reports/jbmt_real_b_losses_projection.md`
(190 rows), and the 149 opposite-direction units in `reports/jbmt_real_b_opposed.md`. Two of them are the same
`store_returns` pair that tier a also skipped, and several are TPC-DS `store_sales × {catalog,web}_sales`
item-key joins that the prior campaign likewise documented as exceeding `max_execution_time = 600`.

One tier-b unit (`tpch__customer_c_nationkey__supplier_s_nationkey__T16__tierb`) was first recorded
`INVALID` after 1,200.8 s — 600 s on each arm — because the original time box was checked *after* the
error branch, so a warmup that fails by exhausting `max_execution_time` escaped it. That was fixed
(the budget is now checked before the error branch) and the unit re-ran as `OVER_BUDGET` at 600.6 s,
i.e. one timeout instead of two. Both rows are in the results file; the scorer keeps the last
attempt per unit id. The pre-fix file is preserved as `results/jbmt_real_b/results.pre_timebox_fix.bak`.

### 5.2 Both tiers side by side

| | fleet_ab synthetic (78 cells) | real tier a (368 units) | real tier b (365 units) |
| --- | --- | --- | --- |
| `probe_cost` W/T/L | 68 / 4 / 6 | 161 / 51 / 156 | 164 / 50 / 151 |
| `probe_cost` aggregate | **−35.2 %** | −8.8 % | −8.8 % |
| `probe_cost` median unit | −35.5 % | +0.4 % | −0.4 % |
| `projection_cost` W/T/L | 2 / 5 / 71 | 43 / 127 / 198 | 49 / 126 / 190 |
| `projection_cost` aggregate, all scored units | **+26.7 %** | +2.8 % | **+10.0 %** |
| `projection_cost` aggregate, 365 units common to both tiers | n/a | **+7.3 %** | **+10.0 %** |
| probe total (recorded), common units | not aggregated¹ | +4.17 % | +6.11 % |
| wall clock (recorded), common units | not aggregated¹ | +7.49 % | +8.01 % |
| opposite-direction cells | 63 | 142 | 149 |

¹ fleet_ab's JSONL carries `duration_us` per run and the probe total per run, so these are
computable there too; they are not aggregated in this report for the fleet suite because that
suite's cells are synthetic shapes whose wall time is dominated by table fills rather than by the
join, which would make a wall-clock aggregate misleading rather than informative. The per-cell
probe-total columns are published for every fleet cell in `reports/fleet_abba.tsv`.

### 5.1 Decomposition and algorithm gates on the real suite (a second, non-vacuous origin)

    $ python3 probe_ab_report.py --results 'results/jbmt_real_a/results.jsonl' \
        --arm-a baseline --arm-b candidate --check-decomposition --check-path-event
      gather events absent on both arms => projection_cost is an unsplit residual
      rows checked: 8096   violations: 0
      timed runs checked: 8096   violations: 0
    CHECK SUMMARY: PASS   → exit 0

This matters beyond repeating Unit 1's result: jbmt records the **full** ProfileEvents map of every
timed run, whereas fleet_ab records a fixed 7-event subset. So on this suite the absence of
`HashJoinResultBuildOutputMicroseconds` / `HashJoinResultFilterLeftMicroseconds` is genuine runtime
evidence rather than a tautology of the harness's schema — which closes the "vacuous check" concern
the independent verifier raised about G0-b on fleet_ab input.

### 4.2 Deviation: 11 timed runs and a per-unit time budget (jbmt only)

Two changes were made to the campaign's own copy of `join_bench_mt.py`; neither touches how a
measured query runs, and both are disclosed here and in `WORKLOG.md`:

- `--min-timed-runs 11` raises each unit's timed-run count from the plan's 5. This was a response to
  the synthetic A/A being red at n=5 with heavily overlapping per-run distributions: the fix was
  more samples, not a wider band.
- `--unit-time-budget 30` skips a unit **before any timed run** if its first warmup exceeds 30 s,
  recording `OVER_BUDGET`. Without it, one tier-a unit (`tpch__customer × supplier` on
  `nationkey`, ~540 s per query) would have consumed ~4 h of a bounded run across its
  26 queries. The decision reads **wall clock only, never either metric**, so it cannot favour an
  arm — borne out by it firing on both arms roughly evenly.

---

## 6. Independent verification (doer ≠ grader)

Two refute passes were attempted before consolidation.

**Pass 1** (fresh subagent, `verifier` type) could not execute commands in this host — its shell
was refused — so it delivered a static review only. Its two usable leads (that `--expect-cells` is
count-only, and that an absent dispatch/lookup key is read as 0 µs) were both carried into pass 2
and are both now closed. That pass's verdict is not counted as verification, because a verifier
that cannot run the gates cannot verify them.

**Pass 2** (fresh subagent with shell, given only the prompt, the scorer and the evidence) ran a
full refutation. Verdict: **FIX-THEN-RESHIP**, with four blocking findings. **All four are fixed
and re-verified**; the fixes are in the commit "Fix defects the independent verifier found".

| # | Blocking finding | Status |
| --- | --- | --- |
| B1 | The stated mechanism for the red synthetic A/A was false: the two arms' `bench.*` parts are **byte-identical** (`hash_of_all_files` equal on separate data roots; `diff -r` clean over 578 MB). The real-vs-synthetic discriminator collapsed with it. | **Fixed** — claim withdrawn in §4, cause recorded as unlocated, raw refutation quoted. |
| B2 | A cheap checkable-but-unrun avenue existed for the UNSETTLED Unit 2: swap the arm→port assignment (no code change, ~25 min) to test whether the offset is a fixed per-server bias. Also: a duration floor would void one offender (8 ms median query). The claim that fixing it needed a harness redesign was unsupported. | **Fixed** — the port-swap control was **run**; see §4.1. |
| B3 | The order-effect "refutation" used signed group means, which cancel; on magnitudes the groups separate (p ≈ 0.033) and the hypothesis is confounded by `lead_flip = crc32(unit_id) & 1`. | **Fixed** — downgraded to "not tested" in §4 with the magnitudes and the confound stated. |
| B4 | `REPORT.md` named two `projection_cost` winners with wrong ids and wrong percentages, one of them a cell that **never existed in the plan**, while the real winners (both `.anti` shapes) went unmentioned — including the one cell that wins on `projection_cost` and loses on `probe_cost`. | **Fixed** — replaced with the scorer's actual output in §3. |

Non-blocking findings, all now addressed in this report or the scorer: `--compare-order` could not
go red while its exit code was cited as evidence (added `--fail-on-order-effect`, re-ran G1-b with
enforcement — still green); `--expect-cells` is count-only (added `--expect-unit-set-seen`, ran set
equality for both orders — green); "the cost gate" was cited loosely (now names the teardown log
and the independent inventory query); G0-b's gather-symmetry check is **vacuous on fleet_ab input**
because that harness writes a fixed 7-event map, so the binary-level grep is the real evidence for
the unsplit-residual assumption (stated in §1 and in `WORKLOG.md`); the A/A control covers one cell shape (§3 caveat
2); and the floor-voided cells lean slightly worse than the scored set (§3 caveat 1).

What pass 2 tried and could **not** refute, each re-derived independently of the scorer: every gate
exit code; every noise-floor figure to the digit; the decomposition identity and non-negativity
over all 3,760 rows with its own Python; no arm asymmetry in event-key presence; the gather events
absent from both binaries and from both commits' sources; all 16 NO-VERDICT cells having **zero**
valid rows with harness-owned reasons; pre-registration commits predating every measured sweep with
an append-only `WORKLOG.md`; the measured cell set equalling the regenerated 94-cell
`parallel_hash` plan and matching neither `hash_inband` nor `all`; `cell_axes.algo` =
`parallel_hash` on all 1,880 rows per sweep; every loss row in every report table verifying
numerically; and its own from-scratch fixtures confirming the two metrics are verdicted
independently. It also chased and dismissed a suspicion that the fleet numbers came from this host
(the `host` field records the driver, not the shard; 3.52 h of cell time in a 38.4 min window
implies 5.5× parallelism, impossible on one machine).

**Limits of the verification, stated plainly.** The fleet was already torn down when pass 2 ran, so
Unit 1 rests on the delivered JSONL: pass 2 attacked it for internal consistency and found nothing
inconsistent, but it could not re-measure a cell and so cannot exclude fabricated JSONL on evidence
alone. `/mnt/data/fleet_ab/fleet_ab.py` is not under version control at that path, so it could only
be shown to predate the campaign by mtime rather than proven unmodified.

**Pass 3** (fresh subagent, scoped to Unit 3 tier a) returned **FIX-THEN-RESHIP**. It re-derived
every tier-a figure independently and could not break one of them, and it confirmed two properties
nobody had checked: the strict ABAB interleave is real (reconstructed from `run_start_us`, which no
gate reads — 0 of 368 units non-alternating), and 31 units show strong within-run drift that the
interleave is exactly what neutralises. Its blocking findings, both fixed:

| # | Finding | Status |
| --- | --- | --- |
| C1 | §5 omitted the two directly measured quantities that make the `probe_cost` line readable in context — the probe total (+1.69 %) and wall clock (+4.46 %, 287 of 368 units slower). Neither is a verdict metric, but omitting them let a −8.8 % headline stand unqualified. | **Fixed** — both stated in §5, with the concentration analysis. |
| C2 | The tier-a regression table attributed +211.7 % to T16 and +129.6 % to T96; they are the other way round. | **Fixed** — and the root cause addressed: `make_report_tables.py` now generates every quotable table from the TSV. |

It also corrected two claims of mine that were wrong in the campaign's favour: the "4 baseline / 4
candidate" argument for the time box being direction-blind (the arm named is always the crc32-chosen
lead arm, so it argues nothing — replaced by the structural argument), and the cross-arm oracle being
"re-confirmed empirically from the JSONL" (the JSONL stores the shared reference in both arms, so
that comparison cannot fail — the oracle is the code path plus `status: OK`).

**Pass 4** (fresh subagent, full delivery) returned **FIX-THEN-RESHIP** with seven blocking findings.
It reproduced every headline number for both tiers under its own from-scratch scorer, confirmed both
earlier fix sets held, demonstrated the TSV-corruption fix is load-bearing (reverting it reproduces
382 rows and 9 malformed rows), and established that the fleet's ABBA and BAAB sweeps are genuinely
independent (disjoint nonces, non-overlapping time windows, 0 byte-identical event maps of 1,880
shared keys). All seven findings are fixed:

| # | Finding | Status |
| --- | --- | --- |
| D1 | §5.2 compared tier a's 368 units against tier b's 365; three tier-a-only units carry 59 % of tier a's `projection_cost` mass and are all near-zero TIEs, so the printed +2.8 % vs +10.0 % gap was largely a unit-set artifact. The offered mechanism ("the regression scales with materialized output") is contradicted by the per-dataset data — TPC-DS, 189 of 365 units, got *better*. And TPC-H scales 2.5× while **JOB does not scale at all**, so 126 of 376 units run on identical data in both tiers, which makes "measured twice over independently" too strong. | **Fixed** — like-for-like table added (+7.34 % vs +10.02 %), mechanism withdrawn with the per-dataset table, JOB non-scaling and TPC-H 2.5× disclosed, §1 reworded. |
| D2 | "Those same units got slower" was false for the majority: 9 of 20 (tier a) and 8 of 20 (tier b) are slower on the probe total; the group total is driven by three TPC-H `lineitem` outliers while the other seventeen net faster. The error ran *against* arm B. | **Fixed** — restated per unit with the outlier decomposition. |
| D3 | §4's "the same two units offend both times" — it is one unit. | **Fixed.** |
| D4 | A 36.0 s exclusion was printed as "3xx s" under a convention that preserves digit count, and "35.9 s" was unsupported. | **Fixed** with the recorded values; tier b's three near-boundary exclusions also disclosed. |
| D5 | The port-swap control changed two variables (assignment *and* host load), so "refutes" was overstated; the cheapest settling control — the unswapped run on a quiet host — was unrun. | **Fixed by running it**: §4.2. It removes the confound and changes Unit 2's stated reason from validity to time. |
| D6 | The self-test assertion count was given as 24 in one place and 33 in another; it is 32. | **Fixed** in both. |
| D7 | "not a measurement artifact — see the per-run separation evidence in §6" pointed at evidence that did not exist. | **Fixed** — evidence produced in `reports/top_regression_per_run.md`: the top regressions' per-arm per-run distributions are **disjoint** with stable leave-one-out deltas. |

Pass 4's non-blocking items are also fixed: the orphaned Unit 4 table row, assumption mis-numbering,
a wrong section pointer, brittle harness line citations replaced by function names, two stale figures
in the §4.1 floor table, and tier b's missing full loss lists and opposite-direction count (now 151,
190 and 149 rows in `reports/jbmt_real_b_*.md`).

Independence: **not degraded** — passes 2, 3 and 4 were fresh contexts that had not done the work,
were given the prompt plus the artifacts, and each produced findings that changed this report. Unit 0,
Unit 1 and Unit 3 were not self-passed. Pass 1 is excluded from that claim because it had no shell.

**What no verifier could check.** The fleet was torn down before pass 2 ran, so Unit 1 rests on the
delivered JSONL: three passes attacked it for internal consistency (nonce disjointness, disjoint time
windows, `proc_exe_sha256` on all 3,760 rows, plan set equality, 5.5× implied parallelism impossible
on one host) and found nothing inconsistent, but no amount of internal consistency excludes fabricated
JSONL. `/mnt/data/fleet_ab/fleet_ab.py` is not under version control at that path, so it could only be
shown to pre-date the campaign by mtime. Pass 4 could not re-verify the AWS teardown, since creating
or querying cloud resources was outside its brief; that gate rests on the two origins in §8.

---

## 7. Evidence matrix

Every invocation is copy-paste re-runnable from the campaign directory
`/mnt/ch/ClickHouse/tmp/probe_campaign_20260729/`. `→ exit N` is the recorded exit code.

| Criterion | Gate invocation (command) | Result (raw) | Non-gate sources (origins) | Verdict |
| --- | --- | --- | --- | --- |
| **G0-a** A/A control, fleet venue: identical binary both arms, ≥8 cells, every cell TIE on both metrics | `python3 probe_ab_report.py --results 'results/aa_fleet/results.shard*.jsonl' --arm-a aaA --arm-b aaB --metric both --aa-control` | `probe_cost: 9 scored, 0 non-TIE, floor 9.86% (707,280 us)` · `projection_cost: 9 scored, 0 non-TIE, floor 0.38% (8,894 us)` · `CHECK SUMMARY: PASS` **→ exit 0** | Both arms' `binary_sha256` and `proc_exe_sha256` = `0d32ef1c96e6…` on all 180 rows; `fleet/deployed.tsv`; verifier pass 2 re-ran it and reproduced every figure | **GREEN** |
| **G0-a power** — the control can go red | `python3 probe_ab_report.py --results 'results/aa_fleet/results.shard*.jsonl' --arm-a aaA --arm-b aaB --metric both --aa-control --band-override 0` | verifier pass 2: 18 failed checks, all 9 cells non-TIE on both metrics **→ exit 1** | `scorer_selftest.py` case 5 | **GREEN** |
| **G0-a** A/A control, jbmt **synthetic** venue | `python3 probe_ab_report.py --results 'results/aa_jbmt11/results.jsonl' --arm-a aaA --arm-b aaB --metric both --aa-control` | `probe_cost: 1 non-TIE, floor 3.86%` · `projection_cost: 1 non-TIE, floor 5.73%` · `CHECK SUMMARY: FAIL (2)` **→ exit 1** | Same result at 5 runs (`results/aa_jbmt/`, exit 1), an independent sweep: sign agrees 8/10 units, r = +0.872 | **RED — reported as RED**; Unit 2 UNSETTLED |
| **G0-a** A/A control, jbmt **real** venue | `python3 probe_ab_report.py --results 'results/aa_real/results.jsonl' --arm-a aaA --arm-b aaB --metric both --aa-control` | `probe_cost: 10 scored, 0 non-TIE, floor 1.13%` · `projection_cost: 10 scored, 0 non-TIE, floor 1.97%` · `PASS` **→ exit 0** | 10 units × 5 datasets × 2 thread ladders; port-swap control §4.1 | **GREEN** |
| **G0-b** decomposition: `probe_cost + projection_cost == ConcurrentHashJoinProbeMicroseconds`, residual ≥ 0, gather events absent-or-identical on both arms | `python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' --arm-a baseline --arm-b candidate --check-decomposition` (and the BAAB glob) | `rows checked: 1560   violations: 0` · `gather events absent on both arms => projection_cost is an unsplit residual` **→ exit 0** (both sweeps) | `strings -a` on both binaries: both gather events `0` occurrences; `git grep` at both commits finds neither event in `src/`; verifier pass 2's own Python over all 3,760 rows: 0 identity violations, 0 negative residuals, 0 rows missing a key, no arm asymmetry | **GREEN** (note: the gather-symmetry half is vacuous on fleet_ab's fixed 7-event map — the binary grep is the load-bearing origin) |
| **G0-c** only `parallel_hash` ran | `python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' --arm-a baseline --arm-b candidate --check-path-event` (and BAAB) | `timed runs checked: 1560   violations: 0` **→ exit 0** (both sweeps) | `cell_axes.algo` = `parallel_hash` on all 1,880 rows/sweep; 0 `.hash` ids; 0 `Partitioned*` events anywhere; measured set ≠ `hash_inband`, ≠ `all`; `PartitionedHashJoinBuildMicroseconds` absent from both binaries | **GREEN** |
| **G1** fleet_ab coverage, ABBA | `python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' --arm-a baseline --arm-b candidate --metric both --expect-cells 94` | `coverage: 78 cells with a verdict …, expected 94 (total cells seen: 94)` · `CHECK SUMMARY: FAIL (1)` **→ exit 1** | All 16 gaps carry the harness's own `below-duration-floor` reason (`fleet_ab.py:420,1379-1390`, pre-dating the campaign); verifier: all 16 have **zero** valid rows | **RED — 78/94, quantified, N not lowered** |
| **G1** fleet_ab coverage, BAAB | same with `results/fleet_baab/results.shard*.jsonl` | `coverage: 78 … expected 94` · `FAIL (1)` **→ exit 1** | same 16 cells as ABBA | **RED — 78/94** |
| **G1-set** the measured cells really are the 94-cell plan (closes `--expect-cells` being count-only) | `python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' --arm-a baseline --arm-b candidate --metric both --expect-unit-set reports/fleet_plan94.json:cell --expect-unit-set-seen` (and BAAB) | `expected 94, scored 94, missing 0, extra 0` · `set equality: YES` **→ exit 0** (both) | Verifier regenerated the plan from the harness itself: `n = 94`, equal to both measured sets, unequal to `hash_inband` (12) and `all` (106) | **GREEN** |
| **G1-b** block-order effect | `python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' --compare-order 'results/fleet_baab/results.shard*.jsonl' --arm-a baseline --arm-b candidate --metric both` | `probe_cost: 78 cells …, 0 disagree` · `projection_cost: 78 cells …, 0 disagree` · `(empty list …)` **→ exit 0** | Tally identity across orders (68/4/6 and 2/5/71 both), aggregates −35.2 % / +26.7 % vs −35.2 % / +26.6 % | **GREEN** |
| **G1-b power** — it can go red | same plus `--fail-on-order-effect` | on real data **→ exit 0** (still green); on a fixture with a forced flip **→ exit 1** | `scorer_selftest.py` case 17; the verifier proved the un-enforced form could not fail | **GREEN** |
| **Regression gate** — no win reported without beating the band on THAT metric; every loss listed with raw components | `python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' --arm-a baseline --arm-b candidate --metric both --out-tsv reports/fleet_abba.tsv` | 6 `probe_cost` losses and 71 `projection_cost` losses listed with dispatch / lookup / probe-total; 63 opposite-direction cells listed in both | Verifier re-derived all tallies and verified all 26 in-report loss rows and all 71 rows of `reports/fleet_abba_losses_projection.md` (0 failures), including the BAAB cross-check column | **GREEN** |
| **Scorer power** — a genuine LOSS is reported on each metric independently, and every check can fail | `python3 scorer_selftest.py` | `scorer_selftest: PASS (0 case(s) failed)` — 32 assertions **→ exit 0** | Verifier read the self-test for tricks (it drives the real scorer as a subprocess and asserts on its TSV), ran it, and built its own from-scratch fixtures reaching the same conclusion | **GREEN** |
| **Cost gate** — tag-filtered EC2 inventory empty after the fleet | `RUN_TAG=fleet-ab-202607291848 fleet/teardown.sh` then the independent query in §8 | `instances by RUN_TAG (want empty): <empty>` · `volumes … <empty>` · `sgs … <empty>` · `TEARDOWN COMPLETE 2026-07-29T20:19:38Z` **→ exit 0** | Independent `aws ec2 describe-instances/volumes/security-groups` by tag value returned empty, and the only running instance in the region is this orchestration host | **GREEN (two origins)** |
| **G2** legacy coverage, 347 cells with set equality | `python3 probe_ab_report.py --results 'results/jbmt_legacy/results.jsonl' --arm-a baseline --arm-b candidate --metric both --expect-cells 347 --expect-unit-set jbmt/join_bench_mt_legacy_cells.json:cell_id` | **NOT RUN — there is no legacy results file to run it against**, because no legacy sweep was measured (~11.5 h unavailable) | The 347-id anchored `--only` regex is built and delivered (`logs/legacy_only_regex.txt`, 13,752 bytes); `join_bench_mt_legacy_cells.json` verified to hold exactly 347 unique `cell_id`s; the venue question is settled favourably by the four A/A controls in §4.1–4.2 | **NO RESULT — measurement not taken; gate unrunnable, not weakened** |
| **G3** real coverage, tier a | `python3 probe_ab_report.py --results 'results/jbmt_real_a/results.jsonl' --arm-a baseline --arm-b candidate --metric both --expect-cells 376` | `coverage: 368 cells with a verdict …, expected 376 (total cells seen: 376)` · `FAIL (1)` **→ exit 1** | All 8 gaps are `OVER_BUDGET` with the harness's recorded reason, and the rule fired on arm baseline for 4 and arm candidate for 4 — direction-blind in practice; the same units are the ones the prior campaign documented as exceeding `max_execution_time = 600` | **RED — 368/376, quantified, N not lowered** |
| **G3** real coverage, tier b | `python3 probe_ab_report.py --results 'results/jbmt_real_b/results.jsonl' --arm-a baseline --arm-b candidate --metric both --expect-cells 376` | `coverage: 365 cells with a verdict …, expected 376 (total cells seen: 376)` · `FAIL (1)` **→ exit 1** | All 11 gaps are `OVER_BUDGET` before any timed run; several are the same units tier a skipped and the prior campaign documented as exceeding `max_execution_time = 600` | **RED — 365/376, quantified, N not lowered** |
| **G0-b / G0-c** on real tier b | `python3 probe_ab_report.py --results 'results/jbmt_real_b/results.jsonl' --arm-a baseline --arm-b candidate --check-decomposition --check-path-event` | `rows checked: 8030   violations: 0` · `timed runs checked: 8030   violations: 0` · `PASS` **→ exit 0** | Full ProfileEvents map recorded per run, so gather-event absence is non-vacuous here too | **GREEN** |
| **G0-a** on the exact measured channel (9005 vs 9006, baseline on both) | `python3 probe_ab_report.py --results 'results/aa_real_pair/results.jsonl' --arm-a aaA --arm-b aaB --metric both --aa-control` | `probe_cost: 10 scored, 0 non-TIE, floor 1.24%` · `projection_cost: 10 scored, 0 non-TIE, floor 1.59%` · `PASS` **→ exit 0** | Each port's running binary hashed from `/proc/<pid>/exe`, both `0d32ef1c96e6…`; candidate restored afterwards | **GREEN** |
| **B2 port-swap control** (verifier-mandated) | `--arm aaA=<baseline>:9007 --arm aaB=<baseline>:9005` then `--aa-control` | `probe_cost: 1 non-TIE, floor 5.22%` · `projection_cost: 0 non-TIE, floor 4.05%` · `FAIL (1)` **→ exit 1** | Compared against the unswapped quiet control below, which isolates the port variable | **RED — reported; sharpens Unit 2** |
| **B5 quiet unswapped control** (verifier-mandated; isolates port from host load) | `--arm aaA=<baseline>:9005 --arm aaB=<baseline>:9007` on a quiet host, then `--aa-control` | `probe_cost: 1 non-TIE, floor 5.01%   FAIL D65536_K0…: LOSS +5.0%` · `projection_cost: 0 non-TIE, floor 2.70%` · `FAIL (1)` **→ exit 1** | Same sign and magnitude as the swapped run (+5.01 % vs +5.22 %) ⇒ not a per-port offset; with fleet_ab's 200 ms floor both quiet runs pass at 3 % | **RED as run; establishes the venue IS measurable under the floor** |

---

## 8. Cost gate — raw proof that every fleet this task launched is gone

One fleet was launched: `RUN_TAG=fleet-ab-202607291848`, 8× `m8g.24xlarge` in `ap-south-2`, up
2026-07-29T18:48Z, torn down 20:19:38Z. Teardown used the guard pattern that fires only for a tag
this run's own launch created (the pre-existing tag is captured first and refused), so no other
campaign's fleet could be affected. A 10 h watchdog was armed at launch as a second safety net.

Origin 1 — `fleet/teardown.sh`, which re-queries the inventory after deleting and exits non-zero if
anything remains (`logs/fleet_teardown.log`):

    === TEARDOWN fleet-ab-202607291848 2026-07-29T20:18:15Z  ===
    --- BEFORE: live inventory by RUN_TAG (this is the proof's power to fail) ---
    instances: i-06d6419c2203bed10 i-09363263dcdb1e4e8 i-06199c22ca3aafe91 i-0b20d193b78a04feb
               i-0851bfb86daf48124 i-04673930737530300 i-0339fd309c7c61258 i-0827ec518b92989b4
    volumes:   vol-08b109a338f307933 vol-0373b1d7c68aa9476 vol-0b4689a47d970c4cf vol-06e795a6433cd7b19
               vol-0cf785a5205f33c61 vol-0a46cc464fcf93782 vol-0b22ce18f416b2649 vol-0ee9e8fef49fcd944
    sgs:       sg-03356cc73c56a7f2e
    --- terminating instances ---
    all instances terminated
    no volumes to delete
    deleted sg-03356cc73c56a7f2e

    === PROOF ===
    instances by RUN_TAG (want empty): <empty>
    volumes by RUN_TAG   (want empty): <empty>
    sgs by RUN_TAG       (want empty): <empty>
    TEARDOWN COMPLETE 2026-07-29T20:19:38Z

Origin 2 — an independent tag-filtered query, run separately from the teardown script:

    $ export AWS_PROFILE=Dev_AWS_Admin AWS_REGION=ap-south-2; TAG=fleet-ab-202607291848
    $ aws ec2 describe-instances --region ap-south-2 --filters "Name=tag-value,Values=$TAG" \
        --query 'Reservations[].Instances[?State.Name!=`terminated`].[InstanceId,State.Name]' --output text
    (empty)
    $ aws ec2 describe-volumes --region ap-south-2 --filters "Name=tag-value,Values=$TAG" \
        --query 'Volumes[].[VolumeId,State]' --output text
    (empty)
    $ aws ec2 describe-security-groups --region ap-south-2 --filters "Name=tag-value,Values=$TAG" \
        --query 'SecurityGroups[].[GroupId]' --output text
    (empty)
    $ aws ec2 describe-instances --region ap-south-2 \
        --filters 'Name=instance-state-name,Values=pending,running' \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType]' --output text
    i-00c8778b1ae41598f     m8g.24xlarge        # this orchestration host, pre-existing, not launched here

No EBS volume was ever created from snapshot `snap-021cbdc2484f86607`, because the real-suite data
was already present locally; so there is no volume to account for either.
