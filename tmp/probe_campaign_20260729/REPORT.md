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
| **Unit 0** — metrics, scorer, A/A control | **GREEN** (fleet venue) · **RED→documented** (jbmt synthetic venue) | G0-a/b/c all exit 0 on the fleet A/A; G0-a exits 1 on the jbmt *synthetic* venue at both 5 and 11 timed runs, and exits 0 on the jbmt *real* venue. Scorer delivered with a 24-assertion self-test proving every gate can go red. |
| **Unit 1** — fleet_ab, 94 cells × ABBA + BAAB | **GREEN except coverage** | G0-b, G0-c and G1-b exit 0 on both sweeps, and the fleet is torn down with empty-inventory proof from two independent origins. **G1 is RED**: 78 of 94 cells carry verdicts; the 16 others are NO-VERDICT with the harness's own `below-duration-floor` reason. `--expect-cells 94` was not lowered. Set equality of the *measured* 94 cells against the plan is separately green. |
| **Unit 2** — jbmt legacy, 347 cells | **UNSETTLED / NO RESULT** | Blocked on validity, not on effort: G0-a is red on the only venue available to this suite. No legacy `probe_cost` or `projection_cost` verdict is issued. Gap and settling evidence named in §6. |
| **Unit 3** — jbmt real, 376 units × tiers a, b | **see §5** | Venue validated (G0-a green, floors 1.13 % / 1.97 %). Tier a measured; tier b as labelled below. |
| **Unit 4** — consolidation | this document | |

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
4. **`system.build_options` GIT_HASH is stale on an incremental build** and must not be used to
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

## 4. Suite 2 — jbmt legacy synthetic (Unit 2): NO RESULT, on validity grounds

**No verdict is issued for any of the 347 legacy cells, on either metric.** This is a deliberate
null result, not an omission.

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
`projection_cost` delta sign agrees for 8 of 10 units, with Pearson r = +0.872, and the same two
units offend both times.

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
   (`D262144_K5…mp256`: `projection_cost` +5.13 % → +0.49 %; `D32000000_K4`: +0.87 % → −0.06 %),
   and a different unit became the offender. So the earlier suspicion of a per-instance bias is
   refuted, and with it the remedy that would have followed from it.
2. **The suite is still red on a quiet host, and the offender is now the micro-unit.**
   `D65536_K0_mb1_mp16` has an **8 ms** median query — at that duration a few per cent is a couple
   of hundred microseconds. This sweep also ran with no fleet sweep being orchestrated from this
   host, which the earlier 11-run control had running concurrently; that concurrency is the most
   plausible cause of the *earlier* magnitudes, and it is a LEAD, not a settled claim.

**Would fleet_ab's own 200 ms duration floor rescue the suite? Only partly — so no.** Applying that
floor to the three A/A sweeps already collected (an analysis of existing data, offered as a proposal
for a future run and explicitly **not** a retroactive pass for Unit 2):

| A/A sweep | units dropped by a 200 ms floor | worst remaining \|delta\| among KEPT units | would G0-a pass? |
| --- | --- | --- | --- |
| 5 timed runs | 2 (9 ms, 73 ms) | `probe_cost` 1.94 %, `projection_cost` **4.46 %** | **no** |
| 11 timed runs | 2 (9 ms, 74 ms) | `probe_cost` **3.86 %**, `projection_cost` **5.13 %** | **no** |
| 11 runs, swapped ports | 2 (8 ms, 74 ms) | `probe_cost` 2.02 %, `projection_cost` 1.31 % | yes |

So a duration floor rescues one sweep of three. **Unit 2 stays UNSETTLED**, and the honest
characterization is now this: on this host the jbmt *synthetic* suite's own A/A exceeds the
campaign's 3 % band in all three sweeps somewhere, the cause is **not** per-arm table construction
(the parts are byte-identical), **not** a fixed per-port offset (swapping did not invert it), and
**not** shown to be arm ordering (that test is confounded by design and was not run). Its per-unit
noise on this venue sits around 4–5.5 %. The settling path is a quiet fleet venue — where fleet_ab's
own A/A passes cleanly — combined with a duration floor to exclude the micro-units, plus the 11.5 h
of sweep time 347 cells at ~2 min each require. That is a follow-up run, not a caveat on this one.

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

| | `probe_cost` | `projection_cost` |
| --- | --- | --- |
| verdicts | 368 | 368 |
| **WIN / TIE / LOSS** | **161 / 51 / 156** | **43 / 127 / 198** |
| aggregate | 230,542 ms → 210,259 ms (**−8.8 %**) | 2,284,640 ms → 2,347,582 ms (**+2.8 %**) |
| median per-cell delta | **+0.4 %** | **+4.5 %** |
| noise floor (G0-a, measured pair) | 1.24 % | 1.59 % |

**Real queries tell a different story from the synthetic microbenchmarks, and it is much less
favourable.** On the fleet's synthetic cells `probe_cost` improved in 68 of 78 cells at −35.2 %
aggregate; on real queries the wins and losses are nearly balanced by count (161 vs 156), the median
unit moves +0.4 % — i.e. nowhere — and the −8.8 % aggregate is carried by a minority of large units
rather than by a broad improvement. `projection_cost` regresses on real queries too (198 losses
against 43 wins, median +4.5 %), but its aggregate regression is much smaller (+2.8 %) than on the
synthetic cells (+26.7 %), because on real data the residual is a far larger share of a much bigger
probe total. **142 units move in opposite directions on the two metrics.**

Worst `probe_cost` regressions (full list of all 156 in `reports/jbmt_real_a_losses_probe.md`):

| unit | delta | band |
| --- | --- | --- |
| `tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__…T16` | **+211.7 %** | 5.0 % |
| `job__movie_keyword_movie_id__title_id__filtered__T96__tiera` | +156.8 % | 17.3 % |
| `tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__…T96` | +129.6 % | 19.8 % |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T96__tiera` | +128.0 % | 19.4 % |
| `job__movie_keyword_movie_id__title_id__T96__tiera` | +105.2 % | 9.4 % |
| `job__movie_keyword_movie_id__movie_companies_movie_id__T96__tiera` | +94.5 % | 8.8 % |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T16__tiera` | +91.3 % | 3.1 % |
| `job__movie_companies_movie_id__title_id__T96__tiera` | +66.8 % | 6.4 % |

Worst `projection_cost` regressions (full list of all 198 in
`reports/jbmt_real_a_losses_projection.md`):

| unit | delta | band |
| --- | --- | --- |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T16__tiera` | +57.9 % | 3.0 % |
| `tpcds__household_demographics_hd_income_band_sk__income_band_ib_income_band_sk__…` | +56.0 % | 19.2 % |
| `tpcds__customer_address_ca_state__store_s_state__filtered__T16__tiera` | +48.2 % | 5.8 % |
| `tpcds__customer_address_ca_state__store_s_state__T16__tiera` | +37.4 % | 8.8 % |
| `job__movie_companies_movie_id__title_id__filtered__T96__tiera` | +37.3 % | 6.9 % |
| `tpcds__web_sales_ws_bill_customer_sk__customer_c_customer_sk__…` | +36.6 % | 3.0 % |
| `tpcds__catalog_sales_cs_ship_customer_sk__customer_c_customer_sk__…` | +35.4 % | 3.0 % |
| `tpcds__store_returns_sr_cdemo_sk__customer_demographics_cd_demo_sk__…` | +35.2 % | 3.0 % |

Opposite-direction units: all 142 in `reports/jbmt_real_a_opposed.md`.

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
| `tpcds__store_sales_ss_item_sk__catalog_sales_cs_item_sk__T96__tiera` | arm candidate warmup 0 took 3xx s |
| `tpcds__store_sales_ss_item_sk__web_sales_ws_item_sk__T16__tiera` | arm baseline warmup 0 took 75.x s |

These are the same units the prior campaign documented as inherently exceeding the harness's
hard-coded `max_execution_time = 600`; they would have produced no verdict in any case. The skip
rule fired on **arm baseline for 4 of them and arm candidate for the other 4**, which is direct
evidence that it is direction-blind in practice and not only in principle.

### Tier b — see §5.2

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
assumption 2 (stated in §1 and in `WORKLOG.md`); the A/A control covers one cell shape (§3 caveat
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

Independence: **not degraded** — pass 2 was a fresh context that had not done the work, was given
the prompt plus the artifacts, and produced findings that changed this report. Unit 0 and Unit 1
were not self-passed.

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
| **Scorer power** — a genuine LOSS is reported on each metric independently, and every check can fail | `python3 scorer_selftest.py` | `scorer_selftest: PASS (0 case(s) failed)` — 33 assertions **→ exit 0** | Verifier read the self-test for tricks (it drives the real scorer as a subprocess and asserts on its TSV), ran it, and built its own from-scratch fixtures reaching the same conclusion | **GREEN** |
| **Cost gate** — tag-filtered EC2 inventory empty after the fleet | `RUN_TAG=fleet-ab-202607291848 fleet/teardown.sh` then the independent query in §8 | `instances by RUN_TAG (want empty): <empty>` · `volumes … <empty>` · `sgs … <empty>` · `TEARDOWN COMPLETE 2026-07-29T20:19:38Z` **→ exit 0** | Independent `aws ec2 describe-instances/volumes/security-groups` by tag value returned empty, and the only running instance in the region is this orchestration host | **GREEN (two origins)** |
| **G2** legacy coverage, 347 cells with set equality | `python3 probe_ab_report.py --results 'results/jbmt_legacy/results.jsonl' --arm-a baseline --arm-b candidate --metric both --expect-cells 347 --expect-unit-set jbmt/join_bench_mt_legacy_cells.json:cell_id` | **NOT RUN** — no measured legacy sweep exists, because G0-a is red for this suite and a red gate stops the unit | The 347-id anchored `--only` regex is built and delivered (`logs/legacy_only_regex.txt`, 13,752 bytes); `join_bench_mt_legacy_cells.json` verified to hold exactly 347 unique `cell_id`s | **UNSETTLED / NO RESULT** — gap and settling evidence in §4 |
| **G3** real coverage, 376 units per tier | `python3 probe_ab_report.py --results 'results/jbmt_real_a/results.jsonl' --arm-a baseline --arm-b candidate --metric both --expect-cells 376` (and tier b) | see §5 | see §5 | see §5 |
