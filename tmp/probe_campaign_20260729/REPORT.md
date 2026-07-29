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

Coverage attempted: 0 of 347 measured cells scored. The `--only` regex that would select exactly
the 347 ids was built and is delivered (`logs/legacy_only_regex.txt`, 13,752 bytes, anchored
`^(id1|…|id347)$`), so the sweep is a single command away once the venue question is settled.

---

## 5. Suite 3 — jbmt real (Unit 3)

*(filled in below once the tier sweeps complete — see §5.1)*

---

## 7. Evidence matrix

*(see §7 table below)*
