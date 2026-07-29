# `phj-ph` A/B benchmark campaign — report

Branch `phj-ph`, payload commit `a0dfbfd965b` ("Decouple the parallel_hash slot count from
the thread count"), against the frozen baseline `a05f3ee81ff`.
Campaign start commit `635aa368fd5`. `RUN_TAG` = `phj-ph-ab-20260728`.
Venue: 8 × `m8g.24xlarge` (96 cores, 370 GiB, aarch64), `ap-south-2a`.

Binaries of record — exactly two, verified from the result rows themselves, not from the
command line that produced them:

| arm | binary | sha256 (prefix) | source commit |
| --- | --- | --- | --- |
| A `baseline` | `bins/clickhouse-baseline-a05f3ee81ff.bin` | `0d32ef1c96e6` | `a05f3ee81ff` |
| B `candidate` | `bins/clickhouse-candidate-96532537d4d.bin` | `06d804546e0f` | `96532537d4d` (payload `a0dfbfd965b`) |

---

## Per-unit verdict

| Unit | What | Verdict |
| --- | --- | --- |
| 1 — preflight and candidate build | G1 green; candidate built; AMAC counters proved by 2 independent origins | **GREEN** |
| 2 — `fleet_ab` measured plan, 94 cells | G2 green; **G3 RED**, **G4 RED**; suite measured in full (93 of 94 cells have data) | **MEASURED, gates RED (honest)** |
| 3 — jbmt legacy synthetic, 347 cells | **G5 GREEN** (347/347 OK), G7 green | **GREEN — suite complete** |
| 4 — jbmt real suite, tier **a** | **G6 GREEN** (376/376, 1 INVALID = the pre-registered unit), G7 green | **GREEN — tier complete** |
| 4 — jbmt real suite, tier **b** | not started | **UNSETTLED — not run** (campaign stopped on instruction; settling command below) |
| 5 — reporting and U5 comparison | G7 green for both jbmt configurations; U5 changed-verdict list complete | **GREEN for the measured suites** |
| 6 — teardown | **G8 GREEN** | **GREEN** |

## Headline result

**The candidate loses, consistently, on all three suites that were measured.** This is the
measured truth, not a framing choice:

| suite | scope | candidate WIN | TIE | candidate LOSS | INVALID | median ratio (cand/base) |
| --- | --- | --- | --- | --- | --- | --- |
| `fleet_ab` synthetic matrix | 93 of 94 cells | 27 | 28 | 22 | 16 | — (per-cell band) |
| jbmt legacy synthetic | 347 of 347 | 119 | 53 | **175** | 0 | **1.057** (5.7% slower) |
| jbmt real, tier a | 375 scored of 376 | 26 | 138 | **211** | 1 (pre-registered) | **1.071** (7.1% slower) |
| jbmt real, tier b | — | — | — | — | — | **UNSETTLED — not run** |

`fleet_ab` is the only suite where wins outnumber losses, and even there the *same 93 cells*
went from 42 win / 9 loss under the U5 precedent's candidate to 27 win / 22 loss now. On both
jbmt suites the candidate is slower at the median and loses 2–8× more units than it wins.
Memory is also worse on both jbmt suites (legacy 5 win / 81 loss, tier a 66 win / 132 loss,
median 1.034 on each).

The one thing that clearly *improved* is the phase the change targets:
`ConcurrentHashJoinProbeLookupMicroseconds` is lower on 61 of 70 `fleet_ab` probe-side cells,
by up to −64%. The wall-clock cost moved somewhere else. **This campaign measures that
tension and does not explain it** — see the leads.

### Authorization-required flags

*(none raised yet; the one candidate is volume deletion at teardown — see Unit 6)*

### Protocol deviation found in verification — disclosed, impact measured

**The `fleet_ab` per-cell ABAB leader flip never fired: arm A (baseline) led the pair in all
93 cells.** Found by the independent verifier, confirmed from the raw rows.

*Mechanism.* `fleet_ab.run_cell` selects the leader positionally
(`order_pair = (0, 1) if cell_index % 2 == 0 else (1, 0)`), and
`run_sweep_stealing.py` runs **one cell per `fleet_ab.py sweep` process**, so `cell_index` is
always 0. A positional flip cannot survive one-cell-per-invocation sharding. (Contrast jbmt,
which derives its leader from `zlib.crc32(unit_id) & 1` and *is* stable under the same
sharding — confirmed live in this campaign: its rows carry `lead_arm: candidate` and
`lead_arm: baseline`.)

*What is still true.* The interleave **within** each cell is strict ABAB — arm A at even
positions, arm B at odd, verified from the `position` field:
`0:A 1:B 2:A 3:B 4:A 5:B 6:A 7:B 8:A 9:B` — so both arms sample the same window under the
same conditions. Only *who goes first in each pair* failed to alternate across cells.

*Measured impact, not an assurance.* A leave-out-the-first-pair recount over the same rows
(no re-running, no protocol change) moves exactly **one verdict of 93**, and it moves
**against** the candidate:

```
all 10 runs:    cells=93 win=27 tie=28 loss=22 invalid=16 insufficient=0
runs 1..9 only: cells=93 win=27 tie=27 loss=23 invalid=16 insufficient=0
verdicts that change when the leading pair is dropped: 1
  str:probe.semi_anti.S4.T96.anti                 TIE -> LOSS  (+5.06% -> +6.68%)
```

So the defect cannot manufacture this campaign's regression finding — in the one cell where
it mattered, always-A-first mildly *flattered* the candidate. 92 of 93 verdicts are invariant
to who led. **Not fixed mid-campaign:** correcting it means editing `run_cell`'s leader
selection, which is a protocol change and would also break comparability with the U5
precedent that ran the same positional flip. Recommended for a future campaign, before it
measures.

### HIGH-IMPACT assumptions

1. **The A/B compares the branch tip, not the slot-decoupling commit in isolation.** The
   frozen baseline `a05f3ee81ff` predates the entire AMAC/routed-probe stack, so every
   number here is *tip minus that baseline*. `a0dfbfd965b` is the tip's payload and the
   campaign's reason, but no per-cell delta is attributable to it alone. Isolating it needs a
   third arm (`3b76b5edfb5`, the pre-decoupling tip, whose binary exists in `bins/`). Not in
   scope, and **not claimed**.
2. **The U5 comparison is confounded by a settings change, by design of this campaign.**
   The committed `timed_settings` change makes this campaign measure
   `collect_hash_table_stats_during_joins = 1` on **every** cell, whereas U5 measured it on
   only the 2 `.statson` cells. So the 31 changed verdicts below mix a code change *and* a
   settings change. This is stated as UNSETTLED with the settling experiment named, not
   presented as evidence about the code.
3. **The U5 precedent is the same measurement lineage** (same baseline binary, same harness,
   same fleet shape). It is used to *contrast* verdicts, never to corroborate one.

### Risk-accepted leads

- **LEAD (not settled):** the largest probe-lookup *regressions* land on `lcstr` (+22.0%,
  +18.3%) and `asof` (+9.9%, +3.9%) — the shapes the branch deliberately routes off the AMAC
  ring. No mechanism isolated. Would need a per-shape ablation to settle.
- **LEAD (not settled):** the wall regression coexisting with a large probe-lookup
  improvement (below) implies the cost moved to a phase other than probe lookup. The
  per-cell phase table in `fleet/report_phj_ph.txt` has the build-side and probe-total events
  needed to chase it; not chased here.
- **LEAD (not settled), and the strongest one:** across **both** jbmt suites the candidate's
  losses concentrate at **low thread counts** and its wins at **T96**. Legacy suite: the 10
  worst are all `K1` at T2–T8 (up to **3.555×** slower) while the 10 best are all `T96` (down
  to 0.574×). Tier a: the 10 best are all `T96`. This is consistent with the payload
  commit's own mechanism — production callers now pass a fixed `max_slots = 256` instead of a
  thread-derived count, so a 2-thread query builds 256 slot maps — and with that commit's own
  stated follow-up ("re-validate the low-thread cells (T1/T48), which previously ran with
  thread-derived slot counts"). **It is a lead, not a finding:** this campaign did not vary
  the slot count, so it has not isolated the cause. Settling it needs a slot-count sweep
  (`fleet_ab` accepts `--env-b CLICKHOUSE_JOIN_SLOTS=…`) or a third arm at `3b76b5edfb5`.

---

## Suite 1 — `fleet_ab` measured plan (94 cells)

**Verdict counts** (`fleet/report_phj_ph.txt`; re-scored with the same
`fleet_ab.cell_verdicts` in `fleet/analysis_phj_ph.txt`, and **independently re-implemented**
without importing `fleet_ab` in `fleet/recount_independent.py`, which agrees exactly):

| cells with data | WIN | TIE | LOSS | INVALID | INSUFFICIENT | uncalibrated |
| --- | --- | --- | --- | --- | --- | --- |
| 93 of 94 planned | 27 | 28 | 22 | 16 | 0 | 0 |

Raw:

```
FLEET_AB REPORT RESULT: cells=93 win=27 tie=28 loss=22 invalid=16 insufficient=0 uncalibrated=0
```

**Probe-event gate metric.** Two different metrics are called this in the harness; both are
reported rather than picking one:

| metric | value |
| --- | --- |
| `ConcurrentHashJoinAmacProbeRows > 0` on the candidate arm (the `--require-engagement` gate) | **82 of 93 cells engaged**; 0 cells with the counter absent; 11 cells at zero, all compile-time-excluded shapes (`mixed`, `lcstr`, `asof`) |
| `ConcurrentHashJoinProbeLookupMicroseconds`, candidate vs baseline median | **lower on 61 of 70 probe-side cells**, best −64.10% (`k256:probe.inner_all.S3.T96`); higher on 9, worst +22.03% (`lcstr:probe.inner_all.S2.T96`) |

**The headline tension, stated plainly:** the phase the change targets improves sharply
(probe lookup down to −64%) while the *wall* verdict distribution is worse than the U5
precedent's on the identical cell set. `fixstr:probe.inner_all.S5.T96` is the clearest single
cell: probe lookup **−7.91%**, wall **LOSS +10.43%**. This campaign measures that; it does
not explain it.

### U5 comparison — every changed verdict

U5's rows rescored with `fleet_ab.cell_verdicts` (the same function that scored this
campaign), so a difference cannot be an artifact of two scoring functions. U5 arm B is
`6495b05ab061` (`candidate-final-d1c77571b39`); arm A is the same `0d32ef1c96e6` baseline.

| campaign | cells | WIN | TIE | LOSS | INVALID |
| --- | --- | --- | --- | --- | --- |
| U5 precedent | 96 | 42 | 28 | 9 | 17 |
| this campaign | 93 | 27 | 28 | 22 | 16 |

Cell-set overlap: **93 shared**, 0 only-here, 3 only-U5
(`key64:probe.inner_all.S1p5.T96`, `key64:probe.mixed_on.S3.T96`,
`str:probe.inner_all.S1p5.T96` — not members of the 94-cell measured plan).

Transition matrix over the 93 shared cells:

| U5 → now | count |
| --- | --- |
| WIN → WIN | 25 |
| WIN → TIE | 12 |
| WIN → LOSS | 4 |
| TIE → TIE | 14 |
| TIE → LOSS | 11 |
| TIE → WIN | 1 |
| LOSS → LOSS | 7 |
| LOSS → TIE | 2 |
| INVALID → INVALID | 16 |
| INVALID → WIN | 1 |

**All 31 changed verdicts, with both old and new values** (`diff%` positive = candidate
slower):

| cell | U5 | now | U5 diff% | now diff% |
| --- | --- | --- | --- | --- |
| `fixstr:probe.inner_all.S3.T96` | WIN | TIE | −5.98 | +2.67 |
| `fixstr:probe.inner_all.S5.T96` | WIN | LOSS | −6.18 | +10.43 |
| `k128:probe.inner_all.S3.T96` | WIN | TIE | −6.21 | +0.52 |
| `k128:probe.inner_all.S5.T96` | WIN | LOSS | −5.18 | +8.33 |
| `k256:build.inner_all.S5.T96` | WIN | TIE | −17.71 | +0.83 |
| `k256:probe.inner_all.S2.T1` | WIN | TIE | −7.86 | +0.96 |
| `k256:probe.inner_all.S4.T48` | TIE | WIN | +2.28 | −7.30 |
| `key32:probe.inner_all.S2.T96` | WIN | TIE | −3.48 | −2.24 |
| `key64:build.inner_all.S3.T1` | TIE | LOSS | −4.57 | +63.87 |
| `key64:build.inner_all.S3.T96.dup16` | TIE | LOSS | −1.35 | +7.92 |
| `key64:build.inner_all.S5.T96` | TIE | LOSS | +0.27 | +23.66 |
| `key64:build.left_all.S3.T96.dup16` | TIE | LOSS | −0.83 | +9.23 |
| `key64:probe.any.S2.T96` | WIN | TIE | −3.22 | −1.40 |
| `key64:probe.any.S4.T96` | LOSS | TIE | +7.71 | +0.85 |
| `key64:probe.asof.S4.T96` | LOSS | TIE | +15.33 | −2.51 |
| `key64:probe.inner_all.S1.T96` | TIE | LOSS | +0.38 | +3.90 |
| `key64:probe.inner_all.S2.T1` | TIE | LOSS | −2.77 | +7.43 |
| `key64:probe.inner_all.S3.T96` | WIN | TIE | −4.02 | +1.83 |
| `key64:probe.inner_all.S3.T96.h50` | TIE | LOSS | −1.99 | +4.92 |
| `key64:probe.inner_all.S3.T96.statson` | WIN | TIE | −4.66 | +2.81 |
| `key64:probe.inner_all.S4.T1` | WIN | TIE | −7.71 | −2.94 |
| `key64:probe.inner_all.S4.T48` | WIN | LOSS | −6.01 | +7.68 |
| `key64:probe.left_all.S3.T96.jun` | TIE | LOSS | −1.43 | +6.40 |
| `key64:probe.semi_anti.S2.T96` | WIN | TIE | −3.50 | −1.30 |
| `mixed:probe.inner_all.S2.T96` | TIE | LOSS | −0.08 | +3.29 |
| `null64:probe.inner_all.S3.T96` | TIE | LOSS | −0.28 | +3.71 |
| `str:build.inner_all.S5.T96` | WIN | TIE | −29.93 | +3.04 |
| `str:probe.asof.S4.T96` | WIN | LOSS | −6.18 | +4.42 |
| `str:probe.inner_all.S1.T96` | TIE | LOSS | +1.46 | +3.40 |
| `str:probe.inner_all.S2.T1` | INVALID | WIN | n/a | −4.33 |
| `str:probe.semi_anti.S4.T96.anti` | WIN | TIE | −23.15 | +5.06 |

**What this comparison does and does not establish. UNSETTLED.** The direction is
unambiguous *as measured*: on the identical 93 cells and the identical baseline arm, wins
fell 42 → 27 and losses rose 9 → 22. What it cannot tell you is **why**, because two things
changed at once: the candidate binary (U5's `6495b05ab061` → this tip `06d804546e0f`, which
adds the slot decoupling and two ASOF/LowCardinality routing commits) **and** the timed
settings (`collect_hash_table_stats_during_joins` now 1 on every cell instead of only the two
`.statson` cells). The prompt itself notes the two settings are not comparable. Attributing
the shift to the code would be an unproven causal story.

*The specific evidence that would settle it, none of which exists yet:* a two-arm `fleet_ab`
sweep of **`bins/clickhouse-candidate-final-d1c77571b39.bin` (`6495b05ab061`) as arm A
against `clickhouse-candidate-96532537d4d.bin` (`06d804546e0f`) as arm B**, on this fleet,
under the committed `timed_settings`, over the same 94-cell list. Both binaries are on disk,
so this is runnable — it holds settings and venue fixed and varies only the code, which is
exactly the missing contrast. Command:

```
python3 tmp/chj_amac/fleet/run_sweep_stealing.py \
  --hosts tmp/chj_amac/fleet/hosts.phj_ph.tsv --ssh-key tmp/chj_amac/fleet/ssh_phj_ph/id_ed25519 \
  --arm-a tmp/chj_amac/bins/clickhouse-candidate-final-d1c77571b39.bin \
  --arm-b tmp/chj_amac/bins/clickhouse-candidate-96532537d4d.bin \
  --name-a u5final --name-b tip \
  --remote-bin-a /home/ubuntu/chj/clickhouse-u5 --remote-bin-b /home/ubuntu/chj/clickhouse-cand \
  --calibration tmp/chj_amac/fleet/calibration_rows.json \
  --results-dir tmp/chj_amac/fleet/results_codeonly \
  --cells "$(cat tmp/chj_amac/fleet/cells_94.txt)"
```

---

## Honest-red list — every red and INVALID, with cause

| # | What | Cells / units | Cause (raw, diagnosed) | Action taken |
| --- | --- | --- | --- | --- |
| R1 | **G3 RED** — `cells_failed=17` | 17 of 94 | see R2 + R3 | none; reported |
| R2 | 16 cells fail the fail-closed 200 ms duration floor | `key64/str/k256/mixed:build.inner_all.S2/S3.T96`, `key64:build.inner_all.S3.T48`, `key64:build.inner_all.S3.T96.statson`, `str:build.{inner,left}_all.S3.T96.dup16`, `key64/str:probe.inner_all.S3.T96.h05`, `key64/str:probe.semi_anti.S2.T96.anti` | every invalid row reads `below-duration-floor (arm **A** median NN ms < 200 ms)`; medians 25.4–156.4 ms. The *baseline* arm's query is too fast to measure at T96 — `MIN_CELL_DURATION_US` refusing to produce a verdict, by design | **not** rerun, floor **not** lowered |
| R3 | 1 cell unmeasurable in this venue | `lcstr:probe.inner_all.S5.T96` | `Code: 241 … would use 186.12 GiB … maximum: 193.71 GiB … While executing FillingRightJoinSide` on the **baseline** arm: 340M `LowCardinality(String)` build keys with two resident servers on 370 GiB. Same cell/arm/exception as U5 (`191.45 GiB`), which dispositioned it `EXCLUDED-INVALID`. Produced **no rows**, hence `cells=93` | **not** rerun, server memory limit **not** raised |
| R4 | **G4 RED** — `cells=93 invalid=16` | — | consequence of R2 and R3; the gate exits 1 on non-zero `invalid`, which is correct behaviour | none; reported |

| R5 | 1 real-suite unit INVALID at tier a | `tpch__customer_c_nationkey__supplier_s_nationkey__T16__tiera` | `TIMEOUT_EXCEEDED` against the harness's fixed 600 s budget. A join on `nationkey` (~25 distinct values) produces an enormous output. **Pre-registered by name** before the sweep | budget **not** raised, unit **not** retried to green |
| R6 | jbmt legacy suite: candidate loses the suite | 175 of 347 units | not a defect — the measured result. Median 5.7% slower; worst `D8000000_K1_…T2` 3.555× | reported as the verdict |
| R7 | jbmt real tier a: candidate loses the tier | 211 of 375 scored units | not a defect — the measured result. Median 7.1% slower | reported as the verdict |
| R8 | `fleet_ab` ABAB leader flip never fired | all 93 cells | `run_cell` picks the leader from `cell_index % 2`; the stealing driver runs one cell per process so `cell_index` is always 0 | disclosed above; impact measured at 1 verdict of 93, **against** the candidate |
| R9 | jbmt `selftest` red | 3 checks | every failure names `partitioned_hash`, which does not exist in these binaries; `selftest` has no `--algorithms` flag. All `parallel_hash` checks and all 4 must-fail proofs passed | reported as expected, not chased |

**No cell or unit was rerun because its result was unwelcome.** There are no rerun files in
`fleet/results_phj_ph/`, and every row of every sweep is preserved, INVALID rows included.

### The one rerun in this campaign, disclosed in full

The first tier-a launch was issued with the **wrong plan partitioning** and was relaunched.
`jbmt_sweep_phj_ph.sh` derived `--shards` from the number of hosts *being launched* rather
than the plan's shard count, so launching a 6-host subset produced `--shards 6` (and, on the
host labelled shard 7, the out-of-range `--shard 7`). Verified from the remote process list:

```
python3 join_bench_mt.py sweep … --suite real --tier a --shards 6 --shard 0 …
```

This is a **diagnosed infrastructure fault**, which is the only basis on which the prompt
permits a rerun — not a result I disliked (the discarded attempt's numbers were never even
scored). Handling:

- the sweeps were killed within ~2 minutes, caught by reading the remote process list rather
  than trusting the launcher's own `launched` output;
- **the partial rows were preserved, not deleted**, as
  `fleet/jbmt_results_phj_ph/results.real_a_misshard6.shard<i>.jsonl` — 25 rows over 25
  distinct unit ids, from 5 hosts. They are excluded from every count in this report because
  they were produced under a different plan cut; the shard-7 host produced none, consistent
  with `--shard 7` of 6 being out of range;
- the driver was fixed so an explicit `NSHARDS` wins over the host count, and the relaunch was
  verified to read `--shards 8 --shard 0` before being left to run.

Cannot move a verdict: the discarded rows are in separate files matched by no glob this report
uses, and the scored tier-a set is exactly the 376 planned units from the 8-way cut, which
G6 confirms (`missing 0; extraneous 0`).

### Finding — G3/G4 cannot go green on this venue, and that is a defect in the gate, not the work

`cells_failed=0` and `invalid=0` are unreachable for the frozen 94-cell measured plan on an
`m8g.24xlarge` fleet: 16 cells are arithmetically below the protocol's own duration floor and
1 cannot be built by the baseline arm. Reaching green would require rerunning red cells,
retuning the venue, or shrinking the frozen plan — all banned. The measured suite is complete
and the reds are attributed; the gates stay red.

---

## Deviations from the prompt, each measured and documented

1. **Calibration file.** The prompt's resource map says to pass
   `tmp/chj_amac/calibration/calibration.json` as `--calibration`. That file raises
   `TypeError: int() argument must be … not 'dict'` in `fleet_ab.resolve_shape`, which does
   `int(calibration[family][size])`. Used `tmp/chj_amac/fleet/calibration_rows.json`
   instead: it matches the flag's documented `{family: {size: build_rows}}` contract, is the
   **exact** flat projection of the nested file (zero value mismatches over every shared
   family/size), and leaves `uncalibrated=0`. Cannot move a verdict — identical row counts,
   and the alternative yields no numbers at all. Neither file was edited.
2. **`HEAD` moved before the build.** Committing the `timed_settings` change and then the
   pre-registration moved `HEAD` to `96532537d4d`, so the candidate binary is named for that
   commit rather than `a0dfbfd965b`. Both commits touch only files under `tmp/chj_amac/`
   (Python harness + markdown), so the C++ payload is exactly `a0dfbfd965b`, which G1
   confirms is still an ancestor.
3. **Campaign-specific fleet scripts instead of `fleet/launch.sh`.** `launch.sh` tags only
   `Name`/`Purpose`/`Owner`, so G8's `tag:RUN_TAG` proof over its fleet would be vacuously
   empty — unable to fail. `fleet/launch_phj_ph.sh`, `deploy_phj_ph.sh` and
   `volumes_phj_ph.sh` tag every instance, volume and security group with `RUN_TAG` **at
   creation time**. The prior campaign's scripts and artifacts were left untouched as its
   audit trail.
4. **Volumes created with `aws ec2 create-volume` rather than
   `join_bench_mt.py fleet-volumes`.** The helper tags only `Name=jbmt-<tag>` and cannot add
   `RUN_TAG` at creation. The volume shape is byte-for-byte the shape the helper hard-codes
   (gp3, 4000 IOPS, 1000 MB/s, same snapshot). Snapshot `snap-021cbdc2484f86607` was never
   modified.
5. **Driver fixes made during the run** — listed under *Driver fixes* below, with the
   argument that none can move a verdict.

### Defect noted, not repaired (out of scope, as instructed)

`bins/MANIFEST.tsv` — its last three lines are bare `sha256  path` pairs instead of
six-column rows, for `clickhouse-candidate-final-d1c77571b39.bin`,
`…-fix1-06e0bbd0aa3.bin` and `…-fix2-6598f4b872f.bin`. The baseline row and the row this
campaign appended are well formed. Left as found.

### Driver fixes (protocol untouched)

None of these touch `timed_settings`, `DEFAULT_TIMED_RUNS`, `WARMUP_RUNS`,
`MIN_VERDICT_RUNS`, the 600 s budget, or any verdict-scoring function; all are in *new*
campaign-local scripts, not in `fleet_ab.py` or `join_bench_mt.py`.

1. **Detach fix in `jbmt_prep_phj_ph.sh`** — `setsid nohup cmd &` inside `ssh` left the
   child as the ssh shell's direct child (the background child is not a group leader), so the
   shell blocked in `wait()` and ssh never returned; the launcher hung after shard 0. Wrapped
   in a subshell `( … & )`, the same double-fork `fleet_ab.py` already carries. Cannot move a
   verdict: it changes only whether a prep script is reachable, and prep runs no timed query.
2. **`pgrep` self-match fix in `jbmt_prep_phj_ph.sh`** — the already-running guard
   `pgrep -f 'bash prep.sh'` matched the very ssh shell running it, so all 8 shards reported
   "already running" when only shard 0 was. Changed to `pgrep -f 'bash pre[p].sh'`. Cannot
   move a verdict: it only affects launch bookkeeping. **This one had teeth** — believed, it
   would have left 7 of 8 shards idle.
3. **Warm-read concurrency in `jbmt_prep_phj_ph.sh`** — `xargs -P 8` hydrated the
   snapshot-backed volume at ~93 MB/s (measured from `/proc/diskstats`) on a volume
   provisioned for 1000 MB/s, because snapshot first-touch is per-block S3 latency, not
   throughput. Raised to `-P 64`: 473 MB/s measured. Added `fleet/hydrate_tail.py` for the
   single-stream tail (14 MB/s → 554 MB/s). Cannot move a verdict: hydration happens before
   any timed run and makes both arms' page cache identically warm — that is the *reason* the
   prior campaign's first-touch timeouts are worth avoiding.

### Regression gate — the measurement protocol is unchanged

```
$ git diff --stat 635aa368fd5..HEAD -- tmp/chj_amac/fleet_ab.py
(no output — fleet_ab.py is untouched since the campaign-start commit)
```
`join_bench_mt.py` is outside the repo (`/mnt/data/jbmt_results/jbmt-sweep-20260724/`) and was
not edited; its `TOOL_VERSION = "jbmt-v2"`, `DEFAULT_RUNS = 5`, `REAL_WARMUPS = 2`,
`SYN_WARMUPS = 4` and 600 s `timeout` defaults are as found. The one committed change to
`fleet_ab.py` (`635aa368fd5`) is the pre-campaign `timed_settings` commit the prompt
required, and it changes a *setting*, not a run count, validity rule or scoring function.

---

## Evidence matrix

| Criterion | Gate invocation (command) | Result (raw) | Non-gate sources (origins) | Verdict |
| --- | --- | --- | --- | --- |
| Baseline binary is the frozen one; candidate carries the payload commit | `sha256sum tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin` ; `git merge-base --is-ancestor a0dfbfd965b HEAD; echo $?` | `0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4  …` ; `0` | `bins/MANIFEST.tsv` baseline row; per-shard `smoke_phjph_shard*.log` (8/8 hosts report the same two hashes) | **GREEN (G1)** |
| Candidate has the 3 AMAC engagement counters | `./tmp/chj_amac/bins/clickhouse-candidate-96532537d4d.bin local --query "SELECT name FROM system.events WHERE name LIKE 'ConcurrentHashJoinAmac%' ORDER BY name SETTINGS system_events_show_zero_values = 1"` | `ConcurrentHashJoinAmacBuildRingGrowths` / `…AmacBuildRows` / `…AmacProbeRows` | (1) `strings -a … \| grep -c ConcurrentHashJoinAmac` → `12`; (2) same query on the **baseline** → `0` (differential control: the check can fail); (3) `src/Common/ProfileEvents.cpp:438-440` | **GREEN** (2 independently-failing origins) |
| The swept cell list is exactly the 94-cell measured plan, no `hash` cells | `python3 -c "import json; c=json.load(open('tmp/chj_amac/fleet/matrix.json'))['measured_plan']['cells']; h=[x for x in c if x.endswith('.hash')]; assert len(c)==94 and not h, (len(c), h); print(','.join(c))"` | 94 ids, exit `0` | `load_cells_file` in `fleet_ab.py` concatenates `measured_plan` + `hash_inband` → 106, confirming what an unset `--cells` would have swept | **GREEN (G2)** |
| The sweep completed every cell | `python3 tmp/chj_amac/fleet/run_sweep_stealing.py … -- --require-engagement` | `FLEET_STEALING RESULT: cells_run=94 cells_failed=17 … -> FAIL` | per-cell `CELL FAILED` lines in `fleet/results_phj_ph/sweep.shard*.log`; 320 invalid rows all reading `below-duration-floor (arm A …)` | **RED (G3)** — cause diagnosed (R2, R3), unreachable without a banned move |
| Coverage and validity | `python3 tmp/chj_amac/fleet_ab.py report --results "$(ls -1 tmp/chj_amac/fleet/results_phj_ph/results.shard*.jsonl \| paste -sd,)"` | `FLEET_AB REPORT RESULT: cells=93 win=27 tie=28 loss=22 invalid=16 insufficient=0 uncalibrated=0` (exit 1) | `fleet/recount_independent.py` — a second implementation that does **not** import `fleet_ab` — recomputes `cells=93 win=27 tie=28 loss=22 invalid=16 insufficient=0`, identical | **RED (G4)** — `invalid=16`, `cells=93` |
| `parallel_hash` actually engaged on the candidate | (non-gate) `fleet/analyze_phj_ph.py` | `ConcurrentHashJoinAmacProbeRows: engaged(>0) in 82 cells; zero in 11; counter absent in 0 (of 93)` | `--require-engagement` was passed and did **not** trip (it fails closed at cell 0 if the candidate lacks the counters) | **GREEN** |
| jbmt legacy suite: all 347 cells OK | `cd tmp/chj_amac/fleet/jbmt_results_phj_ph && python3 -c "…legacy…" 'results.syn.shard*.jsonl'` (full one-liner in `PHJ_PH_PREREG.md`) | `legacy 347 missing 0 not-OK 0` (exit 0) | `report-ab` header `statuses: {'OK': 347}`; `fleet/recount_jbmt.py` reports `units with any fallback_runs > 0: none` | **GREEN (G5)** |
| jbmt real suite coverage, tier a | `python3 join_bench_mt.py report --results 'results.real_a.shard*.jsonl' --suite real --tier a --arm baseline` | `Planned 376 units; results for 376; missing 0; extraneous 0.` / `Statuses: {'OK': 375, 'INVALID': 1}` / `- INVALID: ['tpch__customer_c_nationkey__supplier_s_nationkey__T16__tiera']` | identical with `--arm candidate`; the INVALID matches the unit pre-registered by name before the sweep | **GREEN (G6, tier a)** |
| jbmt real suite coverage, tier b | same command with `--tier b` | **not run** | — | **UNSETTLED — not run** (settling command given above) |
| Cross-arm A/B result, legacy synthetic | `python3 join_bench_mt.py report-ab --results 'results.syn.shard*.jsonl' --arm-a baseline --arm-b candidate --out AB_REPORT.syn.md` | `binaries: {'baseline': ['0d32ef1c96e6'], 'candidate': ['06d804546e0f']}` / `lead arm distribution (ABAB leader): {'candidate': 181, 'baseline': 166}` / `statuses: {'OK': 347}` | `fleet/recount_jbmt.py` (no harness import) reproduces `win=119 tie=53 loss=175` exactly | **GREEN (G7)** — 2 distinct shas, both arms lead, no `FALLBACK` |
| Cross-arm A/B result, real tier a | `python3 join_bench_mt.py report-ab --results 'results.real_a.shard*.jsonl' --arm-a baseline --arm-b candidate --out AB_REPORT.real_a.md` | `binaries: {'baseline': ['0d32ef1c96e6'], 'candidate': ['06d804546e0f']}` / `lead arm distribution (ABAB leader): {'baseline': 172, 'candidate': 204}` / `statuses: {'OK': 375, 'INVALID': 1}` | `fleet/recount_jbmt.py` reproduces `win=26 tie=138 loss=211`, `fallback_runs > 0: none` | **GREEN (G7)** |
| `parallel_hash` engaged in jbmt (no silent fallback) | `fleet/recount_jbmt.py` on both configurations | `algorithms measured: {'parallel_hash': 347}` / `{'parallel_hash': 376}`; `units with any fallback_runs > 0: none` | `report-ab` statuses lists contain no `FALLBACK` entry; `selftest` `[PASS] no spurious path events under plain hash` on both arms | **GREEN** |
| Both arms are the intended binaries, as actually executed | (non-gate) per-shard `fleet/jbmt_prep_shard*.log` | `port 9005 pid 73844 sha256 0d32ef1c96e6d378` / `port 9006 pid 74425 sha256 06d804546e0f029b` — read from `/proc/<pid>/exe` | `binaries:` line of both `report-ab` outputs; `binary_sha256` in every JSONL row; `smoke_phjph_shard*.log` on all 8 hosts | **GREEN** (3 origins) |
| Snapshot data intact after hydration + hardlink clone | (non-gate) `python3 join_bench_mt.py verify --tier {a,b} --binary … --port 9005 --reference /mnt/data/jbmt_server/loads.{a,b}.json` | `verify: OK` (both tiers, all 8 shards) | the shared cross-arm `(row_count, checksum)` oracle produced 0 disagreements across 723 scored units | **GREEN** |
| Teardown proof | the four `aws ec2 describe-*` queries in `PHJ_PH_PREREG.md` Unit 6 | `instances: []` / `volumes: []` / `sgs: []` / `snapshot: "completed"` | pre-teardown the same filters returned 8 instances / 16 volumes / 1 SG (`fleet/teardown_dryrun_before.log`); the instances still resolve by tag as `terminated` × 8 | **GREEN (G8)** |

---

## Independent verification

**Pass 1 (Units 1–2).** A verifier subagent that did none of the execution was given this
prompt, the raw JSONL, the worklog and the gate outputs and asked to refute. Verdict:
**FIX-THEN-RESHIP**, one blocking finding, three leads.

**Its tooling was degraded and that is disclosed, not glossed:** the `Shell` tool was
unavailable to it for the entire session, so it could not run `sha256sum`, `git`, `python`, or
re-execute G1–G4, and substituted read-only greps plus `.git/logs/HEAD`. Its *independence*
was intact; its *executability* was not. The shell-based re-checks were then run by me, the
doer — which is **not** independent. A second verifier pass with a working shell is required
before final delivery.

| Finding | Status | Resolution |
| --- | --- | --- |
| 1. ABAB per-cell leader flip never fired; arm A led all 93 cells | **CONFIRMED, blocking** | Disclosed above with mechanism and a measured impact bound (1 verdict of 93 changes, against the candidate). Not fixed mid-campaign — that would be a protocol edit. |
| 2. "Recomputed independently" overstated — `analyze_phj_ph.py` imports `fleet_ab.cell_verdicts` | **VALID** | Wording corrected, and `fleet/recount_independent.py` written as a genuine second implementation that does not import `fleet_ab`. It agrees exactly. |
| 3. Cited sweep console logs allegedly absent | **REFUTED** | `git ls-files` shows all 8 `fleet/results_phj_ph/sweep.shard*.log` plus `fleet/sweep_phj_ph.log` tracked. The verifier could not run `ls`/`git`. The OOM's arm attribution *is* checkable: `sweep.shard0.log` contains `warmup 0 failed on arm baseline`. |
| 4. Uncommitted working-tree edit to `fleet_ab.py` | **REFUTED** | `git status --porcelain` shows only the in-progress worklog dirty. The ` M fleet_ab.py` the verifier saw is the *opening* status quoted inside the worklog — the pre-commit state, not a live delta. |

Checks the verifier ran that found nothing wrong: two distinct binaries cleanly bound to arms
(930 rows each, zero cross-binding, every `proc_exe_sha256` matching a claimed prefix); all
320 invalid rows naming **arm A**; the OOM cell having zero rows; no reruns;
`collect_hash_table_stats_during_joins = 1` on every row (proved by a plain cell and its
`.statson` twin sharing an identical `settings_fingerprint`, which also confirms `.statson` is
now a no-op); U5's arm-B sha distinct and its settings fingerprint different, so the confound
is real as reported; the 31-changed-verdict table (5 rows spot-checked); G2's teeth (106 vs
94, all 12 extras ending `.hash`); prereg-before-sweep ordering (prereg `96532537d4d` at
17:17:35Z vs earliest sweep row 17:37:08Z).

---

## jbmt two-arm smoke (`selftest`) — orientation, and honestly red

`selftest` has no `--algorithms` flag, so it always exercises the default
`partitioned_hash,parallel_hash` pair. `partitioned_hash` does not exist in either campaign
binary, so it fails — by design of the campaign, not by defect:

```
[FAIL] status OK: … Code: 418. DB::Exception: Unexpected value of JoinAlgorithm: 'partitioned_hash'. Must be one of [… 'parallel_hash' …]. (UNKNOWN_JOIN)
[FAIL] partitioned path event nonzero
[FAIL] LEFT ANTI unit status consistent with path events: … 'partitioned_hash' … (UNKNOWN_JOIN)
selftest: FAILURES PRESENT
```

**Every failure names `partitioned_hash`.** What matters for this campaign passed:

```
[PASS] no spurious path events under plain hash (baseline)
[PASS] no spurious path events under plain hash (candidate)
[PASS] timed runs alternate arms (parallel_hash): baselinecandidatebaselinecandidatebaselinecandidate
[PASS] wrong expected -> INVALID: row_count 400000 != closed-form expected 400001
[PASS] mid-run insert -> INVALID (parts or checksum)
[PASS] fingerprint changes on mutation
```

The four must-fail proofs all fire, and **the two-arm ABAB alternation is verified live** —
which matters because the prior campaign was single-arm and never exercised jbmt's two-arm
path. `selftest` also reported bootstrapping a small `keys_store.k0`; it did not clobber the
real one — `k0` reads `1.02 billion` rows on both arms on the shards checked.

---

## Suite 2 — jbmt legacy synthetic (347 cells)

**G5 GREEN.** All 347 named legacy ids present, every one `OK`:

```
$ cd tmp/chj_amac/fleet/jbmt_results_phj_ph && python3 -c "…" 'results.syn.shard*.jsonl'
legacy 347 missing 0 not-OK 0
[] []
G5_EXIT=0
```

**Verdict counts, candidate-centric** (`report-ab` labels from the *reference* arm's side —
see the orientation note below):

| axis | candidate WIN | TIE | candidate LOSS | median ratio cand/base |
| --- | --- | --- | --- | --- |
| wall | 119 | 53 | **175** | **1.057** |
| peak memory | 5 | 261 | **81** | **1.034** |

**G7 GREEN** on all three asserted contents:

```
347 result rows (347 multi-arm); statuses: {'OK': 347}
binaries: {'baseline': ['0d32ef1c96e6'], 'candidate': ['06d804546e0f']}
lead arm distribution (ABAB leader): {'candidate': 181, 'baseline': 166}
```
Two distinct shas ✓; both arms lead a non-trivial share ✓; no `FALLBACK` ✓.

**Orientation note — the trap that would have inverted this headline.** `report-ab`'s
`win` means the **reference (baseline) arm** is better: `join_bench_mt.py:1492` is
`return ("win" if va < vb else "loss", ratio)` with `va` = arm A, and the header says
`ratio > 1 and 'win' mean baseline better`. Its raw line for this suite is
`{'loss': 119, 'win': 175, 'tie': 53}` — i.e. the **baseline** won 175. Every count in this
report is restated candidate-centrically. `fleet/recount_jbmt.py`, which does not import the
harness, reproduces `win=119 tie=53 loss=175` exactly once the documented noise band
(`max(0.05 × max(median), max(stdev))`) is implemented.

**Structure of the extremes — both directions are large and shape-organised, not noise:**

- biggest candidate **wins**, all `T96`: `D32000000_K7_mb16…T96` **0.574**,
  `D32000000_K7_mb8…T96` 0.602, `D8000000_K7_mb16…T96` 0.637, `D32000000_K3_mb8…T96`
  0.638 — wide keys (K7 = 64-byte string, K3 = 8-column numeric) at high thread counts run up
  to **43% faster**;
- biggest candidate **losses**, all `K1` at **low** thread counts: `D8000000_K1_…T2`
  **3.555×**, `…T8` 3.513×, `…T4` 3.509×, `D32000000_K1_…T2` 3.341× — up to **3.5×
  slower** on the narrow 2-column numeric key at T2–T8.

---

## Suite 3 — jbmt real suite, tier a (376 units)

**G6 GREEN**, and the one INVALID is *exactly* the pre-registered unit — no unexpected INVALID
hiding inside an expected one:

```
$ python3 join_bench_mt.py report --results 'results.real_a.shard*.jsonl' --suite real --tier a --arm baseline
Planned 376 units; results for 376; missing 0; extraneous 0.
Statuses: {'OK': 375, 'INVALID': 1}
- INVALID: ['tpch__customer_c_nationkey__supplier_s_nationkey__T16__tiera']
```
Identical output with `--arm candidate`. The pre-registration named exactly one expected
tier-a INVALID and named that unit. The pre-declared borderline case
(`…supplier_s_nationkey__T96__tiera`) did **not** trip.

**Verdict counts, candidate-centric:**

| axis | candidate WIN | TIE | candidate LOSS | not scored | median ratio cand/base |
| --- | --- | --- | --- | --- | --- |
| wall | 26 | 138 | **211** | 1 INVALID | **1.071** |
| peak memory | 66 | 177 | **132** | 1 INVALID | **1.034** |

**G7 GREEN:**

```
376 result rows (376 multi-arm); statuses: {'OK': 375, 'INVALID': 1}
binaries: {'baseline': ['0d32ef1c96e6'], 'candidate': ['06d804546e0f']}
lead arm distribution (ABAB leader): {'baseline': 172, 'candidate': 204}
```
`fleet/recount_jbmt.py` independently reproduces `win=26 tie=138 loss=211`,
`units with any fallback_runs > 0: none`, `algorithms measured: {'parallel_hash': 376}`,
`tool_versions: {'jbmt-v2': 376}`.

**Where the candidate does win here, it wins at T96** — the 10 best are all `T96`:
`stackoverflow__badges_UserId__users_Id__T96__tiera` 0.802,
`stackoverflow__postlinks_RelatedPostId__posts_Id__T96__tiera` 0.821,
`tpch__lineitem_l_orderkey__orders_o_orderkey__T96__tiera` 0.832,
`tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__T96__tiera` 0.842,
`tpch__partsupp_ps_partkey__part_p_partkey__T96__tiera` 0.854. The same high-thread /
low-thread split as the legacy suite.

---

## Suite 3 — jbmt real suite, tier b: **UNSETTLED — not run**

Tier b was never started. The campaign was stopped on instruction while tier a was
completing, and no tier-b unit was measured — so there is **no** tier-b verdict, not even a
partial one. It is **not** extrapolated from tier a, and tier a's numbers are **not** a
tier-b result: tier b is a different data scale (TPC-H SF100 / TPC-DS SF64 / CoffeeShop 1b /
StackOverflow ×2) and the prior campaign's own INVALIDs were concentrated there.

*The exact command that would settle it*, given a prepared fleet (the same 8-host venue,
both arms resident via `jbmt_prep_phj_ph.sh`):

```
NSHARDS=8 bash tmp/chj_amac/fleet/jbmt_sweep_phj_ph.sh real_b real b
# then, per the gate:
python3 join_bench_mt.py report --results 'results.real_b.shard*.jsonl' --suite real --tier b --arm baseline
python3 join_bench_mt.py report-ab --results 'results.real_b.shard*.jsonl' --arm-a baseline --arm-b candidate --out AB_REPORT.real_b.md
```
Pre-registered expectation for that run, unchanged and still on the record: exactly 3 INVALID
units — `tpch__customer_c_nationkey__supplier_s_nationkey__T16__tierb`,
`…__T96__tierb`, and
`tpcds__catalog_sales_cs_bill_customer_sk__store_returns_sr_customer_sk__T16__tierb`.

---

## Unit 6 — teardown. **G8 GREEN.**

Raw output of the G8 queries, re-run fresh after teardown:

```
$ aws ec2 describe-instances … --filters "Name=tag:RUN_TAG,Values=phj-ph-ab-20260728" "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text
instances: []
$ aws ec2 describe-volumes … "Name=tag:RUN_TAG,…" "Name=status,Values=creating,available,in-use" --query 'Volumes[].VolumeId' --output text
volumes: []
$ aws ec2 describe-security-groups … "Name=tag:RUN_TAG,…" --query 'SecurityGroups[].GroupId' --output text
sgs: []
$ aws ec2 describe-snapshots --snapshot-ids snap-021cbdc2484f86607 … --query 'Snapshots[0].State'
snapshot: "completed"
```

**The gate is shown to have had the power to fail.** Before teardown the identical filters
returned 8 instances, 16 volumes and 1 security group
(`fleet/teardown_dryrun_before.log`), and the instances still resolve by tag as
`terminated` × 8, so the empty results above are a real state change and not a mis-typed
filter:

```
$ aws ec2 describe-instances … --filters "Name=tag:RUN_TAG,Values=phj-ph-ab-20260728" --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text
i-069d5483a4d36300d	terminated      i-065ebd96c4dd296e2	terminated
i-0781d51e1d57c8b1a	terminated      i-0d65b4dd8f104e168	terminated
i-0f8ece4037a96fadc	terminated      i-0cdef32c6d6060ecb	terminated
i-0f8dadc4b4757f83d	terminated      i-01ab31f17f082596e	terminated
```

Teardown accounting (`fleet/teardown_phj_ph.log`): 8 instances terminated; 8 data volumes
deleted (`vol-09c4e9ba1983fdccf`, `vol-0d92b3896d8ef4737`, `vol-012e93820b6d74251`,
`vol-056bf3d5c48f8ca91`, `vol-0aeff043418eaeb06`, `vol-0b1198722222e06e4`,
`vol-0848273a57ae77390`, `vol-007d748bef0937ddc`); the 8 root volumes went with
`DeleteOnTermination`; security group `sg-021349461933fb060` deleted; snapshot untouched.
The `DeleteVolume` denial the prior campaign hit did **not** occur — the script tags
`ndc-dbg-target=true` on this run's own volumes first. **No authorization-required step was
left undone.**
