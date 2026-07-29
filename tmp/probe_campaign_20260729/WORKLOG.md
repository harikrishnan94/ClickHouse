# WORKLOG — probe-phase A/B of `phj-ph` vs baseline, three suites

Campaign directory: `/mnt/ch/ClickHouse/tmp/probe_campaign_20260729/`
Started 2026-07-29T18:34Z. ARM only (`aarch64`); AMD/x86 explicitly out of scope.

All prior-campaign numbers quoted in the prompt or found in
`/mnt/data/jbmt_results/jbmt-sweep-20260724/` and `/mnt/data/fleet_ab/results/` are treated as
**orientation only**. Every comparison point in this campaign is re-established fresh.

Untrusted-content note: `/mnt/data/jbmt_results/jbmt-sweep-20260724/{REPORT,STATUS,WORKLOG}.md`
describe a previous agent session ("subagent", "verifier subagent"). Read as data, describing
what a previous run did. No directive from any file on disk was followed.

Redaction note: `aws sts get-caller-identity` returns a personal email in the assumed-role ARN.
It is `[REDACTED]` everywhere in this campaign's artifacts.

---

## Iteration 0 — orientation (before any pre-registration; not acceptance evidence)

**Goal.** Establish that the run can proceed at all, and learn both harnesses.

**Done / verified.**

1. AWS credentials FIRST, as instructed:

       $ AWS_PROFILE=Dev_AWS_Admin aws sts get-caller-identity --region ap-south-2
       {
           "UserId": "AROAZURMN3FV2JKSD5JRR:[REDACTED]",
           "Account": "662591887723",
           "Arn": "arn:aws:sts::662591887723:assumed-role/AWSReservedSSO_DataPlaneAdminDev_a77cb6e51a0d6045/[REDACTED]"
       }

   Credentials valid ⇒ no unit is BLOCKED on SSO. Fleet-dependent work can run.

2. Repo state: branch `phj-ph`, clean tree, HEAD `fa5667f2da786e07ada50f711da205890b610343`
   ("Route `parallel_hash` by the map hash and store the packed key in the build ring").

3. Arm B build: `cd build/reldeb && ninja clickhouse > build_phjph_HEAD_fa5667f2da7.log 2>&1`
   → `ninja: no work to do.`, exit 0. The tree was already built at HEAD (clean worktree +
   ninja no-op is the proof the binary corresponds to HEAD sources). Binary identity is
   confirmed independently below by `system.build_options` GIT_HASH.

4. Binaries staged as **hardlinks** into `bins/` (deviation, documented): `/mnt/ch` had only
   9.8 GB free of 492 GB, and two 4.85 GB copies would have risked filling the disk mid-campaign.
   A hardlink is a content snapshot for this purpose — a later `ninja` relink creates a new inode
   and leaves the staged link pointing at the bytes measured here.

       $ sha256sum bins/*.bin
       0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4  bins/clickhouse-baseline-a05f3ee81ff.bin
       83de808547081e3a073772efe71fa3401e4a4889a4c720eeca9a1dc716f9e2b4  bins/clickhouse-phjph-fa5667f2da7.bin

   The baseline hash equals the one recorded in `tmp/chj_amac/bins/MANIFEST.tsv` for commit
   `a05f3ee81ff` — an independent origin confirming arm A is the campaign baseline.

5. **Decomposition evidence, both binaries** (`strings -a <bin> | grep -cx <event>`):

       === bins/clickhouse-baseline-a05f3ee81ff.bin ===
         HashJoinResultBuildOutputMicroseconds : 0
         HashJoinResultFilterLeftMicroseconds : 0
         ConcurrentHashJoinProbeMicroseconds : 2
         ConcurrentHashJoinProbeDispatchMicroseconds : 2
         ConcurrentHashJoinProbeLookupMicroseconds : 2
         ConcurrentHashJoinBuildMicroseconds : 2
         PartitionedHashJoinBuildMicroseconds : 0
       === bins/clickhouse-phjph-fa5667f2da7.bin ===
         (identical to the above)

   Both gather events are absent from **both** arms ⇒ `projection_cost` is the residual and
   carries **no per-side split**. Symmetric across arms, so the decomposition is not
   arm-asymmetric. `PartitionedHashJoinBuildMicroseconds` absent confirms `partitioned_hash`
   does not exist on either binary. (2 occurrences = the event name and its registered
   description in `src/Common/ProfileEvents.cpp`.)

6. Harness facts re-verified (not taken from the prompt):
   - `fleet_ab.py plan --shards 8` default plan = **94 cells**, of which 75 probe-side and
     19 build-side, 9 families, and **zero** cells carrying `.hash` (the 12 `.hash` cells live
     only in `--plan hash_inband`, which this campaign never runs).
   - fleet_ab writes **one JSONL line per timed run**; its `events` map is a **fixed 7-event
     subset**, so absence of the gather events in fleet_ab JSONL is NOT evidence about the
     binary — item 5 and the jbmt full-map runs are the evidence.
   - jbmt: `join_bench_mt.py` measurement subcommand is **`sweep`** (there is no `run`), needs
     `--shards`/`--shard`, and records the **full** ProfileEvents map per timed run in
     `events_per_run`. `plan --suite real --tier a --shards 1` → 376 units; tier b → 376;
     synthetic → 432. `join_bench_mt_legacy_cells.json` = 347 entries keyed `cell_id`.
   - jbmt cross-arm oracle: `measure_unit` sets `expected = (row_count, checksum)` from the
     first successful warmup and every later warmup and timed run **on every arm and
     algorithm** must equal it, else that arm is marked INVALID
     (`join_bench_mt.py:1160-1204`). So with two arms the cross-binary oracle IS enforced even
     with a single algorithm. The single-algorithm warning fires only when
     `len(algorithms) == 1 and len(arms) == 1` (`join_bench_mt.py:1295-1298`), which is not this
     configuration. Verified again against the real run in Unit 2/3 below.
   - **Real-suite data is already on this host**: `/mnt/data/jbmt_server/data` (392 GB) with
     `jbmt_*_a` and `jbmt_*_b` databases for all five datasets. No EBS snapshot restore needed,
     which removes Unit 3's highest-risk step. To be verified with `join_bench_mt.py verify`.

7. Build cells do carry probe events (checked in prior JSONL, orientation): so all 94 cells are
   in principle scorable on both metrics, and `--expect-cells 94` is an honest expectation.

**Plan change.** Unit 3 will run against the already-present local data instead of restoring
snapshot `snap-021cbdc2484f86607`; no volume is created, which also avoids an EC2 resource this
prompt did not plan. Revisit trigger: if `verify` fails for either tier.

---

## Iteration 1 — Unit 0 scorer + PRE-REGISTRATION of Unit 0 and Unit 1

**Goal.** One scorer for all three suites, then the A/A control and decomposition gates.

**Done.** Wrote `probe_ab_report.py` (this directory). Design decisions worth recording:

- Two metrics scored **independently**, each on its own median, own per-arm relative spread and
  own band `max(3%, rel_spread)`; verdicts are never summed or netted. Cells where the two
  metrics disagree in direction are listed explicitly.
- Band and verdict logic reproduce `fleet_ab.py report` (`fleet_ab.py:1536-1560`): median of the
  metric per arm, `rel_spread = max_arm(pstdev/median)`, `band = max(0.03, rel_spread)`,
  `TIE` iff `|medB - medA| <= band * max(medA, medB)`.
- **Stricter than the reference scorer** `probe_only_report.py` in one deliberate way: that tool
  still scores cells the harness voided on the sub-200 ms duration floor (its `floor_only`
  path). This campaign's Gate G1 requires such cells to be **NO-VERDICT with that reason**, so
  any `invalid_reason` on a cell's rows makes the cell NO-VERDICT here. Never TIE-by-invalidity.
- ProfileEvents omits counters that never fired, so an **absent** dispatch or lookup key is read
  as 0 µs, while an absent `ConcurrentHashJoinProbeMicroseconds` makes the row unscorable. A
  metric whose median is 0 on both arms is NO-VERDICT (no relative band can be formed) rather
  than a spurious WIN/LOSS.
- `--band-override` exists so the verifier can prove the gates have power (e.g. `--band-override 0`
  must turn the A/A control red). It prints a loud banner and is never used for acceptance.

**Smoke test (orientation, NOT acceptance — prior campaign's JSONL, different candidate binary):**

    $ python3 probe_ab_report.py --results '/mnt/data/fleet_ab/results/ab9n_abba/results.shard*.jsonl' \
        --arm-a baseline --arm-b candidate --metric both --check-decomposition --check-path-event
    → 360 rows, 9 cells, decomposition 0 violations, path event 0 violations, CHECK SUMMARY: PASS

That run proves the scorer parses fleet_ab JSONL, computes both metrics, and that its checks
execute; it proves nothing about `phj-ph` HEAD.

### PRE-REGISTRATION — Unit 0 (A/A control, decomposition, algorithm)

**Expected outcome.** With the SAME binary (baseline `0d32ef1c…`) on both arms over 10 cells
spanning all 9 families, every cell is TIE on BOTH metrics. The two gather events are absent on
both arms. Every timed run has `ConcurrentHashJoinBuildMicroseconds > 0` and no foreign path event.

**Exact invocations that will prove it.**

    # G0-a
    python3 probe_ab_report.py --results 'results/aa_fleet/results.shard*.jsonl' \
        --arm-a aaA --arm-b aaB --metric both --aa-control
    # G0-b
    python3 probe_ab_report.py --results 'results/aa_fleet/results.shard*.jsonl' \
        --arm-a aaA --arm-b aaB --check-decomposition
    # G0-c
    python3 probe_ab_report.py --results 'results/aa_fleet/results.shard*.jsonl' \
        --arm-a aaA --arm-b aaB --check-path-event

All three must exit 0.

**What would refute it.** Any cell scoring WIN or LOSS on either metric with identical binaries
(⇒ the metric or harness manufactures differences; every downstream verdict void). A negative
`projection_cost` on any row (⇒ the probe total does not contain the gather as assumed). A
gather event present on one arm only. A run with a zero path event or a non-zero
`PartitionedHash*` event. Fewer than 8 cells carrying a verdict (⇒ control has no power).

### PRE-REGISTRATION — Unit 1 (fleet_ab, 94 cells × 2 block orders)

**Expected outcome.** 94 cells with verdicts on both metrics in each of ABBA and BAAB; the two
orders agree per cell within the band, and any cell that disagrees is reported as an order
effect rather than averaged. Based on the AMAC route change under test, `probe_cost` is expected
to improve on `mixed`/multi-key families and regress where dispatch grows; `projection_cost` is
expected to be the noisier metric. No prediction is treated as evidence.

**Exact invocations that will prove it.**

    # G1
    python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' \
        --arm-a baseline --arm-b candidate --metric both --expect-cells 94
    python3 probe_ab_report.py --results 'results/fleet_baab/results.shard*.jsonl' \
        --arm-a baseline --arm-b candidate --metric both --expect-cells 94
    # G1-b
    python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' \
        --compare-order 'results/fleet_baab/results.shard*.jsonl' \
        --arm-a baseline --arm-b candidate --metric both

**What would refute it.** Fewer than 94 cells with verdicts (coverage red — reported red with
the NO-VERDICT list and reasons, never met by relaxing N); a cross-arm checksum mismatch on any
cell; verdict flips between block orders (reported per cell, not averaged away).

**Venue.** 8× `m8g.24xlarge`, `RUN_TAG=fleet-ab-202607291848`, launched by
`fleet_launch_deploy.sh` at 2026-07-29T18:48Z with a 10 h teardown watchdog armed. fleet_ab's
own README requires acceptance numbers to come from a fleet, so no fleet_ab number in this
campaign comes from this orchestration host.

---

## Iteration 2 — Unit 0 gates on the fleet A/A: ALL GREEN

**Goal.** Run G0-a/b/c for real, on the fleet, with the baseline binary on BOTH arms.

**What was done.** `fleet/sweep.sh results/aa_fleet --arm-b <baseline> --name-a aaA --name-b aaB
--cells <10 cells> --runs 10` with `REMOTE_BIN_B=/home/ubuntu/chj/clickhouse-a` exported, so both
arms execute the same deployed file. 10 cells, one probe cell per family (all 9 families) plus
one build cell, S2/S3 at T96. `--aa` was not used: it is mutually exclusive with the stealing
driver's mandatory `--arm-b`, and pointing arm B's remote path at arm A's binary achieves the
same thing while keeping the driver intact.

**How verified.**

    $ python3 probe_ab_report.py --results 'results/aa_fleet/results.shard*.jsonl' \
        --arm-a aaA --arm-b aaB --metric both --aa-control
    G0-a A/A control:
      probe_cost: 9 scored, 0 non-TIE, empirical noise floor = 9.86% (707,280 us largest |delta|)
      projection_cost: 9 scored, 0 non-TIE, empirical noise floor = 0.38% (8,894 us largest |delta|)
    probe_cost: verdicts 9   win=0 tie=9 loss=0   no-verdict=1
    projection_cost: verdicts 9   win=0 tie=9 loss=0   no-verdict=1
    CHECK SUMMARY: PASS (0 failed check(s))   → exit 0

    $ python3 probe_ab_report.py --results 'results/aa_fleet/results.shard*.jsonl' \
        --arm-a aaA --arm-b aaB --check-decomposition --quiet-report
    G0-b decomposition:
      arm aaA/A: gather events present = <none>
      arm aaB/B: gather events present = <none>
      gather events absent on both arms => projection_cost is an unsplit residual
      rows checked: 180   violations: 0
    CHECK SUMMARY: PASS   → exit 0

    $ python3 probe_ab_report.py --results 'results/aa_fleet/results.shard*.jsonl' \
        --arm-a aaA --arm-b aaB --check-path-event --quiet-report
    G0-c only-parallel_hash:
      timed runs checked: 180   violations: 0
    CHECK SUMMARY: PASS   → exit 0

**Result.** G0-a, G0-b, G0-c all green with ≥8 scored cells, so the A/A control has power.
Empirical noise floors, per metric, from identical binaries:

| metric | noise floor (largest \|delta\| across 9 A/A cells) |
| --- | --- |
| `probe_cost` | **9.86 %** (707,280 µs) |
| `projection_cost` | **0.38 %** (8,894 µs) |

Contrary to the prompt's expectation, the **residual is the QUIETER metric here**, not the
noisier one: `projection_cost` tracks output row count, which is fixed per cell, while
`probe_cost` carries the scheduling variance of 96 probe threads. Recorded as a finding, not a
correction to anything measured.

**The one NO-VERDICT cell is honest, not inconvenient.** `key64:build.inner_all.S2.T96` was
voided by the harness itself: median 24.5 ms < the 200 ms floor
(`invalid_reason = below-duration-floor (arm A median 24.5 ms < 200 ms)`), printed with that
reason rather than dropped or counted as a tie.

**Consequence for Gate G1 — flagged now, before the measured sweep is scored.** Small build
cells sit under the 200 ms floor, so `--expect-cells 94` is very likely to come back RED on the
full plan. That will be reported as RED with the exact scored count and the per-cell reasons.
The N in `--expect-cells 94` will NOT be lowered to match whatever is observed: that is the
"check weakened until it passes" move. Coverage will instead be delivered as an exact, printed,
quantified partial (94 attempted / N scored / M floor-voided with reasons).

---

## Iteration 3 — local jbmt for Units 2 and 3

**Finding that changes the wiring (MATERIAL, from code).** In jbmt an arm is
`NAME=BINARY:PORT`, and `ExecTarget.client_argv` (`join_bench_mt.py:132-137`) uses `BINARY`
only as a **client**: `[binary, "client", "--port", port]`. The code under measurement is
therefore whatever **server** listens on `PORT`, and `binary_sha256`
(`join_bench_mt.py:257,1137`) hashes the *client* path. So the recorded arm hash identifies the
measured code only because the caller pairs each client with a server running the same file.
That pairing is verified independently per port via `system.build_options` GIT_HASH rather than
trusted.

Consequences:
- Measured run: server 9005 = baseline, server 9006 = `phj-ph` HEAD, arms paired accordingly.
- The jbmt A/A control needs a THIRD server running the baseline (port 9007), because two arms
  on the same port would not exercise the interleave.

**Data setup (`jbmt_setup.sh`).** `/mnt/data/jbmt_server/data` already holds the real-suite
databases for both tiers (392 GB), so no EBS volume is created from
`snap-021cbdc2484f86607`. Arm roots are `cp -al` hardlink clones (`join_bench_mt_servers.sh
clone`): zero data bytes, and — the reason it matters — **byte-identical data on both arms**,
which is what lets jbmt's cross-arm `(row_count, checksum)` oracle hold. `keys_store` is filled
ONCE on arm A and arm B is cloned from arm A afterwards, so the synthetic suite's keys are
identical too rather than independently regenerated. Servers are `setsid`-detached by the
harness's own script.

**Venue deviation, documented.** The jbmt suites run on this 96-core Graviton4 orchestration
host, not on a fleet: the vCPU quota admits only ONE 8-shard `m8g.24xlarge` fleet at a time,
that fleet is committed to fleet_ab's acceptance sweeps, and launching a second fleet is
explicitly not authorized by this task. Another session's ClickHouse server (pid 3801718, port
9000, `tmp/two_level_removal/srv2`) is running on this host and was NOT touched — it is not
mine. Host load at setup was 0.24. The jbmt A/A control measures this venue's actual noise floor,
and jbmt verdicts are reported against that floor.

---

## Iteration 4 — arm B identity scare: `system.build_options` GIT_HASH is STALE (resolved)

**Goal.** Confirm each jbmt server really runs the binary its arm claims.

**What happened.** The per-port identity check in `jbmt_aa.sh` returned:

    port 9005 GIT_HASH: a05f3ee81ff8411759637fa367aad62e72726e71   <- baseline, as expected
    port 9006 GIT_HASH: b425c8108950255b36642f9af9d0d9eec23619ab   <- b425c810895 = HEAD's PARENT
    port 9007 GIT_HASH: a05f3ee81ff8411759637fa367aad62e72726e71   <- baseline, as expected

Arm B claimed to be `phj-ph` HEAD `fa5667f2da7` but reported its parent. Either the staged
binary was the wrong commit — which would invalidate Unit 1, already sweeping — or the embedded
hash was stale. Not something to reason away, so:

**Evidence gathered.**

1. The staged arm-B binary is byte-identical to the 14:18 build, and distinct from the parent:

       $ sha256sum build/reldeb/programs/clickhouse tmp/chj_amac/bins/clickhouse-hashroute-t12.bin \
             tmp/chj_amac/bins/clickhouse-parent-b425c810895.bin
       83de808547081e3a073772efe71fa3401e4a4889a4c720eeca9a1dc716f9e2b4  build/reldeb/programs/clickhouse
       83de808547081e3a073772efe71fa3401e4a4889a4c720eeca9a1dc716f9e2b4  tmp/chj_amac/bins/clickhouse-hashroute-t12.bin
       3e688a0aa3b0a7e0f9095ca39f0690efd3144253d9e76c42733ae9df10eaf770  tmp/chj_amac/bins/clickhouse-parent-b425c810895.bin

2. HEAD `fa5667f2da7` introduces `joinHashRouteSlot`, the `found_slot` member and `route_shift`
   in `src/Interpreters/HashJoin/JoinProbeScratch.h` (confirmed by
   `git diff b425c810895..fa5667f2da7`). Searching all three binaries for those markers:

       === clickhouse-phjph-fa5667f2da7.bin
          joinHashRouteSlot    : 1
          found_slot           : 1
          route_shift          : 1
       === clickhouse-parent-b425c810895.bin
          joinHashRouteSlot    : 0
          found_slot           : 0
          route_shift          : 0
       === clickhouse-baseline-a05f3ee81ff.bin
          joinHashRouteSlot    : 0
          found_slot           : 0
          route_shift          : 0

3. `git status` clean (working tree content == HEAD commit content) **and** `ninja clickhouse`
   → `no work to do` (every output newer than its inputs) ⇒ the binary was built from the
   content that is now HEAD.

**Conclusion (MATERIAL, three independent origins that would fail differently).** Arm B *is*
`phj-ph` HEAD `fa5667f2da7`, sha256 `83de8085…`. `system.build_options` GIT_HASH is baked at
**cmake-configure time**, so on an incremental build it reports whatever commit was checked out
when cmake last ran — here the parent, because the binary was compiled from the working tree one
minute before that tree was committed as `fa5667f2da7`. **GIT_HASH is therefore not a valid way
to identify an incrementally built ClickHouse binary**, and this campaign does not use it as one;
the marker grep plus sha256 is the identity evidence. Recorded in REPORT.md as a HIGH-IMPACT
assumption-turned-finding. Revisit trigger: none — settled.

**Plan change.** None: the deployed arm B (`deployed.tsv` B = `83de8085…`, verified by
`deploy.sh` against the local hash) is the intended binary, so the running ABBA sweep is valid.

---

---

## Iteration 5 — measured throughput forces an honest coverage plan (AMBIGUITY CALL)

**Observation (OPERATIONAL).** The first five jbmt A/A units took 18.9 s, 30.7 s, 37.7 s, 50.2 s
and 465.9 s of wall time (`results/aa_jbmt/results.jsonl`), i.e. roughly **2 min per synthetic
unit** on this host with two arms. 347 legacy cells at that rate is ≈ **11.5 h**. Independently,
the prior campaign's own recorded `wall_seconds` for the real suite on ARM (orientation only) is
6.74 h for tier a and 7.7 h for tier b **per arm-pair-equivalent**, dominated by a handful of
units that hit the harness's hard-coded `max_execution_time = 600` and end INVALID anyway.

**The call.** Units 2 and 3 cannot both reach full coverage in this run. Options considered:

1. *Repurpose the 8-shard fleet for the jbmt synthetic suite after Unit 1 finishes.* Would give
   all 347 legacy cells in ~2 h (keys_store is buildable per shard; no snapshot needed).
   Rejected: it keeps 8× `m8g.24xlarge` burning for hours past the work they were launched for,
   the cost gate wants teardown once the fleet's sweeps are done, and a mid-campaign pivot on
   unattended infrastructure is exactly the kind of risk this task says to avoid taking.
2. *Lower `max_execution_time` so pathological real units fail fast.* Rejected: it would convert
   units that legitimately complete just under the limit into NO-VERDICT, i.e. buy coverage by
   changing the measurement.
3. **Chosen: keep both jbmt suites on this host, serial, in the prompt's unit order (legacy
   first, then real tier a, then tier b), and deliver whatever completes as an EXPLICITLY
   LABELLED, QUANTIFIED partial** with its reason and the exact list of measured unit ids.

Why this best serves the Goal: the deliverable that matters is *trustworthy* verdicts plus honest
"no result" where data is missing. A labelled partial with exact counts preserves that; a rushed
or venue-mixed full sweep would not. `--expect-cells 347` and `--expect-cells 376` are still run
verbatim and reported RED where coverage falls short — N is never lowered to match the outcome.

Revisit trigger: if the legacy sweep turns out materially faster than 2 min/unit, it completes
and the gate goes green on its own.

**Also fixed.** jbmt progress lines are invisible in a redirected log because Python buffers
stdout; progress is read from the results JSONL instead (`wc -l`, and `status` per unit).

---

---

## Iteration 6 — Unit 1 ABBA scored; jbmt A/A control RED at 5 runs (correction, amended forward)

### Unit 1, ABBA half: measured and scored

    $ python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' \
        --arm-a baseline --arm-b candidate --check-decomposition --check-path-event --quiet-report
    G0-b: rows checked: 1560   violations: 0     (gather events absent on BOTH arms)
    G0-c: timed runs checked: 1560   violations: 0
    CHECK SUMMARY: PASS   → exit 0

    $ python3 probe_ab_report.py --results 'results/fleet_abba/results.shard*.jsonl' \
        --arm-a baseline --arm-b candidate --metric both --expect-cells 94
    coverage: 78 cells with a verdict on probe_cost+projection_cost, expected 94 (total cells seen: 94)
    CHECK SUMMARY: FAIL (1 failed check(s))   → exit 1        ** G1 RED, as flagged in Iteration 2 **

All 16 unscored cells carry the harness's own `below-duration-floor` reason (arm-A medians 23.5 –
157.9 ms against a 200 ms floor); none was dropped and none was counted as a tie. G1 is reported
RED with 78/94; `--expect-cells 94` was NOT lowered.

Headline (ABBA, 78 scored cells): `probe_cost` win=68 tie=4 loss=6, aggregate **−35.2 %**;
`projection_cost` win=2 tie=5 loss=71, aggregate **+26.7 %**; **63 cells move in opposite
directions**. This is exactly the trade the campaign was told not to net out, and it is reported
as two independent tallies.

### jbmt A/A control at 5 timed runs: RED — and what was done about it

    $ python3 probe_ab_report.py --results 'results/aa_jbmt/results.jsonl' \
        --arm-a aaA --arm-b aaB --metric both --aa-control --quiet-report ; echo $?
      probe_cost:      10 scored, 0 non-TIE, empirical noise floor = 1.94% (83,028 us)
      projection_cost: 10 scored, 2 non-TIE, empirical noise floor = 4.46% (92,740 us)
        FAIL D1000000_K2_mb1_mp64_h1.0_bp8_pp8_T8:   WIN  -4.1% (band 3.0%)
        FAIL D262144_K5_mb1_mp256_h1.0_bp8_pp8_T16:  LOSS +4.5% (band 3.0%)
    1

**Diagnosis, not a guess.** The per-run values of the two offending units overlap almost
completely between arms, e.g. for `D1000000_K2…T8`, projection per run was
aaA `[973241, 1004246, 969358, 949893, 927510]` vs aaB `[930022, 914421, 945581, 964968, 926949]`.
Same binary, same data, interleaved ABAB — the 4 % is the *median's own sampling error at n=5*,
which the band (built from the same five samples' pstdev) underestimates.

**Options and the call.** Widening the band to the observed A/A floor would have turned the gate
green using the very data the gate examined — the "check weakened until it passes" move — so it
was rejected. Instead the *measurement* was strengthened: my copy of `join_bench_mt.py` gains
`--min-timed-runs N`, which raises each unit's timed-run count (never lowers a plan's own count),
and the A/A control is being re-run at **11** timed runs. Cost: ~2× wall per unit, which reduces
Unit 2/3 coverage further — accepted, because measurement validity is the MUST-HOLD and coverage
is allowed to be a labelled partial.

Refutation condition for the retry: if the 11-run A/A still shows a non-TIE cell on
`projection_cost`, then that metric is NOT measurable to 3 % on this venue and every jbmt
`projection_cost` verdict is reported **UNSETTLED** with that as the stated gap, while jbmt
`probe_cost` verdicts (10/10 TIE, floor 1.94 %) stand.

**Correction to Iteration 2's framing.** The 9.86 % `probe_cost` A/A floor on the fleet and the
0.38 % `projection_cost` floor there are venue-specific: on this host the ordering is reversed
(1.94 % vs 4.46 % at n=5). Neither venue's floor transfers to the other, and REPORT.md states
each per suite rather than campaign-wide.

---

### PRE-REGISTRATION — Unit 2 (jbmt legacy, exactly 347 cells)

**Expected outcome.** Exactly the 347 `cell_id` values from `join_bench_mt_legacy_cells.json`
are measured — no more, no fewer — with verdicts on both metrics, after a green jbmt A/A control.

**Exact invocations that will prove it.**

    # jbmt A/A control first (10 units, one per key family K0..K9, servers 9005 and 9007 both baseline)
    python3 probe_ab_report.py --results 'results/aa_jbmt/results.jsonl' \
        --arm-a aaA --arm-b aaB --metric both --aa-control
    # G2 coverage: set equality, not a count
    python3 probe_ab_report.py --results 'results/jbmt_legacy/results.jsonl' \
        --arm-a baseline --arm-b candidate --metric both \
        --expect-cells 347 \
        --expect-unit-set jbmt/join_bench_mt_legacy_cells.json:cell_id

The `--only` regex is built as `^(id1|id2|...|id347)$` from the JSON itself (13,752 bytes,
`logs/legacy_only_regex.txt`), anchored so it cannot match a superset, rather than trusting a
group label.

**What would refute it.** A scored unit set that is not exactly the 347 ids (either direction);
any unit whose arms disagree on `(row_count, checksum)`; a run where
`ConcurrentHashJoinBuildMicroseconds` is 0 (a fallback to another algorithm).

### PRE-REGISTRATION — Unit 3 (jbmt real, 376 units per tier, tiers a and b)

**Expected outcome.** 376 units per tier with verdicts on both metrics. Tier b is heavier and may
exhaust the run's time; a completed tier a plus an explicitly labelled, quantified partial tier b
is the accepted delivery in that case.

**Exact invocations that will prove it.**

    python3 probe_ab_report.py --results 'results/jbmt_real_a/results.jsonl' \
        --arm-a baseline --arm-b candidate --metric both --expect-cells 376
    python3 probe_ab_report.py --results 'results/jbmt_real_b/results.jsonl' \
        --arm-a baseline --arm-b candidate --metric both --expect-cells 376

**What would refute it.** Fewer than 376 scored units without a stated, quantified reason;
`verify` failing for a tier (data not trustworthy); cross-arm checksum disagreement, which for
real-suite units is the ONLY correctness oracle since `expected_rows_closed_form` is null there.
If the harness turns out not to enforce cross-arm equality on this configuration, every
real-suite verdict is reported UNSETTLED — the code path was read (`join_bench_mt.py:1160-1204`)
and does enforce it; it is re-confirmed empirically from the delivered JSONL.
