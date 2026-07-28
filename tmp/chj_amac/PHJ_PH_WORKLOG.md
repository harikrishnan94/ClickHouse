# Worklog — `phj-ph` A/B benchmark campaign

`RUN_TAG` = `phj-ph-ab-20260728`. Campaign start commit `635aa368fd5`.
All times UTC. Raw command output is quoted; large artifacts are cited by path + sha256.

---

## Unit 0 — orientation (before any pre-registration)

**Goal.** Re-verify the prompt's "verified starting points" rather than trusting them, and
learn the two harnesses' actual contracts.

**What was done / how verified.**

Branch, HEAD, ancestry, working tree:

```
$ git status
On branch phj-ph
Your branch is up to date with 'origin/phj-ph'.
Changes not staged for commit:
	modified:   tmp/chj_amac/fleet_ab.py
$ git log --oneline -2
a0dfbfd965b Decouple the parallel_hash slot count from the thread count
3b76b5edfb5 Record the U5 fleet acceptance campaign and the mission report
$ git merge-base --is-ancestor a0dfbfd965b HEAD; echo $?
0
```

Baseline binary sha256 (G1, first half) — matches the manifest row:

```
$ sha256sum tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin
0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4  tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin
```

The uncommitted `timed_settings` change was present exactly as the prompt described
(`collect_hash_table_stats_during_joins` moved from `1 if cell.statson else 0` to
unconditional `1`), verified with `git diff tmp/chj_amac/fleet_ab.py`.

`MANIFEST.tsv` tail defect confirmed and **left alone** (out of scope, reported): the last
three lines are bare `sha256  path` pairs instead of six-column rows, for
`clickhouse-candidate-final-d1c77571b39.bin`, `…-fix1-06e0bbd0aa3.bin`,
`…-fix2-6598f4b872f.bin`. The baseline row is well formed.

AWS preflight (nothing created yet):

```
$ aws service-quotas get-service-quota … --quota-code L-1216C47A --query 'Quota.Value'
1152.0
$ # running vCPUs, summed CoreCount*ThreadsPerCore over running/pending
96
$ aws ec2 describe-snapshots --snapshot-ids snap-021cbdc2484f86607 … '[State,VolumeSize]'
completed	1536
```
So 8 × `m8g.24xlarge` = 768 vCPUs fits under 1152 with the 96-vCPU orchestration host
counted (headroom 1056). No fleet instance from any prior campaign is alive — the only
running instance is `dev-vm-harik` (`i-00c8778b1ae41598f`), this orchestration host, which
is untagged and **not mine to touch**.

**Two prior-art claims from the prompt/subagent reading that I checked myself, because they
would have broken a gate:**

1. A reading subagent reported that `fleet/matrix.json` has no `measured_plan` key and that
   `fleet/dispositions.json` does not exist. **Both are wrong.** `matrix.json` top-level
   keys are `['generated_at', 'generator', 'notes', 'universe', 'measured_plan',
   'hash_inband']`, and `dispositions.json` is present (537,446 bytes, 1,819 entries). G2
   runs green verbatim as written in the prompt (94 cells, no `.hash`). Subagent output is
   data, not truth; this is why the gate is run rather than believed.
2. The three AMAC counters are registered on this branch —
   `grep -n "ConcurrentHashJoinAmac" src/Common/ProfileEvents.cpp` returns lines 438/439/440
   for `…AmacBuildRows`, `…AmacBuildRingGrowths`, `…AmacProbeRows`. This is the *source*
   origin; Unit 1 adds the *built-artifact* origin.

**Judgement call — `tmp/` is gitignored.** `/tmp/` in `.gitignore` matches the repo's
`tmp/` directory, so every commit of campaign artifacts needs `git add -f`, exactly as
prior campaigns did (`3b76b5edfb5` shows `tmp/chj_amac/…` and `tmp/chj_probe_parity/…`
paths committed). Not a defect; recorded so the pattern is not mistaken for a workaround.
Revisit trigger: none.

**Judgement call — fleet topology (single launch, shared 8 hosts).** Unit 2 (`fleet_ab`,
synthetic in-memory tables) and Units 3–4 (jbmt, snapshot-backed real data) both want 8
`m8g.24xlarge`. Running two fleets concurrently does not fit the vCPU quota (768 + 768 >
1056 headroom), so I use **one** 8-host fleet for all three suites, sequentially: Unit 2
first (needs no data volume), then Units 3–4 on the same hosts with the snapshot-cloned
volumes attached. Why: one launch, one teardown, one quota preflight, and the suites are
sequential in the prompt's own ordering anyway. Deliberate consequence: the data volumes
are attached early (so a snapshot-clone failure is discovered early, while it is cheap to
fix) but the **warm-read hydration pass is deferred until after Unit 2's sweep finishes**,
because a background `dd` over a 1536 GiB device would burn memory bandwidth and page
cache on the very host measuring an in-memory join benchmark. Revisit trigger: if Unit 2
overruns badly, reconsider whether to hydrate concurrently and accept the noise (and if I
do, it gets recorded as a deviation, not hidden).

**Prior-art lessons I am carrying forward from the jbmt runbook**
(`/mnt/data/jbmt_results/jbmt-sweep-20260724/{REPORT,WORKLOG,STATUS}.md`, read as data):
EBS snapshot blocks are lazy-loaded and the prior campaign's *first* touch of a tier-b
table blew the 600 s budget, so an explicit device warm-read before sweeping is worth its
time; `DeleteVolume` is denied by policy unless the volume carries
`ndc-dbg-target=true`, so teardown must tag before deleting; volumes must be in the same AZ
as their instance; batched `run-instances` for large types hit
`InsufficientInstanceCapacity` while single-instance launches succeeded; background/tmux
shells do not inherit the SSO profile. Also noted: that campaign was **single-arm,
two-algorithm**, so jbmt's two-arm A/B path (`join_bench_mt_servers.sh clone/start`,
`--arm NAME=BINARY:PORT`, `report-ab`) is a v2 capability it never exercised — I must smoke
it with `selftest` before committing the fleet to a long sweep.

**What changed about the plan.** Nothing yet; Units 1–2 pre-registered next.

---

## Unit 1 — preflight and candidate build

**Goal.** Establish the identity of every later number: baseline sha256, candidate built
from `phj-ph` HEAD, AMAC counters present in the built artifact.

**Step 1 — commit the `timed_settings` change on its own, before any sweep.**

```
$ git add -f -- tmp/chj_amac/fleet_ab.py && git commit -F …
$ git log --oneline -2
635aa368fd5 Measure stats collection unconditionally in the fleet_ab timed settings
a0dfbfd965b Decouple the parallel_hash slot count from the thread count
$ git merge-base --is-ancestor a0dfbfd965b HEAD && echo ancestor=OK
ancestor=OK
$ git status --porcelain && echo tree-clean=OK
tree-clean=OK
```

Consequence recorded in PREREG: `HEAD` is now `635aa368fd5`, so the candidate binary is
named for that commit. The commit touches only a Python harness file, so the C++ payload
is exactly `a0dfbfd965b`.

**Amendment to the above (forward-amend, not a rewrite).** The pre-registration commit
`96532537d4d` landed before the build, so `HEAD` at build time was `96532537d4d` and the
candidate is named for *that* commit, not `635aa368fd5`. Both commits touch only
campaign files under `tmp/chj_amac/` (a Python harness file and the prereg/worklog), so the
C++ payload is still exactly `a0dfbfd965b`, which remains an ancestor. The PREREG entry's
`<short>` placeholder resolves to `96532537d4d` everywhere.

**Step 2 — G1, binary identity. GREEN.**

```
$ sha256sum tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin
0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4  tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin
$ git merge-base --is-ancestor a0dfbfd965b HEAD; echo $?
0
```
Both halves match the pre-registered expectation exactly.

**Step 3 — candidate build.**

```
$ bash tmp/chj_amac/snapshot_candidate.sh
BUILT candidate-96532537d4d: /mnt/ch/ClickHouse/tmp/chj_amac/bins/clickhouse-candidate-96532537d4d.bin sha256=06d804546e0f029b459fbfa806debb4f33c688fe55dd18770c3ed79957dcb971
real	0m44.125s
```

44 seconds, because the tree was already compiled at this source state and only the
`GIT_HASH` re-configure forced a relink; the ninja log's last lines are
`[37/38] Linking CXX executable programs/clickhouse` after relinking `src/libdbms.a` (which
contains `ConcurrentHashJoin.cpp.o`, the payload's translation unit). Manifest row appended
by the script and re-verified against the file on disk:

```
$ tail -1 tmp/chj_amac/bins/MANIFEST.tsv
clickhouse-candidate-96532537d4d.bin	06d804546e0f029b459fbfa806debb4f33c688fe55dd18770c3ed79957dcb971	4847564584	96532537d4d04fa19c3b00c47739e03d56052e1c	/mnt/ch/ClickHouse/build/reldeb/build_candidate-96532537d4d.log	2026-07-28T17:18:28Z
$ sha256sum tmp/chj_amac/bins/clickhouse-candidate-96532537d4d.bin
06d804546e0f029b459fbfa806debb4f33c688fe55dd18770c3ed79957dcb971  tmp/chj_amac/bins/clickhouse-candidate-96532537d4d.bin
```

**Step 4 — the three AMAC counters are in the built artifact. Two independent origins, and
the check is demonstrated to have the power to fail.**

Origin 1, the ELF string table (fails if the translation unit was not linked in):
```
$ strings -a tmp/chj_amac/bins/clickhouse-candidate-96532537d4d.bin | grep -c ConcurrentHashJoinAmac
12
```

Origin 2, the built binary enumerating its own event registry (fails if the events were not
*registered*, which the string table cannot distinguish):
```
$ ./tmp/chj_amac/bins/clickhouse-candidate-96532537d4d.bin local --query "SELECT name FROM system.events WHERE name LIKE 'ConcurrentHashJoinAmac%' ORDER BY name SETTINGS system_events_show_zero_values = 1"
ConcurrentHashJoinAmacBuildRingGrowths
ConcurrentHashJoinAmacBuildRows
ConcurrentHashJoinAmacProbeRows
$ ./tmp/chj_amac/bins/clickhouse-candidate-96532537d4d.bin local --query "SELECT version(), revision()"
26.8.1.1	54513
```

Differential control — the same query on the **baseline** arm returns zero, which is both
the reason `--require-engagement` inspects `servers[1]` only and the proof that origin 2 is
not a tautology:
```
$ ./tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin local --query "SELECT count() FROM system.events WHERE name LIKE 'ConcurrentHashJoinAmac%' SETTINGS system_events_show_zero_values = 1"
0
```

**Note on what this A/B actually compares.** The frozen baseline `a05f3ee81ff` predates the
whole AMAC/routed-probe stack, so candidate-minus-baseline is the *branch tip* versus that
baseline, not the slot-decoupling commit in isolation. `a0dfbfd965b` is the tip's payload
and the reason for the campaign, but attributing a per-cell delta to that commit alone would
need a third arm (`3b76b5edfb5`, the pre-decoupling tip, whose binary exists in `bins/`).
That is not in this campaign's scope and is **not** claimed; the U5 changed-verdict
comparison in Unit 5 is the closest available signal and it shares this campaign's
measurement lineage, so it is contrast, never corroboration.

**Unit 1 verdict: GREEN.** G1 green; candidate built and identified; counters present in
the artifact by two independently-failing origins.

---

## Unit 2 — `fleet_ab` measured plan, 94 cells

**G2 — the cell list is exactly the measured plan. GREEN**, run verbatim as pre-registered:

```
$ python3 -c "import json; c=json.load(open('tmp/chj_amac/fleet/matrix.json'))['measured_plan']['cells']; h=[x for x in c if x.endswith('.hash')]; assert len(c)==94 and not h, (len(c), h); print(','.join(c))"
key32:probe.inner_all.S2.T96,key32:probe.inner_all.S3.T96,…,key64:probe.inner_all.S3.T96.statson
(94 comma-separated ids; exit 0)
```
Power-to-fail check: `fleet_ab.default_plan_cells()` returns 106 (94 + the 12
`hash_inband` guards) — confirmed in code at `load_cells_file`, which concatenates
`measured_plan.cells` with `hash_inband.cells`. Leaving `--cells` unset would sweep 12
cells whose algorithm does not exist in either binary, so `--cells` is passed explicitly
from this gate's output.

**Deviation, measured and documented — the calibration file the prompt names does not
work.** The prompt's resource map says to pass `tmp/chj_amac/calibration/calibration.json`
as `--calibration`. `fleet_ab.resolve_shape` does `int(cal)` on
`calibration[family][size]`, and in that file the value is a **dict**
(`{"build_rows": …, "dup": …, …}`), so it raises rather than degrading quietly:

```
$ python3 -c "… fleet_ab.resolve_shape(parse_cell('key64:probe.inner_all.S3.T96'), nested) …"
nested RAISES: TypeError int() argument must be a string, a bytes-like object or a real number, not 'dict'
flat  OK calibration-file 24000000
```

The file that matches the documented `--calibration` contract
(`JSON {family: {size: build_rows}}`) is `tmp/chj_amac/fleet/calibration_rows.json`, which
is what the prior campaign passed. It is the **exact flat projection** of the nested file —
zero mismatches across every family and size present in both — and it leaves no cell
uncalibrated:

```
mismatches: []
94-cell uncalibrated-under-flat: 0 []
```

So the campaign uses `fleet/calibration_rows.json`. This cannot move a verdict: the
row counts it yields are identical to the nested file's `build_rows`, and had I passed the
nested file there would be no numbers at all. Recorded as a deviation from the prompt's
resource map, not as a fix to either file (neither is edited).

**Fleet launch — 8 shards, all tagged.**

```
$ bash tmp/chj_amac/fleet/launch_phj_ph.sh
quota=1152.0 running_vcpus=96 need=768
ami=ami-0feea81f640baa6fb sg=sg-021349461933fb060 vpc=vpc-0dbe56e29f84d24a3
shard 0..7: i-01ab31f17f082596e … i-0f8ece4037a96fadc in subnet-07fde41a9bd25176e
FLEET LAUNCHED: 8 shards; SG sg-021349461933fb060; RUN_TAG phj-ph-ab-20260728
LAUNCH_EXIT=0
```

All 8 landed in `ap-south-2a` on the first try (no `InsufficientInstanceCapacity`, so the
one-at-a-time AZ-fallback path was not exercised beyond its first choice). `hosts.phj_ph.tsv`:

```
0	i-01ab31f17f082596e	172.31.9.199	ap-south-2a
1	i-065ebd96c4dd296e2	172.31.2.102	ap-south-2a
2	i-069d5483a4d36300d	172.31.1.38	ap-south-2a
3	i-0781d51e1d57c8b1a	172.31.10.61	ap-south-2a
4	i-0cdef32c6d6060ecb	172.31.11.84	ap-south-2a
5	i-0d65b4dd8f104e168	172.31.10.36	ap-south-2a
6	i-0f8dadc4b4757f83d	172.31.6.162	ap-south-2a
7	i-0f8ece4037a96fadc	172.31.3.104	ap-south-2a
```

**Null result worth recording — my own readiness poll was broken, not the fleet.** A first
reachability loop reported `1/8 reachable` for five minutes. Cause: `ssh` inside a
`while read … done < hosts.tsv` loop consumes the loop's stdin, so the host list was eaten
after the first iteration. Re-run with `ssh -n`: `reachable=8/8`. No instance was
recreated and nothing was "fixed" on the fleet; the deploy script was already immune
(`</dev/null` on every ssh/scp, inherited from `deploy.sh`). Recorded because a
misdiagnosis here would have wasted a relaunch.

**Data volumes created early, deliberately.**

```
$ bash tmp/chj_amac/fleet/volumes_phj_ph.sh
snapshot snap-021cbdc2484f86607 state=completed
shard 0: created vol-09c4e9ba1983fdccf in ap-south-2a for i-01ab31f17f082596e
… (8 volumes, one per shard)
```

Judgement call: volumes are created and attached **now**, before Unit 2 sweeps, so a
snapshot-clone failure surfaces while it is cheap; but the warm-read hydration pass is
deferred until Unit 2's sweep is done (see Unit 0). Deliberate deviation from the runbook's
`join_bench_mt.py fleet-volumes` helper: that helper tags only `Name=jbmt-<tag>` and cannot
add `RUN_TAG`, while G8's proof filters on `tag:RUN_TAG` and the prompt requires the tag at
creation time. `volumes_phj_ph.sh` therefore issues `create-volume` directly with the
helper's hard-coded shape (gp3 / 4000 IOPS / 1000 MBps, same snapshot) plus the tags. The
snapshot is read-only to this run. Revisit trigger: none.

**Sweep launched** (the pre-registered command, with the calibration deviation applied):

```
$ python3 tmp/chj_amac/fleet/run_sweep_stealing.py \
    --hosts tmp/chj_amac/fleet/hosts.phj_ph.tsv --ssh-key tmp/chj_amac/fleet/ssh_phj_ph/id_ed25519 \
    --arm-a tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin \
    --arm-b tmp/chj_amac/bins/clickhouse-candidate-96532537d4d.bin \
    --name-a baseline --name-b candidate \
    --remote-bin-a /home/ubuntu/chj/clickhouse-base --remote-bin-b /home/ubuntu/chj/clickhouse-cand \
    --calibration tmp/chj_amac/fleet/calibration_rows.json \
    --results-dir tmp/chj_amac/fleet/results_phj_ph \
    --cells "$(cat tmp/chj_amac/fleet/cells_94.txt)" -- --require-engagement
stealing sweep: 94 cells planned, 0 already complete, 94 queued (costliest first) across 8 shards
…
real	30m39.273s
SWEEP_EXIT=1
```

Deploy was verified first: all 8 shards report both binaries at the expected sha256
(`0d32ef1c96e6…` base, `06d804546e0f…` cand), `aarch64`, `96` cores, 370 GiB RAM —
8/8 matches for each hash.

**G3 — sweep completed every cell. RED (honest).** Raw final lines:

```
FLEET_STEALING RESULT: cells_run=94 cells_failed=17 shard0=14 shard1=18 shard2=3 shard3=6 shard4=17 shard5=2 shard6=18 shard7=16 -> FAIL
  FAILED lcstr:probe.inner_all.S5.T96 on shard 0 rc=1
  FAILED str:probe.inner_all.S3.T96.h05 on shard 7 rc=1
  FAILED str:probe.semi_anti.S2.T96.anti on shard 4 rc=1
  FAILED key64:probe.inner_all.S3.T96.h05 on shard 1 rc=1
  FAILED key64:probe.semi_anti.S2.T96.anti on shard 6 rc=1
  FAILED str:build.inner_all.S3.T96.dup16 on shard 7 rc=1
  FAILED str:build.left_all.S3.T96.dup16 on shard 1 rc=1
  FAILED mixed:build.inner_all.S3.T96 on shard 0 rc=1
  FAILED k256:build.inner_all.S3.T96 on shard 7 rc=1
  FAILED str:build.inner_all.S3.T96 on shard 1 rc=1
  FAILED key64:build.inner_all.S3.T96 on shard 4 rc=1
  FAILED key64:build.inner_all.S3.T48 on shard 7 rc=1
  FAILED key64:build.inner_all.S3.T96.statson on shard 6 rc=1
  FAILED mixed:build.inner_all.S2.T96 on shard 7 rc=1
  FAILED k256:build.inner_all.S2.T96 on shard 6 rc=1
  FAILED str:build.inner_all.S2.T96 on shard 4 rc=1
  FAILED key64:build.inner_all.S2.T96 on shard 1 rc=1
```

`cells_run=94`, so **every cell was attempted** — nothing was skipped by resume or lost with
a worker. The 17 failures are two diagnosed causes, neither of which is a candidate
regression, and neither of which I may act on:

- **16 cells: the protocol's own fail-closed 200 ms duration floor.** Each of these logged
  `CELL FAILED: <cell>: one or more runs INVALID (see invalid_reason in results)`, and every
  one of the 320 invalid rows carries a `below-duration-floor (arm A median NN ms < 200 ms)`
  reason — arm **A**, the baseline, i.e. the cell's timed query is too fast to measure at
  T96, exactly what `MIN_CELL_DURATION_US` exists to refuse. Full reason histogram:

```
$ # invalid_reason histogram over all 320 invalid rows
   20 x below-duration-floor (arm A median 87.6 ms < 200 ms)      … (16 distinct medians,
   20 x below-duration-floor (arm A median 156.4 ms < 200 ms)         20 rows each: 10 runs
   20 x below-duration-floor (arm A median 25.4 ms < 200 ms)          x 2 arms x 16 cells)
```

- **1 cell: `lcstr:probe.inner_all.S5.T96` cannot be built by the baseline arm here.**

```
CELL FAILED: lcstr:probe.inner_all.S5.T96: warmup 0 failed on arm baseline: Received exception from server (version 26.8.1):
Code: 241. DB::Exception: … (total) memory limit exceeded: would use 186.12 GiB (attempt to allocate chunk of 128.00 MiB), current RSS: 193.70 GiB, maximum: 193.71 GiB. … While executing FillingRightJoinSide. (MEMORY_LIMIT_EXCEEDED)
```
340M `LowCardinality(String)` build keys, on a 370 GiB host running **two** resident
servers. The U5 precedent hit the same cell, same arm, same exception (`would use
191.45 GiB`) and dispositioned it `EXCLUDED-INVALID`. It produced **no rows at all** (the
failure is in warmup 0), which is why the report sees 93 cells and not 94.

**No iteration is possible on this red, and that is the finding.** Turning G3 green would
require one of: rerunning a red cell hoping it flips; raising the server memory limit
(changing the venue mid-campaign); or dropping cells from the frozen 94-cell plan. All three
are banned moves. I did none of them and I am not spending the unit's 3 iteration cycles
re-running a deterministic OOM and a deterministic arithmetic floor. **G3 stays RED with
`cells_failed=17`.**

**G4 — coverage and validity. RED (honest).**

```
$ python3 tmp/chj_amac/fleet_ab.py report --results "$(ls -1 tmp/chj_amac/fleet/results_phj_ph/results.shard*.jsonl | paste -sd,)"
FLEET_AB REPORT RESULT: cells=93 win=27 tie=28 loss=22 invalid=16 insufficient=0 uncalibrated=0
G4_EXIT=1
```

Two pre-registered predictions are **refuted** and I record both as such: `invalid=0` (it is
16) and `cells=94` (it is 93). Two hold: `insufficient=0` and `uncalibrated=0` — the latter
confirming the calibration deviation worked as argued. The gate exits 1 because `invalid`
is non-zero, which is the gate behaving correctly, not a tooling problem.

**Unit 2 measured result (the deliverable, red included).**
93 cells with data out of 94 planned: **27 WIN / 28 TIE / 22 LOSS / 16 INVALID /
0 INSUFFICIENT**, 0 uncalibrated. Arm→binary mapping recomputed from the rows themselves,
930 rows per arm, exactly two distinct binaries:

```
arm -> binary sha256 prefix: {('A', '0d32ef1c96e6'): 930, ('B', '06d804546e0f'): 930}
planned=94 with_rows=93 no_rows=1: ['lcstr:probe.inner_all.S5.T96']
```

**Probe-event gate metric.** Two distinct things share that name in this harness, so I
report both rather than guess which was meant:

1. *Engagement* (what `--require-engagement` actually gates on):
   `ConcurrentHashJoinAmacProbeRows > 0` on the candidate arm in **82 of 93** cells; zero in
   11; the counter was **never absent** on the candidate (`counter absent in 0`). The 11
   zero-engagement cells are the compile-time-excluded families and shapes (`mixed`, `lcstr`,
   `asof`), which is the documented expectation, not a miss.
2. *Phase attribution*, `ConcurrentHashJoinProbeLookupMicroseconds` — the event a probe-side
   win is supposed to be carried by. The candidate's probe-lookup median is lower on
   **61 of 70** probe-side cells, by up to −64% (`k256:probe.inner_all.S3.T96` −64.10%,
   `k256:probe.inner_all.S2.T96` −58.68%) — **while several of those same cells lose on
   wall**. `fixstr:probe.inner_all.S5.T96` is the sharpest example: probe lookup −7.91%,
   wall verdict **LOSS** +10.43%. The routed/AMAC probe is doing its job on the phase it
   claims; the wall regression is elsewhere. Recorded as a measured tension, not resolved.

   The 9 probe-side cells where lookup is *higher* on the candidate, named rather than
   summarised: `lcstr:probe.inner_all.S2.T96` +22.03%, `lcstr:probe.inner_all.S3.T96`
   +18.31%, `key64:probe.semi_anti.S4.T96.anti` +15.08%, `key64:probe.asof.S4.T96` +9.92%,
   `key64:probe.inner_all.S1.T96` +5.83%, `key64:probe.asof.S2.T96` +3.85%,
   `mixed:probe.inner_all.S5.T96` +2.04%, `mixed:probe.inner_all.S2.T96` +1.46%,
   `mixed:probe.inner_all.S3.T96` +0.80%. The three `mixed` cells are the compile-time
   AMAC-excluded family (engagement 0 by design), so their ~1–2% is noise; the `lcstr` and
   `asof` cells are the shapes the branch deliberately routes off the ring, and they carry
   the largest lookup regressions. That is a **LEAD**, not a settled claim: I have not
   isolated a mechanism.

   **Correction, amending forward:** an earlier draft of this entry claimed lookup was lower
   on "every probe-side cell measured (68/68)". That was wrong — it is 61/70, and the nine
   exceptions are named above. The error was caught by re-running the analysis output rather
   than trusting the sentence; the corrected figure is what the report carries.

**Judgement call — I did not rerun anything.** Two shards (2 and 5) ran only 3 and 2 cells
respectively while shards 1 and 6 ran 18 each. That is the work-stealing driver behaving as
designed (costliest-first, so a shard that draws an S5 cell is busy for minutes), not an
imbalance to correct. No cell was rerun; there are no rerun files in
`fleet/results_phj_ph/`, and every row of the sweep is preserved.

---

## Units 3–4 venue preparation

**Goal.** Get both arms resident on one snapshot-cloned volume per shard, with the
`keys_store` tables the legacy suite reads, without touching the measurement protocol.

**Warm-read hydration — measured, and the reason it was worth doing.**

```
$ # shard 0, first attempt, xargs -P 8
35.3497 GB read … 37.2199 GB read after 20s     ->  93 MB/s
$ # after raising to xargs -P 64
throughput: 473 MB/s   cats: 64
$ # single-stream tail (1 cat left, 396 GB of ~417 GB done)
396.1 GB read … 14 MB/s
$ # after launching fleet/hydrate_tail.py (64 concurrent 64 MiB ranges)
rate now: 554 MB/s ; 436.3 GB read total
```
A snapshot-backed volume serves each first-touch block with an S3 round trip, so the limit
is latency, not the 1000 MB/s the volume is provisioned for; only concurrency hides it. All
8 shards finished (`warm read done: 2026-07-28T18:51:25Z`, `442G` used / `999G` free). This
is insurance against the prior campaign's recorded failure where a *first* touch of a
tier-b table blew the fixed 600 s budget.

**Judgement call — venue left alone rather than tuned, on evidence.**
`prepare-keys` bottlenecked on `OPTIMIZE TABLE keys_store.k0 FINAL`: 817 s elapsed, 16 parts
/ 11.48 GiB, 90% progress. My first instinct was that the gp3 volume's 4000 IOPS was the
constraint and that `modify-volume` (my own tagged resource, applied identically to both
arms, therefore unable to bias a within-host A/B) would buy back an hour. The evidence says
otherwise:

```
$ cat /proc/loadavg
1.07 2.67 6.69 1/1650 73294
```
Load 1.07 = exactly one core busy, i.e. the merge is single-thread CPU bound, not I/O bound,
so more IOPS would buy nothing. Decision: **do not modify the volumes**, keep the shape the
runbook's helper hard-codes, and absorb the wait. Revisit trigger: if a later phase shows
device saturation (sustained >900 MB/s or IOPS at the 4000 ceiling) rather than a single busy
core, revisit — and if I do change it, it gets applied to all 8 shards *before* any timed run
so no measurement straddles the change, and recorded as a deviation.

**Teardown readiness proven early, with the gate's power to fail recorded.**

```
$ bash tmp/chj_amac/fleet/teardown_phj_ph.sh --dry-run
=== TEARDOWN phj-ph-ab-20260728 … (DRY RUN) ===
instances: i-069d5483a4d36300d	i-065ebd96c4dd296e2	i-0781d51e1d57c8b1a	i-0d65b4dd8f104e168	i-0f8ece4037a96fadc	i-0cdef32c6d6060ecb	i-0f8dadc4b4757f83d	i-01ab31f17f082596e
volumes:   vol-0274f5096443069ca vol-09c4e9ba1983fdccf vol-0a24edfb74a2bb4a8 vol-0d92b3896d8ef4737 vol-012e93820b6d74251 vol-07ecb7ef8433fc040 vol-00922d1ef1d9576c9 vol-056bf3d5c48f8ca91 vol-0aeff043418eaeb06 vol-0b1198722222e06e4 vol-016d8aa7ded5fbd2b vol-0d4006d03541ec926 vol-0d6f450b79172d1c1 vol-0f899edf947e6ba79 vol-0848273a57ae77390 vol-007d748bef0937ddc
sgs:       sg-021349461933fb060
```
16 volumes = the 8 data clones plus the 8 `DeleteOnTermination` root volumes, all tagged at
creation. Recording this **before** teardown is what makes the post-teardown empty result
meaningful rather than vacuous.

---

## Independent verification pass 1 (Units 1–2) and the blocking finding it produced

A verifier subagent that did none of the execution was given the prompt, the raw JSONL, the
worklog and the gate outputs, and asked to refute. Its verdict was **FIX-THEN-RESHIP** with
one blocking finding and three leads. **Its tooling was degraded** — the `Shell` tool was
unavailable to it for the whole session, so it could not run `sha256sum`, `git`, or `python`
and substituted read-only greps plus `.git/logs/HEAD`. Its independence was intact; its
ability to execute the gates was not. Recorded here because that materially weakens two of
its non-findings, and because the shell-based re-checks below were then run by me, the doer —
which is *not* independent. A second verifier pass with a working shell is required before
final delivery and is noted as such.

### Finding 1 — BLOCKING, and it is real. The per-cell ABAB leader flip never fired.

Verified myself from the raw rows:

```
$ python3 tmp/chj_amac/fleet/recount_independent.py 'tmp/chj_amac/fleet/results_phj_ph/results.shard*.jsonl' tmp/chj_amac/fleet/report_phj_ph.txt
arm leading at run 0 (position 0): {'A': 93}
within-cell positions for one cell (should strictly alternate A,B,A,B,...):
  0:A 1:B 2:A 3:B 4:A 5:B 6:A 7:B 8:A 9:B 10:A 11:B
```

**Mechanism.** `fleet_ab.run_cell` picks the leader positionally:
`order_pair = (0, 1) if cell_index % 2 == 0 else (1, 0)`. `run_sweep_stealing.py` runs
**one cell per `fleet_ab.py sweep` invocation** (it passes `--cells <single id>`), so
`enumerate(cells)` always yields `cell_index = 0` and the leader is always arm A. The flip is
positional, not content-derived, so it cannot survive being sharded one-cell-per-process.
Note the contrast the campaign prompt itself draws: jbmt flips its leader with
`zlib.crc32(unit_id) & 1`, which *is* stable under that sharding; `fleet_ab` is not.

**This is a genuine, previously undisclosed deviation from the protocol I pre-registered**
("ABAB with the per-cell leader flip"). It is disclosed, not argued away.

**Measured impact, rather than an assurance.** Two facts bound it. First, the interleave
*within* a cell is still strict ABAB — arm A at even positions, arm B at odd, shown above —
so both arms sample the same time window and the same thermal/neighbour conditions; what did
not vary is only *who goes first in each pair*. Second, a leave-out-first-pair sensitivity
recount over the same rows (no re-running, no protocol change) moves exactly **one** verdict
of 93:

```
--- sensitivity: drop the first A/B pair, recompute every verdict ---
all 10 runs:    cells=93 win=27 tie=28 loss=22 invalid=16 insufficient=0
runs 1..9 only: cells=93 win=27 tie=27 loss=23 invalid=16 insufficient=0
verdicts that change when the leading pair is dropped: 1
  str:probe.semi_anti.S4.T96.anti                 TIE -> LOSS  (diff +5.06% -> +6.68%)
```

The single affected cell moves **against** the candidate (TIE → LOSS), i.e. the
always-A-first ordering was, in the one place it mattered, mildly *flattering* to the
candidate. So the defect cannot be responsible for the campaign's regression finding — it
works the other way. 92 of 93 verdicts are invariant to who led.

**Not fixed by re-running.** Correcting the flip means editing `fleet_ab.run_cell`'s leader
selection — a change to the measurement protocol mid-campaign, which is forbidden, and which
would also make the results incomparable to the U5 precedent that ran the same positional
flip. The defect is reported, its impact is measured, and the recommendation (re-express the
leader as `crc32(cell_id) & 1` so it survives one-cell-per-invocation) is left for a future
campaign that can adopt it before measuring.

### Finding 2 — VALID, and now settled. "Recomputed independently" was overstated.

The verifier correctly noticed that `fleet/analyze_phj_ph.py` calls
`fleet_ab.cell_verdicts`, so it re-invokes the *same* scoring function and cannot disagree
with it. Fixed by writing `fleet/recount_independent.py`, which does **not** import
`fleet_ab` and re-implements the rule from its documented semantics (both arms present, all
rows valid, ≥5 valid runs per arm, median `duration_us` of B vs A against the per-cell band
the harness printed). It agrees exactly:

```
--- independent recount (no fleet_ab import) ---
all 10 runs: cells=93 win=27 tie=28 loss=22 invalid=16 insufficient=0
```

Identical to `FLEET_AB REPORT RESULT: cells=93 win=27 tie=28 loss=22 invalid=16
insufficient=0`. The report's wording is corrected to distinguish the two.

### Finding 3 — REFUTED with evidence. The sweep console logs exist and are committed.

The verifier reported that the cited `fleet/results_phj_ph/sweep.shard*.log` and
`fleet/sweep_phj_ph.log` do not exist, so the `FLEET_STEALING` line and the OOM exception
lived only in prose. They exist and are tracked:

```
$ git ls-files tmp/chj_amac/fleet/results_phj_ph/ | grep -c 'sweep.shard.*log'
8
$ git ls-files tmp/chj_amac/fleet/sweep_phj_ph.log
tmp/chj_amac/fleet/sweep_phj_ph.log
```
The finding is an artifact of the verifier being unable to run `ls`/`git`. The OOM's arm
attribution *is* independently checkable from `sweep.shard0.log`, which contains
`warmup 0 failed on arm baseline`.

### Finding 4 — REFUTED with evidence. No uncommitted harness edit.

```
$ git status --porcelain
 M tmp/chj_amac/PHJ_PH_WORKLOG.md
```
Only this worklog (being written) is dirty. `fleet_ab.py` was committed as `635aa368fd5`
before any sweep; the ` M tmp/chj_amac/fleet_ab.py` the verifier saw is the *opening* git
status quoted in this worklog's Unit 0 section, i.e. the pre-commit state, not a live delta.

### Findings the verifier looked for and did not find

Binary identity and arm binding (930 rows each, zero cross-binding, every `proc_exe_sha256`
matching a claimed prefix); the 320 invalid rows all naming **arm A**; the OOM cell having
zero rows; absence of reruns; `collect_hash_table_stats_during_joins = 1` on every row
(proved via the plain cell and its `.statson` twin carrying an identical
`settings_fingerprint`, which also confirms `.statson` is now a no-op); U5's arm-B sha being
distinct and its settings fingerprint differing (so the confound is real, as reported); the
31-changed-verdict table transcription (5 rows spot-checked); G2's teeth (106 vs 94, all 12
extras ending `.hash`); and prereg-before-sweep ordering (prereg `96532537d4d` at 17:17:35Z,
earliest sweep row `recorded_at` 17:37:08Z).

---

## Unit 3 — jbmt legacy synthetic, 347 cells

**Pre-sweep smoke: `selftest` with two arms.** Red overall, and every failure names
`partitioned_hash`, which does not exist in either binary:

```
[FAIL] status OK: … Code: 418 … Unexpected value of JoinAlgorithm: 'partitioned_hash'. Must be one of [… 'parallel_hash' …]. (UNKNOWN_JOIN)
[FAIL] partitioned path event nonzero
[FAIL] LEFT ANTI unit status consistent with path events: … 'partitioned_hash' … (UNKNOWN_JOIN)
selftest: FAILURES PRESENT
```
`selftest` has no `--algorithms` flag (verified: passing one is `unrecognized arguments`), so
it always runs the default algorithm pair and cannot pass on these binaries. What the
campaign depends on all passed:

```
[PASS] no spurious path events under plain hash (baseline)
[PASS] no spurious path events under plain hash (candidate)
[PASS] timed runs alternate arms (parallel_hash): baselinecandidatebaselinecandidatebaselinecandidate
[PASS] wrong expected -> INVALID: row_count 400000 != closed-form expected 400001
[PASS] mid-run insert -> INVALID (parts or checksum)
[PASS] fingerprint changes on mutation
```
The two-arm ABAB alternation passing here matters: the prior campaign was single-arm and
never exercised this path. Checked that `selftest`'s "bootstrapping a small keys_store.k0"
did not clobber the real one — `k0` reads `1.02 billion` on **both** arms on two shards.
It does leave a small `bench.build_t`/`probe_t` behind; the sweep drops and recreates
`bench` tables per cell, so it is inert.

**Venue verification before sweeping (per shard, from `jbmt_prep_shard*.log`).** The
strongest identity evidence in the campaign, because it reads the *running process*:

```
--- binary identity as the servers actually run them ---
port 9005 pid 73844 sha256 0d32ef1c96e6d378
port 9006 pid 74425 sha256 06d804546e0f029b
--- verify both tiers against the snapshot reference ---
verify: OK
verify: OK
```
Plus `cloned /mnt/data/jbmt_server/data -> …/data_b (hardlinks: shared inodes, shared page
cache, zero data bytes)`. All 8 shards ended `prep OK`, `rc=0`.

**Sweep launched** 19:50 UTC, `--only` built from the 347 legacy ids:

```
python3 join_bench_mt.py sweep \
  --arm baseline=/home/ubuntu/chj/clickhouse-base:9005 \
  --arm candidate=/home/ubuntu/chj/clickhouse-cand:9006 \
  --algorithms parallel_hash --suite synthetic --tier a \
  --shards 8 --shard <i> --results results.syn.shard<i>.jsonl \
  --only "$(cat only.syn.txt)"
```

jbmt's leader flip **does** work (contrast the `fleet_ab` defect): rows carry
`lead_arm: candidate` and `lead_arm: baseline`, because it is derived from
`zlib.crc32(unit_id) & 1` rather than from a positional index.

Six shards finished with `sweep done: 46 run, 0 not OK/FALLBACK` (and 44/44/46/43/46). Shards
4 and 6 ran long: both sat in `OPTIMIZE TABLE bench.probe_t FINAL` for over 30 minutes
(shard 6: 2208 s) on the largest cells — the same single-threaded merge that dominated
`prepare-keys`, not a hang.

**Judgement call — Unit 4 tier a started on the 6 idle hosts while 4 and 6 finished Unit 3.**
Each shard is a *separate physical machine*, and each machine runs exactly one sweep at a
time, so this cannot contaminate either suite's measurements; it only stops six 96-core hosts
idling for an hour. Unit 3 was **not** abandoned — it ran to completion on the two hosts that
still needed it, and its result is reported in full. Revisit trigger: if any host were ever
asked to run two sweeps at once, stop and discard the overlap.

**Incident, disclosed — the first tier-a launch used the wrong plan partitioning.**
`jbmt_sweep_phj_ph.sh` derived `--shards` from the *number of hosts being launched*
(`NSHARDS=$(grep -c . "$HOSTS")`), which is 6 when launching a 6-host subset, so it issued
`--shards 6` and, on the host labelled shard 7, the out-of-range `--shard 7`:

```
python3 join_bench_mt.py sweep … --suite real --tier a --shards 6 --shard 0 …
```
Caught within ~2 minutes by reading the remote process list rather than trusting the
launcher's own "launched" output. Actions taken, in order:
1. killed the sweeps;
2. **preserved** the partial rows as `results.real_a_misshard6.shard<i>.jsonl` on each host
   (5 hosts had rows; the shard-7 host had none, consistent with `--shard 7` of 6 being
   out of range) — they are *not deleted* and *not scored*, because they were produced under
   a different plan cut than the campaign's;
3. fixed the driver so an explicit `NSHARDS` wins over the host count;
4. relaunched, and verified from the remote process list that it now reads
   `--shards 8 --shard 0`.

This is an infrastructure fault I diagnosed, not a result I disliked, which is the only
basis on which the prompt permits a rerun; both the discarded attempt and the relaunch are
disclosed here and in the report.

**Unit 3 complete — G5 GREEN.**

```
$ cd tmp/chj_amac/fleet/jbmt_results_phj_ph && python3 -c "…" 'results.syn.shard*.jsonl'
legacy 347 missing 0 not-OK 0
[] []
G5_EXIT=0
```
Every one of the 347 named legacy ids is present and `OK`. Original cell ids preserved
verbatim (they are what lets these results join against the other harnesses).

**G7 for this suite — GREEN on all three asserted contents.**

```
$ python3 join_bench_mt.py report-ab --results 'results.syn.shard*.jsonl' --arm-a baseline --arm-b candidate --out AB_REPORT.syn.md
347 result rows (347 multi-arm); statuses: {'OK': 347}
binaries: {'baseline': ['0d32ef1c96e6'], 'candidate': ['06d804546e0f']}
lead arm distribution (ABAB leader): {'candidate': 181, 'baseline': 166}
```
Two distinct shas ✓; both arms lead a non-trivial share (181/166) ✓; no `FALLBACK` in the
statuses ✓ (`{'OK': 347}`, and my own recount confirms `units with any fallback_runs > 0:
none`).

**Orientation trap caught before it could invert the headline.** `report-ab` labels verdicts
from the *reference* arm's point of view — `join_bench_mt.py:1492` is
`return ("win" if va < vb else "loss", ratio)` with `va` = arm A = **baseline**, and the
header says so: `ratio = candidate/baseline; ratio > 1 and 'win' mean baseline better`. So
its `win: 175` means the **baseline** won 175 units, not the candidate. Quoting it directly
would have reported the campaign's biggest result backwards. Every suite in this report is
therefore stated candidate-centrically, with the raw `report-ab` output kept alongside.

**Independent recount (`fleet/recount_jbmt.py`, does not import `join_bench_mt`).** First
attempt disagreed by 2 units (tie 51 vs 53) because I banded on 5% of the baseline median;
the documented rule (`join_memory_bench._noise_band_tie`) is
`max(0.05 * max(median_a, median_b), max(stdev_a, stdev_b))` — 5% of the **larger** median.
Implementing the documented rule correctly gives exact agreement:

```
--- candidate-centric wall verdicts (WIN = candidate better) ---
units scored=347 win=119 tie=53 loss=175
units with any fallback_runs > 0: none
median ratio candidate/baseline = 1.057 (>1 means the candidate is slower/larger)

--- candidate-centric memory verdicts (WIN = candidate better) ---
units scored=347 win=5 tie=261 loss=81
median ratio candidate/baseline = 1.034
```
`arm -> binary sha256 prefixes: {'baseline': ['0d32ef1c96e6'], 'candidate': ['06d804546e0f']}`,
`tool_versions: {'jbmt-v2': 347}`, `algorithms measured: {'parallel_hash': 347}`.

**Unit 3 measured result: the candidate loses this suite.** 119 WIN / 53 TIE / 175 LOSS on
wall, median 5.7% slower; on memory 5 WIN / 261 TIE / 81 LOSS, median 3.4% larger. The
extremes are wide in both directions and are *shape*-structured, not noise:

- biggest candidate wins, all `T96`: `D32000000_K7_mb16…T96` 0.574, `D32000000_K7_mb8…T96`
  0.602, `D8000000_K7_mb16…T96` 0.637, `D32000000_K3_mb8…T96` 0.638 — i.e. the wide-key
  (K7 = 64-byte string, K3 = 8-column numeric) high-thread shapes go up to **43% faster**;
- biggest candidate losses, all `K1` at **low** thread counts: `D8000000_K1_…T2` **3.555×**,
  `…T8` 3.513×, `…T4` 3.509×, `D32000000_K1_…T2` 3.341× — i.e. up to **3.5× slower** on the
  narrow 2-column numeric key at T2–T8.

That low-thread `K1` cluster is a **LEAD**, not a settled cause. It is, however, exactly the
population the payload commit's own message flagged as needing re-validation ("Follow-up:
re-validate the low-thread cells (T1/T48), which previously ran with thread-derived slot
counts") — the candidate now pins 256 slots regardless of thread count, so a T2 query builds
256 slot maps. Consistent with that mechanism, but this campaign has not isolated it: doing
so needs a third arm or a slot-count sweep, neither of which is in scope. Named, not claimed.
