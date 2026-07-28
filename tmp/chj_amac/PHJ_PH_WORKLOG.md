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
