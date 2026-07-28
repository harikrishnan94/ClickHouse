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
