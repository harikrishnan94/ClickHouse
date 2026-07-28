# Pre-registration — `phj-ph` A/B benchmark campaign

Campaign: branch `phj-ph`, payload commit `a0dfbfd965b` ("Decouple the parallel_hash slot
count from the thread count") against the frozen baseline `a05f3ee81ff`.
Campaign start commit: `635aa368fd5` (the committed `timed_settings` change).
`RUN_TAG` for every AWS resource this campaign creates: **`phj-ph-ab-20260728`**.

Rules I bind myself to here, before any acceptance evidence is gathered:

- Each unit's entry below is written and committed **before** the run it predicts. Git
  history is the proof; orientation and smoke output gathered before an entry is
  orientation only and is never cited as acceptance evidence.
- The measurement protocol is frozen: `fleet_ab` 10 timed runs / 4 warmups / ABAB with the
  per-cell leader flip / `MIN_VERDICT_RUNS = 5`; jbmt's own frozen constants
  (`DEFAULT_RUNS`, `REAL_WARMUPS`, `SYN_WARMUPS`) and its fixed 600 s per-query budget.
  No red or INVALID cell is rerun in the hope it flips.
- Effects inside `max(5%, 1 stdev)` of run-to-run variance are **no result** and are not
  claimed as wins or losses. `fleet_ab` enforces its own band; I do not widen it.

---

## Unit 1 — preflight and candidate build (profile BUILD, risk high)

**Predicted outcome.** `HEAD` on `phj-ph` has `a0dfbfd965b` as an ancestor; the frozen
baseline binary still hashes to `0d32ef1c96e6…`; the `timed_settings` change is committed;
a candidate binary builds from `HEAD` and its three AMAC engagement counters
(`ConcurrentHashJoinAmacBuildRows`, `ConcurrentHashJoinAmacBuildRingGrowths`,
`ConcurrentHashJoinAmacProbeRows`) are registered in the built artifact. Nothing is
launched and nothing is measured.

**Gate invocations that will prove it.**

```
sha256sum tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin
git merge-base --is-ancestor a0dfbfd965b HEAD; echo $?
```
Expected: `0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4` and exit `0`
(this is **G1**).

Candidate identity and counter registration (non-gate, MATERIAL, two independent origins
that fail differently — a string table in the ELF vs. a live server enumerating its own
event registry):

```
bash tmp/chj_amac/snapshot_candidate.sh                       # builds + appends MANIFEST row
tail -1 tmp/chj_amac/bins/MANIFEST.tsv
sha256sum tmp/chj_amac/bins/clickhouse-candidate-<short>.bin  # matches the MANIFEST row
strings -a tmp/chj_amac/bins/clickhouse-candidate-<short>.bin | grep -c ConcurrentHashJoinAmac
tmp/chj_amac/bins/clickhouse-candidate-<short>.bin local --query \
  "SELECT name FROM system.events WHERE name LIKE 'ConcurrentHashJoinAmac%' ORDER BY name \
   SETTINGS system_events_show_zero_values = 1"
```
Expected: the `strings` count is ≥ 3, and the live query prints exactly the three counter
names.

**What would refute it.** A baseline sha256 other than `0d32ef1c96e6…`; `merge-base` exit
non-zero; a build failure; fewer than three counters in the built binary (which would make
`--require-engagement` fail closed at cell zero in Unit 2 — that is the failure mode this
check exists to catch *before* an 8-instance fleet is spending money).

**Deviation declared up front.** The `timed_settings` change had to be committed before
building, so `HEAD` is `635aa368fd5`, not `a0dfbfd965b`; the candidate binary is therefore
named for `635aa368fd5`. That commit touches only `tmp/chj_amac/fleet_ab.py` (a Python
harness file), so the C++ payload is exactly `a0dfbfd965b`, which remains an ancestor as G1
requires. Recorded rather than worked around.

---

## Unit 2 — `fleet_ab` measured plan, 94 cells (profile OPTIMIZATION, risk high)

**Predicted outcome — honestly unknown, and ties are a legitimate result.** The payload
decouples the `parallel_hash` slot count from the thread count, so I expect the largest
effects at high thread counts (`T96`) and large sizes (`S4`/`S5`) where slot count drives
hash-table sizing and contention, and near-ties at `T1` where the two are already
equivalent. I do **not** predict a net win. I predict specifically:

- All 94 cells reach a verdict: `invalid=0 insufficient=0` (the below-200 ms duration floor
  is the known INVALID risk; the U5 precedent had 17 floor-INVALIDs over its 105-cell plan,
  but that plan included small build-side cells that the 94-cell measured plan also
  contains, so `invalid=0` is a genuine prediction that may well fail).
- No cell is `DEFAULT-UNCALIBRATED` (`uncalibrated=0`), because `--calibration` is passed.
- The candidate arm carries all three AMAC counters on every cell where the family is not
  compile-time excluded, so `--require-engagement` passes rather than failing at cell zero.

**Gate invocations.**

```
# G2 — the cell list is exactly the measured plan, no `hash` cells
python3 -c "import json; c=json.load(open('tmp/chj_amac/fleet/matrix.json'))['measured_plan']['cells']; h=[x for x in c if x.endswith('.hash')]; assert len(c)==94 and not h, (len(c), h); print(','.join(c))"

# G3 — the stealing driver completed every cell
python3 tmp/chj_amac/fleet/run_sweep_stealing.py --hosts tmp/chj_amac/fleet/hosts.tsv \
  --ssh-key tmp/chj_amac/fleet/ssh/id_ed25519 \
  --arm-a tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin \
  --arm-b tmp/chj_amac/bins/clickhouse-candidate-<short>.bin \
  --name-a baseline --name-b candidate \
  --remote-bin-a /home/ubuntu/chj/clickhouse-base \
  --remote-bin-b /home/ubuntu/chj/clickhouse-cand \
  --calibration tmp/chj_amac/calibration/calibration.json \
  --results-dir <FRESH DIR> --cells "<the 94 ids from G2>" \
  -- --require-engagement
# expect final line: FLEET_STEALING RESULT: cells_run=94 cells_failed=0 shard0=... -> PASS

# G4 — coverage and validity
python3 tmp/chj_amac/fleet_ab.py report --results "$(ls -1 <FRESH DIR>/results.shard*.jsonl | paste -sd,)"
# expect: FLEET_AB REPORT RESULT: cells=94 win=... tie=... loss=... invalid=0 insufficient=0
```

**What would refute it.** `cells_run < 94` or `cells_failed > 0` (G3 red); any non-zero
`invalid` or `insufficient` in G4 — which I will report as a red with its per-cell reasons,
**not** rerun; a `--require-engagement` failure at cell zero (would refute Unit 1's counter
claim); any `DEFAULT-UNCALIBRATED` cell.

**Reruns.** A cell is rerun only for a diagnosed infrastructure fault (ssh/instance/server
crash), never because its verdict is unwelcome; both attempts stay in the JSONL and the
rerun is disclosed in the report.

**Addendum, written before the sweep runs — the calibration file changes.** The G3
invocation above names `tmp/chj_amac/calibration/calibration.json`, following the prompt's
resource map. That file cannot be used: `fleet_ab.resolve_shape` evaluates
`int(calibration[family][size])` and the value there is a dict, so it raises `TypeError`
instead of degrading. The sweep therefore passes
`--calibration tmp/chj_amac/fleet/calibration_rows.json`, which matches the flag's
documented `{family: {size: build_rows}}` contract, is the exact flat projection of the
nested file (zero value mismatches), and leaves 0 of the 94 cells uncalibrated. Neither
file is edited. The prediction `uncalibrated=0` is unchanged and now rests on this file.

**Addendum — a measured red I am recording as a prediction failure before the sweep ends.**
At cell 20 of 94, `lcstr:probe.inner_all.S5.T96` failed on the **baseline** arm with
`Code: 241 … (total) memory limit exceeded: would use 186.12 GiB … maximum: 193.71 GiB …
While executing FillingRightJoinSide`. My pre-registered prediction of `cells_failed=0` is
therefore **refuted**, and G3 will be red with `cells_failed=1`. This is the same cell,
same arm, and same failure mode the U5 precedent recorded (`would use 191.45 GiB`), where
it was dispositioned `EXCLUDED-INVALID`. I am **not** rerunning it, **not** raising the
server memory limit, and **not** removing it from the 94-cell list — all three would be
banned moves. Consequence to state plainly: **G3 and G4 as written cannot go green on this
venue**, because one cell of the frozen measured plan cannot be built by the baseline arm
within one of two co-resident servers' memory. That conflict between the gate's expectation
and the frozen plan is itself a finding for the report, not something to engineer away.

**Addendum — the smoke run that preceded the sweep (orientation, never acceptance).**
One cell (`key64:probe.inner_all.S2.T1`) was run against shard 0 with `--runs 2
--warmups 1` into a **separate** file, `fleet/smoke_phj_ph/smoke.jsonl`, purely to prove the
remote path and the engagement gate before committing 8 hosts to a long sweep
(`fleet_ab.py` itself warns `remote mode is UNTESTED`). Its reduced run count is why it is
orientation only: it is not in the campaign results directory, is not scored, and appears in
no verdict count. It did establish two things the sweep depends on: `--require-engagement`
does **not** trip on the candidate arm (only the expected
`SKIPPED: AMAC engagement counters absent … (arm=baseline)` line appears), and
`rows_source=calibration-file`.

---

## Unit 3 — jbmt legacy synthetic, 347 cells (profile OPTIMIZATION, risk low)

**Cell selection.** The plan's `LEGACY` group is *exactly* the 347 `cell_id`s in
`join_bench_mt_legacy_cells.json` — verified, not assumed:

```
$ python3 join_bench_mt.py plan --suite synthetic 2>&1 >/dev/null | tail -1
-- 432 units (432 cells, 0 queries), 1 shards
$ # plan LEGACY set == legacy json set?  True ; overlap of the other 85 with it: 0
```
I select them with `--only "^(id1|id2|…)$"` built from that JSON (13,752 chars, matching
347/347 LEGACY and 0/85 non-LEGACY) rather than sweeping the 432 superset, so the fleet
spends no time on 85 cells this campaign does not report. Original cell ids are preserved
verbatim — they are the join key against the other harnesses — and nothing is renamed.

**Declared prerequisite and its cost.** The synthetic suite reads `keys_store.k0…k9`, which
the sweep does **not** create; `prepare-keys` does, and the snapshot does not contain them
(the prior campaign never ran this suite). The 347 legacy cells use all of K0–K9, whose
`keys_store` tables are 256M–1.024B rows each (K0 alone is
`INSERT … FROM numbers(1024000000)`). This is a multi-hour, previously-unmeasured
prerequisite with no prior timing to plan against, and it is the main risk to this unit
completing. It is run **once per host on the pre-clone data root**, then the root is cloned,
so both arms share the same keys at zero extra bytes — preparing them per arm would double
both the time and the disk.

**Predicted outcome.** All 347 cells `OK` on both arms; no `FALLBACK` (which would be a hard
failure, not a datapoint); the cross-arm `(row_count, checksum)` reference agrees, so no
`INVALID`. Wall verdicts honestly unknown. I explicitly do **not** predict a net win.

**Gate invocation (G5), run from the directory holding `join_bench_mt_legacy_cells.json`:**

```
python3 -c "import json,glob,sys; leg={c['cell_id'] for c in json.load(open('join_bench_mt_legacy_cells.json'))}; rows=[json.loads(l) for f in glob.glob(sys.argv[1]) for l in open(f) if l.strip()]; st={}; [st.setdefault(r['unit_id'],set()).add(r['status']) for r in rows]; missing=sorted(leg-set(st)); bad=sorted(u for u in leg if st.get(u)!={'OK'}); print('legacy',len(leg),'missing',len(missing),'not-OK',len(bad)); print(missing[:10], bad[:10]); sys.exit(1 if missing or bad else 0)" 'RESULTS_GLOB'
```
Expected: `legacy 347 missing 0 not-OK 0`, exit 0.

**What would refute it.** Any missing id (incomplete sweep), any `FALLBACK` (the
`parallel_hash` path event stayed zero — a hard failure), any `INVALID` (cross-arm
row_count/checksum disagreement, or a run over the 600 s budget), any `ERROR`.

---

## Unit 4 — jbmt real suite, 376 units per tier, tiers a and b (profile OPTIMIZATION, risk high)

**Predicted outcome.** `Planned 376 units; results for 376; missing 0; extraneous 0.` at each
tier, with the only non-`OK` statuses being `TIMEOUT_EXCEEDED`-caused `INVALID` rows on the
pre-registered units below. Wall verdicts honestly unknown.

**Expected INVALID units, named — 4 in total on this ARM-only fleet.** These are the
prompt's accepted tradeoff, and the names are taken from the prior ARM fleet's raw JSONL
(`results.arm.{a,b}.shard*.jsonl`), not from its prose:

Tier a (1):
- `tpch__customer_c_nationkey__supplier_s_nationkey__T16__tiera`

Tier b (3):
- `tpch__customer_c_nationkey__supplier_s_nationkey__T16__tierb`
- `tpch__customer_c_nationkey__supplier_s_nationkey__T96__tierb`
- `tpcds__catalog_sales_cs_bill_customer_sk__store_returns_sr_customer_sk__T16__tierb`

Cause in both families: a join on a ~25-distinct-value key (`nationkey`) produces an
enormous output, and the tier-b `catalog_sales × store_returns` customer join is inherently
over budget. The 600 s budget is **not** raised and these are **not** retried to green.

**Declared borderline case, so it cannot be passed off as a surprise later.**
`tpch__customer_c_nationkey__supplier_s_nationkey__T96__tiera` was `OK` on the prior ARM
fleet but at the edge (that campaign recorded a run at `600000.86 ms`, 0.86 ms *over* on the
T16 variant). This campaign runs **two** resident servers instead of one, so if a fifth
INVALID appears I predict it is that unit. That would be a documented near-miss of the same
cause, not a new finding. Anything INVALID **outside** the `tpch customer × supplier` and
`tpcds catalog_sales × store_returns` families **is** a finding and will be reported as one.

**Gate invocation (G6), per tier:**

```
python3 join_bench_mt.py report --results RESULTS_FILES --suite real --tier TIER --arm ARMNAME
```
Expected: `Planned 376 units; results for 376; missing 0; extraneous 0.` then a `Statuses:`
line whose only non-`OK` entries are the units named above.

**What would refute it.** `missing > 0` or `extraneous > 0`; an INVALID outside the two named
families; any `FALLBACK`.

---

## Unit 5 — reporting and U5 comparison (profile AUTHORING, risk low)

**Predicted outcome.** Per-suite verdict counts recomputable from the raw JSONL alone; the
`fleet_ab` probe-event gate metric reported alongside the wall verdicts; every changed
verdict versus the U5 precedent named with both old and new values; an honest-red list.

**Gate invocation (G7), per jbmt configuration:**

```
python3 join_bench_mt.py report-ab --results RESULTS_FILES --arm-a baseline --arm-b candidate --out AB_REPORT.md
```
`report-ab` always exits 0, so the gate is on **content**, asserted explicitly: the
`binaries:` line names exactly two distinct sha256 prefixes (`0d32ef1c96e6` and the
candidate's `06d804546e0f`); the `lead arm distribution (ABAB leader):` line shows both arms
leading a non-trivial share of units; the statuses list contains no `FALLBACK`.

**Constraint I bind myself to now.** The U5 precedent is the *same measurement lineage* as
this campaign (same baseline binary, same harness family, same fleet shape). It is used only
to name **changed verdicts**, never as independent corroboration of any verdict here. Every
comparison point is re-established fresh: both arms resident on the same host, same volume,
same settings, ABAB-interleaved. No inherited baseline number is compared against a fresh
candidate number.

**What would refute it.** A `binaries:` line with one sha (two ports of the same build); a
lead distribution concentrated on one arm (the ABAB interleave did not alternate); any
`FALLBACK`; a verdict count that a recount from raw JSONL does not reproduce.

---

## Unit 6 — teardown (risk high)

**Predicted outcome.** Every instance, volume and security group this run created is gone,
and the shared snapshot is untouched. This unit runs even if earlier units failed.

**Gate invocation (G8):**

```
aws ec2 describe-instances --profile Dev_AWS_Admin --region ap-south-2 --filters "Name=tag:RUN_TAG,Values=phj-ph-ab-20260728" "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text
aws ec2 describe-volumes --profile Dev_AWS_Admin --region ap-south-2 --filters "Name=tag:RUN_TAG,Values=phj-ph-ab-20260728" "Name=status,Values=creating,available,in-use" --query 'Volumes[].VolumeId' --output text
aws ec2 describe-security-groups --profile Dev_AWS_Admin --region ap-south-2 --filters "Name=tag:RUN_TAG,Values=phj-ph-ab-20260728" --query 'SecurityGroups[].GroupId' --output text
aws ec2 describe-snapshots --snapshot-ids snap-021cbdc2484f86607 --region ap-south-2 --profile Dev_AWS_Admin --query 'Snapshots[0].State'
```
Expected: first three empty, fourth `"completed"`.

**Known obstacle, pre-declared.** The prior campaign recorded `DeleteVolume` being denied by
an identity policy (`DenyDeleteVolumeExceptNdcDbgTagged`) unless the volume carries
`ndc-dbg-target=true`. Teardown tags the 8 campaign volumes with that marker before
deleting. If deletion is still denied, I do **not** escalate: I leave the volumes, list
exactly what would be deleted, and flag it at the top of the report as requiring
authorization.

**Power-to-fail note.** These filters are only meaningful because every instance, volume and
SG was tagged `RUN_TAG` **at creation time** (`launch_phj_ph.sh`, `volumes_phj_ph.sh`); a
tag-filtered proof over an untagged fleet would be vacuously green. Before teardown I record
the same queries returning the **non-empty** live inventory, so the gate is shown to have
the power to fail.
