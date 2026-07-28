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
