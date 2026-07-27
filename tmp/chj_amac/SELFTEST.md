# SELFTEST — fleet_ab.py + fleet/{matrix_gen.py,check_matrix.py}

> **2026-07-27 adversarial review verdict: FIX — all fixes applied and
> re-proven; see the "Post-review fixes" section at the end.** Sections
> marked [SUPERSEDED] below describe pre-review behavior kept for the
> record; the post-review section carries the current transcripts. All
> original selftest result files were quarantined to
> `fleet/results/selftest_archive/` (they contain rows that are invalid as
> evidence under the fixed rules: `--threads-override` rows counted for the
> nominal-T cell id, and A/A rows carried identical `binary_sha256` on both
> arms). Checksum values in archived files predate the NULL-aware checksum
> and are not comparable to new runs.

Component: fleet A/B driver (Unit-1 harness). Date: 2026-07-27 (UTC).
Host: local 96-core aarch64 Graviton (orientation only, never acceptance
evidence). All commands run from `/mnt/ch/ClickHouse/tmp/chj_amac`.

Binary used for self-testing: `bins/clickhouse-baseline-a05f3ee81ff.bin`
(sha256 `0d32ef1c96e6...` per `bins/MANIFEST.tsv`). The task brief named
`build/reldeb/programs/clickhouse` as the candidate binary, but that binary
PREDATES the profile-events commits: probed via `clickhouse local`, its
`system.events` has 1380 events and ZERO `ConcurrentHashJoin*` entries
(the events exist in the source of both branches). `selftest --check-events`
against it correctly FAILED fail-closed (raw output below) — an accidental
but real power check of the gate. All subsequent self-tests used the
baseline binary from `bins/`, which carries all 7 events.

## (0) Bonus power check — stale binary fails closed [NOT RE-RUN]

Not re-run post-review: `build/reldeb/programs/clickhouse` is being
relinked concurrently and is off-limits; the transcript below is the
original (historical) capture.

    $ python3 fleet_ab.py selftest --check-events --local --bin /mnt/ch/ClickHouse/build/reldeb/programs/clickhouse
    check-events: 0/7 shared events present in system.events
    check-events: MISSING (fail-closed): ['ConcurrentHashJoinBuildMicroseconds', ... all 7 ...]
    SKIPPED: AMAC engagement counters absent in system.events ([...]); expected until Unit 2 lands
    FLEET_AB SELFTEST RESULT: events=0/7 amac=absent not-run -> FAIL
    rc=1

## (1) selftest --check-events — 7 events present, AMAC absent [RE-RUN]

Re-run 2026-07-27 post-review with the exact command below from
`tmp/chj_amac` (review finding 4 showed the relative `--bin` path could not
have worked as written before binary paths were resolved to absolute at
argparse time); output identical — see Post-review fixes (iv).

    $ python3 fleet_ab.py selftest --check-events --local --bin bins/clickhouse-baseline-a05f3ee81ff.bin
    check-events: 7/7 shared events present in system.events
    SKIPPED: AMAC engagement counters absent in system.events (['ConcurrentHashJoinAmacBuildRows',
      'ConcurrentHashJoinAmacBuildRingGrowths', 'ConcurrentHashJoinAmacProbeRows']); expected until Unit 2 lands
    FLEET_AB SELFTEST RESULT: events=7/7 amac=absent not-run -> PASS
    rc=0

`--require-amac` / `--forbid-amac` flip absence/presence into failure
(fail-closed both directions, for the Unit-4 fleet smoke).

## (2) One tiny --aa cell end-to-end (local, two servers) [SUPERSEDED]

SUPERSEDED: `--threads-override` was REMOVED by review findings 1-2 (its
rows counted for the nominal-T cell id and poisoned resume and MEASURED
evidence); the results file is archived. The replacement is a real
nominal-T4 cell run — Post-review fixes (vi). The empirical path-assertion
finding below remains valid evidence.

Cell `key64:probe.inner_all.S1.T48` with hidden `--threads-override 4`,
default 10 timed runs + 4 warmups per arm, strict ABAB.

    $ python3 fleet_ab.py sweep --local --aa --arm-a bins/clickhouse-baseline-a05f3ee81ff.bin \
        --cells key64:probe.inner_all.S1.T48 --threads-override 4 --results fleet/results/aa_selftest.jsonl
    sweep: 1 cells planned, 0 already complete (resume via fleet/results/aa_selftest.jsonl), 1 to run
    === cell 1/1: key64:probe.inner_all.S1.T48 rows_build=32720 rows_probe=130880 expected_rows=130880
        threads=4 rows_source=DEFAULT-UNCALIBRATED ===
    SKIPPED: AMAC engagement counters absent in system.events (arm=aaA); engagement recorded as null
    SKIPPED: AMAC engagement counters absent in system.events (arm=aaB); engagement recorded as null
      cell OK (2.4s)
      A/A key64:probe.inner_all.S1.T48: TIE (diff +8.09%, band 7.9%)
    FLEET_AB SWEEP RESULT: cells_run=1 cells_ok=1 cells_failed=0 results=fleet/results/aa_selftest.jsonl -> PASS
    FLEET_AB AA RESULT: cells=1 tie=1 nontie=0 -> PASS
    rc=0

(The printed diff% is relative to median A while the band is applied to
max(A, B); the TIE is arithmetically consistent: 7.9% * 1.0809 = 8.54% > 8.09%.)

EMPIRICAL PATH-ASSERTION FINDING (first run of this test FAILED, by design):
with all three probe events asserted > 0, all 20 runs were INVALID with
`path assertion: expected > 0 for ['ConcurrentHashJoinProbeDispatchMicroseconds']`.
Root cause read from the baseline source (`ConcurrentHashJoin::joinBlock`,
branch `concurrent-hash-join-profile-events`): when `twoLevelMapIsUsed()` the
scatter is skipped entirely (`dispatched_blocks.emplace_back`, sub-us) and
small per-block dispatches round to 0 us. `ProbeDispatchMicroseconds` is
therefore RECORD-ONLY (still in every row and in the attribution table).
The T=1 run in (8) later demoted BuildDispatch/BuildMerge as well; the final
asserted set is stated there. Observed sample events (valid run at T=4):
Build=837 Dispatch=112 Insert=675 Merge=44 Probe=3183 ProbeDispatch=0 Lookup=1250.

## (3) --verdict-selftest — deliberate A != B must be non-TIE [SUPERSEDED]

SUPERSEDED: re-run post-review on the real nominal cell
`key64:probe.inner_all.S2.T4` (no override) — new transcript in
Post-review fixes. The original transcript (override-based) below:

Same binary both arms; arm B's settings_overlay halves max_threads (4 -> 2)
on cell `key64:probe.inner_all.S2.T48` (threads-override 4, 5 runs, 2 warmups).

    $ python3 fleet_ab.py selftest --local --verdict-selftest --bin bins/clickhouse-baseline-a05f3ee81ff.bin
    === cell 1/1: key64:probe.inner_all.S2.T48 rows_build=1048560 rows_probe=4194240 expected_rows=4194240
        threads=4 rows_source=DEFAULT-UNCALIBRATED ===
      cell OK (2.9s)
    FLEET_AB SWEEP RESULT: cells_run=1 cells_ok=1 cells_failed=0 ... -> PASS
    verdict-selftest: A(T4) vs B(T2) -> LOSS (diff +77.76%, band 15.9%)
    FLEET_AB SELFTEST RESULT: not-run verdict=LOSS -> PASS
    rc=0

The verdict machinery CAN fail: a real slowdown produces LOSS, not TIE.

## (4) matrix_gen + check_matrix on empty results

    $ python3 fleet/matrix_gen.py
    wrote /mnt/ch/ClickHouse/tmp/chj_amac/fleet/matrix.json
    MATRIX_GEN RESULT: universe=1800 measured=94 hash_inband=12 -> OK

    $ python3 fleet/check_matrix.py
    NOTE: no dispositions file at .../fleet/dispositions.json; every universe cell is undispositioned
    disposition counts: MEASURED=0 INFERRED=0 PARITY-ONLY=0 EXCLUDED-INVALID=0 NOT-CLAIMED=0 UNDISPOSITIONED=1800
    1800 undispositioned
    rc=1  (gate red until dispositioned — correct)

DECISION (documented in matrix_gen.py): the universe is the 1800 BASE cells
only; modifier cells (.dup16/.h50/.h05/.jun/.statson) and the 12 .hash
algo-override cells are auxiliary measured evidence, not universe members.
Hence '1800 undispositioned' on empty input.

The 94 measured cells encode the approved plan's 9 blocks as data with a
rationale string per block: probe_grid 27, size_ladder 6, thread_ladder 12,
kind_strictness 24, build 14, dup_heavy 4, hit_rate 4, join_use_nulls 1,
stats_on 2 (disjointness and the total are assert-checked in matrix_gen.py).

## (5) report on the A/A JSONL — TIE [SUPERSEDED]

SUPERSEDED: the input file is archived; `report` now also emits
`insufficient=` (finding 7) and hard-errors on missing files (finding 5) —
current transcripts in Post-review fixes. Original:

    $ python3 fleet_ab.py report --results fleet/results/aa_selftest.jsonl
    CELL key64:probe.inner_all.S1.T48 verdict=TIE A[aaA]=4656us B[aaB]=5033us diff=+8.09% band=7.9%
        spread(A/B)=366/239us runs=10/10 [UNCALIBRATED-SIZE]
      phase attribution (median us per arm):
        ConcurrentHashJoinBuildMicroseconds                         828          837           +8
        ConcurrentHashJoinBuildDispatchMicroseconds                 107          113           +6
        ConcurrentHashJoinBuildInsertMicroseconds                   652          686          +35
        ConcurrentHashJoinBuildMergeMicroseconds                     40           40           +0
        ConcurrentHashJoinProbeMicroseconds                        2637         2796         +158
        ConcurrentHashJoinProbeDispatchMicroseconds                   0            0           +0
        ConcurrentHashJoinProbeLookupMicroseconds                  1202         1228          +26
    WARNING: 1 cell(s) used DEFAULT-UNCALIBRATED size->rows mapping; ...
    FLEET_AB REPORT RESULT: cells=1 win=0 tie=1 loss=0 invalid=0 uncalibrated=1
    rc=0

## (6) plan (LPT sharding over measured 94 + hash 12)

    $ python3 fleet_ab.py plan --shards 8 --out fleet/results/plan_selftest.json
    shard 0..7: cells=11..16, est_cost 2.084e+11..2.088e+11 each
    FLEET_AB PLAN RESULT: cells=106 shards=8 load_balance=1.002 -> OK

## (7) resume check [SUPERSEDED]

SUPERSEDED: resume now requires resumed rows to match the CURRENT
invocation's binary shas (both arms), the cell's nominal threads, and the
settings fingerprint (finding 2) — the run below resumed off rows that the
fixed code correctly refuses (threads_effective=4 under a T48 cell id).
Current positive and negative resume transcripts: Post-review fixes (ii).
Original:

    sweep: 1 cells planned, 1 already complete (resume via fleet/results/aa_selftest.jsonl), 0 to run
      A/A key64:probe.inner_all.S1.T48: TIE (diff +8.09%, band 7.9%)
    FLEET_AB SWEEP RESULT: cells_run=0 cells_ok=0 cells_failed=0 ... -> PASS
    FLEET_AB AA RESULT: cells=1 tie=1 nontie=0 -> PASS

## (8) T=1 path-assertion sanity (second empirical assertion finding) [RE-RUN]

Re-run 2026-07-27 post-review (fresh `fleet/results/t1_selftest.jsonl`; the
original is archived): `cell OK (2.1s)`, `A/A key64:probe.inner_all.S1.T1:
TIE (diff -1.21%, band 3.0%)`, `FLEET_AB AA RESULT: cells=1 tie=1 nontie=0
-> PASS`, rc=0. The empirical narrative below is unchanged.

First run FAILED, by design: all 10 runs INVALID with
`path assertion: expected > 0 for ['ConcurrentHashJoinBuildDispatchMicroseconds']`.
Sample events at T=1: Build=581 BuildDispatch=0 Insert=577 Merge=1
Probe=2361 ProbeDispatch=0 Lookup=1081. With a single slot the build
dispatch is a pass-through (rounds to 0 us) and merge is one timer tick
from zero. Final asserted set (constants block, dated comment):
Build+BuildInsert always > 0; Probe+ProbeLookup > 0 for probe-side cells;
BuildDispatch/BuildMerge/ProbeDispatch record-only. This deviates from the
task brief's literal "Build*>0 always" on empirical grounds; all seven
events are still recorded per run and reported in the attribution table.

Re-run after the fix:

    $ python3 fleet_ab.py sweep --local --aa --arm-a bins/clickhouse-baseline-a05f3ee81ff.bin \
        --cells key64:probe.inner_all.S1.T1 --runs 5 --results fleet/results/t1_selftest.jsonl
      cell OK (1.9s)
      A/A key64:probe.inner_all.S1.T1: TIE (diff -1.00%, band 3.0%)
    FLEET_AB SWEEP RESULT: cells_run=1 cells_ok=1 cells_failed=0 ... -> PASS
    FLEET_AB AA RESULT: cells=1 tie=1 nontie=0 -> PASS
    rc=0

(The 3% band floor engaged: T=1 spread was below the floor.)

## (9) check_matrix validation rules (scratch dispositions) [SUPERSEDED]

SUPERSEDED: the MEASURED/INFERRED gate now additionally requires different
per-arm binaries, nominal threads_effective, and consistent per-arm
settings fingerprints (finding 1) — the MEASURED=1 INFERRED=1 acceptance
below was the review's BLOCKER example. Current transcript (same scratch
dispositions, now rejected): Post-review fixes (i). Original:

Scratch fleet/results/dispositions_scratch.json: a MEASURED cell backed by
the A/A results (10 valid runs/arm), an INFERRED cell from it, a MEASURED
cell with no backing results, a PARITY-ONLY without evidence, and a
non-universe (.dup16) cell:

    $ python3 fleet/check_matrix.py --dispositions fleet/results/dispositions_scratch.json \
        --results fleet/results/aa_selftest.jsonl
    WARNING: disposition for non-universe cell ignored: key64:probe.inner_all.S3.T96.dup16
    disposition counts: MEASURED=1 INFERRED=1 PARITY-ONLY=0 EXCLUDED-INVALID=0 NOT-CLAIMED=0 UNDISPOSITIONED=1798
    ISSUE: key64:probe.inner_all.S2.T96: MEASURED unsupported -- results have 0 arm(s) ... need >= 2
    ISSUE: key64:probe.inner_all.S4.T96: PARITY-ONLY requires non-empty 'evidence'
    1798 undispositioned

## KNOWN GAPS

- REMOTE MODE UNTESTED: `sweep --shard k --shards N --ssh-host ... --ssh-key
  ... --remote-bin-a/-b` is implemented (jbmt ssh pattern, shares all cell
  logic with local mode via the Server interface) but has never run — no
  fleet exists yet. First fleet smoke must exercise: remote config upload,
  PID capture on launch, /proc/<pid>/exe sha check, stop-by-PID (PID-only
  now, NO pgrep fallback — finding 10), remote data-dir wipe, and the
  pre-resume remote `sha256sum` of both deployed binaries.
- SIZE->ROWS DEFAULTS UNCALIBRATED: `FAMILY_SPECS[*].map_bytes_per_row` are
  rough estimates; every output row carries `rows_source=default-uncalibrated`
  and sweep/report print loud warnings until a `--calibration` JSON
  ({family: {size: build_rows}}) is frozen. The plan's ">=200 ms per cell"
  requirement is NOT yet enforced (calibration's job).
- PATH ASSERTION SET verified empirically only on key64 S1/S2 inner at
  T in {1, 4} (see (2) and (8)); the asserted events are the weakest set
  consistent with that evidence (Build/BuildInsert + Probe/ProbeLookup).
  This DEVIATES from the task brief's literal "Build*>0 always" -- see (8).
  If a future binary/shape legitimately zeroes an asserted event, the
  constants block at the top of fleet_ab.py is the single place to adjust.
- ASOF/SEMI/ANY/FULL groups and the non-key64 families have NOT been
  executed end-to-end locally (closed forms are unit-consistent by
  construction but untested against a server). parallel_hash support for
  ASOF on both arms is assumed; if unsupported the path assertion will mark
  those cells INVALID (fail-closed) rather than silently timing hash.
- S5 (16GB) fills are single-threaded by design (FILL_SETTINGS, vendored) —
  slow but deterministic; acceptable per-cell cost on the fleet is
  calibration's problem to confirm.
- The A/A noise band on sub-10ms cells is wide (7.9% observed). The 6-cell
  noise-band freeze (PREREG-002c) should use S2+ cells; S1 at low T is jitter-
  dominated by client startup granularity.
- `build/reldeb/programs/clickhouse` is STALE (predates the profile-events
  commits; zero ConcurrentHashJoin events). Unit-2 self-tests must rebuild it
  before using it as a candidate arm.

## Post-review fixes (2026-07-27, review verdict FIX)

All commands run from `/mnt/ch/ClickHouse/tmp/chj_amac`, binary
`bins/clickhouse-baseline-a05f3ee81ff.bin` only, ports 19510/18510/19520/
18520 only.

### Fixes applied

1. BLOCKER `fleet/check_matrix.py` — MEASURED gate rewritten: per arm role
   it now requires >= 5 valid runs, exactly one `binary_sha256`, exactly one
   `settings_fingerprint`, and `cell_axes.threads_effective` equal to the
   cell id's nominal `T` on every valid run; across arms the two binaries
   must DIFFER unless the disposition entry sets `"aa_acceptable": true`
   (no plan cell does). Identical validation applied to INFERRED from-cells.
   Missing results files are a hard error; dedup is by last attempt nonce.
2. BLOCKER `fleet_ab.py` `completed_cells` — resume counts only rows
   matching the CURRENT invocation: per-role `binary_sha256` (computed
   before the resume decision via `LocalServer.binary_sha` /
   `RemoteServer.binary_sha`), per-role `settings_fingerprint`, and
   `threads_effective` == the cell's nominal threads (`expected_row_filters`).
   The poisoned artifacts (`--threads-override` rows for
   `key64:probe.inner_all.S1.T48`, arm-B `max_threads=2` rows for
   `key64:probe.inner_all.S2.T48`, and all other originals) were moved to
   `fleet/results/selftest_archive/`.
3. MAJOR `fleet_ab.py` sweep — `--require-engagement` is now also enforced
   on RESUMED rows before anything runs: every resumed candidate-arm row
   must carry non-null engagement, else the sweep FAILs with `cells_run=0`.
4. MAJOR `fleet_ab.py` — `--arm-a`, `--arm-b`, and selftest `--bin` are
   resolved to absolute paths at argparse time (`abs_path` type); the exact
   SELFTEST section-1 command was re-run as written and the affected
   transcripts regenerated (sections 1, 3, 8 marked, section 2 superseded).
5. MAJOR `fleet_ab.py` `load_result_rows` — a missing results file is a
   hard error (`SystemExit`), in report, sweep A/A pooling, and selftest;
   fresh-start resume is handled solely by `completed_cells`' existence
   check. Mirrored in `check_matrix.load_results_rows`.
6. MINOR `fleet_ab.py` `dedup_last_attempt` — now keeps only rows of the
   LAST attempt nonce per (cell, arm_role, host), so shrinking `--runs`
   can never pool a longer earlier attempt's tail rows. Mirrored in
   `check_matrix.dedup_last_attempt`.
7. MINOR `fleet_ab.py` — `cell_verdicts` gained `min_runs`
   (`MIN_VERDICT_RUNS = 5`); `report` marks cells below it as INSUFFICIENT
   (no WIN/TIE/LOSS, nonzero rc) with `--min-runs` as the override; the
   sweep A/A path uses `min_runs=args.runs`.
8. MINOR `fleet_ab.py` `join_query_sql` — checksum rewritten NULL-aware via
   `checksum_expr`: `cityHash64` over `(isNull(c), ifNull(toString(c), ''))`
   pairs for every output column (parity-harness pattern), applied to EVERY
   cell (uniform and A/B-symmetric), because `sum(cityHash64(*))` is blind
   to NULL-bearing rows (proof below).
9. MINOR `fleet_ab.py` — `plan` refuses `--shards` > cells (and < 1);
   `sweep --shard` additionally refuses out-of-range shard indices.
10. MINOR `fleet_ab.py` `RemoteServer` — `stop` kills ONLY the PID captured
    at launch (pgrep fallback removed entirely) and raises if the PID
    survives SIGTERM+SIGKILL; `wipe_data` raises on failure and sweep now
    wipes data dirs on BOTH transports, not only `--local`.

The hidden `--threads-override` flag was removed outright (root cause of
the BLOCKER-1/2 poison); selftests use real low-T cell ids (`...T4`,
`...T1`) — the cell grammar accepts any T and plan cells are untouched.

### Re-proof (i) — check_matrix rejects the archived A/A rows as MEASURED

    $ python3 fleet/check_matrix.py --dispositions fleet/results/selftest_archive/dispositions_scratch.json \
        --results fleet/results/selftest_archive/aa_selftest.jsonl
    WARNING: disposition for non-universe cell ignored: key64:probe.inner_all.S3.T96.dup16
    disposition counts: MEASURED=0 INFERRED=0 PARITY-ONLY=0 EXCLUDED-INVALID=0 NOT-CLAIMED=0 UNDISPOSITIONED=1800
    ISSUE: key64:probe.inner_all.S1.T1: INFERRED from-cell not measured -- arm A: 10 run(s) with threads_effective != nominal T of key64:probe.inner_all.S1.T48; arm B: 10 run(s) with threads_effective != nominal T of key64:probe.inner_all.S1.T48; both arms ran the SAME binary (A/A rows are not A/B evidence; set "aa_acceptable": true only for explicitly-A/A cells)
    ISSUE: key64:probe.inner_all.S1.T48: MEASURED unsupported -- arm A: 10 run(s) with threads_effective != nominal T of key64:probe.inner_all.S1.T48; arm B: 10 run(s) with threads_effective != nominal T of key64:probe.inner_all.S1.T48; both arms ran the SAME binary (A/A rows are not A/B evidence; set "aa_acceptable": true only for explicitly-A/A cells)
    ISSUE: key64:probe.inner_all.S2.T96: MEASURED unsupported -- results have 0 arm(s) for key64:probe.inner_all.S2.T96, need >= 2
    ISSUE: key64:probe.inner_all.S4.T96: PARITY-ONLY requires non-empty 'evidence'
    1800 undispositioned
    rc=1

The exact rows the review used (MEASURED=1 INFERRED=1 in section 9) are now
rejected on BOTH new grounds: threads_effective 4 != nominal 48, and
identical binaries on both arms.

### Re-proof (ii) — resume no longer accepts the poisoned T48 rows

Scratch copy of the archived file, then the sweep (runs the cell at REAL
nominal T48 instead of resuming):

    $ cp fleet/results/selftest_archive/aa_selftest.jsonl fleet/results/reproof_resume_t48.jsonl
    $ python3 fleet_ab.py sweep --local --aa --arm-a bins/clickhouse-baseline-a05f3ee81ff.bin \
        --cells key64:probe.inner_all.S1.T48 --runs 5 --results fleet/results/reproof_resume_t48.jsonl
    sweep: 1 cells planned, 0 already complete (resume via fleet/results/reproof_resume_t48.jsonl), 1 to run
    === cell 1/1: key64:probe.inner_all.S1.T48 rows_build=32720 rows_probe=130880 expected_rows=130880 threads=48 rows_source=DEFAULT-UNCALIBRATED ===
      cell OK (2.2s)
      A/A key64:probe.inner_all.S1.T48: TIE (diff -0.39%, band 8.1%)
    FLEET_AB SWEEP RESULT: cells_run=1 cells_ok=1 cells_failed=0 results=fleet/results/reproof_resume_t48.jsonl -> PASS
    FLEET_AB AA RESULT: cells=1 tie=1 nontie=0 -> PASS
    rc=0

Legitimate resume still works — the identical re-invocation:

    sweep: 1 cells planned, 1 already complete (resume via fleet/results/reproof_resume_t48.jsonl), 0 to run
      A/A key64:probe.inner_all.S1.T48: TIE (diff -0.39%, band 8.1%)
    FLEET_AB SWEEP RESULT: cells_run=0 cells_ok=0 cells_failed=0 ... -> PASS
    FLEET_AB AA RESULT: cells=1 tie=1 nontie=0 -> PASS
    rc=0

Nonce-dedup evidence (finding 6): the file holds 30 rows (old 10-run
attempt at threads_effective=4 plus the fresh 5-run T48 attempt); report
pools ONLY the fresh nonce:

    $ python3 fleet_ab.py report --results fleet/results/reproof_resume_t48.jsonl --no-phases
    CELL key64:probe.inner_all.S1.T48 verdict=TIE A[aaA]=20494us B[aaB]=20414us diff=-0.39% band=8.1% spread(A/B)=1668/647us runs=5/5 [UNCALIBRATED-SIZE]
    FLEET_AB REPORT RESULT: cells=1 win=0 tie=1 loss=0 invalid=0 insufficient=0 uncalibrated=1
    rc=0

(`reproof_resume_t48.jsonl` was then moved into `selftest_archive/`.)

### Re-proof (iii) — --require-engagement on a fully-resumed sweep

Third invocation of the same sweep, now with --require-engagement; the
resumed rows carry engagement=null (Unit-2 counters absent), so a fully-
resumed sweep must NOT print PASS:

    $ python3 fleet_ab.py sweep --local --aa --arm-a bins/clickhouse-baseline-a05f3ee81ff.bin \
        --cells key64:probe.inner_all.S1.T48 --runs 5 --results fleet/results/reproof_resume_t48.jsonl \
        --require-engagement
    sweep: 1 cells planned, 1 already complete (resume via fleet/results/reproof_resume_t48.jsonl), 0 to run
    FLEET_AB SWEEP RESULT: cells_run=0 cells_ok=0 cells_failed=0 -> FAIL (--require-engagement: resumed rows lack AMAC engagement for: ['key64:probe.inner_all.S1.T48'])
    rc=1

### Re-proof (iv) — SELFTEST section-1 exact command, from tmp/chj_amac

    $ python3 fleet_ab.py selftest --check-events --local --bin bins/clickhouse-baseline-a05f3ee81ff.bin
    check-events: 7/7 shared events present in system.events
    SKIPPED: AMAC engagement counters absent in system.events (['ConcurrentHashJoinAmacBuildRows', 'ConcurrentHashJoinAmacBuildRingGrowths', 'ConcurrentHashJoinAmacProbeRows']); expected until Unit 2 lands
    FLEET_AB SELFTEST RESULT: events=7/7 amac=absent not-run -> PASS
    rc=0

### Re-proof (v) — report with a missing results file is a hard error

    $ python3 fleet_ab.py report --results fleet/results/nonexistent.jsonl
    ERROR: results file missing: fleet/results/nonexistent.jsonl (fail-closed; fix --results)
    rc=1

### Re-proof (vi) — fresh tiny --aa at REAL T4 nominal (no override anywhere)

    $ python3 fleet_ab.py sweep --local --aa --arm-a bins/clickhouse-baseline-a05f3ee81ff.bin \
        --cells key64:probe.inner_all.S1.T4 --runs 5 --results fleet/results/aa_selftest_t4.jsonl
    sweep: 1 cells planned, 0 already complete (resume via fleet/results/aa_selftest_t4.jsonl), 1 to run
    === cell 1/1: key64:probe.inner_all.S1.T4 rows_build=32720 rows_probe=130880 expected_rows=130880 threads=4 rows_source=DEFAULT-UNCALIBRATED ===
      cell OK (2.0s)
      A/A key64:probe.inner_all.S1.T4: TIE (diff -2.06%, band 11.8%)
    FLEET_AB SWEEP RESULT: cells_run=1 cells_ok=1 cells_failed=0 results=fleet/results/aa_selftest_t4.jsonl -> PASS
    FLEET_AB AA RESULT: cells=1 tie=1 nontie=0 -> PASS
    rc=0

### Additional evidence

Finding 8 — NULL-blindness of the old checksum, measured on the baseline
binary (`clickhouse local`): `cityHash64(NULL)` is `NULL`, and the old
expression cannot see NULL-bearing rows at all, while the new pair pattern
distinguishes them:

    SELECT cityHash64(toNullable(materialize(NULL)))                      -> \N
    old: sum(cityHash64(k)) over {NULL, 1, 2}                             -> 10328741490158169205
    old: sum(cityHash64(k)) over {1, 2}                                   -> 10328741490158169205  (IDENTICAL: NULL row invisible)
    new: sum(cityHash64(isNull(k), ifNull(toString(k), ''))) over {NULL,1,2} -> 2736885994253685567
    new: same over {1, 2}                                                 -> 2565701566193993295  (differs: NULL row seen)

Finding 8 end-to-end — a NULL-producing cell (h50 misses + join_use_nulls=1
+ Nullable key; LEFT JOIN emits 52320 unmatched rows with NULL right-side
columns) through the new checksum, closed form exact:

    $ python3 fleet_ab.py sweep --local --aa --arm-a bins/clickhouse-baseline-a05f3ee81ff.bin \
        --cells null64:probe.left_all.S1.T4.h50.jun --runs 5 --results fleet/results/aa_selftest_jun.jsonl
    === cell 1/1: null64:probe.left_all.S1.T4.h50.jun rows_build=26160 rows_probe=104640 expected_rows=104640 threads=4 rows_source=DEFAULT-UNCALIBRATED ===
      cell OK (2.1s)
      A/A null64:probe.left_all.S1.T4.h50.jun: TIE (diff +0.02%, band 4.6%)
    FLEET_AB SWEEP RESULT: cells_run=1 cells_ok=1 cells_failed=0 results=fleet/results/aa_selftest_jun.jsonl -> PASS
    FLEET_AB AA RESULT: cells=1 tie=1 nontie=0 -> PASS
    rc=0

Finding 7 — a 3-run file is INSUFFICIENT by default; --min-runs overrides:

    $ python3 fleet_ab.py report --results fleet/results/selftest_archive/reproof_minruns.jsonl --no-phases
    CELL key64:probe.inner_all.S1.T4 verdict=INSUFFICIENT valid_runs={'A': 3, 'B': 3} (need >= 5/arm for a verdict; --min-runs overrides)
    FLEET_AB REPORT RESULT: cells=1 win=0 tie=0 loss=0 invalid=0 insufficient=1 uncalibrated=1
    rc=1
    $ python3 fleet_ab.py report --results ... --no-phases --min-runs 3
    CELL key64:probe.inner_all.S1.T4 verdict=TIE ... runs=3/3 [UNCALIBRATED-SIZE]
    FLEET_AB REPORT RESULT: cells=1 win=0 tie=1 loss=0 invalid=0 insufficient=0 uncalibrated=1
    rc=0

Finding 9 — plan refuses shards > cells:

    $ python3 fleet_ab.py plan --shards 200 --out /dev/null
    plan: --shards 200 > 106 cells; refusing -- some shards would be empty and load_balance meaningless (review finding 9)
    rc=1
    $ python3 fleet_ab.py plan --shards 8 --out fleet/results/plan_reproof.json
    FLEET_AB PLAN RESULT: cells=106 shards=8 load_balance=1.002 -> OK
    rc=0

Verdict selftest re-run on the real nominal-T4 cell (section 3 replacement):

    $ python3 fleet_ab.py selftest --local --verdict-selftest --bin bins/clickhouse-baseline-a05f3ee81ff.bin
    === cell 1/1: key64:probe.inner_all.S2.T4 rows_build=1048560 rows_probe=4194240 expected_rows=4194240 threads=4 rows_source=DEFAULT-UNCALIBRATED ===
      cell OK (4.3s)
    verdict-selftest: A(T4) vs B(T2) -> LOSS (diff +86.61%, band 6.6%)
    FLEET_AB SELFTEST RESULT: not-run verdict=LOSS -> PASS
    rc=0

Section 8 re-run (real T1 nominal, fresh file) and section 4 gate re-check:

    A/A key64:probe.inner_all.S1.T1: TIE (diff -1.21%, band 3.0%)
    FLEET_AB AA RESULT: cells=1 tie=1 nontie=0 -> PASS
    $ python3 fleet/check_matrix.py
    disposition counts: MEASURED=0 INFERRED=0 PARITY-ONLY=0 EXCLUDED-INVALID=0 NOT-CLAIMED=0 UNDISPOSITIONED=1800
    1800 undispositioned
    rc=1  (gate red until dispositioned — correct)

### Not fixed / not re-run

- Section (0) (stale `build/reldeb` fails closed) was NOT re-run:
  `build/reldeb/programs/clickhouse` is being relinked concurrently and is
  off-limits; the original transcript stands as historical evidence.
- Remote mode remains UNTESTED (no fleet yet); findings 2/5/10 remote-side
  changes (`RemoteServer.binary_sha` before resume, PID-only stop, wipe on
  both transports, hard-error wipe) are code-reviewed but must be exercised
  by the first fleet smoke, as already listed in KNOWN GAPS.

## Hygiene-fix re-proofs (2026-07-27, fixer for commit 91469b6b22e)

Fixes applied per hygiene/91469b6.{humanize,reduce}.md: band file stores
FRACTIONS (per-shape, derived from noise_band_002c_rev1.jsonl; all three
shapes' same-binary rel_spread 0.0123/0.0127/0.0125 < floor -> 0.03) with a
fail-closed >0.5 unit guard in `load_band_file`; `matrix_gen.py` re-encodes
MATRIX.md's 9 blocks verbatim (new `.anti` cell modifier carries block 4's
LEFT ANTI instantiation; validated on the baseline binary: `ANTI LEFT JOIN`
closed form = probe_rows - hits, right columns projectable); missing
`fleet/matrix.json` is now a hard error and deploy.sh ships
`fleet/{matrix.json,band_local.json}` to `<remote_dir>/fleet/`;
`check_matrix.py` imports fleet_ab row semantics; `lpt_assignment` /
`_server_config_text` / `_server_users_text` / `detect_amac` single-sourced
inside fleet_ab; parity_driver's post-FLUSH sleep removed (flush proven
synchronous: 5/5 tagged rows visible immediately).

### (a) matrix_gen + MATRIX.md cross-check

    $ python3 fleet/matrix_gen.py
    MATRIX_GEN RESULT: universe=1800 measured=94 hash_inband=12 -> OK

Mechanical cross-check (independent transcription of MATRIX.md's table rows,
set-compared against matrix.json's blocks):

    block 1..9: emitted == MATRIX.md, set-equal=True for all 9
    total emitted=94  MD-union=94  emitted-cells-not-mapping-to-any-MD-block=0  MD-cells-missing=0
    MD blocks pairwise disjoint: True
    strzero MEASURED cells (must be none; PARITY-ONLY): none
    per-block counts: 27/6/12/24/14/4/4/1/2 (required 27/6/12/24/14/4/4/1/2)
    MATRIX_MD_CROSSCHECK: PASS

### (b) check_matrix on empty dispositions (run from fleet/)

    disposition counts: MEASURED=0 INFERRED=0 PARITY-ONLY=0 EXCLUDED-INVALID=0 NOT-CLAIMED=0 UNDISPOSITIONED=1800
    1800 undispositioned
    rc=1  (gate red until dispositioned — correct)

### (c) report with the fraction band file + unit-guard proof

    $ python3 fleet_ab.py report --results fleet/results/noise_band_002c_rev1.jsonl --band-file fleet/band_local.json
    (all 6 cells: verdict=TIE ... band=3.0%)
    FLEET_AB REPORT RESULT: cells=6 win=0 tie=6 loss=0 invalid=0 insufficient=0 uncalibrated=0
    rc=0
    $ python3 fleet_ab.py report --results ... --band-file tmp/doctored_band.json   # values 3.0
    ERROR: band file tmp/doctored_band.json: key64:probe.inner_all = 3.0; band file value looks like a percentage; store fractions
    rc=1

### (d) selftest --check-events (now includes the contract cross-check)

    contract-check: constants match parity/parity_gen.py (primary copy)
    check-events: 7/7 shared events present in system.events
    FLEET_AB SELFTEST RESULT: events=7/7 amac=absent not-run -> PASS
    rc=0

### (e) tiny A/A cell end-to-end

    $ python3 fleet_ab.py sweep --aa --local --arm-a bins/clickhouse-baseline-a05f3ee81ff.bin \
        --cells key64:probe.inner_all.S1.T4 --runs 5 --warmups 2 \
        --calibration fleet/calibration_rows.json --results fleet/results/aa_hygiene_t4.jsonl
    A/A key64:probe.inner_all.S1.T4: TIE (diff -0.16%, band 3.9%)
    FLEET_AB SWEEP RESULT: cells_run=1 cells_ok=1 cells_failed=0 results=fleet/results/aa_hygiene_t4.jsonl -> PASS
    FLEET_AB AA RESULT: cells=1 tie=1 nontie=0 -> PASS

### (f) bash -n on every touched shell script

    bash -n fleet/deploy.sh: OK
    bash -n fleet/launch.sh: OK
    bash -n order/run_order.sh: OK
    bash -n order/broken_run_order.sh: OK

### Additional evidence

- Missing frozen plan fails closed (matrix.json moved aside, then restored):

      ERROR: .../fleet/matrix.json missing (frozen plan; fail-closed). Ship fleet/matrix.json next to the driver or pass --cells/--cells-file; regenerate only deliberately via fleet/matrix_gen.py.
      rc=1

- `.anti` end-to-end on the baseline binary: closed forms exact at three
  scales (S1.T1 h100: 0 rows; S1.T1 h05: 1,900,000 rows; S3.T4 h100:
  0 rows); tiny zero-output cells trip the 200 ms duration floor and are
  fail-closed INVALID as designed (13.6/169.8/172.1 ms medians). At a
  measurable scale, 10-run protocol:

      A/A key64:probe.semi_anti.S3.T4.anti: TIE (diff +0.02%, band 3.1%)
      FLEET_AB AA RESULT: cells=1 tie=1 nontie=0 -> PASS

  (A 5-run attempt at the same shape verdicted LOSS at diff +5.38% vs band
  3.7% — same-binary jitter on an unbanded ad-hoc shape, the PREREG-002c
  lesson; the 10-run protocol resolves it.) NOTE for Unit 4: block 4's ANTI
  cells emit zero rows at hit=1.0; if an S2xT96 anti cell runs under the
  duration floor on the fleet, it will fail closed and needs
  re-dispositioning or a probe-rows raise, per MATRIX.md caveat 6.

- Verdict power intact after the refactors:

      verdict-selftest: A(T4) vs B(T2) -> LOSS (diff +90.59%, band 3.1%)
      FLEET_AB SELFTEST RESULT: not-run verdict=LOSS -> PASS

- Plan re-proof regenerated through the extracted `lpt_assignment` (the new
  94+12 plan):

      FLEET_AB PLAN RESULT: cells=106 shards=8 load_balance=1.008 -> OK

- `parity_gen.py` edit is comment-only: regenerated cases byte-identical to
  the committed `parity/cases.jsonl` (636 cases, `diff -q` clean).
- CALIBRATION.md's cited method scripts exist on disk with matching names
  (`calibration/calibrate.py`, `calibration/make_json.py`); no doc fix
  needed — they only need to be added to the commit.
