# SELFTEST — parity harness (tmp/chj_amac/parity/)

Date: 2026-07-27. Host: aarch64 Graviton, 96 cores.
Self-test binary: `/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse`
(aarch64 RelWithDebInfo, sha256 `1676de14c894b1a95f69446745510d3bbfc7f96067102c9c2771833f524f2c59`),
used as BOTH arms unless noted. Secondary binary for the audit/engage
plumbing tests: `tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin`
(sha256 `0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4`,
read-only; it is the only local binary containing the seven shared
`ConcurrentHashJoin*` ProfileEvents).

## 1. Matrix generation

    $ python3 parity_gen.py --out cases.jsonl --stats
    wrote 636 cases to .../cases.jsonl
    cases: 636
    families (10): fixstr=42, key32=92, key64=114, keys128=68, keys256=46,
                   lcstr=42, mixed=84, null64=42, nullstr=42, string=64
    kind-strictness-variant combos: 23
    threads: t4=318 t32=318

636 cases (target 600-900). 16 table shapes (13 standard + 3 dup16),
23 (kind, strictness, variant) combos, join_use_nulls x {0,1},
max_threads alternating {4,32} (both values hit within every shape and jun).

## 2. SQL-shape validity probes (clickhouse local, before generator was frozen)

All 24 candidate join shapes were probed with `clickhouse local` on the
self-test binary. Raw result: every shape returned rows EXCEPT
`ANY FULL JOIN` without `any_join_distinct_right_table_keys=1`:

    ANY FULL JOIN | l.k = r.k | join_use_nulls=0
      -> Code: 48. DB::Exception: ANY FULL JOINs are not implemented. (NOT_IMPLEMENTED)
    ANY FULL JOIN | l.k = r.k | any_join_distinct_right_table_keys=1
      -> 300  17812813007374118174        (works)

Consequence encoded in the generator: FULL/ANY exists only as the RightAny
variant. Also verified: `SETTINGS ... INTO OUTFILE '<f>' TRUNCATE FORMAT TSV`
clause order parses; TSV escapes an embedded zero byte as `\0` (two bytes,
deterministic); UInt64 multiplication wraps silently (bijective mixers are
safe); `leftPad`+`toFixedString` yields exact 16-byte keys; ASOF with
composite keys and Int64/UInt8 keys all run under parallel_hash.

## 3. cityHash64 / Nullable empirical check (mandated)

    $ clickhouse local (build/reldeb), 2026-07-27T17:33:01Z
    SELECT cityHash64(CAST(NULL AS Nullable(UInt64)))  -> NULL
           cityHash64(CAST(0 AS Nullable(UInt64)))     -> 4761183170873013810
           cityHash64(toUInt64(0))                     -> 4761183170873013810
           sum(cityHash64(k))          over (0, NULL, 2) -> 4512574814367629943
           sum(cityHash64(isNull(k), ifNull(toString(k), '')))
                                       over (0, NULL, 2) -> 4989331284489142888

Findings: `cityHash64(NULL)` is NULL and `sum` SKIPS NULLs, so a naive
checksum silently drops every row containing a NULL. Also
`cityHash64(Nullable(0)) == cityHash64(0)` (NULL-vs-0 would collide even if
sum did not skip). Therefore `verdict_chk` wraps every column as the pair
`isNull(c), ifNull(toString(c), '')` — total and collision-free.
Raw output kept in `scratch/cityhash_null_check.txt`.

## 4. Full runs (all 636 cases, identical arms)

Run A — build/reldeb as both arms (exit 0, wall 2m27s; raw log
`scratch/full_run2.log`):

    baseline RUN {"executed": 636, "n_errors": 2, "duration_s": 31.0}
    candidate RUN {"executed": 636, "n_errors": 2, "duration_s": 31.0}
    compared: 636 cases, 10 families, 23 combos, 0 divergences, 2 matched-errors
    WARNING: 2 case(s) raised IDENTICAL exceptions on both arms (...)
    audit[base]: SKIPPED (shared event ConcurrentHashJoinProbeMicroseconds absent from binary)
    audit[cand]: SKIPPED (shared event ConcurrentHashJoinProbeMicroseconds absent from binary)
    AMAC-FORCE PASS: SKIPPED (counters absent)
    PARITY OK (636 cases, 10 families, 23 kind-strictness combos, force-pass: SKIPPED)

Run B — bins/clickhouse-baseline-a05f3ee81ff.bin as both arms (exit 0, wall
2m33s; raw log `scratch/full_run_baseline.log`): identical verdict and the
SAME 2 matched-errors (confirming the product bug pre-exists in the true
baseline), and — because this binary HAS the seven shared events — the
positive audit path fired:

    audit[base]: parallel_hash engaged in 10/10 families: fixstr,key32,key64,keys128,keys256,lcstr,mixed,null64,nullstr,string
    audit[cand]: parallel_hash engaged in 10/10 families: ...
    PARITY OK (636 cases, 10 families, 23 kind-strictness combos, force-pass: SKIPPED)

Wall time budget: ~2.5 min per full pass (vs < 15 min target). ~50 s of that
is sha256 identity hashing of the ~4.8 GB binaries (bin file + /proc/pid/exe
per server). Results occupy ~2.3 GB per arm (`srv_*/out/results`); wiped at
the start of the next run.

Fixed along the way by the smoke test (12 cases): the initial
`start_server` captured a wrapper-subshell pid from `$!` and the default
clickhouse-watchdog forked the real server — pid bookkeeping was wrong and
cleanup orphaned servers. Fix: `( cd dir && exec env
CLICKHOUSE_WATCHDOG_ENABLE=0 ... )` so `$!` IS the server, plus a cross-check
that `$dir/data/status` (`PID: n`, written by the server itself) matches, and
stop_server refuses to kill any pid whose /proc/<pid>/cwd is not our own
server dir.

## 5. Product bug found by the harness (pre-existing, both arms)

The first full run reported 2 cases erroring IDENTICALLY on both arms:
`key8.left.any.nonequi.jun0.t4` and `lcstr.left.any.nonequi.jun0.t4`:

    Code: 49. DB::Exception: Invalid number of rows in Chunk
    UInt8(size = 1673), UInt64(size = 1673), UInt64(size = 1673), UInt8(size = 12500)
    column UInt8 at position 3: expected 1673, got 12500:
    While executing JoiningTransform. (LOGICAL_ERROR)

Bisection of the trigger (all on the BASELINE binary a05f3ee81ff, so this is
pre-existing upstream behavior, NOT a phj-ph regression; it also reproduces
with `join_algorithm='hash'` and `max_threads=1`, so it is not
parallel_hash-specific; it is an exception, not a crash):

  * requires materializing the joined rows (`SELECT *`; `count()` passes),
  * requires the RIGHT key column in the projection (`r.k`; without it, passes),
  * requires an extra unprojected column in the tables (`t`; without it, passes),
  * requires the non-equi residual condition (`AND l.v < r.v`) with ANY LEFT,
  * requires heavy key duplication (key8: 50 rows/key; lcstr: 20 rows/key).

Minimal deterministic repro (fails on baseline a05f3ee81ff, build/reldeb
phj-ph, hash AND parallel_hash):

    CREATE TABLE b (k UInt8, v UInt64, t UInt64) ENGINE = Memory;
    CREATE TABLE p (k UInt8, v UInt64, t UInt64) ENGINE = Memory;
    INSERT INTO b SELECT number % 200, intHash64(number % 200) % 1000, number FROM numbers(10000);
    INSERT INTO p SELECT number % 256, intHash64(bitXor(number % 256, 7)) % 1000, number FROM numbers(50000);
    SELECT * FROM
      (SELECT l.k AS lk, r.k AS rk, l.v AS lv, r.v AS rv
       FROM p AS l ANY LEFT JOIN b AS r ON l.k = r.k AND l.v < r.v)
    SETTINGS join_algorithm = 'hash', enable_analyzer = 1, max_threads = 1
    FORMAT Null;

Harness consequence: cases where BOTH arms raise the identical normalized
exception are classified `matched-error` (parity-preserving, loud WARNING,
`logs/<id>.matched-error.txt`, gate still OK); an error on one arm only, or
differing errors, remains a gate-failing divergence. The two cases stay in
the matrix deliberately: if a candidate ever makes the exception disappear
(or change) on one arm only, the gate fails.

## 6. Divergence-path self-test (injected faults)

`scratch/divergence_selftest.sh` copies both run dirs, then injects three
faults into the fake candidate: appends bytes to one case's TSV, overwrites
another case's chk, and deletes a third case's outputs while planting a
synthetic error in its run_summary.json. Compare must report exactly those
three, with the right statuses, without stopping at the first:

    DIVERGENCE key32.inner.all.std.jun0.t4: tsv-mismatch-chk-match
    DIVERGENCE key32.inner.any.std.jun0.t32: tsv-match-chk-mismatch
    DIVERGENCE key32.left.all.std.jun0.t4: error-candidate
    COMPARE {"cases": 636, ..., "divergences": 3, ..., "matched_errors": 2, ...}
    DIVERGENCE-SELFTEST OK (3 injected, 3 reported, all files present)

All three `logs/<id>.divergence.txt` + `.repro.sql` files were written; the
2 matched-errors were preserved independently. (First attempt of this script
guessed a nonexistent victim id — fixed to select victims from cases.jsonl.)

## 7. --require-engagement absence-is-fatal self-test

    $ bash run_parity.sh <reldeb> <reldeb> --limit 8 --require-engagement ; echo EXIT=$?
    ...
    AMAC-FORCE PASS: SKIPPED (counters absent)
    PARITY FAIL (1 divergences, see parity/logs/)
    EXIT=1

with `logs/engagement-required.divergence.txt` stating
`STATUS: amac-counters-absent` and listing the three counters.

## 8. Engage-mode plumbing (failure path) against the baseline binary

Manual: started the baseline binary (which has the seven shared events but
not the AMAC counters) on scratch ports 19301/18301 with
`CLICKHOUSE_JOIN_AMAC=force` in the environment (ignored by this binary, as
expected — env probing is harmless), then ran `parity_driver.py engage`:

    engaged: 0 / 10
    failures: ['fixstr', 'key32', 'key64', 'keys128', 'keys256', 'lcstr', 'mixed', 'null64', 'nullstr', 'string']
    key64 family report: { "ConcurrentHashJoinAmacBuildRows": 0, ...,
      "ConcurrentHashJoinBuildMicroseconds": 517,
      "ConcurrentHashJoinProbeMicroseconds": 2527, ... }

This validates: per-family subset selection, log_comment tagging,
SYSTEM FLUSH LOGS + system.query_log ProfileEvents extraction (shared events
positive), the >0 assertion, and per-family divergence-file writing. Note
run_parity.sh never reaches engage for such a binary (the binary grep gates
it); this was a driver-level test of the machinery.

## KNOWN GAPS

1. The AMAC force-pass SUCCESS path (counters present AND incrementing) is
   untestable today: no binary implements the Units 2-3 contract. Validated
   pieces: binary auto-detection (grep for the counter strings), the SKIPPED
   line, --require-engagement fatality, and the engage assertion/reporting
   machinery (failure path, section 8). When Unit 2 lands, the first
   `--require-engagement` run is the real test.
2. Counter presence is detected by grepping the candidate binary for the
   event-name strings (ProfileEvents names are literal strings in .rodata).
   A binary that contains the strings but never registers the events would
   pass detection and then fail the engage assertion (fail-close, correct
   direction). system.events cannot distinguish absent from zero, hence the
   grep.
3. `verdict_chk` checksums only the projected columns; it is a SECONDARY
   check and byte-diff of (a) is primary. The `tsv-match-chk-mismatch`
   status exists to catch harness anomalies between the two.
4. ANY/SEMI/RightAny determinism across arms relies on the generator's
   invariant that every projected non-key column is a function of the join
   key (injective idx->key mappings). Anyone editing key or value
   expressions in parity_gen.py must preserve this or arms can legitimately
   pick different (but equally valid) ANY matches -> false divergences.
   Documented in the module docstring.
5. The two `left.any.nonequi` matched-error cases (key8, lcstr) execute no
   verdict queries while the pre-existing product bug stands (section 5), so
   LEFT/ANY/nonequi coverage for low-cardinality keys is reduced to the
   error-signature comparison. The other 10 nonequi-capable shapes cover the
   combo fully.
6. `max_threads` is alternated (4 or 32 per case, both values within every
   shape and every jun), not fully crossed — full crossing would double the
   matrix beyond the 900-case target.
7. FULL/ANY without any_join_distinct_right_table_keys is NOT_IMPLEMENTED
   upstream and is deliberately absent from the matrix (section 2).
8. Runs are wiped at start (`srv_base`, `srv_cand`, `logs`); evidence from a
   run survives only until the next invocation. Copy out anything you need
   before re-running.
9. (resolved during self-test) Invoking the script through the symlinked
   checkout path (`/home/ubuntu/ClickHouse` -> `/mnt/ch/ClickHouse`) made the
   un-canonicalized SCRIPT_DIR mismatch the kernel-resolved /proc/<pid>/cwd
   in stop_server's ownership check, which then refused to kill and leaked
   both servers (run itself still PARITY OK). Fixed with `pwd -P` +
   `readlink -f`; re-verified from both the repo root and the symlinked
   path (`scratch/cwd_check2.log`: PARITY OK, exit 0, no leaked processes,
   no WARN lines).

## Post-review fixes (2026-07-27, adversarial review verdict FIX)

All self-tests below use ONLY
`tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin` (both arms; the
same-binary guard is overridden with the new `--allow-identical` flag).
Sections 4/7 above predate these fixes; the final-line format has changed
(counts included, see below).

### Fixes applied

1. BLOCKER — matched-error classification (`parity_driver.py`):
   `normalize_error` now returns a signature ONLY for genuine
   `Code: <N>. DB::Exception:` lines and None otherwise, so connection
   failures (`DB::NetException`), client/harness errors, timeouts, and the
   Phase-3 sentinel can never be matched-errors. The error Code is preserved
   verbatim; the `Received from host:port.` clause is REMOVED (ports cannot
   survive into the signature); only genuinely varying tokens are collapsed
   (hex addresses, remaining digit runs — row counts/chunk sizes). `compare`
   additionally enforces a matched-error budget (fail if > 4 or > 2% of
   cases; override only via explicit `--acknowledge-matched-errors N`) and a
   minimum-compared floor (fail unless >= 90% of cases produced real
   comparisons); violations are emitted as `gate_failures` in the COMPARE
   JSON and fail the gate. The final line now reports
   compared/matched-error/failed counts.
2. MAJOR — same-binary guard (`run_parity.sh`): sha256(baseline) ==
   sha256(candidate) is FATAL (exit 2, before any server starts) unless
   `--allow-identical` is given; when allowed, the final line carries an
   `identical-binaries` marker.
3. MAJOR — stop_server fail-close (`run_parity.sh`): a refused stop (pid
   whose /proc/<pid>/cwd is not our server dir) now KEEPS the pid file,
   returns failure, and every caller aborts (exit 3) BEFORE the
   `rm -rf` of the data dirs; the cleanup trap propagates the failure.
4. MINOR — cleanup trap kills the phase-1 background drivers
   (`DRIVER_PIDS`), so an interrupted run leaves no drivers grinding.
5. MINOR — force-pass auto-detect greps the candidate binary only for the
   ASSERTED counters (`AMAC_ASSERT_POSITIVE_EVENTS`); informational
   RingGrowths may be absent without disabling the force pass.
6. MINOR — `run_client` catches `subprocess.TimeoutExpired` and converts it
   to a failed invocation (rc 124): a timed-out chunk falls back to per-case
   execution, a timed-out case becomes a case error (fail-close; never
   matched-error because the text carries no `DB::Exception`).

New flags: `--allow-identical`, `--acknowledge-matched-errors N`,
`--cases-override FILE` (self-test seam: skip generation, use FILE — used by
test (e) below so the shipped generator stays untouched).

`normalize_error` unit checks (raw): connection-refused blobs on ports
19101/19201 -> None (never eligible); Phase-3 sentinel -> None; timeout text
-> None; the Code 49 product bug with differing ports AND differing chunk
sizes -> identical signatures; same text with a different Code -> different
signatures. Signature produced:

    Code: 49. DB::Exception: Invalid number of rows in Chunk UIntN(size = N),
    UIntN(size = N), UIntN(size = N), UIntN(size = N) column UIntN at position
    N: expected N, got N: While executing JoiningTransform. (LOGICAL_ERROR)

### (a) Full 636-case run, identical arms, --allow-identical (raw log `scratch/postfix_a_full.log`)

    compared: 636 cases (634 compared, 2 matched-error, 0 failed), 10 families, 23 combos, 0 divergences
    WARNING: 2 case(s) raised IDENTICAL exceptions on both arms (...)
    audit[base]: parallel_hash engaged in 10/10 families: fixstr,key32,key64,keys128,keys256,lcstr,mixed,null64,nullstr,string
    audit[cand]: parallel_hash engaged in 10/10 families: fixstr,key32,key64,keys128,keys256,lcstr,mixed,null64,nullstr,string
    asserted counter 'ConcurrentHashJoinAmacBuildRows' not found in candidate binary
    AMAC-FORCE PASS: SKIPPED (counters absent)
    PARITY OK (636 cases: 634 compared, 2 matched-error, 0 failed; 10 families, 23 kind-strictness combos, force-pass: SKIPPED, identical-binaries)
    EXIT=0

### (b) Same binaries WITHOUT --allow-identical (raw log `scratch/postfix_b_fatal.log`)

    FATAL: baseline and candidate are the SAME binary (sha256 0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4). A parity gate over identical binaries is vacuous; pass --allow-identical only for harness self-tests.
    EXIT=2

(No server was started; the FATAL fires in phase 0 right after hashing.)

### (c) Injected connection-refused (raw log `scratch/postfix_c_connrefused.log`, script `scratch/connrefused_selftest.sh`)

Part 1 — candidate server stopped mid-run (by its own pid file, cwd
verified, kill by exact pid): every remaining candidate case got Connection
refused; gate exit 1:

    killed candidate server pid 225035 (cwd verified: .../srv_cand)
    gate final line: PARITY FAIL (604 divergences, 1 gate failure(s); 636 cases: 32 compared, 0 matched-error, 604 failed; see parity/logs/, identical-binaries)
    part1: OK (rc=1, no connection-refused matched-errors)

Part 2 — the reviewer's exact vector: both arms carry the
identical-after-digit-collapse refusal blob differing only in port
(19101 vs 19201). Pre-fix this became a matched-error; now:

    DIVERGENCE key32.inner.all.std.jun0.t4: error-both-arms-different
    DIVERGENCE key32.inner.any.std.jun0.t32: error-both-arms-different
    GATE-FAILURE only 0/2 cases produced real comparisons (< 90.0% floor)
    COMPARE {"cases": 2, "compared": 0, "failed": 2, ..., "divergences": 2, ..., "matched_errors": 0, ..., "gate_failures": ["only 0/2 cases produced real comparisons (< 90.0% floor)"]}
    CONNREFUSED-SELFTEST OK (mid-run kill => PARITY FAIL; both-arms-refused => divergence, not matched-error)

### (d) Injected-fault divergence selftest re-run (raw log `scratch/postfix_d_divergence.log`)

    DIVERGENCE key32.inner.all.std.jun0.t4: tsv-mismatch-chk-match
    DIVERGENCE key32.inner.any.std.jun0.t32: tsv-match-chk-mismatch
    DIVERGENCE key32.left.all.std.jun0.t4: error-candidate
    COMPARE {"cases": 636, "compared": 633, "failed": 1, ..., "divergences": 3, ..., "matched_errors": 2, ..., "gate_failures": []}
    DIVERGENCE-SELFTEST OK (3 injected, 3 reported, all files present)

(The synthetic non-DB::Exception error planted for V3 is classified
`error-candidate` — one-sided errors remain divergences.)

### (e) Matched-error budget (raw log `scratch/postfix_e_budget.log`, script `scratch/budget_selftest.sh`)

12 cases of a COPY of the matrix (`scratch/budget/cases_budget.jsonl`, one
per shape, never the 2 known product-bug cases; shipped generator untouched)
got `, parity_selftest_bogus = 1` injected into both verdict queries and were
run through the full gate via `--cases-override`: 12 UNKNOWN_SETTING
matched-errors + 2 pre-existing = 14 > budget 4:

    GATE FAILURE(S): matched-errors 14 > budget 4 (override requires --acknowledge-matched-errors N)
    final: PARITY FAIL (0 divergences, 1 gate failure(s); 636 cases: 622 compared, 14 matched-error, 0 failed; see parity/logs/, identical-binaries)
    exit:  1

Same run with the explicit acknowledgment:

    final: PARITY OK (636 cases: 622 compared, 14 matched-error, 0 failed; 10 families, 23 kind-strictness combos, force-pass: SKIPPED, identical-binaries)
    exit:  0
    BUDGET-SELFTEST OK (14 matched-errors: gate FAILS by default, PASSES only with explicit acknowledgment)

### Extra proofs for fixes 3 and 4 (raw logs `scratch/postfix_f3_failclose.log`, `scratch/postfix_f4_sigterm.log`)

Fix 4 — SIGTERM during phase 1 (bash runs the EXIT trap on untrapped
SIGTERM at `wait`, probed empirically first):

    gate exit after SIGTERM: 143
    drivers still running: 0; servers still alive: 0; pid files left: 0
    SIGTERM-CLEANUP OK

Fix 3 — planted a live foreign pid (a `sleep` with cwd=scratch) into
`srv_base/pid` plus a canary file in `srv_base`, then ran the gate:

    gate exit: 3
    FATAL: pid 257807 cwd '.../parity/scratch' is not '.../parity/srv_base'; refusing to kill, keeping .../srv_base/pid; investigate manually
    pid file kept: yes; srv_base contents survived (canary): yes; foreign process untouched: yes
    FAILCLOSE-SELFTEST OK

### Closing state

A final clean full run after all fault-injection tests confirms the harness
is green (raw log `scratch/postfix_final_clean.log`):

    PARITY OK (636 cases: 634 compared, 2 matched-error, 0 failed; 10 families, 23 kind-strictness combos, force-pass: SKIPPED, identical-binaries)
    EXIT=0

Scratch run-dir copies (`scratch/divtest` etc.) were removed after the
self-tests; all raw logs listed above are kept.

## 10. Staged per-side force-pass detection (2026-07-27, contract-staging fix)

Defect: force-pass detection required ALL of `AMAC_ASSERT_POSITIVE_EVENTS`
(build+probe) in the candidate binary, so a candidate carrying only the
Unit-2 BUILD ring (`ConcurrentHashJoinAmacBuildRows` present,
`...AmacProbeRows` not yet landed) was SKIPPED, and `--require-engagement`
miscounted the skip as a divergence (evidence: `gate_amacbuild.log`).

Fix (parity_gen.py / parity_driver.py / run_parity.sh only):
per-side contract (`AMAC_ASSERT_BUILD_EVENTS` / `AMAC_ASSERT_PROBE_EVENTS`,
RingGrowths stays report-only) + `AMAC_EXPECTED_ENGAGE_FAMILIES` (8) /
`AMAC_EXCLUDED_FAMILIES` (lcstr, mixed); run_parity.sh detects each side
independently and force-asserts every present side (`engage --assert-sides`);
absence of ALL sides under `--require-engagement` is a GATE FAILURE (own
line), never a divergence; engage asserts per family (expected > 0,
excluded == 0 — load-bearing exclusions) on the family's PRIMARY shape
(pinned: key8 is FixedHashMap and never cursor-engages, so family key32
must not select it). Backward-compat names kept; fleet_ab.py cross-checks
`AMAC_ENGAGEMENT_EVENTS`/`AMAC_ENV_VAR`/`SHARED_PROFILE_EVENTS` only —
unchanged, so no fleet_ab.py edit needed; order/run_order.sh carries its
own bash copy of the (unchanged) counter names — checked, no edit needed.

(a) Exact failed gate re-run, baseline a05f3ee81ff vs uncommitted-amacbuild
(build side only), `--require-engagement` (raw `gate_amacbuild2.log`):

    side 'build': asserted counter(s) present in candidate binary: ConcurrentHashJoinAmacBuildRows
    side 'probe': asserted counter 'ConcurrentHashJoinAmacProbeRows' not found in candidate binary
    AMAC side(s) present: build; restarting candidate with CLICKHOUSE_JOIN_AMAC=force
    AMAC-FORCE PASS: engaged 8/8+2x0 (build) (expected engaged: 8/8, excluded at zero: 2/2)
    PARITY OK (636 cases: 634 compared, 2 matched-error, 0 failed; 10 families, 23 kind-strictness combos, force-pass: engaged 8/8+2x0 (build))
    EXIT=0

Per-family counters under force (from `logs/engage.log`): BuildRows equals
the build-table row count for every expected family (fixstr 20000, key32
24000, key64 30000, keys128 30000, keys256 20000, null64 21600 = 24000
minus 10% NULL keys, nullstr 18000, string 20000; RingGrowths 12 each);
lcstr and mixed are 0/0/0 — exclusions verified load-bearing.

(b) Baseline as both arms, `--allow-identical`, WITHOUT
`--require-engagement` (raw `gate_selftest_skip.log`) — skip path intact:

    side 'build': asserted counter 'ConcurrentHashJoinAmacBuildRows' not found in candidate binary
    side 'probe': asserted counter 'ConcurrentHashJoinAmacProbeRows' not found in candidate binary
    AMAC-FORCE PASS: SKIPPED (no AMAC side counters present in candidate binary)
    PARITY OK (636 cases: 634 compared, 2 matched-error, 0 failed; 10 families, 23 kind-strictness combos, force-pass: SKIPPED, identical-binaries)
    EXIT=0

(c) `--require-engagement` with the BASELINE as candidate (raw
`gate_selftest_absence.log`) — negative proof retains teeth, and the
failure is a GATE FAILURE, not a divergence:

    AMAC-FORCE PASS: SKIPPED (no AMAC side counters present in candidate binary)
    GATE FAILURE: --require-engagement given but the candidate binary has NO AMAC side counters (build: ConcurrentHashJoinAmacBuildRows; probe: ConcurrentHashJoinAmacProbeRows)
    PARITY FAIL (0 divergences, 1 gate failure(s); 636 cases: 634 compared, 2 matched-error, 0 failed; see parity/logs/, identical-binaries)
    EXIT=1

(d) `fleet_ab.py selftest --check-events --local --bin <baseline>` (raw
`fleet_selftest_after.log`) — contract cross-check still passes:

    contract-check: constants match parity/parity_gen.py (primary copy)
    check-events: 7/7 shared events present in system.events
    SKIPPED: AMAC engagement counters absent in system.events (['ConcurrentHashJoinAmacBuildRows', 'ConcurrentHashJoinAmacBuildRingGrowths', 'ConcurrentHashJoinAmacProbeRows']); expected until Unit 2 lands
    FLEET_AB SELFTEST RESULT: events=7/7 amac=absent not-run -> PASS
    EXIT=0
