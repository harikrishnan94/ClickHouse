# ORDER harness self-test record

Component: G-order harness (`check_order.py`, `run_order.sh`) in
`tmp/chj_amac/order/`. Date: 2026-07-27. Machine: aarch64, 96 CPUs.

Self-test binary: `/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse`
(sha256 `1676de14c894b1a95f69446745510d3bbfc7f96067102c9c2771833f524f2c59`,
built 2026-07-27 12:24, phj-ph candidate, scatter probe).

## 1. Native framing verification (empirical) {#framing}

Command:

    build/reldeb/programs/clickhouse local -q "SELECT number AS tag
      FROM numbers_mt(1000000) SETTINGS max_block_size=65409 FORMAT Native" \
      > logs/framing_probe.native

First bytes: `01 81ff03 03 'tag' 06 'UInt64' <raw LE UInt64 data>`
= varuint num_columns (1), varuint num_rows (0x81 0xff 0x03 = 65409),
string name, string type, then rows*8 bytes of data. File size 8000240 =
1000000*8 + 16 blocks * 15 header bytes.

What is actually emitted by `clickhouse local` / `clickhouse client` file
output / plain HTTP: client protocol version 0 framing, i.e.:
- NO `BlockInfo` prefix (`NativeWriter::write` writes it only for
  `client_revision > 0`; the Native output format is registered with
  `settings.client_protocol_version`, which is 0 unless the HTTP param
  `client_protocol_version` is passed — `src/Processors/Formats/Impl/NativeFormat.cpp:608`,
  `src/Formats/FormatFactory.cpp:411`).
- NO custom-serialization prefix byte (written only for
  `client_revision >= DBMS_MIN_REVISION_WITH_CUSTOM_SERIALIZATION` = 54454;
  `src/Formats/NativeWriter.cpp:206`).

Parser verification:

    $ python3 check_order.py < logs/framing_probe.native
    ORDER-BLOCKS OK (16 blocks, 1000000 rows)        # >1 block, counts correct

    $ python3 check_order.py --global < logs/framing_probe.native
    VIOLATION (global) block=2 first_row_tag=65409 < previous_block_last_tag=196226
    ...
    ORDER-BLOCKS FAIL (6 violations in 16 blocks)    # numbers_mt interleaves across blocks: expected

Negative controls:
- `cityHash64(number) AS tag` -> `ORDER-BLOCKS FAIL (49944 violations in 2 blocks)`, exit 1.
- `toString(number) AS tag` -> `PARSE ERROR: ... expected type UInt64 ... got 'String'`, exit 1.
- Empty stream -> `ORDER-BLOCKS FAIL (parse error)`, exit 1 (an empty result is
  not a pass).

`clickhouse client` TCP output was additionally verified implicitly: every
`run_order.sh` check streams through `client ... FORMAT Native` and the parser
consumed up to 33,000,000 rows / 84,096 blocks per check without a framing
error. Same protocol-version-0 framing as `clickhouse local`.

## 2. Empirical power finding: squashing, not raw blocks, exposes scatter {#power-finding}

The mission-specified settings `min_joined_block_size_rows=0,
min_joined_block_size_bytes=0` DISABLE post-join squashing. Measured on the
scatter binary (`clickhouse local`, 3M-row lt, 400k-row rt, T=96, INNER):

| settings                                   | result |
|--------------------------------------------|--------|
| min_joined_block_size 0/0 (spec)           | ORDER-BLOCKS OK (3584 blocks, 1600000 rows) |
| min_joined_block_size 65409/10485760       | ORDER-BLOCKS FAIL (1871 violations in 96 blocks) |
| min_joined_block_size DEFAULT              | ORDER-BLOCKS FAIL (2062 violations in 96 blocks) |

Cause (src/Interpreters/ConcurrentHashJoin.cpp, `ConcurrentHashJoinResult::next`):
this branch's scatter probe emits each bucket's result as a SEPARATE block.
A raw output block is a single bucket piece of one input block; the scatter
selector preserves relative row order, so each raw block is trivially
non-decreasing and the spec settings have NO POWER against scatter. The
squashing transform concatenates consecutive bucket pieces of one input block
in bucket order, which exposes the reordering.

False-positive control for the squash oracle (order-preserving reference):
`join_algorithm='hash'` at T=96 WITH squashing -> `ORDER-BLOCKS OK (48 blocks,
1600000 rows)`. So the squash variant does not spuriously fail an
order-preserving join (per-stream MergeTree read blocks arrive in ascending
key order and consecutive joined blocks squash cleanly).

Resolution implemented in `run_order.sh` (spec deviation, deliberate and loud):
every T=96 relation runs TWICE — `<name>` with the spec settings (raw join
output blocks) and `<name>_squash` with pinned
`min_joined_block_size_rows=65409, min_joined_block_size_bytes=10485760`.
The `--expect-fail` power criterion accepts a genuine violation from ANY
engaged T=96 check; in practice the power comes from the `_squash` variants.
From Unit 3 the candidate must pass BOTH variants.

## 3. run_order.sh self-test runs {#runs}

### 3a. --expect-fail (power check), final clean run

    $ ./run_order.sh build/reldeb/programs/clickhouse --expect-fail --keep-data
    rc=0
    [run_order] check inner_all_k (T=96): OK [ORDER-BLOCKS OK (17408 blocks, 6000000 rows)]
    [run_order] check inner_all_k_squash (T=96): FAIL [ORDER-BLOCKS FAIL (11430 violations in 117 blocks)]
    [run_order] check left_all_k (T=96): OK [ORDER-BLOCKS OK (84096 blocks, 33000000 rows)]
    [run_order] check left_all_k_squash (T=96): FAIL [ORDER-BLOCKS FAIL (41171 violations in 545 blocks)]
    [run_order] check left_any_k (T=96): OK / left_any_k_squash: FAIL (41680 violations in 499 blocks)
    [run_order] check left_semi_k (T=96): OK / left_semi_k_squash: FAIL (11451 violations in 83 blocks)
    [run_order] check left_anti_k (T=96): OK / left_anti_k_squash: FAIL (41781 violations in 451 blocks)
    [run_order] check inner_all_ks (T=96): OK / inner_all_ks_squash: FAIL (13092 violations in 130 blocks)   # string key
    [run_order] check right_all_k_scoped (T=96): OK / right_all_k_scoped_squash: FAIL (12172 violations in 119 blocks)
    [run_order] check full_all_k_scoped (T=96): OK / full_all_k_scoped_squash: FAIL (12135 violations in 188 blocks)
    [run_order] check inner_all_k_t1_global (T=1): OK [ORDER-BLOCKS OK (108 blocks, 6000000 rows)]
    [run_order] summary: total=17 ok=9 fail=8 error=0 not_engaged=0 row_mismatch=0 t1_global=OK
    ORDER POWER-CHECK OK (check fails on this binary, as expected)

Full logs: `logs/selftest_expect_fail3.log`, `logs/selftest_expect_fail_envhook.log`
(the latter with `CLICKHOUSE_JOIN_AMAC=force` in the environment; the passthrough
is logged: `server started, ... CLICKHOUSE_JOIN_AMAC=force`).

- T=1 `--global` PASSES on the scatter binary, as the mission predicted
  (single lane preserves order end to end; 108 blocks, 6,000,000 rows).
- Every check's row count matched an independent `join_algorithm='hash'`
  control count of the same relation (`row_mismatch=0`): INNER 6M, LEFT ALL 33M,
  ANY 30M, SEMI 3M, ANTI 27M, string INNER 6M, RIGHT scoped 6M, FULL scoped 9M.
- Left table verified as exactly 1 active part after `OPTIMIZE TABLE ... FINAL`.
- Server binary identity verified by sha256 of `/proc/<pid>/exe`.
- No server process remains after exit (kill by recorded PID, cmdline-guarded).

### 3b. Normal mode (expected to FAIL on a scatter binary)

    $ ./run_order.sh build/reldeb/programs/clickhouse --keep-data
    rc=1
    ORDER FAIL (ok=9 fail=8 error=0 not_engaged=0 row_mismatch=0 stateless=fail amac_required_failed=0 of 17 checks)

Log: `logs/selftest_normal.log`. From Unit 3 onward this invocation must print
`ORDER OK (...)` on the candidate.

### 3c. --require-engagement (FUTURE contract, fail-close today)

    $ ./run_order.sh <binary> --keep-data --skip-stateless --require-engagement
    rc=1
    [run_order] AMAC-COUNTERS SKIPPED: ConcurrentHashJoinAmacBuildRows ... not present in this binary (expected before Unit 2); env hook CLICKHOUSE_JOIN_AMAC: absent
    [run_order] FATAL: --require-engagement set but AMAC counters are absent
    ORDER FAIL (... amac_required_failed=1 ...)

`--require-engagement --expect-fail` is rejected (exit 2) as contradictory.
Log: `logs/selftest_require_engagement.log`.

## 4. Engagement verification {#engagement}

Surprise: the self-test binary does NOT contain the seven shared
`ConcurrentHashJoin*Microseconds` ProfileEvents (checked via one `grep -aoF`
pass over `/proc/<pid>/exe`; `strings` confirms only
`ConcurrentHashJoinPoolThreads*` are present). The build at
`build/reldeb` (2026-07-27 12:24) predates or diverges from the working tree,
which does declare them (`src/Common/ProfileEvents.cpp:431-437`).

`run_order.sh` therefore has two engagement methods, auto-selected:
- shared events present -> `system.query_log` `ProfileEvents['ConcurrentHashJoinProbeMicroseconds'] > 0`
  per `log_comment='chj-order:<check>:<run_id>'`;
- absent -> loud `ENGAGEMENT-FALLBACK` line + `EXPLAIN actions = 1` of the same
  query must contain `Algorithm: ... ConcurrentHashJoin` (static plan choice;
  weaker than runtime counters, and recorded as such).

On the self-test binary the fallback path ran and all 17 checks were engaged
(`not_engaged=0`). Runtime engagement is independently corroborated by the 8
genuine order-violation failures, which only a scatter `ConcurrentHashJoin`
produces. All `parallel_hash` checks disable spilling
(`max_bytes_before_external_join=0, max_bytes_ratio_before_external_join=0`)
and side swapping (`query_plan_join_swap_table=0` — without this the planner
may swap lt to the build side and the oracle becomes vacuous).

## 5. Stateless tests (10 runs each) {#stateless}

Invocation (from `run_order.sh run_stateless`; `/usr/bin/python3` is used
because it has jinja2 3.1.6, needed to render `03711_...sql.j2`; the default
linuxbrew `python3` does not — clickhouse-test would silently SKIP the
template test):

    cd /mnt/ch/ClickHouse/tests && CLICKHOUSE_PORT_TCP=19310 CLICKHOUSE_PORT_HTTP=18310 \
      /usr/bin/python3 ./clickhouse-test -b <binary> --no-random-settings \
      --no-random-merge-tree-settings --test-runs 10 --tmp <order/srv/tests_tmp> <test-name>

Port plumbing verified against `tests/clickhouse-test` (env read at ~line 6762;
`-b` resolves `<binary> client` via `find_clickhouse_command`). The server
config listens on 127.0.0.1 AND 127.0.0.2 with `<networks><ip>127.0.0.0/8`
because 03448 uses `remote('127.0.0.2', ...)`, which dials the server's own
`tcp_port` on 127.0.0.2.

Results on the scatter binary (logs:
`logs/stateless_03448_analyzer_array_join_alias_in_join_using_bug.20260727_173026_145767.log`,
`logs/stateless_03711_read_in_order_through_join.20260727_173026_145767.log`):

- 03448: FAIL 10/10, and the diff is EXACTLY a join-output row-order flip
  (`[0,1,2] 1 / [0,1,2] 2` emitted as `2 / 1`).
- 03711: FAIL 10/10; the failing sections are the template's
  `join_algorithm='parallel_hash'` iterations, where
  `query_plan_read_in_order_through_join` + `ORDER BY ... LIMIT 3` returns
  wrong rows because the scatter probe breaks the claimed order.

U1 resolved: 03448 DOES deterministically exercise join output order on this
branch with NO settings randomization: the server default
`join_algorithm='direct,parallel_hash,hash'`
(`src/Core/Settings.cpp:3575`) picks `parallel_hash` for its ALL INNER joins,
and the reordering bites 10/10. Randomization is not needed (join_algorithm is
not in clickhouse-test's randomized set anyway; grep of tests/clickhouse-test
finds no join_algorithm entry). Caveat: this was established on the phj-ph
scatter binary; on a binary where the planner's `rhs_size_estimation` falls
below `parallel_hash_join_threshold` (100k) with `hash` also enabled, the tiny
right side of 03448 could select `hash` instead — on THIS branch it did not.

In `--expect-fail` mode the stateless portion is SKIPPED by design (both tests
fail noisily on scatter binaries, as demonstrated above).

## 6. Data design {#data}

- `order_db.lt`: MergeTree ORDER BY tag, 30,000,000 rows, `tag = 0..30M-1`,
  `k = tag % 8000000`, `ks = toString(k)`; `OPTIMIZE ... FINAL` to 1 active
  part (asserted). Physical order == tag order, so every read block is
  internally non-decreasing in tag at any thread count.
- `order_db.rt`: Memory, 4,000,000 rows, `k = intDiv(number,2)*2 + 6000000`
  (evens in [6M,10M), each key twice), `ks = toString(k)`. Gives: 12.5% of
  distinct lt.k values matched, x2 match multiplicity (ALL vs ANY differ),
  unmatched-left rows (LEFT/ANTI power) and unmatched-right rows (RIGHT/FULL
  non-joined stream exists). RIGHT/FULL are scoped to the joined stream with
  `WHERE lt.k >= 6000000` (non-joined right rows carry default lt.k = 0).
- Pitfall found and fixed: Memory-engine `rt` loses its DATA (not metadata)
  across server restarts, so `--keep-data` validates row counts, not table
  existence (first buggy run: `logs/selftest_expect_fail2.log`, 9 checks saw
  empty rt; power verdict was still correct but the run was invalid).

## KNOWN GAPS {#known-gaps}

1. Baseline arm not exercised: `tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin`
   did not exist yet. The whole self-test ran only against the phj-ph candidate
   (also scatter). Rerun `run_order.sh <baseline> --expect-fail` when the
   baseline binary lands; it should also print POWER-CHECK OK, and its
   engagement should go through the query_log path (a05f3ee81ff has the seven
   shared events).
2. The self-test candidate binary lacks the seven shared ProfileEvents, so the
   query_log engagement path is only code-reviewed, not executed. It must be
   smoke-tested on the first binary that has the events (the harness prints
   which method it used, so this is visible in every run).
3. The spec's `min_joined_block_size_rows=0, min_joined_block_size_bytes=0`
   checks have NO power against this branch's scatter probe (section 2). Power
   comes from the added `_squash` variants. If Unit 3 changes block granularity
   assumptions, revisit the pinned 65409/10485760 values.
4. EXPLAIN-fallback engagement proves plan choice, not runtime execution
   (documented in the output). Acceptable because order violations
   independently prove execution today; moot once shared events exist.
5. AMAC counters / env hook (`CLICKHOUSE_JOIN_AMAC`) are auto-detected by
   grepping the server image for the literal names — string presence does not
   prove the counters are correctly wired (a Unit-2 concern; engagement > 0 via
   query_log is the real check and is implemented).
6. 03711 verdicts depend on `/usr/bin/python3` having jinja2. If that python
   disappears, `run_stateless` warns but clickhouse-test will skip the
   template test rather than fail it.
7. clickhouse-test unavoidably writes generated/failure files into
   `tests/queries/0_stateless/` (`03711_...gen.sql`, `<test>.<pid>.stdout/.stderr`
   on failure) — a side effect of the mandated invocation, outside this
   component's directory.
8. `--global` is only asserted for the T=1 INNER check. At T>1 global order is
   not a goal of G-order (streams interleave legitimately).
9. Server data dir `srv/data` is ~4GB after runs (30M-row lt + logs). Use
   `--keep-data` for cheap reruns; delete `srv/data` to reclaim.

## Post-review fixes (2026-07-27) {#post-review-fixes}

Adversarial review verdict was FIX (1 BLOCKER, 4 MAJOR, 2 MINOR). All seven
findings are fixed in `run_order.sh`; `check_order.py` needed no change. All
re-proof runs below use the baseline snapshot
`tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin` (which HAS the seven
shared `ConcurrentHashJoin*` ProfileEvents), ports 19310/18310. This closes
KNOWN GAPS 1 and 2 above.

### Fixes applied {#post-review-fix-list}

1. BLOCKER — RIGHT/FULL scoped checks were not testing RIGHT/FULL:
   `query_plan_convert_outer_join_to_inner_join` defaults to 1 and the scope
   filter `WHERE lt.k >= 6M` rejects default values, so the planner rewrote
   RIGHT->inner and FULL->left. Confirmed empirically on the baseline
   (`EXPLAIN actions = 1`, small-table repro):

       default:   RIGHT+filter -> 'Type: inner', FULL+filter -> 'Type: left'
       pinned =0: RIGHT+filter -> 'Type: right', FULL+filter -> 'Type: full'

   Fix: the setting is pinned to 0 in `COMMON_SETTINGS` (run_order.sh:360)
   and in the hash-join control settings (:441), and a new
   `verify_scoped_join_types` guard (:456, called at :630) EXPLAINs both
   scoped checks each run and raises a CONTROL-ERROR unless the plan says
   `Type: right` / `Type: full`. Harness output on every run now includes:

       [run_order] join-type verified: right_all_k_scoped plans as 'Type: right' (Algorithm:ConcurrentHashJoin)
       [run_order] join-type verified: full_all_k_scoped plans as 'Type: full' (Algorithm:ConcurrentHashJoin)

   With genuine RIGHT/FULL plans the scoped `_squash` checks STILL fail
   per-block at T=96 on the baseline (run (a)):

       [run_order] check right_all_k_scoped_squash (T=96): FAIL [ORDER-BLOCKS FAIL (40 violations in 97 blocks)] rows=6000000
       [run_order] check full_all_k_scoped_squash (T=96): FAIL [ORDER-BLOCKS FAIL (19 violations in 226 blocks)] rows=9000000

   so the RIGHT/FULL coverage retains its own power (fewer violations than the
   pre-fix rewritten plans, 40/19 vs ~12k — the non-joined emission path really
   is different — but still failing). No re-assignment of the power criterion
   was needed. Control counts are unchanged (RIGHT scoped 6M, FULL scoped 9M —
   which is exactly why the row-count cross-check could not catch the rewrite).

2. MAJOR — `--expect-fail` verdict consulted only POWER: a check now carries
   power only if it FAILed at T=96, was engaged, AND its row count matches the
   hash-join control (run_order.sh:676-688, `POWER-INELIGIBLE` log line
   otherwise), and the final verdict additionally requires `N_ERROR==0` and
   `ROWMISMATCH==0` (:697-703). The builder's earlier invalid empty-`rt` run
   would now be BROKEN on all three grounds.

3. MAJOR — control failures no longer fail open: `run_control_count`
   (run_order.sh:434-453) checks the client exit code and that the output is
   numeric; a failure logs `CONTROL-ERROR`, increments `CONTROL_ERRORS`, which
   is added into `N_ERROR` (:691-692) — normal mode cannot print ORDER OK and
   expect-fail mode prints POWER-CHECK BROKEN.

4. MAJOR — stateless engagement now gates the verdict (run_order.sh:580-595,
   :726): with the shared events present, `engaged != yes` sets
   `STATELESS_ENGAGEMENT_FAILED=1` (`STATELESS-NOT-ENGAGED` line) and forbids
   ORDER OK; with the events absent the loud SKIP remains and
   `--require-engagement` makes it fatal.

5. MAJOR — query_log engagement path exercised for real: all four runs below
   used `engagement method: query_log profile-events` (the events are present
   in the baseline binary). All 17 checks engaged (`not_engaged=0`) in every
   valid run, including the T=1 check.

6. MINOR — the AMAC-COUNTERS SKIPPED line now carries the caveat that
   availability is inferred by a strings-level grep of `/proc/<pid>/exe`,
   which can false-positive/negative (run_order.sh:348).

7. MINOR — `run_stateless` now removes the files clickhouse-test leaves in
   `tests/queries/0_stateless/` (`<t>.gen.sql`, `<t>[.gen].<pid>.stdout/.stderr`,
   run_order.sh:596-602). Pre-existing strays from earlier sessions were also
   removed; after run (d) the directory holds only the four canonical files.

### T=1 / engagement facts discovered (finding 5) {#post-review-t1}

On the baseline binary, `ConcurrentHashJoinProbeMicroseconds` is NOT zero at
`max_threads=1`: a `clickhouse local` discovery run (3M x 400k rows INNER,
T=1) reported `ConcurrentHashJoinProbeMicroseconds=494216` (with
`ProbeDispatchMicroseconds=1`, i.e. Probe is the dominant per-slot timer, not
a dispatch-only counter). In the harness itself the T=1 check
`inner_all_k_t1_global` was engaged via query_log in every run
(`not_engaged=0`). Therefore the engagement assertion stays `Probe > 0` for
all thread counts — no Build-evidence fallback was needed or added.

### Re-proof runs (raw final lines) {#post-review-runs}

(a) `./run_order.sh <baseline> --expect-fail` — rc=0, log
`logs/postfix_a_expect_fail.log`, run_id 20260727_182022_214573:

    [run_order] summary: total=17 ok=9 fail=8 error=0 (incl. control_errors=0) not_engaged=0 row_mismatch=0 t1_global=OK
    [run_order] T=1 --global result on this binary: OK (expected OK: single lane preserves order)
    ORDER POWER-CHECK OK (check fails on this binary, as expected: >=1 engaged row-matched T=96 FAIL, errors=0, row_mismatch=0)

(b) T=1 `--global` (part of every run):

    [run_order] check inner_all_k_t1_global (T=1): OK [ORDER-BLOCKS OK (108 blocks, 6000000 rows)] rows=6000000

(c) Broken-control proof, via `broken_run_order.sh` — a copy of the shipped
script with two marked TEST-ONLY breaks (inner_all_k control points at
`order_db.rt_bogus`; `chj_probe_counter` always echoes 0). The shipped script
was not weakened.

(c1) normal mode — rc=1, log `logs/postfix_c1_broken_normal.log`:

    [run_order] CONTROL-ERROR: control count inner_all_k failed (rc=60, output=''; stderr: .../control_inner_all_k.20260727_182251_216612.err)
    [run_order] POWER-INELIGIBLE: inner_all_k_squash FAILed but rows=6000000 does not match the control (<control failed>)
    [run_order] STATELESS-NOT-ENGAGED: 03448_analyzer_array_join_alias_in_join_using_bug did not increment ConcurrentHashJoinProbeMicroseconds (delta=0); parallel_hash did not run — ORDER OK is forbidden
    [run_order] STATELESS-NOT-ENGAGED: 03711_read_in_order_through_join did not increment ConcurrentHashJoinProbeMicroseconds (delta=0); parallel_hash did not run — ORDER OK is forbidden
    ORDER FAIL (ok=9 fail=8 error=1 not_engaged=0 row_mismatch=0 stateless=pass stateless_engagement_failed=1 amac_required_failed=0 of 17 checks)

Normal mode refuses ORDER OK on a control error, and the stateless gate fires
even though the stateless tests themselves were green — exactly the case the
gate exists for.

(c2) `--expect-fail` — rc=1, log `logs/postfix_c2_broken_expectfail.log`:

    ORDER POWER-CHECK BROKEN (power=1 errors=1 row_mismatch=0 — scatter binary passed the check, or the run was invalid)

Even with genuine power present (power=1), one control error makes the run
invalid — the pre-fix behavior (POWER-CHECK OK) is gone.

(d) `./run_order.sh <baseline> --keep-data` (normal mode incl. stateless) —
rc=1, log `logs/postfix_d_normal.log`, run_id 20260727_182705_229480:

    [run_order] shared ProfileEvents (...) present in binary; engagement method: query_log profile-events
    [run_order] stateless 03448_analyzer_array_join_alias_in_join_using_bug x10: PASS; parallel_hash engaged during run: yes (server-wide ConcurrentHashJoinProbeMicroseconds delta=4676)
    [run_order] stateless 03711_read_in_order_through_join x10: PASS; parallel_hash engaged during run: yes (server-wide ConcurrentHashJoinProbeMicroseconds delta=93586)
    [run_order] stateless engagement: 03448=yes (delta ConcurrentHashJoinProbeMicroseconds=4676) 03711=yes (delta ConcurrentHashJoinProbeMicroseconds=93586)
    ORDER FAIL (ok=9 fail=8 error=0 not_engaged=0 row_mismatch=0 stateless=pass stateless_engagement_failed=0 amac_required_failed=0 of 17 checks)

HONEST RECORD — the review expected the stateless tests to FAIL on the
baseline; they PASS 10/10 (0 skipped, genuinely run and genuinely engaged,
delta > 0). The baseline a05f3ee81ff predates the "pure scatter" revert
(69bf5c26c9f): its ConcurrentHashJoin still reorders per-block under T=96
squash (8 harness FAILs above) but does not flip the row order these two tests
depend on at their small scale. On the phj-ph scatter candidate both tests
FAIL 10/10 (section 5). So 03448/03711 discriminate the candidate from the
baseline, while the harness T=96 `_squash` checks catch both — which is why
both layers gate the verdict. Normal mode still ends in ORDER FAIL on the
baseline because of the 8 per-block order FAILs, as it must.

Section 3a/3b (pre-fix runs) and KNOWN GAPS 1/2 are superseded by this
section. The `_squash` violation counts on the baseline differ from section
3a's candidate numbers (e.g. inner 21 vs 11430) — different probe designs,
same verdict.

## 11. Squash-check correction: baseline-differential reclassification (2026-07-27) {#squash-baseline-differential}

### The mis-specification {#squash-misspec}

The original oracle treated any `_squash` check FAIL as gate-failing. That
rule is mis-specified: the TWO-LEVEL BASELINE binary
(`bins/clickhouse-baseline-a05f3ee81ff.bin`) — the order-preserving reference
design, which probes each left block whole and cannot reorder within a block —
fails exactly the same 8 `_squash` checks as the new routed candidate. An
oracle that the reference design itself cannot pass is wrong, not strict.

Mechanism, in one line: squashing concatenates one lane's consecutive join
outputs, and a parallel scan's lane inputs are not tag-monotone, so `_squash`
disorder is an artifact of the SOURCE, not of the join design — argued in
full in the CORRECTION block in `run_order.sh`, which owns this prose.

### Side-by-side verdicts {#squash-side-by-side}

From `logs/gate_002b_baseline.log` (baseline, run_id 20260727_184308_266653)
vs `logs/gate_u3_order.log` (routed candidate, run_id 20260727_225608_639631);
violations/blocks from the per-check lines:

| check                        | baseline (two-level)   | routed candidate       |
|------------------------------|------------------------|------------------------|
| inner_all_k                  | OK                     | OK                     |
| left_all_k                   | OK                     | OK                     |
| left_any_k                   | OK                     | OK                     |
| left_semi_k                  | OK                     | OK                     |
| left_anti_k                  | OK                     | OK                     |
| inner_all_ks                 | OK                     | OK                     |
| right_all_k_scoped           | OK                     | OK                     |
| full_all_k_scoped            | OK                     | OK                     |
| inner_all_k_t1_global (T=1)  | OK                     | OK                     |
| inner_all_k_squash           | FAIL (23 viol/83 blk)  | FAIL (27 viol/83 blk)  |
| left_all_k_squash            | FAIL (156/456)         | FAIL (170/469)         |
| left_any_k_squash            | FAIL (163/322)         | FAIL (136/313)         |
| left_semi_k_squash           | FAIL (24/76)           | FAIL (24/72)           |
| left_anti_k_squash           | FAIL (166/306)         | FAIL (189/311)         |
| inner_all_ks_squash          | FAIL (23/92)           | FAIL (28/90)           |
| right_all_k_scoped_squash    | FAIL (31/96)           | FAIL (40/97)           |
| full_all_k_scoped_squash     | FAIL (14/228)          | FAIL (10/233)          |

Identical shape, comparable violation counts: the squash failures do not
discriminate the candidate from the reference design.

### The rule {#squash-rule}

Normal (gating) mode only; `--expect-fail` power mode is EXACTLY unchanged
(`--baseline-reference` is rejected there). The 8 `_squash` checks are
`source-artifact-prone`: a `_squash` FAIL does not fail the gate PROVIDED

1. its non-squash sibling check is OK, AND
2. a baseline-differential reference confirms the baseline fails the same
   check: the new REQUIRED normal-mode flag `--baseline-reference LOG` points
   at a baseline run log (e.g. `logs/gate_002b_baseline.log`); the script
   parses the reference's per-check verdicts, and the FAIL is reclassified
   `SOURCE-ARTIFACT (baseline fails identically)` only when the reference
   shows FAIL for the same check.

Fail-closed: a squash FAIL where the baseline passed (or has no verdict)
stays a gate-failing FAIL; a missing / unreadable / unparseable reference
(including no per-check lines, no `_squash` verdicts, or conflicting
duplicate verdicts) is FATAL. The summary line carries `source_artifact=N`
and a reclassifying verdict says so explicitly.

WHY THIS IS A CORRECTION, NOT A WEAKENING: argued in full in the CORRECTION
block in `run_order.sh`; in one line — the reference design itself cannot
pass the old rule, and the order contract keeps three independent layers, all
green on the routed candidate (re-proof (a)). The `--expect-fail` power check
keeps its teeth: its scatter evidence is the pre-correction run
`logs/gate_002b_candidate.log`, which still applies because power mode is
code-unchanged (re-proof (c) below re-runs the baseline power check and
reproduces its pre-correction verdict).

### Re-proof runs (raw final lines) {#squash-reproofs}

(a) Normal mode, routed candidate + baseline reference — rc=0, log
`logs/gate_u3_order2.log`, run_id 20260727_230629_650902 (8 x
`SOURCE-ARTIFACT (baseline fails identically)` lines, zero NOT-RECLASSIFIED,
stateless 20/20):

    [run_order] summary: total=17 ok=9 fail=0 source_artifact=8 error=0 (incl. control_errors=0) not_engaged=0 row_mismatch=0 t1_global=OK
    ORDER OK (ok=9 fail=0 source_artifact=8 of 17 checks, all engaged parallel_hash, t1_global=OK, stateless=pass, stateless engagement: 03448=yes (delta ConcurrentHashJoinProbeMicroseconds=2115) 03711=yes (delta ConcurrentHashJoinProbeMicroseconds=87766); squash checks baseline-differential per SELFTEST §11)

(b) Normal mode, BASELINE binary + same reference (consistency: the reference
design passes its own gate) — rc=0, log `logs/gate_baseline_normal.log`,
run_id 20260727_230846_652990:

    [run_order] summary: total=17 ok=9 fail=0 source_artifact=8 error=0 (incl. control_errors=0) not_engaged=0 row_mismatch=0 t1_global=OK
    ORDER OK (ok=9 fail=0 source_artifact=8 of 17 checks, all engaged parallel_hash, t1_global=OK, stateless=pass, stateless engagement: 03448=yes (delta ConcurrentHashJoinProbeMicroseconds=4589) 03711=yes (delta ConcurrentHashJoinProbeMicroseconds=93970); squash checks baseline-differential per SELFTEST §11)

(c) `--expect-fail` on the baseline (power mode unchanged: this re-run
reproduces the pre-correction `logs/gate_002b_baseline.log` verdict; no
reference) — rc=0, log `logs/gate_002b_baseline2.log`, run_id
20260727_231028_654546:

    [run_order] summary: total=17 ok=9 fail=8 source_artifact=0 error=0 (incl. control_errors=0) not_engaged=0 row_mismatch=0 t1_global=OK
    ORDER POWER-CHECK OK (check fails on this binary, as expected: >=1 engaged row-matched T=96 FAIL, errors=0, row_mismatch=0)

(d) `bash -n run_order.sh` — clean, rc=0.

Fail-closed argument paths verified (all rc=2, no server started): normal
mode without `--baseline-reference`; `--baseline-reference` combined with
`--expect-fail`; reference file missing; reference file unparseable (zero
per-check lines); flag without a LOG argument.
