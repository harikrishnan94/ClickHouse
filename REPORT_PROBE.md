# REPORT_PROBE — `RadixHashJoin` probe-time accounting and optimization

Branch `radix-join-probe-perf`, baseline `b1fa64c7286`. Full per-iteration record:
`WORKLOG_PROBE.md`. Raw artifacts: `tmp/rhj-probe-perf/`.

## Verdicts

- **Unit 0 (wait and orient): DONE.** WAIT MODE honored (poll log
  `tmp/rhj-probe-perf/wait_poll.log`); see deviations.
- **Unit 1 (accounting): GREEN.** Delta-over-floor attribution 91.0–104.3% on 14/15 cells
  (B@T96: 85.4% of a 25 ms delta, residual explicitly attributed to the 1 ms `--time`
  quantization); probe-period tiling 98.8–101.3% everywhere; instrumentation overhead within
  band on all five paired subset cells.
- **Unit 2 (experiments): DONE — five preregistered, implemented, measured.** One kept:
  E2 wave-worker output merging, target cell **−33.9%** plus four protected cells improved
  outright; E1/E3/E4/E5 refuted within band and reverted (numbers in their sections).
- **Unit 3 (consolidation): GREEN.** Correctness (hash verification PASS at full size on two
  shapes and on the wide shape at quarter scale; assertions PASS everywhere; zero mismatches/
  errors/fallbacks), liveness 10/10, lifecycle large-shape PASS with exact counts,
  early-termination 6/6, stateless 5/5 — all on the final binary `4b55481c…`.
- **Independent verification: SHIP** (fresh subagent; target win reproduced at −33.68%; all gates re-run green; findings section at the bottom).

## Flags / deviations (none hidden)

- **Adopted the T16 remedy without its final verdict.** The prerequisite T16 campaign finished
  its own evidence matrix fully green but its session was interrupted before the
  independent-verification verdict was appended to `REPORT_T16.md` and before committing.
  Per the baseline rule this campaign branched from `radix-join-bandwidth-model` tip
  `b8557c1f56b`, then adopted the uncommitted remedy verbatim as attributed commit
  `b1fa64c7286` (provenance: worktree diff matched the report; the gated candidate binary
  `cf662d4c9619…` hash-matched the built tree). This campaign's Unit-3 gate set re-validates
  the remedy independently. Rationale: discarding the worktree diff would have violated
  "do not undo its remedy".
  RECONCILIATION (post-campaign): the T16 session was later resumed and landed its own
  commits on `radix-join-bandwidth-model` (`3fccb389f77` remedy — byte-identical to the
  adopted `b1fa64c7286` — and `9f985d69105` docs including its completed independent
  verification, VERDICT: SHIP). The merge `9c56f97e008` keeps that branch's REPORT_T16.md /
  WORKLOG_T16.md as authoritative; the remedy is thus verified by both campaigns.
- **WAIT MODE poller v1 fired prematurely** on the mission's literal check B (report present
  with a verdict line) while the T16 verification was still benchmarking; waiting continued
  under a stricter condition until the T16 session was confirmed interrupted and the host
  quiet (worklog Unit 0).
- **Overhead-check subset**: instrumentation overhead is bounded with the full paired protocol
  on a regime-spanning subset (A@T1, A@T16, A@T96, B@T96, C@T96), not on all 15 accounting
  cells (the T1 wide cell costs ~25 min per invocation).

## Unit 1 — probe-time accounting

Method: instrumented binary (`4ddd1a81d5ae…`, throwaway commit `06e70fea709`) logs one record
per wave (`RADIXSTART`/`RADIXWAVE2`), per delayed flush (`RADIXFLUSH`/`RADIXDELAYED`) and per
query (`RADIXBUDGET`/`RADIXTOTALS`/`RADIXTEARDOWN`) to stderr; the probe period (post-build to
leaf destruction) tiles exactly into telescoping wall segments from these timestamps
(coverage 98.8–101.3% on all cells). Floors are the probe/build subqueries run scan-only with
identical settings on the clean binary. Walls are `clickhouse local --time` medians (1 warmup
+ 3 timed; C@T1 2 timed). Wave phases are wall segments; drain-internal splits (leaf match vs
queue-push wait; lane pop wait / in-`next` time) are CPU-sums across threads and labeled so.


### Shape A — D=268435456 r=2 bp=pp=1 (wave-heavy; ~86% of probe rows via waves)

| phase (ms) | T1 | T16 | T32 | T64 | T96 |
|---|---|---|---|---|---|
| wall (median, ms) | 60725 | 3294 | 1926 | 1414 | 1449 |
| floor: probe scan (ms) | 23714 | 1632 | 961 | 960 | 938 |
| floor: build scan (ms) | 1791 | 147 | 143 | 139 | 141 |
| delta over floor (ms) | 35220 | 1515 | 822 | 315 | 370 |
| pre-period (setup+build scan+build phase) | 9379 | 872 | 406 | 240 | 244 |
| &nbsp;&nbsp;of which build phase | 7444 | 725 | 292 | 170 | 178 |
| probe period | 51561 | 2422 | 1547 | 1160 | 1205 |
| &nbsp;&nbsp;gaps (pure scan/refill) | 23353 | 572 | 254 | 169 | 148 |
| &nbsp;&nbsp;wave scatter | 5458 | 457 | 302 | 271 | 349 |
| &nbsp;&nbsp;wave drain | 14440 | 1079 | 811 | 579 | 589 |
| &nbsp;&nbsp;wave teardown | 0 | 1 | 0 | 3 | 5 |
| &nbsp;&nbsp;scan to flush (pure scan) | 4179 | 1 | 1 | 1 | 1 |
| &nbsp;&nbsp;flush scatter | 973 | 71 | 51 | 34 | 30 |
| &nbsp;&nbsp;delayed drain (+gap) | 3144 | 191 | 104 | 72 | 70 |
| &nbsp;&nbsp;finalize | 1 | 2 | 3 | 4 | 7 |
| &nbsp;&nbsp;leaf destroy | 31 | 22 | 16 | 12 | 9 |
| attribution coverage of delta | 100.7% | 98.3% | 102.5% | 91.0% | 100.8% |
| waves / pushes / rows-delayed-share | 3 / 49152 / 15.1% | 3 / 49152 / 15.1% | 3 / 49152 / 14.8% | 3 / 49152 / 14.4% | 3 / 49152 / 14.1% |

Raw logs: `tmp/rhj-probe-perf/u1/logs/acct_A_T{1,16,32,64,96}.err` and `floor_A_T*_{probe,build}.err`; regenerate with `python3 tmp/rhj-probe-perf/u1/acct_report.py`.

### Shape B — D=16777216 r=1 bp=pp=1 (100% delayed path)

| phase (ms) | T1 | T16 | T32 | T64 | T96 |
|---|---|---|---|---|---|
| wall (median, ms) | 1762 | 167 | 115 | 91 | 85 |
| floor: probe scan (ms) | 732 | 56 | 35 | 40 | 43 |
| floor: build scan (ms) | 122 | 13 | 13 | 16 | 17 |
| delta over floor (ms) | 908 | 98 | 67 | 35 | 25 |
| pre-period (setup+build scan+build phase) | 418 | 67 | 54 | 38 | 33 |
| &nbsp;&nbsp;of which build phase | 296 | 28 | 17 | 12 | 11 |
| probe period | 1344 | 103 | 62 | 50 | 48 |
| &nbsp;&nbsp;gaps (pure scan/refill) | 0 | 0 | 0 | 0 | 0 |
| &nbsp;&nbsp;wave scatter | 0 | 0 | 0 | 0 | 0 |
| &nbsp;&nbsp;wave drain | 0 | 0 | 0 | 0 | 0 |
| &nbsp;&nbsp;wave teardown | 0 | 0 | 0 | 0 | 0 |
| &nbsp;&nbsp;scan to flush (pure scan) | 736 | 56 | 33 | 23 | 21 |
| &nbsp;&nbsp;flush scatter | 126 | 12 | 9 | 7 | 6 |
| &nbsp;&nbsp;delayed drain (+gap) | 482 | 33 | 18 | 15 | 13 |
| &nbsp;&nbsp;finalize | 0 | 1 | 2 | 5 | 7 |
| &nbsp;&nbsp;leaf destroy | 2 | 2 | 1 | 1 | 1 |
| attribution coverage of delta | 100.2% | 104.3% | 103.1% | 93.1% | 85.4% |
| waves / pushes / rows-delayed-share | 0 / 0 / 100.0% | 0 / 0 / 100.0% | 0 / 0 / 100.0% | 0 / 0 / 100.0% | 0 / 0 / 100.0% |

Raw logs: `tmp/rhj-probe-perf/u1/logs/acct_B_T{1,16,32,64,96}.err` and `floor_B_T*_{probe,build}.err`; regenerate with `python3 tmp/rhj-probe-perf/u1/acct_report.py`.

### Shape C — D=268435456 r=4 bp=pp=7 (wide payload; ~95% via waves)

| phase (ms) | T1 | T16 | T32 | T64 | T96 |
|---|---|---|---|---|---|
| wall (median, ms) | 252326 | 17964 | 12621 | 11586 | 11646 |
| floor: probe scan (ms) | 77993 | 5801 | 3773 | 3846 | 3893 |
| floor: build scan (ms) | 9127 | 827 | 896 | 825 | 902 |
| delta over floor (ms) | 165206 | 11336 | 7952 | 6915 | 6851 |
| pre-period (setup+build scan+build phase) | 33812 | 2805 | 1664 | 1056 | 977 |
| &nbsp;&nbsp;of which build phase | 20120 | 1893 | 1122 | 626 | 613 |
| probe period | 218514 | 14992 | 11070 | 10520 | 10626 |
| &nbsp;&nbsp;gaps (pure scan/refill) | 96551 | 524 | 287 | 200 | 176 |
| &nbsp;&nbsp;wave scatter | 52101 | 4008 | 2365 | 2361 | 2263 |
| &nbsp;&nbsp;wave drain | 58375 | 9795 | 7893 | 7442 | 7712 |
| &nbsp;&nbsp;wave teardown | 0 | 5 | 12 | 23 | 38 |
| &nbsp;&nbsp;scan to flush (pure scan) | 4688 | 2 | 2 | 2 | 3 |
| &nbsp;&nbsp;flush scatter | 2618 | 179 | 95 | 86 | 73 |
| &nbsp;&nbsp;delayed drain (+gap) | 3991 | 330 | 257 | 251 | 264 |
| &nbsp;&nbsp;finalize | 2 | 3 | 4 | 6 | 8 |
| &nbsp;&nbsp;leaf destroy | 188 | 154 | 139 | 116 | 81 |
| attribution coverage of delta | 100.0% | 98.6% | 101.2% | 99.4% | 99.3% |
| waves / pushes / rows-delayed-share | 12 / 393216 / 4.8% | 12 / 393216 / 4.6% | 12 / 393216 / 4.5% | 12 / 393216 / 4.3% | 12 / 393216 / 4.1% |

Raw logs: `tmp/rhj-probe-perf/u1/logs/acct_C_T{1,16,32,64,96}.err` and `floor_C_T*_{probe,build}.err`; regenerate with `python3 tmp/rhj-probe-perf/u1/acct_report.py`.

### Findings (MATERIAL, from the tables and drain diagnostics)

1. **Where drain is exposed, it is consumer-quantum-bound, and block granularity is the
   cause.** Every (partition, wave) pair emits exactly one output block (A: 9.4k rows at
   fanout 16384; C: 2.6k rows at fanout 32768). Consumers pop one block per executor quantum;
   at C@T96 that is 393216 quanta at ~235 us per quantum (only ~6.6 us inside `next`), so the
   wave drain occupies 7.7 s of an 11.6 s query (66%) while pool workers sit 89% blocked on
   the full queue (push-wait 662 s vs leaf match 78 s CPU-sum). On shape A at T>=64 the drain
   hides under the probe scan (negative drain-beyond-scan); at C it does not — the scan
   finishes early (gaps ~2%) and the drain is the critical path.
2. **Wave scatter is a serial wall segment of 14–24% of the query** (lanes park on the
   coordinator mutex meanwhile: 194 s CPU-sum at C@T96). Fanouts above the 8192 per-pass cap
   (A: 16384, C: 32768) force two radix passes — the window is traversed twice.
3. **The delayed path is cheap on every measured shape** (delayed drain <= 2.2% of wall on C,
   13–33 ms on the pure-delayed shape B; pending-mutex wait <= 0.1 ms). Wave teardown, loser
   quanta, weightless quanta and window-append cost are all immaterial (<= 38 ms / <= 63 /
   <= 12 / <= 1.6 s CPU-sum across 96 lanes respectively).
4. **At T1 the probe scan itself runs 16–30% slower inside the join than scan-only**
   (A: +3.8 s, C: +23.2 s) — an interleaving effect absent at T>=16.
5. Build phase (out of probe scope): 8–22% of wall depending on T.

Hypothesis verdicts and refuters: WORKLOG_PROBE.md Unit 1. Instrumentation overhead: pending
(gate decision recorded when the paired check completes).

## Unit 2 — experiments

Five experiments, each preregistered in `WORKLOG_PROBE.md` before implementation (the file's
git history orders preregistration ahead of each implementation commit). Paired protocol
throughout: 5 position-balanced invocations per binary per cell (order R C C R R C C R R C),
band = max(5%, 1 stdev) of the reference samples, judged on `radix_join` `median_ms`;
references rebuilt fresh in-session and hashed (builds were byte-deterministic:
the fresh tip rebuilds reproduced `cf662d4c…` and `4b55481c…` exactly).

### E1 — disable the radix-only lazy-columns-indexing default: REFUTED
Motivation: consumer-side drain cost ~1.6–1.9 us/KB; `RadixHashJoin` alone defaults
`enable_lazy_columns_indexing = true` (planner opt-in only for every other join), wrapping
left columns as `ColumnReplicated` whose unwrap was suspected on the consumer crew.
Change: one-line default flip (candidate `e8b66689…`). Result: C@T96 ref median 11472
(sd 202) -> cand 11606, **+1.17%, within band** -> reverted. Value: eliminated
materialization placement as the drain mechanism.

### E2 — merge wave-worker output blocks before queueing: KEPT (commit `a8b4c058483`)
Motivation: exactly one output block per (partition, wave) — 393216 blocks of ~2.6k rows at
C@T96 — with ~235 us consumer turnaround per popped block (~6.6 us inside `next`), producers
89% blocked on the full queue. Change: per-worker merge to 65409 rows / 2 MiB before pushing
(threads > 1 only; T1 is producer-critical). Result (candidate `4b55481c…`):
**target C@T96 11785 -> 7790 ms (−33.9%)**; floors all green with outright wins on
D=67108864 r=2 T96 (−5.66%) and T64 (−5.49%), D=268435456 r=2 T96 (−7.61%) and r=4 bp=pp=1
T96 (−9.09%); T32 −2.56%, T16 +1.01%, T1 +0.29% within band. Liveness oracle 10/10;
early-termination 6/6. Mechanism verified with the instrumented build: pushes 393216 -> 50881,
drain wall 7712 -> 4313 ms.

### E3 — LEAF_TARGET_BYTES 1 MiB -> 2 MiB (single-pass scatter for A-class fanouts): REFUTED
Motivation: wave scatter = 25% of wall post-E2 on A and C; fanout 16384 forces two radix
passes on the A shape. Change: one constant (candidate `83a4dff7…`). Result: target A@T96
ref 1339 (sd 10.9) -> cand 1296, **−3.21%, within the 5% band** -> reverted per protocol.
LEAD: all five candidate samples sit below all five reference samples — a real-looking
sub-band effect; the [7,7] two-pass plan runs below the SWWC fanout threshold while the
single [13] pass runs above it, so pass elimination bought less than the traffic model
predicted.

### E4 — MERGE_TARGET_BYTES 2 MiB -> 8 MiB (consumer-quantum discriminator): REFUTED
Motivation: post-E2, were consumers still a partial bound? Change: one constant (candidate
`308b6596…`). Result: C@T96 ref 7761 (sd 77.4) -> cand 7663, **−1.26%, within band** ->
reverted. Discrimination value: a 4x further quantum reduction moved the wall ~1% — the
post-E2 drain is **producer-bound**; consumer-side tuning is exhausted.

### E5 — delayed-path output merging: REFUTED
Motivation: the delayed path returns one per-partition block per `nextImpl` call; at B@T96
the delayed drain is 15% of the wall with lanes half-idle inside the stream. Change: the E2
merge applied inside `RadixDelayedBlocks::nextImpl` (candidate `e9f082a1…`). Result: B@T96
ref 84 (sd 1.3) -> cand 82, **−2.38%, within band** -> reverted (delayed drain is partially
leaf-bound and small in absolute terms).

## Cumulative comparison (final branch state vs baseline vs `parallel_hash`)

Kept set = E2 only; final tip `ea8da8ed924` rebuilds byte-identical to the E2 candidate
`4b55481c…`, so the numbers below are measurements of the shipped binary
(`tmp/rhj-probe-perf/u2/e2_*_{ref,cand}_*.log`; `parallel_hash` medians from the same
harness invocations, pooled across both binaries' logs since the join under test does not
affect them):

| cell | baseline radix (`cf662d4c`) | kept set radix (`4b55481c`) | delta | `parallel_hash` |
|---|---|---|---|---|
| D=67108864 r=2 bp=pp=1 T96 | 371 | 350 | −5.66% | 784 |
| D=67108864 r=2 bp=pp=1 T64 | 401 | 379 | −5.49% | 723 |
| D=67108864 r=2 bp=pp=1 T32 | 663 | 646 | −2.56% | 802 |
| D=67108864 r=2 bp=pp=1 T16 | 895 | 904 | +1.01% | 1307 |
| D=67108864 r=2 bp=pp=1 T1 | 14164 | 14205 | +0.29% | 14246 |
| D=268435456 r=2 bp=pp=1 T96 | 1433 | 1324 | −7.61% | 2588 |
| D=268435456 r=4 bp=pp=1 T96 | 2705 | 2459 | −9.09% | 3948 |
| D=268435456 r=4 bp=pp=7 T96 (target) | 11785 | 7790 | −33.90% | 16914 |

`radix_join` beats `parallel_hash` on every protected cell; the kept set widens the
wide-payload margin to 2.17x.

## Risk-accepted leads (not attempted or sub-band)

1. **Pipelining the next wave's scatter under the current wave's drain** — the largest
   remaining lever (scatter is a serial 19–25% of wall: 1949 ms at C@T96, 339 ms at A@T96
   post-E2). Not attempted: the naive design is blocked by pool-capacity coupling (all
   `max_threads` pool slots are held by wave workers parked on the full output queue, so
   scatter sub-jobs queued on the same pool cannot run until the drain tail) and needs a
   worker-budget split plus staged-wave teardown paths in `~RadixHashJoin`; liveness analysis
   sketched in WORKLOG_PROBE.md. Producers measured ~50% idle post-E2 — capacity for a split
   exists.
2. **LEAF_TARGET_BYTES 2 MiB** (E3): consistent −3.2% at A@T96, sub-band; worth a sweep
   where a 3% effect can clear a noise band, jointly with the SWWC fanout threshold.
3. **T1 in-join scan excess** (Unit 1): the probe scan runs 16–30% slower inside the join
   than scan-only at T1 (A +3.8 s, C +23.2 s), vanishing at T>=16 — unexplained interleaving
   effect, diagnosis would start with alternating-phase cache/TLB behavior.
4. Wave-drain producer work is now half leaf match, half merge-copy + materialization
   (220 s CPU-sum at C@T96); fusing the materialize+append double copy would need an
   append-from-index primitive on `IColumn` (out of scope here).

## Evidence matrix

All rows measured on the final branch state `ea8da8ed924` (binary `4b55481c22d0…`, byte-identical
fresh rebuild, `build/reldeb/build_final_tip.log`); raw outputs under `tmp/rhj-probe-perf/`.

| Criterion | Gate invocation (command) | Result (raw) | Verdict |
| --- | --- | --- | --- |
| Accounting recomputes | `python3 tmp/rhj-probe-perf/u1/acct_report.py tmp/rhj-probe-perf/u1/logs` over `acct_*.err` + `floor_*.err` | attribution 91.0–104.3% (14/15 cells); B@T96 85.4% of 25 ms with residual attributed; tiling 98.8–101.3% | GREEN |
| Instrumentation overhead | 5+5 paired, unmodified harness, cells A@{T1,T16,T96}, B@T96, C@T96 (`tmp/rhj-probe-perf/u1/overhead/verdicts.txt`) | −1.2%…+1.6%, all within max(5%, 1 sd) | GREEN |
| ≥5 preregistered experiments | WORKLOG_PROBE.md entries (preregistration precedes implementation in git history) | E1 −1.17% refuted; E2 −33.9% KEPT; E3 −3.21% refuted; E4 −1.26% refuted; E5 −2.38% refuted | GREEN |
| Kept win (E2 target) | `run_paired.sh e2_C_T96 …` D=268435456 r=4 bp=pp=7 T96, 5 pairs (`tmp/rhj-probe-perf/u2/e2_C_T96_*.log`) | ref median 11785 (sd 137.7) → 7790; band 589.2; every cand sample < every ref sample | GREEN (WIN) |
| Protected floors | `run_floors_e.sh e2 …` 7 cells (`tmp/rhj-probe-perf/u2/e2_floors_verdicts.txt`) | D67 T96 −5.66%, T64 −5.49%, T32 −2.56%, T16 +1.01%, T1 +0.29%; D268 r2 T96 −7.61%; r4 T96 −9.09% — no red | GREEN |
| Correctness (verification on) | harness runs with `--verify-max-output-rows` raised (A 600M, B 20M, wide-at-D67 300M) + full-size C at default cap (`tmp/rhj-probe-perf/u3/verify{A,B,Cfull,Cscaled}.log`) | verify PASS/PASS/PASS + SKIP-by-cap (documented); assertions PASS ×4; `fallback=0 invalid=0 errors=0 hash_mismatch=0` ×4 | GREEN |
| Liveness oracle | `ninja -C build/asan unit_tests_dbms` then 10× `timeout --signal=TERM --kill-after=10s 30s build/asan/src/unit_tests_dbms --gtest_filter=RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave` (`build/asan/u3_liveness_run_{1..10}.log`) | 10/10 rc=0, each log exactly `[  PASSED  ] 1 test` | GREEN |
| Lifecycle (large shape) | `tmp/radix-wave-deadlock/run_large_fixed_gate.sh build/reldeb/programs/clickhouse 268435456 4` | exit 0; `Assertions: PASS`; probe/build/joined = 1073741824/268435456/1073741824; clean summary (~60 s) | GREEN |
| Lifecycle (early termination) | `tmp/radix-wave-deadlock/run_early_termination_gate.sh build/reldeb/programs/clickhouse` | 6/6: 3× early_stop PASS (exit 0, LIMIT row), 3× exc PASS (exit 241, `MEMORY_LIMIT_EXCEEDED`), nothing left running | GREEN |
| Stateless tests | scratch server on 9131/8161 (`/proc/<pid>/exe` sha256 = `4b55481c…`), `CLICKHOUSE_PORT_TCP=9131 CLICKHOUSE_PORT_HTTP=8161 ./tests/clickhouse-test -b build/reldeb/programs/clickhouse 04508… 04512…` (`tmp/rhj-probe-perf/u3/stateless.log`) | `5 tests passed. 0 tests skipped. 1.60 s elapsed` | GREEN |

## Independent verification — VERDICT: SHIP

Performed by a fresh subagent that did none of the work, instructed to refute it; full
findings at `tmp/rhj-probe-perf/u3/independent_verification.md`, raw artifacts under
`tmp/rhj-probe-perf/u3/indep/`.

Reproduced (refutation attempts failed):
- **Target win real**: fresh paired run ref 11789 -> cand 7818 ms = **−33.68%** (claimed
  −33.9%), all five candidate samples below all five reference samples; floors re-measured
  green (D268r2@T96 −6.11%, D67@T96 −3.77%, D67@T16 +0.43%).
- **Provenance proven both ways**: forced recompile of the tip reproduces `4b55481c…`
  byte-identically, and rebuilding the baseline source reproduces `cf662d4c…` — both sides of
  every paired comparison are proven builds of their claimed sources; the adopted T16 remedy
  patch-id-matches the preserved patch.
- **Accounting recomputes**: by-hand recomputation of A@T96 and B@T96 from the raw `RADIX*`
  timestamps matches every report row; tiling telescopes to 100.0%.
- **Gates on the tip binary**: liveness 10/10, early-termination 6/6, wide-shape hash
  verification PASS, stateless 5/5 with the server binary hash-verified; the src diff since
  baseline is the E2 change only (reverts clean).

Verifier findings accepted and recorded (none invalidating):
1. **E5 preregistration is file-order but not commit-order attested**: the E5 preregistration
   text was appended before the E5 build (per this campaign's sequence), but the commit that
   carries it also carries the E5 result, so git history alone cannot prove the ordering for
   E5 the way it does for E1–E4. E5 was refuted and reverted, so no kept change relies on it.
   Process fix for future campaigns: commit each preregistration entry before implementing.
2. The worklog's timestamp-correction entry itself misstated E4's measurement window
   (said 21:19–21:45; the driver log shows 21:42–21:52).
3. The D=67108864 r=2 T96 floor improvement re-measured at −3.77% (sub-band) vs the E2-batch's
   −5.66% — cross-suite drift; the cell is a protected floor and stays green in both
   measurements, but only the target-cell −33.9% and the re-reproduced D268r2@T96 win should
   be quoted as beyond-noise wins.
4. E2 hardening notes: the merge byte-cap undercounts lazily-replicated blocks (the row cap
   still bounds memory), and a theoretical `ColumnConst` positional-append hazard exists that
   no current leaf output can trigger — both recorded for a follow-up hardening pass.

Both documented deviations (T16-remedy adoption; correctness SKIP-by-cap for the full-size
wide shape) were judged sound and honestly reported.
