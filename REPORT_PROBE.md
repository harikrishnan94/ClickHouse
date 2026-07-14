# REPORT_PROBE — `RadixHashJoin` probe-time accounting and optimization

Branch `radix-join-probe-perf`, baseline `b1fa64c7286`. Full per-iteration record:
`WORKLOG_PROBE.md`. Raw artifacts: `tmp/rhj-probe-perf/`.

## Verdicts (updated as units complete)

- **Unit 0 (wait and orient): DONE.** WAIT MODE honored (poll log
  `tmp/rhj-probe-perf/wait_poll.log`); see deviations.
- **Unit 1 (accounting): tables below; instrumentation-overhead gate PENDING.**
- **Unit 2 (experiments): not started.**
- **Unit 3 (consolidation): not started.**

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

(preregistrations and results land here)

## Evidence matrix

(final gate table lands here in Unit 3)
