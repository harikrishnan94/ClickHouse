# WORKLOG_PROBE — `RadixHashJoin` probe-time accounting and optimization campaign

Branch `radix-join-probe-perf`. Artifacts under `tmp/rhj-probe-perf/`.
Session: claude-e7403e25 (unattended).

## Unit 0 — wait and orient

### Iteration 0.1 (2026-07-14 ~15:37–18:35)
Goal: honor WAIT MODE while the T16 campaign ran in this checkout; orient; plan.

Wait protocol evidence (`tmp/rhj-probe-perf/wait_poll.log`, one line per 10-min poll):
- 15:37 first poll: no `REPORT_T16.md` anywhere (A, B empty); T16 session claude-ffd8242a
  ACTIVE with `Status: in-progress` (C); host quiet at that instant (D). WAIT MODE entered.
- 15:39–17:39 poller v1 (`tmp/rhj-probe-perf/wait_poll.sh`): T16 ran benchmarks throughout
  (its harness invocations visible in the poll log's `busy:` lines).
- 17:39 poller v1 exited on the mission's literal check B (`REPORT_T16.md` present with a
  verdict line) + quiet x2. **Premature**: the T16 session was still active — its state file
  demanded an idle host for its independent-verification subagent's benchmarks, and its work
  was uncommitted. DEVIATION (conservative): stayed in WAIT MODE per "never barge into a live
  benchmark"; launched stricter poller v2 (`wait_poll2.sh`: exit only on report committed |
  `Status: done` | state stale >45 min with verdict present, AND quiet x2).
- 17:41–18:01 verification-subagent floor re-runs observed; 18:01 last host activity.
- 18:31 poller v2 exited: state stale 52 min + quiet x3. agent-memory shows the T16 session
  **interrupted** (not merely idle) — it cannot return on its own. Its independent
  verification never appended a verdict; its remedy + report + worklog were left uncommitted.

Orientation performed during the wait (all read-only; notes in
`tmp/rhj-probe-perf/orientation.md`, plan in `tmp/rhj-probe-perf/instrumentation_plan.md`):
read `RadixHashJoin.cpp` in full (pre- and post-remedy), `ConcurrentBoundedQueue.h`,
`ColumnsScatter.h`, settings defaults, delayed-path pipeline wiring; three read-only Explore
subagents digested prior probe art, harness semantics, and the wave-deadlock campaign's gates.

### Baseline decision (ambiguity rule — recorded)
The mission's rule: branch from the T16 task branch tip if it landed commits with a SHIP or
FIX-THEN-RESHIP verdict, else from `radix-join-bandwidth-model` tip. The T16 task landed **no
commits and no final verdict** (session interrupted after its own evidence matrix was fully
green, before the independent-verification verdict was appended). Branching from the clean tip
while discarding the worktree remedy would violate the mission's other instruction ("do not
undo its remedy"), and the remedy modifies exactly the machinery this campaign measures.

Decision: branch `radix-join-probe-perf` from `radix-join-bandwidth-model` tip `b8557c1f56b`
(per the rule), then **adopt the uncommitted T16 remedy verbatim as the first, attributed
commit** `b1fa64c7286` (includes their `REPORT_T16.md`/`WORKLOG_T16.md`, unmodified, with the
verification section still reading "pending"). Provenance verified before adopting:
- Worktree diff was `RadixHashJoin.cpp` +74/−6 only, matching the remedy described in
  `REPORT_T16.md`; preserved at `tmp/rhj-probe-perf/adopted/t16_remedy_adopted.patch`
  (sha256 `95828ded7dbf73d7c231baeec3332df65b9ab2ddb1f8e0a9359200b5077ed148`).
- `build/reldeb/programs/clickhouse` hashed to
  `cf662d4c9619c7c26b1a713b57353a9024749bc9ad1d6b70f767d3bbe926bb5a` — exactly the candidate
  binary their evidence matrix gated, i.e. that binary was built from this exact source.
FLAG (goes at top of REPORT_PROBE.md): the adopted remedy carries green gates from its own
campaign but a pending independent verification; this campaign's own Unit-3 gate set
(liveness oracle, lifecycle, stateless, correctness verification, protected cells including
D=67108864 r=2 T16) re-validates it independently.

Baseline: **`b1fa64c7286`** (= `b8557c1f56b` + adopted remedy). Reference binaries for all
paired protocols are fresh in-session builds of this tip, hashed at measurement time.

### Plans fixed in Unit 0
- Accounting model v2 (post-remedy semantics): per-wave wall segments — refill gap, scatter
  (serial, under coordinator mutex), drain (overlaps scan via surplus lanes), teardown — from
  per-wave stderr records; within-drain worker split (leaf match vs queue-push wait) and
  lane-side counters (pop wait, in-next time, empty quanta); delayed path: flush scatter wall +
  per-call leaf/pending counters; rows-per-path. Extends the T16 campaign's instrumentation
  (`tmp/t16/instr.patch`, `analyze_instr.py`) — same RADIX* stderr-line pattern.
- Input-scan floor (harness has no floor mode): per cell, median of >=3 scan-only runs of the
  probe subquery and build subquery (identical PREWHERE/projection/settings/threads,
  `FORMAT Null`), floor = build_scan + probe_scan.
- Instrumented accounting runs via a runner importing the harness module for byte-identical
  SQL; protected-cell gates use the unmodified harness + `tmp/t16/judge.py` semantics
  (band max(5%, 1 stdev), radix_join `median_ms`).
- Accounting cells: A wave-dominated D=268435456 r=2 bp=pp=1; B delayed-dominated D=16777216
  r=1 bp=pp=1 (probe ~256 MiB < 512 MiB budget floor; validate with rows-per-path counters);
  C wide D=268435456 r=4 bp=pp=7. Threads {1,16,32,64,96} each.
- Overhead check (instrumented vs clean): full paired protocol (>=5+5 position-balanced) on a
  regime-spanning subset — A@T16, A@T96, B@T96, C@T96, A@T1 — subset choice documented here;
  remaining cells rely on these bounds (flagged in the report).
- Unit-2 idea space: orchestration-level only (leaf `HashJoin` internals are out of scope;
  prior AMAC/bulk-gather wins live there and are not candidates).

## Unit 1 — probe-time accounting

### Iteration 1.1 — instrumentation (2026-07-14 18:35–18:49)
- Throwaway branch `radix-join-probe-instr` off task-branch tip `4f0ef35d365`; instrumentation
  commit `06e70fea709` (RadixHashJoin.cpp only): per-wave stderr records `RADIXSTART`/
  `RADIXWAVE2` (created/publish/scatter/finish/teardown steady-clock timestamps, rows in/out,
  worker leaf-match vs queue-push-wait split, pops, attaches, losers, window-at-end),
  `RADIXBUDGET`, `RADIXFLUSH` (delayed-flush size + scatter wall), `RADIXDELAYED` (per-stream
  drain totals incl first/last call timestamps, leaf time, pending-mutex wait), `RADIXTOTALS`
  (lane-side per-query totals: in-next time, coordinator-mutex wait, pop wait, quantum counts,
  window-append time), `RADIXTEARDOWN` (parallel leaf destruction wall). No per-row timing;
  finest granularity is per block/partition/wave. First build failed (`AcctQuantum` aggregate
  init vs explicit `Stopwatch` ctor), fixed; rebuild green with 0 warnings
  (`build/reldeb/build_instr_probe{,2}.log`).
- Binaries: instrumented sha256 `4ddd1a81d5ae…` (`tmp/rhj-probe-perf/bin/instr/clickhouse`),
  clean baseline `cf662d4c9619…` (`tmp/rhj-probe-perf/bin/baseline/clickhouse`).
- Behavior-neutrality note: one instrumentation edit initially moved the delayed-path rejoin
  `joinBlock` outside `pending_mutex`; reverted before building (baseline holds the lock).

### Iteration 1.2 — accounting matrix (18:49–19:19)
- SQL generated by `tmp/rhj-probe-perf/u1/gen_cells.py` importing the harness module
  (byte-identical `measurement_script` SQL; dataset metadata bucket_width=4194304,
  max_cardinality=524288000). Cells: A = D=268435456 r=2 bp=pp=1 (wave-heavy), B = D=16777216
  r=1 bp=pp=1 (probe 268 MB < 512 MiB budget floor -> 100% delayed path, confirmed by
  counters), C = D=268435456 r=4 bp=pp=7 (wide), each at T {1,16,32,64,96}; 1 warmup + 3 timed
  runs per cell (C_T1: 2). Floors: the probe and build subqueries scan-only with identical
  PREWHERE/projection/SETTINGS, `FORMAT Null`, same warmup+runs, on the CLEAN baseline binary.
- Runner `tmp/rhj-probe-perf/u1/run_matrix.sh`: 45/45 invocations rc=0
  (`tmp/rhj-probe-perf/u1/logs/matrix_driver.log`); raw logs `acct_<cell>.err`,
  `floor_<cell>_{probe,build}.err` under `tmp/rhj-probe-perf/u1/logs/`.
- Analyzer `tmp/rhj-probe-perf/u1/acct_report.py` (regenerate:
  `python3 tmp/rhj-probe-perf/u1/acct_report.py tmp/rhj-probe-perf/u1/logs`); output snapshot
  `tmp/rhj-probe-perf/u1/acct_report_full.txt`. One fix after first full run: the
  "in-join scan excess over floor" term (pure-scan segments exceeding the floor at T1) was
  initially subtracted; it is a genuine delta component (the scan runs slower interleaved
  with join phases) and is now added.

### Iteration 1.2 results
- Timeline tiling (probe period from `RADIXBUDGET` to `RADIXTEARDOWN`, telescoping segments):
  98.8–101.3% coverage on all 15 cells.
- Delta-over-floor attribution coverage: 91.0–104.3% on 14/15 cells. B_T96: 85.4% of a 25 ms
  delta — the 3.7 ms residual is attributed to the client `--time` 1 ms quantization and
  run-to-run swing (timed walls 77/85/102 ms); MARKED as noise-residual, not unknown work.
- Hypothesis verdicts (preregistered refuters in tmp/rhj-probe-perf/instrumentation_plan.md):
  * H1 consumer-quantum-bound drain: CONFIRMED where drain is exposed. C_T96: workers spend
    89% of drain CPU blocked on the full queue (push-wait 662 s vs match 78 s CPU-sum);
    393216 pops over 7.7 s drain with crew 12 = ~235 us per consumer quantum, of which only
    ~6.6 us inside `next` — the rest is executor/downstream turnaround.
  * H2 scatter share: CONFIRMED material. Wave scatter is a serial wall segment (lanes park on
    the coordinator mutex: 194 s CPU-sum at C_T96): A 14–24% of wall, C 19–22%. Fanout 16384
    (A) / 32768 (C) forces 2 radix passes (per-pass cap 8192) — window bytes traversed twice.
  * H3 teardown/losers: REFUTED as material — wave teardown <= 38 ms, loser quanta <= 63,
    weightless quanta <= 12 on every cell.
  * H4 wide-payload gather dominating drain: REFUTED at orchestration level — producers
    (leaf match+gather) are mostly idle/blocked; the drain limiter is consumer-side.
  * H5 output block granularity: CONFIRMED as the root cause behind H1 — exactly one output
    block per (partition, wave): A 9.4k rows/block (fanout 16384), C 2.6k rows/block
    (fanout 32768); C pays 393216 consumer quanta per query.
  * H6 delayed-path contention: REFUTED — pending-mutex wait <= 0.1 ms everywhere; delayed
    drain 13–33 ms on B, 247–327 ms on C (~2% of wall).
  * H7 window-append cost: REFUTED as material — <= 1.6 s CPU-sum across 96 lanes (C_T96),
    trivial per lane.
  * New (unhypothesized): in-join scan excess at T1 — the probe scan takes 3.8 s (A) / 23.2 s
    (C) longer inside the join than the scan-only floor at T1; at T>=16 the effect vanishes.
- Row shares wave/delayed: A ~86%/14%, C ~95.5%/4.5%, B 0%/100%.
- Instrumentation overhead check: RUNNING (paired 5+5 position-balanced, unmodified harness,
  cells A_T96/B_T96/C_T96/A_T16/A_T1, `tmp/rhj-probe-perf/u1/overhead/`, judge =
  `tmp/t16/judge.py`). Gate decision recorded when done.

## Unit 2 — preregistered experiments

### E1 preregistration — worker-side output block merging (written BEFORE implementation)
Preregistered: 2026-07-14 ~19:30, before any E1 code was written. Implementation follows only
after the Unit-1 overhead gate is green.

- Motivating Unit-1 evidence (tmp/rhj-probe-perf/u1/acct_report_full.txt): wave-drain wall is
  consumer-quantum-bound where exposed. C_T96: 393216 output blocks = one per
  (partition, wave) at ~2.6k rows each; consumers spend ~235 us per popped block (~6.6 us of
  it inside `next`), drain = 7712 ms = 66% of an 11646 ms wall, drain-beyond-scan 3998 ms;
  workers are 89% blocked on the full queue (push-wait 662 s vs match 78 s CPU-sum) — i.e.
  producers have massive idle capacity while consumers are quantum-starved. Same pattern at
  C_T32/C_T64 (drain-beyond-scan 4410/3799 ms) and C_T16 (4520 ms).
- Mechanism: in `ActiveWave::worker`, accumulate leaf output blocks into a per-worker merge
  buffer and push merged blocks of up to 65409 rows / 2 MiB (whichever first), flushing the
  remainder at worker exit. Only when `threads > 1`: at T1 the drain is producer-bound
  (A_T1: drain 14440 ms ~= match 14089 ms; the consumer pop-waits 13.6 s), so the extra
  producer-side copy would be pure cost on a protected T1 cell.
- Expected effect: C_T96 wall improves >= 10% (drain quanta drop ~6x; model predicts drain
  7.7 s -> ~1.5-4.5 s => wall -25% +- wide); C_T16/32/64 improve beyond band; A and B cells
  within band (A drain is scan-hidden at T>=64, producer-limit-adjacent at T16; B has no
  waves). Memory overhead bounded: <= (2T+1 + T) x 2 MiB ~= 0.6 GiB at T96.
- Gate invocation (paired protocol, band max(5%, 1 stdev), >= 5 position-balanced pairs,
  reference = task-branch tip rebuilt fresh in-session and hashed):
  target cell `--cardinalities 268435456 --ratios 4 --build-payload-columns 7
  --probe-payload-columns 7 --threads 96`; protected floors D=67108864 r=2 bp=pp=1
  T{1,16,32,64,96}, D=268435456 r={2,4} bp=pp=1 T96, D=268435456 r=4 bp=pp=7 T96 (the last
  doubles as the target cell); liveness oracle 10/10 + early-termination gate 6/6 before any
  keep decision.
- Refuting outcome: C_T96 median within band of reference -> the ~235 us/quantum is not
  fixed-overhead-dominated (downstream is per-row/per-byte bound) -> REVERT and record. Any
  protected floor red -> REVERT regardless of target-cell win.

### Preregistration amendment (before any implementation)
Post-preregistration static analysis of the drain diagnostics reinterprets the consumer
quantum cost: per-quantum time scales with block BYTES (~1.6-1.9 us/KB on both A and C), and
the blocks in the wave queue carry lazily-replicated LEFT payload columns. `RadixHashJoin`
hard-codes `enable_lazy_columns_indexing = true` (RadixHashJoin.cpp State default) — unlike
`HashJoin`/`parallel_hash` (default false; only the planner pass
`optimizeJoinLazyIndexing.cpp` enables it, and only under a small LIMIT/Sorting or a stacked
join). With m=1 identity replication, `HashJoinResult` (`appendRightColumns`,
`force_lazy_replication` at HashJoinResult.cpp:189-207) wraps every left column as
`ColumnReplicated`, and the full-column conversion at :319-324 is skipped — so the unwrap
(index gather) happens downstream on the FEW consumer lanes while the 96 producers idle.
Reordering (neither experiment implemented yet; the block-merge preregistration above is
retained verbatim and renumbered E2):

### E1 preregistration (revised) — disable the radix-only lazy-columns-indexing default
Preregistered before implementation, 2026-07-14 ~19:45.

- Motivating evidence: (a) Unit-1 C_T96 drain diagnostics — 235 us per consumer quantum with
  only 6.6 us inside `next`, producers 89% push-blocked, drain 66% of wall and
  drain-beyond-scan ~4 s at T>=32 (tmp/rhj-probe-perf/u1/acct_report_full.txt); (b) consumer
  quantum cost proportional to block bytes across shapes (A@T96 144 us over ~150 KB, C@T96
  235 us over ~170 KB); (c) static: radix's default-true lazy indexing wraps left columns as
  `ColumnReplicated` even for identity replication, deferring the gather to consumer lanes
  (crew max(2, T/8)) — for `parallel_hash` the same queries run with lazy indexing OFF.
- Mechanism: change `State::enable_lazy_columns_indexing` default in RadixHashJoin.cpp from
  `true` to `false`. The planner still calls `setEnableLazyColumnsIndexing(true)` for the
  genuinely-lazy cases (small LIMIT above, stacked join), identical to other join algorithms.
  Left-column materialization then happens inside leaf `joinBlock`/`next` on the pool workers
  (idle capacity) in the wave path; the delayed path materializes on the same lane as before.
- Expected effect: C_T96 wall improves >= 10% (target cell); C_T16/32/64 improve; A cells and
  B cells within band or better (A drain is scan-hidden; producers have headroom at every T
  except T1 where threads=1 has no crew asymmetry — watch the protected D=67108864 T1 floor).
- Gate invocation: as in the E2 entry above (paired protocol, target C-shape
  `--cardinalities 268435456 --ratios 4 --build-payload-columns 7 --probe-payload-columns 7
  --threads 96`; then all protected floors if the target wins; liveness + early-termination
  gates before keep).
- Refuting outcome: C_T96 within band -> consumer cost is not the lazy unwrap (or producer
  materialization offsets it) -> revert and fall back to E2 (block merge) as the lever.
