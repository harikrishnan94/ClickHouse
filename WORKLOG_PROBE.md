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
