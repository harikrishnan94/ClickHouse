# `RadixHashJoin` wave-lifecycle deadlock — repair report

No irreversible action was taken or required authorization. One environment
deviation from the mission text (below) is flagged for awareness.

## Deviation: `build/x86_reldeb` is a cross-compile

This host is aarch64; `build/x86_reldeb` produces x86-64 binaries that fail
locally with `Exec format error`. Every runnable gate therefore used
`build/asan` and `build/reldeb` (aarch64, the same tree Gate 1 used). The
x86 build was still performed and its log audited (the source compiles for
x86); its unrunnable copy is kept at
`tmp/radix-wave-deadlock/bin/perf-baseline-x86-cross/`. The runnable
performance baseline is the frozen Gate-1 binary (SHA-256
`f97d41279797aed3623bcd9d703286019e9cd3d1c7257a7a6213a089790a26b3`), which
is the exact binary that exhibited the deadlock.

## The defect and the fix

`RadixHashJoin::joinBlock` acquired `State::wave_mutex` on a query-executor
thread and moved the held `std::unique_lock` into the returned
`WaveJoinResult` for its whole lifetime. Only that result's `next` drained
the wave's bounded output queue. When every executor worker piled into
`joinBlock` behind the mutex, no thread remained to drain the active wave:
the dedicated `RadixJoin` pool workers parked forever on the full queue
(Gate 1: two dumps 30 s apart, 364 threads, all 10 `QueryPipelineEx`
workers blocked at the wave mutex, all 94 `RadixJoin` workers blocked in
`ConcurrentBoundedQueue<Block>::emplaceImpl`, zero drain frames). The
retained lock could also be destroyed on a different executor thread than
the locker (work stealing) — invalid for `std::mutex`.

The fix (all in `src/Interpreters/RadixHashJoin/RadixHashJoin.cpp`) makes
the wave a shared object drained cooperatively by every probe lane:

- `State` holds a `WaveCoordinator` (mutex + `shared_ptr<ActiveWave>`).
  `ActiveWave` owns the scattered partitions, the bounded queue, per-wave
  worker accounting (mutex/cv, joined per wave rather than pool-wide), the
  stored worker exception, and a teardown-winner flag; workers hold the
  owning `shared_ptr`.
- `WaveJoinResult` is a stateless view. Its `next` starts a wave from the
  shared window when none is active (swap happens only at wave start, so
  pending lanes hold no data — a lane holding a result also pulls no
  further input, which bounds in-flight memory at roughly one accumulating
  plus one probing window), then pops from the active wave's queue whoever
  started it; a finished queue elects one lane to join the workers, clear
  the coordinator, rethrow a stored wave exception, and chain into the next
  wave. Cross-lane emission of INNER-join rows matches what
  `RadixDelayedBlocks` already does across delayed-worker transforms.
- The liveness invariant, stated in the file: no executor lane ever waits
  on progress that only another executor lane can make. Queue pops wait on
  dedicated-pool producers; the coordinator mutex is held only across the
  pool-driven scatter.
- Hardening in the same file: `parallelRun` joins already-scheduled jobs
  before rethrowing a mid-loop scheduling failure (pre-existing
  use-after-free, reachable via the stress-CI thread fault injector);
  `~RadixHashJoin` tears down an abandoned active wave (cancellation path)
  and logs its stored exception; the destructor's pool wait cannot throw;
  `getDelayedBlocks` asserts no wave is active; `State::pool` documents the
  no-escaping-jobs invariant.

Two earlier designs were implemented, gated, refuted, and recorded in
WORKLOG.md: pending results that carried their own window (OOM: all 96
lanes could hold a full probe budget), and an atomic admission token with
empty-non-final retries (busy-spinning lanes starved the pool workers:
`radix_join` 30-55x slower). The shipped design removes both failure modes
and measurably improves the drain (single consumer became many).

## Per-unit verdicts

- Unit 1 (repro + oracle): DONE. Gate 1 accepted (prior session, exact
  current-code dumps); the deterministic public-API gtest fails on pre-fix
  code only via its 10-second liveness expectation and exits cleanly.
- Unit 2 (design + repair): DONE at iteration 3, after two honest
  refutations; design validated by four independent read-only reviewers
  (executor contract, delayed-blocks ordering, primitives,
  exception/cancellation) and the iteration-3 diff adversarially reviewed
  by two more (both SOUND; one must-address fixed, notes taken or
  risk-accepted below).
- Unit 3 (verification): Gates B-G all green on the final binary; one
  localized non-blocking performance finding at T16 surfaced below.

## Evidence matrix

| Criterion | Runnable command | Raw result/artifact | Verdict |
| --- | --- | --- | --- |
| Gate 1: pre-fix hang is real, mechanism confirmed | `tmp/radix-wave-deadlock/run_large_gate1.sh tmp/radix-wave-deadlock/bin/baseline/clickhouse 268435456 4` | Assertions PASS (1073741824/268435456/1073741824), then ≤1 CPU-s/30 s for 120 s; dumps `tmp/radix-wave-deadlock/gate1_d268435456_r4_stacks_{1,2}.txt` (364 threads; 10 executor lanes at the wave mutex; 94 workers in queue push; 0 drain frames) | PASS (accepted, independently audited) |
| Gate A: new test genuinely red on unmodified code | `ninja -C build/asan unit_tests_dbms` + 30 s-timeout gtest run | Exit 1 in 10.06 s, only the liveness expectation failed; `tmp/radix-wave-deadlock/gateA_baseline_red.log` (SHA-256 `56b19dc7...`) | PASS |
| Gate B: fixed oracle 10/10 | `for i in 1..10: timeout 30s build/asan/src/unit_tests_dbms --gtest_filter=RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave` | 10x exit 0, each log `[  PASSED  ] 1 test` (~2 ms); `build/asan/test_radix_wave_fixed_{1..10}.log` | PASS |
| Gate C: accepted large shape completes | `tmp/radix-wave-deadlock/run_large_fixed_gate.sh build/reldeb/programs/clickhouse 268435456 4` | Exit 0 in ~90 s; Assertions PASS with exact counts; clean summary; `radix_join` wins 1.496x (11424 ms vs 17095 ms); `build/reldeb/test_radix_wave_gatec_d268435456_r4.log` | PASS |
| Gate D: adjacent stateless tests | `CLICKHOUSE_PORT_TCP=9131 CLICKHOUSE_PORT_HTTP=8161 ./tests/clickhouse-test -b build/reldeb/programs/clickhouse 04508... 04512...` | 5/5 OK; server binary verified via `/proc/<pid>/exe` = `148ee743...`; `build/reldeb/test_radix_wave_stateless.log` | PASS |
| Gate E: early termination + exception propagation | `tmp/radix-wave-deadlock/run_early_termination_gate.sh build/reldeb/programs/clickhouse` | 6/6: LIMIT row + exit 0 (x3); `MEMORY_LIMIT_EXCEEDED` reaches client, exit 241 (x3); no leftover processes; `build/reldeb/test_radix_wave_early_term.log` | PASS |
| Gate F: negative flip | `tmp/radix-wave-deadlock/run_negative_flip.sh` | Red on restored pre-fix (exit 1, liveness line, no outer timeout); fix restored byte-identically (SHA-256 `5b0a6b74...`); green (exit 0); `build/asan/test_radix_wave_flip_{red,green}.log` | PASS |
| Gate G: performance guard | `tmp/radix-wave-deadlock/run_perf_guard.sh tmp/radix-wave-deadlock/bin/perf-baseline/clickhouse build/reldeb/programs/clickhouse` | `perfguard verdict: PASS`; fixed medians -14.97%, -14.81%, -28.15% vs baseline on the three frozen T96 shapes (5 paired samples each; band max(5%, 1 baseline stdev)); `build/reldeb/test_radix_wave_perf_guard.tsv` | PASS |

## Performance

Primary guard (payload-narrow shapes, T96, 5 paired position-balanced
samples per binary, `radix_join` median ms, band = max(5%, 1 baseline
stdev); `build/reldeb/test_radix_wave_perf_guard.tsv`):

| D | ratio | baseline median (stdev) | fixed median | delta | verdict |
| --- | --- | --- | --- | --- | --- |
| 67108864 | 2 | 481 (3.4) | 409 | -14.97% | PASS |
| 268435456 | 2 | 1917 (23.9) | 1633 | -14.81% | PASS |
| 268435456 | 4 | 3957 (9.8) | 2843 | -28.15% | PASS |

The fix is significantly faster at T96 on every frozen shape: the
cooperative drain replaced the pre-fix single-consumer queue bottleneck.
The Gate-C wide shape (bp=pp=7), which pre-fix never completed, now
completes with `radix_join` winning 1.496x over `parallel_hash`.

Thread sweep on D=67108864 ratio=2 (secondary, per the standing
thread-sweep requirement): T32 -5.7%, T64 -10.3%, T96 -14.2% (fixed
faster); T1 +0.37% (settled as noise with 5 paired samples after a
single-invocation flag); **T16 +11.21% — a real, localized regression**
(baseline median 892 ms, stdev 6.9; fixed median 992 ms; 5 paired samples;
`tmp/radix-wave-deadlock/perf-guard/sweep_settle.tsv`).

**Finding and recommendation (T16):** at mid thread counts on this one
shape the shared-wave drain costs ~11% wall time, while every higher
thread count improves by double digits and T1 is neutral. Recommendation:
accept as a documented tradeoff — `radix_join`'s planner gate targets
large builds probed at high thread counts, correctness/liveness is the
MUST-HOLD property, and the binding T96 guard improved on all shapes. A
follow-up lead for iteration: reduce per-quantum coordinator/queue
round-trips at low consumer counts (e.g. batched pops), measured against
this exact cell.

## Cancellation and exception status

- Mid-wave `LIMIT` early termination: completes, three consecutive runs
  (Gate E). The old LIMIT-hang lead from `tmp/u2-reviewA` does not
  reproduce on the fixed binary.
- Mid-wave exception (`MEMORY_LIMIT_EXCEEDED`): reaches the client with
  nonzero exit, three consecutive runs (Gate E); a worker exception is
  stored per wave and rethrown by the teardown-electing lane after the
  workers are joined.
- Query cancellation with a live wave: transform results are inert to
  destroy in any order; `~RadixHashJoin` unparks and joins the abandoned
  wave's workers and logs its stored exception.

## Risk-accepted leads (reported, not fixed — out of scope or accepted)

- `JoiningTransform::prepare`'s early-finish path (downstream port closed)
  can strand an undrained result until pipeline teardown. Proven
  unreachable as a deadlock under the current `DelayedPortsProcessor` +
  `ResizeProcessor` wiring (all-or-none port closure), and the delayed
  flush asserts no active wave; reviewers suggested a defensive
  `join_result.reset()` in `JoiningTransform::prepare` — that file is
  outside this task's scope.
- `startWave`'s partial-scheduling failure rethrows the scheduling error,
  which can mask a concurrently stored worker exception (both fail the
  query; only the reported root cause differs).
- After any exception escapes a pool job (scatter/build/teardown paths),
  the dedicated pool shuts down permanently for that query; subsequent
  waves fail fast with `CANNOT_SCHEDULE_TASK` and the error propagates
  (verified degradation, not a hang).
- Cross-lane output emission skews per-stream statistics (a lane may emit
  another lane's joined rows); totals are unaffected. `RadixDelayedBlocks`
  set this precedent.
- The `getDelayedBlocks` assertion's proof rests on three consumer-side
  facts (documented in WORKLOG); a future wiring change would trip the
  debug assert rather than hang silently in release only if debug builds
  exercise the path.

## Independent verification

Verdict: **SHIP** (full independence — the verifier was a fresh context that
did not implement the fix and was given only the mission, the diff,
WORKLOG.md, REPORT.md, and the gate artifacts).

- Gates B, C, D, E, F re-run from a clean state: all green (Gate B 10/10
  with per-log confirmation of exactly one passing test; Gate C exit 0 with
  exact counts and `radix_join` winning 1.465x on the re-run; Gate D 5/5
  with the server binary identity re-verified; Gate E 6/6; Gate F
  controlled red then green with byte-identical restoration re-hashed).
- Gate G statistics recomputed from the raw TSVs: every claimed median,
  stdev, and delta reproduces, including the T16 finding.
- Gate-1 dumps re-counted (364 threads; 10 executor lanes at the pre-fix
  wave mutex; 94 workers in queue push; zero drain frames).
- Code inspection found no lifecycle gap; the worker accounting, teardown
  election, destruction order, and liveness invariant were each verified
  against the primitives.
- Additional leads named by the verifier (non-blocking, recorded here):
  the T16 fixed-side samples are bimodal (897 vs 985-1012 ms), supporting
  the batched-pop follow-up; the `getDelayedBlocks` tripwire is debug-only
  (release builds would hang on a future wiring change — the out-of-scope
  `JoiningTransform` hardening remains the right follow-up); explicit
  `KILL QUERY` mid-wave is exercised only indirectly (LIMIT/exception
  paths); cancellation response while parked in a queue pop is delayed by
  up to one wave's largest partition — a latency property, strictly better
  than the pre-fix behavior.

## Deliverables

- Fix: `src/Interpreters/RadixHashJoin/RadixHashJoin.cpp` (single file).
- Regression oracle: `src/Interpreters/tests/gtest_radix_hash_join.cpp`.
- WORKLOG.md: full preregistrations, refuted iterations, raw gate results.
- Gate helpers and artifacts under `tmp/radix-wave-deadlock/`.
