# `RadixHashJoin` wave-lifecycle deadlock worklog

## Scope and safe defaults

- Branch: `codex/radix-join-wave-deadlock`, created from `35cb5a53569`.
- Product scope is limited to the wave/probe-result lifecycle in
  `src/Interpreters/RadixHashJoin/RadixHashJoin.cpp` and directly required
  tests. Scatter kernels, pass planning, leaf tables, settings semantics,
  `parallel_hash`, and `hash` are out of scope.
- No history rewrite, force-push, merge, pull-request mutation, or benchmark
  data deletion is authorized. Temporary evidence lives under
  `tmp/radix-wave-deadlock`.
- The existing deterministic LIMIT repro is only a lead. It passes Gate 1 only
  if a fresh current-code dump shows both executor lanes blocked acquiring the
  wave admission mutex and `RadixJoin` workers blocked pushing to the bounded
  output queue. A hang with different stacks is a failed gate, not evidence for
  the proposed mechanism.

## Unit 1: confirm and minimize the reproduction

### Iteration 1: LIMIT candidate rejected before execution

Current-code observations before execution:

- `RadixHashJoin::joinBlock` moves a blocking `wave_mutex` lock into
  `WaveJoinResult`.
- `WaveJoinResult::next` is the only bounded-queue consumer.
- `JoiningTransform` retains its `join_result` between scheduler calls and
  resets it only after `next` reports the last result.
- The dedicated radix pool can therefore block on a full queue while sibling
  query-executor lanes block acquiring the lifetime-held mutex.

The preregistered LIMIT candidate was not run. A read-only audit of its preserved
historical dump refuted it as a primary Gate-1 oracle:

```text
$ rg -c 'DB::RadixHashJoin::joinBlock' tmp/u2-reviewA/baseline_hang_stacks.txt
76
$ rg -c '"QueryPipelineEx"' tmp/u2-reviewA/baseline_hang_stacks.txt
96
$ rg -c 'RadixJoin|WaveJoinResult::worker|ConcurrentBoundedQueue' \
    tmp/u2-reviewA/baseline_hang_stacks.txt
<no matches>
```

Decision: preserve `gate1.sql` as a deterministic cancellation/LIMIT gate for
Unit 3, but do not run or claim it as proof of bounded-queue exhaustion.

### Iteration 2 preregistration: first known-large point

The smallest existing completed wide-payload evidence is
D=134217728, ratio=4, bp=pp=7, T96, so it cannot be the negative gate. Start
with the first point extracted from the known reliable four-point hang:
D=268435456, ratio=2, bp=pp=7, T96.

Expected outcome: count assertions pass, then the normal (no LIMIT and no forced
exception) `radix_join` measurement stops making progress. After 120 seconds
with no log change, two dumps 30 seconds apart must both show:

1. multiple `QueryPipelineEx` threads in `RadixHashJoin::joinBlock` waiting
   on the wave mutex;
2. `RadixJoin` workers waiting in `ConcurrentBoundedQueue<Block>::push`; and
3. the pipeline executor unable to schedule the active result's queue drain.

The no-progress clock starts only after `Assertions: PASS`. The harness runs in
its own process group; after both dumps the whole group is sent `SIGTERM`,
given ten seconds to stop, then sent `SIGKILL` only if still alive and reaped.

Refutation: the point completes, the log keeps advancing, or either dump lacks
one side of the cycle. In that case the full known four-point command must be
rerun and minimized from whichever point actually hangs.

Exact commands:

```bash
cp build/reldeb/programs/clickhouse \
  tmp/radix-wave-deadlock/bin/baseline/clickhouse
sha256sum build/reldeb/programs/clickhouse \
  tmp/radix-wave-deadlock/bin/baseline/clickhouse

tmp/radix-wave-deadlock/run_large_gate1.sh \
  tmp/radix-wave-deadlock/bin/baseline/clickhouse 268435456 2
```

Result: REFUTED as a deterministic Gate-1 shape. Assertions passed inside
30 seconds. The 266-byte log was then unchanged for 90 seconds, but the harness
completed normally before the 120-second dump threshold. The final log:

```text
Assertions: PASS for D=268435456 m=1 ratio=2 hit=1 bp=7 pp=7;
probe/build/joined=536870912/268435456/536870912
radix_join     OK  median_ms=7221.000  leaf_builds=32768
parallel_hash  OK  median_ms=13089.000
Winner: radix_join (1.813x)
Summary: wins=1 losses=0 ties=0 fallback=0 invalid=0 errors=0 hash_mismatch=0
```

An independent log reader confirmed the same counts, timings, and absence of
fallbacks/errors. Verification was explicitly skipped; exact count assertions
remained enabled. No dump was taken because the candidate completed.

Plan correction: log silence alone is not progress evidence—the harness prints
algorithm results only after the subprocess exits. Remaining iterations also
track the direct ClickHouse child's CPU ticks. The dump threshold requires both
an unchanged log and no more than one CPU-second of progress per 30-second
interval for 120 consecutive seconds. A child PID change resets the clock.

### Iteration 3 preregistration: D=268435456, ratio=4

Expected outcome and stack oracle are unchanged from Iteration 2. The next
original-sweep point doubles the probe rows while retaining bp=pp=7 and T96.
It passes only if the monitored stall produces the same two-sided cycle in two
dumps. Normal completion refutes this candidate.

```bash
tmp/radix-wave-deadlock/run_large_gate1.sh \
  tmp/radix-wave-deadlock/bin/baseline/clickhouse 268435456 4
```

Result: ACCEPTED. Gate 1 is settled on this shape against the frozen pre-fix
binary (`tmp/radix-wave-deadlock/bin/baseline/clickhouse`, SHA-256
`f97d41279797aed3623bcd9d703286019e9cd3d1c7257a7a6213a089790a26b3`).

- Assertions passed with probe/build/joined row counts
  1073741824/268435456/1073741824.
- The direct ClickHouse child (PID 110847) was initially active (22698, then
  41392 CPU ticks per 30-second interval), then made only 3-4 ticks per
  interval for 120 consecutive seconds with an unchanged harness log — the
  preregistered CPU-stall criterion (unchanged log plus at most one
  CPU-second of child progress per 30-second interval for 120 seconds).
- Two gdb dumps were captured 30 seconds apart:
  `tmp/radix-wave-deadlock/gate1_d268435456_r4_stacks_1.txt` and
  `tmp/radix-wave-deadlock/gate1_d268435456_r4_stacks_2.txt`, with matching
  child records in `gate1_d268435456_r4_child_{1,2}.txt`.
- Both dumps contain 364 threads. In both: all 10 `QueryPipelineEx` workers
  are blocked in `RadixHashJoin::joinBlock` at the same wave mutex
  (line 1291), all 94 `RadixJoin` pool workers are blocked in
  `ConcurrentBoundedQueue<Block>::emplaceImpl` (line 47) pushing output for
  the same `WaveJoinResult`, and there are zero `WaveJoinResult::next` or
  queue-pop frames anywhere. The coordinator thread waits for the pipeline
  pool. This is the complete two-sided cycle from the preregistered oracle.
- After the second dump the isolated process group was terminated with
  `SIGTERM`, the harness exited 143, and no workload process remained.
- Independent audit verdict: ACCEPT (both dumps re-checked by a reviewer that
  did not run the harness; frame counts re-verified in this session:
  364 threads, 10 `joinBlock` frames, 188 `emplaceImpl` frames across the
  94 `RadixJoin` threads, 0 `next`/`popImpl` frames in each dump).

The earlier D=268435456 ratio=2 candidate remains a recorded refutation (it
completed normally). The old LIMIT dump under `tmp/u2-reviewA` has mutex
waiters but no queue-push frames; it is preserved only as a later
early-termination/cancellation gate. No further large-shape minimization is
warranted.

## Unit 1b: deterministic small oracle (Gate A)

### Preregistration

A public-API gtest, `RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave`
in `src/Interpreters/tests/gtest_radix_hash_join.cpp`, added before any product
change. Shape: `max_threads = 2` (bounded output queue capacity
`2 * threads + 1 = 5`), probe budget forced to 1 byte
(`probe_buffer_fraction = 0`, min = max = 1), two radix partitions (build side
small enough that fanout stays at the lower bound `max(2, bit_ceil(2)) = 2`),
single `UInt64` key. The two keys are selected at runtime so their scatter
routes differ (partition = `(routeWord(key) >> 31) & 1` for the 1-pass 1-bit
plan; `route_shift = 32 - bits` per `RadixHashJoin.cpp:279`). Each key is
duplicated 4 times on the build side (duplicates also prevent the
`All`-to-`RightAny` promotion), and each probe lane has 8 rows (4 per key), so
one wave emits 32 rows. With `max_joined_block_size_rows = 1` each output block
carries exactly one probe row's match set (`numLeftRowsForNextBlock`,
`HashJoinResult.cpp:333`): 8 blocks per wave against a 5-slot queue, so the
wave's producers must park.

Derived expectations (from the constructed blocks, not hard-coded guesses):
64 joined rows across both lanes (2 lanes x 8 probe rows x 4 matches), and the
exact output multiset of `(k, probe_id, rk, build_id)` tuples computed
programmatically in the test; any dropped, duplicated, or cross-wired row
changes the multiset. The originally proposed 32-row / probe-id-sum-16 numbers
did not survive re-derivation and were replaced by this construction.

Flow: lane A `joinBlock` + one non-final `next` (leaves the wave mid-flight
with a full queue and parked producers), then lane B runs `joinBlock` plus its
immediate `next` on another thread and must return within 10 seconds. On the
old code the failure is controlled: the abandon flag is set, lane A's result is
destroyed (releasing the wave), and lane B drains its result on its own thread
(on the old code that thread owns the moved wave lock; draining elsewhere would
be a cross-thread `std::mutex` unlock). On green, both results are drained
concurrently and the multiset identity is asserted.

Expected Gate A outcome: the test fails only via the 10-second liveness
expectation and the binary exits cleanly within the outer 30-second timeout.
Refutation: the test passes on unmodified code, hangs past the outer timeout,
or fails on any other assertion.

### Result: ACCEPTED (test is genuinely red, in the controlled way)

First build attempt failed (`routeWord` needed its `DB::ColumnsScatter`
namespace qualifier) — recorded as a null result; fixed and rebuilt cleanly.

```text
$ ninja -C build/asan unit_tests_dbms > build/asan/build_radix_wave_gtest_baseline.log 2>&1; echo ninja_exit=$?
ninja_exit=0
$ timeout --signal=TERM --kill-after=10s 30s build/asan/src/unit_tests_dbms \
    --gtest_filter=RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave \
    > build/asan/test_radix_wave_baseline.log 2>&1; echo exit=$?
exit=1
```

Raw log (`build/asan/test_radix_wave_baseline.log`, preserved as
`tmp/radix-wave-deadlock/gateA_baseline_red.log`, SHA-256
`56b19dc79450c57af49e292604b954c1a96e34010ceabace87b5050fab7dc74e`):

```text
[ RUN      ] RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave
src/Interpreters/tests/gtest_radix_hash_join.cpp:240: Failure
Value of: lane_b_returned
  Actual: false
Expected: true
lane B's joinBlock+next quantum did not return while lane A's wave was mid-flight
[  FAILED  ] RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave (10002 ms)
```

The only failure is the preregistered liveness expectation; the run took
10.06 seconds total (10-second deadline plus controlled cleanup), the gtest
process exited with code 1, and the outer `timeout` did not fire. This proves
the oracle exercises executor-lane blocking, not merely final output.

## Environment deviation: `build/x86_reldeb` is a cross-compile

The host is aarch64. `build/x86_reldeb/programs/clickhouse` is an x86-64
binary (`file` verified) and fails locally with `Exec format error`; the
mission's literal `x86_reldeb` gate commands cannot run here. All runnable
gates therefore use the aarch64 trees: `build/asan` for the unit-test gates
and `build/reldeb` (RelWithDebInfo, same tree Gate 1 used) for the large
shape, stateless, early-termination, flip, and performance gates. The x86
build was still performed and its log audited (proves the source compiles
for x86); its frozen copy is kept unrunnable at
`tmp/radix-wave-deadlock/bin/perf-baseline-x86-cross/`.

The runnable performance baseline is the already-frozen Gate-1 binary
(`tmp/radix-wave-deadlock/bin/baseline/clickhouse`, SHA-256
`f97d41279797aed3623bcd9d703286019e9cd3d1c7257a7a6213a089790a26b3`), copied
by ordinary copy to `tmp/radix-wave-deadlock/bin/perf-baseline/clickhouse`
(hash verified equal, link count 1). A fresh
`ninja -C build/reldeb clickhouse` on unchanged product source relinked to a
different hash (`7f8e3ce8...`) solely because the CMake re-glob for the new
gtest regenerated `tzdata`/`GitHash` objects (log:
`build/reldeb/build_radix_wave_perf_baseline.log`); using the proven
Gate-1 binary as the perf baseline avoids that ambiguity entirely.

## Unit 2: design and repair

### Design validation (read-only, before preregistration)

Four independent read-only reviewers validated the admission-token lead
against (1) the executor/`JoiningTransform` contract, (2) delayed-blocks
ordering, (3) `ConcurrentBoundedQueue`/`ThreadPool` primitive semantics, and
(4) exception/cancellation lifecycles. All four verdicts: SOUND, with
must-address refinements folded into the preregistration below (full
findings with file:line evidence preserved in the session workflow journal).
Key verified facts:

- `finish`/`clearAndFinish` wake parked pushers and make their push return
  false; `pop` drains a finished non-empty queue before reporting end; the
  final drain must therefore use worker-side `finish` (not `clearAndFinish`,
  which drops rows) — as the current code already does.
- `ThreadPool::wait` is callable from any thread, waits for all pool jobs,
  and rethrows only exceptions that escaped a job body. Wave workers swallow
  theirs into `wave_exception`; `parallelRun` jobs do not, and one escaped
  job exception permanently shuts the pool (`shutdown_on_exception` default).
- `scheduleOrThrow` on this pool never blocks (queue_size 0) but can throw
  (thread-creation failure and the stress-CI `CannotAllocateThreadFaultInjector`),
  so partial-scheduling failure is a real, CI-exercised path. Today it
  escapes the `WaveJoinResult` constructor, skipping the destructor while
  scheduled workers still reference the dying object (a latent
  use-after-free) and leaving `active_workers` unable to reach zero.
- The executor serializes the quanta of one transform (even across stolen
  threads), destruction happens after executor threads are joined, an empty
  non-final result keeps the transform Ready with no lost wakeup and never
  pushes an empty chunk downstream, and `JoinProbeTableRowCount` counts only
  `joinBlock` creation. Non-admitted lanes busy-spin at full CPU for the
  duration of the active wave: accepted for a liveness fix, and measured by
  the performance gate including a thread sweep.
- The delayed-blocks flush starts only after every `JoiningTransform` is
  finished, which cannot happen while any lane holds an undrained result, so
  `getDelayedBlocks` runs with the admission free. The one exotic exception
  (a lane force-finished by a closed downstream port while holding an active
  result) is proven unreachable as a deadlock with the current
  `ResizeProcessor` wiring; a defensive reset in `JoiningTransform::prepare`
  was suggested but is outside this task's scope — recorded as a lead for
  REPORT.md.

### Preregistration: atomic wave admission with a pending/active result

Mechanism (all in `src/Interpreters/RadixHashJoin/RadixHashJoin.cpp`):

1. `State::wave_mutex` is replaced by `std::atomic<bool> wave_active{false}`.
   Admission is `!wave_active.exchange(true, std::memory_order_acq_rel)`;
   release is `wave_active.store(false, std::memory_order_release)` and may
   happen only after the output queue is finished-and-empty and
   `pool.wait()` has returned. Cross-wave happens-before rides this
   acquire/release pair (worker-side ordering additionally rides the pool
   and queue mutexes).
2. `joinBlock` swaps the ready window into a `WaveJoinResult` and returns
   without blocking, scattering, or scheduling: the result starts *pending*
   and owns only the window blocks. The pre-build/schema-join gates are
   untouched.
3. `WaveJoinResult::next` on a pending result attempts admission. If busy it
   returns an empty non-final `JoinResultBlock` (`next_block` stays null)
   promptly, so the executor retains scheduling control. If admitted, the
   result becomes *active*: the probe `Stopwatch` restarts (a long pending
   period must not inflate `RadixHashJoinProbeMicroseconds`), then the
   activation sequence — scatter the window on the pool, store
   `active_workers`, schedule the workers — runs under a try/catch. On any
   activation throw (scatter or a partial `scheduleOrThrow` failure): finish
   and clear the queue (already-running workers observe push()==false and
   exit), `pool.wait()`, release admission, mark the result terminal, and
   rethrow. The cleanup must not depend on the `active_workers`/`finish`
   chain. Activating on a fully constructed object also removes the current
   constructor-throw use-after-free.
4. The admitted drain loop is unchanged: pop returns output blocks as
   non-final results; when the queue reports finished-and-empty, `pool.wait()`,
   release admission, mark terminal, then rethrow `wave_exception` if a
   worker stored one (release-before-rethrow, so a poisoned pool cannot trap
   the remaining pending lanes — they get admitted, fail fast, and
   propagate), otherwise account the probe time and return `is_last`.
5. Destruction: a pending result is inert (it must not touch the pool or
   the token — a pending destructor `pool.wait()` could deadlock teardown
   behind another lane's parked workers). An active result clears/finishes
   the queue, joins the workers with the wait wrapped against rethrow (this
   destructor runs on every mid-wave cancellation), and releases admission.
   A terminal result is inert. The release-once flag is a plain member:
   quanta of one transform are serialized by the executor and destruction
   happens after executor threads are joined.
6. `worker()` moves the `ThreadGroupSwitcher` construction inside its
   try/catch so no wave job can ever escape an exception into the pool.
7. Same-file hardening riders: `chassert` in `getDelayedBlocks` that the
   admission is free (the wiring guarantees it; the assert catches future
   wiring changes in debug builds), and the pre-existing
   `~RadixHashJoin` catch-path `pool->wait()` gets wrapped so a teardown
   exception cannot escape a destructor. The `State` and `WaveJoinResult`
   comment blocks are rewritten for the new semantics.

Gate commands that will prove or refute this:

```bash
ninja -C build/asan unit_tests_dbms > build/asan/build_radix_wave_gtest_fixed.log 2>&1
for i in 1 2 3 4 5 6 7 8 9 10; do timeout --signal=TERM --kill-after=10s 30s \
  build/asan/src/unit_tests_dbms \
  --gtest_filter=RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave \
  > build/asan/test_radix_wave_fixed_$i.log 2>&1 || exit 1; done
```

Expected: 10/10 pass with the exact row-count and multiset assertions green,
no run hitting the outer timeout. Refutation: any failed or timed-out run,
any multiset mismatch (dropped/duplicated rows), or the large-shape Gate C
run stalling again. A refuted design returns to this preregistration step;
it must not be patched into shape by weakening a gate.

### Implementation and Gate B result: ACCEPTED (10/10)

The preregistered design was implemented in
`src/Interpreters/RadixHashJoin/RadixHashJoin.cpp` (148 insertions, 34
deletions; single file). The humanize pass ran over the diff; the only
remaining `std::unique_lock` in the file is `RadixDelayedBlocks`'
pre-existing `eof_mutex` use, and no `wave_mutex`/`wave_lock` references
remain. Both builds succeeded with `RadixHashJoin.cpp` recompiled and no
warnings (`build/asan/build_radix_wave_gtest_fixed.log`,
`build/reldeb/build_radix_wave_clickhouse_fixed.log`, exit 0 each,
independently read). Fixed reldeb binary SHA-256 recorded below.

```text
$ for i in 1 2 3 4 5 6 7 8 9 10; do timeout --signal=TERM --kill-after=10s 30s \
    build/asan/src/unit_tests_dbms \
    --gtest_filter=RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave \
    > build/asan/test_radix_wave_fixed_$i.log 2>&1 || break; done
run 1..10: exit=0 each
```

All ten logs contain `[  PASSED  ] 1 test` and
`[       OK ] RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave (2 ms)`;
no `Failure` lines; no outer timeout fired. The 2 ms fixed-code runtime
against the 10-second pre-fix red confirms lane B's quantum now returns
immediately, and the multiset identity (64 exact tuples across both
concurrently drained lanes) held in every run.

### Gate C, iteration 1: REFUTED — the v1 pending design blows the memory envelope

```text
$ tmp/radix-wave-deadlock/run_large_fixed_gate.sh build/reldeb/programs/clickhouse 268435456 4
gatec assertions passed; monitoring for completion or stall
gatec FAIL: harness exit=1
```

Raw log `build/reldeb/test_radix_wave_gatec_d268435456_r4.log`: assertions
passed, `parallel_hash` completed (median 17037 ms), but the `radix_join`
measurement failed 5.3 s in with
`Code: 241 ... Query memory limit exceeded: would use 93.15 GiB ... maximum: 93.13 GiB`
(`MemoryTrackerPeakUsage` 99.5 GB). No hang — the deadlock is gone — but the
fix as implemented replaced it with an OOM on the same shape.

Diagnosis (v1 defect, not measurement noise): pre-fix, a lane that swapped a
full probe window immediately parked on `wave_mutex`, so at most
executor-pool-size (~10) swapped windows could exist. The v1 pending design
returns each lane's result without blocking, so all 96 probe streams
eventually swap and hold a full window while pending. With
`radix_join_probe_buffer_fraction = 0.15`, min 512 MB, max uncapped
(`Settings.cpp:7931-7939`) the budget for this shape is ~3.9 GB
(post-build ~26 GB), so in-flight window memory went from ~10 x budget
(~47 GB, fits) to up to 96 x budget (>90 GB before the probe finishes
reading — exceeds the harness's 100 GB `max_memory_usage`).

### Preregistration, iteration 2: weightless pending results

Revision: a pending `WaveJoinResult` carries no window at all.

1. `joinBlock` appends to the shared window as today; when the budget is
   crossed it returns a pending result holding only references (shared
   probe state, admission token, and the window trio
   `window_mutex`/`window_blocks`/`window_bytes`). No swap happens in
   `joinBlock`.
2. On admission inside `next`, the result swaps the *current* shared window
   under `window_mutex` — whatever has accumulated, possibly more than one
   budget (bounded: every lane stops pulling input once it holds a result,
   so overshoot is at most one in-flight chunk per lane) and possibly empty
   (another admitted wave already took the rows). An empty swap releases
   admission immediately and returns `is_last` — the lane's rows were or
   will be probed by another wave or the delayed flush.
3. Everything downstream of the swap (activation try/catch, drain,
   exception, destructor phases) is unchanged from iteration 1; the
   destructor of a pending result stays inert.

Memory envelope: one active window being probed plus one accumulating shared
window, ~2-3 budgets total — strictly below the pre-fix ~10-11 budgets.
Exactly-once probing is preserved: rows leave the shared window only via one
admitted wave's swap or the final delayed flush.

Gate commands unchanged (Gate B 10x, then Gate C on the same shape).
Expected: Gate B 10/10 and Gate C completes with green assertions and clean
summary within the 100 GB harness limit. Refutation: any Gate B failure, a
Gate C stall, OOM, or dirty summary.

### Iteration 2 results: Gate B 10/10; Gate C completes but is REFUTED on performance

Gate B (iteration 2): 10/10 pass, every log `[  PASSED  ] 1 test` at ~2 ms,
no outer timeout (`build/asan/test_radix_wave_fixed_{1..10}.log`).

Gate C (iteration 2): the monitor reported PASS by the letter of the gate —
harness exit 0, `Assertions: PASS` with exact counts
1073741824/268435456/1073741824, `Summary: ... fallback=0 invalid=0 errors=0
hash_mismatch=0`, ~17.5 minutes total, no stall
(`build/reldeb/test_radix_wave_gatec_d268435456_r4.log`). The deadlock is
gone and no OOM occurred. But the same log shows `radix_join` at
506553 ms vs `parallel_hash` 16822 ms (`Winner: parallel_hash (30.113x)`),
with the child pegged at ~95 cores the whole time while the join's own
accounted work is only a few hundred core-seconds. A control run on the
smallest frozen narrow shape (D=67108864 r=2 bp=pp=1 T96, fixed binary,
`tmp/radix-wave-deadlock/perf-explore/fixed_iter2_d67108864_r2.log`)
measured `radix_join` 26312 ms vs the pre-fix 478 ms — a 55x regression with
`parallel_hash` unaffected (834 vs 674 ms). Diagnosis: every non-admitted
lane burns a full executor quantum per empty non-final result, and the
executor re-runs it immediately; with ~an executor thread per stream, the
spinning lanes starve the dedicated radix pool workers of CPU for the whole
wave. Iteration 2 satisfies liveness but violates the mission goal that the
fix must not destroy `radix_join`'s wins. REFUTED; on to iteration 3.

### Preregistration, iteration 3: shared active wave, cooperative drain

The busy wait is removed by making waiting lanes useful instead of polite:
the admitted wave becomes shared state, and every lane's `next` drains it.

1. `State` gains an anonymous-namespace `WaveCoordinator` member: a small
   mutex plus `std::shared_ptr<ActiveWave> active_wave`. The iteration-2
   atomic token disappears.
2. `ActiveWave` owns what `WaveJoinResult` used to: scattered parts, drain
   order, the bounded output queue, worker bookkeeping (a mutex/cv-guarded
   worker counter instead of pool-wide waits), the wave `Stopwatch`, the
   stored worker exception, and a `torn_down` flag. Workers capture the
   `shared_ptr`, so the wave outlives any consumer teardown order.
3. `WaveJoinResult` becomes a stateless view (references to the coordinator,
   the shared window fields, and the probe budget). Its destructor is
   trivially inert — nothing to leak, nothing to release.
4. `next` loops: under the coordinator mutex, if no wave is active, swap the
   shared window (only when it has reached the probe budget — below-budget
   rows keep waiting exactly as pre-fix) and start a new wave (scatter and
   schedule while holding the coordinator mutex: lanes that park on it wait
   only on pool-driven scatter progress, never on another executor task);
   if there is nothing to start, return `is_last`. Then pop from the active
   wave's queue: a block is returned as a non-final result — *any* lane
   emits any wave's output, exactly like `RadixDelayedBlocks` already does
   across delayed-worker transforms. A finished-and-empty queue elects a
   teardown winner via `torn_down`: the winner joins the wave's workers (cv
   wait on the per-wave counter), clears the coordinator pointer, rethrows
   the wave exception if any, accounts the probe time, and loops (either
   starting the next wave or reporting `is_last`); losers return an empty
   non-final result and re-check on their next quantum (bounded by the
   winner's brief join).
5. The blocking queue pop is safe where the old wave-mutex park was not: the
   pop's producers are dedicated pool workers that need no executor thread,
   whereas the old mutex's release required an executor-driven drain. The
   liveness invariant, stated once for the file: no executor lane ever waits
   on progress that only another executor lane can make.
6. Partial `scheduleOrThrow` failure: the worker counter is preset and
   corrected downward for never-scheduled workers, the queue is finished and
   cleared, the scheduled workers are cv-joined, and the error propagates
   without publishing the wave.
7. `~RadixHashJoin` tears down a leftover active wave (cancellation path:
   clear-and-finish the queue, cv-join the workers) before the partition
   teardown; `getDelayedBlocks` asserts no wave is active under the
   coordinator mutex.
8. The gtest changes one assertion: per-lane drained counts are no longer
   32/32 (lanes share every wave's output arbitrarily), so it asserts the
   total (64) plus the unchanged exact multiset identity, which still
   detects any dropped, duplicated, or corrupted row. The red-path shape is
   untouched.

Expected: Gate A stays red on pre-fix code (unchanged mechanism), Gate B
10/10, Gate C completes with `radix_join` within the same order of
magnitude as its pre-fix wins (the 1.8x-win shape must not become a loss
purely from wave-lifecycle overhead), and Gate G confirms parity within the
band on the frozen narrow shapes. Refutation: any hang, row-count or
multiset failure, OOM, or another order-of-magnitude `radix_join` collapse.

### Iteration 3 results: Gate B 10/10, Gate C PASS with radix_join winning

Implemented as preregistered (`+242/-67` on the single product file; diff
snapshot `tmp/radix-wave-deadlock/iter3_product.diff`). Both builds clean;
fixed reldeb binary SHA-256
`db8f5120f15870307a6b3857f76ff8a323200d0bdef519b560d9a9034115825b`.

- Gate B: 10/10 pass, every log `[  PASSED  ] 1 test`, no outer timeout
  (`build/asan/test_radix_wave_fixed_{1..10}.log`). The multiset identity
  now tolerates the by-design arbitrary per-lane split (total 64 rows plus
  the exact 64-tuple multiset).
- Narrow-shape control (the probe that refuted iteration 2): D=67108864,
  ratio=2, bp=pp=1, T96 — `radix_join` 424 ms vs pre-fix 478 ms, winner
  `radix_join` 1.745x (pre-fix 1.410x); the cooperative multi-consumer
  drain removed the old single-consumer queue bottleneck
  (`tmp/radix-wave-deadlock/perf-explore/fixed_iter3_d67108864_r2.log`).
- Gate C: PASS. The accepted Gate-1 hang shape (D=268435456, ratio=4,
  bp=pp=7, T96) completed in ~90 s wall with harness exit 0,
  `Assertions: PASS`, counts 1073741824/268435456/1073741824, clean summary
  — and `radix_join` WON it: 11424 ms vs `parallel_hash` 17095 ms
  (`Winner: radix_join (1.496x)`;
  `build/reldeb/test_radix_wave_gatec_d268435456_r4.log`). Pre-fix this
  shape never completed (permanent deadlock), and under iteration 2 it
  completed 30x slower than `parallel_hash`.

A two-reviewer adversarial pass over the iteration-3 diff (concurrency
protocol; consumer semantics and row accounting) runs before the remaining
gates; its findings gate Unit 3.

### Adversarial review of the iteration-3 diff: 2x SOUND, one must-address

Two independent reviewers (concurrency protocol; consumer semantics and row
accounting) each returned SOUND on `tmp/radix-wave-deadlock/iter3_product.diff`
(full JSON findings preserved in the session workflow journal). Findings and
dispositions:

- MUST-ADDRESS (pre-existing, but exercised by every wave the new `startWave`
  runs): `parallelRun` had no cleanup when `scheduleOrThrow` failed mid-loop —
  it unwound without `pool.wait()`, leaving already-scheduled jobs running
  while they referenced the caller's `std::function` temporary and stack
  locals (scatter histograms/parts; the destructor's teardown lambda): a
  use-after-free reachable through the stress-CI thread fault injector or a
  `MEMORY_LIMIT_EXCEEDED` on the job allocation. Fixed in the same change:
  the schedule loop is wrapped so any failure joins the scheduled jobs
  before rethrowing. (`startWave`'s own scheduling loop was verified correct;
  its workers capture the wave `shared_ptr` by value.)
- Taken as cheap hardening: the abandoned-wave teardown in `~RadixHashJoin`
  now logs a stored wave exception instead of dropping it silently
  (diagnosability; no client-visible error is lost — that path only runs
  when the query already ended), and `State::pool` documents the liveness
  invariant that no exception-escaping job may share the pool with queued
  wave workers (an escaped exception shuts the pool and destroys queued
  jobs unrun, which would strand the wave's consumers).
- Accepted as-is (recorded for REPORT.md): `startWave`'s partial-failure
  path rethrows the scheduling error, which can mask a live worker's stored
  probe exception (both fail the query); the post-shutdown pool degradation
  is safe and the destructor's nested catch is load-bearing; the
  `getDelayedBlocks` chassert's proof rests on three consumer-side facts
  (JoiningTransform's has_input-before-isFinished order, the delayed-root
  short-circuit on closed outputs, the DelayedPortsProcessor+Resize wiring);
  cross-lane emission skews only per-stream statistics; the loser-lane retry
  was verified bounded (the queue finishes only after the last worker's
  decrement, so the winner's join returns immediately).

Because the review fixes touch product code, Gates B/C/D/E are re-run on the
final binary; the Gate G run that had started on the previous binary was
killed (partial results discarded) and runs once, complete, on the final
binary.

### Gate E (iteration 3): PASS 6/6

```text
$ tmp/radix-wave-deadlock/run_early_termination_gate.sh build/reldeb/programs/clickhouse
gatee early_stop run 1..3: PASS (exit 0, LIMIT row present, 5s each)
gatee exc run 1..3: PASS (exit=241, exception reached client, 5s each)
gatee PASS: all six runs behaved
```

`tmp/u2-reviewA/early_stop.sql` (LIMIT mid-wave) completed with the correct
`early_stop 5` row three times; `exc.sql` (mid-wave `MEMORY_LIMIT_EXCEEDED`)
propagated the exception to the client with nonzero exit three times; no
process remained (raw outputs `tmp/radix-wave-deadlock/gatee_*.out`, driver
log `build/reldeb/test_radix_wave_early_term.log`). The old LIMIT-hang lead
from `tmp/u2-reviewA` does not reproduce on the fixed binary.

### Gate D (iteration 3): PASS 5/5

A scratch server on ports 9131/8161 (this host runs another server on the
default ports and Keeper raft 9234) served the tests; its binary identity
was verified via `/proc/<pid>/exe` SHA-256 = `db8f5120...825b` (the version
banner's githash is configure-time-stale and was ignored).

```text
$ CLICKHOUSE_PORT_TCP=9131 CLICKHOUSE_PORT_HTTP=8161 ./tests/clickhouse-test \
    -b build/reldeb/programs/clickhouse 04508_radix_join_gate_and_fallback \
    04509_radix_join_distinct_estimate 04510_radix_join_payload_gate \
    04511_radix_join_multi_pass 04512_radix_join_wide_fixed_types
5 tests passed. 0 tests skipped. 1.62 s elapsed
```

All five green (`build/reldeb/test_radix_wave_stateless.log`); the server
was stopped cleanly afterwards.

### Final binary re-gates after the review fixes

The review fixes (the `parallelRun` cleanup, the abandoned-wave exception
log, the pool-invariant comment) changed product code, so every cheap gate
was re-run on the final binaries (product file SHA-256
`5b0a6b7401ca1115603970ffbd3ac29a9823adc1ffc6310e583f44dfd2ef311b`, reldeb
binary `148ee743a09f4031c2e3cfbfe50448fd7d522e3b53f6792032c68bd687497a3a`,
builds audited clean — `build/{asan,reldeb}/build_radix_wave_*_final.log`).
The Gate G run that had started on the previous binary was killed; its
partial results were discarded unread.

- Gate B: 10/10, every log `[  PASSED  ] 1 test` (~2 ms each), no timeout.
- Gate C: PASS again (~90 s, exit 0, exact counts, clean summary,
  `wins=1` — `radix_join` still wins the formerly hanging shape).
- Gate D: 5/5 against a scratch server whose `/proc/<pid>/exe` hashed to the
  final binary.
- Gate E: 6/6 (LIMIT row + exit 0 three times; `MEMORY_LIMIT_EXCEEDED` to
  the client with exit 241 three times; nothing left running).

### Gate F: PASS (the oracle tests the fix, not luck)

```text
$ tmp/radix-wave-deadlock/run_negative_flip.sh
gatef fixed snapshot sha256=5b0a6b7401ca1115603970ffbd3ac29a9823adc1ffc6310e583f44dfd2ef311b
gatef pre-fix product restored from 35cb5a535699b9d5d6a4b19a6ffb151e13f64197
gatef red phase: exit=1 controlled_red=1
gatef fixed product restored byte-identically
gatef green phase: exit=0 green=1
gatef PASS: controlled red on pre-fix, green on restored fix, worktree byte-identical
```

The red phase failed only via the preregistered liveness expectation
(`lane B's joinBlock+next quantum did not return...`, 10002 ms, exit 1 — the
outer timeout did not fire); the restored fix passed in 2 ms. The product
file's post-flip SHA-256 equals the pre-flip snapshot. Logs:
`build/asan/test_radix_wave_flip_{red,green}.log`,
`build/asan/build_radix_wave_flip_{prefix,fixed}.log`,
`build/asan/test_radix_wave_negative_flip.log`.

### Gate G: PASS — fixed binary faster on every primary shape

```text
$ tmp/radix-wave-deadlock/run_perf_guard.sh \
    tmp/radix-wave-deadlock/bin/perf-baseline/clickhouse build/reldeb/programs/clickhouse
perfguard verdict: PASS (no regression beyond max(5%, 1 baseline stdev))
```

Primary shapes (`bp=pp=1`, T96, 5 paired position-balanced samples per
binary, `radix_join` median ms; band = max(5%, 1 baseline stdev)):

| D | ratio | baseline median (stdev) | fixed median | delta | verdict |
| --- | --- | --- | --- | --- | --- |
| 67108864 | 2 | 481 (3.4) | 409 | -14.97% | PASS |
| 268435456 | 2 | 1917 (23.9) | 1633 | -14.81% | PASS |
| 268435456 | 4 | 3957 (9.8) | 2843 | -28.15% | PASS |

The fixed binary is significantly *faster* at T96 on all three shapes —
the cooperative drain replaced the old single-consumer queue bottleneck.
Per-sample data: `build/reldeb/test_radix_wave_perf_guard.tsv`; raw logs
under `tmp/radix-wave-deadlock/perf-guard/`.

Thread sweep on D=67108864 ratio=2 (report-only, one `--runs 3` invocation
per cell, no stdev): T32 -5.7%, T64 -10.3%, T96 -14.2% (all fixed faster);
T1 +5.9% and T16 +11.8% flagged slower. Because single invocations carry no
error bars, the two flagged cells were settled with 5 paired
position-balanced samples per binary
(`tmp/radix-wave-deadlock/perf-guard/sweep_settle.tsv`):

- T1: baseline median 14216 ms (stdev 689), fixed 14269 ms, +0.37% —
  inside the band; the single-invocation flag was noise. SETTLED, PASS.
- T16: baseline median 892 ms (stdev 6.9, samples 887-904), fixed 992 ms
  (samples 897-1012), +11.21% — beyond the max(5%, 1 stdev) band with tight
  baselines. A REAL, localized regression at this mid thread count on this
  shape, surfaced as a finding in REPORT.md with a recommendation (the
  binding T96 guard passed with double-digit improvements on all three
  frozen shapes, and T32/64 also improved; `radix_join`'s planner gate
  targets large high-thread shapes). Not silently accepted, not hidden.

## Independent verification: SHIP

A fresh, fully independent verifier (no involvement in the implementation;
given only the mission, the diff, WORKLOG.md, REPORT.md, and the gate
artifacts) re-ran Gates B-F from a clean state — all green, with raw
evidence per gate — recomputed every Gate G statistic from the raw TSVs,
re-counted the Gate-1 dumps, audited the evidence matrix (every row entails
its criterion), and inspected the final code for lifecycle gaps (none
blocking). Verdict: SHIP. Its named non-blocking leads are folded into
REPORT.md. One operational note from the verification: Gate F's `cp`
restore leaves the asan binary mtime-stale relative to the file, so the
verifier's first build re-compiled an identical `RadixHashJoin.cpp.o` —
content-identical, no impact on any verdict.

## Close-out

All gates green on the final source (product file SHA-256
`5b0a6b7401ca1115603970ffbd3ac29a9823adc1ffc6310e583f44dfd2ef311b`); the
independent verdict is SHIP; the fix and its regression test are committed
as small green commits on `codex/radix-join-wave-deadlock`; no push and no
pull-request action was taken (not authorized). REPORT.md holds the
evidence matrix, performance numbers, the T16 finding with recommendation,
and the risk-accepted leads.

## Unit 3: verification and performance

### Gate G shape freeze (screened pre-fix, before any fixed-binary run)

One screening run per candidate payload-narrow shape (`bp=pp=1`, T96,
multiplicity 1, hit 1, `--runs 1 --no-verify`) on the frozen aarch64 pre-fix
baseline (`bin/perf-baseline/clickhouse`, SHA-256 `f97d4127...26b3`), each in
an isolated process group with a 900-second kill guard
(`tmp/radix-wave-deadlock/explore_perf_shapes.sh`; raw logs under
`tmp/radix-wave-deadlock/perf-explore/`). All five candidates completed
pre-fix (no hang) with green assertions:

| D | ratio | radix_join ms | parallel_hash ms | winner |
| --- | --- | --- | --- | --- |
| 67108864 | 2 | 478 | 674 | radix_join 1.410x |
| 67108864 | 4 | 894 | 975 | radix_join 1.091x |
| 268435456 | 2 | 1896 | 2745 | radix_join 1.448x |
| 268435456 | 4 | 3823 | 3931 | radix_join 1.028x |
| 524288000 | 2 | 4882 | 4446 | parallel_hash 1.098x |

Frozen Gate G shapes (favor `radix_join`, complete pre-fix):

1. D=67108864, ratio=2, bp=pp=1, T96 (clear win, small/fast)
2. D=268435456, ratio=2, bp=pp=1, T96 (largest win)
3. D=268435456, ratio=4, bp=pp=1, T96 (thinnest margin — the most sensitive
   canary for busy-spin overhead introduced by the fix; same D/ratio as the
   accepted Gate 1 hang shape, at narrow payload)

Additionally, per the standing thread-sweep requirement, shape 2 will also be
measured at threads 1/16/32/64/96 on both binaries (the admission busy-spin
scales with lane count, so a single thread count could hide a regression).
