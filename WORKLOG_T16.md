# WORKLOG_T16 — recovering the T16 `radix_join` regression

Campaign: wave-deadlock fix follow-up. Branch `radix-join-bandwidth-model`, tip
`b8557c1f56b` at mission start, tree clean. Unattended session
`claude-ffd8242a-2ac7-413a-a38a-245e200d0d21`, started 2026-07-14.

## Fixed reference points (verified in-session)

- PRE-FIX binary: `tmp/radix-wave-deadlock/bin/perf-baseline/clickhouse`,
  SHA-256 `f97d41279797aed3623bcd9d703286019e9cd3d1c7257a7a6213a089790a26b3`
  (matches the mission-pinned hash).
- SHIPPED binary: `build/reldeb/programs/clickhouse`, SHA-256
  `148ee743a09f4031c2e3cfbfe50448fd7d522e3b53f6792032c68bd687497a3a`, built from
  the clean tip `b8557c1f56b`. Note: the first `ninja -C build/reldeb clickhouse`
  rebuilt `RadixHashJoin.cpp.o` and relinked (stale object mtime left by another
  session), producing a byte-identical binary; the second run was a no-op
  (`build/reldeb/build_t16_noop_check{,2}.log`). Rebuild determinism verified —
  the SHIPPED binary can be restored bit-exactly after instrumented rebuilds by
  reverting the source and rebuilding, verified by this hash.
- Host: aarch64, 96 cores, `perf_event_paranoid=4` (no HW counters). Other agent
  sessions are active on this box; the host is checked for idleness (load, no
  ninja/clickhouse/pytest processes) before every timing batch, and load is
  recorded before/after. Deviation risk flagged: if a foreign process starts
  mid-batch, the batch is discarded and re-run.
- Benchmark protocol (mission-fixed): >=5 paired position-balanced invocations
  per binary per cell; judge `radix_join` `median_ms` (4th column of the harness
  result table); band = max(5%, 1 stdev) of the reference binary's samples.
  Harness: `bep/tools/join_mergetree_bench.py`, dataset `/mnt/data/join_bench_data`
  (read-only).

## Design facts (read from code, commit `02b7ae9d93d` diff)

Pre-fix wave lifecycle at T16:
- `joinBlock` swapped the window out immediately on reaching budget, then parked
  the lane on `wave_mutex` holding the pre-staged window; the wave started
  (scatter + 16 workers) under that held lock when the previous wave's result
  died. Exactly ONE consumer lane (the owner) drained each wave's bounded queue
  (capacity `2*16+1 = 33`), one block per `next()` quantum. Parked lanes
  pipelined subsequent windows; up to `threads` windows could be staged.
Fixed wave lifecycle at T16:
- `joinBlock` appends under `window_mutex`; on budget it returns a weightless
  `WaveJoinResult`. `next()` under `coordinator.mutex` swaps the window and runs
  the scatter (pool-parallel) with the coordinator mutex held, then schedules 16
  workers. EVERY lane holding a result co-drains the active queue, one block per
  quantum, each pop preceded by a `coordinator.mutex` lock + `shared_ptr` copy.
  A consumer lane stays attached across chained waves until it observes
  "no active wave AND window below budget" (then `is_last`, resumes scanning).
  Teardown: election via `torn_down.exchange`; losers return an empty non-last
  block (an extra executor quantum) while the winner runs `joinWorkers`.
- Window budget is a fraction of `post_build_bytes` — independent of threads.

## Iteration 1 — re-establish the T16 regression fresh (Unit 1)

**Goal.** Reproduce the preregistered cell fresh, in-session: D=67108864 m=1
ratio=2 hit=1 bp=pp=1, `max_threads=16`.

**Pre-registered protocol.** 5 paired position-balanced invocations per binary,
sequence P F F P P F F P P F (P = PRE-FIX, F = SHIPPED), serial, idle host.
Command per invocation (only `--binary` varies):

    python3 bep/tools/join_mergetree_bench.py run \
      --path /mnt/data/join_bench_data --binary <bin> \
      --cardinalities 67108864 --multiplicities 1 --ratios 2 --hit-rates 1 \
      --build-payload-columns 1 --probe-payload-columns 1 \
      --threads 16 --runs 1 --no-verify --max-memory 100000000000

Logs: `tmp/t16/fresh_t16_{prefix|shipped}_{1..5}.log`.

**Pre-registered expectations.**
- If the regression is real and stable: SHIPPED median above PRE-FIX median +
  band (campaign record: 892 vs 992, band would be ~max(5%,7ms) = ~45ms), with
  SHIPPED samples bimodal (one or more near PRE-FIX levels, rest ~985-1012).
- Refuting outcome: SHIPPED median within PRE-FIX median + band → the regression
  does not reproduce today → stop, re-measure with more samples before any
  attribution work; if it still fails to reproduce, report UNSETTLED with data.

**Result.** Regression REPRODUCED (MATERIAL, fresh in-session data). Host idle
before batch (load 0.03/0.23/1.03), all 10 invocations rc=0
(`tmp/t16/run_fresh_t16.out`), logs `tmp/t16/fresh_t16_*.log`:

| binary | samples (ms, run order) | median | stdev |
| --- | --- | --- | --- |
| PRE-FIX | 884, 901, 898, 893, 896 | 896 | 6.5 |
| SHIPPED | 1030, 1018, 884, 999, 1008 | 1008 | 58 (bimodal) |

Band = max(5% of 896, 6.5) = 44.8 ms → threshold 940.8. SHIPPED median 1008 is
+112 ms (+12.5%) — red, matching the campaign record. Bimodality reproduced:
`shipped_3` (7th position, mid-batch) = 884 ms, indistinguishable from the
PRE-FIX distribution; the other four are 999-1030. Interpretation: the fixed
binary has two regimes — a fast regime at full pre-fix speed (~1 in 5 runs) and
a slow regime costing ~+120 ms (~4 in 5 runs). The remedy target is regime
elimination, not a constant tax.

**Free evidence from campaign logs** (`settle_t16_fixed_{1..5}.log`, LEAD):
comparing the fast fixed sample (897) against slow ones (985-1012):
`build_ms`, `probe_ms`, `pack_ms`, `leaf_ms` are flat across regimes;
`hash_match_ms`/`hash_gather_ms` (thread-summed leaf work) are slightly LOWER in
slow runs (3020-3031 vs 3086; baseline 3092). Same or less summed work with
+100 ms wall → the slow regime is a pipeline/overlap loss (gaps), not extra
per-block CPU. This weakens any "added per-block cost" candidate on its own.

## Iteration 2 — attribution probes (Unit 1) — PRE-REGISTERED before running

**Candidate mechanisms** (from the design diff, § Design facts):

- **H1 — consumer-set accumulation starves the scan.** Fixed-design lanes that
  co-drain a wave pull no input and stay attached across chained waves; if most
  lanes attach, scan parallelism collapses, the window refills slowly, and pure
  scan gaps appear between waves (pre-fix kept ~15 lanes scanning and pipelined
  pre-staged windows). Signature: slow runs show high per-wave consumer attach
  counts AND wave-end shared-window bytes below budget (starved chain) with
  inter-wave gaps; fast runs show few attaches and/or window-at-end >= budget.
- **H2 — queue-full producer parking.** 16 producers outpace quantum-paced
  consumers; capacity-33 queue fills; producers park; wave elongates.
  Signature: producer park time (sum) in slow runs on the order of
  16 x 100 ms; near zero in fast runs. Refuted if park time is small or
  regime-independent.
- **H3 — scatter serialization under the coordinator mutex.** Signature:
  per-wave scatter duration large (tens of ms) and bigger/more frequent in slow
  runs. (Both designs scatter serially per wave, so only a regime-dependent
  difference counts.)
- **H4 — teardown-election loser spin.** Losers burn empty executor quanta
  while the winner joins workers. Signature: loser-quantum counts per wave in
  the many-thousands in slow runs. Refuted if counts are a handful per wave.
- **H5 — wave-count / delayed-flush share regime (bimodality).** Wall time
  depends on how many budget waves run vs. how much lands in the final
  delayed flush (which drains without a bounded queue). Signature: fast runs
  have fewer waves and a larger flush; slow runs more waves. Refuted if wave
  count and flush bytes are regime-independent.

A probe outcome compatible with every candidate is vacuous; the per-wave
structure below separates them: H1 (attaches + window-at-end + gaps),
H2 (park_us), H3 (scatter_us), H4 (loser quanta), H5 (wave count + flush bytes).

**Probe: throwaway instrumented build** (never ships; product tree restored and
hash-verified afterwards). Patch `RadixHashJoin.cpp` only, on top of the clean
tip, adding stderr counters:

- `RADIXBUDGET` once per query: window budget, post-build bytes, fanout, threads.
- `RADIXSTART` per wave: id, steady-clock t0, scatter_us, window bytes/blocks.
- `RADIXWAVE` per wave at teardown: id, created/published/finished/torn-down
  timestamps, pushes, full-pushes, producer park_us, pops, consumer attaches,
  loser quanta, shared-window bytes at teardown.
- `RADIXFLUSH` once: delayed-flush bytes/blocks.

Runner: replicate the harness measurement pipeline exactly (same
`measurement_script` = 1 warmup + 1 timed query, same
`clickhouse local --print-profile-events --time --progress=off
--profile-events-delay-ms=-1` flags, same dataset path), capturing stderr.
12 invocations at T16 to catch both regimes (expected ~2-3 fast, ~9-10 slow).
Classify each run fast/slow by its timed wall; compare per-wave structure
across regimes WITHIN the same instrumented binary (no pre/post confound).

**Pre-registered decision rule.** The attributed mechanism is the candidate
whose signature separates fast from slow runs by an amount that accounts for
the ~+120 ms delta (order of magnitude, not exact). If no candidate separates,
Unit 1 loops with a new probe design (<= 5 cycles). If the instrumented binary
stops being bimodal (observer effect), that itself is reported and the probe
redesigned with lighter counters.

**Probe execution note — first batch invalid (recorded null result).** The
first 12-run instrumented batch (archived in `tmp/t16/invalid_fullscan/`) was
run with SQL generated from `BenchmarkPoint(..., bucket_width=1)`. The loaded
dataset's metadata has `bucket_width=4194304`, `max_cardinality=524288000`, so
the harness's real probe subquery prunes with `PREWHERE cycle < 2 AND
card_bucket < 16`; with `bucket_width=1` the predicate became `card_bucket <
67108864` — no granule pruning, a 58.7 GB whole-table read
(`CompressedReadBufferBytes`), ~7.8 s walls. Those runs measure a different
cell and are discarded. The corrected SQL is regenerated through the harness's
own `read_metadata` + `validate_points` path (verified: `card_bucket < 16`,
point label `D=67108864 m=1 ratio=2 hit=1 bp=1 pp=1`), so the query text is
byte-identical to what `measurement_script` gives the harness. Incidental
observation from the invalid runs (not evidence for the T16 cell): waves
alternated `staged`/`STARVED` window states with 0.6 s inter-wave gaps and
multi-second producer park times — the instrumentation and analyzer work.

**Result.** ATTRIBUTED (MATERIAL for Unit 1). Corrected batch: 12/12 rc=0,
logs `tmp/t16/instr_t16_{1..12}.err`, analyzer `tmp/t16/analyze_instr.py`.
Query structure at this cell: budget 610 MB (0.15 x post_build 4064 MB),
fanout 4096, 3 waves (~610 MB, 4096 output blocks each) + 311 MB delayed flush.

| run | wall | total inter-wave gap | att per wave | wend(wave2) | park sum | losers | scatter sum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| instr_3 (FAST) | 0.920 s | 17.4 ms | 10,10,1 | 520 MB | 374 ms | 0 | 106 ms |
| 11 slow runs | 0.992-1.017 s | 115.1-118.2 ms | 16,16,1 | 0 MB | 110-328 ms | 0 | 101-107 ms |

- **H1 CONFIRMED.** In every slow run the first timed wave captures all 16
  lanes (`att=16`); the second wave then runs with ZERO scanning lanes and ends
  with the shared window at 0 bytes; every lane detaches (`is_last`), and a
  ~116 ms pure-scan refill gap (one full budget at 16-lane scan rate
  ~5.3 GB/s) sits exposed before the third wave. Gap delta fast-vs-slow
  (~99 ms) matches the wall delta (~80-97 ms) and the fresh regression
  (+112 ms). The fast run is exactly the incomplete-capture case: `att=10`
  leaves 6 scanners, the window refills to 520/610 MB during wave 2, and the
  gap collapses to 17 ms. Bimodality = the capture race: whether the window
  re-crosses budget early enough during wave 1 for every lane's next
  `joinBlock` to convert it into an attached consumer.
- **H2 REFUTED**: producer park time does not separate regimes and is
  anti-correlated with wall (fast run has the largest park sum, 374 ms).
- **H3 REFUTED**: per-run scatter sum is flat (101-107 ms) across regimes.
- **H4 REFUTED**: teardown loser quanta are zero in all 24 executions.
- **H5 REFUTED**: wave count (3) and flush share (311 MB) are identical in
  every run; the one fast run differs only in gap structure.

Mechanism statement: the cooperative drain makes wave-consumership sticky and
unbounded — any lane whose `joinBlock` lands while the window is >= budget and
a wave is active attaches and stops pulling input until it observes
no-wave-and-below-budget. At T16 the capture usually saturates (16/16), which
serializes one full window's scan behind each captured wave pair. Pre-fix
bounded consumers to 1 per wave (the owner) and parked at most a few lanes
holding pre-staged windows, so >= ~14 lanes always kept scanning and the scan
never surfaced on the critical path. T1 is immune (single lane in both
designs); T32+ increasingly hides the gap because more lanes mean faster
refill and the executor overlaps other work.

## Iteration 3 — T96 drain structure (Unit 1, sizing input for Unit 2) — PRE-REGISTERED

Any remedy that bounds the consumer set must not undo the fix's T96 win (the
pre-fix single-consumer drain was the T96 bottleneck per the campaign). Before
designing: measure the same per-wave counters at T96 on the frozen primary
shape D=67108864 r=2 bp=pp=1 (4 runs) and on D=268435456 r=2 (2 runs).

**Pre-registered expectations.** If the T96 win comes from multi-consumer
drain relief: T96 waves show large attach counts and, per wave, aggregate pop
rate = pops / wave duration far above what one consumer could sustain
(~1 block / 15 us quantum => ~66k blocks/s). The minimum consumer count the
remedy must allow at T96 is then ~pop_rate x 15 us with margin. If instead T96
waves show small attach counts or single-consumer-compatible pop rates, the
consumer cap can be uniform and small. Also check whether T96 waves chain
staged (wend >= budget) — if T96 also starves but hides it, a cap must not
re-expose it.

**Result.** (6/6 runs rc=0, logs `tmp/t16/instr_t96_*.err`.)

| cell | wall | waves | att per wave | gaps | wave dur (att=96 / att=1) |
| --- | --- | --- | --- | --- | --- |
| D=67108864 r2 T96 (4 runs) | 0.380-0.416 s | 3 | 96,96,1(2) | ~37 ms | 30-33 ms / 74-79 ms (att=2: 37 ms) |
| D=268435456 r2 T96 (2 runs) | 1.622-1.634 s | 3 | 96,96,1 | ~142 ms | 125-132 ms / 298-304 ms |

- T96 shows the SAME capture-and-starve pattern (att=96, second wave ends with
  window at 0, pure-scan gap before the last wave) — the fix's T96 win survives
  it because multi-consumer drain relief there outweighs the gap.
- Consumer throughput quantified: a single attached consumer pops ~55k blocks/s
  (4096 pops in 74-79 ms; 16384 in 298-304 ms). Producer-side demand at T96 is
  ~136k blocks/s (D=67M: 4096/30 ms) and ~126k blocks/s (D=268M: 16384/130 ms).
  One run's last wave had att=2 and drained in 37 ms — consumer scaling is
  ~linear at small counts. So the T96 multi-consumer win needs only ~3
  concurrent consumers; 96 is enormously past saturation.
- Scan-refill saturates with few lanes: the T16 fast run refilled at ~5.4 GB/s
  with only 6 free lanes (520 MB during a 97 ms wave), the same rate as 16
  free lanes in the pure gap (610 MB / 116 ms). Freeing "most" lanes is not
  required — freeing "more than a handful" is.

## Unit 2 — remedy, PRE-REGISTERED before implementation

**Attributed mechanism being fixed** (Unit 1, MATERIAL): unbounded, sticky
wave-consumership. Any lane whose `joinBlock` lands while the shared window is
at budget and a wave is active converts into a queue consumer and stops
pulling input until it observes no-wave-and-below-budget; at T16 the capture
usually saturates all 16 lanes, the following wave runs with zero scanners,
and a full window refill (~116 ms) is exposed on the critical path.

**The change** (smallest lifecycle change targeting the mechanism): bound the
number of consumer lanes attached to one wave.

- `ActiveWave` gains `std::atomic<size_t> consumers{0}`.
- `WaveJoinResult` gains `std::shared_ptr<ActiveWave> attached_wave` and
  `const size_t max_consumers = std::max<size_t>(2, threads / 8)`; on seeing a
  wave it is not attached to, it detaches from the previous wave (atomic
  decrement) and tries a CAS-attach; if the wave's consumer slots are full it
  returns `{Block{}, nullptr, true}` — the exact semantics of the existing
  below-budget path ("another wave took the rows this result was created
  for"), sending the lane straight back to pulling input. `~WaveJoinResult`
  detaches (plain atomic decrement — still inert: no waiting, no teardown).
- Nothing else changes: window accounting, wave startup/scatter, teardown
  election, worker lifecycle, abandoned-wave reaping, and the delayed flush
  are untouched. `joinBlock` already appends unconditionally, so the window
  pre-stages past the budget while a wave runs and the winner's loop chains
  the next wave gap-free.

**Why the liveness invariant survives:** capped-out lanes never wait — they
return immediately with `is_last` (their rows are already in the shared
window, probed by a later wave or the delayed flush — same as today's
below-budget path). Attached lanes wait only on the queue (producers are
dedicated pool workers) and the coordinator mutex (scatter holder), unchanged.
A live wave always has >= 1 attached consumer: slots only report full when
`consumers == max_consumers >= 2`, and an attached transform cannot finish
while its wave is live (its `next` never returns `is_last` before teardown),
which also preserves the `getDelayedBlocks` no-active-wave invariant.
Cancellation paths are unchanged (destructor reaping; detach is a decrement).

**Cap sizing, from Iteration 2/3 measurements:** consumer drain scales
~linearly at ~55k blocks/s each; the largest measured producer demand is
~136k blocks/s (T96). `max(2, threads/8)` gives: T16 -> 2 (drain 110k/s =
2.1x the T16 demand of ~52k/s; leaves 14 scanning lanes where >= 6 already
achieve the full ~5.3 GB/s refill rate), T32 -> 4, T64 -> 8, T96 -> 12
(660k/s = 4.8x measured demand, 84 lanes free). T1: a single lane attaches
exactly as today (cap >= 1 lane present) — no behavior change.

**Oracle interaction check (read, not modified):** the liveness gtest runs
`max_threads=2` -> cap 2; lane B attaches and pops exactly as today; the
test's total-and-multiset assertions are split-agnostic by design. The test is
NOT modified, so the negative-flip re-run requirement does not trigger.

**Mechanism confirmation probe (LEAD, before gates):** rebuild the remedy WITH
the Iteration 2 instrumentation on top (never ships), 12 x T16 runs: expect
per-wave `att <= 2`, no zero-scan starved generation (wend of the middle wave
well above 0), total inter-wave gaps <= ~40 ms in every run, walls unimodal
near the fast regime. Refuting outcome: att still saturating, or gaps still
~116 ms bimodal -> remedy does not act on the attributed mechanism; stop and
re-design (cycle 2 of <= 5).

**Gate invocations and refuting outcomes (run in this order, each red stops
the unit):**

1. **Gate T16**: fresh paired batch, sequence P C C P P C C P P C (P=PRE-FIX,
   C=candidate `build/reldeb/programs/clickhouse` with the remedy), harness
   cell D=67108864 m=1 r=2 hit=1 bp=pp=1 T16, idle host. Green: candidate
   median <= PRE-FIX median + max(5%, 1 stdev of PRE-FIX samples). Red:
   anything else.
2. **Gate floors**: fresh paired batches candidate-vs-SHIPPED
   (`tmp/t16/bin/shipped/clickhouse`, a preserved copy of hash `148ee743...`),
   cells: same shape at T1, T32, T64, T96; D=268435456 r=2 T96; D=268435456
   r=4 T96. Green: candidate median <= SHIPPED median + max(5%, 1 stdev of
   SHIPPED samples) in EVERY cell. Red: any cell outside.
3. **Gate liveness**: `ninja -C build/asan unit_tests_dbms` (log in build
   dir), then 10 consecutive
   `timeout --signal=TERM --kill-after=10s 30s build/asan/src/unit_tests_dbms
   --gtest_filter=RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave`
   -> 10/10 exit 0 AND each log contains exactly `[  PASSED  ] 1 test`.
4. **Gate large**: `tmp/radix-wave-deadlock/run_large_fixed_gate.sh
   build/reldeb/programs/clickhouse 268435456 4` -> exit 0, `Assertions: PASS`,
   counts 1073741824/268435456/1073741824.
5. **Gate early-term**: `tmp/radix-wave-deadlock/run_early_termination_gate.sh
   build/reldeb/programs/clickhouse` -> 6/6.
6. **Gate stateless**: 04508..04512 via `tests/clickhouse-test -b
   build/reldeb/programs/clickhouse` against the scratch server
   (`tmp/radix-wave-deadlock/testserver/config.xml`, ports 9131/8161, server
   binary verified by hashing `/proc/<pid>/exe`) -> 5/5.

**Implementation.** Done on top of the clean tip; diff `tmp/t16/remedy.patch`
(+54/-6 lines, `RadixHashJoin.cpp` only). Clean candidate binary
`build/reldeb/programs/clickhouse` SHA-256
`466b8a2609b6e88bcaeaae3a9654c15fb71b0a31e8847f6c77a38b3e1f535285`
(rebuild determinism re-verified: instrumented build layered on top, then
source restored to remedy-only and rebuilt to the identical hash). SHIPPED
reference preserved at `tmp/t16/bin/shipped/clickhouse` (hash re-verified
`148ee743...`). Instrumented-remedy binary `tmp/t16/bin/instr_remedy/clickhouse`
(`628e9e80...`, never ships).

**Mechanism confirmation (LEAD, instrumented-remedy, 12 x T16 runs,
`tmp/t16/instr_remedy_t16_*.err`).** All pre-registered expectations met:
walls 0.882-0.904 s (median 0.897), UNIMODAL at the fast-regime level; per-wave
attach counts [2,2,1] (one run [1,1,1]) — never above the cap; total
inter-wave gaps 1.4-16.0 ms in every run (shipped slow regime: ~116 ms);
middle wave ends with 517-604 MB staged (shipped slow regime: 0 MB);
losers=0. (Analyzer note: the instrumented-remedy binary accidentally carried
two RADIXWAVE dump sites — the old patch hunk plus the adapted one — so waves
appear twice per file; analysis dedupes by wave id. Incidentally the second
dump shows the next wave's window swap landing within ~24 us of teardown:
gap-free chaining in action.)

**Gate results.**

1. **Gate T16: GREEN.** Idle host (load 0.86 before batch), paired
   P C C P P C C P P C, logs `tmp/t16/gate_t16_{prefix,cand}_{1..5}.log`,
   all rc=0. PRE-FIX samples 909/902/915/901/890 (median 902, stdev 9.4);
   candidate samples 905/913/904/911/898 (median 905). Band = max(45.1, 9.4)
   = 45.1 -> threshold 947.1; 905 <= 947.1. Candidate is unimodal at the
   pre-fix level; no slow-regime sample in 5 runs (shipped showed 4/5).
   Judge: `python3 tmp/t16/judge.py gate_t16 ...` (exit 0; the judge's power
   to fail was demonstrated on the Iteration 1 data, where it returns RED).

2. **Gate floors: GREEN (all six cells).** Paired S C C S S C C S S C per
   cell vs the preserved SHIPPED binary, 60/60 invocations rc=0
   (`tmp/t16/run_floors.out`), logs `tmp/t16/floor_<cell>_{shipped,cand}_*.log`,
   judged by `tmp/t16/judge.py`:

   | cell | SHIPPED median (stdev) | candidate median | threshold | verdict |
   | --- | --- | --- | --- | --- |
   | T1 D=67108864 r2 | 14098 (160.5) | 14234 | 14802.9 | GREEN |
   | T32 D=67108864 r2 | 688 (8.3) | 649 | 722.4 | GREEN (cand -5.7%) |
   | T64 D=67108864 r2 | 448 (13.3) | 367 | 470.4 | GREEN (cand -18%) |
   | T96 D=67108864 r2 | 417 (5.9) | 370 | 437.9 | GREEN (cand -11%) |
   | T96 D=268435456 r2 | 1618 (19.6) | 1419 | 1698.9 | GREEN (cand -12%) |
   | T96 D=268435456 r4 | 2866 (38.9) | 2151 | 3009.3 | GREEN (cand -25%) |

   The remedy does not pay for T16 with the fix's wins — it extends them: the
   consumer cap removes the same capture-starvation gap at every mid-to-high
   thread count (consistent with Iteration 3, which measured the identical
   pattern at T96).

3. **Gate liveness: GREEN (10/10).** `ninja -C build/asan unit_tests_dbms`
   clean (6-line incremental log, `build/asan/build_t16_asan.log`). 10
   consecutive runs of `timeout --signal=TERM --kill-after=10s 30s
   build/asan/src/unit_tests_dbms --gtest_filter=RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave`
   -> every run rc=0 AND its log (`build/asan/t16_liveness_run_{1..10}.log`)
   contains exactly one `[  PASSED  ] 1 test` line (zero-test-filter
   impossible). The oracle test was not modified, so the negative-flip re-run
   requirement does not trigger.

4. **Gate large: RED (cycle 1 remedy rejected).**
   `tmp/radix-wave-deadlock/run_large_fixed_gate.sh build/reldeb/programs/clickhouse
   268435456 4` -> exit 1. Assertions PASS with exact counts
   (probe/build/joined = 1073741824/268435456/1073741824) and the shape does
   not hang, but the `radix_join` measurement died with rc=241:
   `Query memory limit exceeded: would use 93.13 GiB ... maximum: 93.13 GiB
   (MEMORY_LIMIT_EXCEEDED)` while executing `MergeTreeSelect`
   (`build/reldeb/test_radix_wave_gatec_d268435456_r4.log`). The SHIPPED
   binary passes this same gate at the same limit (campaign record, and it is
   the gate's contract). Per the gate rules the unit stops: cycle 1's remedy
   is rejected as-is.

## Unit 2, cycle 2 — diagnose the memory red, then bounded-overshoot remedy — PRE-REGISTERED

**Hypothesis for the red (H-mem).** The cycle-1 cap removed the design's input
back-pressure. In SHIPPED, a lane that attaches to a wave stops pulling input,
so the shared window overshoots the budget by at most ~1 in-flight block per
lane. With the cap, capped-out lanes keep scanning for the entire wave, and the
window grows by scan_rate x wave_duration past the budget. At the bp=pp=7
D=268435456 r=4 T96 shape (multi-second waves, wide rows, ~90+ lanes free) the
overshoot reaches tens of GiB and blows the 100 GB harness cap. The bp=pp=1
floor cells passed because their budgets and row widths are ~10x smaller.

**Probe (pre-registered).** One run of the instrumented cycle-1 remedy binary
(`tmp/t16/bin/instr_remedy/clickhouse`) on the failing cell (D=268435456 r=4
bp=pp=7 T96), capturing `RADIXBUDGET` (budget) and `RADIXSTART winb` (bytes
actually swapped into each wave). Expected if H-mem true: memory failure
reproduces AND at least one wave's `winb` (or the shared window at failure)
is several times the budget. Expected if H-mem false: `winb` stays ~budget
and the failure reproduces anyway -> different mechanism, re-diagnose.

**Remedy v2 (pre-registered, implemented only if the probe confirms H-mem).**
Make surplus-lane scanning conditional on the shared window needing input —
the symmetric invariant "a surplus lane scans iff the next window is below
budget":

- Rule A (memory bound, restores the shipped back-pressure): a capped-out
  lane checks the shared window; if `window_bytes >= budget` (the next wave
  is fully staged) it attaches PAST the cap — stopping its input pull exactly
  like SHIPPED — instead of returning to scan. Window overshoot bound returns
  to ~budget + one in-flight block per lane.
- Rule B (throughput, keeps the T16 payoff): an attached lane, on entering a
  pop quantum while `consumers > max_consumers` and `window_bytes < budget`,
  CAS-releases its slot (floor = max_consumers) and returns `is_last`,
  going back to scanning. So after a wave swap empties the window, the
  over-cap consumers drain back into scanners, and the guaranteed crew of
  `max_consumers` keeps draining the wave.
- Liveness unchanged: no new waits (Rule A attaches; Rule B is a CAS +
  bounded `window_mutex` lock); a live wave keeps >= max_consumers >= 1
  attached consumers at all times; capped-out lanes still return immediately.

**Gate plan for cycle 2:** the mechanism confirmation at T16 (instrumented v2:
still att<=cap during below-budget phases, gaps <= ~40 ms, unimodal walls) and
then the FULL gate suite re-run from scratch on the v2 binary (Gate T16, all
six floors, liveness 10/10, large, early-term, stateless) — timing gates
re-measured fresh, not carried over from cycle 1. Refuting outcomes: as in
cycle 1, plus Gate large red again -> cycle 3 or UNSETTLED with data.

**Cycle-2 probe result: H-mem CONFIRMED.** One instrumented cycle-1-remedy run
on the failing cell (`tmp/t16/instr_remedy_large_1.err`, rc=241):
`RADIXBUDGET budget=5464316530 post_build=36428775424`; a single
`RADIXSTART id=1 bytes=5465346048` (first wave, exactly one budget) and then
`MEMORY_LIMIT_EXCEEDED` at 93.13 GiB **while reading probe column `p_p5` in
`MergeTreeSelect`** — during wave 1's drain, before any second wave. The ~90
surplus lanes scanned for the whole multi-second wave and grew the shared
window without bound (post_build 36.4 GB + window overshoot + wave in flight
blew the 100 GB cap). Discriminates exactly as pre-registered: the failure is
input accumulation during a wave, not leaf or queue growth. Proceeding with
remedy v2 (Rules A and B) as pre-registered.

**Cycle-2 implementation.** v2 implemented as pre-registered (Rules A and B),
clean build SHA-256 `113ca763c163d084bc08122d26d68cccddf6df4f0384b64a8d51fbded3d74d33`,
diff `tmp/t16/remedy_v2.patch`. Instrumented v2
(`tmp/t16/bin/instr_v2/clickhouse`, `1179f4a7...`) probe on the failing cell
(`tmp/t16/instr_v2_large_1.err`): **rc=0, zero MEMORY_LIMIT lines** — the red
is repaired. Every wave swaps in `winb` ~ 5.5 GB (= budget, bounded) and ends
with the window staged at budget (`wend` = 5.5 GB -> gap-free chaining);
`att=96, capped=84, overcap=84` shows Rule A engaging: 84 surplus lanes stop
pulling input once the window is staged.

**Amendment before gates — Rule B removed as unreachable (v2.1).** The probe
showed `released=0` everywhere, and inspection proves it structurally: within
one wave's lifetime the shared window only grows (its only decrease is the
swap in `startWave`, which runs strictly after this wave's teardown-and-reset,
or in `getDelayedBlocks`, which asserts no active wave), and an over-cap
consumer attached only because the window was >= budget — so `consumers > cap
AND window < budget` can never hold for the wave a consumer is attached to.
The slot handback Rule B aimed at happens naturally at the wave switch: an
over-cap consumer fails `tryAttach` on the next wave, finds its window
freshly-swapped below budget, and returns to scanning. Rule B is deleted
(dead code holding a mutex in the pop path); Rule A and the cap are the whole
remedy. The full gate suite runs fresh on the v2.1 binary.

**v2.1 mechanism confirmation (LEAD).** Clean v2.1 binary SHA-256
`cf662d4c9619c7c26b1a713b57353a9024749bc9ad1d6b70f767d3bbe926bb5a`
(diff `tmp/t16/remedy_v2_1.patch`, +74/-6); instrumented v2.1
`tmp/t16/bin/instr_v2_1/clickhouse` (`4fe55141...`). Probes
(`tmp/t16/instr_v21_t16_{1..12}.err`, `instr_v21_large_1.err`):
- T16: 12/12 rc=0, walls 0.890-0.934 s (median 0.921), no slow-regime sample;
  inter-wave gaps 13-45 ms (shipped slow regime: ~116 ms). Counter narrative:
  when wave 1 refills to budget mid-wave, Rule A captures the surplus lanes
  (att=16, overcap=14) and the window stops at ~617 MB; wave 2 starts staged
  (zero gap) and immediately caps them back out (capped=14) so 14 lanes scan
  during wave 2; wave 2 ends at 380-450 MB, leaving only a 30-45 ms residual
  gap. Runs where the window never hit budget mid-wave show att<=3 and 13-16 ms
  gaps. The all-lanes-captured starved generation no longer exists.
- Large cell: rc=0, zero MEMORY_LIMIT lines after Rule B's removal.

**Cycle-2 gates (v2.1, all fresh; cycle-1 logs archived in `tmp/t16/cycle1/`).**

1. **Gate T16: GREEN.** Paired P C C P P C C P P C, idle host, 10/10 rc=0,
   logs `tmp/t16/gate_t16_{prefix,cand}_{1..5}.log`. PRE-FIX 894/909/886/911/916
   (median 909, stdev 12.6); candidate 924/911/902/942/927 (median 924).
   Band = max(45.5, 12.6) = 45.5 -> threshold 954.5; 924 <= 954.5. Candidate
   unimodal; no slow-regime sample.
2. **Gate floors: GREEN (all six cells).** 60/60 rc=0, logs
   `tmp/t16/floor_<cell>_{shipped,cand}_*.log`:

   | cell | SHIPPED median (stdev) | candidate median | threshold | verdict |
   | --- | --- | --- | --- | --- |
   | T1 D=67108864 r2 | 14329 (486.3) | 14443 | 15045.5 | GREEN |
   | T32 D=67108864 r2 | 702 (8.4) | 669 | 737.1 | GREEN (cand -4.7%) |
   | T64 D=67108864 r2 | 453 (7.6) | 404 | 475.6 | GREEN (cand -11%) |
   | T96 D=67108864 r2 | 408 (7.9) | 372 | 428.4 | GREEN (cand -8.8%) |
   | T96 D=268435456 r2 | 1583 (10.1) | 1444 | 1662.2 | GREEN (cand -8.8%) |
   | T96 D=268435456 r4 | 2892 (73.3) | 2705 | 3036.6 | GREEN (cand -6.5%) |

3. **Gate liveness: GREEN (10/10).** `ninja -C build/asan unit_tests_dbms` on
   v2.1 clean (5-line incremental log, `build/asan/build_t16_asan_v21.log`);
   10 consecutive runs of the 30 s-timeout oracle -> every run rc=0 AND its
   log (`build/asan/t16_liveness_v21_run_{1..10}.log`) contains exactly one
   `[  PASSED  ] 1 test` line. Oracle test not modified; no negative flip
   required.
4. **Gate large: GREEN.** `run_large_fixed_gate.sh build/reldeb/programs/clickhouse
   268435456 4` -> exit 0; `Assertions: PASS` with exact counts
   probe/build/joined = 1073741824/268435456/1073741824; clean summary
   (`wins=1 losses=0 ... errors=0`); `Winner: radix_join (1.429x)`
   (`build/reldeb/test_radix_wave_gatec_d268435456_r4.log`). The cycle-1
   memory failure is repaired on the exact shape that exposed it.
5. **Gate early-term: GREEN (6/6).** `run_early_termination_gate.sh` -> 3/3
   `early_stop` (exit 0, LIMIT row present, ~5 s) and 3/3 `exc` (exit 241,
   `MEMORY_LIMIT_EXCEEDED` reached the client); "all six runs behaved",
   nothing left running (`tmp/radix-wave-deadlock/gatee_*.out`).
6. **Gate stateless: GREEN (5/5).** Scratch server started from the candidate
   binary on ports 9131/8161; serving process identity verified by
   SHA-256(`/proc/607199/exe`) = `cf662d4c...` (the v2.1 candidate hash).
   `CLICKHOUSE_PORT_TCP=9131 CLICKHOUSE_PORT_HTTP=8161 ./tests/clickhouse-test
   -b build/reldeb/programs/clickhouse 04508... 04512` -> "5 tests passed.
   0 tests skipped. 1.65 s elapsed" (`build/reldeb/test_t16_stateless_v21.log`).
   Server stopped cleanly; no clickhouse servers left running.

**Cycle-2 verdict: every gate GREEN on the v2.1 candidate
(`cf662d4c9619c7c26b1a713b57353a9024749bc9ad1d6b70f767d3bbe926bb5a`).**
Unit 2 complete pending independent verification.

## Independent verification and delivery

A fresh verification subagent (doer != grader) re-ran Gate T16 and all six
floors with its own pairings (all GREEN; its raw numbers are in
REPORT_T16.md's verification section), re-ran liveness (10/10), large
(exit 0, exact counts, after voiding one dataset-lock collision), early-term
(6/6, after voiding one attempt contaminated by a foreign binary swap it
caught via `/proc/<pid>/exe` hashing) and stateless (5/5, server identity
hash-verified), confirmed the judge's power to fail, audited the diff for
harmed lifecycle paths (none found), and returned **VERDICT: SHIP**.
Artifacts: `tmp/t16/vfy_*`, `build/asan/vfy_liveness_*.log`.

Delivery note: after this campaign's gates were green, a concurrent agent
session (`tmp/rhj-probe-perf` campaign) switched the shared checkout to its
new branch `radix-join-probe-perf`, committed this campaign's then-uncommitted
remedy and doc snapshots as `b1fa64c7286` (verified byte-identical to
`tmp/t16/remedy_v2_1.patch`), and took over `build/reldeb` for its own
builds. To deliver without disturbing that active session, these commits were
made on `radix-join-bandwidth-model` via a linked git worktree
(`git worktree add`), and the worktree was removed afterwards. No rebase, no
force-push, no deletion of any other session's work.
