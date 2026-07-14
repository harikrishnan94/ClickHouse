# REPORT_T16 — the T16 `radix_join` regression: attributed and recovered

Branch `radix-join-bandwidth-model`, mission start at tip `b8557c1f56b`.
Full raw data and per-iteration pre-registrations: `WORKLOG_T16.md`.

## Verdicts

- **Unit 1 (attribution): SETTLED.** The T16 cost is unbounded, sticky
  wave-consumership starving the probe-side scan (details below), confirmed by
  discriminating instrumented probes; the bimodality is the capture race.
- **Unit 2 (remedy): GREEN.** A bounded consumer set per wave with
  staged-window back-pressure (`max_consumers = max(2, threads/8)` + attach
  past the cap once the shared window reaches the budget) recovers Gate T16
  within the PRE-FIX band and passes every floor and correctness gate.
  Cycle 1 of the remedy (cap without the back-pressure) was **rejected red**
  at Gate large (`MEMORY_LIMIT_EXCEEDED`) and repaired in cycle 2; both
  cycles are fully documented in the worklog.
- **Independent verification:** see the bottom of this report.

**Flags / deviations (none hidden):**
- The first instrumented probe batch was run with wrong SQL
  (`bucket_width=1` instead of the dataset's `4194304`) — discarded as a null
  result, archived in `tmp/t16/invalid_fullscan/`, and the query generation
  switched to the harness's own metadata path.
- The instrumented cycle-1 remedy binary carried two per-wave dump sites
  (patch hunk + manual edit); analysis dedupes by wave id. No effect on any
  gate (gates run clean binaries only).
- Cycle-1 timing gates (T16, floors, liveness) were green but are VOID —
  superseded by the fresh cycle-2 runs on the final binary; their logs are
  archived in `tmp/t16/cycle1/`.
- Pre-registration ordering is evidenced by worklog structure and session
  history; the work was intentionally left uncommitted until all gates were
  green, so there are no per-step commit timestamps.

## The attributed mechanism (Unit 1, MATERIAL)

The wave-deadlock fix made wave consumership **cooperative, sticky and
unbounded**: any probe lane whose `joinBlock` lands while the shared window is
at budget and a wave is active becomes an attached queue consumer, pulls no
further input, and stays attached across chained waves until it observes
no-wave-and-below-budget. At T16 this capture usually saturates all 16 lanes
(the window re-crosses the budget mid-wave, converting every lane), so the
following wave runs with **zero scanning lanes**, ends with an empty window,
and a full window refill (~610 MB at ~5.3 GB/s ≈ 116 ms) sits exposed on the
critical path — matching the measured +112 ms. The bimodality is the capture
race: in the rare fast run only ~10 lanes were captured, 6 kept scanning, and
the gap collapsed to 17 ms.

Discriminating evidence (instrumented builds, within-binary regime
comparison, `WORKLOG_T16.md` Iteration 2): slow runs show `att=16,16,1`,
wave-2 window-at-end = 0 MB, gap ≈ 116 ms; the fast run shows `att=10,10,1`,
window-at-end 520 MB, gap 17 ms. Producer park time is anti-correlated with
wall (kills the queue-capacity hypothesis), scatter time is flat (kills
scatter serialization), teardown-loser quanta are zero (kills the loser-spin
hypothesis), wave count and flush share are identical (kills the wave-count
hypothesis). T96 probes (Iteration 3) show the same capture pattern and
quantified the drain: one consumer pops ~55k blocks/s versus a producer-side
demand of ~136k blocks/s — so the fix's T96 win needs only ~3 consumers, not
96, which is what makes a small cap safe.

## The remedy (Unit 2)

`src/Interpreters/RadixHashJoin/RadixHashJoin.cpp` only (+74/−6):

- `ActiveWave` gains an attached-consumer counter; `WaveJoinResult` gains
  `max_consumers = max(2, threads/8)`, a CAS attach, and a detach (also in the
  destructor — a plain atomic decrement, still inert).
- A capped-out lane returns `is_last` (the existing "another wave took my
  rows" semantics) and goes back to pulling input **while the shared window is
  below budget**; once the window is staged to budget it attaches **past the
  cap**, stopping its input pull exactly like the shipped design — restoring
  the shipped memory bound (window overshoot ≤ one in-flight block per lane).
  Over-cap consumers dissolve at the next wave switch (they fail the attach on
  the new wave, find its window freshly swapped below budget, and resume
  scanning). Governing invariant: **a surplus lane scans iff the shared window
  is below budget.**
- Liveness invariant preserved: no new waits anywhere; a live wave always has
  ≥ 1 attached consumer (slots only read full with `max_consumers ≥ 2`
  attached, and an attached transform cannot finish while its wave lives,
  which also preserves the delayed-flush no-active-wave invariant);
  cancellation paths unchanged (abandoned waves still reaped by the
  destructor).

Why it works: the guaranteed crew of 2 (T16) drains at ~110k blocks/s against
a ~52k blocks/s producer demand, while ≥ 14 lanes keep the scan at its
saturated ~5.3 GB/s; waves chain either gap-free (window staged) or with only
the residual 13-45 ms tail. At T96 the crew of 12 provides ~4.8× the measured
drain demand — and the floors show the remedy is *faster* than SHIPPED at
every thread count ≥ 32 because the same starvation gap existed there too.

## Evidence matrix (all cycle-2, candidate binary `cf662d4c9619...bb5a`)

| Criterion | Gate invocation (command) | Result (raw) | Verdict |
| --- | --- | --- | --- |
| T16 recovery | 5 paired position-balanced runs/binary, P C C P P C C P P C: `python3 bep/tools/join_mergetree_bench.py run --path /mnt/data/join_bench_data --binary <bin> --cardinalities 67108864 --multiplicities 1 --ratios 2 --hit-rates 1 --build-payload-columns 1 --probe-payload-columns 1 --threads 16 --runs 1 --no-verify --max-memory 100000000000`; judged by `tmp/t16/judge.py` | PRE-FIX 894/909/886/911/916 (median 909, stdev 12.6); candidate 924/911/902/942/927 (median 924); band max(45.5, 12.6)=45.5, threshold 954.5 | GREEN |
| Floor T1 | same, `--threads 1`, candidate vs SHIPPED | SHIPPED median 14329 (stdev 486.3); candidate 14443; threshold 15045.5 | GREEN |
| Floor T32 | same, `--threads 32` | SHIPPED 702 (8.4); candidate 669; threshold 737.1 | GREEN |
| Floor T64 | same, `--threads 64` | SHIPPED 453 (7.6); candidate 404; threshold 475.6 | GREEN |
| Floor T96 | same, `--threads 96` | SHIPPED 408 (7.9); candidate 372; threshold 428.4 | GREEN |
| Floor D=268435456 r=2 T96 | same, `--cardinalities 268435456 --ratios 2 --threads 96` | SHIPPED 1583 (10.1); candidate 1444; threshold 1662.2 | GREEN |
| Floor D=268435456 r=4 T96 | same, `--cardinalities 268435456 --ratios 4 --threads 96` | SHIPPED 2892 (73.3); candidate 2705; threshold 3036.6 | GREEN |
| Liveness oracle | `ninja -C build/asan unit_tests_dbms`, then 10× `timeout --signal=TERM --kill-after=10s 30s build/asan/src/unit_tests_dbms --gtest_filter=RadixHashJoin.ConcurrentJoiningQuantumDoesNotWaitForPreviousWave` | 10/10 rc=0; each log exactly one `[  PASSED  ] 1 test` (`build/asan/t16_liveness_v21_run_{1..10}.log`) | GREEN |
| Large shape completes | `tmp/radix-wave-deadlock/run_large_fixed_gate.sh build/reldeb/programs/clickhouse 268435456 4` | exit 0; `Assertions: PASS`; probe/build/joined = 1073741824/268435456/1073741824; `errors=0`; `Winner: radix_join (1.429x)` | GREEN |
| Early termination | `tmp/radix-wave-deadlock/run_early_termination_gate.sh build/reldeb/programs/clickhouse` | 6/6: 3× `early_stop` PASS (exit 0, LIMIT row), 3× `exc` PASS (exit 241, `MEMORY_LIMIT_EXCEEDED` at client); nothing left running | GREEN |
| Stateless tests | `CLICKHOUSE_PORT_TCP=9131 CLICKHOUSE_PORT_HTTP=8161 ./tests/clickhouse-test -b build/reldeb/programs/clickhouse 04508_radix_join_gate_and_fallback 04509_radix_join_distinct_estimate 04510_radix_join_payload_gate 04511_radix_join_multi_pass 04512_radix_join_wide_fixed_types` (server binary verified: SHA-256(`/proc/607199/exe`) = candidate hash) | `5 tests passed. 0 tests skipped. 1.65 s elapsed` | GREEN |

Reference binaries, measured fresh in-session: PRE-FIX
`tmp/radix-wave-deadlock/bin/perf-baseline/clickhouse`
(`f97d41279797aed3623bcd9d703286019e9cd3d1c7257a7a6213a089790a26b3`, matches
the mission pin); SHIPPED `tmp/t16/bin/shipped/clickhouse`
(`148ee743a09f4031c2e3cfbfe50448fd7d522e3b53f6792032c68bd687497a3a`, built
from the clean tip and preserved before the first product edit). Candidate:
`build/reldeb/programs/clickhouse`
(`cf662d4c9619c7c26b1a713b57353a9024749bc9ad1d6b70f767d3bbe926bb5a`).

## Independent verification

(pending — appended below when complete)
