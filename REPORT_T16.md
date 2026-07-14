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
- **Independent verification: VERDICT SHIP** — a fresh subagent (genuine
  doer/grader separation, not self-verified) re-ran Gate T16 and all six
  floors with its own pairings, re-ran every correctness gate against a
  hash-pinned candidate copy, confirmed the judge's power to fail, and found
  no lifecycle hazard in the diff. Full report at the bottom.

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
- **Concurrent-session interleaving (after all doer gates were green):** a
  third agent session running its own RadixHashJoin probe campaign
  (`tmp/rhj-probe-perf`) switched the shared checkout to a new branch
  `radix-join-probe-perf`, committed this campaign's then-uncommitted remedy
  and doc snapshots as `b1fa64c7286` ("adopt the T16 bounded-consumer
  remedy") — verified byte-identical to `tmp/t16/remedy_v2_1.patch` — and
  replaced `build/reldeb/programs/clickhouse` with its own instrumented
  build. The independent verifier caught the swap via the mandated
  `/proc/<pid>/exe` hash checks, voided and re-ran two contaminated gate
  attempts against a hash-pinned candidate copy, and all timing batches
  predate the interference. These deliverables were therefore committed to
  `radix-join-bandwidth-model` through a linked git worktree, leaving the
  foreign session's active checkout untouched. Nothing was force-pushed,
  rebased, or deleted.

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

## Independent verification — VERDICT: SHIP

Performed by a fresh subagent that did not write the change, given the mission
contract, the diff, the worklog/report, and the artifact paths; it re-measured
everything itself. Its report, verbatim:

> ### 1. Static integrity — PASS
>
> - Hashes (verified myself): PRE-FIX = `f97d4127...90a26b3` OK; SHIPPED =
>   `148ee743...` OK; CANDIDATE = `cf662d4c...926bb5a` OK.
> - Diff: +74/−6 in `src/Interpreters/RadixHashJoin/RadixHashJoin.cpp` only;
>   byte-identical to `tmp/t16/remedy_v2_1.patch`; only untracked files were
>   `WORKLOG_T16.md`/`REPORT_T16.md`.
> - Rebuild determinism: `ninja -C build/reldeb clickhouse` -> "no work to
>   do", hash unchanged.
> - Environment deviation (not the doer's fault, fully contained): a third
>   session committed the remedy as `b1fa64c7286` and replaced the reldeb
>   binary with its instrumented build. I verified the adopted commit is
>   byte-identical to the remedy patch, took a private hash-verified
>   candidate copy (`tmp/t16/vfy_bin/clickhouse`), and ran all subsequent
>   gates against it. All my timing batches predate the swap.
>
> ### 2. Code refutation hunt — no hazard found
>
> Traced cancellation mid-wave (destructor detach is a lock-free decrement on
> a shared_ptr-held wave; abandoned-wave reaping unchanged), worker exception
> (teardown winner rethrows; a live wave always has >= 1 attached consumer),
> partial scheduling failure (unchanged; stale attachment released by the
> destructor), the delayed-flush invariant (attached lanes cannot finish
> while the wave lives; a capped-out lane's early `is_last` is the
> pre-existing below-budget semantics and its rows are already in the shared
> window), back-pressure (the shared window shrinks only via the two swaps,
> so the over-cap attach decision cannot be invalidated; overshoot bounded at
> one in-flight block per lane), and liveness (no new waits; lock order
> `coordinator.mutex -> window_mutex` only; relaxed atomics are pure
> counters; the unmodified liveness gtest at `max_threads=2` -> cap 2 still
> exercises the original deadlock scenario).
>
> ### 3. Gate T16 fresh — GREEN
>
> Idle host verified. PRE-FIX 906/895/897/892/895 (median 895, stdev 5.3);
> candidate 907/934/937/919/896 (median 919); threshold 939.8
> (`tmp/t16/vfy_t16_*.log`). Judge power to fail confirmed: RED (exit 1) on
> the doer's original regression data. Bimodality eliminated: no ~1000 ms
> sample anywhere.
>
> ### 4. Six floor cells fresh — all GREEN
>
> 60/60 invocations rc=0, no foreign overlap (per-batch foreign-process logs
> clean; batches ended before the foreign campaign's first bench).
>
> | cell | SHIPPED median (stdev) | cand median | threshold | verdict |
> | --- | --- | --- | --- | --- |
> | T1 | 14216 (104.7) | 14217 | 14926.8 | GREEN |
> | T32 | 695 (8.6) | 666 | 729.8 | GREEN (−4.2%) |
> | T64 | 453 (12.8) | 413 | 475.6 | GREEN (−8.8%) |
> | T96 | 413 (3.1) | 372 | 433.6 | GREEN (−9.9%) |
> | D=268435456 r2 T96 | 1620 (18.3) | 1427 | 1701.0 | GREEN (−11.9%) |
> | D=268435456 r4 T96 | 2884 (28.8) | 2690 | 3028.2 | GREEN (−6.7%) |
>
> ### 5. Correctness gates — all GREEN
>
> - Liveness: ASan rebuild from the remedy source; 10/10 runs rc=0, each log
>   with exactly one `[  PASSED  ] 1 test` (`build/asan/vfy_liveness_*.log`).
> - Large: attempt 1 VOID — dataset status-lock collision with the foreign
>   campaign (`Cannot lock file .../status`, nothing about `radix_join`);
>   attempt 2 in a clean window: exit 0, `Assertions: PASS`,
>   probe/build/joined = 1073741824/268435456/1073741824, `errors=0`,
>   `Winner: radix_join (1.545x)` (`tmp/t16/vfy_large_gate_try2.out`).
> - Early-term: first run VOID — the foreign session had swapped the reldeb
>   binary (caught via the `/proc/<pid>/exe` hash check); re-run on the
>   verified copy: 6/6 (`tmp/t16/vfy_earlyterm_gate2.out`).
> - Stateless: serving process verified `sha256(/proc/<pid>/exe)` =
>   candidate hash; 5/5 passed, 1.65 s (`tmp/t16/vfy_stateless.log`); server
>   stopped cleanly.
>
> ### 6. Process checks — clean, with the disclosed caveat
>
> Pre-registrations precede results throughout the worklog; the cycle-1 red
> at Gate large is documented as a rejection with the full suite re-run
> fresh on v2.1; no banned moves found; the doer's raw gate logs reproduce
> every claimed number. Caveat, noted honestly: pre-registration ordering
> rests on document structure (the work was uncommitted by design).
>
> ### Discrepancies with the doer's claims
>
> None found in the doer's own work. All discrepancies encountered were
> caused by a concurrent third session after the doer finished and were
> contained via hash-pinned binary copies.
>
> VERDICT: SHIP
