# REPORT — WaveJoin cooperative probe replacement

Branch `radix-join-bandwidth-model`; final tree at commit `05949c6f588` plus this report.
All raw evidence paths below are rerunnable on this machine (96-core Neoverse-V2, 370 GB,
single NUMA node; OpenJDK Temurin 17.0.19; tla2tools-1.7.4; clang-22 reldeb build).

## Unit verdicts

| unit | verdict |
| --- | --- |
| 1 — formal contract + evidence import | **GREEN**; independent verifier round 1: **SHIP** |
| 2 — independent design + failing tests | **GREEN** (sealed before the implementation was opened; red-test evidence recorded) |
| 3 — replacement implementation | **GREEN** (all contract tests pass; old coordination removed) |
| 4 — gates | **PARTIAL**: gates 1–5, 7, 8 (amended), 11 GREEN; gate 6 **SKIPPED-BY-USER**; gates 9–10 **UNSETTLED by user ruling** (structural evidence below) |
| final independent verification | see the appended verdict section |

**The task is NOT DONE by its own definition**: the performance gates are not green.
This is reported as UNSETTLED per the user's explicit ruling, not softened into success.

## Assumptions, blockers, rulings, deviations

1. **Gate 6 (ASan)**: skipped at the user's explicit mid-run instruction
   ("Skip asan build gate"). No sanitizer evidence exists for the new code.
2. **Gate 8 register amendment v2** (user sign-off): the frozen −25 %
   probe-control-line criterion was structurally unreachable because the old
   design outsourced its probe data plane to build-shared scatter machinery
   while the cooperative contract requires it as claimable jobs on executor
   lanes. Amended: ≤ 115 % of baseline lines AND ≥ 25 % fewer
   synchronization-primitive declarations. Both the original red result
   (712 vs limit 467) and the amendment are recorded in WORKLOG.md.
3. **Gates 9/10 ruling (UNSETTLED)**: the first suite run proved a structural
   asymmetry — the OLD design runs its wave scatter and probe on a dedicated
   radix pool of `max_threads` threads ON TOP of the `max_threads` executor
   lanes (≈ 2× compute at T16; oversubscribed 192→96 at T96), which the
   cooperative contract forbids the candidate to use. `A_T1` parity proves
   per-row-work parity. The user declined the offered contract amendments
   (two-wave pipeline; pool participation in data-plane stages) and ruled:
   report UNSETTLED with everything exactly as frozen.
4. **Prereg-ordering limitation (Unit 1 only)**: prereg text and artifacts
   landed in one commit; from Unit 2 on, preregs were committed before the
   governed work (verifier finding, adopted).
5. **A real deadlock was found by Gate 5 and fixed** (first run: 04509 hung
   600 s; root cause: a failed seal CAS overwrote the waiter's control-word
   snapshot with the terminal state; fix `a02a024c904`; gdb stacks archived).
   The synthetic regression test could not hit the nanosecond window pre-fix;
   the hang itself plus stacks are the red evidence. Honest limitation: the
   formal model could not catch this class — it lives below the spec's
   atomic-action granularity.
6. **Mission/requirements tension surfaced**: the mission sentence
   ("alternate between filling the one active wave and draining the one
   sealed wave") admits a two-wave-slot reading that would restore the
   refill/drain overlap; the requirements pinned "exactly one shared wave".
   The user kept the one-wave contract.

## What was delivered (all committed)

- Corrected `src/Interpreters/RadixHashJoin/WaveJoinProbe.tla`: worker-local
  state and worker-parameterized transitions (F1), honest budget accounting
  with bounded overshoot (F5), `FinalRefinement` checked by TLC (F6), full
  failure/cancellation/EOF protocol, per-worker outputs, plus 5 positive TLC
  configs, 1 liveness witness, 2 mutation witnesses, 3 reachability
  witnesses, wrapped by `verify_tla.sh` (exit 0 only when positives pass AND
  witnesses fail as expected).
- Sealed `INDEPENDENT_DESIGN.md` (committed before the old implementation
  was first opened) and the replacement implementation: one shared wave,
  packed-word admission (one CAS = budget check + reservation + in-flight
  count), packed-word claims bound to phase+generation, last-finisher
  barriers, per-worker output merging, thin delayed-blocks adapter over the
  same machine, first-exception-wins poisoning with fail-close abandonment.
- Six deterministic contract tests (multiset exactness, cooperative help —
  the preregistered red test, multi-wave + final partial wave, multi-pass
  refinement, concurrent delayed pulls termination, abandonment poisoning).
- Frozen-before-change `RadixHashJoin.before.cpp`, `complexity_gate.py`
  (with the signed amendment), `candidate_verdict.py` (frozen and
  null-tested before any benchmark ran), `WORKLOG.md`, and all raw evidence
  under `tmp/wave-join-cooperative/` and `build/reldeb/*.log`.

## TLC state counts (final green run, `build/reldeb/test_wave_join_tla.log`)

| config | states generated / distinct | checked |
| --- | --- | --- |
| MC_Normal | 26,339 / 11,832 | 19 invariants incl. FinalRefinement; Termination; ParticipationLive; PrimaryStable |
| MC_PL1 | 219 / 142 | same battery |
| MC_MultiWave | 119,576 / 41,231 | same battery (duplicate-result multiset) |
| MC_Fail | 2,697 / 1,152 | same battery + TerminationFail (two racing distinct-error faults) |
| MC_CancelRace | 27,087 / 10,944 | same battery (cancel races completion) |
| MC_NoSteal (witness) | 143 / 122 | EXPECTED temporal violation of ParticipationLive |
| mutation witnesses | — | last-exception-wins mutant fails PrimaryStable; budget-ignoring mutant fails MemBound |
| reachability | — | full ownership, two in-flight reservations, crossed-with-inflight all reachable |

## Structural before/after (frozen definitions; amended thresholds)

| metric | baseline | candidate |
| --- | --- | --- |
| probe-control NCNB | 623 | 714 (+14.6 %, amended limit 716) |
| synchronization-primitive declarations | 20 | 11 (−45 %, amended limit 15) |
| coordination machinery | 3 classes + bounded queue + capped drain crew + coordinator | 1 engine, 1 result type, 18-line delayed adapter |
| known baseline hang class (lock across scheduler steps) | present | eliminated (no lock held across quanta) |

## Performance (gates 9/10): UNSETTLED

Diagnostic A-sweep plus a C_T96 oracle pair (raw:
`evidence/diag_u3_prebatch_Asweep.{jsonl,log}`, `evidence/diag_C_oracles.{jsonl,log}`);
the full frozen 10-cell acceptance run was aborted by design once the
structural cause was proven (user ruling):

| cell | candidate | frozen baseline | band | goal (beat by > band) |
| --- | ---: | ---: | ---: | --- |
| A_T96 | 1.485 | 1.343 | 1.37 % | FAIL (+10.6 %) |
| A_T64 | 1.63 | 1.368 | 1.83 % | FAIL (+19 %) |
| A_T32 | 2.55 | 2.179 | 2.18 % | FAIL (+17 %) |
| A_T16 | 4.239 | 3.289 | 1.07 % | FAIL (+28.9 %) |
| A_T1 (2 pairs) | 61.43/62.23 | 61.678 | 1.08 % | FAIL (parity, not −1.08 %) |
| C_T96 (1 pair) | 11.06 | 7.835 | 1.17 % | FAIL (+41 % single pair; note: the in-run baseline arm read 8.281, +5.7 % off the frozen median — ENV-DRIFT territory under the frozen rules; diagnostic-only, and the candidate is +33.5 % even against the drifted arm) |
| C_T64/T32/T16/T1 | not run | — | — | UNSETTLED (no data) |

Scaling: candidate A T16→T96 ratio ≈ 2.86 vs baseline 2.45 — but the
comparison is confounded by the baseline's hidden pool threads at every T.

Correctness at scale is green in the same runs: engagement exact on every
run (A 16384, C 32768 leaves), counts exact (C: 1,073,741,824 joined rows),
radix-vs-parallel_hash fingerprints equal, candidate-vs-baseline radix
fingerprints equal, /mnt/data integrity snapshots identical, binaries stable.

What would settle gates 9/10, if ever resumed: either the two-wave-pipeline
contract amendment (restores refill/drain overlap without pool, queue, or
crews) or a re-baselined register that normalizes the thread budget; then a
full frozen suite run plus `candidate_verdict.py`.

## Evidence matrix

| # | criterion | rerunnable command | raw result | verdict |
| --- | --- | --- | --- | --- |
| 1 | baseline identity | `test "$(sha256sum tmp/wave-join-cooperative/baseline/clickhouse \| awk '{print $1}')" = "4b55481c…dce"` | exit 0 | PASS |
| 2 | formal | `bash tmp/wave-join-cooperative/verify_tla.sh` | exit 0; 13/13 expected behaviors | PASS |
| 3 | reldeb build | `ninja -C build/reldeb clickhouse unit_tests_dbms` | exit 0 (`build_wave_join_final.log`) | PASS |
| 4 | focused tests | `build/reldeb/src/unit_tests_dbms --gtest_filter='RadixHashJoin.*'` | 6/6 (`test_wave_join_cooperative_gtest.log`) | PASS |
| 5 | stateless | `tests/clickhouse-test -b build/reldeb/programs/clickhouse '045(08\|09\|10\|11\|12)_radix_join'` (scratch server, ports 9131/8161) | 5/5 in 1.71 s (`test_wave_join_cooperative_stateless.log`) | PASS |
| 6 | ASan | — | — | SKIPPED-BY-USER |
| 7 | removal | `! rg -n 'ActiveWave\|WaveJoinResult\|ConcurrentBoundedQueue\|output_queue\|max_consumers\|attached_wave\|consumers' src/Interpreters/RadixHashJoin/RadixHashJoin.cpp` | exit 0 | PASS |
| 8 | structural (amended v2) | `python3 tmp/wave-join-cooperative/complexity_gate.py --baseline tmp/wave-join-cooperative/RadixHashJoin.before.cpp --candidate src/Interpreters/RadixHashJoin/RadixHashJoin.cpp` | PASS (714 ≤ 716; 11 ≤ 15) | PASS |
| 9 | paired suite | frozen suite invocation (WORKLOG prereg) | first run RED at A cells; aborted after root cause proven; ruling: UNSETTLED | UNSETTLED |
| 10 | verdict | `python3 tmp/wave-join-cooperative/candidate_verdict.py --baseline tmp/wave-join-impl/baseline_u0.jsonl --candidate …` | not run on acceptance data (none exists) | UNSETTLED |
| 11 | hygiene | `git diff --check` | exit 0 | PASS |
| — | red-test evidence (Unit 2) | `build/reldeb/test_wave_join_red_before.log` | EXPECT failure for the intended reason, 18 ms | PASS |
| — | deadlock incident | `evidence/deadlock_04509_gdb_bt.txt` + fix `a02a024c904` + 04509 re-runs | fixed, 6× green | PASS |

## Final independent verification (fresh context, refutation mandate)

**Verdict: SHIP**, judged against the delivered scope as ruled (gates 9/10
UNSETTLED per the user; gate 6 skipped per the user). The verifier re-ran
every gate itself, recomputed all perf medians/deltas/bands from the raw
JSONL (they match), audited the TLA properties for tautologies and the gate
script for counting tricks (none), confirmed the prereg/design/before-capture
commit ordering, and — decisively — REPRODUCED the fixed deadlock by locally
reverting the fix: a 4-client concurrent radix-join hammer wedged a normally
0.4 s query for 298 s parked in `ProbeWave::tailOrWait` with exactly the
archived pre-fix stack, then restored the tree byte-identically (rebuilt
binary bit-for-bit equal to the delivered one, 6/6 tests green). Repro
artifacts: `tmp/wave-join-cooperative/verifier-final/`. Its findings:
one LOW (this report and the C-oracle evidence were uncommitted — fixed in
the closing commit) and four INFO notes (a rounding nit and the C-oracle
drift note, both folded in above; the regression-test gap and the stale
configure-time version stamp, both already disclosed).

## Failures and nulls retained (nothing laundered)

- Gate 9 first run: RED (full table above; raw JSONL archived).
- Gate 8 original criterion: RED (712 vs 467) before the signed amendment.
- Gate 5 first run: RED (deadlock; 600 s timeout) before the fix.
- First TLA witness run: deadlock detector masked the liveness counterexample
  (fixed with `-deadlock` on witness runs only).
- Unit-2 red test: red for the preregistered reason before the implementation.
- Scatter-claim batching: implemented, measured (no effect on the
  thread-asymmetry-dominated cells), and dropped.
- The synthetic deadlock-regression test did not reproduce the race pre-fix
  (window too narrow at gtest scale); kept as a broad termination guard.
