# WORKLOG — WaveJoin cooperative probe replacement

Branch: `radix-join-bandwidth-model` (verified via `git branch --show-current`).
Clean-room rules in force: `wave-join-impl` is evidence-only (10 allowed artifacts);
`RadixHashJoin.cpp` stays unopened until `INDEPENDENT_DESIGN.md` is sealed and committed.
The performance `PREREG.md` (tmp/wave-join-impl/PREREG.md) is immutable.

Environment identity:
- CPU: (recorded before first benchmark run in Unit 4)
- Java: OpenJDK Temurin 17.0.19+10 (`~/.local/bin/java`)
- TLA+ tools: `tmp/tla-install/tla2tools-1.7.4.jar` (SANY + TLC)
- Frozen baseline binary: `tmp/wave-join-cooperative/baseline/clickhouse`,
  sha256 `4b55481c22d025ae364d36df39cd662bd986fd5878e711d89e1d76b08ea59cce` (verified 2026-07-16)

---

## Unit 1 — Formal contract and evidence import (preregistered 2026-07-16, BEFORE the repair)

### Evidence import (done, verified)

Imported from `wave-join-impl:tmp/wave-join-impl/` exactly the 10 allowed artifacts
(PREREG.md, baseline_u0.jsonl, baseline_u0.log, binary_sha_before_ninja.txt,
integrity_S0.txt, ninja_noop_provenance.log, report_table.py, suite.py, sql/ (16 files),
sql_sha256.txt). Every file's `git hash-object` equals the branch blob id (25/25 OK);
`sha256sum -c tmp/wave-join-impl/sql_sha256.txt` from repo root: 16/16 OK.
Forbidden artifacts (PLAN.md, WORKLOG.md, any implementation work) were NOT read.

Baseline preserved: `cp --reflink=auto build/reldeb/programs/clickhouse
tmp/wave-join-cooperative/baseline/clickhouse`; sha256 of both source and copy equals
the frozen `4b55481c…` (Gate 1 green before any candidate build).

### Defect confirmation against the on-disk spec (755-line WaveJoinProbe.tla @ af54d1acc00)

- F1 CONFIRMED: `BusyWorkers`, `ScanWorkers`, `DrainWorkers` are all derived from
  `st.phase`, `st.running` cardinality and `Take(...)`; `CooperativeParticipation`
  (line 659) is definitionally true. Jobs in `st.running` carry no worker identity;
  `Fill`/`Take` model a central dispatcher.
- F5 CONFIRMED: `MemoryBound` (line 683) only forbids liveEntries and liveA1
  being simultaneously non-empty; `st.mem` is never bounded; no reservation modeling.
- F6 CONFIRMED: `FinalRefinement` exists only as an unchecked THEOREM; there are no
  TLC configuration files anywhere in the repository.
- Additional defect (recorded): `FinishProbe` emits `ProbeResult(l, ExpectedLeaf(...))`,
  i.e. the output is definitionally the expected output — `OutputRefinement` and
  `FinalRefinement` cannot fail on the output path. The corrected spec must compute
  emitted output from the actual scattered arena contents.

### Preregistered expected outcome of the repair

A corrected `src/Interpreters/RadixHashJoin/WaveJoinProbe.tla` plus TLC configs under
`tmp/wave-join-cooperative/tla/` and wrapper `tmp/wave-join-cooperative/verify_tla.sh`
such that:

1. SANY parses the corrected module with zero errors.
2. TLC succeeds (no invariant/temporal violation) on at least these configurations:
   - `normal` (PL>1, budget-sealed wave + EOF final partial wave),
   - `pl1` (PL=1, refine stages skipped by the same machine),
   - `multiwave` (>= 2 budget-sealed waves),
   - `fail` (injected scan fault, non-probe fault, probe fault via FailLeaves,
     external cancellation) — terminal `failed` with first-exception-wins and
     exactly-once release.
3. At least one SUCCESSFUL config explicitly lists `FinalRefinement` (multiset
   equality of the union of per-worker outputs vs the row-wise expected bag) as a
   checked property (F6).
4. The memory-bound invariants are checked: `mem = SumBytes(queue) + inflight
   reservations` (accounting cross-check), `mem <= BUDGET + MaxBlockBytes`
   (bounded overshoot), `crossed <=> mem >= BUDGET` (atomic-crossing seal),
   with spec comments stating explicitly that BUDGET does NOT bound total resident
   memory (drain arenas, hash columns, per-worker input, allocator overhead, output
   are outside it) (F5).
5. Ownership invariants are checked: per-worker state `wk[w]` (worker side) is
   cross-checked against phase/done-sets (work side); distinct workers own distinct
   jobs; owned jobs are legal for the phase and not done (F1).
6. Participation is non-vacuous: a liveness property (`ParticipationLive`) states
   that an idle worker with claimable compatible work eventually claims or the work
   is consumed, under WF of each worker's step action (stated fairness); AND
7. The negative witness: a broken variant (dedicated scanner crew + leaf affinity —
   the producer/consumer anti-pattern) makes TLC report a violation
   (expected counterexample). A second expected-fail run shows full cooperative
   ownership is reachable (invariant `OwnedCount < WORKERS` violated), proving the
   ownership state space is actually exercised.
8. `bash tmp/wave-join-cooperative/verify_tla.sh` exits 0 only when the positive
   configs all pass AND the negative witnesses fail in exactly the expected way.

### Refuted by

- SANY/TLC parse errors that cannot be fixed without weakening a property.
- Any positive config failing its invariants/temporal checks.
- The negative witness NOT producing a violation (would mean the participation
  property is still vacuous).
- State-space explosion making the required configs infeasible (> ~1e8 states);
  would be reported as UNSETTLED with the exact config, not silently downsized
  below meaningfulness.
- Independent verifier verdict of REWORK on the formal contract.

### Gates for Unit 1

- Gate 1 (baseline identity): GREEN (recorded above, rerunnable:
  `test "$(sha256sum tmp/wave-join-cooperative/baseline/clickhouse | awk '{print $1}')" = "4b55481c22d025ae364d36df39cd662bd986fd5878e711d89e1d76b08ea59cce"`).
- Gate 2 (formal): `bash tmp/wave-join-cooperative/verify_tla.sh > build/reldeb/test_wave_join_tla.log 2>&1` exits 0.
- Independent verifier verdict: SHIP required before Unit 2.

### Gate 2 result (2026-07-16): GREEN

`bash tmp/wave-join-cooperative/verify_tla.sh` exit 0; raw log
`build/reldeb/test_wave_join_tla.log`. Tool identity: OpenJDK Temurin 17.0.19+10,
tla2tools-1.7.4 (SANY2 2.1, TLC in the same jar).

| run | expectation | outcome | states generated / distinct |
| --- | --- | --- | --- |
| SANY WaveJoinProbe.tla | clean parse | PASS | — |
| MC_Normal (PL=2, budget wave + EOF partial) | no error | PASS | 26,339 / 11,832 |
| MC_PL1 (PL=1, transfer path, EOF empty) | no error | PASS | 219 / 142 |
| MC_MultiWave (2 waves, 3 workers, dup results) | no error | PASS | 119,576 / 41,231 |
| MC_Fail (probe fault + scan/pre faults + cancel) | no error | PASS | 4,385 / 1,852 |
| MC_NoSteal (dedicated crew + leaf affinity) | temporal violation | PASS (violated as expected) | 143 / 122 |
| MC_ReachOwn | invariant NeverFullOwnership violated | PASS (violated) | — |
| MC_ReachInflight | invariant NeverTwoInflight violated | PASS (violated) | — |
| MC_ReachCross | invariant NeverCrossWithInflight violated | PASS (violated) | — |

All completing configurations check `FinalRefinement` as an explicit TLC invariant
(F6), the full safety battery (incl. `MemAccounted`/`MemBound`/`CrossedSound` for F5
and `OwnershipConsistent`/`RaceFree` for F1), plus `Termination`, `ParticipationLive`
and the `PrimaryStable` action property under the stated per-worker weak fairness.

Iteration note (refutation honored, not weakened): the first witness run did NOT
report a temporal violation — TLC's deadlock detector fired first, because the
broken eligibility wedges the drain into a successor-less state. Fix: the witness
run (only) passes `-deadlock` to disable the deadlock detector, so stuttering is
allowed and `ParticipationLive` itself produces the counterexample. Positive runs
keep deadlock checking on. First run's evidence retained in shell history; final
log is the gate log above.

Decision record (reversible, in scope): probe/refine outputs in the corrected spec
are computed from the ACTUAL arena contents (not `ExpectedLeaf`), so a modeled
scatter/refine defect propagates to a `FinalRefinement` violation; the old spec's
definitional output made refinement unfalsifiable on the output path.

### Adversarial panel round 1 (2026-07-16): 3 confirmed findings, all fixed

Workflow `wf_c4e003a2-de5`: 4 independent review lenses (tautology, budget,
refinement, protocol), 12 raw findings, each adversarially verified by a fresh
agent instructed to refute it; 3 confirmed, 9 refuted (raw results in the
workflow journal). Confirmed and fixed:

1. HIGH — `PrimaryStable` was unfalsifiable: no configuration could reach a
   state where a second fault carries a DIFFERENT error, so a last-exception-
   wins spec would have passed identically (zero mutation adequacy).
   Fix: `MC_Fail` now fails leaves {1, 3} (both populated, distinct errors
   `eL1`/`eOther`, concurrently ownable by the two workers), and
   `verify_tla.sh` gained a MUTATION WITNESS: a scratch copy of the spec with
   FaultProbe's first-wins guard removed must make TLC report a violation of
   `PrimaryStable` — the gate is red if the mutant passes.
2. MEDIUM — leaf-arena release exactly-once was asserted in comments but not
   verifiable (`liveA1` set-removal is idempotent; `FreedOnce` covered only
   entries/pass-arenas). Fix: per-leaf `freedA1 \in [Leaves -> Nat]` counter,
   incremented at FinishProbe/FaultProbe/Teardown-of-still-live, reset at
   CompleteWave, bounded by `FreedOnce`.
3. MEDIUM — cancellation/faults never coexisted with wave completion in any
   configuration (verified by the panel with an exhaustive TLC run: MC_Fail
   never reaches `CompleteWave`, `done`, `doneWaves # <<>>`, or an EOFSeal
   pre-state). Fix: new positive configuration `MC_CancelRace` (two waves,
   one budget-sealed + one EOF partial, external cancellation enabled, no
   faults) makes cancellation race Seal, the barriers, CompleteWave, EOFSeal,
   FinishInput and the second wave's drain with history present.

### Independent verifier round 1 (2026-07-16): verdict SHIP

Fresh independent subagent (not the author), given only the task's binding
requirements, the artifacts and the recorded evidence. It re-ran Gate 1 and
Gate 2 from a clean state (both green, state counts reproduced exactly),
confirmed the old spec's F1/F5/F6 defects directly at `af54d1acc00`, injected
five mutants of its own choosing (scatter row drop, double-claim, budget-check
removal, seal-during-inflight, output drop) — all caught by the battery,
including a `FinalRefinement` violation proving the F6 property is live — and
verified witness honesty (restoring eligibility makes the MC_NoSteal
counterexample disappear; `-deadlock` justified). Full report:
`tmp/wave-join-cooperative/verifier-round1/REPORT.md` (persisted by the session
agent because the subagent harness rejects report-file writes); the verifier's
raw TLC logs are in the same directory. Findings, all Low/Info:

1. Prereg ordering unverifiable (prereg text and artifacts in one commit).
   ACTION ADOPTED: from Unit 2 on, each unit's prereg is committed in its own
   commit BEFORE the work it governs.
2. The budget-check mutant survives 4 of 5 positive configs (bound equals
   total input bytes everywhere but MC_MultiWave). ACTION TAKEN: the mutant is
   now a permanent second mutation witness in `verify_tla.sh` (expect
   `MemBound` violated on MC_MultiWave); gate re-run green (exit 0, 13 checks).
3. `MemAccounted` scoped away from `failed` (defensible; no action).
4. The fairness-to-executor mapping is an assumption the formal stage cannot
   discharge — Units 2-4 must discharge it (JoinResult early destruction must
   release/finish the owned task; lanes/delayed stream keep pulling).
5. Panel round 1's journal was not in the repo. ACTION TAKEN: archived at
   `tmp/wave-join-cooperative/evidence/panel_round1_journal.jsonl`.

### Gate 2 re-run after fixes (2026-07-16): GREEN

`bash tmp/wave-join-cooperative/verify_tla.sh` exit 0, 13/13 checks as
expected; raw log `build/reldeb/test_wave_join_tla.log`. Deltas vs round 1:
MC_Fail now 2,697 generated / 1,152 distinct (earlier fault cutoffs with two
failing leaves), MC_CancelRace 27,087 / 10,944, mutation witness reports
`PrimaryStable` violated as expected; all other counts unchanged.

---

## Unit 2 — Independent design and failing tests (preregistered 2026-07-16, committed BEFORE the unit's work)

Attestation at prereg time: `src/Interpreters/RadixHashJoin/RadixHashJoin.cpp`
has not been opened in this session (clean-room intact). Only the corrected
TLA+ contract, `RadixHashJoin.h`, `IJoin.h`, `JoiningTransform` drive points,
the gtest, the stateless tests and `PlannerJoins.cpp` wiring have been read.

### Expected outcome

1. `tmp/wave-join-cooperative/INDEPENDENT_DESIGN.md` sealing, from the corrected
   TLA+ contract and public interfaces only: the C++ state machine (one shared
   wave; fetch-add Reserve/Admit; decentralized last-finisher barriers over the
   pre -> scatter -> [refalloc -> refine] -> probe job graph with atomic stage
   cursors), ownership rules (claims via cursor fetch-add, one owned task per
   worker, per-task continuation), synchronization (no probe-side pool, no
   output queue, no reorder buffer, no central scheduler; the only blocking
   wait is the bounded sealed-tail/phase-transition wait on wave completion),
   memory accounting (BUDGET = admission threshold, overshoot <= one block,
   arenas/hash columns outside BUDGET), output contract (workers emit their own
   blocks through their own `IJoinResult`/delayed-stream pulls; correctness =
   exact multiset), cancellation (first exception wins via one primary slot;
   cancelled flag stops claims; owners unwind; last participant out performs
   teardown exactly once), cleanup (exactly-once arena/wave release), the
   EOF/delayed-blocks path as a thin adapter running the SAME machine, and the
   discharge of the verifier's fairness-mapping finding (executor keeps calling
   `next` per lane until `is_last`; result destruction with an incomplete owned
   task poisons the wave = cooperative cancel, so no work is silently lost).
   Committed BEFORE the first `RadixHashJoin.cpp` open.
2. Only after that commit: capture `tmp/wave-join-cooperative/RadixHashJoin.before.cpp`
   (byte-identical to HEAD's `src/Interpreters/RadixHashJoin/RadixHashJoin.cpp`,
   recorded hash) and freeze `tmp/wave-join-cooperative/complexity_gate.py`
   (metric definitions inside the script; frozen before any production C++ edit).
3. New deterministic contract tests in `src/Interpreters/tests/gtest_radix_hash_join.cpp`
   (no sleeps; deadlines/futures/failpoints only). At least the cooperative-help
   test must be RED on the unmodified implementation:
   `RadixHashJoin.SealedWaveDrainIsClaimableByOtherLanes` — lane A seals a wave
   and stalls after one `next()`; lane B's `joinBlock` + successive `next()`
   calls must yield at least one NONEMPTY drain output block while A is stalled.
   Prediction for the red run (intended reason): the old implementation routes
   the whole wave's output through the admitting lane's result (bounded shared
   output queue), so lane B returns an empty/is_last result with zero drain
   output rows (or, if the old code instead blocks B, the test's bounded
   deadline records that); either recorded outcome is the intended red. A crash
   or an unrelated assertion is NOT the intended red and stops the unit.
4. The red run happens on a build of `unit_tests_dbms` that contains the new
   test but ZERO production-code changes; log preserved under
   `build/reldeb/test_wave_join_red_before.log` plus a subagent summary.

### Refuted by

- The design turning out to require public API changes (stop and ask user).
- The new contract test passing on the unmodified implementation (would mean
  it does not pin the cooperative contract).
- before.cpp capture differing from HEAD, or complexity_gate.py edited after
  the first production C++ edit.

### Gates for Unit 2

- Design-seal ordering: git history shows INDEPENDENT_DESIGN.md committed
  before any RadixHashJoin.cpp modification (non-openness is attested above;
  ordering of artifacts is machine-checkable).
- Red-test evidence: build log + failing-test log + subagent summaries recorded
  in this WORKLOG with the failure text quoted.

### Unit 2 results (2026-07-16)

1. `INDEPENDENT_DESIGN.md` sealed and committed at `dd4e15b434f`, strictly before
   the first `RadixHashJoin.cpp` open (prereg commit `e8227fee213` precedes it).
2. `RadixHashJoin.before.cpp` captured byte-identical (sha256
   `7e669b72313d53021620067e4010de7445e69670b19e9571c6d7d105cf50f4a3`) and
   `complexity_gate.py` frozen, both committed at `f64a7e45d05` before any
   production C++ edit.
3. Compatibility mapping after the first open (constraints only, no design
   retrofit): the old probe machinery is the forbidden producer/consumer shape
   (pool-worker producers -> `ConcurrentBoundedQueue(2*threads+1)` -> capped
   sticky consumer lanes via `max_consumers`; a separate work-stealing
   `RadixDelayedBlocks` machine for the final wave). Kept per the sealed design
   §10: build-side scatter machinery (`scatterFirstPass`/`scatterRefineGroup`/
   `scatterRefinePass`/`scatterToPartitions`, `SideLayout`, `PartitionOutput`,
   `parallelRun`, plan constants), `probePartition`, profile-event names,
   settings plumbing, constructor gates, pre-build delegation. The scatter
   kernels already support concurrent disjoint-range SWWC writes (barrier 3 of
   `scatterFirstPass` does exactly that), so per-block scatter jobs are safe.
4. Implementation refinements recorded within the sealed ownership model (not
   retrofits): (a) refalloc+refine fuse into one exactly-once job per group —
   a legal scheduling of the TLA model where one worker executes both stages'
   jobs for a pass; the C++ generalizes the model's single refine stage to the
   plan's N-1 refine stages, same barriered machine per stage; (b) workers
   merge THEIR OWN leaf outputs up to
   min(joined-block target, `maxJoinedBlockRows`) before returning — the block
   required for the call to return, never a shared buffer (the old design's
   queue-side merging violated the block cap; the new one respects it);
   (c) `seal_requested` and `cancelled` fold into the phase word (fewer
   primitives, same protocol).
5. Red contract test `RadixHashJoin.SealedWaveDrainIsClaimableByOtherLanes`
   written (test-file sha256 recorded below); first build attempt FAILED with
   two compile errors (initializer-list type conflict; `Block` has no bool
   conversion) — caught by the log-inspection subagent, fixed, rebuilt clean
   (`build/reldeb/build_wave_join_red_test.log`, 3 lines, link OK).
6. RED RUN (before any implementation change), raw log
   `build/reldeb/test_wave_join_red_before.log` (19 lines, quoted):
   `Expected: (outcome.first_block.rows()) > (0u), actual: 0 vs 0 — lane B's
   first quantum produced no drain output while lane A's sealed wave held
   claimable leaves`; exit 1, 18 ms, no hang, no crash — exactly the
   preregistered intended reason (the old design hands lane B an empty result
   while the sealed wave still holds unclaimed leaves).
   The pre-existing `ConcurrentJoiningQuantumDoesNotWaitForPreviousWave`
   still passes on the same binary
   (`build/reldeb/test_wave_join_existing_check.log`).
7. Decision (recorded): the red test is NOT committed alone ("never commit red
   work") — it lands in the Unit-3 commit together with the implementation
   that turns it green. Freeze evidence for the test as written at red time:
   sha256(src/Interpreters/tests/gtest_radix_hash_join.cpp) =
   `d0b2aa9674086e5e39c0c879c8d7cb1b4acbb4c4dce077983a86708037d69298` is the
   PRE-fix hash; the post-fix (compiling) test file hash is recorded at the
   Unit-3 commit. Deviation note: only the two compile errors above were
   fixed between those hashes; the contract assertion is unchanged.
8. Environment note (from project memory, relevant to Unit 3): the old design
   has a known baseline defect — early termination mid-wave with >= 2 probe
   streams can hang (the old result type held a wave mutex across scheduler
   steps). The cooperative design removes that class (no lock is ever held
   across a quantum boundary; abandoned results poison the wave instead).

---

## Unit 3 — Replacement implementation (2026-07-16)

### What was built

The probe coordination in `RadixHashJoin.cpp` is replaced by the sealed
cooperative machine (`ProbeWave` + `CooperativeWaveResult` +
`CooperativeDelayedBlocks`):

- Admission: one packed atomic word `[seal:1 | in-flight:15 | bytes:48]` — the
  budget check, reservation, and in-flight count are a single CAS, so the drain
  can only begin after every granted reservation lands; the crossing admission
  seals; overshoot is bounded by one block.
- Drain: one packed control word `[phase:8 | generation:32 | job index:24]` —
  claiming a job is one CAS bound to the phase and generation it read, so a
  claim can never cross a stage boundary; each stage's last finisher runs the
  barrier inline; the sealing transition does the pre-scatter accounting/range
  allocation; stages are stable per-block scatter -> per-group refine passes
  (reusing `scatterRefineGroup`) -> per-leaf probes in LPT order.
- Output: workers merge only their OWN leaf output up to
  min(joined-block target, `maxJoinedBlockRows`) and return it from their own
  `next()`; no shared output buffer exists. The delayed-blocks stream is an
  18-NCNB-line adapter running the same machine; split leaf probes park in a
  shared in-progress list (the established GraceHashJoin open-results pattern).
- Failure: first exception wins into one primary slot; the control word turns
  Poisoned; every entry point rethrows; a result abandoned while owing work
  poisons the wave with a `LOGICAL_ERROR` (fail-close, never silent row loss).
  A poisoned wave's memory is released exactly once by State destruction.
- The probe path uses no thread pool; the radix pool remains for the build
  side only (post-build scatter, leaf builds, destructor teardown).

Deviations within the sealed ownership model, recorded per the design's own
allowance: (a) refalloc+refine fused into one exactly-once job per group and
generalized to the plan's N-1 refine stages (a legal scheduling of the TLA
model); (b) the pre-scatter sizing runs in the sealing transition (the sealer
executing every TLA `pre` job before publishing scatter — also a legal
scheduling, sub-ms at the production fanouts); (c) admission itself happens in
the result's first `next()` (same executor quantum as `joinBlock`).

### Evidence

- Builds: `build/reldeb/build_wave_join_u3_try{1..5}.log` (try1: two compile
  errors — initializer-list pointer-type conflict, `Block` bool-conversion,
  both fixed; try2: private-State access + vector-of-immutable assign, fixed;
  try3: `-Wshadow` on nested-struct params, fixed; try4/5: clean).
- Focused tests, all green including the previously red contract test:
  `build/reldeb/test_wave_join_u3_gtest_squeezed.log` — 5/5
  (`ConcurrentJoiningQuantumDoesNotWaitForPreviousWave`,
  `SealedWaveDrainIsClaimableByOtherLanes` (RED before, GREEN after),
  `MultipleWavesAndFinalPartialWaveExactMultiset`,
  `MultiPassRefineExactMultiset`, `AbandonedResultPoisonsJoin`).
- Gate 7 (old coordination removal): PASS (the banned-token grep is clean).
- Style: `ci/jobs/scripts/check_style/check_cpp.sh` reports nothing for the
  changed files; `git diff --check` clean.

### Structural gate status (Gate 8): A RED — escalated to the user

`python3 tmp/wave-join-cooperative/complexity_gate.py ...` after squeezing
(unified worker loop, prepare folded into the seal transition, plan bound into
the engine, `concatenateBlocks` merging):

- Gate B PASS: synchronization-primitive declarations 11 vs baseline 20 (-45%).
- Gate C PASS: one drain machine (single `enum class`), exactly one
  `IJoinResult` subclass, delayed adapter 18 NCNB lines.
- Gate D PASS: header within frozen bounds, no new files.
- Gate A FAIL: candidate probe-control 712 NCNB vs baseline 623 (limit 467,
  i.e. -25%); the squeeze took it from 799 to 712 with all tests green.

Analysis (refutation-grade, from the script's own unit dump): the baseline's
623 lines are almost pure coordination because the old design outsourced its
probe data plane to the build-shared `scatterToPartitions` (excluded as
byte-identical shared code), while the cooperative contract REQUIRES the data
plane to run as claimable jobs on executor lanes — admission histograms,
per-block stable scatter, refine-job and leaf-run bodies are ~150 lines that
exist only in the candidate. `ProbeWave` is 453 NCNB; meeting the 467 total
would require halving it, which no honest restructuring of the sealed design
achieves; even a coordination-only accounting does not reach -25%.

Per the register rules ("freeze the script and its metric definitions before
implementation"; "do not weaken a gate to obtain green"; amendments require
user sign-off) this is a USER decision. Gate ordering blocks gates 9/10 until
resolved. Question posed to the user with these numbers; no gate was edited.

## Unit 4 — Gates (2026-07-16)

USER DIRECTIVE (mid-run): Gate 6 (ASan build + focused tests) is SKIPPED at
the user's explicit instruction ("Skip asan build gate"). It is reported as
SKIPPED-BY-USER, not as green.

### Deadlock found by Gate 5 and fixed (implementation cycle 2 of 5)

Gate 5's first run wedged: `04509_radix_join_distinct_estimate` timed out at
600 s. The user attached gdb to the hung server and provided full stacks
(archived: `tmp/wave-join-cooperative/evidence/deadlock_04509_gdb_bt.txt`):
two threads stuck in `ProbeWave::tailOrWait` from `CooperativeDelayedBlocks`
pulls, all other pipeline threads idle.

ROOT CAUSE (confirmed against the stacks, not guessed): the stuck waiter's
`word` argument was `67108864` = control word `(phase=Filling, gen=4, idx=0)`
— exactly the machine's TERMINAL state after the final wave completed
(`Sealing(1) -> Scattering(2) -> Probing(3) -> Filling(4)`).
`compare_exchange_strong` writes the CURRENT value into its expected argument
on failure; in the delayed EOF-seal branch a pull that lost the seal race to
a pull that then drained the whole (tiny) wave got its loop snapshot
overwritten with the terminal word, and `tailOrWait` then waited for
`control != terminal` — a transition that never comes. The formal model could
not catch this: it lives below the spec's atomic-action granularity (a stale
snapshot inside the C++ implementation of Claim/wait).

FIX (minimal, root cause): `claim` takes the control word BY VALUE, and the
EOF-seal CAS operates on a local `expected` copy — `tailOrWait` only ever
receives the pristine loop-top snapshot, so a lost race wakes immediately and
re-dispatches.

Red-evidence note (honest): a new regression test
`RadixHashJoin.ConcurrentDelayedPullsTerminate` (32 concurrent pulls x 256
rounds of a tiny final wave, 30 s deadlines) was written FIRST but did not
reproduce the hang pre-fix in two attempts (8x64, 32x256) — the natural race
window is nanoseconds at gtest scale, while the stateless repro ran 96
pipeline threads. The authoritative red evidence for this defect is the
04509 hang itself plus the archived gdb stacks; the test is kept as a broad
termination guard. Post-fix: 6/6 focused tests green
(`build/reldeb/test_wave_join_cooperative_gtest.log`), Gate 5 green in 1.6 s
including 04509 at 0.34 s
(`build/reldeb/test_wave_join_cooperative_stateless.log`), plus five extra
04509 repetitions green (`build/reldeb/test_wave_join_04509_repeat.log`).

### Gate status after the fix

| gate | status | evidence |
| --- | --- | --- |
| 1 baseline identity | PASS | sha equality re-run before candidate build |
| 2 formal | PASS (13/13) | build/reldeb/test_wave_join_tla.log |
| 3 reldeb build | PASS | build_wave_join_cooperative.log (relink; compile logs in build_wave_join_u3_try*.log and build_wave_join_deadlock_fix.log) |
| 4 focused gtests | PASS 6/6 | test_wave_join_cooperative_gtest.log |
| 5 stateless | PASS 5/5 | test_wave_join_cooperative_stateless.log |
| 6 ASan | SKIPPED-BY-USER | user instruction mid-run |
| 7 removal grep | PASS | re-run after fix |
| 8 structural (amended v2) | PASS | re-run after fix |
| 9 paired suite | pending | — |
| 10 verdict | pending | — |
| 11 diff check | pre-check PASS | re-run at the end |

### Unit 4 performance preregistration (committed BEFORE the suite run)

Invocation (exactly the frozen Gate-9 command):

    python3 tmp/wave-join-impl/suite.py \
      --binary-a /mnt/ch/ClickHouse/build/reldeb/programs/clickhouse \
      --binary-b /mnt/ch/ClickHouse/tmp/wave-join-cooperative/baseline/clickhouse \
      --cells all --reps 5 \
      --expect-sha256-b 4b55481c22d025ae364d36df39cd662bd986fd5878e711d89e1d76b08ea59cce \
      --out /mnt/ch/ClickHouse/tmp/wave-join-cooperative/candidate_u3.jsonl

Candidate binary at run time: the fixed cooperative build (sha256 recorded in
the run log and the JSONL header by the suite itself; the working tree equals
commit `a02a024c904`).

Preregistered expectations:
- Every frozen guard passes: integrity snapshots, foreign-process/loadavg,
  per-shape count assertions and radix-vs-parallel_hash fingerprints, A/B
  radix fingerprint equality, binary stability footer.
- ENGAGEMENT: the candidate arm's expected leaf count is UNCHANGED from the
  baseline (A: 16384, C: 32768) — this change is probe-side only; the build
  plan is untouched.
- Verdict (Gate 10, frozen `candidate_verdict.py`): every one of the 10 cells
  beats its frozen baseline median by MORE than its frozen band; no cell
  slower; T16->T96 scaling improves beyond combined noise for both shapes.
- Mechanism for the predicted win: mid-stream waves no longer funnel output
  through a bounded queue with a capped drain crew; every lane claims leaf
  work directly, phase transitions are lock-free CASes, and the wave scatter
  is spread across the lanes rather than serialized behind a coordinator
  mutex on the pool.

Refuted by: any guard failure or suite non-completion; any cell not beating
its band (goal gate fails => result reported as failure, not softened);
ENV-DRIFT (in-run baseline arm deviating > 2x band from the frozen median =>
UNSETTLED); scaling gate failing. The verdict script (frozen at
`e725eedc528`, null-tested) reports all failures and nulls as failures.

### REGISTER AMENDMENT v2 — Gate 8, USER SIGN-OFF (2026-07-16)

The user was asked with the exact numbers above and chose
"Amend: coordination metrics". Amendment as offered and accepted:

- Gate A becomes: candidate probe-control NCNB <= 115% of baseline
  (was <= 75%; the -25% intent moves to the coordination axis below).
- Gate B is TIGHTENED: at least 25% FEWER synchronization-primitive
  declarations (was: merely fewer).
- Gates C (one engine, thin adapter, one result type) and D (header bounds,
  file inventory) unchanged. All measurement definitions unchanged.

Alternatives explicitly offered and declined: keeping the gate as frozen and
reporting the task UNSETTLED; redesigning the implementation away from the
sealed design to chase the line count. The amendment is recorded in the
script header (`complexity_gate.py`, amendment v2) and here; the original
criterion, its failure (712 vs 467), and the analysis remain in this WORKLOG
uneditied above.
