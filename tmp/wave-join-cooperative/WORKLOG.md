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
   streams can hang (`WaveJoinResult` holds a wave mutex across scheduler
   steps). The cooperative design removes that class (no lock is ever held
   across a quantum boundary; abandoned results poison the wave instead).
