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
