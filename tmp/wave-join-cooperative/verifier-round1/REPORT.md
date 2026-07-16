# Independent verifier report — Unit 1 (formal contract), round 1

(Persisted verbatim by the session agent from the verifier subagent's final message;
the subagent harness rejected its own report-file write. Verifier agent id
a698893b561884367; all raw evidence logs in this directory were produced by the
verifier itself.)

Verifier: independent (not the author). Branch `radix-join-bandwidth-model`, HEAD `e725eedc528`.
Clean-room respected: `RadixHashJoin.cpp` not opened; `tmp/wave-join-impl/PLAN.md`/`WORKLOG.md` not read (absent on this branch); branch `wave-join-impl` not inspected. No tracked file was modified.

## Verdict: SHIP

Both gates green on my own re-run; every binding requirement (1-9) discharged by definitions I located and re-checked myself; all five of my own injected mutants caught by the existing battery; both witnesses honest; the F1/F5/F6 defect claims about the old spec confirmed real by direct inspection of `af54d1acc00`. Remaining findings are Low/Info (process and margin notes), none unrefuted critical/high. The preregistration-ordering limitation is real and documented below; it weakens the evidentiary force of the WORKLOG's prereg, not the correctness of the formal contract, which I re-derived independently.

## Evidence table

| # | Check | Command (abridged) | Raw outcome |
|---|-------|--------------------|-------------|
| 1 | Gate 1 baseline sha256 | `test "$(sha256sum tmp/wave-join-cooperative/baseline/clickhouse \| awk '{print $1}')" = "4b55481c…dce"` | `GATE1 PASS` |
| 2 | Gate 2 clean re-run | `bash tmp/wave-join-cooperative/verify_tla.sh > verifier-round1/gate2_rerun.log 2>&1` | `EXIT=0`; log read in full: SANY PASS; MC_Normal 26,339/11,832 no error; MC_PL1 219/142 no error; MC_MultiWave 119,576/41,231 no error; MC_Fail 2,697/1,152 no error; MC_CancelRace 27,087/10,944 no error; MC_NoSteal `Error: Temporal properties were violated`; wrapper mutant `Error: Action property PrimaryStable is violated`; all 3 reach configs `Invariant … is violated`. Counts identical to the WORKLOG's claims. |
| 3 | Old-spec defects (F1/F5/F6) | `git show af54d1acc00:src/Interpreters/RadixHashJoin/WaveJoinProbe.tla` | F1: `CooperativeParticipation` (old line 659) compares `BusyWorkers` to `ScanWorkers`/`DrainWorkers`, all derived from phase/counts via `Take`/`Fill` (a central dispatcher); jobs in `st.running` carry no worker identity — tautology confirmed. F5: old `MemoryBound == ~(liveEntries # {} /\ liveA1 # {})`; `st.mem` never bounded — confirmed. F6: `FinalRefinement` only in an unchecked THEOREM; zero TLC `.cfg` files at that commit (only jepsen `zoo.cfg`) — confirmed. Also: old `FinishProbe` emitted `ProbeResult(l, ExpectedLeaf(…))` into a SHARED `st.output` — definitional output and a shared buffer, both fixed in the new spec. |
| 4 | Mutant A: scatter drops a row | TLC MC_Normal | `Error: Invariant StableAtBarriers is violated.` (exit 12) |
| 5 | Mutant B: double-claim | TLC MC_Normal | `Error: Invariant OwnershipConsistent is violated.` (exit 12) |
| 6 | Mutant C: `Reserve` ignores budget | TLC MC_Normal, MC_MultiWave | MC_Normal: **no error** (33,743/16,292) — survives; MC_MultiWave: `Error: Invariant MemBound is violated.` — caught by the battery |
| 7 | Mutant D: `Seal` skips in-flight quiescence | TLC MC_Normal | `Error: Invariant NoAdmitDuringDrain is violated.` (exit 12) |
| 8 | Mutant E: probe drops its output | TLC MC_Normal | `Error: Invariant FinalRefinement is violated.` — the F6 property itself is live, not shadowed by construction |
| 9 | Witness honesty (MC_NoSteal) | re-ran `MC_NoSteal` with the `ClaimEligible <- broken_ClaimEligible` line removed from the cfg, `-deadlock` kept | `Model checking completed. No error has been found.` (26,339/11,832) — the counterexample disappears exactly when eligibility is restored |
| 10 | MC_CancelRace reachability honesty | scratch module + invariants `~(cancelled /\ doneWaves # <<>>)` and `~(cancelled /\ phase="probe" /\ probeDone=Leaves)` under MC_CancelRace constants, `SPECIFICATION Spec` | both `Invariant … is violated` — cancellation genuinely races wave history and a CompleteWave-ready state (panel finding 3's fix verified) |
| 11 | Budget-honesty grep | `grep -inE "resident\|RSS\|total memory" WaveJoinProbe.tla` | single hit: line 38, the disclaimer "NOT a bound on total process resident memory" — no RSS ≤ BUDGET claim anywhere |
| 12 | Wrapper mutant sed target | `grep -c '!.primary = IF st.primary = NoError THEN ErrorOf(l) ELSE @,'` | exactly 1 occurrence (`FaultProbe`); the wrapper also `cmp`-checks that the mutation applied |
| 13 | Prereg ordering | `git show --stat 7a1c9b39664 01407136160 e725eedc528` | WORKLOG (prereg + first results) committed in `01407136160` (11:45:07) TOGETHER with the corrected spec, configs and wrapper — 24 s after the import commit `7a1c9b39664` (11:44:43, no WORKLOG). Gap-closure in `e725eedc528` (12:07:52). |
| 14 | Imported artifact integrity (partial) | `sha256sum -c tmp/wave-join-impl/sql_sha256.txt` | exit 0, 16/16 OK |

## Requirements 1-9 — where each is discharged

1. **One shared wave**: single `st.queue`; `Reserve`/`Admit` guarded on `phase = "active"`; `Seal` requires every in-flight reservation admitted; invariants `NoAdmitDuringDrain`, `HashOnce`, `SealedJustified`, `WaveJustified`. Mutant D proves overlap is caught.
2. **Cooperative workers, no crews/queues/scheduler**: `ClaimEligible(w, kind, id) == TRUE`; workers self-claim from `UnownedClaimable`, derived purely from phase + done-sets + queue (no scheduler state); `out[w]` appended only by `w`; the old `Fill`/`Take` dispatcher and shared `st.output` are gone. Barriers are unattributed global transitions — an abstraction, but they hold no worker-busy bookkeeping.
3. **F1 fixed**: worker-local `st.wk[w]` (`res`/`job`/`stopped`); worker-parameterized `Claim`/`FinishPre`/`FinishScatter`/`FinishRefAlloc`/`FinishRefine`/`FinishProbe`/`Fault*`/`ReleaseJob`/`ReleaseRes`/`StopWorker`. `OwnershipConsistent` cross-checks ownership against the independently derived work side; `ParticipationLive` compares `Idle`/`Acquired` (worker side) with `UnownedClaimable`/`ScanOpen` (work side, eligibility-independent). Capable of failing: MC_NoSteal fails it; restoring eligibility makes it pass (check 9); mutant B caught.
4. **F5 fixed**: `MemAccounted` (`mem = SumBytes(queue) + InflightBytes`, in-flight reservations included), `MemBound` (`mem <= BUDGET + MaxBlockBytes`, explicit bounded overshoot), `CrossedSound`. The header states BUDGET is the admission/sealing threshold and explicitly NOT an RSS bound; grep confirms no contrary claim. Mutant C caught (see Finding 2 for the margin).
5. **F6 fixed**: `FinalRefinement` is a checked INVARIANT in MC_Normal, MC_PL1, MC_MultiWave, MC_CancelRace (verified in the cfg files and passing TLC runs); `done` is genuinely reached (no-fault/no-cancel configs + `Termination`); mutant E proves the property itself can fail.
6. **Explicit drain work graph**: `PreJob` → `ScatterJob` → (`RefAllocJob` → `RefineJob` iff PL > 1) → `ProbeJob` with phase barriers; exactly-once claims/completion via `OwnershipConsistent` + done-sets + `HashOnce` + `FreedOnce` (per-leaf `freedA1` counters); disjoint writes via `RaceFree` + `Footprint` + `RankInjective`; stable scatter via `Rank0`/`Rank1` + `CellSafety` + `StableAtBarriers`; probe-per-leaf is the smallest task. Mutant A caught. MC_Normal's input genuinely exercises multi-block passes/leaves, so rank stability is load-bearing, not vacuous.
7. **Per-worker output**: only `FinishProbe(w, l)` appends, and only to `out[w]`; `EmittedAll` appears in properties only; `FinalRefinement` is exact multiset equality, order unconstrained (MC_MultiWave's cross-wave duplicate result exercises multiplicity).
8. **Failure semantics**: first-exception-wins guard on `primary` + `PrimaryStable` action property + wrapper mutation witness (MC_Fail has two concurrently-ownable failing leaves with distinct errors — I confirmed the two faults can race, so the property is genuinely exercised); cancellation visibility (`cancelled`), no new work (`UnownedClaimable = {}` when cancelled; `Reserve`/`Admit`/barriers guarded), unwind (`ReleaseJob`/`ReleaseRes`/`StopWorker`), exactly-once release (`FreedOnce`), terminal validity (`TerminalClean`), propagation (`FailureSafety`, `primary` preserved to the terminal state). EOF (`EOFSeal`/`FinishInput`), normal completion, injected faults (`FaultScan`/`FaultStep`/`FaultProbe`), external cancel all modeled; PL=1, multi-wave, EOF and the final partial wave all run through the one machine — I checked the full action set: no second machine exists.
9. **Interleavings/fairness**: safety invariants are checked over the full reachable state graph (fairness constrains only liveness); reach configs use plain `Spec`. Progress (`Termination`, `ParticipationLive`) explicitly states per-worker WF + WF(`Transition`).

The `-deadlock` flag is used ONLY for the MC_NoSteal witness and is justified: the broken eligibility wedges the drain (leaf 0 becomes claimable by nobody), so without `-deadlock` TLC's deadlock detector fires before the temporal check; with it, stuttering is allowed and `ParticipationLive` — the only property in that cfg — produces the counterexample. All positive runs keep deadlock checking on.

## Findings

1. **[Low, process] Preregistration ordering is unverifiable; the WORKLOG's "preregistered … BEFORE the repair" claim cannot be substantiated from the record.** The prereg text and the delivered spec/configs/wrapper/first-results landed in one commit (`01407136160`), and the prereg names the delivered design in detail. Consistent with honest write-then-execute in a single sitting, but of no independent evidentiary force. Mitigation: nothing in this verification rests on the prereg. To make green in later units: commit the prereg in its own commit before producing the artifacts it governs. Related retention gap: the first (failed) witness run is documented honestly in the WORKLOG iteration note, with the `-deadlock` rationale, but its raw log lives only in "shell history", not as a preserved artifact. Deviations prereg-vs-delivered (5th positive config `MC_CancelRace`, 3 reach configs instead of 1, the wrapper mutation witness) all strengthen the gate and are documented in the WORKLOG's panel section.
2. **[Low, adequacy margin] The budget-check mutant survives 4 of 5 positive configs.** Removing `st.mem < BUDGET` from `Reserve` passes MC_Normal (33,743 states, no error — `BUDGET + MaxBlockBytes = 5` equals total input bytes), and by the same arithmetic survives MC_PL1, MC_Fail and MC_CancelRace; only MC_MultiWave violates `MemBound`. Caught by the battery, so the requirement is met, but the margin is a single config. Suggested (non-blocking): add this mutant to `verify_tla.sh`'s witness set, or give one more config input bytes exceeding `BUDGET + MaxBlockBytes`.
3. **[Info] `MemAccounted` excludes `phase = "failed"`.** Numerically the identity would hold there too; the exclusion is defensible (entries are freed at `Teardown`) but slightly weakens the invariant. No action required.
4. **[Info] The fairness-to-implementation mapping is an assumption, not a proven fact.** The header maps per-worker WF to "the executor contract guarantees the remaining lanes / the delayed-blocks stream keep pulling" even when a caller destroys a result early. Nothing at the formal stage can discharge this (and clean-room forbids checking the C++). The spec flags it and defines a falsification protocol; Units 2-4 must actually discharge it.
5. **[Info] Panel round 1 is only partially verifiable.** The WORKLOG cites workflow `wf_c4e003a2-de5` (12 raw findings, 3 confirmed); the journal is not in the repo. I could not verify the panel's process, but I independently re-verified all three fixes: the `PrimaryStable` mutation witness (Gate 2; unique sed target), the per-leaf `freedA1`/`FreedOnce` accounting (present in the spec), and MC_CancelRace's race reachability (check 10).

## Mutation results (all mine, scratch copies only, under `verifier-round1/`)

| Mutant | Diff snippet | Caught by | Raw TLC line |
|---|---|---|---|
| A: scatter drops a row | `LET occs == SeqSet(BlockOccs(b))` → `… \ {<<b, 1>>}` | MC_Normal | `Error: Invariant StableAtBarriers is violated.` |
| B: two workers own one job | `ELSE ExistingJobs \ OwnedJobs` → `ELSE ExistingJobs` | MC_Normal | `Error: Invariant OwnershipConsistent is violated.` |
| C: reserve ignores budget | delete `/\ st.mem < BUDGET` from `Reserve` | MC_MultiWave (survives MC_Normal) | `Error: Invariant MemBound is violated.` |
| D: seal during in-flight admission | delete `\A w \in WorkerIds : st.wk[w].res = NoBlock` from `Seal` | MC_Normal | `Error: Invariant NoAdmitDuringDrain is violated.` |
| E: probe emits nothing | `!.out[w] = @ \o ConcatResults(DropNoRow(st.arena1[l])),` → `!.out[w] = @,` | MC_Normal | `Error: Invariant FinalRefinement is violated.` |

Logs on disk: `gate2_rerun.log`, `mutA_MC_Normal.log`, `mutB_MC_Normal.log`, `mutC_MC_Normal.log`, `mutC_MC_MultiWave.log`, `mutD_MC_Normal.log`, `mutE_MC_Normal.log`, `honestE1_restored.log`, `honestE2_history.log`, `honestE2_probedone.log`, `sql_integrity.log`.

## What I could NOT verify, and why

- **Wall-clock prereg ordering** (Finding 1): same-commit; no independent timestamp.
- **The evidence-import blob-id cross-check (25/25) against branch `wave-join-impl`**: inspecting that branch is prohibited to me. Partially compensated: `sha256sum -c sql_sha256.txt` → 16/16 OK on this branch, and `tmp/wave-join-impl/` contains no forbidden PLAN/WORKLOG files.
- **Panel round 1's process** (Finding 5): journal not in the repo; the fixes were verified independently.
- **The C++ mapping comments and the fairness assumption** (Finding 4): clean-room forbids opening `RadixHashJoin.cpp`; deferred to later units by design.
- **The first witness run's raw output**: not preserved as a file; the documented behavior (deadlock detector firing first) is consistent with TLC semantics and with what `-deadlock` changes, and the final witness behavior reproduces on my re-run.
