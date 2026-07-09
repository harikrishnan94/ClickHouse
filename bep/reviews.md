# RADIX-JOIN-V1 — Review record (§12)

> Cadence change (D-0011, user directive, 2026-07-09): per-unit reviews are deferred. U2–U5 close
> on acceptance evidence alone; ONE consolidated adversarial review of the full implementation
> runs after U5, in parallel with U6. The U2 fan-out that was mid-flight was stopped unfinished;
> its scope is folded into the consolidated pass.

## U1 — Component port — single adversarial pass (low-risk tier) — 2026-07-09T18:30Z

Reviewer: independent subagent, clean context (spec + artifacts + §9 only).
Method highlights: normalized donor diff (applied the three declared renames to donor files and
diffed — residue zero except declared adaptations); `rowRefFromWord` bit-exactness proven against
both definitions; independent gtest reruns (reldeb, ASan); reviewer REBUILT the stale TSan binary
and ran the suites + 20× repeats of the barrier/cancellation tests → 0 TSan warnings; 100×
reldeb repeats of the wave tests → 100/100; x86-64 compile of all 6 production TUs clean; CMake
object + gtest-glob pickup confirmed; barrier-coherence and SWWC bounds traced in code.

Findings (none blocking; no open LEADs):
- R1 (C, should-fix): PartitionOutput used resize→PODArray pow2 rounding — "exact allocation"
  property broken (up to 2×) vs bench's alloc_for_num_elements. Fix: resize_exact. → APPLIED.
- R2 (C, should-fix): scatterColumns holds 2 B/row pids for the WHOLE side across phases 1–3
  (bench bounded pids per batch). scatterWaves (U3's path) is per-window and fine.
  → DOCUMENTED in ColumnScatter.h + recorded as U3 budget-accounting obligation.
- R3 (A, should-fix): validateChunks admitted isFixedAndContiguous columns (Decimal/UUID/IPv4/6,
  DateTime64) that resizeUninitialized later rejects mid-scatter on a worker thread. Fail-close but
  a phase too late. Fix: upfront allocability probe in validation. → APPLIED.
- R4 (C/E, should-fix): no test ran scatterWaves through SWWC fanout (≥256) to completion with
  content verification. → TEST ADDED (WavesSwwcCompletionFingerprint).
- R5 (A, minor): computePassBits(p_star≤1) divergence vs bench (bench UB/SIGFPE, port returns {}).
  → Contract decided & documented: {} means "no partitioning needed"; caller must not call
  scatter entry points with empty bits; gtest pins computePassBits(1)=={}.
- R6 (E, minor): MAX_FANOUT_PER_PASS exported but not enforced. → computePassBits clamps f_max to
  it; comment states the enforcement locus.
- R7 (D, minor): header overstated stop promptness (consume steal-loop keeps draining the wave
  after a sibling throws). → header corrected + best-effort relaxed stop check in the steal loop
  (does not change barrier arrival counts).
- R8 (B, note): phase-body duplication between scatterColumns pass 1 and scatterWaves —
  revisit when U3 touches the file. → ACCEPTED as-is (recorded).

Axis verdicts: A PASS-with-items · B PASS · C PASS-with-items · D PASS · E PASS-with-items.
Bottom line: SHIP-WITH-FIXES. Disposition: all items applied/documented (see L0008); gates re-run
green post-fix before commit. Unit U1 declared GREEN after post-fix gate re-run.
