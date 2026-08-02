# FINAL VERIFY — Units 0–2

**Verdict:** `SHIP` (provenance caveat only; not a parity blocker)

**Verifier:** [Final Unit 2 verifier](ca3b20ca-6f8b-471b-aa96-ca2de0102e36)  
**Tip checked:** `uhj-parity` @ `13ec290c6c6d` (code `f0420d93d31`)  
**Mode note:** verifier ran read-only Ask mode and could not write this file; parent session transcribed the verdict here.

## Must-hold checks

| # | Check | Result |
| --- | --- | --- |
| 1 | Unit 1 SHIP before Unit 2 (`U1_VERIFY_R2.md`) | PASS (transcript proves run at `5b659f8f24a` pre-U2; see caveat) |
| 2 | U2-PRE: PREREG before code | PASS — `3a8d41e12fc` before `f0420d93d31` |
| 3 | U2-SERIAL wall | PASS — uhj 251 ms ≤ hash 267 ms (`bench_serial_f1c.log`) |
| 4 | U2-PARALLEL wall | PASS — uhj 157 ms ≤ phash 121 + 49.6 ms noise (`bench_parallel_f1c.log`) |
| 5 | Parallel CPU residual stop criterion | PASS — documented EXCLUDED two-level in `REPORT.md` |
| 6 | No UHJ-only invent; no per-row `scoped_lock` in MethodsImpl | PASS |
| 7 | Correctness 04658/04659 | PASS — OK, `JOB_EXIT=0` |
| 8 | No push / no PR / not master | PASS — no upstream `uhj-parity`; `gh pr list --head uhj-parity` empty |
| 9 | B2 plumbing after F2 | PASS — uhj ≈ phash for spill 0 and large spill (`probe_b2_after_f2.txt`) |

## Blocking findings

None.

## Caveat (non-blocking)

`U1_VERIFY_R2.md` content says SHIP and the verifier transcript shows it ran at `5b659f8f24a` before Unit 2 code, but the file was committed later in `13ec290c6c6` with the final report. Weaker committed provenance; not a parity/correctness blocker.

## Independence

HEAD stayed stable during verification. Existing committed logs/code were inspected; benchmarks/tests were not re-run in this pass.

## Suggested next (out of mission)

None for parity mission. Do not push/PR unless asked. Flat-map A/B remains out of scope (EXCLUDED two-level).
