# Unit 1 independent verification

**Verdict: REWORK** (from verifier agent a8148942-6dc8-4de8-88e7-f4a947824166)

Verifier could not write this file (Ask/read-only mode); doer records the verdict verbatim and remediates below.

## Blocking list (as returned)

1. Serial A2 `mutex_related` counts polluted by `pthread_cond_timedwait` / `condition_variable` stacks (keyword match on `mutex` inside demangled templates). `serial_uhj mutex_related=110/170` is not valid build-lock domination evidence.
2. B2 lacks discriminating raw evidence in `probe_b2.txt` (only `B2_DONE`).
3. `04659` direct run `JOB_EXIT=36` (duplicate `--max_threads`); proxy lacked `JOB_EXIT`.
4. Branch/artifact mtimes moved during verification (`d0faf9f5158` → `3a8d41e12fc`); independence degraded.

## What verifier confirmed (keep)

- U1-BASE logs `JOB_EXIT=0`, RESULT lines match WORKLOG.
- Code: UHJ `addBlockToJoin` always passes `&bucket_locks`; HashJoin insert has no bucket-lock branch.
- Parallel B1 directionally strong (pthread_mutex under UHJ insert).
- Differential: `JOB_EXIT=0` but `DIFF_RESULT=FAIL` with 3 spill mismatches.

## Remediation plan (this follow-up)

- Fix probe keyword filter; re-run serial (+ confirm parallel) U1-DISC.
- Rewrite `probe_b2.txt` with raw timings.
- Re-run `04659` without duplicate `--max_threads`; ensure `JOB_EXIT=` on proxy.
- Freeze Unit 1 evidence commit; re-verify before Unit 2.
