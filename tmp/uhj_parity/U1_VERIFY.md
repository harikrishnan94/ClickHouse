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

## Remediation status (doer follow-up)

| Blocker | Status | Evidence |
| --- | --- | --- |
| 1 Serial A2 polluted metric | Fixed | `probe_summaries_v2.txt`: lock_in_insert 33 vs 0 |
| 2 B2 missing raw probe | Fixed | `probe_b2.txt` RESULT lines + JOB_EXIT=0 |
| 3 04659 JOB_EXIT=36 | Fixed | `test_04659.log`: OK / JOB_EXIT=0 via `run_04659.sh` |
| 4 Branch moved mid-verify | Mitigated | Remediation frozen in dedicated commit for re-verify |

Re-verify required before Unit 2.
