# Unit 1 verification after REWORK remediation

## Verdict

SHIP.

Unit 1 is ready to proceed to Unit 2. I found no remaining blocking issue from the prior REWORK list. This verification checked the repository state, committed artifact inventory, profiler summaries/raw stacks, correctness logs, and the relevant `HashJoin` / `UnifiedHashJoin` code paths. I did not modify `src/` or push.

## Blocking list

None.

Reproduce the checked gates / evidence from `/mnt/ch/ClickHouse`:

```bash
git -C /mnt/ch/ClickHouse rev-parse --abbrev-ref HEAD
git -C /mnt/ch/ClickHouse rev-parse HEAD
git -C /mnt/ch/ClickHouse merge-base --is-ancestor 207a664a6aa HEAD; echo ANCESTOR_EXIT=$?
git -C /mnt/ch/ClickHouse status --short

cat tmp/uhj_parity/bench_serial.log
cat tmp/uhj_parity/bench_parallel.log
cat tmp/uhj_parity/probe_summaries_v2.txt
cat tmp/uhj_parity/probe_b2.txt
cat tmp/uhj_parity/test_04659.log
cat tmp/uhj_parity/diff_unified.log
```

If rerunning runtime gates, use the existing scripts under `tmp/uhj_parity/` and preserve unique logs for comparison:

```bash
UHJ_PORT=9101 bash tmp/uhj_parity/bench_serial.sh
UHJ_PORT=9101 bash tmp/uhj_parity/bench_parallel.sh
UHJ_PORT=9101 bash tmp/uhj_parity/run_04659.sh
```

## What I checked

- Frozen tip: start and end HEAD were `5b659f8f24ac0ebeaf0e3b7f9d81a63d0da81984` on branch `uhj-parity`; `207a664a6aa` is an ancestor (`ANCESTOR_EXIT=0`). No HEAD movement was observed during this verification.
- U1-BASE: `bench_serial.log` and `bench_parallel.log` both have `JOB_EXIT=0` and their `RESULT` lines match `WORKLOG.md`: serial `hash` 272ms vs `unified_hash` 1839ms; parallel `parallel_hash` 144ms vs `unified_hash` 31251ms.
- Prior blocker 1: the v2 profiler filter in `probe_profile.sh` excludes `condition_variable` / `pthread_cond` frames and records `lock_in_insert`. `probe_summaries_v2.txt` shows serial `hash` `lock_in_insert=0` vs serial `unified_hash` `lock_in_insert=33`, clearing the polluted `mutex_related` claim.
- Prior blocker 2: `probe_b2.txt` now has raw `RESULT` lines for both algorithms and both spill settings, plus `JOB_EXIT=0`.
- Prior blocker 3: `test_04659.log` is `OK`, `JOB_EXIT=0`, `WRAPPER_EXIT=0`; `run_04659.sh` no longer combines the original shell-config `COMMON --max_threads=16` with additional `--max_threads` overrides.
- Prior blocker 4: remediation is frozen in commits after `207a664a6aa`; no mid-verify branch churn was observed.
- Code path: `src/Interpreters/UnifiedHashJoin/HashJoin.cpp` unconditionally passes `&bucket_locks[onexpr_idx]` into `Unified::HashJoinMethods::insertFromBlockImpl`; `src/Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h` treats non-null locks as `parallel_build` and locks per row. The legacy `src/Interpreters/HashJoin` insert path passes only `data->pool` / `Arena & pool` and has no `bucket_locks` row-loop branch.
- Parallel B1: `probe_summaries_v2.txt` shows `parallel_uhj_v2 lock_in_insert=5055/5929` vs `parallel_phash_v2 lock_in_insert=0/43`. `probe_parallel_uhj_v2_stacks.txt` top stacks show `pthread_mutex_lock` / `std::__1::lock` under `Unified::HashJoinMethods::insertFromBlockImplTypeCase`; `probe_parallel_phash_v2_stacks.txt` does not show matching insert-path lock domination.
- Inventory honesty: `INVENTORY.md` retracts the old polluted serial `mutex_related=110/170` overclaim, marks A2 as `CONFIRM MATERIAL (path); magnitude partial`, marks B2 wall impact `UNSETTLED pending B1 fix`, and keeps A3/B3 as `UNSETTLED`.
- Spill differential: `diff_unified.log` still records the three spill mismatches explicitly (`INNER`, `RIGHT`, `FULL`), with `DIFF_RESULT=FAIL` and `JOB_EXIT=0`; they were not silently dropped.
- PREREG / no Unit 2 source changes: commit `3a8d41e12fc` modifies only `tmp/uhj_parity/PREREG.md` and pre-registers Unit 2 F1/F2 before implementation. `git diff --name-status d0faf9f5158..HEAD` and `git diff --name-status 3a8d41e12fc..HEAD` show only `tmp/uhj_parity` artifacts, no `src/` changes.

## Independence note

Independence is intact for branch state: HEAD remained `5b659f8f24ac0ebeaf0e3b7f9d81a63d0da81984` throughout the verification. This report file is the only write performed by this verifier.

## Unverified / risks

- I reviewed existing runtime logs and raw profiler artifacts but did not rerun the long benchmark/profile gates during this pass, to avoid regenerating additional evidence beyond this report.
- Unit 1 only identifies and ranks causes; it does not validate the future Unit 2 fixes.

## Suggested next steps

- Proceed to Unit 2 from the pre-registered F1/F2 plan.
- Keep spill differential mismatches visible until Unit 2/related correctness work explicitly resolves or reclassifies them.
