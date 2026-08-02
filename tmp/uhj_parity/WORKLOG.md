# UHJ parity — worklog

## Environment
- Host: `Linux ip-172-31-5-72 7.0.0-1008-aws aarch64` (96 CPUs)
- Repo: `/mnt/ch/ClickHouse`
- Build: `build/reldeb` (RelWithDebInfo) — always, per user
- Binary: `/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse` (rebuilt at tip)
- Server: `:9101` via `tmp/uhj_parity/start_server.sh` (minimal config, no Keeper)
- Work branch: `uhj-parity` @ `d0faf9f5158`
- Preserve: `uhj-parity-preserve-20260802` @ `86e2f1b07b3`
- Foundation tip (unchanged): `unified-hash-join-foundation` @ `86e2f1b07b3`
- Note: `tmp/` is gitignored; `git add -f` when committing artifacts.

---

## Unit 0 — Branch restore

**Goal:** Preserve post-Aug-1 work; park work tip at restore SHA without moving shared foundation.

**What was done**
- User: defaults + always `build/reldeb`.
- `uhj-parity-preserve-20260802` @ `86e2f1b07b3`; `uhj-parity` @ `d0faf9f5158`.

**How verified:** U0-A equivalent green; U0-B ANCESTOR_OK=1 with 6 post-restore commits; no push.

**Verdict:** green.

---

## Unit 1 — Bidirectional root-cause

**Goal:** Catalogue hot-path diffs; classify; rank MATERIAL causes with discriminating probes.

**What was done**
- Harnesses: `bench_serial.sh`, `bench_parallel.sh`, `diff_inventory.sh`, `probe_profile.sh`.
- Rebuild clickhouse at tip; dedicated server :9101.
- U1-BASE benches (5 runs, median+stdev).
- U1-DISC Real profiler probes serial/parallel.
- Inventory: `INVENTORY.md`.
- Correctness: `diff_unified_p9101.sh`; 04658/04659 launched.

**How verified — U1-BASE**
```
# bench_serial.log JOB_EXIT=0
RESULT algo=hash wall_median_ms=272 wall_stdev_ms=101.726 cpu_median_us=159771
RESULT algo=unified_hash wall_median_ms=1839 wall_stdev_ms=130.448 cpu_median_us=1733076
# bench_parallel.log JOB_EXIT=0 threads=16
RESULT algo=parallel_hash wall_median_ms=144 wall_stdev_ms=65.7092 cpu_median_us=972678
RESULT algo=unified_hash wall_median_ms=31251 wall_stdev_ms=685.396 cpu_median_us=45971115
```

**How verified — U1-DISC**
```
# probe_summaries.txt
SUMMARY label=serial_hash total=36 mutex_related=18 insert_related=18
SUMMARY label=serial_uhj total=170 mutex_related=110 insert_related=84
SUMMARY label=parallel_phash total=42 mutex_related=10 insert_related=5
SUMMARY label=parallel_uhj total=6733 mutex_related=6674 insert_related=5815
# parallel_uhj top stack: std::lock + pthread_mutex_lock under insertFromBlockImplTypeCase
# Code: HashJoin.cpp always passes &bucket_locks → parallel_build always true
```

**How verified — U1-INV:** see `INVENTORY.md`. MATERIAL: A2/B1 (per-row locks). EXCLUDED: A1 two-level. UNSETTLED: A3, B3 pending A2/B1 fix. B2 plumbing CONFIRM avoidable.

**Correctness:** `diff_unified.log`: checks=44 mismatches=3 — all three are spill cases where `hash` throws `LIMIT_EXCEEDED` (grace buckets) and UHJ returns rows. NON_SPILL_MISMATCHES=0.

**Plan changes:** Primary fix target for Unit 2 = align insert locking with baselines (nullptr/`hash` serial path; ConcurrentHashJoin-style batch lock for parallel), then plumb max_threads (B2).

**Authority calls:** none new.

---

## Unit 1 — REWORK remediation (after verifier a8148942)

**Blockers addressed**
1. Fixed `probe_profile.sh` lock filter (exclude `condition_variable` / `pthread_cond`; require real lock ops). Re-ran as v2:
   ```
   serial_hash_v2  lock_in_insert=0
   serial_uhj_v2   lock_in_insert=33   (lock_ops=33, all under insert)
   parallel_phash_v2 lock_in_insert=0
   parallel_uhj_v2   lock_in_insert=5055/5929
   ```
   Retracted prior `mutex_related=110/170` claim.
2. `probe_b2.txt` now has raw RESULT lines (UHJ ~1.3s both spill settings; parallel_hash ~60ms).
3. `run_04659.sh` rewritten to avoid duplicate `--max_threads` (original test + shell_config clash).
4. Evidence refresh committed as its own commit after this remediation (freeze for re-verify).

**INVENTORY:** A2 path CONFIRM MATERIAL (magnitude partial); B1 CONFIRM MATERIAL; B2 plumbing CONFIRM / wall UNSETTLED pending B1.
