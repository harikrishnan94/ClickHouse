# UHJ parity — pre-registration

Environment of record (filled as measured):
- Host: (see WORKLOG `uname -a`)
- Binary: `/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse`
- Build dir: `build/reldeb` (user: always)
- Settings defaults for benches: `enable_join_runtime_filters=0`, `max_bytes_before_external_join=0`
- Noise band (Unit 2, declare before fixes): effects within `max(5%, 1 stdev)` of run-to-run variance are **NO RESULT**

---

## Unit 0 — Branch restore (pre-registered before branch ops)

**Expected outcome**
- Preservation branch `uhj-parity-preserve-20260802` points at tip `86e2f1b07b3` and contains all commits after `d0faf9f5158`.
- Work branch `uhj-parity` checked out at `d0faf9f5158`.
- `unified-hash-join-foundation` left at `86e2f1b07b3` (user-approved U0-A equivalent; do not move shared tip).
- No push; no PR; status shows no upstream push pending created by this task.

**Gate invocations**
- U0-A equivalent: `git -C /mnt/ch/ClickHouse rev-parse uhj-parity` and `git -C /mnt/ch/ClickHouse log -1 --format='%h %ci %s' uhj-parity` → must be `d0faf9f5158…` with date ≤ 2026-08-01 23:30 +0530. Also record `unified-hash-join-foundation` still at `86e2f1b07b3`.
- U0-B: `git -C /mnt/ch/ClickHouse merge-base --is-ancestor d0faf9f5158 uhj-parity-preserve-20260802 && git -C /mnt/ch/ClickHouse log --oneline d0faf9f5158..uhj-parity-preserve-20260802 | head`
- Regression: `git -C /mnt/ch/ClickHouse status -sb` → no `[ahead N]` created by push of this task; confirm read-only `remote show origin | head -5`.

**Refute outcomes**
- U0-A red if `uhj-parity` is not `d0faf9f5158` or date is after cutoff.
- U0-B red if preserve branch missing post-restore commits or ancestor check fails.
- Authority red if any push/PR occurred.

**Authority call (user 2026-08-02)**
- Defaults + always `build/reldeb`. Leave foundation tip; work on `uhj-parity`.

---

## Unit 1 — Bidirectional root-cause (pre-registered BEFORE U1-BASE / probes)

**Environment of record**
- Host: aarch64 Linux, 96 CPUs (`ip-172-31-5-72`)
- Branch/tip: `uhj-parity` @ `d0faf9f5158`
- Binary: `/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse` rebuilt at this tip (mtime after rebuild)
- Server: dedicated port `UHJ_PORT=9101` via `tmp/uhj_parity/start_server.sh` (does not use :9000)
- Bench settings: `enable_join_runtime_filters=0`, `max_bytes_before_external_join=0`, serial `max_threads=1`, parallel `max_threads=16`
- Noise band (for later Unit 2): `max(5%, 1 stdev)` — NO RESULT inside band

**Expected outcome (U1-BASE)**
- Fresh medians+stdevs for wall_ms and UserTimeMicroseconds for:
  - serial: `hash` vs `unified_hash` (`bench_serial.sh`)
  - parallel: `parallel_hash` vs `unified_hash` (`bench_parallel.sh`)
- Recorded in WORKLOG and this file's results section after the run (results are not predictions).

**Expected outcome (U1-INV / U1-DISC)** — orientation LEADs to probe, not acceptance yet
- Candidate A1 (serial): unconditional TwoLevel vs flat HashMap — EXCLUDED if cost alone is two-level required; else measure.
- Candidate A2 (serial): UHJ serial insert path does per-row `impls[0]` buffer-size reads / different byte accounting vs HashJoin Arena-only — probe via assembly/profile if slowdown remains after excluding two-level.
- Candidate B1 (parallel): per-row `blocks_mutex`+bucket lock vs ConcurrentHashJoin per-shard mutex + scatter — CONFIRM if profile shows lock domination unique to UHJ.
- Candidate B2 (parallel): `createInMemoryHashJoin` / Spilling path drops `max_threads` for Unified when spill wrapper used — probe by comparing with/without `max_bytes_before_external_join=0` (benches force 0 so direct ctor path; separate probe for default spill).
- Candidate B3: `BUCKETS_PER_THREAD` / scan-from-zero contention vs ConcurrentHashJoin's shard count = threads.

**Gate invocations**
- U1-BASE: `bash tmp/uhj_parity/bench_serial.sh` and `bash tmp/uhj_parity/bench_parallel.sh` → raw RESULT/RAW_* lines in logs under `tmp/uhj_parity/`.
- U1-INV: `bash tmp/uhj_parity/diff_inventory.sh` → `tmp/uhj_parity/diff_inventory_out.txt`
- U1-DISC: per-cause probes (profile / with-vs-without setting / code path assert) recorded before MATERIAL claims.
- Regression: `UHJ_PORT=9101 bash tmp/diff_unified_p9000.sh` adapted, or `clickhouse-test 04658 04659` — document which.

**Refute**
- U1-BASE red if <5 runs, missing median/stdev, or wrong tip binary.
- U1-DISC red if probe outcome is compatible with every cause (vacuous).
- U1-INV red if a claimed slowdown has no mapped non-two-level-required diff and is not EXCLUDED/UNSETTLED.

---


## Unit 2 — Parity fixes (PRE-REGISTERED before any fix commit)

Noise band (declared): effects within `max(5%, 1 stdev)` of run-to-run variance are **NO RESULT**.

### Fix F1 — Align serial/parallel insert locking with baselines (closes A2/B1)

**Expected delta:** UHJ serial wall/cpu within noise of `hash`; UHJ parallel wall/cpu within noise of `parallel_hash` (or residual attributable only to EXCLUDED two-level after F1).

**Baseline being matched**
- Serial: `src/Interpreters/HashJoin/HashJoinMethodsImpl.h` insert path with `Arena &` and no locks (~243–311); pass `bucket_locks=nullptr` when not doing a multi-threaded build.
- Parallel: `src/Interpreters/ConcurrentHashJoin.cpp` ~298–351 — hold shard/bucket mutex around a *batch* of rows for that shard (restore scatter-then-lock-per-group), not per-row `std::lock(blocks_mutex, bucket)`. Index `bucket_locks` by bucket alone (not clause×bucket) so arena races are covered without `blocks_mutex` on the row path — matching ConcurrentHashJoin's one-mutex-per-shard model.

**Gate command:** `bash tmp/uhj_parity/bench_serial.sh` and `bash tmp/uhj_parity/bench_parallel.sh`; `bash tmp/uhj_parity/diff_inventory.sh` → no new UHJ-only divergences.

**Refute:** if after F1, serial UHJ still > hash + noise with profile still showing per-row mutex in insert, F1 failed. If parallel still mutex-dominated at 99%, F1 failed.

**Must not:** add UHJ-only fast paths absent from baselines; remove TwoLevel.

### Fix F1b — Stagger + bucket sizing (closes B3 residual after F1)

**Expected delta:** U2-PARALLEL wall within noise of `parallel_hash` after F1 left ~7× wall gap (CPU already ≤ baseline).

**Baseline:** ConcurrentHashJoin uses `slots ≈ max_threads` and try_lock drain across shards starting independently. Preserve commits `4ea37fd4777` / `ce6d1d7150d` align UHJ bucket count and drain start with that model.

**Gate:** `bench_parallel.sh` after rebuild.

**Refute:** parallel UHJ wall still > parallel_hash + noise after F1b.

### Fix F1c — Drop per-row bucket re-routing (align with HashJoin cost model)

**Expected delta:** parallel/serial UserTime within noise of baselines (F1b wall OK; CPU ~33% high).

**Baseline:** HashJoin does not re-derive two-level bucket per row after scatter; ConcurrentHashJoin probes per-shard maps without extra UHJ routing.

**Gate:** `bench_parallel.sh` / `bench_serial.sh` CPU medians ≤ baseline + noise.

**Refute:** CPU still > baseline + noise after F1c.

### Fix F2 — Plumb `max_threads` through `createInMemoryHashJoin` / SpillingHashJoin for Unified (closes B2)

**Expected delta:** with default spill wrapper, UHJ `num_buckets` / parallel build track `max_threads` like ConcurrentHashJoin path.

**Baseline:** `PlannerJoins.cpp` ConcurrentHashJoin / SpillingHashJoin concurrent ctor passes `params.max_threads`; `InMemoryHashJoin.cpp` must pass through to `UnifiedHashJoin` ctor.

**Gate:** after F1, probe with `max_bytes_before_external_join>0` vs `=0` at same threads — wall should be comparable (not forced to 1-bucket). Re-run U2-PARALLEL.

**Refute:** UHJ under spill still reports/behaves as 1 bucket when max_threads=16.

---
