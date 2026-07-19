# WORKLOG — multi-pass scatter port into `PartitionedHashJoin`

Branch: `ahj`. Build dir: `build/reldeb`. Dataset: `/mnt/data/join_bench_data`.

## Iteration 0 — setup & environment verification (pre-work)

Goal: verify starting points from the mission prompt before any change.

Commands and raw results:

- `git status --porcelain` → only `?? contrib/base64/` (untracked directory, not a registered submodule — `git submodule status contrib/base64` errors with "pathspec did not match"). The tracked tree is CLEAN. Decision: proceed; the prompt's "stop if dirty" clause targets the requester's uncommitted tracked work, and there is none. `contrib/base64/` is pre-existing local state; I will never `git add` it. Revisit trigger: if any commit ever shows contrib changes, abort and investigate.
  - Note: the IDE-provided git snapshot at session start listed 5 modified files (`src/Interpreters/HashJoin/AddedColumns.{cpp,h}`, `PartitionedHashJoinProbeImpl.h`, `RowRefs.{cpp,h}`), but the live `git status` shows them clean — the snapshot was stale (taken before the requester committed). Live state is authoritative.
- `git branch --show-current` → `ahj`. `git log --oneline -1` → `fd53e4e604e` — matches the commit the baselines (`tmp/quick_compare_fixed.csv`, `tmp/bp_pp_sweep_fixed.csv`) were produced on.
- `git stash list` → exactly one stash: `stash@{0}: On radix-join-bandwidth-model: radix-join multi-pass scatter: ...` — matches the prompt. Will only `git stash show -p`, never pop/apply.
- Existence checks all pass: `build/reldeb/programs/clickhouse`, `build/reldeb/src/unit_tests_dbms`, `/mnt/data/join_bench_data/{data,metadata,...}`, `bep/tools/{join_mergetree_bench.py,parse_sweep_log.py,summarize_sweep.py}`, `tmp/{q_lose_cell.sql,lose_cell_ab.log,quick_compare_fixed.csv,bp_pp_sweep_fixed.csv}`.

Status: setup complete, moving to discovery (read-only).

## Iteration 1 — discovery (read-only)

Studied:

- `tmp/multipass_port/ref_RadixHashJoin.cpp` (exported from `radix-join-bandwidth-model`) and `tmp/multipass_port/stash0.diff` (`git stash show -p 'stash@{0}'` — read only, never popped): `computePassBits(p_star, f_max)` balanced split; `scatterFirstPass` consumes the top bits, `scatterRefinePass` group-claims partitions dynamically (atomic counter) and slices `(route >> shift) & mask`; group-major output makes the final index equal `route >> (32 - total_bits)` regardless of the split.
- `PartitionedHashJoin` current pipeline: `decidePartitionPlan` clamps `bits` to `countr_zero(MAX_FANOUT_PER_PASS) = 13` with the warning G3 greps for. `postBuildPartitioned` waves: histogram → allocate → scatter → hash-table plan+slab → leaf builds. Bucket ids from saved 16-bit routes, `routes[i] >> (16 - bits)`, drop bucket at index `partitions`. Fixed mode scatters raw key bytes cooperatively; generic mode scatters per-worker Layer-1 pieces; locators (narrow 4-byte / wide 8-byte) always cooperative.
- Probe side (`PartitionedHashJoinProbeImpl.h` ~300): `leaf_ids[i] = UInt16(route_words[i] >> (32 - bits))` — valid for any `bits <= 16`; the plan loop already bounds `bits < 16`+1 = 16. **Conclusion: no probe change needed; the saved 16 route bits are sufficient for any reachable plan (bits <= 16).** This answers the prompt's width question: no widening required, but the ceiling must be asserted.
- Drop-bucket subtlety found: a hypothetical single-pass 16-bit plan would overflow the UInt16 bucket-id at drop index 65536; multi-pass never has per-pass fanout above 8192+1, so all per-pass pids stay UInt16-safe.

## Iteration 2 — Unit 1 PRE-REGISTRATION (before any implementing change)

### Design decisions (ambiguity calls)

- **D1 — carry routes vs recompute:** the refine pass derives sub-bucket pids from the pass-1-scattered 16-bit route words (2 B/row extra scatter traffic in non-final passes) instead of recomputing `computeJoinRouteWords` on the scattered key columns. Why: uniform across every key type (fixed/generic/String/LowCardinality), provably consistent with pass 1 (same word), and 2 B/row is cheaper than a re-hash. Revisit trigger: refine-pass scatter bandwidth showing up as a regression in G5/G7.
- **D2 — total-bits ceiling stays 16** (the existing plan-loop bound): forced by the saved 16-bit routes and the UInt16 probe leaf ids; enough for the entire acceptance surface (15 bits at D=524M). Assert `bits <= 16` and per-pass `bits_done + b <= 16`.
- **D3 — refine output is final-leaf-indexed, group-major** (`leaf = (g << b) | p`), so `planAndAllocateHashTables` and `leafBuildWorker` run on the refined arrays with minimal change; the drop bucket is dropped after pass 1 (refine never sees skipped rows). Generic mode after refine has ONE piece per (column, leaf) — a small `refined` branch in `leafBuildWorker`.
- **D4 — single-pass plans (`pass_bits.size() == 1`) take the existing code path**: pass-1 parametrization uses `pass1_bits`/`pass1_partitions` which equal `bits`/`partitions` there; routes are NOT scattered; the refine wave does not run.
- **D5 — test hook:** `setMaxFanoutPerPassForTests(size_t)` overrides the per-pass ceiling in `decidePartitionPlan` (mirrors `setReserveSafetyFactorForTests` convention); `BuildStats` gains `pass_bits` and per-leaf `leaf_row_counts` so tests can assert the split and per-leaf row parity directly.
- **D6 — `computePassBits` lives in `ColumnsScatter`** (next to `MAX_FANOUT_PER_PASS`), ported from the reference with the same balanced split (15 → 8+7).

### Expected outcome

Build (G1) exits 0 with no warnings in touched files; all existing `PartitionedHashJoin.*` gtests stay green; new gtests (forced multi-pass via the fanout hook: 2-pass, 4-pass, wide-locator, and a generic-mode/String-key variant) pass, asserting: partitions NOT capped by the forced per-pass ceiling, `pass_bits` split correct, per-leaf row counts identical to the same build's single-pass plan, one slab allocation, `region_carves == partitions`, zero heap fallbacks, exact multiset join output.

### Gate invocations (exact)

- G1: `ninja -C build/reldeb clickhouse > build/reldeb/build_multipass.log 2>&1; echo $?` → `0` (subagent scans log for warnings in touched files; also build `unit_tests_dbms`).
- G2: `build/reldeb/src/unit_tests_dbms --gtest_filter='PartitionedHashJoin.*'` → all pass, including the new multi-pass tests.
- Negative case: with the port in place, temporarily re-introduce the single-pass clamp inside `decidePartitionPlan` (bits = min(bits, forced-cap bits) — the pre-port behavior generalized to the hook) and re-run the new gtest → it must FAIL (partitions capped ≠ expected / parity broken). This proves the test detects the absence of multi-pass, not plumbing.

### What would refute the design

- A refined leaf's row count differing from the single-pass plan's same leaf (routing inconsistency between `(g << b) | p` and `route >> (16 - total_bits)`).
- Any existing gtest breaking (single-pass path perturbed).
- The forced-cap build reporting `partitions` equal to the forced per-pass ceiling (cap still effective — port not engaged).
