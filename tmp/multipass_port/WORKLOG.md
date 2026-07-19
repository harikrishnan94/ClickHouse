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

## Iteration 3 — Unit 1 implementation + gates (GREEN)

Implemented per the registered design:

- `ColumnsScatter::computePassBits` (`src/Columns/ColumnsScatter.{h,cpp}`) — balanced MSB-first split, ported from the reference.
- `decidePartitionPlan`: cap + warning REMOVED; `pass_bits = computePassBits(partitions, max_fanout_per_pass)`; `max_fanout_per_pass` member (default `ColumnsScatter::MAX_FANOUT_PER_PASS`), test hook `setMaxFanoutPerPassForTests`.
- `postBuildPartitioned`: pass-1 fanout = `2^pass_bits[0] + 1`; multi-pass scatters the saved 16-bit route words as an extra cooperative stream (2 B/row); drop bucket freed before refine; refine wave loop; refined stage in the trace log.
- `refinePassWave`: dynamic group claim, per-group pid derivation `(route >> shift) & mask`, exact per-sub-bucket allocation, SWWC scatter of locators/routes/fixed keys, Layer-1 `scatter` of generic pieces (worker-major pid spans for the first refine, single piece after), group-major output, eager input freeing.
- `planAndAllocateHashTables` skips drop-bucket clearing on refined builds; fills `stats.leaf_row_counts`.
- `leafBuildWorker`: refined-generic branch (one piece per leaf).
- 4 new gtests: `MultiPassForcedPlanLeafParity`, `MultiPassWideLocatorsManyPassesWithDuplicates` (3+ passes, wide locators, duplicates), `MultiPassRightJoinNonJoined` (used flags + non-joined over refined leaves), `MultiPassGenericStringKeys` (generic-mode refine).

### Gate results (raw)

- **G1**: `ninja -C build/reldeb clickhouse unit_tests_dbms > build/reldeb/build_multipass.log 2>&1; echo $?` → `exit=0`; `grep -c warning build/reldeb/build_multipass.log` → `0`. All touched TUs rebuilt (log lines `[2/17] ... ColumnsScatter.cpp.o`, `[4/17] ... PartitionedHashJoin.cpp.o`, `[10/17] ... PartitionedHashJoinBuild.cpp.o`, `[7/17] ... gtest_partitioned_hash_join.cpp.o`).
- **G2**: `build/reldeb/src/unit_tests_dbms --gtest_filter='PartitionedHashJoin.*'` → `[  PASSED  ] 13 tests.` (9 pre-existing + 4 new; log `build/reldeb/test_gtest_multipass.log`).
- **Negative case**: temporarily re-added `bits = min(bits, countr_zero(bit_floor(max_fanout_per_pass)))` in `decidePartitionPlan`, rebuilt, ran `--gtest_filter='PartitionedHashJoin.MultiPass*'` → `4 FAILED TESTS` (all four new tests; log `build/reldeb/test_negcase.log`). Reverted the probe; full suite green again (13/13, `build/reldeb/test_gtest_multipass2.log`); `clickhouse` binary relinked against the reverted source (`build/reldeb/build_multipass_final.log`, exit 0).

Verdict: Unit 1 GREEN. Committing.

## Iteration 4 — Unit 2 PRE-REGISTRATION (correctness under multi-pass)

No production code changes expected in this unit; the work is building and running the checks.

### Expected outcome

The losing-cell query plans 15 bits / 32768 partitions with no cap warning; hash verification agrees with `parallel_hash` on every check, including a genuinely >13-bit multi-pass plan; the shared `HashJoin` machinery is untouched.

### Gate invocations (exact)

- **G3**: `build/reldeb/programs/clickhouse local --path=/mnt/data/join_bench_data --multiquery --send_logs_level=warning < tmp/q_lose_cell.sql 2>&1 | grep -c "Partition plan capped"` → `0`. Plan observation: same query with `--send_logs_level=trace 2>&1 | grep "Partition plan"` → `bits = 15, partitions = 32768, 2 scatter pass(es)`. Refuted by: any cap warning, or a plan below 15 bits.
- **G4a (bench small cells, bit-exact oracle)**: `python3 bep/tools/join_mergetree_bench.py run --path=/mnt/data/join_bench_data --cardinalities=4194304 --multiplicities=1 --ratios=1 --hit-rates=0.5,1 --build-payload-columns=7 --probe-payload-columns=3 --threads=32 --runs=1` → every point `Verification: PASS`, summary `hash_mismatch=0`. (These plan single-pass; the sorted `FORMAT Hash` oracle is capped at 10M output rows, so multi-pass cells cannot use it.)
- **G4b (multi-pass hash verification)**: `tmp/multipass_port/verify_multipass.sql` via `clickhouse local`: LEFT JOIN with duplicate build keys — build `SELECT intDiv(number, 2) AS k, number AS bv FROM numbers(1048576000)` (524288000 distinct keys, every key twice → the `RowRefList` path), probe `SELECT number AS k, number AS pv FROM numbers(600000000)` (75.7M non-matching rows exercise LEFT semantics) — comparing `count()`, `sum(cityHash64(k, pv, ifNull(bv, 42)))` between `join_algorithm='partitioned_hash'` and `'parallel_hash'`. Mechanism proof: `--send_logs_level=trace` grep shows `Partition plan: bits = 15 ... 2 scatter pass(es) (bits per pass [8, 7])` for the partitioned run. Refuted by: any aggregate differing, or the plan line showing 1 pass.
- **Regression gate**: `build/reldeb/src/unit_tests_dbms --gtest_filter='PartitionedHashJoin.*'` → 13/13; `git diff --stat ahj -- src/Interpreters/HashJoin/` → empty AND `git diff --stat fd53e4e604e..HEAD -- src/Interpreters/HashJoin/` → empty.

### What would refute

A hash/aggregate mismatch on any check (routing or locator corruption in the refine pass); a cap warning or sub-15-bit plan on the losing cell (cap not actually lifted); any diff under `src/Interpreters/HashJoin/`.

## Iteration 5 — Unit 2 gates (GREEN)

- **G3**: `... --send_logs_level=warning < tmp/q_lose_cell.sql 2>&1 | grep -c "Partition plan capped"` → `0` (query exit 0; log file `tmp/multipass_port/g3_warning.log` is EMPTY — no warnings of any kind). Plan observation (`tmp/multipass_port/g3_trace.log`): `Partition plan: bits = 15, partitions = 32768, 2 scatter pass(es) (bits per pass [8, 7]), 524288000 rows in 8722 blocks, estimated 524090366 distinct keys`; `Built 32768 leaf hash tables: 524288000 keys ... 32768 carved from one 16.02 GiB slab, 0 heap fallbacks`; stage trace shows `refine passes 120.8/3659.8` wall/thread ms.
- **G4a**: bench run (`tmp/multipass_port/g4a_bench_verify.log`): both points `Verification: PASS (identical sorted output)`, `Summary: wins=2 losses=0 ties=0 fallback=0 invalid=0 errors=0 hash_mismatch=0`.
- **G4b**: `tmp/multipass_port/verify_multipass.sql` via `clickhouse local` (`tmp/multipass_port/g4b_out.tsv`): `partitioned_hash 1124288000 8110256178567749413` vs `parallel_hash 1124288000 8110256178567749413` — identical count and row-hash sum (expected count checks out: 524288000×2 matched + 75712000 LEFT-non-matched). Mechanism proof (`tmp/multipass_port/g4b_trace.log`): `Partition plan: bits = 15, partitions = 32768, 2 scatter pass(es) (bits per pass [8, 7]) ... estimated 562217958 distinct keys` — the verification genuinely exercised the multi-pass path, on the `RowRefList` (duplicate-key) shape and LEFT semantics.
- **Regression gate**: gtests 13/13 (`build/reldeb/test_gtest_unit2.log`); `git diff --stat ahj -- src/Interpreters/HashJoin/` → empty; `git diff --stat fd53e4e604e..HEAD -- src/Interpreters/HashJoin/` → empty.

Verdict: Unit 2 GREEN.

## Iteration 6 — Unit 3 PRE-REGISTRATION (performance validation)

**Noise band (declared up front): effects within max(5%, run-to-run spread of the 3 medians) are "no result".** Comparisons: same binary path (`build/reldeb/programs/clickhouse`), same dataset, same thread counts, `--runs=3` medians, re-established fresh.

### Expected outcome

The D=524M bp=0 cell flips to `partitioned_hash` (mechanism: 32768 partitions put the per-leaf hash table + build rows within the 1.6 MiB L2 budget, restoring the lookup advantage); D=436M/470M margins improve vs `tmp/lose_cell_ab.log`; the other 33 cells unchanged within the noise band, and their logs must show the multi-pass path did NOT trigger (partitions <= 8192 single-pass at D <= 268M; the two other D=524M quick-compare cells also plan 15 bits and may improve — they were already wins).

### Gate invocations (exact)

- **G5**: `python3 bep/tools/join_mergetree_bench.py run --path=/mnt/data/join_bench_data --cardinalities=524288000 --multiplicities=1 --ratios=2 --build-payload-columns=0 --probe-payload-columns=7 --threads=32 --runs=3` → `Winner: partitioned_hash`, margin outside the noise band. Refuted by: `Winner: parallel_hash` or a within-noise margin.
- **G6**: same with `--cardinalities=436207616,469762048` → both `Winner: partitioned_hash`; margins at D≥470M better than the recorded ladder (`tmp/lose_cell_ab.log`: 436M won 1.045x, 470M lost 1.040x).
- **G7**: the 10 quick-compare invocations exactly as the baseline headers (threads 1: [D=33554432 with (ratio=2,bp=1,pp=0), (ratio=1,bp=3,pp=3), (ratio=1,bp=1,pp=7), (ratio=2,bp=7,pp=0)]; threads 16: [D=67108864 ratio=4 bp=0 pp=1, D=134217728 ratio=4 bp=0 pp=7, D=134217728 ratio=2 bp=7 pp=0]; threads 32: [D=524288000 ratio=2 bp=0 pp=7, D=524288000 ratio=4 bp=1 pp=7, D=268435456 ratio=2 bp=7 pp=0]; all m=1, hit=1, runs=3) appended into `tmp/quick_compare_multipass.log`; the 4 sweep invocations (`--cardinalities=67108864,134217728,268435456 --ratios=1,4 --threads={32,64}` × bp=pp={3,7}, runs=3) into `tmp/bp_pp_sweep_multipass_threads_{32,64}.log`; parse with `bep/tools/parse_sweep_log.py`; `grep -c "FALLBACK\|ERROR\|INVALID\|CANNOT_SCHEDULE\|hash mismatch"` over the three logs → `0`. Pass: `partitioned_hash` wins >= 33 of 34 with the D=524M bp=0 cell among the wins; no cell's partitioned median regresses beyond the noise band vs `tmp/quick_compare_fixed.csv` / `tmp/bp_pp_sweep_fixed.csv`.

### What would refute

G5/G6 wins inside the noise band (mechanism not proven); any sweep cell slower by more than the band (refine passes leaking into single-pass plans — would demand a diff review of the bits <= 13 path); nonzero FALLBACK/ERROR/INVALID/CANNOT_SCHEDULE/hash-mismatch count.
