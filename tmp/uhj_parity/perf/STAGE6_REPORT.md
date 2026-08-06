# Stage 6 — final verification and report

Baseline `clickhouse.base3` (`fa77c89eb39`, BuildID `481f8ae4514186425cad2d2db85a3cdafcbc5884`).
Final HEAD: `60c07733196` (Stage 5), BuildID `2552d1e642d296d5078d88f40d9b8b292c65da0a`.
Stage 6 makes no source changes - verification only.

## Goldens on the final binary

`GOLDENS_MATCH cells=166` (same result already re-verified at the end of Stage 5, on the
same binary this report covers - no intervening code change since).

## Stateless join suite

**UHJ smoke tests, via the real `clickhouse-test` harness** (not the ad-hoc comparison
scripts used during earlier exploration) - pointed at the already-running Stage-5-binary
server via `CLICKHOUSE_PORT_TCP`/`CLICKHOUSE_PORT_HTTP`, with `--no-random-settings
--no-random-merge-tree-settings` (the randomizer otherwise duplicates client arguments the
test scripts already set, e.g. `--max_threads`, which the client rejects):

- `04658_unified_hash_join_equivalence`: **OK** (2.74 s)
- `04659_unified_hash_join_parallel_build`: **OK** (6.45 s)

**Join-name subset**, 155 tests matching
`(hash_join|any_.*join|all_.*join|semi.*join|anti.*join|full.*join|inner.*join|left.*join|right.*join)`
(same harness, `-j 8 -r 3 -t 90`):

**137 passed, 6 skipped, 12 failed** - every failure traced to a missing local
dependency, none to UHJ probe logic:

| failure category | count | example |
|---|---|---|
| `NO_ZOOKEEPER` (ReplicatedMergeTree needs ZooKeeper) | 2 | `03275_pr_any_join`, `03261_pr_semi_anti_join` |
| `CLUSTER_DOESNT_EXIST` (parallel-replicas / distributed test cluster not configured) | 4 | `04303_empty_left_join_parallel_replicas`, `03709_parallel_replicas_right_join_with_distributed`, `03452_array_join_global_right_join_parallel_replicas`, `03574_parallel_replicas_last_right_join` (connection refused to a second node) |
| missing `test.hits` / `test.visits` reference dataset (not loaded in this sandbox) | 5 | `00042_any_left_join`, `00043_any_left_join`, `00044_any_left_join_string`, `00074_full_join`, `00075_left_array_join` |
| `ACCESS_STORAGE_FOR_INSERTION_NOT_FOUND` (no writable user-directory configured) | 1 | `04519_select_access_rights_join_union_rewrite` |

This is exactly the plan's pre-acknowledged "known ZooKeeper/cluster environment failures,"
documented here rather than investigated further - none of the 12 exercise the UHJ probe
restructure at all (they fail before a join even executes, on missing infrastructure).

## Full timed matrix A/B, 7 reps, vs `clickhouse.base3`

- **A1 (arm assertion): 0 wrong arms** across 302 assertion runs (zero misattributed to the
  wrong probe family); 12 `UNKNOWN` are all sub-2-CPU-sample tiny cells, same pattern every
  stage has shown.
- **A2 (arms agree): GREEN**, no mismatches.
- **A3 (A/A calibration): GREEN**, both calibration cells within the noise band.
- **A4 (the actual comparison):**

| slice | `probe_us` median | `cpu_us` median |
|---|---|---|
| Full 144-cell matrix | -0.1% | +0.2% |
| Non-SEMI/ANTI (96 cells) | -0.3% (worst +3.5%, best -4.0%) | -0.2% (worst +2.9%, best -5.6%) |
| SEMI/ANTI slice (48 cells) | +2.6% (worst +14.9%, best -11.5%) | +1.3% (worst +7.4%, best -6.3%) |

By kind: `INNER`/`LEFT`/`RIGHT`/`FULL` all sit within +-0.8% median (single-clause code Stages
3-5 never touch); `LEFT SEMI` +2.4%/`LEFT ANTI` +2.8% - the same Stage 2 regression, still
unchanged in magnitude at 7 reps as it was at 5.

**Real-sized special cells** (the two cells that exercise the restructure's own logic
directly, at matrix scale):

- `multi2all|u64|t1|medium` (2-disjunct ALL, the clause-major driver's target shape):
  **-37.7%** `probe_us`, pre = [633.6k .. 657.2k] vs post = [394.5k .. 400.9k] - fully
  non-overlapping across 7 reps, confirming Stage 3's win as real and stable.
- `addfilter|u64|t1|medium` (the additional-filter collect pass): **+0.3%**, flat as
  Stage 4 predicted.

**Verdict**: the restructure is a net performance improvement outside the one deliberately
accepted, previously-reported SEMI/ANTI regression (Stage 2's decision, unchanged by every
subsequent stage), and a substantial win (-37.7%) on the multi-disjunct ALL shape the
clause-major design specifically targets.

## Final report against the Stage 0 baseline

### LOC

| file | Stage 0 baseline | final (Stage 5) |
|---|---|---|
| `HashJoinMethodsImpl.h` | not recorded pre-restructure (baseline table only tracked `.o`/`.text`/symbol counts) | 1996 |
| `HashJoinMethods.h` | " | 318 |
| `ProbeLookup.h` | " | 237 |

(The plan's "Verified current state" section did not include a Stage-0 LOC baseline for
these files, so only the final counts are reported; Stage 5's report already carries the
`processMatch`-deletion LOC delta for `HashJoinMethodsImpl.h` specifically.)

### Per-TU `.o` / `.text` (`build/reldeb/src/CMakeFiles/dbms.dir/Interpreters/UnifiedHashJoin/`)

| TU | Stage 0 `.o` | final `.o` | Stage 0 `.text` | final `.text` |
|---|---|---|---|---|
| `LeftHashJoinAll.cpp.o` | 12,724,008 | 9,144,192 (-28.1%) | 2,454,491 | 1,292,284 (**-47.3%**) |
| `LeftHashJoinSemi.cpp.o` | 8,399,128 | 7,515,008 (-10.5%) | 1,538,039 | 1,110,856 (-27.8%) |

`LeftHashJoinAnti.cpp.o` was not separately measured at Stage 0 (the plan's baseline table
only broke out `All` and `Semi`); final: `.o` = 7,241,936, `.text` = 942,544.

**UHJ totals**: `.text` 62,015,808 -> **37,485,308 (-39.6%, -24.5 MB)**. `.o` ~315.0 MB ->
266.5 MB (-15.4%).

### Instantiation counts (canonical `llvm-nm --defined-only <obj> | c++filt | rg -c '<fully-qualified pattern>'`)

Per the plan's Stage 0 table, `LeftHashJoinAll.cpp.o` baseline vs final:

| symbol | Stage 0 baseline (`LeftHashJoinAll.cpp.o`) | final (`LeftHashJoinAll.cpp.o`) |
|---|---|---|
| `probeTwoPhase` | 128 | **0** (deleted Stage 2) |
| `EmitSink` | 0 (was 256 on `LeftHashJoinSemi.cpp.o`) | **0** (deleted Stage 2, all TUs) |
| `MultiEmitSink` | 256 | **0** (deleted Stage 3) |
| `PreSelectSink` | 256 | **0** (deleted Stage 4) |
| `consumeProbeBatch` | 2 | 2 (unchanged) |
| `consumeFusedBatch` | 2 | 2 (unchanged) |
| `SequentialLookup` | 384 | 128 (-66.7%; Stage 1's `run` dispatcher got inlined into `lookupBatch`, its only call site) |
| `SequentialMultiLookup` | 512 | **0** (deleted Stage 4) |
| `lookupBatch` | 0 (did not exist) | **64** (introduced Stage 1; multi-clause and additional-filter both reuse it, adding nothing - confirmed at every stage's landing gate) |

Global totals across every UHJ TU (not part of the plan's per-TU table, included for
completeness): `lookupBatch` = 2624, `SequentialLookup` = 5248, `consumeProbeBatch` = 68,
`consumeFusedBatch` = 4, the new Stage 3/4 functions `emitBatch` = 60 and
`collectAdditionalFilterBatch` = 32, and `probeTwoPhase`/`EmitSink`/`MultiEmitSink`/
`PreSelectSink`/`SequentialMultiLookup` all **0**.

### Emit-body additive-set arithmetic (F14 follow-up, per the plan's Stage 5 note)

The plan's original "Target instantiation shape" claimed emit bodies form "a small additive
set (`need_filter` x `single_clause`, plus `Selector` for ASOF)." After Stages 2-5 the actual
call sites are: `consumeProbeBatch` (non-ASOF, `need_filter` x 1 = 2 per TU), `consumeProbeBatch`
(ASOF, `need_filter` x 1 = 2 per TU, only in ASOF TUs), `consumeFusedBatch` (fused output,
`need_filter` x 1 = 2 per TU, only where `outputIsProbeOutcomes` applies), `emitBatch`
(multi-clause, one per TU), and `collectAdditionalFilterBatch` (additional-filter, one per
TU, further split by `KnownRows` at the call site into 2 instantiations). The single-clause
consume and the multi-clause emit were **not** unified into one function (Stage 5 confirmed
this was optional and left them separate); the additive-set claim holds in shape (nothing
multiplies by key type or join kind, since all live behind `MapsTemplate`/`KIND`/`STRICTNESS`
template parameters at the class level, which is what Stage 0's 32-way `UNIFIED_APPLY_FOR_JOIN_VARIANTS`
macro already fixes per TU) - the emit side was always additive per TU; the win the plan
targeted was specifically de-multiplying the *lookup*, verified throughout at exactly 64 per
TU from Stage 1 onward.

## Summary across all 6 stages

| stage | key deletion | landing highlight | perf verdict |
|---|---|---|---|
| 0 | - | base3 + 166 goldens | A/A GREEN |
| 1 | (lambda closures) | `lookupBatch` = 64 | neutral to faster |
| 2 | `EmitSink`, `probeTwoPhase` | SEMI/ANTI on recording path | SEMI/ANTI regression, accepted per explicit user decision |
| 3 | `MultiEmitSink` | clause-major multi-clause driver | `multi2all` -39% (genuine win) |
| 4 | `PreSelectSink`, `SequentialMultiLookup` | F7 UB-fix; addfilter clause-major | flat (correctness change, not perf-targeted) |
| 5 | `processMatch` | 7 arms folded into 3 emit sites | exactly neutral (-356 B `.text`) |
| 6 | - | final verification | -37.7% on the target shape at 7 reps; SEMI/ANTI regression unchanged and isolated |

Every stage's own correctness gate (`GOLDENS_MATCH cells=166`) and, where required, codegen-
parity gate passed before that stage was committed. The one accepted regression (SEMI/ANTI,
~+2-3% `probe_us`) was explicit user policy from the plan itself ("unify unconditionally; no
fused fallback even if these cells regress"), stayed the same size across every stage that
followed it rather than compounding, and has a named, unimplemented fallback on record for a
future change that wants to revisit the trade-off: P9's byte-recording sink for
`is_anti_join` only - which would help ANTI but not LEFT SEMI, whose recorded word is
genuinely read for its right-column emit (N21); see `STAGE2_REPORT.md` for the full
rationale.
