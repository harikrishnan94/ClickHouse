# Stage 0b — correctness net and base binary for the clause-major probe

Baseline: `fa77c89eb39b99d61f94670755a3c18a81025691`, BuildID
`481f8ae4514186425cad2d2db85a3cdafcbc5884` (asserted via `readelf -n` before copying; the
assertion is what the whole gate chain compares against, so Stage 0 would have aborted on
mismatch). `clickhouse.base3` is a byte copy of that binary; `clickhouse.pre` untouched.

## Gate results

- BuildID: asserted, matches.
- Goldens: **166 cells** recorded on the base binary (`STAGE0B_GOLDENS_RECORDED`), then
  re-verified against `clickhouse.base3` on port 8122 **and** the current build on 8121:
  `GOLDENS_MATCH cells=166` both, `JOB_EXIT=0` both.
- Timed matrix: **151 cells** = 144 matrix + 7 special timed (3 tiny + 4 new real-sized).
- Rig: `DUAL_SERVERS_READY` (8121 = current build, 8122 = `clickhouse.base3`, both report
  the same BuildID). A/A report `stage0b-rig-204011`: **A1/A2/A3/A4 all GREEN**
  (cross-arm checksums agree exactly; A/B deltas within noise, as identical binaries must be).

## Golden additions (23 -> 22 special, 166 total)

| group | cells | what it pins down |
|---|---|---|
| `multi2` | 3 | 2-disjunct `ON l.k = r.k OR l.a = r.a` on `p_medium_hi`/`b_medium`, INNER / LEFT / LEFT SEMI, full-column checksum incl. right cols |
| `multi3` | 3 | 3-disjunct `... OR l.s = r.s`, same kinds; mixed key kinds merge to the `hashed` map |
| `semi_right` | 2 | single-clause LEFT SEMI / LEFT ANTI checksumming **right columns** (N21; the matrix never does, `Cell.has_right_cols` is False there) |
| `nullable_multi` | 2 | multi-clause where the `k` disjunct carries a null map and the `a` disjunct does not (per-clause skip, N3): INNER, LEFT SEMI |
| `c13_split` | 1 | high-duplication multi (8 keys x 250 dup rows) with `max_joined_block_size_rows=1024`: probe stops mid-block, `joinBlockImpl` splits; deterministic (512000, chk) |
| `addfilter_extra` | 5 | `flag_per_row` true via RIGHT / FULL / multi-disjunct filter, false with `max_joined_block_size_rows=8` (early exit + resize path), medium INNER on new `uhj_stage0_af_*` tables (~50% filter pass) |

New timed cells (`all_timed_cells`): `multi2all|u64|t1|medium`, `multi2semi|u64|t1|medium`,
`addfilter|u64|t1|medium` (on the new `uhj_stage0_af_left/right`, 200k/100k rows; on the
matrix tables every column is a function of `k`, so any `l.x < r.y` filter is degenerate
there), `multi2semi|u64|t16|large` (short-circuit fold at scale).

## Reachability findings (why some planned cells do not exist)

- **K = 0 (N22) is unreachable from SQL.** `key_getter_vector` gets one getter per
  `join_on_keys` entry and `join_on_keys` gets one entry per `getClauses()` entry, so empty
  getters require zero clauses. Zero-equality-key joins never reach UHJ: the planner routes
  them to the constant/cartesian result path before `tryCreateJoin` is consulted
  (CPU-profiler evidence on the base binary: `INNER JOIN ... ON l.a < r.a` over 44.5M result
  rows shows only `DB::ConstantJoinCartesianResult::next` + `DB::JoiningTransform`, zero
  `DB::Unified` samples in 57; `LEFT JOIN ... ON false` shows
  `DB::ConstantJoinUnmatchedLeftRowsResult::next`). Even if a query got there, the UHJ
  constructor itself rejects it: `HashJoin.cpp:307-308,327-328` throw for
  `getClauses().empty()`. Stage 3 must still implement the K=0 emit semantics defensively
  (one miss per row), but no golden can observe it from SQL.
- **Multi-disjunct is only supported for INNER / LEFT / LEFT SEMI.** RIGHT, FULL, ANY and
  ANTI with >1 clause throw `assertHasOneOnExpr` ("Expected to have only one join clause",
  LOGICAL_ERROR) — measured on the base binary. The planned RIGHT/FULL/ANTI/ANY multi cells
  are therefore impossible by design, not skipped.
- **Multi-disjunct on NULLABLE keys is only INNER / LEFT SEMI.** LEFT and RIGHT both throw
  the same `assertHasOneOnExpr` error. (RIGHT appears to work with default settings only
  because `query_plan_join_swap_table` rewrites it; the pinned harness settings disable the
  swap, which is also why the nullable SEMI checksum differs between pinned and default
  settings — goldens are recorded under the pinned settings, as intended.)
- **The 3rd disjunct uses the string column.** On the benchmark tables `b = k % 1000` has
  1000 distinct values, so `l.b = r.b` fans out to ~1000 matches per probe row at medium
  cardinality (measured: >120 s for both `hash` and `unified_hash`). `l.s = r.s` is 1:1 with
  `k`, keeps the output bounded (dedup to 2 rows/left row, same checksum as 2-disjunct), and
  exercises the merged-`hashed`-map multi path (`mergeJoinMethods`).

## What is in this commit

Harness + goldens only. No `src/` changes. The perf gate for Stages 1+ compares against
`clickhouse.base3` on 8122 (`start_ab_servers.sh` `PRE_BIN`).
