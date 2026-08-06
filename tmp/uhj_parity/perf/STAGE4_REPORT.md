# Stage 4 — additional filter onto clause-major, delete `PreSelectSink` and `SequentialMultiLookup`

Baseline `clickhouse.base3` (BuildID `481f8ae4514186425cad2d2db85a3cdafcbc5884`);
post BuildID `c41db5d3c8eae83e777dba81b9e3a51aa98f30a4`.

## What changed

The additional-filter pre-select phase becomes a clause-major collect pass, mirroring
Stage 3's shape: `lookupBatch` fills `outcomes[k]` per clause per batch, then the new
`collectAdditionalFilterBatch` (`NO_INLINE`) expands every match into `selected_rows` /
`row_replicate_offset`, exactly what `PreSelectSink` used to do.

- **No short-circuit fold on this path.** `stop_after_first_match` is hard `false` here -
  the filter pass needs every pre-selected right ref regardless of clause order (SEMI/ANY's
  first-match rule is applied AFTER filtering, not during collection) - so there is no
  `sc_matched`/`sc_combined` machinery at all, unlike Stage 3's plain multi-clause driver.
- **`flag_per_row` stays a runtime bool**, selecting `KnownRowsHolder<true>` or
  `KnownRowsHolder<false>` via two explicit branches at the call site, exactly as before -
  only what runs inside each branch changed (a templated lambda now, calling the clause-major
  batch loop instead of `SequentialMultiLookup::run`).
- **F7: `find_results` (a `std::vector<FindResult>`) is replaced with plain
  `std::vector<size_t> selected_offsets`.** The old shape was latently UB-adjacent under a
  clause-major rewrite: a `FindResult`'s `Mapped*` pointer would have to point at a
  stack-local rebuilt from a recorded word, dead the moment the collecting function returns,
  while the ONE thing ever read back from it (`used_flags.setUsed<need_flags, false>(...)`
  reading only `getOffset()`, never `getMapped()` - verified against `JoinUsedFlags.h`)
  doesn't need the pointer at all. Storing the plain offset removes the dangling-pointer
  shape entirely. Populated only when `!flag_per_row` (the single-clause case this is ever
  read in), same guard `consumeProbeBatch` uses for `ProbeOutcomes::offset`
  (`join_features.need_flags ? outcomes[k].offset[j] : 0`). The one remaining read site
  reconstructs a `FindResultImpl<const Mapped, need_flags>(nullptr, true, offset)` inline and
  passes it to `setUsed` - the `nullptr` value pointer is never dereferenced on this path.
- **Per-clause prefetch**, matching Stage 3's plain multi-clause driver: one `ProbePrefetch`
  instance per clause, constructed once for the whole call.
- `buildAdditionalFilter` and the post-filter emit loop are **unchanged in behavior** - only
  the producer of `selected_rows` / `selected_offsets` / `row_replicate_offset` changed. Both
  asserts (`selected_rows.size() == current_added_rows`,
  `left_block_rows == row_replicate_offset.size()`) are untouched.
- Deleted `PreSelectSink` (kept `PreSelectedRows`) and, finally, `SequentialMultiLookup`
  (both `run` overloads and `runImpl`) from `ProbeLookup.h` - additional-filter was its last
  caller.

## Landing assertions

| symbol (all UHJ TUs) | before | after | verdict |
|---|---|---|---|
| `DB::Unified::PreSelectSink` | 256 (`LeftHashJoinAll`) + more | **0** | PASS |
| `DB::Unified::SequentialMultiLookup` | 4096 total | **0** | PASS |
| `DB::Unified::lookupBatch` (`LeftHashJoinAll.cpp.o`) | 64 | **64** (unchanged) | PASS |

`joinRightColumnsWithAdditionalFilter` count on `LeftHashJoinAll.cpp.o`: 224.

`.text`, `LeftHashJoinAll.cpp.o`: 1,670,632 (Stage 3) -> **1,292,284 (-22.7%)**. UHJ total
`.text`: 43,742,420 (Stage 3) -> **37,485,664 (-14.3% this stage, -39.6% cumulative from the
Stage 0 baseline's 62,015,808)**. UHJ total `.o`: 297.7 MB -> 266.5 MB.

## Correctness gate

`GOLDENS_MATCH cells=166`. The decisive cells are the `addfilter_extra` group: RIGHT / FULL /
multi-disjunct (all `flag_per_row == true`), INNER with `flag_per_row == false` (the branch
that now reads `selected_offsets` instead of `find_results`), and the small-limit case
(`max_joined_block_size_rows = 8`, exercising the early-exit / resize path this stage
preserved). Every one matches the base binary's checksum.

## Performance gate

Full matrix (151 cells, reps=5) vs `clickhouse.base3`:

| slice | `probe_us` median | `cpu_us` median |
|---|---|---|
| Non-SEMI/ANTI matrix (96 cells) | -0.2% (worst +4.1%) | -0.2% (worst +4.5%) |
| SEMI/ANTI slice (48 cells) | +3.0% | +1.7% |

- **Non-SEMI/ANTI is unaffected**, as expected - this stage only touches the
  additional-filter path, which none of INNER/LEFT/RIGHT/FULL's plain (non-filtered) cells in
  the matrix exercise. `INNER`'s best -35.8% is Stage 3's `multi2all` clause-major win,
  unrelated to this stage's changes and unchanged by them.
- **SEMI/ANTI slice is the same pre-existing Stage 2 regression, not compounded**: +3.0%/
  +4.0%(ANTI)/+2.2%(SEMI) here vs the +2.4%/+2.6%/+1.0% measured in Stage 3 and the original
  +2.9%/+3.7% from Stage 2 - all within run-to-run noise of each other, no trend across
  stages. This path is untouched by Stage 4.
- **The real-sized additional-filter timed cell, the actual target of this stage, is flat**:

  | cell | `probe_us` pre -> post | `cpu_us` pre -> post |
  |---|---|---|
  | `addfilter\|u64\|t1\|medium` (200k/100k rows, ~50% filter pass) | 3,974 -> 3,914 us (-1.5%) | 7,451 -> 7,213 us (-3.2%) |

  Both deltas are well inside the noise band for cells this size. Stage 4 is a correctness/
  robustness change (F7's UB-adjacency fix, plus adopting the same clause-major shape as
  Stage 3) rather than a targeted performance change, so flat is the expected and desired
  result - unlike Stage 3's `multi2all`, there is no locality win to expect here since the
  additional-filter path was never row-major-interleaved across clauses in a way the
  clause-major batching would improve (it already visited clauses in a fixed per-row loop
  whose cost is dominated by `buildAdditionalFilter`'s expression evaluation, not the probe).

**Verdict**: neutral outside the already-reported, unchanged SEMI/ANTI regression - exactly
what a pure structural change (recording-path unification + the F7 correctness fix) should
produce.
