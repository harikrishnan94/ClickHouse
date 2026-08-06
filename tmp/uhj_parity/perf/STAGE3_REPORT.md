# Stage 3 — clause-major multi-clause driver, delete `MultiEmitSink`

Baseline `clickhouse.base3` (BuildID `481f8ae4514186425cad2d2db85a3cdafcbc5884`);
post BuildID `53821ace7cd6abb58d98b5ab71b3a4ca221a8aeb`.

## What changed

Replaced the row-major multi-clause driver (`SequentialMultiLookup::run` + `MultiEmitSink`,
one row visiting all its clauses before the next row starts) with the clause-major shape:
per batch of `PROBE_BATCH_ROWS`, `lookupBatch` runs once per clause over the WHOLE batch
(filling `outcomes[k]`), then `emitBatch` (new, `NO_INLINE`) consumes all K outcomes for that
batch, one row at a time. `lookupBatch` is the exact same function the single-clause probe
uses - multi-clause is just K calls to it, so it adds nothing to the 64-instantiation count.

- **Short-circuit fold (P8, N20, F1).** ANY/SEMI joins used to `break` out of the row-major
  clause loop on first match. The clause-major driver instead folds already-matched rows
  into the NEXT clause's skip mask: `sc_matched` (batch-position indexed, never handed to
  the lookup) tracks which rows in the current batch already matched; when any row has,
  `sc_combined` (sized to the SOURCE row domain, same domain `buildRowSkipData` already
  uses) is built by ORing the existing skip byte with `sc_matched` at each row's absolute
  index, and that becomes the skip pointer for the next clause. No second mask parameter or
  template axis reaches the lookup (P8) - it is always the same single `skip_data` pointer.
  `sc_combined` needs no clearing between batches: every position is written before it is
  read, in the same batch, since each absolute row position is unique across the whole probe
  call.
- **Per-clause prefetch (F11), a deliberate change.** The old driver only ever prefetched
  `mapv[0]`. Each clause now gets its own `ProbePrefetch` instance, constructed once for the
  whole probe call (not per batch, so the look-ahead calibration - which fires once at an
  absolute row - still works for `begin > 0`).
- **No offset arrays (N23).** `lookupBatch</*need_flags=*/false>` is called unconditionally
  on this path, regardless of the TU's `need_flags`, exactly as the plan requires.
- **`emitBatch`** reconstructs a `FindResult` from each clause's recorded word (the same
  `mappedFromWord` / reinterpret-cast technique the single-clause `consumeProbeBatch` uses),
  with `is_last_disjunct = (k + 1 == num_clauses)` (N5, positional) and `current_offset <
  max_joined_rows` checked at the START of each row (F12) so the row that crosses the limit
  is still fully emitted. The offset passed into the reconstructed `FindResult` is always 0:
  verified against `JoinUsedFlags.h` that `setUsed`/`setUsedOnce` with `flag_per_row = true`
  never call `getOffset()`, so the value is inert regardless of the TU's `need_flags`.
- Deleted `MultiEmitSink`. **`SequentialMultiLookup` is untouched** - its only remaining
  callers are the two `run<...>` calls inside `joinRightColumnsWithAdditionalFilter`
  (verified by source grep), exactly as F2 requires; it is deleted in Stage 4.

## Landing assertions

| symbol | before | after | verdict |
|---|---|---|---|
| `DB::Unified::MultiEmitSink` (all UHJ TUs) | 256 (`LeftHashJoinAll`) + more | **0** | PASS |
| `DB::Unified::SequentialMultiLookup` (all UHJ TUs) | 512 (`LeftHashJoinAll`) + more | **4096** (nonzero, additional-filter only) | PASS |
| `DB::Unified::lookupBatch` (`LeftHashJoinAll.cpp.o`) | 64 | **64** (unchanged - multi reuses it) | PASS |

`.text`, `LeftHashJoinAll.cpp.o`: 1,976,908 (Stage 2) -> **1,670,632** (-15.5%). UHJ total
`.text`: 62,015,808 (Stage 0 baseline) -> **43,742,420 (-29.5%, -17.4 MB)**. UHJ total `.o`:
~315.0 MB -> 297.7 MB.

## Correctness gate

`GOLDENS_MATCH cells=166`. The decisive cells here are `multi2`/`multi3` (2- and 3-disjunct
INNER/LEFT/LEFT SEMI on the medium tables), `c13_split` (the high-duplication + small
`max_joined_block_size_rows` cell, which proves the batch-boundary early exit and
`joinBlockImpl`'s block-split still land on the same row), and `nullable_multi` (per-clause
skip via a null map on one clause and none on the other) - every one of these matches the
base binary's checksum byte-for-byte, on both the fold path (SEMI, which exercises the
short-circuit merge) and the non-folding path (INNER/LEFT, which exercises the
"skip straight through" branch).

## Codegen-parity gate

`SequentialLookup::runImpl`, u64 TwoLevel + String, matched by exact non-closure template
signature (map-grower type x key-getter's nullable-flag x sink) rather than address order -
see the methodology note below.

- **u64**: 4/4 matched groups have identical loop and branch counts (2 groups exact
  instruction count, 2 groups -8 - the same register-allocation delta seen in every prior
  stage, not a new effect).
- **String**: 4/4 matched groups are **byte-for-byte identical** (instructions, branches,
  loops, calls all equal).

No loop-count change anywhere (the P1 fingerprint stays absent).

**Methodology note - a finding worth recording.** A naive `llvm-nm | rg <pattern> | head -N`
comparison across binaries broke at this stage: the post-Stage-3 binary has far FEWER unique
`runImpl` symbol names matching a given `(KeyGetter, Map)` substring than the base binary (6
vs 64, for both u64 and String). This is not a regression - it is Stages 1-3 paying off more
than their own per-TU counts suggested. Before Stage 1, `PrefetchAt` was a lambda whose
closure type embedded the enclosing `joinRightColumns` instantiation
`(KIND, STRICTNESS, need_filter, ...)`, so every one of the ~32 join-kind TUs minted its own
distinct `runImpl` symbol even for an identical `(Map, KeyGetter, Selector)` - hence base3's
64. Since Stage 1, `PrefetchAt` is the named `ProbePrefetch<Map, KeyGetter, Selector>`, whose
type depends on NOTHING kind/strictness-specific; the mangled name for a given
`(Map, KeyGetter, Selector, sink)` is now IDENTICAL across every TU that uses it, so the
linker's ODR/COMDAT folding collapses all ~32 TU-local instantiations into the single kept
definition - global sharing, not just the per-TU 128 -> 64 the plan measured. The gate was
still applied correctly: matched groups by their exact non-closure signature (verified
identical map-grower type, nullable flag, and sink on both sides) rather than trusting
address order, per the DO NOT list's warning about identical-code-folding aliasing.

## Performance gate

Full matrix (151 cells, reps=5) vs `clickhouse.base3`:

| slice | `probe_us` median | `cpu_us` median |
|---|---|---|
| Non-SEMI/ANTI matrix (96 cells) | -0.2% (worst +5.5%, noise-tested clean) | -0.3% (worst +3.9%, noise-tested clean) |
| SEMI/ANTI slice (48 cells) | +2.4% | +1.1% |
| Real-sized multi timed cells (3) | see below | see below |

- **Non-SEMI/ANTI is unaffected**, as expected - Stage 3 only touches the multi-clause path,
  and every single-clause kind (INNER/LEFT/RIGHT/FULL) shows medians within +-0.6%. The one
  worst-case cell, `INNER|str|lo|t1|medium` at +3.9% `cpu_us`, passes the noise test
  (overlapping pre/post ranges, delta within IQR band) - single-clause code is untouched by
  this stage, so this is measurement noise, not an effect.
- **SEMI/ANTI slice is the same pre-existing regression from Stage 2**, not made worse:
  `probe_us` +2.4% here vs +2.9%/+3.7% (SEMI/ANTI) measured in Stage 2 at 10 reps; `cpu_us`
  +1.1% here vs +1.3%/+1.8% there. Both within run-to-run noise of each other. Stage 3's
  clause-major change did not touch the single-clause SEMI/ANTI path at all, so this is
  exactly the expected outcome - the STAGE2_REPORT.md fallback discussion still applies
  unchanged.
- **Real-sized multi cells - a genuine, large win**:

  | cell | `probe_us` pre -> post | delta |
  |---|---|---|
  | `multi2all\|u64\|t1\|medium` (2-disjunct ALL) | 647,914 -> 395,044 | **-39.0%** |
  | `multi2semi\|u64\|t1\|medium` (2-disjunct SEMI) | 92,921 -> 92,363 | -0.6% (noise) |
  | `multi2semi\|u64\|t16\|large` (2-disjunct SEMI, 16 threads) | 2,335,201 -> 2,205,385 | -5.6% |

  `multi2all`'s -39% is not noise: pre = [640,526 .. 660,635], post = [389,189 .. 403,966],
  non-overlapping ranges, both `probe_us` and `cpu_us` (-35.0%) agree. This is the payoff the
  restructure targets directly - the old row-major driver interleaved both clauses' hash-table
  walks per row (no locality between one row's two lookups); the clause-major driver walks
  clause 0 across the whole batch, then clause 1 across the whole batch, which is exactly the
  batching effect `lookupBatch`'s prefetcher and cache locality were designed around. The SEMI
  variant sees a smaller effect because ANY/SEMI's short-circuit fold means the second clause
  is looked up for far fewer rows to begin with (most rows already matched on clause 0), so
  there is less batched work to speed up.

**Verdict**: G-probe/G-cpu pass cleanly outside the SEMI/ANTI slice (which is Stage 2's
already-reported, already-accepted regression, unchanged by this stage); the new
multi-clause path shows a large real speedup on the shape it targets (2-disjunct ALL) and no
measurable movement on the SEMI variant beyond noise.
