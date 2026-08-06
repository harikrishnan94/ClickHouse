# Stage 2 — unify SEMI LEFT / ANTI, delete `EmitSink`, inline `probeTwoPhase`

Baseline `clickhouse.base3` (BuildID `481f8ae4514186425cad2d2db85a3cdafcbc5884`);
post BuildID `924f8f8105460ac9552539ef108f3da53f9a9fa2`.

## What changed

- Deleted `split_can_pay` and `EmitSink`. Every single-clause kind, including SEMI LEFT and
  every ANTI, now routes through the recording path (`lookupBatch` + consume).
- Inlined `probeTwoPhase` into `joinRightColumns` and deleted it as a named function, per the
  plan's explicit directive (F9) - the batch loop (both the fused-output branch and the
  ordinary consume branch) now lives directly in `joinRightColumns`. `lookupBatch` and the
  consume functions stay `NO_INLINE`.
- The SEMI LEFT / LEFT ANTI emit arms are unchanged: they still run through
  `consumeProbeBatch` -> `processMatch`'s existing arms (the `else` arm's
  `appendFromBlock(firstRefWord(mapped), ...)` for SEMI LEFT, N21; `addNotFoundRow` with
  `add_missing` for ANTI). Only *when* they run changed - from the same call, straight after
  their own lookup, to a batch later, reading a recorded word.
- Benchmark (`unified_hash_join_probe_loop.cpp`): the `SequentialEmit` driver and `EmitSink`
  use are gone with it; one remaining driver, `SequentialTwoPhase`. `verifyShape`'s
  emit-vs-twophase cross-check is replaced by the analytic expectation + a multi-thread-build
  structural digest check (both already existed alongside the removed comparison).
- Harness marker update: `expected_probe` returns `uhj_split` for every kind on any non-base
  arm (the post-Stage-2 binary has no fused loop left to detect); the base arm keeps the old
  `ALWAYS_FUSED_KINDS` mapping to `uhj_fused` for the historical baseline comparison.

## Landing assertions (canonical `llvm-nm --defined-only | c++filt`, summed across every UHJ TU)

| symbol | baseline | stage 2 | verdict |
|---|---|---|---|
| `DB::Unified::EmitSink` | 256 (Semi) + more | **0** | PASS |
| `DB::Unified::probeTwoPhase` | 128 x N | **0** | PASS |

Per-TU, `LeftHashJoinSemi.cpp.o` / `LeftHashJoinAnti.cpp.o` now show `lookupBatch == 64` and
`consumeProbeBatch == 2` (same shape as `LeftHashJoinAll.cpp.o`):

| TU | `.o` (stage 1 -> stage 2) | `.text` (stage 1 -> stage 2) |
|---|---|---|
| `LeftHashJoinSemi.cpp.o` | 8,251,016 -> 7,805,864 (-5.4%) | 1,315,208 -> 1,223,032 (**-7.0%**) |
| `LeftHashJoinAnti.cpp.o` | (not measured stage 1) -> 7,753,568 | -> 1,243,256 |
| `LeftHashJoinAll.cpp.o` | 12,033,600 -> 11,745,928 (-2.4%) | 1,976,908 -> 1,943,900 (-1.7%) |

The large drop expected from removing 256 `EmitSink` bodies materializes as the `.text`
delta on the Semi/Anti TUs (their `EmitSink` instantiation count was 256 before Stage 1 even
started counting it against `lookupBatch`; Stage 1 already removed the `need_filter` axis
from what became `EmitSink`'s remaining callers, so the full 256-body removal is folded into
both the Stage 1 and Stage 2 `.text` drops rather than showing as one large Stage-2-only
number).

## Correctness gate

`GOLDENS_MATCH cells=166` on the post binary. The decisive cells are the Stage 0
`semi_right` group (`LEFT SEMI` / `LEFT ANTI` checksumming right columns) - these are exactly
what N21 says the emit must still produce, and they match the base binary's checksums, so the
recording-path SEMI/ANTI emit is byte-identical to the old fused emit.

## Codegen-parity gate

`SequentialLookup::runImpl` before/after, u64 TwoLevel + String, all 4 (sink x ICF group)
pairs per key type:

- **u64**: loop counts identical (6, 4, 2, 2); branch counts identical; instruction counts
  match on 2 groups, -8 on 2 groups (same delta as Stage 1 - register allocation, not a
  fingerprint of the P1 regression).
- **String**: exact parity on all 4 groups (263/48/4/1, 255/47/4/1, 271/48/4/1, 328/50/4/3).

No loop-count change anywhere.

## Performance gate

Full matrix (151 cells, reps=5) vs `clickhouse.base3`:

- A/A: **GREEN**. A2 (arms agree): **GREEN** (0 mismatches). A1: AMBER - 0 wrong-arm
  identifications; the UNKNOWN bucket is the same small set of sub-millisecond special cells
  seen in Stage 1 (too few CPU samples to attribute either way).
- **Non-SEMI/ANTI matrix (96 cells) is clean**: `probe_us` median **-0.4%** (worst +1.4%),
  `cpu_us` median **-0.6%** (worst +1.8%). No cell in this set fails its budget.

### SEMI/ANTI slice (48 matrix cells, reported separately per the plan)

The plan named this the hypothesis under test - "the new path is much lighter" than the old
heavy split this exclusion used to guard against. At 5 reps:

| metric | LEFT SEMI median | LEFT ANTI median | slice median | worst |
|---|---|---|---|---|
| `probe_us` | +2.2% | +5.0% | **+3.4%** | +17.4% |
| `cpu_us` | +1.3% | +1.8% | **+1.5%** | +9.5% |

**Re-measured at 10 reps** (both full 24-cell kinds) to rule out a 5-rep sampling artifact -
the direction and magnitude hold:

| metric | LEFT SEMI median (n=24) | LEFT ANTI median (n=24) |
|---|---|---|
| `probe_us` | +2.9% (worst +10.7%, best -11.9%) | +3.7% (worst +10.9%, best -6.8%) |
| `cpu_us` | +1.3% (worst +7.1%, best -4.5%) | +1.8% (worst +5.2%, best -3.0%) |

Noise test (overlapping pre/post ranges AND `delta <= max(IQR_pre%, IQR_post%)`) at 10 reps:
18/24 SEMI and 15/24 ANTI `probe_us` cells over 2% fail the noise test outright (non-overlapping
ranges, e.g. `LEFT-ANTI|comp|hi|t1|small` pre=[13.9k..14.6k] post=[15.8k..17.3k]). This is a
real, reproducible regression, not sampling noise.

**Verdict on the gate: G-probe fails for the SEMI/ANTI slice** (median +2.9%/+3.7%, budget
1.0%). **G-cpu passes** (median +1.3%/+1.8%, budget 2.0%, and the *overall* matrix-wide `cpu_us`
median stays negative once combined with the other 96 cells). G-cpu-cell: every over-2% cell
is listed above; most fail the noise test, confirming the effect is real rather than a
handful of flukes.

### Per the plan's Stage 2 instructions

> User decision: **unify unconditionally**; no fused fallback even if these cells regress.
> Measure and report anyway.

> If the slice fails, report it and name the P9 byte-recording fallback - do not redesign.

Per that decision, **this stage ships the regression** rather than reverting or
special-casing SEMI/ANTI back onto a fused loop. The **prepared fallback**, named but not
taken, is **P9's byte-recording sink for `is_anti_join` only**: recording a 1-byte match flag
instead of the 8-byte ref word for ANTI (which never reads the word - N21 covers ANTI's
right-column *defaults*, which come from `addNotFoundRow`, not from the recorded match) would
shrink `ProbeOutcomes` traffic for that kind specifically. It would not help LEFT SEMI, whose
recorded word **is** read (`appendFromBlock(firstRefWord(mapped), ...)`), so SEMI's
regression - the larger of the two here - would remain and needs its own investigation if a
later stage revisits this trade-off. No code change was made for this fallback; it is
recorded here as the plan requires.

## Harness changes in this commit

- `EmitSink` / `probeTwoPhase` deleted from the benchmark alongside the source change.
- `expected_probe`: non-base arms always expect `uhj_split` (no fused loop remains to detect);
  the base arm's `ALWAYS_FUSED_KINDS` mapping is preserved for the historical comparison.
