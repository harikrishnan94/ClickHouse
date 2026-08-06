# Stage 1 — de-multiply the lookup (128 -> 64)

Baseline `clickhouse.base3` (fa77c89eb39, BuildID `481f8ae4514186425cad2d2db85a3cdafcbc5884`);
post BuildID `0bd14d2d97dc02336767ad160771275a6524c988`.

## What changed

- `ProbePrefetch<Map, KeyGetter, Selector>`: the probe's software prefetch as a named type,
  replacing the lambda minted inside `joinRightColumns` whose closure type carried
  `need_filter` into every lookup body. One instance per probe call (calibration fires once
  at an absolute row, F11); multi binds `mapv[0]`/`key_getter_vector[0]` like the old lambda
  (per-clause prefetch is Stage 3's deliberate change, not this stage's).
- `lookupBatch<need_flags, Map, KeyGetter, Selector, PrefetchAt>`: the batched phase-1
  lookup, NO_INLINE, keyed only on (Map, KeyGetter, Selector) + TU-constant `need_flags`.
  Builds `RecordOutcomeSink<need_flags>` and calls `SequentialLookup::run`.
- `probeTwoPhase` keeps `need_filter` (the consume needs it) and is now a thin batch loop
  over `lookupBatch` + consume. `LookupDriver` template param dropped (benchmark call site
  updated). Deleted in Stage 2 as planned.
- `runImpl` stays NO_INLINE (P1). No semantic change.

## Landing assertions (`LeftHashJoinAll.cpp.o`, canonical `llvm-nm --defined-only | c++filt`)

| metric | baseline | stage 1 | verdict |
|---|---|---|---|
| `DB::Unified::lookupBatch` | 0 | **64** (68 B each) | PASS |
| `DB::Unified::SequentialLookup` | 384 | **128** | deviates, explained below |
| `DB::Unified::probeTwoPhase` | 128 | **128** (640 B each) | count PASS, size deviates |
| `.o` bytes | 12,724,008 | 12,033,600 | -5.4% |
| `.text` bytes | 2,454,491 | 1,976,908 | **-19.5%** |

- `SequentialLookup` = 128, not the predicted 192: clang inlined the two-branch `run`
  dispatcher into `lookupBatch` (its only call site). The measured substance is exactly the
  plan's target: 64 lookup bodies per TU keyed on (Map, KeyGetter, Selector), `need_filter`
  gone (template args verified: `ProbePrefetch<...>&`, `RecordOutcomeSink<need_flags>&`).
- `probeTwoPhase` shells are 640 B, above the predicted <512 B (the shell still holds the
  fused-output branch + two `PODArray::resize` inlines). Still small; 128 x 640 B = 80 KB.
- `LeftHashJoinSemi.cpp.o`: `.o` 8,399,128 -> 8,251,016, `.text` 1,538,039 -> 1,315,208
  (-14.5%). UHJ totals: `.text` 62,015,808 -> 49,802,876 (**-19.7%, -12.2 MB**),
  `.o` ~315.0 MB -> 310.7 MB.

## Correctness gate

`GOLDENS_MATCH cells=166` on the post binary (8121), `JOB_EXIT=0`.

## Codegen-parity gate

`SequentialLookup::runImpl` before/after, per (sink x ICF group), u64 TwoLevel + String:

- **String: exact parity on all 4 groups** (insns/branches/loops/calls identical:
  263/48/4/1, 255/47/4/1, 271/48/4/1, 328/50/4/3).
- **u64: loop counts identical** (6, 4, 2, 2 per group), branch counts identical;
  instruction counts exact on 2 groups, -8 on 2 groups (register-allocation churn from the
  struct layout, fewer instructions not more).

No loop-count change anywhere (the P1 fingerprint).

## Performance gate (sweep `stage1ab2-20260806`, 151 cells x 2 arms, reps=5, vs base3)

- **A/A: GREEN.** A1: **0 wrong arms**; 14 UNKNOWN (tiny ~1 ms cells with 0-2 samples,
  multi-clause cells the split/fused markers do not cover by design, one marginal small
  cell). A2: **0 disagreeing cells.**
- **G-probe**: overall median **-0.6%**; per-family medians all in [-1.3%, +0.1%] (LEFT ANTI
  -1.3%, LEFT SEMI -1.1% — faster). 9 cells >2% slower / 26 >2% faster.
- **G-cpu**: overall median **-0.5%**; every per-family median <= 2.0%. 1 cell slower beyond
  band, 1 faster.
- **G-cpu-cell / probe-cell resolution**: every >2% cell listed and noise-tested
  (overlapping ranges AND delta <= max(IQR_pre%, IQR_post%)). Two slower cells failed at 5
  reps and were re-measured at 10 reps: `FULL|str|lo|t1|medium` probe_us **+5.4% -> +0.0%**
  (5-rep noise, consistent with the exact String codegen parity);
  `addfilter|filter|t1|timed` cpu_us +5.9% -> **+4.3% within IQR band** (us-quantized tiny
  cell; its real-sized counterpart `addfilter|u64|t1|medium|timed` is within 2% on both
  metrics). All remaining >2% cells pass the noise test or are faster.

**Verdict: perf-neutral to slightly faster.** Stage 1 ships no regression.

## Harness changes pulled forward (were Stage 2's item 5)

Stage 1's thin `probeTwoPhase` is inlined into `joinRightColumns` by clang, so the split
probe marker vanished from stacks one stage earlier than the plan scheduled. Pulled forward:

- `PROBE_SYMBOL_RULES`: split marker = `probeTwoPhase` OR `lookupBatch` (either counts, so
  the same judge reads base3 and post-Stage-1 binaries); fused marker stays `EmitSink`
  (matches via the sink template argument in runImpl names).
- `ab_report.a1`: `verdict=UNKNOWN` with <20 samples goes to the unknown bucket, not
  wrong-arm (tiny special cells collect ~1 sample).
- `multi2semi` timed cells aggregate `sum(r.k)` instead of `sum(r.v)`: which duplicate
  becomes SEMI's first match is parallel-build-order dependent, so `sum(r.v)` was
  nondeterministic at 16 threads (measured +-11 on both binaries — pre-existing, not a
  Stage 1 effect; verified 6 runs each on base3 and post).
