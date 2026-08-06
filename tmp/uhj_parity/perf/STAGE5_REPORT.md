# Stage 5 — delete the `processMatch` layer

Baseline `clickhouse.base3` (BuildID `481f8ae4514186425cad2d2db85a3cdafcbc5884`);
post BuildID `2552d1e642d296d5078d88f40d9b8b292c65da0a`.

## What changed

`processMatch`'s seven `if constexpr` arms (ASOF / ALL / (ANY|SEMI)&&right / ANY&&inner /
ANY&&full-TODO / ANTI / else) are folded directly into all three call sites - pure
relocation, no behavior change, including the empty `ANY && full` arm and the SEMI LEFT
append (N21).

**All three call sites, per F14** (`collectAdditionalFilterBatch` from Stage 4 was checked
and confirmed NOT a fourth site - it calls `addFoundRowAll` directly and never went through
`processMatch`):

- the non-ASOF `consumeProbeBatch` overload,
- the ASOF `consumeProbeBatch` overload,
- the multi-clause `emitBatch` (Stage 3).

Each site previously called `processMatch` from inside two branches
(`if constexpr (probe_mapped_fits_word<MappedValue>)` / `else`), once per branch, each
constructing its own identical-shape `FindResult`. Folding gave the chance to also unify
that duplication: the branch now only decides how to obtain the `Mapped*` pointer
(`mappedFromWord` into a local, or a `reinterpret_cast` of the word), and the `FindResult`
construction plus the seven-arm chain run exactly once afterward. This is a step beyond
literal copy-paste, but not a behavior change - `if constexpr` still discards the exact same
branches per instantiation it always did (verified: `RowRef`/`RowRefList` have `= default`
constructors and ASOF's `unique_ptr` mapped type default-constructs to null, so declaring
the storage unconditionally is always well-formed), and the two former call sites were
never reachable simultaneously for one `Mapped` type to begin with.

Per-site substitutions (only the wiring the original `processMatch` calls already carried):

| site | `i` / `row` | `ind` | `known_rows` | `is_last_disjunct` |
|---|---|---|---|---|
| non-ASOF `consumeProbeBatch` | `i` | `0` (unreachable arm, `static_assert(!is_asof_join)`) | `dummy_known_rows` | `true` (hardcoded, as before) |
| ASOF `consumeProbeBatch` | `i` | real `selectorIndexAt(selector, i)` (pre-existing local, not redeclared) | `dummy_known_rows` | `true` (hardcoded, as before) |
| multi `emitBatch` | `row` | `0` (unreachable, multi never has ASOF) | `known_rows` | `(k + 1 == num_clauses)`, computed per clause - the one substitution that genuinely differs across sites |

The single-clause and multi-clause emit passes were **not** merged into one function (the
plan frames that as optional and reportable-if-done, not required); all three remain
separate `NO_INLINE` functions, each now holding its own copy of the folded arm logic.

**Minor factual correction, non-blocking.** The plan's landing-assertion note attributes the
"assert by source grep, not by symbol" methodology to `processMatch` being `ALWAYS_INLINE`.
The pre-Stage-5 source did not carry an explicit `ALWAYS_INLINE` attribute on `processMatch`
- it was an ordinary template function, presumably inlined by the optimizer as a matter of
course (single call site per translation context, small body) rather than by directive. The
methodology itself is unaffected: after this stage `processMatch` does not exist in source
at all, so a source grep for its definition is the only sensible check regardless of what
attribute it used to carry.

## Landing assertions

- `grep -rn 'void processMatch(' src/Interpreters/UnifiedHashJoin/` -> **no matches**
  (assert by source grep, as the plan specifies; the function produced no distinct linkable
  symbol before or after, so there was never a meaningful `nm` count to compare).
- LOC: `HashJoinMethodsImpl.h` = 1996, `HashJoinMethods.h` = 318, `ProbeLookup.h` = 237
  (total 2551).

## `.text` - the neutrality check the plan asks for

`LeftHashJoinAll.cpp.o`: 1,292,284 -> 1,292,284 (unchanged). UHJ total `.text`: 37,485,664
(Stage 4) -> **37,485,308 (-356 bytes, -0.001%)**. This is the expected result for a pure
relocation with `NO_INLINE` boundaries unchanged: the fold moved code across a function
boundary that the compiler was already inlining in practice, so codegen is unaffected.

## Correctness gate

`GOLDENS_MATCH cells=166`.

## Codegen-parity gate

Not required - the plan restricts this gate to Stages 1-3.

## Performance gate

Full matrix (151 cells, reps=5) vs `clickhouse.base3`:

| slice | `probe_us` median | `cpu_us` median |
|---|---|---|
| Non-SEMI/ANTI matrix (96 cells) | -0.4% (worst +3.4%) | -0.5% (worst +2.6%) |
| SEMI/ANTI slice (48 cells) | +2.4% | +0.9% |

Both slices land within the same band every prior stage measured for them
(non-SEMI/ANTI has hovered around -0.2% to -0.6% median since Stage 2; the SEMI/ANTI slice
has hovered around +1-4%/+1-2% since the Stage 2 regression was first measured). Neither
moved outside that band here, confirming the fold changed nothing observable - exactly the
"exactly neutral" the plan calls for, modulo the pre-existing, already-reported SEMI/ANTI
effect this stage does not touch.

**Verdict**: neutral, as required.
