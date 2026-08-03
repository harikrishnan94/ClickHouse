# Worklog — per-row-loop and locking cost analysis of `unified_hash` vs `hash` vs `parallel_hash`

Working record. Per iteration: goal / what was done / how verified with exact commands
/ what changed about the plan. Failed ablations, void nulls and refuted hypotheses are
recorded — they are results. Corrections amend forward with a new entry referencing the
old; nothing is edited away.

---

## F0 — Mission start, environment re-verification

**Goal.** Establish the starting state and re-verify every inherited premise cheap
enough to re-verify, before forming any hypothesis.

**Mission-start commit.** The brief names `543efb61fb9850e3c715def8085ce522db71651d`.
Verified it exists and is an ancestor of HEAD:

```
$ git log --oneline -5
7ec1e520fbe uhj-perf: mark the false llvm-mca limitation superseded in PREREG
55cac1722dd uhj-perf: correct the false "llvm-mca unavailable" limitation
543efb61fb9 uhj-perf: commit the instruction-class counter behind the codegen numbers
```

**Deviation from the brief, documented rather than silent.** HEAD is **two commits
past** the stated mission-start commit. Both are doc-only:

```
$ git log --stat 543efb61fb9..HEAD | grep -E '^ tmp|^ src'
 tmp/uhj_parity/perf/PREREG.md | 5 +++++
 tmp/uhj_parity/perf/WORKLOG.md | ...
 tmp/uhj_parity/perf/bin/... (wrappers)
```

They add the LLVM-22 tool wrappers and correct the prior mission's false "llvm-mca
unavailable" record. No `src/` change. I work from HEAD; this does not move any
baseline, and G5.1 is diffed against `0945a745399` as the gate text specifies, which
is unaffected either way.

**G5.1 is green at mission start** (it must also be green at delivery):

```
$ git diff --stat 0945a745399 -- src/Interpreters/HashJoin/ \
      src/Interpreters/ConcurrentHashJoin.cpp src/Interpreters/ConcurrentHashJoin.h
(empty)
```

**Environment re-verified** — recorded in PREREG P0.0 rather than duplicated here.
Headline: `aarch64` / `Neoverse-V2` / 96 CPUs; `llvm-mca` **is** present at
`/opt/llvm-22/bin/llvm-mca` (LLVM 22.1.8, aarch64 default target, host CPU reported as
`neoverse-v2`), confirming the prior mission's correction; `build/reldeb` is current
(`ninja -n clickhouse` shows no pending compile or link steps).

**Reference binary.** `build/reldeb/programs/clickhouse` copied to
`tmp/uhj_parity/perf2/bin/clickhouse.ref`. Note this is **not** the prior mission's
`clickhouse.pristine`: that one predates the kept A7 fix
(`offsetInternalAtBucket`), so before/after codegen diffs in this mission must use
`clickhouse.ref`, not the older pristine copy. Using the older one would attribute the
A7 fix to whatever ablation was under test.

**What changed about the plan.** Nothing yet.

---

## F1 — Reading the prior record, and one premise correction

**Goal.** Absorb the prior mission's corrections before forming hypotheses, per the
brief, and re-verify the specific premises this mission is asked to build on.

**What was done.** Four parallel reading passes: the prior worklog/report/prereg and
candidate inventory; the build hot path in all three trees; the probe and non-joined
hot paths in all three trees; and a static lock/atomic enumeration.

**Premise correction — the "identical probe path" claim is partly false.** The brief
says probe match/non-match/gather "is reported textually identical between the two
trees apart from namespace" and asks me to verify it. Verified by normalised diff:

- **Identical** modulo namespace and include path: `AddedColumns.{h,cpp}`,
  `HashJoinResult.cpp`, `KnownRowsHolder.h`, `joinDispatch.h`, `processMatch` body.
- **Materially different:** `HashJoinMethodsImpl.h` and `KeyGetter.h`.
- **Also different:** `JoinUsedFlags.h` — unified **removed** `allOffsetFlagsSet`.

Registered as PREREG P0.3 before any measurement was designed around it. This matters
because the brief drew an inference from the identity ("if true it bounds where a
probe difference can live"); the bound does not hold as stated, and the lookup and
key-getter layers stay in scope — which is where A6 already points.

**Other inherited premises re-verified from code by the reading passes:**

| Premise | Verdict |
| --- | --- |
| `parallel_hash` merges all slots into slot 0 at build finish; probe walks one shared 256-bucket map | **Confirmed**, `ConcurrentHashJoin.cpp:805-904`, merge at `:828-832`, copy-back at `:896-900`, probe at `:468-474` |
| baseline hardcodes `constexpr bool use_offset = true` | **Confirmed**, `HashJoin/KeyGetter.h:19`; unified instead sets `needs_offset = JoinFeatures<...>::need_flags` at `UnifiedHashJoin/HashJoinMethods.h:90` |
| `unified_hash` `sole` short-circuit at `num_buckets == 1` | **Confirmed**, `TwoLevelHashTable.h:544-545`, used at `:556-557` and `:585-586` |
| `hash` build takes no join-level mutex | **Confirmed statically**; to be confirmed dynamically by G0.2 |

**What changed about the plan.** The probe-path scope widened: `KeyGetter.h` and the
`Prober` indirection in `HashJoinMethodsImpl.h` are now first-class enumerated loops
rather than assumed-identical background.

---

## F2 — Unit 0: loop enumeration and the G0.1 completeness gate

**Goal.** Enumerate every per-row/per-cell loop on the hot path and prove the
enumeration complete against profiler samples (G0.1).

**What was done.** `tmp/uhj_parity/perf2/loops.py` holds the enumeration as data: 44
loops, each with file:line evidence, multiplicity formula, the implementations that
have it, and the regexes naming the symbols it compiles into. Written from reading the
code, not from looking at samples — which is what gives the gate power to fail.
`enumerate.py` collects samples and judges.

**Sample collection.** 132 profiled runs: 3 algorithms x 3 key families (`u64`, `str`,
`comp`) x 4 (threads, cardinality) points x kinds `INNER` / `FULL` (both match rates)
/ `LEFT SEMI`, minus the 12 `parallel_hash` SEMI cells that have no comparator.
550,754 in-join CPU samples at a 1 ms sampling period.

```
$ python3 tmp/uhj_parity/perf2/enumerate.py collect --tag u0a   # -> results/samples_u0a.jsonl
$ python3 tmp/uhj_parity/perf2/enumerate.py gate --tag u0a
```

**First run: RED, as pre-registered (PREREG P0.1).** 61 unexplained symbols, 6,778
samples (1.23%). I registered in advance that I expected the first run to be red and
that gather/allocator symbols were the likely cause; the actual misses were only
partly that, and three were real:

| Missed symbol | Samples | What it actually was |
| --- | --- | --- |
| `ScatteredBlock::filterBySelector` | 2,654 (0.48%) | **A per-row loop the reading missed entirely** — left-block materialisation. Now enumerated as **G5**. |
| `ConcurrentHashJoin::dispatchBlock` | 2,524 (0.46%) | the `parallel_hash` scatter driver; my B5/B6/B9 regexes named the helpers but not the caller they were inlined into |
| `MapsTemplate<>::getBucketBufferSizeInBytes` | 6 | **B16's regex was simply wrong** — I had guessed `getBucketBytes` |

The rest were per-block or per-query scaffolding (transform `work`/`prepare`,
`ScatteredBlock` moves and destruction, `JoinCommon` type plumbing, container growth),
each now excluded with a written reason.

**G5 is worth its own note**, because it is the one loop the reading missed and it
turns out to be cheap to settle: `find src -name ScatteredBlock.h` returns exactly one
path, `src/Interpreters/HashJoin/ScatteredBlock.h`. There is no Unified copy. All
three implementations therefore run *literally the same code*, and the codegen delta
for G5 is zero by construction rather than by measurement.

**Final: GREEN.**

```
$ python3 tmp/uhj_parity/perf2/enumerate.py gate --tag u0a
  distinct in-join syms: 310
  in-join samples      : 550754
  mapped to a loop     : 242 syms, 542956 samples (98.58%)
  explicitly excluded  : 68 syms, 7798 samples (1.42%)
  UNEXPLAINED          : 0 syms, 0 samples (0.00%)
G0.1: GREEN
```

**Gate power, checked rather than assumed** (`enumerate.py power --tag u0a`): three
injected control symbols all come back unexplained, so no pattern is a catch-all; the
exclusions absorb only 1.42% of samples; and a per-loop knockout reports, for each
loop, how many samples go unexplained when its patterns are removed. 12 loops are
individually load-bearing (G2 alone accounts for 6.23%). The other 32 are either
unsampled or share a symbol with a loop they were inlined into — recorded, not hidden.

**Seven loops are enumerated but unsampled**: B1, B19, B21, G4, P4, P6, P9. Not a
gate failure and not the same situation in each case — B19/B21 are once-per-build,
P9 is not exercised by this matrix (no residual filter), and P4/P6/G4 are *inlined
into a caller that was sampled*, which the knockout confirms. Each still needs an
unconditional codegen artefact under G1.1.

---

## F3 — Correction: the inherited algorithm assertion is unsound (identical-code folding)

**This corrects the prior mission's accepted method for Gate G0.1**, `perf/harness.py:347-378`.

**How it surfaced.** The very first collection run reported nine "algorithm
mismatches" of the form *asked `hash`, got `unified_hash`* — with `unified_hash: 1`
sample against `hash: 13`. A single frame was deciding the verdict.

**Cause, established independently of the profiler.** The offending stack is
`HashJoinResult::generateBlock` calling something that demangles to
`DB::Unified::HashJoin::canRemoveColumnsFromLeftBlock` inside a plain `hash` query.
That is impossible as a call, and it is not one — it is **identical-code folding**:

```
$ llvm-nm --defined-only -C tmp/uhj_parity/perf2/bin/clickhouse.ref \
      | grep canRemoveColumnsFromLeftBlock
0000000014289180 T DB::Unified::HashJoin::canRemoveColumnsFromLeftBlock(DB::TableJoin const&)
0000000014289180 T DB::HashJoin::canRemoveColumnsFromLeftBlock(DB::TableJoin const&)
```

One address, two names. `addressToSymbol` returns one of them arbitrarily, so a
`hash` run genuinely does report `DB::Unified::` frames. A presence test on such a
frame is not a proof of anything.

**Fix.** `algoassert.py` excludes frames whose demangled name lives at a cross-tree
folded address. The ambiguous set is computed from the binary by `icf_census.py`, so
it is derived from the mechanism, not tuned against the samples.

**Verified over all 132 runs**, where the requested algorithm is known:

```
$ python3 tmp/uhj_parity/perf2/algoassert.py recheck --tag u0a
runs rechecked                       : 132
mismatches under the INHERITED rule  : 9
mismatches under the CORRECTED rule  : 0
G0.1b algorithm identity: GREEN
```

**Why this matters beyond bookkeeping.** The prior mission reported 264/264 agreement
and treated this rule as the one sound way to tell the implementations apart. It is
not sound, and nine counterexamples now exist. Any inherited result that rests on it
rests on a rule that can mislabel a run.

**Second, larger consequence — ICF is also evidence, and the strongest kind
available.** `icf_census.py` finds **31 addresses** where a `DB::Unified::` join
symbol and a baseline join symbol are the same instructions, including
`AddedColumns<false>::appendFromBlock`, `AddedColumns<{true,false}>::applyLazyDefaults`
and several `processMatch` instantiations. For those loops the codegen delta is not
"measured small" — it is **zero by identity**, and no disassembly, instruction count
or `llvm-mca` run can add anything. This is folded into Unit 1 as the cheapest tier of
codegen evidence.

**Caveat recorded now, before it can bite G5.3.** ICF cuts the other way for the
inertness proof: an edit to a shared header that changes one tree's code can *de-fold*
a previously shared address, which makes baseline symbol names appear in a
"changed symbols" diff even though no baseline source changed. G5.3's symbol diff must
therefore distinguish "the baseline's instructions changed" from "the baseline stopped
sharing an address with unified". Handled explicitly in the G5.3 implementation.
