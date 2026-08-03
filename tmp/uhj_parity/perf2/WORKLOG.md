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

**Caveat recorded now, before it can bite G5.3 (see F3-note).** ICF cuts the other way for the
inertness proof: an edit to a shared header that changes one tree's code can *de-fold*
a previously shared address, which makes baseline symbol names appear in a
"changed symbols" diff even though no baseline source changed. G5.3's symbol diff must
therefore distinguish "the baseline's instructions changed" from "the baseline stopped
sharing an address with unified". Handled explicitly in the G5.3 implementation.

---

## F4 — Unit 2: lock counts, granularity and hold times measured (G0.2, G2.1, G2.2 green)

**Goal.** Measure, not estimate, every hot-path lock's acquisition count, hold-time
distribution and contention, and check the counts against the formulas.

**Instrument.** `src/Common/JoinLockProbe.h` (commit `79a9eee2619`, labelled
`INSTRUMENTATION`, reverted in `2f4a...` before delivery): thread-local per-site
counters and a log2 hold-time histogram read from `cntvct_el0`. Thread-local so the
probe adds no synchronisation to the hot path; a raw counter rather than
`clock_gettime` because the intervals being measured are of the same order as a
`clock_gettime` call. Totals dumped as one JSON line per join destruction;
`lockmeas.py` diffs consecutive lines.

**Which binary produced which number, stated because the probe perturbs timing:** every
number in this entry comes from `bin/clickhouse.instr`. No timing verdict anywhere in
this report uses that binary.

**G2.1 — counts match the formulas exactly, at 1, 16 and 64 threads.** All 12 cells,
ratio 1.000:

```
cell                       algo          B(meas)   K  bucket acq  K*B pred  ratio
INNER|u64|hi|t1|medium     unified_hash       18   1          18        18  1.000
INNER|u64|hi|t16|large     unified_hash      586  32       18752     18752  1.000
INNER|u64|hi|t64|large     unified_hash      563 128       72064     72064  1.000
INNER|comp|hi|t64|large    unified_hash      759 128       97152     97152  1.000
```

`B` is measured independently by a *different* site (`UNI_BLOCKS_MUTEX`, one acquisition
per block) and used to predict `UNI_BUCKET_TRY`, so this is a real cross-check rather
than a tautology. `parallel_hash`'s slot acquisitions equal its dispatched-block count
exactly (ratio 1.000), which is `B*S`.

**G0.2 — the sets agree, after one corrected prediction.** `hash` takes only the two
`StoredColumnsIndex` locks, confirming dynamically that its build has no join-level
mutex. `parallel_hash` takes the per-slot lock. `unified_hash` takes the per-bucket lock
and `blocks_mutex`. **All three probe paths take zero join-level mutexes.**

The one red: `UNI_BUCKET_EMPTY` (`UnifiedHashJoin/HashJoin.cpp:168-173`, the
zero-rows-in-any-bucket special case) never fired. Because the per-bucket selectors
partition the block's rows, it can only fire for a block with zero rows. Two attempts to
produce one — a right side filtered to empty, and one filtered to a single key — both
left the counter at 0 while a dump line was still appended, i.e. the join ran and the
site did not. Recorded as a corrected prediction with the gate still asserting the site
stays at zero, so it retains power in both directions. **LEAD:** it may be unreachable
through the pipeline at all, since empty chunks are dropped upstream.

**G2.2 — hold times, measured.** The headline comparison, 64 threads, `INNER|u64|large`:

| | acquisitions | p50 hold | p99 hold | total held | try_lock failures |
| --- | --- | --- | --- | --- | --- |
| `unified_hash` per-bucket | 72,064 | 11.6 us | 46.3 us | 1.30 s | 50,522 (**0.70 per acquisition**) |
| `parallel_hash` per-slot | 35,840 | 46.3 us | 92.7 us | 1.76 s | 11,074,336 (**309 per acquisition**) |

**This is the largest single POSITIVE for `unified_hash` in the mission.** It takes twice
as many locks and holds each about four times more briefly, for 26% less total time
inside critical sections — and its drain loop fails a `try_lock` 0.7 times per success
against `parallel_hash`'s 309. `parallel_hash` burns roughly 11 million failed `try_lock`
attempts on one 30M-row build at 64 threads; `unified_hash` burns fifty thousand.

The mechanism is in the loop structure, not luck. `unified_hash` rotates its scan start
by `stored_block_no % num_buckets` (`HashJoin.cpp:175`), so concurrent threads begin at
different buckets, and it blocks rather than spins when a pass makes no progress
(`:199-213`). `parallel_hash` always scans slots from index 0 and calls
`std::this_thread::yield()` (`ConcurrentHashJoin.cpp:330,365-366`), so every thread
retries the same slot order and the yield returns it straight into another failing pass.

**Contention is small in absolute terms for `unified_hash`:** 614 blocked waits out of
72,064 acquisitions (0.85%), totalling 0.23 s of wait summed across 64 threads.

**A caution recorded so the report cannot misuse this data:** at 1 thread the measured
"hold time" is 1.48 ms per acquisition, which is simply the whole per-block insert. An
uncontended lock's hold time is not a cost. Hold time measures critical-section
*granularity*, which is what the mission asks for; the *cost* is the acquire/release
overhead times the count, plus the blocked-wait time. Those are reported separately.

**An over-served invariant, found by looking for one.** `UNI_BLOCKS_MUTEX` is held
2.90 us at p50 (median across all cells; 563 acquisitions at 64 threads, 1.83 ms total).
Inside it, `StoredColumnsIndex::add` — the only part that genuinely needs a lock shared
with other threads — takes 32 ns. The other ~99% is
`assertBlocksHaveEqualStructureAllowReplicated`, `doDebugAsserts()`,
`JoinCommon::getCurrentQueryMemoryUsage()` and the `data->columns.emplace_back`
(`UnifiedHashJoin/HashJoin.cpp:878-899`). This is a CONTINGENT finding and it is in the
necessity table.

---

## F5 — Two hypotheses refuted, one of them by a defect in my own instrument

Both are recorded because a refuted hypothesis is a result, and the second is a
methodological correction that would have contaminated every per-loop share in the
report.

**Refuted hypothesis 1: "the result-gather loop is inlined differently between the
trees."** The profiler showed `hash` and `parallel_hash` spending 38-48% of in-join
samples in `HashJoinResult::generateBlock` while `unified_hash` spent 38-48% in
`LazyOutput::buildOutputFromBlocks<true>` — which reads exactly like the baseline having
inlined the callee and unified not. On the hottest loop in the join, that would have
been a major finding.

**It is false.** `codegen/P1_G2_probe_and_gather.md`: both symbols exist on both sides,
both are `0x634` bytes, `generateBlock` is `0x1cf4` bytes on *both* sides, and
`buildOutputFromBlocks<true>` disassembles to 397 instructions on each side with **zero
positional mismatches** after namespace normalisation. Neither `generateBlock` calls it
directly; both reach it through `buildOutput`.

**The cause was my own attribution rule.** `IN_JOIN_MARKERS` in `loops.py` carries a
blanket `DB::Unified::` that catches the entire unified tree, but the baseline has no
common prefix and every counterpart must be named individually. `DB::LazyOutput::` was
missing. A missing marker does not lose a sample — the leaf-in-join scan walks one frame
further out and credits the *caller*. So completeness (G0.1) stayed green throughout
while the per-tree shares were biased.

**Fix and verification.** Added the missing markers plus a mechanical symmetry audit
over known counterpart pairs, which immediately found a second one
(`DB::processMatch`). Re-collected as `--tag u0c` and the prediction registered in
PREREG P1.1 came out exactly:

```
before (u0a): hash generateBlock 44.7% / buildOutputFromBlocks  0.0%
              unified generateBlock 1.1% / buildOutputFromBlocks 47.8%
after  (u0c): hash 0.0% / 44.3%   parallel 0.0% / 43.3%   unified 0.0% / 49.0%
```

All three now agree, which is what byte-identical code should look like.

**Refuted hypothesis 2: "`hash` inlines the composite key getter and the others do
not."** Also false. `hash` and `parallel_hash` are the same `DB::HashJoin` code with the
same `use_offset=true`, and call the *same* out-of-line constructor at `0x13ab1280`;
`hash` does have samples in it. The two specialisations (`need_offset` true and false)
are instruction-for-instruction identical, 107 instructions, 644 bytes each. The
difference is not codegen at all — it is **how many times the function is called**, which
is what F6 is about.

---

## F6 — The headline finding: whole-block key packing, once per partition

**How it was found.** Chasing the refuted inlining hypothesis into the constructor.

**The mechanism, from code.** `HashMethodKeysFixed`'s constructor calls `packFixedBatch`
whenever `usePreparedKeys` holds — no nullable or LowCardinality keys, `sizeof(Key) <= 16`,
every key size in {1,2,4,8,16} — which is exactly the two-UInt64 composite key
(`Common/ColumnsHashing/HashMethod.h:456-459`). `packFixedBatch` calls `fillFixedBatch`,
and `fillFixedBatch` sizes its output by

```65:66:src/Interpreters/AggregationCommon.h
            const auto * column = key_columns[i];
            size_t num_rows = column->size();
```

**`column->size()` — the whole block, not the selector.** So one construction packs every
row of the block regardless of how few of them that partition will insert.

And `unified_hash` constructs a key getter **once per bucket**
(`UnifiedHashJoin/HashJoinMethodsImpl.h:356`, reached from `insertIntoBuckets` once per
bucket) plus once for the scatter pass (`:166`). `hash` constructs one per block.
`parallel_hash` constructs one per slot per block.

So the composite-key build cost carries a term **linear in the partition count**:
1 for `hash`, `S = threads` for `parallel_hash`, `K = 2 x bit_ceil(threads)` for
`unified_hash`. The single-UInt64 key goes through `HashMethodOneNumber` and packs
nothing, which is the discriminator.

**Measured, pre-registered as PREREG P1.2 before the run.** Sweep `max_threads` over
4/8/16/32/64 at fixed 30M-row build data, 7 reps, algorithms interleaved within each
repetition. Fit `ns_per_build_row = a + b*K`:

```
key   algo             a (ns/row)   b (ns/row/part)     R^2   n
comp  hash                  40.05            0.0000   0.000   5     (K == 1 always; flat 39.1-41.1 across 4..64 threads)
comp  parallel_hash         48.68            1.7459   0.993   5
comp  unified_hash          57.95            1.1808   0.964   5
u64   hash                  29.31            0.0000   0.000   5     (flat 27.0-31.5)
u64   parallel_hash         33.01            0.5746   0.988   5
u64   unified_hash          38.52            0.1577   0.572   5
```

All four registered predictions held. The decisive one: the `comp`-to-`u64`
per-partition coefficient ratio is **7.49** for `unified_hash` against a registered
threshold of 4. Both arms see the same thread count and therefore the same contention
change; only `comp` packs.

**What it costs, at 64 threads with composite keys** (build phase, ns per build row):

| | partitions | measured | per-partition term | share |
| --- | --- | --- | --- | --- |
| `hash` | 1 | 39.9 | — | — |
| `parallel_hash` | 64 | 162.6 | 1.746 x 64 = 111.7 | 69% |
| `unified_hash` | 128 | 202.5 | 1.181 x 128 = 151.1 | 75% |

`unified_hash` is **5.07x** the flat-map build cost and **+24.6%** over `parallel_hash` —
which independently reproduces the inherited deficit map's "+21%..+38% build CPU at 64
threads with composite keys" from a completely different measurement.

Note the shape carefully, because it inverts the obvious reading: **`unified_hash`'s
per-partition coefficient is 32% *lower* than `parallel_hash`'s** (1.181 vs 1.746) — it is
the more efficient implementation per partition. It loses because it uses **twice as many
partitions** (`K = 2*bit_ceil(T)` against `S = T`).

**A third origin, failing differently from both, agrees.** After the F5 marker fix,
`fillFixedBatch` became visible to the profiler and holds 7.5% of *all* in-join samples
across the whole matrix. Per cell:

```
INNER|comp|hi|t64|large   hash 35.5%   parallel_hash 52.1%   unified_hash 67.0%
INNER|comp|hi|t16|large   hash 33.5%   parallel_hash 44.7%   unified_hash 51.6%
INNER|comp|hi|t1|medium   hash 43.1%   parallel_hash 41.3%   unified_hash 37.3%
INNER|u64 |hi|t64|large   hash  0.0%   parallel_hash  0.0%   unified_hash  0.0%
```

Three origins that fail differently — a static read of the sizing expression, a
regression on a settings-only natural experiment, and a sampling profiler — and they
agree, including on the null: zero for `u64`, and equal across all three
implementations at one thread where every implementation has one or two partitions.

**This loop was not in the enumeration until the marker fix exposed it**, and it is now
`B22`. It is the largest cost difference found in this mission.

---

## F7 — Ablation A-K1 refuted its own prediction, and two defects in my validity tool

**Goal.** Test whether the K-linear term measured in F6/P1.2 is a live lever, by halving
the bucket count (`BUCKETS_PER_THREAD` 2 -> 1) at fixed thread count. Pre-registered as
PREREG P3.1 *before* the patch, with the prediction derived from the P1.2 fit rather
than from an instruction ratio.

**Two defects in `symdiff.py` found while validating the ablation binary. Both are
recorded because each would have produced a confidently wrong result.**

*Defect 1 — `adrp` annotation false positive.* `llvm-objdump` annotates each operand
with the nearest preceding symbol. For a call that names the callee; for `adrp` it names
whatever data symbol happens to sit at that page, so it changes whenever the data
section shifts. It reported `DB::ConcurrentHashJoin::addBlockToJoin` as changed — 826
instructions, identical size, three differing annotations and nothing else. Left alone,
this would have said the ablation perturbed a baseline. Fixed by keeping the annotation
only for branch and call instructions.

*Defect 2 — the serious one. The normaliser could not see the ablation.* To absorb
addresses moving, the first version rewrote every `0x...` token to `ADDR`. Ablation A-K1
compiles to exactly one instruction change:

```
ref:  165d0cbc:  mov  w9, #0x2      // = BUCKETS_PER_THREAD
ak1:  165d0cbc:  mov  w9, #0x1
```

`0x2` and `0x1` were both being rewritten to `ADDR`, so the tool reported the ablated
constructor as **byte-identical to the reference** — a validity check that could not see
the change it existed to confirm. Had I trusted it, I would have concluded the ablation
never took effect and thrown away a real result. Fixed by normalising only hex tokens of
five or more digits, plus precise handling of the `adrp`+`add` relocation pair (whose low
half legitimately is a short hex immediate). **And the gate now asserts the ablation
target DIFFERS (`--expect-differ`), not only that the baselines do not** — the direction
that was missing is the one that failed.

**Validity, after the fixes — GREEN in both directions:**

```
added 0, removed 0, resized 0 outside DB::Unified::
expect-differ 'DB::Unified::HashJoin::HashJoin(std':  1 symbol, 1 DIFFER  -> OK
byte-compare  'fillFixedBatch<unsigned long':         4 symbols, 4 opcode-identical
byte-compare  'DB::ConcurrentHashJoin::addBlockToJoin': 1 symbol, 1 opcode-identical
byte-compare  'DB::HashJoin::addBlockToJoin':         2 symbols, 2 opcode-identical
```

`fillFixedBatch` matters specifically: it is shared code that both baselines run, and it
is opcode-identical, so the ablation cannot have moved the packing rather than changing
how often it happens.

**Result: the prediction is REFUTED, and the refutation is the finding.**

```
comp + unified_hash, 64 threads, build phase
  measured before      :  202.5 ns per build row   (model predicted 209.1 -- good fit)
  PREDICTED after      :  133.5 ns per build row   (registered before the build)
  measured after       :  220.7 ns per build row
  predicted change     :  -34.1%
  measured  change     :   +9.0%   (band 5.0%, 7 reps, stdev 0.7%)
  model/measured ratio :  -3.79    -- wrong sign, not merely wrong magnitude
```

The control arms behaved: `hash` and `parallel_hash` moved -1.6%, -0.8%, +1.7%, +0.8%
at 64 threads, all inside the band, so the ablation was isolated and the result is not
drift.

**What it means, stated carefully.** Halving K removes per-partition work *and* removes
the slack the `try_lock` drain loop needs. At `BUCKETS_PER_THREAD = 1` and 64 threads
there are exactly as many buckets as threads, so a thread that finds its bucket busy has
nowhere else to go; at 2 it has twice as many chances per pass. The `u64` arm, where
there is no packing to save, shows this in the clear: **+90.1% at 64 threads.** The two
effects have opposite signs and contention wins.

So the existing `BUCKETS_PER_THREAD = 2` is a justified choice, not an arbitrary one,
and **K is not the lever**.

**What it does NOT mean, and this is the distinction the tiering exists for.** It does
not refute B22. The packing is in the code (`fillFixedBatch` sizes by `column->size()`),
it runs once per partition, and the profiler independently attributes 67% of in-join
samples to it for `unified_hash` at 64 threads with composite keys against **0%** for
`u64`. What the ablation refutes is the *causal reading of the P1.2 coefficient*: P1.2
varied K by varying thread count, so its `b` confounds per-partition cost with
thread-count effects, and the clean fixed-thread test disagrees. **B22's magnitude is
therefore UNSETTLED, not MEASURED**, and the named experiment that would settle it is in
the report: hoist the key-getter construction out of the per-bucket loop, which changes
the packing count without changing K or contention at all.

**Amendment to F6.** The F6 entry states the per-partition term as though `b` were the
packing cost. That reading is superseded by this entry. The comp-vs-`u64` *discriminator*
in P1.2 survives — both arms saw the same thread-count change, so the difference between
them is still comp-specific — but the absolute attribution of `b * K` nanoseconds to
packing does not.

**LEAD, not used for acceptance:** the ablation's effect on `unified_hash` is
non-monotone in thread count (+36%, -22%, +19%, -15%, +90% for `u64` at 4/8/16/32/64)
while the controls are flat. Something bimodal is happening in the bucket layout that
neither the packing story nor the contention story predicts on its own. Not chased.

---

## F8 — G3.2 closed: the one-thread deficit is an instruction-count cause, and the cache-footprint candidate is REFUTED

**Goal.** The gate the prior mission never ran and the brief named as the first step:
separate an instruction-count cause from a cache-footprint cause behind the one-thread
probe deficit, with hardware counters, before believing any codegen story.

**Three measurement routes tried. The first two are unsound on this host and are recorded
so nobody re-pays them.**

1. **`perf stat -p <server pid>`** — *wrong, and quietly so.* `-p` follows only the
   threads that exist when it attaches, and ClickHouse spawns per-query threads, so the
   join's work is never counted. It reported **178,633 cycles** for a query that
   executes about **385 million** — a 2000x undercount, and the derived per-row numbers
   (0.0387 cycles/row) looked small but not obviously impossible. Caught by sanity-
   checking the magnitude against a rough cycles-per-row estimate, not by any error.
2. **ClickHouse's own `metrics_perf_events_enabled=1`** — the right idea, but its
   per-thread counter deltas **UInt64-underflow**: values around 2^64 appear routinely.
   Only 2 of 15 reps were usable for `hash` and 6 of 15 for `unified_hash`, and the
   surviving samples disagreed by two orders of magnitude. Abandoned.
3. **`perf stat` LAUNCHING the process** — sound, because perf then follows every thread
   the process creates. Requires the data locally, so cell `FULL|u64|hi|t1|medium` was
   reproduced in `clickhouse local` with the harness's own `SCATTER` constant; the
   reproduction returns **3,600,000 output rows**, identical to the server-side cell.
   Process startup is subtracted using a `SELECT 1` control measured in the same loop.

**A scoping error of mine, caught and corrected.** The first run used `INNER`, which the
deficit map reports at **+0.64% median** — a cell with essentially no deficit. It duly
measured +2.1% cycles and +0.5% instructions, i.e. nothing. The deficit lives in the
**used-flag kinds**. Re-run on `FULL`:

```
per probe row              hash       unified    delta%   hash sd%
cycles                  209.675       226.718     +8.1%      2.9%
instructions            431.116       460.005     +6.7%      0.4%
cache-misses              5.252         5.323     +1.4%      0.7%
cache-references        160.031       171.952     +7.4%      0.3%
dTLB-load-misses          7.174         7.089     -1.2%      0.3%
branch-misses             1.362         1.345     -1.2%      0.6%
IPC                       2.056         2.029     -1.3%
```

**The deficit is reproduced** (+8.1% cycles/row against the map's +4.6%..+9.7% range,
mean +6.54%) and **the accounting closes almost exactly**: 1.067 / 0.987 = **1.081**.
The 8.1% more cycles is 6.7% more instructions executed at 1.3% lower IPC — that is all
of it. Memory behaviour is flat: cache misses +1.4%, dTLB misses **−1.2%**, branch
misses −1.2%, all inside the run-to-run spread.

**Verdicts against PREREG P3.2, scored honestly including where I set the threshold
badly:**

- The registered **cache-footprint refutation condition** was "IPC more than 10% lower
  while instructions are within 10%". Measured IPC delta is **−1.3%**. **Candidate A5
  — `per_offset_flags` being wider and sparser on the partitioned map — is REFUTED as
  the mechanism for the one-thread deficit.** This is a standing candidate from the
  prior mission's inventory and it is now closed.
- The registered **instruction-count signature** was "instructions up >10%, IPC flat,
  misses flat". Two of three hold outright; the first is **+6.7%, below the 10% I
  registered**. That threshold was miscalibrated on my part — I set it above the size of
  the deficit being explained (~6.5% wall), so it could not have been met by a correct
  answer. The discriminating comparison, which is *which* counter moved, is unambiguous:
  instructions moved, memory did not.

**So: the mechanism behind the one-thread deficit is settled as an instruction-count
cause, and the deficit itself is now MEASURED rather than BOUNDED.** The codegen story in
`codegen/P1_G2_probe_and_gather.md` — the `Prober` not staying in registers, dependent-load
depth 3→5, two spill stores and nine spill reloads per row, and `offsetInternal` costing
8 instructions and 3 loads even on the `sole` fast path — is the story the counters
support.

**Remaining honesty about scope.** This measures the whole query (build, probe and the
non-joined scan), not the probe in isolation: the `full − buildonly` subtraction was
abandoned with route 2 because buildonly runs are short and underflowed. The
attribution of the +6.7% *within* the query to the probe loop specifically still rests on
the codegen artefact and the profiler, not on the counters. What the counters settle is
the **mechanism class**, which is exactly what G3.2 exists to do.
