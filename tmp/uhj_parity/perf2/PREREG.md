# Pre-registration — per-row-loop and locking cost analysis

Every entry is written **before** the change or measurement it predicts. Entries are
appended, never edited; a correction is a new entry that references the old one.
The git history is the proof of ordering.

Mission-start commit: `543efb61fb9850e3c715def8085ce522db71651d`
HEAD at first entry: `7ec1e520fbe33351697d036c47fca3b1feb51950` (two doc-only commits
past mission start; `git diff` confirmed they touch only `tmp/uhj_parity/perf/PREREG.md`).

---

## P0.0 — Environment, re-verified at pre-registration

Re-confirmed rather than inherited, per the mission's "re-verify; do not trust".

| Fact | Command | Result |
| --- | --- | --- |
| Host arch/uarch | `lscpu` | `aarch64`, `Neoverse-V2`, 96 CPUs, 1 NUMA node, L1d 64 KiB/core, L2 2 MiB/core, L3 36 MiB shared |
| `llvm-mca` present | `/opt/llvm-22/bin/llvm-mca --version` | LLVM 22.1.8, default target `aarch64-unknown-linux-gnu`, host CPU `neoverse-v2`. **Present** — the prior mission's "unavailable" was a PATH failure and is superseded. |
| Binary current | `ninja -n clickhouse` in `build/reldeb` | only "Re-running CMake"; no compile or link steps pending ⇒ binary matches the tree |
| G5.1 already green | `git diff 0945a745399 -- src/Interpreters/HashJoin/ src/Interpreters/ConcurrentHashJoin.{h,cpp}` | empty |

**Prediction:** these hold for the whole mission. **Refuted if** any later `ninja -n`
shows pending compile steps for a binary already used for a measurement, or G5.1
becomes non-empty.

---

## P0.1 — Unit 0, loop-enumeration completeness (gate G0.1)

**Claim under test:** the static enumeration of per-row/per-cell loops, derived by
reading `addBlockToJoin`, `joinBlock` and the non-joined scan in all three trees,
covers every symbol that actually consumes CPU inside the join.

**Instrument:** ClickHouse's own sampling profiler via `system.trace_log`
(`query_profiler_cpu_time_period_ns`), over a spread of cells covering all three
algorithms, all four (threads, cardinality) points, and every key-getter family
(`u64`, `str`, `comp`). Symbols are demangled server-side with
`demangle(addressToSymbol(...))`.

**Gate invocation:** `python3 tmp/uhj_parity/perf2/enumerate.py --gate g01`

**Predicted outcome:** every sampled symbol whose frame lies inside the join maps to
an enumerated loop or to an explicit exclusion with a recorded reason; **zero
unexplained symbols**.

**Predicted failure mode, registered in advance so it is not rationalised later:** I
expect the *first* run to be RED, and I expect the unexplained symbols to be
concentrated in (a) column `insertFrom`/`insertRangeFrom` instantiations reached from
the result-gather loops, and (b) arena/allocator symbols reached from `RowRefList`
append. Both are real per-row work and belong in the enumeration; discovering them is
the gate doing its job, not a defect. What would genuinely refute the enumeration
approach is an unexplained symbol inside the *lookup* or *insert* path, because that
is the part I claim to have read exhaustively.

**Refuted if:** a symbol with >=1% of in-join samples cannot be mapped to any
enumerated loop even after extending the enumeration — that means the entry points I
read are not the whole hot path.

**Threshold for "inside the join":** a sampled stack is in-join if any frame matches
`DB::HashJoin`, `DB::Unified::`, `DB::ConcurrentHashJoin`, `DB::JoinStuff`,
`NotJoinedHash`, `AddedColumns`, `HashJoinResult`, `RowRefList`, `TwoLevelHashTable`,
`HashTable<`, `ColumnsHashing`, `JoiningTransform`, `FillingRightJoinSide`, or
`NonJoinedBlocksTransform`. This rule is fixed here, before the samples are looked at,
so it cannot be tuned to make the gate pass. Attribution is by **leaf frame** (the
innermost in-join frame), because that is where the cycles are.

---

## P0.2 — Unit 0, lock-enumeration completeness (gate G0.2)

**Claim under test:** the static grep-plus-reading enumeration of locks and atomics
finds the same **set** of locks/atomics that a dynamically instrumented binary
observes being taken on the hot path.

**Instrument:** an `INSTRUMENTATION` patch adding a per-site counter and a
cycle-counter hold-time histogram to every enumerated lock site, plus a catch-all: I
will additionally confirm the set with `perf` and by asserting that sites the static
enumeration says should *never* fire on a given path have a zero count.

**Predicted outcome:** the sets agree. Specifically I predict:
- `hash` build takes **zero** join-level mutexes (only `StoredColumnsIndex`'s, which
  is shared infrastructure, and which I therefore expect to be non-zero for all three);
- `unified_hash` build takes the per-bucket lock and `blocks_mutex`;
- `parallel_hash` build takes the per-slot mutex;
- **all three probe paths take zero join-level mutexes**, the probe being lock-free
  apart from `StoredColumnsIndex::resolveEmitColumns` once per probe batch and the
  relaxed/seq_cst atomics in `JoinUsedFlags`.

**Refuted if:** instrumentation records a non-zero count at a site the static
enumeration marked unreachable on that path, or a lock fires that the enumeration
does not list at all.

**Registered risk:** a static grep misses locks reached through templates and inlined
helpers. That is exactly why this gate is dynamic as well as static, and I expect
`StoredColumnsIndex`'s mutex (`RowRefs.h:482`) to be the one a naive grep of the three
join directories would have missed — it is reached from all three and lives in neither.

---

## P0.3 — Correction registered before measurement: the "identical probe path" premise

The mission brief states probe match/non-match/gather is "reported textually
identical between the two trees apart from namespace" and asks me to verify it.

**Registered finding, from reading only (no measurement yet), so that it is on record
before any measurement is designed around it:** the premise is **partly false**.

- **Identical modulo namespace/include:** `AddedColumns.{h,cpp}`, `HashJoinResult.cpp`,
  `KnownRowsHolder.h`, `joinDispatch.h`, and the body of `processMatch`.
- **Materially different:** `HashJoinMethodsImpl.h` (unified adds `scatterByBucket*`,
  routes probe lookups through `map->prober()` rather than `findKey(*map, ...)`, and
  prefetches through the prober) and `KeyGetter.h` (unified templates `use_offset`
  where the baseline hardcodes it `true` at `HashJoin/KeyGetter.h:19`, and replaces
  the LowCardinality `getHash` with `routingHashForRow`).
- **Also different:** `JoinUsedFlags.h` — unified **removed** `allOffsetFlagsSet()`,
  the baseline's all-matched early-out.

**Consequence for the mission's own reasoning, registered now:** the brief's inference
that identity "bounds where a probe difference can live" does not hold as stated. The
probe difference can live in the lookup and key-getter layers, which is where claim A6
already points. I will treat the gather layer (`AddedColumns`, `HashJoinResult`) as
verified-identical and therefore excluded, and the lookup/key-getter layer as in scope.

**Refuted if:** a normalised diff run as part of Unit 1 shows any of the files I have
listed as identical in fact differing in a way that changes emitted code — which I
will check by the symbol-level byte comparison of G5.3's technique, not by reading.

---

## P1.1 — Correction: `IN_JOIN_MARKERS` was asymmetric between the trees

Registered as a correction to P0.1, before the re-run it justifies.

`unified_hash` symbols all carry the `DB::Unified::` prefix, so one blanket marker
catches the whole tree. The baseline has no such prefix, so every baseline counterpart
must be named individually. Two were missing: `DB::LazyOutput::` and `DB::processMatch`.

A missing marker does not *lose* samples — the leaf-in-join scan simply walks one frame
further out and attributes the sample to the caller. So the effect is invisible in
G0.1's completeness count (which stayed green throughout) and shows up only as a
**biased per-loop share between the two trees**. It produced a false finding: the gather
loop appeared to be inlined in the baseline and out-of-line in unified, on the hottest
loop in the join. It is not — the two symbols are byte-identical, 397 instructions with
zero positional mismatches (`codegen/P1_G2_probe_and_gather.md`).

**Change:** add `DB::LazyOutput`, `DB::processMatch`, `DB::addFoundRowAll`,
`DB::CollectorNonJoined`, `DB::JoinOnKeyColumns`, `DB::Inserter` and the
`packFixed`/`fillFixedBatch` family; add a mechanical symmetry audit that asserts, for a
list of known counterpart pairs, that the unified and baseline names are matched
identically. The audit is what stops this recurring, since it fails on a pair rather
than relying on someone noticing.

**Prediction for the re-run (`--tag u0b`):** G0.1 stays green (completeness was never
the failing property), and the baseline's gather samples move from
`HashJoinResult::generateBlock` to `LazyOutput::buildOutputFromBlocks`, bringing the two
trees' G2 shares into agreement. **Refuted if** the two trees' G2 shares still differ by
more than the sampling error after the fix — that would mean there IS a real difference
and the byte-comparison is somehow wrong.

---

## P1.2 — The per-partition key-getter packing: a K-scaling natural experiment

**The finding to test.** `HashMethodKeysFixed`'s *constructor* calls `packFixedBatch`
when `usePreparedKeys` holds (no nullable/LowCardinality keys, `sizeof(Key) <= 16`, all
key sizes in {1,2,4,8,16}) — true for the two-UInt64 composite key and false for the
single-UInt64 key, which uses `HashMethodOneNumber` and packs nothing.
`fillFixedBatch` sizes its output by **`column->size()`** — the whole block — not by the
selector (`src/Interpreters/AggregationCommon.h:65-66`). And `unified_hash` constructs a
key getter **once per bucket** inside `insertFromBlockImplTypeCase`
(`UnifiedHashJoin/HashJoinMethodsImpl.h:356`), plus once for the scatter pass (`:166`).

So the composite-key build cost should contain a term **linear in the partition count**,
present in `parallel_hash` (S = threads partitions) and `unified_hash` (K = 2 x
bit_ceil(threads) partitions) and absent in `hash` (one partition), and it should be
absent for `u64` in all three.

**Experiment (natural, no patch, runs on the delivered binary).** Sweep `max_threads`
over 4, 8, 16, 32, 64 at fixed data (`large`), for `u64` and `comp`, all three
algorithms, measuring build-phase time (`FillingRightJoinSide`) and CPU per build row.
`K = bucketCountForThreads(t)` takes the values 8, 16, 32, 64, 128; `S = t`.

**Predictions, written before the run:**

1. For **`comp` + `unified_hash`**, build CPU per build row fits `a + b*K` with
   `b > 0` well outside noise; extrapolating to `K=1` recovers roughly the `hash` cost.
2. For **`u64` + `unified_hash`**, the same fit gives a `b` at least 4x smaller,
   because no packing happens on that path.
3. **`hash`** shows no K term at all in either key type (it has one partition).
4. `parallel_hash` shows the same effect against `S`, since it constructs one key getter
   per slot per block for the same reason.

**Refuted if:** `comp` and `u64` show the same per-partition coefficient (then the
scaling is contention or cache footprint, not key packing), or if `comp`'s coefficient
is inside the noise band (then the packing is real in the code and irrelevant in time).

**Why this is a natural experiment and not an ablation:** thread count is a setting, so
nothing is rebuilt and nothing can leak into the tree. The confound it must survive is
that thread count changes contention as well as K — which is exactly why the `u64` arm
is run: it has the same contention change and no packing.

---

## P3.1 — ABLATION A-K1: halve the bucket count (`BUCKETS_PER_THREAD` 2 -> 1)

Written before the patch is applied. The patch is a one-token change to
`BUCKETS_PER_THREAD`, which feeds `bucketCountForThreads`
(`UnifiedHashJoin/HashJoin.cpp:66-74`), so `K` goes from `2*bit_ceil(T)` to
`bit_ceil(T)`: 128 -> 64 at 64 threads, 32 -> 16 at 16 threads.

**What it tests.** Whether the per-partition term measured in P1.2 is load-bearing, and
therefore whether "K = 2 x bit_ceil(threads)" is a FUNDAMENTAL requirement of the
one-shared-map design or a CONTINGENT tuning choice.

**Prediction, derived from the P1.2 fit rather than from an instruction ratio.** The fit
is `ns_per_build_row = a + b*K` with, for `comp` + `unified_hash`, `a = 57.95` and
`b = 1.1808` (R^2 = 0.964, n = 5). Halving K at 64 threads therefore predicts

    before (K=128): 57.95 + 1.1808*128 = 209.1   (measured 202.5)
    after  (K= 64): 57.95 + 1.1808* 64 = 133.5
    predicted change: -75.6 ns per build row, i.e. -34.4% of build-phase time

and for `u64` + `unified_hash` (`a = 38.52`, `b = 0.1577`) only

    before (K=128): 58.7  (measured 54.4)      after (K=64): 48.6
    predicted change: -10.1 ns/row, -17%   -- and this arm is a weak fit (R^2 = 0.572),
    so it is registered as "much smaller than comp", not as a number.

**Registered direction and rough magnitude:** build phase for `comp` at 64 threads gets
FASTER by 25-40%; `u64` at 64 threads gets faster by much less; `hash` and
`parallel_hash` are unchanged (the constant is not in their code path).

**What would refute it.** Build time for `comp` at 64 threads unchanged, or changed by
less than the noise band -- that would mean the K-linear term measured in P1.2 is not
caused by anything K actually controls, and the mechanism attribution is wrong.
A change in `hash` or `parallel_hash` beyond noise would mean the ablation is not
isolated and the result must be discarded.

**Validity check, required before the result is believed (an unvalidated null is not
evidence):** the ablation binary must be shown to differ from the reference binary only
in unified code. Checked with the G5.3 technique -- `llvm-nm` symbol-table diff, and the
`fillFixedBatch` symbol itself byte-compared, since it is shared code and must NOT
change. Additionally, the measured bucket count is observable at runtime: acquisitions
per block in the lock instrumentation would be 64 rather than 128, but since the
ablation binary is not instrumented, the check used here is the symbol diff plus the
build-side effect being confined to `unified_hash`.

**This is an ablation, not a proposed fix.** Halving the bucket count changes contention
as well as the packing cost, so a favourable result does not by itself recommend the
change; it establishes that K is a live lever and that the per-partition term is real.

---

## P3.2 — G3.2: hardware counters on the one-thread probe deficit

The gap the prior mission left and the brief's stated first step. Written before any
`perf` run.

**Question.** The one-thread deficit (10 cells, mean +6.54% wall) is attributed by the
codegen artefact to the probe lookup path: the `Prober` does not stay in registers, so
per row `unified_hash` reloads `shift`/`max_bucket`/`prefix` from the stack, and its
sub-table pointer depends on the hash (dependent-load depth 3 -> 5, 2 spill stores and
9 spill reloads against 0 and 1). Two mechanisms predict the same *sign* and must be
told apart:

| mechanism | signature in the counters |
| --- | --- |
| **instruction-count cause** (the codegen story) | instructions per probe row UP, IPC roughly flat, `LLC-load-misses` per row roughly unchanged |
| **cache-footprint cause** (candidate A5: `per_offset_flags` sized by summed bucket capacities, so wider and sparser) | `LLC-load-misses` and/or `dTLB-load-misses` per row UP, **IPC DOWN**, instructions per row roughly unchanged |

**Method.** `perf stat -p <server pid> -e cycles,instructions,cache-misses,LLC-load-misses,dTLB-load-misses`
around each query, server otherwise idle. The probe phase is isolated by subtraction:
the harness's `buildonly` mode reduces the probe side to nearly nothing
(`WHERE k < 0`, not `LIMIT 0`, so the planner cannot prune the join), so
`probe counters = full - buildonly`. Cell `INNER|u64|hi|t1|medium`: 1M build rows,
2M probe rows, no non-joined phase. Algorithms interleaved, 7 reps each.

**Prediction, registered:** instructions per probe row for `unified_hash` exceed `hash`
by **more than 10%**; IPC differs by **less than 10% relative**; `LLC-load-misses` per
probe row differ by **less than 20% relative**. That is the instruction-count signature,
and it is what the mca analysis implies, since mca reports both sides
dispatch-width-limited (`Block RThroughput` = uops/6 exactly on each) rather than
latency-bound.

**Refuted if:** IPC for `unified_hash` is more than 10% lower while instructions per row
are within 10% — that is the cache-footprint signature, and it would mean the codegen
story is the wrong mechanism and A5 is the right one. Per the brief, a codegen story
contradicted by the counters is REFUTED or downgraded, not accepted with a caveat.

**Also possible and registered in advance:** both signatures together (instructions up
*and* IPC down). That is not a fudge — it would mean both mechanisms are live, and the
report must then say the split is unresolved rather than pick one.

---

## P3.3 — Loop P2 (software prefetch): a settings-only ablation, no rebuild

Raised as a blocking finding by the independent verifier: P2 was UNSETTLED while a
checkable experiment was available in this environment. Registered before running it.

`enable_software_prefetch_in_join` turns the per-row prefetch (loop P2) off for all
three implementations without a rebuild, so nothing can leak into the tree.

**Prediction.** The codegen artefact says prefetch costs 14 instructions per row on the
baseline and 20 on unified — so if prefetch were pure overhead, disabling it would make
both faster and unified faster by more. It is not overhead: it exists to hide hash-table
misses, and the medium cell's table (1M rows x ~48 B) is far larger than the 2 MiB L2.
**I predict disabling it makes BOTH slower, and that the `hash`-to-`unified_hash` gap
does not close** — i.e. P2 is not a source of the one-thread deficit.

**Refuted if** the gap closes materially with prefetch off (that would make P2 a
contributor), or if disabling prefetch makes either implementation faster (that would
mean the prefetch is mistuned for this cell).
