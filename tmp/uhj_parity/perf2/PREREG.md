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
