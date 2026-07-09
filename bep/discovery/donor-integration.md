# Donor integration surface outside `src/Interpreters/RadixHashJoin/`

Scope: everything `origin/phj5-real` changed relative to its merge-base with our branch
(`e69a9d5ba75`), excluding `src/Interpreters/RadixHashJoin/` itself. Our branch:
`radix-join-bandwidth-model`, HEAD `0709d550de9`, branch base `2834291df98` (a descendant of
`e69a9d5ba75`; verified with `git merge-base --is-ancestor`).

Method: `git diff e69a9d5ba75..origin/phj5-real -- . ':(exclude)src/Interpreters/RadixHashJoin'`
read hunk-by-hunk for every file; drift quantified with `git diff e69a9d5ba75..HEAD -- <file>`
and an exact 3-way simulation `git merge-tree --write-tree HEAD origin/phj5-real`.

## Headline merge result {#headline}

`git merge-tree --write-tree HEAD origin/phj5-real` produces exactly **one content-conflicted
file**: `src/Interpreters/ConcurrentHashJoin.cpp` (2 conflict regions, both trivial — see below).
Every other file auto-merges. However there are **two semantic mis-merges the tool does not
flag**: the `SettingsChangesHistory.cpp` entries land in the already-released `26.6` block, and
`FillingRightJoinSideTransform` on HEAD calls `onBuildPhaseFinish` from `prepare` (not `work`),
which changes where the donor's heavy eager post-build executes.

## Complete touched-file list (41 files, 2571 insertions, 12 deletions) {#file-list}

Full stat (donor vs base):

```
src/CMakeLists.txt                                 |   1 +
src/Common/CurrentMetrics.cpp                      |   3 +
src/Common/ProfileEvents.cpp                       |  10 +
src/Common/ScopedLLCMissCounter.cpp                |  78 + (new)
src/Common/ScopedLLCMissCounter.h                  |  33 + (new)
src/Common/setThreadName.h                         |   1 +
src/Core/Joins.cpp / Joins.h                       |   1 + each
src/Core/Settings.cpp                              |   6 +
src/Core/SettingsChangesHistory.cpp                |   2 +
src/Core/SettingsEnums.cpp                         |   3 +-
src/Interpreters/CMakeLists.txt                    |   4 +
src/Interpreters/ConcurrentHashJoin.{cpp,h}        |  17 + / 2 +
src/Interpreters/{Direct,GraceHash,Merge,Paste,FullSortingMerge,SpillingHash}Join.h, JoinSwitcher.h | 2-3 + each
src/Interpreters/HashJoin/HashJoin.h               |   1 +
src/Interpreters/HashTablesStatistics.{cpp,h}      |   7 + / 13 +
src/Interpreters/IJoin.h                           |  11 +
src/Interpreters/JoinOperator.{cpp,h}              |  10 + / 2 +
src/Interpreters/benchmarks/CMakeLists.txt         |   7 + (new)
src/Interpreters/benchmarks/bench_build_bandwidth.cpp | 565 + (new)
src/Interpreters/benchmarks/bench_rhj_vs_chj.cpp   | 881 + (new)
src/Interpreters/benchmarks/bench_scatter_perf.cpp | 665 + (new)
src/Planner/PlannerJoins.{cpp,h}                   | 101 + / 4 +
src/Processors/QueryPlan/QueryPlanSerializationSettings.cpp | 2 +
src/Processors/Transforms/JoiningTransform.{cpp,h} |  12 +- / 11 +-
src/QueryPipeline/QueryPipelineBuilder.cpp         |  11 +-
tests/queries/0_stateless/04316_radix_hash_join_gate_and_fallback.{sql,reference} (new)
tests/queries/0_stateless/04337_radix_hash_join_distinct_estimate.{sql,reference} (new)
```

**No docs files are touched** (setting docs are the `DECLARE` descriptions in `Settings.cpp`).
No integration tests, no performance tests, no `.md`.

---

## 1. `src/Interpreters/IJoin.h` — lane-aware overloads {#ijoin}

Donor adds two non-pure virtuals with forwarding defaults (donor lines 115–118 and 132–134):

- `virtual bool addBlockToJoin(const Block & block, size_t num_rows, bool check_limits, size_t /*build_lane*/)`
  — default forwards to the 3-arg overload. Comment: 0-based build lane `< max_threads`, lets a
  join bind stable per-lane build state without a thread-local slot cache.
- `virtual JoinResultPtr joinBlock(Block block, size_t /*lane*/)` — default forwards to
  `joinBlock(Block)`. Same rationale for per-lane probe scratch without locking.

Drift on HEAD: +3 lines only — a new unrelated virtual `setEnableLazyColumnsIndexing(bool)` at
IJoin.h:186 (lazy columns indexing on hash join variants). No overlap; auto-merges.
**Port difficulty: trivial.**

## 2. `using IJoin::...` in every implementer (silences `-Woverloaded-virtual`) {#using-decls}

Because derived classes override only some `addBlockToJoin`/`joinBlock` overloads, the donor adds
`using IJoin::addBlockToJoin;` / `using IJoin::joinBlock;` to: `ConcurrentHashJoin.h`,
`DirectJoin.h`, `FullSortingMergeJoin.h`, `GraceHashJoin.h`, `HashJoin/HashJoin.h`
(`addBlockToJoin` only), `JoinSwitcher.h`, `MergeJoin.h`, `PasteJoin.h`, `SpillingHashJoin.h`.
The set of `IJoin` implementers on HEAD is identical (verified: no new `joinBlock` overriders
appeared on master), so no extra class needs the treatment.

Drift: `DirectJoin.h`, `GraceHashJoin.h`, `MergeJoin.h`, `PasteJoin.h` are byte-identical to base
on HEAD. Others drifted for unrelated reasons (`JoinSwitcher.h` + `SpillingHashJoin.h` gained
`StatsCollectingParams` ctor args; `FullSortingMergeJoin.h` gained
`isMergeAlgorithmStrictnessAndKindSupported`; `ConcurrentHashJoin.h` lost deferred-build members;
`HashJoin.h` ctor now takes `reserve_num_/instance_id_/use_two_level_maps_/stats_collecting_params_`).
All hunks non-overlapping; auto-merge clean. **Port difficulty: trivial.**

## 3. `src/Core/Joins.{h,cpp}`, `src/Core/SettingsEnums.cpp` — the enum {#join-algorithm-enum}

`JoinAlgorithm::RADIX_HASH` appended as the last enumerator in `src/Core/Joins.h` (after
`FULL_SORTING_MERGE`), `toString` case in `Joins.cpp`, and `{"radix_hash", JoinAlgorithm::RADIX_HASH}`
in `IMPLEMENT_SETTING_MULTI_ENUM(JoinAlgorithm, ...)` in `SettingsEnums.cpp`.
Drift on HEAD: tiny, non-overlapping (other enum lists changed). **Port difficulty: trivial.**

Note: the **old analyzer** (`src/Interpreters/ExpressionAnalyzer.cpp` has its own
`tryCreateJoin`/`chooseJoinAlgorithm`) was NOT taught about `RADIX_HASH`; with
`enable_analyzer = 0` and `join_algorithm = 'radix_hash'` the algorithm is simply never matched
there. Donor tests pin `enable_analyzer = 1`. Decide whether to leave the old path untouched
(donor's choice) or add an explicit fallback there too.

## 4. Settings — `src/Core/Settings.cpp`, `SettingsChangesHistory.cpp`, `QueryPlanSerializationSettings.cpp`, `JoinOperator.{cpp,h}`, `PlannerJoins.h` {#settings}

Two new query settings (both `DECLARE(..., 0)`, no tier flags):

- `max_partitions_per_pass` (UInt64, default **8192**): "the maximum number of partitions (fanout)
  produced by a single radix scatter pass. The total leaf count is split into the minimum number
  of passes that respect this cap."
- `radix_hash_join_size_tables_by_distinct_estimate` (Bool, default **true**): size each leaf hash
  table by a HyperLogLog distinct-key estimate computed during the build scatter (only ever
  shrinks) instead of raw row count.

Plumbing (all auto-merges cleanly):
- `SettingsChangesHistory.cpp`: 2 entries — `{"max_partitions_per_pass", 8192, 8192, ...}` and
  `{"radix_hash_join_size_tables_by_distinct_estimate", false, true, ...}`.
- `QueryPlanSerializationSettings.cpp`: matching `DECLARE`s (needed so the settings survive plan
  serialization).
- `JoinOperator.h` `struct JoinSettings`: two fields; `JoinOperator.cpp`: both ctors
  (`Settings` and `QueryPlanSerializationSettings`) plus `updatePlanSettings` copy them.
- `PlannerJoins.h` `struct JoinAlgorithmParams`: two fields; `PlannerJoins.cpp`: both
  `JoinAlgorithmParams` ctors populate them (from `Context` settings and from `JoinSettings`).

**Semantic mis-merge (must fix by hand):** on HEAD `SettingsChangesHistory.cpp` was restructured
into `addSettingsChanges(settings_changes_history, "26.7", {...})` blocks and the donor's anchor
entry was renamed (`enable_join_runtime_filter_shared_fixed_hash_table` →
`join_runtime_filter_from_fixed_hash_table`, now in the **26.6** block, HEAD line ~82). The
auto-merge drops the donor's two entries into the **released 26.6 block** (merged-file line ~86,
right after `ai_function_embedding_max_batch_size`). They must go into the current dev block
(**"26.7"**, starts at HEAD line 42) instead. CI has a settings-history consistency check that can
catch this, but do not rely on it.

Drift elsewhere: `Settings.cpp` on HEAD drifted heavily (+350/-121) but the donor's insertion
anchor (`parallel_hash_join_threshold`) is intact; merged file places `max_partitions_per_pass`
at ~line 7916. `JoinOperator.{cpp,h}` drift is only the runtime-filter setting rename — no overlap.
**Port difficulty: low (trivial except the history-block placement).**

Naming risk: `max_partitions_per_pass` has no `radix_hash_join_` prefix; expect review pushback
and consider renaming while porting (donor kept the generic name in both `Settings.cpp` and
`QueryPlanSerializationSettings.cpp` — they must stay in sync, and the
`SettingsChangesHistory` entry must match the final name).

## 5. `src/Planner/PlannerJoins.cpp` — gate + creation + fallback {#planner}

Donor adds (all in the anonymous/static section before `tryCreateJoin`):

### 5a. Exact gate predicate — `radixHashJoinApplicable(table_join, right_table_expression_header)` (donor lines 1174–1214)

Returns true iff ALL of:
1. `table_join->oneDisjunct()`
2. `table_join->kind() == JoinKind::Inner`
3. `table_join->strictness() == JoinStrictness::All` (v1 is ALL-inner only; ANY/SEMI/ANTI/ASOF
   excluded on purpose — comment says they "would not be bit-identical to `hash` under the
   parallel passthrough")
4. `!table_join->isSpecialStorage()`
5. `key_names_right = table_join->getOnlyClause().key_names_right` is non-empty
6. for every right key column (looked up by name in `right_table_expression_header`):
   - column exists in the header
   - `!type->isNullable() && !type->lowCardinality()`
   - `type->haveMaximumSizeOfValue()` (fixed width)
   - `width = type->getMaximumSizeOfValueInMemory()` and `width != 0`
7. `packed_key_width = Σ width` satisfies `packed_key_width % 4 == 0 && packed_key_width >= 4 && packed_key_width <= 64`
   (4-byte scatter granularity; 64-byte leaf-cell template bound). So a lone
   `UInt8`/`UInt16`/`Date`/`Enum8`/`Enum16` fails (not a multiple of 4), as does any composite
   whose sum is not a multiple of 4 or exceeds 64 bytes.

### 5b. Fallback — `createRadixHashJoinFallback(table_join, right_table_expression_header, params)` (donor lines 1217–1241)

When the gate fails: if `table_join->oneDisjunct() && !isSpecialStorage() && strictness != Asof &&
kind ∈ {Left, Inner, Right, Full}` → `ConcurrentHashJoin(table_join, params.max_threads,
right_table_expression_header, StatsCollectingParams{...})`; otherwise
`HashJoin(table_join, right_table_expression_header, params.join_any_take_last_row)`.
Notably it **ignores `parallel_hash_join_threshold` / `rhs_size_estimation`** (always
parallel_hash when shape allows) and does not consult `table_join->allowParallelHashJoin()`.

### 5c. Dispatch in `tryCreateJoin` (donor lines 1267–1285)

Inserted before the `HASH/PREFER_PARTIAL_MERGE/PARALLEL_HASH/DEFAULT` branch:
`if (algorithm == JoinAlgorithm::RADIX_HASH)` → if gate passes, construct
`RadixHashJoin(table_join, right_table_expression_header, params.max_threads,
params.rhs_size_estimation, params.max_partitions_per_pass,
params.radix_hash_join_size_tables_by_distinct_estimate, StatsCollectingParams{...})`; else the
fallback. There is no size-threshold auto-selection: `radix_hash` only runs when explicitly listed
in `join_algorithm` (and `chooseJoinAlgorithm` tries algorithms in the order listed).
Both the old planner path and the new `JoinStepLogical` path funnel through this same
`chooseJoinAlgorithm`/`tryCreateJoin` (verified: `src/Processors/QueryPlan/JoinStepLogical.cpp:1423`
calls `chooseJoinAlgorithm` from `PlannerJoins`).

Drift on HEAD (+22/-20): `tryCreateJoin` was refactored — `StatsCollectingParams` hoisted to the
top of the HASH-family and AUTO branches; `HashJoin` ctor call sites now pass
`/*reserve_num_=*/0, /*instance_id_=*/"", /*use_two_level_maps_=*/false, stats_collecting_params`;
`GraceHashJoin`/`JoinSwitcher` now take `stats_collecting_params`. None of these hunks overlap the
donor insertion points — auto-merge is clean. But **the donor fallback's plain
`HashJoin(table_join, header, join_any_take_last_row)` should be updated to pass
`stats_collecting_params` like HEAD's other call sites** (the 3-arg call still compiles because the
extra params have defaults, but it silently loses hash-table stats). `PlannerJoins.h` is unchanged
on HEAD. **Port difficulty: low; small deliberate adaptation recommended.**

## 6. `src/Processors/Transforms/JoiningTransform.{h,cpp}` — lane plumbing {#joining-transform}

- `JoiningTransform` ctor gains trailing `size_t stream_index_ = 0`; stored as `stream_index`;
  `readExecute` calls `join->joinBlock(std::move(block), stream_index)` instead of the 1-arg form.
- `FillingRightJoinSideTransform` ctor gains trailing `size_t build_lane_ = 0`; stored;
  `work()` calls `join->addBlockToJoin(block, num_rows, true, build_lane)`.

Drift on HEAD (+6/-25) — **the important semantic drift**: master commit `95b3b5b07e2` ("Scope
this pull request to the compact row references", merged via PR #107189) **removed** the
`finish_build_phase`/`build_phase_finished` state machine that the donor base had. On HEAD,
`FillingRightJoinSideTransform::prepare` calls `join->onBuildPhaseFinish()` **directly inside
`prepare`** (when `finish_counter->isLast()`), then sets `post_build_phase` if
`join->hasPostBuildPhase()`; `runPostBuildPhase` still runs in `work()` (timed by
`JoinBuildPostProcessingMicroseconds`). The donor's hunks don't overlap the removed code, so the
auto-merge is textually clean.

**Consequence for the port:** donor `RadixHashJoin` does its entire heavy post-build (histogram
merge, scatter, leaf-table builds, parallelized over its own thread pool) inside
`onBuildPhaseFinish` (`RadixHashJoin.h:80`, `RadixHashJoin.cpp:722`) and does NOT implement
`hasPostBuildPhase`/`runPostBuildPhase`. On HEAD that heavy work would execute inside `prepare` of
the last filling transform. ConcurrentHashJoin's bucket merge already runs there on HEAD, so it
"works", but for RadixHashJoin consider moving the eager post-build behind
`hasPostBuildPhase() = true` + `runPostBuildPhase()` to get work-context execution (and the
existing profile event) instead of resurrecting the reverted state machine.
**Port difficulty: low textually; medium semantically (decide the post-build hook).**

## 7. `src/QueryPipeline/QueryPipelineBuilder.cpp` — lane assignment {#pipeline-builder}

In `joinPipelinesRightLeft`:
- `concurrent_right_filling_transform` lambda: a `size_t build_lane = 0` counter, incremented per
  resized outport in both the squashing and non-squashing loops, passed to each
  `FillingRightJoinSideTransform` (one filling transform per outport == one lane).
- probe side: the per-stream loop index `i` is passed as `stream_index` to each `JoiningTransform`.
- `joinPipelinesByShards`: shard index `i` passed as `stream_index`.

Drift on HEAD (+16/-1): only `addDefaultTotals` (const-column totals fix) — different function, no
overlap; auto-merges. **Port difficulty: trivial.**

Lane-coverage caveats found by grep (call sites that keep the default lane 0):
- `FilledJoinStep::transformPipeline` (`src/Processors/QueryPlan/JoinStep.cpp:402`) creates
  `JoiningTransform` without a stream index for filled joins (StorageJoin/dictionary). Unreachable
  for RadixHashJoin because the gate rejects `isSpecialStorage()`, and `HashJoin::isFilled()` is
  `from_storage_join`. Same for the donor branch (unchanged there too).
- `QueryPipelineBuilder.cpp:768` (`joinPipelinesYShapedByShards` area on HEAD) creates a
  `FillingRightJoinSideTransform` with default lane — verify reachability for `radix_hash` when
  porting; if reachable, all builds land on lane 0 and per-lane state must tolerate that (donor's
  IJoin comment implies lanes must be `< max_threads`, not necessarily distinct).

## 8. `src/Interpreters/ConcurrentHashJoin.{cpp,h}` — instrumentation only (THE one conflict) {#chj}

Donor changes are pure benchmarking instrumentation (+17):
`ProfileEventTimeIncrement<Microseconds>` build watches in `addBlockToJoin` and
`onBuildPhaseFinish` (`ConcurrentHashJoinBuildMicroseconds`), probe watches + `ScopedLLCMissCounter`
in `joinBlock` and in `ConcurrentJoinResult::next` (`ConcurrentHashJoinProbeMicroseconds`,
`ConcurrentHashJoinProbeLLCMisses`), plus the header's `using` declarations.

Drift on HEAD is large (+68/-308 vs base): the deferred exact-size build (`deferred_build`,
`reserveBucketsBySize`, `HashJoinDeferredPreallocatedElementsInHashTables`) was scoped out of the
merged PR #107189; our branch additionally added `ConcurrentHashJoinProbeDispatchMicroseconds`
timing in `joinBlock` (commit range `2834291df98..HEAD`).

The 2 conflict regions in the merge simulation:
1. `namespace ProfileEvents` extern block (~line 63): HEAD added
   `ConcurrentHashJoinProbeDispatchMicroseconds`, donor added its 3 events + the (now-deleted)
   `HashJoinDeferredPreallocatedElementsInHashTables`. Resolution: union, drop the deferred one
   (it no longer exists in HEAD's `ProfileEvents.cpp` — re-adding the extern would fail to link).
2. `onBuildPhaseFinish` head (~line 776): HEAD deleted the whole `deferred_build` replay block that
   base had; donor kept it and prepended the `build_watch` line. Resolution: HEAD body + the
   `build_watch` line.

**Port difficulty: low** — and the whole file is optional: this instrumentation exists to compare
CHJ vs RHJ in benchmarks. Skipping it (and the LLC counters) shrinks the port.

## 9. `src/Common/*` — events, metrics, thread name, LLC counter {#common}

- `ProfileEvents.cpp` (+10): `ConcurrentHashJoin{Build,Probe}Microseconds`,
  `ConcurrentHashJoinProbeLLCMisses`, `RadixHashJoin{Build,Probe}Microseconds`,
  `RadixHashJoinProbeLLCMisses`, `RadixHashJoinProbePermMicroseconds`,
  `RadixHashJoinProbeCollectMatchesMicroseconds`, `RadixHashJoinProbePackHashRouteMicroseconds`.
  Inserted after `JoinBuildPostProcessingMicroseconds` (HEAD line 1517). Auto-merges. Our branch
  already added 3 events elsewhere in the file (`HashJoinProbe{Match,Gather}Microseconds`,
  `ConcurrentHashJoinProbeDispatchMicroseconds` at ~line 425) — no name collisions.
- `CurrentMetrics.cpp` (+3): `RadixHashJoinPoolThreads{,Active,Scheduled}` for the RHJ post-build
  pool. Auto-merges.
- `setThreadName.h` (+1): `M(RADIX_JOIN, "RadixJoin")` in the thread-name macro list. Auto-merges.
- `ScopedLLCMissCounter.{h,cpp}` (new, 111 lines): per-thread lazily-opened
  `perf_event_open(PERF_TYPE_HW_CACHE, LL|READ|MISS, exclude_kernel)` counter; RAII scope adds the
  delta to a ProfileEvent on destruction; silent no-op if the fd can't be opened (non-Linux or
  `perf_event_paranoid`). Donor history note: it was stripped from production once (`9bbf8516de9`)
  and re-added (`e02113ce338`, `4573e142011`) — it is benchmarking instrumentation; decide whether
  to port it at all. Production code opening perf_event fds may raise review/security questions.

**Port difficulty: trivial (all optional except the RHJ events/metrics that RadixHashJoin.cpp
references — check which ones the RHJ sources actually use).**

## 10. `src/Interpreters/HashTablesStatistics.{h,cpp}` — warm-run distinct-keys cache {#stats}

New `RadixHashJoinEntry { size_t distinct_keys; }` with
`shouldBeUpdated = new < old/2 || old < new` and `dump`; registered in
`getHashTablesCacheStatistics` aggregation and explicitly instantiated
(`template class HashTablesStatistics<RadixHashJoinEntry>`). Purpose: cache the estimated distinct
build keys per join so a warm run sizes leaf tables directly and skips the per-leaf HyperLogLog
estimation (donor commit `8cfd6341243`). Files unchanged on HEAD → **trivial**.

## 11. Build system {#build}

- `src/CMakeLists.txt`: `add_object_library(clickhouse_interpreters_radix_hash_join Interpreters/RadixHashJoin)`
  (after the `Interpreters/HashJoin` line). HEAD drifted but not around that anchor; auto-merges.
- `src/Interpreters/CMakeLists.txt`: `if (ENABLE_BENCHMARKS) add_subdirectory(benchmarks) endif()`.
  Unchanged on HEAD; the `ENABLE_BENCHMARKS` option already exists (top-level CMakeLists.txt:148)
  and is already used by our branch's `src/Common/benchmarks`.
- `src/Interpreters/benchmarks/CMakeLists.txt` (new): builds ONLY `bench_rhj_vs_chj` (links `dbms`).
  **`bench_build_bandwidth.cpp` and `bench_scatter_perf.cpp` are present but not wired into any
  CMake target** — dead in the donor build as committed.

## 12. Donor benchmarks (new, 2111 lines total, research harnesses) {#benchmarks}

- `bench_rhj_vs_chj.cpp` (881): drives real `RadixHashJoin` vs `ConcurrentHashJoin` through the
  IJoin API (`addBlockToJoin(..., build_lane)`, `onBuildPhaseFinish`, `joinBlock(block, lane)`) with
  its own pinned worker pool, for `SELECT count() ... INNER JOIN ... USING (k0)`, UInt64 key.
- `bench_build_bandwidth.cpp` (565): isolates build-side data movement — CHJ zero-copy dispatch
  replica vs RHJ scatter vs RHJ memcpy baseline.
- `bench_scatter_perf.cpp` (665): PMU-instrumented (perf_event groups, SPR raw configs)
  phase-attribution of the RHJ scatter vs memcpy.

Overlap warning: our branch already has its own benchmark suite in `src/Common/benchmarks/`
(`hash_join_bandwidth_model.cpp` 2022 lines, `radix_hash_join_bench.{h,cpp}`,
`concurrent_hash_join_bench.*`, `hash_join_bench.*`) — same problem domain, different location.
No path or symbol collisions found, but porting the donor benches verbatim duplicates
functionality; consider porting only `bench_rhj_vs_chj` (the only one CMake builds) or none.

## 13. Stateless tests {#tests}

- `04316_radix_hash_join_gate_and_fallback.sql` (+ .reference, 8 rows): tags `no-random-settings`,
  `SET enable_analyzer = 1`. Two Memory tables with UInt64/UInt32/UInt8/String/LowCardinality(String)/
  Nullable(UInt64)/payload columns, overlapping unequal key ranges, many-to-many duplicates. Eight
  cases each printing `1` when `radix_hash` equals `hash` on `(count(), sum(cityHash64(p.pay, b.pay)))`:
  `single_u64`, `single_u32`, `composite_u64_u32` (gate passes) and `fallback_u8`,
  `fallback_string`, `fallback_lowcard`, `fallback_nullable`, `fallback_composite_nm4`
  (gate fails → parallel_hash fallback; `nm4` = UInt64+UInt8 = 9 bytes, not a multiple of 4).
- `04337_radix_hash_join_distinct_estimate.sql` (+ .reference, 5 rows): duplicate-heavy build
  (200 distinct keys / 100k rows) and unique build; asserts `radix_hash` (estimate on/off) ==
  `hash` and on == off.

Drift: no filename collisions on HEAD (other `04316_*`/`04337_*` tests exist with different
names), but the current max test number is **04507** — per project convention, renumber with
`./tests/queries/0_stateless/add-test <name>` when porting. **Port difficulty: trivial.**

---

## Port-difficulty summary table {#summary}

| File | Donor change | HEAD drift vs donor base | Conflict risk | Verdict |
|---|---|---|---|---|
| IJoin.h | 2 lane overloads | +3 unrelated | none | trivial |
| Joins.{h,cpp}, SettingsEnums.cpp | enum + name | tiny | none | trivial |
| Settings.cpp | 2 settings | heavy but disjoint | none | trivial |
| SettingsChangesHistory.cpp | 2 entries | restructured blocks | **semantic** | manual: put in "26.7" |
| QueryPlanSerializationSettings.cpp | 2 settings | +1/-1 | none | trivial |
| JoinOperator.{cpp,h} | JoinSettings plumbing | rename-only | none | trivial |
| PlannerJoins.{cpp,h} | gate+fallback+dispatch+params | tryCreateJoin refactor | none textually | low; pass stats params to fallback HashJoin |
| JoiningTransform.{cpp,h} | stream_index/build_lane | onBuildPhaseFinish moved to prepare | **semantic** | low; rethink RHJ post-build hook |
| QueryPipelineBuilder.cpp | lane assignment | unrelated hunk | none | trivial |
| ConcurrentHashJoin.{cpp,h} | instrumentation | rewritten (deferred build removed; our dispatch timing) | **2 real conflicts** | low; optional — consider dropping |
| Other IJoin implementers (7 headers) | using-decls | some drift, disjoint | none | trivial |
| ProfileEvents/CurrentMetrics/setThreadName | events/metrics/name | disjoint | none | trivial |
| ScopedLLCMissCounter.{h,cpp} | new | n/a | none | optional (perf_event in prod) |
| HashTablesStatistics.{h,cpp} | RadixHashJoinEntry | unchanged | none | trivial |
| CMake (3 files) | radix obj lib + benches | disjoint | none | trivial |
| benchmarks (3 cpp + cmake) | new | n/a (we have our own suite in Common/benchmarks) | none | optional; 2 of 3 not even built |
| tests 04316/04337 | new | numbering moved on (max 04507) | none | renumber via add-test |

## Cross-cutting risks for the RadixHashJoin port itself (context, not this task's scope) {#cross-cutting}

- Donor merged the **unscoped** `compact-hash-join-row-refs` branch (`9ffceb9d301`); master merged
  a **scoped** version (PR #107189) that removed `ColumnsInfo` (replaced by `StoredColumnsIndex`
  emit tables, see `src/Interpreters/HashJoin/ScatteredBlock.h:373-375`) and the deferred CHJ
  build. Donor RHJ sources reference `BuildRefList`/`ColumnsInfo`
  (`RadixHashJoin/{BuildSide.h,KeyRefScatter.h,LeafTable.*,RadixHashJoin.cpp}` grep-hit) — these
  internals WILL need adaptation to HEAD's scoped row-ref shapes even though that code lives inside
  the excluded directory.
- Donor RHJ relies on `onBuildPhaseFinish` running in a `work()` context (see §6).
- New HEAD virtual `setEnableLazyColumnsIndexing` — RHJ can keep the default no-op, but check the
  probe result path doesn't assume lazy-column behavior it doesn't implement.
