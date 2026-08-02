# INVENTORY_U3 — exhaustive UHJ divergence inventory (Unit 1)

Tree: branch `uhj-parity`, code tip `f0420d93d31` (artifacts tip `13ec290c6c6`).

Baselines: `src/Interpreters/HashJoin/**` (`DB::HashJoin`, `join_algorithm='hash'`) and
`src/Interpreters/ConcurrentHashJoin.{h,cpp}` (`DB::ConcurrentHashJoin`, `join_algorithm='parallel_hash'`).
Fork under audit: `src/Interpreters/UnifiedHashJoin/**` (`DB::Unified::HashJoin`, `join_algorithm='unified_hash'`).

## Method — how completeness is established

`UnifiedHashJoin` is a whole-directory fork of `HashJoin`, so divergence is enumerable mechanically
rather than by sampling. Two artifacts:

- `U3_rawdiff.txt` — `diff -ru src/Interpreters/HashJoin src/Interpreters/UnifiedHashJoin` (3420 lines).
- `U3_normdiff.txt` — produced by `U3_normdiff.sh`, which strips the three mechanical fork
  transformations (the `namespace Unified { ... }` wrapper, `Interpreters/UnifiedHashJoin/` →
  `Interpreters/HashJoin/` include rewrites, and `Unified::` qualifier removal) and re-diffs.

Residual after normalization: **1464 changed lines in 6 files**. The other 36 of 42 files are
byte-identical to their baseline counterparts modulo the fork wrapper, so they contain no divergence
and need no row here.

```
626 HashJoin.cpp   364 HashJoin.h   152 HashJoinMethodsImpl.h
148 KeyGetter.h    108 JoinUsedFlags.h    66 HashJoinMethods.h
```

Plumbing outside all three implementation directories was audited separately (`src/Core`,
`src/Planner`, `src/Interpreters/{ExpressionAnalyzer,InMemoryHashJoin,SpillingHashJoin,GraceHashJoin,TableJoin}`,
`src/Processors`, `src/QueryPipeline`, `src/CMakeLists.txt`).

## Classification scheme

| Label | Meaning |
| --- | --- |
| **EXCLUDED** | Exists *only* because the fork's maps are unconditionally bucketed (`TwoLevelHashMap`/`PartitionedFixedHashMap`): bucket maps, per-bucket arenas, scatter-by-bucket, bucket-indexed locks, bucket-offset synthesis. Mission-excluded forever. Each row states the forcing mechanism. |
| **FORK-MECHANICAL** | Pure consequence of a second copy coexisting in one binary (macro redefinition clash, `getName()` string). Not TwoLevel-required, but unavoidable without deleting the fork — which is not this mission. Counted separately, never folded into EXCLUDED. |
| **MATERIAL** | Avoidable divergence, not TwoLevel-required. Must be aligned to baseline. |
| **LEAD** | Real but low-value / latent; no behavior difference today. |
| **UNSETTLED** | Cannot be aligned without violating MUST-HOLD; blocker named. |

`EXCLUDED` is used only where the *bucketing* forces the difference. Where the fork builds one
shared map from several threads and therefore does not use the `ConcurrentHashJoin` wrapper, the
forcing mechanism is the per-bucket lock — which is bucket-indexed and exists only because the map
is partitioned — so those rows are EXCLUDED with that mechanism spelled out, per the mission's
"bucket-indexed locks **when** they exist only because of TwoLevel".

---

## A. EXCLUDED — TwoLevel-required (grouped; 44 diff regions)

| ID | Divergence | Forcing mechanism |
| --- | --- | --- |
| E1 | `JoinHashMap`/`JoinHashMapWithSavedHash`/`JoinFixedHashMap` replace flat `HashMap`/`FixedHashMap`; `BITS_FOR_BUCKET=-1` runtime-sized | The map type *is* the excluded construct |
| E2 | `two_level_*` enum variants + their `KeyGetterForTypeImpl` specializations deleted; `chooseMethod(..., use_two_level_maps)` overload deleted; `twoLevelMapIsUsed()` deleted | Every map is already two-level, so a separate two-level *type family* is meaningless |
| E3 | `bucketCountForThreads`, `BUCKETS_PER_THREAD=2`, `num_buckets`, ctor takes `max_threads_` where baseline takes `use_two_level_maps_` | Bucket count is the runtime sizing parameter of the partitioned table |
| E4 | `BucketLock` (cache-line-padded mutex vector), `bucket_locks` | Bucket-indexed locks; exist only because the map is partitioned |
| E5 | `RightTableData::pools[]` per-bucket arenas (baseline: single `Arena pool`) | One arena per bucket so bucket-parallel inserts never share an allocator |
| E6 | `scatterByBucket` / `scatterByBucketTypeCase` / `insertIntoBuckets`; `bucket` parameter threaded through `insertFromBlockImpl*`; `map.impls[bucket]` addressing | Routing rows to the bucket whose lock is held |
| E7 | `BuildResult` struct replaces `bool & is_inserted, bool & all_values_unique`; `new_keys` out-param on `Inserter::insert*` | One insert call per bucket, so every build output must be reducible (OR/AND/+) across buckets |
| E8 | `use_offset`/`needs_offset` promoted from global `constexpr bool` to template parameter of `KeyGetterForType*` | On a partitioned map a cell's global offset must be synthesized against the other buckets' cells, so joins that never read offsets do not pay for them |
| E9 | `JoinUsedFlags::{setUsed,getUsed,setUsedOnce}` rewritten from early-`return` to `if constexpr (use_flags) {...}` + `[[maybe_unused]]` | Downstream of E8: code *after* an `if constexpr` is still instantiated, so `f.getOffset()` must not be named when the getter has no offset |
| E10 | `computeBucketPrefix` / `freezeMapsForProbing`; `getBucketBufferSizeInBytes` | Global cell offsets are numbered from a prefix sum over bucket capacities, computed once after the last insert so the build writes nothing shared |
| E11 | `prober()` handles in probe loops; `findKey(prober, ...)`; build prefetch via `cells.prefetchByHash` | Resolving the partition routing state once per block rather than per row |
| E12 | `bucket_bytes` running sum + `recomputeBucketBytes`; `getTotalByteCount` uses it instead of walking pools+maps | Byte total over N sub-tables and N arenas cannot be walked on the hot path |
| E13 | Counters (`rows_to_join`, `keys_to_join`, `all_values_unique`, `shrink_blocks`) become atomics; `blocks_mutex`/`totals_mutex`; `*Unlocked()` accessors; `columns.erase(it)` instead of `pop_back()` | Several threads mutate one shared map under bucket locks |
| E14 | `supportParallelJoin() == true` | Same as `ConcurrentHashJoin` (baseline-faithful for the parallel path); enabled by bucket locks |
| E15 | CHJ-only entry points removed: `joinScatteredBlock`, `getUsedFlags`/`setUsedFlags`, `hasNonJoinedRows`, `updateNonJoinedRowsStatus`, `allOffsetFlagsSet`; `addBlockToJoin(Block, Selector, bool)` made private | The fork has no `ConcurrentHashJoin` wrapper, so these have no caller. `hash` never calls them either, so no baseline behavior is lost |
| E16 | `create(type, buckets, reserve)`; post-build rerange uses `poolForBucket(0)`; converted range maps are bucket-partitioned; `reuseJoinedData` resizes `bucket_locks` + refreezes | Bucket count is part of construction and of any imported `RightTableData` |
| E17 | LowCardinality map gate loses `!use_two_level_maps`; `mergeJoinMethods` ranks lose the two-level family; `emplace` always passes an explicit hash; `routingHashForRow` replaces `getHash` | Bucket routing needs the hash that `emplace` will place by |

## B. FORK-MECHANICAL (2)

| ID | Divergence | Why unavoidable |
| --- | --- | --- |
| F1 | `APPLY_FOR_JOIN_VARIANTS` → `UNIFIED_APPLY_FOR_JOIN_VARIANTS` (and `_LIMITED`) | Verified **not** `#undef`'d in either header and used from `HashJoin.cpp`, `HashJoinMethodsImpl.h`, `ConcurrentHashJoin.cpp`, `StorageJoin.cpp` — a redefinition clash is real |
| F2 | `getName()` returns `"UnifiedHashJoin"` | Must be distinguishable in `EXPLAIN`/logs |

## C. MATERIAL — avoidable, must be aligned

| ID | Divergence | Baseline file/symbol | One-line align plan |
| --- | --- | --- | --- |
| **M1** | **Parallel non-joined block processing absent.** UHJ does not override `supportParallelNonJoinedBlocksProcessing()` (inherits `false` from `IJoin.h:158`) and does not override the 5-arg `getNonJoinedBlocks(..., bucket_idx, num_buckets)`, so it inherits the `IJoin.h:170` default that ignores the partitioning. `NotJoinedHash` lost the `isBucketInRange`/`isBlockInRange` filtering and the `bucket_idx != 0` nullmap guard. RIGHT/FULL joins therefore emit non-joined rows from a single stream. | `ConcurrentHashJoin::supportParallelNonJoinedBlocksProcessing` (`ConcurrentHashJoin.cpp:525`) and `ConcurrentHashJoin::getNonJoinedBlocks(...,stream_idx,num_streams)` (`:535`), whose **two-level branch** (`:555-560`) delegates to `HashJoin::getNonJoinedBlocks(..., stream_idx, num_streams)` (`HashJoin.cpp:1520`) | Port the baseline bucket-partitioned non-joined path into UHJ and override `supportParallelNonJoinedBlocksProcessing` with the same predicate as CHJ. **Not TwoLevel-required — TwoLevel is the enabler**: the baseline reaches this path *because* the map is two-level, and UHJ's map always is. |
| **M2** | `JoinUsedFlags::finalizePerRowFlags` drops the `source` parameter | `HashJoin/JoinUsedFlags.h:90` `finalizePerRowFlags(JoinUsedFlags & source, size_t num_blocks)` | Restore the baseline signature and the `HashJoin.cpp:2380` self-merge call shape |
| **M3** | `doDebugAsserts()` no longer runs on the public byte-count path: UHJ's `getTotalByteCount` takes `blocks_mutex` then calls `getTotalByteCountUnlocked`, which omits the assert | `HashJoin.cpp:533-538` calls `doDebugAsserts()` inside `getTotalByteCount()` | Call `doDebugAsserts()` in UHJ's public `getTotalByteCount()` under the lock it already holds |
| **M4** | `KEYGETTER_RANGE_IMPL` gratuitously renamed to `UNIFIED_KEYGETTER_RANGE_IMPL` | `HashJoin/KeyGetter.h:270-284` | Rename back — verified `#undef`'d at `KeyGetter.h:284` in both copies, so no clash exists (unlike F1) |
| **M5** | `clone()` propagates `stats_collecting_params` and `max_threads`; the baseline clone propagates neither its stats nor its map-shape knob | `HashJoin.h:129-134` | **Align direction ambiguous — see Open questions.** Dropping `stats_collecting_params` matches baseline; dropping `max_threads` would silently make every cloned UHJ single-bucket |
| **M6** | `optimize_read_in_order` disabled for UHJ: the gate is `typeid_cast<HashJoin *>(join.get())`, which a `Unified::HashJoin` fails | `ExpressionAnalyzer.cpp:2255` | Admit `Unified::HashJoin` to the same gate. UHJ's probe path is the baseline's, so left block order is preserved. Matches `hash`; diverges from `parallel_hash`, which also fails the cast |

## D. LEAD (no behavior difference today)

| ID | Item | Why not MATERIAL |
| --- | --- | --- |
| L1 | `GraceHashJoin::prepareRightBlock` (`GraceHashJoin.cpp:759`) hardcodes the baseline static `HashJoin::prepareRightBlock` even when `in_memory_kind == Unified` | Verified the two statics are **textually identical**, so no behavioral divergence today; latent coupling only |
| L2 | `join_algorithm='unified_hash'` missing from the docs of `max_bytes_before_external_join`, `join_overflow_mode`, `query_plan_join_shard_by_pk_ranges`, `parallel_non_joined_rows_processing`, and from the BuzzHouse fuzz value list | Documentation, not behavior |
| L3 | `JoinAlgorithm::AUTO` never selects `unified_hash`; `unified_hash` absent from the default `join_algorithm` list | Deliberate staging of an experimental algorithm, not an accidental divergence |
| L4 | `calculateHashTableCacheKeys` skips UHJ (gated on `allowParallelHashJoin()`, which requires `PARALLEL_HASH` in the setting list) | Cache is a `parallel_hash` slot-preallocation mechanism; UHJ still receives `StatsCollectingParams` from the planner |

## E. Null results (checked, no divergence)

- `ExpressionAnalyzer` legacy factory: `hash`, `parallel_hash` and `unified_hash` **all** get
  `any_take_last_row=false` and `StatsCollectingParams{}` (`ExpressionAnalyzer.cpp:1085-1092`). Parity.
- Runtime filters: `UNIFIED_HASH` is in `supportsRuntimeFilter` alongside both baselines
  (`joinRuntimeFilter.cpp:142-149`); no per-algorithm branching elsewhere.
- `isFilled()`, `alwaysReturnsEmptySet()`, `pipelineType()`, `hasPostBuildPhase()`: identical.
- `joinGet` / `joinGetCheckAndGetReturnType`, `joinBlock` dispatch, ASOF NULL-key build filter,
  `getMinBytesForPrefetchInJoin` threshold: textually identical.
- ProfileEvents: none in either `HashJoin.cpp`.
- 36 of 42 forked files: identical modulo the fork wrapper.
- `optimizeJoinByShards`, `optimizeJoin` outer-swap, `PlannerCorrelatedSubqueries`: UHJ explicitly
  listed alongside the baselines.

---

## Open questions for the user (Unit 2 is gated on these)

1. **M1 scope.** Porting the bucket-partitioned parallel non-joined path is the one substantial
   implementation item in this inventory. Confirm it is in scope rather than a risk-accepted LEAD.
2. **M5 align direction.** Baseline `clone()` propagates neither stats nor map-shape knob. Dropping
   `max_threads` to match would make cloned UHJ instances single-bucket. Options: (a) drop only
   `stats_collecting_params`; (b) drop both; (c) risk-accept as LEAD.
3. **FORK-MECHANICAL as a separate bucket.** These 2 rows are not TwoLevel-required, so folding them
   into EXCLUDED would be the reclassification the mission bans; they are also not removable without
   deleting the fork. Confirm they may stay outside the `AVOIDABLE_MATERIAL` count.

## Summary line

```
AVOIDABLE_MATERIAL=6        (M1..M6, none yet aligned — Unit 1 is inventory only)
EXCLUDED=17 groups / 44 diff regions
FORK_MECHANICAL=2
LEAD=4
UNSETTLED=0
```
