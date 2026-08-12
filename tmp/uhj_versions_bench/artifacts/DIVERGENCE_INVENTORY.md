# Divergence inventory: `Interpreters/UnifiedHashJoin` vs `Interpreters/HashJoin` + `ConcurrentHashJoin`

Read-only audit. Repo `/mnt/ch/ClickHouse`, branch `cursor/uhj-versions-bench-4f2a`, merge-base `3218492309c`.
No source file under `src/` was modified; nothing was built or benchmarked.

**Scope reminder.** The two-phase / batched probe restructuring (`ProbeLookup.h`, `lookupBatch`,
`consumeProbeBatch`, `SequentialLookup`, `ProbeOutcome`, `PROBE_BATCH_ROWS`, removal of
`processMatch`/`EmitSink`/`MultiEmitSink`/`PreSelectSink`, the batched `joinRightColumns` rewrite) and
purely mechanical fork artifacts (`namespace Unified`, include-path rewrites, renamed-but-equivalent
identifiers, comment rewording, clang-tidy edits) are **excluded** and are not inventoried below.

**A note on what "baseline" means.** The in-repo `src/Interpreters/HashJoin/**` is *not* pristine — the
branch touched it. `git diff 3218492309c -- src/Interpreters/HashJoin/` is 17 lines across two files and
is entirely mechanical (`map.NUM_BUCKETS` → `map.numBuckets()`, `IJoin` → `IInMemoryHashJoin` base class
plus `override` markers). So diffing inside `/mnt/ch/ClickHouse` is safe. `ConcurrentHashJoin.cpp`,
`SpillingHashJoin.*`, `GraceHashJoin.*`, `ExpressionAnalyzer.cpp`, `TreeRewriter.cpp` and `Aggregator.cpp`
*were* changed non-trivially on the branch; those changes are inventoried (D14, D18, D19, D20, D22)
because they alter the baseline arm of any A/B.

**Impact classes are reasoned estimates, not measurements**, except where explicitly marked
"measured" (A and B only). No performance number appears below that was not supplied in the prompt.

---

## 1. Summary table

Ranked by expected performance impact. "Regime" is the condition under which the divergence is live.

| id | area | one-line description | baseline `file:line` | uhj `file:line` | impact | measured? |
|----|------|----------------------|----------------------|-----------------|--------|-----------|
| **A** | build path | `scatterBlockBySlot` is an extra per-row pass over every build block; no counterpart in serial `hash` (there *is* one in `parallel_hash`) | `ConcurrentHashJoin.cpp:721` (analogue) / none in `HashJoin/HashJoin.cpp` | `UnifiedHashJoin/SlotScatter.cpp:30`, called from `UnifiedHashJoin/HashJoin.cpp:1011` | **High** | **yes (A)** |
| **B** | hash table types | `BITS_FOR_BUCKET_SERIAL = 0` makes the `max_threads == 1` map a 1-bucket two-level map; `parallel_hash` at `max_threads=1` uses 256 sub-tables with `TwoLevelHashTableGrower` | `HashJoin/HashJoin.h:323`, `ConcurrentHashJoin.cpp:230` | `UnifiedHashJoin/HashJoin.h:48,83-84` | **High** | **yes (B)** |
| **D1** | concurrency / planner | `parallel_hash_join_threshold` + `rhs_size_estimation` gate is bypassed for `unified_hash`: UHJ always goes 256-bucket + N slots when `max_threads > 1` | `Planner/PlannerJoins.cpp:1244-1256` | `Planner/PlannerJoins.cpp:1259-1264` (`!unified` guards at 1211, 1244) | **High** | no |
| **D2** | memory accounting | `bucket_bytes` accumulates *insert deltas only*; the initial map allocation (256 buckets × 256 cells, or the whole fixed-map buffer) is never counted until `recomputeBucketBytes` at end of build | `HashJoin/HashJoin.cpp:533-557` (recomputed every call) | `UnifiedHashJoin/HashJoin.cpp:699-714`, `435-437` | **High** | no |
| **D3** | concurrency | UHJ serializes the parallel build on one global `blocks_mutex` (≥2 acquisitions per build block); `ConcurrentHashJoin` uses a per-slot mutex and lock-free counters | `ConcurrentHashJoin.cpp:340` (`hash_join->mutex`, per slot) | `UnifiedHashJoin/HashJoin.cpp:907, 1045, 1054, 1062, 1070, 1101` | **High** | no |
| **D4** | hash table types | all fixed/direct-addressed maps become `PartitionedFixedHashMap`, which permanently calls `disableMinMaxOptimization()`; iteration then scans the whole key range | `HashJoin/HashJoin.h:310-311, 332-339` | `UnifiedHashJoin/HashJoin.h:112, 435-442`; `Common/HashTable/PartitionedFixedHashMap.h:35-49`; `Common/HashTable/TwoLevelHashTable.h:204-208` | **Medium** | no |
| **D5** | semantics / non-joined rows | `hasNonJoinedRows` / `updateNonJoinedRowsStatus` / `allOffsetFlagsSet` removed; RIGHT/FULL always runs the full non-joined scan even when every right row matched | `HashJoin/HashJoin.cpp:1192-1239`, `HashJoin/JoinUsedFlags.h:260-266`, `ConcurrentHashJoin.cpp:559` | absent in `UnifiedHashJoin/HashJoin.cpp` (see `2466-2490`) and `UnifiedHashJoin/JoinUsedFlags.h` | **Medium** | no |
| **D6** | key getters | `use_offset` becomes a template parameter driven by `JoinFeatures::need_flags`; UHJ skips `offsetInternal` per probe row when flags are unused. Baseline hard-codes `use_offset = true` | `HashJoin/KeyGetter.h:19, 171` | `UnifiedHashJoin/KeyGetter.h:18, 164-171`; `UnifiedHashJoin/HashJoinMethods.h:135` | **Medium** (UHJ favoured) | no |
| **D7** | hash table types | offset computation switched from lazy `std::call_once` + re-hash to `offsetInternalUnsafe` / `offsetInternalAtBucket` with an explicit `computeBucketPrefix` barrier | `HashJoin/KeyGetter.h:171`, `HashJoin/HashJoin.cpp:1431,1450` | `UnifiedHashJoin/KeyGetter.h:167`, `UnifiedHashJoin/HashJoin.cpp:1489`, `2070-2082` | **Medium** (UHJ favoured) | no |
| **D8** | hash table types | UHJ adds `two_level_low_cardinality_key_{string,fixed_string}` and permits the dictionary-aware map in parallel builds; baseline forbids LowCardinality maps whenever `use_two_level_maps` | `HashJoin/HashJoin.cpp:202` (`&& !use_two_level_maps`) | `UnifiedHashJoin/HashJoin.cpp:346`, `UnifiedHashJoin/HashJoin.h:302-314, 407-408` | **Medium** | no |
| **D9** | build path / concurrency | `key8`/`key16` parallel build: UHJ shares one `PartitionedFixedHashMap` under per-bucket locks; `parallel_hash` keeps N separate `FixedHashMap`s and scatters probe blocks per slot | `HashJoin/HashJoin.cpp:418` (`default: return type`), `ConcurrentHashJoin.cpp:455-464` | `UnifiedHashJoin/HashJoin.cpp:418` + `UnifiedHashJoin/HashJoin.h:112` | **Medium** | no |
| **D10** | memory accounting | per-block byte accounting: UHJ runs `slot_bytes()` twice per slot per block (`O(buckets/slots)` switch-dispatched calls); `ConcurrentHashJoin` calls `getTotalByteCount()`/`getTotalRowCount()` once per block per slot | `ConcurrentHashJoin.cpp:750-772`, `HashJoin/HashJoin.cpp:490-504` | `UnifiedHashJoin/HashJoin.cpp:130-136, 140, 166`, `642-649` | **Medium** | no |
| **D11** | build path / sizing | statistics-driven reserve moved from map construction (`create(type, reserve)`) to a lazy per-slot `reserveSlot`, and is now clamped by `max_bytes_before_external_join` on every path | `HashJoin/HashJoin.h:341-357`, `HashJoin/HashJoin.cpp:471`, `ConcurrentHashJoin.cpp:115-171` | `UnifiedHashJoin/HashJoin.h:451-505`, `UnifiedHashJoin/HashJoin.cpp:142-148, 589-605` | **Medium** | no |
| **D12** | concurrency | `SpillingHashJoin` single-join mode now reports `supportParallelJoin() == true` for UHJ, so the pipeline feeds one UHJ instance from `max_threads` threads | `SpillingHashJoin.h:89` (was `concurrent_join != nullptr`) | `SpillingHashJoin.h:97`, `SpillingHashJoin.cpp:37-60` | **Medium** | no |
| **D13** | build path | `BlockKeyGetter` shares one key getter across slots when `reads_whole_block_at_construction`; without it the LowCardinality dictionary cache would be rebuilt per slot per block | no counterpart | `UnifiedHashJoin/HashJoinMethods.h:102-131`; `Common/ColumnsHashing/HashMethod.h:411` | **Medium** | no |
| **D14** | measurement bias | seven `ProfileEventTimeIncrement<Microseconds>` timers added to `ConcurrentHashJoin`'s build and probe hot paths on this branch — a `clock_gettime` pair per block on the *baseline* arm | `ConcurrentHashJoin.cpp:300, 308, 324, 420, 431, 452, 456, 797-798` | n/a (UHJ has none) | **Medium** | no |
| **D15** | prefetching | multi-clause (OR) probe: baseline prefetches only `mapv[0]`; UHJ builds one `ProbePrefetch` per clause | `HashJoin/HashJoinMethodsImpl.h:692-705`, `929-943` | `UnifiedHashJoin/HashJoinMethodsImpl.h:1151-1162`, `1495-1505` | Low | no |
| **D16** | key getters | LowCardinality `emplaceKey` builds the key holder twice and decodes the dictionary index twice per build row when the dictionary has no saved hash | `HashJoin/KeyGetter.h:136-143` | `UnifiedHashJoin/KeyGetter.h:119-137` | Low | no |
| **D17** | semantics | `joinGet` support removed from UHJ (`is_join_get`, `buildJoinGetOutput`, `nullable_column_ptrs`, `joinGetCheckAndGetReturnType`, `joinGet`) | `HashJoin/HashJoin.h:159-162`, `HashJoin/AddedColumns.h:117, 148, 198-205, 306-313`, `HashJoin/HashJoinResult.h:23` | removed in the corresponding UHJ files | Low | no |
| **D18** | spilling | `GraceHashJoin::makeInMemoryJoin` passes `max_threads = 1`, so `unified_hash` under grace is always the 1-bucket serial map (regime B) | `GraceHashJoin.cpp:735` (old) | `GraceHashJoin.cpp:739-750` | Low | no |
| **D19** | spilling | `GraceHashJoin::joinBlock` now takes `hash_join_mutex` per probe block whenever `getNumBuckets() <= 1`, including when `hash_join == nullptr` | `GraceHashJoin.cpp:454` (old: guarded by `hash_join &&`) | `GraceHashJoin.cpp:458-462` | Low | no |
| **D20** | spilling | `SpillingHashJoin::addBlockToJoin` now takes a `shared_lock` on the single-in-memory-join path too (previously only the concurrent path), and `switchToGraceHashJoin` takes the matching `unique_lock` | `SpillingHashJoin.cpp:147-160` (old) | `SpillingHashJoin.cpp:170-180`, `233-238` | Low | no |
| **D21** | non-joined rows | UHJ overrides `supportParallelNonJoinedBlocksProcessing()` unconditionally, so even a 1-bucket serial map gets `num_streams` non-joined streams, `num_streams − 1` of which emit nothing | not overridden by `HashJoin`; only `ConcurrentHashJoin.cpp:515` | `UnifiedHashJoin/HashJoin.cpp:1573-1578`, `1441-1445` | Low | no |
| **D22** | thread safety | UHJ guards `setTotals`/`getTotals` with a mutex and erases the stored block by iterator instead of `pop_back` | `HashJoin/HashJoin.cpp` (no lock), `IJoin::setTotals` | `UnifiedHashJoin/HashJoin.cpp:718-730`, `1081` | Low | no |
| **D23** | build path | UHJ owns `num_slots` arenas routed by `slotForBucket`; serial UHJ and serial baseline both have exactly one, `parallel_hash` has one per `HashJoin` instance | `HashJoin/HashJoin.h:440` | `UnifiedHashJoin/HashJoin.h:637-647` | Low | no |
| **D24** | analyzer plumbing | `TableJoin::isHashFamilyEnabled` replaces `isEnabledAlgorithm(HASH)` in three places; strictly widens (adds `UNIFIED_HASH`), no change for `hash`/`parallel_hash` | `TreeRewriter.cpp:657,781`, `JoinStepLogical.cpp:1160`, `convertJoinToIn.cpp:146` | same files | Unknown (believed nil) | no |

Counts: **High 5** (A, B, D1, D2, D3) · **Medium 10** (D4–D14) · **Low 9** (D15–D23) · **Unknown 1** (D24).
Already measured: 2 (A, B).

---

## 2. Detailed entries

### A — `scatterBlockBySlot`: an extra per-row pass over every build block *(already diagnosed)*

Already root-caused and measured on JOB q64: **+1.58 G retired instructions per query, 6.73 % of all
instructions, ≈46 instructions/row**. Recorded here for completeness and to state the comparison basis
precisely.

**Baseline.** Serial `hash` does no scatter at all — `HashJoin::addBlockToJoin` hands the block's
selector straight to `insertFromBlockImpl`. `parallel_hash` *does* have a per-row scatter
(`ConcurrentHashJoin::dispatchBlock`, `ConcurrentHashJoin.cpp:721`), which computes a hash per row via
a key getter, derives a shard, and then either builds per-shard index columns
(`scatterBlocksWithSelector`, line 701) or physically copies the block (`scatterBlocksByCopying`).

**UHJ.** `UnifiedHashJoin/HashJoin.cpp:1011` calls `scatterBlockBySlot` whenever `slots > 1`; the
`slots == 1` path at line 1002 just aliases the original selector.

```150:1013:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
                    if (slots == 1)
                    {
                        per_slot[0] = &stored_columns->selector;
                    }
                    else
                    {
                        ...
                        scattered = scatterBlockBySlot(
                            data->type, maps_kind, key_columns, key_sizes[onexpr_idx], stored_columns->selector, slots,
                            strictness_ == JoinStrictness::Asof);
```

The per-row loop is `SlotScatter.cpp:67-82`: construct a key holder, compute the routing hash, derive
the bucket, derive the slot, increment a count. A second pass at line 93 fills the per-slot index
columns. Optionally a third pass materialises `dense_keys` (line 109) — only when the summed key width
is ≤ `sizeof(IColumn::Selector::value_type)` and the incoming selector is the identity range.

**Why it matters.** Against serial `hash` it is pure added work. Against `parallel_hash` it is a
*different* scatter: UHJ moves only a selector (plus optionally the narrow key columns), where
`dispatchBlock` may copy all columns; but UHJ's routing hash is computed through the full key getter
per row, and it happens for every clause of every block.

**Regime.** Parallel build only (`slots > 1`, i.e. `max_threads > 1`). Proportional to build-side rows,
independent of build-side cardinality. Worst for wide/expensive keys (strings) where the key getter is
not cheap; least bad for narrow fixed keys that also qualify for the `dense_keys` fast path.

**How to test.** Compare `unified_hash` at `max_threads=1` vs `max_threads=2` on a build-heavy query
and look at the delta in `FillingRightJoinSide` time per build row; or flip `slotCountForThreads`
(`UnifiedHashJoin/HashJoin.cpp:70-75`) to return `1` unconditionally and re-measure with `max_threads`
unchanged, which isolates the scatter from the rest of the parallel path.

---

### B — `BITS_FOR_BUCKET_SERIAL = 0`: the serial map is one flat table *(already diagnosed)*

Already root-caused and measured at `max_threads=1`: **+12.5 % instructions, +14.9 % LL cache misses,
+8.3 % dTLB walks, +34 % cycles**.

**One clarification worth stating precisely**, because it changes what the number means. UHJ's serial map:

```77:84:src/Interpreters/UnifiedHashJoin/HashJoin.h
/// Serial maps use the flat-table grower; the two-level grower added two rehashes on full-size maps
/// (+35–44% `FillingRightJoinSide` in the measured 500k-key case).
template <typename Key, typename Mapped, typename Hash = DefaultHash<Key>>
using JoinHashMap
    = TwoLevelHashMap<Key, Mapped, Hash, HashTableGrowerWithPrecalculation<>, HashTableAllocator, HashMapTable, BITS_FOR_BUCKET_SERIAL>;
```

With `bits_for_bucket == 0` the `TwoLevelHashTable` template folds to a single inline `HashMapTable`
(`Common/HashTable/TwoLevelHashTable.h:19-21, 104, 135-136`), and the grower is
`HashTableGrowerWithPrecalculation<>` — which is exactly what baseline's plain `HashMap` uses by
default (`Common/HashTable/HashMap.h:384, 391`). **So UHJ's serial map is structurally identical to the
serial `hash` algorithm's map.** The measured +34 % cycles is therefore a comparison against
`parallel_hash` at `max_threads = 1`, where `ConcurrentHashJoin` builds each of its (one) `HashJoin`
instances with `use_two_level_maps = true` (`ConcurrentHashJoin.cpp:230`) and `slots =
toPowerOfTwo(min(1, 256)) = 1` (`ConcurrentHashJoin.cpp:196`), giving 256 sub-tables under
`TwoLevelHashTableGrower`.

The actionable statement is therefore: **at `max_threads == 1`, UHJ can only produce the flat layout,
while baseline can produce either, and the 256-bucket layout measured faster.** `useTwoLevelMaps` is a
hard function of `max_threads` with no setting behind it:

```53:56:src/Interpreters/UnifiedHashJoin/HashJoin.h
inline bool useTwoLevelMaps(size_t max_threads)
{
    return max_threads > 1;
}
```

**Regime.** `max_threads == 1` (or `GraceHashJoin`'s in-memory joins, which are hard-wired to
`max_threads = 1` — see D18). Grows with build-side cardinality: the flat table's resize copies the
whole buffer, the 256-bucket layout resizes one bucket at a time.

**How to test.** Change `BITS_FOR_BUCKET_SERIAL` to `8` (which makes serial UHJ use 256 buckets with
`HashTableGrowerWithPrecalculation`) or change `useTwoLevelMaps` to `return true`, and re-run at
`max_threads=1`. The two variants separate "bucket count" from "grower", which the current pair of
aliases conflates.

---

### D1 — `parallel_hash_join_threshold` is bypassed for `unified_hash` *(High, reasoned estimate)*

**Baseline.** Both the new-analyzer planner and the old analyzer gate `parallel_hash` on an estimated
right-hand-side size. If the estimate is available and below `parallel_hash_join_threshold`
(default **100 000**, `Core/Settings.cpp:8066`), the serial `HashJoin` is used instead:

```1244:1256:src/Planner/PlannerJoins.cpp
        if (table_join->allowParallelHashJoin() && !unified)
        {
            const bool use_parallel_hash = !table_join->isEnabledAlgorithm(JoinAlgorithm::HASH) || !params.rhs_size_estimation
                || (*params.rhs_size_estimation >= params.parallel_hash_join_threshold);
            if (use_parallel_hash)
            {
                return std::make_shared<ConcurrentHashJoin>(
                    table_join,
                    params.max_threads,
                    ...
```

**UHJ.** The `&& !unified` clause routes `unified_hash` past that gate entirely; it is then constructed
with the raw `params.max_threads`:

```1259:1264:src/Planner/PlannerJoins.cpp
        if (unified)
        {
            return std::make_shared<UnifiedHashJoin>(
                table_join, right_table_expression_header, params.join_any_take_last_row, /*reserve_num_=*/0, /*instance_id_=*/"",
                stats_collecting_params, params.max_threads);
        }
```

and `max_threads > 1` unconditionally selects 256-bucket maps plus `slotCountForThreads(max_threads)`
lock/arena slots (`UnifiedHashJoin/HashJoin.cpp:292, 418`). The same asymmetry exists on the spilling
path (`PlannerJoins.cpp:1211`) and in the old analyzer (`ExpressionAnalyzer.cpp:1059, 1088`).

**Why it matters.** For every query whose right side is estimated below 100 k rows, baseline
`join_algorithm='hash,parallel_hash'` runs the *serial* algorithm while `unified_hash` runs the *parallel*
one — different map layout, plus the scatter of finding A, plus slot locking. Any A/B that leaves
`max_threads` at its default is comparing two different algorithms on those queries, not two
implementations of one.

**Regime.** Small-to-medium build sides with a usable `rhs_size_estimation`; multiplied by query count
in a benchmark suite like JOB where most build sides are small.

**How to test.** Set `parallel_hash_join_threshold = 0` on the baseline arm to force `parallel_hash`
everywhere, or `parallel_hash_join_threshold = 18446744073709551615` to force serial `hash`; either
removes the confound. Alternatively run both arms at `max_threads = 1`, where UHJ is serial by
construction. `EXPLAIN` will name the chosen algorithm.

---

### D2 — `bucket_bytes` never sees the initial map allocation *(High, reasoned estimate)*

**Baseline.** `getTotalByteCount` recomputes the map contribution from scratch on every call:

```533:557:src/Interpreters/HashJoin/HashJoin.cpp
size_t HashJoin::getTotalByteCount() const
{
    ...
    res += data->allocated_size;
    res += data->nullmaps_allocated_size;
    res += data->pool.allocatedBytes();
    ...
            [&](auto, auto, auto & map_) { res += map_.getTotalByteCountImpl(data->type); });
```

**UHJ.** It reads a running atomic that is only ever fed by insert deltas:

```709:714:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
size_t HashJoin::getTotalByteCountUnlocked() const
{
    ...
    return data->allocated_size + data->nullmaps_allocated_size + data->bucket_bytes.load(std::memory_order_relaxed);
}
```

and `bucket_bytes` is deliberately not seeded:

```439:440:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
    /// `bucket_bytes` tracks insert deltas only; do not seed it from empty map buffers here or
    /// `SpillingHashJoin` sees ~1 MiB before the first row and spills immediately.
```

The deltas come from `insertIntoSlots`, which snapshots `slot_bytes(slot)` *before* the first insert
(`UnifiedHashJoin/HashJoin.cpp:140`) and again after (line 166). Both `HashTable` and `FixedHashTable`
allocate their buffer in their default constructor (`Common/HashTable/HashTable.h:798-803`,
`Common/HashTable/FixedHashTable.h:225`), i.e. inside `MapsTemplate::create` at
`UnifiedHashJoin/HashJoin.h:464-479`, which runs before any insert. So that first allocation is inside
`bytes_before` and cancels out of every delta. The correct total is only restored by
`recomputeBucketBytes` at `UnifiedHashJoin/HashJoin.cpp:2482`, after the build has finished.

Concretely, the un-accounted amount is:
* two-level maps: 256 buckets × the grower's initial 256 cells = 65 536 cells;
* fixed maps: the whole buffer — `2^size_bits` cells, up to `2^18` for `range18_key64`.

**Why it matters.** `getTotalByteCount` is what drives `SpillingHashJoin`'s
`max_bytes_before_external_join` check (`SpillingHashJoin.cpp:158`), `TableJoin::sizeLimits()` against
`max_bytes_in_join` (`UnifiedHashJoin/HashJoin.cpp:1099`), and `shrinkStoredBlocksToFit`'s
half-the-budget heuristic. Under-reporting delays the spill and delays block shrinking; the comment
above shows the trade-off was made knowingly, but it is a divergence from baseline in both directions
(baseline over-reports the empty buffer at time zero, UHJ under-reports it forever during the build).

A second, subtler consequence: `slot_bytes` sums `map.getBucketBufferSizeInBytes(type, bucket)` over the
slot's buckets, and for a fixed map every bucket index returns the *same* flat table
(`Common/HashTable/TwoLevelHashTable.h:215-216`), so `slot_bytes` returns `256/num_slots` copies of one
value. That is harmless today only because a `FixedHashTable`'s buffer never grows, so the term cancels
in the delta — a correctness-by-coincidence that will break if fixed maps ever gain a resize path.

**Regime.** Any query with `max_bytes_in_join` or `max_bytes_before_external_join` set. Worst for
`range*`/`key8`/`key16` maps and for many-clause joins, where the un-accounted buffer is largest
relative to the data.

**How to test.** Run with a small `max_bytes_before_external_join` (say 64 MiB) and compare the row
count at which each arm switches to `GraceHashJoin` — `system.events`'
`JoinSpillingHashJoinSwitchedToGraceJoin` and the `SpillingHashJoin` log line give the switch point.
Alternatively call `recomputeBucketBytes()` right after `dataMapInit` and see whether the spill point
moves.

---

### D3 — the parallel build serializes on one global `blocks_mutex` *(High, reasoned estimate)*

**Baseline.** `ConcurrentHashJoin` gives each shard its own `HashJoin` and its own mutex, and takes it
with `try_to_lock` so a thread that loses moves on to a different shard:

```340:344:src/Interpreters/ConcurrentHashJoin.cpp
                    std::unique_lock<std::mutex> lock(hash_join->mutex, std::try_to_lock);
                    if (!lock.owns_lock())
                        continue;
```

Global totals are maintained with relaxed atomics (`updateTotalRowsAndBytesUnlocked`,
`ConcurrentHashJoin.cpp:750-772`). There is no lock shared by all shards.

**UHJ.** Slot inserts use the same `try_to_lock` + yield pattern
(`UnifiedHashJoin/HashJoin.cpp:205-221`), but everything *around* the insert goes through a single
`blocks_mutex` owned by the join. Per build block, on every build thread:

| site | `file:line` | what it guards |
|---|---|---|
| stored-block registration | `UnifiedHashJoin/HashJoin.cpp:907` | `data->columns.push_back`, `stored_columns_index->add`, `allocated_size` |
| per-row flag init | `1045` | `used_flags->reinit` (first clause, `flag_per_row` only) |
| nullmap storage | `1054`, `1062` | `data->nullmaps.emplace_back` |
| empty-insert rollback | `1070` | `data->columns.erase` |
| `shrinkStoredBlocksToFit` | `1101` | taken *before* the `shrink_blocks` early-out |

The last one is the notable case: `addBlockToJoin` ends with an unconditional
`shrinkStoredBlocksToFit(total_bytes)`, and the function takes the lock before deciding it has nothing
to do:

```1097:1108:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
void HashJoin::shrinkStoredBlocksToFit(size_t & total_bytes_in_join, bool force_optimize)
{
    /// Rewrites every stored block in place, so it must not run while another build thread is
    /// appending one. The decision itself is cheap and the rewrite happens at most once per join.
    std::lock_guard lock(blocks_mutex);
    ...
        if (shrink_blocks)
            return; /// Already shrunk
```

Baseline's equivalent (`HashJoin/HashJoin.cpp:866-888`) takes no lock at all — it runs under
`ConcurrentHashJoin`'s per-shard mutex.

`HashJoin::getTotalByteCount` also takes `blocks_mutex` (`UnifiedHashJoin/HashJoin.cpp:701`), and
`SpillingHashJoin::addBlockToJoin` calls it once per block from the pipeline
(`SpillingHashJoin.cpp:158`), adding a third acquisition on the spilling path.

**Why it matters.** Two mandatory global critical sections per build block put a hard ceiling on build
scalability that `ConcurrentHashJoin` does not have. The sections are short, so this should not show up
at 2–4 threads, but it is the first thing that will bite at 16–64.

**Regime.** Parallel build, high `max_threads`, small blocks (more blocks ⇒ more acquisitions). Invisible
at `max_threads = 1`.

**How to test.** Sweep `max_threads` (1, 2, 4, 8, 16, 32) on a build-dominated query and plot
`FillingRightJoinSide` throughput per thread for both arms; the divergence should appear as a knee in
the UHJ curve. `perf lock` or a `std::mutex` contention counter around `blocks_mutex` would confirm
directly. Hoisting the `shrink_blocks` check above the lock is a one-line experiment that isolates the
`shrinkStoredBlocksToFit` acquisition.

---

### D4 — fixed maps become `PartitionedFixedHashMap` with min/max iteration disabled *(Medium)*

**Baseline.** Direct-addressed maps are plain flat tables:

```310:339:src/Interpreters/HashJoin/HashJoin.h
        std::shared_ptr<FixedHashMap<UInt8, Mapped>>                          key8;
        std::shared_ptr<FixedHashMap<UInt16, Mapped>>                         key16;
        ...
        std::shared_ptr<FixedHashMapWithSizeBits<UInt64, Mapped, 18>>         range18_key64;
```

`FixedHashTable` tracks `min`/`max` on every `emplace` and uses them to bound iteration
(`Common/HashTable/FixedHashTable.h:118-123, 185-186, 358-362, 394-397, 406`).

**UHJ.** All of them become `JoinFixedHashMap = PartitionedFixedHashMap`
(`UnifiedHashJoin/HashJoin.h:112, 435-442`), which is a `TwoLevelHashTable` whose `ImplTable` is a
`FixedHashMap` and whose `BucketHash` is `FixedRangeBucketHash`
(`Common/HashTable/PartitionedFixedHashMap.h:34-49`). That combination selects `FixedRangeStorage`,
which keeps one flat buffer and disables the min/max optimisation in its constructor:

```203:208:src/Common/HashTable/TwoLevelHashTable.h
        /// Direct-addressed storage keeps one flat buffer; buckets route locks instead of owning regions.
        FixedRangeStorage()
        {
            /// Do not cache `min`/`max`: parallel inserts race on them.
            flat.disableMinMaxOptimization();
        }
```

**Why it matters.** Every full traversal of a fixed map now walks the entire key range instead of
`[min, max]`. The traversals are: the RIGHT/FULL non-joined scan (`UnifiedHashJoin/HashJoin.cpp:1483`
onwards), `forEachMapped`, and `firstPopulatedCell`/`lastPopulatedCell`. For `range18_key64` that is
262 144 cells regardless of how many are populated; `tryConvertToFixedHashMap` only fires when the key
range is ≤ `MAX_RANGE`, but nothing forces the range to be *densely* populated, so a converted map with
1 000 live keys in a 262 144-cell range pays 262× the scan. `key8`/`key16` are small enough not to
matter.

Note the disabling is unconditional — it is required for the parallel build (D9) but is also paid at
`max_threads = 1` and after `tryConvertToFixedHashMap`, which runs on a single thread post-build.

Two related, smaller effects of the same substitution: `PartitionedFixedHashMap` never prefetches
(`TwoLevelHashTable::prefetchByHash` returns early for non-void `BucketHash`,
`Common/HashTable/TwoLevelHashTable.h:473-479`) and `isEmptyCell` always returns `false` (line 481-487).
Baseline's `FixedHashMap` has neither entry point either, so those two are parity, not divergence.

**Regime.** RIGHT/FULL joins, or any path that iterates the map, with a `key8`/`key16`/`range*` map —
i.e. a small-integer or dense-integer join key. Cost scales with the key *range*, not the row count.

**How to test.** `SELECT ... RIGHT JOIN` on a `UInt64` key with a dense range of ~250 000 but only a few
hundred distinct values, with `join_fixed_hash_table_conversion` enabled, and compare the non-joined
phase duration. Making `FixedRangeStorage`'s constructor conditional on `bucketCount() > 1` (it cannot
be, as written, since `bucketCount()` is always 256 here) or on a runtime "parallel build" flag would
isolate it.

---

### D5 — the non-joined-rows short-circuit is gone *(Medium)*

**Baseline.** `HashJoin` caches whether any right row went unmatched, and `ConcurrentHashJoin` uses the
answer to skip building the stream at all:

```1211:1239:src/Interpreters/HashJoin/HashJoin.cpp
void HashJoin::updateNonJoinedRowsStatus()
{
    ...
        else if (used_flags)
        {
            if (needUsedFlagsForPerRightTableRow(table_join))
                found_non_joined = true;
            else if (table_join->oneDisjunct())
                found_non_joined = !used_flags->allOffsetFlagsSet();
            else
                found_non_joined = true;
        }
```

```559:564:src/Interpreters/ConcurrentHashJoin.cpp
        if (hash_join->data->hasNonJoinedRows())
        {
            if (auto s = hash_join->data->getNonJoinedBlocks(left_sample_block, result_sample_block, max_block_size))
                streams.push_back(std::move(s));
        }
```

`allOffsetFlagsSet` is a single linear pass over the per-offset flag array
(`HashJoin/JoinUsedFlags.h:260-266`), run once at `onBuildPhaseFinish`.

**UHJ.** `hasNonJoinedRows`, `updateNonJoinedRowsStatus`, `has_non_joined_rows`,
`has_non_joined_rows_checked` and `allOffsetFlagsSet` are all absent. `onBuildPhaseFinish`
(`UnifiedHashJoin/HashJoin.cpp:2466-2490`) does not compute the status, and `getNonJoinedBlocks`
(line 1580) always constructs a `NotJoinedHash` when `JoinCommon::hasNonJoinedBlocks(*table_join)` is
true.

**Why it matters.** For a RIGHT or FULL join where every right row matched — a common shape when the
right side is a dimension table fully covered by the fact table — baseline pays one pass over the flag
array and then nothing, while UHJ pays a full hash-table traversal (which, per D4, may be a full key
range for fixed maps), pushing empty blocks through the non-joined pipeline stages.

Note the removal is not gratuitous: `allOffsetFlagsSet` reads `per_offset_flags` in bulk, and UHJ's
probe may run concurrently with... — actually the flags are final by the time the non-joined stream is
built on both sides, so the reason for the removal is not visible in the code. Marked as a real
divergence rather than a necessary consequence of the fork.

**Regime.** RIGHT / FULL joins where the match rate is high. No effect on INNER/LEFT.

**How to test.** `SELECT count() FROM small_dim RIGHT JOIN big_fact ON ...` where every `small_dim` key
appears in `big_fact`; compare the tail of the query (the delayed-blocks phase) between arms. The
`NotJoinedBlocks` transform's `elapsed_us` in `system.processors_profile_log` isolates it exactly.

---

### D6 — `use_offset` becomes a template parameter *(Medium, UHJ favoured)*

**Baseline.** A file-scope constant, so every key getter always computes the map offset:

```19:19:src/Interpreters/HashJoin/KeyGetter.h
constexpr bool use_offset = true;
```

```171:171:src/Interpreters/HashJoin/KeyGetter.h
        const size_t offset = found ? data.offsetInternal(it) : 0;
```

**UHJ.** It is threaded through as a template parameter, driven by whether the join actually needs used
flags:

```135:135:src/Interpreters/UnifiedHashJoin/HashJoinMethods.h
    static constexpr bool needs_offset = JoinFeatures<KIND, STRICTNESS, MapsTemplate>::need_flags;
```

```164:167:src/Interpreters/UnifiedHashJoin/KeyGetter.h
        size_t offset = 0;
        /// Offset only for used flags; needs current bucket-prefix state.
        if constexpr (use_offset)
            offset = found ? data.offsetInternalUnsafe(it) : 0;
```

This is enabled by the `JoinUsedFlags` restructure: `setUsed`/`getUsed`/`setUsedOnce` were rewritten
from early-`return` to `if constexpr (use_flags) { ... }` with `[[maybe_unused]]` parameters
(`UnifiedHashJoin/JoinUsedFlags.h:118-155, 185-198, 200-234`), specifically so the no-flags
instantiation stops requiring `FindResult::getOffset` — the comment at line 187 says so.

**Why it matters.** For INNER/LEFT ALL joins with a single disjunct and no `flag_per_row`, `need_flags`
is false, so UHJ eliminates one `offsetInternal` per matched probe row — a prefix-array load plus a
pointer subtraction for a flat map, and a re-hash of the key for a 256-bucket map. Baseline pays it and
then discards the result. This is a divergence in UHJ's favour and should be subtracted before
attributing any UHJ probe win to the batched probe loop.

**Regime.** Probe side, INNER/LEFT joins where `JoinFeatures::need_flags` is false. Scales with matched
probe rows.

**How to test.** Compare INNER JOIN (no flags) against RIGHT JOIN (flags) on the same data in both arms;
the UHJ advantage should exist in the first and vanish in the second. Alternatively hard-code
`needs_offset = true` in `HashJoinMethods.h:135` and re-measure.

---

### D7 — `offsetInternal` → `offsetInternalUnsafe` / `offsetInternalAtBucket` *(Medium, UHJ favoured)*

**Baseline.** Both the probe and the non-joined scan use the lazily-initialised entry point, which does a
`std::call_once` load on every call and re-derives the bucket by re-hashing the stored key:

```569:577:src/Common/HashTable/TwoLevelHashTable.h
    size_t offsetInternal(ConstLookupResult ptr) const
    {
        if constexpr (isFixedRangeStorage())
            return impls.offsetInternal(ptr);
        else if constexpr (bucketCount() == 1)
            return impls.offsetInternal(ptr, 0);
        else
            return impls.offsetInternal(ptr, getBucketFromHash(bucketRoutingHash(ptr->getKey(), ptr->getHash(*this))));
    }
```

Call sites: `HashJoin/KeyGetter.h:171` (probe), `HashJoin/HashJoin.cpp:1431, 1450` (non-joined scan).

**UHJ.** It establishes the prefix sums once, explicitly, at the build/probe boundary:

```2070:2082:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
void HashJoin::freezeMapsForProbing()
{
    ...
            [this](auto, auto, auto & map_) { map_.computeBucketPrefix(data->type); });
```

called from `onBuildPhaseFinish` (line 2468) and again from `runPostBuildPhase` (line 2513) after the
map may have been replaced. The probe then uses `offsetInternalUnsafe` (`UnifiedHashJoin/KeyGetter.h:167`)
and the non-joined scan uses `offsetInternalAtBucket`, which takes the bucket from the iterator instead
of re-hashing:

```1487:1489:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
                /// Prefer `offsetInternalAtBucket`: the iterator already knows its bucket, and the
                /// prefix sums were established by `freezeMapsForProbing()`.
                size_t offset = map.offsetInternalAtBucket(it.getPtr(), it.getBucket());
```

**Why it matters.** For 256-bucket maps this removes one `std::call_once` and one full key hash per
offset computation. The `call_once` is an acquire load on a fast path (uncontended after the first
call), and the re-hash is a real hash of the key. Combined with D6 it means UHJ's probe pays for offsets
strictly less often and strictly less per occurrence.

The cost side: `freezeMapsForProbing` must be called after every mutation of bucket capacity, or
`offsetInternalUnsafe` reads stale prefix sums. It is called twice, and the `chassert(computed)` in
`BucketPrefixSums::offsetUnsafe` (`Common/HashTable/TwoLevelHashTable.h:91`) only fires in debug builds.
That is a new correctness obligation with no analogue in baseline, though I found no path that violates
it.

**Regime.** Parallel builds (256-bucket maps) with used flags, i.e. RIGHT/FULL, plus the non-joined scan
on both serial and parallel.

**How to test.** RIGHT JOIN at `max_threads=8`, compare probe-phase instruction counts. Reverting
`UnifiedHashJoin/KeyGetter.h:167` to `offsetInternal` isolates the `call_once`+re-hash cost.

---

### D8 — LowCardinality maps are now allowed in parallel builds *(Medium)*

**Baseline.** The dictionary-aware map is explicitly restricted to non-two-level, i.e. serial `hash`:

```200:207:src/Interpreters/HashJoin/HashJoin.cpp
    /// Detect a single non-nullable LowCardinality key before the keys are materialized below, so it
    /// can use a dictionary-aware map. Restricted to a single disjunct and non-two-level maps for now.
    std::optional<Type> low_cardinality_method;
    if (table_join->oneDisjunct() && !use_two_level_maps && strictness != JoinStrictness::Asof)
```

`parallel_hash` on a LowCardinality key therefore materialises the dictionary and uses a plain
`two_level_key_string` map. There is no `two_level_low_cardinality_*` variant in
`APPLY_FOR_JOIN_VARIANTS` (`HashJoin/HashJoin.h:209-239`, 29 variants).

**UHJ.** The restriction is dropped and two variants are added (31 variants total):

```343:346:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
    /// Detect a single non-nullable LowCardinality key before the keys are materialized below, so it
    /// can use a dictionary-aware map. Restricted to a single disjunct for now.
    std::optional<Type> low_cardinality_method;
    if (table_join->oneDisjunct() && strictness != JoinStrictness::Asof)
```

```302:314:src/Interpreters/UnifiedHashJoin/HashJoin.h
    #define UNIFIED_APPLY_FOR_TWO_LEVEL_JOIN_VARIANTS(M) \
        ...
        M(two_level_low_cardinality_key_string)  \
        M(two_level_low_cardinality_key_fixed_string)
```

with the key getter supplied by a generic single-level → two-level forwarding macro
(`UnifiedHashJoin/KeyGetter.h:247-254`), and `toTwoLevelType` (`UnifiedHashJoin/HashJoin.h:390-402`)
performing the promotion.

**Why it matters.** For a LowCardinality join key at `max_threads > 1`, UHJ keeps the per-block
dictionary-index dedup on the probe side (`LowCardinalityKeyGetterForJoin::findKey`'s `visit_cache` /
`mapped_cache`, `UnifiedHashJoin/KeyGetter.h:139-172`) where baseline throws it away and materialises.
That should be a substantial UHJ win on LowCardinality(String) keys with a small dictionary and many
probe rows — and it means such queries are not comparing like with like unless the arms are pinned to
the same map type.

It also increases UHJ's instantiation count: the same 30 explicit-instantiation `.cpp` files exist on
both sides with byte-identical contents modulo the namespace, but each expands `APPLY_FOR_JOIN_VARIANTS`
over 31 variants instead of 29, and adds the fixed/single/two-level sub-macros. Compile-time and code
size only.

**Regime.** Single-disjunct, non-ASOF join on one non-nullable `LowCardinality(String)` or
`LowCardinality(FixedString)` key, `max_threads > 1`.

**How to test.** Build a join on `LowCardinality(String)` with a ~1 000-entry dictionary and 100 M probe
rows; compare `unified_hash` vs `parallel_hash` at `max_threads=8`. The `LOG_TRACE` "Using a
dictionary-aware hash map for the single LowCardinality join key"
(`UnifiedHashJoin/HashJoin.cpp:408`) confirms which map each arm chose.

---

### D9 — `key8`/`key16` parallel build: shared map vs sharded maps *(Medium)*

**Baseline.** `chooseMethod(..., use_two_level_maps = true)` has no two-level form for `key8`/`key16` and
falls through:

```417:419:src/Interpreters/HashJoin/HashJoin.cpp
        default:
            return type;
```

so each of `ConcurrentHashJoin`'s `slots` instances holds its *own* `FixedHashMap<UInt8>`.
`twoLevelMapIsUsed()` is then false, so the post-build merge is skipped and the probe side must scatter
every probe block across slots (`ConcurrentHashJoin.cpp:455-464`).

**UHJ.** `toTwoLevelType` likewise leaves `key8`/`key16` alone (`UnifiedHashJoin/HashJoin.h:390-402`,
which only covers `UNIFIED_APPLY_FOR_SINGLE_LEVEL_JOIN_VARIANTS`), but the map itself is a
`PartitionedFixedHashMap` — one flat table with 256 routing buckets. All slots therefore insert into the
*same* buffer, each holding a different `BucketLock`.

That is safe: routing is deterministic per key (`FixedRangeBucketHash` on `key >> block_shift`, so a
given key always lands in one bucket and hence one slot), buckets are routed a cache line at a time
(`Common/HashTable/PartitionedFixedHashMap.h:9-27`), the size counter is
`FixedHashTableStoredSize` with `m_size.fetch_add` (`Common/HashTable/FixedHashTable.h:53-63`), and
`min`/`max` — the one genuinely racy pair — is what `disableMinMaxOptimization` turns off (D4).

**Why it matters.** UHJ's probe reads one map; baseline's probe must scatter every probe block into
`slots` pieces and run the join machinery `slots` times. For a small-integer join key that is a large
structural advantage for UHJ on the probe side, and a small disadvantage on the build side (lock
routing plus the min/max loss). Any comparison on `UInt8`/`UInt16`/`Enum8`/`Enum16` keys is measuring
this, not the probe-loop rewrite.

**Regime.** `max_threads > 1` with a 1- or 2-byte numeric join key; also `range*` maps after
`tryConvertToFixedHashMap`, though those are produced post-build and only ever probed single-map on
both sides.

**How to test.** Join on an `Enum8` or `UInt8` key with `max_threads=8` and compare; `EXPLAIN` plus the
`LOG_TEST` "datatype:" line names the map on each side.

---

### D10 — per-block byte/row accounting has different asymptotics *(Medium)*

**Baseline.** `ConcurrentHashJoin` calls, once per dispatched block per slot:

```753:759:src/Interpreters/ConcurrentHashJoin.cpp
    const size_t rows_delta = hash_join->data->getTotalRowCount() - hash_join->local_total_rows;
    ...
    const size_t updated_local_bytes = hash_join->data->getTotalByteCount();
```

Both walk all 256 buckets of that slot's map. Serial `hash` sets `data->keys_to_join = total_rows` per
block (`HashJoin/HashJoin.cpp:861`) where `total_rows` came from `getTotalRowCount()` — O(1) for a flat
map.

**UHJ.** Row counting becomes O(1) via an explicitly-propagated `new_keys`:

```1041:1041:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
                    data->keys_to_join.fetch_add(result.new_keys, std::memory_order_relaxed);
```

fed by `Inserter::insertOne/insertAll/insertAsof`, each of which now takes `size_t & new_keys`
(`UnifiedHashJoin/HashJoinMethods.h:28-115`). Byte counting instead runs a bucket loop *twice per slot
per block*:

```130:136:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
    auto slot_bytes = [&](size_t slot)
    {
        size_t res = pools[slot]->allocatedBytes();
        for (size_t bucket = slot; bucket < num_buckets; bucket += num_slots)
            res += map.getBucketBufferSizeInBytes(type, bucket);
        return res;
    };
```

`getBucketBufferSizeInBytes` is a `switch` over 31 map variants
(`UnifiedHashJoin/HashJoin.h:551-560`), so each iteration is a jump-table dispatch, not an inlined load.

Totals per build block, per clause:

| map | baseline (per slot) | UHJ (per slot) |
|---|---|---|
| serial flat | 1 row + 1 byte call | 2 × 1 bucket |
| 256-bucket, `slots` slots | 256 row + 256 byte iterations | 2 × 256/`slots` iterations |
| fixed (`key8`/`range*`) | 1 (`FixedHashMap`) | 2 × 256/`slots` iterations, all returning the same value |

**Why it matters.** UHJ is cheaper for two-level maps and strictly more expensive for fixed maps, where
it does 512/`slots` dispatched calls per block to compute a delta that is always zero. The row-count
change is a clear UHJ win everywhere.

**Regime.** Build phase, proportional to block count. Worst for fixed maps with many small blocks.

**How to test.** `UInt8`-key build with `max_insert_block_size` set small (many blocks) — the per-block
constant should dominate. Short-circuiting `slot_bytes` for `isFixedRangeStorage()` maps is a
two-line experiment.

---

### D11 — reserve policy: construction-time → lazy per-slot, and now clamped *(Medium)*

**Baseline.** The map is sized at construction if the type supports it:

```341:357:src/Interpreters/HashJoin/HashJoin.h
        void create(Type which, size_t reserve)
        {
            ...
                    if constexpr (HasConstructorOfNumberOfElements<typename decltype(NAME)::element_type>::value) \
                        NAME = reserve ? std::make_shared<typename decltype(NAME)::element_type>(reserve)         \
```

Statistics-driven reserve for `parallel_hash` happens separately in `reserveSpaceInHashMaps`
(`ConcurrentHashJoin.cpp:115-171`), once per slot, guarded by `space_was_preallocated`, clamped by
`external_join_threshold / (8 * cell_size)`. Serial `hash` (including the instance
`GraceHashJoin` creates per bucket with a real `reserve_num`) is **not** clamped.

**UHJ.** `create` takes no reserve at all (`UnifiedHashJoin/HashJoin.h:464-479`); the reserve happens
lazily on first insert into each slot:

```142:148:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
        if (reserve_hint && !slot_space_reserved[slot])
        {
            const size_t reserved = map.reserveSlot(type, slot, num_slots, reserve_hint, max_reserve_bytes);
```

with the hint chosen by

```589:605:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
size_t HashJoin::sizeHintForMaps() const
{
    if (reserve_num)
        return reserve_num;

    /// Use statistics only for bucket-parallel builds; serial builds retain the normal grower's policy.
    ...
    if (num_slots <= 1)
        return 0;
    if (const auto hint = getSizeHint(stats_collecting_params))
        return hint->ht_size;
```

and `max_reserve_bytes = table_join->maxBytesBeforeExternalJoin()`
(`UnifiedHashJoin/HashJoin.cpp:435`), applied through `clampReserve`
(`UnifiedHashJoin/HashJoin.h:451-462`) using the same `reserve / (8 * cell_size)` formula as baseline.

**Divergences.**
1. The reserve is deferred to the first block, so the *first* block's inserts happen into an unreserved
   map. Immaterial for correctness, mildly different for the first-block cost.
2. An explicit `reserve_num` (the `GraceHashJoin` per-bucket reserve, `GraceHashJoin.cpp:739-750`) is now
   **clamped by `max_bytes_before_external_join`**, where baseline's serial path applied it verbatim.
   Under grace, `max_bytes_before_external_join` is always non-zero, so this always bites: the grace
   bucket's map may be reserved smaller than requested.
3. `StatsCollectingParams` is now propagated through `clone()` (`UnifiedHashJoin/HashJoin.h:218`) where
   baseline drops it (`HashJoin/HashJoin.h:133`). Cloned joins in UHJ will therefore both consume and
   update the statistics table; baseline's clones do neither.

**Regime.** (2) applies to every `GraceHashJoin` bucket under `unified_hash`; (3) applies wherever
`IJoin::clone` is used.

**How to test.** For (2), a grace-hash query with a large per-bucket `reserve_num` and a small
`max_bytes_before_external_join`; watch `HashJoinPreallocatedElementsInHashTables` in `system.events` —
it should be smaller on the UHJ arm. For (3), run the same query twice with
`collect_hash_table_stats_during_joins=1` and compare `HashJoinPreallocatedElementsInHashTables` on the
second run.

---

### D12 — `SpillingHashJoin` now reports parallel support for the single in-memory join *(Medium)*

**Baseline.**

```89:89:src/Interpreters/SpillingHashJoin.h
    bool supportParallelJoin() const override { return concurrent_join != nullptr; }
```

The single-`HashJoin` mode is single-threaded by construction, and `addBlockToJoin` takes the
`switch_mutex` only on the concurrent path.

**UHJ.**

```97:97:src/Interpreters/SpillingHashJoin.h
    bool supportParallelJoin() const override { return concurrent_join ? true : in_memory_hash_join->supportParallelJoin(); }
```

and `UnifiedHashJoin::supportParallelJoin()` is hard-coded `true`
(`UnifiedHashJoin/HashJoin.h:240`, vs baseline's `HashJoin` which does not override the `IJoin` default).
The pipeline therefore feeds one UHJ instance from `max_threads` threads, and UHJ serialises internally
via `blocks_mutex` + `BucketLock`s (D3).

Coupled change: `addBlockToJoin` now takes a `shared_lock` on *both* paths and `switchToGraceHashJoin`
takes the matching `unique_lock` with a re-check (`SpillingHashJoin.cpp:170-180, 233-238`) — that part is
listed separately as D20 because it changes the baseline arm too.

**Regime.** `max_bytes_before_external_join > 0` with `unified_hash`.

**How to test.** `EXPLAIN PIPELINE` on a query with `max_bytes_before_external_join` set — the number of
`FillingRightJoinSide` streams differs between the arms.

---

### D13 — `BlockKeyGetter`: one key getter shared across slots *(Medium)*

**Baseline.** No counterpart. In `ConcurrentHashJoin` each shard is a separate `HashJoin` that
constructs its own key getter for its own piece of the block.

**UHJ.**

```102:131:src/Interpreters/UnifiedHashJoin/HashJoinMethods.h
/// Share a getter only when construction reads the whole block; otherwise build one per bucket.
class BlockKeyGetter
{
public:
    template <typename KeyGetter, typename Build>
    KeyGetter & getOrBuild(Build && build)
    ...
template <typename KeyGetter>
constexpr bool shareKeyGetterAcrossBuckets()
{
    if constexpr (requires { KeyGetter::reads_whole_block_at_construction; })
        return KeyGetter::reads_whole_block_at_construction;
```

The only type declaring `reads_whole_block_at_construction = true` today is
`HashMethodSingleLowCardinalityColumn` (`Common/ColumnsHashing/HashMethod.h:411`), reached via
`LowCardinalityKeyGetterForJoin`. Without sharing, each slot would rebuild the dictionary-sized
`visit_cache` / `mapped_cache` / `offset_cache` arrays
(`UnifiedHashJoin/KeyGetter.h:71-75`) for every block.

**Why it matters.** This is the machinery that makes D8 (LowCardinality maps in parallel builds)
affordable. It is a UHJ-only capability with no baseline equivalent, so it is a divergence in
capability, not a regression. The `std::shared_ptr<void>` + `typeid` erasure at
`HashJoinMethods.h:117-119` costs one allocation and one `typeid` comparison per block per clause even
for the non-sharing case — small but non-zero, and paid on every map type.

**Regime.** LowCardinality single-key joins with `slots > 1`.

**How to test.** Compare block-level build throughput on a LowCardinality key at `max_threads=8` with
`shareKeyGetterAcrossBuckets` forced to `false`.

---

### D14 — profiling timers added to `ConcurrentHashJoin`'s hot paths *(Medium — measurement bias)*

Added on this branch, to the **baseline** arm only:

```61:69:src/Interpreters/ConcurrentHashJoin.cpp
extern const Event ConcurrentHashJoinBuildMicroseconds;
extern const Event ConcurrentHashJoinBuildDispatchMicroseconds;
extern const Event ConcurrentHashJoinBuildInsertMicroseconds;
extern const Event ConcurrentHashJoinBuildMergeMicroseconds;
extern const Event ConcurrentHashJoinProbeMicroseconds;
extern const Event ConcurrentHashJoinProbeDispatchMicroseconds;
extern const Event ConcurrentHashJoinProbeLookupMicroseconds;
```

Instrumentation points: `addBlockToJoin` entry (line 300), the dispatch scope (308), the insert loop
(324), `ConcurrentHashJoinResult::next()` (420) and its lookup scope (431), `joinBlock` (452) and its
dispatch scope (456), `onBuildPhaseFinish` (797-798).

`ProfileEventTimeIncrement<Microseconds>` is a `Stopwatch` — a `clock_gettime` at construction and
another at destruction, plus an atomic increment. `next()` runs **once per output block**, and
`joinBlock` once per probe block.

**Why it matters.** This is not a divergence in the join algorithm, but it is a systematic bias in any
`unified_hash` vs `parallel_hash` comparison run from this branch: the baseline arm pays two `vDSO`
clock reads plus an atomic per block on the probe path and three nested pairs on the build path, and the
UHJ arm pays none. On small blocks this is not negligible relative to the per-block overhead being
measured.

**Regime.** Every `parallel_hash` measurement taken from this branch. Also affects `SpillingHashJoin`'s
concurrent mode, which wraps a `ConcurrentHashJoin`.

**How to test.** Build the baseline arm from `3218492309c` instead of from this branch, or stub the
`ProfileEventTimeIncrement` constructor, and re-run the same comparison.

---

### D15 — multi-clause probe prefetch *(Low)*

**Baseline.** One prefetcher, on the first clause's map only:

```692:705:src/Interpreters/HashJoin/HashJoinMethodsImpl.h
    /// Software prefetch for multi-map variant. Only prefetch the first map.
    chassert(!mapv.empty());
    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;

    bool use_prefetch = false;
    if constexpr (can_prefetch)
        use_prefetch = shouldUseJoinPrefetch(added_columns.enable_prefetch, mapv[0]);
```

(the same shape at `HashJoin/HashJoinMethodsImpl.h:929-943` for the `flag_per_row` path).

**UHJ.** One `ProbePrefetch` per clause, each with its own `PrefetchingHelper` calibration:

```1151:1162:src/Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h
    /// One prefetcher per clause for the whole call (each clause's map; absolute-row calibration).
    constexpr bool can_prefetch = join_prefetch_supported<KeyGetter, Map>;
    std::vector<ProbePrefetch<Map, KeyGetter, Selector>> prefetchers;
    prefetchers.reserve(num_clauses);
    for (size_t k = 0; k < num_clauses; ++k)
    {
        bool use_prefetch_k = false;
        if constexpr (can_prefetch)
            use_prefetch_k = shouldUseJoinPrefetch(added_columns.enable_prefetch, mapv[k]);
```

(same at line 1495-1505). The build-side prefetch is unchanged between the two
(`HashJoin/HashJoinMethodsImpl.h:270-288` vs `UnifiedHashJoin/HashJoinMethodsImpl.h:341-359`), as are
`shouldUseJoinPrefetch` and `getMinBytesForPrefetchInJoin`.

**Why it matters.** Only for multi-disjunct (`OR`) joins, which are rare and already excluded from
`parallel_hash` by `allowParallelHashJoin` (`TableJoin.cpp:1304`). UHJ covers all clauses (better
latency hiding) at the cost of `num_clauses` prefetch instructions per row instead of one.

**Regime.** `JOIN ON a = b OR c = d` with a map larger than L2. Nil for single-clause joins.

**How to test.** A two-disjunct join with a large right side; toggle `enable_software_prefetch_in_join`.

---

### D16 — LowCardinality `emplaceKey` does the work twice *(Low)*

**Baseline.**

```136:143:src/Interpreters/HashJoin/KeyGetter.h
        if (saved_hash)
            data.emplace(key_holder, it, inserted, saved_hash[row]);
        else
            data.emplace(key_holder, it, inserted);
```

The `else` branch lets the table compute the hash from the key holder it already has.

**UHJ.**

```119:131:src/Interpreters/UnifiedHashJoin/KeyGetter.h
    ALWAYS_INLINE EmplaceResult emplaceKey(Data & data, size_t row_, Arena & pool)
    {
        ...
        const size_t row = getIndexAt(row_);

        auto key_holder = base.getKeyHolder(row, pool);

        typename Data::LookupResult it;
        bool inserted = false;
        data.emplace(key_holder, it, inserted, routingHashForRow(data, row_, pool));
```

and `routingHashForRow` (line 99-114) re-runs `getIndexAt(row_)` and, when `saved_hash == nullptr`,
constructs a **second** key holder before hashing it.

So per build row on a LowCardinality key: two `getIndexAt` dictionary-index decodes (a `switch` on index
width plus an `assert_cast` and an element load) always, and two `getKeyHolder` constructions when the
dictionary has no cached hash.

**Why it matters.** Small constant per build row, only on the LowCardinality path. Worth noting because
D8 makes that path much more reachable than it is in baseline.

**Regime.** Build side, single LowCardinality key, dictionary without a saved hash
(`ColumnLowCardinality::getDictionary().tryGetSavedHash()` returning null).

**How to test.** Hoist `const size_t hash = saved_hash ? saved_hash[row] : data.hash(keyHolderGetKey(key_holder));`
into `emplaceKey` and reuse `key_holder`; compare build throughput on a LowCardinality(String) key.

---

### D17 — `joinGet` removed from UHJ *(Low)*

**Baseline.** `joinGetCheckAndGetReturnType` / `joinGet` (`HashJoin/HashJoin.h:159-162`), the
`is_join_get` flag threaded through `AddedColumns` (`HashJoin/AddedColumns.h:148, 190, 309`) and
`HashJoinResult::Properties` (`HashJoin/HashJoinResult.h:23`), `buildJoinGetOutput`
(`HashJoin/AddedColumns.h:117`), and the `nullable_column_ptrs` array used to widen non-nullable stored
columns for `joinGetOrNull` (`HashJoin/AddedColumns.h:198-205, 306-313`).

**UHJ.** All removed. `HashJoinResult.h`'s only divergence from baseline is the deleted `bool
is_join_get;`, and `AddedColumns.{h,cpp}`'s only divergences are this removal.

**Why it matters.** Not a regression for the `unified_hash` *algorithm*: `joinGet` is served by
`StorageJoin`, which constructs `DB::HashJoin` unconditionally
(`Storages/StorageJoin.cpp:88, 170, 194, 305`) and has no `unified_hash` path. It is a scope limitation
to record, not a behaviour change. UHJ does still carry `isFilled()`/`from_storage_join`
(`UnifiedHashJoin/HashJoin.h:236`) and `UNIFIED_APPLY_FOR_JOIN_VARIANTS_LIMITED`
(`UnifiedHashJoin/HashJoin.h:337-351`), which are unreachable today.

Small side effect: UHJ's `AddedColumns` is a few bytes smaller and its constructor skips the
`nullable_column_ptrs` build loop, so per-probe-block construction is marginally cheaper.

**Regime.** n/a — `joinGet` cannot currently reach UHJ.

**How to test.** Not testable from SQL without wiring `StorageJoin` to `unified_hash`.

---

### D18 — `GraceHashJoin`'s in-memory joins are pinned to `max_threads = 1` *(Low)*

```737:750:src/Interpreters/GraceHashJoin.cpp
    return createInMemoryHashJoin(
        in_memory_kind,
        ...
        /// `addBlockToJoinImpl` inserts under `hash_join_mutex`, so this instance only ever sees one
        /// thread at a time however many feed the grace join.
        /*max_threads=*/1);
```

Baseline made a `HashJoin` directly with no thread parameter, so this is the same effective behaviour —
but for `unified_hash` it means every grace bucket uses the 1-bucket serial map, i.e. regime B, for a
workload that is by definition large. Recorded so it is not mistaken for a UHJ parallel-build
measurement.

**Regime.** `join_algorithm='unified_hash'` with `grace_hash` or with
`max_bytes_before_external_join` triggering a switch.

**How to test.** Force `GraceHashJoin` (`join_algorithm='grace_hash'` is not a `unified_hash` path;
use `unified_hash` + a small `max_bytes_before_external_join`) and check that the per-bucket map is
`key64`, not `two_level_key64`, in the `LOG_TEST` "datatype:" line.

---

### D19 — `GraceHashJoin::joinBlock` locks unconditionally *(Low)*

```455:462:src/Interpreters/GraceHashJoin.cpp
    /// Check if hash join post build optimizations could be performed.
    if (getNumBuckets() <= 1)
    {
        std::lock_guard lock(hash_join_mutex);
        if (hash_join)
            hash_join->runPostBuildPhase();
    }
```

Previously the `hash_join &&` test was outside the lock, so a null `hash_join` cost nothing. Now
`hash_join_mutex` is acquired on **every probe block** whenever the grace join degenerated to one
bucket. This closes a race (`hash_join` can be reset by a rehash concurrently with the read) at the
cost of a mutex acquisition per probe block on the `GraceHashJoin` fast path — which affects the
baseline arm as much as UHJ.

**Regime.** Grace-hash queries that end up with a single bucket. Uncontended acquisition, so cheap, but
it is on the per-block path.

**How to test.** `join_algorithm='grace_hash'` with a right side that fits in one bucket; compare probe
throughput against `3218492309c`.

---

### D20 — `SpillingHashJoin` now locks on the single-join path too *(Low)*

```167:180:src/Interpreters/SpillingHashJoin.cpp
    /// Shared lock: several threads may be adding blocks concurrently, and it keeps them out of a
    /// join that switchToGraceHashJoin is draining.
    std::shared_lock lock(switch_mutex);

    if (state.load(std::memory_order_acquire) != State::COLLECTING)
        return chosen_join->addBlockToJoin(block, check_limits);

    if (concurrent_join)
        return concurrent_join->addBlockToJoin(block, check_limits);

    return collectingJoin().addBlockToJoin(block, check_limits);
```

and the matching exclusive side:

```233:238:src/Interpreters/SpillingHashJoin.cpp
    /// Drain the in-memory join under the exclusive lock after shared `addBlockToJoin` holders leave.
    std::unique_lock lock(switch_mutex);

    if (state.load(std::memory_order_relaxed) != State::COLLECTING)
        return;
```

Baseline took the shared lock only when `concurrent_join` was set and did the single-join drain with no
lock at all. The change is required by D12 (the single in-memory join can now be multi-threaded), but it
also adds a `SharedMutex` acquire/release per build block to the `hash` +
`max_bytes_before_external_join` baseline path, where none existed.

**Regime.** `max_bytes_before_external_join > 0`, both arms.

**How to test.** Compare `FillingRightJoinSide` throughput for `join_algorithm='hash'` +
`max_bytes_before_external_join` against `3218492309c`.

---

### D21 — parallel non-joined streams even for a single-bucket map *(Low)*

**Baseline.** `HashJoin` does not override `supportParallelNonJoinedBlocksProcessing`; only
`ConcurrentHashJoin` does (`ConcurrentHashJoin.cpp:515-519`). Serial `hash` therefore gets one
non-joined stream.

**UHJ.**

```1573:1578:src/Interpreters/UnifiedHashJoin/HashJoin.cpp
bool HashJoin::supportParallelNonJoinedBlocksProcessing() const
{
    return table_join->allowParallelNonJoinedRowsProcessing()
        && JoinCommon::hasNonJoinedBlocks(*table_join)
        && anyClauseHasRightKeys(*table_join);
}
```

with no `max_threads` / bucket-count condition. `NotJoinedHash` partitions by
`bucket % num_streams == stream_idx` (`UnifiedHashJoin/HashJoin.cpp:1441-1445`). With the serial
1-bucket map only bucket 0 exists, so stream 0 does all the work and the other `num_streams − 1` streams
find nothing (`skipToNextOwnedBucket` walks to `end()` immediately, line 1467-1479).

**Why it matters.** Correct, but it materialises `num_streams` pipeline stages that produce nothing.
Setup cost plus a differently-shaped pipeline versus baseline. The `flag_per_row` path partitions by
`block_no % num_streams` (line 1402-1405) and does distribute properly.

**Regime.** RIGHT/FULL at `max_threads > 1` with `unified_hash` and a serial map — i.e. a
`GraceHashJoin` in-memory join (D18), or a `clone()`d/`reuseJoinedData` join.

**How to test.** `EXPLAIN PIPELINE` a RIGHT JOIN under `unified_hash` and count the non-joined stages.

---

### D22 — assorted thread-safety hardening in UHJ *(Low)*

Three small changes with no baseline counterpart, all consequences of UHJ accepting concurrent
`addBlockToJoin`:

* `setTotals` / `getTotals` are overridden and guarded by `totals_mutex`
  (`UnifiedHashJoin/HashJoin.cpp:718-730`, declared at `UnifiedHashJoin/HashJoin.h:246-247`). Baseline
  uses `IJoin`'s unguarded implementation. `ConcurrentHashJoin` guards `setTotals` but not `getTotals`
  (`ConcurrentHashJoin.cpp:474-486`).
* The failed-insert rollback uses `data->columns.erase(stored_columns_it)`
  (`UnifiedHashJoin/HashJoin.cpp:1081`) instead of baseline's `pop_back`, because another thread may
  have appended in the meantime. `data->columns` is a `std::list`, so the erase is O(1).
* `RightTableData`'s counters (`allocated_size`, `nullmaps_allocated_size`, `rows_to_join`,
  `keys_to_join`, `bucket_bytes`) and the join's `shrink_blocks` /
  `memory_usage_before_adding_blocks` / `all_values_unique` became `std::atomic`
  (`UnifiedHashJoin/HashJoin.h:649-657`) where baseline has plain scalars
  (`HashJoin/HashJoin.h:442-447`). `memory_usage_before_adding_blocks` is now sampled with a
  `compare_exchange_strong` so the *first* build thread wins
  (`UnifiedHashJoin/HashJoin.cpp:894-899`), where baseline samples on the single thread's first block.

Impact: atomic RMWs on a shared cache line instead of plain increments in the build loop. Mitigated by
`memory_order_relaxed` throughout, but `rows_to_join.fetch_add` and `keys_to_join.fetch_add` are once
per block, not once per row, so the contention is bounded.

---

### D23 — arena ownership *(Low)*

**Baseline.** One `Arena pool` per `HashJoin` (`HashJoin/HashJoin.h:440`), i.e. one for serial `hash`
and `slots` for `parallel_hash`.

**UHJ.** `num_slots` arenas, routed by slot (`UnifiedHashJoin/HashJoin.h:637-647`,
`slotForBucket` at line 56-59), created in `RightTableData`'s setup at
`UnifiedHashJoin/HashJoin.cpp:605-610`. `poolsAllocatedBytes()` sums them.

Counts match on both regimes (1 for serial, `slots` for parallel), and the default `Arena` chunk sizing
is unchanged, so this is parity rather than divergence. Recorded because the checklist asks about arena
count and ownership, and because `poolForBucket` routing means a string key's arena residence now
depends on its bucket rather than on which shard's block it arrived in — a difference in *locality*,
not in *quantity*, that I could not evaluate by reading.

---

### D24 — `isHashFamilyEnabled` widening *(Unknown, believed nil)*

`TableJoin::isHashFamilyEnabled` (`TableJoin.h:294-303`) returns `HASH || UNIFIED_HASH` and replaces
`isEnabledAlgorithm(JoinAlgorithm::HASH)` at three call sites:

* `TreeRewriter.cpp:657` (`tryJoinOnConst`) and `781` (the multi-OR check, whose message also changed);
* `JoinStepLogical.cpp:1160` (`can_convert_to_cross`);
* `convertJoinToIn.cpp:146` (was `HASH || PARALLEL_HASH`, now `(HASH || UNIFIED_HASH) || PARALLEL_HASH`).

Each is a strict widening — the predicate returns `true` in every case it returned `true` before — so I
believe there is no behaviour change for `hash` or `parallel_hash`. Marked Unknown only because
"strictly widening predicate ⇒ no behaviour change" assumes no call site depends on the predicate being
*false*, which I verified by reading but not by test.

**How to settle.** Run the `0_stateless` join tests with `join_algorithm` unset and with
`join_algorithm='parallel_hash'` against `3218492309c` and against this branch, and diff the results.

---

## 3. Checked and clean

File pairs diffed with the normalisation script at
`tmp/uhj_versions_bench/artifacts/nd2/gen.sh` (strips `Unified::`, the `namespace Unified { }` wrapper,
`UnifiedHashJoin/` → `HashJoin/` include paths, the `UNIFIED_APPLY_FOR` prefix, and trailing
blank-line/brace noise), then read by hand. "Clean" means the only remaining differences fall in the
excluded categories.

| pair | diff lines (normalised) | verdict |
|---|---|---|
| `HashJoin/JoinFeatures.h` ↔ `UnifiedHashJoin/JoinFeatures.h` | 2 | **clean** — a `// NOLINT(readability-identifier-naming)` comment |
| `HashJoin/KnownRowsHolder.h` ↔ `UnifiedHashJoin/KnownRowsHolder.h` | 4 | **clean** — `addFoundRowAll`'s template parameter renamed `Map` → `Mapped` and takes the mapped type directly; equivalent |
| `Interpreters/joinDispatch.h` ↔ `UnifiedHashJoin/joinDispatch.h` | 7 | **clean** — three deleted comments and two redundant `typename` keywords removed |
| `HashJoin/HashJoinResult.h` ↔ `UnifiedHashJoin/HashJoinResult.h` | 1 | only the `is_join_get` field (→ D17) |
| `HashJoin/HashJoinResult.cpp` ↔ `UnifiedHashJoin/HashJoinResult.cpp` | 26 | only the `is_join_get` branches and brace/comment style (→ D17) |
| `HashJoin/AddedColumns.h` ↔ `UnifiedHashJoin/AddedColumns.h` | 44 | only `joinGet` removal (→ D17) |
| `HashJoin/AddedColumns.cpp` ↔ `UnifiedHashJoin/AddedColumns.cpp` | 52 | only `joinGet` removal (→ D17) |
| the 30 explicit-instantiation units (`{Inner,Left,Right,Full}HashJoin{All,Any,Anti,Semi,Asof,RightAny,…}.cpp`) | 0 after namespace normalisation | **clean** — verified byte-identical apart from `namespace Unified { }`; the *set* of files is identical on both sides (same 30 names). The set of *instantiations* differs only because `APPLY_FOR_JOIN_VARIANTS` gained two variants (→ D8) |
| `HashJoin/ScatteredBlock.h` | n/a — shared, UHJ includes the baseline header | **clean** — `git diff 3218492309c` is empty |
| `Interpreters/RowRefs.h` | +12 vs merge-base | **clean for this inventory** — adds `RowRef::fromWord` (line 74-79), used only by `UnifiedHashJoin/ProbeLookup.h:58-60` and therefore part of the excluded probe machinery. `Columns/IColumn.cpp:628` and `RowRefs.h:397,405` use the pre-existing `RowRefList::fromWord` |
| `Common/HashTable/HashSet.h`, `HashTable.h`, `StringHashTable.h`, `TwoLevelStringHashTable.h` | 4 / 4 / 2 / 9 vs merge-base | **clean** — `NUM_BUCKETS` → `numBuckets()` and `size_t` → `Int32` on the `bits_for_bucket` template parameter; no semantic change |
| `Interpreters/Aggregator.cpp` | 8 vs merge-base | **clean** — four `Method::Data::NUM_BUCKETS` → `numBuckets()` |
| `HashJoin/HashJoin.h`, `HashJoin/HashJoin.cpp` | 12 / 5 vs merge-base | **clean** — `IJoin` → `IInMemoryHashJoin` base plus `override` markers, and `map.NUM_BUCKETS` → `map.numBuckets()` in one `requires` clause |
| `HashJoin/KeyGetter.h`, `JoinUsedFlags.h`, `HashJoinMethods.h`, `HashJoinMethodsImpl.h`, `AddedColumns.*`, `HashJoinResult.*`, `JoinFeatures.h`, `KnownRowsHolder.h`, `ScatteredBlock.h`, `Interpreters/joinDispatch.h`, `Storages/StorageJoin.cpp` | 0 vs merge-base | **unchanged on the branch** — confirms the baseline arm inside this repo is the merge-base behaviour |
| `Common/ColumnsHashing.h`, `Common/ColumnsHashingImpl.h`, `Common/ColumnsHashing/HashMethod.h` | 0 vs merge-base | **unchanged** — the `use_offset` template parameter and `reads_whole_block_at_construction` already existed |
| `Common/HashTable/FixedHashTable.h` | 0 vs merge-base | **unchanged** — `disable_min_max_optimization` / `disableMinMaxOptimization` are pre-existing; UHJ only *calls* them (→ D4) |
| `Common/HashTable/TwoLevelHashMap.h`, `HashTableTraits.h` | 26 / 25 vs merge-base | **clean** — plumbing the `bits_for_bucket` / `BucketHash` template parameters through; `forEachMapped` moved to the base class and now asks the storage instead of looping buckets, which is behaviour-identical for `FixedStorage` |
| `Common/HashTable/TwoLevelHashTable.h` | 453 vs merge-base | **behaviour-preserving for baseline** — the storage split (`FixedStorage` / `FixedRangeStorage`) reproduces the merge-base semantics exactly when `isFixedRangeStorage()` is false, which is every instantiation the baseline uses. The new `FixedRangeStorage` path is reached only by `PartitionedFixedHashMap`, i.e. only by UHJ (→ D4, D7, D9) |
| `Common/HashTable/BucketPartitionedTable.h` | +60, new file | UHJ-only concept definitions used by the `static_assert`s at `UnifiedHashJoin/HashJoin.h:114-119` — no runtime effect |
| `Planner/PlannerCorrelatedSubqueries.cpp`, `PlannerJoinsLogical.cpp` | 7 / 2 vs merge-base | **clean** — `UNIFIED_HASH` added to algorithm lists; no change for existing algorithms |
| `Processors/QueryPlan/Optimizations/optimizeJoin.cpp`, `optimizeJoinByShards.cpp`, `joinRuntimeFilter.cpp` | 3 / 4 / 1 vs merge-base | **clean** — `typeid_cast<UnifiedHashJoin *>` added alongside the existing `HashJoin` checks so UHJ gets the same optimisations; existing behaviour unchanged |
| `Core/Joins.{h,cpp}`, `Core/SettingsEnums.cpp`, `Core/Settings.cpp` | 1 / 1 / 3 / 4 | **clean** — the `unified_hash` enum value, its name, and its documentation |
| runtime filters: `publishSharedRuntimeFilters`, `buildSharedFilterProbeFn`, `SharedFixedHashTableRuntimeFilter` construction | `HashJoin/HashJoin.cpp:2127-2340` ↔ `UnifiedHashJoin/HashJoin.cpp:2216-2429` | **clean** — line-for-line equivalent, only the ~89-line offset. `hasPostBuildPhase` (`2396`/`2492`) and `runPostBuildPhase` (`2408`/`2506`) select the same conditions; UHJ additionally calls `recomputeBucketBytes` and `freezeMapsForProbing` afterwards (→ D2, D7) |
| `tryRerangeRightTableData`, `tryConvertToFixedHashMap`, `canConvertToFixedHashMap`, `rightTableCanBeReranged` | both sides | **clean** — same predicates, same `MAX_RANGE` walk; the only difference is the destination map type (→ D4) |
| ASOF: `Inserter::insertAsof`, `AsofRowRefs`, `SortedLookupVectorBase` | `HashJoin/HashJoinMethods.h:52-88` ↔ `UnifiedHashJoin/HashJoinMethods.h:76-113` | **clean** — the only change is the added `key_row`/`row_no` split and `size_t & new_keys` (→ D10) |
| `reinitUsedFlags`, `finalizePerRowFlags`, per-row flag sizing | `HashJoin/HashJoin.cpp:1995-2014` ↔ `UnifiedHashJoin/HashJoin.cpp:2084-2103` | **clean** — identical; both size to `getBufferSizeInCells(type) + 1`. UHJ calls `finalizePerRowFlags` unconditionally where baseline skips it for two-level maps and lets `ConcurrentHashJoin` do it during the merge — same net effect since UHJ owns its buckets |

---

## 4. Open questions / could not determine

1. **Why was `hasNonJoinedRows` / `allOffsetFlagsSet` dropped (D5)?** Nothing in the UHJ code explains it,
   and the flags are final before the non-joined stream is built on both sides, so the concurrency
   argument that motivates most of the other removals does not obviously apply. *Evidence needed:* the
   commit that removed it, or a test showing `allOffsetFlagsSet` racing with a concurrent probe.

2. **Is the `blocks_mutex` contention in D3 material at realistic thread counts?** I can bound the number
   of acquisitions per block (≥2, ≥3 under `SpillingHashJoin`) but not the hold time, which depends on
   `StoredColumnsIndex::add` and `std::list::push_back`. *Evidence needed:* a `max_threads` sweep with
   lock contention counters, or `perf lock contention` on a build-dominated query.

3. **Does the D2 under-reporting actually move a spill decision in practice?** The magnitude
   (65 536 cells for two-level, up to 262 144 for `range18_*`) is small next to a multi-gigabyte
   `max_bytes_before_external_join`, but the check is `getTotalByteCount() * 2 >= threshold`, so the
   error is doubled and the relevant comparison is against the *threshold*, not the data. *Evidence
   needed:* a spill-point comparison at a small `max_bytes_before_external_join` (D2's "how to test").

4. **Is `poolForBucket` arena routing (D23) better or worse for locality than baseline's per-shard
   arena?** UHJ interleaves keys from all incoming blocks into the arena belonging to their bucket's
   slot; baseline appends each shard's keys to that shard's arena in block order. Which produces better
   probe-time locality for string keys is not decidable by reading. *Evidence needed:* a string-key
   build with `perf stat -e dTLB-load-misses,LLC-load-misses` on the probe phase.

5. **What is the compile-time / icache cost of the two extra map variants (D8)?** 31 vs 29 variants
   across 30 instantiation units multiplies through the whole `HashJoinMethods` template. *Evidence
   needed:* `size(1)` of the two sets of object files, and `perf stat -e icache_64b.iftag_miss` on a
   probe-heavy query.

6. **Does `freezeMapsForProbing`'s prefix-sum barrier (D7) hold on every path?** `offsetInternalUnsafe`
   has only a debug `chassert(computed)`. `freezeMapsForProbing` is called at
   `onBuildPhaseFinish` (2468), at the end of `runPostBuildPhase` (2513) and at `reuseJoinedData`
   (1651). I did not find a path that mutates bucket capacity after the last of these, but I could not
   prove one does not exist — `StorageJoin`'s `OPTIMIZE`/insert path in particular reaches
   `shrinkStoredBlocksToFit` after the build. *Evidence needed:* a debug-build run of the `StorageJoin`
   stateless tests under `unified_hash`, which would trip the `chassert`. (Not currently reachable —
   `StorageJoin` hard-codes `DB::HashJoin`; see D17.)

7. **`ConcurrentHashJoin`'s `reserveSpaceInHashMaps` recomputes `actual_reserve_size` inside the
   per-map lambda but reads it outside for the `ProfileEvents::increment`
   (`ConcurrentHashJoin.cpp:137, 159, 169`).** UHJ returns the value from `reserveSlot` instead
   (`UnifiedHashJoin/HashJoin.cpp:144-146`). Whether the baseline pattern can report a stale zero when
   the map type has no `cell_type` is a pre-existing question, not a fork divergence, but it makes
   `HashJoinPreallocatedElementsInHashTables` not directly comparable between the arms. *Evidence
   needed:* compare the event value against the actual reserved capacity on both arms.
