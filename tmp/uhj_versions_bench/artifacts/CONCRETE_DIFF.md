# The concrete difference between baseline and uhj on JOB q64

I previously wrote "same code, same work". That was wrong at the query level, and it is what
sent me chasing symptoms. Only the **leaf** function `RowRefList::insert` is shared and
identical. Everything that calls it is a different class, and the counters I had already
collected said so:

| per query, max_threads=16 | baseline | uhj | delta |
|---|---:|---:|---:|
| inst_retired | 22.44 G | 23.46 G | **+1.02 G instructions** |
| mem_access | 6.84 G | 7.25 G | **+406 M accesses** |

A billion extra instructions is not "the same work". Below is where they come from. There
are **two independent differences**, in two different regimes, both in source you control.

---

## Difference 1 — `max_threads >= 2`: uhj runs an extra full pass over every build row

Retired-instruction profile, attributed per symbol (`perf record -e inst_retired`):

| symbol | baseline | uhj | delta |
|---|---:|---:|---:|
| **`DB::Unified::scatterBlockBySlot`** | **0.00 G (absent)** | **1.58 G (6.73%)** | **+1.58 G** |
| `DB::calculateHashes<...>` | 0.36 G | 0.00 G | −0.36 G |
| `IColumnHelper<ColumnVector<int>>::scatter` | 0.97 G | 0.57 G | −0.40 G |
| `HashJoinMethods::joinRightColumns` | 0.31 G | 0.00 G | −0.31 G |
| `ConcurrentHashJoin::addBlockToJoin` + `dispatchBlock` | 0.19 G | 0.00 G | −0.19 G |
| `RowRefList::insert` | 3.30 G | 3.49 G | +0.19 G |

`scatterBlockBySlot` is the single largest line item and has **no counterpart in baseline**.
It sums to the measured +1.02 G once the baseline-only functions are netted off.

What it does (`src/Interpreters/UnifiedHashJoin/SlotScatter.cpp`, called from
`HashJoin.cpp:1011`), per build block, before any insert happens:

```cpp
for (size_t i = 0; i < rows; ++i)              // pass over EVERY row
{
    auto key_holder = key_getter.getKeyHolder(selector[i], scratch_pool);  // re-read the key
    hash_value = ...routingHashForRow(...);                                // recompute the hash
    const size_t bucket = Traits::getBucketFromHash(...);
    row_to_slot[i] = slotForBucket(bucket, num_slots);                     // write 4B/row
    ++counts[slot];
}
for (size_t i = 0; i < rows; ++i)
    indexes[row_to_slot[i]]->getData().push_back(selector[i]);             // write per-slot index vectors
...
    auto parts = column->scatter(num_slots, column_selector);              // MATERIALISE dense key columns
```

So per build row uhj additionally extracts the key, computes a routing hash, writes a slot
id, appends to a per-slot index vector, and physically copies the key values into per-slot
columns. That is **~46 extra instructions per row** (1.58 G / 34.5 M rows) and it is the
source of the extra 406 M memory accesses.

Baseline's `ConcurrentHashJoin` also dispatches blocks to buckets, but it costs ~1.04 G
against uhj's ~2.15 G for the equivalent stage.

**Guard:** the call is skipped when `slots == 1` (`HashJoin.cpp:1001`), which is why this
difference disappears at `max_threads=1` — and why there had to be a second cause.

---

## Difference 2 — `max_threads == 1`: the two arms build a different hash table

`slotCountForThreads(1) == 1`, so uhj takes its **serial** map. The two map aliases are
declared in `src/Interpreters/UnifiedHashJoin/HashJoin.h`:

```cpp
constexpr Int32 BITS_FOR_BUCKET_SERIAL   = 0;                          // line 48
constexpr Int32 BITS_FOR_BUCKET_TWO_LEVEL = DEFAULT_BITS_FOR_BUCKET;   // line 49  (= 8)

/// Serial maps use the flat-table grower; the two-level grower added two rehashes on full-size
/// maps (+35-44% `FillingRightJoinSide` in the measured 500k-key case).
using JoinHashMap = TwoLevelHashMap<Key, Mapped, Hash,
    HashTableGrowerWithPrecalculation<>, HashTableAllocator, HashMapTable,
    BITS_FOR_BUCKET_SERIAL>;                                           // line 84

/// Parallel maps use the two-level grower, which avoids oversized bucket growth.
using TwoLevelJoinHashMap = TwoLevelHashMap<Key, Mapped, Hash,
    TwoLevelHashTableGrower<>, HashTableAllocator, HashMapTable,
    BITS_FOR_BUCKET_TWO_LEVEL>;                                        // line 99
```

`BITS_FOR_BUCKET_SERIAL = 0` means **2⁰ = one bucket**: a single flat table holding all
42,296,370 rows. Baseline always uses 2⁸ = **256 sub-tables** of ~165k entries each. Confirmed
from the running binaries — the hot build-loop instantiation at `max_threads=1`:

```
baseline  TwoLevelHashMapTable<UInt32, ..., TwoLevelHashTableGrower<8ul>,          ...>
uhj       TwoLevelHashMapTable<UInt32, ..., HashTableGrowerWithPrecalculation<8ul>, ...>
```

Different grower *and* different bucket count. The measured consequence, per query at
`max_threads=1`:

| metric | baseline (256 sub-tables) | uhj (1 flat table) | delta |
|---|---:|---:|---:|
| inst_retired | 17.16 G | 19.30 G | **+12.5%** |
| ll_cache_miss_rd | 259.7 M | 298.4 M | **+14.9%** |
| dtlb_walk | 79.4 M | 86.0 M | **+8.3%** |
| stall_backend_mem | 8.87 G | 12.58 G | +41.8% |
| IPC | 1.213 | 1.017 | −16.1% |
| cycles | 14.15 G | 18.98 G | **+34.1%** |

Note this regime behaves differently from `max_threads=16`: here uhj really does take **more
misses** (+14.9%) and **more page walks** (+8.3%), which is what one flat multi-hundred-MB
table predicts against 256 small ones. The extra 12.5% instructions are consistent with
rehashing one giant table instead of 256 independent small ones.

The code comment says this choice was tuned on a **500k-key** case. At 42 M keys the
trade-off inverts: the rehash saving is swamped by the locality loss.

---

## Summary

| regime | concrete difference | where | measured cost |
|---|---|---|---|
| `max_threads >= 2` | uhj runs `scatterBlockBySlot`, an extra per-row pass (key re-read, routing hash, slot write, per-slot index vectors, dense key column materialisation) | `SlotScatter.cpp`; call site `HashJoin.cpp:1011` | +1.58 G instructions/query (6.73%), +406 M memory accesses, ~46 instructions/row |
| `max_threads == 1` | uhj builds **one flat 42 M-entry table** (`BITS_FOR_BUCKET_SERIAL = 0` + flat grower); baseline builds **256 sub-tables** | `HashJoin.h:48, 82-84` | +12.5% instructions, +14.9% LL misses, +8.3% dTLB walks, +34% cycles |

Both are single-constant / single-call-site changes, and both are testable directly:

1. Set `BITS_FOR_BUCKET_SERIAL = 8` (or make the serial map use `TwoLevelJoinHashMap` above
   some key-count threshold) and re-measure `max_threads=1`.
2. Make the `slots >= 2` path insert straight from the original columns with a per-slot
   selector — i.e. avoid materialising `dense_keys` — and re-measure `max_threads=16`.

The earlier prefetch observation (uhj capturing ~25% of the prefetch benefit) is a
*consequence* of these two, not a third cause: a flat 42 M-entry table and an extra pass that
evicts the build working set both reduce what a row-distance-calibrated prefetch can hide.

## Reproduce

```bash
ARM=uhj QIDX=64 SAMPLE=30 bash tmp/uhj_versions_bench/instr_profile.sh            # per-symbol instructions
ARM=uhj QIDX=64 SAMPLE=30 MT=1 TAG=_mt1 bash tmp/uhj_versions_bench/instr_profile.sh
ARM=uhj QIDX=64 SAMPLE=40 MT=1 TAG=_mt1 bash tmp/uhj_versions_bench/deep_metrics_norm.sh
```
