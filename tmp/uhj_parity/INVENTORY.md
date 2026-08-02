# Divergence inventory — Unit 1 (tip `d0faf9f5158`, evidence refresh after REWORK)

Raw inventory: `diff_inventory_out.txt`. Profiles v2: `probe_summaries_v2.txt` (fixed lock filter).

## Comparison A — UHJ vs `hash` (non-parallel)

| ID | Diff | Two-level required? | Slowdown role | Verdict |
| --- | --- | --- | --- | --- |
| A1 | Unconditional `TwoLevelHashMap` vs flat `HashMapTable` | YES | Residual expected | **EXCLUDED** |
| A2 | Always pass `&bucket_locks` → `parallel_build=true` at t=1; per-row `std::lock(blocks_mutex, bucket)` for `RowRefList`; prefetch off | NO | Discriminating v2: `serial_uhj lock_in_insert=33` vs `serial_hash lock_in_insert=0`. (Old `mutex_related=110/170` **retracted** — cond_wait pollution.) Serial CPU still ~10×; A2 is a confirmed avoidable path tax, not proven sole cause of the full 10× | **CONFIRM MATERIAL (path); magnitude partial** |
| A3 | Per-row `getBufferSizeInBytes` on serial branch | NO | Masked until A2 fixed | **UNSETTLED** |
| A4 | `supportParallelJoin()==true` vs HashJoin `false` | NO | Pipeline LEAD | **LEAD** |

**Baseline:** HashJoin `HashJoinMethodsImpl.h` ~243–311 (no locks). UHJ `HashJoinMethodsImpl.h` ~430–449; `HashJoin.cpp` ~900–912.

## Comparison B — UHJ vs `parallel_hash`

| ID | Diff | Two-level required? | Slowdown role | Verdict |
| --- | --- | --- | --- | --- |
| B1 | Per-row `std::lock(blocks_mutex, bucket)` vs ConcurrentHashJoin scatter + per-shard try_lock per block | NO | v2: `parallel_uhj lock_in_insert=5055/5929`; `parallel_phash lock_in_insert=0`. Dominates ~217× wall | **CONFIRM MATERIAL** |
| B2 | `createInMemoryHashJoin` omits `max_threads` for Unified; spill ctor without concurrent_slots | NO | Code divergence CONFIRM. Raw wall (`probe_b2.txt`): UHJ spill=0 best 1390ms vs spill=1e9 best 1319ms (both ≫ parallel_hash ~60ms) — B1 masks any B2 wall effect here | **CONFIRM plumbing; wall impact UNSETTLED pending B1 fix** |
| B3 | `BUCKETS_PER_THREAD=1` / scan-from-0 | partial | Pending B1 | **UNSETTLED** |

## Ranked Unit 2 targets

1. **F1 = A2/B1** — align locking with baselines (nullptr when serial; batch lock like ConcurrentHashJoin when parallel).
2. **F2 = B2** — plumb `max_threads` (after F1 so wall can show it).

## U1-BASE medians (unchanged)

Serial: hash 272ms/160kcpu; uhj 1839ms/1.73Mcpu.
Parallel t16: phash 144ms/973kcpu; uhj 31251ms/46Mcpu.
