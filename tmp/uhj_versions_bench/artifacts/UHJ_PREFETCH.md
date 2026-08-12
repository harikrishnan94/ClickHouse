# Exactly why uhj is slower in `RowRefList::insert` — JOB q64

**uhj's build is 11% faster than baseline with software prefetch disabled, and 20% slower
with it enabled. The regression is not the insert path — it is that uhj converts only about
a quarter of the prefetch into actual memory-level parallelism.**

Everything below is JOB q64 with `collect_hash_table_stats_during_joins=0` on both arms, so
the plan and the work are identical.

## The decisive experiment

`enable_software_prefetch_in_join` toggled on both arms (3 hot runs each):

| | prefetch ON | prefetch OFF | what prefetch is worth |
|---|---:|---:|---:|
| baseline, max_threads=1 | 5.06s | 7.34s | **1.45×** |
| uhj, max_threads=1 | 6.09s | 6.55s | **1.08×** |
| baseline, max_threads=16 | 0.399s | 0.537s | **1.35×** |
| uhj, max_threads=16 | 0.497s | 0.551s | **1.11×** |

Read the columns, not the rows:

- **With prefetch off the arms invert.** uhj is *faster*: 6.55s vs 7.34s at one thread
  (−11%), and level at 16 threads (0.551 vs 0.537). uhj's insert path is not the problem;
  on raw no-prefetch throughput it is the better of the two.
- **With prefetch on, baseline pulls ahead** because it converts prefetch into a 1.35–1.45×
  speedup while uhj gets 1.08–1.11×.

## Confirmed by the counters

Adding a third configuration — baseline with prefetch disabled — places uhj on the scale
(per query, iteration-normalised):

| metric | baseline +PF | **uhj +PF** | baseline −PF |
|---|---:|---:|---:|
| cycles | 16.35 G | **20.47 G** | 22.54 G |
| IPC | 1.372 | **1.146** | 1.022 |
| stall_backend_mem | 9.48 G | **13.02 G** | 14.58 G |
| ll_cache_miss_rd | 215.2 M | **214.2 M** | 219.6 M |
| **memory-stall cycles per LL miss** | **44.0** | **60.8** | **66.4** |

The miss count is the same in all three. Only the cost per miss moves. On the
44.0 → 66.4 scale that prefetch spans, uhj sits at 60.8:

> **uhj captures ~25% of the prefetch benefit** (5.6 of 22.4 cycles/miss).
> The wall-clock numbers agree: (0.537 − 0.497) / (0.537 − 0.399) = **29%**.

This is why the earlier measurements looked paradoxical — identical misses, identical TLB
walks, identical branch misses, identical call count, but +37% memory-stall cycles. A
prefetched miss still counts as a miss; it just doesn't stall. uhj takes the same misses as
demand misses.

## What it is not (each refuted with evidence)

| hypothesis | test | result |
|---|---|---|
| uhj does more work | uprobe call count | **34,494,831 on both**, exactly equal |
| different code | disassembly diff | **instruction-identical**, 200 instructions |
| the parallel build scatter / slot count | max_threads sweep 1→16 | gap present at **1 slot** (+19%) |
| prefetch not compiled into uhj | disassemble the hot build loop | **one `prfm pldl1keep` in both** |
| different hash map or key method | demangle hot build symbols | both `TwoLevelHashMapTable<UInt32, …, TwoLevelHashTableGrower<8>>` with `HashMethodOneNumber` |
| smaller per-call batches defeating the look-ahead | uprobe on the build loop | 9,537 vs 11,568 entries per query — **3.6k vs 3.0k rows per call**, far too similar to matter |

## Where the stall lands

`RowRefList::insert` performs two dependent dereferences:

```
site A   0x0050  ldr x8, [x22]        ; hash cell word -> CELL node (Arena)
site B   0x0168  ldr x8, [x24]        ; cell node -> OVERFLOW node (keys with >= 8 rows)
```

Share of the function's last-level misses: baseline **27.9% A / 65.4% B**, uhj
**49.0% A / 46.2% B**. Site B's address comes from what site A loads, so an unhidden miss at
A serialises the pair. The prefetch in the build loop covers the *map cell*, not the Arena
node — so when it works it keeps site A resident and only site B misses; when it doesn't,
both hops miss back to back. That is precisely the difference the two arms show.

## Conclusion

The regression decomposes into two opposing effects:

1. uhj's build path is **~11% faster** than baseline's without prefetch.
2. uhj loses **~75% of the software-prefetch benefit**, worth 1.35–1.45× to baseline.

Net: +19% at one thread, +22–26% at sixteen. Fix the second and uhj should be ahead on this
query, not behind.

## Next step

The narrow remaining question is why the same `map.prefetch()` call, in the same loop shape,
over the same map type, is 4× less effective under uhj. The refutations above rule out
compilation, map type, slot count and batch size, which leaves the timing of the prefetch
relative to its use — i.e. `PrefetchingHelper`'s look-ahead calibration against uhj's
different per-row cost, or the prefetched address not being the one subsequently demanded
(the `dense_keys` path recomputes the key from materialised columns).

The measurement to run next is a look-ahead sweep: force a fixed prefetch distance on both
arms and see whether uhj's optimum is simply at a different distance. If it is, the fix is
calibration, not architecture.

Worth noting for whoever picks this up: neither engine prefetches the Arena node at site B,
which owns the majority of baseline's remaining misses. That is an optimisation opportunity
for both.

## Reproduce

```bash
ARM=uhj QIDX=64 bash tmp/uhj_versions_bench/prefetch_test.sh       # the decisive toggle
ARM=baseline QIDX=64 PF=0 TAG=_nopf bash tmp/uhj_versions_bench/deep_metrics_norm.sh
ARM=uhj QIDX=64 bash tmp/uhj_versions_bench/count_calls.sh         # call-count equality
python3 tmp/uhj_versions_bench/annotate_insert.py                  # site A / site B split
```
