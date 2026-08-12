# JOB +11% regression — investigation

**The +11% is the same statistics artifact as TPC-H q8 and TPC-DS q54. At equal plans
`unified_hash` is 2.9% *faster* on JOB. A smaller, genuine regression survives on 7 of 113
queries, all with very large build sides.**

## Method

Full 113-query JOB suite, both arms, two variants, `TRIES=6` + `drop_caches` per query
(ClickBench contract), fresh server per pass so each starts with an empty statistics cache:

- `default` — as originally benchmarked (baseline gets the statistics-driven join reorder).
- `nostats` — `collect_hash_table_stats_during_joins=0` on **both** arms, which forces both
  onto the same plan and makes the comparison about the join engine.

Raw per-query timings: `followup/job/job_{baseline,uhj}_{default,nostats}.tsv`.

## Result

| variant | baseline geomean | uhj geomean | delta | outside noise |
|---|---|---|---|---|
| `default` (as benchmarked) | 76.3ms | 84.2ms | **+10.4%** | 36 reg / 8 imp |
| `nostats` (same plan) | 87.3ms | 84.8ms | **−2.9%** | 6 reg / 20 imp |

Note which arm moves: disabling statistics costs **baseline** 14.6% (76.3 → 87.3ms) and leaves
**uhj** unchanged (84.2 → 84.8ms). The original +11% was baseline gaining from a plan
optimisation uhj cannot receive, not uhj losing performance.

Per-query decomposition (ms, min-of-hot):

| q | base default | base nostats | uhj default | uhj nostats | baseline's stats gain | uhj at same plan |
|---|---|---|---|---|---|---|
| q2 | 17 | 130 | 113 | 116 | 86.9% | **−10.8%** |
| q4 | 17 | 125 | 109 | 112 | 86.4% | **−10.4%** |
| q23 | 94 | 378 | 493 | 475 | 75.1% | **+25.7%** |
| q64 | 115 | 395 | 503 | 499 | 70.9% | **+26.3%** |
| q68 | 54 | 160 | 157 | 158 | 66.3% | −1.3% |
| q88 | 63 | 166 | 163 | 161 | 62.0% | −3.0% |
| q106 | 105 | 218 | 190 | 189 | 51.8% | −13.3% |
| q61 | 68 | 135 | 79 | 96 | 49.6% | −28.9% |

q2 and q4 are the clearest: they looked like +565% and +541% regressions, and at equal plans
uhj is ~10% *faster* than baseline on both.

## The genuine residual

Six to seven queries are still slower under `unified_hash` with statistics off on both arms:

| q | baseline | uhj | delta |
|---|---|---|---|
| q64 | 395ms | 499ms | +26.3% |
| q23 | 378ms | 475ms | +25.7% |
| q56 | 228ms | 258ms | +13.2% |
| q58 | 255ms | 285ms | +11.8% |
| q57 | 256ms | 286ms | +11.7% |
| q29 | 139ms | 155ms | +11.5% |
| q59 | 240ms | 266ms | +10.8% |

They are one family — the `cast_info` / `name` / `title` / `movie_keyword` many-way joins —
and they are exactly the queries with the largest build sides. Both arms build an identical
number of rows (same plan), so the delta is pure engine cost:

| q | build rows (both arms) | uhj delta |
|---|---|---|
| q64 | 42,296,370 | +26% |
| q23 | 41,793,296 | +26% |
| q57 | 24,042,103 | +12% |
| q56 | 23,404,282 | +13% |
| q61 | 5,149,965 | −29% |
| q2 | 4,306,134 | −11% |
| q106 | 2,986,122 | −13% |

The sign flips with build-side size: at or below ~5M build rows uhj is equal or faster; at
23M+ it is 11–26% slower.

## Where the time goes

`perf record`, q64, statistics off on both arms, 40s of back-to-back runs
(`followup/job/perf/*.flat.txt`):

```
baseline (0.399s/query)                     uhj (0.499s/query)
  57.10%  DB::RowRefList::insert              65.85%  DB::RowRefList::insert
   5.83%  HashJoinMethods::insertFromBlock…    4.97%  Unified::HashJoinMethods::insertFromBlock…
   4.63%  BloomFilter::addHashPairs            3.69%  BloomFilter::addHashPairs
   4.30%  CityHash64WithSeed                   3.35%  CityHash64WithSeed
   1.90%  ColumnVector<int>::scatter           1.81%  Unified::scatterBlockBySlot
```

Both arms sit in the *same* function. This is not a different algorithm being chosen; it is
the same row-list append costing more.

`perf stat` over a 30s loop of the same query (ratios are independent of iteration count):

| metric | baseline | uhj |
|---|---|---|
| IPC | 1.355 | **1.142** (−16%) |
| backend-stalled cycles / cycles | 68.5% | **74.5%** |
| frontend-stalled cycles / cycles | 5.2% | 3.6% |
| cache-miss / cache-reference | 3.82% | 3.36% |

uhj retires fewer instructions per cycle and spends a larger share of cycles stalled in the
backend, with no worse miss *rate* — the signature of memory-latency-bound pointer chasing
rather than extra work. `RowRefList` builds a chain of row batches in an `Arena`; on an
ALL-strictness join over `cast_info` (tens of millions of rows, many rows per key) that path
dominates, and uhj's per-slot scatter/arena layout appears to make it more latency-bound.

## Conclusions

1. The reported JOB **+11% regression is not real**. It is the third instance of the
   statistics/plan artifact: `unified_hash` never receives the statistics-driven join
   reorder, so any comparison against a `parallel_hash`-containing default measures plans.
   With that controlled, uhj is **2.9% faster** on JOB.
2. A real regression remains, an order of magnitude smaller and narrowly scoped: **+11% to
   +26% on 7 of 113 queries**, all `RowRefList`-heavy builds of 23M+ rows. Offset by
   **20 queries that get genuinely faster** (up to −29%).
3. The remaining question for the residual is why the identical `RowRefList::insert` runs at
   1.14 IPC under uhj versus 1.36 under baseline. The evidence points at the build-side
   arena/slot layout rather than the probe path, which is where the next investigation should
   start — not at the join algorithm selection.

## Reproduce

```bash
for v in default nostats; do for a in baseline uhj; do
  ARM=$a VARIANT=$v bash tmp/uhj_versions_bench/job_suite.sh
done; done
python3 tmp/uhj_versions_bench/job_compare.py

ARM=uhj QIDX=64 bash tmp/uhj_versions_bench/job_perf.sh      # hotspots
ARM=uhj QIDX=64 bash tmp/uhj_versions_bench/job_perfstat.sh  # IPC / stalls
```
