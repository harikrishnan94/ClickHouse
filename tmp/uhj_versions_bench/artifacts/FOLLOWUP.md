# Follow-up investigation: runtime filters, warm-run state, and the two outliers

Three questions, answered with the command output recorded under
`tmp/uhj_versions_bench/artifacts/followup/`.

---

## (a) Is the runtime filter active under `unified_hash`? — **Yes, and it is load-bearing**

`supportsRuntimeFilter` in `src/Processors/QueryPlan/Optimizations/joinRuntimeFilter.cpp`
lists `JoinAlgorithm::UNIFIED_HASH` alongside `HASH`/`PARALLEL_HASH`/`GRACE_HASH`, and the
runtime behaviour matches:

| evidence | baseline | unified_hash |
|---|---|---|
| `BuildRuntimeFilter` steps in `EXPLAIN PLAN` | 6 | 6 |
| `RuntimeFiltersCreated` (every run) | 6 | 6 |
| `RuntimeFilterRowsChecked`, TPC-H q8 | 240.6M | 240.5M |
| TPC-H q8 with `enable_join_runtime_filters=0` | — | **123.1s vs 0.73s (168× slower)** |

So the filter is not merely planned, it is doing the work: removing it costs unified_hash
two orders of magnitude on q8. `UnifiedHashJoin` also implements the fixed-hash-table filter
hand-off (`publishSharedRuntimeFilters`).

On TPC-DS q54 the filter is created but *dynamically disabled* for a poor pass ratio on both
arms in the cold plan (`RuntimeFilterBlocksSkipped` ≈ 9,100 of ~9,800 blocks), which is why
`enable_join_runtime_filters=0` changes nothing there (29.26s vs 29.29s).

---

## (b) Do warm runs use cached state under `unified_hash`? — **Partly: preallocation yes, plan feedback no**

The versions benchmark runs each query 6× against one long-lived server, so runs 2–6 inherit
two kinds of state: the OS page cache, and the server's `HashTablesStatistics` cache.

**What unified_hash does use:** hash-map preallocation. `HashJoinPreallocatedElementsInHashTables`
on q8 is `0` on run 1, `24,453,056` on runs 2–6, and back to `0` under
`collect_hash_table_stats_during_joins=0`. It is worth nothing measurable here (0.74s with,
0.70s without — inside noise).

**What it does not use:** the statistics → join-reorder feedback loop. The query plan is
byte-identical cold vs warm under `unified_hash`, while under the default algorithm list it
is restructured after run 1:

```
tpch q8, baseline default, cold: supplier nation nation region lineitem part orders customer
tpch q8, baseline default, warm: customer orders part lineitem supplier nation nation region
tpch q8, unified_hash, cold==warm (plan identical)
```

The gate is in `calculateHashTableCacheKeys.cpp`, which only assigns a statistics cache key
when `allowParallelHashJoin(...)` is true, and that returns false unless `parallel_hash` is in
`join_algorithm`:

```cpp
if (std::ranges::none_of(join_algorithms, [](auto algo) { return algo == JoinAlgorithm::PARALLEL_HASH; }))
    return false;
```

`unified_hash` alone never satisfies it — and neither does plain `hash`, which is the control
that proves this is not UHJ-specific code (see below).

---

## (c) TPC-H q8 and TPC-DS q54 — **both deltas are a plan change, not an algorithm difference**

### Reproduced exactly

| run | baseline q8 | uhj q8 | baseline q54 | uhj q54 |
|-----|------------|--------|--------------|---------|
| 1 (cold) | 9.25s | 9.27s | 31.0s | 29.9s |
| 2–6 (hot) | **353–357s** | **0.73s** | **0.38s** | **29.4s** |

`JoinBuildTableRowCount` from `system.query_log` explains all of it:

| | cold | warm |
|---|---|---|
| baseline q8 | 25,916,097 | **1,212,609,641** (47× bigger build side) |
| uhj q8 | 25,916,097 | 25,916,097 (unchanged) |
| baseline q54 | 91,971,348 | **166,051** (554× smaller build side) |
| uhj q54 | 91,969,267 | 91,973,481 (unchanged) |

### Root cause, isolated by settings

Turning off the statistics cache removes the anomaly **in both directions** and makes the two
engines agree:

| variant | baseline q8 warm | baseline q54 warm |
|---|---|---|
| default | 353.0s | 0.38s |
| `collect_hash_table_stats_during_joins=0` | **0.74s** | **31.0s** |
| `join_runtime_filter_size_from_hash_table_stats=0` | 353.2s | 0.40s |
| `enable_join_runtime_filters=0` | 351.0s | 0.39s |

Only `collect_hash_table_stats_during_joins` matters. Runtime-filter sizing and the runtime
filter itself are not the cause.

### Control: the same baseline binary, only the algorithm list changed

| `join_algorithm` | q8 warm | q8 plan | q54 warm | q54 plan |
|---|---|---|---|---|
| `parallel_hash,hash` | 358.7s | **flips** | 0.39s | **flips** |
| `hash` | 1.53s | stable | 36.3s | stable |
| `unified_hash` (branch binary) | 0.73s | stable | 29.4s | stable |

Plain `hash` behaves exactly like `unified_hash`. The differentiator is the presence of
`parallel_hash` in the list, not anything in the UnifiedHashJoin implementation.

### Hotspot evidence (perf, mid-query, `--call-graph fp`)

Both slow cases are dominated by hash-table **build** on the oversized build side, which is
what the plan difference predicts:

```
baseline q8 (353s, 1.2B-row build side)
  33.16%  DB::RowRefList::insert(unsigned long, DB::Arena&)
   8.16%  DB::HashJoinMethods<...>::insertFromBlockImplTypeCase<...>
   7.31%  DB::countColumnsSizeInSelector(...)
   4.50%  LZ4_compress_fast_extState
   2.51%  DB::BloomFilter::addHashPairs(...)

uhj q54 (29s, 92M-row build side)
  33.34%  DB::Unified::HashJoinMethods<...>::insertFromBlockImplTypeCase<... UInt256 ... TwoLevelHashMapTable ...>
   7.12%  LZ4_compress_fast_extState
   5.26%  DB::IColumnHelper<DB::ColumnVector<long>>::scatter(...)
   4.18%  HashTable<wide::integer<256ul, unsigned int>, ...>
```

The two fast cases (uhj q8 at 0.73s, baseline q54 at 0.38s) show no join symbols at all —
the query finishes well inside the sampling window and the profile is background threads.

### Apples-to-apples engine comparison (same plan, same warm state)

With the plan flip neutralised, the engines are equivalent on exactly these two queries:

| query | baseline `parallel_hash` + stats off | `unified_hash` | delta |
|---|---|---|---|
| tpch q8 (warm) | 0.74s | 0.73s | −1% (noise) |
| tpcds q54 (warm) | 31.0s | 29.2s | −6% |

## What this means for the earlier benchmark report

The two headline outliers in `artifacts/SUMMARY.md` are measurement artifacts:

- tpch q8 `−99.8%` is **not** a uhj win — it is baseline re-planning itself into a
  1.2-billion-row build side on warm runs.
- tpcds q54 `+7330%` is **not** a uhj regression — it is baseline re-planning itself into a
  166k-row build side that uhj never sees.

Since q8 drives the TPC-H geomean, the `−18%` TPC-H figure does not survive either. The JOB
`+11%` regression was re-tested separately and does not survive either — see
`JOB_REGRESSION.md`: at equal plans uhj is 2.9% faster on JOB, with a genuine but much
smaller regression left on 7 of 113 queries.

Any future A/B against a `parallel_hash`-containing default must either set
`collect_hash_table_stats_during_joins=0` on both arms, or wire `unified_hash` into
`allowParallelHashJoin`, or the comparison measures query plans rather than join algorithms.
