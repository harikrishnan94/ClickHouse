# Measuring the map-selection divergences: D4, D8, D9, D13, D16

Five entries of `artifacts/DIVERGENCE_INVENTORY.md` concern which hash table
`src/Interpreters/UnifiedHashJoin/` builds and how it is filled and walked. This document
specifies how to price each of them. It is a design; nothing here has been run.

Scripts: `setup_synth.sh` (fixtures), then `m_D4.sh`, `m_D8.sh`, `m_D9.sh`, `m_D13.sh`,
`m_D16.sh`, each independent and each safe to re-run. Shared harness: `_maps_common.sh`.
Results land in `/mnt/data/uhj_versions_bench/measure/<id>/`.

**Two corrections to the inventory are baked into this design.** Both were found by re-reading
the source while writing it, and both change what the measurement can conclude:

1. **D13 does not apply to LowCardinality.** The inventory says the only type declaring
   `reads_whole_block_at_construction` is `HashMethodSingleLowCardinalityColumn`, reached via
   `LowCardinalityKeyGetterForJoin`. It is not: the flag lives on `HashMethodKeysFixed`
   (`Common/ColumnsHashing/HashMethod.h:411`), the composite fixed-width key getter, and it is
   the only declaration in the tree. `LowCardinalityKeyGetterForJoin`
   (`UnifiedHashJoin/KeyGetter.h:26`) does not declare it and does not inherit it, so
   `shareKeyGetterAcrossBuckets` is *false* for it and its dictionary-sized
   `visit_cache` / `mapped_cache` / `offset_cache` are rebuilt **per slot per block**. D13 is
   therefore a benefit for composite-key joins and a silent cost for LowCardinality ones — the
   opposite sign from what the inventory predicts. See §4.
2. **D16's extra hash is dead code.** `tryGetLowCardinalityMethod` accepts only `String` and
   `FixedString` nested types, `ReverseIndex::use_saved_hash` is `!is_numeric_column`, and
   `tryGetSavedHash` computes the hash array on first call rather than returning null. So
   `saved_hash` is always non-null for every dictionary the join can reach, and the branch that
   builds a second key holder and hashes it a second time is unreachable. What remains live is
   one extra `getIndexAt` per build row, which the compiler may well eliminate. §5 is designed
   to measure a quantity that may legitimately be zero, and to tell "zero" apart from "not
   measured".

---

## 0. Common ground

### 0.1 Arms

| arm | binary | why |
| --- | --- | --- |
| `hash` | `clickhouse-baseline` | merge-base serial hash join |
| `parallel_hash` | `clickhouse-baseline` | merge-base `ConcurrentHashJoin` |
| `unified_hash` | `clickhouse-uhj` | the branch |

`hash` and `parallel_hash` **must** come from the merge-base build. The branch adds seven
`ProfileEventTimeIncrement` timers to `ConcurrentHashJoin`'s build and probe hot paths (D14);
running the baseline arm on the branch binary charges it a `clock_gettime` pair per block and
per scattered sub-block, which is exactly the granularity D9 is trying to resolve.

### 0.2 Which map each arm builds

This table is the whole reason the five divergences need separate scripts — the arms do not
disagree about one map, they disagree about a different map for each key shape.

| build key | `hash` | `parallel_hash` (per shard) | `unified_hash`, `max_threads=1` | `unified_hash`, `max_threads>1` |
| --- | --- | --- | --- | --- |
| `UInt8` | `key8` — `FixedHashMap`, min/max on | `key8` — N private `FixedHashMap`s, probe blocks scattered | `key8` — `PartitionedFixedHashMap`, min/max **off** | same |
| `UInt16` | `key16` — as above | `key16` — as above | `key16` — as above | same |
| `UInt64`, small range | `key64` → converted to `range*_key64` (`FixedHashMapWithSizeBits`, min/max on) | `two_level_key64`, **never converted** | `key64` → `range*_key64` (`PartitionedFixedHashMap`, min/max off) | `two_level_key64` → `range*_key64`, same |
| `LowCardinality(String)` | `low_cardinality_key_string` | `two_level_key_string` on the **materialised** `String` column | `low_cardinality_key_string` | `two_level_low_cardinality_key_string` |
| two `UInt64`s | `keys128` | `two_level_keys128` × N | `keys128` | `two_level_keys128` |

Sources: `UnifiedHashJoin/HashJoin.h:112` (`JoinFixedHashMap` = `PartitionedFixedHashMap`),
`HashJoin/HashJoin.cpp:202` (`&& !use_two_level_maps`, absent at `UnifiedHashJoin/HashJoin.cpp:346`),
`HashJoin/HashJoin.cpp:418` (`default: return type` — no two-level form for `key8`/`key16`),
`UnifiedHashJoin/HashJoin.cpp:2062` vs `HashJoin/HashJoin.cpp:1989`
(`canConvertToFixedHashMap` accepts the two-level types only on the branch),
`UnifiedHashJoin/HashJoin.h:53` (`useTwoLevelMaps(max_threads) = max_threads > 1`).

### 0.3 Confirming the map that was actually built

Reading a number before confirming the map type is how this whole family goes wrong. Three
methods, in order of preference:

1. **The construction log line.** Both arms emit, at `LOG_TEST`, from the `HashJoin` constructor:

   ```
   <instance> Keys: ..., datatype: <Type>, kind: ..., strictness: ..., right header: ...
   ```

   It reaches the client with `--send_logs_level=test`. `capture_maptype` in `_maps_common.sh`
   runs every case once with that flag and appends the extracted `datatype:` values to
   `maptypes.txt`, keeping the raw log next to it. **`parallel_hash` prints one line per shard**,
   each tagged `(concurrentN)`, so the line count is also the shard count — which is precisely
   the D9 question, and the cheapest way to answer it.

2. **Two more lines, captured when present.** `Using a dictionary-aware hash map for the single
   LowCardinality join key` (`LOG_TRACE`, `UnifiedHashJoin/HashJoin.cpp:409`) settles D8, and
   `Converted join hash map to fixed hash map (range: R, keys: K)` (`LOG_DEBUG`,
   `UnifiedHashJoin/HashJoin.cpp:2059`) settles which `range*` map D4 got and prints the two
   numbers the whole of §1 turns on.

3. **`perf` symbol names**, when the log is not enough — for instance to confirm that a shard
   really instantiated a *private* `FixedHashMap` rather than a bucket of a shared one. Record
   the server while the query loops:

   ```bash
   sudo perf record -F 299 -g -p "$(pgrep -f 'clickhouse-uhj server')" -o tmp/d9.perf -- sleep 20
   sudo perf report -i tmp/d9.perf --stdio --sort symbol | head -60
   ```

   The map type appears in the mangled `insertFromBlockImplTypeCase` and `joinRightColumns`
   instantiations: `PartitionedFixedHashMap` / `TwoLevelHashTable`, `FixedHashMap`,
   `FixedHashMapWithSizeBits` and the key type are all template arguments and survive
   demangling. `../job_perf.sh` already performs this recording; point it at a case query.

**The runner must check `maptypes.txt` before reading `timings.tsv`.** If a case shows
`UNKNOWN`, or a `datatype:` other than the one this document names for it, that case's numbers
are void, not interesting.

### 0.4 Settings held fixed on every query

`MAPS_SETTINGS` in `_maps_common.sh`:

| setting | value | why |
| --- | --- | --- |
| `collect_hash_table_stats_during_joins` | `0` | equal reserve policy on both arms; the harness rule for this whole comparison (D11) |
| `query_plan_join_swap_table` | `false` | the right table must stay the build side, or a `RIGHT JOIN` stops iterating the map the SQL says it builds |
| `query_plan_optimize_join_order_limit` | `0` | no reordering |
| `query_plan_convert_join_to_in` | `0` | the join must stay a join; several build sides here are small enough to tempt the rewrite |
| `parallel_hash_join_threshold` | `0` | `rhs_size_estimation >= threshold` always holds, so `parallel_hash` is never silently downgraded to serial `hash`. Several build sides here are 64 rows; without this the baseline arm would not be running the algorithm it is named after, and D1 would contaminate every number |
| `enable_join_runtime_filters` | `0` | on by default, and for exactly the `key8`/`key16`/`range*` maps of §1 and §3 it publishes a shared fixed-hash-table filter that can prune the probe source. Separate mechanism, separate divergences. §1 and §3 each add one as-shipped case with it back on |
| `max_memory_usage` | `0` | no accidental spill differences |
| `log_processors_profiles` | `1` | the primary instrument, see §0.5 |

A case that needs to override one of these passes it as an extra argument;
`_merge_settings` drops the default of the same name first, because the ClickHouse client
rejects a repeated option rather than letting the last one win.

### 0.5 Metrics

Per case × arm × `max_threads`, `REPS` repetitions (default 5) after one untimed warm-up.

* **Wall clock** — `client --time`, min and median. `timings.tsv`.
* **Stage time** — `system.processors_profile_log`, summed over streams and grouped by
  processor name. `stages.tsv`.
  * `FillingRightJoinSide` → build,
  * `JoiningTransform` → probe,
  * `DelayedJoinedBlocksWorkerTransform` and `*NonJoined*` → the non-joined scan.

  **For D4 this is the primary instrument, not wall clock.** The whole effect is a few hundred
  microseconds of extra cell walking; against a query that also has to read and decode a probe
  table it is invisible, but it is the entire content of the non-joined processor's
  `elapsed_us`. The stream *count* matters as much as the time: §1's leading-scan penalty is
  paid once per non-joined stream, so `nonjoined_streams` is what the penalty multiplies by.
* **Per-query hardware counters** — `metrics_perf_events_enabled=1` with six events
  (`PerfInstructions`, `PerfCPUCycles`, `PerfCacheMisses`, `PerfBranchMisses`,
  `PerfStalledCyclesBackend`, `PerfDataTLBMisses`), read from
  `system.query_log.ProfileEvents`. Attributed to the query's own threads, so no iteration
  counting and no server-wide window. `qlog.tsv` also derives instructions per build row and
  per probe row, which is the only figure comparable across row counts. The harness probes
  once at startup and silently drops to wall clock and stage times if
  `perf_event_paranoid` forbids it.
* **`memory_usage`** — a 65536-cell `key16` map is about 1.5 MB and a 262144-cell `range18`
  map about 6 MB, per slot-set. Cheap sanity check that the arms built the map sizes this
  document claims.
* **`perf stat`, optional** (`PERF=1`): two 6-event groups, `cpu_cycles,inst_retired,
  stall_backend,stall_backend_mem,br_mis_pred_retired,mem_access` and `cpu_cycles,inst_retired,
  l1d_cache_refill,ll_cache_miss_rd,dtlb_walk,mem_access`, with `cpu_cycles` and `inst_retired`
  in both so the two passes can be cross-checked. Same loop-and-count shape as
  `../deep_metrics_norm.sh`. Only worth spending on §1 and §3, where the question is memory
  behaviour rather than instruction count.
* **Result agreement** — every case is written so all three arms must return identical
  scalars. `results_check.tsv`; `summarize` prints `MISMATCH` when they do not. A mismatch
  voids the timings beside it.

### 0.6 Exact cell sizes

Several claims below are in units of cells. To turn them into bytes without guessing:

```bash
.claude/tools/cppexpr.sh -i Common/HashTable/FixedHashMap.h -i Interpreters/RowRefs.h \
  'OUT(sizeof(FixedHashMapCell<UInt16, DB::RowRefList>)) OUT(sizeof(FixedHashMapCell<UInt8, DB::RowRef>))'
```

Do this before interpreting §1, and record the answer in the results directory. (It compiles
against the tree, so do not run it while a timed measurement owns the machine.)

---

## 1. D4 — fixed maps lose the min/max optimisation

### 1.1 Mechanism

Every fixed / direct-addressed map in UHJ is a `PartitionedFixedHashMap`
(`UnifiedHashJoin/HashJoin.h:112`), whose `FixedRangeStorage` constructor calls
`flat.disableMinMaxOptimization()` unconditionally and permanently, because parallel inserts
would race on the cached `min` and `max` (`Common/HashTable/TwoLevelHashTable.h:204-208`).
`FixedHashTable::begin` then walks forward from cell 0 until it finds a populated cell instead
of jumping to `buf + min`, and `end` is `buf + NUM_CELLS` instead of `buf + max + 1`
(`Common/HashTable/FixedHashTable.h:403-417`), so a full traversal costs the whole key range
rather than the observed span.

### 1.2 What forces the path, and how to verify

A single non-nullable integer join key, plus something that traverses the map. Two traversals
exist:

* the `RIGHT`/`FULL` non-joined scan (`UnifiedHashJoin/HashJoin.cpp:1462`, `map.begin()`);
* `tryRerangeRightTableDataImpl`'s `forEachMapped` (`UnifiedHashJoin/HashJoin.cpp:1866`), on
  `INNER`/`LEFT` joins — but only under `allow_experimental_join_right_table_sorting`, which is
  off by default, and only when the build side is at most `join_to_sort_maximum_table_rows`
  (10000) with at least `join_to_sort_minimum_perkey_rows` (40) rows per key.

`UInt8`/`UInt16` keys select `key8`/`key16` directly. `UInt32`/`UInt64`/`Int32`/`Int64` keys
reach a `range*` map only through `tryConvertToFixedHashMapImpl`, whose gates are:
`enable_join_fixed_hash_table_conversion` on, one disjunct, not `ASOF`, at most `2^18` keys,
`max_key - min_key < 2^18`, and — above `2^16` — at least 25% fill
(`if ((1ULL << bits) > key_count * 4) return false`).

Verify with `Converted join hash map to fixed hash map (range: R, keys: K)` and the
`datatype:` line. Expected `datatype:` per case is in the table below; anything else voids the
case.

**Two penalties, not one, and they scale differently.** Naming them separately is what makes
the worst case designable:

* *leading* — `firstPopulatedCell`'s linear search from cell 0. Paid at `map.begin()`, i.e.
  **once per non-joined stream**, including the streams that own no iteration bucket and
  return immediately after. `FixedRangeStorage::iterationBuckets()` is 1, so at
  `max_threads = 16` fifteen of the sixteen streams do nothing *except* this scan.
* *trailing* — the walk from the last real key to `NUM_CELLS`. Paid once, by the owning stream.

Converted `range*` maps store `key - min_key`, so they are always populated from cell 0 and
have no leading penalty. `key8`/`key16` store the raw key, so placing the keys at the top of
the key space is what makes the leading penalty exist at all.

### 1.3 Worst case

`d4_dim_k16_top64`: 64 `UInt16` keys, 65472–65535, at the very top of a 65536-cell `key16`
buffer; `RIGHT JOIN` against a 4-row probe that matches none of them, so every build row is
non-joined and the traversal *is* the query.

```sql
SELECT count(), sum(r.v)
FROM bench_synth.d4_probe_k16_nomatch AS l
RIGHT JOIN bench_synth.d4_dim_k16_top64 AS r ON l.k = r.k
```

Result on every arm: `64  2016`.

Why this is the worst case: it maximises the leading penalty (65472 of 65536 cells scanned
before the first key) *and* multiplies it by the stream count (16 at `max_threads = 16`, so
about 1.05 M cell probes, roughly 16 MB touched at a `FixedHashMapCell<UInt16, RowRefList>` of
about 24 bytes), while the baseline scans 64 cells once. There is no larger multiplier
available: `key16` is the widest direct-addressed key type, `key8` caps at 256 cells, and the
`range*` maps cannot have a leading penalty at all.

`d4_dim_k16_bot64` is the same 64 keys at the bottom of the range and isolates the trailing
penalty alone; the pair separates the two components. `d4_dim_u64_r262k` is the largest
*absolute* gap the conversion permits — 65536 keys two apart span 131073, overflow `range17`
into a 262144-cell `range18` buffer, and `262144 > 65536 * 4` is false by exactly one cell, so
the fill guard passes: 131071 wasted cells, trailing only.

### 1.4 Case matrix

`nomatch` probes are 4 rows that match nothing, except against the two `_full` controls where
by construction no `UInt8`/`UInt16` value can miss.

| case | build table | rows | expected `datatype:` | leading + trailing wasted (UHJ) | baseline scan |
| --- | --- | --- | --- | --- | --- |
| `k16_top64_right` | `d4_dim_k16_top64` | 64 | `key16` | 65472 + 0 | 64 |
| `k16_bot64_right` | `d4_dim_k16_bot64` | 64 | `key16` | 0 + 65472 | 64 |
| `k16_top4096_right` | `d4_dim_k16_top4096` | 4096 | `key16` | 61440 + 0 | 4096 |
| `k16_full_right` | `d4_dim_k16_full` | 65536 | `key16` | 0 + 0 (**control**) | 65536 |
| `k8_top16_right` | `d4_dim_k8_top16` | 16 | `key8` | 240 + 0 | 16 |
| `k8_full_right` | `d4_dim_k8_full` | 256 | `key8` | 0 + 0 (**control**) | 256 |
| `u64_r257_right` | `d4_dim_u64_r257` | 257 | `range16_key64` | 0 + 65279 | 257 |
| `u64_r65k_right` | `d4_dim_u64_r65k` | 65536 | `range16_key64` | 0 + 0 (**control**) | 65536 |
| `u64_r131k_right` | `d4_dim_u64_r131k` | 65537 | `range17_key64` | 0 + 65535 | 65537 |
| `u64_r262k_right` | `d4_dim_u64_r262k` | 65536 | `range18_key64` | 0 + 131071 | 131073 |
| `u64_sparse_right` | `d4_dim_u64_sparse` | 4096 | `key64` / `two_level_key64`, **no conversion** (**control**) | n/a | n/a |
| `k16_top64_full` | `d4_dim_k16_top64` | 64 | `key16` | as `k16_top64_right`, `FULL JOIN` | |
| `k16_top64_inner` | `d4_dim_k16_top64` | 64 | `key16` | **control**: `INNER`, nothing traverses | |
| `k16_top64_right_rtf` | `d4_dim_k16_top64` | 64 | `key16` | as-shipped: `enable_join_runtime_filters=1` | |
| `k16_rerange_inner` | `d4_rerange_k16` | 10000 | `key16` | second path: `INNER` + `allow_experimental_join_right_table_sorting=1` | ~250 |

`u64_sparse` spans 131041 with only 4096 keys, fails the 25% guard, and stays a plain
`key64`. It is the control that shows the effect belongs to the fixed map and not to the shape
of the data: it must show no arm gap beyond the family baseline.

`k16_rerange_k16` is the only shape satisfying `rightTableCanBeReranged()` — 250 distinct keys
at the top of the range × 40 rows. `sum(r.v)` is in the select list on purpose:
`tryRerangeRightTableData` returns early when `sample_block_with_columns_to_add.columns() == 0`.

### 1.5 Average case

Sweep `max_threads ∈ {1, 4, 16}` across the whole matrix — the leading penalty is the only one
that scales with threads, so the `top` vs `bot` gap widening with `max_threads` while the
`bot` gap stays flat is the signature that confirms the mechanism rather than merely observing
a difference.

Sweep the probe side too, `{4 rows, 10 M rows}` (`d4_probe_k16_nomatch`,
`d4_probe_k16_10m`), to place the penalty on a scale: the 4-row probe gives D4's absolute cost
in isolation, the 10 M probe gives the fraction of a query that actually does work.

Free axes already in the matrix: key range (256 / 65536 / 131072 / 262144 cells), fill
(64 / 4096 / 65536 keys), strictness (`RIGHT` / `FULL` / `INNER`).

### 1.6 Real-world exposure

**Almost certainly none, and the script must confirm it rather than assume it.**

* No `UInt8`, `Int8`, `UInt16`, `Int16`, `Bool`, `Enum8` or `Enum16` column exists in any table
  of `job`, `tpch`, `tpcds` or `coffeeshop`. Checked twice: against
  `/mnt/ch/ClickBench-master/versions/create/schema/*.columns`, where every such column belongs
  to `hits`, `ontime`, `logs2`, `logs3`, `lineorder_flat`, `trips` or `uk_price_paid` — none of
  which are loaded — and against the 56 `ATTACH TABLE` statements actually on disk under
  `/mnt/data/uhj_versions_bench/server_shared/data/metadata/{job,tpch,tpcds,coffeeshop}/`,
  which use only `Int32` (66 columns), `Int64` (158), `UInt32` (46), `Date`, `FixedString`,
  `String` and floats. So `key8` and `key16` are unreachable in the four suites.
* The `range*` half *is* reachable. `job` keys are `Int32` and its dimension tables are tiny
  (`comp_cast_type` 4 rows, `company_type` 4, `kind_type` 7, `role_type` 12, `link_type` 18,
  `info_type` 113 — all `range8_key32`); `tpch.nation` (25) and `tpch.region` (5) are `UInt32`;
  `tpcds.date_dim.d_date_sk` is `UInt32` with about 73049 contiguous values, which converts to
  `range17_key32` because `131072 <= 73049 * 4`.
* But neither traversal is reached. `grep -icE '(right|full)[[:space:]]+(outer[[:space:]]+)?join'`
  gives 0 for `job`, `tpch` and `coffeeshop`, and 2 for `tpcds` — queries 55 and 101 — and both
  of those join on **two** columns (`(web.item_sk, web.d_date)` and
  `(customer_sk, item_sk)`), so both build a `keys*` map and neither can be a fixed map at all.
  `allow_experimental_join_right_table_sorting` is off by default, so the `forEachMapped` path
  is not reached either.

**Conclusion to state in the report: D4 has zero real-world exposure in the four loaded
suites — the maps are built, but nothing ever walks them.** The runner must still answer, from
the live server, the four questions `realworld_report` writes to
`measure/setup/realworld_exposure.txt`, because the static reading above could be wrong about
which tables are actually loaded.

### 1.7 Expected direction and confounds

Direction: **against UHJ**, by a constant per join build (per non-joined stream for the
leading part), independent of probe size. Prediction in §6.

Confounds:

* `enable_join_runtime_filters` publishes a filter built from exactly these maps. Off
  everywhere; one as-shipped case has it on.
* `parallel_hash` never converts `key64` to a `range*` map at all
  (`canConvertToFixedHashMap` rejects `two_level_key64` on the baseline), so for the `u64_*`
  cases the `parallel_hash` arm is running a genuinely different algorithm and its gap is not
  D4. Compare `unified_hash` against `hash` there, and read `parallel_hash` only as context.
* At `max_threads = 1` UHJ still uses `PartitionedFixedHashMap` for `key8`/`key16` — the map is
  not two-level-on-demand, it is always partitioned — so D4 is present even single-threaded.
  That makes `max_threads = 1` the clean isolate for the trailing penalty.
* `used_flags` is sized from `getBufferSizeInCells`, which is `2^size_bits` on both arms for
  the same map type, so it is not a confound between arms — but it is between *cases*, which
  is why the row-count controls exist.

---

## 2. D8 — LowCardinality maps permitted in parallel builds

### 2.1 Mechanism

The baseline refuses the dictionary-aware map whenever the build is parallel —
`if (table_join->oneDisjunct() && !use_two_level_maps && ...)` at
`HashJoin/HashJoin.cpp:202` — so `parallel_hash` materialises the `LowCardinality` key column
and builds `two_level_key_string` over full strings. UHJ drops the `!use_two_level_maps`
clause (`UnifiedHashJoin/HashJoin.cpp:346`) and adds
`two_level_low_cardinality_key_{string,fixed_string}` (`HashJoin.h:302-314, 407-408`), so it
keeps the dictionary and the per-block dedup cache in a parallel build.

### 2.2 What forces the path, and how to verify

A single non-nullable `LowCardinality(String)` or `LowCardinality(FixedString)` join key, one
disjunct, not `ASOF`. Numeric dictionaries are rejected by `tryGetLowCardinalityMethod` on both
sides, so `LowCardinality(UInt32)` will not do.

Verify: `datatype: two_level_low_cardinality_key_string` plus `Using a dictionary-aware hash
map for the single LowCardinality join key` on `unified_hash` at `max_threads > 1`;
`datatype: two_level_key_string` and no dictionary line on `parallel_hash`;
`datatype: low_cardinality_key_string` on `hash`. Three different maps in one case is normal
here and is the point.

Fixtures pin `low_cardinality_use_single_dictionary_for_part=1` and
`low_cardinality_max_dictionary_size = 2 * D + 8192` at insert time, then `OPTIMIZE FINAL`. At
the default the writer starts a fresh dictionary every 8192 entries, and the per-block
dictionary size — which is what the whole of this section scales with — would be an artifact
of that default rather than a property of the data. `setup_synth.sh` prints `uniqExact(k)` per
table so the runner can confirm the dictionary is the size intended.

### 2.3 Worst case

Two of them, in opposite directions, because D8 is a capability rather than a regression and
its sign depends on one ratio: **dictionary size versus rows per block**.

*Best for UHJ* — tiny dictionary, probe-heavy. Every block of 65409 probe rows resolves 16
distinct dictionary indices, so `visit_cache` turns 65409 hash-table lookups into 16, and
`parallel_hash` additionally materialises 50 M × 48 bytes of string:

```sql
SELECT count(), sum(r.v)
FROM bench_synth.lc_probe_d16 AS l
INNER JOIN bench_synth.lc_dim_d16 AS r ON l.k = r.k
SETTINGS max_threads = 16
```

*Worst for UHJ* — dictionary far larger than a block, build-heavy. At 1 M dictionary entries
each block touches at most 65409 of them, so the cache never repays itself, and because
`LowCardinalityKeyGetterForJoin` is **not** shared across slots (§4) the 1 M-entry
`visit_cache` + `mapped_cache` (+ `offset_cache`) — roughly 17 MB — is allocated and zeroed
once per slot per block, sixteen times over at `max_threads = 16`:

```sql
SELECT count()
FROM bench_synth.lc_nomatch AS l
INNER JOIN bench_synth.lc_sweep_d1m AS r ON l.k = r.k
SETTINGS max_threads = 16
```

### 2.4 Case matrix and average-case sweep

Probe-heavy, `INNER JOIN`, probe 50 M rows, build = the matching `lc_dim_dN`:

| case | probe | dictionary | index width | dict vs 65409-row block |
| --- | --- | --- | --- | --- |
| `lc_d16` | `lc_probe_d16` | 16 | `UInt8` | 1/4000 — maximal dedup |
| `lc_d1k` | `lc_probe_d1k` | 1000 | `UInt16` | 1/65 |
| `lc_d100k` | `lc_probe_d100k` | 100000 | `UInt32` | 1.5× — dedup gone |
| `lc_d1m` | `lc_probe_d1m` | 1000000 | `UInt32` | 15× — cache is pure cost |
| `str_d1k` | `str_probe_d1k` | n/a | n/a | the materialised twin of `lc_d1k`, same bytes, `String` column: prices the dictionary path against the materialised path within one arm |

Build-heavy, `lc_nomatch` probe, build = `lc_sweep_d{1k,10k,100k,1m}` (20 M rows each, dictionary
the only axis).

Sweep `max_threads ∈ {1, 4, 16}` throughout. `max_threads = 1` is important: there UHJ and
baseline `hash` build the *same* `low_cardinality_key_string` map, so it is the zero point
against which the parallel arms are read. The crossover to look for is where the `unified_hash`
curve crosses `parallel_hash` as the dictionary grows.

### 2.5 Real-world exposure

**None.** No `LowCardinality` column exists in any of the 64 schema files under
`/mnt/ch/ClickBench-master/versions/create/schema/`, nor in any of the 56 `ATTACH TABLE`
statements on disk under
`/mnt/data/uhj_versions_bench/server_shared/data/metadata/{job,tpch,tpcds,coffeeshop}/` — the
datasets were deliberately built with old-compatible types. So no query in `job`, `tpch`,
`tpcds` or `coffeeshop` can reach any `low_cardinality_*` map on either arm, and D8, D13 and
D16 are synthetic-only.

The runner must still confirm this from the live server, because metadata on disk describes how
the tables were *attached*, and a materialised view or a temporary table could still introduce
the type at query time:

```sql
SELECT database, table, name, type FROM system.columns
WHERE database IN ('job','tpch','tpcds','coffeeshop') AND type LIKE '%LowCardinality%'
ORDER BY database, table, name
```

`realworld_report` runs exactly this and writes it to
`measure/setup/realworld_exposure.txt` §1. **An empty result confirms zero real-world exposure
for D8, D13 and D16 and must be stated as such in the report.** A non-empty result invalidates
this section's "synthetic only" framing and the affected queries should be measured directly.

### 2.6 Expected direction and confounds

Direction: **for UHJ** at small dictionaries, **against** at large ones, crossing somewhere
near dictionary ≈ block size. Prediction in §6.

Confounds:

* The arms build different maps *and* read different column representations. Materialisation
  cost belongs to `parallel_hash`'s side of the ledger legitimately — it is what the baseline
  must do — but it should be reported separately, which is what the `str_d1k` twin is for.
* Dictionary size at read time is a property of the part, not the query. Check
  `setup_synth.sh`'s `uniqExact` output before believing any point on the sweep.
* §4's per-slot construction cost is inside every parallel `unified_hash` number here. D8 and
  D13 cannot be fully separated on the LowCardinality axis; §4 measures the shared component
  so it can be subtracted.
* `max_joined_block_size_rows` and `max_block_size` set the denominator of "dictionary versus
  block". Both left at their defaults; do not vary them here.

---

## 3. D9 — `key8`/`key16` parallel build: shared map versus sharded maps

### 3.1 Mechanism

Baseline `chooseMethod(..., use_two_level_maps = true)` has no two-level form for `key8` or
`key16` and returns them unchanged (`HashJoin/HashJoin.cpp:418`, `default: return type`), so
each `parallel_hash` shard owns a private single-level `FixedHashMap`, `twoLevelMapIsUsed()` is
false, and `ConcurrentHashJoin::joinBlock` routes every probe block through `dispatchBlock`
(`ConcurrentHashJoin.cpp:455-464`). UHJ's `key8`/`key16` are `PartitionedFixedHashMap`
(`UnifiedHashJoin/HashJoin.h:112`), one shared flat buffer whose 256 buckets route
`BucketLock`s rather than owning storage, so the build takes locks and the probe does not
scatter at all.

### 3.2 What forces the path, and how to verify

A single `UInt8`/`UInt16` (or `Int8`/`Int16`/`Enum8`/`Enum16`/`Bool`) join key, plus
`max_threads > 1` and an algorithm that shards. Verify by counting `datatype:` lines: on
`parallel_hash` there is one per shard, all saying `key8` or `key16`; on `unified_hash` there
is one, saying the same. Confirm the probe-side difference from
`system.processors_profile_log` — the baseline's scatter shows up as `JoiningTransform`
`elapsed_us` that has no counterpart in the UHJ arm — and, if that is not conclusive, from
`perf` symbols (`ConcurrentHashJoin::dispatchBlock`, `scatterBlocksWithSelector`).

Note that UHJ *does* scatter the **build** side for these key types:
`scatterBlockBySlot` produces `dense_keys` when the key columns total no more than
`sizeof(IColumn::Selector::value_type)` = 8 bytes per row (`SlotScatter.cpp:101-123`), and
`UInt8`/`UInt16` qualify. So D9 is not "scatter versus no scatter"; it is "scatter on both
sides versus scatter on the build side only, plus bucket locks".

### 3.3 Worst case

Two, again in opposite directions.

*Best for UHJ* — probe-heavy, so the baseline's per-block `dispatchBlock` dominates and the
shared map is never contended:

```sql
SELECT count(), sum(r.v)
FROM bench_synth.d9_probe_k8 AS l
INNER JOIN bench_synth.d9_dim_k8 AS r ON l.k = r.k
SETTINGS max_threads = 16
```

100 M probe rows over a 256-row build side. The baseline scatters 100 M rows into 16 selectors
per block for a map that fits in 6 KB of L1; UHJ does one lookup per row into one shared
buffer.

*Worst for UHJ* — build-heavy with every key present in every block, so every build block has
rows for every slot and the shared map must take all 16 bucket locks per block, while the
baseline's shards each fill a private buffer:

```sql
SELECT count(), sum(r.v)
FROM bench_synth.d9_probe_k8_small AS l
INNER JOIN bench_synth.d9_build_k8 AS r ON l.k = r.k
SETTINGS max_threads = 16
```

50 M build rows, `k = number % 256`, so each of 65409 rows per block hits all 256 keys and
therefore all 16 slots. There is no shape with more lock traffic per byte of data: with a
1-byte key the payload per lock acquisition is as small as it can be.

### 3.4 Case matrix and sweep

| case | probe | build | note |
| --- | --- | --- | --- |
| `k8_probe` | `d9_probe_k8` 100 M | `d9_dim_k8` 256 | probe-heavy, `key8` |
| `k16_probe` | `d9_probe_k16` 100 M | `d9_dim_k16` 65536 | probe-heavy, `key16`, map no longer L1-resident |
| `k64_probe` | `d9_probe_k64` 100 M | `d9_dim_k64` 256 | **control**: same 256 values 8 bytes wide, with `enable_join_fixed_hash_table_conversion=0` so both arms use `key64`/`two_level_key64` and D9 and D4 are both off. Whatever gap remains belongs to the rest of the fork and must be subtracted from the other rows |
| `k8_build` | `d9_probe_k8_small` 1 M | `d9_build_k8` 50 M | build-heavy, maximal lock traffic |
| `k8_probe_rtf` | as `k8_probe` | | as-shipped: `enable_join_runtime_filters=1` |

`max_threads ∈ {1, 2, 4, 8, 16}` on all of them. At `max_threads = 1` both arms are serial and
the gap should be ≈ 0; the slope in `max_threads` is the divergence. Report throughput per
thread, not just wall clock: the interesting failure mode is a shared map that stops scaling
at 8 threads, which a wall-clock table at 16 threads alone would hide.

### 3.5 Real-world exposure

**None**, for the reason in §1.6: no `UInt8`/`Int8`/`UInt16`/`Int16`/`Bool`/`Enum` column
exists in any of the 56 tables of the four loaded suites, so `key8` and `key16` are
unreachable. Same `system.columns` query, §2 of `realworld_exposure.txt`:

```sql
SELECT database, table, name, type FROM system.columns
WHERE database IN ('job','tpch','tpcds','coffeeshop')
  AND (type IN ('UInt8','Int8','UInt16','Int16','Bool') OR type LIKE 'Enum8%' OR type LIKE 'Enum16%')
```

An empty result confirms it. Note that this is a property of *these* datasets, not of the
divergence: `hits.OS`, `hits.Age`, `ontime.Month` and similar are exactly the columns real
users join on, and they are one dataset away.

### 3.6 Expected direction and confounds

Direction: **for UHJ** on probe-heavy shapes, **against** on build-heavy contended ones.
Prediction in §6.

Confounds:

* D4 rides along on every `key8`/`key16` case, but only through `INNER` joins here, so nothing
  traverses the map and D4 contributes nothing. Keep the strictness at `INNER` for all of §3;
  a `RIGHT` variant here would measure §1, not §3.
* D1 (the `parallel_hash_join_threshold` gate) is neutralised by
  `parallel_hash_join_threshold = 0`, which matters because `d9_dim_k8` is 256 rows and would
  otherwise be built serially by the baseline.
* Runtime filters would prune the probe source and change the row counts, not just the times.
  Off except in `k8_probe_rtf`.
* `d9_probe_k8` compresses extremely well (256 distinct 1-byte values), so a large share of
  the query is decompression that both arms share. Use instructions-per-probe-row from
  `qlog.tsv` rather than wall clock when the wall-clock gap looks small.

---

## 4. D13 — one key getter shared across slots

### 4.1 Mechanism

`insertIntoSlots` calls `insertFromBlockImpl` once per non-empty slot for the same block, and
`blockKeyGetter` gives each of those calls either a shared getter or a private one depending on
`shareKeyGetterAcrossBuckets<KeyGetter>()`, which is true only for key getters declaring
`reads_whole_block_at_construction` (`UnifiedHashJoin/HashJoinMethods.h:102-131`,
`HashJoinMethodsImpl.h:286-295`). The sole declaration in the tree is on `HashMethodKeysFixed`
(`Common/ColumnsHashing/HashMethod.h:411`), whose constructor packs the entire block into
`prepared_keys` — so composite fixed-width keys pack once per block instead of once per slot,
and everything else, **including `LowCardinalityKeyGetterForJoin`**, does not.

### 4.2 What forces the path, and how to verify

Sharing is active when all of: `max_threads > 1` (otherwise `num_slots == 1` and the question
is moot), the map is a `keys32`/`keys64`/`keys128`/`keys256` variant, and
`scatterBlockBySlot` did **not** produce `dense_keys` — because when it does,
`insertFromBlockImplTypeCase` builds a private getter over the scattered columns and never
consults the shared one (`HashJoinMethodsImpl.h:320-330`). `dense_keys` appear when the key
columns total at most `sizeof(IColumn::Selector::value_type)` = 8 bytes per row. Hence:

| key | map | `usePreparedKeys` | `dense_keys` | sharing effective? |
| --- | --- | --- | --- | --- |
| 2 × `UInt32` (8 B) | `keys64` | yes | yes | **no** — bypassed by the dense path |
| 2 × `UInt64` (16 B) | `keys128` | yes | no | **yes** — the case sharing exists for |
| 4 × `UInt64` (32 B) | `keys256` | no (`sizeof(Key) > 16`) | no | shared, but the constructor is cheap, so ≈ no effect |
| `LowCardinality(String)` | `low_cardinality_key_string` | n/a | no (LC excluded at `SlotScatter.cpp:104`) | **no** — caches rebuilt per slot per block |

There is no direct A/B: the flag is `constexpr` and cannot be turned off from SQL. The
measurement is therefore a **counterfactual by contrast** — compare key shapes where sharing is
active against shapes where it is not, and read the *slope in `max_threads`*, which is the
thing sharing removes. A per-slot constructor cost of `C` shows up as `C × slots` per block; a
shared one shows up as `C` per block regardless of slots.

Verify the map from `datatype:`; verify slot count from `slotCountForThreads`
(`min(bit_ceil(max_threads), 256)`), i.e. 1, 2, 4, 8, 16 for the sweep below.

### 4.3 Worst case

The worst case for D13 is the case where sharing is *absent* and its absence is most
expensive: a `LowCardinality(String)` build with a huge dictionary at maximum slot count. Every
one of the 16 slots constructs its own `LowCardinalityKeyGetterForJoin` per block, and each
constructor allocates and zeroes `visit_cache` (1 B/entry), `mapped_cache` (8 B/entry) and,
when used flags are needed, `offset_cache` (8 B/entry) sized by the **whole** dictionary — not
by the rows in that slot:

```sql
SELECT count()
FROM bench_synth.lc_nomatch AS l
INNER JOIN bench_synth.lc_sweep_d1m AS r ON l.k = r.k
SETTINGS max_threads = 16
```

20 M build rows in about 306 blocks × 16 slots × 17 MB of cache allocation and zeroing. Note
this is UHJ-versus-UHJ arithmetic: there is no baseline that does this, because the baseline
never uses a LowCardinality map in a parallel build at all (§2). The comparison that gives it
meaning is against `str_sweep_d1m`, the identical data as plain `String`, whose getter
constructor is O(1) and whose build time must therefore be flat in dictionary size.

The worst case for the mechanism *working* — i.e. the largest saving attributable to D13 — is
`keys128` at 16 slots, where a shared getter packs 65409 rows into `UInt128` once instead of
sixteen times per block:

```sql
SELECT count(), sum(r.a)
FROM bench_synth.d13_nomatch_keys128 AS l
INNER JOIN bench_synth.d13_build_keys128 AS r ON l.a = r.a AND l.b = r.b
SETTINGS max_threads = 16
```

### 4.4 Case matrix and sweep

Build-heavy throughout: 4-row non-matching probe, so build cost is the query.

| case | build | key getter | sharing | prediction |
| --- | --- | --- | --- | --- |
| `lc_d1k` … `lc_d1m` | `lc_sweep_d{1k,10k,100k,1m}` 20 M | LC | no | build time grows with dictionary **and** with slots |
| `str_d1k`, `str_d1m` | `str_sweep_d{1k,1m}` 20 M | `HashMethodString` | n/a (O(1) ctor) | flat in dictionary size — the control |
| `keys128` | `d13_build_keys128` 50 M | `HashMethodKeysFixed<UInt128>` | yes | flat in slots |
| `keys64` | `d13_build_keys64` 50 M | `HashMethodKeysFixed<UInt64>` | bypassed by `dense_keys` | packs per slot, but over 1/N of the rows, so also flat |
| `keys256` | `d13_build_keys256` 20 M | `HashMethodKeysFixed<UInt256>` | yes, cheap ctor | flat, and equal to `keys128` in slope |

`max_threads ∈ {1, 2, 4, 8, 16}` on all rows. **The deliverable of this section is a slope, not
a ratio**: `d(build_us) / d(slots)` at fixed rows, for each key shape. A non-zero slope for the
`lc_*` rows and a zero slope for `str_*` and `keys128` is the finding; the size of the LC slope
is what sharing would be worth if it were extended to `LowCardinalityKeyGetterForJoin`.

Arms: `unified_hash` is the subject. `parallel_hash` and `hash` are run on the same cases as
context — `parallel_hash` shows what the baseline pays instead (a real scatter of the key
columns, then one cheap getter per shard over 1/N of the rows), and `hash` gives the
single-threaded floor.

### 4.5 Real-world exposure

For the LowCardinality half: **none**, per §2.5.

For the composite-key half: **present, and this is the one section of this document with real
coverage.** Every `keys*` map in the four suites reaches the shared path. Confirm with:

```sql
SELECT database, table, name, type FROM system.columns
WHERE database IN ('job','tpch','tpcds','coffeeshop') AND type IN ('UInt64','Int64')
```

and by reading the two `tpcds` `FULL OUTER JOIN`s (queries 55 and 101), whose two-column
`Int64` keys are exactly the 16-bytes-per-row `keys128` shape where sharing is effective and
`dense_keys` does not apply. Any suite query joining on two `Int64` columns qualifies; the
`census_explain` helper in the sibling `_common.sh` enumerates them if a full census is wanted.
Because the effect is a slope in slot count rather than a level, a real-world confirmation
needs the same `max_threads` sweep on those queries, not a single run.

### 4.6 Expected direction and confounds

Direction: **for UHJ** on composite keys — but bounded, because the baseline's per-shard
getters also pack each row exactly once in total; sharing buys UHJ parity, not an advantage.
**Against UHJ** on LowCardinality, where the mechanism does not apply and the per-slot cost is
real. Prediction in §6.

Confounds:

* This is the hardest section to attribute, because there is no switch. Anything that changes
  per-block work with `max_threads` — block size, arena growth, `bucket_bytes` accounting (D2),
  lock traffic (D9) — also produces a slope. The `str_*` control is what separates them: it
  shares nothing, has an O(1) constructor, and any slope it shows is the floor to subtract from
  the others.
* `keys64`'s dense-key bypass makes it look like the shared case even though it is not. That is
  a genuine finding about `scatterBlockBySlot`, not noise, and should be reported as one.
* At `max_threads = 1` all rows collapse to one slot and one getter; that point anchors the
  regression line and must be included.
* The whole LowCardinality half rests on an assumption that has to be checked rather than
  believed: that a block read from MergeTree carries the *part's* dictionary, so
  `dictionary.getNestedNotNullableColumn()->size()` is 1 M and not the number of distinct values
  in that block. The fixtures force one part with one dictionary — `OPTIMIZE FINAL` plus
  `low_cardinality_use_single_dictionary_for_part` and `low_cardinality_max_dictionary_size` in
  the server's default profile, so that background merges cannot undo the INSERT's sizing — and
  `setup_synth.sh` prints `uniqExact(k)` and the part count per table. If the `lc_*` build time
  turns out flat in dictionary size, the two candidate explanations are that the getter is
  shared after all (contradicting `HashJoinMethods.h:123-127`) or that the per-block dictionary
  is not the part dictionary. Distinguish them by comparing `memory_usage` across the sweep
  before concluding anything about D13.

---

## 5. D16 — LowCardinality `emplaceKey` does the index decode twice

### 5.1 Mechanism

UHJ's `LowCardinalityKeyGetterForJoin::emplaceKey` computes `getIndexAt(row_)`, builds a key
holder from it, and then passes `routingHashForRow(data, row_, pool)` as the hash argument to
`data.emplace` — and `routingHashForRow` calls `getIndexAt(row_)` again, and, when the
dictionary has no saved hash, builds a *second* key holder and hashes it
(`UnifiedHashJoin/KeyGetter.h:119-137`). The baseline instead passes `saved_hash[row]` directly
and falls back to the no-hash `emplace` overload, decoding the index once
(`HashJoin/KeyGetter.h:136-143`).

### 5.2 What forces the path, and how to verify

Any single `LowCardinality(String)` build side. But — as established in the preamble — the
second key holder and second hash are **unreachable**: `tryGetLowCardinalityMethod` admits only
`String` and `FixedString` nested types, `ReverseIndex::use_saved_hash` is
`!is_numeric_column`, and `tryGetSavedHash` materialises the hash array on first call rather
than returning null (`Columns/ReverseIndex.h:349-367`). So `saved_hash` is always non-null and
the live divergence is one extra `getIndexAt` — a switch on `size_of_index_type` plus an
indexed load — per build row, which the compiler may CSE away entirely since both calls read
the same const column element with no intervening store.

**This section is therefore designed to measure a quantity that may be zero, and to
distinguish zero from unmeasured.** The discriminator is key width:

* an extra full string hash scales with key width;
* an extra index decode does not.

So run the same dictionary (1000 entries), the same row count, and two key widths, 16 and 48
bytes. If the `unified_hash` − `hash` gap per build row is the *same* at both widths, the extra
hash is not being paid and the source reading is confirmed. If it grows with width, the extra
hash is live and the source reading is wrong.

Verify the map with `datatype: low_cardinality_key_string` on **both** arms — this is the one
case in the family where the arms agree on the map, which is what makes the isolation possible.

### 5.3 Worst case

`max_threads = 1`, build-heavy, 4-row non-matching probe:

```sql
SELECT count()
FROM bench_synth.lc_nomatch AS l
INNER JOIN bench_synth.lc_build_w48_d1k AS r ON l.k = r.k
SETTINGS max_threads = 1
```

`max_threads = 1` is not an incidental choice: it is the only setting at which UHJ and baseline
`hash` build the *same* single-level `low_cardinality_key_string` map with one slot, one getter
and no scatter, so the build inner loop is the only thing that differs. At `max_threads > 1`
the arms diverge on the map itself (§2) and D16 disappears into D8.

Why this is the worst case for D16 specifically: the effect is strictly per build row, so
maximise build rows and minimise everything else. 50 M build rows against a 4-row probe that
matches nothing puts essentially the whole query in `emplaceKey`, and the 1000-entry dictionary
keeps `size_of_index_type` at `UInt16` — the middle branch of the `getIndexAt` switch, neither
the branch-predictor-friendliest nor the rarest.

### 5.4 Case matrix

| case | build | dictionary | key width | purpose |
| --- | --- | --- | --- | --- |
| `lc_w16` | `lc_build_w16_d1k` 50 M | 1000 | 16 B | the width discriminator, narrow |
| `lc_w48` | `lc_build_w48_d1k` 50 M | 1000 | 48 B | the width discriminator, wide |
| `str_w48` | `str_build_w48_d1k` 50 M | n/a | 48 B | **control**: plain `String`, no LC key getter on either arm, so any `unified_hash` − `hash` gap here is the rest of the fork and must be subtracted |

Arms `unified_hash` and `hash` at `max_threads = 1`; `parallel_hash` at
`max_threads = 1` as a third reading of the same thing. The whole matrix repeated at
`max_threads = 16` for context only — clearly labelled as D8 territory, not D16.

### 5.5 Average case

The parameter that matters is dictionary size, because it selects the `getIndexAt` branch:
`UInt8` below 256 entries, `UInt16` below 65536, `UInt32` beyond. Re-run `lc_w48` against
`lc_sweep_d1k`, `lc_sweep_d100k` and `lc_sweep_d1m` (20 M rows, width 48) at
`max_threads = 1`. A D16 cost that is real should be roughly constant across these; one that
moves a lot with dictionary size is measuring cache behaviour of `saved_hash[row]`, which both
arms pay.

Row count is the other axis, and it is a linearity check rather than a sweep: the effect is
per build row, so `(uhj − hash)` per row must be constant between the 20 M and 50 M tables. If
it is not, the measurement is dominated by something fixed and should not be reported as a
per-row cost.

### 5.6 Real-world exposure

**None**, per §2.5 — no `LowCardinality` column in the four suites.

### 5.7 Expected direction and confounds

Direction: **against UHJ**, and predicted to be too small to separate from noise. Prediction in
§6.

Confounds:

* The `unified_hash` − `hash` gap at `max_threads = 1` on a LowCardinality build contains every
  other single-threaded build-path divergence, not only D16. `str_w48` is the subtraction term;
  the reportable number is the *difference of differences*,
  `(uhj − hash)_lc_w48 − (uhj − hash)_str_w48`, and even that keeps whatever is specific to the
  LC getter.
* `tryGetSavedHash` computes the dictionary hash array lazily on first call — an O(dictionary)
  SipHash pass over the strings. Both arms trigger it, but only once, and on a 20 M-row build
  it is noise; on a small build it would not be. Keep the build sides large.
* Turbo/thermal drift over a 50 M-row build dwarfs a per-row instruction. Interleave the arms
  rather than running all `unified_hash` points then all `hash` points, use the median of five,
  and prefer `PerfInstructions` per build row over wall clock — an extra `getIndexAt` is an
  instruction-count question, and the instruction count is deterministic where the clock is
  not.

---

## 6. Predictions

Written before any run, so they can be wrong. Magnitudes are order-of-magnitude, on the
16-vCPU / 32 GiB cgroup these scripts use.

| divergence | direction | predicted magnitude | what would falsify it |
| --- | --- | --- | --- |
| **D4** | against UHJ | Per join build: ~130 µs at `max_threads = 1` for `k16_top64` (65472 cells × ~24 B ≈ 1.5 MB streamed), rising to ~2 ms of CPU across 16 non-joined streams at `max_threads = 16`. `u64_r262k` similar per stream but trailing-only, so flat in threads. As a fraction of a real query: under 1% for anything with a probe side, ~10–40% for the 4-row-probe cases. Zero on `INNER`. | a `top` vs `bot` gap that does not widen with `max_threads`, which would mean `firstPopulatedCell` is not on the per-stream path |
| **D8** | for UHJ at small dictionaries, against at large | `lc_d16`: UHJ 1.5–3× faster than `parallel_hash` on probe-heavy 50 M-row joins, most of it avoided string materialisation rather than avoided lookups. `lc_d1k`: 1.2–1.8×. `lc_d100k`: roughly parity. `lc_d1m` build-heavy: UHJ 1.5–3× *slower*, dominated by the per-slot cache zeroing of §4 | a crossover far from dictionary ≈ block size, which would mean the dedup cache is not what drives the sign |
| **D9** | for UHJ on probe-heavy, against on contended builds | `k8_probe` at 16 threads: UHJ 1.3–2× faster, the baseline's per-block `dispatchBlock` being pure overhead for a 6 KB map. `k16_probe`: smaller, 1.1–1.4×, the map no longer fits L1 so lookups dominate. `k8_build`: UHJ 1.2–2× slower on bucket-lock traffic. `k64_probe` control: within noise | a `k64_probe` control that is *not* within noise, which would mean the other rows are measuring the rest of the fork |
| **D13** | for UHJ on composite keys (parity, not advantage); against on LowCardinality | `keys128` build slope in slots ≈ 0; the counterfactual saving, extrapolated from the LC slope, would be ~15× the per-block pack cost at 16 slots. LC slope: build time growing ~linearly to roughly 16 × (dictionary × 17 B) of zeroing per block, which at 1 M entries and 20 M rows should be the majority of build time — call it 3–10× the `str_sweep_d1m` control | an LC build time flat in `max_threads`, which would mean the caches are being shared after all and the source reading in the preamble is wrong |
| **D16** | against UHJ | Below 1% of build time, and quite possibly exactly zero after compiler CSE. Expect the width discriminator to show no width dependence, confirming the extra hash is dead. If the extra `getIndexAt` survives CSE, ~2–4 instructions per build row against a build loop that costs 50–200 | a gap that grows with key width, which would mean `saved_hash` is null somewhere and the second hash is live |

Two of these predictions are structural rather than numeric and matter more than the numbers:
**D8, D13 and D16 have zero real-world exposure in the four loaded suites** (no
`LowCardinality` column exists anywhere in them), and **D4 and D9 have zero exposure too**
(no `UInt8`/`UInt16` column exists in them, and the only two `FULL OUTER JOIN`s in the suites
use two-column keys, so no fixed map is ever traversed). Every number this family produces is
synthetic. That is a finding about the benchmark, not about the fork — the columns these
divergences fire on, `hits.OS` or `ontime.Month` or any dictionary-encoded string, are
ordinary — but it does mean none of the five can be validated against the loaded suites, and
the report should say so rather than implying the suites confirmed anything.
