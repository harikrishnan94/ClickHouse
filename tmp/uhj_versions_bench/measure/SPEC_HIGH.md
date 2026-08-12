# Measuring the five high-impact UHJ divergences

Design document for `m_A.sh`, `m_B.sh`, `m_D1.sh`, `m_D2.sh`, `m_D3.sh` in this directory. The
divergences themselves are catalogued in `../artifacts/DIVERGENCE_INVENTORY.md`; this file only
covers how to put numbers on five of them.

Repository `/mnt/ch/ClickHouse`, branch `cursor/uhj-versions-bench-4f2a`, merge-base `3218492309c`.

## 0. Shared setup {#shared-setup}

### Arms {#arms}

| arm | binary | algorithms it may be asked for |
|---|---|---|
| `baseline` | `/mnt/data/uhj_versions_bench/bin/clickhouse-baseline` | `hash`, `parallel_hash` |
| `uhj` | `/mnt/data/uhj_versions_bench/bin/clickhouse-uhj` | `unified_hash` |

`hash` and `parallel_hash` must be measured on the merge-base binary. The branch added seven
`ProfileEventTimeIncrement` timers to `ConcurrentHashJoin`'s build and probe hot paths (divergence
D14), so measuring `parallel_hash` from the branch build charges the baseline arm a `clock_gettime`
pair per block that the shipped merge base does not pay.

Both binaries read the same data directory `/mnt/data/uhj_versions_bench/server_shared/data`
(`job`, `tpch`, `tpcds`, `coffeeshop`, plus the synthetic `bench_synth`), so every script stops the
server before switching arms. `_common.sh` takes an exclusive `flock` on
`/mnt/data/uhj_versions_bench/measure/.lock` for exactly that reason: `stop_server` kills every
process in the `uhj_versions_bench` cgroup, so two measurements running at once destroy each other.

### What every measurement pins {#pinned-settings}

`COMMON_SETTINGS` in `_common.sh`:

* `collect_hash_table_stats_during_joins=0` — the statistics cache changes the join order on one arm
  only and then dominates every other effect. This is a harness convention inherited from
  `../job_perf.sh`, not a preference.
* `enable_join_runtime_filters=0` — a runtime filter can prune the probe side outright, which swamps
  the build-side effects all five of these divergences live in.
* `enable_join_fixed_hash_table_conversion=0` — post-build conversion to `PartitionedFixedHashMap`
  is divergence D4 and is measured elsewhere. Note this does **not** disable the `key8`/`key16`
  fixed maps chosen by `chooseMethod` from the key type, which D2 depends on.
* `query_plan_join_swap_table=false` — pins the right-hand table as the build side, so "build side"
  means what the SQL says on both arms.
* `query_plan_convert_join_to_in=0` — a build-only probe query must stay a join.

### Metrics {#metrics}

Two independent instruments, used in that order of preference:

1. **Per-query hardware counters.** `metrics_perf_events_enabled=1` with
   `metrics_perf_events_list=PerfInstructions,PerfCPUCycles,PerfCacheMisses,PerfBranchMisses,PerfStalledCyclesBackend,PerfDataTLBMisses`
   (six events, no PMU multiplexing). These are opened by the query's own threads, so they attribute
   exactly, need no iteration counting, and land in `system.query_log.ProfileEvents` next to
   `JoinBuildTableRowCount` — which makes "retired instructions per build row" a single SQL
   expression. `perfev_available` probes them at the start of each script; they come back as zeros
   when `perf_event_paranoid` forbids self-monitoring.
2. **Server-wide `perf stat`.** Two 6-event groups, `cpu_cycles` and `inst_retired` in both so they
   can be cross-checked:
   * `PERF_CORE` = `cpu_cycles, inst_retired, stall_backend, stall_backend_mem, br_mis_pred_retired, mem_access`
   * `PERF_MEM` = `cpu_cycles, inst_retired, l1d_cache_refill, ll_cache_miss_rd, dtlb_walk, mem_access`
   * `PERF_LOCK` = `syscalls:sys_enter_futex, syscalls:sys_enter_sched_yield` (D3 only)

   A window counts completed iterations of a loop so counters can be divided by iterations and rows.

3. **`system.processors_profile_log`.** `FillingRightJoinSide.elapsed_us` summed over streams is the
   build phase, `JoiningTransform` the probe, `NonJoinedBlocksTransform` the RIGHT/FULL tail. A
   thread blocked on a mutex inside a transform still accrues `elapsed_us`, which is what makes this
   the right instrument for D3.

**Retired instructions, not wall time, is the primary metric wherever the comparison crosses a
thread count.** Instructions are additive across build threads and therefore comparable between
`max_threads=1` and `max_threads=16`; wall time is not.

### Synthetic fixtures {#fixtures}

`ensure_synth` in `_common.sh` creates them in `bench_synth`, idempotently (a table is repopulated
only when its row count does not match). ~2.5 GB total, well inside the 20 GB budget. Key columns
use `CODEC(NONE)` deliberately: LZ4 decompression would add a few hundred instructions per build row
and dilute the per-row effects under study.

| table | rows | purpose |
|---|---|---|
| `build_u64` | 64 M | `UInt64` key, PK-ordered so `WHERE id < N` is an exact and cheap cardinality knob |
| `build_str48` | 16 M | 48-byte `String` key: most expensive single-column key getter |
| `build_keys256` | 16 M | four `UInt64` columns -> 32-byte packed key (`UInt256HashCRC32`) |
| `build_u64_null` | 16 M | `Nullable(UInt64)`, ~1% NULLs, for the RIGHT-join nullmap path |
| `build_u16` | 1 000 | `UInt16` key -> the `key16` fixed map |
| `probe_one` | 1 | one row, so a query is build-only |
| `probe_10m` | 10 M | keys 0..999, one match per probe row against every `dim_*` |
| `dim_1000` … `dim_1000000` | 1 k … 1 M | build sides straddling `parallel_hash_join_threshold` |
| `tiny_dim`, `tiny_fact` | 100 each | a query that is nothing but join setup |

### Running them {#running}

```bash
cd /mnt/ch/ClickHouse/tmp/uhj_versions_bench/measure
ARM=baseline ./m_A.sh        # or omit ARM to run both arms back to back
ARM=uhj      ./m_A.sh
```

Each script writes to `/mnt/data/uhj_versions_bench/measure/<id>/`, prints a compact summary, and
ends with `M_<id>_DONE`. Re-running a script replaces the rows of the cells it re-measures rather
than appending a second copy of them, so a partial run can simply be repeated. Only one script may
run at a time.

---

## A — `Unified::scatterBlockBySlot` {#a-scatter}

### Mechanism {#a-mechanism}

When `slots > 1`, `UnifiedHashJoin/HashJoin.cpp:1011` sends every build row through a full key getter
to compute a routing hash, derives a bucket and a slot from it, and then makes a second pass filling
the per-slot index columns (`SlotScatter.cpp:63-99`); when the summed key width is at most
`sizeof(IColumn::Selector::value_type)` = 8 bytes and the incoming selector is the identity range, a
third pass materialises scattered copies of the key columns as `dense_keys` (`SlotScatter.cpp:101-123`).
Serial `hash` does no scatter at all and `parallel_hash` does a different one
(`ConcurrentHashJoin::dispatchBlock`), so this is a UHJ-only per-row pass over the build side —
already measured once at ~46 instructions/row, +1.58 G instructions on JOB q64.

### Rebuild required {#a-rebuild}

**No.** `slots == 1` at `max_threads=1` disables the scatter, and the per-row cost is visible as the
difference in instructions per build row between `max_threads=1` and `max_threads>1` on the UHJ arm,
with `parallel_hash`'s own dispatch as the reference for what the equivalent work costs on the
baseline.

An optional variant would separate the scatter from everything else that `slots > 1` turns on (slot
locks, arenas, 256-bucket maps) by forcing `slotCountForThreads` to return 1 while leaving
`useTwoLevelMaps` alone, in `src/Interpreters/UnifiedHashJoin/HashJoin.cpp:70-76`:

```c++
size_t slotCountForThreads(size_t /*max_threads*/)
{
    return 1;
}
```

That is a diagnostic build only — it serialises all inserts behind one slot lock — so it answers
"how much of the mt=1 to mt=16 delta is the scatter" and nothing else. The main script does not need
it.

### Worst case {#a-worst-case}

48-byte `String` key, `ALL INNER`, one probe row, `max_threads=16`:

```sql
SELECT count()
FROM bench_synth.probe_one AS p
INNER JOIN (SELECT k FROM bench_synth.build_str48 WHERE id < 16000000) AS r
    ON toString(p.id) = r.k
```

Worst case because all three factors are maximised at once: the routing hash is a full hash over 48
bytes rather than a single integer mix; a variable-width key can never satisfy `max_bytes_per_row <=
8`, so `dense_keys` never fires and the scatter is pure overhead with no downstream saving; the probe
side is one row, so nothing amortises the build; and the insert itself is comparatively cheap per
row, which maximises the scatter's share.

The opposite corner is in the same grid on purpose: a single `UInt64` key takes the `dense_keys`
path, where the scatter's third pass produces a scattered copy of the key column that the insert
then reads instead of chasing the selector — so the net effect there may well be **negative**, i.e.
in UHJ's favour.

### Average case {#a-average-case}

Full grid, best of `REPEATS`, 8 M build rows, one probe row:

* key shape ∈ {`u64` (dense-keys path), `keys256` (32-byte packed key), `str48` (variable width)}
* strictness ∈ {`ALL INNER`, `LEFT ANY`}
* `max_threads` ∈ {1, 2, 4, 16}
* algorithm ∈ {`hash`, `parallel_hash`} on baseline, {`unified_hash`} on uhj

The reported quantity is retired instructions per build row (`PerfInstructions /
JoinBuildTableRowCount`). `max_threads=1` is the within-arm control: UHJ at mt=1 has `slots == 1` and
skips the scatter entirely, so `instr/row(mt=16) − instr/row(mt=1)` on the UHJ arm is the scatter plus
whatever else parallelism adds, and the same difference on `parallel_hash` is the baseline's price
for the same thing.

### Real world {#a-real-world}

Exposure is exactly `JoinBuildTableRowCount` whenever `max_threads > 1`, because the scatter runs
once per build row per clause. Pass 3 executes all four suites once on the UHJ arm and reports, per
query, `46 * JoinBuildTableRowCount` as a share of that query's own `PerfInstructions` — an estimate
at the previously measured rate, to be replaced by the rate this script's own grid produces for the
matching key shape. Queries at the top of that list (large build sides, little else going on) are
where the scatter is worth attacking; queries at the bottom are where it is noise.

### Metrics {#a-metrics}

`PerfInstructions`, `PerfCPUCycles`, `PerfCacheMisses`, `PerfDataTLBMisses`,
`PerfStalledCyclesBackend` per query, normalised by `JoinBuildTableRowCount`; wall time best-of-N;
`FillingRightJoinSide.elapsed_us`; server-wide `PERF_CORE` and `PERF_MEM` at the worst-case corner.

### Expected direction {#a-direction}

Against UHJ for wide and variable-width keys, plausibly **for** UHJ for single 8-byte keys where
`dense_keys` fires. This must be reported per key shape; a single average over shapes would hide a
sign change.

### Confounds {#a-confounds}

* At `max_threads > 1` the UHJ arm also gets 256-bucket maps and N slot locks and arenas. The scatter
  cannot be separated from those without the `slotCountForThreads` variant above; the grid measures
  their sum and says so.
* Reading the build table is part of every measurement. `CODEC(NONE)` keeps it small, and it is
  identical across arms at a fixed shape, but it inflates the denominator when instructions per row
  are compared against the ~46 figure measured on JOB q64.
* `toString(p.id)` on the one-row probe side is evaluated once and is not part of the build.

---

## B — `BITS_FOR_BUCKET_SERIAL = 0` {#b-serial-map}

### Mechanism {#b-mechanism}

`UnifiedHashJoin/HashJoin.h:48` sets `BITS_FOR_BUCKET_SERIAL = 0`, so at `max_threads == 1` the
`JoinHashMap` alias folds to a `TwoLevelHashMap` with a single bucket — one flat table — grown by
`HashTableGrowerWithPrecalculation`; `useTwoLevelMaps` (line 53) makes this a hard function of
`max_threads` with no setting behind it. `parallel_hash` at `max_threads == 1` instead builds one
shard whose `HashJoin` was constructed with `use_two_level_maps=true`, i.e. 256 sub-tables under
`TwoLevelHashTableGrower`, so a resize rehashes one bucket instead of the whole buffer and each
bucket's probe sequence stays inside a much smaller region.

Note there are two knobs and three layouts in play, which is why the script measures all three:

| configuration | buckets | grower |
|---|---|---|
| baseline `hash` (any `max_threads`) | 1 (single-level `HashMap`) | `HashTableGrower` |
| baseline `parallel_hash` at `max_threads=1` | 256 | `TwoLevelHashTableGrower` |
| `unified_hash` at `max_threads=1` | 1 | `HashTableGrowerWithPrecalculation` |

The comment at `UnifiedHashJoin/HashJoin.h:82-83` records that the two-level grower cost +35–44%
`FillingRightJoinSide` on a 500 k-key case, while the inventory records +12.5% instructions and +34%
cycles the other way on JOB q64 at `max_threads=1`. Both can be true at different cardinalities;
resolving that is the point of the sweep.

### Rebuild required {#b-rebuild}

**Yes for a clean isolation** — nothing at run time can make a serial UHJ build a 256-bucket map.
Two one-line variants, both in `src/Interpreters/UnifiedHashJoin/HashJoin.h`:

**V-B1** — bucket count only, keeping the flat grower (line 48):

```diff
-constexpr Int32 BITS_FOR_BUCKET_SERIAL = 0;
+constexpr Int32 BITS_FOR_BUCKET_SERIAL = 8;
```

**V-B2** — exactly `parallel_hash`'s layout, 256 buckets under `TwoLevelHashTableGrower` (lines 53-56):

```diff
-inline bool useTwoLevelMaps(size_t max_threads)
-{
-    return max_threads > 1;
-}
+inline bool useTwoLevelMaps(size_t /*max_threads*/)
+{
+    return true;
+}
```

V-B1 keeps `slotCountForThreads(1) == 1`, so the scatter stays off and only the map layout changes:
it is the clean single-variable experiment. V-B2 changes bucket count and grower together and lands
on the same layout `parallel_hash` uses, which makes it the right variant for asking "would adopting
the baseline's layout wholesale help".

Build into a separate directory from `/mnt/ch/ClickHouse/build/reldeb` and point
`M_B_VARIANT_BIN=/path/to/clickhouse-uhj-b1` at the result; `m_B.sh` picks it up as a third arm
(`ARM=variant`) automatically and skips it when unset.

**The no-rebuild proxy** is `unified_hash` at `max_threads=1` against `parallel_hash` at
`max_threads=1`, which the script always runs. Its caveats, all of which the proxy attributes to B
even though they are not B:

* `parallel_hash` at one shard still goes through `ConcurrentHashJoin::addBlockToJoin` and its
  `dispatchBlock` short-circuit, a different code path from `UnifiedHashJoin::addBlockToJoin`.
* Insert, probe and result assembly differ between the two implementations in ways catalogued as
  other divergences (D5–D24); the proxy cannot tell those from the map layout.
* Both arms are different binaries, so any compiler-level difference is folded in as well.

The proxy establishes the size of the total gap at `max_threads=1`; only V-B1 attributes it.

### Worst case {#b-worst-case}

64 M distinct `UInt64` keys, `max_threads=1`, one probe row:

```sql
SELECT count()
FROM bench_synth.probe_one AS p
LEFT JOIN (SELECT id AS k FROM bench_synth.build_u64 WHERE id < 64000000) AS r
    ON p.id = r.k
```

Worst case because a flat table reaching 64 M entries crosses ~18 power-of-two resizes, and the last
few allocate a buffer twice the size of a live one that is already far larger than the last-level
cache, then rehash every key in one pass over it. A 256-bucket layout does the same total work in
256 pieces, each of which fits in cache and each of which is rehashed independently. The single
`UInt64` key also means the per-row work outside the map is minimal, so the layout is as large a
share of the query as it can be.

The script's `perf stat` corner uses 16 M keys rather than 64 M so a 25–30 s window contains several
complete iterations.

### Average case {#b-average-case}

Cardinality sweep at `max_threads=1`, for each of `hash`, `parallel_hash`, `unified_hash` (and
`variant` when built):

* `u64` — 100 k, 1 M, 4 M, 16 M, 64 M distinct keys, build-only.
* `str48` — 100 k, 1 M, 4 M, 16 M. `JoinHashMapWithSavedHash` stores the hash, so a resize is a pure
  copy with no re-hashing: if the gap persists here it is layout and locality, not hashing.
* `probe_u64` — the same `u64` build sides with the 10 M-row probe side, so the lookup side of the
  layout question is measured too, not just the build.

The expected shape is a gap near zero while the whole table fits in cache and opening up as the
buffer outgrows it. `q_maptype` records the `datatype:` each configuration actually chose (from the
`LOG_TEST` line), so a cell where the planner picked something unexpected is visible rather than
silently averaged in.

### Real world {#b-real-world}

Serial UHJ is reachable in two ways, and only the second is common:

1. `max_threads=1` queries.
2. Every in-memory `GraceHashJoin` join, which `GraceHashJoin.cpp:739-750` pins to `max_threads = 1`
   (divergence D18). So **anything that spills** runs its in-memory joins in regime B.

Pass 3 therefore runs all four suites at `max_threads=1` on both arms and aggregates per suite. To
detect the second route in a real workload, look for `JoinSpillingHashJoinSwitchedToGraceJoin > 0` in
`system.query_log` — those queries are in regime B regardless of their `max_threads`.

### Metrics {#b-metrics}

Instructions, cycles, LL cache misses (`ll_cache_miss_rd`), dTLB walks (`dtlb_walk`),
`l1d_cache_refill`, all per build key; wall time best-of-N; the chosen map type. The four counters
are chosen to reproduce the originally reported B result (+12.5% instructions, +14.9% LL misses,
+8.3% dTLB walks, +34% cycles on JOB q64).

### Expected direction {#b-direction}

Regime-dependent, and the sweep exists because of that. Against UHJ at high cardinality where resize
and locality dominate; possibly for UHJ at low cardinality, where one small flat table beats 256
sub-tables that each pay their own indirection and each hold a nearly empty buffer.

### Confounds {#b-confounds}

* `collect_hash_table_stats_during_joins=0` is essential here: a size hint would pre-size the buffer
  and remove the resizes that are the whole subject.
* `HashJoinPreallocatedElementsInHashTables` should be 0 in `system.query_log` for every cell; a
  non-zero value means something pre-reserved and the cell is not measuring what it claims.
* The `str48` build side is 16 M rows, so the two largest cardinality points are skipped for it.
* At `max_threads=1` the pipeline is serial end to end, so the read side is on the critical path.
  Both arms read identically, but it caps the observable ratio.

---

## D1 — `parallel_hash_join_threshold` is bypassed {#d1-threshold}

### Mechanism {#d1-mechanism}

`Planner/PlannerJoins.cpp:1244` guards the `rhs_size_estimation >= parallel_hash_join_threshold`
decision with `&& !unified`, so `unified_hash` never sees the gate and is constructed with the raw
`params.max_threads` (lines 1259-1264); the same `&& !unified` appears on the spilling path at line
1211. Baseline `join_algorithm='direct,parallel_hash,hash'` runs the serial `HashJoin` for every join
whose right side is estimated below 100 000 rows, while `unified_hash` runs the parallel machinery —
256-bucket maps, `slotCountForThreads(max_threads)` slots and arenas, the scatter of divergence A,
and `max_threads` `FillingRightJoinSide` streams instead of one.

### Rebuild required {#d1-rebuild}

**No**, and this is the divergence where that matters most: `parallel_hash_join_threshold` is a
setting, so the baseline arm can be made to emulate either side of the gate. That separates the
*decision* from the *implementation*:

| configuration | arm | settings |
|---|---|---|
| `bgate` | baseline | `join_algorithm=direct,parallel_hash,hash`, `parallel_hash_join_threshold=100000` (default) |
| `bpar` | baseline | same, `parallel_hash_join_threshold=0` — always parallel |
| `bser` | baseline | `join_algorithm=hash` — always serial |
| `u` | uhj | `join_algorithm=unified_hash` |
| `umt1` | uhj | `join_algorithm=unified_hash`, `max_threads=1` |

`bpar − bgate` is the price of ignoring the gate, priced in the baseline's own implementation, and is
therefore the cleanest attribution of D1 as such. `u − bgate` is what a user actually experiences when
switching to `unified_hash`, D1 and everything else included. `umt1` is on the list because lowering
`max_threads` for the whole query is the only workaround UHJ offers, and the cost of that workaround
(a serial probe and a serial scan) belongs on the record.

### Worst case {#d1-worst-case}

A join whose entire cost *is* the join setup — 100 build rows, 100 probe rows, `max_threads=16`:

```sql
SELECT count()
FROM bench_synth.tiny_fact AS f
INNER JOIN bench_synth.tiny_dim AS d ON f.k = d.id
```

Worst case because serial `hash` allocates one 256-cell flat table (~4 KiB) and one arena and gets one
`FillingRightJoinSide`, while `unified_hash` at `max_threads=16` allocates a 256-bucket map
(256 × 256 cells, ~1 MiB), 16 arenas, 16 bucket locks and 16 build streams, and runs the scatter — to
insert 100 rows. Nothing amortises any of it.

The star-schema variant multiplies that by eight, which is the shape a real dashboard query has:

```sql
SELECT count() FROM bench_synth.tiny_fact AS f
INNER JOIN bench_synth.tiny_dim AS d1 ON f.k = d1.id
INNER JOIN bench_synth.tiny_dim AS d2 ON f.k = d2.id
-- ... d3 .. d8
```

run with `query_plan_optimize_join_order_limit=0`: nine relations is inside the reordering limit, and
without pinning it the two arms could be handed different join orders and the comparison would
measure the optimizer.

A third shape, 10 M probe rows against `dim_1000`, is the realistic version of the same thing: far
below the threshold, so the baseline runs it serially and `unified_hash` does not, but with enough
probe work that the setup cost is a small share rather than the whole query.

### Average case {#d1-average-case}

Build-side sweep across the threshold: `dim_1000`, `dim_10000`, `dim_50000`, `dim_99000`,
`dim_101000`, `dim_200000`, `dim_1000000`, each joined to the 10 M-row `probe_10m`. Separate physical
tables so `rhs_size_estimation` is the table's own row count and the gate decision is unambiguous;
`probe_10m`'s keys are 0..999, which every dimension contains, so each probe row matches exactly one
build row at every size and the result is a constant 10 M rows — the only thing that varies across
the sweep is the build side.

`EXPLAIN actions=1` (`Algorithm:`) and `EXPLAIN PIPELINE` (count of `FillingRightJoinSide`) are
recorded per cell, which is what makes the gate crossing visible rather than inferred: `bgate` should
flip from `HashJoin`/1 stream to `ConcurrentHashJoin`/16 streams between 99 000 and 101 000, and `u`
should show 16 streams everywhere.

### Real world {#d1-real-world}

Plan-only, therefore free: run `EXPLAIN actions=1` over every query in all four suites on the
baseline arm with default settings, and count the queries whose plan contains a bare `Algorithm:
HashJoin`. Every one of those is a join that `unified_hash` will run in parallel instead. The census
also records the plans with the gate disabled (`parallel_hash_join_threshold=0`) and the
`unified_hash` plans, so the three can be diffed per query. `EXPLAIN PIPELINE` corroborates: the
serial `HashJoin` has one `FillingRightJoinSide`, the parallel algorithms have `max_threads`.

### Metrics {#d1-metrics}

Wall time best-of-N (5× more repeats for the tiny shapes, where the fixed per-join cost is exactly
what noise hides); the chosen algorithm and build-stream count per cell; per-query instructions,
cycles and peak memory; `FillingRightJoinSide.elapsed_us`. Peak memory matters here independently of
time: 256 buckets and N arenas for a 100-row build side is a memory regression even when it is not a
time regression.

### Expected direction {#d1-direction}

Against UHJ below the threshold, by a roughly fixed per-join amount rather than a percentage —
which means it is invisible on a long query and dominant on a short one. Neutral above the threshold,
where both arms go parallel anyway.

### Confounds {#d1-confounds}

* `collect_hash_table_stats_during_joins=0` on both arms, or the cached statistics change the
  baseline's join order and nothing else is measurable.
* Nine-relation star query: pin `query_plan_optimize_join_order_limit=0`.
* At these sizes the query's own fixed overhead (parsing, planning, pipeline construction) is a large
  share of the wall time. `bser` versus `bpar` on the same binary brackets it: both pay the same
  fixed overhead, so their difference is the join machinery alone.
* `rhs_size_estimation` comes from the table's statistics, so the sweep must use separate physical
  tables — a `WHERE id < N` filter on one big table would not move the estimate the same way.

---

## D2 — `bucket_bytes` under-reports during the build {#d2-accounting}

### Mechanism {#d2-mechanism}

`insertIntoSlots` samples `slot_bytes(slot)` before and after each insert and adds only the
difference to `data->bucket_bytes` (`UnifiedHashJoin/HashJoin.cpp:140,166,222`), while both
`HashTable` and `FixedHashTable` allocate their buffer in their constructor — inside
`MapsTemplate::create`, before any insert — so the initial allocation sits inside `bytes_before` and
cancels out of every delta, deliberately (`UnifiedHashJoin/HashJoin.cpp:438-439`), until
`recomputeBucketBytes` restores it at the end of the build. Baseline `HashJoin::getTotalByteCount`
(`HashJoin/HashJoin.cpp:533-557`) recomputes the map contribution from the maps themselves on every
call and therefore always includes it.

Derived sizes of the un-accounted amount, from the source (`RowRefList` is 8 bytes by `static_assert`
at `RowRefs.h:390`, so a `key64` cell is 16 bytes):

* two-level map, any `max_threads > 1`: 256 buckets × 256 initial cells = 65 536 cells ≈ **1 MiB**
  — the "~1 MiB" the code comment refers to;
* one-bucket serial map, `max_threads == 1`: 256 cells, a few KiB;
* fixed maps (`key8`, `key16`, `range*`): the **whole** buffer, forever, because a `FixedHashTable`
  never grows and so the delta is exactly zero — 2^16 cells for `key16`, 2^18 for `range18_key64`.

`slot_bytes` also covers the per-slot `Arena`, whose growth *is* counted; only its initial chunk
(4 KiB per slot) cancels the same way, which is negligible next to the map term.

### Rebuild required {#d2-rebuild}

**No.** `getTotalByteCount` is observable from SQL in two independent ways.

1. `max_bytes_in_join` with `join_overflow_mode=throw`. The check is `bytes > max_bytes`
   (`QueryPipeline/SizeLimits.cpp:40`) against exactly the value under study, so **the smallest limit
   at which the query still succeeds is the peak the join reported during its build**. A bisection on
   that limit reads the accounting out to 4 KiB. The exception text — `Limit for JOIN exceeded, max
   bytes: 1.00 B, current bytes: 1.25 MiB` — corroborates it to two decimal digits, and with
   `max_bytes_in_join=1` it reports the state of the accounting after the very first block, which is
   where the un-accounted initial allocation is the whole story.
2. `max_bytes_before_external_join`. `SpillingHashJoin::addBlockToJoin` spills when
   `getTotalByteCount() * 2 >= max_bytes_before_external_join` (`SpillingHashJoin.cpp:153,158`), so
   under-reporting moves the spill point to a larger build side, and
   `ProfileEvents['JoinSpillingHashJoinSwitchedToGraceJoin']` in `system.query_log` says per query
   whether it moved.

### Worst case {#d2-worst-case}

A `UInt16` join key, which selects `key16` — a `PartitionedFixedHashMap` whose 2^16-cell buffer is
allocated in the constructor and never grows, so UHJ's delta accounting reports **zero** bytes for the
map for the entire build, no matter how many rows go in:

```sql
SELECT count()
FROM bench_synth.probe_one AS p
LEFT JOIN bench_synth.build_u16 AS r ON toUInt16(p.id) = r.k
```

Worst case because the error is 100% of the map term and permanent rather than transient, and because
1 000 build rows keep the stored columns at a few KiB, so essentially the whole reported number is the
map and the arm-to-arm difference is the un-accounted buffer undiluted.

The consequence side has its own worst case: the smaller `max_bytes_before_external_join` is, the
larger a fraction of it the fixed ~1 MiB error represents. The script uses 64 MiB, which is small
enough to be crossed by a few-million-row build side and large enough that the error is a percentage
rather than the whole budget.

### Average case {#d2-average-case}

Two sweeps.

*Accounting*, one number per cell in bytes: shapes `u16fixed` (1 000 rows, fixed map), `u64small`
(100 000 keys) and `u64mid` (2 000 000 keys) × configurations `bh` (`hash`, mt=1), `bhmt16` (`hash`,
mt=16), `bph` (`parallel_hash`, mt=16), `umt1` (`unified_hash`, mt=1), `umt16` (`unified_hash`,
mt=16). Reported per cell: bytes after the first block, peak reported bytes (bisection), and real
peak `memory_usage` with no limit set. The gap is then taken between configurations with the *same*
map layout — `umt1` against `bh`, `umt16` against `bph` — so the comparison is not confounded by the
layout difference of divergence B. `u64mid` is in the list to show the absolute gap staying constant
while the relative error shrinks, which is the honest framing.

*Consequence*: `max_bytes_before_external_join` fixed at 64 MiB, build side swept over 250 k … 6 M
rows against the 10 M-row probe side, recording `JoinSpillingHashJoinSwitchedToGraceJoin`, peak
`memory_usage` and wall time. The output is the first build size that spills, per configuration.

### Real world {#d2-real-world}

D2 is unobservable unless a memory bound is set, so the real-world question is not "which query
touches this code" but "of the queries that have a bound, how many land on the wrong side of it". The
suites are run twice per arm — unbounded and with `max_bytes_before_external_join = 512 MiB` — with
each arm configured the way a user would configure it (`direct,parallel_hash,hash` on baseline,
`unified_hash` on uhj). The per-query output lists exactly the queries that spill on one arm and not
the other under the same cap.

### Metrics {#d2-metrics}

Bytes: reported-after-first-block, peak-reported (bisection), peak `memory_usage`.
Counts: `JoinSpillingHashJoinSwitchedToGraceJoin`. Time: wall time per cell, which is the thing a
delayed spill ultimately changes.

### Expected direction {#d2-direction}

UHJ reports fewer bytes than baseline at every cell, by ~1 MiB for two-level maps and by the whole
buffer for fixed maps. Whether that is *good* is a separate question the script does not prejudge: a
delayed spill can be faster (the join stays in memory and never pays the partitioning) or it can
overshoot the memory budget the user set. Both outcomes are visible in the same table — peak
`memory_usage` against the configured cap on one side, wall time on the other.

### Confounds {#d2-confounds}

* `shrinkStoredBlocksToFit` fires when the reported total exceeds half of `max_bytes_in_join`, so the
  limit under test changes the stored bytes. This is why the headline shape is `u16fixed`, where the
  stored columns are a few KiB and there is nothing to shrink.
* Setting `max_bytes_before_external_join > 0` makes `isUsedByAnotherAlgorithm` true on both arms,
  which saves the key columns and increases the stored bytes on both. Never compare a capped cell to
  an uncapped one across arms; compare capped to capped.
* UHJ passes `map_reserve_bytes_cap = maxBytesBeforeExternalJoin()`
  (`UnifiedHashJoin/HashJoin.cpp:435`), so the cap also changes how much the map reserves. This is a
  second, independent effect of the same setting and is why the accounting pass uses
  `max_bytes_in_join` rather than `max_bytes_before_external_join`.
* `collect_hash_table_stats_during_joins=0`, or a size hint pre-allocates and the initial allocation
  is no longer the initial allocation.
* The baseline arm's `parallel_hash` shards the map across N `HashJoin` instances, each with its own
  buffer, and reports all of them: N MiB of initial two-level allocation, or N × 2^16 cells for the
  `u16fixed` shape, since `chooseMethod(..., use_two_level_maps=true)` leaves `key16` a
  `FixedHashMap` (`HashJoin/HashJoin.cpp:418-419`). That is a real difference in the same direction,
  not an artefact, but it means the `bph` column is measuring N buffers against UHJ's one and the
  per-shard figure is the like-for-like one.

---

## D3 — the parallel build serialises on one global `blocks_mutex` {#d3-lock}

### Mechanism {#d3-mechanism}

Every UHJ build thread takes the join-wide `blocks_mutex` at least twice per build block — once to
register the stored block (`UnifiedHashJoin/HashJoin.cpp:907`) and once inside
`shrinkStoredBlocksToFit`, which is called unconditionally at the end of `addBlockToJoin` (line 1093)
and takes the lock *before* deciding it has nothing to do (line 1101) — with more for RIGHT/FULL joins
and nullable keys (lines 1045, 1054, 1062) and a third one per block from
`SpillingHashJoin::addBlockToJoin`'s `getTotalByteCount()` call when
`max_bytes_before_external_join` is set (`SpillingHashJoin.cpp:158` into
`UnifiedHashJoin/HashJoin.cpp:701`). `ConcurrentHashJoin` has no join-wide lock at all: each shard has
its own mutex taken with `try_to_lock` so a thread that loses moves to another shard
(`ConcurrentHashJoin.cpp:340`), and the global totals are relaxed atomics.

### Rebuild required {#d3-rebuild}

**No**, and the reason is the core of the design: the cost is **per block**, while every other
build-side divergence between the arms — the scatter of divergence A above all — is **per row**. Hold
the row count fixed and shrink `max_block_size`: the number of global critical sections multiplies
while the per-row work is untouched. The slope of build time against block count is D3; the intercept
absorbs A and everything else per-row.

### Worst case {#d3-worst-case}

`max_threads=16`, `max_block_size=512`, 16 M build rows (31 250 blocks), a `Nullable` key and a RIGHT
join so the nullmap store adds a third mandatory acquisition per block:

```sql
SELECT count()
FROM bench_synth.probe_one AS p
RIGHT JOIN (SELECT k FROM bench_synth.build_u64_null WHERE id < 16000000) AS r
    ON p.id = r.k
```

Worst case because it maximises acquisitions per unit of work in three independent ways at once: 16
threads contending, the smallest practical block so the critical sections come as fast as possible,
and three global sections per block instead of two — all while the per-row work (one `UInt64` compare
and insert) is as cheap as it gets, so the lock is the largest possible share of the total.

`RIGHT JOIN` on a one-row probe side does mean ~16 M non-joined rows come out through
`NonJoinedBlocksTransform`. Both arms pay that identically, and the `u64` shape below has no such
tail, so the two together bracket it.

### Average case {#d3-average-case}

Grid at 16 M build rows, build-only:

* shape `u64` (two acquisitions per block): `max_block_size` ∈ {65536, 8192, 2048, 512} ×
  `max_threads` ∈ {1, 2, 4, 8, 16}
* shape `right` (three acquisitions per block): the same block sizes × `max_threads` ∈ {1, 16}
* arms: `parallel_hash` on baseline, `unified_hash` on uhj

Two readings come out of the same grid. Across `max_threads` at a fixed block size it is a
scalability curve — but UHJ's `max_threads=1` point is also where divergence B lives, so curve
*shapes* are comparable and mt=2 is the honest normalisation point, not mt=1. Across block size at a
fixed `max_threads` it is the per-block cost, reported as
`(t[bs=512] − t[bs=65536]) / (blocks[512] − blocks[65536])` in microseconds per block, for wall time
and for summed `FillingRightJoinSide.elapsed_us` separately.

A second pass measures the acquisition that the spilling wrapper adds: the same query with
`max_bytes_before_external_join` at 64 GiB (large enough never to spill) against 0. The quantity of
interest is the within-arm ratio, because enabling the wrapper also makes both arms save the key
columns.

### Real world {#d3-real-world}

Exposure is (number of build blocks) × (number of build threads), and the per-query proxy for the
first factor is `JoinBuildTableRowCount / max_block_size`. The suites are executed once per arm at
`max_threads=16` and the build-side processor time is read from `system.processors_profile_log`:
`FillingRightJoinSide.elapsed_us` summed over streams includes the time a thread spends blocked on
`blocks_mutex`, so at equal build rows the arm-to-arm ratio of that number is what D3 looks like in a
real query. The output is sorted by that ratio.

Queries with a large build side split into many small blocks are the candidates — which in these
suites means the fact-table-on-the-build-side joins, and anything whose build side comes out of a
subquery that emits small blocks rather than out of a `MergeTree` scan.

### Metrics {#d3-metrics}

Wall time best-of-N; summed `FillingRightJoinSide.elapsed_us` and `input_wait_elapsed_us`;
`syscalls:sys_enter_futex` and `syscalls:sys_enter_sched_yield` and `context-switches` per build
block from a server-wide window; instructions and cycles per build row for reference. The futex count
is the direct evidence: an uncontended `std::mutex` issues no syscall at all, so a futex rate that
rises with thread count and block count is the lock and nothing else.

### Expected direction {#d3-direction}

Against UHJ, growing with thread count and with the number of build blocks, and near zero at 2–4
threads with default 65 536-row blocks. The `sched_yield` counter may move in UHJ's favour: the
`try_to_lock`-and-yield loop in `insertIntoSlots` is the same strategy `ConcurrentHashJoin` uses, so
if yields dominate futexes the bottleneck is the slot locks, which is not D3.

### Confounds {#d3-confounds}

* Shrinking `max_block_size` also increases per-block overhead everywhere else in the pipeline —
  more `Chunk`s through the executor, more virtual calls, worse vectorisation. Both arms pay it,
  which is why only the arm-to-arm *difference* in the slope may be attributed to D3, never the slope
  itself.
* `preferred_block_size_bytes=0` is required or the adaptive block sizing overrides `max_block_size`
  for narrow rows and the block-count axis silently collapses.
* Divergence A also scales with `max_threads` on the UHJ arm. It does not scale with block count,
  which is exactly why the block-size axis is the estimator and the thread axis is only context.
* The read side is also parallel, so at high thread counts the measurement can become read-bound.
  `CODEC(NONE)` and a key-only build side keep that as far away as possible; `build_streams` from
  `EXPLAIN PIPELINE` is recorded per cell so a pipeline that did not actually get 16 build streams is
  visible rather than averaged in.
* At `max_threads=1` UHJ has one slot and one bucket, so that column carries divergence B. It is
  reported, not used as the normalisation base.
