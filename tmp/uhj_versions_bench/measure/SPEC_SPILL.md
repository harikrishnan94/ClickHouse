# Measuring the spilling, per-block-accounting and arena divergences

Design document for `m_D14.sh`, `m_D10.sh`, `m_D11.sh`, `m_D12.sh`, `m_D18.sh`, `m_D19.sh`,
`m_D20.sh`, `m_D23.sh` and the helper `_spill_common.sh` in this directory. The divergences
themselves are catalogued in `../artifacts/DIVERGENCE_INVENTORY.md`; this file only covers how to
put numbers on eight of them.

Repository `/mnt/ch/ClickHouse`, branch `cursor/uhj-versions-bench-4f2a`, merge-base `3218492309c`.
Nothing here has been executed. Every number that appears below is labelled as a **prediction**.

Read section 0 first: two of its findings change what the other sections can claim, and one of them
(section 0.2) contradicts an assumption in the task that produced this document.

---

## 0. Shared setup {#shared-setup}

### 0.1 Arms {#arms}

| arm | binary | what it is |
|---|---|---|
| `baseline` | `/mnt/data/uhj_versions_bench/bin/clickhouse-baseline` | merge-base `3218492309c`: `hash`, `parallel_hash`, `grace_hash`, no `unified_hash` |
| `uhj` | `/mnt/data/uhj_versions_bench/bin/clickhouse-uhj` | this branch: all four |

Most cells here follow the house rule — `hash` / `parallel_hash` / `grace_hash` from
`clickhouse-baseline`, `unified_hash` from `clickhouse-uhj` — precisely because of D14. Three
measurements deliberately break it and run **the same algorithm on both binaries**, because the
divergence under test is a branch change to the baseline arm rather than a difference between the
two join implementations:

| cell | algorithm | both binaries? | isolates |
|---|---|---|---|
| `m_D14` null control | `hash`, no spill wrapper | yes | nothing — the binary-to-binary noise floor |
| `m_D14` treatment | `parallel_hash`, no spill wrapper | yes | the seven added `ConcurrentHashJoin` timers |
| `m_D20` | `hash` + spill wrapper | yes | the added `shared_lock` on the single-in-memory-join path |
| `m_D19` | `grace_hash` | yes | the unconditional `hash_join_mutex` in `GraceHashJoin::joinBlock` |

Both binaries read the same data directory, so every script stops the server before switching arms
and takes an exclusive `flock` on `/mnt/data/uhj_versions_bench/measure/.lock` — the same lock file
`_common.sh` and `_maps_common.sh` use, so all three measurement families exclude each other.
`sp_stop` kills every process in the `uhj_versions_bench` cgroup; two scripts running at once would
destroy each other's server.

### 0.2 Spilling is on by default, and that changes the real-world answer {#spilling-by-default}

`max_bytes_before_external_join` defaults to `0`, but `max_bytes_ratio_before_external_join`
defaults to **`0.5`** (`Core/Settings.cpp:8248`), and the effective threshold is the smaller of the
two non-zero values, resolved at plan time against `getMostStrictAvailableSystemMemory()`
(`JoinOperator.cpp:301-334`). With a `tmp_path` configured — which the benchmark server config sets
— the planner therefore takes the `params.max_bytes_before_external_join > 0` branch
(`Planner/PlannerJoins.cpp:1209`) and wraps **every eligible hash join** in `SpillingHashJoin`. The
in-tree comment at `Processors/QueryPlan/Optimizations/topKThroughJoin.cpp:375` says so in as many
words: "`max_bytes_ratio_before_external_join` defaults to `0.5` and wraps every hash join in
`SpillingHashJoin`".

Two consequences.

**The premise "the suites run with `max_memory_usage=0` and are unlikely to spill at all, so the
real-world exposure of D12/D18/D19/D20 is zero" is only half right.** `max_memory_usage=0` removes
the per-query memory *limit*; it does nothing to the spill *threshold*, which comes from the ratio.
So the suites almost certainly run with the wrapper present on every join. What they very probably
do not do is *cross* the threshold (~half of the visible memory, i.e. ~16 GiB in the 32 GiB cgroup).
That splits the four spill divergences cleanly in two:

* **D12 and D20 need only the wrapper, so their real-world exposure is ~100 % of eligible suite
  joins.** D12 in particular stops being a spill-only curiosity: with the wrapper always present,
  `unified_hash`'s `supportParallelJoin() == true` is what gives every `unified_hash` suite join
  `max_threads` build streams, and it is the branch of the planner where the `&& !unified` gate
  (D1) also lives.
* **D18 and D19 need the threshold to be crossed, so their exposure is plausibly zero.** Each
  script verifies rather than assumes: `ProfileEvents['JoinSpillingHashJoinSwitchedToGraceJoin']`
  is non-zero exactly when a switch happened, and `sp_default_threshold_report` records the
  effective threshold and the join object the planner picks under untouched spill settings.

**Every query this harness issues therefore passes both settings explicitly.** `sp_cap 0` sets
`max_bytes_before_external_join=0` *and* `max_bytes_ratio_before_external_join=0`, which is the only
way to get a genuinely unwrapped join; `sp_cap N` sets the absolute threshold to exactly `N` and the
ratio to `0`, so the threshold is a number the script chose rather than a function of how much
memory the machine happened to have free. Sibling harnesses that pin only
`max_bytes_before_external_join` are running through `SpillingHashJoin` whether they intend to or
not; their "wrapper off" cells are "wrapper on with a machine-dependent threshold".

### 0.3 What every cell pins {#pinned-settings}

`sp_settings` in `_spill_common.sh` composes one array from five per-cell knobs
(`join_algorithm`, `max_threads`, `max_block_size`, the spill cap, the statistics flag) plus:

* `max_bytes_ratio_before_external_join=0` — see 0.2. The single most important line in the harness.
* `enable_join_runtime_filters=0` — a runtime filter can prune the probe side outright.
* `enable_join_fixed_hash_table_conversion=0` — post-build conversion to `PartitionedFixedHashMap`
  is divergence D4, measured elsewhere. It does **not** disable the `key8`/`key16` fixed maps that
  `chooseMethod` picks from the key type, which D10's worst case depends on.
* `query_plan_join_swap_table=false` — "build side" means what the SQL says, on both arms.
* `query_plan_convert_join_to_in=0` — a build-only query must stay a join.
* `preferred_block_size_bytes=0` — nothing adaptive behind `max_block_size`.
* `collect_hash_table_stats_during_joins=0` everywhere **except `m_D11`**, whose subject is the
  statistics path itself.

`join_algorithm` is always a **single** algorithm. That is not cosmetic: the
`parallel_hash_join_threshold` gate's first term is `!isEnabledAlgorithm(JoinAlgorithm::HASH)`
(`PlannerJoins.cpp:1244`), so a one-element list makes `parallel_hash` unconditionally parallel and
`hash` unconditionally serial, and divergence D1 cannot leak into any cell here.

### 0.4 Every shape comes from `numbers()` {#shapes}

No fixture tables. `numbers()` / `numbers_mt()` emits exactly `max_block_size` rows per block, so
the block count — the independent variable for six of these eight divergences — is known exactly
rather than inferred from a part layout, granule size or adaptive block sizing. It also removes
page-cache and decompression variance, and makes every script idempotent with nothing to create.

The price is that row generation is included in the per-row cost. That is harmless for the slope
method (0.5), and for the level comparisons the primary metric is a per-processor time that excludes
the source entirely.

Three shapes (`_spill_common.sh`):

* `sql_build_only <rows> [keyexpr]` — one probe row, so the query is a build. Per-build-block
  effects are undiluted. Selects `max(r.k)` rather than `count()` so that no plan rewrite can
  decide the join is redundant and drop the build side.
* `sql_probe_heavy <build_rows> <probe_rows>` — small build, large probe, 1:1 match. The probe block
  count is `probe_rows / max_block_size`.
* `sql_str_build <rows> <width>` — a `String` key, one `keyHolderPersistKey` arena copy per distinct
  key. The only shape whose footprint is dominated by the arenas (D23).

`sp_check_shape` validates each one once per pass: the `EXPLAIN` `Algorithm` field must name the
expected join object and `JoinBuildTableRowCount` must equal the row count the SQL asked for. A cell
whose build side had been optimised away would otherwise report a beautifully stable number that
measures nothing.

### 0.5 Method: the per-block slope {#slope-method}

Every divergence in this group except D11, D18 and D23 is a **per-block** cost, while nearly every
other difference between the arms — the `scatterBlockBySlot` pass of divergence A above all — is
**per-row**. So: hold the row count fixed, shrink `max_block_size`, and the number of per-block
events multiplies while the per-row work stays exactly the same. The **slope** of time against block
count is the per-block cost; the intercept absorbs everything per-row, including A and B.

`sp_slope` fits it by least squares over four block sizes (`65505, 8192, 2048, 512`) instead of
differencing two points, so the residual also says whether the relationship is linear — which is the
check that the design is sound rather than an artefact.

For the two cells that run the same algorithm on both binaries, the slope is differenced twice
(difference-in-differences):

```
D14  =  [ slope(parallel_hash, uhj-binary) - slope(parallel_hash, baseline-binary) ]
      - [ slope(hash,          uhj-binary) - slope(hash,          baseline-binary) ]
```

The second bracket is the null control. It cancels everything that differs between the two binaries
but is not on the `ConcurrentHashJoin` path: code layout and i-cache placement (the branch binary is
335 MB larger), the 453-line `Common/HashTable/TwoLevelHashTable.h` rewrite, the
`NUM_BUCKETS` → `numBuckets()` conversions in `HashTable.h` / `HashSet.h` / `StringHashTable.h` /
`Aggregator.cpp`, and the `IJoin` → `IInMemoryHashJoin` base-class insertion. None of those is
per-block, so the null control's slope difference should be ~0; if it is not, that number is the
error bar on D14 and on D20.

### 0.6 Metrics {#metrics}

| metric | source | why |
|---|---|---|
| `best_sec` | `client --time`, best of `REPEATS` | the headline, minimum over repeats to suppress scheduler noise |
| `build_us` | `sum(elapsed_us)` over `FillingRightJoinSide` in `system.processors_profile_log` | build-transform time only: excludes the `numbers()` source, and is a CPU-time-like quantity that does not shrink when the work is spread over more threads |
| `probe_us` | same, over `JoiningTransform` | the probe equivalent |
| `nonjoined_us`, `delayed_us` | same, over `NonJoinedBlocksTransform` / `DelayedJoinedBlocksWorkerTransform` | the RIGHT/FULL tail and the grace delayed-blocks phase |
| `build_streams`, `probe_streams` | processor counts, and `EXPLAIN PIPELINE` | D12's whole content is a stream count |
| `mem_bytes` | `system.query_log.memory_usage` | D23; the MemoryTracker peak, independent of the join's own (D2-broken) accounting |
| `instructions`, `cycles`, `cache_misses`, `dtlb_misses` | per-query `Perf*` events | six events, no PMU multiplexing; zeros if `perf_event_paranoid` forbids self-monitoring, which `sp_perfev_ok` probes for |
| `chj_*_us` | `ConcurrentHashJoinBuild/Probe*Microseconds` | **only exists on the branch binary** — the D14 timers reporting on themselves |
| `prealloc` | `HashJoinPreallocatedElementsInHashTables` | D11: what was actually reserved |
| `spilled` | `JoinSpillingHashJoinSwitchedToGraceJoin` | D18/D19: did the switch happen at all |
| `ext_parts`, `ext_comp_bytes`, `ext_uncomp_bytes` | `ExternalJoinWritePart`, `ExternalJoin*Bytes` | how much grace I/O each arm did, i.e. whether two grace runs are comparable |

One flush and one dump per pass rather than per query: `SYSTEM FLUSH LOGS` costs more than most
cells here. Wall time is echoed live; the counters are joined onto the grid at the end by `sp_join`.

`m_D14` adds a fourth instrument: a compiled C micro-benchmark of one
`ProfileEventTimeIncrement<Microseconds>` scope (section 1.4). It is the only one that gives the
per-timer cost directly instead of by differencing two noisy wall times.

### 0.7 Running order {#running}

`m_D14.sh` **first**, and read its verdict before trusting any other `parallel_hash` number in this
campaign. The rest are independent. Each takes the lock, so run them one at a time:

```bash
cd /mnt/ch/ClickHouse/tmp/uhj_versions_bench/measure
./m_D14.sh            # then D10, D11, D12, D18, D19, D20, D23
```

Common environment knobs: `ARM=baseline|uhj` (default both), `REPEATS`, `ROWS`, `PROBE_ROWS`,
`SKIP_SUITES=1`, `Q_TIMEOUT`. Results in `/mnt/data/uhj_versions_bench/measure/<id>/`, with an
`M_<id>_DONE` sentinel on success. Re-running a script replaces the rows of the cells it
re-measures (`sp_tsv_prune` keys on the trailing `tag` column) rather than appending a second copy.

---

## 1. D14 — the seven `ConcurrentHashJoin` timers {#d14}

### 1.1 Mechanism {#d14-mechanism}

The branch added seven `ProfileEvents` and nine `ProfileEventTimeIncrement<Microseconds>` scopes to
`ConcurrentHashJoin.cpp`. `git diff 3218492309c -- src/Common/ProfileEvents.cpp` is exactly those
seven event declarations and nothing else; `grep -c ProfileEventTimeIncrement` is `9` in
`ConcurrentHashJoin.cpp` and **`0`** in both `src/Interpreters/HashJoin/` and
`src/Interpreters/UnifiedHashJoin/`. So the instrumentation exists on exactly one of the two
algorithms being compared, and it is the baseline one.

Per-block counts, which is what matters:

| path | site | scopes per event |
|---|---|---|
| build | `addBlockToJoin` entry (`:300`), dispatch (`:308`), insert loop (`:324`) | **3 per build block** |
| probe | `joinBlock` entry (`:452`), probe dispatch (`:456`) | **2 per probe block** |
| probe | `ConcurrentHashJoinResult::next()` (`:420`), its lookup scope (`:431`) | **1 per output block, plus 1 per dispatched block** |
| build, once | `onBuildPhaseFinish` (`:797-798`) | 2 per query per instance |

So ~3 scopes per build block and ~3–4 per probe block. Each scope is a
`Stopwatch(CLOCK_MONOTONIC)` constructed and stopped — two vDSO `clock_gettime` calls — plus one
`ProfileEvents::increment`, which is a `sched_getcpu` (an rseq TLS read on Linux,
`Common/PerCPU.h:26-30`, not a syscall) and one relaxed `fetch_add` per level of the counter chain
(thread, thread group, global), each into a per-CPU row so there is no cross-thread line sharing.

Worth recording as a second-order point: `elapsedMicroseconds()` truncates, so a scope shorter than
a microsecond reports **0** while still costing its two clock reads. On small blocks these timers
therefore pay their full price and report nothing usable — they are not only overhead, they are
overhead in exchange for a systematically low reading.

### 1.2 Worst case {#d14-worst-case}

Maximise blocks per second of join work: the cheapest possible join, the smallest sensible block,
and one thread so that the bias is not divided by the thread count before it reaches wall time.

```sql
-- 10 M probe rows against a 1000-row build side: the map is ~40 KB, i.e. L1-resident, so the
-- per-row probe cost is as small as a hash join gets and the per-block constant dominates.
SELECT count()
FROM (SELECT number % 1000 AS k FROM numbers_mt(10000000)) AS p
INNER JOIN (SELECT number AS k FROM numbers(1000)) AS r ON p.k = r.k
```

Settings: `join_algorithm=parallel_hash`, `max_threads=1`, `max_block_size=256`,
`max_bytes_before_external_join=0`, `max_bytes_ratio_before_external_join=0`. Run on **both**
binaries. 39 063 probe blocks, 4 build blocks.

Why this is the worst case: the bias is `blocks × scopes × ns`, and `blocks` is maximised by the
smallest block size while `ns` is fixed, so the *fraction* is maximised when the per-block work is
minimised — which is a tiny L1-resident map and no payload columns. `max_threads=1` converts the
whole bias into wall time; the same cell at `max_threads=16` shows the wall-time fraction after
16-way division, which is the number a threaded benchmark would see.

### 1.3 Average case {#d14-average-case}

`max_block_size` ∈ {65505, 8192, 2048, 512} × `max_threads` ∈ {1, 4, 16} × two shapes (build-heavy
8 M rows with a one-row probe; probe-heavy 10 M rows against 1 000), on both binaries, for both
`hash` (control) and `parallel_hash` (treatment). 65505 is the default, so the leftmost column of
every table is the magnitude that matters for anything anyone actually ran.

### 1.4 Direct calibration {#d14-calibration}

`sp_timer_cost` compiles and runs a C program that does exactly what one scope does — two
`clock_gettime(CLOCK_MONOTONIC)` calls, a `sched_getcpu`, three relaxed `fetch_add`s — a few million
times inside the same cgroup, and reports nanoseconds per scope and, separately, nanoseconds per
bare clock pair. Multiplying by the exactly-known block count gives a *predicted* bias per query
that is independent of the wall-clock differencing, and the two estimates cross-check each other. If
they disagree by more than ~2× the null control's slope, something else differs between the binaries
on the `parallel_hash` path and the DiD number should not be trusted.

### 1.5 Real world {#d14-real-world}

The four loaded suites at the default `max_block_size=65505`, `max_threads=16`, on the branch
binary with `parallel_hash`. Per query, the predicted bias is
`(build_rows/65505 × 3 + probe_rows/65505 × 3.5) × ns_per_timer`, compared against
`query_duration_ms`. Also read back the branch-only `chj_*_us` events, which give the same block
counts a second way. `SKIP_SUITES=1` skips this pass; it is the only expensive one.

### 1.6 Expected direction and magnitude (prediction) {#d14-prediction}

The branch binary is **slower** on `parallel_hash` — that is, the bias flatters `unified_hash`.

* Per scope: **50–80 ns** (two vDSO clock reads at ~20–25 ns each, plus ~15 ns of increment).
* Per build block ~3 scopes: **150–240 ns**. Per probe block ~3.5 scopes: **175–280 ns**.
* At the default `max_block_size=65505`: a 240 M-row probe is 3 663 blocks, so **under 1 ms per
  query**, i.e. **< 0.05 %** of anything that takes a second. Invisible.
* At `max_block_size=512` on a cheap join at `max_threads=1`: 19 531 probe blocks × ~225 ns ≈
  **4.4 ms**, against a query that should take 60–150 ms — so **3–7 %**.
* At `max_block_size=256`, `max_threads=1`, the worst case above: ~39 000 blocks × ~225 ns ≈
  **8.8 ms** on a query in the 100–250 ms range, so **4–9 %**.

**Verdict (prediction): it does not invalidate the published comparisons, but it does invalidate a
specific class of them.** Any comparison at default or near-default `max_block_size` is biased by
less than a tenth of a percent, which is far below the run-to-run noise of the suites. Any
comparison that (a) shrinks `max_block_size`, (b) normalises by block rather than by row, or (c)
reports a per-block constant — which is exactly what several measurements in this campaign do — is
biased in `unified_hash`'s favour by a per-block constant of ~150–280 ns and must subtract it.
Measurements taken through `_common.sh` / `_maps_common.sh` / `_spill_common.sh` are immune by
construction, because those run `parallel_hash` on the merge-base binary; the exposure is to any
earlier A/B that took both arms from a single branch build.

### 1.7 Confounds and inseparability {#d14-confounds}

* The two binaries differ in more than the timers (0.5). The null control bounds that; if the
  control's own slope difference is comparable to the treatment's, D14 is not resolvable by timing
  and only the calibration number stands.
* The spill wrapper must be off in both control and treatment, or D20 contaminates the control.
* The timers on `onBuildPhaseFinish` are per query, not per block, and are therefore in the
  intercept, not the slope. They are not measured separately; ~2 scopes per query is ~0.1 µs.
* D14 cannot be separated from D20 in a cell that has the wrapper on, since both are per-build-block
  costs added to the baseline arm by this branch. That is why `m_D14` runs with `sp_cap 0` and
  `m_D20` with the cap on, using the same control.

---

## 2. D10 — per-block byte accounting {#d10}

### 2.1 Mechanism {#d10-mechanism}

UHJ's `insertIntoSlots` brackets each slot's insert with `slot_bytes(slot)`
(`UnifiedHashJoin/HashJoin.cpp:130-136`, called at `:140` and `:166`), and `slot_bytes` loops over
that slot's buckets calling `map.getBucketBufferSizeInBytes(type, bucket)` — a `switch` over 31 map
variants (`UnifiedHashJoin/HashJoin.h:551-560`), so a jump-table dispatch per bucket, not an inlined
load. `num_buckets` is `map.getBucketCount(data->type)` (`:1033`), which is `bucketCount()` of the
map type.

Summing over the slots that a block actually touches, the per-build-block totals are:

| map | `bucketCount()` | UHJ dispatched calls per block (all slots) | baseline per block |
|---|---|---|---|
| serial two-level (`BITS_FOR_BUCKET_SERIAL = 0`) | 1 | **2** | serial `hash`: 1 `getTotalRowCount()`, O(1) |
| parallel two-level, `slots` slots | 256 | **512** | `parallel_hash`: `slots × 2 × 256` bucket iterations |
| fixed (`key8`/`key16`/`range*`) | 256 | **512**, every one returning the same value | `parallel_hash`: `slots × 2`, O(1); serial `hash`: 1, O(1) |

Note the asymmetry in the middle row: UHJ's total is `2 × 256` *regardless of slot count*, because
each slot walks `256/slots` buckets, while `ConcurrentHashJoin` calls
`updateTotalRowsAndBytesUnlocked` once per slot per block and each call walks all 256 buckets of
that slot's map (`ConcurrentHashJoin.cpp:753-759`, `HashJoin/HashJoin.cpp:491-504` and `533-557`).
So **for two-level maps UHJ is cheaper by a factor of `slots`, and for fixed maps it is more
expensive by a factor of ~256/`slots`.** The row-count half of the change (an explicitly propagated
`new_keys`, `UnifiedHashJoin/HashJoin.cpp:1041`) is a clear UHJ win everywhere.

The fixed-map case is the pathological one: `PartitionedFixedHashMap` keeps one flat buffer whose
buckets all return the same `getBufferSizeInBytes()` (`Common/HashTable/TwoLevelHashTable.h:215-216`),
and that buffer never grows — so UHJ does 512 dispatched calls per block to compute a delta that is
always exactly zero.

### 2.2 Worst case {#d10-worst-case}

A `UInt8` key, so `chooseMethod` picks `key8`, and **`max_threads=1`**, which is what makes this
separable from D3:

```sql
-- 8 M build rows, 256 distinct UInt8 keys -> key8, i.e. PartitionedFixedHashMap with 256 buckets
-- and one slot. 512 switch-dispatched getBucketBufferSizeInBytes calls per build block on UHJ;
-- one O(1) call per block on serial `hash`.
SELECT max(r.k)
FROM (SELECT 0 AS k) AS p
LEFT JOIN (SELECT toUInt8(number) AS k FROM numbers_mt(8000000)) AS r ON p.k = r.k
```

`max_block_size` ∈ {65505, 8192, 2048, 512}; `hash` on `baseline`, `unified_hash` on `uhj`. At
`max_block_size=512` that is 15 625 build blocks × 512 dispatched calls = **8 M calls** for a query
that inserts 8 M rows into a 256-cell table.

Why worst: the per-block constant is maximal (256 buckets, one slot, so one slot walks all of them),
the per-row work is minimal (a direct-addressed store into an L1-resident table), and at
`max_threads=1` there is no scatter (divergence A is `slots > 1` only) and no lock contention.

### 2.3 Separating D10 from D3 {#d10-vs-d3}

D3 (the global `blocks_mutex`, measured by a sibling) is also a per-build-block cost on the UHJ
build path, and it has the same worst-case shape. One lever separates them: **D3's acquisitions are
independent of the key type; D10's cost is a function of the map's `bucketCount()`.** So run the
same block-size sweep with two key types at `max_threads=1`:

| cell | UHJ map | `bucketCount()` | D10 per block | D3 per block |
|---|---|---|---|---|
| `UInt64` key | serial two-level | 1 | 2 calls (≈ nil) | 2 uncontended acquisitions |
| `UInt8` key | `PartitionedFixedHashMap` | 256 | 512 calls | 2 uncontended acquisitions |

`slope(u8) − slope(u64)` is D10's fixed-map cost with D3 subtracted; `slope(u64)` alone bounds D3's
uncontended component. `m_D10.sh` reports both, and a `UInt16` cell (`key16`, also 256 buckets)
confirms the number is a property of the bucket count and not of the key width.

At `max_threads > 1` the two are **not** separable: both are per-block, both are on the same code
path, and A's per-row scatter joins the intercept. Every `max_threads > 1` row in `m_D10`'s output
is therefore labelled as bounding **D3 + D10 together**. The thread sweep is still worth running for
its own reason: the two-level shape should show baseline's accounting cost *growing* with slot count
while UHJ's stays flat, which is D10's one clear win and is not visible at `max_threads=1`.

### 2.4 Average case {#d10-average-case}

`max_block_size` ∈ {65505, 8192, 2048, 512} × `max_threads` ∈ {1, 2, 4, 8, 16} × key type ∈
{`UInt64`, `UInt8`, `UInt16`}, 8 M build rows, build-only.

### 2.5 Real world {#d10-real-world}

The fixed-map worst case needs a 1- or 2-byte join key (`UInt8`, `UInt16`, `Enum8`, `Enum16`). JOB
joins on `Int32` ids, TPC-H and TPC-DS on `Int32`/`Int64` keys; a 1-byte join key is unusual in all
four suites. `m_D10.sh` reports the map each suite query chose (`sp_maptype`, the `LOG_TEST`
"datatype:" line) so the claim is checked rather than asserted. Expected finding: **zero exposure to
the pathological case, and a small UHJ win everywhere else** from the two-level and row-count
halves.

### 2.6 Expected direction and magnitude (prediction) {#d10-prediction}

* `UInt8`/`UInt16` key: UHJ **slower** by 512 dispatched calls per build block. A jump-table
  dispatch plus a load and add is ~3–5 ns, so **1.5–2.5 µs per build block** — enormous relative to
  a per-block cost, and at `max_block_size=512` roughly **3–5 ns per build row**, i.e. tens of
  percent of a `key8` build. This should be the largest single per-block effect in this whole group.
* `UInt64` key at `max_threads=1`: **~nil** (2 calls per block).
* `UInt64` key at `max_threads=16`: UHJ **faster**, by `slots × 512 − 512` ≈ 7 700 bucket iterations
  per block; baseline's are inlined rather than dispatched, so at ~1 ns each that is **~7 µs per
  build block** of baseline cost that UHJ does not pay. This is a real and underrated `parallel_hash`
  inefficiency, and it will partly mask D3 in any two-level comparison.
* Fixed maps at `max_threads=16`: UHJ slower by 512 dispatched calls against baseline's ~32 O(1)
  calls.

---

## 3. D11 — statistics-driven reserve {#d11}

### 3.1 Mechanism, with two corrections to the inventory {#d11-mechanism}

UHJ removed the reserve argument from `MapsTemplate::create` (`UnifiedHashJoin/HashJoin.h:464-479`)
and moved it to a lazy per-slot `reserveSlot` called on that slot's first insert
(`HashJoin.cpp:142-148`), with the hint from `sizeHintForMaps()` (`:589-605`) and the cap from
`table_join->maxBytesBeforeExternalJoin()` (`:435`), applied by `clampReserve`
(`HashJoin.h:449-462`) as `min(reserve, cap / (8 × sizeof(cell_type)))`.

Reading the baseline side changes two of the inventory's three claims:

1. **"The reserve is deferred to the first block" is parity, not divergence.** `ConcurrentHashJoin`
   also reserves lazily, inside the slot lock on the first insert into that slot, guarded by
   `space_was_preallocated` (`ConcurrentHashJoin.cpp:341-347`). Both arms therefore build their
   first block into an unreserved map.
2. **"Baseline is unclamped" is true only off the wrapper path.** `reserveSpaceInHashMaps` clamps by
   `external_join_threshold` with the same `reserve / (8 × cell_size)` formula
   (`ConcurrentHashJoin.cpp:137-158`), and `SpillingHashJoin` passes its cap in as that argument. So
   with the wrapper present both arms clamp identically. The divergence survives where UHJ reads
   `table_join->maxBytesBeforeExternalJoin()` but no wrapper exists, i.e. when
   `GraceHashJoin::isSupported` is false — ASOF strictness or a multi-disjunct `ON` — and inside
   `GraceHashJoin`, where the per-bucket `reserve_num` (`GraceHashJoin.cpp:739-750`) is clamped by a
   cap that is always non-zero.

Both arms do participate in the statistics cache: UHJ reads it at `HashJoin.cpp:601` and writes it
at `:1291`, `ConcurrentHashJoin` at `:122` and `:262`. Two live preconditions for the UHJ side:
`collect_hash_table_stats_during_joins=1` (default) and `num_slots > 1` — `sizeHintForMaps` returns
`0` when `num_slots <= 1` unless an explicit `reserve_num` was passed, so **serial `unified_hash`
never uses statistics at all**.

### 3.2 Why this one may not use the `stats=0` control {#d11-no-control}

Every other cell in this family pins `collect_hash_table_stats_during_joins=0`, which makes the two
arms plan identically. Here the setting *is* the subject, so it is `1`, and the equal-plan control is
unavailable. Two things replace it:

* **Cold vs warm within one arm.** The statistics cache is process-global and in-memory, so it is
  empty at server start. Run 1 of a shape gets no hint; runs 2 and 3 do. `prealloc`
  (`HashJoinPreallocatedElementsInHashTables`) is `0` on run 1 and non-zero afterwards, which both
  proves the transition happened and says what was reserved. The interesting quantity is the *warm
  uplift* `run1 / median(run2, run3)` per arm, which is a within-arm ratio and immune to any
  binary-level difference.
* **A server restart before every cell**, so "cold" is guaranteed rather than assumed. The cache key
  is the right-hand-side plan hash (`JoinStepLogical::getRightHashTableCacheKey`), so varying a
  literal would probably do, but a restart costs seconds and removes the doubt.

`prealloc` is comparable across arms in aggregate: baseline increments
`actual_reserve_size / slots` once per slot and UHJ increments `clamped / slots` once per slot, so
both sum to approximately the whole reserve.

### 3.3 Cells {#d11-cells}

**A — the statistics path (primary).** 8 M distinct `UInt64` build keys, build-only,
`max_threads=16`, `max_block_size=65505`, `sp_cap 0`, `stats=1`, three consecutive runs from a fresh
server. `parallel_hash` on `baseline`, `unified_hash` on `uhj`. Report `prealloc` and `build_us` per
run. This answers "does the hint arrive, and is it worth the same on both arms".

**B — the clamp.** The same shape with `sp_cap` ∈ {0, 4 GiB, 512 MiB, 64 MiB}, chosen so the join
never actually spills at 4 GiB but the clamp `cap / (8 × cell)` bites well below the 8 M-element
hint at the smaller caps. Prediction, from the code: **both arms clamp, and clamp to the same
value.** Any difference is either a `cell_type` mismatch between the two map families or the audit's
open question 7 (baseline reading `actual_reserve_size` outside the per-map lambda) showing up as a
stale value.

**C — the no-wrapper clamp (diagnostic, not timed).** `ON a = b OR c = d` with `sp_cap` set: grace
is unsupported for multiple disjuncts, so no `SpillingHashJoin` is created, so baseline's
`ConcurrentHashJoin` gets `external_join_threshold = 0` and does not clamp — but a multi-disjunct
join is also excluded from `parallel_hash` by `allowParallelHashJoin`, so baseline is serial `hash`
and reserves nothing at all, while `unified_hash` goes parallel and reserves a clamped amount. The
cell records `EXPLAIN`'s `Algorithm`, `prealloc` and the time on each arm, and states plainly that
the difference is **D11 entangled with D1** and cannot be attributed to either alone.

**D — the grace per-bucket `reserve_num` clamp.** Measured in `m_D18` (section 5), because it needs
a grace join. Cross-arm comparison of `prealloc` is impossible there: baseline's serial
`create(type, reserve)` does not increment the event at all. Only UHJ's own cap-sensitivity is
measurable.

**E — `StatsCollectingParams` propagated through `clone()`.** Not reachable from SQL on these
shapes; recorded as unmeasured.

### 3.4 Expected direction and magnitude (prediction) {#d11-prediction}

* Warm uplift, both arms: **5–20 %** off build time for a multi-million-key two-level build, from
  the rehashes the reserve avoids. Roughly equal on the two arms — this is a parity check, and the
  interesting outcome would be UHJ's uplift being *smaller*, which would point at
  `reserveSlot`'s `clamped / Table::bucketCount()` per-bucket division rounding the reserve down
  more aggressively than baseline's `actual_reserve_size / map.numBuckets()`.
* Cell B: **no difference** between the arms. Prediction stated so it can be falsified.
* Cell C: `unified_hash` reserves where baseline reserves nothing, and is parallel where baseline is
  serial. Direction depends entirely on D1, not on D11.
* Memory: a hinted run holds a larger buffer earlier; `mem_bytes` should rise by roughly the reserved
  cell count × cell size on both arms.

---

## 4. D12 — `SpillingHashJoin` reports parallel support for one UHJ instance {#d12}

### 4.1 Mechanism {#d12-mechanism}

`supportParallelJoin()` went from `concurrent_join != nullptr` (`SpillingHashJoin.h:89`, merge-base)
to `concurrent_join ? true : in_memory_hash_join->supportParallelJoin()` (`:97`), and
`UnifiedHashJoin::supportParallelJoin()` is hard-coded `true`. `SpillingHashJoin`'s single-in-memory
-join constructor now also forwards `max_threads_` into `createInMemoryHashJoin`
(`SpillingHashJoin.cpp:51-59`), and `createInMemoryHashJoin` passes it to `UnifiedHashJoin` while
**ignoring it for `HashJoin`**, which it always builds with `use_two_level_maps = false`
(`InMemoryHashJoin.cpp:21-38`). So, with the wrapper present:

| algorithm | join object | build streams |
|---|---|---|
| `hash` | `SpillingHashJoin(HashJoin)` | **1** |
| `parallel_hash` | `SpillingHashJoin(ConcurrentHashJoin)` | `max_threads` |
| `unified_hash` | `SpillingHashJoin(UnifiedHashJoin)` | `max_threads` |

Per 0.2 the wrapper is present by default, so this is not a spill-only regime — it is the normal
one.

### 4.2 What is and is not measurable {#d12-measurable}

D12 has **no within-arm control**: there is no setting that makes `UnifiedHashJoin` report
`supportParallelJoin() == false`, so the N-thread feed cannot be switched off without a rebuild.
What can be measured is the consequence, against the two baseline shapes it sits between:

1. `EXPLAIN PIPELINE` stream counts for all three rows of the table above at
   `max_threads` ∈ {1, 4, 16} — a structural fact, exactly reproducible, zero measurement error.
   This is the deliverable that actually pins D12 down.
2. Build throughput for the same three, `max_threads` ∈ {1, 2, 4, 8, 16}, 8 M build rows,
   `sp_cap` = 64 GiB (wrapper present, never trips). `build_us / build_rows` and wall time.

The comparison against `SpillingHashJoin(HashJoin)` answers "is feeding one instance from N threads
better than feeding it from one", and the comparison against `SpillingHashJoin(ConcurrentHashJoin)`
answers "is one shared instance better than N sharded ones".

### 4.3 Worst case {#d12-worst-case}

Build-heavy at high thread count, which is where feeding one instance from many threads either wins
(real parallel build) or loses (the internal `blocks_mutex` and `BucketLock`s): 8 M-row build-only
at `max_threads=16` with `max_block_size=512`, so there are 15 625 blocks to contend over instead of
122. Small blocks are the worst case because UHJ's serialisation is per block, not per row.

### 4.4 Real world {#d12-real-world}

~100 % of eligible suite joins are wrapped (0.2), so D12 is live everywhere — but it is not
separable from D1 there. D1 is the planner-level `&& !unified` gate that sends `unified_hash` past
the `parallel_hash_join_threshold` check (`PlannerJoins.cpp:1211` on the spilling branch); D12 is the
pipeline-level mechanism that then gives it `max_threads` streams. For a suite query whose right
side is estimated below 100 000 rows, baseline runs `SpillingHashJoin(HashJoin)` with one build
stream and `unified_hash` runs `SpillingHashJoin(UnifiedHashJoin)` with sixteen. Attributing that to
D12 or to D1 is meaningless; `m_D12.sh` reports it as a joint effect and points at `m_D1.sh`.

### 4.5 Expected direction and magnitude (prediction) {#d12-prediction}

* Against `SpillingHashJoin(HashJoin)` (1 stream): UHJ **faster**, 2–6× on build-dominated shapes at
  `max_threads=16` — a genuinely parallel build against a serial one, discounted by UHJ's internal
  serialisation.
* Against `SpillingHashJoin(ConcurrentHashJoin)` (N streams): **within ±20 %**, with UHJ falling
  behind as blocks shrink and thread count rises, because D3's two mandatory global critical
  sections per block do not scale while `ConcurrentHashJoin`'s per-shard `try_to_lock` does. Expect
  the crossover somewhere between 8 and 32 threads at `max_block_size=512`.
* Not separable from: D1 (in the suites), D3 and A (in the synthetic cells — they are what the
  N-thread feed runs into).

---

## 5. D18 — grace's in-memory joins are pinned to `max_threads = 1` {#d18}

### 5.1 Mechanism {#d18-mechanism}

`GraceHashJoin::makeInMemoryJoin` passes `/*max_threads=*/1` (`GraceHashJoin.cpp:739-750`). For
`InMemoryHashJoinKind::Hash` that argument is ignored anyway, so the baseline behaviour is
unchanged; for `Unified` it means every grace bucket is a `UnifiedHashJoin` with `num_slots = 1`,
which by `useTwoLevelMaps(max_threads)` (`UnifiedHashJoin/HashJoin.h:53-56`) is the 1-bucket serial
map — regime B — for a workload that is by definition too big for memory.

Note what this does **not** mean. Baseline's grace buckets are `HashJoin` with
`use_two_level_maps = false`, i.e. also a flat map, and per the inventory's own clarification of
divergence B the UHJ serial map is structurally identical to it. So D18 is not "UHJ gets a worse map
than baseline under grace" — it is "**UHJ can never get its parallel map under grace**", which makes
grace the one regime where all of UHJ's parallel machinery is dead code and the comparison reduces
to serial UHJ versus serial `HashJoin`.

That in turn means **D18 is a regime selector, not a cost**. It is not separable from B (the serial
map), D10 (which at 1 slot is 2 calls per block for the two-level map but 512 for a fixed map), D21
(non-joined streams for a 1-bucket map) or the batched probe rewrite. `m_D18.sh` measures the
aggregate and says so.

### 5.2 Design {#d18-design}

Force grace through the wrapper on both arms, so the switch logic, the thresholds, the initial
bucket count and the rehash policy are all identical and the only difference is the bucket join:

| arm | settings | ends up as |
|---|---|---|
| `baseline` | `join_algorithm=hash`, `sp_cap 64 MiB` | `GraceHashJoin` with `HashJoin` buckets |
| `baseline` | `join_algorithm=parallel_hash`, `sp_cap 64 MiB` | `GraceHashJoin` with `HashJoin` buckets, converted from `ConcurrentHashJoin` |
| `uhj` | `join_algorithm=unified_hash`, `sp_cap 64 MiB` | `GraceHashJoin` with `UnifiedHashJoin` buckets, `max_threads = 1` |

Shape: 8 M distinct `UInt64` build keys against 8 M probe rows, so both phases are real. A 64 MiB
cap against a map of ~8 M × ~40 B guarantees the switch. `join_algorithm=grace_hash` is
deliberately *not* used for the baseline arm: it constructs `GraceHashJoin` directly with the
default `in_memory_kind = Hash` and `external_join_threshold = 0`, so its rehash decisions come from
`max_bytes_in_join` instead of the cap (`GraceHashJoin.cpp:326-352`) and the two arms would not be
bucketed alike. It is used in `m_D19`, where that is the point.

Verifications the script performs, because a grace measurement that did not spill is worthless:
`spilled` (`JoinSpillingHashJoinSwitchedToGraceJoin`) must be ≥ 1 on every arm; `ext_parts` and
`ext_uncomp_bytes` must be within ~2× across arms, or the arms bucketed differently and the times
are not comparable; and `sp_maptype` must report `key64` (not `two_level_key64`) on the UHJ arm,
which is D18 itself made visible.

### 5.3 The within-arm signature {#d18-signature}

D18 has one clean within-arm test, and it needs no baseline at all. Sweep `max_threads` ∈
{1, 2, 4, 8, 16} on the `uhj` arm twice — once with `sp_cap 64 MiB` (grace) and once with
`sp_cap 0` (in-memory):

* in-memory: build time falls with `max_threads`, because `num_slots = bit_ceil(max_threads)`;
* grace: build time is **flat** in `max_threads` as far as the bucket join is concerned, because
  `makeInMemoryJoin` hard-codes 1.

A flat grace curve next to a falling in-memory curve is D18, measured, with no cross-binary
comparison to defend.

### 5.4 Real world {#d18-real-world}

Needs the threshold crossed. The threshold under default settings is ~half of the visible memory,
~16 GiB in the 32 GiB cgroup; `sp_default_threshold_report` records the actual figure from the
`JoinSettings` TRACE line. The largest suite build side is `coffeeshop.fact_sales` at 500 M rows,
but no suite query builds a hash table on all of it. `m_D18.sh` reports the effective threshold
against the largest `memory_usage` any suite query reached, so the margin is a number rather than an
assertion. Expected: **zero real-world exposure**, and if so, D18 is a correctness/latency note
about a regime nobody in this benchmark reaches.

### 5.5 Expected direction and magnitude (prediction) {#d18-prediction}

* UHJ grace vs baseline grace: **within ±15 %**, dominated by grace's own I/O
  (`ExternalJoin*Bytes`), which is identical machinery on both arms. Any difference is the serial
  UHJ-vs-`HashJoin` gap (regime B plus the probe rewrite) diluted by the spill traffic.
* Grace vs in-memory on the same arm: grace **3–10× slower**, per bucket write-read-rehash cycle.
  Not a divergence, just context for how much the ±15 % is worth.
* The `max_threads` flatness signature: strongly expected to hold. If UHJ's grace build time *does*
  fall with `max_threads`, `makeInMemoryJoin`'s hard-coded 1 is not reaching the bucket join and the
  inventory entry is wrong.

---

## 6. D19 — `GraceHashJoin::joinBlock` locks unconditionally {#d19}

### 6.1 Mechanism, and why the delta is nearly nil {#d19-mechanism}

```
-    if (hash_join && getNumBuckets() <= 1)
+    if (getNumBuckets() <= 1)
     {
         std::lock_guard lock(hash_join_mutex);
-        hash_join->runPostBuildPhase();
+        if (hash_join)
+            hash_join->runPostBuildPhase();
     }
```

The merge base **already** took `hash_join_mutex` on every probe block whenever
`getNumBuckets() <= 1` and `hash_join` was non-null — which is the normal case, since
`grace_hash_join_initial_buckets` defaults to `1` (`Core/Settings.cpp:8314`). The branch moved the
null check inside the lock, closing a genuine race (`hash_join` can be reset by a concurrent
rehash). **The only behavioural difference is in the window where `hash_join == nullptr`**, and no
SQL-level workload reaches that window deterministically.

So D19's branch-vs-merge-base delta is expected to be **zero**, and the honest deliverable is a
bound plus a verification, not a measurement of a difference.

### 6.2 A larger, unlisted divergence on the same line {#d19-unlisted}

While bounding D19 it is worth measuring what that per-probe-block critical section actually costs,
because for `unified_hash` it costs **more than it does for baseline**, and the inventory does not
record this. `runPostBuildPhase` is called once per probe block under the global mutex, and UHJ's
override does two things baseline's does not (`UnifiedHashJoin/HashJoin.cpp:2505-2514` vs
`HashJoin/HashJoin.cpp:2408-2413`):

```cpp
    recomputeBucketBytes();     // UHJ only
    freezeMapsForProbing();     // UHJ only
```

For a 1-bucket serial map both are O(1)-ish — `poolsAllocatedBytes()` over one arena, one
`getBufferSizeInBytes()`, and a prefix-sum pass over one bucket — so the *predicted* cost is small.
But it is inside a mutex taken by every probe thread on every probe block, so at `max_threads=16`
with small blocks it is a serialisation point that baseline does not have, and it is worth a number.
This is measurable on the `uhj` arm only (baseline has no `unified_hash`), so it is reported as a
level, not a delta.

### 6.3 Design {#d19-design}

**Cell A — the D19 delta.** `join_algorithm=grace_hash` on **both binaries** (it uses `HashJoin`
buckets on both, so this is a clean branch-vs-merge-base isolation), `grace_hash_join_initial_buckets=1`,
a build side small enough that no rehash occurs so `getNumBuckets()` stays at 1, 10 M probe rows,
`max_block_size` ∈ {65505, 8192, 2048, 512}, `max_threads=16`. The slope difference against probe
block count is D19. Predicted: **0 ± the null control**.

**Cell B — the absolute level.** Same sweep, one binary: the slope itself is the per-probe-block cost
of the whole `getNumBuckets() <= 1` prologue including the lock. That is the ceiling on what D19
could ever cost if the null-`hash_join` window were always taken.

**Cell C — the UHJ prologue.** `unified_hash` + a cap that trips, so grace is reached with UHJ
buckets, same probe-block sweep. `probe_us` per probe block versus the baseline grace arm's is the
`recomputeBucketBytes` + `freezeMapsForProbing` addition, plus everything else that differs about a
UHJ bucket.

### 6.4 Expected direction and magnitude (prediction) {#d19-prediction}

* Cell A: **0**, within noise. An uncontended `std::mutex` is two atomics, ~20 ns, and both binaries
  pay it.
* Cell B: **50–300 ns per probe block** for the prologue at `max_threads=16`, rising with thread
  count because the mutex is shared by every probe thread. At `max_block_size=512` and 10 M probe
  rows that is 19 531 blocks × ~150 ns ≈ **3 ms**, still negligible against grace's I/O.
* Cell C: UHJ's prologue **1.5–3× more expensive** than baseline's, i.e. a few hundred nanoseconds
  per probe block, and worse under contention. Small in absolute terms; worth fixing by hoisting the
  once-only work out of the per-block path, since it is trivially hoistable.
* Real-world exposure: **zero** unless the threshold is crossed (0.2, 5.4).

---

## 7. D20 — `SpillingHashJoin` locks on the single-join path too {#d20}

### 7.1 Mechanism {#d20-mechanism}

The merge base took the shared lock only when `concurrent_join` was set; the branch takes it on both
paths (`SpillingHashJoin.cpp:170-180`) and adds the matching `unique_lock` with a re-check in
`switchToGraceHashJoin` (`:233-238`). The per-build-block delta on the single-in-memory-join path is
therefore exactly:

* one `SharedMutex::lock_shared()` / `unlock_shared()` pair, and
* one extra `state.load(acquire)`.

The `getTotalByteCount()` call per build block was already there in the merge base (`:155` old,
`:158` new), so it is **not** part of D20 — a useful distinction, because on the UHJ arm that call is
D3's third `blocks_mutex` acquisition and it would otherwise be miscounted here.

Because `HashJoin::supportParallelJoin()` is false on both binaries, the baseline
`hash` + wrapper path has exactly **one** build stream on both, so the added shared lock is
uncontended. That makes D20 the smallest and cleanest thing in this group to measure.

### 7.2 Design {#d20-design}

`join_algorithm=hash`, `sp_cap` = 64 GiB (wrapper present, never trips), 8 M-row build-only,
`max_block_size` ∈ {65505, 8192, 2048, 512}, **both binaries**. Slope difference against build block
count is D20. The null control is the same `hash` cell with `sp_cap 0` on both binaries — identical
to `m_D14`'s control, so `m_D20.sh` runs it again rather than depending on `m_D14`'s output file.

`max_threads` is swept {1, 16} only to show that it does not matter: the build stream count is 1
either way, so a thread-dependent result would mean the wrapper is not in the mode the cell assumes.

### 7.3 Real world {#d20-real-world}

Live on every eligible suite join, on **both** arms — this is a branch change to the baseline arm, so
it makes the baseline arm slower, in `unified_hash`'s favour, exactly like D14. Magnitude bounds it
to irrelevance (7.4), but it is the second of the two systematic biases in this campaign and it is
worth stating that both point the same way.

### 7.4 Expected direction and magnitude (prediction) {#d20-prediction}

* **20–50 ns per build block**, uncontended. At the default `max_block_size=65505` a 240 M-row build
  is 3 663 blocks, so **under 0.2 ms per query**. At `max_block_size=512` and 8 M rows, 15 625
  blocks ≈ **0.5 ms**.
* Direction: branch binary **slower**, i.e. biased in `unified_hash`'s favour.
* **Not separable from D14** in any `parallel_hash` + wrapper cell, since both are per-build-block
  additions to the baseline arm; the design avoids that by using `hash` for D20 (no timers) and
  `sp_cap 0` for D14 (no wrapper).

---

## 8. D23 — arena ownership {#d23}

### 8.1 Mechanism, and why the count is parity {#d23-mechanism}

UHJ owns `num_slots` arenas routed by `slotForBucket` (`UnifiedHashJoin/HashJoin.h:637-647`,
created at `HashJoin.cpp:605-610`); baseline has one `Arena` per `HashJoin`
(`HashJoin/HashJoin.h:440`), i.e. one for serial `hash` and `slots` for `parallel_hash`. The counts
match exactly: `slotCountForThreads` is `bit_ceil(max_threads)` capped at 256
(`UnifiedHashJoin/HashJoin.cpp:70-76`) and `ConcurrentHashJoin`'s is
`toPowerOfTwo(min(max_threads, 256))` (`ConcurrentHashJoin.cpp:197`) — the same function.

So D23 is not a quantity divergence. What differs is *residence*: a string key's arena is chosen by
its bucket under UHJ and by which build thread's block it arrived in under `parallel_hash`. That is
a locality difference (the audit's open question 4) and possibly a fragmentation difference, and
neither is a time effect that can be attributed by reading.

### 8.2 Why a time measurement would not work, and what to measure instead {#d23-design}

Any query that exercises the arenas also exercises the map layout (B, D4), the accounting (D2, D10),
the scatter (A) and the probe rewrite. There is no setting that changes arena routing alone, so
**D23 is not separately measurable as a time effect** — stated rather than attempted. What *is*
measurable is the footprint, and one subtraction isolates the arena from everything else in it:

```
arena_bytes(arm, threads) ≈ mem_bytes(String key) − mem_bytes(UInt64 key)
```

at equal row counts, where the `String` shape allocates one `keyHolderPersistKey` copy per distinct
key into an arena and the `UInt64` shape allocates none. Both shapes share the same block plumbing,
the same `RowRefList` nodes and the same stored columns, so the difference is dominated by the arena
plus the string columns themselves — and the string columns are identical across arms, so the
**arm-to-arm difference of that difference** is the arena effect.

Cells: `sql_str_build 4M 48` and `sql_build_only 4M` at `max_threads` ∈ {1, 4, 16}, `sp_cap 0`,
`baseline`/`parallel_hash` and `uhj`/`unified_hash`, reporting `mem_bytes`, `build_us` and — for the
locality question — `dtlb_misses` and `cache_misses` per build row. The `max_threads=1` row is the
control: one arena on both arms, so the arm difference there is *not* D23 and whatever it is must be
subtracted from the 4- and 16-thread rows.

### 8.3 Expected direction and magnitude (prediction) {#d23-prediction}

* Footprint: **within ±5 %**, i.e. nil. `Arena` grows geometrically from 4 KiB, and both arms split
  the same key set over the same number of arenas by a hash, so the per-arena chunk overshoot is
  statistically identical.
* Locality: **nil to slightly UHJ-favourable**. UHJ's bucket-routed arena means keys that live in
  one map bucket also live in one arena, so a probe that hits a bucket touches fewer arena pages;
  baseline's thread-routed arena interleaves keys from all buckets. If `dtlb_misses` per probe row is
  measurably lower on UHJ for the string shape at `max_threads=16`, that is the mechanism — but it
  is confounded with the map layout and should be reported as an observation, not an attribution.
* If the footprint difference exceeds ±5 %, the likely cause is not D23 but D2 (UHJ's
  `bucket_bytes` under-reporting changing when `shrinkStoredBlocksToFit` fires), which is a sibling's
  measurement.

---

## 9. What cannot be separated from what {#separability}

| divergence | inseparable from | why, and what the script says instead |
|---|---|---|
| **D14** | D20 (in any `parallel_hash` + wrapper cell) | both are per-build-block additions to the baseline arm; avoided by measuring D14 with `sp_cap 0` |
| **D14** | binary-level code layout | not separable in principle; bounded by the `hash` null control and cross-checked by the C calibration |
| **D10** | D3, at `max_threads > 1` | both per-build-block on the same path; separable only at `max_threads=1`, via the key type (bucket count 1 vs 256) |
| **D10** | A, B | per-row, so they sit in the intercept of the block-size slope and do not bias it |
| **D11** | D1, in cell C | the OR-join cell changes the algorithm *and* the reserve; reported jointly |
| **D11** | D2 | the reserve changes the buffer size, which is what D2 mis-accounts; `prealloc` is read instead of inferring it from bytes |
| **D12** | D1 | in the suites they are the same effect seen at two levels (planner gate, pipeline stream count); reported jointly |
| **D12** | D3, A | the N-thread feed's cost *is* the internal serialisation and the scatter |
| **D18** | B, D10, D21, the probe rewrite | D18 selects the serial regime rather than costing anything; the measurement is of the aggregate |
| **D19** | nothing — but there is almost nothing to separate | the delta is confined to the `hash_join == nullptr` window; only a bound is reported |
| **D20** | D14 (see above), D3 (on the UHJ arm, where the same per-block `getTotalByteCount` is D3's third acquisition) | measured with `hash`, which has neither |
| **D23** | B, D2, D4, A, the probe rewrite | not separable as a time effect at all; only the `String` − `UInt64` footprint subtraction is attributable |

Two divergences in this group are **systematic biases in the same direction**: D14 and D20 both make
the baseline arm slower on a branch build, and both flatter `unified_hash`. They are additive, both
per-block, and together bound the "free" advantage of a single-binary A/B at roughly
**200–330 ns per build block and 175–280 ns per probe block**.

---

## 10. Corrections to `DIVERGENCE_INVENTORY.md` {#corrections}

Found while reading for this design; none of them changes an impact class, all of them change what a
measurement can claim.

1. **D11 divergence (1) is parity.** `ConcurrentHashJoin` also reserves lazily on the first insert
   per slot (`ConcurrentHashJoin.cpp:341-347`), guarded by `space_was_preallocated`.
2. **D11 divergence (2) is narrower than stated.** Baseline clamps too, by the same formula, whenever
   a `SpillingHashJoin` supplies `external_join_threshold` (`ConcurrentHashJoin.cpp:137-158`). The
   divergence survives only off the wrapper path (ASOF, multi-disjunct) and inside `GraceHashJoin`.
3. **D19's delta is confined to `hash_join == nullptr`.** The merge base already took the lock per
   probe block whenever `getNumBuckets() <= 1` and `hash_join` was set, which is the default
   (`grace_hash_join_initial_buckets = 1`).
4. **D10's sign depends on the map.** UHJ is cheaper than `parallel_hash` by a factor of `slots` for
   two-level maps and more expensive by ~`256/slots` for fixed maps. The inventory says this in its
   table but the summary line ("Medium") does not convey that the two-level case is a UHJ win at
   every thread count above 1.
5. **Not in the inventory:** `GraceHashJoin::joinBlock` calls `runPostBuildPhase()` once per probe
   block under `hash_join_mutex`, and UHJ's override adds `recomputeBucketBytes()` +
   `freezeMapsForProbing()` to it (section 6.2). Predicted small, but it is a UHJ-only per-probe-block
   cost inside a global lock, and it is trivially hoistable.
6. **Cross-cutting, affects every script in this directory:**
   `max_bytes_ratio_before_external_join` defaults to `0.5`, so `SpillingHashJoin` wraps every
   eligible join unless the ratio is explicitly zeroed (section 0.2). A cell that sets only
   `max_bytes_before_external_join=0` is not measuring an unwrapped join.
