# Working-branch benchmark scatter/probe kernels (input to U1 productionization)

Branch: `radix-join-bandwidth-model` (all citations are plain files on this branch).
Files covered:

- `src/Common/benchmarks/hash_join_bench.h` (190 lines) — public API of the kernels + driver.
- `src/Common/benchmarks/hash_join_bench.cpp` (951 lines) — scatter kernels, `streamingWaveProbe`, `makeTableJoin`, driver.
- `src/Common/benchmarks/radix_hash_join_bench.{h,cpp}` — `RadixHashJoinBench` (`IJoinBench` impl): scatter + per-partition `HashJoin`, `probeWaves`.
- `src/Common/benchmarks/hash_join_bandwidth_model.cpp` (2022 lines) — measurement kernels, analytical model, `runBepWaveSweep`, `htBytesForDistinctReserved`, driver `main`.
- `src/Common/benchmarks/concurrent_hash_join_bench.{h,cpp}` — NPHJ competitor (real `ConcurrentHashJoin`), referenced for symmetry.

Everything lives in `namespace DB::JoinBench`. All data is `ColumnUInt64` only: column 0 is the key, the rest are 8-byte payloads.

---

## 1. `scatterSide` + `ScatterScratch`

### Entry point

`scatterSide(WorkerPool &, const std::vector<Block> &, const std::vector<size_t> & pass_bits)` — `hash_join_bench.h:144`, impl `hash_join_bench.cpp:656-684`.

- Hard precondition (`hash_join_bench.cpp:658-664`): total rows per side must be `<= 2^32-1` because histogram/offset counters are `UInt32`; violation throws `std::runtime_error` (needs `DB::Exception` in production).
- Wraps caller `Block`s into `Chunk {Columns columns; size_t rows}` (`hash_join_bench.h:114-120`) — shares `ColumnPtr` refs, no copy.
- Runs `scatterPass` once per entry in `pass_bits` (`cpp:677-683`); passes slice disjoint bit ranges of a single 32-bit route word, high bits first (`shift = 32 - bits_done - bits`, `cpp:410`).

### Pass planning

- `computePassBits(p_star, f_max)` — `cpp:638-654`: `total_bits = log2(bit_ceil(p_star))`, per-pass bits `= ceil(total_bits / n_pass)` where `n_pass = ceil(total_bits / log2(bit_floor(f_max)))`. Single pass is the common case.
- `MAX_FANOUT_PER_PASS = 8192` — `hash_join_bench.h:129`. This is a *memory-correctness* ceiling, not a tuning knob: per worker per partition the SWWC state is ~76 B (64 B staging line + 8 B cursor + 4 B fill), so 8192 partitions ≈ 608 KiB, only fitting an L2 ≥ ~1 MiB. Effective runtime fanout = `min(measured F_max, MAX_FANOUT_PER_PASS, bit_floor(L2/128))` (model wiring at `hash_join_bandwidth_model.cpp:1986-1988`; on the sweep-free `--join-nb`/`--bep-nb` paths the same clamp minus the measured term at `:1414`, `:1528`).

### Routing

- `routeWord(UInt64 key)` — `cpp:111-118`: **must be independent of `HashCRC32` (CRC32C)** used by the real join hash tables, else partition assignment correlates with bucket placement. aarch64: ISO-poly `__crc32d`; elsewhere multiply-shift (`key * 0x9E3779B97F4A7C15 >> 32`).
- Key column (j==0) routes from keys and emits 2-byte partition ids (`RouteFromKey`, `cpp:255-268`); payload columns re-route via the ids (`RouteFromPids`, `cpp:270-275`) — a 2 B id read replaces an 8 B key re-read, and the ids end all routing uses of the key column, enabling eager input drop. `pids` is null when there are no payload columns (`scatterChunkColumn`, `cpp:342-361`).

### Histogram

- `histogramChunk` — `cpp:225-244`: per-worker histograms into disjoint slices of one flat `threads * fanout` `UInt32` array. At fanout ≤ `HIST_INTERLEAVE_MAX_FANOUT = 2048` (`cpp:198`), 4 interleaved `UInt32` lanes (row i increments lane `i & 3`) break the load-increment-store serialization (~1.9x measured at fanout 2); reduced by `reduceHistogramLanes` (`cpp:246-250`). Above 2048 collisions are rare and lanes would blow cache, so direct increments.

### Fused prefix-sum + exact allocation

- First pass, barrier 2 (`cpp:445-467`): each worker owns a contiguous disjoint partition range `[fanout*tid/threads, fanout*(tid+1)/threads)` and for each partition computes the cross-worker running offsets, the total, and immediately allocates `PartitionOutput` — **no single-threaded prefix-sum phase**.
- `PartitionOutput` (`cpp:363-398`): exactly-sized `ColumnUInt64::create(rows)` (POD contents uninitialized — no memset; pages first-touched by the scatter writes), raw base pointers for direct placement, `toChunk()` moves out. Refine passes call `allocateColumn` just-in-time per column round so the allocator reuses the extents of the input column just dropped.

### SWWC (software write-combining) + non-temporal stores

- `SWWC_MIN_FANOUT = 256` (`cpp:120-124`): below this, direct per-partition cursor stores (`scatterChunkDirect`, `cpp:277-282`); at ≥256 the SWWC path (`scatterChunkSwwc`, `cpp:284-326`).
- `ScatterScratch` — `cpp:126-193`: per worker, `fanout` × 64-byte staging lines (64-aligned inside a padded buffer), `cursors[fanout]` (`UInt64*`), `fill[fanout]` (`UInt32`, bytes staged).
  - Invariant (`cpp:130-136`): staged bytes for partition p live at `staging_line + [m, fill)` where `m = cursor & 63`. `seed(p, cursor)` sets `fill[p] = cursor & 63` so the first flush of a misaligned stream emits the partial head with `memcpy`; after that the cursor is line-aligned and each full line goes out via `__builtin_nontemporal_store` of a 64-byte vector (`NtLine`, `cpp:126`, store at `cpp:319`).
  - `drain()` (`cpp:169-192`) flushes residual bytes (only when `fill > m`) and issues one `std::atomic_thread_fence(seq_cst)` to publish the weakly-ordered NT stores before outputs are read.
  - Hoisting of `staging`/`cursors`/`fill` into locals in the row loop (`cpp:287-292`) is load-bearing: NT store through `char*` defeats TBAA, measured ~1.07x (clang) / ~1.65x (GCC).
  - A narrow-loads variant of the flush was measured and rejected (`cpp:311-318`).

### First-pass batching and the ~fanout × 4 KiB window-fill constraint

- `SCATTER_BATCH_MIN_ROWS = 256Ki`, `SCATTER_BATCH_LINES_PER_PARTITION = 64` → `scatterBatchRowsTarget(fanout) = max(256Ki, fanout * 64 * 8)` rows (`cpp:200-217`). Meaning: each (batch, column) window writes ≥ 64 lines = **4 KiB per partition**, so the per-boundary cost (seed/save cursor sweeps + up to one partial-line flush + one head-realignment memcpy per partition) stays ≤ ~1.5% of lines written. The row floor keeps low-fanout batches large enough to amortize the sweeps. Also bounds transient memory: batch input rows + 2 B/row pids ≤ ~4M rows/worker at fanout 8192.
- First pass structure (`cpp:417-544`): exactly 3 pool.run barriers — histogram, fused prefix-sum/alloc, one fused all-columns scatter. In the scatter, each worker walks its chunk stripe (`c = tid; c += threads`) in batches; for each batch: key column scatter emits pids, payload columns scatter via pids, then **the batch's input chunks are dropped** (`chunks[b].columns = {}`, `cpp:534-537`) — on pass 0 this releases the caller's block references (upstream recycling point in a real pipeline). Each worker writes only its `[offset, offset+hist)` range of each (partition, column) buffer — disjoint, so no barrier between columns or batches; the final pool.run join + per-worker `drain()` fences suffice.
- Refine passes (`cpp:546-630`): groups pulled by an atomic counter (dynamic scheduling — skew defense, since group sizes diverge), each group processed entirely worker-locally: histogram, per-group pids for the whole group (bounded: ≤ 1/fanout_so_far of the side), then per column: JIT-allocate outputs → scatter → **drop the consumed input column** (`chunk.columns[j] = nullptr`, `cpp:619-620`), keeping a group in flight at ~(C+1)/C of its size instead of 2x. `groups[g].clear()` at `cpp:628`.
- `chassert(fanout <= 2^16)` (`cpp:409`) — pids are `UInt16`. `chassert` is debug-only; production needs a real check.

---

## 2. `streamingWaveProbe` — fused single-dispatch wave loop

Decl `hash_join_bench.h:170-177`, impl `hash_join_bench.cpp:686-846`. This is the benchmark shape of the **streaming budget-bounded probe** (BEP, evict-all-at-budget): the probe side is consumed in `waves` consecutive windows; each window is radix-scattered to leaf depth (single pass of `bits` bits, `chassert(1 <= bits <= 16)` at `cpp:701`) and every non-empty partition chunk is probed and dropped before the next window.

### Why fused

Header comment `hash_join_bench.h:158-164`: the whole wave loop runs inside **one** `pool.run`; phases are separated by `std::barrier` instead of per-phase pool dispatches, and per-worker scratch persists across waves. This removed the per-wave overhead that dominated small budgets (~4 dispatches/wave, measured ~1.9 ms/wave at 96 threads).

### State

Shared, allocated once before the pool.run (`cpp:726-738`): `hist[threads*fanout]`, `offsets[threads*fanout]` (both `UInt32`), `totals[fanout]` (`UInt64`), `parts[fanout]` (`PartitionOutput`), `std::atomic<size_t> next_partition` (probe work stealing; reset by tid 0 during the alloc phase, barrier-separated from both neighboring uses), `rows`/`digest` atomics, `std::barrier<> barrier(threads)`.

Per-worker, persistent across all waves (`cpp:740-750`): `ScatterScratch` (SWWC staging/cursors/fill), histogram `lanes`, `pids` buffer, `col_cursors[num_columns*fanout]`, `local_rows`, `local_digest`. `Stopwatch` consulted on tid 0 only; an initial `barrier.arrive_and_wait()` (`cpp:753`) aligns the start so tid 0's first span excludes pool ramp-up.

### Exact phase structure per wave `w` (window = chunks `[size*w/waves, size*(w+1)/waves)`)

1. **Histogram** (`cpp:762-770`): each worker histograms its strided chunks (`c = begin + tid; c += threads`) into its `hist` slice (lanes if `fanout <= 2048`). → `barrier.arrive_and_wait()` (`cpp:771`).
2. **Fused prefix-sum + exact allocation** (`cpp:773-788`): each worker owns partitions `[fanout*tid/threads, fanout*(tid+1)/threads)`: cross-worker offsets, `totals[p]`, `parts[p] = PartitionOutput{}` then `allocate(num_columns, total)` if non-empty. tid 0 resets `next_partition = 0`. → barrier (`cpp:789`).
3. **Fused all-columns scatter** (`cpp:791-813`): same structure as `scatterPass` barrier 3 but without intra-window batching (the window *is* the batch): resize `pids` to the stripe's rows, seed cursors from `totals/offsets`, scatter column-major over the worker's stripe, `drain()` per column. → barrier (`cpp:814`); tid 0 adds elapsed to `stats.scatter_sec` and restarts the watch.
4. **Probe with work stealing** (`cpp:821-829`): `p = next_partition.fetch_add(1)`; skip `totals[p] == 0`; `probe_partition(p, parts[p].toChunk(), digest?)` — the callback **receives ownership** of the window's partition chunk (freed on return) and returns output row count. → barrier (`cpp:830`); tid 0 adds to `stats.probe_sec`.

After all waves each worker adds `local_rows` to `g_sink` and the shared atomics (`cpp:838-840`).

Notes for the port:
- `num_waves = max(1, min(waves, chunks.size()))` (`cpp:723`) — window granularity is whole chunks.
- Unlike `scatterPass`, the streaming path does **not** drop the window's input chunk references after scattering (the benchmark reuses the probe blocks across runs); a production port should release each window's input after phase 3.
- The `>2^32-1` probe-rows precondition throws `std::runtime_error` (`cpp:720-721`).
- Timing model: tid 0's `Stopwatch` spans are ~wall time only because the barriers align all workers.

---

## 3. `RadixHashJoinBench::probeWaves` (radix_hash_join_bench.cpp)

- `build` (`radix_hash_join_bench.cpp:39-77`): `scatterSide` (timed to `build_scatter_sec`), then per-partition `HashJoin`s created with **dynamic scheduling** (atomic counter — static striping inflates the phase ~1.5x on non-power-of-two core counts and is the skew defense, `:46-50`). Each `HashJoin` gets `reserve_num = exact partition row count from the scatter histogram` (no rehash growth, up to 2x smaller table than the growth ladder), `use_two_level_maps = false`, name `radix{p}`, `addBlockToJoin(..., check_limits=false)`, `onBuildPhaseFinish()`. `build_parts[p].clear()` releases only `Chunk`/`ColumnPtr` wrappers — `addBlockToJoin` COW-shares the same columns into the join's stored blocks, so the scattered build side stays resident until `teardown()` (`:70-74`).
- `probe` = `probeWaves(blocks, waves=1, fingerprint)` (`:79-82`).
- `probeWaves` (`:84-152`):
  - **Single-pass case (`pass_bits.size() == 1`, the common case `p_star <= f_max`)**: delegates the entire wave loop to `streamingWaveProbe` with `probe_partition = drainJoinResult(partition_joins[p]->joinBlock(toBlock(chunk, left_header)), digest)` (`:93-104`). Scatter/probe timings from `StreamingWaveStats`.
  - **Multi-pass fallback (`p_star > f_max`)**: legacy per-wave loop (`:113-148`): window of blocks → `scatterSide` (timed) → one `pool.run` with atomic-counter work stealing over partitions; probes every chunk of `probe_parts[p]`, then `probe_parts[p].clear()` frees the consumed scattered probe input before the next partition (`:137-142`). ~4 pool dispatches per wave — the overhead the fused path removed.
- `teardown` (`:154-164`): parallel `partition_joins[p].reset()` via atomic counter — symmetric with NPHJ's parallelized destructor.
- `toBlock` (`:13-22`) re-wraps a `Chunk` with the header's names/types to feed `HashJoin::joinBlock`.

---

## 4. `runBepWaveSweep`, the budget rule, `htBytesForDistinctReserved` (hash_join_bandwidth_model.cpp)

### `htBytesForDistinctReserved(n)` — `:637-643`

Exact byte size of a `HashMap<UInt64, UInt64>` constructed with a size hint of `n` (the reserve path the radix benches use): `degree = max(8, floor(log2(n-1)) + 2)`, size `= 2^degree * 16` (16-byte cells, `static_assert` at `:155`) — i.e. 2n..4n cells. Contrast `htBytesForDistinct` (`:626-632`), the insertion-growth ladder (degree 8, +2 up to 23, then +1; 2n..8n cells). **Do not unify them**: the model's L2-fit test uses `Reserved`, but curve *lookups* use `htBytesForDistinct` because that is the label the sweeps record points with (`predict()`, `:1175-1181`).

### P* / F_max selection (repeated at `:1400-1414` in `runSingleJoin`, `:1516-1528` in `runBepWaveSweep`, and `predict()` `:1135-1147`)

- `partition_bytes(P) = htBytesForDistinctReserved(max(1, N_b/P)) + (N_b/P) * w_b`; P* = smallest power of two with `partition_bytes(P*) <= L2`, bumped to `>= bit_ceil(threads)` when > 1, capped at `bit_ceil(max_partitions)`, floor 2 on the join paths.
- `f_max = min(MAX_FANOUT_PER_PASS, bit_floor(max(2, L2/128)))` on the no-kernel paths (`:1414`, `:1528`); the full-model path also intersects with the measured "largest contiguous-prefix fanout still ≥ 80% of peak scatter bandwidth" (`:1976-1988`).

### `runBepWaveSweep(cfg, pool, cache, n_b, n_p, extra_budget)` — `:1505-1617`

Invoked by `--bep-nb` (+ optional `--bep-np`, `--bep-budget`) from `main` (`:1940-1946`), skipping all kernels.

1. Selects P*/f_max as above; generates build side (`uniqueKeys(n_b)`) and probe side (`probePermutationKeys(n_b, n_p, hit_rate)` — an exact permutation of the build key space, every build key exactly `n_p/n_b` times).
2. **NPHJ reference** (`:1541-1557`): builds one `ConcurrentHashJoinBench`, medians the probe over `cfg.runs`, tears down. Gives `np_probe_sec` and `np_matches`.
3. Builds one `RadixHashJoinBench` **once**; all budget rows probe the same prebuilt partition tables.
4. **The budget rule** (`:1564-1584`):
   - Reference quantity `build_accumulated_bytes = n_b * w_b + p_star * htBytesForDistinctReserved(max(1, n_b/p_star))` — everything the build phase has accumulated by probe time (stored scattered build rows + reserved per-partition hash tables).
   - `probe_bytes = n_p * w_p`.
   - Budget list: `PHJ` (budget 0 = unbounded, 1 wave); for percent ∈ {5,10,15,20,25}: `budget = max(512 MiB, build_accumulated_bytes * percent / 100)` (`min_budget = 512ULL << 20`, `:1569`); optional explicit `extra_budget`.
   - `waves = ceil(probe_bytes / budget)` (`:1588`).
5. Per budget row (`:1586-1611`): one discarded warmup `probeWaves`, then `cfg.runs` timed repetitions; median by total (scatter+join); prints waves, `rows/part/wave = n_p / (waves * p_star)` (how well each partition visit amortizes the working-set reload), scatter/probe/total ms, ns/row, `vs NP = np_probe_sec / total`, and a `matches == np_matches` check.
6. `rp.teardown()` and a legend (`:1612-1616`).

Related model context an implementer will want:
- `predict()` (`:1118-1186`): T_RP = scatter term (`n_pass * (N_b w_b + N_p w_p) / B_scatter(per-pass fanout)`, or a directly measured 2-pass bandwidth when `n_pass == 2`) + RP build/probe curve lookups at the per-partition label.
- Fraction crossover `printFractionCrossover` (`:1294-1384`): closed-form minimal probe fraction `f* = (c_s w_b - dBuild) / (dPG - c_s w_p)`, with a `C_reload = (P* htBytesReserved(D/P*) + N_b w_b) / B_read` variant charged for the compulsory partition reload — directly relevant to when the streaming BEP probe pays off.
- Fail-close peak-memory guard before running (`estimatePeakBytes` + `MemAvailable` 80% check, `:1752-1806`, `:1905-1927`).

---

## 5. Benchmark entanglements inventory (must be removed/replaced for production)

| # | Entanglement | Where | Production replacement |
|---|---|---|---|
| E1 | `g_sink` global anti-DCE sink | `hash_join_bench.h:24`, `hash_join_bench.cpp:54`, added to at `hash_join_bench.cpp:838`, `radix_hash_join_bench.cpp:143`, `concurrent_hash_join_bench.cpp:45`, model `:518,583,878,929,1042`, printed in `main` `:1936,1944,2020` | Delete. Production output blocks flow to downstream processors; nothing to sink. |
| E2 | Private fixed `WorkerPool` (`ThreadPoolImpl` with `CurrentMetrics::LocalThread*`, threads == max_free_threads, blocking fork-join `run`) | `hash_join_bench.h:29-44`, `hash_join_bench.cpp:56-70`; every kernel and `RadixHashJoinBench` hold a `WorkerPool &` | The `IJoin` implementation must not own a pool: probe is driven by pipeline threads (`joinBlock` called concurrently). Where internal parallelism is genuinely needed (build partitioning, teardown), use `ThreadPoolCallbackRunner` on `GlobalThreadPool`/the context's pool with the query's `ThreadGroup` attached (`ThreadGroupSwitcher`), honoring `max_threads`. |
| E3 | `std::barrier` phase transitions assume exactly `threads` dedicated, never-failing participants; a worker that throws (e.g. allocation failure inside `HashJoin::joinBlock`) leaves the rest blocked in `arrive_and_wait` forever | `hash_join_bench.cpp:738,753,771,789,814,830` | Either restructure phases as processor/pipeline states, or keep the barrier inside a dedicated bounded pool with: try/catch around the phase body that records the exception and still arrives; a `std::atomic<bool>` stop flag checked after every barrier; rethrow after the join. Never let an exception skip an arrive. |
| E4 | No cancellation anywhere: wave loop, scatter loops, per-partition build/probe/teardown loops run to completion | `hash_join_bench.cpp:757-836` (waves), `:404-634` (scatterPass), `radix_hash_join_bench.cpp:51-76,126-146,157-162` | `std::atomic<bool>`/`isCancelled()` checks at wave boundaries, per chunk batch, and per partition in work-stealing loops; propagate as `DB::Exception(ErrorCodes::QUERY_WAS_CANCELLED)` through the E3 mechanism. |
| E5 | Memory not attributed to the query: pool threads carry `ThreadStatus` but no query `ThreadGroup`, so `PODArray`/`ColumnUInt64` allocations (hist, offsets, pids, SWWC staging, partition outputs, scattered sides) hit only `total_memory_tracker`; `addBlockToJoin(..., check_limits=false)` everywhere; no limits/spill | `radix_hash_join_bench.cpp:68`, `concurrent_hash_join_bench.cpp:29`, all scratch allocations in `hash_join_bench.cpp` | Attach workers to the query `ThreadGroup` so `CurrentMemoryTracker` accounting/limits apply automatically; pass `check_limits=true` (or enforce `max_{rows,bytes}_in_join` explicitly); account the scatter's transient budget (window bytes + pids + staging) up front against the probe-buffer budget. No raw `mmap`/`malloc` exists in these kernels — everything already goes through `PODArray`/`IColumn`, which is the right base. |
| E6 | `std::runtime_error` + `chassert` for hard preconditions (2^32-1 rows/side; `fanout <= 2^16`; `bits <= 16`) | `hash_join_bench.cpp:663-664,720-721` (throw), `:408-409,700-701` (chassert, debug-only) | `DB::Exception` with proper `ErrorCodes` (e.g. `LOGICAL_ERROR`/`LIMIT_EXCEEDED`); the row-count limit needs either 64-bit counters or a documented fallback to the non-partitioned join. |
| E7 | All-`UInt64`-columns assumption: `assert_cast<const ColumnUInt64 &>` on every column, 8 B fixed stride baked into SWWC/cursor arithmetic, `routeWord(UInt64)` on the raw key | `hash_join_bench.cpp:328-336` (`keyData`/`columnData`), `:120-126` (LINE_BYTES/ELEMS_PER_LINE), `:111-118` | Production scatter must route on the join key *hash* (any key type; keep it independent of the table's CRC32C — same requirement as `routeWord`'s comment) and either (a) restrict the SWWC fast path to fixed-width columns with per-column element size, falling back to `IColumn::scatter`/selector-based dispatch otherwise, or (b) scatter row references + late materialization. |
| E8 | Unique/uniform-key assumptions: `makeTableJoin` pins INNER ALL with duplicate-free build keys so `onBuildPhaseFinish` promotes to `RightAny` (one output row per probe row) — the whole model's output-size assumption; benchmarks generate keys via bijections (`uniqueKeys`, `probePermutationKeys`); per-partition per-worker offsets fit `UInt32`; no defense against a partition exceeding the budget except refine passes | `hash_join_bench.cpp:848-890` (makeTableJoin comment `:855-872`), model `:390-444` (generators), `:78-88` (header comment) | Production must handle duplicate keys (`RowRefList` chains, output amplification ⇒ the budget must bound *output* too, not just scattered input), arbitrary strictness/kinds, and skew: re-split or spill oversized partitions, and don't size P* assuming D = N_b. |
| E9 | Count-only output draining: `drainJoinResult` materializes every output block then drops it; `probe_partition` returns row counts; fingerprint only for verification | `hash_join_bench.cpp:917-930`, `radix_hash_join_bench.cpp:98-99,133-135` | Stream `JoinResultPtr` blocks to the caller/downstream (the `IJoin::joinBlock` contract already returns `JoinResultPtr`); respect `max_joined_block_rows`; back-pressure instead of draining in a tight loop. |
| E10 | `makeTableJoin` fabricates `TableJoin` from `static const Settings` + chasserted defaults (`enable_software_prefetch_in_join` etc.) | `hash_join_bench.cpp:861-872` | Production receives the planner's `TableJoin`; the pinned-default chasserts disappear. |
| E11 | Wall-clock phase stats via tid-0 `Stopwatch` + mutable members (`probe_scatter_sec`, `build_scatter_sec`), not thread-safe, benchmark-report only | `hash_join_bench.cpp:752-756,815-819,831-835`, `radix_hash_join_bench.h:46-48` | `ProfileEvents` (the pattern already exists: `ProbeProfile`/`currentProbeProfile`, `hash_join_bench.h:53-67`) — add e.g. `RadixJoinScatterMicroseconds`, `RadixJoinProbeMicroseconds`. |
| E12 | Whole-side batch interface: `build(vector<Block>)`, `probe(vector<Block>)` — the entire probe side is materialized in memory before `streamingWaveProbe` slices it into waves by *chunk count* | `hash_join_bench.h:93-99`, `hash_join_bench.cpp:723,759-760` | The core U1 re-architecture: accumulate incoming probe blocks into a budget-bounded buffer (bytes, not chunk fractions) and trigger a wave when the buffer reaches the budget (`max(512 MiB floor, fraction of build accumulated bytes)` per `runBepWaveSweep:1564-1584`); final partial wave on `onBuildPhaseFinish`-style flush. |
| E13 | Streaming path never releases the window's input chunk refs after scattering (benchmark reuses inputs across runs) | `hash_join_bench.cpp:708-719` vs `scatterPass`'s eager drops `:534-537` | Drop each window's input references right after phase 3 (scatter), as `scatterPass` already does — required for the budget accounting to hold. |
| E14 | Machine introspection in the driver: sysfs cache detection, `/proc/meminfo` fail-close guard, `je_mallctl`, `getNumberOfCPUCoresToUse` | `hash_join_bandwidth_model.cpp:252-317,1756-1806,1894-1927` | P*/f_max selection in the server should read cache sizes via existing ClickHouse facilities (or a setting with a sane default) and derive the probe budget from settings/memory limits, not `MemAvailable`. |
| E15 | Benchmark-only measurement scaffolding: `medianTime`, warmup runs, order alternation to dodge jemalloc extent-reuse asymmetry, size-hint `stats_key` warm patterns | model `:320-329,1455-1472,1590-1603,1674-1681` | Not ported; noted so nobody mistakes them for functional requirements. |

### Constants to carry over (with their justifications)

- `SWWC_MIN_FANOUT = 256` (`hash_join_bench.cpp:124`) — direct path wins below.
- `MAX_FANOUT_PER_PASS = 8192` (`hash_join_bench.h:129`) — SWWC state ~76 B/partition/worker must fit L2.
- L2-derived fanout cap `bit_floor(L2/128)` (`hash_join_bandwidth_model.cpp:1986-1988`) — headroom over the 76 B for histogram/cursors.
- `HIST_INTERLEAVE_MAX_FANOUT = 2048`, 4 lanes (`hash_join_bench.cpp:198`).
- Batch fill: ≥ 64 lines (4 KiB) per partition per column per window ⇒ `scatterBatchRowsTarget = max(256Ki rows, fanout * 512)` (`hash_join_bench.cpp:211-217`).
- Budget rule: `max(512 MiB, 5..25% of (stored build rows + reserved HT bytes))`; `waves = ceil(probe_bytes / budget)` (`hash_join_bandwidth_model.cpp:1564-1588`).
- P* rule: smallest pow2 with `htBytesForDistinctReserved(D/P) + (N_b/P) w_b <= L2`, `>= bit_ceil(threads)`, `<= max_partitions` (`:1516-1527`).
