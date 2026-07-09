# Census of donor `src/Interpreters/RadixHashJoin/` on `origin/phj5-real`

Read-only survey (all content read via `git show origin/phj5-real:<path>`; nothing checked out).
Donor merge-base with our HEAD: `e69a9d5ba75`. Our branch `radix-join-bandwidth-model` is based on a
newer master (`2834291df98` lineage) — see the "Divergences vs our HEAD" section at the end; several
donor dependencies were renamed/refactored upstream and the port must adapt.

## File inventory {#inventory}

| File | Lines | Role |
|---|---|---|
| `RadixHashJoin.h` | 109 | `IJoin` facade, pimpl `State` |
| `RadixHashJoin.cpp` | 926 | IJoin lifecycle, probe/gather phases, thread pool |
| `BuildSide.h` / `.cpp` | 203 / 878 | build accumulation, histograms, deferred radix scatter |
| `LeafTable.h` / `.cpp` | 190 / 707 | per-leaf open-addressing tables, AMAC build insert + AMAC probe |
| `KeyRefScatter.h` / `.cpp` | 122 / 359 | fused `[ref|key]` record scatter kernels (DIRECT and SWWC+NT) |
| `Arena.h` / `.cpp` | 78 / 102 | jemalloc-backed bag of exact-sized aligned blocks |
| `PackedKeyHash.h` | 111 | 32-bit CRC32C hash of packed key; route/bucket bit split |
| `KeyLayout.h` | 61 | row-major multi-column key packers (`ColumnPackFn`) |
| `Hll.h` | 102 | header-only dense HyperLogLog (precision 4–6) |
| `PartitionPlan.h` / `.cpp` | 59 / 66 | leaf-count/pass planning from RHS estimate + L2 size |
| `ParallelFor.h` | 28 | `std::function` parallel-for contract (dense worker ids) |
| `tests/gtest_radix_hash_join.cpp` | 856 | unit tests (no server needed) |
| `benchmarks/CMakeLists.txt` | 7 | wires ONLY `bench_rhj_vs_chj` |
| `benchmarks/bench_rhj_vs_chj.cpp` | 881 | RHJ-vs-CHJ IJoin-level driver (links `dbms`) |
| `benchmarks/bench_build_bandwidth.cpp` | 565 | build-side data-movement comparison (NOT in CMake) |
| `benchmarks/bench_scatter_perf.cpp` | 665 | PMU-attributed scatter diagnostics (NOT in CMake) |

There is no `CMakeLists.txt` inside `src/Interpreters/RadixHashJoin/` on the donor.

## CMake wiring {#cmake}

- `src/CMakeLists.txt:339` (donor): `add_object_library(clickhouse_interpreters_radix_hash_join Interpreters/RadixHashJoin)` — the directory compiles as one more dbms object library. Our HEAD has the analogous list at `src/CMakeLists.txt:342` (`clickhouse_interpreters_hash_join`); add the radix line next to it.
- gtest: picked up automatically — `src/CMakeLists.txt` (donor :874–881) globs `gtest*.cpp` recursively under `src/` into `unit_tests_dbms`. No explicit registration needed; the same glob exists on our HEAD.
- `src/Interpreters/CMakeLists.txt` (donor): adds a `if (ENABLE_BENCHMARKS) add_subdirectory(benchmarks) endif()` block. Our HEAD's `src/Interpreters/CMakeLists.txt` has only tests/examples/fuzzers — the benchmarks block must be added.
- `src/Interpreters/benchmarks/CMakeLists.txt` (donor, 7 lines): `clickhouse_add_executable(bench_rhj_vs_chj bench_rhj_vs_chj.cpp)` + `target_link_libraries(... PRIVATE dbms)`. **`bench_build_bandwidth.cpp` and `bench_scatter_perf.cpp` are present in the tree but NOT wired into CMake.**

## Per-file details {#files}

### RadixHashJoin.h (109 lines) {#radixhashjoin-h}

Class `DB::RadixHashJoin : public IJoin`, pimpl (`struct State` in the .cpp). Constructor:

```
RadixHashJoin(std::shared_ptr<TableJoin>, SharedHeader right_sample_block, size_t max_threads,
              std::optional<UInt64> rhs_size_estimation, UInt64 max_partitions_per_pass,
              bool size_tables_by_distinct_estimate, const StatsCollectingParams &);
```

Overrides: `getName`, `getTableJoin`, `supportParallelJoin() -> true`, three `addBlockToJoin` overloads
(the third carries `size_t build_lane`), `checkTypesOfKeys`, `joinBlock(Block)` and
`joinBlock(Block, size_t lane)` (both return `JoinResultPtr`), `setTotals` (mutex-guarded, base does
unguarded assign), `getTotalRowCount`, `getTotalByteCount`, `alwaysReturnsEmptySet`,
`getNonJoinedBlocks` (returns `{}` — inner join only), `onBuildPhaseFinish`. Private `runPostBuild`.
Includes: `Core/Block_fwd.h`, `Interpreters/HashTablesStatistics.h`, `Interpreters/IJoin.h`.
The header doc records the planner gate contract (single-disjunct inner ALL equi-join; fixed-width,
non-nullable, non-LowCardinality keys; packed width multiple of 4 in [4, 64]); the ctor re-checks and
throws `LOGICAL_ERROR`.

### RadixHashJoin.cpp (926 lines) {#radixhashjoin-cpp}

Entangled with the IJoin lifecycle — this is the port's main adaptation surface.

- **Anonymous namespace helpers**: `detectL2Bytes` (`sysconf(_SC_LEVEL2_CACHE_SIZE)`, lines 71–78);
  `runParallelFor(ThreadPool&, num_workers, ThreadGroupPtr, total, UnitFn)` (93–149) — the production
  `ParallelFor`: schedules ≤ num_workers tasks via `scheduleOrThrow`, each wrapped in
  `ThreadGroupSwitcher(thread_group, ThreadName::RADIX_JOIN)` + a
  `ProfileEventTimeIncrement<Microseconds>(RadixHashJoinBuildMicroseconds)` watch; dynamic work
  stealing on `std::atomic<size_t> next`; first exception captured and rethrown after `pool.wait()`.
- **Output plan** (`LeftOut`/`PayOut`/`ReqOut`/`OutputPlan`, 151–175; `buildOutputPlan` 233–270):
  precomputed once (`std::call_once` on first `joinBlock`, i.e. the header path); left columns filtered
  by `HashJoin::canRemoveColumnsFromLeftBlock` + `TableJoin::getOutputColumns(JoinTableSide::Left)`,
  then payload columns (dedup by output name via `TableJoin::renamedRightColumnName`), then required
  right keys copied from the left source (`TableJoin::getRequiredRightKeys`) with `castColumn` when
  types differ.
- **ProbeScratch** (191–213): per-lane reusable buffers — packed keys, `left_rows`/`refs` match arrays,
  counting-sort arrays (`block_start`, `cursor`, `sorted_left_rows`, `sorted_row_no`, `runs`
  (`BlockRun{block_no, begin, end}`)), reused `ColumnUInt32` index columns.
- **probeBlock** (332–386): single-col keys probe the raw column data directly (no packing); multi-col
  keys packed via `packBatch<W>` in 1024-row tiles; `collectMatches` fills matches; then hybrid gather
  decision — if `leaf_tables.any_duplicates`, counting-sort matches by build block
  (`sortMatchesByBlock`, 274–304) for bulk per-block gathers, else keep probe order.
- **gatherLeft** (390–417): one `IColumn::index(left_index, 0)` bulk gather per output left column;
  required right keys gathered from the left source + optional `castColumn`.
- **gatherRight** (488–523): duplicate-free path = `gatherPayloadDirect` per payload column — typed
  no-virtual-dispatch loop for the 10 fixed-width numeric types (`gatherNumericDirect<T>`, 424–440,
  `assert_cast<ColumnVector<T>>`), per-row `insertFrom` fallback otherwise; duplicate path = per
  `BlockRun` one `IColumn::index` + `insertRangeFrom`. Payload resolved via
  `stored_columns[ref.blockNo()]->columns[col_idx]` where `stored_columns` is
  `StoredColumnsIndex::blocksData()` and `col_idx = payload_right_indexes[payload_idx]`
  (position in `right_sample_block`).
- **State** (527–579): `PartitionPlan plan`; key names/positions/widths/offsets/packers;
  `unique_ptr<BuildSide>`; dedicated `unique_ptr<ThreadPool> pool` (created in ctor with the three
  `RadixHashJoinPool*` CurrentMetrics, max_free_threads=0, queue_size=max_threads);
  `atomic<bool> built`; `LeafTables leaf_tables`; `StoredColumnsIndexPtr stored_columns_index` +
  `vector<unique_ptr<ColumnsInfo>> columns_infos` + `payload_right_indexes`; `block_base`,
  `total_rows`, `total_bytes`; output-plan blocks (`right_table_keys`, `columns_to_add`,
  `required_right_keys` + name vectors, `remove_left_columns`, `left_output_names`);
  `once_flag plan_once`; `vector<unique_ptr<ProbeScratch>> lane_scratch` (size max_threads, lazily
  filled, lock-free per lane).
- **Constructor** (581–674): re-checks gate; computes `key_offsets` (`std::exclusive_scan`), packers
  (`chooseColumnPacker`); `PartitionPlan::choose(rhs_size_estimation, detectL2Bytes(), max_partitions_per_pass)`;
  creates `BuildSide`; splits schema via `JoinCommon::splitAdditionalColumns`.
- **addBlockToJoin(block, num_rows, check_limits, build_lane)** (693–709): normalises the right block
  to `right_sample_block` column order by name, `materializeBlock`, then `build_side->add(normalized, lane)`.
  Always returns true — **no memory-limit check (`check_limits` ignored) and no memory-tracking-based
  rejection**. Charged to `RadixHashJoinBuildMicroseconds`.
- **onBuildPhaseFinish** (722–737): `finishBuild()` (histogram merge barrier) then `runPostBuild()`.
  Called by `JoiningTransform::work()` exactly once before any real probe.
- **runPostBuild** (739–829): captures `getCurrentThreadGroup()`; wraps pool in a `ParallelFor` lambda;
  consults `getHashTablesStatistics<RadixHashJoinEntry>().getSizeHint(stats_collecting_params)` — warm
  run skips HLL and reconstructs per-leaf estimates proportionally from cached `distinct_keys`
  (775–789); `scatterToLeaves(parallel_for, num_workers, run_hll)`; `buildLeafTables(...)`; cold run
  publishes summed estimate via `.update(...)` (796–803); registers every stored block into a fresh
  `StoredColumnsIndex` (block_no == accumulation index, chasserted); `total_bytes =
  leaf_tables.arena.bytesReserved()`; frees the scatter `LeafArrays`; `built.store(true, release)`.
- **joinBlock(Block, lane)** (836–902): `materializeBlock`; `call_once` output plan; `can_probe =
  built.load(acquire) && n > 0` (header path emits schema only); lane bounds-checked against
  `lane_scratch.size()` (throws LOGICAL_ERROR); `probeBlock` + `gatherLeft` + `gatherRight`; returns
  `IJoinResult::createFromBlock(Block(std::move(out)))`. Wrapped in
  `RadixHashJoinProbeMicroseconds` watch + `ScopedLLCMissCounter(RadixHashJoinProbeLLCMisses)`.

Depends on (outside the dir): `Interpreters/RowRefs.h` (BuildRef/BuildRefList/StoredColumnsIndex/
ColumnsInfo), `Interpreters/TableJoin.h`, `Interpreters/JoinUtils.h`, `Interpreters/HashJoin/HashJoin.h`
(only for `canRemoveColumnsFromLeftBlock`), `Interpreters/HashTablesStatistics.h`,
`Interpreters/castColumn.h`, `Core/Block.h`, `Columns/*`, `DataTypes/IDataType.h`,
`Common/ElapsedTimeProfileEventIncrement.h`, `Common/ProfileEvents.h`, `Common/ScopedLLCMissCounter.h`
(donor-added file), `Common/Stopwatch.h`, `Common/ThreadPool.h`, `Common/ThreadGroupSwitcher.h`,
`Common/assert_cast.h`, `Common/setThreadName.h` (ThreadName::RADIX_JOIN, donor-added enum member).
6 ProfileEvents + 3 CurrentMetrics referenced (see wiring section).

### BuildSide.h (203) / BuildSide.cpp (878) {#buildside}

Self-contained given `Block` + the sibling headers; no IJoin coupling (good porting unit).

- `LeafArrays` (h:30–60): per-leaf fused-record arrays. Fields: `num_leaves`, `key_width`,
  `record_width` (= key_width + 8), `record_base` (null for empty leaf), `leaf_rows` (== global
  histogram), `distinct_key_estimates` (empty when HLL off; clamped to `[1, leaf_rows]`), diagnostics
  `alloc_count` / `bytes_scattered`, owning `Arena`. Accessors `keyAt(leaf,i)` / `refAt(leaf,i)`
  (record layout is ref-first: `PACKED_KEY_OFFSET_IN_RECORD = sizeof(BuildRef)` = 8).
- `class BuildSide` — 3 phases:
  - `add(const Block &, size_t lane)` (cpp:331–381): lane-indexed `LocalState` (throws LOGICAL_ERROR if
    `lane >= max_threads`); COW-moves the block (zero copy); computes 32-bit route words per row into
    reused scratch (single-col: hash the raw column; multi-col: pack 1024-row chunks then hash);
    accumulates a **replicated histogram** (`chooseReplicas`: up to 4 replicas while ≤32 KiB, cpp:33–37)
    to dodge store-to-load-forward stalls. Hash never stored per row.
  - `finishBuild()` (cpp:383–433): concatenates per-lane block stores in slot order (final `block_no` =
    concatenation index; `chassert(num_blocks <= BuildRef::BLOCK_NO_MASK)`), folds replicated histograms
    into `global_hist`, computes `block_base` prefix sums (`numBlocks()+1`, back()==totalRows()),
    records each used slot's contiguous block range (`used_slots`/`slot_block_begin`/`slot_block_end`).
  - `scatterToLeaves(const ParallelFor &, size_t num_workers, bool estimate_distinct_keys)`
    (cpp:834–876): sets up transient `HllScatterState` (per-worker × per-leaf flat register array,
    cpp:269–287) when HLL on; dispatches single-pass (`pass_bits.size() <= 1`) or multi-pass; then
    merges per-worker HLL sketches per leaf (register-wise max, parallel over leaves) into
    `distinct_key_estimates` clamped to `[1, rows]`.
- `scatterSinglePass` (cpp:577–619): `allocExactPartitions` — ONE exact-sized line-padded allocation per
  non-empty leaf (the no-churn property, `alloc_count` asserted by tests); per-(slot, leaf) start
  offsets from replaying per-slot histograms; then `scatterBlockRanges`.
- `scatterBlockRanges` (cpp:446–575): parallel over used slots; each worker walks its own block range in
  `SCATTER_CHUNK_ROWS = 1024` chunks: build `BuildRef(block_idx, row)` array, pack keys (multi-col) or
  use raw column, `computeRoutes` / `computeRoutesAndBuckets` (HLL final pass shares one hash for route
  + bucket), `fuseKeyRefChunk` into `[ref|key]` records, then `appendColumnSwwc` (if
  `shouldUseSwwc(num_parts)`, seeding cursors + head-peel counts for line alignment) or
  `appendColumnDirect`.
- `scatterMultiPass` (cpp:729–832): pass 0 scatters blocks into `2^pass_bits[0]` intermediate
  partitions (own arena, per-slot offsets); then parallel per partition: recursive depth-first
  `refine` (cpp:621–727) re-derives routes from records (`computeRoutesStrided`), scatters into child
  arrays (`allocExactPartitions` on a local child arena) or, on the last pass, directly into the final
  leaf `record_base` slots; consumed intermediates freed via `Arena::release` as recursion unwinds
  (peak memory ~ live working set). HLL accumulated only on the last pass. `RefineScratch`
  (cpp:252–262): reused `ScatterScratch` + route/bucket vectors sized to max refine fanout.
- 16-way width dispatch tables (4..64 step 4) for `computeRoutes*`, `fuseKeyRefChunk` (cpp:41–213).

Depends on: `Core/Block.h`, `Columns/IColumn.h`, `Common/Exception.h`, siblings
(Arena/KeyLayout/KeyRefScatter/ParallelFor/PartitionPlan/Hll/PackedKeyHash), `Interpreters/RowRefs.h`
via KeyRefScatter.h (`BuildRef`).

### LeafTable.h (190) / LeafTable.cpp (707) {#leaftable}

The heart of the algorithm; self-contained given `RowRefs.h` + `Common/Arena.h`.

- **Cell layout**: `[ BuildRefList word (8 B) | packed key (key_width B) ]`;
  `leafCellBytes(kw) = sizeof(BuildRefList) + kw` (h:100–103). Empty sentinel: word == 0. Unique key:
  the word IS the encoded singleton `BuildRef` (bit 63 = inline flag) — probe emits with no extra load.
  First duplicate allocates a `BuildRefList::Batch` node from the owning build worker's `DB::Arena`
  (per-worker arenas in `LeafTables::build_arenas`, single-writer).
- **Key widths**: every multiple of 4 in [4, 64] — 16 template instantiations in both the build fill
  (`RHJ_FILL_DISPATCH`, cpp:226–238) and probe (`RHJ_PIPE_DISPATCH`, cpp:503–516); unsupported width →
  LOGICAL_ERROR.
- **Grouped leaves** (h:47–97): `MAX_GROUP_BITS = 8` → at most `MAX_UNIQUE_BUCKET_SIZES = 256` groups;
  each group = consecutive leaf range sharing one bucket count (snapped to its largest member) and ONE
  arena allocation; `LeafHT{char * cells, UInt64 num_buckets}` (16 B, static_asserted); group
  metadata ≤4 KB so it stays L1-resident. Routing: `g = leaf >> local_shift`,
  `local = leaf & ((1<<local_shift)-1)`, leaf stride = `roundUpToLine(nb * stride)`.
- **`LeafTables`** (h:114–158): `GroupedLeaves grouped`, `build_arenas`
  (`vector<unique_ptr<DB::Arena>>`), `atomic<bool> any_duplicates` (selects the probe's grouped-gather
  path), `num_rows`, `max_bucket_bits` (probe uses UInt32 ring slots iff ≤ 31), `cell_alloc_count`,
  owning `RadixJoin::Arena arena`. Move-only with hand-written moves (atomic member).
- **`buildLeafTables(const LeafArrays &, UInt64 num_rows, size_t key_width, size_t num_workers, const ParallelFor &)`**
  (cpp:550–682): per-group sizing = `bit_ceil(max_member_sizing * 2)` (load factor 0.5) where sizing =
  distinct estimate if present else row count; one allocation per non-empty group (parallel over
  groups); fill parallel over leaves (memset the leaf range, then `fillLeafDispatch`); **overflow
  guard**: the AMAC build refuses to claim the last empty cell (`claimed == mask`, cpp:167–171) — an
  undersized leaf (HLL under-estimate) flags `overflowed` and the whole group is REBUILT with safe
  row-count sizing (`bit_ceil(max_rows * 2)`, cpp:637–664) — this guarantees every linear-probe walk
  terminates.
- **AMAC build insert** (`amacRing` cpp:55–103 + `BuildPolicy` cpp:114–188): generic power-of-two ring
  (ring_size 32) keeping N rows in flight, ONE fused read→act step per visit, prefetch next cell for
  the next visit. Correctness note: the read and act MUST be one indivisible step or two in-flight
  same-key rows could both claim a cell (documented cpp:49–54; stress-tested by
  `BuildProbeHeavyDuplicatesFewKeys`). `PosT` templated UInt32/UInt64 (leaf > 2^31 buckets fallback).
- **AMAC probe** (`collectMatchesPipelined`, cpp:343–489): the single probe path for all widths /
  PosT / dup-ness. Ring of 32 `PipelineSlot{cells, pos, mask, row}`; `probeHomeCell` (cpp:259–288)
  hashes + routes + decodes home cell inline on admit (no seed pre-pass — CRC32 is cheap enough);
  steady phase sweeps the ring with no per-visit active check, then a drain phase; match emission via
  raw cursors (`OutPtrs`, grow ~2x out-of-line `growOutPtrs` / `emitMatchListCold` for multi-row keys,
  cpp:295–327); output buffers pre-resized to +n (singleton lower bound) and shrunk after.
- **Entry point**: `collectMatches(key_width, grouped, leaf_shift, total_bits, packed_keys, n, pos_fits_u32, out_left_rows, out_refs)`
  (h:179–188, cpp:684–705).

Depends on: `Interpreters/RowRefs.h` (via BuildSide.h/KeyRefScatter.h: `BuildRef`, `BuildRefList` with
`word`, `insert(UInt64 ref_word, DB::Arena&)`, iterator `begin()/ok()/++`, `refWordIsInline`),
`Common/Arena.h` (DB::Arena for Batch nodes), `Common/Exception.h`, `base/types.h`.

### KeyRefScatter.h (122) / KeyRefScatter.cpp (359) {#keyrefscatter}

Pure scatter kernels; only external deps `Interpreters/RowRefs.h` (re-exports `using DB::BuildRef`),
`Common/Exception.h`, `Common/TargetSpecific.h`. `LINE_BYTES = 64`, `roundUpToLine`.

- `appendColumnDirect(route, shift, mask, n, src, elem_width, void** cursors)` — incremental typed
  scatter, compile-time widths {4,8,16,32,64} + 4-byte-lane fallback (cpp:33–71).
- SWWC+NT path (x86 only, `DECLARE_MULTITARGET_CODE` v3/v4, cpp:73–235): `appendTiledSwwc<W>` for
  widths dividing 64; `appendStreamSwwc` for whole-line multiples; `appendGenericSwwc` for the rest
  (handles line-straddling records + head alignment); flush = `__builtin_nontemporal_store` of a
  64-byte vector; `drainColumnSwwc` writes residual partial lines + `seq_cst` fence (mfence).
- `ScatterScratch` (h:68–98, cpp:253–310): per-worker staging — `posix_memalign` staging lines +
  cursors + fills + head-peel counters; move-only; `resetFills`.
- `ntStoresAvailable()` = `isArchSupported(x86_64_v4 || x86_64_v3)`; `shouldUseSwwc(p)` = NT available
  && p >= 256 (measured crossover, cpp:248–251). **ARM: always DIRECT.**

### Arena.h (78) / Arena.cpp (102) {#arena}

`DB::RadixJoin::Arena` — bag of exact-sized aligned blocks from `Allocator<false, false>` (plain
jemalloc: no zero-fill, no mmap/THP — deliberate, documented rationale h:15–29). Thread-safe (mutex
only around the block list; alloc itself lock-free). `allocate(bytes, alignment)` (0 rounds to 1),
`allocateArray<T>`, `release(void*)` (O(n) scan, swap-with-back; used to drop consumed multi-pass
intermediates), `blockCount()`, `bytesReserved()` (drives `getTotalByteCount`). Movable via
`unique_ptr<mutex>`. Depends only on `Common/Allocator.h`, `base/types.h`, `base/defines.h`.
NOTE: allocations are NOT tracked by any per-query `MemoryTracker` beyond what `Allocator` itself does
(Allocator<false,false> uses CurrentMemoryTracker via alloc — actually `Allocator::alloc` does track;
but there is no `check_limits` integration at the join level).

### PackedKeyHash.h (111) {#packedkeyhash}

`HashT = UInt32`; one CRC32C hash serves routing (top `total_bits`) and bucketing (low bits) —
invariant: `total_bits + log2(buckets) <= 32`. `hashPackedKey<width>` uses `HashCRC32<T>` for widths
with an integer type (4→UInt32, 8→UInt64, 16→UInt128, 32→UInt256; 1/2 for table completeness) and
`updateWeakHash32(bytes, width, -1)` (byte-span CRC32C from `Common/HashTable/Hash.h`) for composite
widths (12, 20, 24, …). Runtime-width switch entry too. `routeBits(h) == bucketBits(h) == h`.
Depends: `Common/HashTable/Hash.h`, `base/types.h`.

### KeyLayout.h (61) {#keylayout}

`ColumnPackFn = void(*)(src_raw, row_begin, rows, dst, stride, dst_offset, width)`;
`packColumnFixed<4|8|16|32|64>` (`__builtin_memcpy_inline`) + `packColumnLanes` (4-byte lanes);
`chooseColumnPacker(width)`. Shared by build and probe so both produce byte-identical packed keys.
Single-column keys skip packing entirely (raw column data IS the key). Header-only, deps `base/types.h`.

### Hll.h (102) {#hll}

Header-only dense HLL, namespace `Hll`: precision 4–6 (`MAX` = 64 registers = one cache line),
32-bit input (`INPUT_BITS`), `MEMORY_BUDGET_BYTES = 32 MiB` bounds `workers × leaves × 2^p`
(`choosePrecision`). `add(registers, precision, hash)` (top-p bits index, remaining bits rank),
`merge` (register-wise max — exactly mergeable across worker partials), `estimate` (α_m harmonic mean
+ linear-counting small-range correction). Storage caller-provided (flat byte array). Deps `base/types.h`.

### PartitionPlan.h (59) / PartitionPlan.cpp (66) {#partitionplan}

`PartitionPlan{num_leaves, total_bits, leaf_shift, pass_bits}` + constants: `ROUTE_BITS = 32`,
`MAX_LEAVES = 2^20`, `CELL_BYTES = 16`, `LOAD_FACTOR = 0.5`, `L2_HEADROOM = 0.8`,
`DEFAULT_LEAVES = 256` (no estimate), `L2_FALLBACK_BYTES = 256 KiB`.
`choose(rhs_rows_estimation, l2_bytes, max_partitions_per_pass)`: table bytes ≈ rows × 32 B; leaves =
`ceil(table_bytes / (0.8 × L2))` rounded up to power of two, clamped to [1, 2^20]; per-pass bits =
`floor(log2(cap))` clamped [1,16]; bits spread evenly across passes (max−min ≤ 1, remainder on early
passes). Deps: `base/types.h`, `base/defines.h`.

### ParallelFor.h (28) {#parallelfor}

`UnitFn = std::function<void(size_t unit, size_t worker)>`; `ParallelFor = std::function<void(size_t
total, const UnitFn &)>`. Contract (documented): dense stable worker ids in [0, num_workers) enabling
single-writer per-worker resources; dynamic load balancing; exception propagation after all workers
stop; total==0 no-op. Production impl = `runParallelFor` over the join's ThreadPool; tests use
`std::thread`s.

### tests/gtest_radix_hash_join.cpp (856) {#gtest}

Server-free unit tests of BuildSide + LeafTable + scatter + HLL + plan (no RadixHashJoin/IJoin
instantiation — no TableJoin/Context needed). Auto-collected into `unit_tests_dbms` by the gtest glob.
Includes `Columns/ColumnsNumber.h`, `Columns/ColumnFixedString.h`, `Core/Block.h`, DataTypes, gtest.
Key tests: `PackedKeyHashDeterministicAndSpread`, `PartitionPlanSizingAndPasses`,
`ScatterColumnRoundTripDirect`, `BuildProbeUniqueKeys`, `BuildProbeManyToManyParallel` (4 build lanes ×
4 post-build workers), `BuildProbeHeavyDuplicatesFewKeys` (AMAC same-key in-flight stress),
`BuildProbeForcedMultiPass`, `BuildProbeMultiPassSwwc` (skipped without NT stores),
`HllEstimateAccuracy`, `HllMergeEqualsSingleSketch`, `DistinctEstimateShrinksDuplicateHeavyBuild`,
`DistinctEstimateNeverUndersizesLeaf` (regression: overflow rebuild / infinite-loop guard),
`BuildProbeGroupedLeaves4096`, `GroupedLeavesEmptyLeafSlotInitialized` (F1 regression: probing empty
sibling leaves in non-empty groups), `BuildProbeGroupedLeaves64ByteKeys` (stride 72),
`GroupedLeavesLoadInvariant` (load ≤ 0.5), and parametrised `FusedKeyWidthFanout` over widths
{4,8,12,16,24,32,56,64} × leaves {16,512,8192} (DIRECT + SWWC). Asserts the no-churn property
(`alloc_count == non-empty leaves`) and grouped-metadata bounds (≤256 groups, `cell_alloc_count ≤ 512`).

### Benchmarks {#benchmarks}

- `bench_rhj_vs_chj.cpp` (881; the only one wired into CMake): standalone `main()`, links `dbms`;
  constructs real `RadixHashJoin` and `ConcurrentHashJoin` through the IJoin API (mirrors
  PlannerJoins construction), pinned worker threads (assumes 48 physical cores), perf_event PMU groups
  for probe-phase attribution, reference query `SELECT count() FROM probe INNER JOIN build USING (k0)`
  with UInt64 keys. Uses `TableJoin` built manually (`makeTableJoin`), `ThreadStatus`. Modes:
  instrumented run / report sweep.
- `bench_build_bandwidth.cpp` (565; NOT wired): compares CHJ zero-copy dispatch (re-implemented
  in-bench since `dispatchBlock` is private; uses `Interpreters/HashJoin/ScatteredBlock.h`) vs
  `BuildSide` scatter vs a flat memcpy baseline over identical blocks.
- `bench_scatter_perf.cpp` (665; NOT wired): PMU-attributed (perf_event + `getrusage(RUSAGE_THREAD)`)
  breakdown of scatter vs memcpy phases; drives genuine `BuildSide`. Both un-wired benches include
  only RadixHashJoin headers + Columns/Core/DataTypes + `Common/ThreadStatus.h` + fmt.

## Out-of-directory wiring on the donor (must be re-created when porting) {#wiring}

Full donor-vs-merge-base change list outside the directory (from `git diff --name-status
e69a9d5ba75..origin/phj5-real`):

1. **`src/CMakeLists.txt`**: `add_object_library(clickhouse_interpreters_radix_hash_join Interpreters/RadixHashJoin)` (line 339).
2. **`src/Interpreters/CMakeLists.txt`**: `ENABLE_BENCHMARKS` → `add_subdirectory(benchmarks)`.
3. **`src/Core/Joins.h/.cpp`**: `JoinAlgorithm::RADIX_HASH` enum member + `toString` case.
4. **`src/Core/SettingsEnums.cpp`**: `"radix_hash"` mapping.
5. **`src/Core/Settings.cpp`**: `max_partitions_per_pass` (UInt64, default 8192) and
   `radix_hash_join_size_tables_by_distinct_estimate` (Bool, default true).
6. **`src/Core/SettingsChangesHistory.cpp`**: entries for both settings.
7. **`src/Planner/PlannerJoins.cpp`**: `radixHashJoinApplicable` gate (single-disjunct, Inner, All
   strictness, no special storage, all keys fixed-width/non-nullable/non-LC, packed width %4==0 in
   [4,64]) + `createRadixHashJoinFallback` (parallel_hash when shape allows, else hash) + `tryCreateJoin`
   branch constructing `RadixHashJoin`; `JoinAlgorithmParams` gains both settings (also
   `src/Planner/PlannerJoins.h:251`).
8. **`src/Interpreters/JoinOperator.h/.cpp`** + **`src/Processors/QueryPlan/QueryPlanSerializationSettings.cpp`**:
   both settings plumbed through `JoinSettings` and plan serialization.
9. **`src/Interpreters/IJoin.h`**: two new virtual overloads with lane defaults —
   `addBlockToJoin(block, num_rows, check_limits, build_lane)` and `joinBlock(block, lane)`.
10. **`src/QueryPipeline/QueryPipelineBuilder.cpp`**: `joinPipelinesRightLeft` assigns a unique
    `build_lane` to each `FillingRightJoinSideTransform` and passes stream index `i` to each
    `JoiningTransform`; `joinPipelinesByShards` passes shard index.
11. **`src/Processors/Transforms/JoiningTransform.h/.cpp`**: `JoiningTransform` gains `stream_index_`
    ctor arg → `joinBlock(block, stream_index)`; `FillingRightJoinSideTransform` gains `build_lane_`
    → `addBlockToJoin(block, num_rows, true, build_lane)`.
12. **Other IJoin implementors** (`HashJoin.h`, `ConcurrentHashJoin.h`, `GraceHashJoin.h`,
    `MergeJoin.h`, `DirectJoin.h`, `FullSortingMergeJoin.h`, `PasteJoin.h`, `JoinSwitcher.h`,
    `SpillingHashJoin.h`): add `using IJoin::addBlockToJoin; using IJoin::joinBlock;` to unhide the
    new base overloads.
13. **`src/Interpreters/HashTablesStatistics.h/.cpp`**: `RadixHashJoinEntry{size_t distinct_keys}`
    with `shouldBeUpdated` (update when halved or grew) + explicit template instantiation +
    cache-stats aggregation.
14. **`src/Common/setThreadName.h`**: `M(RADIX_JOIN, "RadixJoin")` thread-name enum entry.
15. **`src/Common/ProfileEvents.cpp`**: 6 RHJ events (`RadixHashJoinBuildMicroseconds`,
    `ProbeMicroseconds`, `ProbeLLCMisses`, `ProbePermMicroseconds` (declared but the perm stage was
    removed — only referenced in RadixHashJoin.cpp's externs), `ProbeCollectMatchesMicroseconds`,
    `ProbePackHashRouteMicroseconds`) + 3 CHJ events (Build/Probe/ProbeLLCMisses) added to
    ConcurrentHashJoin.cpp for A/B accounting.
16. **`src/Common/CurrentMetrics.cpp`**: `RadixHashJoinPoolThreads{,Active,Scheduled}`.
17. **`src/Common/ScopedLLCMissCounter.h/.cpp`** (new, 33+78 lines): per-thread lazily-opened
    perf_event LLC-miss counter; silent no-op when unavailable.
18. **Stateless tests**: `tests/queries/0_stateless/04316_radix_hash_join_gate_and_fallback.sql/.reference`
    (radix_hash vs hash equality across gate/fallback shapes incl. composite keys, String/LC/Nullable
    fallbacks) and `04337_radix_hash_join_distinct_estimate.sql/.reference` (HLL sizing on/off result
    equality). Both `no-random-settings`, `SET enable_analyzer = 1`.

## Divergences: donor base vs our HEAD (port adaptation points) {#divergences}

Verified against our HEAD (`radix-join-bandwidth-model`, clean tree):

1. **`BuildRef` / `BuildRefList` were renamed upstream to `RowRef` / `RowRefList`**
   (`src/Interpreters/RowRefs.h` on HEAD: `RowRef` at :40 with the same 8-byte
   `{row_no, block_no|INLINE_FLAG}` encode/decode, `refWordIsInline` at :74, `RowRefList` at :96 with
   `word`, `insert(UInt64 ref_word, Arena & pool)` and 64-byte `Batch`). Same semantics, different
   names; the donor code's `DB::BuildRef`/`DB::BuildRefList` and `BuildRef::fromWord` usages must be
   renamed. HEAD's `RowRef` ctor **throws** (`throwRowRefOutOfRange`) instead of chasserting.
2. **`ColumnsInfo` no longer exists**: HEAD's `StoredColumnsIndex::add` takes
   `const StoredBlock *` (defined in `src/Interpreters/HashJoin/ScatteredBlock.h:376`, has `Columns
   columns` + `replicated_columns` + `selector` + `block_no`). `blocksData()` returns
   `const StoredBlock * const *`. The donor's `State::columns_infos` +
   `stored_columns[b]->columns[col_idx]` gathers must switch to `StoredBlock` (note
   `ColumnReplicated` handling may be needed if stored blocks can contain replicated columns; for RHJ's
   own materialized normalized blocks they will not be).
3. **Lane plumbing absent on HEAD**: `IJoin.h` has no 4-arg `addBlockToJoin` / 2-arg `joinBlock`;
   `FillingRightJoinSideTransform` has no `build_lane`; `JoiningTransform` has no stream index passed
   to `joinBlock` (HEAD's `stream_index` at JoiningTransform.h:188 belongs to
   `NonJoinedBlocksTransform`, a different class). Items 9–12 above must be re-applied, watching for
   drift in `QueryPipelineBuilder::joinPipelinesRightLeft` (HEAD version has evolved).
4. **`JoinAlgorithm` enum on HEAD** ends at `FULL_SORTING_MERGE` (Joins.h:120–131); adding
   `RADIX_HASH` is additive.
5. Present-and-compatible on HEAD (verified): `JoinCommon::splitAdditionalColumns`
   (`JoinUtils.h:117`), `HashJoin::canRemoveColumnsFromLeftBlock` (`HashJoin/HashJoin.h:538`),
   `updateWeakHash32` (`Common/HashTable/Hash.h:145`), `ThreadGroupSwitcher(ThreadGroupPtr,
   ThreadName, bool)` + `getCurrentThreadGroup` (`Common/ThreadGroupSwitcher.h:20,41`),
   `IJoinResult::createFromBlock` (`IJoin.h:66`), `HashTablesStatistics<Entry>` template
   (`HashTablesStatistics.h:90`), `StatsCollectingParams`.
6. **Neither `src/Interpreters/RadixHashJoin/` nor `src/Interpreters/benchmarks/` exists on our
   branch**, and our `src/Interpreters/CMakeLists.txt` has no `ENABLE_BENCHMARKS` block.
7. `ScopedLLCMissCounter`, `ThreadName::RADIX_JOIN`, the ProfileEvents/CurrentMetrics, and
   `RadixHashJoinEntry` do not exist on HEAD — port items 13–17 wholesale.

## Notable behavioural facts / gotchas {#gotchas}

- `addBlockToJoin` ignores `check_limits` and always returns true; memory is only accounted via
  jemalloc/CurrentMemoryTracker inside `Allocator` — there is no join-level `max_bytes_in_join`
  enforcement. Relevant to the planned streaming budget-bounded probe.
- The join is **eagerly and fully built in `onBuildPhaseFinish`**; `joinBlock` never builds and before
  the barrier emits schema only (header path). `getDelayedBlocks` is NOT overridden (no such donor
  concept; non-joined blocks are `{}` since inner-only).
- The probe emits the ENTIRE match set of a block as one output Block (no `max_block_size` splitting
  inside `joinBlock` — `IJoinResult::createFromBlock` wraps a single block). With duplicate-heavy
  builds output can be much larger than the input block.
- One 32-bit hash serves route + bucket: any port change to hashing must preserve
  `total_bits + max_bucket_bits <= 32` (checked implicitly by plan; MAX_LEAVES 2^20 leaves 12 bits).
- SWWC/NT path is x86-only (`USE_MULTITARGET_CODE`); ARM silently uses DIRECT everywhere
  (`ntStoresAvailable() == false`), including for fanouts ≥ 256.
- `PartitionPlan` sizing is payload-independent (~32 B/row) and uses the private L2 via sysconf
  (0 → 256 KiB fallback).
- The dedicated `ThreadPool` (max_threads, queue max_threads) is created per join instance in the
  constructor — mirrors ConcurrentHashJoin's pool pattern.
- `RadixHashJoinProbePermMicroseconds` is declared/registered but the "perm" stage no longer exists;
  only `Pack`/`CollectMatches` are actually incremented in `RadixHashJoin.cpp`.
- The build normalises + materializes every right block (`materializeBlock`), so Sparse/Const columns
  are expanded at build time; probe side also materializes each block.
- `BuildSide::add` requires `n <= UInt32::max` per block and `finishBuild` chasserts
  `num_blocks <= 2^31` (BLOCK_NO_MASK).
- Multi-pass scatter (`pass_bits.size() > 1`) is only triggered when `total_bits >
  floor(log2(max_partitions_per_pass))` — with default cap 8192 (13 bits) and MAX_LEAVES 2^20 up to
  2 passes in practice.
