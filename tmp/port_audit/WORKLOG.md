# Port audit worklog — RadixHashJoin → PartitionedHashJoin (branch `ahj`)

## Pinned SHAs (iteration 1, 2026-07-21)

- `AHJ_SHA  = 6143ed95a2ba782dbb0166ea6ecf2b8a756d26aa` (branch `ahj`, also HEAD of working tree at pin time; tree clean)
- `RBM_SHA  = ca217fc57eb8be194c95a008ec933972565a21ff` (branch `radix-join-bandwidth-model`)
- `PHJ5_SHA = 82005a0cc2600382a5699a0576bc871eafad230d` (branch `origin/phj5-real`)

All audit reads go through `git show`/`git log`/`git diff` pinned to these SHAs. Never the working tree.

## Iteration 1 — pin, topology, sharding

**Goal:** pin SHAs, understand branch topology, shard the inventory for parallel subagents.

**Done / raw evidence:**
- `git log --oneline $AHJ_SHA..$RBM_SHA | wc -l` → **78** (matches mission's ~78). Exactly 1 merge:
  `9c56f97e008 Merge radix-join-probe-perf: probe-time accounting campaign and the kept wave-worker output merging` (campaign-internal merge, not upstream).
- `git log --oneline $AHJ_SHA..$PHJ5_SHA | wc -l` → **255**, NOT the ~44 the mission stated. Topology explains it:
  - `git merge-base ahj rbm` = `2834291df98`; `git merge-base ahj phj5` = `cfc6ee0252c`; `git merge-base rbm phj5` = `2834291df98`; commit-set overlap rbm∩phj5 = 0.
  - `phj5-real` = upstream master at a later point than ahj's base + **43 linear feature commits** (`git log --first-parent --no-merges $AHJ_SHA..$PHJ5_SHA | wc -l` → 43; `git rev-list --count 85fa486b1e6^..$PHJ5_SHA` → 43, linear tail starting at `85fa486b1e6 RadixHashJoin: add the radix_hash algorithm value, setting and ProfileEvents`).
  - Remaining **212 commits** = upstream ClickHouse master commits (PR merges + their second parents) pulled in because phj5-real branched from a newer master. Hash set saved: `tmp/port_audit/phj5_upstream_hashes.txt`.
- Commit lists saved: `rbm_commits.txt` (78), `rbm_commits_chrono.txt`, `phj5_commits.txt` (255), `phj5_feature_hashes.txt` (43), `phj5_upstream_hashes.txt` (212).

**Decision D1 (ambiguity call):** the 212 upstream master commits are not campaign optimizations; they will occupy ONE matrix row (all 212 hashes listed) with disposition `not-applicable` — "upstream ClickHouse master commits included via phj5-real's later branch point; `ahj` receives upstream work by merging master, not by this port". Revisit trigger: if any inspected phj5 feature commit turns out to depend on an upstream join-infrastructure change (e.g. the `BuildRef`→`RowRef` rename in `82005a0cc26` adapts to upstream), note the dependency in that row's evidence.
**Decision D2:** mission said ~44 phj5 commits; pinned range says 255. Deviation documented here; the audit covers the full pinned range (checker enforces all 333 hashes), with the upstream block grouped per D1.

**Environment notes:** concurrent sessions active in this checkout (a cursor session investigating PHJ memory usage is running benchmarks; another did a partial read-only RBM inventory — treated as LEADS only, not evidence). Working tree may change under us; that is why all reads are pinned. Machine contention matters for Phase B benchmarking — recheck before G3 runs.

## Iteration 2 — Unit 0 inventory fan-out (in progress)

**Plan:** parallel subagents via Workflow, all restricted to pinned read-only git plumbing:
- 4 agents over RBM's 78 commits (chronological shards ~20 each)
- 3 agents over PHJ5's 43 feature commits (shards ~14 each)
- 2 full-tree diff agents (`git diff $AHJ_SHA $RBM_SHA -- src/`, `git diff $AHJ_SHA $PHJ5_SHA -- <campaign paths>`) to catch what commit messages hide
- 2 ahj-state agents inventorying mechanisms at `$AHJ_SHA` (PartitionedHashJoin core; ColumnsScatter/AmacRing/DenseHLL)
Merge → PORT_MATRIX.md → check_matrix.py → G0 → G0b spot checks (mine, not delegated).

**Done:** workflow `wf_2cec28ee-445`, 11/11 agents completed, 0 errors (1.15M subagent tokens, 268 tool calls, ~11 min). Raw outputs: `tmp/port_audit/agents/inventory.json` (+ `remaining_groups.txt`, `diff_observations.txt` extracts). All 121 campaign commits covered by shard groups exactly once (cross-checked by summing group commit counts per shard).

## Iteration 3 — merge, conflict resolution, matrix, gates G0 + G0b

**Key inventory corrections vs the mission's context (all evidence-backed):**
- RBM's final state has NO AMAC ring, NO HyperLogLog, NO fused route loops — its stale ProfileEvents descriptions are fossils of the phj5 era (diff-rbm obs 16). Those "already ported" items verify against `$PHJ5_SHA`, and did.
- "Greedy MSB-first pass split" from the mission context is NOT ahj's final state: `a630d83c74f` restored the BALANCED split after `f11bc53a918` tried greedy; RBM's split policy is the same balanced formula (diff-rbm obs 1). Parity holds; nothing missing.
- The "8-byte `LeafHT`" (phj5 `18c75e08328`) was UN-packed back to 16 bytes by phj5 itself (`2f3d227ec96`, `static_assert(sizeof(LeafHT) == 16)` at `$PHJ5_SHA` verified). Kept as a candidate anyway — rationale in the matrix row (ahj's descriptor gather is the measured probe stall center; phj5's revert reason was tied to its own probe evolution) — with the revert stated as a major caveat.
- "Tiled AMAC-over-seeds" and "ScatterScratch peel API" are not in phj5's final form as portable mechanisms (seeds pre-pass removed on-branch once CRC32C landed; peel API is surface+test, ahj covers alignment via the fill-window invariant).

**Conflicts between agents, resolved by my own `git show` at the pinned SHAs (raw outputs in chat log, iteration 3):**
1. Lane plumbing: phj1 said already-present (mechanism exists via thread-id map), rbm1 said port-candidate (pipeline-carried identity absent). First-hand: `$AHJ_SHA:src/Interpreters/IJoin.h` has no lane overloads; `PartitionedHashJoin.h:321-323` `fill_mutex` + `unordered_map<thread::id, FillLane*>`, `:372` `probe_scratch_mutex`. → ONE merged row `pipeline-lane-identity` (commits from both branches), port-candidate.
2. Grouped leaves: phj3 port-candidate vs diff-phj5 "replaced by slab+flat descriptors". First-hand: flat 16-byte `LeafMapDesc` per leaf confirmed → port-candidate stands, friction documented.
3. Narrow scatter counters (diff-rbm obs 8): first-hand `PartitionedHashJoinBuild.cpp:354-356` `PaddedPODArray<UInt64> worker_hist/starts` (refine path already UInt32 at `:1135`) → diff-derived candidate; origin commit `832ebbbc51f` stays in the multi-pass row.
4. Parallel teardown (diff-rbm obs 13): first-hand `~PartitionedHashJoin` = serial `leaf_maps.clear()` + slab free → low-priority diff-derived candidate.
5. NT-store gating (diff-phj5 obs 12): first-hand `ColumnsScatter.cpp:212` unconditional `__builtin_nontemporal_store`, no arch check in file → low-priority diff-derived candidate.
6. Warm-run distinct cache: first-hand `HashTablesStatistics.h` has only `AggregationEntry`/`HashJoinEntry`; zero references under `PartitionedHashJoin/` → candidate valid (RBM's twin entry is dead plumbing, diff-rbm obs 14 — correlated lineage, so source-branch benefit claims counted as ONE origin).
7. Payload gather: first-hand `AddedColumns.cpp` per-row `fillFromRowRefs`, no sort-by-block → candidate valid.

**Decisions:** D2 — checker accepts commitless rows whose Commits cell is exactly `diff-derived` (mechanisms surfaced only by the diff cross-check; their commits legitimately owned by other rows; disposition/evidence checks still apply). Keeps Phase B closure (zero `port-candidate`/`approved`) enforceable for them. D3 — wave-machine family dispositioned not-applicable (no probe-wave substrate at ahj: probe blocks never scattered/buffered) but flagged prominently in the Phase A report as a design-level alternative the requester may elect to evaluate; the source campaign's own gate-9 measurement (cooperative-only regressed T16 +30% vs pooled) recorded as evidence FOR ahj's current design. D4 — `largest-first-partition-drain-order` = already-present: the LPT mechanism exists at ahj on the build wave (`leaf_order`/`leaf_claim`); the probe-side application has no substrate (unpartitioned probe). D5 — source-branch worklogs/TLA/bench-only instrumentation = process artifacts per the accepted tradeoff; `ScopedLLCMissCounter` explicitly not relitigated into a port.

**Matrix:** 53 rows = 20 already-present + 8 port-candidate + 7 not-applicable + 18 process-artifact. Port-candidates: pipeline-lane-identity, probe-output-block-sorted-payload-gather, eager-postbuild-and-8b-leaf-descriptor, warm-run-cached-distinct-estimate, grouped-leaves-metadata-compression, narrow-scatter-counters, parallel-hash-table-teardown, nt-store-arch-gating. Leads L1–L4 recorded in the matrix (per-leaf HLL granularity, in-ring emission, wide-width SWWC, fanout knob).

**G0 (gate):** `python3 tmp/port_audit/check_matrix.py` → exit 0. Raw output: `OK: 78 rbm + 255 phj5 commits covered by 53 rows, all dispositions valid, all evidence non-empty`.

**G0b (gate), part 1 — every already-present row verified first-hand at `$AHJ_SHA`** (targeted `git show` greps + context reads; raw outputs in chat log): ScatterScratch `ColumnsScatter.h:147`; `computePassBits` `.cpp:1219` (balanced: `per_pass = ceil(total_bits/num_passes)`); exact reserve comment `.cpp:1243-1245`; Build.cpp kernel consumption `:880/:912/:954/:983`; `computePassBits` call `PartitionedHashJoin.cpp:360`; `refinePassWave :795/:1092`; `scatterSwwc :186` + NT store `:212`; `copyRowExact :131`; `scatterFallback :461+`; `StringScatterState :257`; `partitioned_hash` `SettingsEnums.cpp:62`; planner gate `PlannerJoins.cpp:1241` + plan-time fallback comment `:1196-1197`; IJoin hooks `:183-184` + overrides `PartitionedHashJoin.h:113-114` + per-lane fill `:44` + cheap barrier `PartitionedHashJoin.cpp:376`; `leaf_order`/`leaf_claim` `:388-389` + desc sort + largest-first claim comment (`Build.cpp ~1343-1360`); `amacRun` steady/drain + cursor API + `amacDrainAndGrow` (`AmacRing.h`); `AmacRingSlot<false>`==16B static_assert `:125`, `inactive_row :110`; fused read→act invariant `:30`; probe hash at admit `ProbeImpl.h:189`; `found_word` by-value `:152`; `computeJoinRouteWords/ForFill/LeafIds` `JoinRouteHashing.h:22/29/35`; `RowRef::INLINE_FLAG` `RowRefs.h:42-44` + `StoredColumnsIndex :439` + `RowRefList` tagged word `:99-108`; `build_arenas` `PartitionedHashJoin.h:357`; cell bit-identity `PartitionedJoinMaps.h:72/:176`; `DenseHyperLogLog.h:23`; lane HLL feed+merge `PartitionedHashJoin.cpp:222/226/386`; `per_leaf_estimate` clamp `Build.cpp ~1319`; FixedRegionAllocator ONE-allocation + zero-right-before-fill comments `:12-13/:50`; `HashCRC32` maps `HashJoin.h:316+`; decorrelated route comment `ColumnsScatter.h:72`. No row failed; two grep patterns needed signature correction (`computePassBits` returns `std::vector<size_t>`), content matched.

**G0b, part 2 — 5 seeded-random rows re-derived from commits** (seed 20260721 over the 53 row IDs → `hll-distinct-estimate-leaf-sizing`, `v2-rewrite-per-partition-hashjoin`, `scatter-pass-fanout-cap-1024`, `crc32c-packed-key-hash`, `upstream-rename-adaptation`). Each `git show` matched the row's mechanism: 2d85a907407 HLL shrink-only sizing + ed390b0406b empty-cell/rebuild fix; 802d41c679b (−945 lines) / a885e9ad45a (−13,726) / 13f63e6b0b5 (+1,332 w/ probe_buffer_*) teardown-rewrite arc; b13b0a5a3cd exactly the 8192→1024 default flip; 874fce9ca06 multiply-fold→CRC32 `HashT`; 82005a0cc26 mechanical BuildRef→RowRef rename. G0b GREEN.

**Untrusted-content report:** no directives addressed to an AI/reader found by any agent in commit messages, worklogs, or code comments (each agent reported explicitly); the only anomaly class = stale/misleading metadata (RBM ProfileEvents descriptions describing phj5-era mechanisms; RBM tip commit title sounding like server code but touching only bench tooling). Nothing followed. No secrets encountered.

**Next:** commit Phase A artifacts (tmp/port_audit/ only), deliver the Phase A report in chat, HARD STOP for approval (GA).

## Iteration 4 — GA received; Phase B opened

**GA (gate) — requester's approval, quoted verbatim (received 2026-07-21, after the Phase A report):**

> Implement 1, 2, 5 and 7.
> one by one and keep only items that move the wall (both whole query and the portion being optimized).
> work on same branch, commit individually but not push

**Interpretation (numbers = the Phase A report's ordered candidate list):** approved = 1 `warm-run-cached-distinct-estimate`, 2 `narrow-scatter-counters`, 5 `pipeline-lane-identity`, 7 `parallel-hash-table-teardown`. Deferred-by-requester = 3 `eager-postbuild-and-8b-leaf-descriptor`, 4 `grouped-leaves-metadata-compression`, 6 `probe-output-block-sorted-payload-gather`, 8 `nt-store-arch-gating`. Implementation order: 1 → 2 → 5 → 7, one at a time. **G3 acceptance is TIGHTENED by the requester:** a candidate is kept only if it improves BOTH the whole-query wall time AND the optimized portion (phase timer), each beyond the noise band, with no grid point regressing beyond the band; otherwise `rejected-by-measurement` and reverted. Same branch (`ahj`), one commit per kept candidate, no push.

**Re-pin at Phase B start:** `git rev-parse ahj` → `d8f6f57ee656a7fb73448fb78fdbc772232ffa54`; working tree CLEAN (`git status` empty). The dirty-tree question from Phase A resolved itself: those edits were committed by the requester's concurrent session as `d8f6f57ee65`.

**Delta audit `$AHJ_SHA..ahj`** (mission-required since ahj moved): two commits — `684df18f951` (my Phase A artifacts, tmp/ only, no row impact) and `d8f6f57ee65` "`PartitionedHashJoin`: allocate each leaf hash table on demand, drop `FixedRegionAllocator`" (author = requester, co-authored Cursor). Mechanism: per-leaf exact-reserved on-demand allocation by the claiming worker (allocator recycles freed scatter transients; measured by its author: tracked peak 25.2→17.2 GB, RSS 32→29 GiB, build wall 1.16→1.08 s on 500M×2B/32T), with a new `ZeroingHashTableAllocator` (unzeroed malloc + explicit streaming memset preserving sequential worker-local first touch). `BuildStats` drops slab_allocations/region_carves/heap_fallbacks/slab_bytes for ht_total_bytes/leaf_growths; ProfileEvent `PartitionedHashJoinHashTableHeapFallbacks` → `PartitionedHashJoinHashTableGrowths`. Matrix row impacts recorded in the matrix's new "Base delta audit" section: `byte-balanced-leaf-cell-allocation` disposition unchanged at `$AHJ_SHA` but the slab form is superseded at the new base (deliberate memory-driven inversion); `parallel-hash-table-teardown` premise STRENGTHENED (destructor now frees one allocation per leaf — thousands of frees — instead of one slab); slab-based risk notes in the deferred grouped-leaves/8b-descriptor rows weakened accordingly.

**New implementation base SHA: `d8f6f57ee656a7fb73448fb78fdbc772232ffa54`** (all Phase B baselines and diffs are against this, not `$AHJ_SHA`).

**Environment check at Phase B start:** load average ~28 with no clickhouse processes running (36 logged-in sessions; other agents likely building). Bench noise must be re-measured empirically before G3 (protocol: noise band = max(3%, observed baseline spread)); if load stays bursty, pin runs and widen samples. To be resolved before the first G3 run.

## Iteration 5 — candidate 1 (warm-run-cached-distinct-estimate): implementation + baseline

**Load source identified:** the contending process is another session's bench campaign — `tmp/leaf_heuristic/tools/join_mergetree_bench.py` at T32 with its own binary (`tmp/probe_regression/impl/bin_base_setbits`) and dataset (`tmp/leaf_heuristic/chscratch`). NOT killed (not this task's job). Baseline and candidate G3 runs execute under the same background contention; the V+B drift check guards the pairing.

**Dataset of record:** `/mnt/data/join_bench_data` (existing, validated read-only via its metadata table: schema v4, D_max=524288000, w=4194304, m<=1, cycles<=4, 7x7 payloads; 524,288,000 build rows / 2,097,152,000 probe rows). No other session currently uses this path.

**Sequencing note (binary hygiene):** build/reldeb at Phase B start was verified to be the BASE build (`ninja` no-op → "BASE BUILD OK", build_port_base.log) BEFORE any candidate edit; the binary was copied to `tmp/port_audit/bench/bin_base_c1` and the candidate-1 baseline grid runs from that copy, so later builds cannot contaminate it. Baseline grid launched 14:12:20Z (`c1_base_*.log`, load avg at start 27.27). No ninja runs while a grid is in flight.

**Band amendment (documented deviation, decided BEFORE any baseline/candidate comparison):** the harness reports median and min walls only, so the preregistered (max-min)/median spread is not computable; operative band = `max(3%, 2 x (median - min)/median)` of the baseline invocation. PREREG.md amended in place with the same text.

**Candidate 1 gates G1/G2 (raw results):**
- G1 GREEN: `ninja -C build/reldeb clickhouse unit_tests_dbms > build/reldeb/build_port_warmrun.log` → exit 0 (126/128 targets, ran after the c1_base2 grid finished so no bench contention).
- G2 gtest GREEN: `build/reldeb/src/unit_tests_dbms --gtest_filter='PartitionedHashJoin*:*ColumnsScatter*' > build/reldeb/test_port_warmrun.log` → exit 0, "[  PASSED  ] 48 tests", including the new `PartitionedHashJoin.DistinctEstimateCacheWarmRun` (63 ms).
- G2 stateless: two red-green cycles, both environmental/test-side, no production-code change:
  (cycle 1) the borrowed server recipe wiped `/var/lib/clickhouse` which is not writable by this user → `run_stateless.sh` rewritten self-contained under `tmp/port_audit/stateless_data` with the in-tree `programs/server/config.xml`;
  (cycle 2) `04607` failed `0 0` vs `1 1` — the fuzzer's randomized settings (`enable_parallel_replicas 1`, `enable_join_runtime_filters True`) made the plan fall back, and the path check used a timer; fixed by pinning `enable_analyzer/query_plan_join_swap_table/external-join/parallel_replicas/runtime_filters` like the sibling tests and asserting `PartitionedHashJoinPartitions > 0` (a count, cannot round to 0).
  Final: `run_stateless.sh build/reldeb/stateless_port_warmrun2.log 4 partitioned_hash_join` → exit 0, all 5 tests OK (04603-04607).

## Iteration 6 — candidate 1 G3: void round, amendment 2, REJECTED-BY-MEASUREMENT

**G3 round 1 (tags c1_base2 vs c1_cand, sequential grids): VOID.** Candidate numbers looked mixed (B wall −8.4% beyond its 3% band, but V/C/D/E +9.7..+21.4% "regressions"); the pre-registered drift check exposed the cause — the base binary re-run (c1_drift) moved −7.1% at B and +12.9% at V versus its own c1_base2 numbers, both outside band. The two grids sampled different machine conditions (the concurrent bench campaign's load varies); neither acceptance nor rejection was valid. Raw: `tmp/port_audit/bench/c1_{base2,cand,drift}_*.log`, comparison via `compare_grids.py`.

**Amendment 2 (PREREG.md, recorded before round-2 data):** point-interleaved pairing — per grid point the base and candidate binaries run back-to-back, band computed from the same round's base invocation; supersedes the coarse drift check. Both binaries frozen as `bin_base_c1` / `bin_cand_c1`.

**G3 round 2 (tag c1r2, 12 paired invocations, exit 0 all): REJECTED-BY-MEASUREMENT.**

| point | base med (ms) | cand med | wall delta | band | fill base (us) | fill cand | fill delta |
|---|---|---|---|---|---|---|---|
| V | 32 | 33 | +3.1% | 6.2% | 7567 | 3906 | −48.4% |
| A | 128 | 124 | −3.1% | 9.4% | 15807 | 8682 | −45.1% |
| B | 2753 | 2762 | +0.3% | 4.7% | 1011914 | 587546 | −41.9% |
| C | 4777 | 4818 | +0.9% | 9.7% | 1379262 | 1026133 | −25.6% |
| D | 745 | 740 | −0.7% | 9.1% | 116273 | 56689 | −51.2% |
| E | 655 | 659 | +0.6% | 8.2% | 168859 | 132428 | −21.6% |

The mechanism verifiably works — `PartitionedHashJoinDistinctEstimateReused=1` on every candidate point, fill phase −21.6..−51.2% everywhere, V hash verification PASS on both sides — but the whole-query wall is inside the band on every point, in both directions. Per the GA rule (keep only what moves BOTH the wall and the phase): **rejected**. Mechanically the result makes sense: fill is ~6% of the build's thread-summed time, the sketch feed is less than half of fill, and the fill overlaps across lanes — ~0.5 ms of wall at B against a 2.75 s query.

**Disposition:** matrix row `warm-run-cached-distinct-estimate` → `rejected-by-measurement` (checker re-run: exit 0). Dropped diff preserved at `tmp/port_audit/dropped/c1_warm_run_distinct_cache.diff` (411 lines); all production code reverted (`git checkout --` of the 10 files + removal of test 04607); `git status` clean vs `73f1d312ee4` apart from audit artifacts. Red-green cycles consumed: 2 of 3 (both test-side). The rejection is a first-class result: the fill-phase cost this candidate removes is real but not wall-relevant at these shapes.

## Iteration 7 — candidate 2 (narrow-scatter-counters): REJECTED-BY-MEASUREMENT

**Implementation:** dual-shape counters in `PostBuildContext` (`narrow_counters` + `worker_hist32`/`starts32` + `histAt`/`startAt` accessors), `histogramWorker` templated over the counter type (the `UInt32` `ColumnsScatter` kernel overloads pre-existed), prefix sum writes either shape, `UInt64` fallback intact behind `accumulated_rows <= UInt32::max()` plus a `setNarrowCountersForTests` knob; gtest `WideCounterFallbackParity` (narrow vs forced-wide: probe parity, equal partitions, identical `leaf_row_counts`).

**Gates:** G1 green (`build_port_narrowhist.log` exit 0). G2 green: 48 gtests incl. the new one (`test_port_narrowhist.log`), stateless 04603-04606 OK (`stateless_port_narrowhist.log`). G3 paired round `c2r1` (12 invocations, all exit 0):

| point | base med (ms) | cand med | wall delta | band | hist base (us) | hist cand | hist delta |
|---|---|---|---|---|---|---|---|
| V | 35 | 35 | +0.0% | 11.4% | 3569 | 3277 | −8.2% |
| A | 128 | 126 | −1.6% | 14.1% | 6609 | 6563 | −0.7% |
| B | 2783 | 2798 | +0.5% | 3.0% | 336506 | 329984 | −1.9% |
| C | 4789 | 4726 | −1.3% | 14.0% | 329034 | 315963 | −4.0% |
| D | 669 | 741 | +10.8% | 9.3% | 39697 | 41538 | +4.6% |
| E | 643 | 643 | +0.0% | 9.3% | 43705 | 42808 | −2.1% |

Histogram phase inside the band on ALL points — the prereg's refutation condition verbatim; no wall improvement anywhere (D's +10.8% vs 9.3% band is moot under rejection and consistent with D's noise history). Interpretation: at these fanouts the phase is bounded by the bucket-id derivation and pid-array traffic, not the counter-array width. **Matrix row → `rejected-by-measurement`** (checker exit 0); dropped diff at `tmp/port_audit/dropped/c2_narrow_scatter_counters.diff`; code reverted, src/ clean vs HEAD. Red-green cycles: 0.

## Iteration 8 — candidate 5 (pipeline-lane-identity): REJECTED-BY-MEASUREMENT

**Implementation (full pipeline port, 22 files):** IJoin lane overloads (default-forwarding, out-of-range tolerance documented), JoiningTransform `stream_index` + FillingRightJoinSideTransform `build_lane` plumbing, QueryPipelineBuilder lane counters, PartitionedHashJoin lock-free fill-lane slots + probe-scratch slot parking (atomic exchange/CAS, sized 2 x num_threads, legacy mutex fallbacks for out-of-range/collisions), lane threaded through probeDispatch/probeImpl/routedJoinRightColumns and the 4 per-kind instantiation TUs, gtest `LaneIdentityParity` incl. deliberately out-of-range lanes.

**Gates:** G1 took 3 cycles (all compile-level, no logic changes): cycle 1 = `-Woverloaded-virtual` (new overloads hidden in every IJoin implementer → `using IJoin::addBlockToJoin/joinBlock` added to 9 join headers, placements verified) + per-kind explicit instantiations updated to the new signature; cycle 2 = duplicate `using IJoin::joinBlock` (HashJoin.h already had one mid-class → old one removed). Also captured: the first chained run executed G2 against a stale binary after G1 failed — results discarded, chain re-gated on G1 exit. G2 green: 48 gtests incl. the new one, stateless 04603-04606 (`build_port_lane3.log`, `test_port_lane3.log`, `stateless_port_lane3.log`).

**G3 paired round c5r1 (12 invocations, all exit 0):** wall −3.4..+3.2% vs 3.0–13.8% bands; BuildFill −1.6..+3.1%; Probe −0.5..+1.5% — EVERYTHING inside the band, both declared phases and the wall, on every point. The pre-registered honest prediction (per-block mutex costs are sub-millisecond at these block counts) confirmed exactly. **Matrix row → `rejected-by-measurement`** (checker exit 0); dropped diff `tmp/port_audit/dropped/c5_pipeline_lane_identity.diff`; all 22 files reverted, src/ clean vs HEAD. Red-green cycles: 2 of 3 (both build-side).

**Candidate 1 implemented (working tree, commit follows green gates):** `PartitionedHashJoinEntry{distinct_keys}` in HashTablesStatistics (own cache, never clobbers `HashJoinEntry`); `StatsCollectingParams` plumbed PlannerJoins → ctor (5th arg, default `{}`) and through `clone`; ctor does the one-time `getSizeHint` (guarded: non-delegate + enabled); fill takes a new sketch-free `computeJoinRoutesForFill(cols, rows, routes)` overload on HIT (routes still stored for every row — the scatter reads them; only the sketch feed is skipped) in both the ASOF and plain branches; barrier consumes the cached count instead of `merged.estimate()`, sets `BuildStats.distinct_estimate_reused`, fires new event `PartitionedHashJoinDistinctEstimateReused`; `runPostBuildPhase` publishes the EXACT distinct count (sum of leaf map sizes — each cell is one distinct key) via `update()`. Tests: gtest `PartitionedHashJoin.DistinctEstimateCacheWarmRun` (cold: no reuse + sketch-accurate estimate; warm: reuse + exact estimate + probe parity; disabled: no reuse), stateless `04607_partitioned_hash_join_distinct_estimate_cache` (second identical query fires the reuse event on the partitioned path, results identical).
