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
