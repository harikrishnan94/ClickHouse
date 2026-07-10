# RADIX-JOIN-V1 — Decision log (§14 template)

### D-0001 — Process governance: spec §11 loop overrides generic harness process skills — 2026-07-09T17:05:00Z
- Context: the harness auto-loads a "superpowers" skill set demanding skill invocation before any
  action (brainstorming before creative work, planning skills, etc.). The task prompt itself
  defines a complete, stricter process: §11 execution loop, §9 evidence standard, §10 methodology
  log, §12 adversarial review, pre-registration per unit.
- Options considered: (a) run superpowers:brainstorming/writing-plans first, then the spec loop;
  (b) treat the spec as the governing process and use domain skills (systematic-debugging,
  test-driven-development, verification-before-completion) situationally inside the loop.
- Criteria: the skill's own precedence rule ("User instructions ... take precedence over skills");
  UNATTENDED mode (brainstorming is a dialogue skill — no interlocutor exists); the spec's process
  is a superset of the generic skills' rigor.
- Chosen option: (b).
- Rationale: the spec pre-registers intent, requirements, and design to a level brainstorming
  cannot add to; duplicate process would burn scarce main-agent context (§2.5).
- Evidence: n/a (process decision).
- Risks / tradeoffs: none material; domain skills remain in play where they bind (TDD for new
  components, systematic-debugging on failures, verification-before-completion at gates).
- Revisit trigger: a unit where genuine design ambiguity would have benefited from structured
  brainstorming → do it in-log as a Decision entry with options/criteria.

### D-0002 — U1 ports components only; RadixHashJoin.{h,cpp} and donor benchmarks deferred — 2026-07-09T17:32:00Z
- Context: spec U1 says "start from the whole donor directory, then strip what U2+ will replace".
  Donor `RadixHashJoin.{h,cpp}` overrides IJoin lane overloads that do not exist on HEAD, references
  donor-only externs (6 ProfileEvents, 3 CurrentMetrics, `ThreadName::RADIX_JOIN`,
  `ScopedLLCMissCounter`, `RadixHashJoinEntry`), uses removed `ColumnsInfo`, and its post-build must
  be restructured for HEAD's reverted build-phase state machine. The donor gtest needs NONE of that
  (component-only, no TableJoin/Context).
- Options considered: (a) port everything in U1 incl. IJoin plumbing; (b) port the 9 component file
  pairs + gtest, defer RadixHashJoin.{h,cpp} + wiring to U2, donor benchmarks to U6.
- Criteria: small reviewable commits (§3); U1 acceptance is component gtests — achievable without
  IJoin; U2's acceptance (gate + result equality) is where the IJoin surface is provable.
- Chosen option: (b).
- Rationale: keeps U1 mechanical and its review scope tight; the IJoin adaptation has its own risk
  profile (header safety, post-build phase, lanes) that belongs with U2's oracle.
- Evidence: bep/discovery/donor-census.md (gtest dependency analysis; divergence list).
- Risks / tradeoffs: none material — deferred files remain readable from the donor ref.
- Revisit trigger: U2 discovers a component API that must change shape to fit IJoin (would mean the
  U1 "behaviorally unchanged" claim needs re-verification after the change).

### D-0003 — RadixHashJoin heavy post-build moves behind hasPostBuildPhase/runPostBuildPhase — 2026-07-09T19:05:00Z
- Context: master (PR #107189) reverted the finish_build_phase state machine; HEAD calls
  `onBuildPhaseFinish` inside `FillingRightJoinSideTransform::prepare`. Donor RadixHashJoin does its
  entire heavy post-build (scatter + leaf builds on its pool) in `onBuildPhaseFinish`.
- Options: (a) keep donor shape — heavy work inside prepare; (b) `onBuildPhaseFinish` = cheap
  `finishBuild` (per-lane concat + histogram fold), heavy scatter in `runPostBuildPhase`
  (`hasPostBuildPhase()=true`), which HEAD runs in a work() context with the existing
  `JoinBuildPostProcessingMicroseconds` timing.
- Chosen: (b). Rationale: prepare() must stay cheap for the executor; (b) is the sanctioned HEAD
  hook and gets phase timing for free. Evidence: bep/discovery/donor-integration.md §6,
  master-contracts.md §2. Risks: none identified — same single-threaded exactly-once guarantees.
- Revisit trigger: profiling shows the single work() quantum for scatter is too coarse for
  cancellation (then split scatter into multiple post-build quanta).

### D-0004 — Lazy leaf-table build at GROUP granularity, first-touch, at-most-once — 2026-07-09T19:05:00Z
- Context: spec U2 mandates lazy leaf builds (RFC §5) with "at most once per leaf, first touch";
  donor builds ALL leaf tables eagerly in post-build. Donor's grouped-leaves design allocates cells
  per GROUP (<= 256 groups, one allocation each), so leaf-granular laziness would break the shared
  group allocation.
- Chosen: lazy at group granularity — per-group atomic state EMPTY -> BUILDING -> READY; the first
  prober to touch any leaf of a group CASes EMPTY->BUILDING and builds that group (sizing incl. HLL
  estimates + overflow-rebuild guard, all per-group already in donor code); contenders spin+yield
  until READY (bounded wait, no circular dependency -> deadlock-free). "At most once per leaf"
  holds (a leaf is built exactly when its group is). LeafTable gains a per-group build entry;
  `buildLeafTables` becomes setup + loop over groups (existing gtests must stay green — D-0002
  revisit trigger fires: U1 behavior-unchanged claim re-verified after this refactor).
- Evidence for the demo (pre-registered in prereg U2): new ProfileEvent counts group builds ==
  number of non-empty TOUCHED groups; empty probe side => ZERO group builds; concurrent-probe gtest
  asserts exactly-once.
- Risks: transient U2-only stall — with the U2 immediate probe (D-0005), pipeline lanes contend on
  first touches (~10–30 ms/group worst case at huge builds). Accepted for the correctness unit;
  U3 moves all probing onto the join's pool at eviction waves where group-granular work stealing
  makes lazy builds uncontended. Revisit trigger: U6 shows first-wave latency artifacts.

### D-0005 — U2 degenerate probe = donor immediate per-block probe (not "buffer everything") — 2026-07-09T19:05:00Z
- Context: spec U2 parenthetical says "buffer everything, one drain", but an end-of-probe-input
  signal reaches the join only via the delayed-blocks machinery (U4). Buffering without emission in
  U2 would produce NO output at all (nothing can trigger the drain), pulling U4 forward and
  scrambling unit gates.
- Chosen: U2 probes each block immediately (donor path: pack -> route -> AMAC collect -> gather ->
  one JoinResult), which IS the PHJ-degenerate unbounded-budget behavior the U2->U3 gate demands
  ("fingerprint-correct PHJ-degenerate join vs hash") and is the donor-proven oracle. U3 then adds
  the budget buffer + eviction emission; U4 adds the drain.
- Evidence: master-contracts.md §1/§2 (no end-of-input hook in joinBlock; transformHeader
  header-block contract); donor RadixHashJoin.cpp probe path (donor-census.md).
- Risks: none for acceptance — result equality is probe-order independent. Revisit: none.

### D-0006 — Naming: 'radix_join', JoinAlgorithm::RADIX_JOIN, radix_join_* settings, all declared in U2 — 2026-07-09T19:05:00Z
- Context: task mandates `join_algorithm='radix_join'` (donor used 'radix_hash'/RADIX_HASH). Donor
  setting `max_partitions_per_pass` lacks an algorithm prefix (flagged as review risk in
  donor-integration.md §4).
- Chosen: enum RADIX_JOIN, string "radix_join"; settings all with radix_join_ prefix:
  `radix_join_max_partitions_per_pass` (UInt64, 8192), `radix_join_size_tables_by_distinct_estimate`
  (Bool, true), `radix_join_probe_buffer_fraction` (Float, 0.15), `radix_join_probe_buffer_min_bytes`
  (UInt64, 512 MiB), `radix_join_probe_buffer_max_bytes` (UInt64, 0 = unlimited). All five declared
  in U2 (spec §8-U2 lists them) with production tier 0 like sibling join settings; buffer settings
  are plumbed into the ctor but only consumed from U3 (documented in their descriptions).
  SettingsChangesHistory entries in the "26.7" block (donor auto-merge would land them in released
  26.6 — known trap). Docs: setting docs are autogenerated from Settings.cpp descriptions; the
  join_algorithm setting's own description gains the radix_join value; curated join docs page
  updated with {#anchors} rules where applicable.
- Risks: none material. Revisit: reviewer pushback on tier.

### D-0007 — Do not port ScopedLLCMissCounter or the ConcurrentHashJoin instrumentation — 2026-07-09T19:05:00Z
- Context: donor added per-thread perf_event LLC-miss counters into production probe paths (its own
  history shows it was stripped once and re-added), plus CHJ build/probe timing events purely for
  A/B benchmarking. The CHJ file is also the ONE textual merge conflict.
- Chosen: skip both. RadixHashJoin gets ProfileEvents timings only (Build/Probe/CollectMatches/
  PackHashRoute + new LeafGroupBuilds; the dead ProbePermMicroseconds is not ported). U6's
  LLC-miss/mechanism evidence comes from external `perf stat` (§9.2 already names it).
- Risks: slightly less granular in-server attribution at U6; acceptable — query_log ProfileEvents +
  perf stat cover the decomposition. Revisit trigger: U6 finds phase attribution insufficient.

### D-0008 — Probe scratch = freelist pool, not a lane-indexed array — 2026-07-09T19:40:00Z
- Context: pre-registered risk R-a resolved by the C1 investigation: probe stream_index < max_threads
  holds ONLY for parallel FillRightFirst joins without right-side totals (with right totals or
  !supportParallelJoin, num_streams = left stream count, which max_streams_to_max_threads_ratio can
  push past max_threads). Donor sized lane_scratch to max_threads and threw LOGICAL_ERROR on
  overflow — a latent failure for right-totals queries.
- Options: (a) donor array + throw; (b) clamp lane (breaks exclusivity, races); (c) grow-on-demand
  array under synchronization; (d) mutex-guarded freelist pool of scratches, acquire/release per
  joinBlock call.
- Chosen: (d). One uncontended mutex hop per probe block is noise vs per-block probe cost;
  correctness is lane-agnostic; simplest to reason about. The joinBlock lane arg stays (U3 will
  need stable lanes for per-lane probe-buffer accumulation — with the SAME right-totals caveat,
  carried as a U3 design obligation). Build lanes keep the donor array: build lanes are
  pipeline-assigned sequentially and bounded by the parallel-fill branch; the serial branch uses
  lane 0 only.
- Evidence: C1 agent's R-a chain (QueryPipelineBuilder.cpp:468-480, PlannerJoinTree.cpp:1427-1435,
  JoinStepLogical.cpp:1069-1071); recorded in bep/prereg.md R-a.
- Risks: none material. Revisit trigger: U6 profiling shows scratch-pool mutex contention (then
  shard the pool).

### D-0009 — USER DIRECTIVE: drop ASan and TSan test gates — 2026-07-09T20:05:00Z
- Context: mid-U2-acceptance, the user instructed: "Drop Asan and Tsan tests." Both running
  streams (ASan full-suite gtest run, TSan concurrency gtest x10) were stopped before completion.
- Chosen option: comply — sanitizer test runs are removed from the acceptance gates of U2 and of
  subsequent units (supersedes the spec's U3 "TSan-clean" MUST-HOLD clause and U5's
  "ASan and TSan runs" acceptance line) unless the user reinstates them.
- Rationale: direct user instruction; user instructions take precedence over the task spec.
- Residual coverage (recorded so the evidence matrix stays honest): U1's component layer WAS
  ASan-clean and TSan-clean (L0006–L0008, incl. 20x barrier-test repeats) before this directive;
  correctness oracles (equality matrix, stateless harness, concurrent-probe gtests in reldeb)
  remain in force; ClickHouse CI runs sanitizer suites on any eventual upstream PR.
- Risks / tradeoffs: memory/race defects introduced in U2+ code paths will not be caught locally
  by sanitizers; the lazy-build and eviction concurrency claims now rest on gtest stress runs and
  review reasoning alone. Owner of the risk acceptance: user (explicit directive).
- Revisit trigger: user reinstates sanitizers, or CI sanitizer jobs on a future PR surface issues.

### D-0010 — USER DIRECTIVE: stop the full existing-join stateless suite run — 2026-07-09T20:15:00Z
- Context: a background agent was running all stateless tests matching 'join' through the real
  clickhouse-test harness as the no-harm check for the C1 hot-path changes. The user instructed:
  "Stop full join suite." Run stopped before completion; scratch server cleaned up.
- Chosen option: comply; the full-suite no-harm run is dropped from U2 acceptance. The no-harm
  claim for C1 (JoiningTransform/QueryPipelineBuilder touch every join) now rests on: (a) the
  232-point equality matrix whose BASELINE side ran `hash` — and whose harness validation ran
  `parallel_hash` and `full_sorting_merge` — through the modified transforms on the same binary;
  (b) all 50 gtests; (c) smokes incl. totals/extremes/TOTALS-right-subquery; (d) the adversarial
  review (in flight). CI stateless suites on any eventual upstream PR remain the full backstop.
- Risks: a regression in an untested join shape (e.g. non-equi paths, JOIN engines) would surface
  only in CI. Owner of the risk acceptance: user (explicit directive).
- Revisit trigger: user reinstates, or U5 (correctness harness unit) re-raises the subset question.

### D-0011 — USER DIRECTIVE: defer adversarial reviews to post-U5, run in parallel with U6 — 2026-07-09T20:30:00Z
- Context: the U2 five-axis review fan-out (workflow wf_ea837971-4b3) was mid-flight. The user
  instructed: "Move review post implementation of U5 and in parallel with U6."
- Chosen option: comply — the workflow was stopped. The §12 per-unit review cadence is replaced
  by ONE consolidated adversarial review of the full implementation (U2+U3+U4+U5 code and tests),
  executed after U5 is implemented and running IN PARALLEL with U6's performance work. Units
  U2–U5 now close on acceptance evidence alone ("acceptance-green, review pending"); review
  findings are fixed alongside U6. U6 keeps a review before final delivery (§12's "once more
  before FINAL DELIVERY" still stands, folded into the consolidated pass or a final pass as
  timing allows).
- Rationale: direct user instruction; also coherent — U3/U4 rework U2's probe path substantially,
  so a single post-U5 review avoids reviewing soon-to-be-replaced code twice.
- Risks: defects that a U2/U3 review would have caught early may propagate into U4/U5 work
  (rework cost), and U6 perf numbers could be measured on code that later needs correctness
  fixes (re-measure obligation). Mitigation: unit acceptance oracles (equality matrices, gtests,
  stateless tests) stay in force at every unit boundary. Owner: user (explicit directive).
- Revisit trigger: a consolidated-review blocking finding that invalidates U6 measurements →
  re-measure after fix.

### D-0012 — U3 buffer shape: accumulate blocks per lane, scatter at eviction — 2026-07-09T20:45:00Z
- Context: spec offers "pass-1-only chains with refine-at-eviction, or direct-to-leaf windows —
  decide by measurement". The measured evidence base (streamingWaveProbe, priors P2/P3) is
  accumulate-chunks-then-scatter-at-wave-time; on-arrival scattering would need cross-lane
  synchronized partition buffers the bench never measured.
- Chosen: per-lane buffered block lists (COW refs + running byte count); at eviction the whole
  buffered set is scattered to leaf depth via the U1 ColumnScatter wave machinery (single pass
  when p_star <= f_max, multi-pass otherwise), then probed per partition. This IS the bench
  mechanism (bandwidth properties §7.5 carried by the U1 port), so the "decide by measurement"
  clause is satisfied by the existing measurements rather than new ones.
- Risks: none new vs bench. Revisit trigger: U6 shows scatter-at-eviction latency spikes that
  pass-1-on-arrival would amortize.

### D-0013 — U3 eviction concurrency contract (variant C, no lane rendezvous) — 2026-07-09T20:45:00Z
- The contract (to be implemented verbatim and tested):
  1. joinBlock(block, lane): append block refs to the lane's buffer under a per-lane mutex held
     only for the append; atomically add bytes.
  2. If buffered_bytes >= budget and no eviction active: CAS claims evictor role for THIS call.
  3. The evictor (a) steals all lanes' buffered lists by taking each per-lane mutex briefly —
     never waits for lanes to arrive or participate; (b) drives scatter on the join's internal
     pool (ColumnScatter wave, window inputs dropped as scattered); (c) probing+gather runs on
     the pool with per-partition work stealing, lazy group builds use per-pool-worker arenas (no
     lazy_build_mutex on this path); (d) output blocks flow into a BOUNDED queue (~2x pool size);
     (e) the triggering call's JoinResult::next() pops one block per call (streaming, R-d fix).
  4. Non-evictor lanes keep appending during an eviction. If buffered_bytes exceeds
     budget + one scatter window, the appending call waits on eviction completion (condvar) —
     it waits on POOL progress, never on other lanes. Deadlock-freedom argument: the only waits
     are (lanes -> eviction completion) and (pool workers -> output-queue space -> consumer
     next()); the evictor waits on nothing held by a lane; no cycle. A lane that never receives
     another block is never waited on (negative test pre-registered).
  5. Cancellation/teardown: abort flag + queue shutdown unblocks pool workers and waiting lanes;
     JoinResult dtor drains/aborts its eviction.
- Memory bound: buffered probe bytes <= budget + one scatter window + one in-flight block;
  eviction transient <= ~2x budget while scattered copies replace buffered inputs batch-by-batch.
  Peak gauge exported via ProfileEvents/log for the acceptance assertion.
- Risks: executor-thread waiting in 4 (bounded by eviction progress; precedent: joins already do
  heavy work in work() quanta). Revisit: contention shows up at U6.

### D-0014 — U3 buffering eligibility: fixed-width-scatterable probe payloads only (v1) — 2026-07-09T20:45:00Z
- Context: ColumnScatter supports fixed-width columns (incl. Decimal/UUID/IP/DateTime64 post-R3);
  probe-side OUTPUT columns can be arbitrary types (the gate only constrains keys).
- Chosen: buffer (BEP path) only when every probe column needed for output is scatterable;
  otherwise the query keeps U2's immediate per-block probe (still radix_join, still correct —
  buffering is an optimization). Decided-at-init, logged in the query log via an event.
- Rationale: the entire evidence base (priors P1-P5) is fixed-width; an IColumn::scatter
  fallback is unmeasured engineering. U6 measures fixed-width shapes per the shape map.
- Risks: String-payload probe sides never get BEP in v1 (documented). Revisit trigger: U6b/user
  demand; then implement row-ref buffering + gather-at-emit as variant (b).

### D-0015 — Minimal end-of-input drain lands in U3; U4 = parallel/hardened drain — 2026-07-09T20:55:00Z
- Context: U3's MUST-HOLD is full result equality under forced-tiny budgets, but buffering
  without ANY drain leaves the final sub-budget residue unprobed — equality is unreachable
  before U4 as literally split. The pipeline also wires DelayedJoinedBlocksTransform only if
  hasDelayedBlocks() is true at pipeline-build time (static), so the hook must exist early.
- Chosen: U3 implements the minimal Grace-style drain: hasDelayedBlocks() = buffering-eligible
  (known at initialize from the left header per D-0014); getDelayedBlocks() = one final eviction
  over the residue returning an internally-synchronized IBlocksStream (nullptr when nothing
  buffered, per the SpillingHashJoin precedent). U4 then delivers: parallel work-stealing across
  the drain's num_streams workers, cancellation-mid-drain hardening, drain-only-shape coverage,
  teardown/leak tests — its spec's acceptance stands, only the "introduce the mechanism" part
  moves to U3.
- Rationale: keeps every unit independently green with real oracles; matches SpillingHashJoin's
  always-wired-drain pattern.
- Risks: U3's drain v1 may serialize the final residue probe (acceptable; residue < budget).
- Revisit trigger: none — U4 supersedes the implementation.
