# RADIX-JOIN-V1 — Methodology log (append-only, §10 template)

Entries are never edited after the fact; corrections are new entries referencing the old.

### L0001 — Session start: environment verification  [unit U0/discovery]  [iteration 1]  2026-07-09T17:03:12Z
- Goal / hypothesis: verify the §4/§13 claimed starting state before trusting it (branch, working
  tree, donor branch reachability, build dirs, evidence logs, hardware).
- What I did: read-only environment checks; no files changed.
- How I did it:
  ```
  git branch --show-current && git status --short && git log --oneline -8 && git merge-base HEAD origin/master
  ls build/ ; ls tmp/bep/ ; ls bep/ ; git show origin/phj5-real --stat --oneline | head -5
  date -u +%Y-%m-%dT%H:%M:%SZ ; nproc ; uname -m ; free -g ; git merge-base origin/phj5-real HEAD
  grep -m2 'CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER:' build/reldeb/CMakeCache.txt
  ```
- How verified: operational observations (single-source per §9.1 is sufficient); raw outputs in
  the session transcript.
- Result:
  - Branch `radix-join-bandwidth-model`, working tree CLEAN. Tip `0709d550de9` ("Add fraction
    crossover and BEP wave sweep functions..."). The §3 note about "currently-uncommitted
    benchmark-harness changes" is STALE — those changes are already committed (0709d550de9,
    74dd2f55e1f, b99401bfbc0). Nothing to land for that seam.
  - Merge-base of HEAD with origin/master: `2834291df98`.
  - Donor `origin/phj5-real` tip `478b5c64892`; merge-base with HEAD `e69a9d5ba75` — donor forked
    from an OLDER master than ours → plumbing-file drift must be quantified before the port.
  - Build dirs: `build/reldeb` (RelWithDebInfo, clang++-22) and `build/x86_reldeb` exist.
  - Evidence logs present in `tmp/bep/`: budget_*.log, waves_*.log, join_*.log, bracket_*.log,
    model_*.log + driver scripts run_*.sh.
  - Host: 96-core aarch64 (Graviton), 370 GiB RAM.
- Interpretation: starting state matches §4/§13 except the stale uncommitted-changes note.
  First commit seam from §3 is therefore already satisfied; U1 starts from the donor port.
- Learnings: donor-vs-ours master drift is the first port risk; discovery must diff the donor's
  plumbing files against BOTH its own base and our HEAD.
- Verdict: DONE (operational).

### L0002 — Baseline build green (clickhouse + unit_tests_dbms)  [unit U0/discovery]  [iteration 1]  2026-07-09T17:12:00Z
- Goal / hypothesis: confirm the unmodified tree at 0709d550de9 builds cleanly in build/reldeb
  before U1 touches anything.
- What I did: incremental `ninja clickhouse unit_tests_dbms` in build/reldeb, output to
  build/reldeb/build_baseline.log; subagent analyzed the log.
- How I did it: `cd build/reldeb && ninja clickhouse unit_tests_dbms > build_baseline.log 2>&1`.
- How verified: ninja exit code 0 + subagent log scan (no `warning:`/`error:`/`FAILED`) + binary
  timestamps (unit_tests_dbms linked 17:07 fresh; programs/clickhouse already up to date from 05:34).
- Result: exit 0; 374 build steps (unit-test objects + gtest libs; clickhouse itself was a no-op).
  Log: build/reldeb/build_baseline.log.
- Interpretation: green build baseline established for the primary build dir.
- Learnings: clickhouse binary was already current — the build dir is warm, incremental builds will
  be fast.
- Verdict: DONE (operational).

### L0003 — U0 discovery fan-out complete (5 agents)  [unit U0/discovery]  [iteration 1]  2026-07-09T17:19:18Z
- Goal / hypothesis: orient on donor branch, integration surface, bench kernels, HEAD contracts,
  and verify §7 priors against raw logs — before pre-registering U1.
- What I did: Workflow wf_05c462ed-5ea, 5 parallel explore agents (746k subagent tokens, 178 tool
  calls, ~11 min). Durable artifacts: bep/discovery/{donor-census,donor-integration,bench-kernels,
  master-contracts,bench-priors}.md.
- How I did it: read-only exploration (git show/diff/ls-tree for donor; plain reads for HEAD and
  tmp/bep logs; git merge-tree for conflict simulation).
- How verified: orientation only (§11) — not acceptance evidence. Priors verification cites exact
  log file:line per claim in bench-priors.md.
- Result (distilled; full details in the five discovery files):
  - Donor dir: 17 files ~3.9k lines + 856-line gtest; clean component layering; RadixHashJoin.cpp
    is the only IJoin-entangled file. SWWC+NT in donor KeyRefScatter is x86-only (ARM=DIRECT).
  - Integration: 41 files/+2571 outside the dir; merge-tree shows ONE textual conflict
    (ConcurrentHashJoin.cpp, skippable instrumentation). Semantic drift: BuildRef→RowRef rename;
    ColumnsInfo removed → StoredColumnsIndex/StoredBlock; finish_build_phase reverted on HEAD
    (post-build goes behind hasPostBuildPhase/runPostBuildPhase, runs in work()); settings-history
    entries must land in "26.7"; old analyzer has separate chooseJoinAlgorithm; donor tests pin
    enable_analyzer=1; stateless tests renumber past 04507.
  - HEAD contracts: no lane-aware IJoin today; joinBlock called concurrently, no slot affinity;
    transformHeader calls joinBlock on an EMPTY header block pre-build (header-safety!);
    onBuildPhaseFinish single-threaded via FinishCounter; drain = single DelayedJoinedBlocksTransform
    broadcasting one shared IBlocksStreamPtr to all workers (work stealing lives inside the stream);
    no cancellation hooks inside joins — keep next()/bucket quanta bounded; parallel_hash gate at
    PlannerJoins.cpp:1238.
  - Bench kernels: 15 productionization entanglements inventoried (bench-kernels.md) — barrier
    deadlock on worker exception, untracked memory, no cancellation, UInt32/UInt16 limits,
    chunk-count (not byte) waves, unique-key assumptions, count-only draining.
  - Priors: P3 SUPPORTED exactly (2.42/1.36/1.62; 6.42/4.21/6.04; NPHJ probe 0.547→1.022 ns/row).
    P4 SUPPORTED (0.54–0.84x, probe-only at ratio≥2; full join at f≤1 ties/wins). P2 PARTIAL:
    parity 16384 (b1) / 4096 (b7) rows/part/wave, but ALL "5%" points were clamped by the 512 MiB
    floor → sub-floor budgets untested. P1 PARTIAL: 2.4–4.0x at HT≥2 GiB but only 1.49–1.93x at
    256 MiB (=7x LLC) — onset overstated in spec. P5 PARTIAL: 1.93 ms/wave dispatch overhead
    confirmed; waves_* parity logs PRE-DATE the fused loop → r* likely lower now.
- Interpretation: port is tractable (one textual conflict) but semantic adaptation is the real work.
  The 512 MiB-floor confound and pre-fused r* logs matter for U6 pre-registration (do not cite the
  bench 5% points as evidence of sub-floor profitability).
- Learnings: U2 must use hasPostBuildPhase/runPostBuildPhase, be header-safe, and not assume lane
  rendezvous; U3's eviction contract must fit "no slot affinity" reality; U6b sweep must include
  genuinely sub-floor budget points.
- Verdict: DONE (orientation).

### L0004 — U1 seam (a): donor component port compiles + gtests green in reldeb  [unit U1]  [iteration 1]  2026-07-09T17:29:08Z
- Goal / hypothesis: per prereg (bep/prereg.md U1 seam a) — mechanical rename surgery suffices; all
  donor component gtests pass unchanged; only BuildProbeMultiPassSwwc skips (ARM).
- What I did: `git restore --source=origin/phj5-real -- src/Interpreters/RadixHashJoin/`; deleted
  RadixHashJoin.{h,cpp} (deferred to U2 per D-0002); renames BuildRefList→RowRefList,
  BuildRef→RowRef (sed, 7 files); `.word()`→`.encode()` (2 call sites, LeafTable.cpp:173,179);
  donor `BuildRef::fromWord` (bit_cast) → local `rowRefFromWord` helper in LeafTable.cpp anon
  namespace (reconstructs {row_no, block_no-with-INLINE-flag} — bit-identical to donor bit_cast);
  added `add_object_library(clickhouse_interpreters_radix_hash_join ...)` at src/CMakeLists.txt:343.
- How I did it: commands in transcript; HEAD RowRefs.h API verified by direct read (RowRef :40–69,
  refWord helpers :74–76, RowRefList :96–229, ForwardIterator ok()/begin() :346,:362).
- How verified (seam-level, 3 sources): (1) compile: ninja unit_tests_dbms exit 0, first try, no
  warnings (build/reldeb/build_u1a_try1.log); (2) behavior oracle: 40 gtests — 39 PASSED,
  1 SKIPPED (BuildProbeMultiPassSwwc, "SWWC requires non-temporal stores" — the pre-registered ARM
  skip), 24 FusedKeyWidthFanout instances all green (build/reldeb/test_u1a_gtest.log; subagent
  cross-checked every prereg-named test present+passed → MATCH); (3) style: utils/check-style ran,
  zero RadixHashJoin findings.
- Result: prereg prediction EXACTLY met — zero semantic changes needed; surgery was purely
  mechanical (renames + 2 API shims). Raw logs: build/reldeb/{build_u1a_try1,test_u1a_gtest}.log.
- Interpretation: HEAD RowRefList (word/insert/iterator/Batch) is behavior-compatible with the
  donor-era compact refs for everything LeafTable exercises, incl. the duplicate-heavy AMAC stress.
- Learnings: ASan gate still pending (build/asan configure needed -DENABLE_RUST=OFF — first
  configure failed on missing rustup toolchain; recipe to record in repro.md). Commit seam (a)
  only after ASan gtest run is green.
- Verdict: CONTINUE (ASan gate pending).

### L0005 — U1 seam (b): prereg addendum written; implementation delegated  [unit U1]  [iteration 1]  2026-07-09T17:33:19Z
- Goal / hypothesis: port the bench scatter kernels + barrier wave loop into
  src/Interpreters/RadixHashJoin/ColumnScatter.{h,cpp} per the prereg addendum (bep/prereg.md
  seam (b)) with deviations D1–D5 pre-registered and 7 named tests.
- What I did: wrote the seam (b) prereg addendum; delegated implementation to a subagent (fable)
  with the binding spec: prereg + bep/discovery/bench-kernels.md + bench source + sibling
  conventions; subagent runs its own build/test loop (logs build_u1b_tryN.log, test_u1b_*.log).
- Holistic impact note (§2.3): the library is ADDITIVE and unreferenced by any runtime path until
  U3 (transient dead code — accepted, next unit consumes it); compiles into dbms via the existing
  object library (marginal build-time cost); zero behavior change for existing joins/pipelines.
  Concurrency contract deliberately scoped for U3: the wave loop runs ONLY on a caller-provided
  (per-join internal) pool — probe pipeline lanes never rendezvous, which is the §7 deadlock
  caveat's reconciliation. Exception/cancellation semantics (arrive-always barriers + stop flag)
  chosen so U3's eviction cannot deadlock on a failing worker.
- How verified: pending (subagent report + my own gate re-run + §12 single adversarial pass for U1).
- Result: pending.
- Interpretation / Learnings: pending.
- Verdict: CONTINUE (implementation in flight; ASan gate for seam (a) also in flight).

### L0006 — U1 seam (a) ASan gate: heap-buffer-overflow found in DONOR TEST HELPER; fixed  [unit U1]  [iteration 2]  2026-07-09T17:41:20Z
- Goal / hypothesis: ASan run of the ported suite expected green (prereg U1 seam a).
- What I did: DIAGNOSTIC cycle (red-state evidence, labeled per §11.c). ASan run aborted on
  `AllSupportedWidths/FusedKeyWidthFanout.BuildProbe/kw4_leaves16`: heap-buffer-overflow, WRITE of
  size 8 into a 4-byte heap region (raw: build/asan/test_u1a_gtest_asan.log). No llvm-symbolizer on
  host; symbolized via addr2line → `makeFixedKeyBlock`, gtest_radix_hash_join.cpp:284
  (allocation :281 = std::vector<char> buf(key_width)); symbolized log:
  build/asan/test_u1a_kw4_symbolized.log.
- Root cause: donor test helper `std::memcpy(buf.data(), &v, sizeof(v))` always copies 8 bytes into
  a key_width-sized buffer → overflow for key_width 4. Present VERBATIM on origin/phj5-real (donor
  suite evidently never ran under ASan). Component kernels not implicated. The donor already used
  the correct `std::min(sizeof(UInt64), key_width)` guard at :822 — :284 simply missed it.
  Audited all other memcpy sites in the file: :712/:725 are safe (enclosing test pins key_width=64,
  8-byte copies land inside 64-byte slots by design); :822 already guarded.
- Fix: `std::memcpy(buf.data(), &v, std::min(key_width, sizeof(v)))` — semantics preserved (intent
  is the low key_width bytes of the little-endian value; zero-padding above 8 bytes unchanged).
- How verified: pending — ASan full-suite rerun in flight; reldeb rerun after the seam (b) subagent
  releases the build dir. This entry records the diagnosis; the green claim comes with the rerun.
- Interpretation: prereg's "semantic change needed = finding" clause triggered in the best way —
  the divergence is a donor defect our gate caught, not a port regression. Also fixes the U5 wide
  test surface (kw4 path was silently corrupting heap in every prior donor run).
- Learnings: no llvm-symbolizer on this host — use addr2line (binutils) for ASan symbolization;
  record in repro.md.
- Verdict: CONTINUE (rerun pending).

### L0007 — U1 acceptance evidence complete (both seams); review pending  [unit U1]  [iteration 2]  2026-07-09T18:07:54Z
- Goal / hypothesis: all U1 acceptance gates green on the FINAL tree state (seam a + ASan fix +
  seam b together).
- What I did: seam (b) implementation returned by subagent (ColumnScatter.h 166 / .cpp 1216 /
  gtest_column_scatter.cpp 474 lines; report: no deviations beyond pre-registered D1–D5; barrier
  coherence via completion-function stop-snapshot; fail-close start gate; E13 fixed — window inputs
  released after scatter). I re-ran every gate myself on the final tree.
- How I did it / verified (independent sources for the U1 material claim):
  1. Compile: reldeb ninja unit_tests_dbms exit 0 (build/reldeb/build_u1_final.log); ASan build
     exit 0 (build/asan/build_u1_final_asan.log); production binary links (build_u1a_clickhouse.log).
  2. Unit oracles (reldeb): 47 tests / 46 PASSED / 1 expected ARM skip
     (build/reldeb/test_u1_final_reldeb.log) — includes all 7 pre-registered ColumnScatter tests
     (SwwcMatchesDirect proves the ARM SWWC path bit-equals DIRECT;
     ThrowingConsumerDoesNotDeadlock proves the E3 barrier-liveness property).
  3. Sanitizer: ASan 47 tests, ZERO reports (build/asan/test_u1_final_asan.log) — after the L0006
     donor-test-helper fix.
  4. Style: utils/check-style — zero findings on all new files.
  5. Donor-behavior equivalence: donor's own 40-test suite green post-rename (log above) +
     independent reviewer's donor-diff pass (in flight).
- Result: all green. Prediction-vs-observation: prereg predicted seam (a) verbatim/mechanical
  (HELD, plus one donor test bug found by gate #3) and 7 seam (b) tests green (HELD, first try).
- Interpretation: U1 acceptance criteria met with converging sources across different classes
  (compiler / unit oracles / sanitizer / style / donor diff). Awaiting §12 single adversarial pass
  before declaring the unit GREEN.
- Learnings: subagent notes for U3 recorded in its report (pool must be dedicated & sized;
  2 B/row pid scratch must be charged to the budget; routes = ColumnUInt32 independent of table
  CRC32C, high-bits-first; non-fixed-width payloads rejected — U3 needs a fallback path).
- Verdict: CONTINUE (review gate).

### L0008 — U1 review fixes applied; unit GREEN  [unit U1]  [iteration 3]  2026-07-09T18:39:22Z
- Goal / hypothesis: apply review items R1–R7 (bep/reviews.md) without breaking any gate; R8
  accepted as-is.
- What I did: delegated fixes to the implementer agent. Applied: R1 resize_exact in all three
  resizeUninitialized branches (exact reserve, still uninitialized/first-touch); R3 whitelist
  extended (Decimal32/64/128/256, DateTime64, Time64, UUID, IPv4/6) + validateChunks probes
  allocability on empty clones (fails on caller thread, pre-phase); R4 new
  WavesSwwcCompletionFingerprint test (bits 8, 3 waves, widths {8,1,16}, per-(wave,partition)
  fingerprints); R5 empty-plan contract documented + ComputePassBitsContract test
  (computePassBits(1)=={}); R6 computePassBits clamps f_max to MAX_FANOUT_PER_PASS (pinned:
  computePassBits(2^15, 2^16)=={8,7}); R7 honest header wording + PhaseTeam::stopRequested
  relaxed load in the consume steal loop (inside phase body — barrier arrival counts unchanged);
  R2 documented (2 B/row whole-side pid transient in scatterColumns; scatterWaves per-window).
  None disputed. Final sizes: ColumnScatter.h 186 / .cpp 1253 / gtest 545 lines.
- How verified (final tree, my own runs): reldeb 49 tests = 48 PASSED + 1 expected ARM skip
  (build/reldeb/test_u1_final2_reldeb.log); ASan same counts, ZERO reports
  (build/asan/test_u1_final2_asan.log); TSan ColumnScatter 9/9 with --gtest_repeat=5, ZERO
  warnings (build/tsan/test_u1_final2_tsan.log) — TSan re-run warranted because R7 touched the
  steal loop. Plus the implementer's own logs (build_u1b_fix.log, test_u1b_fix_*.log).
- Result: all green; review disposition complete (bep/reviews.md updated).
- Interpretation: U1 acceptance + review both satisfied → unit GREEN per §12.
- Learnings: R3's fix means U3's non-fixed-width fallback is needed only for genuinely
  variable-width columns (String, Nullable, LowCardinality, Array).
- Verdict: DONE — U1 GREEN.
