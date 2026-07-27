# U3 DRAFT NOTES — order-preserving routed probe + AMAC probe find ring

Status: done (implementation + self-tests). NOT committed (per task: no git write commands).
Branch: phj-ph @ 837247e57fc (working tree carries the whole unit).
Frozen design: PREREG-007 + plan "Design summary" (routed probe + AMAC core paragraphs).

## Files
New:
- src/Interpreters/HashJoin/HashJoinRoutedMethods.h        (RoutedHashJoinMethods decl + 30 extern templates)
- src/Interpreters/HashJoin/HashJoinRoutedMethodsImpl.h    (joinBlockImpl / switch / routed loop with `precomputed` seam + word_loop + AMAC wiring)
- src/Interpreters/HashJoin/RoutedHashJoin{All,Any,RightAny,Semi,Anti,Asof}.cpp  (6 TUs, 30 instantiations mirroring HashJoinMethods.h)
- src/Interpreters/HashJoin/AmacProbe.h                    (mapped-word contract, SlotMapDesc, amac_probe_supported, amacFindPass decl, 64 extern instantiations)
- src/Interpreters/HashJoin/AmacProbe.cpp                  (AmacFindPolicy + amacFindPass + event; chunk = 8192 rows)
Modified:
- src/Interpreters/ConcurrentHashJoin.cpp   (step 1 shared StoredColumnsIndex; step 5 flip: joinBlock derives routes only, RoutedJoinResult replaces deleted ConcurrentHashJoinResult; selectDispatchBlock fwd-decl)
- src/Interpreters/ConcurrentHashJoin.h     (class comment: probe no longer scatters)
- src/Interpreters/HashJoin/HashJoin.h/.cpp (joinRoutedBlock static + RoutedHashJoinMethods friend; joinScatteredBlock REMOVED — only caller was CHJ)
- src/Interpreters/HashJoin/HashJoinMethods.h/Impl.h (RoutedProbeContext + threading through joinRightColumnsWithAdditionalFilter: per-row map/flags by route slot, aggregate-bytes prefetch gate; friend RoutedHashJoinMethods; createKeyGetter member removed)
- src/Interpreters/HashJoin/KeyGetter.h     (hoisted free `createKeyGetter<KeyGetter, is_asof_join>` incl. key_range; also replaced a dead file-local copy in HashJoin.cpp)
- src/Interpreters/HashJoin/ResumableHashMap.h (const overload of `cursorCells`)
- src/Common/ProfileEvents.cpp              (new ConcurrentHashJoinAmacProbeRows with the frozen description; 3 probe event descriptions re-sited, names frozen)
- src/Interpreters/tests/gtest_concurrent_hash_join_amac.cpp (5 new tests; helpers parametrized by kind/strictness/slots; drain handles next_block re-feed)

## Design-point notes / deviations argued
- Shared index (step 1): master pattern restored after pool->wait(); StoredColumnsIndex::add is
  mutex-protected (src/Interpreters/RowRefs.cpp:273) — concurrent build inserts stay correct.
- AMAC probe instantiations = 8 families x {MapsOne, MapsAll} x need_flags x selector = 64.
  ASOF is excluded by the by-word gate (`amac_mapped_fits_word`; AsofRowRefs is a unique_ptr) —
  mirrors AmacBuild.cpp's MapsOne/MapsAll-only list; ahj's pointer-scheme lane for non-word
  mapped types was NOT ported (no acceptance anchor needs it; ASOF keeps the plain routed loop).
- found_offset is the slot-LOCAL offset ((cell-buf)+1, matching offsetInternal); NO found_slot
  output array — emit re-derives the slot from slot_ids[ind]. The ring keeps a slot field
  internally for recordHit's descriptor lookup only.
- Chain-wrap guard (frozen plan design summary): computed at slot-desc build time per probe call
  (last pad cell occupied => disengage), instead of post-build plumbing — equivalent (maps are
  immutable during probe), O(num_slots) loads, and applies under Force too (correctness, not a
  threshold).
- No flat-descriptor fallback loop (ahj's flat_loop): the sub-threshold path is the plain routed
  loop with look-ahead prefetch gated on AGGREGATE map bytes, per the U3 work list. Perf gate
  (c) arbitrates in verification.
- Engagement predicate spec-exact: mode!=Off && slot_joins[0]->amacEnabled() && (Force ||
  (aggregate bytes > getMinBytesForPrefetchInJoin() && rows >= amac_min_rows)) && !chain_may_wrap.
- word_loop gate: isLazy && word-mapped && !need_flags && !asof && !any (ahj port minus its
  use_direct_typed_gather, which does not exist on this branch).
- Routed loop is single-disjunct (flag_per_row=false) — CHJ supports one clause (chasserted).
  Additional-filter path (mixed ON) reuses HashJoinMethods::joinRightColumnsWithAdditionalFilter
  with nullable RoutedProbeContext (per-row map+flags by slot; find_results/flag ops in the
  second loop use the row's slot's flags via selector[i]); no new instantiations.
- LC probe getter cache is safe across slots: same dictionary index => same key => same route.
- Remainders: RoutedJoinResult forwards inner next_block to JoiningTransform (re-feeds via
  joinBlock, routes re-derived). Wrapper keeps `inner` alive (pointer points into it).

## Gate outputs (verbatim)
(i) builds (logs in build/reldeb/):
    build_u3_step1..6.log, build_u3_step7_gtest.log, build_u3_final2.log — all "rc=0", 0 errors.
(ii) gtests (build/reldeb/src/unit_tests_dbms --gtest_filter='*Amac*'):
    default env: "[==========] 10 tests from 1 test suite ran. (1945 ms total) / [  PASSED  ] 10 tests."
    CLICKHOUSE_JOIN_AMAC=0: "[==========] 10 tests from 1 test suite ran. (1967 ms total) / [  PASSED  ] 10 tests."
(iii) local smoke (tmp/u3_smoke/probe_smoke.sh, out in tmp/u3_smoke/probe_smoke.out):
    force/off/auto: all six shapes byte-identical, e.g. "inner_uint 1000000 3600575050460955328"
    ... "right_str 1000000 11759732130608910995" in every arm.
    force: "amac_probe_rows 3567986" (>0; < 6M because join runtime filters prune probe rows
    before the join), "amac_build_rows 6000000". off: both 0. auto: probe 3567986, build 5223431.
    hash algorithm under force: results equal parallel_hash, "amac_events_under_hash 0".
(iv) ORDER smoke (check_order.py READ-ONLY, clickhouse local, max_threads=96, harness-pinned
    settings min_joined_block_size_rows=0, min_joined_block_size_bytes=0, query_plan_join_swap_table=0):
      force INNER: "ORDER-BLOCKS OK (137 blocks, 6000000 rows)"
      force LEFT:  "ORDER-BLOCKS OK (184 blocks, 9000000 rows)"
      force RIGHT: "ORDER-BLOCKS OK (137 blocks, 6000000 rows)"
      off   INNER/LEFT/RIGHT: same three OK lines.
      T=1 --global force INNER: "ORDER-BLOCKS OK (137 blocks, 6000000 rows)"
    NOTE for verification: with DEFAULT min_joined_block_size_* the per-block check reports 1-2
    seam violations in ~530 blocks in BOTH arms — that is the join-output squashing merging
    chunks of different input blocks, exactly why run_order.sh pins SQUASH0 for its raw checks.
(v) clang-tidy (clang-tidy-22, repo .clang-tidy, build/reldeb compile_commands.json) over
    AmacProbe.cpp, RoutedHashJoinAll.cpp, RoutedHashJoinAsof.cpp, ConcurrentHashJoin.cpp,
    gtest_concurrent_hash_join_amac.cpp with header-filter over the new headers: ZERO findings
    on our lines. Two justified NOLINTs in AmacProbe (found_word/found_offset written through
    the dependent Policy; a required `typename` in a template argument). Pre-existing findings
    on untouched lines of HashJoin.cpp / KeyGetter.h / HashJoinMethods*.h (redundant-typename,
    ifelse-braces, C-casts, ConcatStreams nextImpl visibility) left alone. utils/check-style: clean.
(vi) binary size: before 4,748,738,592 (== bins/clickhouse-candidate-7e64a6cf4d5 modulo docs-only
    commits), after 4,842,488,712 => +93,750,120 (+1.97%). llvm-size text: 474,931,892 ->
    486,236,880 => code +11.3 MiB (+2.38% text; 30 routed instantiations + 64 find-pass
    instantiations); the remaining ~82 MiB is DWARF (RelWithDebInfo). New-TU compile times
    (.ninja_log): Any 53.9s, RightAny 53.4s, All 46.5s, Semi 43.0s, Anti 40.5s, Asof 25.6s,
    AmacProbe 8.5s.

## Extra self-tests beyond DoD
- Shape matrix hash vs parallel_hash(force), tmp/u3_smoke/shape_matrix.out: OK for
  inner/left/right/full mixed-ON (additional-filter routed path), semi/anti left+right,
  any_right, asof_inner, keys256 (3-col), Nullable keys, LowCardinality keys, join_use_nulls=1.
- any_left/any_inner "DIFF" vs hash investigated: ANY-choice under the U2 BUILD ring is
  run-to-run nondeterministic under force on the PRE-U3 binary too (checksum 9156238750851762467
  produced by both binaries; off arm stable and equal to hash). ANY INNER force pre-U3 == post-U3
  exactly. Pre-existing, U2-accepted ("rows reordered - safe for an unordered join",
  AmacRing.h); ANY semantics permit any matching row. NOT a U3 regression.

## Known gaps (for the verification checkpoint, PREREG-007)
- G-parity full matrix (--require-engagement), run_order.sh full 17-check suite + 03448/03711
  x10 + baseline power check, probe A/B vs candidate-7e64a6cf4d5, and G-disasm-probe vs
  bins/clickhouse-ahj-cf465cfbe23.bin: not run here (verification checkpoint's gates).
- Stateless 03448/03711 not run locally (need a scratch server; ports were left untouched).
- Step 8 of the plan's U3 commit list ("extend parallel_hash correctness coverage, 04107
  pattern") was NOT in this task's step list and was not done.
