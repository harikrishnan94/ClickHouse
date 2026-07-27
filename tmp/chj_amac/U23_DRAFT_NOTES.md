# U2.3 draft notes — AMAC build-insert ring for ConcurrentHashJoin

Implementer draft (uncommitted; orchestrator commits). Base: `phj-ph` @ `4a32708e08a`.
Design: approved plan + PREREG-006. Reference: `ahj` branch (`AmacRing.h`,
`PartitionedHashJoinBuild.cpp`), harness pattern from
`6613c51c2d4:src/Interpreters/tests/gtest_partitioned_hash_join.cpp`.

## Files created
- `src/Interpreters/HashJoin/AmacRing.h` — ring driver (`amacRun`), growth cancellation
  (`amacDrainAndGrow`), constants (`amac_ring_size`=32, `amac_min_rows`=256,
  `amac_inactive_row`), `AmacStepResult`, `AmacResumableMap` concept, getter-exclusion traits,
  `amac_join_supported`. Provenance cited in the header.
- `src/Interpreters/HashJoin/AmacMode.h` / `.cpp` — process hook `CLICKHOUSE_JOIN_AMAC`
  (Off/Auto/Force), `joinAmacMode` (env read once), `setAmacModeForTests`. Diagnostic hook,
  not a user-facing Setting (requester decision, stated in the comment).
- `src/Interpreters/HashJoin/AmacBuild.h` / `.cpp` — `AmacBuildInsertResult`,
  `amacBuildInsert<KeyGetter, Map, selector_is_range>`, `AmacBuildInsertPolicy` (in the .cpp,
  anonymous namespace), 32 explicit instantiations (8 families x {MapsOne, MapsAll} x
  {range, indexes}) shared between header externs and .cpp definitions via one macro.
- `src/Interpreters/tests/gtest_concurrent_hash_join_amac.cpp` — 4 tests (3 specified + 1
  regression, see below).

## Files modified
- `src/Interpreters/HashJoin/HashJoinMethodsImpl.h` — engagement branch in
  `insertFromBlockImplTypeCase` before the prefetch block: compile-time gate
  (`!is_asof_join && amac_join_supported`), runtime predicate
  (mode != Off && `join.amacEnabled()` && (Force || (buffer bytes > `getMinBytesForPrefetchInJoin()`
  && rows >= `amac_min_rows`)) && rows < `amac_inactive_row`), merged skip-byte array
  (null_map ∪ join_mask, with `saw_null_row`), call, exact `is_inserted`/`all_values_unique`
  accumulation, single ProfileEvents increment per section.
- `src/Interpreters/HashJoin/HashJoin.h` — `amac_enabled` member + inline
  `setAmacEnabled`/`amacEnabled` (no .cpp change needed; brief allowed the accessor form).
- `src/Interpreters/ConcurrentHashJoin.cpp` — `setAmacEnabled(true)` per slot at construction
  (only `parallel_hash` opts in).
- `src/Common/ProfileEvents.cpp` — `ConcurrentHashJoinAmacBuildRows`,
  `ConcurrentHashJoinAmacBuildRingGrowths` (the existing seven events untouched).
- CMake: nothing to edit — `Interpreters/HashJoin/*.cpp` and `gtest*.cpp` are both globbed
  (`add_object_library` / `grep_gtest_sources` with CONFIGURE_DEPENDS).

## Deviations from the reference / brief (argued)
1. **`reseed` policy hook** (driver calls `policy.reseed`, not `policy.start`, in
   `amacDrainAndGrow`; the `chassert(restarted)` is gone). The frozen brief stores the SOURCE
   row index in the ring while `start` takes SECTION positions (skip bytes are
   section-indexed); re-seeding through `start` would translate/skip-check in the wrong
   domain. `reseed` = seed-only (no skip/zero checks; rows handled synchronously never entered
   the ring), so re-seeding still cannot fail.
2. **Slot-preserving re-seed (bug fix vs the `ahj` reference).** The reference re-seeds pending
   rows into "the first non-skip slots". In the steady phase a failed refill leaves one slot
   inactive while the sweep still finishes; a later `DoneNeedsGrow` in the same sweep then
   moves a row into the dead slot and leaves a not-yet-swept slot empty, and the tail of the
   sweep steps it — dereferencing `amac_inactive_row` (segfault on string keys via
   `offsets[2^32-2]`, silent garbage insert on numeric). Reproduced on the pre-fix build:
   `local` string join, `max_threads=96` (~500-row sections), deterministic SIGSEGV; core
   backtrace in `build/reldeb/core_bt.log` shows `step` at `s=29` with `row=4294967295`.
   Fix: collect (row, slot) pairs and re-seed each row into its own slot — the active set is
   invariant under growth. The latent bug exists in `ahj` (its compact leaf sections made the
   exhaust-then-grow-in-one-sweep window practically unreachable). Teeth check: the new
   regression gtest was built once against the ported (buggy) logic and crashed (rc=134,
   `build/reldeb/gtest_amac_teeth.log`), then passes with the fix.
3. **Row-order (sorted) re-seed.** Pending rows are sorted ascending before re-seeding: slot
   order at collection is not row order (a mid-sweep refill can put a later row into a lower
   slot), and a growth erases the earlier row's visit-count lead, so collection-order re-seed
   could let a later duplicate claim a cell before an earlier one — observable through
   first-wins `RowRef` maps (RightAny/Any parity). Sorting ≤31 UInt32s on the growth-only path
   is free. Ring rows are source indexes and every engaged caller's selector is monotonic
   (`parallel_hash` scatter emits ascending indexes), so ascending == sequential insert order.
4. **`AmacBuildInsertResult` carries {growths, any_inserted, all_unique}; the
   rows-through-ring count lives at the call site** (`rows - skipped_rows`, computed while the
   skip array is built anyway). Threading a row counter through the policy would add a mutated
   field to the refill path for no information gain. Event semantics: rows actually applied to
   the map (zero-key sync rows included, null/masked rows excluded), incremented once per
   section.
5. **Traits in `AmacRing.h`**: no `is_low_cardinality_join_key_getter` existed in
   `KeyGetter.h`; written next to the gate as in the reference.
6. **`AmacMode` as its own small .h/.cpp** (brief explicitly allowed either).
7. **clang-format macro hint not applied**: 4 suggestions inside the `AmacBuild.h`
   instantiation-macro continuation; clang-format's proposal folds the template-id onto one
   long line, which is less readable and not required by the CI style check (which passes).
8. **gtest additions beyond the brief**: (a) 4th test `TinySectionGrowthDuringSweepTail` —
   regression for deviation 2 (tiny 256-row build blocks make a growth in a section's final
   partial sweep frequent; uint + string arms, exact multisets); (b) key `0` added to the
   duplicate-heavy test — the true zero-sentinel synchronous path (empty strings are NOT the
   string zero sentinel: a column-backed `StringRef` has a non-null data pointer — the
   empty-string key in the string test covers zero-length persist instead, comment corrected).

## Gate outputs (raw final lines)
All logs below were re-produced against the FINAL source (after the drain fix and the
style-brace fix; `build/reldeb/build_amacbuild_draft.log` is the final rebuild of
`clickhouse` + `unit_tests_dbms`).
1. Build: `build/reldeb/build_amacbuild_draft.log` → `ninja rc=0`, zero errors/warnings
   (subagent-checked on the first full build; final incremental rebuild also rc=0).
2. Gtests (`build/reldeb/gtest_amac_draft.log`, default env):
   `[  PASSED  ] 4 tests.`
   Off-env arm (`CLICKHOUSE_JOIN_AMAC=0`, `build/reldeb/gtest_amac_draft_offenv.log`):
   `[  PASSED  ] 4 tests.` (Force via `setAmacModeForTests` overrides env; Off arms assert
   counters == 0.)
3. Smoke (`build/reldeb/smoke_amacbuild_final.log`), binary
   `tmp/chj_amac/bins/uncommitted-amacbuild.tmp.bin`:
   - uint 3M force: `3000000`, `AmacBuildRows 3000000`, `RingGrowths 512`; off: `3000000`.
   - string 3M force: `3000000`, `AmacBuildRows 3000000`, `RingGrowths 512`; off: `3000000`.
   - string 300k T96 force (former crasher): `300000`, `AmacBuildRows 300000`, `RingGrowths 384`.
   - `join_algorithm='hash'` under force env: `3000000`, no AMAC events (per-join opt-in works).
   - uint 30M T8, env unset (Auto): `30000000`, `AmacBuildRows 29672957`, `RingGrowths 24`
     (early cache-resident sections stay sequential — the intended Auto disengage).
4. clang-tidy-22 (`-p build/reldeb`): `AmacMode.cpp` — 0 findings on my lines (1 justified
   NOLINT for `getenv`); `AmacBuild.cpp` — 0 findings on my lines (352 pre-existing header
   findings recorded in `build/reldeb/tidy_amacbuild.log`, all in base/, Columns/, etc.);
   gtest — 0 findings on my lines (`build/reldeb/tidy_gtest_amac.log`).
5. Style: `ci/jobs/scripts/check_style/check_cpp.sh` → clean (one Allman finding on the
   concept brace was fixed).

## Disasm sanity (bonus, not the formal 006d gate)
All 32 `amacBuildInsert` instantiations (plus the 64 `PosT` lambda bodies) are present in the
final binary. The key64/`RowRefList`/range/`PosT=UInt32` steady loop (symbol @ `0x140f4bc0`)
contains only `prfm pstl1keep, [x26, xN]` prefetches — write intent, L1, keep — indexed off a
register-resident cells base (`x26`), i.e. no policy-field reloads feeding the prefetch
address. `llvm-objdump-22` symlinks for `.claude/tools/analyze-assembly.py` are in
`tmp/bintools/` (the tool needs them on PATH).

## Known gaps
- PREREG-006 gates (c) fleet A/B perf and (d) G-disasm-build vs the `ahj` anchors are the
  orchestrator's; not run here (the bonus check above is a sanity read, not the normalized
  anchor comparison).
- Auto never engages when the whole build fits under L2 per slot (e.g. 96-slot 3M-row local
  runs) — by design; Force exists for tests/harnesses.
- The parity harness's force pass should see AmacBuildRows > 0 in exactly the 8 families;
  lcstr/mixed/hashed/key8/key16/range are compile-time excluded (`amac_join_supported`).
- `build/reldeb/build_unit_tests_teeth.log` and `gtest_amac_teeth.log` were produced against
  the TEMPORARY buggy drain variant (teeth check) and do not reflect the final source.
- The empty gtest run under ASan/TSan was not performed (no such build dir configured in this
  task's scope).
