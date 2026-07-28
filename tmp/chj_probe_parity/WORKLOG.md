# Worklog: probe-side win-or-parity mission (`phj-ph`)

Mission: DESIGN.md REV 3 (approved by requester 2026-07-28).
Items 1-5 + 7; item 6 dropped. Gate: per-cell median thread-summed
`ConcurrentHashJoinProbeMicroseconds` B ≤ A within A/A band on every
probe cell + build guard + parity/order/tests/disasm + honest-red rule.

## 2026-07-28 — U0

- Design REV 3 approved by requester after two revision rounds
  (rev 2: narrow slot-ids on zero-copy scatter now, pool rationale,
  regime map, ASOF build-impact; rev 3: item 7 `AmacWalk` policy).
- Preconditions verified:
  - Tree clean at `21f6d8043396` (phj-ph HEAD).
  - `tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin` sha256
    `0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4`
    == MANIFEST. Baseline arm binary of record.
  - `tmp/chj_amac/bins/clickhouse-ahj-cf465cfbe23.bin` sha256
    `c8260c682b78ea7cd9beb789b9d517d7c4d810ea73f131b6e31fc482dbf36f6e`
    == MANIFEST. Disasm reference binary of record.
  - `build/reldeb` present (clang-22).
- Evidence tree: `tmp/chj_probe_parity/` (this dir). Prior-mission
  artifacts referenced in place under `tmp/chj_amac/` (matrix, bands,
  raw sweep JSONLs, joinblock diff JSONs).
- Matrix frozen: `MATRIX.md` — 83 gate cells (8 prior probe blocks +
  1 new mixed-ON cell + 2 threshold-boundary cells with 3 hook arms) +
  14 build guard cells; all 20 prior loss cells verified members of
  the gate blocks. Coverage boundaries recorded (wrapped-plan =
  gtest-only; x86 route word NOT-CLAIMED unless spot-checked).
- Build-at-HEAD check: `ninja -C build/reldeb clickhouse` rc=0
  (relink only), log `build/reldeb/build_u0_noop_check.log`.
- PREREG 001 (U1a) registered.

## U1a — JoinSlotRouting fold family (dead code)

- `src/Interpreters/HashJoin/JoinSlotRouting.{h,cpp}` + 
  `src/Interpreters/tests/gtest_join_slot_routing.cpp`.
- G-build: rc=0, 0 errors (`build/reldeb/build_u1a.log`).
- gtests: 9/9 PASSED (`build/reldeb/test_u1a_gtest.log`), incl. the
  PREREG 001 contract checks: single-numeric == `routeWord`; LC ==
  plain-string words; unrolled == reference chain (2/3/4/5 cols);
  wide-numeric byte fold; embedded-zero strings don't collide;
  slot ids == `word >> (32 - bits)` for bits 1..8; distribution
  max/mean < 1.5 at 1M rows x 256 slots (sequential AND random).
- Containment: nothing calls the new code yet; `hash` untouched.

## U1a hygiene pass

- Reports: `hygiene/7dfe941a6d0.reduce.md` (clean; 1 unused include)
  and `hygiene/7dfe941a6d0.humanize.md` (10 findings). Fixer applied
  findings 1-9 + include removal + the same-class `default:` brace;
  finding 10 (evidence file in commit) is the mission's deliberate
  evidence convention - won't fix.
- Re-gates: build rc=0 / 0 errors (`build_u1a_hyg{,2}.log`); gtests
  9/9 (`test_u1a_hyg2_gtest.log`). G-parity re-run deliberately
  deferred to the U1b gate: the commit is comment/name/brace-only on
  code nothing calls yet (documented deviation, not silent).

## U1b — dispatch flip to the fold + single key preparation (in progress)

Edits (all landed in working tree, building):
- `ConcurrentHashJoin.cpp`: `joinBlock` builds `JoinOnKeyColumns` once
  (keep-LC for LC map types) and derives UInt8 slot ids over the
  routed key prefix (`routeKeyColumns`; ASOF drops the trailing
  inequality column) via `computeDispatchSlotIds` (`key8`/`key16`:
  value low bits, bit-identical to the old identity-hash selector;
  everything else: `JoinSlotRouting` fold). `RoutedJoinResult` owns
  the prepared keys + slot ids and hands both to the lookup.
  Build side: `prepareDispatchKeyColumns` mirrors the probe prep
  (`JoinCommon::materializeColumns{,KeepLowCardinality}` + nested
  extraction); `scatterBlocksWithSelector` consumes UInt8 ids
  directly; `scatterBlocksByCopying` gets a one-shot widened
  `IColumn::Selector` (core `IColumn::scatter` signature).
  DELETED: `routeByHighBits`, `hashToSelector`, `calculateHashes`,
  `DispatchKeyShape`/`getDispatchKeyShape`, `selectDispatchBlock`
  (the per-family KeyGetter switch), `BlockHashes`.
- `HashJoin.{h,cpp}`: `joinRoutedBlock` takes `const UInt8 *` +
  `std::vector<JoinOnKeyColumns>` (fwd-declared in the header).
- `HashJoinRoutedMethods{.h,Impl.h}`: `joinBlockImpl` consumes the
  caller's `join_on_keys` (local construction deleted); `slot_ids`
  narrowed to `UInt8 *` through `switchJoinRightColumns` /
  `joinRightColumnsRouted` / `joinRightColumns`.
- `HashJoinMethods.h`: `RoutedProbeContext::slot_ids` -> `UInt8 *`.
- `AmacProbe.{h,cpp}`: find-pass `slot_ids` -> `UInt8 *`.
- `ProfileEvents.cpp`: `ProbeDispatch` description now says key
  preparation + route-word fold (names frozen).
- gtest comment reworded (cited the deleted `hashToSelector`).
970c2db6189a828c421f05ae702b5468363423b843f6ded9295aa5c09f87e79b  tmp/chj_amac/bins/uncommitted-u1b.tmp.bin
- U1b snapshot sha256 970c2db6189a... = `bins/uncommitted-u1b.tmp.bin`.
- G-order: ORDER OK (ok=9 fail=0 source_artifact=8/17, all engaged,
  t1_global=OK, 03448/03711 x10 pass with engagement; log
  tmp/chj_probe_parity/order_u1b.log). GREEN.
- gtests: 19/19 (JoinSlotRouting 9 + ConcurrentHashJoinAmac 10).
- G-parity: PARITY OK (636 cases: 634 compared, 2 matched-error,
  0 failed; force-pass engaged 8/8+2x0 build,probe). GREEN.
- G-tests differential (join selector, 893 tests, both arms — closes
  the prior mission's named gap): 119 baseline failures (all
  environmental on the scratch server) are a subset of the
  candidate's 120; ONE candidate-only failure,
  `03567_max_joined_block_size_bytes` — established PRE-EXISTING at
  `5b276c5fb88` (single-test runs: baseline PASS, 5b276c5fb88 FAIL,
  u1b FAIL). Root cause + fix pre-registered as PREREG 003; the fix
  lands as the next commit and the differential re-runs there.

## U1c — whole-join bytes-per-row estimate fix (PREREG 003)

- Fix: `joinBlockImpl` sums `allocated_size`/`rows_to_join` across all
  `slot_joins` for `HashJoinResult::Properties` (was slot 0 only).
- 03567 x3 PASS on the fixed snapshot (`uncommitted-u1c.tmp.bin`,
  sha a5c83da8c59f...). Build rc=0; gtests 19/19.
- Join differential re-run: cand2 failures = 119 == baseline set;
  candidate-only EMPTY. G-tests GREEN (prior mission's named gap
  closed; found + fixed one pre-existing product defect on the way).
- G-order on u1c: ORDER OK (ok=9 fail=0, all engaged, stateless x10
  pass). GREEN.
- hash-NFC (U1b binary, report hash_nfc_u1b.md): PASS — 14064
  non-Routed `HashJoinMethods` symbols pair 1:1, zero size mismatches;
  anchors instruction-identical modulo GOT/address shifts; the one
  semantic diff is the intended `ldr`->`ldrb` at 2 sites double-guarded
  by the routed-context null check the `hash` path never passes.
  Transfers to u1c (the fix touches a routed-only template).
- G-parity on u1c: running.
- G-parity on u1c: PARITY OK (636: 634 compared, 2 matched-error,
  0 failed; force-pass 8/8+2x0). GREEN. All U1c gates green.
