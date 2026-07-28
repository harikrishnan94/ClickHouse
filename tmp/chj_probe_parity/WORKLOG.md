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
