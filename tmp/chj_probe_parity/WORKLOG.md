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
