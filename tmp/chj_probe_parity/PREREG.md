# Pre-registrations: probe-side win-or-parity mission

Rules: every implementing change gets an entry BEFORE the gated action:
expectation, exact invocation, refutation criterion, action on refute.
Numbers from the fleet only for acceptance; local numbers are
orientation and labeled as such.

## 000 — Mission gate (frozen at approval, 2026-07-28)

- Gate: for EVERY probe cell in the frozen cell list (`MATRIX.md`),
  per-cell median of thread-summed
  `ConcurrentHashJoinProbeMicroseconds` satisfies B ≤ A × (1 + band),
  band = per-cell A/A noise band (max(3%, spread)), duration floors
  ≥200 ms/cell and ≥2M probe rows/thread. Wall = secondary sanity.
  `ProbeDispatch`/`ProbeLookup` reported for attribution.
- Guards: build cells in-band (wall + Build events); G-parity; G-order;
  G-tests (candidate failures ⊆ baseline failures); G-disasm (bare ring
  + flat loop anchors vs ahj reference `c8260c682b78...`; wrap_aware +
  ASOF-ring anchors standalone review — no ahj counterpart exists).
- Honest-red: any cell still red after ≤5 pre-registered fix cycles is
  reported red. Banned: weakened checks, local-as-fleet numbers,
  silent deviations, amend/rebase/push.
- Arms: A = saved baseline binary
  `0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4`;
  B = candidate built from phj-ph at the U5 acceptance commit.
  Binary identity by sha256 + /proc/<pid>/exe, never GIT_HASH.

## 001 — U1a: JoinSlotRouting fold family (dead code + gtests)

- Change: new `src/Interpreters/HashJoin/JoinSlotRouting.{h,cpp}` —
  fold primitives (`routeWord` = `__crc32d(-1U, key)` on ARM /
  golden-ratio multiply-shift elsewhere; `mixStep`; `foldBytes` with
  constant-size tail switch; `finalizeRoute = h >> 32`) and the
  route-word computation over prepared key columns (single-numeric
  fast path for widths 1/2/4/8; all-fixed width-8 unrolled fold for
  2/3/4 columns; ColumnString byte fold; live-LowCardinality fold via
  getDataAt value bytes; generic per-column computeHashInto + mixStep).
  Two sinks over one implementation: probe (UInt8 slot ids,
  slot = word >> (32 - bits)) and build (narrow ids or Selector).
  Dead code until U1b. Plus gtest_join_slot_routing.cpp.
- Expectation: G-build green; new gtests green; zero behavior change
  (nothing calls the new code); `hash`-join codegen byte-identical.
- Invocation: ninja clickhouse + unit_tests_dbms (logs in build dir,
  subagent-analyzed); gtest filter JoinSlotRouting*.
- Refutation: any existing test/codegen change → the change is not
  contained; fix before proceeding.
- Contract pinned by gtests: (1) LC column and its materialized plain
  sibling produce identical words per row; (2) Nullable handled by
  caller (nested columns in, same words as plain); (3) all-fixed
  unrolled fold == column-outer accumulation fold (bit-identical);
  (4) slot ids uniform-ish across 2^bits slots for sequential AND
  random UInt64 keys (chi-square sanity bound, prereg: max/mean slot
  fill < 1.5 at 1M rows, 256 slots); (5) build sink and probe sink
  agree on the shared top bits for every bits in [1, 8].
