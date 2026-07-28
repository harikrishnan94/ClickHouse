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

## 002 — U1b: dispatch flip to the fold + single key preparation

- Change (one commit, build+probe together — any consistent route is
  parity-neutral only if both sides flip at once):
  (a) probe: `ConcurrentHashJoin::joinBlock` builds the block's
  `JoinOnKeyColumns` once (materialize + nested unwrap + mask, LC kept
  for LC map types), routes over its prepared `key_columns` (ASOF
  equality prefix) via `JoinSlotRouting::computeJoinSlotIds` into a
  `PaddedPODArray<UInt8>`, stores both in `RoutedJoinResult`;
  `HashJoin::joinRoutedBlock` takes the vector + `const UInt8 *`
  (slot_ids UInt64->UInt8 through RoutedHashJoinMethods /
  RoutedProbeContext / AmacProbe);
  (b) build: `dispatchBlock` routes via a shared prep+fold helper into
  UInt8 ids; `scatterBlocksWithSelector` consumes them directly;
  `scatterBlocksByCopying` widens once at the `IColumn::scatter`
  boundary;
  (c) `key8`/`key16` keep low-bit-of-key routing (`value & (slots-1)`,
  bit-identical to today's `FixedHashTable::hash` identity selector);
  (d) dead code deleted: `calculateHashes`, `hashToSelector`,
  `routeByHighBits`, `selectDispatchBlock`'s per-family KeyGetter
  switch; event descriptions updated (names frozen).
- Expectation: G-build green; G-parity `PARITY OK` (636 cases incl.
  force-engagement staged counters); G-order `ORDER OK`; G-tests
  candidate failures subset of baseline; gtests all green
  (JoinSlotRouting.* + ConcurrentHashJoinAmac.*); `hash` join
  algorithm codegen NFC (routing lives in ConcurrentHashJoin only).
- Local orientation A/B (labeled local, not acceptance): mixed S2/S3,
  key64 S2, fixstr S2 probe cells — expect ProbeDispatch to collapse
  (mixed: SipHash-128 pass -> fold); no PLook regression outside local
  noise.
- Refutation: any parity/order diff -> route inconsistency between
  build and probe (fold contract broken) or prep divergence; fix
  before proceeding, never weaken the check. Slot-balance check
  (PREREG 001 gtest) already pins distribution.

## 003 — Fix: routed probe's block-size estimate reads slot 0 only

- FINDING (from the join-selector differential this mission added —
  the prior mission's named gap): `03567_max_joined_block_size_bytes`
  fails on routed-probe binaries (pre-existing at `5b276c5fb88`,
  passes on the two-level baseline). Mechanism, confirmed by
  arithmetic: `joinBlockImpl` fills
  `HashJoinResult::Properties::avg_joined_bytes_per_row` from SLOT 0
  (`join.data->allocated_size / max(1, join.data->rows_to_join)`);
  with few distinct keys most slots are empty, slot 0's ratio is 0,
  the estimate collapses to the left block's bytes/row (8), and
  `max_rows = 1MiB / 8 = 131072` — the exact unsplit block the test
  sees. The two-level baseline read the merged whole-join map.
- Change: aggregate `allocated_size` and `rows_to_join` across ALL
  `slot_joins` for the estimate (equals the baseline's whole-join
  semantics). O(slots) per block; hoisted once-per-build later by
  item 4.
- Expectation: 03567 passes on the fixed binary (x3 runs); full join
  differential becomes candidate ⊆ baseline (empty cand-only list);
  parity + order stay green on the new snapshot.
- Refutation: 03567 still failing -> mechanism wrong, re-diagnose
  before any further commit; never weaken the test.

## 004 — U2: probe-lane plumbing + pooled ProbeScratch

- Change: (a) `IJoin::joinBlock(Block, size_t lane)` defaulted virtual
  (forwards to the lane-less overload; no other join touched);
  `JoiningTransform` gains a `stream_index` constructor parameter
  (the `joinPipelinesRightLeft` per-stream loop index) and passes it
  per `joinBlock` call; (b) `ConcurrentHashJoin::ProbeScratch`
  {`PaddedPODArray<UInt8>` slot_ids, `PaddedPODArray<UInt64>`
  found_word, found_offset} parked in a fixed table of
  2 x slots-hint atomic slots (acquire = `exchange(nullptr)`,
  release = `compare_exchange`; mutexed pool fallback for collision
  losers, out-of-range lanes, and the lane-less `invalid_lane`
  entries); (c) the scratch is owned by `RoutedJoinResult` and
  CAS-released in its destructor (the lookup is lazy - this is the
  deliberate deviation from the eager AHJ scope-exit release);
  (d) acquisition stays lazy: slots == 1 and empty blocks allocate
  nothing; (e) the AMAC find pass's found_word/found_offset arrays
  come from the same scratch (no per-call PaddedPODArray locals).
- Expectation: parity/order/tests all stay green (affinity-only
  change, zero correctness role); pool gtests prove collision safety
  (two results alive on one lane), invalid_lane fallback, and
  capacity reuse across blocks (no per-block allocation after
  warm-up, asserted via allocation probe or capacity identity);
  local orientation A/B on S1/S2 floor cells (key64 S1, str S1,
  key64 S2) - expect small PDisp/alloc shaving, no regressions
  outside local noise.
- Refutation: any parity diff -> ownership/lifetime bug (scratch
  reused while a result still reads it); fix before proceeding.
  Memory accounting note: scratches live on the join until dtor,
  uncounted in getTotalByteCount (AHJ precedent) - documented.

## 005 — U3: once-built slot tables, then the flat descriptor loop

- Commit (a), once-built tables: after the build finishes (all slots
  through `onBuildPhaseFinish`; collection runs AFTER it because
  shrink-to-fit mutates `allocated_size`), `ConcurrentHashJoin`
  collects once per join: per-map-type slot descriptor arrays
  ({buf, mask} per slot), type-erased map pointers, used-flags
  pointers, per-slot wrap bits (last tail-pad cell occupied) OR'd
  into the plan bool, the AMAC engagement decision (aggregate bytes
  threshold), and the whole-join bytes-per-row estimate - replacing
  the per-probe-block O(slots) passes (descriptor build in the AMAC
  arm, `maps_by_slot`/`flags_by_slot` vectors, the estimate loop).
- Commit (b), flat descriptor fused loop: for `has_cheap_key_
  calculation` cursor-capable families (key32/64, keys32/64/128/256)
  below the AMAC engagement gate, replace the plain routed loop with
  a fused find+emit loop reading `cell = desc.buf + (hash &
  desc.mask)` from the once-built descriptor array (wrap-aware walk
  through the tail pad - also the wrapped-chain fallback), with its
  own adaptive look-ahead prefetcher (home-cell line, same
  L2-outgrowth gate, mutually exclusive with AMAC). Strings,
  `hashed`, LC, `key8`/`key16` keep the plain loop.
- Expectation: G-build green; gtests all green (existing 23 + any
  added flat-loop parity tests); PARITY OK; ORDER OK; G-disasm: flat
  loop key64 + keys256 anchors instruction-equivalent to the ahj
  reference binary `c8260c682b78...` (llvm-nm/objdump address ranges,
  never the analyze-assembly cache); local orientation A/B on
  below-threshold cells (key64 S1/S2, k128 S2) - expect PLook
  improvement or in-band, no regressions outside local noise; the
  hoist alone (commit a) must be perf-neutral-or-better.
- Refutation: any parity/order diff -> descriptor/lifetime bug; any
  unexplained disasm delta -> port infidelity; fix before proceeding.

## 006 — U4: ASOF pointer-recording ring, then the AmacWalk policy

- Commit (a), ASOF ring (probe-only; the single-slot ASOF plan is
  explicitly NOT adopted): the find ring gains the ASOF maps by
  recording the matched cell's mapped POINTER bits in `found_word`
  (never 0 for a match; the probe maps are immutable, so the pointer
  stays valid into phase B). Phase B stays the full precomputed loop,
  rebuilding the `FindResult` from the pointer and running `findAsof`
  as today. The dispatch-free `word_loop` continues to exclude ASOF.
  New `amacFindPass` instantiations for the ASOF maps land in
  `AmacProbe.cpp`; the compile-time and binary-size deltas are
  measured and reported. Tests: ring-vs-plain ASOF parity gtests
  (duplicate-heavy keys, inequality boundaries for all four
  inequalities), order; the full parity/order harnesses re-run.
- Commit (b), `AmacWalk` policy (DESIGN item 7): a second compile-time
  axis {bare, wrap_aware} on the find pass. `bare` stays byte-for-byte
  today's steady loop (the ahj-anchored one); `wrap_aware` adds a
  per-frame descriptor lane and wraps step and step-prefetch at the
  pad end, replicating the grower's `next`. The engagement gate drops
  the `!chain_may_wrap` term - a wrapped plan selects the wrap-aware
  instantiation per `joinBlock` instead of disengaging; force mode
  now engages on wrapped plans too. Tests: a degenerate-hash gtest
  forcing a pad-spanning cluster (asserts the dispatcher never picks
  `bare` on a wrapped map and ring results equal the sequential
  find); SQL cannot reach wrapped plans deterministically - recorded
  as the coverage boundary.
- Expectation: all gates stay green (build, gtests, parity 636,
  order); G-disasm: the BARE anchors must remain instruction-equal to
  the ahj reference (the policy axis must not perturb them);
  `wrap_aware` and the ASOF ring get standalone review anchors (no
  ahj counterpart exists - ahj disengages on wrapped plans and rings
  ASOF only under its single-leaf plan). Local orientation A/B on
  asof cells (key64 S2-equivalent asof, str asof) - expect
  `ProbeLookup` improvement or in-band.
- Refutation: any parity diff -> pointer-lifetime or boundary bug;
  any bare-anchor delta -> the axis leaked into the hot loop; fix
  before proceeding, never weaken.
