# REPORT — Unit U1: Component port — GREEN

Closed 2026-07-09. Iterations: 3 (port → ASan finding fix → review fixes). Review: single
adversarial pass (low-risk tier), verdict SHIP-WITH-FIXES, all items applied/documented, no
blocking findings, no open leads (bep/reviews.md).

## What landed
- `src/Interpreters/RadixHashJoin/` component library ported from `origin/phj5-real`:
  Arena, BuildSide, Hll, KeyLayout, KeyRefScatter, LeafTable, PackedKeyHash, ParallelFor,
  PartitionPlan (+ donor gtest, 40 tests). Adaptations were purely mechanical
  (`BuildRef`→`RowRef`, `BuildRefList`→`RowRefList`, `.word()`→`.encode()`, donor `fromWord`
  → bit-exact local `rowRefFromWord`), verified by normalized diff (residue zero) and the
  donor's own suite.
- `ColumnScatter.{h,cpp}` + `gtest_column_scatter.cpp`: production port of the working branch's
  bench scatter kernels (histogram + 4-lane interleave, fused prefix-sum + exact one-shot
  allocation, DIRECT/SWWC+NT paths, multi-pass refine, fused single-dispatch barrier wave loop)
  per pre-registered deviations D1–D5, with exception-safe barriers (arrive-always + stop
  snapshot), cancellation, no globals, tracker-visible allocations, element widths
  {1,2,4,8,16,32,64} incl. Decimal/UUID/IP/DateTime64 columns.
- One CMake line (`add_object_library` at src/CMakeLists.txt:343).

## Findings of note
- Donor test-helper heap-buffer-overflow (gtest_radix_hash_join.cpp:284, kw4) found by the ASan
  gate — a pre-existing donor defect, fixed here (L0006).
- Review findings R1–R8: exact-allocation regression (fixed via resize_exact), whole-side pid
  transient in scatterColumns (documented; U3 must charge it or use scatterWaves), late type
  validation (fixed), missing wave×SWWC completion test (added), p_star≤1 contract (documented +
  pinned), fanout ceiling enforcement (clamped in computePassBits), post-failure consume
  promptness (documented + best-effort stop). R8 (phase-body duplication) accepted, revisit at U3.

## Evidence (final tree)
| Gate | Result | Raw |
|---|---|---|
| reldeb build + suite | 49 tests: 48 PASS, 1 pre-registered ARM skip | build/reldeb/{build_u1_final2,test_u1_final2_reldeb}.log |
| ASan | same counts, 0 reports | build/asan/test_u1_final2_asan.log |
| TSan (ColumnScatter ×5 repeats) | 9/9 ×5, 0 warnings | build/tsan/test_u1_final2_tsan.log |
| x86-64 compile (reviewer) | 6 production TUs clean | tmp/review_x86_compile.log |
| Style | check-style: zero findings | (transcript) |
| Donor equivalence | normalized diff residue zero + donor suite green | bep/reviews.md |

## §5 Intentions status
- Intention 1 (replicate bench wins in SQL): not yet exercised — U1 provides the verified
  substrate (leaf tables + AMAC probe behaviorally donor-equal; scatter kernels with the
  measured properties preserved: exact allocation, ≥4 KiB/partition/column windows, no
  inner-loop atomics, SWWC≥256 — the ARM SWWC path proven bit-equal to DIRECT).
- Intention 2 (do no harm): no runtime path references the new code yet; every existing test
  surface untouched; production binary builds.

## Carried obligations for U2/U3
- U2: port RadixHashJoin.{h,cpp} with hasPostBuildPhase/runPostBuildPhase, header-safe joinBlock,
  StoredColumnsIndex/StoredBlock migration, lane plumbing, settings history block 26.7.
- U3: dedicated fully-schedulable pool for the wave loop; budget must charge 2 B/row pid scratch
  (or use per-window scatterWaves) + ~76 B/partition/worker SWWC state; routes = ColumnUInt32,
  CRC-independent, high-bits-first; variable-width payloads (String/Nullable/LC/Array) need a
  fallback path; consume callbacks must tolerate non-prompt stop after a sibling failure.
