# REPORT — Unit U2: Algorithm skeleton + build side — ACCEPTANCE-GREEN (review deferred)

Closed 2026-07-09. Status per D-0011 (user directive): acceptance evidence complete and green;
the adversarial review is deferred into the consolidated post-U5 pass that runs in parallel
with U6. Sanitizer gates and the full existing-join-suite run were waived by user directives
D-0009/D-0010 (residual coverage recorded there).

## What landed (four commits)
- `6503c7cfa9a` C1: lane/stream identity plumbing (defaulted `IJoin` overloads; zero behavior
  change for existing joins).
- `d8bd320606c` C2: `join_algorithm='radix_join'` + five `radix_join_*` settings (26.7 history
  block, plan-serialization + `JoinSettings`/`JoinAlgorithmParams` plumbing).
- `4c039eb816c` C3: `RadixHashJoin : IJoin` — donor port adapted to master (`StoredBlock`
  registration, post-build behind `runPostBuildPhase`, LAZY group-granular leaf builds with
  exactly-once CAS protocol and record-memory release, lane-agnostic probe-scratch freelist,
  ProfileEvents/CurrentMetrics/thread name, `RadixHashJoinEntry` stats cache; no perf_event
  counters).
- `0dee4ea0d38` C4: planner gate (`radixHashJoinApplicable`), fallback (ConcurrentHashJoin /
  HashJoin with stats params), dispatch, stateless tests 04508/04509.

## Acceptance evidence (L0011, L0012)
| Gate | Result |
|---|---|
| Equality matrix vs `hash` (pre-registered) | 232/232 PASS: 156 @1e5 (full cross + edges, threads 1 & 32), 72 @1e7 (full cross), 4 @1e8 (subset) |
| gtests (reldeb) | 50 tests: 49 pass + 1 expected aarch64 SWWC skip; incl. lazy exactly-once under 16 threads |
| Stateless 04508/04509 | all rows correct, deterministic ×3 |
| Gate/fallback structure | `EXPLAIN` shows RadixHashJoin / ConcurrentHashJoin / HashJoin exactly per pre-registered row |
| Lazy leaf builds | empty probe ⇒ 0 group builds; probed ⇒ builds == non-empty groups |
| Edge smokes | Sparse/Const right columns, TOTALS, extremes, right-TOTALS subquery, empty sides, max_threads=1 — all equal `hash` |

## Notable deviations from donor (all logged; fold into consolidated review)
`Arena::releaseMany`; lazy builds serialized by `lazy_build_mutex` (U3 removes); reporting-only
`getTotalByteCount` change; overflow-rebuild releases discarded cells; conservative
`max_bucket_bits`; eager path parallel-over-groups; OR-disjunct test uses `'radix_join,hash'`
(multi-disjunct check precedes per-algorithm gates).

## Confirmed hazard carried to U3 (R-d)
Duplicate-heavy probe (20M output rows) emits ONE `JoinResult` block peaking ~1 GB tracked
memory (vs ~18 MB for `hash`). U3's budgeted, streaming emission is the fix; U3 prereg must
include a bounded-emission acceptance check.

## Other U3 obligations
Remove the probe-side double-hash pre-pass (absorb into partition routing); replace
`lazy_build_mutex` with pool-worker-keyed arenas under wave work-stealing; probe-buffer settings
(`fraction/min/max`) validated and stored, unconsumed; lane identity unreliable for right-totals
shapes (freelist pattern or per-lane buffers with the same caveat).

## §5 Intentions status
- Intention 1 (bench wins in SQL): not yet measured (U6); the full algorithm skeleton is live
  end-to-end behind the gate.
- Intention 2 (do no harm): every non-radix_join path behavior-identical by construction
  (defaulted overloads, additive settings) + matrix baselines ran hash/parallel_hash/
  full_sorting_merge through the modified transforms; full-suite/CI verification remains the
  backstop per D-0010.
