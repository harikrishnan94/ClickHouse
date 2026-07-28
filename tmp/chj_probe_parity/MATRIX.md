# Gate cell list: probe-side win-or-parity mission (FROZEN)

Frozen 2026-07-28 at design approval (DESIGN.md REV 3). Derived from the
prior campaign's measured plan (`tmp/chj_amac/fleet/matrix.json`, 94 cells)
so every gate verdict is comparable to the prior campaign's raws. Cell name
semantics (family:probe.<kind>.<size>.<threads>[.modifier]) unchanged.

## Gate cells (win-or-parity required on Probe-event medians)

### probe_core_grid (27 cells) — family x map-residency are the first-order AMAC axes; T96 is the headline
- key32:probe.inner_all.S2.T96
- key32:probe.inner_all.S3.T96
- key32:probe.inner_all.S5.T96
- key64:probe.inner_all.S2.T96
- key64:probe.inner_all.S3.T96
- key64:probe.inner_all.S5.T96
- str:probe.inner_all.S2.T96
- str:probe.inner_all.S3.T96
- str:probe.inner_all.S5.T96
- fixstr:probe.inner_all.S2.T96
- fixstr:probe.inner_all.S3.T96
- fixstr:probe.inner_all.S5.T96
- k128:probe.inner_all.S2.T96
- k128:probe.inner_all.S3.T96
- k128:probe.inner_all.S5.T96
- k256:probe.inner_all.S2.T96
- k256:probe.inner_all.S3.T96
- k256:probe.inner_all.S5.T96
- null64:probe.inner_all.S2.T96
- null64:probe.inner_all.S3.T96
- null64:probe.inner_all.S5.T96
- lcstr:probe.inner_all.S2.T96
- lcstr:probe.inner_all.S3.T96
- lcstr:probe.inner_all.S5.T96
- mixed:probe.inner_all.S2.T96
- mixed:probe.inner_all.S3.T96
- mixed:probe.inner_all.S5.T96

### size_ladder (6 cells) — full 5-point ladder on 3 sentinel families locates the engagement knee
- key64:probe.inner_all.S1.T96
- key64:probe.inner_all.S4.T96
- str:probe.inner_all.S1.T96
- str:probe.inner_all.S4.T96
- k256:probe.inner_all.S1.T96
- k256:probe.inner_all.S4.T96

### thread_ladder (12 cells) — ring + ordered-probe costs scale with lanes; T1 isolates single-slot
- key64:probe.inner_all.S2.T1
- key64:probe.inner_all.S2.T48
- key64:probe.inner_all.S4.T1
- key64:probe.inner_all.S4.T48
- str:probe.inner_all.S2.T1
- str:probe.inner_all.S2.T48
- str:probe.inner_all.S4.T1
- str:probe.inner_all.S4.T48
- k256:probe.inner_all.S2.T1
- k256:probe.inner_all.S2.T48
- k256:probe.inner_all.S4.T1
- k256:probe.inner_all.S4.T48

### kind_strictness (24 cells) — one measured point per instantiation group per sentinel family per residency class
- key64:probe.left_all.S2.T96
- key64:probe.left_all.S4.T96
- str:probe.left_all.S2.T96
- str:probe.left_all.S4.T96
- key64:probe.rf_all.S2.T96
- key64:probe.rf_all.S4.T96
- str:probe.rf_all.S2.T96
- str:probe.rf_all.S4.T96
- key64:probe.any.S2.T96
- key64:probe.any.S4.T96
- str:probe.any.S2.T96
- str:probe.any.S4.T96
- key64:probe.semi_anti.S2.T96
- key64:probe.semi_anti.S4.T96
- str:probe.semi_anti.S2.T96
- str:probe.semi_anti.S4.T96
- key64:probe.semi_anti.S2.T96.anti
- key64:probe.semi_anti.S4.T96.anti
- str:probe.semi_anti.S2.T96.anti
- str:probe.semi_anti.S4.T96.anti
- key64:probe.asof.S2.T96
- key64:probe.asof.S4.T96
- str:probe.asof.S2.T96
- str:probe.asof.S4.T96

### dup_heavy (4 cells) — duplicate chains change ring occupancy and `RowRefList` appends
- key64:build.inner_all.S3.T96.dup16
- key64:build.left_all.S3.T96.dup16
- str:build.inner_all.S3.T96.dup16
- str:build.left_all.S3.T96.dup16

### hit_rate (4 cells) — miss-dominated probes stress the ring differently
- key64:probe.inner_all.S3.T96.h50
- key64:probe.inner_all.S3.T96.h05
- str:probe.inner_all.S3.T96.h50
- str:probe.inner_all.S3.T96.h05

### join_use_nulls (1 cells) — nullable output path interacts with ordered gather
- key64:probe.left_all.S3.T96.jun

### stats_on (2 cells) — protocol-sensitivity check for the stats-off measurement decision
- key64:build.inner_all.S3.T96.statson
- key64:probe.inner_all.S3.T96.statson

### new_mixed_on (1 cell) — item 6 drop-decision check: remainder route re-derivation must be in-band after the fold
- key64:probe.mixed_on.S3.T96   (ANY LEFT + additional non-equi ON filter; max_joined_block_rows default)

### new_threshold_boundary (2 cells x 3 hook arms) — item 7 companion: ring-vs-flat boundary re-measured against the flat loop
- key64:probe.inner_all.S1p5.T96   (arms: default / CLICKHOUSE_JOIN_AMAC=force / =0; S1p5 = aggregate map bytes ~2x getMinBytesForPrefetchInJoin, calibrated at U5 and recorded)
- str:probe.inner_all.S1p5.T96   (arms: default / CLICKHOUSE_JOIN_AMAC=force / =0; S1p5 = aggregate map bytes ~2x getMinBytesForPrefetchInJoin, calibrated at U5 and recorded)

Total gate cells: 83

## Guard cells (in-band required; not gate wins)

### build (14 cells) — build events must not regress; kind is second-order on build (shared insert path)
- key64:build.inner_all.S2.T96
- key64:build.inner_all.S3.T96
- key64:build.inner_all.S5.T96
- str:build.inner_all.S2.T96
- str:build.inner_all.S3.T96
- str:build.inner_all.S5.T96
- k256:build.inner_all.S2.T96
- k256:build.inner_all.S3.T96
- k256:build.inner_all.S5.T96
- mixed:build.inner_all.S2.T96
- mixed:build.inner_all.S3.T96
- mixed:build.inner_all.S5.T96
- key64:build.inner_all.S3.T1
- key64:build.inner_all.S3.T48

Total guard cells: 14

## Coverage boundaries (recorded honestly)

- Wrapped-plan probe (item 7 wrap_aware) is unreachable from SQL
  deterministically; covered by gtest with a degenerate-hash map. No cell.
- x86 route-word quality (multiply-shift variant) is not measured by the
  ARM fleet; recorded NOT-CLAIMED unless an x86 spot-check is run.
- The 20 prior loss cells are all members of the blocks above (verified:
  they came from this same measured plan); no separate block needed.
