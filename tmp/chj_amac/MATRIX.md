# Coverage matrix — frozen (Unit 1)

Scope: `ConcurrentHashJoin` (`join_algorithm = 'parallel_hash'`) build and
probe paths, candidate (`phj-ph`) vs baseline
(`concurrent-hash-join-profile-events` @ `a05f3ee81ff`). This file is the
authoritative freeze of the disposition universe and the measured subset;
`fleet/matrix.json` (generated) must agree with it — on any conflict this
file wins and the generator gets fixed.

## Axes

- side ∈ {build, probe} (2)
- family (10): key32 (repr. also for key8/16), key64, str, strzero, fixstr,
  k128, k256, null64, lcstr, mixed
- group (6): probe-instantiation classes over kind × strictness —
  inner_all (INNER ALL), left_all (LEFT ALL), rf_all (RIGHT/FULL ALL,
  used-flags), any (INNER/LEFT ANY + RightAny), semi_anti (LEFT/RIGHT
  SEMI + ANTI), asof (ASOF INNER/LEFT)
- size (5): S1..S5 = nominal aggregate map bytes 1 MiB / 32 MiB / 1 GiB /
  4 GiB / 16 GiB (see calibration caveats)
- threads (3): T1, T48, T96

Universe: 2 × 10 × 6 × 5 × 3 = **1,800 cells**, plus the 12-cell
G-hash-inband list (below), which is tracked separately (different
algorithm, not part of the parallel_hash universe).

## Disposition vocabulary (every cell gets exactly one; G-coverage requires 0
undispositioned)

- `MEASURED(cell_id)` — ≥5 valid fleet runs per arm on one shard.
- `INFERRED(from=<measured cell>, rule=<family-repr | group-repr |
  size-interp | thread-interp>)` — covered by a measured representative.
- `PARITY-ONLY` — correctness axis only, no perf claim (all strzero cells;
  most lcstr cells).
- `EXCLUDED-INVALID(reason)` — no such SQL shape (e.g. asof × lcstr) or
  structurally unreachable size (see calibration).
- `NOT-CLAIMED(reason)` — AMAC intentionally disengaged (e.g. cache-resident
  sizes); sampled by G-force-engage.

## Measured subset — 94 parallel_hash cells in 9 blocks

| # | block | cells | rationale |
|---|-------|------:|-----------|
| 1 | Probe core grid: {key32,key64,str,fixstr,k128,k256,null64,lcstr,mixed} × {S2,S3,S5} × T96 × inner_all | 27 | family × map-residency are the first-order AMAC axes; T96 is the headline |
| 2 | Size-ladder completion: {key64,str,k256} × {S1,S4} × T96 × inner_all | 6 | full 5-point ladder on 3 sentinel families locates the engagement knee |
| 3 | Thread ladder: {key64,str,k256} × {S2,S4} × {T1,T48} × inner_all | 12 | ring + ordered-probe costs scale with lanes; T1 isolates single-slot |
| 4 | Kind/strictness: {left_all, rf_all(RIGHT ALL), any(INNER ANY), semi_anti(LEFT SEMI), semi_anti(LEFT ANTI), asof(INNER ASOF)} × {key64,str} × {S2,S4} × T96 | 24 | one measured point per instantiation group per sentinel family per residency class |
| 5 | Build side: {key64,str,k256,mixed} × {S2,S3,S5} × T96 × inner_all, plus key64 × S3 × {T1,T48} | 14 | build events must not regress; kind is second-order on build (shared insert path) |
| 6 | Duplicate-heavy build (dup=16): {key64,str} × S3 × T96 × {inner_all,left_all} | 4 | duplicate chains change ring occupancy and `RowRefList` appends |
| 7 | Hit-rate: h ∈ {0.5, 0.05}: {key64,str} × S3 × T96 × inner_all | 4 | miss-dominated probes stress the ring differently |
| 8 | join_use_nulls=1: key64 × S3 × T96 × left_all | 1 | nullable output path interacts with ordered gather |
| 9 | Stats-on sensitivity: key64 × S3 × T96 × {build,probe} with `collect_hash_table_stats_during_joins=1` | 2 | protocol-sensitivity check for the stats-off measurement decision |

Blocks 1-4 and 7-9 are probe-side; blocks 5-6 build-side. Every other
universe cell is INFERRED from its block representative, PARITY-ONLY,
EXCLUDED-INVALID, or NOT-CLAIMED — assigned per cell in
`fleet/dispositions.json` during Unit 4 and checked by
`fleet/check_matrix.py`.

## G-hash-inband list (12 cells, `join_algorithm='hash'` on BOTH arms)

{key64, str, k256} × {S2, S4} × {T1, T96} — guards the tail-padded-grower
change to the shared join maps (requester decision 4). PASS = every cell
in-band vs baseline.

## Dataset parameters

Frozen row counts per (family, size): `calibration/calibration.json`
(58 measured points on the baseline binary, method + residuals in
`calibration/CALIBRATION.md`). Defaults: dup=1, probe rows = max(4×D, 50M)
capped by memory sanity, hit rate 1.0 unless the block says otherwise;
dup16/h50/h05 per block. Memory-engine tables, deterministic `numbers()`
fills.

## Calibration caveats (recorded, binding on interpretation)

1. The ladder was calibrated on the BASELINE binary, whose slots merge into
   ONE two-level map (256 buckets, load 0.5). The candidate keeps per-slot
   maps; for identical D its aggregate map bytes differ by power-of-two
   quantization across slots. Cells are therefore defined by **frozen row
   counts**, and S-labels are nominal residency classes, not per-arm byte
   guarantees. Both arms of a cell always run identical datasets.
2. Sizes land on plateaus (cells are 16/24/32/40 B; growers are
   power-of-two): S2 for the 16 B families (key32/key64/null64) realizes
   16 MiB (−50% vs nominal); still the L3-resident class on both venues
   (local L3 36 MiB, m8g.24xlarge similar). Recorded per cell in
   calibration.json.
3. S1 (1 MiB, L2-resident) is reachable only for key32/key64/null64. For
   str/strzero/fixstr/k128/mixed the map floor is ~1.5-2.5 MiB (still
   per-core-L2-adjacent on Graviton4: 2 MiB L2/core), k256 ~2.5 MiB, lcstr
   ~14 MiB (L3). S1 cells for lcstr are EXCLUDED-INVALID(size floor); S1
   for the mid families is retained with the actual floor recorded.
   G-force-engage's designated cell key64:probe.inner_all.S1.T96 is
   unaffected (true 1 MiB).
4. lcstr's join-reported byte counter double-counts shared per-block
   dictionaries (~2× vs tracked peak). lcstr is an expected AMAC exclusion
   family (LowCardinality getter); its measured cells guard against
   regression, not for AMAC wins.
5. S5 rows are analytic extrapolation one grower step beyond the measured
   range (S4 empirically validated for key64/str/lcstr); if a fleet S5 cell
   shows unexpected memory, recalibrate before trusting its verdict.
6. Duration floor (from the PREREG-002c attempt-1 failure): cells whose
   timed query runs under 200 ms at their thread count are jitter-bound
   (same-binary S2×T96 medians of 23-42 ms produced an 11% spurious "LOSS";
   the ≥200 ms cell was tight at −0.17%). `fleet_ab.py` enforces this two
   ways: probe-side cells get ≥2M probe rows per thread, and every cell's
   per-arm median must clear 200 ms or all its runs are marked invalid
   (fail-closed). Build-side cells cannot inflate build rows without
   changing the map size, so small-size build cells at high T (e.g. S2×T96
   in block 5) are EXPECTED to trip the floor; they will be re-dispositioned
   during the campaign with recorded rationale (S3/S5 coverage substitutes)
   rather than measured as noise.

## Venue and acceptance rules

Local host (96-core Graviton4) = orientation + noise-band only. Acceptance
numbers exclusively from the 8× m8g.24xlarge fleet, paired ABAB, ≥5 valid
runs/arm/cell, band = max(3%, frozen per-shape A/A spread), wins must appear
in the claimed phase event (`ConcurrentHashJoinBuildInsertMicroseconds` /
`...ProbeLookupMicroseconds`).
