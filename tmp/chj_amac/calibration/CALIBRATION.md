# parallel_hash size -> build-row calibration (binary a05f3ee81ff, aarch64)

Calibrates build-table row counts `D` per key family so the aggregate hash-map bytes
of a `parallel_hash` join build land near S1=1MiB, S2=32MiB, S3=1GiB, S4=4GiB,
S5=16GiB (targets interpreted as binary units; tolerance +-30%).

- Binary: `tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin`
  (`26.8.1.1`, `GIT_HASH a05f3ee81ff8411759637fa367aad62e72726e71`,
  sha256 `0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4`)
- Server: single instance, `127.0.0.1`, tcp 19710 / http 18710, config in `srv/`,
  log level `trace`; logs left in `srv/server.log*`.
- Probe query (per family, key expressions adapted):
  `SELECT count() FROM (SELECT <keys> FROM numbers(1000)) t INNER JOIN calib_build b ON <cond>`
  with `join_algorithm='parallel_hash'`, `max_threads=32`,
  `collect_hash_table_stats_during_joins=0`, `max_bytes_before_external_join=0`,
  `max_bytes_ratio_before_external_join=0`, `parallel_hash_join_threshold=1`,
  tagged via `log_comment`. Path asserted via
  `ProfileEvents['ConcurrentHashJoinBuildMicroseconds'] > 0` on every point.
- Build tables: `Memory` engine, key columns only, deterministic `numbers_mt`-based
  generators (see `calibration.json` meta.generators), distinct keys, dup=1.
- Driver: `calibrate.py`; raw data: `measurements.jsonl` (58 points);
  ladder emitter: `make_json.py`.

## What "aggregate hash-map bytes" means here {#definition}

On this binary (which still has the two-level hash-join machinery; it predates the
pure-scatter revert at branch HEAD), `parallel_hash` at `max_threads=32` scatters the
build over 32 slots and then **merges everything into ONE two-level hash map**
(256 buckets) on a single slot at build finish. The measured quantity is
`HashJoin::getTotalByteCount` of that data-carrying slot, parsed from the per-shard
`Join data is built, <size> and <rows> rows in hash table` trace lines
(`ReadableSize`, 2 decimals => <=0.5% parse error). The other 31 slots report empty
1-2.5MiB "skeletons" that are nominal only — query peak memory proves they are freed
by the merge.

For `count()`-only probes the join stores **zero right-table columns**, so the
data-slot byte count = map buffers + join pool = exactly the map footprint.
Cross-check: `query_log.memory_usage` tracks map bytes + transients for every
family except `lcstr` (see caveats). Row counts in the log lines were verified
against inserted key counts on every point (including the 0.9 factor for `null64`).

## Empirically established structure {#structure}

All 58 measured points are explained, with zero residual for fixed-width keys, by:

- Merged two-level map: 256 buckets, each an independent hash table,
  min 256 cells, resize when `keys > cells/2` (max load factor 0.5).
- Grower: cells go 256 -> 1K -> 4K -> 16K -> 64K (x4 per resize), then
  x2 per resize from 64Ki cells/bucket on (observed through 2^20 cells/bucket).
  Aggregate plateau = `256 buckets * cells/bucket * cell_size`.
- Cell sizes (mapped value is this branch's tagged 8-byte `RowRefList` word):
  - 16 B: `key32`, `key64`, `null64` (`HashMap<UInt32/UInt64>`; NULLs are not inserted:
    map keys = 0.9*D for `null64`)
  - 24 B: `k128` (`keys128`), `mixed` (`hashed`, UInt128 key hash),
    `fixstr` (**`FixedString(16)` packs into `keys128`**, not `key_fixed_string`:
    `chooseMethod`'s all-fixed <=16B check fires first)
  - 32 B: `str`, `strzero`, `lcstr` (saved-hash string cells; **string keys are NOT
    copied into the arena** — only ~1.0-1.8 B/row of smooth extra on top of cells)
  - 40 B: `k256` (`keys256`)
- `lcstr` (100k distinct, rows duplicated beyond 100k): join-reported bytes are
  linear in rows: `~9.2 MiB + 50.5 B/row` (fit residual <1% over 0.5M..85M rows).
  Duplicate handling verified via probe counts scaling exactly with rows/100k.

Consequence: achievable aggregate sizes are **quantized to the plateau staircase**
(x4 below 256 MiB-equivalent, x2 above, per family cell size). Where a target falls
between plateaus it cannot be hit with dup=1 — those entries carry the nearest
plateau and an out-of-tolerance flag.

## Final ladder (build_rows; deviation of expected map bytes vs target) {#ladder}

| family | S1 (1MiB) | S2 (32MiB) | S3 (1GiB) | S4 (4GiB) | S5 (16GiB) |
|---|---|---|---|---|---|
| key32 | 24,000 (+0%) | 260,000 (**-50%**) | 24,000,000 (+0%) | 96,000,000 (+0%) | 384,000,000 (+0%, extrap.) |
| key64 | 24,000 (+0%) | 260,000 (**-50%**) | 24,000,000 (+0%) | 96,000,000 (+0%, meas.) | 384,000,000 (+0%, extrap.) |
| null64 | 27,000 (+0%) | 290,000 (**-50%**) | 26,700,000 (+0%) | 107,000,000 (+0%) | 427,000,000 (+0%, extrap.) |
| str | 24,000 (**+105%**) | 260,000 (+0.8%) | 12,000,000 (+2.0%) | 48,000,000 (+1.5%, meas.) | 192,000,000 (+1.5%, extrap.) |
| strzero | 24,000 (**+103%**) | 260,000 (+0.8%) | 12,000,000 (+2.0%) | 48,000,000 (+1.5%) | 192,000,000 (+1.5%, extrap.) |
| fixstr | 24,000 (**+50%**) | 260,000 (-25%) | 12,000,000 (-25%) | 48,000,000 (-25%) | 192,000,000 (-25%, extrap.) |
| k128 | 24,000 (**+50%**) | 260,000 (-25%) | 12,000,000 (-25%) | 48,000,000 (-25%) | 192,000,000 (-25%, extrap.) |
| k256 | 24,000 (**+150%**) | 260,000 (+25%) | 12,000,000 (+25%) | 48,000,000 (+25%) | 192,000,000 (+25%, extrap.) |
| mixed | 24,000 (**+50%**) | 260,000 (-25%) | 12,000,000 (-25%) | 48,000,000 (-25%) | 192,000,000 (-25%, extrap.) |
| lcstr | 100,000 (**+1303%**) | 500,000 (+3.9%) | 21,000,000 (-0.5%) | 85,000,000 (+0.2%, meas.) | 340,000,000 (+0.1%, extrap.) |

Bold = out of the +-30% tolerance (structurally unreachable, see caveats).
All `build_rows` sit mid-plateau of the resize staircase (fill ~0.36-0.47 of the
half-load limit), so they are robust to small key-distribution jitter.
`lcstr` rows ride a linear (smooth) model instead, so they hit targets near-exactly.

## Raw measured points {#raw-points}

map = data-slot `getTotalByteCount`; peak = `query_log.memory_usage`;
tbl = Memory-table `total_bytes`; build_us = `ConcurrentHashJoinBuildMicroseconds`.

| family | D | map MiB | peak MiB | tbl MiB | build_us | probe count |
|---|---|---|---|---|---|---|
| fixstr | 24,000 | 1.50 | 49.9 | 0.5 | 1,658 | 1,000 |
| fixstr | 260,000 | 24.00 | 74.2 | 4.0 | 20,625 | 1,000 |
| fixstr | 480,000 | 24.00 | 92.4 | 7.5 | 29,227 | 1,000 |
| fixstr | 2,000,000 | 96.00 | 238.1 | 31.0 | 184,743 | 1,000 |
| fixstr | 12,000,000 | 768.00 | 965.7 | 183.5 | 1,377,944 | 1,000 |
| fixstr | 16,000,000 | 768.00 | 977.8 | 245.0 | 1,877,562 | 1,000 |
| k128 | 24,000 | 1.50 | 50.0 | 0.4 | 1,891 | 1,000 |
| k128 | 260,000 | 24.00 | 66.2 | 4.0 | 20,971 | 1,000 |
| k128 | 400,000 | 24.00 | 67.1 | 6.1 | 22,299 | 1,000 |
| k128 | 12,000,000 | 768.00 | 932.1 | 183.3 | 1,208,168 | 1,000 |
| k128 | 16,000,000 | 768.00 | 968.8 | 244.6 | 1,965,193 | 1,000 |
| k256 | 24,000 | 2.50 | 82.1 | 0.9 | 1,707 | 1,000 |
| k256 | 260,000 | 40.00 | 114.7 | 8.0 | 31,591 | 1,000 |
| k256 | 400,000 | 40.00 | 118.5 | 12.2 | 28,031 | 1,000 |
| k256 | 4,200,000 | 640.00 | 751.6 | 128.4 | 722,547 | 1,000 |
| k256 | 12,000,000 | 1280.00 | 1482.4 | 366.8 | 2,022,623 | 1,000 |
| key32 | 24,000 | 1.00 | 33.9 | 0.1 | 1,155 | 1,000 |
| key32 | 260,000 | 16.00 | 43.0 | 1.0 | 9,885 | 1,000 |
| key32 | 786,432 | 64.00 | 71.9 | 3.0 | 39,683 | 1,000 |
| key32 | 16,000,000 | 512.00 | 514.7 | 61.2 | 637,710 | 1,000 |
| key32 | 24,000,000 | 1024.00 | 1088.6 | 91.8 | 1,316,327 | 1,000 |
| key64 | 24,000 | 1.00 | 33.9 | 0.2 | 1,061 | 1,000 |
| key64 | 24,576 | 1.00 | 33.9 | 0.2 | 1,112 | 1,000 |
| key64 | 260,000 | 16.00 | 44.1 | 2.0 | 9,677 | 1,000 |
| key64 | 786,432 | 64.00 | 74.7 | 6.0 | 43,755 | 1,000 |
| key64 | 16,000,000 | 512.00 | 537.2 | 122.1 | 691,223 | 1,000 |
| key64 | 24,000,000 | 1024.00 | 1088.6 | 183.2 | 1,261,276 | 1,000 |
| key64 | 96,000,000 | 4096.00 | 4196.8 | 732.6 | 10,036,547 | 1,000 |
| lcstr | 32,000 | 5.46 | 66.8 | 1.1 | 2,163 | 1,000 |
| lcstr | 100,000 | 14.03 | 83.1 | 6.0 | 4,739 | 1,000 |
| lcstr | 500,000 | 33.25 | 135.7 | 25.0 | 21,194 | 5,000 |
| lcstr | 1,000,000 | 56.43 | 197.4 | 47.9 | 40,803 | 10,000 |
| lcstr | 2,400,000 | 124.62 | 345.9 | 115.6 | 105,313 | 24,000 |
| lcstr | 4,800,000 | 239.18 | 326.9 | 229.2 | 216,172 | 48,000 |
| lcstr | 21,000,000 | 1019.31 | 592.2 | 1003.3 | 1,049,752 | 210,000 |
| lcstr | 85,000,000 | 4106.24 | 2087.6 | 4062.2 | 4,398,983 | 850,000 |
| mixed | 24,000 | 1.50 | 54.2 | 0.9 | 3,653 | 1,000 |
| mixed | 260,000 | 24.00 | 86.3 | 8.9 | 39,590 | 1,000 |
| mixed | 400,000 | 24.00 | 104.4 | 13.7 | 64,425 | 1,000 |
| mixed | 12,000,000 | 768.00 | 933.3 | 412.1 | 3,268,072 | 1,000 |
| mixed | 16,000,000 | 768.00 | 954.9 | 549.6 | 4,657,395 | 1,000 |
| null64 | 27,000 | 1.00 | 33.9 | 0.2 | 1,331 | 900 |
| null64 | 290,000 | 16.00 | 54.2 | 2.5 | 8,357 | 900 |
| null64 | 900,000 | 64.00 | 104.9 | 7.7 | 28,332 | 900 |
| null64 | 17,800,000 | 512.00 | 764.7 | 152.9 | 770,574 | 900 |
| null64 | 26,700,000 | 1024.00 | 1353.0 | 229.3 | 1,614,319 | 900 |
| str | 24,000 | 2.05 | 66.0 | 0.7 | 1,926 | 1,000 |
| str | 260,000 | 32.25 | 100.4 | 7.9 | 20,083 | 1,000 |
| str | 480,000 | 32.50 | 125.9 | 14.7 | 33,564 | 1,000 |
| str | 2,000,000 | 130.00 | 331.5 | 61.0 | 198,205 | 1,000 |
| str | 4,200,000 | 516.00 | 707.5 | 128.2 | 883,025 | 1,000 |
| str | 12,000,000 | 1044.48 | 1730.4 | 366.3 | 2,794,182 | 1,000 |
| str | 48,000,000 | 4157.44 | 6748.4 | 1465.0 | 8,800,877 | 1,000 |
| strzero | 24,000 | 2.03 | 66.0 | 1.2 | 1,650 | 1,000 |
| strzero | 260,000 | 32.25 | 104.4 | 10.0 | 20,573 | 1,000 |
| strzero | 480,000 | 32.50 | 125.9 | 18.8 | 31,345 | 1,000 |
| strzero | 4,200,000 | 516.00 | 720.8 | 160.6 | 610,778 | 1,000 |
| strzero | 12,000,000 | 1044.48 | 1736.7 | 458.8 | 1,785,821 | 1,000 |

`null64` probe count 900 is correct: multiples of 10 are NULL on the build side and
NULL never matches in an INNER join.

## Model residuals {#residuals}

- Fixed-width families (key32/key64/null64/k128/k256/mixed/fixstr): every measured
  point equals the plateau model **exactly** (0.00% residual; the trace-line format
  quantizes at ~0.5%). The one "surprise" during fitting — 512 MiB instead of 1 GiB
  at key64 D=16M — identified the x4->x2 grower transition; after that, no exceptions.
- str/strzero: cells match plateaus exactly; the smooth extra is ~1.0 B/row at
  <=4.2M rows, 1.79 B/row at 12M, 1.34 B/row at 48M (drifting, sub-2% of totals;
  S5 uses the S4 rate, so S5's extra term carries ~+-0.5% total uncertainty).
- strzero == str byte-for-byte at every common D (zero bytes in keys are free).
- lcstr: linear fit 9.15 MiB + 50.54 B/row; residuals +0.9 MiB at 100k rows
  (bucket-resize granularity of the 100k-distinct cells), <1% at 0.5M..85M.
- Validated S4 points: key64 96M -> 4096.00 MiB (0.0%), str 48M -> 4157.44 MiB
  (+1.5%), lcstr 85M -> 4106.24 MiB (+0.2% vs fit prediction 4096+9 MiB).

## Caveats / shaky families {#caveats}

1. **Plateau quantization**: with dup=1 the achievable sizes are discrete; deviations
   in the ladder table are structural, not noise. Out-of-tolerance cells:
   - S1 for everything except key32/key64/null64 (floors: 1.5/2/2.5/14 MiB).
   - S2 for key32/key64/null64: only 16 MiB (-50%) or 64 MiB (+100%) exist; the
     ladder uses 16 MiB. If +100% is preferable for the matrix, use D=786,432
     (measured 64.00 MiB).
2. **lcstr is the shaky family**: its byte counter re-counts per-block
   `LowCardinality` dictionary copies referenced from stored blocks (shared with the
   source table); tracked query peak is ~half the reported map bytes (592 MiB vs
   1019 MiB at S3, 2088 vs 4106 at S4). The ladder calibrates the join-reported
   counter. If the matrix measures process RSS deltas instead, lcstr sizes are
   roughly 2x optimistic. Also, `map_rows` in its log lines counts distinct cells
   (100k), not rows; duplicate insertion was verified via probe-match counts.
3. **S5 is extrapolated** (flagged in json): one x2 grower step beyond the largest
   observed bucket (2^20 cells/bucket); the x2 regime was observed at 2^17..2^20.
   No S5 run was performed (by design, to keep wall time bounded).
4. **Build-phase transient floor**: all 32 per-slot skeletons are real during build
   (~32-78 MiB depending on cell size) and freed at merge; a build can never peak
   below ~34 MiB at max_threads=32 whatever D is.
5. Peak memory >> map bytes for insert-heavy families (str S4 peaked at 6.6 GiB for
   a 4.06 GiB map); size memory limits off peak, not map bytes.
6. Calibration assumes count()-style probes (no right columns stored). Selecting
   right-table columns adds stored-block bytes on top of the map.
7. The ladder is specific to this binary's two-level+merge design. Branch HEAD
   (after 69bf5c26c9f / 0d06aaf2933) uses 32 independent per-slot single-level maps;
   plateau positions and floors differ — recalibrate for HEAD-built binaries.
8. Analytic priors from HEAD sources that measurement **corrected**: upstream-style
   24 B key64 cells (actually 16 B tagged-word mapped), arena-copied string keys
   (actually zero-copy), FixedString(16) using the string map (actually keys128),
   x4-only grower (actually x4 then x2), and 32 live per-slot maps at probe time
   (actually one merged map).
