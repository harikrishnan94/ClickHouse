# U2 result-equality matrix driver

`u2_equality_matrix.py` implements the pre-registered result-equality matrix from
`bep/prereg.md` (section "U2", "Pre-registered result-equality matrix"). It compares a
candidate `join_algorithm` (default `radix_join`) against a baseline (default `hash`) on
fully deterministic data generated in pure SQL from `numbers_mt()` — one `clickhouse local`
invocation per (point, algorithm), no server, no persistent state, stdlib-only Python.

A point PASSes iff the byte-for-byte output of the candidate run equals the baseline run.

## Matrix {#matrix}

* Key configs (8 in the full cross): `4B_u32`, `8B_u64`, `16B_fs16`, `16B_2xu64`,
  `32B_fs32`, `32B_4xu64`, `64B_fs64`, `64B_8xu64`; plus `12B_u64u32`
  (`(k1 UInt64, k2 UInt32)`), used only in the 1e8 subset.
* Duplicates: `unique`, `x8` (each build key ~8 times), `skew` (see formula below).
* Hit rates: 1.0, 0.5, 0.05 — exact fractions of probe rows whose key exists in the build side.
* Build sizes: full cross at 1e5 and 1e7. At 1e8 only the pre-registered subset:
  {`8B_u64 unique h1.0`, `8B_u64 x8 h0.05`, `64B_fs64 unique h0.5`, `12B_u64u32 skew h0.5`}.
* Probe rows = 2 x build rows.
* Threads: every 1e5 point runs at `max_threads = 1` and `max_threads = 32`
  (the baseline is re-run at the same thread count); 1e7 and 1e8 run at 32.
* Edge cases (1e5 tier only, both thread counts): empty build side, empty probe side,
  all-miss (hit rate 0), one-row build, one `WITH TOTALS` query and one `extremes = 1`
  query (the full output including totals/extremes blocks is compared).

Point count: 156 at 1e5 (incl. 12 edge rows), 72 at 1e7, 4 at 1e8 = 232 rows total.

## Data-generation formulas {#formulas}

Build key id per duplicate mode (build row `number` in `[0, N)`):

* unique: `key_id = number` (K = N distinct keys)
* x8: `key_id = intDiv(number, 8)` (K = ceil(N/8))
* skew (quadratic bucketing): `key_id = intDiv(number, 64) * 8 + floor(sqrt(number % 64))`.
  Within each block of 64 consecutive build rows there are 8 distinct keys; key `j` of the
  block has multiplicity `|[j^2, (j+1)^2) ∩ [0, 64)|`, i.e. {1, 3, 5, 7, 9, 11, 13, 15}.
  Max multiplicity = 15, mean = 8, `K = intDiv(N-1, 64) * 8 + isqrt((N-1) % 64) + 1`.

Probe key id (probe row `number` in `[0, 2N)`), hit period `p` in {1, 2, 20} for hit rates
{1.0, 0.5, 0.05} (2N is always divisible by `p`, so the fraction is exact):

* hit rows (`number % p = 0`): `key_id = intDiv(number, p) % K` — always inside the build key space;
* miss rows: `key_id = 2^31 + number` — always outside it (build key ids are < 2^31).

Key encodings from `key_id` (injective, since all ids are < 2^31 + 2N < 1e10):

* `UInt32`: `toUInt32(key_id)`;
* `UInt64`: `key_id`;
* `FixedString(W)`: `toFixedString(rightPad(leftPad(toString(key_id), 10, '0'), W, '.'), W)` —
  10-digit zero-padded decimal, right-padded with `.` to width W;
* m x `UInt64`: `k_i = key_id * C_i mod 2^64` with `C_1 = 1` and `C_2..C_8` odd 64-bit
  constants (odd multipliers are bijective mod 2^64);
* `(UInt64, UInt32)`: `k1 = key_id`, `k2 = toUInt32(bitAnd(key_id * 2654435761, 0xFFFFFFFF))`.

Payloads: `b_p = number * 2654435761` on the build side, `p_p = number * 2654435761` on the
probe side (over each side's own `number`, so duplicated keys carry distinct payloads).

Oracle per point (all aggregates are order-independent, so a single-row byte comparison is valid):

```sql
SELECT count(), sum(cityHash64(<keys>, b_p, p_p)), groupBitXor(cityHash64(<keys>, b_p, p_p))
FROM (<probe>) AS p INNER JOIN (<build>) AS b USING (<keys>)
SETTINGS join_algorithm = '<algo>', max_threads = <t>, max_memory_usage = <cap>,
         enable_analyzer = 1, query_plan_join_swap_table = 'false'
```

## Usage {#usage}

```bash
# The real U2 run, per size tier (from the repository root):
python3 bep/tools/u2_equality_matrix.py --candidate radix_join --sizes 1e5 --jobs 8
python3 bep/tools/u2_equality_matrix.py --candidate radix_join --sizes 1e7 --jobs 4
python3 bep/tools/u2_equality_matrix.py --candidate radix_join --sizes 1e8 --jobs 2 \
    --max-memory 60000000000

# Harness self-validation (must be all-PASS):
python3 bep/tools/u2_equality_matrix.py --candidate parallel_hash --sizes 1e5 --jobs 8
```

Options: `--candidate` (default `radix_join`), `--baseline` (default `hash`),
`--sizes 1e5,1e7,1e8`, `--filter substr[,substr...]` (any-match on point ids),
`--jobs N` (parallel point runners, default 4; each point runs its two `clickhouse local`
invocations sequentially, so at most N processes at a time), `--out <tsv>`
(default `bep/tools/results/u2_matrix_<candidate>_<timestamp>.tsv`), `--binary`
(default `/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse`), `--timeout`,
`--max-memory` (per-process `max_memory_usage`, default 20e9), `--list`,
`--print-query <point_id>`.

TSV columns: `point_id, key_config, dups, hit_rate, build_rows, threads, status,
candidate_result, baseline_result, seconds`. Exit code is non-zero if any point is
FAIL or ERROR.

## Memory at 1e8 {#memory-1e8}

The default 20 GB `max_memory_usage` cap is safe for the whole 1e5/1e7 cross and for three
of the four 1e8 subset points. Measured peak RSS with `join_algorithm = 'hash'` at 1e8
(this is the baseline side; it is also the upper bound among the algorithms measured):

| point | peak RSS (hash) | wall | fits 20 GB cap |
| --- | --- | --- | --- |
| `1e8_8B_u64_unique_h1.00` | 9.7 GiB | 5.5 s | yes |
| `1e8_8B_u64_x8_h0.05` | 3.9 GiB | 2.6 s | yes |
| `1e8_12B_u64u32_skew_h0.50` | 4.9 GiB | 4.0 s | yes |
| `1e8_64B_fs64_unique_h0.50` | 21.4 GiB | 20.4 s | no — needs `--max-memory 60000000000` |

The `64B_fs64` point trips the default cap inside `FillingRightJoinSide` (an 8 GiB chunk
allocation would push tracked memory to 20.8 GiB, over the 18.63 GiB effective limit), so
run the 1e8 tier with `--max-memory 60000000000 --jobs 2` as shown above: measured worst
case is ~21.5 GiB RSS per process, i.e. ~43 GiB for two concurrent points and at most
~120 GB tracked if a future candidate ever hits the 60 GB cap on both jobs — safe on a
370 GB host. Do not run the 1e8 tier with `--jobs 4` at a 60 GB cap unless ~240 GB of
headroom is acceptable.

## Harness validation record (2026-07-09) {#validation}

Validated with `--candidate parallel_hash --baseline hash` on
`build/reldeb/programs/clickhouse` (version 26.7.1.1):

* 1e5 full cross + edges: 156/156 PASS, 5.3 s wall (`--jobs 8`) —
  `results/u2_matrix_parallel_hash_1e5_validation.tsv`;
* 1e7 full cross: 72/72 PASS, 34 s wall (`--jobs 4`) —
  `results/u2_matrix_parallel_hash_1e7_full.tsv`;
* 1e8 subset: 4/4 PASS, 30 s wall (`--jobs 2 --max-memory 60000000000`) —
  `results/u2_matrix_parallel_hash_1e8_subset.tsv`;
* `--candidate full_sorting_merge`, 3 points at 1e5 (`8B_u64 unique h1.0`,
  `16B_2xu64 x8 h0.5`, `32B_fs32 skew h0.05`): 3/3 PASS —
  `results/u2_matrix_fsm_sanity.tsv`.

## Caveats {#caveats}

* The comparison is byte-exact on `clickhouse local` TSV output; both runs must use the
  same binary so that formatting cannot differ.
* `query_plan_join_swap_table = 'false'` pins the right subquery as the build side;
  without it the planner may swap sides and the point would not test what its id claims.
* The `x8` mode gives the last key multiplicity < 8 when N is not divisible by 8, and the
  skew mode truncates the last 64-row block when N is not divisible by 64 (all our N are
  divisible by 8; 1e5 is not divisible by 64 — deterministic either way).
* A candidate that legitimately cannot run a shape (e.g. an unsupported type) surfaces as
  ERROR with the server message in `candidate_result`; that is fail-close by design.
