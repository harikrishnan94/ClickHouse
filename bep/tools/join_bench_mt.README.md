# join_bench_mt - MergeTree join benchmark (`partitioned_hash` vs `parallel_hash`)

Successor to `join_memory_bench.py` on persistent `MergeTree` tables. Ships as five files
that must sit together on every instance (stdlib-only Python):

| file | role |
|---|---|
| `join_bench_mt.py` | the harness (all subcommands) |
| `join_memory_bench.py` | imported as a library (key machinery, fill SQL, settings, tie band) |
| `join_bench_mt_queries.json` | 188 real join specs extracted from TPC-H/TPC-DS/JOB/CoffeeShop/StackOverflow |
| `join_bench_mt_schemas.json` | 62 table schemas + `ORDER BY` + tier-a row counts |
| `join_bench_mt_jobcsv.py` | vendored JOB CSV re-encoder (ClickBench `job_convert.py`) |

`join_bench_mt_manifest.py` regenerates the two JSON files from a ClickBench checkout;
only needed when the upstream queries change.

## What it measures

Two workload families, both algorithms per unit, hot protocol only:

- **real**: one pairwise join per extracted benchmark edge, with the projections each side
  carries in the source queries, in the source join kind (INNER / LEFT / LEFT SEMI /
  LEFT ANTI), plus filtered-dimension variants (36) and the CoffeeShop SCD-2 band
  (applied as a post-join WHERE). Tiers: `a` = TPCH SF40, TPCDS SF32, CoffeeShop 500m,
  StackOverflow 1x, JOB; `b` = SF100 / SF64 / 1b / SO x2 (id-offset second copy) / JOB.
- **synthetic**: controlled-axis cells, 432 total: LEGACY (the full 347-cell plan of the
  Memory-engine sweeps, regenerated deterministically from `join_memory_bench.py plan` and
  shipped as `join_bench_mt_legacy_cells.json`; original cell_ids preserved so results join
  across harnesses, old group letters in the note) + the new gap groups: K (tiny dims x
  huge m_p, up to D=16 / m_p=2M), L (filtered dims h<1 at small D), M (m:n m_b x m_p cross),
  N (K10 = UInt32 keys), O (KF2 = FixedString(2) + 8B-string dims), S (K8 Nullable keys,
  10% NULLs), P (join kinds). Legacy cells with m_p <= 256 keep byte-parity insertion order
  with the Memory sweeps (MAX_PROBE_PASSES = 256).

Per timed run: wall (`query_duration_ms`), peak memory (`memory_usage`), and the **complete
ProfileEvents map** (no allow-list). Protocol guarantees: `OPTIMIZE FINAL`-ed tables,
`SYSTEM STOP MERGES` + part-count-stability assertion around timing, warmups (4 synthetic /
2 real), JIT contamination recorded, cross-algorithm (row_count, checksum) agreement gate,
closed-form row counts additionally asserted for synthetic cells (incl. LEFT/SEMI/ANTI),
FALLBACK status whenever the intended algorithm's path event stayed zero (never silently
measured). `report` scores wall AND peak memory with the same noise band.

## Single-host quickstart

```bash
CH=build/reldeb/programs/clickhouse
# server must have query_log enabled (the embedded default config does NOT):
# see /mnt/data/jbmt_smoke/config.xml for a minimal example. Then:
python3 bep/tools/join_bench_mt.py selftest     --binary $CH --port 9005   # 5 checks, 3 must-fail
python3 bep/tools/join_bench_mt.py prepare-keys --binary $CH --port 9005   # keys_store (synthetic)
python3 bep/tools/join_bench_mt.py load-real    --binary $CH --port 9005 --tier a \
    --datasets tpch,tpcds,coffeeshop,stackoverflow,job --workdir /mnt/data/jbmt_work
python3 bep/tools/join_bench_mt.py sweep  --binary $CH --port 9005 --suite all --tier a \
    --shards 1 --shard 0 --results results.shard0.jsonl
python3 bep/tools/join_bench_mt.py report --results 'results.*.jsonl' --suite all --tier a --out REPORT.md
```

`plan` prints the full unit list with shard assignment (LPT by cost). `run-unit <unit_id>`
runs one unit. `sweep --only <regex>` restricts to matching unit ids; sweeps are resumable
(units already OK/FALLBACK in the results file are skipped). `verify --reference loads.json`
recomputes table fingerprints and compares against the loader's (the fleet readiness gate;
`--emit` writes them).

## Data loading notes

- TPC-H/TPC-DS: generated with DuckDB extension generators (`fetch-duckdb` downloads the
  static CLI v1.3.2 per arch) into parquet, then piped into the server. SF100 needs ~30 GB
  transient parquet in `--workdir`.
- CoffeeShop: `icebergS3` from the public bucket (500m or 1b fact).
- StackOverflow: public parquet; tier `b` inserts a second copy with all id columns
  offset by 1e9.
- JOB: canonical `imdb.tgz` (CSV), re-encoded by the vendored converter.
- Every table: `OPTIMIZE FINAL`, then fingerprint (row count + `cityHash64` over the
  `ORDER BY` key columns) stored in `jbmt_meta.fingerprints`.
- Disk: tier a ~210 GB, tier b ~420 GB, OPTIMIZE peak + largest table (see tmp/disk_sizing.md).
- `--limit-rows N` caps S3-sourced loads for smoke tests (`--tier smoke` uses SF1).

## Fleet playbook (tri-arch, snapshot-cloned shards)

Per architecture (m8g / r7a / m7i .24xlarge, ap-south-2):
1. Launch ONE loader instance with a data volume; run `prepare-keys` + `load-real` for the
   tier(s); `verify --emit loads.<arch>.json`; stop the server cleanly.
2. `fleet-snapshot --volume-id vol-... --tag <run>` -> wait `snapshot-completed`.
3. `fleet-volumes --snapshot-id snap-... --attach az:i-...,az:i-...` for N shard instances;
   attach + mount each clone; start servers.
4. On each shard: `verify --reference loads.<arch>.json` (readiness gate), then
   `sweep --shards N --shard i --results results.<arch>.shard<i>.jsonl`.
5. Collect results files; `report` per fleet; terminate instances, delete volumes+snapshot.

The binary under test is shipped by the operator (same BuildID per ISA across fleets, as in
the tri-fleet Memory sweep; record provenance the same way).

## Deliberately deferred (v1)

- Zipf multiplicity distributions (JOB/StackOverflow realism) - uniform only.
- `nulls_pct` other than {0, 10} (generator limitation inherited from jmb).
- CoffeeShop band as a true join condition (currently post-join WHERE, INNER-equivalent).
- Cold-cache passes; trace_log memory timelines (peak-only per user decision).
- Huge-m_p fills for Nullable keys (batched fill requires nulls_pct=0; plan respects this).

## Validation status (2026-07-24, dev box, 26.7.1.1)

Selftest ALL PASS (incl. must-fail proofs: wrong closed form, mid-timing mutation,
fingerprint mismatch). Smoke-verified end-to-end: CoffeeShop tier-smoke load + all 3 specs
(incl. SCD-2 band), sweep resume, report; synthetic KF2 D64 m_p=500K (32M rows, 9s),
K10 D65536 m_p=1600 (104,857,600 rows = closed form), K8 10%-NULL cell (28.8M = closed
form). Branch finding: with the harness settings, `partitioned_hash` executes LEFT /
LEFT SEMI / LEFT ANTI natively (no fallback); FULL/RIGHT fall back to `hash`.
