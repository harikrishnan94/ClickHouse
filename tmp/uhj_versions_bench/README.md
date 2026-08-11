# uhj-parity vs merge-base on ClickBench `versions/`

Measures what the `uhj-parity` branch does to query performance on **TPC-H**,
**TPC-DS**, **JOB**, and **Coffeeshop** from
[ClickHouse/ClickBench `versions/`](https://github.com/ClickHouse/ClickBench/tree/main/versions).

## Arms

| Arm | Binary | Settings |
|-----|--------|----------|
| **baseline** | merge-base of `uhj-parity` | defaults |
| **uhj** | `uhj-parity` tip | `join_algorithm = unified_hash` only (`uhj` shorthand) |

Same datasets, same host, same cgroup wrapper; arm order interleaved
(`B1,U1,B2,U2,...`).

## Machine emulation

Published results use `"machine": "c7a.4xlarge"` (16 vCPU, 32 GiB). The wrapper
pins whole sibling groups and sets `memory.max=32GiB`, `memory.swap.max=0`.

## How to run

```bash
# after binaries + prepare-data/*.native.zst are ready under work/
ROUNDS=5 bash tmp/uhj_versions_bench/run_interleaved.sh
bash tmp/uhj_versions_bench/verify_arms.sh
python3 tmp/uhj_versions_bench/analyze.py
```

Raw command output for every verification step lives under
`work/verify/` and `work/logs/` (symlinked to `/mnt/data/uhj_versions_bench`).
