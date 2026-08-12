# uhj-parity vs merge-base — ClickBench versions (TPC-H / TPC-DS / JOB / Coffeeshop)

> **Superseded — read `FOLLOWUP.md` and `JOB_REGRESSION.md` first.** Every headline number
> below is confounded. Both arms re-plan differently on warm runs because the
> hash-table-statistics cache is only wired up when `parallel_hash` is in `join_algorithm`,
> so the baseline arm changes its join order after run 1 and the uhj arm never does. With
> `collect_hash_table_stats_during_joins=0` on both arms:
>
> - tpch `−18%` — does not survive (driven by q8, a baseline plan flip to a 1.2B-row build).
> - tpcds q54 `+7330%` — not a uhj regression (baseline plan flip to a 166k-row build).
> - job `+11%` — **also does not survive**; at equal plans uhj is 2.9% *faster* on JOB.
>   A genuine but much smaller regression remains on 7 of 113 JOB queries
>   (+11% to +26%, all with 23M+ row build sides), offset by 20 genuine improvements.

## Verdict

On an emulated `c7a.4xlarge` (16 vCPU / 32 GiB cgroup) with the ClickBench
`versions/` runner contract (`TRIES=6`, `drop_caches`), forcing
`join_algorithm=unified_hash` on the `uhj-parity` binary vs the merge-base
default (`direct,parallel_hash,hash`):

| Dataset | Baseline geomean (min-hot) | UHJ geomean | Δ | Outside noise |
|---------|---------------------------|-------------|---|---------------|
| coffeeshop | 4.73s | 4.86s | **+2.8%** | 1 regression (q5 +5.1%) |
| tpch | 0.57s | 0.47s | **−18.1%** | 3 reg / 1 imp (driven by q8) |
| tpcds | 0.27s | 0.28s | **+3.9%** | 17 reg / 5 imp |
| job | 0.076s | 0.085s | **+11.1%** | 41 reg / 7 imp |

Noise rule (declared before comparison): NO RESULT if
`|rel_delta| ≤ max(5%, 1×stdev)` of the 5 baseline hot tries (`TRIES=6`).
One interleaved suite pair (B1, U1). Suite-level ×5 was not completed (≈3.5h
per arm on this host).

## Verification (raw commands/output in `artifacts/`)

1. **Emulation binds** — `artifacts/verify_emulation_final.log`:
   `nproc=16`, `memory.max=34359738368`, `swap.max=0`, OOM kill rc=137,
   `max_threads=16`, MemoryWorker “32.00 GiB”.
2. **uhj in use** — settings dump `join_algorithm=unified_hash`; every
   `EXPLAIN` join on the uhj arm is `SpillingHashJoin(UnifiedHashJoin)`
   (`artifacts/explain/`, `artifacts/smoke_explain_uhj.txt`). No fallback
   observed.
3. **Arms comparable** — settings diff is only `join_algorithm`
   (`artifacts/settings_diff.txt`). Shared data dir; checksums in
   `artifacts/dataset_checksums.txt`.
4. **Noise band** — within-suite 5-hot stdev (see above).
5. **Fidelity** vs published `master.json` (`machine=c7a.4xlarge`):
   - coffeeshop −4.5% ✓ / job +4.0% ✓ / tpcds +10.1% (borderline) /
     tpch **+48%** ✗.
   - Cause: host is ARM Neoverse-V2 (no SMT) emulating AMD EPYC c7a.
     **Absolute comparisons unreliable; A/B deltas remain valid.**
6. **All queries** — nulls named below; not averaged in.

## Null / failed queries (both arms unless noted)

- `tpch/q5` — baseline only: timeout >600s after a 503s first try; uhj completed.
- `tpcds/q5` — `Illegal type Variant(Decimal(7,2), Float64)` for `sum` (both).
- `tpcds/q14`, `tpcds/q15` — `INTERSECT ALL` / missing subquery alias (both).
  These match published nulls on awkward SQL; not join-algorithm specific.

## Effects that drive the geomeans

**Improvements (name the drivers):**
- `tpch/q8`: baseline hot stuck ~353s all five tries → uhj ~0.71s (−99.8%).
  Dominates the tpch geomean win.
- `tpcds/q97`: baseline hot ~39–44s → uhj ~0.47s (−98.8%).

**Regressions (report as the result, not tuned away):**
- `tpcds/q54`: baseline hot ~0.37s → uhj hot ~27–29s (+7330%).
- `tpch/q11` (+130%), `tpch/q21` (+128%).
- JOB: broad regressions; geomean **+11%**, with large hits on q2/q4/q23/q64/q68/…

Full per-query table: `artifacts/report/REPORT.md` / `report.json`.
Raw timings: `artifacts/results/{baseline,uhj}_r1.json`.

## Machine / arms

| | |
|-|-|
| Published target | `c7a.4xlarge` from `versions/results/master.json` |
| Emulation | cpuset 0–15, memory.max=32GiB, swap.max=0, cpu.max=1600000 100000 |
| baseline SHA | `3218492309c` (merge-base) |
| uhj binary | RelWithDebInfo from `uhj-parity` tip at measurement time |
| Treatment | `join_algorithm=unified_hash` only |
| Scale | TPC-H SF40, TPC-DS SF32, Coffeeshop 500m, JOB IMDB — from `prepare-data/*.sh` |
