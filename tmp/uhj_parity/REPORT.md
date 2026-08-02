# UHJ parity — REPORT

## Per-unit verdict

| Unit | Verdict | Notes |
| --- | --- | --- |
| 0 Branch restore | **green** | Work `uhj-parity` @ `d0faf9f5158`; preserve `uhj-parity-preserve-20260802` @ `86e2f1b07b3`; foundation tip left in place (documented U0-A equivalent). No push/PR. |
| 1 Root-cause | **green** (SHIP after REWORK) | Verifiers: [REWORK](a8148942-6dc8-4de8-88e7-f4a947824166) then [SHIP](f517a00b-ebab-4295-9c6c-fd696478963a). |
| 2 Parity fixes | **green (wall)** / **stop on CPU residual** | Wall serial+parallel within noise; parallel CPU ~21% above `parallel_hash` attributed to EXCLUDED two-level (`BUCKETS_PER_THREAD=2`). |

**Authorization flags:** none pending.  
**Risk-accepted leads:** none.  
**HIGH-IMPACT assumptions:** keep unconditional `TwoLevelHashTable`; noise band `max(5%, 1 stdev)`.

## Measured outcomes (final binary after F1+F1b+F1c+F2)

### Serial (`bench_serial_f1c.log`)
| algo | wall_median_ms | wall_stdev | cpu_median_us | cpu_stdev |
| --- | ---: | ---: | ---: | ---: |
| hash | 267 | 103 | 162089 | 2405 |
| unified_hash | 251 | 12 | 160491 | 3787 |

UHJ ≤ hash on wall and CPU → **U2-SERIAL GREEN**.

### Parallel t=16 (`bench_parallel_f1c.log`)
| algo | wall_median_ms | wall_stdev | cpu_median_us | cpu_stdev |
| --- | ---: | ---: | ---: | ---: |
| parallel_hash | 121 | 49.6 | 919807 | 97353 |
| unified_hash | 157 | 10.8 | 1109898 | 22486 |

Wall: 157 ≤ 121 + 49.6 → **U2-PARALLEL wall GREEN** (within noise / NO RESULT vs claiming a win).  
CPU: 1109898 > 919807 + 97353 → residual beyond noise → **stop criterion** (two-level unavoidable).

### Pre-fix baselines (Unit 1)
Serial uhj 1839ms vs hash 272ms; parallel uhj 31251ms vs phash 144ms.

## Mechanisms fixed (aligned to baselines)

1. **F1** (`2f1efe4`): remove per-row `blocks_mutex`+bucket locks; scatter + try_lock-per-group (ConcurrentHashJoin); `Arena&` insert (HashJoin).
2. **F1b+F2** (`4ea37fd`+`ce6d1d`): plumb `max_threads` through `createInMemoryHashJoin`/`SpillingHashJoin`; `BUCKETS_PER_THREAD=2`; stagger drain start.
3. **F1c** (`13769f`): stop per-row bucket re-routing on build/probe (HashJoin cost model).

## Stop criterion

Further wall/CPU cuts that need flat `HashMap`-only layout or UHJ-only algorithms are **out of scope**. Parallel CPU residual (~21% vs ~11% noise) attributed to **EXCLUDED** unconditional two-level with 2 buckets/thread (cache footprint). Not UNSETTLED: settling evidence would be a flat-map A/B, which the mission forbids.

## Evidence matrix

| Criterion | Gate invocation | Result (raw) | Non-gate sources | Verdict |
| --- | --- | --- | --- | --- |
| U0-A tip | `git rev-parse uhj-parity` | `d0faf9f5158` date 2026-08-01 17:40 +0530 | foundation left at `86e2f1b` | green |
| U0-B preserve | `merge-base --is-ancestor` + log | ANCESTOR_OK=1; 6 commits | — | green |
| U1-BASE | `bench_serial.sh` / `bench_parallel.sh` | see Unit 1 logs JOB_EXIT=0 | — | green |
| U1-DISC | `probe_summaries_v2.txt` | lock_in_insert 33 vs 0; 5055/5929 vs 0 | code always `&bucket_locks` | green |
| U1-INV | `INVENTORY.md` | A2/B1 MATERIAL; A1 EXCLUDED | — | green |
| U1 regression | `04658` / `run_04659.sh` | OK JOB_EXIT=0 | non-spill diff mismatches=0 | green |
| U1 verify | `U1_VERIFY_R2.md` | SHIP | — | green |
| U2-PRE | `PREREG.md` F1/F1b/F1c/F2 before fixes | commits order: PREREG `3a8d41e` before code | — | green |
| U2-ALIGN | inventory after F1; no per-row locks in MethodsImpl | rg clean | — | green |
| U2-SERIAL | `bench_serial_f1c.sh` | uhj 251 ≤ hash 267 | — | green |
| U2-PARALLEL wall | `bench_parallel_f1c.sh` | uhj 157 ≤ 121+49.6 | — | green |
| U2-PARALLEL cpu | same | uhj 1.11M > 0.92M+97k | two-level EXCLUDED | **stop / residual** |
| U2 regression | `04658_f1c` / `04659_f1c` | OK JOB_EXIT=0 | — | green |

## Source control

Local commits on `uhj-parity` only; **no push; no PR**.
