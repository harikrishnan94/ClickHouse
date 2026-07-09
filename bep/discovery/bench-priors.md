# Bench priors verification — tmp/bep/ raw evidence audit

Scope: verify the task's binding prior findings P1–P5 against the raw logs in `tmp/bep/`
(40 files, generated 2026-07-09 15:18–16:21 UTC on the 96-core Graviton host described in
`bep/repro.md`). All paths relative to `/mnt/ch/ClickHouse`.

## 1. Log naming scheme (decoded from tmp/bep/run_*.sh)

| Token | Meaning | Evidence |
|---|---|---|
| `b<N>` | `--build-payload-columns N`. Build ROW width = 8 B key + N×8 B payload: b1 = 16 B, b3 = 32 B, b7 = 64 B rows | `run_width_grid.sh:6-10`; log headers, e.g. `model_b7_p1.log:4` "build row width: 64 B" |
| `p<N>` | `--probe-payload-columns N`; probe row width likewise (p1 = 16 B, p7 = 64 B) | same |
| `nb<K>` | N_b = 2^K build rows. Absent ⇒ family default: join_*/model_* use N_b = 2^26 | `run_fraction_validation.sh:6`, `run_bracket_points.sh:17` |
| `f<X>` | probe fraction f = N_p/N_b. **Inconsistent encoding**: in `join_*` and in `bracket_b1_p1`/`bracket_b7_p1`, `f<D>` means f = 1/D (N_p = N_b/D; `run_fraction_validation.sh:9-11`). In `bracket_b1_p7_f0.5`/`_f2` the tag is the LITERAL fraction (N_p = N_b/2 resp. 2·N_b; `run_bracket_points.sh:18-19`) | scripts as cited |
| `r<R>` | probe/build ratio R: N_p = R × N_b (budget sweeps) | `run_budget_sweeps.sh:15-18` |
| `waves_*` | BEP wave-COUNT sweep (1..256 waves, powers of 2) — older output format | log bodies |
| `budget_*` | BEP budget-%% sweep (PHJ + 5/10/15/20/25% of build accumulated bytes, 512 MiB floor) — newer format | `budget_*.log:22` footer |

## 2. Binary + flags per log family

Binary: `build/reldeb/src/Common/benchmarks/hash_join_bandwidth_model` (RelWithDebInfo,
clang-22, jemalloc 5.3-RC; source `src/Common/benchmarks/hash_join_bandwidth_model.cpp` +
`hash_join_bench.{h,cpp}` + `radix_hash_join_bench.{h,cpp}` + `concurrent_hash_join_bench.{h,cpp}`).
CLI options: `hash_join_bandwidth_model.cpp:1817-1841`.

**Two binary generations** (neither maps to a clean commit; both were working-tree builds):
- Gen 1 (built ~15:13) produced `model_*` (15:18–22), `join_*` (15:23–24), `bracket_*` (15:26),
  `waves_*` (15:55–57). Its BEP sweep is the wave-count format; its probe wave loop is the
  legacy per-wave-dispatch one.
- Gen 2 (binary mtime 16:18, matches working tree at commit `0709d550de9`, 16:28) produced
  `verify_fused.log` (16:19), `waves_fixed_nb27.log` (16:19), `budget_*` (16:20–21). It has the
  fused single-`pool.run` streaming wave loop (`hash_join_bench.h:159-164`,
  `radix_hash_join_bench.cpp:89-93`) and the budget-% sweep format.

Flags (from scripts; waves_*/fixed/verify reconstructed from output headers — no script saved):
- `model_b<B>_p<P>.log`: `--quick --build-payload-columns B --probe-payload-columns P` (`run_width_grid.sh:10`)
- `join_b<B>_p<P>_f<D>.log`: `--build-payload-columns B --probe-payload-columns P --join-nb $((1<<26)) --join-np $((1<<26>>shift))` (`run_fraction_validation.sh:13-14`)
- `bracket_*.log`: same `--join-nb/--join-np` shape at N_b ∈ {2^23, 2^26, 2^27} (`run_bracket_points.sh:17-23`)
- `budget_*.log`: `--bep-nb/--bep-np` per `run_budget_sweeps.sh:15-18`
- `waves_b1_p1_nb27`: `--bep-nb $((1<<27))` (np defaults to nb); `waves_b1_p1_ratio8`: `--bep-nb $((1<<25)) --bep-np $((1<<28))`; `waves_b7_p1`: `--build-payload-columns 7 --bep-nb $((1<<26)) --bep-np $((1<<27))`; `waves_b1_p7`: `--probe-payload-columns 7`, same sizes
- `waves_fixed_nb27`: `--bep-nb $((1<<27)) --bep-budget 8388608` (the 8 MiB "extra" row, line 21)
- `verify_fused`: `--verify --join-nb $((1<<24)) --join-np $((1<<25)) --runs 1` (fingerprint line only appears with `--verify`, `hash_join_bandwidth_model.cpp:1836`)

Machine header in every log: 96 threads, L1d 64 KiB, L2 2 MiB, LLC 36 MiB (Graviton, per `bep/repro.md`).

## 3. Priors table

Ratios below computed from the cited lines; "warm" = runs 1–2 (run 0 pays first-touch faults:
e.g. `join_b1_p1_f1.log:11` NPHJ build 186.19 ms vs 76.64/73.35 ms warm).

| # | Claim (from spec) | Value found in logs | Log file:lines | Verdict |
|---|---|---|---|---|
| P1 | Partitioned build beats CHJ-style build 2–3.5x once table spills LLC | HT = 2 GiB (57x LLC): b1 2.36–2.63x (76.64/32.46, 73.35/29.70, 84.64/32.30, 76.88/29.21, 87.47/36.42, 76.92/30.67); b7 2.87–3.89x. HT = 4 GiB/8 GiB build side: 3.28–4.01x. **HT = 256 MiB (7.1x LLC): only 1.49–1.93x** (12.64/8.34, 11.96/8.05, 11.80/6.10). Single-run refs in budget logs vary 1.39x–4.48x | `join_b1_p1_f{1,4,16}.log:14,17`; `join_b7_p1_f{1,4,16}.log:15,18`; `bracket_b7_p1_nb27_f{16,4}.log`; `bracket_b1_p1_nb23_f{16,1}.log`; `budget_*.log:11-12` | **PARTIAL** — 2.4–4.0x holds for multi-GiB tables; just past LLC (256 MiB) it is 1.5–1.9x, and the top end exceeds 3.5x. "Once spills LLC" overstates onset; ratio grows with table size. Model concurs: crossover at HT = 16 MiB is only 1.11–2.2x total (`model_b1_p1.log:93-99,144`) |
| P2a | Probe parity threshold r* ≈ 4–16K rows/leaf/visit without AMAC | Parity (vs NP = 1.0) at 16 384 rows/part/wave (vs NP 1.01; 8 192 → 0.81) for b1_p1 N_b = 2^25 r8; at 4 096 rows (0.99; 8 192 → 1.22) for b7_p1. Bench uses production `HashJoin` software prefetch but no AMAC (`hash_join_bench.cpp:848-871`) | `waves_b1_p1_ratio8.log:19-20`; `waves_b7_p1.log:17-18` | **SUPPORTED** (caveat: waves_* logs are pre-fused-loop, so parity threshold includes the since-removed ~2 ms/wave dispatch cost; post-fix r* should be lower — untested) |
| P2b | 5% budget clears parity | 5% rows/part/wave and speedup vs NP: 8 192 → 1.36x (nb27_r8); 16 384 → 1.45x (nb26_r16); 8 192 → 4.21x (b7_p1). NB: 5% of accumulated bytes is below the 512 MiB floor in all four sweeps, so "5%" is really the floor | `budget_b1_p1_nb27_r8.log:16`; `budget_b1_p1_nb26_r16.log:16`; `budget_b7_p1_nb26_r8.log:16` | **SUPPORTED** for the winning shapes (fails only on the P4 shape, 0.55x, `budget_b1_p7_nb26_r4.log:16`) |
| P2c | Gains ~65–95% of unbounded PHJ by 25% budget | speedup@25% / speedup@PHJ: 1.62/2.42 = 67%; 1.66/2.41 = 69%; 6.04/6.42 = 94%; (losing shape: 0.65/0.84 = 77%) | `budget_b1_p1_nb27_r8.log:15,20`; `budget_b1_p1_nb26_r16.log:15,20`; `budget_b7_p1_nb26_r8.log:15,20` | **SUPPORTED** (67–94%) |
| P2d | Diminishing returns past 15–20% | Marginal vs-NP per 5% step, nb27_r8: +0.10, +0.09, +0.04, +0.03 → knee at 15%. b7_p1: +0.72, +0.51, +0.34, +0.26 — diminishing but still material at 25%. nb26_r16: 5–15% identical (512 MiB floor clamps all three to 32 waves) | `budget_b1_p1_nb27_r8.log:16-20`; `budget_b7_p1_nb26_r8.log:16-20`; `budget_b1_p1_nb26_r16.log:16-18` | **PARTIAL** — clear for narrow build; wide build still gains +0.26x per step at 25%; floor confounds the r16 sweep |
| P3a | Ratio-8 narrow: PHJ 2.42x, BEP@5% 1.36x, BEP@25% 1.62x | Exactly 2.42 / 1.36 / 1.62 | `budget_b1_p1_nb27_r8.log:15,16,20` | **SUPPORTED** (exact) |
| P3b | Wide 64 B build: PHJ 6.42x, BEP@5% 4.21x, BEP@25% 6.04x | Exactly 6.42 / 4.21 / 6.04 (b7 = 64 B build ROW = 56 B payload + 8 B key, not "64 B payload") | `budget_b7_p1_nb26_r8.log:15,16,20` | **SUPPORTED** (exact; wording nit on "payload") |
| P3c | parallel_hash probe degrades 0.55 → 1.02 ns/row from ratio 1 → 8 | NPHJ probe+gather 0.547 ns/row (N_b = N_p = 2^27) and 0.545 (same, gen-2 run) → 1.022 ns/row (N_b = 2^27, N_p = 2^30, ratio 8). Also 0.738 at N_b = 2^25 ratio 8 | `waves_b1_p1_nb27.log:11`; `waves_fixed_nb27.log:11`; `budget_b1_p1_nb27_r8.log:11` | **SUPPORTED** |
| P4 | Wide probe over narrow build loses at every budget, 0.5–0.84x | PHJ 0.84; 5% 0.55; 10% 0.54; 15% 0.57; 20% 0.61; 25% 0.65. Wave sweep: 0.49–0.57 at 1–8 waves, down to 0.05 at 256 | `budget_b1_p7_nb26_r4.log:15-20`; `waves_b1_p7.log:15-23` | **SUPPORTED** (min found 0.54 in the budget sweep, 0.49 in waves). Scope caveat: this is probe-only at ratio ≥ 2. The FULL b1_p7 join at f ≤ 1 does NOT lose: tie at f = 1 (111.71 vs 111.68 ms, `join_b1_p7_f1.log:14-15`), wins 1.4–1.6x at f ≤ 1/4 (`join_b1_p7_f4.log`, `bracket_b1_p7_f0.5.log:15,18`); loses 0.75–0.77x at f = 2 (`bracket_b1_p7_f2.log:15,18`) |
| P5a | Small scatter windows collapse bandwidth (SWWC lines under ~fanout×4 KiB/window) | **No "SWWC" lines exist in any tmp/bep log** — SWWC appears only in source comments (`hash_join_bench.h:122-129`, `hash_join_bench.cpp:122-128,203-207`). Numerically the collapse is real: waves_b1_p1_nb27 (P* = 4096, fanout×4 KiB = 16 MiB): scatter 38.27 ms @ 2 GiB window → 108.32 @ 64 MiB → 361.05 @ 16 MiB → 700.40 ms @ 8 MiB. Post-fused, the pure-bandwidth residual at 8 MiB (2 KiB/part) is still 345.42 ms = 15.6x the 1-wave scatter (22.15 ms) | `waves_b1_p1_nb27.log:15-23`; `waves_fixed_nb27.log:15,21` | **PARTIAL** — phenomenon confirmed and severe below ~fanout×4 KiB, but degradation starts earlier (2.8x already at 16 KiB/part), and "SWWC lines" as a log artifact do not exist; half the pre-fix "collapse" was dispatch overhead |
| P5b | Per-window pool dispatch ~2 ms/window at 96 threads | Direct A/B at identical shape (N_b = N_p = 2^27, 256 waves, 8 MiB): legacy 1416.63 ms vs fused 921.45 ms → (1416.63−921.45)/256 = **1.93 ms/wave**. Source comment: "~4 dispatches/wave, measured ~1.9 ms/wave at 96 threads". Legacy per-phase slopes: scatter +2.65, probe +2.72 ms/wave (128→256 waves) | `waves_b1_p1_nb27.log:22-23`; `waves_fixed_nb27.log:21`; `hash_join_bench.h:161-164` | **SUPPORTED** (~1.9 ms/wave) |
| P5c | Fused single-dispatch wave loop keeps overhead flat | At operational budgets (5–25% → 2–4 waves) fused totals 77.33–89.82 ms vs 73.71 ms PHJ (≤ 22% overhead). Correctness: `verify_fused.log:13` "results equal (fingerprint 67975942e241e65a)". BUT the extreme row (256 waves, 8 MiB) is 921.45 ms = 6.865 ns/row vs 0.549 at 1 wave — total cost is NOT flat at tiny windows; the residual is SWWC bandwidth collapse, not dispatch | `waves_fixed_nb27.log:15-21`; `verify_fused.log:10-13`; `radix_hash_join_bench.cpp:89-93` | **PARTIAL** — dispatch overhead removed (P5b), and overhead is near-flat in the 5–25% operating range, but "flat" does not extend below ~4 KiB/part windows |

## 4. Additional load-bearing observations

1. **waves_* logs are stale relative to the shipped kernel.** They were produced by the
   pre-fused (per-wave dispatch) binary; every per-wave number in them overstates small-budget
   cost by ~1.9 ms/wave/phase-pair. The budget_* logs are the authoritative post-fix numbers.
   Only one post-fix extreme-wave point exists (`waves_fixed_nb27.log:21`).
2. **512 MiB budget floor hides the true 5% point** in all four budget sweeps (5% of 3–6 GiB
   accumulated = 154–307 MiB < 512 MiB floor; `budget_*.log:22`). "5% budget" numbers are
   floor numbers.
3. **RPHJ build-time variance in budget logs**: RPHJ build 212.93 ms in `budget_b1_p1_nb27_r8.log:12`
   vs 62.42 ms for the same N_b = 2^27 build in `waves_b1_p1_nb27.log:12` (3.4x spread;
   single measurement, 16 GiB probe side resident). Treat single-run build refs as noisy.
4. **NPHJ teardown is large and unaccounted in the ratios**: ~100–250 ms per join
   (e.g. `join_b7_p1_f1.log` teardown ~122–128 ms vs RPHJ 5.8–8.0 ms); the "total" columns
   in join logs exclude teardown from neither side's total. Porting implication: NPHJ-style
   concurrent slot teardown is a real cost RPHJ avoids.
5. **Model fraction-crossover tables** (`model_b1_p1.log:149-162`, `model_b7_p1`, `model_b1_p7`)
   agree qualitatively with P4/P1: wide build ⇒ RPHJ "always" wins; wide probe ⇒ wins only
   below f ≈ 0.12–0.9 depending on N_b; and NP curves beyond 2 GiB are flat-extrapolated
   (f* upper bounds, `:162`).
6. All BEP comparisons are **probe-only** ("vs NP = NPHJ probe time / BEP probe total",
   `budget_*.log:23`) against the SAME prebuilt partition tables; build cost excluded.
   Unique keys, hit rate 1, INNER promoted to RightAny point lookups
   (`hash_join_bench.cpp:854-860`) — duplicate-key ALL joins are explicitly NOT modeled
   (`model_b1_p1.log:142,146-147`).
