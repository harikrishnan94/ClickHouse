# PREREG — WaveJoin implementation campaign

Pre-registrations are committed BEFORE the runs they gate (verifiable in
git history). A prediction/result mismatch is a finding to investigate,
never to rationalize.

## Baseline binary (frozen)

- Path: `/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse`
- sha256: `4b55481c22d025ae364d36df39cd662bd986fd5878e711d89e1d76b08ea59cce`
- Provenance: clean tree on `wave-join-impl` @ branch point 069c287d9a0;
  `ninja -C build/reldeb clickhouse` rebuilt 1 TU + relinked and
  reproduced the byte-identical sha256
  (log `tmp/wave-join-impl/ninja_noop_provenance.log`, pre/post hashes in
  `tmp/wave-join-impl/binary_sha_before_ninja.txt` and the log's tail).
  This binary is also the prior probe campaign's final gated binary
  (`tmp/wave-join-verify-report_probe.md` cites `4b55481c…`), i.e. v1+E2.

## Protected cells (FROZEN — user ruling 2026-07-16)

Spec §1's descriptor ("A: 2 GiB build; C: 4 GiB build; both plan 8192
leaves") contradicts the shapes its quoted floors were measured on. The
user ruled the probe-campaign shapes bind; the descriptor mismatch is a
spec defect recorded for REPORT.md.

| shape | cardinality D | m | ratio | hit | bp | pp | v1 plan leaves |
|-------|---------------|---|-------|-----|----|----|----------------|
| A     | 268435456     | 1 | 2     | 1   | 1  | 1  | 16384          |
| C     | 268435456     | 1 | 4     | 1   | 7  | 7  | 32768          |

Cells = {A, C} × T ∈ {96, 64, 32, 16, 1} (execution order; 10 cells).
Exact queries: rendered SQL committed under `tmp/wave-join-impl/sql/`
(sha256s in `tmp/wave-join-impl/sql_sha256.txt`); settings string embedded
in each (join_algorithm='radix_join', max_threads=T,
query_plan_join_swap_table=false, enable_analyzer=1,
enable_join_runtime_filters=0, external join disabled,
max_memory_usage=100000000000).

## Measurement protocol (FROZEN)

- Runner: `tmp/wave-join-impl/suite.py` — FROZEN at the Unit-0 prereg
  commit; any later edit is a register amendment requiring user sign-off.
- Session = one `clickhouse local --path=tmp/wave-join-impl/chscratch`
  process, 1 warmup + 1 timed identical query; metric = client `--time`
  wall of the timed query (quantized 1 ms). Radix ProfileEvents are
  diagnostics only — `RadixHashJoinProbeMicroseconds` is wait-inclusive
  and summed across workers (reads up to ~8× wall) and NEVER gates.
- Pairing: position-balanced pairs; order (A,B) on even pair index, (B,A)
  on odd. Unit 0 runs SELF-PAIRED (arm A = arm B = baseline): the
  within-pair log-ratio SE is the null-distribution scale of the exact
  statistic Unit 3 gates on. One "rep" ≔ one self-pair.
- Pairs per cell: 5 at T ∈ {1,16,32}; 9 at T ∈ {64,96} (pre-declared —
  historical single-pair disturbances reach ~40% there). Band computed
  over ALL recorded pairs of a cell.
- Noise band (per cell) ≔ max(1%, 3 × SE(within-pair log-ratios)).
  Pre-declared rules: a frozen band > 5% is surfaced at the approval gate
  as a finding; if Unit 3's observed paired SE materially exceeds the
  frozen band, that is a finding to surface, never a silent band update.
- Cache regime: fully cached (370 GB RAM ≫ working set); no cache
  dropping anywhere; per-shape oracles run first at T96 and double as the
  primer. Suite order frozen and identical in Unit 3.
- Oracles: per timed session — engagement fail-closed
  (`RadixHashJoinLeafGroupBuilds` ≥ 1 via `fallback_reason`) AND exact
  expected leaf count per arm (baseline arms: A 16384, C 32768; a
  candidate arm's expectation is set in the Unit-3 prereg — spec §1's
  build-side clamps may legitimately change it). Per shape at T96 —
  count assertions (probe/build/joined exact) and
  radix-vs-parallel_hash fingerprint (count, sum(cityHash64(payloads)));
  in A/B mode both arms' radix fingerprints must also match.
- Guards: /mnt/data integrity snapshot (`find -printf '%p %s %T@\n'`,
  sorted) vs frozen `integrity_S0.txt`, checked pre / per-shape / post —
  any diff aborts. Foreign-process check (exact-name pgrep) + loadavg <
  1.0 bounded wait at start. Binary sha256 recorded at suite start and
  re-checked at end (footer `binary_stable`).
- Timeouts per session: C_T1 5400 s, A_T1 2400 s, C_T16 1200 s, else 900 s.

Bootstrap evidence (2026-07-16): fresh scratch smoke-proven — /mnt/data
snapshot byte-identical across a `clickhouse local` open
(`integrity_S0_preopen.txt` == `integrity_S0_postopen.txt` ==
`integrity_S0.txt`). Cross-campaign diff vs
`tmp/radix-probe-followup/integrity_S0_FINAL.txt`: only the top-level
directory mtime differs (the rhj-probe-perf campaign opened /mnt/data
directly; known and benign); all 247 file entries identical — no content
drift. Harness smoke (`smoke_harness.jsonl`, A_T96 × 1 pair) is
development evidence only, NOT acceptance evidence.

## Unit-0 gate (pre-registered 2026-07-16, BEFORE the run)

Invocation (copy-paste re-runnable):

    cd /mnt/ch/ClickHouse && python3 tmp/wave-join-impl/suite.py \
      --binary /mnt/ch/ClickHouse/build/reldeb/programs/clickhouse \
      --cells all --reps 5 \
      --out /mnt/ch/ClickHouse/tmp/wave-join-impl/baseline_u0.jsonl

Expected outcome: completes with footer `status=complete`,
`binary_stable=true`; all integrity rows ok; both shapes' assertions and
fingerprints ok; every timed run engaged with the exact expected leaf
count; per-cell medians in the neighborhood of the prior campaign's
(A ≈ 60.7/3.3/1.9/1.4/1.4 s and C ≲ 252/18/12.6/11.6/7.9 s at
T=1/16/32/64/96 — v1+E2 should be ≤ the pre-E2 numbers at C T≥16);
bands mostly at the 1% floor for T ≤ 32.

Refuted by (any of): suite non-completion or timeout; any fallback
(LeafGroupBuilds = 0) or leaf-count mismatch; fingerprint or count
mismatch; integrity violation; `binary_stable=false`; a band > 5%
(escalate as finding); medians wildly off the prior campaign's (> ~2×)
without an identifiable cause.

Results (filled after the run, 2026-07-16; raw data
`baseline_u0.jsonl`, log `baseline_u0.log`): **GATE GREEN.** Suite
completed in 2 h 18 m 44 s (03:34:17–05:53:01 UTC), footer
status=complete, binary_stable=true, binary sha256 constant
(4b55481c22d0…) across all 132 runs. Integrity ok ×4
(pre/post-shape-A/post-shape-C/post). Assertions exact
(A: 536870912/268435456/536870912; C: 1073741824/268435456/1073741824).
Fingerprints radix == parallel_hash both shapes. Engagement exact on
every run (A: 16384 leaves ×66, C: 32768 ×66). Full position-balanced
coverage (9 pairs at T96/T64, 5 at T32/T16/T1; no gaps/dups; 0
alternation violations). No refutation condition fired; medians in the
pre-registered neighborhood (C at T ≥ 16 faster than the pre-E2
figures, as predicted for v1+E2). Largest per-arm median divergence
1.84% (A_T32), under its band. Independent log audit: PASS
(fresh subagent, all 5 checks).

### 10-cell FROZEN baseline (medians + spread + noise bands)

| cell | n pairs | median (s) | min | max | stdev | SE(log) | band |
|------|---------|------------|-----|-----|-------|---------|------|
| A_T96 | 9 | 1.343 | 1.321 | 1.367 | 0.012 | 0.00457 | 1.37% |
| A_T64 | 9 | 1.368 | 1.336 | 1.388 | 0.015 | 0.00610 | 1.83% |
| A_T32 | 5 | 2.179 | 2.137 | 2.246 | 0.031 | 0.00727 | 2.18% |
| A_T16 | 5 | 3.289 | 3.269 | 3.339 | 0.023 | 0.00356 | 1.07% |
| A_T1 | 5 | 61.678 | 61.156 | 62.337 | 0.334 | 0.00360 | 1.08% |
| C_T96 | 9 | 7.835 | 7.754 | 7.943 | 0.056 | 0.00388 | 1.17% |
| C_T64 | 9 | 7.939 | 7.794 | 8.076 | 0.071 | 0.00230 | 1.00% |
| C_T32 | 5 | 9.268 | 9.230 | 9.378 | 0.043 | 0.00252 | 1.00% |
| C_T16 | 5 | 14.195 | 13.890 | 14.551 | 0.196 | 0.01167 | 3.50% |
| C_T1 | 5 | 254.125 | 249.205 | 256.981 | 2.301 | 0.00529 | 1.59% |

All bands ≤ 3.50% — none reaches the pre-declared 5% escalation
threshold. These bands are FROZEN: Unit 3's goal gate requires the
candidate median to beat each cell's baseline median by MORE than its
band; the floor gate forbids exceeding it by more than the band.
