# VERIFICATION_U5 — independent verification of the U5 fleet acceptance campaign

Verifier: independent agent (doer ≠ grader), 2026-07-28T12:35–12:50Z.
Method: every number below was recomputed from the raw JSONLs in
`fleet_results/` with my own scripts (`verify_scratch/verify_gate.py`,
`verify_scratch/verify_aux.py`, `verify_scratch/verify_spots.py`); no
orchestrator code was executed. Raw outputs: `verify_scratch/gate_recompute.txt`,
`verify_scratch/aux_recompute.txt`, `verify_scratch/spots_analysis.txt`,
spot raws `verify_scratch/spot_{win,red,rand}.jsonl` + logs.

Verdict rule re-implemented per PREREG 007 and the campaign's operative band
definition: per-cell median across valid runs per arm of
`events['ConcurrentHashJoinProbeMicroseconds']` (thread-summed by the server);
WIN if diff ≤ −band, LOSS if diff > band, else TIE; band = max(3%, arm-A
pooled spread fraction (max−min)/median); build cells (`:build.` in id) judged
on wall (`duration_us`) with GUARD-OK iff diff ≤ 0 or |diff| ≤ band; ≥5 valid
runs per arm required. Substitutions per PREREG 008/009: `key64 asof S2/S4`
and `str asof S2` from `fix2_rerun` (shipped ASOF code = fix2); `lcstr S3`
from `fix1_rerun` (fix2 did not change LC).

## Summary of checks

| # | Check | Verdict |
|---|-------|---------|
| 1 | Gate recompute (97 frozen cells) | CONFIRMED — zero verdict differences; QUALIFIED on coverage (1 frozen cell missing from the tally) |
| 2 | Validity audit (identity, rowcounts, checksums, floors, ABAB) | CONFIRMED — 0 violations; one auditability caveat (`host` field) |
| 3 | Ablation + boundary recompute | CONFIRMED — and stronger on the pre-registered probe metric than the wall numbers quoted in WORKLOG |
| 4 | Spot re-runs on the fleet (3 cells, 10 runs each) | CONFIRMED — 3/3 reproduce within band (max delta 0.80 pp) |
| 5 | PREREG ordering audit | CONFIRMED — all entries committed before their gated runs; minor drift notes |
| 6 | Fleet accounting pre-check | CONFIRMED — 8 × m8g.24xlarge, launched 2026-07-28T11:00:44Z |

FINAL: **FIX-THEN-SHIP** — every measured claim verifies exactly, but the
campaign accounting silently drops one frozen gate cell (details in check 1);
the fix is reporting, not re-measurement.

## 1. Gate recompute — CONFIRMED (zero verdict differences), QUALIFIED on coverage

My tally over the 96 cells present in the sweep, with substitutions applied:
**54 WIN / 11 TIE / 7 GUARD-OK / 7 LOSS / 17 floor-invalid** — identical to
the orchestrator's `54 win / 18 tie / 7 red / 17 invalid` (their "18 tie"
counts the 7 build GUARD-OKs as ties). Every per-cell verdict, diff, and band
matches `gate_verdicts_fullsweep.txt` and the WORKLOG's substituted red list.
**No cell's verdict differs from the orchestrator's.**

Recomputed honest-red cells (after substitution):

| cell | diff | band | wall | source |
|------|------|------|------|--------|
| key64:probe.asof.S2.T96 | +3.57% | 3.0% | −1.49% | fix2_rerun |
| key64:probe.asof.S4.T96 | +5.89% | 3.0% | +7.22% | fix2_rerun |
| str:probe.asof.S2.T96 | +4.94% | 3.0% | −1.56% | fix2_rerun |
| lcstr:probe.inner_all.S3.T96 | +18.41% | 6.6% | −5.39% (wall WIN) | fix1_rerun |
| key32:probe.inner_all.S5.T96 | +12.58% | 3.0% | +13.19% | sweep |
| key64:probe.inner_all.S5.T96 | +13.30% | 3.7% | +6.25% | sweep |
| key64:probe.semi_anti.S4.T96.anti | +16.02% | 3.0% | +40.26% | sweep |

All match the orchestrator's claimed values to the reported precision.
Fix-cycle progression also recomputed from raw: key64 asof S4 +26.56% (sweep)
→ +5.62% (fix1) → +5.89% (fix2); str asof S2 +4.18% → +11.91% (fix1 blanket
exclusion overshoot) → +4.94% (fix2) — exactly as PREREG 008/009 narrate.
`str asof S4` correctly keeps its sweep TIE (string ASOF under fix2 rings
again = code-equivalent to the sweep's final binary for that cell).

Full recomputed 96-cell table: `verify_scratch/gate_recompute.txt`.

**QUALIFICATION (the one real finding): `lcstr:probe.inner_all.S5.T96` is a
frozen MATRIX gate cell (97 total) but is absent from the sweep results and
from every tally.** It was planned (`u5_cells_shard7.txt`) and attempted:
warmup 0 failed on the **baseline** arm with `MEMORY_LIMIT_EXCEEDED` (187.67
GiB requested vs 194.96 GiB limit `While executing FillingRightJoinSide`,
`sweep_shard7.log`), zero rows were recorded, and the orchestrator's
"54/18/7, rest floor-invalid" accounting (54+18+7+17 = 96) silently omits it.
It is not a candidate defect — the baseline binary OOMed first, the cell is
infeasible on 192 GB hosts at S5×LC — but the gate contract is "every probe
cell in the frozen list", so the cell must be reported as
INFEASIBLE-ON-FLEET/NOT-MEASURED in REPORT.md, not dropped. WORKLOG and
`report_current.txt` do not mention it.

Secondary qualifications (no verdict impact):
- **Band-source drift**: PREREG 007's letter says band = pre-sweep A/A
  calibration; the operative gate used in-sweep arm-A spread (3% floor). The
  A/A run (`aa_u5.shard{0,1}`, 12 cells, recorded 11:05–11:08Z, before the
  sweep) covers only 12 of 97 cells. For the 10 comparable probe cells I
  computed A/A-derived bands (pooled spread): max A/A median diff is ±1.35%,
  and **no verdict flips** under either band definition. Verdict-neutral
  where checkable; should be recorded as a deviation.
- **lcstr S2 code-version note**: fix 1b changed single-column LC routing, so
  the shipped (fix2) code differs from the sweep's final binary on `lcstr`
  cells, but only `lcstr S3` was re-run. `lcstr S2`'s TIE therefore carries
  the pre-fix binary. Risk is low (35% band; lcstr S3 moved only +0.7 pp
  under the fix) but it is an extrapolated verdict.
- **Orchestrator script warts** (`gate_verdicts.py`): the build-guard band is
  computed from the probe-event spread rather than the wall spread it judges
  (line 52 is also dead code). Recomputing guard cells with wall-spread bands
  leaves all 7 GUARD-OK (my wall bands are wider on every guard cell).

## 2. Validity audit — CONFIRMED

- **Invalid rows**: 340/1920 sweep rows invalid — exactly 17 cells × 20 rows,
  every reason "below-duration-floor (arm median < 200 ms)" (16 cells on arm
  A, 1 on arm B: `str:probe.inner_all.S2.T1`). I recomputed both floors
  myself from raw `duration_us`/`rows_probe`: my floor-invalid set is exactly
  the recorded 17 cells — no cell flagged by me was passed, none failed
  spuriously; the 2M rows/thread floor never binds (min observed = 2.0M/thread
  at T96 × 192M rows). fix1/fix2/ablate/boundary files: 0 invalid rows.
  A/A: 40 invalid rows = 2 build cells below floor, consistent.
- **Binary identity**: every valid row in every file has
  `binary_sha256 == proc_exe_sha256 ==` the MANIFEST value for its arm:
  baseline `0d32ef1c96e6…`, final `6495b05ab061…`, fix1 `6edb195b1416…`,
  fix2 `a7515405c7c9…`; aux arms (ringON/OFF, bdefault/bforce/bflatoff) all
  `6495b05ab061…` (final). **0 mismatches** across 2 520 campaign rows.
  I additionally re-hashed the deployed remote binaries over SSH on shards 4
  and 5 — they match MANIFEST.
- **Rowcounts/checksums**: `rows == expected_rows` in every valid row;
  per-cell checksum identical across arms in every file; and identical
  **across files** for the 79 cells appearing in more than one file
  (sweep/fix1/fix2/aux/A-A/my spots): 0 mismatches.
- **Protocol order**: strict ABAB (or BABA) alternation holds for every cell
  in every file (positions 0..19).
- **Caveat (auditability, not validity)**: the `host` field in every row is
  `ip-172-31-5-72` — the **controller's** hostname (`socket.gethostname()`,
  `fleet_ab.py:1537`), not the shard. The rows cannot by themselves prove
  where they ran. Fleet execution is corroborated by: per-shard `--ssh-host`
  invocations in `run_u5_sweep.sh`; the deployed binaries still on the remote
  hosts with mtimes matching the timeline (base/cand 11:02, fix1 12:16, fix2
  12:30) and matching shas; `/proc/<pid>/exe` hashed over SSH by the harness;
  and my own on-fleet spot re-runs reproducing the numbers (check 4).
  Recommendation: record the remote hostname in the row.
- Minor: `MANIFEST.tsv`'s last three entries (final/fix1/fix2) are appended
  as raw `sha256sum` lines, not TSV rows (values correct).

## 3. Ablation and boundary — CONFIRMED (stronger than quoted)

Recomputed from `ablate_ring.jsonl` (A = default ring ON, B =
`CLICKHOUSE_JOIN_AMAC=0`, final binary both, 10/10 valid, identity clean):

| cell | probe diff (OFF vs ON) | wall diff | WORKLOG quoted |
|------|------------------------|-----------|----------------|
| k256:probe.inner_all.S3.T96 | **+114.65%** | +18.88% | +18.88% |
| str:probe.inner_all.S5.T96 | **+62.47%** | +18.48% | +18.48% |
| key32:probe.inner_all.S5.T96 | **+37.41%** | +6.65% | +6.65% "TIE at 8.4% band" |
| key64:probe.inner_all.S5.T96 | **+36.63%** | +4.88% | +4.88% |

The WORKLOG quoted **wall** numbers (they match my wall recompute exactly;
it should say so). On the pre-registered probe metric the ring-off regression
is far larger, so G-ablation holds a fortiori. The claimed-win rule
(OFF-regression ≥ win − band) passes on both claimed-win cells (k256 S3:
+114.65% ≥ 49.61%; str S5: +62.47% ≥ 31.24%), and the ring is confirmed to
HELP at the red S5 cells (exoneration). One nit: I cannot reproduce the
"8.4% band" for key32 S5 (I get 10.6% wall-spread band); the TIE-on-wall
verdict is unaffected.

Boundary (`boundary_force.jsonl`, `boundary_off.jsonl`, final binary both):
- force vs default: 3/3 TIE confirmed (probe +0.27% / −0.07% / −1.98% at
  3–4.8% bands; wall +0.12/+0.21/−0.76%).
- Force-engagement counters present in the force arm's cell-level
  `engagement` (key64 S1: `AmacBuildRows` 24000 + 128 ring growths, S1p5:
  96000 + 256; `AmacProbeRows` 192M) — PREREG 007's force-engage requirement
  satisfied. The bdefault arm's engagement confirms the WORKLOG's honest
  nuance: the probe ring already auto-engages at S1/S1p5 (ON-vs-ON), so the
  real ON-vs-OFF contrast is boundary_off:
- default vs `=0` at S1p5: key64 probe +38.56% (wall +5.35%), str probe
  +76.55% (wall +15.77%) — WORKLOG's wall numbers confirmed; the
  below-threshold engines carry real weight at the boundary. Naming nit: the
  WORKLOG's "flat-loop-off" label is loose — `CLICKHOUSE_JOIN_AMAC=0`
  disables the ring only; the flat descriptor find is unconditional for
  cursor families (`HashJoinRoutedMethodsImpl.h`), so the key64 contrast is
  ring-vs-flat. The substance stands.

## 4. Spot re-runs — CONFIRMED (3/3 reproduced)

Run by me on the live fleet (10 timed runs + 4 warmups each, strict ABAB,
`--require-engagement`, campaign calibration file; identity and rowcount
checked in-row, checksums match the campaign's; raw JSONLs in
`verify_scratch/`). Note the WIN cell was deliberately re-run on a
**different host** than its campaign shard.

| cell | where | spot result | campaign ref | outcome |
|------|-------|-------------|--------------|---------|
| k256:probe.inner_all.S3.T96 (WIN) | shard 5 (orig: shard 0) | **−52.56%** (band 3.0%), wall −16.73% | −52.61% (sweep) | REPRODUCED, Δ 0.05 pp |
| key64:probe.asof.S2.T96 (honest-red) | shard 4, baseline vs fix2 | **+4.09% LOSS** (band 3.0%), wall −0.24% | +3.57% (fix2_rerun) | REPRODUCED, Δ 0.52 pp |
| key64:probe.any.S2.T96 (random 3rd, seed 20260728 over 37 fast probe cells) | shard 5 | **−24.76% WIN** (band 11.9%), wall −3.45% | −25.56% (sweep) | REPRODUCED, Δ 0.80 pp |

PREREG 007 specified the verifier re-runs 3 cells; the two designated ones
plus one random pick satisfy it. All three verdicts and magnitudes reproduce
within band; binary/proc-exe shas in my rows match MANIFEST (baseline
`0d32ef1c…`, final `6495b05a…`, fix2 `a7515405…`).

## 5. PREREG ordering audit — CONFIRMED (minor notes)

Commit history of `tmp/chj_probe_parity/PREREG.md` vs `recorded_at` in raw rows:

| entry | committed (UTC) | gated runs begin | order |
|-------|-----------------|------------------|-------|
| 000–006 + MATRIX freeze | 06:35:02 (`867a6983f0f`) | — | OK |
| 007 (campaign contract) | 11:00:39 (`190d41213a7`) | A/A 11:05:03; sweep 11:10:50; ablate/boundary 12:08 | OK |
| 008 (fix cycle 1) | 12:15:50 (`06e0bbd0aa3`, with the fix code) | fix1_rerun 12:19:12 | OK |
| 009 (fix cycle 2) | 12:30:01 (`6598f4b872f`) | fix2_rerun 12:31:45 | OK |

Expectations and refutation criteria are stated in each entry before the
corresponding runs; 008's trigger cites the interim 7/8-shard tally (sweep
ended 12:09), consistent with a fix cycle. Honesty notes: (a) commit
timestamps come from the local clock — content cross-checks (interim vs final
tallies) corroborate the ordering; (b) the 007 "fleet identity" addendum,
labeled "recorded at launch", was committed at 12:15 inside the fix1 commit —
record-keeping, not a pre-registration; (c) fix 1b (LC dictionary route) was
not among 007's three named red-cell levers (fix 1a is a variant of lever 2)
— permitted since 008 was itself pre-registered before its re-runs, but it is
lever-list drift; (d) the band-source deviation is covered in check 1.

## 6. Fleet accounting pre-check — CONFIRMED

`launch_receipt.json`: one reservation `r-06bd20b53fbe67a2a`, **8 ×
m8g.24xlarge** (arm64, Neoverse-V2 per smoke lscpu), all `LaunchTime`
**2026-07-28T11:00:44Z**, ap-south-2c, SG `sg-0a32c30cee50aa10e`; instance
IDs and private IPs match `hosts.tsv` and the PREREG 007 addendum exactly.
At verification close (12:47Z) the fleet had consumed ≈ **14.2
instance-hours** (1.77 h × 8) and was still up (required for this
verification; teardown pending — REPORT.md must add the final number).
Work performed on the fleet: 12 A/A cells + 97 attempted sweep cells (96
recorded, 1 baseline-OOM) + 4 ablation + 3 force + 2 off + 6 fix1 + 3 fix2 +
3 verifier spot cells ≈ 2 580 recorded runs.

## Final verdict on evidence quality

**FIX-THEN-SHIP.** Everything the campaign measured is real, pre-registered,
internally consistent, binary-identified, and independently reproducible —
my recompute produced zero verdict changes, and all three on-fleet spot
re-runs land within 0.8 pp of the recorded numbers. The single disqualifying
defect is one of **accounting honesty, not measurement**: the frozen gate
cell `lcstr:probe.inner_all.S5.T96` (baseline-arm OOM, infeasible at S5×LC on
192 GB hosts) vanished from the tally instead of being reported as
NOT-MEASURED. Required fixes before ship, all documentation-level:
1. Report `lcstr:probe.inner_all.S5.T96` in REPORT.md as
   INFEASIBLE-ON-FLEET (baseline-arm OOM; coverage boundary), adjusting the
   headline accounting to 97 cells.
2. Record the band-source deviation (in-sweep arm-A spread, not the
   pre-sweep A/A calibration of PREREG 007; A/A cross-check shows it is
   verdict-neutral on the 10 comparable cells).
3. Label the WORKLOG's ablation/boundary numbers as wall-based (probe-metric
   recompute is stronger and should be quoted alongside).
4. Note that `lcstr S2`'s TIE carries the pre-fix binary (fix 1b changed LC
   routing; only lcstr S3 was re-run).
5. Note the `host`-field artifact (controller hostname in every row) as a
   harness limitation for future campaigns.
