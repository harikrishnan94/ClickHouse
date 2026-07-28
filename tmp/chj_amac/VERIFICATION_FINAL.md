# FINAL INDEPENDENT VERIFICATION — AMAC mission report (`tmp/chj_amac/REPORT.md`)

Verifier: independent (did none of the mission work; adversarial stance — tried to
refute the report). Date: 2026-07-28 (01:30–02:00 UTC). Repo `/mnt/ch/ClickHouse`,
branch `phj-ph`. All commands below were run from `/mnt/ch/ClickHouse/tmp/chj_amac`
unless noted; my fresh artifacts live in `tmp/` (this dir) and
`fleet/results/verify_rerun_shard{1,2}.jsonl` + `fleet/verify_rerun_shard{1,2}.log`.
Nothing else was modified; the fleet was NOT terminated (it remains running, as the
report itself flags).

Scope note per the verification charter: SHIP here asserts evidence integrity — that
the raws honestly support the report's claims *including its red G-perf gate* — not
that the perf outcome is good.

## Check 1 — Independent recount of the headline and specific cells

Own script (`tmp/verify_final_recount.py`) reimplementing only the documented rules
(MATRIX.md "Venue and acceptance rules"; dedup = last attempt nonce per
(cell, arm_role, host); INVALID on any invalid row/missing arm; ≥5 valid runs/arm;
band = max(3%, observed rel spread) — the report invocation passed no band file).
`fleet_ab.py report` was NOT used.

    python3 tmp/verify_final_recount.py fleet/results/results.shard{0..7}.jsonl
    → RECOUNT: WIN=30 TIE=38 LOSS=20 INVALID=17 INSUFFICIENT=0 cells= 105

Headline **reproduces exactly**. Per-cell verdict diff vs `fleet/report_sweep.txt`:
all **105/105 identical** (`diff tmp/report_verdicts.txt tmp/my_verdicts.txt` → empty).
TIE split reproduces (38 = 26 parallel_hash + 12 hash). All 17 INVALID reasons are
`below-duration-floor` (fail-closed floor as documented).

Specific cells recomputed from raws (my numbers → report's):

| Cell | mine | report |
|---|---|---|
| `str:probe.rf_all.S4.T96` (WIN, str probe) | −72.61% (A=3165250us B=867119us) | −72.61% ✓ |
| `str:probe.inner_all.S2.T96` (WIN, str probe) | −7.02% | −7.02% ✓ |
| `k256:probe.inner_all.S5.T96` (WIN) | −15.47% | −15.47% ✓ |
| `key64:probe.inner_all.S5.T96` (LOSS, required) | +13.86% | +13.86% ✓ |
| `key64:probe.inner_all.S4.T96` (LOSS) | +12.80% (band 3.1%) | +12.80% ✓ |
| `key64:probe.inner_all.S2.T96.hash` (hash-inband TIE) | +0.30% | +0.30% ✓ |
| force-engage `key64:probe.inner_all.S1.T96` | +1.22% TIE, arms amacFORCE/amacOFF | +1.22% ✓ |

Binary integrity: every valid sweep row has `binary_sha256` == `proc_exe_sha256` ==
`0d32ef1c96e6` (arm A, baseline) / `dc8b1f17e5a7` (arm B, candidate) — uniform across
all 8 shard files. Matches `bins/MANIFEST.tsv`, which matches `sha256sum` on disk for
both bins (recomputed).

Ablation claims (recomputed from `fleet/results/ablate_shard{0,3}.jsonl`, same
candidate sha both arms):
- `str:probe.inner_all.S3.T96` ring-OFF regression **+27.57%** ✓ (band 3.0%).
- `key64:probe.inner_all.S5.T96` OFF +6.15% ✓; **BuildInsert ON=171.7 s / OFF=170.5 s
  thread-s** vs baseline 134.6 s in the sweep → the build gap is ring-independent, ✓
  exactly as claimed.
- key64 S2 OFF +5.09% ✓; key64 S4 +4.30% TIE inside its 6.5% band ✓; str build S5
  +6.75% TIE inside 18.4% ✓; mixed S5 −1.55% with engagement 0/0 on BOTH arms ✓
  (compile-time excluded family).
- Engagement semantics verified in raws: AmacProbeRows = full probe counts ON, 0 OFF;
  force-engage arm A really engaged (ProbeRows=192M, RingGrowths=128).

Phase receipts: key64 S4 LOSS has ProbeLookup 18.6→13.1 s (−29.4%) and BuildInsert
17.4→28.5 s (+63.9%) ✓ — the report's "loss is paid in BuildInsert, not the lookup"
decomposition is faithful to the raws.

## Check 2 — Live-fleet re-runs (fresh results files, shards 1 and 2)

Fleet confirmed alive (ssh; both arms present, on-host `sha256sum` of candidate =
`dc8b1f17e5a7...`). Commands use the exact `fleet/run_sweep_all.sh` shape with
`--cells` and fresh `--results`:

    python3 fleet_ab.py sweep --cells "key64:probe.inner_all.S5.T96" --shard 1 --shards 8 \
      --ssh-host ubuntu@172.31.29.85 --ssh-key fleet/ssh/id_ed25519 \
      --arm-a bins/clickhouse-baseline-a05f3ee81ff.bin --arm-b bins/clickhouse-candidate-5b276c5fb88.bin \
      --remote-bin-a /home/ubuntu/chj/clickhouse-base --remote-bin-b /home/ubuntu/chj/clickhouse-cand \
      --calibration fleet/calibration_rows.json --results fleet/results/verify_rerun_shard1.jsonl
    (analogous on shard 2 / 172.31.18.3 for "str:probe.inner_all.S2.T96,key64:probe.inner_all.S2.T96.hash")

| Cell (arm-swap venue) | re-run | original | stable? |
|---|---|---|---|
| `key64:probe.inner_all.S5.T96` (LOSS; rerun shard 1, orig shard 6) | **LOSS +11.97%** | LOSS +13.86% | ✓ same sign, comparable, far out of band |
| `str:probe.inner_all.S2.T96` (WIN; rerun shard 2, orig shard 0) | **WIN −6.94%** | WIN −7.02% | ✓ |
| `key64:probe.inner_all.S2.T96.hash` (TIE; rerun shard 2, orig shard 0) | **TIE −0.37%** | TIE +0.30% | ✓ in-band |

Verdict stability holds — including across *different* shards than the originals.

## Check 3 — G-coverage / check_matrix

Documented order (aux files FIRST):

    python3 fleet/check_matrix.py --results "fleet/results/ablate_shard0.jsonl,...,results.shard7.jsonl"
    → rc=0; "disposition counts: MEASURED=68 INFERRED=1079 PARITY-ONLY=297
       EXCLUDED-INVALID=71 NOT-CLAIMED=285 UNDISPOSITIONED=0"; "0 undispositioned";
       19 WARNING lines (12 hash + 7 modifier-floor non-universe entries, as documented)

Reversed order (aux last): rc=1, **"187 undispositioned"** with a loud A/A-evidence
explanation — the report's order-sensitivity note (assumption 6 / finding 12) is real
and honestly documented, and the check fails closed rather than silently.

Sampled dispositions:
- 3 random INFERRED (`k128:build.inner_all.S4.T96`, `null64:probe.any.S2.T48`,
  `k256:probe.semi_anti.S4.T48`): all `from=` cells are genuinely MEASURED with raw
  verdicts (WIN/TIE/LOSS) present in the shard JSONLs; the third inherits a **LOSS** —
  losses propagate conservatively as claimed.
- 3 random floor EXCLUDED-INVALID (`str:build.inner_all.S2.T96`,
  `key64:build.inner_all.S2.T96`, `key64:probe.inner_all.S3.T96.h05`): each cites the
  byte-exact `invalid_reason` string AND correct shard found in the raws. There are
  exactly 17 duration-floor entries = my 17 INVALID cells.
- `lcstr:probe.inner_all.S5.T96` OOM evidence verified verbatim in
  `fleet/sweep_shard0.log` (Code 241, "would use 191.45 GiB").
- Plan arithmetic: matrix.json plan = 106 cells (94+12); results contain exactly 105;
  the only missing cell is the OOM'd lcstr S5; zero extra cells.

## Check 4 — PREREG audit

PREREG.md is committed on-branch; `git log -S "## PREREG-nnn"` per entry:

| Entry | added in | implementing commit | order |
|---|---|---|---|
| 001/002a/002b/002c/003 | `b159a96e9a2`/`7708ef69e8e` (16:59/17:07) | first product commit `d2a759e684f`+ | ✓ precedes |
| 004 | `8cf071c284f` | `844ee1a82dd` (route decorrelation) | ✓ immediate parent |
| 005 | `8ce8b831401` | `60b8d1684a1` (cursor layer) | ✓ precedes |
| 006 | `4a32708e08a` | `7e64a6cf4d5` (build ring) | ✓ precedes |
| 007 | `837247e57fc` | `5b276c5fb88` (routed probe + find ring) | ✓ precedes |

Named process gaps: "no U4 PREREG" — confirmed (PREREG.md ends at 007 + the 003
appendix; no U4 gate entry). "No U4 WORKLOG entries" — was true at report-assembly
time (REPORT.md mtime 01:23:58); a U4 WORKLOG entry was appended at 01:27:23 that
self-describes as "written at report time". Stated, then closed — not hidden
(see discrepancy C).

## Check 5 — Negative proofs

- `grep -ac ConcurrentHashJoinAmacBuildRows bins/clickhouse-baseline-a05f3ee81ff.bin`
  → **0** (candidate → 3; `...AmacProbeRows` candidate → 3). ✓
- Disasm anchors: `disasm/U23_build_anchors.md` ends `G-DISASM-BUILD: PASS
  (0 unexplained)`; `disasm/U3_probe_anchors.md` ends `G-DISASM-PROBE: PASS
  (0 unexplained)`; 3 build + 3 probe anchors = 6/6, per-criterion PASS tables present.
- Direct `llvm-nm-22 --demangle --defined-only bins/clickhouse-candidate-5b276c5fb88.bin`
  (no analyze-assembly cache): **64** `DB::amacFindPass<...>` instantiations (matches
  the report's "64 amacFindPass instantiations" and `disasm/nm_u3_findpass.txt`), and
  the string-key probe anchor symbol
  `amacFindPass<HashMethodString<..., RowRefList>, ResumableHashMap<...SavedHash...TailPaddedHashTableGrower...>>`
  exists at the same address (0x15b2de40) as archived. ✓
- Parity harness teeth: `parity/gate_selftest_absence.log` ends `PARITY FAIL
  (0 divergences, 1 gate failure(s); ...)` with the named absent-counter gate failure
  → `--require-engagement` genuinely fails a counterless binary. ✓

## Check 6 — Hunt results

- **Losses downplayed as ties:** none. 105/105 verdicts agree with my independent
  recount; the report's 20-loss list = exactly my 20 LOSS cells; the ablation TIEs
  quoted with wide bands (key64 S4 6.5%, str build 18.4%) verified from raw spreads.
- **Wins without the phase event moving:** 29/30 WINs show the claimed phase event
  improving. ONE violation — see discrepancy B.
- **Local presented as fleet:** not found. All G-perf/hash/ablation/force numbers come
  from fleet JSONLs with the fleet binaries' shas on both `binary_sha256` and
  `proc_exe_sha256`; the U3(c) A/B is a PREREG-007 local orientation gate by
  definition and its raws (`fleet/results/u3_probe_ab.jsonl`) reproduce the claimed
  −43.5%..−57.9% with arm A = sha of `clickhouse-candidate-7e64a6cf4d5.bin` (verified).
  Ablation/force shard attributions (instances i-0fe67352…, i-0e97d083…, i-09c67629…)
  match `fleet/hosts.tsv`.
- **Evidence hashes:** 8 spot-checks, 8 matches, 0 mismatches
  (`noise_band_002c.jsonl` 3dd917…09f8b4a3c full match; `gate_002c_run2.log`
  3de7ae…df481f9449 full match; `noise_band_002c_rev1.jsonl`, `parity/gate_002a_run1.log`,
  `cursorlayer_hash_ab.jsonl` 9e9906f2…, `hash_t96_aa.jsonl` f7599f89…,
  `uncommitted-cursorlayer.tmp.bin` 2d9a0113…, MANIFEST rows a1e71812…/43ef2b74…/c8260c68…).
- **Top-verdict block vs disk:** every artifact named resolves and its final line
  matches verbatim (parity `gate_u3.log` + `gate_hygieneu3.log`; order
  `gate_u3_order2.log` + `gate_baseline_normal.log` + power-check logs;
  `fleet/gate_002c_run1.log` FAIL → `run2` PASS — the attempt-1 failure is real;
  `VERIFICATION_U2.md` line 263 "U2 VERIFICATION: SHIP"; `report_sweep.txt` final line;
  MANIFEST byte sizes → +1.97%/+93,750,120 arithmetic checks out, and the report
  itself discloses the 352-byte snapshot quirk in gap 9).
- gtests re-run by me: `[  PASSED  ] 10 tests.` in default env AND
  `CLICKHOUSE_JOIN_AMAC=0` (rc=0 both). ✓

## Check 7 — Named gaps: honest? runnable?

| Gap | honest? | runnable <10 min? |
|---|---|---|
| 1 broad stateless `join` differential | ✓ no artifact exists (order/logs holds only 03448/03711 + per-check controls) | no (hours) |
| 2 lcstr S5 OOM | ✓ raw log evidence verified | no (structural at venue) |
| 3 h05 floor | ✓ both cells INVALID in raws with floor reasons | no (floor is structural at the frozen row counts) |
| 4 statson build floor | ✓ INVALID in raws; probe statson TIE −2.16% reproduces | no (same) |
| 5 str dup16 build floor | ✓ INVALID in raws | no (same) |
| 6 fleet termination unexecuted | ✓ verified myself: shards alive via ssh at 01:31 | not mine to run (verifier barred from touching the fleet) |
| 7 exact fleet binary never ran G-parity | ✓ stated with bracketing evidence | **YES — RAN IT (closed)** |
| 8 mechanism (i) per-IP run | ✓ no artifact claims it | no (profiling campaign, not a gate) |
| 9 352-byte size-base quirk | ✓ arithmetic verified | n/a |

**Gap 7 closed by running it:**

    bash tmp/chj_amac/parity/run_parity.sh tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin \
      tmp/chj_amac/bins/clickhouse-candidate-5b276c5fb88.bin --require-engagement
    → "verified: server pid ... runs the given binary (sha256 dc8b1f17e5a7fcce614c8d26e25031cf1adb100f60955d7d6ffc63f934aea653)"
      "AMAC-FORCE PASS: engaged 8/8+2x0 (build,probe) (expected engaged: 8/8, excluded at zero: 2/2)"
      "PARITY OK (636 cases: 634 compared, 2 matched-error, 0 failed; 10 families, 23 kind-strictness combos, force-pass: engaged 8/8+2x0 (build,probe))"  [exit 0]
    (full log: tmp/chj_amac/tmp/verify_parity_fleetbin.log)

The exact fleet candidate binary is now parity-green with dual-side engagement; the
report's provenance caveat can be retired.

## Discrepancies (all non-blocking; every load-bearing number reproduces)

A. **Loss-attribution table's shard column is wrong for 9/20 cells** (e.g.
   `key64:probe.inner_all.S5.T96` says shard 0, raws say shard 6;
   `key64:probe.inner_all.S4.T96` says 3, raws say 5; also any/semi_anti/h50/asof/S1/
   fixstr/k128 rows). The per-family tables and the raw JSONLs agree with each other;
   only the loss-table column is mislabeled. Diffs, verdicts, and mechanisms are
   unaffected (all reproduce to the digit). Fix in an erratum.

B. **One of the 30 WINs violates the stated phase-attribution rule:**
   `lcstr:probe.inner_all.S3.T96` (WIN −4.46%) has ProbeLookup WORSE (21.0→24.0
   thread-s); the wall win is carried by the build phase (19.4→11.2). The report's own
   per-family table prints these true phase numbers, and lcstr is a disclosed
   AMAC-excluded regression-guard family (MATRIX caveat 4, engagement 0/0/0) — but the
   G-perf prose "Wins are carried by the claimed phase event … per the acceptance
   rule" overreaches for this cell. Strictly, the headline is 29 phase-attributed WINs
   + 1 wall-only WIN. Since G-perf is already declared RED/MUST-HOLD, this weakens
   nothing that was accepted; it should be recorded as an erratum.

C. **Finding 16 went stale minutes after assembly:** WORKLOG.md now HAS a U4 entry
   (appended 01:27:23, four minutes after REPORT.md's mtime), self-described as
   "written at report time". Direction is benign — the stated gap was closed, not
   hidden — but REPORT finding 16's "no Unit-4 entries" no longer matches the file.

Retracted suspicion (for the record): a first-pass probe suggested nonzero
`ConcurrentHashJoin*` events on hash-inband rows; the detailed pass showed my own
check was wrong (present-but-all-zero engagement dicts). The report's "all zero on
hash cells" claim is correct.

## Verdict rationale

The headline, all 105 per-cell verdicts, the ablation exoneration, the force-engage
mismatch, the coverage counts, and the PREREG ordering all reproduce from raw
evidence under an independent implementation; three live-fleet re-runs (including on
different shards) confirm verdict stability; negative proofs and hash chains hold;
the red G-perf gate is reported with losses fully enumerated and honestly attributed;
the one previously-unrun cheap gate (G-parity on the exact fleet binary) passes when
actually run. The three discrepancies found are presentation-level errata that do not
change any verdict, any accepted claim, or the red gate.

FINAL VERIFICATION: SHIP
