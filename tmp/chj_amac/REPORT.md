# REPORT — AMAC rings + order-preserving routed probe for `ConcurrentHashJoin` (branch `phj-ph`)

Mission: implement (1) AMAC rings for build-insert and probe-lookup and (2) an
order-preserving routed probe for `ConcurrentHashJoin`
(`join_algorithm = 'parallel_hash'`), with acceptance vs the two-level
baseline (branch `concurrent-hash-join-profile-events` @ `a05f3ee81ff`, sha256
`0d32ef1c96e6...`). Final candidate: `5b276c5fb88` (sha256 `dc8b1f17e5a7...`).
Acceptance venue: 8× `m8g.24xlarge` (ap-south-2c, aarch64 Neoverse-V2, 96
cores), paired ABAB, ≥5 valid runs/arm/cell, band = max(3%, frozen per-shape
A/A spread), fail-closed 200 ms duration floor. Report assembled 2026-07-28
(Unit 4). All paths relative to `tmp/chj_amac/` unless absolute.

The mission brief pre-accepted that the scatter design may lose shapes vs the
two-level baseline and required exactly this honest red report rather than
silent acceptance. The in-repo trace of that framing is PREREG-007's action
clause ("if the loss stands the ring/routed probe does not ship for that
family (excluded-measured-loss, force-engage discriminator in Unit 4)",
`PREREG.md`) and MATRIX.md's disposition vocabulary, which reserves a place
for measured losses instead of hiding them.

## Top verdicts

```
U1 — GREEN. PREREG-001 (reference builds), PREREG-002a (G-parity: PARITY OK, 636 cases),
     PREREG-002b (G-order power check fails BOTH scatter arms as required), PREREG-002c
     (noise band: GREEN on attempt 2 after a real attempt-1 failure, fixed fail-closed —
     200 ms duration floor + 2M probe rows/thread now enforced). Matrix + calibration frozen.
U2 — GREEN (independently verified: "U2 VERIFICATION: SHIP"; verifier findings F1–F6 all
     closed or non-blocking, F1 closed with a fresh green gate on the committed snapshot).
U3 — GREEN. All six PREREG-007 gates:
     (a) PARITY OK ... force-pass: engaged 8/8+2x0 (build,probe)
     (b) ORDER OK (ok=9 fail=0 source_artifact=8 of 17) after the oracle correction;
         baseline passes its own gate; power check still fails scatter binaries
     (c) probe A/B vs pre-U3: NO losses; wall −43.5%..−57.9%, ProbeLookup −57%..−78%
     (d) G-DISASM-PROBE: PASS (0 unexplained) — with U2's G-DISASM-BUILD, 6/6 anchors PASS
     (e) gtests: [  PASSED  ] 10 tests. in default env AND CLICKHOUSE_JOIN_AMAC=0
     (f) binary +1.97% (+93,750,120 bytes; +11.3 MiB text), under the ~2% ceiling, accounted
U4 — MIXED; red where it is red:
     G-hash-inband  GREEN — 12/12 TIE (tail-padded grower in-band for join_algorithm='hash').
     G-ablation     GREEN — the AMAC rings are NOT the loss cause: ring-OFF regresses
                    str probe S3 +27.57%, key64 S2 +5.09%, key64 S5 +6.15%; key64 S4 +4.30%
                    and str build S5 +6.75% are in-band ties; mixed unchanged (−1.55%,
                    ring-excluded family). No tested cell is faster with the ring off.
     G-force-engage RAN, with an HONEST PREDICTION MISMATCH — forced ring at
                    key64:probe.inner_all.S1.T96 = +1.22% OFF-vs-FORCE, in-band (band 3%):
                    the predicted small-map ring loss did NOT reproduce. The auto-disengage
                    threshold is precautionary, and the ahj small-map-loss lead does not
                    reproduce at this venue.
     G-coverage     GREEN — check_matrix: "0 undispositioned". 1800-cell universe =
                    68 MEASURED / 1079 INFERRED / 297 PARITY-ONLY / 71 EXCLUDED-INVALID /
                    285 NOT-CLAIMED; + 12 hash-inband and 7 modifier floor cells tracked
                    separately (warned-and-ignored non-universe entries, by design).
     G-perf         RED — MUST-HOLD. 30 WIN / 38 TIE / 20 LOSS / 17 floor-invalid over the
                    106-cell plan (105 ran; lcstr S5 OOM-excluded at the venue). The losing
                    families that MUST HOLD before any ship decision: cheap numeric keys at
                    T96 with the probe Memory table resident (key64/key32/null64 S4–S5:
                    +8.99%..+14.79%; ANTI S4 outlier +42.35%), the hashed-route mixed family
                    (+7.38%..+10.74%), the small-map S1/S2 floor (+3.16%..+5.09%),
                    k256 S4.T48 (+4.33%), key64 asof S4 (+8.38%), key64 h50 (+3.47%).
                    Attribution below; the rings are exonerated by G-ablation.
Authorization flags:
     Fleet LAUNCHED per requester authorization (8× m8g.24xlarge via Dev_AWS_Admin,
     ap-south-2; receipt fleet/launch_receipt.json, LaunchTime 2026-07-27T23:25:56Z;
     hosts fleet/hosts.tsv; authorization text in fleet/launch.sh header and WORKLOG
     requester decision 1).
     ⚠ Fleet TERMINATION NOT EXECUTED: teardown was owed at campaign end (PREREG-003
     appendix: "Teardown owed at campaign end: fleet/teardown.sh (instances + SG),
     accounting into REPORT.md") but has NO execution artifact, and a read-only
     `aws ec2 describe-instances` check during report assembly (2026-07-28) shows all
     8 instances still `running`. Requester action required: run fleet/teardown.sh and
     paste its output here. The report assembler is barred from touching the fleet.
```

Risk-accepted leads (recorded, not shipped):

- `ZeroingHashTableAllocator` (`ahj`) deliberately not ported — grower rebind
  only (PREREG-005; WORKLOG U2.2). Candidate keeps `HashTableAllocator`.
- `ahj`'s pointer-scheme lane for non-word mapped types not ported; ASOF keeps
  the plain routed loop (`U3_DRAFT_NOTES.md` — no acceptance anchor needs it).
- Cheaper consistent route hash for the hashed (`mixed`) family, build+probe
  together — parity-neutral, pre-registerable follow-up; the named fix-path
  for loss mechanism (ii) below.
- Per-IP counter run on a fleet shard — the named settling instrument for loss
  mechanism (i) below (UNSETTLED hypothesis).

HIGH-IMPACT assumptions this report rests on:

1. S-labels are nominal residency classes defined by frozen row counts, not
   per-arm byte guarantees (MATRIX caveat 1; per-slot maps quantize
   differently than the baseline's single two-level map).
2. S5 row counts are analytic extrapolation one grower step beyond the
   measured calibration range (MATRIX caveat 5). The one S5 cell that showed
   unexpected memory (lcstr) was excluded, not trusted.
3. 1079 INFERRED cells inherit the verdict class of their measured block
   representative (family-repr / group-repr / size-interp / thread-interp
   chains recorded per cell in `fleet/dispositions.json`). Losses propagate
   conservatively (e.g. key32/null64 S4 infer from their S5 LOSS).
4. Acceptance is single-venue: ARM (Neoverse-V2) fleet only. No x86 coverage
   in this campaign.
5. The JSONL `host` field records the driver host (`ip-172-31-5-72`) for
   every shard; shard identity comes from the per-shard results file and the
   `shard` field (wired by `fleet/run_sweep_all.sh --ssh-host` per shard).
6. G-coverage validation is order-sensitive when the auxiliary JSONLs are
   pooled: ablation/force-engage rows reuse base cell ids with the candidate
   binary on BOTH arms, and `dedup_last_attempt` is last-write-wins per
   (cell, arm_role, host) in file order — the auxiliary files MUST precede
   the shard files in `--results` (see G-coverage row and findings register).

## G-perf, honestly (the red gate)

Headline: `FLEET_AB REPORT RESULT: cells=105 win=30 tie=38 loss=20 invalid=17
insufficient=0 uncalibrated=0` (`fleet/report_sweep.txt` final line; 38 TIE =
26 parallel_hash + 12 hash-inband). The plan was 106 cells (94 parallel_hash
per MATRIX.md blocks 1–9, including 15 modifier cells, + 12 hash-inband); the
one cell that never produced rows is `lcstr:probe.inner_all.S5.T96` — the
BASELINE arm OOMed in warmup (`Code: 241 ... MEMORY_LIMIT_EXCEEDED ... would
use 191.45 GiB`, `fleet/sweep_shard0.log`), i.e. the shape is structurally
unreachable at this venue (MATRIX caveats 4–5); dispositioned
EXCLUDED-INVALID, named again under gaps.

Where the candidate wins: the entire `str` probe grid (17 str probe cells WIN
across every kind/strictness group at S2 and S4, wall −5.5%..−72.6%, the
largest being `str:probe.rf_all.S4.T96` −72.61%) plus
`str:build` S5 (−24.29%); `k256` 8 WINs (probe S2 across T1/T48/T96, S3, S4
T1/T96, S5 −15.47%, build S5 −17.87%); `key64` DRAM ladder at low lanes (S4
T1 −8.54%, S4 T48 −4.00%); `key64:probe.rf_all.S4.T96` −68.92% (used-flags
shape); `lcstr` S3 −4.46%. Wins are carried by the claimed phase event
(`ProbeLookup`, e.g. k256 S3 −14.6 thread-s of 23.4, k256 S5 −50.0 of 109.4,
str S5 −32.4 of 81.2; `BuildInsert` for the build cells), per the acceptance
rule.

The 20 losses, each attributed (evidence: `fleet/report_sweep.txt` per-cell
phase blocks; `fleet/results/results.shard*.jsonl` raw rows;
`fleet/results/ablate_shard{0,3}.jsonl`):

| Loss cell | diff | shard | mechanism |
|---|---|---|---|
| `key64:probe.inner_all.S4.T96` | +12.80% | 3 | (i) |
| `key64:probe.inner_all.S5.T96` | +13.86% | 0 | (i) |
| `key32:probe.inner_all.S5.T96` | +11.18% | 5 | (i) |
| `null64:probe.inner_all.S5.T96` | +8.99% | 2 | (i) |
| `key64:probe.left_all.S4.T96` | +14.79% | 7 | (i) |
| `key64:probe.any.S4.T96` | +9.67% | 1 | (i) |
| `key64:probe.semi_anti.S4.T96` | +12.46% | 1 | (i) |
| `key64:probe.semi_anti.S4.T96.anti` | +42.35% | 6 | (iii) = (i) on a small cell |
| `k256:probe.inner_all.S4.T48` | +4.33% | 4 | (i) (BuildInsert +51.4%) |
| `key64:probe.inner_all.S3.T96.h50` | +3.47% | 5 | (i) (BuildInsert +43.0%) |
| `key64:probe.asof.S4.T96` | +8.38% | 5 | (i) + (ii): BuildInsert +20.4% AND un-ring-offset routed probe (asof is AMAC-excluded; ProbeLookup +9.0%) |
| `mixed:probe.inner_all.S2.T96` | +10.74% | 3 | (ii) hashed-route serialize |
| `mixed:probe.inner_all.S3.T96` | +7.38% | 6 | (ii) hashed-route serialize |
| `mixed:probe.inner_all.S5.T96` | +7.61% | 1 | (ii) hashed-route serialize |
| `key64:probe.inner_all.S1.T96` | +4.34% | 4 | (ii) small-map floor |
| `str:probe.inner_all.S1.T96` | +3.16% | 7 | (ii) small-map floor |
| `fixstr:probe.inner_all.S2.T96` | +5.02% | 1 | (ii) small-map floor |
| `k128:probe.inner_all.S2.T96` | +4.95% | 2 | (ii) small-map floor |
| `key64:probe.rf_all.S2.T96` | +4.15% | 5 | (ii) small-map floor |
| `str:probe.asof.S2.T96` | +3.43% | 4 | (ii) small-map floor (asof AMAC-excluded) |

First, the exoneration: **the AMAC rings are NOT the cause.** G-ablation
(same candidate binary both arms, `CLICKHOUSE_JOIN_AMAC` auto vs 0) shows the
ring wins or ties on every tested cell *including losing ones*:

- `str:probe.inner_all.S3.T96`: ring OFF regresses **+27.57%** (band 3%).
- `key64:probe.inner_all.S5.T96` (a fleet LOSS +13.86%): ring OFF makes it
  **+6.15% worse still**.
- `key64:probe.inner_all.S2.T96`: OFF +5.09%; `key64` S4 +4.30% (inside its
  6.5% band); `str:build` S5 +6.75% (inside 18.4%); `mixed` S5 −1.55% —
  unchanged, because mixed is a compile-time ring-excluded family
  (engagement counters 0 on both arms).

Raw lines: `python3 fleet_ab.py report --results
fleet/results/ablate_shard0.jsonl --no-phases` → `CELL
str:probe.inner_all.S3.T96 verdict=LOSS A[amacON]=335354us B[amacOFF]=427810us
diff=+27.57% ...` (arm A = ring ON; "LOSS" here means OFF is slower).

The losses decompose into:

**(i) A structural build-phase gap on cheap numeric keys at T96 with the
probe Memory table resident.** In every mechanism-(i) loss cell,
`ConcurrentHashJoinBuildInsertMicroseconds` regresses +29%..+76%
(key64 S4 17.4→28.5 thread-s +63.9%; key64 S5 134.6→178.2 +32.4%; key32 S5
134.2→173.7 +29.4%; null64 S5 142.9→184.7 +29.2%; left_all S4 15.7→27.6
+75.6%; any S4 16.8→29.5 +75.2%; semi S4 16.9→27.5 +62.8%; anti S4 16.1→24.9
+54.2%; k256 S4.T48 11.9→18.0 +51.4%). Three facts pin it as structural, not
ring-induced:

1. It is **ring-independent**: in the ablation at key64 S5 the candidate's
   BuildInsert is ~171 thread-s with the ring ON (171.7) *and* OFF (170.5)
   vs the baseline's 134.6 — turning the ring off does not recover the gap.
2. It is **PARITY in the dedicated build cell**: `key64:build.inner_all.S5.T96`
   is TIE (+0.48%, BuildInsert 129.8→126.5 thread-s, −2.5%) — same key
   family, same map size, same T96; the difference is that probe cells hold
   a much larger probe Memory table resident (e.g. 1536M vs 96M rows)
   during the build phase.
3. The str/k256 families, whose per-row insert is heavier, show BuildInsert
   *improving* at S5 (str −24.3% wall WIN; k256 −17.9% wall WIN) — the gap
   is specific to cheap (16-byte-cell) inserts at 96 lanes.

Mechanism hypothesis: an allocation/page-locality interaction (per-slot map
growth allocating while a huge Memory table occupies the page cache/NUMA
arena) — **UNSETTLED**. Settling instrument, named: a per-IP counter run
(sampled per-instruction-pointer profile) on a fleet shard replaying
`key64:probe.inner_all.S5.T96` on both arms.

The **+42.35% ANTI outlier** is this same mechanism on a small cell: ANTI at
hit-rate 1.0 emits ~zero rows, so wall is dominated by exactly the
build+scan phases where the gap lives (BuildInsert +54.2%); its SEMI sibling
(+12.46%) and the whole key64 S4 group ladder move together.

**(ii) Route-derivation cost the two-level baseline does not pay** — the
baseline's probe has zero dispatch (it probes whole blocks; its two-level
bucket pick rides the hash), while the routed candidate derives a route per
row. For the `mixed` (hashed/serialized-key) family the route requires
serializing the key: `ProbeDispatch` 51 µs → 24.96 **thread-s** at S5
(6.24 at S3, 6.29 at S2) — wall +7.4..+10.7%. Named fix-path: a cheaper
consistent route hash for the hashed family, applied to build and probe
together (parity-neutral, pre-registerable follow-up). The same route +
ordered-emit overhead, un-recouped where lookups are cache-cheap, is the
small-map floor: +3.16%..+5.09% at S1/S2 (six cells above); G-force-engage
shows the ring itself is blameless there (+1.22% in-band when forced).

**(iii)** = (i) on a small cell (the ANTI outlier, above).

Per-phase receipt that the probe machinery itself is sound even in losing
cells: on `key64:probe.inner_all.S4.T96` the candidate's ProbeLookup is
**−29.4%** (18.59→13.13 thread-s) while the wall loses +12.80% — the loss is
paid in BuildInsert (+63.9%) and ProbeDispatch (+1.10 thread-s), not in the
lookup the AMAC ring accelerates.

## Per-family comparison tables

Per measured cell: pooled per-arm wall medians (A = baseline `a05f3ee81ff`,
B = candidate `5b276c5fb88`), per-arm spread (pstdev), band, verdict, the
seven phase-event medians (thread-time summed over lanes), engagement
counters (candidate arm), and the shard that measured it. Source:
`fleet/report_sweep.txt` (verbatim numbers) + `fleet/results/results.shard*.jsonl`
(shard attribution). INVALID = fail-closed duration floor (MATRIX caveat 6).

#### key32

| Cell | Shard | Wall A (baseline) | Wall B (candidate) | diff | band | spread A/B | Verdict |
|---|---|---|---|---|---|---|---|
| `probe.inner_all.S2.T96` | 3 | 264.8 ms | 261.5 ms | -1.26% | 3.0% | 2.6/2.2 ms | **TIE** |
| `probe.inner_all.S3.T96` | 3 | 337.0 ms | 339.7 ms | +0.79% | 3.0% | 2.1/5.7 ms | **TIE** |
| `probe.inner_all.S5.T96` | 5 | 3819.2 ms | 4246.3 ms | +11.18% | 3.0% | 45.4/55.7 ms | **LOSS** |

Phase-event medians, thread-seconds summed over lanes, A→B; engagement[B] = `AmacBuildRows/RingGrowths/AmacProbeRows`:

| Cell | Build | BDispatch | BInsert | BMerge | Probe | PDispatch | PLookup | engagement[B] |
|---|---|---|---|---|---|---|---|---|
| `probe.inner_all.S2.T96` | 0.01→0.010 | 0.001→0.001 | 0.01→0.008 | 0.002→0.000 | 6.8→6.1 | 0.001→0.46 | 4.6→3.2 | 0/0/192M |
| `probe.inner_all.S3.T96` | 3.2→4.9 | 0.25→0.26 | 3.0→4.6 | 0.002→0.000 | 10.8→8.6 | 0.001→0.46 | 8.4→5.5 | 20M/128/192M |
| `probe.inner_all.S5.T96` | 138→179 | 3.6→4.9 | 134→174 | 0.002→0.000 | 85.6→86.8 | 0.002→3.1 | 67.0→64.1 | 380M/384/1536M |

#### key64

| Cell | Shard | Wall A (baseline) | Wall B (candidate) | diff | band | spread A/B | Verdict |
|---|---|---|---|---|---|---|---|
| `build.inner_all.S2.T96` | 2 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 23.4 ms < 200 ms)) |
| `build.inner_all.S3.T1` | 0 | 1554.7 ms | 1550.0 ms | -0.30% | 11.7% | 44.7/182.1 ms | **TIE** |
| `build.inner_all.S3.T48` | 2 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 71.8 ms < 200 ms)) |
| `build.inner_all.S3.T96` | 7 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 75.2 ms < 200 ms)) |
| `build.inner_all.S3.T96.dup16` | 6 | 270.6 ms | 270.5 ms | -0.01% | 3.0% | 3.0/2.8 ms | **TIE** |
| `build.inner_all.S3.T96.statson` | 5 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 66.4 ms < 200 ms)) |
| `build.inner_all.S5.T96` | 1 | 1595.2 ms | 1602.8 ms | +0.48% | 19.0% | 125.6/304.1 ms | **TIE** |
| `build.left_all.S3.T96.dup16` | 6 | 270.9 ms | 270.3 ms | -0.24% | 3.0% | 3.7/3.4 ms | **TIE** |
| `probe.any.S2.T96` | 4 | 310.2 ms | 302.6 ms | -2.46% | 3.0% | 3.5/2.1 ms | **TIE** |
| `probe.any.S4.T96` | 4 | 880.2 ms | 965.4 ms | +9.67% | 4.1% | 21.2/39.1 ms | **LOSS** |
| `probe.asof.S2.T96` | 2 | 427.3 ms | 439.1 ms | +2.77% | 3.0% | 3.2/2.9 ms | **TIE** |
| `probe.asof.S4.T96` | 0 | 1387.3 ms | 1503.6 ms | +8.38% | 5.2% | 22.1/78.7 ms | **LOSS** |
| `probe.inner_all.S1.T96` | 3 | 284.7 ms | 297.1 ms | +4.34% | 3.0% | 1.6/1.8 ms | **LOSS** |
| `probe.inner_all.S2.T1` | 2 | 210.6 ms | 205.4 ms | -2.44% | 3.0% | 1.6/0.9 ms | **TIE** |
| `probe.inner_all.S2.T48` | 1 | 240.9 ms | 240.1 ms | -0.31% | 3.0% | 0.7/1.0 ms | **TIE** |
| `probe.inner_all.S2.T96` | 1 | 309.0 ms | 314.5 ms | +1.78% | 3.0% | 2.5/3.5 ms | **TIE** |
| `probe.inner_all.S3.T96` | 4 | 384.5 ms | 379.9 ms | -1.21% | 3.0% | 1.6/3.7 ms | **TIE** |
| `probe.inner_all.S3.T96.h05` | 2 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 126.1 ms < 200 ms)) |
| `probe.inner_all.S3.T96.h50` | 4 | 246.7 ms | 255.2 ms | +3.47% | 3.0% | 2.1/5.2 ms | **LOSS** |
| `probe.inner_all.S3.T96.statson` | 2 | 373.8 ms | 365.8 ms | -2.16% | 3.0% | 2.3/3.2 ms | **TIE** |
| `probe.inner_all.S4.T1` | 6 | 50115.9 ms | 45836.4 ms | -8.54% | 3.0% | 324.4/208.6 ms | **WIN** |
| `probe.inner_all.S4.T48` | 3 | 1247.5 ms | 1197.5 ms | -4.00% | 3.0% | 27.8/19.0 ms | **WIN** |
| `probe.inner_all.S4.T96` | 5 | 860.2 ms | 970.4 ms | +12.80% | 3.1% | 27.0/21.1 ms | **LOSS** |
| `probe.inner_all.S5.T96` | 6 | 4136.6 ms | 4710.2 ms | +13.86% | 3.0% | 39.8/68.7 ms | **LOSS** |
| `probe.left_all.S2.T96` | 7 | 313.6 ms | 319.1 ms | +1.76% | 3.0% | 2.2/2.5 ms | **TIE** |
| `probe.left_all.S3.T96.jun` | 1 | 500.2 ms | 505.6 ms | +1.08% | 3.0% | 2.0/3.7 ms | **TIE** |
| `probe.left_all.S4.T96` | 7 | 844.7 ms | 969.6 ms | +14.79% | 3.0% | 25.7/17.8 ms | **LOSS** |
| `probe.rf_all.S2.T96` | 5 | 362.5 ms | 377.5 ms | +4.15% | 3.0% | 2.3/3.5 ms | **LOSS** |
| `probe.rf_all.S4.T96` | 4 | 6281.5 ms | 1952.0 ms | -68.92% | 6.7% | 141.2/131.7 ms | **WIN** |
| `probe.semi_anti.S2.T96` | 6 | 308.0 ms | 316.5 ms | +2.76% | 3.0% | 1.9/2.8 ms | **TIE** |
| `probe.semi_anti.S2.T96.anti` | 0 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 35.6 ms < 200 ms)) |
| `probe.semi_anti.S4.T96` | 0 | 850.8 ms | 956.9 ms | +12.46% | 3.0% | 24.3/26.0 ms | **LOSS** |
| `probe.semi_anti.S4.T96.anti` | 6 | 298.3 ms | 424.7 ms | +42.35% | 10.3% | 28.5/43.7 ms | **LOSS** |

Phase-event medians, thread-seconds summed over lanes, A→B; engagement[B] = `AmacBuildRows/RingGrowths/AmacProbeRows`:

| Cell | Build | BDispatch | BInsert | BMerge | Probe | PDispatch | PLookup | engagement[B] |
|---|---|---|---|---|---|---|---|---|
| `build.inner_all.S3.T1` | 0.89→0.92 | 0→0 | 0.89→0.91 | 0.000→0.000 | 0.14→0.12 | 0→0 | 0.09→0.06 | 24M/5/6M |
| `build.inner_all.S3.T96.dup16` | 3.1→2.7 | 0.23→0.22 | 2.8→2.5 | 0.002→0.000 | 4.5→4.5 | 0.000→0.01 | 0.23→0.25 | 0/0/6M |
| `build.inner_all.S5.T96` | 133→131 | 3.5→4.3 | 130→127 | 0.002→0.000 | 6.7→8.2 | 0.000→0.29 | 5.6→6.7 | 380M/384/96M |
| `build.left_all.S3.T96.dup16` | 3.0→2.7 | 0.22→0.21 | 2.8→2.5 | 0.002→0.000 | 4.5→4.6 | 0.000→0.01 | 0.24→0.25 | 0/0/6M |
| `probe.any.S2.T96` | 0.01→0.01 | 0.001→0.001 | 0.01→0.01 | 0.001→0.000 | 8.2→6.5 | 0.000→0.59 | 7.6→5.2 | 0/0/192M |
| `probe.any.S4.T96` | 17.8→30.6 | 0.98→1.1 | 16.8→29.5 | 0.002→0.000 | 27.0→19.0 | 0.001→1.0 | 25.6→16.5 | 92M/256/384M |
| `probe.asof.S2.T96` | 0.04→0.03 | 0.001→0.001 | 0.03→0.03 | 0.009→0.000 | 18.8→21.0 | 0.000→0.46 | 16.8→18.5 | 0/0/0 |
| `probe.asof.S4.T96` | 38.1→45.4 | 1.1→0.92 | 36.9→44.5 | 0.010→0.000 | 52.2→57.3 | 0.001→1.0 | 47.7→52.0 | 0/0/0 |
| `probe.inner_all.S1.T96` | 0.003→0.001 | 0.000→0.000 | 0.001→0.001 | 0.001→0.000 | 4.6→4.9 | 0.001→0.63 | 2.4→1.9 | 0/0/192M |
| `probe.inner_all.S2.T1` | 0.005→0.004 | 0→0 | 0.005→0.004 | 0.000→0.000 | 0.04→0.03 | 0→0 | 0.02→0.01 | 202656/1/2M |
| `probe.inner_all.S2.T48` | 0.01→0.009 | 0.001→0.001 | 0.008→0.008 | 0.001→0.000 | 2.2→2.2 | 0.000→0.19 | 1.3→1.0 | 0/0/96M |
| `probe.inner_all.S2.T96` | 0.01→0.01 | 0.001→0.001 | 0.01→0.01 | 0.001→0.000 | 7.6→6.7 | 0.001→0.62 | 5.3→3.6 | 0/0/192M |
| `probe.inner_all.S3.T96` | 3.3→4.3 | 0.26→0.27 | 3.0→4.0 | 0.002→0.000 | 11.4→9.0 | 0.001→0.61 | 8.9→5.6 | 20M/128/192M |
| `probe.inner_all.S3.T96.h50` | 3.2→4.5 | 0.25→0.27 | 3.0→4.3 | 0.002→0.000 | 8.2→7.2 | 0.000→0.61 | 7.0→5.2 | 20M/128/192M |
| `probe.inner_all.S3.T96.statson` | 2.5→2.5 | 0.27→0.31 | 2.2→2.2 | 0.002→0.000 | 11.4→9.1 | 0.001→0.60 | 8.9→5.7 | 24M/0/192M |
| `probe.inner_all.S4.T1` | 4.7→4.4 | 0.000→0.000 | 4.7→4.4 | 0.000→0.000 | 13.4→9.6 | 0.000→0.000 | 9.8→6.1 | 96M/7/384M |
| `probe.inner_all.S4.T48` | 8.7→8.9 | 0.50→0.56 | 8.2→8.3 | 0.001→0.000 | 14.9→13.1 | 0.000→0.77 | 11.1→8.5 | 94M/128/384M |
| `probe.inner_all.S4.T96` | 18.4→29.6 | 0.96→1.0 | 17.4→28.5 | 0.002→0.000 | 23.3→19.5 | 0.001→1.1 | 18.6→13.1 | 92M/256/384M |
| `probe.inner_all.S5.T96` | 138→183 | 3.8→4.6 | 135→178 | 0.002→0.000 | 90.5→106 | 0.003→4.3 | 71.7→83.4 | 380M/384/1536M |
| `probe.left_all.S2.T96` | 0.01→0.01 | 0.001→0.001 | 0.01→0.01 | 0.001→0.000 | 8.0→7.4 | 0.001→0.61 | 5.1→3.6 | 0/0/192M |
| `probe.left_all.S3.T96.jun` | 3.6→4.7 | 0.28→0.30 | 3.3→4.4 | 0.002→0.000 | 15.4→12.9 | 0.000→0.60 | 9.2→5.6 | 20M/128/192M |
| `probe.left_all.S4.T96` | 16.6→28.7 | 0.90→1.1 | 15.7→27.6 | 0.002→0.000 | 23.9→21.1 | 0.001→1.0 | 18.0→13.0 | 92M/256/384M |
| `probe.rf_all.S2.T96` | 0.03→0.02 | 0.001→0.001 | 0.02→0.01 | 0.01→0.001 | 11.0→11.9 | 0.003→0.64 | 6.5→6.0 | 0/0/192M |
| `probe.rf_all.S4.T96` | 46.6→60.1 | 2.2→2.9 | 40.1→56.9 | 4.2→0.31 | 45.7→45.2 | 0.007→1.2 | 37.2→33.2 | 92M/256/384M |
| `probe.semi_anti.S2.T96` | 0.01→0.01 | 0.001→0.001 | 0.01→0.01 | 0.002→0.000 | 7.3→6.7 | 0.001→0.62 | 5.0→3.6 | 0/0/192M |
| `probe.semi_anti.S4.T96` | 17.9→28.5 | 1.00→1.0 | 16.9→27.5 | 0.002→0.000 | 22.4→19.3 | 0.001→1.1 | 17.7→13.1 | 92M/256/384M |
| `probe.semi_anti.S4.T96.anti` | 17.1→26.0 | 0.97→1.1 | 16.1→24.9 | 0.002→0.000 | 8.9→11.9 | 0.000→1.3 | 8.8→10.5 | 92M/256/384M |

#### str

| Cell | Shard | Wall A (baseline) | Wall B (candidate) | diff | band | spread A/B | Verdict |
|---|---|---|---|---|---|---|---|
| `build.inner_all.S2.T96` | 1 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 27.5 ms < 200 ms)) |
| `build.inner_all.S3.T96` | 1 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 63.2 ms < 200 ms)) |
| `build.inner_all.S3.T96.dup16` | 7 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 128.0 ms < 200 ms)) |
| `build.inner_all.S5.T96` | 1 | 1196.9 ms | 906.2 ms | -24.29% | 12.4% | 148.4/99.3 ms | **WIN** |
| `build.left_all.S3.T96.dup16` | 5 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 128.9 ms < 200 ms)) |
| `probe.any.S2.T96` | 1 | 341.7 ms | 289.2 ms | -15.38% | 3.0% | 1.3/1.2 ms | **WIN** |
| `probe.any.S4.T96` | 2 | 603.2 ms | 453.3 ms | -24.84% | 4.8% | 29.2/12.4 ms | **WIN** |
| `probe.asof.S2.T96` | 4 | 403.2 ms | 417.0 ms | +3.43% | 3.0% | 3.9/2.5 ms | **LOSS** |
| `probe.asof.S4.T96` | 1 | 873.8 ms | 854.6 ms | -2.19% | 3.0% | 17.1/8.9 ms | **TIE** |
| `probe.inner_all.S1.T96` | 7 | 241.9 ms | 249.5 ms | +3.16% | 3.0% | 1.7/2.1 ms | **LOSS** |
| `probe.inner_all.S2.T1` | 1 | - | - | - | - | - | INVALID (below-duration-floor (arm B median 168.4 ms < 200 ms)) |
| `probe.inner_all.S2.T48` | 0 | 238.8 ms | 204.2 ms | -14.51% | 3.0% | 0.6/0.8 ms | **WIN** |
| `probe.inner_all.S2.T96` | 0 | 294.4 ms | 273.8 ms | -7.02% | 3.0% | 3.0/1.8 ms | **WIN** |
| `probe.inner_all.S3.T96` | 7 | 400.1 ms | 337.6 ms | -15.62% | 3.0% | 2.0/1.8 ms | **WIN** |
| `probe.inner_all.S3.T96.h05` | 6 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 160.8 ms < 200 ms)) |
| `probe.inner_all.S3.T96.h50` | 3 | 273.5 ms | 227.5 ms | -16.84% | 3.0% | 3.1/2.0 ms | **WIN** |
| `probe.inner_all.S4.T1` | 4 | 30410.4 ms | 22007.9 ms | -27.63% | 3.0% | 92.5/508.0 ms | **WIN** |
| `probe.inner_all.S4.T48` | 6 | 798.2 ms | 707.7 ms | -11.34% | 3.6% | 19.4/25.6 ms | **WIN** |
| `probe.inner_all.S4.T96` | 2 | 528.9 ms | 426.9 ms | -19.28% | 5.5% | 29.2/17.2 ms | **WIN** |
| `probe.inner_all.S5.T96` | 4 | 2820.5 ms | 2236.5 ms | -20.70% | 12.0% | 339.6/162.3 ms | **WIN** |
| `probe.left_all.S2.T96` | 2 | 326.2 ms | 301.8 ms | -7.50% | 3.0% | 3.1/1.5 ms | **WIN** |
| `probe.left_all.S4.T96` | 7 | 552.0 ms | 453.5 ms | -17.84% | 4.4% | 24.4/15.6 ms | **WIN** |
| `probe.rf_all.S2.T96` | 3 | 430.9 ms | 407.2 ms | -5.52% | 3.0% | 3.0/2.4 ms | **WIN** |
| `probe.rf_all.S4.T96` | 1 | 3165.2 ms | 867.1 ms | -72.61% | 3.0% | 87.1/15.3 ms | **WIN** |
| `probe.semi_anti.S2.T96` | 5 | 294.4 ms | 275.4 ms | -6.44% | 3.0% | 2.3/1.7 ms | **WIN** |
| `probe.semi_anti.S2.T96.anti` | 5 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 59.3 ms < 200 ms)) |
| `probe.semi_anti.S4.T96` | 4 | 521.7 ms | 436.8 ms | -16.29% | 5.8% | 30.4/8.7 ms | **WIN** |
| `probe.semi_anti.S4.T96.anti` | 2 | 276.4 ms | 219.7 ms | -20.51% | 10.4% | 28.8/5.3 ms | **WIN** |

Phase-event medians, thread-seconds summed over lanes, A→B; engagement[B] = `AmacBuildRows/RingGrowths/AmacProbeRows`:

| Cell | Build | BDispatch | BInsert | BMerge | Probe | PDispatch | PLookup | engagement[B] |
|---|---|---|---|---|---|---|---|---|
| `build.inner_all.S5.T96` | 103→76.2 | 1.6→1.8 | 101→74.4 | 0.002→0.000 | 5.5→4.4 | 0.000→0.25 | 4.9→3.6 | 191M/384/48M |
| `probe.any.S2.T96` | 0.03→0.02 | 0.002→0.002 | 0.03→0.02 | 0.002→0.000 | 17.0→9.8 | 0.000→1.0 | 15.5→6.3 | 0/0/192M |
| `probe.any.S4.T96` | 14.6→11.4 | 0.46→0.50 | 14.1→10.9 | 0.002→0.000 | 27.4→15.3 | 0.000→1.0 | 26.0→12.3 | 47M/256/192M |
| `probe.asof.S2.T96` | 0.08→0.05 | 0.002→0.002 | 0.06→0.05 | 0.01→0.000 | 20.5→23.6 | 0.000→0.77 | 18.5→20.8 | 0/0/0 |
| `probe.asof.S4.T96` | 31.6→27.0 | 0.41→0.41 | 31.2→26.6 | 0.01→0.000 | 34.4→37.7 | 0.000→0.77 | 32.4→34.8 | 0/0/0 |
| `probe.inner_all.S1.T96` | 0.004→0.002 | 0.000→0.000 | 0.002→0.002 | 0.002→0.000 | 6.4→6.5 | 0.000→1.2 | 4.2→2.8 | 0/0/192M |
| `probe.inner_all.S2.T48` | 0.02→0.02 | 0.001→0.001 | 0.02→0.02 | 0.001→0.000 | 4.3→2.7 | 0.000→0.33 | 3.3→1.4 | 0/0/96M |
| `probe.inner_all.S2.T96` | 0.03→0.02 | 0.002→0.002 | 0.03→0.02 | 0.002→0.000 | 12.1→8.8 | 0.000→1.1 | 10.1→5.4 | 0/0/192M |
| `probe.inner_all.S3.T96` | 2.6→2.2 | 0.15→0.15 | 2.5→2.0 | 0.002→0.000 | 19.6→13.5 | 0.000→1.0 | 17.6→10.0 | 11M/128/192M |
| `probe.inner_all.S3.T96.h50` | 2.7→2.0 | 0.15→0.14 | 2.5→1.9 | 0.002→0.000 | 14.2→10.4 | 0.000→1.1 | 13.1→8.1 | 11M/128/192M |
| `probe.inner_all.S4.T1` | 5.2→3.7 | 0→0.000 | 5.2→3.7 | 0.000→0.000 | 13.5→6.6 | 0.000→0 | 11.7→4.9 | 48M/6/192M |
| `probe.inner_all.S4.T48` | 9.8→11.6 | 0.31→0.33 | 9.5→11.2 | 0.001→0.000 | 14.8→8.7 | 0.000→0.64 | 13.0→6.2 | 47M/192/192M |
| `probe.inner_all.S4.T96` | 14.6→10.3 | 0.46→0.50 | 14.1→9.8 | 0.002→0.000 | 20.0→13.9 | 0.000→1.0 | 17.9→10.5 | 47M/256/192M |
| `probe.inner_all.S5.T96` | 124→89.8 | 1.7→1.9 | 123→87.9 | 0.002→0.000 | 89.4→62.0 | 0.001→3.9 | 81.2→48.8 | 191M/384/768M |
| `probe.left_all.S2.T96` | 0.03→0.02 | 0.002→0.002 | 0.02→0.02 | 0.002→0.000 | 15.1→11.5 | 0.000→1.1 | 11.4→5.7 | 0/0/192M |
| `probe.left_all.S4.T96` | 14.2→10.2 | 0.47→0.51 | 13.7→9.7 | 0.002→0.000 | 22.8→16.4 | 0.000→1.1 | 19.2→11.1 | 47M/256/192M |
| `probe.rf_all.S2.T96` | 0.05→0.03 | 0.002→0.002 | 0.03→0.03 | 0.01→0.001 | 23.0→19.6 | 0.003→1.2 | 16.1→8.3 | 0/0/192M |
| `probe.rf_all.S4.T96` | 23.1→23.5 | 0.48→0.92 | 20.6→22.5 | 2.0→0.08 | 34.0→32.1 | 0.003→1.1 | 27.2→22.2 | 47M/256/192M |
| `probe.semi_anti.S2.T96` | 0.03→0.02 | 0.002→0.002 | 0.03→0.02 | 0.002→0.000 | 12.0→8.9 | 0.000→1.1 | 9.9→5.4 | 0/0/192M |
| `probe.semi_anti.S4.T96` | 14.3→11.0 | 0.46→0.49 | 13.8→10.5 | 0.002→0.000 | 19.5→13.9 | 0.000→1.0 | 17.5→10.5 | 47M/256/192M |
| `probe.semi_anti.S4.T96.anti` | 13.8→9.4 | 0.45→0.51 | 13.3→8.9 | 0.002→0.000 | 9.9→9.4 | 0.000→0.92 | 9.8→8.5 | 47M/256/192M |

#### fixstr

| Cell | Shard | Wall A (baseline) | Wall B (candidate) | diff | band | spread A/B | Verdict |
|---|---|---|---|---|---|---|---|
| `probe.inner_all.S2.T96` | 4 | 304.8 ms | 320.1 ms | +5.02% | 3.0% | 2.3/1.2 ms | **LOSS** |
| `probe.inner_all.S3.T96` | 7 | 405.1 ms | 403.9 ms | -0.31% | 3.0% | 2.4/2.8 ms | **TIE** |
| `probe.inner_all.S5.T96` | 7 | 2626.5 ms | 2573.4 ms | -2.02% | 6.1% | 67.5/157.1 ms | **TIE** |

Phase-event medians, thread-seconds summed over lanes, A→B; engagement[B] = `AmacBuildRows/RingGrowths/AmacProbeRows`:

| Cell | Build | BDispatch | BInsert | BMerge | Probe | PDispatch | PLookup | engagement[B] |
|---|---|---|---|---|---|---|---|---|
| `probe.inner_all.S2.T96` | 0.04→0.04 | 0.002→0.002 | 0.04→0.04 | 0.001→0.000 | 9.7→10.6 | 0.001→1.5 | 7.5→6.7 | 0/0/192M |
| `probe.inner_all.S3.T96` | 4.4→3.9 | 0.18→0.17 | 4.2→3.8 | 0.001→0.000 | 15.8→14.8 | 0.001→1.4 | 13.4→10.6 | 8M/0/192M |
| `probe.inner_all.S5.T96` | 116→111 | 2.1→2.2 | 114→108 | 0.002→0.000 | 57.8→57.0 | 0.003→6.2 | 48.1→39.6 | 188M/256/768M |

#### k128

| Cell | Shard | Wall A (baseline) | Wall B (candidate) | diff | band | spread A/B | Verdict |
|---|---|---|---|---|---|---|---|
| `probe.inner_all.S2.T96` | 5 | 430.6 ms | 451.9 ms | +4.95% | 3.0% | 2.7/3.0 ms | **LOSS** |
| `probe.inner_all.S3.T96` | 3 | 528.1 ms | 517.8 ms | -1.96% | 3.0% | 1.9/1.6 ms | **TIE** |
| `probe.inner_all.S5.T96` | 7 | 3002.8 ms | 2960.0 ms | -1.43% | 4.9% | 30.4/145.3 ms | **TIE** |

Phase-event medians, thread-seconds summed over lanes, A→B; engagement[B] = `AmacBuildRows/RingGrowths/AmacProbeRows`:

| Cell | Build | BDispatch | BInsert | BMerge | Probe | PDispatch | PLookup | engagement[B] |
|---|---|---|---|---|---|---|---|---|
| `probe.inner_all.S2.T96` | 0.05→0.04 | 0.002→0.002 | 0.04→0.04 | 0.001→0.000 | 10.8→11.5 | 0.000→1.9 | 8.6→7.2 | 0/0/192M |
| `probe.inner_all.S3.T96` | 4.2→3.4 | 0.19→0.18 | 4.0→3.2 | 0.001→0.000 | 16.2→15.0 | 0.000→1.8 | 13.9→10.6 | 8M/0/192M |
| `probe.inner_all.S5.T96` | 112→108 | 1.8→1.8 | 111→106 | 0.002→0.000 | 60.0→61.5 | 0.000→7.5 | 50.9→43.9 | 188M/256/768M |

#### k256

| Cell | Shard | Wall A (baseline) | Wall B (candidate) | diff | band | spread A/B | Verdict |
|---|---|---|---|---|---|---|---|
| `build.inner_all.S2.T96` | 1 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 40.3 ms < 200 ms)) |
| `build.inner_all.S3.T96` | 5 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 84.7 ms < 200 ms)) |
| `build.inner_all.S5.T96` | 5 | 1594.6 ms | 1309.7 ms | -17.87% | 12.9% | 151.0/168.6 ms | **WIN** |
| `probe.inner_all.S1.T96` | 5 | 567.8 ms | 552.5 ms | -2.70% | 3.0% | 4.1/5.0 ms | **TIE** |
| `probe.inner_all.S2.T1` | 1 | 447.8 ms | 419.0 ms | -6.44% | 3.0% | 3.3/3.4 ms | **WIN** |
| `probe.inner_all.S2.T48` | 0 | 495.3 ms | 475.4 ms | -4.01% | 3.0% | 1.8/1.4 ms | **WIN** |
| `probe.inner_all.S2.T96` | 6 | 625.6 ms | 580.3 ms | -7.24% | 3.0% | 4.3/5.2 ms | **WIN** |
| `probe.inner_all.S3.T96` | 0 | 739.6 ms | 638.2 ms | -13.70% | 3.0% | 2.8/3.4 ms | **WIN** |
| `probe.inner_all.S4.T1` | 5 | 61594.8 ms | 52592.4 ms | -14.62% | 3.0% | 324.3/131.5 ms | **WIN** |
| `probe.inner_all.S4.T48` | 4 | 1374.6 ms | 1434.1 ms | +4.33% | 3.0% | 29.4/27.3 ms | **LOSS** |
| `probe.inner_all.S4.T96` | 3 | 889.7 ms | 768.9 ms | -13.57% | 4.3% | 38.2/11.9 ms | **WIN** |
| `probe.inner_all.S5.T96` | 3 | 4541.5 ms | 3839.0 ms | -15.47% | 6.0% | 66.8/228.8 ms | **WIN** |

Phase-event medians, thread-seconds summed over lanes, A→B; engagement[B] = `AmacBuildRows/RingGrowths/AmacProbeRows`:

| Cell | Build | BDispatch | BInsert | BMerge | Probe | PDispatch | PLookup | engagement[B] |
|---|---|---|---|---|---|---|---|---|
| `build.inner_all.S5.T96` | 132→106 | 2.4→2.4 | 129→103 | 0.002→0.000 | 6.7→5.4 | 0.000→0.40 | 6.1→4.4 | 191M/384/48M |
| `probe.inner_all.S1.T96` | 0.004→0.002 | 0.000→0.000 | 0.002→0.002 | 0.002→0.000 | 10.6→8.0 | 0.001→1.7 | 8.5→4.1 | 0/0/192M |
| `probe.inner_all.S2.T1` | 0.01→0.01 | 0→0 | 0.01→0.01 | 0.000→0.000 | 0.09→0.06 | 0→0 | 0.07→0.04 | 202656/1/2M |
| `probe.inner_all.S2.T48` | 0.02→0.02 | 0.003→0.003 | 0.02→0.02 | 0.001→0.000 | 4.7→3.7 | 0.000→0.76 | 3.8→2.0 | 0/0/96M |
| `probe.inner_all.S2.T96` | 0.03→0.02 | 0.003→0.003 | 0.03→0.02 | 0.002→0.000 | 17.6→10.6 | 0.000→1.7 | 15.6→6.5 | 0/0/192M |
| `probe.inner_all.S3.T96` | 3.4→2.7 | 0.20→0.19 | 3.2→2.5 | 0.002→0.000 | 25.5→13.1 | 0.000→1.7 | 23.4→8.8 | 11M/128/192M |
| `probe.inner_all.S4.T1` | 5.6→5.5 | 0→0.000 | 5.6→5.5 | 0.000→0.000 | 22.6→13.9 | 0.000→0.000 | 20.9→12.1 | 48M/6/192M |
| `probe.inner_all.S4.T48` | 12.4→18.5 | 0.51→0.52 | 11.9→18.0 | 0.001→0.000 | 16.1→13.3 | 0.000→1.5 | 14.2→10.0 | 47M/192/192M |
| `probe.inner_all.S4.T96` | 18.4→14.9 | 0.66→0.67 | 17.8→14.2 | 0.002→0.000 | 25.2→14.8 | 0.000→1.7 | 23.0→10.6 | 47M/256/192M |
| `probe.inner_all.S5.T96` | 165→128 | 2.4→2.5 | 162→125 | 0.002→0.000 | 118→75.0 | 0.001→6.2 | 109→59.4 | 191M/384/768M |

#### null64

| Cell | Shard | Wall A (baseline) | Wall B (candidate) | diff | band | spread A/B | Verdict |
|---|---|---|---|---|---|---|---|
| `probe.inner_all.S2.T96` | 7 | 379.1 ms | 385.7 ms | +1.75% | 3.0% | 2.2/2.4 ms | **TIE** |
| `probe.inner_all.S3.T96` | 7 | 452.1 ms | 461.1 ms | +2.00% | 3.0% | 2.4/6.1 ms | **TIE** |
| `probe.inner_all.S5.T96` | 2 | 5201.2 ms | 5669.0 ms | +8.99% | 3.0% | 139.5/77.5 ms | **LOSS** |

Phase-event medians, thread-seconds summed over lanes, A→B; engagement[B] = `AmacBuildRows/RingGrowths/AmacProbeRows`:

| Cell | Build | BDispatch | BInsert | BMerge | Probe | PDispatch | PLookup | engagement[B] |
|---|---|---|---|---|---|---|---|---|
| `probe.inner_all.S2.T96` | 0.02→0.02 | 0.001→0.001 | 0.01→0.01 | 0.001→0.000 | 8.2→7.7 | 0.001→0.64 | 5.8→4.4 | 0/0/192M |
| `probe.inner_all.S3.T96` | 3.6→5.1 | 0.28→0.29 | 3.3→4.8 | 0.002→0.000 | 12.5→9.8 | 0.001→0.65 | 9.9→6.2 | 22M/128/192M |
| `probe.inner_all.S5.T96` | 147→190 | 4.1→4.9 | 143→185 | 0.002→0.000 | 105→122 | 0.003→5.0 | 84.3→95.5 | 423M/384/1708M |

#### lcstr

| Cell | Shard | Wall A (baseline) | Wall B (candidate) | diff | band | spread A/B | Verdict |
|---|---|---|---|---|---|---|---|
| `probe.inner_all.S2.T96` | 6 | 1078.7 ms | 1100.1 ms | +1.98% | 3.0% | 13.3/6.0 ms | **TIE** |
| `probe.inner_all.S3.T96` | 1 | 1330.4 ms | 1271.1 ms | -4.46% | 3.0% | 17.4/12.3 ms | **WIN** |

Phase-event medians, thread-seconds summed over lanes, A→B; engagement[B] = `AmacBuildRows/RingGrowths/AmacProbeRows`:

| Cell | Build | BDispatch | BInsert | BMerge | Probe | PDispatch | PLookup | engagement[B] |
|---|---|---|---|---|---|---|---|---|
| `probe.inner_all.S2.T96` | 0.24→0.06 | 0.005→0.005 | 0.23→0.06 | 0.002→0.000 | 15.1→21.0 | 0.000→2.0 | 12.9→16.8 | 0/0/0 |
| `probe.inner_all.S3.T96` | 19.4→11.2 | 0.39→0.37 | 19.0→10.9 | 0.002→0.000 | 23.4→28.3 | 0.000→2.1 | 21.1→24.0 | 0/0/0 |

#### mixed

| Cell | Shard | Wall A (baseline) | Wall B (candidate) | diff | band | spread A/B | Verdict |
|---|---|---|---|---|---|---|---|
| `build.inner_all.S2.T96` | 1 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 38.3 ms < 200 ms)) |
| `build.inner_all.S3.T96` | 7 | - | - | - | - | - | INVALID (below-duration-floor (arm A median 93.5 ms < 200 ms)) |
| `build.inner_all.S5.T96` | 2 | 1241.3 ms | 1241.4 ms | +0.01% | 7.2% | 81.2/89.8 ms | **TIE** |
| `probe.inner_all.S2.T96` | 3 | 477.7 ms | 529.0 ms | +10.74% | 3.0% | 3.2/4.7 ms | **LOSS** |
| `probe.inner_all.S3.T96` | 6 | 712.2 ms | 764.8 ms | +7.38% | 3.0% | 9.4/5.2 ms | **LOSS** |
| `probe.inner_all.S5.T96` | 1 | 3992.5 ms | 4296.4 ms | +7.61% | 3.0% | 19.1/52.4 ms | **LOSS** |

Phase-event medians, thread-seconds summed over lanes, A→B; engagement[B] = `AmacBuildRows/RingGrowths/AmacProbeRows`:

| Cell | Build | BDispatch | BInsert | BMerge | Probe | PDispatch | PLookup | engagement[B] |
|---|---|---|---|---|---|---|---|---|
| `build.inner_all.S5.T96` | 98.7→93.8 | 6.9→7.0 | 91.7→86.8 | 0.002→0.000 | 11.9→13.8 | 0.000→1.6 | 11.4→11.7 | 0/0/0 |
| `probe.inner_all.S2.T96` | 0.04→0.04 | 0.009→0.009 | 0.03→0.03 | 0.001→0.000 | 22.1→27.7 | 0.000→6.3 | 20.1→19.5 | 0/0/0 |
| `probe.inner_all.S3.T96` | 3.9→3.5 | 0.46→0.45 | 3.4→3.0 | 0.002→0.000 | 41.2→46.9 | 0.000→6.2 | 39.3→38.7 | 0/0/0 |
| `probe.inner_all.S5.T96` | 111→115 | 6.9→7.0 | 104→108 | 0.002→0.000 | 192→219 | 0.000→25.0 | 184→186 | 0/0/0 |

#### G-hash-inband (12 cells, `join_algorithm='hash'` on BOTH arms)

| Cell | Shard | Wall A | Wall B | diff | band | spread A/B | Verdict |
|---|---|---|---|---|---|---|---|
| `k256:probe.inner_all.S2.T1.hash` | 0 | 444.4 ms | 441.6 ms | -0.63% | 3.0% | 2.2/3.5 ms | **TIE** |
| `k256:probe.inner_all.S2.T96.hash` | 2 | 636.0 ms | 638.4 ms | +0.38% | 3.0% | 3.7/1.3 ms | **TIE** |
| `k256:probe.inner_all.S4.T1.hash` | 6 | 60210.0 ms | 61532.0 ms | +2.20% | 3.0% | 192.0/322.7 ms | **TIE** |
| `k256:probe.inner_all.S4.T96.hash` | 5 | 6660.9 ms | 6676.6 ms | +0.24% | 10.1% | 46.2/673.4 ms | **TIE** |
| `key64:probe.inner_all.S2.T1.hash` | 2 | 211.2 ms | 213.1 ms | +0.89% | 3.0% | 0.7/0.9 ms | **TIE** |
| `key64:probe.inner_all.S2.T96.hash` | 0 | 310.0 ms | 310.9 ms | +0.30% | 3.0% | 1.8/1.3 ms | **TIE** |
| `key64:probe.inner_all.S4.T1.hash` | 2 | 49283.8 ms | 49664.5 ms | +0.77% | 3.0% | 205.2/599.6 ms | **TIE** |
| `key64:probe.inner_all.S4.T96.hash` | 1 | 5709.9 ms | 5791.8 ms | +1.43% | 3.0% | 47.6/44.8 ms | **TIE** |
| `str:probe.inner_all.S2.T1.hash` | 0 | 220.3 ms | 223.2 ms | +1.31% | 3.0% | 1.0/2.3 ms | **TIE** |
| `str:probe.inner_all.S2.T96.hash` | 6 | 294.8 ms | 295.1 ms | +0.08% | 3.0% | 2.0/2.6 ms | **TIE** |
| `str:probe.inner_all.S4.T1.hash` | 3 | 29576.1 ms | 30021.4 ms | +1.51% | 3.0% | 193.9/97.6 ms | **TIE** |
| `str:probe.inner_all.S4.T96.hash` | 5 | 5777.6 ms | 5746.8 ms | -0.53% | 3.0% | 84.2/53.6 ms | **TIE** |

(`ConcurrentHashJoin*` phase events and AMAC counters are all zero on hash-algorithm cells — the gate is wall-only by design.)


## Evidence matrix

One row per mission acceptance criterion. Invocations are copy-paste runnable
from the repo root (`/mnt/ch/ClickHouse`); fleet rows name their shard.

| Criterion | Gate invocation (copy-paste) | Result (raw final line) | Non-gate origins | Verdict |
|---|---|---|---|---|
| G-build (all arms build clean) | `bash tmp/chj_amac/build_refs.sh`; per-commit ninja logs named in `bins/MANIFEST.tsv` `build_log` column | baseline: 0 errors/0 warnings, 15,942 edges; `ahj`: 0/0, 16,673 edges; candidate commits incl. final `build_candidate-5b276c5fb88.log`: 38-edge incremental, link OK | WORKLOG U1.2, U1.3 (stale-binary trap caught + fixed), U2/U3 build logs | GREEN |
| G-parity (dual-side engagement) | `bash tmp/chj_amac/parity/run_parity.sh tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin tmp/chj_amac/bins/uncommitted-u3.tmp.bin --require-engagement` | `PARITY OK (636 cases: 634 compared, 2 matched-error, 0 failed; 10 families, 23 kind-strictness combos, force-pass: engaged 8/8+2x0 (build,probe))` (`parity/gate_u3.log`; re-green post-hygiene: `parity/gate_hygieneu3.log`, identical final line) | parity/SELFTEST.md §10 (staged per-side force-pass + negative proof: baseline-as-candidate fails with a named gate failure). PROVENANCE NOTE: the exact fleet binary (`dc8b1f17`) never itself ran G-parity — it is bracketed by parity-green snapshots (22:58 pre-hygiene `2f941e41`, 23:56 post-hygiene `48d258ff`), and the fleet harness enforced cross-arm NULL-aware full-output checksum equality per cell (fail-closed, `fleet_ab.py:1101`) on all 88 valid cells of the exact fleet binaries | GREEN (with provenance note) |
| G-order (incl. oracle correction) | `bash tmp/chj_amac/order/run_order.sh <candidate-bin> --keep-data --baseline-reference tmp/chj_amac/order/logs/gate_002b_baseline.log` | `ORDER OK (ok=9 fail=0 source_artifact=8 of 17 checks, all engaged parallel_hash, t1_global=OK, stateless=pass, stateless engagement: 03448=yes (delta ConcurrentHashJoinProbeMicroseconds=2115) 03711=yes (delta ConcurrentHashJoinProbeMicroseconds=87766); squash checks baseline-differential per SELFTEST §11)` (`order/logs/gate_u3_order2.log`); baseline passes its own gate (`gate_baseline_normal.log`); power mode still fails scatter: `ORDER POWER-CHECK OK (check fails on this binary, as expected: >=1 engaged row-matched T=96 FAIL, errors=0, row_mismatch=0)` | order/SELFTEST.md §11 — baseline-differential justification: the `_squash` per-block rule measured per-lane scan interleaving, which the order-preserving TWO-LEVEL BASELINE fails identically; an oracle the reference design cannot pass is wrong, not strict. Reclassified fail-closed (`--baseline-reference` required; `SOURCE-ARTIFACT` only when the non-squash sibling is OK AND the baseline fails the same check); `--expect-fail` power mode untouched | GREEN |
| G-tests | `build/reldeb/src/unit_tests_dbms --gtest_filter='*Amac*'` (default env, then `CLICKHOUSE_JOIN_AMAC=0`) | `[  PASSED  ] 10 tests.` in BOTH env arms (U3_DRAFT_NOTES gate (ii); U2's 4 tests are a subset, re-run by orchestrator + verifier) | Stateless layer folded into `run_order.sh`: 03448/03711 ×10 on both binaries (scatter fails 10/10 pre-U3; routed candidate passes 10/10 with engagement). **NAMED GAP:** a broad `join`-selector stateless differential run was NEVER executed — no artifact exists (only 03448/03711 stateless logs in `order/logs/`); not claimed | GREEN with named gap |
| G-disasm (6 anchors) | anchor docs built with `asmdiff/asmdiff.py` (disclosed F6 tool deviation) | `G-DISASM-BUILD: PASS (0 unexplained)` (`disasm/U23_build_anchors.md`: key64/keys256/key_string build rings — all-`pstl1keep` prefetch, fused claim, tail-pad advance, inlined refill) + `G-DISASM-PROBE: PASS (0 unexplained)` (`disasm/U3_probe_anchors.md`: 56/56 `pldl1keep`, second-line prefetch >24 B cells, admit-time key pack, SSA-clean) | Two candidate-favoring deltas documented per side (inlined `memequalWide`/SIMD compare vs out-of-line `bcmp`; fewer spill reloads). Reference drain bug found during the port (findings register 2) | GREEN |
| G-hash-inband | `python3 tmp/chj_amac/fleet_ab.py report --results "$(ls tmp/chj_amac/fleet/results/results.shard*.jsonl | paste -sd,)" 2>&1 | grep '\.hash '` (hash cells ran on shards 0, 1, 2, 3, 5, 6) | 12/12 `verdict=TIE` (diffs −0.63%..+2.20%, all in-band; full table above) | matrix.json `hash_inband` (requester decision 4); PREREG-005's local T96 `hash` jitter was honestly deferred to this gate (WORKLOG U2.2) — settled green here | GREEN |
| G-perf | `python3 tmp/chj_amac/fleet_ab.py report --results "$(ls tmp/chj_amac/fleet/results/results.shard*.jsonl | paste -sd,)"` (8 shards; per-cell shard column in the tables above) | `FLEET_AB REPORT RESULT: cells=105 win=30 tie=38 loss=20 invalid=17 insufficient=0 uncalibrated=0` | `fleet/report_sweep.txt`; loss attribution in the G-perf section with ablation cross-evidence | **RED — MUST-HOLD** (attributed; named fix-paths + settling instrument recorded) |
| G-ablation | `python3 tmp/chj_amac/fleet_ab.py report --results tmp/chj_amac/fleet/results/ablate_shard0.jsonl --no-phases` (shard 0, i-0fe67352e5989edf7) and `... ablate_shard3.jsonl` (shard 3, i-0e97d0836ce2798f2) | shard 0: `CELL str:probe.inner_all.S3.T96 verdict=LOSS A[amacON]=335354us B[amacOFF]=427810us diff=+27.57% band=3.0% spread(A/B)=2824/3528us runs=10/10` (+ key64 S5 +6.15%, key64 S4 +4.30% TIE); shard 3: key64 S2 +5.09%, mixed S5 −1.55% TIE, str build S5 +6.75% TIE | Arm semantics: A = ring ON, B = ring OFF; "LOSS" = OFF slower. Engagement verified in raw rows (`AmacProbeRows` = full probe counts ON, 0 OFF; mixed 0/0 = compile-time excluded family) | GREEN |
| G-force-engage | `python3 tmp/chj_amac/fleet_ab.py report --results tmp/chj_amac/fleet/results/force_engage.jsonl --no-phases` (ran on shard 4, i-09c676297620ec10a) | `CELL key64:probe.inner_all.S1.T96 verdict=TIE A[amacFORCE]=299612us B[amacOFF]=303282us diff=+1.22% band=3.0% spread(A/B)=1657/1642us runs=10/10` | PREREG-007's action clause named force-engage as the Unit-4 discriminator for excluded-measured-loss families. **HONEST PREDICTION MISMATCH recorded:** the expectation was that forcing the ring on a cache-resident map would LOSE (justifying auto-disengage); it did not reproduce (+1.22% in-band, FORCE side in fact faster) → the threshold is precautionary, not load-bearing, at this venue; the ahj small-map lead does not reproduce | GREEN as a gate (mismatch recorded, not rationalized) |
| G-coverage | `python3 fleet/check_matrix.py --results "fleet/results/ablate_shard0.jsonl,fleet/results/ablate_shard3.jsonl,fleet/results/force_engage.jsonl,fleet/results/results.shard0.jsonl,fleet/results/results.shard1.jsonl,fleet/results/results.shard2.jsonl,fleet/results/results.shard3.jsonl,fleet/results/results.shard4.jsonl,fleet/results/results.shard5.jsonl,fleet/results/results.shard6.jsonl,fleet/results/results.shard7.jsonl"` | `disposition counts: MEASURED=68 INFERRED=1079 PARITY-ONLY=297 EXCLUDED-INVALID=71 NOT-CLAIMED=285 UNDISPOSITIONED=0` then `0 undispositioned` (rc=0; the 19 WARNING lines are the 12 hash + 7 modifier-floor tracked-separately entries, by design) | ORDER MATTERS: the auxiliary files must come FIRST in `--results`. They reuse base cell ids with the candidate binary on both arms, and `fleet_ab.dedup_last_attempt` is last-write-wins per (cell, arm_role, host) in file order; aux-last clobbers 7 sweep cells' A/B evidence and yields `187 undispositioned` (verified during assembly). The aux rows are env-toggled same-binary experiments, not sweep re-attempts — sweep-last is the semantically correct pooling. `check_matrix` itself was NOT modified | GREEN |

## Compile time and binary size

From `bins/MANIFEST.tsv` (identity of record: sha256):

| Binary | Commit | Bytes | vs baseline |
|---|---|---:|---|
| `clickhouse-baseline-a05f3ee81ff.bin` | `a05f3ee81ff` | 4,849,599,008 | — |
| `clickhouse-ahj-cf465cfbe23.bin` (reference) | `cf465cfbe23` | 4,894,775,184 | +0.93% |
| `clickhouse-candidate-75d431b1d74.bin` (U1) | `75d431b1d74` | 4,740,990,472 | −2.24% |
| `clickhouse-candidate-844ee1a82dd.bin` (U2.1 route fix) | `844ee1a82dd` | 4,740,991,336 | −2.24% |
| `clickhouse-candidate-60b8d1684a1.bin` (U2.2 cursor layer) | `60b8d1684a1` | 4,743,919,256 | −2.18% |
| `clickhouse-candidate-7e64a6cf4d5.bin` (U2.3 build ring) | `7e64a6cf4d5` | 4,748,738,944 | −2.08% |
| `clickhouse-candidate-5b276c5fb88.bin` (U3, fleet arm) | `5b276c5fb88` | 4,842,488,712 | −0.15% |

- The recorded U3 delta: 4,748,738,592 → 4,842,488,712 = **+93,750,120 bytes
  (+1.97%)**, under PREREG-007(f)'s ~2% ceiling. `llvm-size` text:
  474,931,892 → 486,236,880 = **+11.3 MiB code (+2.38% text)** — 30 routed
  instantiations + 64 `amacFindPass` instantiations; the remaining ~82 MiB is
  DWARF (RelWithDebInfo). (`U3_DRAFT_NOTES.md` gate (vi).) The final fleet
  candidate remains 0.15% SMALLER than the baseline binary (the branch's
  earlier two-level-machinery removal more than pays for the rings).
- New-TU compile times (.ninja_log, U3): Any 53.9 s, RightAny 53.4 s, All
  46.5 s, Semi 43.0 s, Anti 40.5 s, Asof 25.6 s, AmacProbe 8.5 s.
- Build-log edge counts: baseline full build 15,942 edges; `ahj` reference
  16,673; U1 candidate re-sync 87-edge incremental (exactly the missing
  commits' TUs); U2.2 full-ripple rebuild (`HashJoin.h` rebind); final fleet
  candidate `5b276c5fb88` 38-edge incremental, 0 errors.

## Findings register

1. **Pre-existing product bug (upstream candidate): `Code: 49` in
   `JoiningTransform`.** `ANY LEFT JOIN` + non-equi residual `ON` + right key
   projected + extra unprojected column + heavy duplication →
   `DB::Exception: Invalid number of rows in Chunk ... While executing
   JoiningTransform. (LOGICAL_ERROR)` — an exception, not a crash. Reproduces
   on baseline AND `phj-ph`, `hash` AND `parallel_hash`, even
   `max_threads=1`; minimal deterministic repro in `parity/SELFTEST.md` §5.
   Affects 2/636 parity cases (classified matched-error, loud warning; the
   cases stay in the matrix so any one-arm change fails the gate).
   **Suggest filing an upstream issue** with the §5 repro.
2. **Real bug found in the `ahj` reference and fixed (teeth-checked):**
   its drain re-seeds pending rows into "first free slots"; a same-sweep
   growth after a failed refill lets the sweep tail step an emptied slot →
   deterministic SIGSEGV on string keys at T96 (`offsets[2^32-2]`), silent
   garbage insert on numeric keys. Fix: slot-preserving, row-sorted re-seed
   (preserves first-wins `RowRef` semantics). The regression gtest was built
   once against the ported buggy logic and crashed (rc=134 — teeth
   verified), then passes with the fix (`U23_DRAFT_NOTES.md` deviations 2–3;
   latent in `ahj`, practically unreachable there due to compact leaf
   sections).
3. **ORDER-oracle correction.** The original `_squash` per-block rule
   measured per-lane scan interleaving: the order-preserving two-level
   baseline itself fails the identical 8 checks. Corrected to
   baseline-differential, fail-closed (`--baseline-reference` mandatory;
   `SOURCE-ARTIFACT` only with a passing non-squash sibling AND a matching
   baseline failure); power mode untouched and still fails scatter binaries.
   Re-proofs: `order/SELFTEST.md` §11. Justification: an oracle the
   reference design cannot pass is wrong, not strict.
4. **U2 verifier findings F1–F6 all closed** (`VERIFICATION_U2.md`):
   F1 uncommitted-gate-binary provenance — closed by equivalence analysis +
   a fresh green run on the committed snapshot (residual nit: the routefix
   gate binary is unarchived); F2 "wall win" wording → corrected to TIE in
   WORKLOG; F3 missing narration (S4→S5 substitution, floor-unverdictable
   S2) → added; F4 pre-hygiene instruction counts not reproducible, the
   load-bearing delta set reproduces; F5 str S5 BuildInsert −20.2% weighted
   lightly (spread-comparable) and settled by the fleet; F6 `asmdiff.py`
   tool substitution — disclosed prereg deviation (the prereg-named tool's
   symbol cache is unsafe on 4.9 GB binaries on this host).
5. **`ZeroingHashTableAllocator` lead** (from `ahj`): deliberately not
   ported in PREREG-005 (grower rebind only); visible in the reference
   disasm anchors as a map-layout delta. Open perf lead for follow-up.
6. **k256 verdict-boundary notes.** Local: `k256 ... +15.13% TIE (band
   14.3%)` was verified legitimate under the pre-registered verdict rule
   (VERIFICATION_U2 check 6) — wide same-binary bands make local T96 `hash`
   cells verdict-insensitive. Fleet: `k256:probe.inner_all.S4.T48` LOSS
   +4.33% sits near its 3.0% band on a quiet shard — boundary-flagged; its
   mechanism-(i) BuildInsert (+51.4%) is what pushes it out.
7. **hash-T96 local jitter, deferred then settled.** PREREG-005's local
   `hash` A/B produced an out-of-band "WIN" that a same-binary A/A exposed
   as noise (−14.12% swings on identical binaries). Not rationalized:
   deferred to the fleet G-hash-inband gate per the pre-stated
   interpretation rule — now GREEN 12/12 TIE on quiet dedicated shards.
8. **ANY-join force/off row-choice nondeterminism (documented, not a
   regression).** ANY-choice under the U2 build ring is run-to-run
   nondeterministic under force (the off arm is checksum-stable and equal to
   `hash`); ANY semantics permit any matching row; pre-U3 == post-U3 exactly
   (`U3_DRAFT_NOTES.md`). The parity generator preserves determinism by
   making every projected column a function of the join key
   (`parity/SELFTEST.md` KNOWN GAPS 4); the fleet harness bans dup16 outside
   `inner_all`/`left_all` for the same reason (`fleet_ab.py` cell grammar).
9. **Remote-launch wait bug + `stop` SIGKILL race (both fixed, both
   evidenced).** (a) `bash -c 'cmd &'` left the server as the ssh shell's
   direct child; the shell then blocks in `wait()` holding the ssh session
   open forever → first remote smoke failed `rc=124: ssh timed out after
   30s` (`fleet/remote_smoke.log`). Fix: double-fork inside a subshell
   (reparents the server to init), pid taken from the server's own status
   file (`fleet_ab.py` `RemoteServer.start`); `remote_smoke2.log` PASS.
   (b) a `kill -0` probe right after SIGKILL races the kernel's teardown of
   a 100-thread server holding multi-GB Memory tables, and a zombie still
   answers `kill -0` → the FIRST fleet sweep failed on ALL 8 shards
   (`... still alive after SIGTERM+SIGKILL`, `fleet/sweep_all.log`). Fix:
   poll `/proc/<pid>/stat` up to ~30 s, SIGKILL after 5 s, state `Z` counts
   as dead; the second sweep (`fleet/sweep_all2.log`) resumed from partial
   results and completed.
10. **U3 commit-split deviation (recorded).** The routed probe and the probe
    ring landed as ONE product commit — one logical change under the
    requester's own unit reordering (the ordered emit is the AMAC probe's
    property); hunk surgery to force the plan's finer split would have
    produced untested intermediate states.
11. **`ahj`-pointer PR-time form decision.** `ahj`'s pointer-scheme lane for
    non-word mapped types was NOT ported; ASOF (`unique_ptr` mapped) keeps
    the plain routed loop, gated by `amac_mapped_fits_word`. No acceptance
    anchor needs it; recorded as the PR-time form (`U3_DRAFT_NOTES.md`).
12. **Fleet dedup sharp edge (harness note).** Auxiliary experiments
    (ablation, force-engage) reuse base cell ids; `dedup_last_attempt`'s
    last-write-wins pooling makes `check_matrix --results` order-sensitive
    (aux files must precede shard files — verified: aux-last yields
    `187 undispositioned`). Recommendation for future campaigns: suffix
    auxiliary cell ids (e.g. `.ablate`, `.force`) so they can never collide
    with sweep evidence.
13. **lcstr S5 venue OOM.** `lcstr:probe.inner_all.S5.T96` OOMed the
    BASELINE arm in warmup (191.45 GiB vs the 194.89 GiB cap) — consistent
    with MATRIX caveats 4 (double-counted dictionary bytes) and 5 (S5 rows
    are extrapolated); per caveat 5's own rule the cell was excluded rather
    than trusted. The lcstr S2/S3 regression guards survived (TIE / WIN).
14. **JSONL `host` field quirk.** Every fleet row records
    `host=ip-172-31-5-72` (the driver host), not the shard instance; shard
    identity rides the results-file name and the `shard` field. Harmless
    here (one cell = one shard) but worth fixing before any multi-host
    pooling relies on host-keyed dedup.
15. **Fleet termination outstanding.** See authorization flags: launch
    receipt exists; teardown script exists; NO teardown artifact; instances
    verified still `running` on 2026-07-28 (read-only describe). Cost is
    accruing until `fleet/teardown.sh` runs; its output is owed to this
    report's accounting per PREREG-003.
16. **WORKLOG gap (process note).** `WORKLOG.md` has no Unit-4 entries; the
    fleet campaign is reconstructed here from `fleet/` artifacts
    (launch/deploy/sweep logs, receipts, results JSONLs). The U4 gate set
    (G-perf/G-coverage/G-ablation/G-force-engage/G-hash-inband) has no
    PREREG entry of its own; its acceptance rules live in MATRIX.md
    ("Venue and acceptance rules"), requester decisions 1/4 (WORKLOG), and
    the mission brief.

## Named gaps (nothing below is claimed)

1. The broad `join` stateless-selector differential run (both arms) was
   never executed; only 03448/03711 ×10 ran inside `run_order.sh`'s
   stateless layer.
2. `lcstr:probe.inner_all.S5.T96` — planned, unmeasurable at this venue
   (baseline-arm OOM); 105 of 106 plan cells produced data.
3. Miss-dominated h=0.05 probes: both `.h05` cells tripped the duration
   floor; no h05 verdict exists (h50 siblings measured: key64 LOSS +3.47%,
   str WIN −16.84%).
4. Build-side stats-on point (`key64:build...statson`) floor-invalid; the
   probe-side statson TIE (−2.16%) is the only stats-on protocol evidence.
5. `str` dup16 build points floor-invalid (the key64 dup16 TIEs survive).
6. Fleet termination unexecuted and unevidenced (instances running at
   report-assembly time).
7. The exact fleet candidate binary never itself ran G-parity (bracketing
   snapshots green; per-cell cross-arm checksums on the fleet cover result
   parity for the exact binaries).
8. Loss mechanism (i) is UNSETTLED — the named per-IP counter run has not
   been executed.
9. The +1.97% binary-size figure was measured against a docs-only-adjacent
   snapshot (4,748,738,592 bytes) rather than the MANIFEST byte size of
   `7e64a6cf4d5` (4,748,738,944); the 352-byte discrepancy is noted, not
   material.

## Errata (from FINAL VERIFICATION, non-blocking; raws are authoritative)

Recorded verbatim in `VERIFICATION_FINAL.md`: (A) the loss-attribution
table's shard column is wrong for 9/20 rows (the per-family tables and the
raw JSONLs agree; diffs/verdicts/mechanisms unaffected); (B) the phrase
"wins are carried by the claimed phase event" holds for 29 of 30 WINs —
`lcstr:probe.inner_all.S3.T96` (−4.46%) is a wall-only win carried by the
build phase (its `ProbeLookup` is worse), as its own table row discloses;
(C) finding 16 went stale when the U4 worklog entry was written minutes
after assembly. Verification also closed named gap 7 by running G-parity on
the exact fleet binary: `PARITY OK (636 cases: 634 compared, 2
matched-error, 0 failed; ... force-pass: engaged 8/8+2x0 (build,probe))`.

## Fleet teardown accounting (2026-07-28T01:48:58Z)

`fleet/teardown.sh` executed rc=0 after the final verification's live
re-runs completed (full transcript: `fleet/teardown_run1.log`):

i-0aacde025557ca905	shutting-down
i-0e97d0836ce2798f2	shutting-down
i-0fe67352e5989edf7	shutting-down
i-034fe227c0c563e19	shutting-down
i-0cc94834290c24619	shutting-down
i-00dc41ff4d2cf8008	shutting-down
i-09c676297620ec10a	shutting-down
i-075f553f3fadcad69	shutting-down
all instances terminated
{
    "Return": true,
    "GroupId": "sg-0426a4e0a113e0985"
}
SG sg-0426a4e0a113e0985 deleted
i-0fe67352e5989edf7	terminated
i-034fe227c0c563e19	terminated
i-0aacde025557ca905	terminated
i-0e97d0836ce2798f2	terminated
i-09c676297620ec10a	terminated
i-075f553f3fadcad69	terminated
i-0cc94834290c24619	terminated
i-00dc41ff4d2cf8008	terminated
TEARDOWN COMPLETE 2026-07-28T01:48:58Z — copy this output into REPORT.md accounting
