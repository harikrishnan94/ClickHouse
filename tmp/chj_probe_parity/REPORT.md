# REPORT: probe-side win-or-parity for `ConcurrentHashJoin` (`phj-ph`)

Mission executed 2026-07-28 per the requester-approved DESIGN.md REV 3
(items 1-5 + 7; item 6 dropped by requester decision). Base:
`phj-ph`; baseline arm for every comparison:
`concurrent-hash-join-profile-events` @ `a05f3ee81ff` (binary
`0d32ef1c96e6...`, identity of record in `bins/MANIFEST.tsv`).

## What landed (16 commits above the U0 base)

| Item | Commits | Substance |
|---|---|---|
| 1. Dedicated route hash + single key prep | `7dfe941a6d0`+hyg `32ffc87991b`, `f8d4826722d` | `JoinSlotRouting` fold (CRC-32 ISO / multiply-shift, decorrelated from the maps' CRC-32C), UInt8 slot ids, `JoinOnKeyColumns` built once per probe block, zero-copy build scatter on narrow ids, old per-family hash dispatch deleted |
| (bonus bug fix) | `72a1e91c99e` | pre-existing `max_joined_block_size_bytes` splitting disabled by the slot-0 estimate; found by the join-selector differential this mission added (the prior mission's named gap, closed here: 893 tests, candidate failures == baseline's 119) |
| 2. Pooled per-lane scratch | `96136264c16`+hyg `3c7ecf108b8` | `IJoin::joinBlock(Block, lane)` + `JoiningTransform` stream index; one parked `JoinProbeScratch` per lane (atomic exchange/CAS, pool fallback); zero per-block allocation at steady state |
| 3+4. Once-built plan + flat descriptor loop | `561c1d21b99`, `aa98041d472`+hyg `9f40ff64257` | `RoutedProbePlan` collected at ctor + `onBuildPhaseFinish`; fused descriptor find for cheap-key cursor families below the ring gate, wrap-aware walk, descriptor-based look-ahead |
| 5. ASOF pointer-recording ring | `633cca3879a` | ring records the ASOF mapped address; phase-B `findAsof` unchanged; ASOF stays routed multi-slot |
| 7. `AmacWalk` policy | `349bafd1e1e`+hyg `d1c77571b39` | {bare, wrap_aware} axis selected per `joinBlock` from the plan's wrap bit; wrapped plans keep the ring; degenerate-hash teeth gtest |
| U5 fix cycles | `06e0bbd0aa3`, `6598f4b872f` | fleet-driven: cheap-key ASOF leaves Auto engagement (string ASOF keeps the ring); dictionary-aware LC route fold |

Compile/binary cost: `amacFindPass` 64 -> 160 symbols across the two
axes; binary +~4 MB total (+0.08%); 25 gtests.

## Evidence matrix

| Gate | Invocation | Result | Verdict |
|---|---|---|---|
| G-build | ninja per commit, logs in `build/reldeb/` | rc=0 every commit (2 transient linker bus-errors diagnosed: disk 98% + concurrent multi-GB links) | GREEN |
| G-parity | `parity/run_parity.sh` baseline vs snapshot, per commit | `PARITY OK` 636 cases x9 runs lifetime (634 compared, 2 matched-error budget, force-pass 8/8+2x0) | GREEN |
| G-order | `order/run_order.sh` w/ baseline reference | `ORDER OK` x6 runs (all engaged, t1_global OK, 03448/03711 x10) | GREEN |
| G-tests | join-selector differential (893 tests, both arms) + 25 gtests | candidate failures == baseline's 119 after the `72a1e91c99e` fix; gtests 25/25 | GREEN |
| G-disasm | llvm-objdump ranges vs ahj `c8260c682b78...` | flat-loop anchors (key64, keys256) PASS 0 unexplained; bare ring anchor opcode-identical pre/post walk axis; `hash` codegen NFC (14064 symbols 1:1) | GREEN |
| G-ablation | `ablate_ring.jsonl` (ring-off arm) | ring-off regresses k256 S3 +18.9%, str S5 +18.5%; ring helps even at the red S5 cells (+4.9/+6.7%) | GREEN |
| Boundary/force-engage | `boundary_{force,off}.jsonl` | force-at-S1/S1p5 3/3 TIE (threshold placed right); ring+flat off at S1p5 costs +5.4/+15.8%; force counters staged. Nuance: S1 auto-engages on 128-slot plans (per-slot minimum buffers aggregate past L2) | GREEN |
| A/A noise | `aa_u5.shard{0,1}.jsonl` | 10/10 probe cells TIE, diffs <=1% (3% floor holds) | GREEN |
| **G-probe-perf (must-hold)** | full sweep 8 shards + fix-cycle substitutions, `gate_verdicts.py`; independently recomputed (VERIFICATION_U5.md, zero verdict differences) | **54 WIN / 18 TIE (11 probe + 7 build-guard-OK) / 7 HONEST-RED / 17 unverdictable-at-floor / 1 INFEASIBLE-ON-FLEET** (= 97 frozen cells) | **RED (must-hold) — 72/79 verdictable cells win-or-parity; every red attributed below** |

## The gate table (headline numbers)

Wins (probe-event medians vs baseline): `k256` −5..−31% across every
size (S3 −15.9%, S5 −16.8% wall; probe −30%+), `str` family −8..−34%
(S5 probe −34%), `fixstr`/`k128`/`mixed`/`key32-64` S1-S3 all
green, `rf_all` wall −69/−73% (the baseline's giant flag-merge is
gone), every prior-campaign dispatch-family loss (mixed, fixstr,
k128, floor cells) now win or tie. The `mixed` family's serialized
route (25 thread-s at S5 in the prior campaign) is a cheap fold.

Honest-red (7 cells, all attributed, fix-paths named):

| Cell | Final diff | Attribution |
|---|---|---|
| `key64:probe.asof.S2.T96` | +3.57% (band 3.0) | ASOF route floor: the route pass + plain-loop residual on a shape whose sorted-vector search dominates; ring OFF for cheap-key ASOF was fleet-measured better (+26.6% -> +5.9% at S4) |
| `key64:probe.asof.S4.T96` | +5.89% | same; improved from the prior campaign's +15.3% wall / +9% lookup |
| `str:probe.asof.S2.T96` | +4.94% | string ASOF keeps the ring (removing it cost +11.9%); residual = route floor |
| `lcstr:probe.inner_all.S3.T96` | +18.41% probe, **wall −5.4% WIN** | the red is the LOOKUP component (LC probe path vs the merged two-level map); dispatch already fixed (dictionary-fold). Metric tension recorded: the user-facing wall wins. Named lead: LC getter cache behavior across 128 slot maps |
| `key32:probe.inner_all.S5.T96` | +12.58% | S5 DRAM residual: route pass (~1.3-2.2 thread-s) + two-phase find/emit overhead vs the zero-dispatch merged map; ring itself helps (+6.7% worse off). Arithmetic in WORKLOG shows lever 1 (fused ring->emit) cannot reach the band |
| `key64:probe.inner_all.S5.T96` | +13.30% | same (ring-off +4.9% worse) |
| `key64:probe.semi_anti.S4.T96.anti` | +16.02% | route floor on an emit-light cell: the route pass alone (0.74 thread-s) exceeds the band (0.29 s); no lever removes routing |

Unverdictable-at-floor (17): small `S2/S3` build guard cells and
`.anti`/`.h05`-class cells under the 200 ms duration floor
(fail-closed, prior-campaign precedent). Guard coverage rests on the
S5/dup16 build cells - all in-band or better (str build S5 wall
−29.9%).

INFEASIBLE-ON-FLEET (1): `lcstr:probe.inner_all.S5.T96` - the
BASELINE arm OOMs at that size on the 192 GB usable budget (the
same cell OOMed identically in the prior campaign). The verifier
caught that the orchestrator's tally had silently dropped it; the
97-cell accounting now closes: 54 + 18 + 7 + 17 + 1 = 97.

## Errata and honest notes

- The per-run `events` capture carries the seven shared events only;
  AMAC counters live in the cell-level `engagement` field.
- S1 auto-engages the ring on 128-slot plans (minimum buffers
  aggregate past L2) - the force-engage contrast is ON-vs-ON there;
  the off-arm supplies the true contrast.
- `amacRun` is compiler-outlined post walk-axis (once-per-chunk call;
  per-visit loop untouched; disasm-verified).
- S1p5 rows (96000) are a pow2-approximate 2x-L2 target.
- Fix cycle 1's blanket ASOF exclusion overcorrected for string ASOF
  (+11.9%) - refined in cycle 2; the refutation branch fired exactly
  as pre-registered.
- Local orientation A/B during development: 12/12 cells in-band; the
  key64 asof S2 local +5.2% lookup note correctly predicted the
  fleet ASOF finding.
- Verifier-required corrections (all applied here): (1) the WORKLOG
  ablation quote used WALL numbers; on the pre-registered probe
  metric ring-off regresses +37% to +115% - stronger, a fortiori;
  (2) the band source drifted from PREREG 007's A/A-letter wording
  to the implemented max(3%, arm-A spread) rule - verdict-neutral on
  every checkable cell, recorded as a deviation; (3) `lcstr` S2's
  sweep verdict binds to the PRE-fix-1 binary; fix 1 only cheapens
  the LC dispatch, so that verdict is conservative; (4) the results'
  `host` field records the controller's hostname, not the shard
  (shard identity lives in the `shard` field and ssh logs).

## Coverage boundaries

- Wrapped-chain ring plans: unreachable from SQL; covered by the
  degenerate-hash gtest (ring == `HashMapTable::find` on a
  pad-spanning wrapped chain).
- x86 route-word quality (multiply-shift arm): NOT-CLAIMED (ARM
  fleet only).
- The `mixed_on` group and `S1p5` size are harness extensions,
  smoke-validated (0 invalid) before the campaign.

## Verification and teardown

- Independent verification: VERIFICATION_U5.md (doer != grader;
  own-code recompute of all 96 measured cells from the raw JSONLs -
  ZERO verdict differences; 2,520-row validity/identity/checksum/
  ABAB audit clean; ablation and boundary recomputed on the probe
  metric; 3 on-fleet spot re-runs reproduce within 0.05-0.80 pp,
  one on a different host; PREREG 007/008/009 committed before
  their runs). VERDICT: **FIX-THEN-SHIP**, with the five
  documentation fixes applied in this report -> SHIP.
- Fleet accounting: 8x m8g.24xlarge ap-south-2 (SG
  `sg-0a32c30cee50aa10e`), launched 11:00:44Z, TEARDOWN COMPLETE
  12:54:51Z (`fleet_teardown.log`; all 8 instances terminated, SG
  deleted) - ~1 h 54 m x 8 = ~15.2 instance-hours. Work performed:
  the 97-cell sweep (10 runs x 2 arms each), 12-cell A/A, 9 aux
  cells (ablation + boundary), 9 fix-cycle re-run cells, 3 verifier
  spot re-runs.
