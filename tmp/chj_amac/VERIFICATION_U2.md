# VERIFICATION_U2 — independent verification of Unit 2 (adversarial checkpoint)

Verifier: independent agent (did none of the Unit 2 work). Date: 2026-07-27 (UTC).
Scope: product commits `844ee1a82dd`, `60b8d1684a1`, `7e64a6cf4d5` + harness/hygiene
commits `99986daeab6`, `3be337d9d24`, `bc5d7a9524f`, and their recorded gates.
Method: assume wrong, refute from raw evidence; every finding below carries the
command that reproduces it. Verifier scratch outputs: `tmp/verify_u2/` (repo tmp,
outside the evidence tree) and `tmp/chj_amac/tmp/verify_*` (namespaced additions;
no recorded evidence file was modified). Re-running the parity harness regenerates
its transient outputs (`parity/logs/`, `parity/scratch/engage_out.txt`,
`parity/cases.jsonl`) — these are harness-managed per-run artifacts, and the
recorded `gate_*.log` files were not touched.

## Check 1 — PREREG ordering: PASS

Command: `git merge-base --is-ancestor <prereg> <impl>` + `git show -s --format='%h %ci %s'`
plus content check `git show <prereg>:tmp/chj_amac/PREREG.md | grep -c PREREG-<n>`.

| prereg commit | impl commit | ancestor | prereg ts | impl ts | entry present / next entry absent |
|---|---|---|---|---|---|
| 8cf071c284f (PREREG-004) | 844ee1a82dd | OK | 18:56:36 | 19:03:59 | 004=1, 005=0 |
| 8ce8b831401 (PREREG-005) | 60b8d1684a1 | OK | 19:11:46 | 19:46:47 | 005=1, 006=0 |
| 4a32708e08a (PREREG-006) | 7e64a6cf4d5 | OK | 19:49:31 | 21:01:40 | 006=1 |

Each PREREG entry was committed before its implementing commit, and each prereg
commit contains its own entry but not the next one (no back-fill).

## Check 2 — Gate power: PASS

(a) gtests re-run by verifier (binary `build/reldeb/src/unit_tests_dbms`, mtime
20:28 — built from the final U2.3 draft source, predates only the hygiene
commit's comment-level header changes):

    ./build/reldeb/src/unit_tests_dbms --gtest_filter='*Amac*'          -> rc=0, "[  PASSED  ] 4 tests."
    CLICKHOUSE_JOIN_AMAC=0 ./build/reldeb/src/unit_tests_dbms --gtest_filter='*Amac*' -> rc=0, "[  PASSED  ] 4 tests."

Logs: `tmp/chj_amac/tmp/verify_gtest_default.log`, `verify_gtest_off.log`. The 4
tests match PREREG-006(b)'s three specified tests + the documented regression
addition.

(b) negative proof, re-established from scratch:
- `grep -ac` on the immutable snapshots: baseline `a05f3ee81ff` = 0/0/0 for
  `ConcurrentHashJoinAmacBuildRows` / `...RingGrowths` / `...AmacProbeRows`;
  pre-ring candidate `60b8d1684a1` = 0/0/0; ring candidate `7e64a6cf4d5` = 3/3/0
  (probe counter correctly absent until Unit 3).
- Recorded log `parity/gate_selftest_absence.log` says exactly what SELFTEST §10(c)
  claims (GATE FAILURE line, `PARITY FAIL (0 divergences, 1 gate failure(s); ...)`).
- RE-RUN by verifier (ports were busy behind a live hygiene-gate run; scheduled after):
  `run_parity.sh <baseline> <baseline> --allow-identical --require-engagement`
  -> rc=1, `GATE FAILURE: --require-engagement given but the candidate binary has
  NO AMAC side counters ...`, `PARITY FAIL (0 divergences, 1 gate failure(s); 636
  cases: 634 compared, 2 matched-error, 0 failed; ...)`.
  Log: `tmp/verify_u2/parity_absence_rerun.log`.

(c) drain-bug teeth: `src/Interpreters/tests/gtest_concurrent_hash_join_amac.cpp:324`
(`TinySectionGrowthDuringSweepTail`) encodes exactly the growth-during-sweep-tail
scenario: `build_block_rows=256` (tiny sections), `EXPECT_GT(amac_ring_growths, 0)`
(the tail window must be exercised), uint + string arms, exact multiset equality.
U23_DRAFT_NOTES records the teeth run (rc=134 against the ported buggy drain);
`build/reldeb/gtest_amac_teeth.log` (292 bytes) ends mid-`[ RUN ]` with no result
line — consistent with SIGABRT (134 = 128+6). The build log of the buggy variant
(`build_unit_tests_teeth.log`) exists alongside.

## Check 3 — Perf claims vs raw data: PASS (with findings F2, F5)

Recomputed with the verifier's own script (`tmp/chj_amac/tmp/verify_medians.py`,
medians over valid runs; does not import fleet_ab.py):

(i) `routefix_ab.jsonl` (arm shas match MANIFEST: A=a1e71812/75d431b1d74,
B=43ef2b74/844ee1a82dd; 10 valid runs/arm):
- `key64:probe.inner_all.S3.T96`: wall −24.30% (claimed −24.3), BuildInsert −38.07%,
  ProbeLookup −57.21%, ProbeDispatch +2.04% (flat). Wall spreads ≤2.7%.
- `str:probe.inner_all.S3.T96`: wall −46.75% (claimed −46.8), BuildInsert −60.54%,
  ProbeLookup −65.35%, ProbeDispatch −5.40% (roughly flat).
- `key64:build...S3.T96`: all rows `valid=false`, reason
  `below-duration-floor (arm A median 135.7 ms < 200 ms)` — matches the WORKLOG's
  "formally INVALID" disposition verbatim; raw direction 135.4→88.5 ms as recorded.
Wins are attributed to exactly the claimed phase events; fail-closed floor worked.

(ii) `amacbuild_ab2.jsonl` (A=4b1c6744/60b8d1684a1, B=9166ec8d/uncommitted ring):
- `str:build.inner_all.S5.T96`: BuildInsert −20.20% (claimed −20.2), wall −15.57%.
- `key64:build.inner_all.S4.T1`: BuildInsert −12.22% (claimed −12.2), wall −7.75%;
  distributions cleanly separated except one cold first run in B (trimmed pstdev
  1.0–1.6%) — the effect is real.
- `key64:build.inner_all.S4.T96`: BuildInsert +0.05%, wall −0.28% — FLAT, and the
  WORKLOG records it verbatim as "PARTIAL REFUTATION ... the pre-registered
  refutation clause fired", with the prereg-mandated codegen checklist run before
  accepting the contention mechanism. Not spun as a win.
- Engagement (raw `engagement` field, arm B rows): S4.T96 ≈ 91.77M/96M rows every
  run; S5.T96 str ≈ 190.90M/192M; S4.T1 = 95,942,656/96M constant across 10 runs;
  `S2.T96` = 0 in all 10 runs (disengage proof). Arm A rows have no engagement
  field (its binary predates the counters) — consistent.
- The S2 cell's timing rows are floor-invalid (26 ms medians) as designed; only its
  counters are used, which is sound (engagement is orthogonal to timing validity).
- str build S4→S5 substitution between `amacbuild_ab.jsonl` and `ab2` is
  evidence-backed (S4 tripped the 200 ms floor in the first run) and pre-authorized
  by MATRIX.md caveat 6 ("S3/S5 coverage substitutes"). See F3.

(iii) LOCAL ORIENTATION labeling: WORKLOG U2.1 — "These are LOCAL ORIENTATION
numbers; acceptance comes from the Unit-4 fleet vs the baseline"; U2.3(c) — "final
disposition at fleet scale in Unit 4". Present as claimed.

## Check 4 — Parity: PASS (with finding F1, closed)

`parity/gate_amacbuild2.log` final line (verified byte-for-byte):
`PARITY OK (636 cases: 634 compared, 2 matched-error, 0 failed; 10 families, 23
kind-strictness combos, force-pass: engaged 8/8+2x0 (build))`. The 2 matched-errors
are `key8.left.any.nonequi.jun0.t4` and `lcstr.left.any.nonequi.jun0.t4`, both
`Code: 49 ... JoiningTransform (LOGICAL_ERROR)`, raised IDENTICALLY on both arms —
exactly the pre-existing `ANY LEFT JOIN` + non-equi residual product bug documented
in WORKLOG U1.3 and SELFTEST §5; within the ≤4 budget.

F1 (found, then closed): the gate's candidate was `uncommitted-amacbuild.tmp.bin`
(sha 9166ec8d), not the committed snapshot (4b4935fd). Closed three ways:
1. Byte-level equivalence: same size (4,748,738,944); 9,574,882 differing bytes are
   the build-id note (first diff at 0x32c), `.rodata`/dynamic-section layout shift,
   and 453,321 `.text` instruction words that differ ONLY in address-materialization
   fields (`adrp` page immediates — the 2,817 "opcode-class" mismatches are all
   ADRP-vs-ADRP whose immlo bits live in the top byte — plus `add`/`ldr` immediates).
   Same instruction stream, relocated constants (relink at a different HEAD).
2. The anchor symbol sits at the identical address+size in both (0x140f4bc0/0xc8c).
3. Decisive: verifier RE-RAN the full gate against the COMMITTED snapshot:
   `run_parity.sh <baseline> bins/clickhouse-candidate-7e64a6cf4d5.bin --require-engagement`
   -> rc=0, `PARITY OK (636 cases: 634 compared, 2 matched-error, 0 failed; ...,
   force-pass: engaged 8/8+2x0 (build))`, with the harness's runtime
   `/proc/<pid>/exe` check confirming sha 4b4935fd. Log:
   `tmp/verify_u2/parity_committed_7e64a6c_try2.log`.
   (First attempt collided with a concurrently launched parity run — see note N3;
   the harness failed CLOSED on the identity check, which is itself gate-teeth
   evidence.)

Same pattern upstream: `gate_routefix.log` ran against the live build path
(sha 347d4c97, never archived) and `gate_cursorlayer2.log` against
`uncommitted-cursorlayer.tmp.bin` (2d9a0113, sha verified on disk). Mitigated by
`gate_postharnessfix.log`: PARITY OK against the COMMITTED
`clickhouse-candidate-60b8d1684a1.bin` snapshot (4b1c6744), which contains both
earlier product commits — and transitively by the re-run above at 7e64a6cf4d5.
The WORKLOG itself records the process error and adopts the immutable-snapshot
rule mid-unit (U2.2 "PROCESS ERROR").

Also verified during this checkpoint: the post-U2 hygiene commit `bc5d7a9524f`
(touches `HashJoin.h`/`ResumableHashMap.h`) finished its own gate green while
verification ran: `gate_hygiene60b.log` -> `PARITY OK (... force-pass: engaged
8/8+2x0 (build))` against `uncommitted-hygiene60b.tmp.bin` (bca06d3b).

## Check 5 — Disasm spot-check: PASS

Verifier's own toolchain run on the COMMITTED snapshot (not the report's binary):

    /usr/local/bin/llvm-nm-22 --defined-only --print-size --demangle bins/clickhouse-candidate-7e64a6cf4d5.bin | grep amacBuildInsert ...
    -> key64/RowRefList, ResumableHashMap<HashMapTable<unsigned long, HashMapCell<..., RowRefList, HashCRC32<...>>, ..., TailPaddedHashTableGrower<8>, ...>, true>
       lambda operator()<unsigned int> @ 0x140f4bc0 size 0xc8c   (same addr/size the report cites)
    /usr/local/bin/llvm-objdump-22 -d --start-address=0x140f4bc0 --stop-address=0x140f584c ...

Results (raw in `tmp/chj_amac/tmp/verify_anchor1.asm`):
- `prfm` count in the symbol = 10, ALL `pstl1keep` (write-intent, L1, keep), all
  `[x26, xN]` (register-resident cells base). `pldl3keep` occurrences = 0.
- Steady-loop advance at 0x140f4f80..0x140f4fa4 matches the report listing
  instruction-for-instruction: `ldr x8,[x19,#0x58]` (grower `precalculated_buf_size`),
  `add/cmp/csinc x8,xzr,x9,eq` (tail-pad wrap), `str w8` (ring pos), `lsl #4`,
  `prfm pstl1keep,[x26,x9]`, `add x28,#1; cmp x28,#0x20; b.ne` back-edge.
The report's `G-DISASM-BUILD: PASS (0 unexplained)` is consistent with what the
verifier sees on the anchor checked.

Bonus (PREREG-005 codegen-diff reproduction, `asmdiff.py` re-run on the archived
pre/post binaries, raw `tmp/verify_u2/asmdiff_rerun.log`): opcode delta set
reproduces exactly (INSERT: ldr+4 cmp+3 csinc+2 and−1 cbz+1 b.eq+1 mov−1 ldrb−1;
PROBE: and−2 cmp+2 csinc+2 ldr+3 mov/ldrb/lsl−1, +14 alignment nops) and stores
are flat (55→55, 69→69) — the walk-advance-only claim holds. See F4 for the
absolute-count mismatch.

## Check 6 — Hash guard: PASS

Recomputed medians (same script):
- `cursorlayer_hash_ab.jsonl` (A=43ef2b74 pre-grower, B=2d9a0113 cursor layer):
  key64 S3.T1 `hash` +0.63% (tight, arm-A max-min spread 1.27%); T96 cells
  −6.99% (str) / +15.13% (k256) / −24.83% (key64) — all match WORKLOG.
- `hash_t96_aa.jsonl` (SAME binary 43ef2b74 both arms): key64 T96 `hash` A/A diff
  −14.12%; harness's own A/A verdicts TIE with pstdev bands 12.0% / 17.5%
  (`fleet/hash_t96_aa.log`) — the T96 `hash` shapes are demonstrably jitter-bound
  on this host, so "locally unresolvable" is supported by same-binary data.
- k256 "+15.13% TIE (band 14.3%)" is legitimate under the pre-registered verdict
  rule: `band_abs = band_frac * max(medA, medB)` (diff 223 ms ≤ band 243 ms). That
  band semantics is unchanged since the original harness commit `91469b6b22e`
  (checked via `git show <commit>:tmp/chj_amac/fleet_ab.py` at all three revisions);
  the hygiene "band-units" fix made bands stricter, not weaker. Not a post-hoc rule
  change; PREREG-005's refutation criterion (a `hash` cell LOSING outside band) did
  not fire.
- WORKLOG defers the in-band acceptance to the Unit-4 fleet G-hash-inband gate in
  so many words ("settled by the fleet G-hash-inband gate ... in Unit 4") rather
  than claiming it settled. Verified verbatim.

## Check 7 — Hunt results

Evidence-hash spot checks (7 checked, all match): `cursorlayer_hash_ab.jsonl`
(9e9906f2828837da...), `hash_t96_aa.jsonl` (f7599f895794486f...),
`gate_002a_run1.log` (15249683fa16...092d8adc), `noise_band_002c.jsonl`
(3dd917202db9...), `noise_band_002c_rev1.jsonl` (1acb6a2cccef...), MANIFEST rows
for `clickhouse-baseline-a05f3ee81ff.bin` (0d32ef1c...) and
`clickhouse-candidate-7e64a6cf4d5.bin` (4b4935fd...); plus the two uncommitted
snapshots match their recorded shas (2d9a0113..., 9166ec8d...).

Gates can fail (not vacuous): two real FAIL logs on disk
(`gate_amacbuild.log` — `PARITY FAIL (1 divergences, ...)`, the recorded initial
engagement-detection failure; `gate_selftest_absence.log` — `PARITY FAIL` with a
named gate failure); injected-divergence selftests recorded in SELFTEST (3 injected
→ 3 reported → PARITY FAIL; matched-error budget 14>4 → FAIL; mid-run kill → FAIL)
with scripts present in `parity/scratch/`; and the verifier's own two fresh FAILs
(absence re-run rc=1; the port-collision run FATALed closed on the server-identity
check).

## Discrepancies (none blocking)

- F1 (closed): U2 parity/perf gates ran against uncommitted/live binaries rather
  than the committed snapshots (`gate_routefix` live path sha 347d4c97 unarchived;
  `gate_cursorlayer2` and `gate_amacbuild2` against `uncommitted-*.tmp.bin`).
  Closed by binary-equivalence analysis + `gate_postharnessfix` (committed
  60b8d1684a1) + the verifier's fresh `PARITY OK` + engagement on the committed
  `7e64a6cf4d5` snapshot. Residual nit: the routefix gate binary is unarchived, so
  its equivalence is only transitively established.
- F2 (wording): WORKLOG U2.3(c) calls key64 S4.T1 a "wall win"; under the
  harness's own verdict rule it is TIE (wall −7.75% vs pstdev band 9.2%, band
  inflated by one cold run). The BuildInsert −12.2% claim is real and cleanly
  separated. Fix the wording before it propagates into REPORT.md; no gate outcome
  changes (PREREG-006 requires only wall-not-regressing).
- F3 (cross-reference nit): the str build S4→S5 cell substitution in ab2 and the
  unverdictable-by-floor status of "S2 stays in-band wall" are evidence-backed and
  pre-authorized by MATRIX caveat 6, but WORKLOG U2.3 does not explicitly narrate
  either disposition.
- F4 (reproducibility nit): WORKLOG U2.2's absolute instruction counts
  (590→598 / 747→749) do not reproduce with the current `asmdiff.py`
  (626→634 / 855→871 incl. nops) — the script was modified afterwards by the
  hygiene commit and the original listings were not archived. The load-bearing
  facts (delta opcode set, stores flat) reproduce exactly.
- F5 (weight): str S5.T96 BuildInsert −20.2% is comparable to its per-arm spread
  (trimmed pstdev 14–18%) on this noisy host; directionally consistent (9/10 B runs
  below the A median) and correctly deferred to the fleet — weight it lightly until
  Unit 4.
- F6 (process nit): PREREG-005's invocation names `analyze-assembly.py`; the diff
  actually shipped with the custom `asmdiff.py` (tool substitution disclosed in
  WORKLOG but not flagged as a prereg deviation).

Environmental notes:
- N1: `unit_tests_dbms` on disk predates the hygiene commit by minutes; the hygiene
  changes to product headers are comment/format-level (verified via `git show
  bc5d7a9524f`), and its parity gate is green.
- N2: verifier re-runs regenerated the parity harness's transient outputs
  (`parity/logs/`, `scratch/engage_out.txt`, `cases.jsonl`); recorded gate logs
  untouched. A leftover baseline server from the collided run (pid 534302, cwd
  `parity/srv_base`, exe = baseline bin) was stopped by the verifier.
- N3: a second `run_parity.sh` instance was launched concurrently by another agent
  at ~21:29 and collided with the verifier's first re-run (shared ports/dirs; the
  harness failed closed). Recommend serializing parity runs (lockfile) before
  Unit 3's heavier gate traffic.

## Verdict

All seven mandated checks pass on raw evidence; every material claim reproduced
under independent recomputation; the one gate-provenance gap (F1) was closed by a
fresh green run against the committed snapshot; remaining findings are wording,
cross-referencing, and archival nits that do not change any gate outcome.

U2 VERIFICATION: SHIP
