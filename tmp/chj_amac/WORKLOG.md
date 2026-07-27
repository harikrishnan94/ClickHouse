# WORKLOG — AMAC build/probe + order-preserving probe for `ConcurrentHashJoin`

Mission: implement (1) AMAC rings for build-insert and probe-lookup and
(2) an order-preserving routed probe for `ConcurrentHashJoin`
(`join_algorithm = 'parallel_hash'`) on branch `phj-ph`. Baseline arm for all
comparisons: branch `concurrent-hash-join-profile-events` @ `a05f3ee81ff`.
Approved plan: see `PREREG.md` header and the per-unit entries below.

Conventions: entries are per iteration (goal / done / verified-how / plan
changes), amended forward, UTC dates. Raw outputs inline when small, else
path + sha256. Secrets redacted as [REDACTED].

---

## 2026-07-27 — Session start, planning phase

Goal: produce the approved implementation plan.

Done:
- 3 Explore agents (phj-ph code state; `ahj` reference design; history docs +
  environment) and 3 Plan agents (routed probe; AMAC rings; harness/fleet/
  process). Findings recorded in the approved plan file
  (`/home/ubuntu/.claude/plans/mission-amac-zany-hammock.md`).
- Verified myself (not delegated): (a) `hashToSelector` on `phj-ph` routes by
  `hashes[i] & (num_shards - 1)` (ConcurrentHashJoin.cpp:522-530) — the same
  low bits `HashTableGrowerWithPrecalculation::place` uses (HashTable.h:301)
  → per-slot home-cell clustering; the two-level baseline routed by
  `getBucketFromHash` = bits 24..31 (TwoLevelHashTable.h:54); (b) upstream
  `master` shares one `StoredColumnsIndex` across slots
  (master ConcurrentHashJoin.cpp:232-237) — revert `69bf5c26c9f` dropped it.

Requester decisions recorded:
1. Fleet = launch 8× m8g.24xlarge (Dev_AWS_Admin, ap-south-2), terminate
   after the campaign.
2. AMAC diagnostic hook = C++-only (env `CLICKHOUSE_JOIN_AMAC` off/auto/force
   + gtest setters). No public setting. Consequence: no stateless SQL test
   for off/force paths; gtests carry off-path coverage.
3. Unit order = AMAC first; order fix lands with the AMAC probe (its
   in-order emit is the fix).
4. Join maps adopt the tail-padded grower (ring disassembly must replicate
   `ahj`); new gate G-hash-inband (12 `hash`-algorithm A/B cells).
5. New gate G-disasm: instruction-semantics equivalence of ring steady loops
   vs an `ahj`-HEAD reference binary (6 anchors).

Plan approved by requester via plan-mode approval (1 revision cycle).

## 2026-07-27 — U1.1: evidence skeleton

Goal: create `tmp/chj_amac/` skeleton (this file, PREREG.md, .gitignore) and
commit it.

Done: directories {bins, parity, order, hygiene, fleet/{ssh,results},
tests_srv}; .gitignore excluding binaries/keys/server dirs/logs; this
worklog; PREREG.md with U1 pre-registration.

Verified: `git status --porcelain` clean before staging (raw: empty output);
HEAD `6cdee22a4554e3935e26165837dedb2b3eb2362a`;
`concurrent-hash-join-profile-events` = `a05f3ee81ff8411759637fa367aad62e72726e71`;
`ahj` = `cf465cfbe23a14f982d1bc36510f3e311ce6379f`.

Deviation: the per-commit hygiene loop's G-build/G-parity re-run
does not apply to this commit — no source changed and the parity harness does
not exist yet (it is U1.5's deliverable). Hygiene report subagents still run.

Interpretation recorded: hygiene-pass commits themselves are terminal — the
loop does not recurse onto them (their content is exactly the reports plus the
accepted fixes). Tooling, docs, test, and code commits all get the loop.

## 2026-07-27 — U1.2: reference builds (baseline + `ahj`)

Goal: build the two comparison-arm binaries per PREREG-001.

Done:
- Commits `7708ef69e8e` (hygiene pass for b159a96) and `d2a759e684f`
  (`worktree_setup.sh` + `build_refs.sh`).
- Baseline worktree `/mnt/ch/ClickHouse-concurrent-hash-join-profile-events`
  @ `a05f3ee81ff` (hardlinked submodules, create-worktree skill recipe);
  full RelWithDebInfo clang-22 build.
- Baseline binary snapshotted:
  `bins/clickhouse-baseline-a05f3ee81ff.bin`
  sha256 `0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4`
  (MANIFEST.tsv row; identity of record).
- G-build (baseline): PASS — subagent analysis of
  `build_baseline-a05f3ee81ff.log`: 0 errors, 0 warnings, single configure,
  Clang 22.1.8 aarch64 RelWithDebInfo, link OK (15,942 compile edges — full
  clean build).
- `ahj` worktree `/mnt/ch/ClickHouse-ahj` @ `cf465cfbe23` ready; build
  running (log `build_ahj-cf465cfbe23.log`).

Hygiene reports for `d2a759e684f`: reduce = essentially clean (restore two
rationale comments dropped from the skill transcription); humanize = 4
actionable (missing `mkdir -p` for `bins/`, partial-worktree resume gap,
garbled recovery message, manifest last-row-wins ambiguity) + 2 commit-message
nits (unfixable — amending is banned). Fixer deliberately DELAYED until
`build_refs.sh` finishes executing: bash reads scripts incrementally, and
editing a mid-run script corrupts its parse.

- `ahj` reference binary snapshotted: `bins/clickhouse-ahj-cf465cfbe23.bin`
  sha256 `c8260c682b78ea7cd9beb789b9d517d7c4d810ea73f131b6e31fc482dbf36f6e`
  (MANIFEST.tsv row). G-build (`ahj`): PASS — subagent analysis of
  `build_ahj-cf465cfbe23.log`: 0 errors, 0 warnings, full 16,673-edge build,
  clean configure, link OK. PREREG-001 is satisfied for both reference arms.
- Disk note: /mnt/ch at 83G free after both build trees; worktree object
  dirs are reclaimable once G-disasm no longer needs in-tree artifacts (the
  snapshot binaries in `bins/` carry the debug info).

## 2026-07-27 — U1.3: harness suite built, adversarially reviewed, fixes in flight

Goal: build the parity/order/fleet harnesses (workflow: 3 builders + 3
adversarial reviewers), then fix what the reviews found.

Done (builders, all self-tested):
- `parity/`: 636-case generator + driver + gate script. Full self-run
  `PARITY OK (636 cases, 10 families, 23 kind-strictness combos)` in 2m27s;
  baseline-as-both-arms run engaged `parallel_hash` 10/10 families.
- `order/`: Native-stream order oracle + 17-check gate. Power check PROVEN:
  scatter binary FAILS per-block at `max_threads=96`, PASSES `--global` at
  T=1; 03448 fails 10/10 with a join row-order flip (U1 identity RESOLVED —
  it exercises join order under stock defaults); 03711 fails 10/10 in its
  `parallel_hash` sections. Deviation: `min_joined_block_size_*=0` has no
  power on the scatter branch (per-bucket blocks are trivially ordered) —
  `_squash` variants added.
- `fleet_ab.py` + `fleet/{matrix_gen,check_matrix}.py`: two-arm ABAB driver,
  1800-cell matrix, LPT plan. A/A cell TIE; deliberate A≠B verdicts LOSS;
  `check-events` 7/7 on baseline. Empirical deviation: `ProbeDispatch` can be
  0 (two-level skips scatter at low T) and `BuildDispatch` 0 at T1 → asserted
  set = Build+BuildInsert always, Probe+ProbeLookup probe-side; dispatch/
  merge sub-phases record-only.
- PRODUCT BUG (pre-existing, NOT phj-ph): `ANY LEFT JOIN` + non-equi residual
  ON + right key projected + extra unprojected column + heavy duplication →
  `Code: 49` exception `Invalid number of rows in Chunk ... JoiningTransform`
  on baseline AND phj-ph, `hash` AND `parallel_hash`, even `max_threads=1`.
  Repro: `parity/SELFTEST.md` §5. Affects 2/636 parity cases (classified
  matched-error, loud warning). ACTION: report upstream (flag to requester).
- STALE-BINARY TRAP caught by fleet selftest: `build/reldeb` predated
  `a05f3ee81ff` (zero `ConcurrentHashJoin*` events). Fixed by re-cmake +
  rebuild: candidate snapshot `bins/clickhouse-candidate-75d431b1d74.bin`
  sha256 `a1e71812b0ab1587dd4dc966a898fa36542804dee6d919276a55c83cc711fefc`.
  G-build (candidate): PASS — 87-edge incremental (exactly the missing
  commits' TUs: `GitHash.generated`, `ProfileEvents.cpp`,
  `ConcurrentHashJoin.cpp`, 30 HashJoin TUs), 0 errors/warnings; log read
  directly (89 lines — deviation from the subagent rule, stronger than
  delegation at this size). Event presence verified: `grep -ac
  ConcurrentHashJoinProbeLookupMicroseconds` = 3 on candidate AND baseline
  snapshots (plain `grep -c` without `-a` returns 0 on ELF — recorded as an
  evidence-recipe pitfall).

Order fixer COMPLETE (all 7 findings fixed in `run_order.sh`, re-proven on
the baseline snapshot): `query_plan_convert_outer_join_to_inner_join=0`
pinned + per-run EXPLAIN assertion of `Type: right`/`Type: full` (the scoped
checks now genuinely exercise RIGHT/FULL and still FAIL per-block at T=96 —
40/19 violations, rows matching controls); power verdict requires engaged +
row-matched + `N_ERROR==0`; controls fail closed; stateless engagement gates
the verdict via query_log deltas. Facts discovered: (1)
`ConcurrentHashJoinProbeMicroseconds` > 0 at `max_threads=1` on the baseline
(494216 µs; Probe is the per-slot timer) — T=1 engagement assertion stands;
(2) HONEST DEVIATION vs PREREG-002b's stateless sub-expectation: the
baseline PASSES 03448/03711 10/10 (its two-level probe reorders only at
harness scale/squash — 8 per-block FAILs at T=96), while the scatter
candidate FAILS them 10/10. The per-block power check therefore fails on
BOTH arms (PREREG-002b's core criterion holds); the stateless layer has
power specifically against the scatter probe. Raw evidence:
`order/SELFTEST.md` "Post-review fixes", `order/logs/postfix_*.log`.

Parity fixer COMPLETE (all 6 findings; re-proofs (a)-(e) + 2 extra in
`parity/SELFTEST.md` "Post-review fixes"): matched-error now only for genuine
`Code: <N>. DB::Exception:` signatures with the code preserved, budget (>4 or
>2% fails, `--acknowledge-matched-errors` to override) and a 90%-compared
floor; identical binaries FATAL without `--allow-identical`; `stop_server`
refusal aborts (pid file kept); cleanup trap kills phase-1 drivers;
force-pass detection keyed on asserted counters only; `TimeoutExpired`
handled per-case fail-close.

Fleet fixer COMPLETE (all 10 findings; re-proofs in `SELFTEST.md`):
MEASURED/INFERRED evidence must show two arms with DIFFERENT binary shas,
`threads_effective` == nominal T, one settings fingerprint per arm, ≥5 valid
runs; resume validates the same against the CURRENT invocation; poisoned
selftest rows quarantined to `fleet/results/selftest_archive/`;
`--require-engagement` applies to resumed rows; binary paths absolute;
missing results file fatal; nonce-based dedup; `INSUFFICIENT` verdict under
5 runs/arm; NULL-aware checksums everywhere; `plan` refuses shards > cells;
remote stop is PID-only (pgrep fallback deleted); hidden `--threads-override`
removed (real T4/T1 selftest cells instead).

Adversarial reviews: ALL THREE returned FIX with demonstrated holes —
parity (vacuous matched-error pass, no same-binary guard, destructive stop
path), order (RIGHT/FULL scoped checks silently plan-rewritten to
inner/left by `query_plan_convert_outer_join_to_inner_join=1` — never
tested; expect-fail ignores errors; controls fail open), fleet (MEASURED
accepts same-binary/threads-overridden rows; resume trusts mismatched rows —
poisoned selftest rows on disk; `--require-engagement` bypassed at
cells_run=0). Three fixer agents launched with the verbatim findings; each
must re-prove via self-tests against the baseline snapshot. Full review
texts: workflow journal (wf_3dc931f5-0d9).

## 2026-07-27 — U1.4: pre-registered gate runs (PREREG-002a/b/c)

- **PREREG-002a (G-parity, baseline bin vs candidate bin) — GREEN.**
  Invocation: `bash tmp/chj_amac/parity/run_parity.sh
  bins/clickhouse-baseline-a05f3ee81ff.bin
  bins/clickhouse-candidate-75d431b1d74.bin` (distinct binaries, no
  `--allow-identical`). Final line (raw): `PARITY OK (636 cases: 634
  compared, 2 matched-error, 0 failed; 10 families, 23 kind-strictness
  combos, force-pass: SKIPPED)`. The 2 matched-errors are exactly the known
  pre-existing `Code: 49` `JoiningTransform` product bug (identical on both
  arms); engagement audit: `parallel_hash` engaged 10/10 families on BOTH
  arms; AMAC force-pass correctly SKIPPED (counters do not exist until
  Unit 2). Log: `parity/gate_002a_run1.log`.
- **PREREG-002b (G-order power check, BOTH arms) — GREEN.** Invocations:
  `bash tmp/chj_amac/order/run_order.sh <bin> --expect-fail` for candidate
  then baseline. Final lines (raw, identical): `ORDER POWER-CHECK OK (check
  fails on this binary, as expected: >=1 engaged row-matched T=96 FAIL,
  errors=0, row_mismatch=0)` — each arm: 8 engaged row-matched per-block
  FAILs at T=96, 0 errors, and T=1 `--global` OK. Logs:
  `order/logs/gate_002b_{candidate,baseline}.log`.
- **PREREG-002c (noise band) — GREEN on attempt 2, after a real failure on
  attempt 1.** Same 6 cells ({key64,str,k256}×S2×T96, {key64,str}×S3×T96,
  key64×S2×T1), baseline binary both arms, calibration override
  `fleet/calibration_rows.json`.
  - Attempt 1 FAILED per the pre-registered criterion (raw final line:
    `FLEET_AB AA RESULT: cells=6 tie=5 nontie=1 -> FAIL`): key64:S2.T96
    verdicted LOSS on same-binary data. Root cause from the raw rows: the
    S2×T96 cells ran 23-42 ms medians (1M probe rows over 96 threads =
    ~10k rows/thread — scheduler jitter), far below the approved plan's
    200 ms floor, which the fleet builder had flagged as not yet enforced;
    the one ≥200 ms cell (key64:S3.T96, 228 ms) was tight at −0.17%.
  - Fix (per the prereg's action clause, before any Unit 2 commit):
    `fleet_ab.py` now enforces `PER_THREAD_MIN_PROBE_ROWS = 2_000_000` for
    probe-side cells and a fail-closed `MIN_CELL_DURATION_US = 200_000`
    per-arm median check (a too-fast cell can never verdict). MATRIX.md
    caveat 6 records the build-side consequence (small-size build cells at
    high T will trip the floor and be re-dispositioned with rationale).
  - Attempt 2 GREEN (raw): `FLEET_AB AA RESULT: cells=6 tie=6 nontie=0 ->
    PASS`; per-cell diffs −1.22%..+0.43%; medians 213-646 ms; same-binary
    spreads 0.59-1.27% ⇒ the frozen band is the 3% floor for all six shapes
    (`fleet/band_local.json`). The deliberate A≠B verdict selftest
    requirement is satisfied by the fleet fixer's re-proof (S2.T4 LOSS
    +86.61%, band 6.6% — `SELFTEST.md` "Post-review fixes").
  Evidence hashes (logs stay uncommitted per `.gitignore`; sha256 recorded):
  `parity/gate_002a_run1.log` 15249683fa16...092d8adc;
  `order/logs/gate_002b_candidate.log` b25dedb3a98b...cba6454d6;
  `order/logs/gate_002b_baseline.log` 3ec3eb283e58...245ca870;
  `fleet/gate_002c_run1.log` ffd571c9d7ae...00d90007;
  `fleet/gate_002c_run2.log` 3de7aefa64dd...df481f9449;
  `fleet/results/noise_band_002c.jsonl` 3dd917202db9...09f8b4a3c;
  `fleet/results/noise_band_002c_rev1.jsonl` 1acb6a2cccef...da1a299e.

Unit 1 exit state: PREREG-001 and PREREG-002a/b/c all green; coverage matrix
frozen (MATRIX.md + fleet/matrix.json); calibration frozen; fleet runbook
written (launch deferred until Units 2-3 pass local gates); harness suite
committed with this entry. Remaining Unit 1 residue carried forward: AMAC
force-pass success path and remote fleet mode are untestable until Unit 2 /
Unit 4 respectively (loudly SKIPPED, fail-closed under --require flags).
