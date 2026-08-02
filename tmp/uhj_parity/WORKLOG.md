# UHJ parity — worklog

## Environment
- Host: `Linux ip-172-31-5-72 7.0.0-1008-aws aarch64` (96 CPUs)
- Repo: `/mnt/ch/ClickHouse`
- Build: `build/reldeb` (RelWithDebInfo) — always, per user
- Binary: `/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse` (rebuilt at tip)
- Server: `:9101` via `tmp/uhj_parity/start_server.sh` (minimal config, no Keeper)
- Work branch: `uhj-parity` @ `d0faf9f5158`
- Preserve: `uhj-parity-preserve-20260802` @ `86e2f1b07b3`
- Foundation tip (unchanged): `unified-hash-join-foundation` @ `86e2f1b07b3`
- Note: `tmp/` is gitignored; `git add -f` when committing artifacts.

---

## Unit 0 — Branch restore

**Goal:** Preserve post-Aug-1 work; park work tip at restore SHA without moving shared foundation.

**What was done**
- User: defaults + always `build/reldeb`.
- `uhj-parity-preserve-20260802` @ `86e2f1b07b3`; `uhj-parity` @ `d0faf9f5158`.

**How verified:** U0-A equivalent green; U0-B ANCESTOR_OK=1 with 6 post-restore commits; no push.

**Verdict:** green.

---

## Unit 1 — Bidirectional root-cause

**Goal:** Catalogue hot-path diffs; classify; rank MATERIAL causes with discriminating probes.

**What was done**
- Harnesses: `bench_serial.sh`, `bench_parallel.sh`, `diff_inventory.sh`, `probe_profile.sh`.
- Rebuild clickhouse at tip; dedicated server :9101.
- U1-BASE benches (5 runs, median+stdev).
- U1-DISC Real profiler probes serial/parallel.
- Inventory: `INVENTORY.md`.
- Correctness: `diff_unified_p9101.sh`; 04658/04659 launched.

**How verified — U1-BASE**
```
# bench_serial.log JOB_EXIT=0
RESULT algo=hash wall_median_ms=272 wall_stdev_ms=101.726 cpu_median_us=159771
RESULT algo=unified_hash wall_median_ms=1839 wall_stdev_ms=130.448 cpu_median_us=1733076
# bench_parallel.log JOB_EXIT=0 threads=16
RESULT algo=parallel_hash wall_median_ms=144 wall_stdev_ms=65.7092 cpu_median_us=972678
RESULT algo=unified_hash wall_median_ms=31251 wall_stdev_ms=685.396 cpu_median_us=45971115
```

**How verified — U1-DISC**
```
# probe_summaries.txt
SUMMARY label=serial_hash total=36 mutex_related=18 insert_related=18
SUMMARY label=serial_uhj total=170 mutex_related=110 insert_related=84
SUMMARY label=parallel_phash total=42 mutex_related=10 insert_related=5
SUMMARY label=parallel_uhj total=6733 mutex_related=6674 insert_related=5815
# parallel_uhj top stack: std::lock + pthread_mutex_lock under insertFromBlockImplTypeCase
# Code: HashJoin.cpp always passes &bucket_locks → parallel_build always true
```

**How verified — U1-INV:** see `INVENTORY.md`. MATERIAL: A2/B1 (per-row locks). EXCLUDED: A1 two-level. UNSETTLED: A3, B3 pending A2/B1 fix. B2 plumbing CONFIRM avoidable.

**Correctness:** `diff_unified.log`: checks=44 mismatches=3 — all three are spill cases where `hash` throws `LIMIT_EXCEEDED` (grace buckets) and UHJ returns rows. NON_SPILL_MISMATCHES=0.

**Plan changes:** Primary fix target for Unit 2 = align insert locking with baselines (nullptr/`hash` serial path; ConcurrentHashJoin-style batch lock for parallel), then plumb max_threads (B2).

**Authority calls:** none new.

---

## Unit 1 — REWORK remediation (after verifier a8148942)

**Blockers addressed**
1. Fixed `probe_profile.sh` lock filter (exclude `condition_variable` / `pthread_cond`; require real lock ops). Re-ran as v2:
   ```
   serial_hash_v2  lock_in_insert=0
   serial_uhj_v2   lock_in_insert=33   (lock_ops=33, all under insert)
   parallel_phash_v2 lock_in_insert=0
   parallel_uhj_v2   lock_in_insert=5055/5929
   ```
   Retracted prior `mutex_related=110/170` claim.
2. `probe_b2.txt` now has raw RESULT lines (UHJ ~1.3s both spill settings; parallel_hash ~60ms).
3. `run_04659.sh` rewritten to avoid duplicate `--max_threads` (original test + shell_config clash).
4. Evidence refresh committed as its own commit after this remediation (freeze for re-verify).

**INVENTORY:** A2 path CONFIRM MATERIAL (magnitude partial); B1 CONFIRM MATERIAL; B2 plumbing CONFIRM / wall UNSETTLED pending B1.

---

## Unit 2 — F1 lock alignment (in progress)

**What was done**
- Cherry-picked `2f1efe4c617` (batch bucket lock / Arena& insert like HashJoin+ConcurrentHashJoin).
- Build JOB_EXIT=0; server restarted.

**F1-alone gates** (`bench_*_f1.log`)
```
SERIAL: hash wall_median=268 cpu=160021; uhj wall_median=254 cpu=157107  → GREEN (uhj ≤ hash)
PARALLEL t16: phash wall_median=121 cpu=915184; uhj wall_median=867 cpu=729459
  → RED wall (uhj 867 > 121+noise); CPU already better
```

**Plan change:** apply F1b (`4ea37fd` bucket sizing + F2 max_threads plumb) and `ce6d1d` stagger before re-measuring parallel; do not claim U2-PARALLEL on F1 alone.

---

# U3 — divergence reduction excluding compulsory TwoLevel

## U3 / Unit 1 — Exhaustive divergence inventory

**Goal:** classify every UHJ-vs-baseline divergence; leave no unlabeled avoidable path.
Performance is explicitly *not* an oracle in this mission.

**Starting state re-verified (not trusted from prior report)**
```
$ git status && git log --oneline -3
On branch uhj-parity
nothing to commit, working tree clean
13ec290c6c6 uhj_parity Unit 2: REPORT, gate logs, stop-criterion for two-level CPU
f0420d93d31 Align UnifiedHashJoin build locks and threading with hash/parallel_hash
5b659f8f24a uhj_parity: record REWORK remediation status in U1_VERIFY.md
$ git branch -v | grep preserve
  uhj-parity-preserve-20260802  86e2f1b07b3 Add buffer size management for HashTable and UnifiedHashJoin
```
Tip matches the briefed `13ec290c6c6` / `f0420d93d31`; preserve branch intact.

**Method change vs prior units.** The prior inventory (`INVENTORY.md`) listed 7 items (A1-A4, B1-B3)
found by profiling hot paths. That is a sampling method and cannot support a "zero avoidable
divergence" claim. Replaced with a mechanical whole-directory diff, since `UnifiedHashJoin` is a
fork of `HashJoin` and therefore fully enumerable:

```
$ diff -ru src/Interpreters/HashJoin src/Interpreters/UnifiedHashJoin > tmp/uhj_parity/U3_rawdiff.txt
$ wc -l tmp/uhj_parity/U3_rawdiff.txt
3420 tmp/uhj_parity/U3_rawdiff.txt
```

3420 lines is mostly the mechanical fork transformation. `tmp/uhj_parity/U3_normdiff.sh` strips the
`namespace Unified` wrapper, the include-path rewrites and the `Unified::` qualifiers, then re-diffs:

```
$ ./tmp/uhj_parity/U3_normdiff.sh
--- residual per-file changed-line counts ---
   626 HashJoin.cpp
   364 HashJoin.h
   152 HashJoinMethodsImpl.h
   148 KeyGetter.h
   108 JoinUsedFlags.h
    66 HashJoinMethods.h
TOTAL_RESIDUAL_LINES=1464
```

36 of 42 forked files are byte-identical modulo the wrapper -> they contain no divergence at all.
This is what makes the inventory exhaustive rather than sampled.

**Claims checked directly rather than taken from subagents** (both plumbing and HashJoin.cpp
catalogues were produced by explore subagents; every load-bearing row below was re-verified by me):

1. `allOffsetFlagsSet` — baseline defines it at `JoinUsedFlags.h:261` and its only use is
   `HashJoin.cpp:1231`, inside `updateNonJoinedRowsStatus`, whose only caller is
   `ConcurrentHashJoin.cpp:569`. `hash` never reaches it. -> removal loses no baseline behavior. EXCLUDED (E15).
2. `prepareRightBlock` — subagent flagged `GraceHashJoin.cpp:759` hardcoding the baseline static for
   the Unified kind as a possible correctness risk. Diffed the two statics:
   ```
   $ diff -u <(sed -n '/^Block HashJoin::prepareRightBlock(const Block & block, const Block & saved_block_sample_)/,/^}/p' src/Interpreters/HashJoin/HashJoin.cpp) \
             <(sed -n '/^Block HashJoin::prepareRightBlock(const Block & block, const Block & saved_block_sample_)/,/^}/p' src/Interpreters/UnifiedHashJoin/HashJoin.cpp)
   IDENTICAL
   ```
   -> no behavioral divergence today. Downgraded MATERIAL -> LEAD (L1). **Null result recorded.**
3. `ExpressionAnalyzer` arg parity — subagent's table implied unified gets weaker args than `hash`
   there. Read `ExpressionAnalyzer.cpp:1085-1092`: `hash` is constructed as
   `HashJoin(analyzed_join, right_sample_block)` (defaults `any_take_last_row=false`, stats `{}`),
   `parallel_hash` gets `StatsCollectingParams{}`, unified gets `false` + `StatsCollectingParams{}`.
   All three identical. -> **REFUTED, null result.** Not an inventory row.
4. Macro renames — the two renames are not equivalent:
   ```
   $ rg -n "KEYGETTER_RANGE_IMPL" src/Interpreters/HashJoin/KeyGetter.h | tail -1
   284:#undef KEYGETTER_RANGE_IMPL          # undef'd -> rename is gratuitous  -> MATERIAL M4
   $ rg -n "APPLY_FOR_JOIN_VARIANTS" src/ | grep -v UnifiedHashJoin | wc -l
   13                                       # not undef'd, used from 4 other TUs -> rename required -> FORK-MECHANICAL F1
   ```
5. M1 (the flagship item) — verified by reading `ConcurrentHashJoin.cpp:525-560` and
   `HashJoin.cpp:1513-1521` directly. CHJ's parallel non-joined path has an explicit
   *two-level* branch (`:555-560`, comment "Two-level maps: partition buckets across pipeline
   streams") that delegates to the baseline's 5-arg `HashJoin::getNonJoinedBlocks`. So the baseline
   implements parallel non-joined processing **on top of** two-level maps. UHJ's maps are always
   two-level yet it overrides neither `supportParallelNonJoinedBlocksProcessing` (inherits `false`
   from `IJoin.h:158`) nor the 5-arg overload (inherits the partition-ignoring default at
   `IJoin.h:170`). TwoLevel is the *enabler* here, not the cause of the gap -> MATERIAL, not EXCLUDED.

**Classification-scheme decision (authority call, raised to user).** Two rows (macro rename required
to avoid redefinition clash; `getName()` string) are neither TwoLevel-required nor removable without
deleting the fork. Folding them into EXCLUDED would be exactly the reclassification the mission
bans, so they are carried in a separate, explicitly labeled FORK-MECHANICAL bucket and excluded from
the `AVOIDABLE_MATERIAL` count. Flagged to the user for confirmation rather than decided silently.

**Outcome:** `INVENTORY_U3.md`. `AVOIDABLE_MATERIAL=6` (M1-M6), EXCLUDED 17 groups / 44 regions,
FORK-MECHANICAL 2, LEAD 4, UNSETTLED 0. Prior A3 (per-row `getBufferSizeInBytes`) is resolved: it no
longer exists — byte accounting is now the `bucket_bytes` running sum (E12). Prior A4
(`supportParallelJoin`) is resolved as EXCLUDED/baseline-faithful (E14): it matches
`ConcurrentHashJoin`, which the mission names as the parallel-path baseline.

**Plan change:** Unit 2 gated on three user decisions recorded in `INVENTORY_U3.md` "Open questions"
(M1 scope, M5 align direction, FORK-MECHANICAL bucket). Asked and waiting; no implementing edit made.

**Non-gates:** no bench was run in this unit and none is required by it.
