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

## U3 / Unit 2 — batch 1: M2, M3, M4 (independent of the three open questions)

**Goal:** close the three MATERIAL items whose align direction is unambiguous, while the user
decides M1/M5/FORK-MECHANICAL.

**Pre-registration:** `PREREG_U3.md` committed as `cab6730b83a`, *before* the implementing commit.

**Changes**
- M2: `UnifiedHashJoin/JoinUsedFlags.h` `finalizePerRowFlags(size_t)` ->
  `finalizePerRowFlags(JoinUsedFlags & source, size_t)`; call site `HashJoin.cpp:2407` -> baseline
  self-merge shape `used_flags->finalizePerRowFlags(*used_flags, ...)`.
- M3: `UnifiedHashJoin/HashJoin.cpp` public `getTotalByteCount()` now null-checks then calls
  `doDebugAsserts()` under the `blocks_mutex` it already holds. Checked first that `doDebugAsserts`
  takes no lock (no deadlock) and that it dereferences `data` unguarded — hence the null check
  ordering copied from baseline `HashJoin.cpp:533-538`.
- M4: `UNIFIED_KEYGETTER_RANGE_IMPL` -> `KEYGETTER_RANGE_IMPL` (9 sites + `#undef`).

**Build**
```
$ ninja -C build/reldeb clickhouse > build/reldeb/u3_build_m234.log 2>&1
[514/515] Linking CXX executable programs/clickhouse
NINJA_EXIT=0
```
Green build is itself M4's refute test: no macro redefinition diagnostic, so the rename was indeed
gratuitous rather than clash-avoiding.

**Gate U2-ALIGN — structural proofs.** Re-ran `U3_normdiff.sh`; residual fell 1464 -> 1437.
```
KeyGetter.h      148 -> 128
JoinUsedFlags.h  108 -> 100
```
The decisive check is *changed* (`+`/`-`) lines, not context lines — a first grep counted context
and produced a false "REFUTED", corrected here:
```
$ awk '/^=== DIFF /{f=$3} /^[+-][^+-]/{print f" | "$0}' tmp/uhj_parity/U3_normdiff.txt \
    | grep -E "finalizePerRowFlags|KEYGETTER_RANGE_IMPL"
HashJoin.cpp | -        used_flags->finalizePerRowFlags(*used_flags, data->stored_columns_index->size());
HashJoin.cpp | +    used_flags->finalizePerRowFlags(*used_flags, data->stored_columns_index->size());
```
- **M2 CLOSED.** The signature is now identical on both sides; the only surviving difference is the
  surrounding baseline guard `if (!twoLevelMapIsUsed())` ("Two-level maps per-row flags will be
  finalized by ConcurrentHashJoin"). UHJ has neither `twoLevelMapIsUsed` (every map is two-level,
  E2) nor a `ConcurrentHashJoin` to defer to (E15), so that residue is EXCLUDED, not M2.
- **M4 CLOSED.** No `+`/`-` line mentions the macro; all its occurrences are now context.
- **M3 CLOSED.** `doDebugAsserts();` call-site count base 9 / uhj 9, and the public
  `getTotalByteCount()` bodies now match modulo the E13 lock split.

`HashJoin.cpp` residual moved 626 -> 627: the one added line is the `std::lock_guard`, inherent to
the E13 parallel-build lock split, which is EXCLUDED.

**Null/negative results:** the first proof attempt was wrong (grep matched context lines) and is
recorded above rather than quietly re-run.

**Not done in this batch:** M1, M5, M6 — awaiting user decisions.

## U3 / Unit 2 — batch 2: M6 (`optimize_read_in_order` gate)

**Goal:** stop `unified_hash` silently losing `optimize_read_in_order` in the legacy-analyzer path.

**Pre-registration:** `b9cbfabc408`, before the implementing commit.

**Change:** `ExpressionAnalyzer.cpp:2255-2256` — the gate
`typeid_cast<HashJoin *>(join.get())` now also accepts `UnifiedHashJoin *`.

**Gate U1-DISC — first attempt was inconclusive; recorded rather than discarded.**
The initial `EXPLAIN PLAN` compared all three algorithms and showed identical plans differing only
in the algorithm name. Cause: with default settings the join is wrapped, so the plan reported
`SpillingHashJoin(HashJoin)` and `join.get()` is a `SpillingHashJoin` — the `typeid_cast` fails for
*all three* algorithms and the gate is never reached. That run discriminated nothing.

Re-ran with `max_bytes_before_external_join=0, max_bytes_ratio_before_external_join=0` so the join
is a bare in-memory one:
```
hash          Algorithm: HashJoin           left ReadFromMergeTree  Read type: InOrder
unified_hash  Algorithm: UnifiedHashJoin    left ReadFromMergeTree  Read type: InOrder
parallel_hash Algorithm: ConcurrentHashJoin left ReadFromMergeTree  Read type: Default  (+ extra Sorting)
```

**Genuine before/after control.** Because the discriminator was first run *after* the edit, the
change was stashed, rebuilt and re-measured, then restored:
```
$ git stash push -q src/Interpreters/ExpressionAnalyzer.cpp && ninja -C build/reldeb clickhouse   # NINJA_EXIT=0
  unified_hash  Algorithm: UnifiedHashJoin  Read type: Default     <-- BEFORE fix
$ git stash pop -q && ninja -C build/reldeb clickhouse             # NINJA_EXIT=0
  unified_hash  Algorithm: UnifiedHashJoin  Read type: InOrder     <-- AFTER fix
```
CONFIRM: the divergence was real and this edit is what removes it. `parallel_hash` remains the
negative control — it fails the same cast and still reads `Default`, which is the baseline-faithful
outcome for it.

**Refute condition not met.** 200-row ordered result set is byte-identical between `hash` and
`unified_hash` under the read-in-order plan:
```
rows: hash=200 unified=200
diff -> no output ; M6 ROWS IDENTICAL
```
So UHJ does preserve the left block ordering the gate assumes; this is aligned, not UNSETTLED.

Artifacts: `tmp/uhj_parity/m6/`.

## U3 / interim correctness (Gate U4 dry run on the M2+M3+M4+M6 binary)

Test server: restarted on `:9101` because the running one (pid 982095, started 14:07) was executing
a **deleted** binary — `readlink /proc/982095/exe` reported `.../clickhouse (deleted)` — so it would
have tested pre-U3 code. This is the campaign's own dedicated server under `tmp/uhj_parity/`; no
other process was touched.

```
$ bash tests/queries/0_stateless/04658_unified_hash_join_equivalence.sh   # via CLICKHOUSE_PORT_TCP=9101
OK
JOB_EXIT=0
04658 MATCHES REFERENCE

$ bash tmp/uhj_parity/run_04659.sh
OK
JOB_EXIT=0
SCRIPT_EXIT=0
04659: harness form required — the upstream .sh passes --max_threads and so does shell_config, and
the client rejects a duplicated option ("option '--max_threads' cannot be specified more than
once", JOB_EXIT=36). run_04659.sh (written in a prior unit for this reason) mirrors the test logic
without the duplicate. Also had to CREATE DATABASE test, which clickhouse-test normally provides.
```

Both green. This is an interim run; Gate U4 will be re-run on the final binary.

## U3 / M1 feasibility study (read-only, pending user decision)

Checked the one thing that could have made the M1 port impossible: whether UHJ's map iterators can
report the bucket a cell lives in, which is what the baseline's `isBucketInRange` /
`skipToNextOwnedBucket` machinery needs.
```
$ rg -n "getBucket\b" src/Common/HashTable/TwoLevelHashTable.h
675:        size_t getBucket() const { return bucket; }
713:        size_t getBucket() const { return bucket; }
```
UHJ's `JoinHashMap`/`JoinHashMapWithSavedHash` are `TwoLevelHashMap`s, so both const and non-const
iterators expose it. UHJ's non-joined scan (`HashJoin.cpp:1452-1479`) already walks
`map.begin()..map.end()` with exactly the iterator type the baseline filters on. Open sub-question
if M1 is approved: whether `JoinFixedHashMap` (`PartitionedFixedHashMap`, used by
`key8`/`key16`/`range*`) exposes the same, and if not, how the baseline handles those — the baseline
reaches its bucket-partitioned path only for the two-level family, so a `requires`/`if constexpr`
guard is the likely shape.

**Status: not implemented. Awaiting the user's answer on M1 scope.**

## U3 / Unit 2 — batch 3: M1 and M5 (user decision: "Fix both M1 and M5")

**Authority call recorded.** User asked how M1/M5 affect performance and functionality. Answered
honestly: **neither changes query results.** M5 affects only hash-table size hints; M1 changes only
how many streams emit the non-joined rows of a RIGHT/FULL join, not which rows or their values.
Both were then implemented on the user's instruction, justified as divergence removal, **not** by
any throughput measurement. No bench was run for either.
M5 used the proposed split (drop `stats_collecting_params`; keep `max_threads` as EXCLUDED under E3),
which the user did not object to.

**Pre-registration:** `9f0b669b7fc`, before the implementing commit.

**The trap M1 nearly walked into.** The baseline gates its partitioned branch on
`if constexpr (requires { it.getBucket(); map.numBuckets(); })` (`HashJoin.cpp:1406-1409`). Copying
that verbatim would have been silently wrong here: `numBuckets()` is declared
`requires(isFixedStorage())` (`TwoLevelHashTable.h:77-82`) and UHJ runs every map in runtime mode
(`BITS_FOR_BUCKET == -1`), so the clause is **false for every UHJ map**. All of them would have
fallen into the unfiltered branch and each stream would have emitted every non-joined row —
duplicates, in a path that aggregate-only tests could easily have missed. Found by reading
`TwoLevelHashTable.h` before writing the port, not by testing afterwards.

Partitioning is therefore by the **iteration** bucket, which is what `iteratorAt`/`getBucket`
actually index (`TwoLevelHashTable.h:733`):
- `JoinHashMap`/`JoinHashMapWithSavedHash` (RuntimeStorage): `iterationBuckets() == num_buckets`
  -> genuine disjoint partition across streams.
- `JoinFixedHashMap` (FixedRangeStorage, `key8`/`key16`/`range*`): `iterationBuckets()` is `1`
  (`TwoLevelHashTable.h:397-398`) -> everything is iteration-bucket 0, so stream 0 emits it all and
  the others emit nothing. Serial for those types, but correct.
Correctness needs only a disjoint, complete cover of the cells; both cases give one.

**Build.** First attempt failed: `-Werror,-Wshadow`, because the baseline's parameter name
`num_buckets` shadows UHJ's `num_buckets` **member** (the map's bucket count, which the baseline has
no equivalent of). Renamed the parameters to `stream_idx`/`num_streams` — which is what
`ConcurrentHashJoin::getNonJoinedBlocks` calls them (`ConcurrentHashJoin.cpp:536-537`), so the fix
is baseline-faithful rather than invented. `NINJA_EXIT=0` after.

**Gate U2-ALIGN — capability now matches the parallel baseline.** Count of
`NonJoinedBlocksTransform` in `EXPLAIN PIPELINE` for a RIGHT join at `max_threads=8`:
```
hash           NonJoined_count=0     (serial, emitted inside JoiningTransform)
parallel_hash  NonJoined_count=8
unified_hash   NonJoined_count=8     <-- was 0 before this change
```

**Refute condition not met.** `tmp/uhj_parity/m1_nonjoined.sh`, 30 cases — RIGHT and FULL x
{aggregate, **full sorted row set**, NULL right keys, string keys, UInt8 direct-addressed keys} x
`max_threads` {1,4,16}, each compared against `hash`:
```
30/30 ok ; FAIL=0 ; M1 ROWSETS IDENTICAL ; JOB_EXIT=0
```
The full-row-set cases matter specifically: a wrong partition duplicates or loses rows in a way
`count()`/`sum()` can cancel out. The UInt8 case covers the direct-addressed single-iteration-bucket
path, and the NULL-key case covers the stream-0 nullmap guard.

**Gate U4 on this binary:** `04658` OK JOB_EXIT=0; `04659` OK JOB_EXIT=0.

**Honest note on the line metric.** Residual went **1435 -> 1445 (+10)** even though M1 *removed* a
behavioral divergence. Reason: UHJ's partitioned scan is textually shorter than the baseline's,
because the baseline keeps an `if constexpr` guard plus a single-level `else` branch that would be
**dead code** in UHJ (every UHJ map is a `TwoLevelHashTable`), and because the parameters had to be
renamed for the shadow error. Padding UHJ with a dead branch purely to shrink the line count would
be gaming the metric, so it was not done. The line count is a proxy; the completion oracle is
`AVOIDABLE_MATERIAL`, and the capability proof above is what settles M1.

Attribution after this batch — still no unexplained divergence:
```
TWOLEVEL         hunks=75   changed_lines=1232
PARALLEL_BUILD   hunks=25   changed_lines=195
FORK_MECHANICAL  hunks=9    changed_lines=18
UNATTRIBUTED     hunks=0    changed_lines=0
```

## U3 / Unit 5 — independent verification (two graders)

### Verifier 1 (static, shell blocked in its environment) — FIX-THEN-RESHIP
No banned move; **no avoidable divergence mislabeled as excluded** (it read every hunk). Three real
gaps, all addressed: W1 prereg miss (recorded as a gap, not backdated), `joinDispatch.h` never
diffed (script fixed), broad attribution regexes (answered with `U3_attribute_strict.sh` + reading
all 347 residue lines). Its stated condition for SHIP was that the runtime gates come back clean.

### Verifier 2 (runtime, adversarial) — RUNTIME VERDICT: PASS
Binary currency confirmed (`readlink /proc/1080068/exe` -> real path, not `(deleted)`).
~50 row-set comparisons vs both `hash` and `parallel_hash`, all PASS, zero mismatches:
UInt8/UInt16 (direct-addressed), UInt64, String, FixedString, LowCardinality, two-column keys,
Nullable mixed, all-NULL keys, empty right, empty left, 3-row right at `max_threads=32`, 99% skew,
`ON 1=0`, `join_use_nulls=1`, spill at 8MB and 100MB, 400k-row full sorted sets, duplicate chains
(RowRefList), `max_block_size=7` (resume + `skipToNextOwnedBucket` stress), odd thread counts 3/5/7.
`04658` exit 0 byte-identical to reference; `04659` exit 0 `JOB_EXIT=0`.

**Finding it caught that my own discriminator did not.** The parallel non-joined path is reachable
only when the optimizer does **not** swap the join: with `query_plan_join_swap_table` at its default,
a RIGHT join over real tables is rewritten to LEFT and the path is dormant (`NonJoined` = 0 for
`parallel_hash` **and** `unified_hash` alike). My Gate U2-ALIGN measurement used a CTE-based query
where the swap did not apply, which is why it showed 0/8/8; both observations are correct, but mine
did not reveal the dependence. With `query_plan_join_swap_table=0` the verifier got, across every
key type and threads {1,2,3,5,7,8,16,32}:
```
hash          NonJoined = 0
parallel_hash NonJoined = max_threads
unified_hash  NonJoined = max_threads   (identical to parallel_hash)
```
This does not weaken M1: `unified_hash` now tracks `parallel_hash` in **both** regimes — dormant
when the plan swaps, parallel when it does not — which is exactly the divergence removal claimed.
It does mean the feature's practical reach is narrower than "all RIGHT/FULL joins", and that is
recorded rather than left implied.
Also N/A, not a gap: ASOF RIGHT/FULL is rejected by every algorithm including `hash`; multi-OR FULL
is `NOT_IMPLEMENTED` in both non-hash algorithms. Consistent, so not divergence.

**Combined verdict: SHIP.** Verifier 1's SHIP condition was met by Verifier 2's clean runtime pass.
Independence: full — neither grader saw the implementer's reasoning; Verifier 1's tool limitation
was compensated by commissioning Verifier 2 rather than by self-passing.

## OPERATIONAL — post-mission performance snapshot (NOT a gate, NOT acceptance evidence)

Measured on request AFTER the U3 mission closed. These numbers were not used to accept or reject
any unit, and no design choice was made because of them. Recorded as context only.

Host 96 CPUs, `build/reldeb`, server :9101 (binary currency confirmed), 7 runs, median + sample
stdev, `enable_join_runtime_filters=0`, `max_bytes_before_external_join=0`.
Logs: `u3_bench_serial.log`, `u3_bench_parallel.log`, `u3_bench_extra.log`.

### Serial, `max_threads=1` (vs `hash`)
```
build-bound INNER   hash 267ms/164324us   unified 257ms/161954us   parity (unified marginally ahead, within stdev ~88/~8k)
probe-bound INNER   hash 378ms/318919us   unified 382ms/324767us   parity (+1.1% wall, +1.8% CPU)
RIGHT + non-joined  hash 1167ms/986486us  unified 1223ms/1049852us +4.8% wall (within stdev 242), +6.4% CPU (beyond stdev)
```

### Parallel, `max_threads=16` (vs `parallel_hash`)
```
build-bound INNER   phash 125ms/908822us   unified 144ms/1138819us  wall within noise (125+43>144); CPU +25% BEYOND noise
probe-bound INNER   phash  91ms/446506us   unified  83ms/ 420393us  unified FASTER: -8.8% wall, -5.8% CPU (stdevs 2.3/0.7, so real)
RIGHT + non-joined  phash 205ms/1246121us  unified 288ms/1511559us  +40% wall, +21% CPU, both beyond noise
```

### What M1 actually bought — clean A/B on the setting that gates it
`parallel_non_joined_rows_processing` toggles `allowParallelNonJoinedRowsProcessing()`, which
`supportParallelNonJoinedBlocksProcessing()` consults, so 0 reproduces pre-M1 behaviour exactly.
RIGHT join, 20M right / 5M left, `max_threads=16`, `query_plan_join_swap_table=0`, 5 runs:
```
pnj=0 (pre-M1)   parallel_hash 667ms    unified_hash 824ms
pnj=1 (post-M1)  parallel_hash 205ms    unified_hash 293ms
```
**M1 is a 2.8x wall improvement for `unified_hash` on this workload** (824 -> 293), and it closes
most of the gap to `parallel_hash`: 824/205 = 4.0x worse before, 293/205 = 1.43x worse after.

**This revises my earlier characterisation to the user.** I had described M1 as throughput-only and
leaned toward risk-accepting it. That was accurate about *results* (M1 changes no rows) but badly
understated the magnitude: declining M1 would have left a ~2.8x regression against the baseline in
place. The user's instruction to fix it was the right call and my lean was wrong.

### Attribution of the two remaining deficits
- Parallel build-bound CPU +25%: consistent with the prior campaign's ~21% figure, attributed to
  EXCLUDED unconditional two-level (per-bucket sub-tables, `BUCKETS_PER_THREAD=2` cache footprint).
- Parallel RIGHT non-joined +40% wall: `parallel_hash` shards the map so each stream scans its own
  small table, whereas UHJ streams walk one shared partitioned map and each `isUsed(offset)` needs
  `offsetInternal`, which on a partitioned table resolves through the bucket prefix sum (E8/E10)
  rather than flat pointer arithmetic. TwoLevel-attributable, and NOT investigated further, because
  chasing it is out of scope for a mission whose anti-goals include optimising for bench deltas.
  Flagged as a candidate for a future, explicitly performance-scoped mission.
