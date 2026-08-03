# `unified_hash` performance attribution — REPORT

Mission-start commit: `0945a745399466911e0b94289fa120b168032d31`
Branch: `uhj-perf-attribution`
Binary of record: `BuildID[sha1]=b7980f6e38fd7fccc6cb883a140c0c0a1b4dbe78`
Host: aarch64 **Neoverse-V2** (Graviton4), 96 CPUs
Server: `127.0.0.1:9111`, started by `tmp/uhj_parity/perf/start_server.sh`

---

## 0. Verdicts, up front

| Unit | Verdict |
| --- | --- |
| **Unit 0 — the measurement instrument** | **GREEN on 5 of 6 gates; G0.4 RED** (see §5). The deficit map is complete: 144/144 cells measured or SKIPPED with a reason. |
| **Unit 1 — attribution** | **PARTIAL.** One pre-registered ablation ran and **refuted** its claim (A1, placement). A second cause (A7) was **CONFIRMED and fixed** — commit `aeddef94b15`, targeted phase -10%, correctness re-verified — but it accounts for only ~2.3pp of a ~12.6pp gap. The bulk of the 1-thread deficit stays **UNSETTLED** as claim A6. See §3 and §7. |

### The headline, stated plainly

**The deficit the mission was written to explain does not exist at 16 or 64
threads.** Against `parallel_hash`, `unified_hash` is never slower on wall time in
any of the 48 cells measured there, and at 64 threads it wins **all 24**. What
remains is smaller and differently shaped than the brief assumed:

1. a **1-thread deficit against `hash`** of +4.6% to +9.7% wall on 13 of 72 cells,
   concentrated entirely in the join kinds that maintain used-flags; and
2. a **64-thread composite-key build CPU excess** of +21% to +38%, which coexists
   with a 15-37% wall *win* — `unified_hash` is buying latency with CPU there.

### Authorization-required flags

None. No irreversible, destructive, production-facing or security-relevant action
was taken or is pending.

### HIGH-IMPACT assumptions

| # | Assumption | Why it matters | Revisit trigger |
| --- | --- | --- | --- |
| A1 | Build tables carry **2 rows per distinct key**. With 1 row/key all three implementations promote strictness to `RightAny` and the whole matrix silently measures `MapsOne` instead of `MapsAll`. | Changes which code path every number describes. | The unique-key/`RightAny` regime is a real shape (primary-key joins) and is **not covered**; see §6. |
| A2 | Join keys are spread over the whole `UInt64` range (`ordinal * 6364136223846793005`), not dense. | Dense keys fire `canConvertToFixedHashMap` and would turn every small-cardinality cell into a direct-addressed-array comparison labelled as a hash-join comparison. | If the conversion's guards change. |
| A3 | Cell-level wall/CPU come from `query_duration_ms` and `UserTimeMicroseconds`, which are **independent of processor attribution**. Phase splits come from `processors_profile_log`. | The phase split had two bugs (§5); the cell-level numbers were never affected by either. | — |

### Risk-accepted leads

None outstanding. Both inherited leads are addressed in §4.4.

---

## 1. What the three implementations actually are

The brief's description was right about one of three. Corrected from code, and
this correction changes which experiments are meaningful:

| | Structure | Evidence |
| --- | --- | --- |
| `hash` | One **flat** `HashMap`. Every standalone caller passes `use_two_level_maps=false`. | `PlannerJoins.cpp:1268,1337`, `GraceHashJoin.cpp:747`, `JoinSwitcher.cpp:22`, `SpillingHashJoin.cpp:58` |
| `unified_hash` | **One shared** `TwoLevelHashMap`, runtime bucket count `bucketCountForThreads(n)` = 1 / 32 / 128 at 1 / 16 / 64 threads. Per-bucket cache-line-padded `std::mutex`, per-bucket arenas. | `UnifiedHashJoin/HashJoin.h:56,67,78`; `HashJoin.cpp:66-74` |
| `parallel_hash` | N shards, each a baseline `HashJoin` built with `use_two_level_maps=true`. **But** a row's shard is derived from the bucket it will occupy, so only **256** partitions are ever populated — and at build finish they are merged into **one shared 256-bucket map** that the probe uses. | `ConcurrentHashJoin.cpp:230` (shard ctor), `:589` (`getBucketFromHash(h) & (slots-1)`), `:817-834` (merge, which throws if the ownership invariant breaks), `:197` (`slots` capped at 256) |

**Populated partitions of the data:**

| max_threads | `unified_hash` | `parallel_hash` |
| --- | --- | --- |
| 16 | 32 | 256 |
| 64 | 128 | 256 |

So `unified_hash` splits the same rows into **8x fewer** partitions at 16 threads
and 2x fewer at 64 — meaning its sub-tables are correspondingly **larger**. The
sharding is a build-phase property only; at probe time both are one shared
partitioned map.

> An earlier version of this report said 4096 and "128x fewer". That counted empty
> bucket *objects* rather than populated partitions and was wrong; it is corrected
> here and in WORKLOG E3. Commit `b8452c4f70b`'s message still carries the wrong
> figure, and history is not rewritten on this branch.

Ruled out cheaply, so no run was spent on it: both sides use **identical hash
functions** (`HashCRC32<UInt32/UInt64>`, `UInt128HashCRC32`, `UInt128TrivialHash`,
`TwoLevelHashMapWithSavedHash` for strings).

---

## 2. The deficit map

144 declared cells = 6 join kinds x 3 key types x 2 match rates x 4 (thread,
cardinality) pairs. 120 classified; 24 have no `parallel_hash` comparator and are
SKIPPED with a reason (§2.3). 7 interleaved reps per point; noise band
`max(5%, 1 sample stdev)`.

Reproduce: `python3 tmp/uhj_parity/perf/gates.py g07 --all`
Full machine-readable map: `tmp/uhj_parity/perf/results/deficit_map.json`

### 2.1 Summary

| threads | comparator | metric | faster | within noise | slower |
| --- | --- | --- | --- | --- | --- |
| 1 | `hash` | wall | 0 | 59 | **13** |
| 1 | `hash` | CPU | 0 | 59 | **13** |
| 16 | `parallel_hash` | wall | 14 | 10 | **0** |
| 16 | `parallel_hash` | CPU | 6 | 18 | **0** |
| 64 | `parallel_hash` | wall | **24** | 0 | **0** |
| 64 | `parallel_hash` | CPU | 16 | 3 | **5** |

### 2.2 The 1-thread deficit is monotone in used-flag work

Median wall delta vs `hash`, over all 12 cells of each kind:

| kind | n | median wall | max |
| --- | --- | --- | --- |
| `RIGHT` | 12 | **+5.27%** | +9.30% |
| `FULL` | 12 | **+4.79%** | +9.70% |
| `LEFT` | 12 | +2.33% | +4.40% |
| `LEFT SEMI` | 12 | +2.06% | +6.67% |
| `LEFT ANTI` | 12 | +1.57% | +7.14% |
| `INNER` | 12 | **+0.64%** | +2.78% |

`INNER`, which maintains no used-flags and emits no non-joined rows, is at parity.
The cost is not on the common lookup path.

### 2.3 Cells slower on wall or CPU (19 of 120)

`p+nj` is probe and non-joined **combined**, which is the only fair comparison for
these kinds — see §3.2 for why comparing them separately is invalid.

| cell | wall% | cpu% | build% | probe% | p+nj% |
| --- | --- | --- | --- | --- | --- |
| `INNER\|comp\|lo\|t64\|large` | **-15.1** | **+18.5** | **+36.2** | -3.1 | - |
| `LEFT\|comp\|lo\|t64\|large` | **-15.6** | **+12.5** | **+37.1** | -9.7 | - |
| `LEFT\|comp\|hi\|t64\|large` | **-16.8** | +8.6 | **+37.6** | -13.5 | - |
| `INNER\|comp\|hi\|t64\|large` | **-16.7** | +8.6 | **+36.3** | -13.4 | - |
| `RIGHT\|comp\|lo\|t64\|large` | **-36.6** | +5.9 | **+21.2** | -0.4 | -5.8 |
| `FULL\|u64\|hi\|t1\|medium` | +9.7 | +10.2 | +4.5 | -2.9 | **+12.6** |
| `RIGHT\|u64\|hi\|t1\|medium` | +9.3 | +10.0 | +3.8 | -3.7 | **+12.7** |
| `RIGHT\|u64\|lo\|t1\|small` | +9.1 | +7.5 | -0.9 | +8.4 | +11.7 |
| `FULL\|u64\|hi\|t1\|small` | +8.9 | +8.5 | -0.9 | +9.0 | +9.4 |
| `RIGHT\|u64\|hi\|t1\|small` | +7.7 | +8.0 | +2.5 | +8.5 | +9.0 |
| `LEFT-ANTI\|u64\|hi\|t1\|small` | +7.1 | +8.8 | +1.9 | +19.1 | **+19.1** |
| `FULL\|u64\|lo\|t1\|small` | +6.8 | +6.2 | +2.1 | +5.3 | +6.8 |
| `LEFT-SEMI\|u64\|lo\|t1\|small` | +6.7 | +1.2 | -2.6 | +5.2 | +5.2 |
| `FULL\|comp\|hi\|t1\|medium` | +6.6 | +6.4 | +0.5 | -5.6 | +9.1 |
| `RIGHT\|str\|hi\|t1\|medium` | +6.0 | +6.0 | +0.8 | -0.9 | +7.6 |
| `RIGHT\|comp\|hi\|t1\|medium` | +5.6 | +5.4 | +0.6 | -7.6 | +7.2 |
| `RIGHT\|str\|hi\|t1\|small` | +5.6 | +5.1 | +0.5 | +7.0 | +7.3 |
| `FULL\|u64\|lo\|t1\|medium` | +5.4 | +5.6 | +3.8 | -44.8 | +5.9 |
| `FULL\|comp\|hi\|t1\|small` | +4.6 | +5.1 | -18.3 | +6.1 | - |

The first five rows are the 64-thread composite-key cells: **slower on CPU,
substantially faster on wall.**

### 2.4 SKIPPED cells

48 (cell, algorithm) pairs, one reason, pre-registered before it was hit:

> no `parallel_hash` comparator: `allowParallelHashJoin()` is false for SEMI/ANTI
> (`TableJoin.cpp:1301-1303`), so the request silently yields plain `hash`.

Zero silently-missing cells (`gates.py g06`: 144 declared, 144 covered, 0 missing,
0 partial, 0 errored).

---

## 3. Attribution

### 3.1 Ranked claims

**No claim is CONFIRMED.** One ablation was run and it **refuted** the claim it
tested. The rest are LEADs from the code inventory with no measurement of their
own.

The refutation is the most useful thing in this table, and its **scope matters**.
A1 tested the *placement* of the non-joined scan in a separate transform. Setting
`parallel_non_joined_rows_processing=0` puts `unified_hash` on the baseline's
pipeline, and `JoiningTransform` then calls the **same** `getNonJoinedBlocks`
(`JoiningTransform.cpp:167,630`). So the ablation moved the scan; it did not
remove its per-cell cost, which is present in both arms. A1-d verified the
transform vanished — true, and enough to make the null valid for the placement
claim — but not enough to clear the scan itself.

So: placement is REFUTED (A1); the scan's per-cell cost was never tested and is
now **A7**, carrying codegen evidence and a bounded estimate rather than a
measured percentage. An earlier draft recorded "13.8% of phase" as A1's cost; that
was a phase share, never an ablation result, and quoting it as a cause would have
been exactly the move the brief bans. It is struck.

| # | Operation | Which impl | Sub-phase / cell | Cost | Ablation | Codegen | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A6** | **Per-matched-row used-flag maintenance** — `offsetInternal` then `setUsed` — on `unified_hash`'s partitioned map versus `hash`'s flat one. Present for every kind that needs flags (RIGHT/FULL/SEMI/ANTI); absent for INNER, which is at parity. | differs in kind (partitioned vs flat addressing) | **probe**, 1 thread | probe **+8.1% to +14.0%** with the pipeline shape equalised; cell wall +7.1% to +11.1% | not run | not yet produced | **UNSETTLED** (localised, not attributed) |
| **A7** | **The non-joined scan's per-cell `offsetInternal`.** It recovered the cell's bucket by re-hashing the key with `crc32cx` (provably **0** at 1 thread — one bucket) and went through `BucketPrefixSums::offset`, storing a `std::call_once` closure and taking an acquire-load every cell, to reach the same `(ptr-buf)+1` the flat baseline computes inline. | in UHJ; absent in `hash` | post-join scan, 1 thread | **FIXED.** Targeted phase **-10.0%/-11.1%** at high match (14,794 -> 13,315 us, ~10 sd), only **-2/-3%** at low match. Closes **~2.3pp of a 12.6pp** probe+non-joined gap. | **FIXED AND RE-MEASURED** (commit `aeddef94b15`): bidirectional in effect — the dose-response splits high vs low match exactly as the mechanism predicts. | `codegen/N1_nonjoined_scan.md` | **CONFIRMED — real, but small** |
| ~~A1~~ | ~~The *placement* of the non-joined scan in a separate `NonJoinedBlocksTransform`~~ | in UHJ only | post-join | the transform is real, but its **placement** does not cause the gap | **RAN.** Removing it closes at most **2.3 of 12.2** points; one cell got worse. A1-d PASS => valid null. | `codegen/N1_nonjoined_scan.md` | **REFUTED** |
| A2 | 64-thread composite-key **build** CPU excess | UHJ vs `parallel_hash` | build, 5 `comp\|t64\|large` cells | build CPU **+21% to +38%**, cell CPU +5.9% to +18.5%, wall **-15% to -37%** | not run | not produced | **UNSETTLED** |
| A3 | per-bucket **mutex acquire/release** in `insertIntoBuckets` (`HashJoin.cpp:187,207`); no counterpart in `hash` | UHJ only | build | not measured | not run | not produced | **UNSETTLED (LEAD)** |
| A4 | per-row bucket routing + `prefix[bucket]` load in `Prober::find` (`TwoLevelHashTable.h:554-563`) | UHJ | probe | not measured; expected ~0 at 1 thread (the `sole` short-circuit) | not run | not produced | **UNSETTLED (LEAD)** |
| A5 | `per_offset_flags` sized from **summed** bucket capacities, wider and sparser than the baseline's | UHJ | probe + post-join | not measured | infeasible in isolation (consequence of the map layout) | not produced | **UNSETTLED (LEAD)** |

**A7's earlier bound was wrong by about 6x, and the fix is what proved it.** This
report previously carried A7 as SUPPORTED with "an upper bound of order 80% of the
probe gap", extrapolated from the 39-vs-102 instruction ratio. The fix recovered
**1,479 us, not ~9,200 us**. Instructions are not cycles, and the 102 covered the
whole per-cell emitting path while the fix removes only the re-hash and the
`call_once`, leaving the bucket-aware iterator and the out-of-line frame. The
estimate is struck, not adjusted. It is the clearest evidence in this report for why
an instruction ratio may not be written as a percentage.

Consequence: **A7 is confirmed and fixed, and it does not explain the 1-thread
deficit.** The non-joined phase is about a tenth of these queries, so the 13
one-thread cells classified slower improve only from a mean +7.28% to +6.54%
(13 cells -> 10). Claim A6 remains the open target.

A6's probe deltas are measured with the pipeline shape **equalised on both sides**
(`parallel_non_joined_rows_processing=0`), so they are not an artefact of the
processor split. They localise the cost; they do not attribute it, because no
ablation has isolated the flag maintenance itself.

The sharp version of the open question: at 1 thread `bucketCountForThreads(1) = 1`,
so `Prober` takes the `sole` short-circuit and bucket routing should cost nothing.
Either that path is not as free as it reads, or the cost is in the used-flag
array's layout rather than the lookup. Those two are distinguishable — an
instruction-count cause raises `instructions`/row at flat IPC, a layout cause
raises `LLC-load-misses`/row and drops IPC — which is exactly what G1.3's counters
would settle, and they have not been gathered.

### 3.2 A phase comparison that was invalid, and the fix

Comparing `probe_us` and `nonjoined_us` as separate columns compares different
partitions of the same work, because the two implementations put the non-joined
rows in different processors:

```
FULL|u64|hi|t1|medium
  hash          JoiningTransform 93,560 us  out_rows 3,885,728   NonJoined: absent
  unified_hash  JoiningTransform 93,167 us  out_rows 3,800,000
                NonJoinedBlocksTransform 14,888 us  out_rows 85,728
```

`hash` emits its 85,728 non-joined rows from inside `JoiningTransform` at almost
no marginal cost (93,560 vs 93,167 us for 85,728 rows more work). Taken
separately, `unified_hash` looked **better** on probe (-2.9%) while being 9.7%
worse overall. Combined, it is +12.6%. `gates.py g07` now reports
`probe_plus_nonjoined_us` for this reason.

### 3.3 Where `unified_hash` wins, and the likely reason

At 16 and 64 threads `unified_hash` wins every wall-time cell. The leading
explanation — a LEAD, not measured — is an asymmetry that favours it:
the baseline hardcodes `constexpr bool use_offset = true` (`HashJoin/KeyGetter.h:19`),
so `parallel_hash` computes an offset for **every matched row**, which on a
two-level map means `TwoLevelHashTable::offsetInternal` — a full re-hash for cells
that do not save their hash, plus a `std::call_once`. `unified_hash` asks for the
offset only when the join kind needs flags (`UnifiedHashJoin/HashJoinMethods.h:85-90`).
Pre-registered in PREREG P1.1 C4 with a symmetric ablation that would settle it;
**not run**.

Also observed, not a timing claim: for a 10,000-row build, reported in-memory size
was `hash` 1.13 MB, `parallel_hash` **16.86 MB**, `unified_hash` 0.47 MB —
`parallel_hash` pays 256 sub-table minimum capacities whether or not it has rows
for them.

### 3.4 The two inherited leads

**LEAD (1)** — "the parallel build-bound CPU excess is the unconditional two-level
map's per-bucket sub-table cache footprint at `BUCKETS_PER_THREAD = 2`".
**Not refuted, but its premise is backwards.** `unified_hash` has *fewer*
partitions than `parallel_hash` (32 vs 256 at 16 threads), so it cannot be paying
for having more sub-tables. A footprint effect may still be real with the opposite
sign — its sub-tables are 8x *larger*. The discriminating probe is pre-registered
(PREREG P0.2R): sweep `BUCKETS_PER_THREAD` 2 -> 8 -> 32; shrinking gap means "too
few, too large", growing gap means LEAD (1) as worded, flat means the cost is
per-operation. **Not run.** Verdict **UNSETTLED**.

**LEAD (2)** — "the parallel RIGHT non-joined excess is `offsetInternal`'s bucket
prefix sum versus flat pointer arithmetic". **The excess it explains no longer
exists**: `RIGHT|u64|lo|t16|large` now shows `unified_hash` 7.9% *faster*. The
mechanism is, however, exactly the one that surfaced at **1 thread** as claim A1.
Verdict **UNSETTLED**, redirected from 16 threads to 1 thread.

### 3.5 Per-cell residual

Sum of CONFIRMED deltas is **zero** in every cell, because there are no CONFIRMED
claims. The unexplained residual is therefore **100% of every measured gap**, with
the SUPPORTED/LEAD candidates in §3.1 listed against it. Stated as a number rather
than left implicit, per Gate G1.6.

---

## 4. Refuted hypotheses and corrected errors

| Hypothesis | Outcome | Evidence |
| --- | --- | --- |
| `parallel_hash` has 4096 sub-tables at 16 threads | **WRONG (mine)** — 256. Slot is derived from bucket. | `ConcurrentHashJoin.cpp:589`, merge invariant `:830` |
| The build-only cross-check is systematically biased for `parallel_hash`, because its drain loop yields differently | **WRONG (mine)** — the deviations were duplicate-`query_id` double-counting. On clean data, 0/29 pairs over tolerance. I had a plausible mechanism ready for a number that was simply corrupt. | WORKLOG E7.3 -> E8 |
| Hash-function difference between the implementations | **Ruled out** from code before measuring | §1 |
| Dense keys are safe for the small-cardinality cells | **WRONG (mine)** — they fire `canConvertToFixedHashMap` | WORKLOG E4 |
| One row per key is a clean cardinality design | **WRONG (mine)** — promotes strictness to `RightAny` in all three implementations | WORKLOG E5.2 |

---

## 5. Evidence matrix

| Criterion | Gate invocation | Result (raw) | Non-gate sources (origins) | Verdict |
| --- | --- | --- | --- | --- |
| G0.1 measured algo == requested | `python3 tmp/uhj_parity/perf/gates.py g01` | `assertion runs: 264 / verdict==requested: 264 / MISMATCH: 0 / UNKNOWN: 0` | symbol-level `trace_log` stacks; independently, `ConcurrentHashJoin*` ProfileEvents and processor topology agree | **GREEN** |
| G0.2 results agree | `... gates.py g02` | `cells with checksum runs: 144 / disagree: 0 / weak-checksum disagree: 0` | strong full-column `cityHash64` checksum + a weak checksum on every timed run | **GREEN** |
| G0.3 A/A calibration | `... gates.py g03` | 1-thread `-0.89%` wall / `-1.14%` CPU; 64-thread `+2.06%` / `+0.45%`; band `+-5%` | — | **GREEN** |
| G0.4 known-signal recovery | `... gates.py g04` | `(a) delta=-4.0% within_noise`; `(b) delta=-7.9% faster`; both expected positive | instrument power shown by A/A and by resolving +5%..+19% and -37% in the same sweep | **RED** (§5.1) |
| G0.5 phase split | `... gates.py g05` | identity `1838 runs / 0 violations`; build-only `29 pairs / 0 over tol`; internal-timer `48 cells / 0 over tol, median dev 0.3%, max 1.5%`; join share median 82.2% | three origins, one of which (`ConcurrentHashJoinBuildMicroseconds`) shares no machinery with processor accounting | **GREEN** |
| G0.6 coverage | `... gates.py g06` | `declared 144 / covered 144 / MISSING 0 / PARTIAL 0 / errors 0 / SKIPPED 48 with reason` | — | **GREEN** |
| G0.7 deficit map | `... gates.py g07 --all` | 120 cells classified; emitted to `results/deficit_map.json` | — | **EMITTED** |
| G1.1 codegen artifact | see `codegen/N1_nonjoined_scan.md` | in progress at time of writing | — | **PARTIAL** |
| G1.2 ablation | — | **not run** | — | **NOT DONE** |
| G1.3 counters | — | **not run** (`perf stat` not exercised) | — | **NOT DONE** |
| G1.4 null-ablation validity | — | not applicable, no ablation run | — | **N/A** |
| G1.7 baselines pristine | `git diff 0945a745399 -- src/Interpreters/HashJoin/ src/Interpreters/ConcurrentHashJoin.cpp src/Interpreters/ConcurrentHashJoin.h` | empty | — | **GREEN** |

### 5.1 Why G0.4 is red, and what would settle it

Neither recorded snapshot effect reproduces; both now show `unified_hash` at
parity or ahead. PREREG P0.5 pre-committed to separating two explanations:

- *The instrument lacks power* — **rejected.** A/A resolves to within 2.1%, and
  the same sweep detects +5% to +19% deficits and a -37% win. An instrument that
  resolves +5% does not miss +40%.
- *The code changed since the snapshot* — **supported, specifically.** The
  snapshot binary predates commit `5362055b4ed`, "Give `unified_hash` the parallel
  non-joined path", which changes exactly the mechanism behind effect (b). The
  pre-existing `build/reldeb` binary was also stale: `ninja` rebuilt **430** dbms
  objects at mission start.

G0.4 is **not re-scored green by argument**. The settling experiment is cheap and
named: commit `5362055b4ed` touches only `UnifiedHashJoin/HashJoin.{cpp,h}` (92+14
lines), so reverting those two files, rebuilding incrementally, and re-measuring
cell (b) would decide it in one build. **It was not run** — see §7.

---

## 6. Declared coverage gaps

1. **The non-joined scan is barely exercised at small/medium cardinality.** 2M
   probe rows over 10k/500k build keys match essentially every build key, so
   RIGHT/FULL there emit few non-joined rows (median 98 us and 7.3 ms, against
   2.4-3.0 s at `large`+`lo`). The match-rate knob controls the fraction of
   *probe* rows that match, which equals the fraction of *build* keys left
   unmatched only when the probe is not much larger than the build. Closing it
   needs a probe-rows-per-build-key knob.
2. **The unique-key / `RightAny` regime is not covered** (assumption A1). It is a
   real and common shape. Closing it is one re-run with `ROWS_PER_KEY = 1`.
3. **The direct-addressed `range*` conversion is UNSETTLED.** It never fired
   despite every readable precondition holding. Settling experiment: a temporary
   `LOG_DEBUG` on each early return in `canConvertToFixedHashMap` /
   `tryConvertToFixedHashMapImpl`. One rebuild, one query.
4. **Duplicate-key `RowRefList` chains are length 2 only.** Long chains are not
   exercised.
5. **`key8`/`key16`/LowCardinality key types are not on the matrix.** Per the
   inventory these are the types where `parallel_hash` stays single-level and
   really is N shards — a large, type-dependent divergence, unmeasured.

---

## 7. Next-mission input

Ranked by expected headroom. Note the honest framing: **the largest single result
of this mission is that four fifths of the presumed problem is not there.**

| Rank | Target | Headroom | Tier | Risk of removing it |
| --- | --- | --- | --- | --- |
| 1 | **Claim A6 — per-matched-row used-flag maintenance** (`offsetInternal` + `setUsed`) on the partitioned vs flat map. The 1-thread gap is in the probe (+8.1%..+14.0%) with the build at only +3.1%..+4.4%, and INNER — which keeps no flags — is at parity. | the whole remaining 1-thread deficit, ~5-9% wall on 10 cells | **UNSETTLED** | Unknown. Start with `perf stat`: it separates an instruction-count cause from a flag-array-layout cause before any code is touched. |
| 2 | ~~A7, the non-joined scan's `offsetInternal`~~ | **DONE** — commit `aeddef94b15`, phase -10%, ~2.3pp of gap | **CONFIRMED** | Shipped; comparators proven inert. |
| 2 | 64-thread composite-key build CPU (claim A2) | +21% to +38% build CPU | UNSETTLED | Unclear — it currently coexists with a large wall win, so a naive fix could trade latency for CPU in the wrong direction. |
| 3 | `BUCKETS_PER_THREAD` sizing | Unknown; the 3-way probe in P0.2R bounds it in one sweep | UNSETTLED | Low — it is already a tunable constant. |
| 4 | Baseline's unconditional `use_offset = true` | Would *remove* a `unified_hash` advantage, so measure before touching anything else, or every other attribution is netted against a moving baseline | LEAD | n/a — this is a baseline property, not a UHJ defect. |

**Do first, before any optimisation:** run the four settling experiments named in
§5.1 and §6. They are all cheap, all runnable in this environment, and three of
them change what the ranked list above should say.

---

## 8. Deviations from the brief

Every one of these is measured and documented rather than silent.

1. **Unit 1 is not complete.** No ablation was built or run, so no claim is
   CONFIRMED and the residual is 100% in every cell. Reported rather than dressed
   up.
2. **G0.4 left RED.** The brief says never to close a unit on red. It is red, it is
   reported as red, and the one-build experiment that would settle it is named.
3. **`llvm-mca` is unavailable** on this host, so loop-body throughput analysis is
   out of reach. Instruction, branch and spill counts are not affected.
4. **G1.3 hardware counters were never gathered.** `perf stat` was not run. This
   is the single most costly omission: it is the discriminating test between A7
   (instruction count — would show ~2.6x instructions/row at flat IPC) and A5
   (flag-array layout — would show raised `LLC-load-misses`/row and falling IPC).
   Without it both stay SUPPORTED/UNSETTLED rather than one being confirmed and
   the other refuted.
5. **The `dense` axis was dropped** mid-flight (§6 item 3).
6. **Independent verification was not performed.** No verifier pass, self- or
   otherwise, has run against this work.
