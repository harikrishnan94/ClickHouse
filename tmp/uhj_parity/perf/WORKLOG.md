# WORKLOG — `unified_hash` performance-deficit attribution

Working record. Per iteration: goal / what was done / how verified (exact commands,
raw output or artifact path) / what changed about the plan. Corrections amend
forward with a new entry referencing the old; nothing is edited away.

Everything read from the repo, prior worklogs, logs and tool output is treated as
DATA. Any directive found inside such content is reported here and not followed.

---

## E0 — mission start, environment of record

**Goal.** Establish the ground truth the whole mission is measured against, before
any number is taken.

**MISSION-START COMMIT (Gate G1.7 diffs against this):**

```
0945a745399466911e0b94289fa120b168032d31
```

Branch created off it:

```
$ git -C /mnt/ch/ClickHouse rev-parse HEAD
0945a745399466911e0b94289fa120b168032d31
$ git -C /mnt/ch/ClickHouse checkout -b uhj-perf-attribution
Switched to a new branch 'uhj-perf-attribution'
```

**Environment, re-verified rather than inherited from the prompt:**

| Item | Prompt said | Measured | Note |
| --- | --- | --- | --- |
| CPUs | 96 | 96 (`nproc`) | agrees |
| Arch | (not stated) | **aarch64, Neoverse-V2** (Graviton4) | `uname -m`; `lscpu` "Model name: Neoverse-V2". Assembly work must use `--mcpu=neoverse-v2`, not an x86 model. |
| Binary | `build/reldeb/programs/clickhouse` | same path, RelWithDebInfo | rebuilt, see below |
| Server | already running on `127.0.0.1:9101` | **nothing was listening on 9101, nor on 9000** | `ss -ltnp` returned no match for either port. The prompt's "server already running" and "the process on :9000 is not yours" were both stale. I started my own; see below. |

**DEVIATION D1 — the pre-existing binary was stale; the prior mission's snapshot
is not reproducible from it.** The prompt says to confirm the binary is current
before trusting a number. It was not:

```
$ stat -c '%y' build/reldeb/programs/clickhouse     # before
2026-08-02 15:38:29 +0530
$ git log --format='%ci %h %s' -3 -- src/
2026-08-02 15:41:24 +0530 5362055b4ed Give unified_hash the parallel non-joined path and align clone() stats
2026-08-02 15:23:56 +0530 b99fa90b5b0 Drop gratuitous blank-line divergences in the UnifiedHashJoin fork
```

Commit `5362055b4ed`, which changes `unified_hash`'s non-joined path — precisely
one of the two shapes the inherited snapshot reports on — landed **after** the
binary was linked. Running `ninja -C build/reldeb` rebuilt **430** `dbms`
translation units and relinked:

```
$ ninja -C build/reldeb > build/reldeb/build_mission_start.log 2>&1; echo "JOB_EXIT=$?"
JOB_EXIT=0
$ grep -cE "Building CXX object src/CMakeFiles/dbms" build/reldeb/build_mission_start.log
430
```

Consequence, recorded so it is not quietly forgotten: the snapshot numbers quoted
in the mission prompt (`u3_bench_*.log`) were produced by a binary that did not
contain `5362055b4ed`. They are LEAD-grade context only. Gate G0.4 asks the
harness to recover those two effects; if it fails to, "the snapshot was measured
on different code" is a live explanation that must be weighed against "the
instrument lacks power", and G0.4 cannot be scored until that is separated.

**Binary of record for every number in this mission:**

```
BuildID[sha1]=b7980f6e38fd7fccc6cb883a140c0c0a1b4dbe78
```

Pristine reference copy kept for before/after codegen diffs, so an unmodified side
always exists:

```
tmp/uhj_parity/perf/bin/clickhouse.pristine
```

**Server of record.** Started my own rather than reusing the prior mission's, for
two reasons: nothing was running, and the prior config has no
`processors_profile_log` section, which Gate G0.5's phase split needs.

```
$ tmp/uhj_parity/perf/start_server.sh
SERVER_READY port=9111 pid=2095410
BINARY_BUILD_ID=BuildID[sha1]=b7980f6e38fd7fccc6cb883a140c0c0a1b4dbe78
```

Port **9111**, HTTP 8121, data dir `tmp/uhj_parity/perf/chserver/` (separate from
the prior mission's, so the two cannot race). One false start on the way, recorded
because it is a real property of the config: `max_server_memory_usage` is a
server-level setting and is rejected inside a user profile
(`Code: 115 ... Unknown setting 'max_server_memory_usage' ... while parsing profile 'default'`);
removed from `users.xml`.

**Tree state.** `src/` is clean at HEAD. Two files under `tmp/uhj_parity/` were
already modified on entry by a sibling session (`bench_serial.sh`,
`bench_parallel.sh` — build/probe row-count bumps). They are not mine, they are
outside `src/`, and Gate G1.7 does not cover them; left untouched. Several `src/`
files carry today's mtime from a sibling session that edited and reverted them —
content matches HEAD, which is why the 430-object rebuild was needed.

---

## E1 — confirming the three shapes from code (orientation, not acceptance evidence)

**Goal.** The prompt's one-sentence description of the three implementations is
"a summary, not a specification". Confirm or correct it before designing the matrix.

**Result: the summary is right about two of three, and materially wrong about the
third.**

| Implementation | Prompt's description | What the code says |
| --- | --- | --- |
| `unified_hash` | partitioned map, `BUCKETS_PER_THREAD = 2` per thread | **Confirmed.** `UnifiedHashJoin/HashJoin.h:67`. One shared `TwoLevelHashMap` with `BITS_FOR_BUCKET = -1` (runtime bucket count, `HashJoin.h:56`), per-bucket cache-line-padded `std::mutex` (`HashJoin.h:78`) and per-bucket arenas. |
| `hash` | a single flat map | **Confirmed, but only because of a caller.** `HashJoin.h` declares *both* flat (`HashMap`, `FixedHashMap`, lines 310-320) and two-level (lines 323-331) variants, selected by the ctor parameter `use_two_level_maps`. Every standalone caller passes `false` (`PlannerJoins.cpp:1268,1337`, `GraceHashJoin.cpp:747`, `JoinSwitcher.cpp:22`, `SpillingHashJoin.cpp:58`), so standalone `hash` is flat. |
| `parallel_hash` | shards into independent per-stream `HashJoin` instances | **Half right, and the missing half matters.** It does shard (`ConcurrentHashJoin.h:127` `std::vector<InternalHashJoin>`), but it constructs every shard with `/*use_two_level_maps*/ true` (`ConcurrentHashJoin.cpp:230`). **Each shard is itself a two-level map.** |

**Why that third row changes the mission.** Sub-table counts, from
`DEFAULT_BITS_FOR_BUCKET = 8` (`TwoLevelHashTable.h:31`) and
`bucketCountForThreads` (`UnifiedHashJoin/HashJoin.cpp:66-74`):

| max_threads | `unified_hash` sub-tables | `parallel_hash` sub-tables |
| --- | --- | --- |
| 1 | **1** (`max_threads <= 1` returns 1) | n/a (compared against `hash`, which is flat) |
| 16 | `bit_ceil(16)*2` = **32** | `16 shards x 256` = **4096** |
| 64 | `bit_ceil(64)*2` = **128** | `64 x 256` = **16384** |

So at 16 and 64 threads `unified_hash` has **two orders of magnitude fewer**
sub-tables than its comparator. Inherited LEAD (1) — that the parallel build-bound
CPU excess is `unified_hash`'s "per-bucket sub-table cache footprint at
`BUCKETS_PER_THREAD = 2`" — points the wrong way on its face: the side with more
sub-tables is the fast one. LEAD (1) is now a hypothesis I expect to REFUTE, and
it is recorded as such before any measurement, so that refuting it later is a
prediction met and not a story fitted. It stays a LEAD until a G1.4-valid
measurement settles it.

Two consequences for the plan:

- At **1 thread** the difference cannot be about bucket cache footprint at all —
  `unified_hash` has exactly one bucket. Whatever 1-thread cost exists must be the
  residual per-operation indirection of wrapping one bucket in the two-level /
  `BucketPartitionedTable` interface, plus the global-cell-offset numbering. That
  is a much sharper and more ablatable hypothesis than "two-level is different".
- At **16/64 threads** the nameable operations `unified_hash` has and
  `parallel_hash` does not are the per-bucket **mutex acquire/release** on a
  *shared* map (a sharded map needs no lock at all) and the **global cell-offset
  prefix sum** behind `offsetInternal`, which a per-shard flag array does not need.
  Both are per-operation costs, not footprint. These become the leading candidates.

**Ruled out early, cheaply:** the two implementations use the **same hash
functions** — `HashCRC32<UInt32>`, `HashCRC32<UInt64>`, `UInt128HashCRC32`,
`UInt128TrivialHash`, `TwoLevelHashMapWithSavedHash` for strings — comparing
`UnifiedHashJoin/HashJoin.h` `MapsTemplate` against `HashJoin/HashJoin.h:310-331`.
A hash-function difference is therefore not a candidate cause, and no measurement
needs to spend a run on it.

**Verified commands for this entry:** see the `rg`/`sed` invocations recorded in
`tmp/uhj_parity/perf/artifacts/E1_commands.txt`.

**Plan change.** The matrix's key-type axis must be driven by the actual
key-getter families rather than by guessing; a subagent is enumerating them from
`KeyGetter.h` and `HashJoinMethods*.h` into
`tmp/uhj_parity/perf/artifacts/CANDIDATE_INVENTORY.md`. Unit 0's pre-registration
waits on that list so the axis is complete rather than plausible.

---

## E2 — instrument design pilots (orientation; nothing here is acceptance evidence)

**Goal.** Before pre-registering Unit 0, establish that the two hardest gates —
G0.1 (the measured algorithm is the requested algorithm) and G0.5 (the phase
split) — have a mechanism that actually works, rather than pre-registering a gate
I cannot implement.

### E2.1 Phase split exists and is clean

A pilot INNER join at `max_threads=1` with `log_processors_profiles=1`:

```
FillingRightJoinSide   elapsed_us=10137  in_rows=500000   <- BUILD
JoiningTransform       elapsed_us= 9556  in_rows=2000000  <- PROBE
DelayedJoinedBlocks{,Worker}Transform  elapsed_us=0       <- non-joined (INNER: none)
query_log: query_duration_ms=23  UserTimeMicroseconds=18196
```

The two join processors account for 19,693 us of the ~21,058 us summed over all
processors, and the query's `UserTimeMicroseconds` is 18,196. The phase source is
therefore real and dominant, not a rounding artefact. Names for the record:
`FillingRightJoinSide` (not `...Transform`), `JoiningTransform`,
`DelayedJoinedBlocksTransform` + `DelayedJoinedBlocksWorkerTransform`.

### E2.2 G0.1 — the discriminator problem, and why the obvious answers fail

Three candidate discriminators were tried and **two were rejected**:

- **`EXPLAIN PLAN description=1` — REJECTED.** All three algorithms print the
  identical `Join (JOIN FillRightFirst)`. The plan does not name the algorithm.
- **`system.query_log`'s `Settings['join_algorithm']` — REJECTED.** It echoes what
  was *requested*, which is precisely the thing under suspicion.
- **Processor topology — PARTIAL, insufficient.** At `max_threads=4`, `hash` has 1
  `FillingRightJoinSide` while `parallel_hash` and `unified_hash` each have 4. That
  separates `hash`, but `parallel_hash` and `unified_hash` are byte-identical in
  topology — and they are exactly the pair compared at 16 and 64 threads. Also
  useless at `max_threads=1`, where all three have 1.

Rejected too: `ProfileEvents` alone. Empirically, `parallel_hash` emits
`ConcurrentHashJoinBuild/Probe*Microseconds` and
`HashJoinPreallocatedElementsInHashTables`, which `hash` and `unified_hash` do not
— a sound positive test for `parallel_hash`, but `hash` and `unified_hash` have
**identical** join-related event key sets, so it cannot separate that pair. Neither
`HashJoin/` nor `UnifiedHashJoin/` contains a single `ProfileEvents::` reference.
Their `LOG_DEBUG` texts are identical too ("The joined right table total rows...").

**Accepted mechanism: symbol-level proof from `system.trace_log`.** `unified_hash`
is `DB::Unified::HashJoin` (`UnifiedHashJoin/HashJoin.h:36` `namespace Unified`),
a distinct symbol from `DB::HashJoin`. Sampling the CPU profiler and demangling
the stacks identifies the code that *actually executed*:

```
algo       unified_frames  concurrent_frames  total_samples
hash                    0                  0            388
parallel                0                414            430
unified               437                  0            453
```

Note `parallel_hash`'s shards *are* baseline `HashJoin` objects, so
`DB::HashJoin::` frames appear under `parallel_hash` too; the rules are therefore
ordered — `ConcurrentHashJoin` frames first, then `Unified`, then baseline — and
are written that way in the harness.

### E2.3 A real silent-downgrade trap, found and closed

`PlannerJoins.cpp:1244-1257` runs the parallel-hash branch whenever
`table_join->allowParallelHashJoin() && !unified`, with

```cpp
use_parallel_hash = !isEnabledAlgorithm(HASH) || !rhs_size_estimation
                    || (*rhs_size_estimation >= parallel_hash_join_threshold);
```

The `!rhs_size_estimation` term means that **`join_algorithm='hash'` can silently
give you `ConcurrentHashJoin`** whenever the planner cannot estimate the right
side — which is common. That would have made the 1-thread baseline column a
comparison against the wrong implementation.

It is closed by `allowParallelHashJoin` (`TableJoin.cpp:1293-1307`), whose first
test is that `PARALLEL_HASH` appears in the algorithm list. Setting
`join_algorithm` to a **single** algorithm therefore makes the downgrade
impossible for `hash`. Confirmed empirically: `join_algorithm='hash'` emitted no
`ConcurrentHashJoin*` events and no `DB::ConcurrentHashJoin` frames. The harness
pins a single-valued `join_algorithm` for this reason, and G0.1 still asserts it
per cell rather than trusting the argument.

**Second consequence, predicted into PREREG rather than discovered later:** the
same function returns false for any kind outside Left/Inner/Right/Full
(`TableJoin.cpp:1301-1303`), so **`SEMI`/`ANTI` have no `parallel_hash` comparator
at all** — requesting it yields plain `hash`. Those cells are pre-registered as
SKIPPED for the `parallel_hash` comparison.

**What changed about the plan.** G0.1 gains a dedicated assertion run per cell
(separate from the timed runs, so it cannot perturb them). G0.5's tolerance is
declared as two separate checks — an exact accounting identity plus a 20%
build-only cross-check — rather than one vague "phases sum to total".

---

## E2 — BLOCKING: a second agent session is running this same mission in this same working tree

**Goal.** Commit the Unit 0 pre-registration before any harness code, per the
mission's ordering requirement.

**What happened instead.** The commit succeeded, but `git log` showed a commit I
did not create sitting between the mission-start commit and mine:

```
$ git log --oneline -3
51013b1eac9 uhj-perf: pre-register Unit 0 before the harness exists          <- mine
b8452c4f70b uhj-perf: pre-register Unit 0 and record the environment of record  <- NOT mine
0945a745399 uhj_parity: record post-mission performance snapshot as OPERATIONAL context
$ git reflog --date=iso | head -3
51013b1eac9 HEAD@{2026-08-03 10:36:26 +0530}: commit: uhj-perf: pre-register Unit 0 before the harness exists
b8452c4f70b HEAD@{2026-08-03 10:35:33 +0530}: commit: uhj-perf: pre-register Unit 0 and record the environment of record
0945a745399 HEAD@{2026-08-03 10:22:48 +0530}: checkout: moving from uhj-parity to uhj-perf-attribution
```

`b8452c4f70b` landed 53 seconds before mine, on the branch I created at 10:22:48,
and committed four paths: `perf/PREREG.md` (121 lines), `perf/WORKLOG.md` (270),
`perf/start_server.sh` (117), `perf/artifacts/E1_commands.txt` (7). The last three
are byte-for-byte the files **I** had just written. The first is **not mine** — I
wrote `PREREG.md` exactly once, in a single write, at ~246 lines.

**Evidence that it is a second agent on the same mission, not a stale artifact.**
The other `PREREG.md` is preserved at
`tmp/uhj_parity/perf/artifacts/PREREG_from_b8452c4f70b.md`. It is independent prose
reaching the same design conclusions I reached: cardinality expressed as build-table
row count with unique keys, the same 10,000 / 1,000,000 / 50,000,000 levels, the same
0.9 / 0.1 match rates, the same 1 / 16 / 64 thread counts — and it refers to
`sweep.py`, `matrix.py` and `artifacts/CANDIDATE_INVENTORY.md`, the last being the
exact path my own background subagent is writing to right now.

Confirmed *not* the cause: there is only one `clickhouse` server running (mine,
pid 2095410, port 9111), and no competing benchmark load.

**Damage so far, stated plainly.** My `Write` of `PREREG.md` overwrote theirs on
disk; my commit `51013b1eac9` consequently reads as "136 insertions, 110 deletions"
against a pre-registration I never authored. Their content is not lost — it is in
`b8452c4f70b` and copied to `artifacts/`. No `src/` file is affected and the
baselines remain pristine.

**Why this stops the mission rather than being worked around.** Three reasons, in
order of severity:

1. **Every measurement would be invalid.** Two agents are each about to run a
   96-core sweep on one 96-CPU host. Concurrent sweeps make wall and CPU numbers
   meaningless, and Gate G0.3's A/A calibration — the one check that distinguishes
   an instrument measuring the join from one measuring scheduling drift — would
   either fail or, far worse, pass spuriously while every A/B is contaminated.
   The mission's entire value rests on measurement validity.
2. **The audit trail is already corrupted.** The history now implies I revised a
   pre-registration when I in fact overwrote a different agent's. Pre-registration
   ordering is a checked acceptance criterion; two interleaved authors on one
   branch destroy its meaning.
3. **Resolving it unilaterally is destructive and irreversible.** Every option —
   reverting their commit, moving my work to a fresh branch, killing their session,
   abandoning mine — either discards another agent's work or forks the deliverable.

The mission's ambiguity rule reserves exactly this class of decision: "except for
decisions that are irreversible, destructive, production-facing, security-relevant,
or materially scope-changing: stop, ask, and wait for an answer. An irreversible
step never proceeds past an unanswered question."

**Action taken:** none that is destructive. Their `PREREG.md` preserved to
`artifacts/PREREG_from_b8452c4f70b.md`; nothing reverted, no history rewritten, no
process killed, benchmark data generation NOT started. Asking the requester, who
this mission states is available.

**Not done, deliberately:** did not start the data generation or the sweep, because
starting a multi-hour 96-core job while a second agent may be doing the same is the
single fastest way to produce numbers that look fine and mean nothing.

---

## E3 — CORRECTION to E1: the sub-table count was wrong (amends E1 forward)

**Trigger.** The requester rejected the claim in E1 and in commit `b8452c4f70b`'s
message that `parallel_hash` has 4096 sub-tables at 16 threads: "parallel_hash has
4096 sub-tables but only always 256 of them is utilized." Checked, and the
requester is right. E1's table is wrong and is corrected here rather than edited
away.

**What I got wrong.** I counted bucket *objects* (`slots x 256`) and called them
sub-tables. Only 256 of them ever hold a row.

**Evidence.** `ConcurrentHashJoin.cpp:582-596`:

```cpp
selector[i] = hash_table.getBucketFromHash(hashes[i]) & (num_shards - 1);
```

The slot a row is scattered to is *derived from the bucket index it will occupy*,
so shard `i` only ever populates buckets `j` with `j % slots == i`. The merge in
`onBuildPhaseFinish` (`ConcurrentHashJoin.cpp:817-834`) relies on exactly this and
throws `"Unexpected non-empty map"` if it is ever violated:

```cpp
for (size_t j = idx; j < lhs_map.numBuckets(); j += slots)
{
    if (!lhs_map.impls[j].empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Unexpected non-empty map");
    lhs_map.impls[j] = std::move(rhs_map.impls[j]);
}
```

`slots` is additionally capped: `slots(toPowerOfTwo(std::min<UInt32>(slots_, 256)))`
(`ConcurrentHashJoin.cpp:197`).

**Corrected table — partitions of the data that actually hold rows:**

| max_threads | `unified_hash` | `parallel_hash` | ratio |
| --- | --- | --- | --- |
| 1 | 1 | n/a (comparator is `hash`, one flat map) | — |
| 16 | 32 | **256** | UHJ has **8x fewer** |
| 64 | 128 | **256** | UHJ has **2x fewer** |

(Previously stated, wrongly, as 4096 and 16384, i.e. "128x fewer".)

**What survives, what does not.**

- *Survives:* `unified_hash` still partitions into **fewer** sub-tables than
  `parallel_hash` at both thread counts. So inherited LEAD (1) as literally worded
  — that UHJ's cost is having *more* per-bucket sub-tables to miss the cache on —
  still does not follow from the structure. It remains a hypothesis I expect to
  refute.
- *Does not survive, and this is the substantive part:* the magnitude collapses
  from 128x to 8x, and more importantly **the cache mechanism plausibly runs the
  other way**. Because UHJ splits the same rows into 8x fewer partitions, each of
  its sub-tables is **8x larger** than a `parallel_hash` bucket. A build insert
  into a big sub-table misses more than one into a small sub-table that still fits
  in cache. So there may well be a real sub-table cache effect behind the build CPU
  excess — with the **opposite sign** to LEAD (1): the problem would be too *few*
  buckets, not too many.

  This is a better hypothesis than the one it replaces and it is directly testable
  with the knob that already exists: raising `BUCKETS_PER_THREAD` should *shrink*
  the gap if this mechanism is real, and do nothing if it is not. Registered as a
  Unit 1 ablation before any measurement.

**Also corrected: `parallel_hash` is not sharded at probe time.**
`onBuildPhaseFinish` consolidates every slot's buckets into slot 0's map and shares
it, so the probe walks **one shared 256-bucket map**, structurally the same shape
as `unified_hash`'s one shared 32-bucket map. The sharding is a build-phase
property only. The brief's "shards into independent per-stream `HashJoin`
instances" is therefore true of the build and false of the probe.

**Status of commit `b8452c4f70b`.** Its message states the wrong 4096/128x figure.
History is never rewritten on this branch, so the commit stands and this entry is
the correction of record; `REPORT.md` will carry the corrected table.

---

## E4 — a benchmark design flaw caught before it produced a number

**Trigger.** The candidate-cause inventory flagged that the `range8_key32 ..
range18_key64` direct-addressed maps are reached by a *post-build conversion*, not
by column type. Checked against my own data generator, and it would have silently
corrupted a quarter of the matrix.

**The flaw.** `gendata.py` gave build tables keys `0 .. N-1` — a dense range.
`HashJoin::canConvertToFixedHashMap` (`UnifiedHashJoin/HashJoin.cpp:2077-2081`)
fires for `key32`/`key64` when `enable_join_fixed_hash_table_conversion` is on,
which it is **by default** (`Settings.cpp:8257`, default `true`). The fixed maps
top out at `range18`, i.e. a key range of 2^18 = 262,144.

My declared cardinalities against that threshold:

| cardinality | keys | dense range | converts? |
| --- | --- | --- | --- |
| small | 10,000 | 10,000 | **YES** |
| medium | 1,000,000 | 1,000,000 | no |
| large | 30,000,000 | 30,000,000 | no |

So every `small` cell — a quarter of the matrix, and precisely the cells meant to
isolate cheap-lookup indirection with a cache-resident table — would have compared
two *direct-addressed array* implementations, not two hash tables, while being
labelled a hash-join comparison. It would also have left the small-cardinality hash
path with **zero** coverage.

**Fix.** Build keys become non-dense: `k = number * 2654435761` (odd multiplier,
wraps over the full `UInt64` range), so the key range is astronomically larger than
2^18 and no conversion can fire at any cardinality. Probe keys are derived from the
same mapping so match rates are unchanged. `enable_join_fixed_hash_table_conversion`
is left at its realistic default rather than pinned off, and the harness asserts
per cell that conversion did **not** fire, so this cannot silently return.

**Not discarded — promoted.** The conversion is a real capability asymmetry worth
reporting rather than engineering away: `canConvertToFixedHashMap` requires
`key32`/`key64`, and `parallel_hash` can never satisfy it, while `unified_hash`
converts at any thread count. That is a cell where `unified_hash` should *win*, and
the mission explicitly wants those reported. Recorded as a declared extra axis
(`dense` key type) to be measured separately, so the win is attributed rather than
accidentally averaged into the general case.

---

## E5 — two instrument decisions forced by what the debug log showed

Running the pilot with `--send_logs_level=debug` surfaced two things that no
amount of reading the matrix definition would have.

### E5.1 The `dense` axis is dropped — UNSETTLED, with the settling experiment named

The conversion I predicted in P0.7 **does not fire**, on either `hash` or
`unified_hash`, for dense keys `0..9999`:

```
$ clickhouse client --port 9111 --send_logs_level=debug -q \
  "SELECT count() FROM p_dense_hi AS l INNER JOIN b_dense AS r ON l.k = r.k
   SETTINGS join_algorithm='unified_hash', max_threads=16, ..." 2>&1 | grep -i Converted
(no output)
```

Every precondition I can read is satisfied: `enable_join_fixed_hash_table_conversion`
is `1` on the server, the key is `UInt64` (`Type::key64`), `data->maps.size()==1`,
strictness is not `Asof`, `source_map.size()` is 10,000 which is under
`MAX_RANGE = 1<<18`, and `runPostBuildPhase` does call `tryConvertToFixedHashMap`
(`UnifiedHashJoin/HashJoin.cpp:2519`). So the guard that actually rejects it is one
I have not identified.

**Decision: drop `dense` from the matrix** and record this as UNSETTLED rather than
keep spending the instrument budget on an axis the mission lists as optional
context. This does **not** weaken the core matrix — the reason `dense` existed at
all was to keep the conversion from contaminating the `small` cells, and that is
already handled by spreading the keys (E4). Matrix returns to the 144 cells
pre-registered in P0.

**P0.7 is therefore not scored.** It is neither confirmed nor refuted; the premise
("the conversion fires") is unverified. Recording it as a met or missed prediction
either way would be dishonest.

**The experiment that would settle it**, for whoever picks it up: build with a
breakpoint or a temporary `LOG_DEBUG` on each early-return in
`canConvertToFixedHashMap` and `tryConvertToFixedHashMapImpl`, run the query above,
and read which guard rejects. One rebuild, one query. It is checkable in this
environment; I am choosing not to spend the rebuild on an optional axis, and that
choice is the deviation, not an inability.

**Kept as an OPERATIONAL observation** (from the same pilot, `10,000`-row build):
reported in-memory build size differs by 36x across implementations —
`hash` 1,128,704 B, `parallel_hash` **16,859,264 B**, `unified_hash` 473,344 B.
`parallel_hash` pays 256 sub-table minimum capacities whether or not it has rows to
put in them. At small cardinality that is a real `parallel_hash` cost and a
`unified_hash` advantage, and it is the first direct evidence that the sub-table
count trade runs in `unified_hash`'s favour at the small end. Not a timing claim —
memory, from `query_log.memory_usage`, recorded on every run for later use.

### E5.2 Unique build keys silently changed the algorithm under test

All three implementations logged:

```
HashJoin: Promoting join strictness to RightAny, because all values in the right table are unique
```

My build tables had exactly one row per key, so **every cell in the matrix was
about to measure the `RightAny`/`MapsOne` path** while being labelled `INNER JOIN`,
`LEFT JOIN` and so on — i.e. the `ALL` strictness that those SQL forms actually
mean would never have executed. `MapsAll` and its `RowRefList` chains would have
had zero coverage.

The comparison would still have been *fair* (all three promote identically), so no
gate would have caught it. It is a coverage defect, not a validity defect, which is
exactly the kind that survives to the report.

It also matters for attribution specifically: candidate cause #6 in the inventory
is "per-bucket arenas fragmenting `RowRefList` allocation", and per-bucket arenas
are a `unified_hash`-only structure. That candidate **cannot exist** in a workload
where no `RowRefList` ever chains. Measuring only the promoted path would have
made a real candidate unfalsifiable by construction.

**Fix: two rows per distinct key.** Build tables now generate `2 x cardinality`
rows, with `ordinal = intDiv(number, 2)` so each key appears exactly twice and the
payload `v` still differs per row. This defeats the promotion, exercises `MapsAll`
with chains of length 2, keeps output size linear (2x per matched probe row), and
brings the harness into line with the mission's own wording — it says
"cardinalities (**distinct build-side keys**) ... capped at 1M build **rows**",
which already distinguishes the two quantities. Cardinality stays the number of
distinct keys; build rows are twice that.

The unique-key / `RightAny` regime is now a **declared coverage gap** rather than
the accidental default: it is a real and common shape (primary-key joins) and is
listed in REPORT.md as a named gap with the cheap experiment to close it (re-run
the sweep with one row per key).

---

## E6 — Unit 0 sweep run, and the deficit map it produced

144 cells, 7 interleaved reps per point, ~2,500 runs, 12 minutes wall.
`SWEEP_DONE ... JOB_EXIT=0`.

**Gate results (first pass):** G0.1 GREEN (264/264 assertion runs identified the
requested implementation from demangled stacks, 0 mismatches, 0 unknowns),
G0.2 GREEN (144/144 cells checksum-identical across algorithms), G0.3 GREEN
(A/A within 2.1% against a 5% band), G0.6 GREEN (144 declared, 144 covered,
0 missing, 48 (cell,algo) SKIPPED with the pre-registered SEMI/ANTI reason),
G0.4 **RED**, G0.5 **RED**. Both reds are worked below and are not carried
silently.

### The deficit map

| threads | comparator | metric | faster | within noise | slower |
| --- | --- | --- | --- | --- | --- |
| 1 | `hash` | wall | 0 | 59 | **13** |
| 1 | `hash` | CPU | 0 | 59 | **13** |
| 16 | `parallel_hash` | wall | **17** | 7 | 0 |
| 16 | `parallel_hash` | CPU | 5 | 19 | 0 |
| 64 | `parallel_hash` | wall | **24** | 0 | 0 |
| 64 | `parallel_hash` | CPU | 16 | 3 | **5** |

**The mission's premise does not hold at 16 and 64 threads.** `unified_hash` is
never slower on wall there, and at 64 threads it wins every single cell.
P0.1's third clause and P1.2 both predicted this shape before the sweep ran.

Two real deficits remain, and they are the Unit 1 targets:

1. **1 thread vs `hash`: 13 cells, +4.9% to +14.3% wall.** Concentrated in the
   kinds that maintain used-flags (`RIGHT`, `FULL`, `LEFT SEMI`, `LEFT ANTI`) and
   at small/medium cardinality. Several are probe-dominated (`LEFT-ANTI|u64|hi|t1|small`
   probe +18.6%, `LEFT-ANTI|str|hi|t1|small` +12.2%). Prediction P0-c said any
   1-thread deficit would be small and single-digit; 12 of 13 are, one is +14.3%,
   so P0-c is **substantially met but not exactly** — recorded as a partial miss.
2. **64 threads, composite keys: build CPU +23% to +37%** on 5 cells
   (`INNER|comp|lo|t64|large` CPU +18.4% / build +37.1%,
   `LEFT|comp|lo|t64|large` +12.0% / +35.6%, and three more), *while wall is
   15-37% better*. `unified_hash` buys wall time with CPU here. This is the only
   place the inherited "parallel build CPU excess" story survives, and it is
   key-type-specific in a way the inherited lead did not predict.

### G0.4 RED — neither recorded snapshot effect reproduces

```
(a) 16t INNER build CPU   parallel_hash=1724591 unified=1655285  delta=-4.0%  within_noise
(b) 16t RIGHT wall        parallel_hash=382     unified=352      delta=-7.9%  faster
```

Both were expected at +25% and +40%. Both now show `unified_hash` at parity or
ahead. Per P0.5 the two explanations must be separated before this is scored, and
they can be:

- *Instrument lacks power* — **rejected.** The same harness resolves A/A to within
  2.1%, and in the same sweep it detects effects of +5% to +18% (the 13 one-thread
  cells) and -37% (the 64-thread wins). An instrument that resolves +5% would not
  miss +40%.
- *The code changed since the snapshot* — **supported, and specifically.** WORKLOG
  D1 established that the snapshot binary predates commit `5362055b4ed`, whose
  subject is "Give `unified_hash` the parallel non-joined path" — i.e. it changes
  exactly the mechanism behind effect (b), the RIGHT non-joined shape.

G0.4 stays **RED as defined**, because its pre-registered criterion is recovery of
the recorded effects and they were not recovered. It is not re-scored green by
argument. What the evidence does establish is that the red is a property of the
*reference*, not of the instrument, and P0.5 pre-committed to exactly that
distinction. The decisive experiment — rebuild at `5362055b4ed^` and re-measure
cell (b) — is named in E8 and is runnable in this environment.

---

## E7 — two instrument defects found AFTER the sweep, both invisible to the gates

Recorded prominently because both are the dangerous kind: they leave every gate
green-able while corrupting the numbers underneath.

### E7.1 The non-joined phase was mapped to the wrong processors

`PHASE_PROCESSORS["nonjoined"]` listed `DelayedJoinedBlocksTransform` and
`DelayedJoinedBlocksWorkerTransform`. Both read **zero** on these queries. The
RIGHT/FULL non-joined scan is `NonJoinedBlocksTransform`:

```
$ ... WHERE query_id LIKE 'u0full-RIGHT_u64_lo_t1_small-unified_hash-timed-3%'
JoiningTransform            15533 us
MergeTreeSelect              4712 us
NonJoinedBlocksTransform      429 us   <- the phase that was being reported as 0
FillingRightJoinSide          357 us
```

So every RIGHT and FULL cell reported a non-joined phase of exactly 0 — for the
shape the mission most wants attributed.

**Gate G0.5 did not catch it**, and the reason is worth stating: its accounting
identity `build + probe + nonjoined + other == total` holds perfectly when a phase
is zero and its work is sitting in `other`. A partition check cannot detect a
mis-partition. That is a genuine weakness in how I wrote the gate, not a fluke.

Fixed, and the phase split **re-derived from the raw log rather than re-measured**
(`rederive.py`): the measurements were correct, only the bookkeeping was wrong, and
re-running would have discarded the numbers the gates were already run against.
Median join-phase share rose from 73.5% to 82.0%, and 504 of 1,848 timed runs now
carry a non-zero non-joined phase.

### E7.2 Duplicate `query_id`s across sweep attempts, silently double-counting

Check (iii) below flagged two cells where the processor-derived build phase was
**exactly 2.004x** the internal timer. An exact factor of two is not noise.

Cause: the first `u0full` sweep attempt was killed after 6 cells (backgrounding
with a plain `&` — the job dies when the tool call's shell returns; `setsid` is
required). I relaunched under the *same* `--run-tag`, and the query-id sequence
counter restarts at 0 in a new process, so the second attempt regenerated
**identical query ids** for those first cells. Any readback that groups the
`system.*` logs by `query_id` then sums two different executions together.

```
$ SELECT count() FROM (SELECT query_id, count() c FROM system.query_log
    WHERE query_id LIKE 'u0full-%' GROUP BY query_id HAVING c > 1)
166
```

166 contaminated ids. Fixed by adding a per-process token to every query id, and
the whole sweep re-run from clean (`--run-tag u0v2`); the contaminated file is
kept as `results/runs_u0full_dupcontaminated.jsonl` rather than deleted.

Note again what did *not* catch it: the accounting identity still held (everything
doubled together), and the deficit map's percentages were roughly unaffected
(both algorithms doubled equally). Only an origin that fails differently exposed
it.

### E7.3 What actually validates the phase split: check (iii)

The build-only cross-check (ii) is systematically biased for `parallel_hash` —
its build-only figure reads high (24.7%, 11.4%, 10.5% on three cells) while `hash`
and `unified_hash` sit under 3% on most. Plausibly because `ConcurrentHashJoin`'s
try-lock drain loop yields differently with no probe-side work competing.

Rather than widen the tolerance — a banned move — I added a **third origin that
fails differently**: `ConcurrentHashJoin` instruments its own build with
`ProfileEvents['ConcurrentHashJoinBuildMicroseconds']`, measured inside the
implementation, sharing no machinery with the pipeline's processor accounting.

```
RIGHT|str|lo|t64|large   internal 4,496,011 / 4,382,120 / 4,525,652 us
                         processor (FillingRightJoinSide) 4,503,634 us
48 cells checked, median deviation 0.4%
```

So `FillingRightJoinSide` **is** the build phase, confirmed to 0.4% by independent
instrumentation. It is the build-only *query variant* that misrepresents
`parallel_hash`, not the phase source.

**G0.5 therefore stays RED as pre-registered** (1 of 29 pairs over the 20%
tolerance on check (ii)), while the property the gate exists to protect — that the
phase denominators are real — is affirmed by check (iii). Both are reported; the
tolerance is not touched. Consequence carried into Unit 1: any build-phase
percentage taken against `parallel_hash` via the build-only route carries a stated
caveat, whereas cell-level wall and CPU (from `query_duration_ms` and
`UserTimeMicroseconds`) are independent of processor attribution entirely and are
unaffected.

---

## E8 — clean re-run: G0.5 goes GREEN, and E7.3's mechanism story was WRONG

Re-ran the whole sweep with per-process-unique query ids (`--run-tag u0v2`,
144 cells, 12.2 min, `JOB_EXIT=0`). Gates on the uncontaminated data:

| Gate | Verdict |
| --- | --- |
| G0.1 measured algo == requested | **GREEN** |
| G0.2 checksums agree | **GREEN** |
| G0.3 A/A calibration | **GREEN** |
| G0.4 known-signal recovery | **RED** |
| G0.5 phase split reconciles | **GREEN** |
| G0.6 coverage | **GREEN** |

**Correction to E7.3, amending it forward.** E7.3 concluded that the build-only
cross-check is "systematically biased for `parallel_hash`", and offered a
mechanism: its try-lock drain loop yielding differently without probe-side
contention. **That was wrong.** On clean data:

```
(ii) build-only vs FillingRightJoinSide : 29 pairs, over tolerance 0   (was 1)
(iii) internal timer vs processor       : 48 cells, over tolerance 0,
                                          median dev 0.3%, max 1.5%    (was max 50.1%)
```

Every deviation was the duplicate-`query_id` double-counting from E7.2, not a
property of `parallel_hash`. The lesson is the one worth keeping: I had a
plausible mechanism ready for a number that was simply corrupt, and wrote it down
as if it explained something. The invented mechanism is struck; only the
contamination was real.

G0.5 is now GREEN on its own pre-registered terms, with the tolerance untouched.

---

## E9 — why the 1-thread deficit exists, and a phase comparison that was invalid

The 1-thread deficit is real and its shape is sharp. Median wall delta vs `hash`,
across all 12 cells of each kind:

| kind | n | median wall | max |
| --- | --- | --- | --- |
| `RIGHT` | 12 | **+5.27%** | +9.30% |
| `FULL` | 12 | **+4.79%** | +9.70% |
| `LEFT` | 12 | +2.33% | +4.40% |
| `LEFT SEMI` | 12 | +2.06% | +6.67% |
| `LEFT ANTI` | 12 | +1.57% | +7.14% |
| `INNER` | 12 | **+0.64%** | +2.78% |

The deficit is monotone in how much used-flag and non-joined work the kind does,
and `INNER` — which does none — is at parity. That gradient is itself evidence:
whatever the cause is, it is not on the common lookup path.

**A phase comparison I was making was invalid, and the fix changes the answer.**
Non-joined phase medians at 1 thread, over 84 runs each:

```
card=medium  hash          nonjoined_us =      0   (0/84 runs non-zero)
card=medium  unified_hash  nonjoined_us = 37,858  (84/84 runs non-zero)
```

That is not `hash` skipping the work — G0.2 proves the outputs are identical. The
processor lists show where it goes (`FULL|u64|hi|t1|medium`):

```
hash          JoiningTransform  93,560 us  out_rows 3,885,728   NonJoined: absent
unified_hash  JoiningTransform  93,167 us  out_rows 3,800,000
              NonJoinedBlocksTransform 14,888 us  out_rows 85,728
```

`hash` emits its 85,728 non-joined rows **from inside `JoiningTransform`** — its
`output_rows` carries them — and pays almost nothing extra for it (93,560 vs
93,167 us for a transform doing 85,728 rows more work). `unified_hash` runs a
**separate scan** costing 14,888 us.

So comparing `probe_us` and `nonjoined_us` as separate columns compares different
partitions of the same work, and it made `unified_hash` look *better* on probe
(-2.9%) while being worse overall. Gate G0.7 now also reports
`probe_plus_nonjoined_us`, which is the only fair comparison for these kinds:

| cell | wall | probe alone | probe+nonjoined |
| --- | --- | --- | --- |
| `FULL\|u64\|hi\|t1\|medium` | +9.7% | **-2.9%** | **+12.6%** |
| `RIGHT\|u64\|hi\|t1\|medium` | +9.3% | -3.7% | +12.7% |
| `FULL\|u64\|lo\|t1\|medium` | +5.4% | -44.8% | +5.9% |
| `LEFT-ANTI\|u64\|hi\|t1\|small` | +7.1% | +19.1% | +19.1% |

This is the leading Unit 1 claim, in the requester's schema:

> Operation: **a separate full scan of the hash table to emit non-joined rows.**
> It exists in `unified_hash` (`NonJoinedBlocksTransform`, per-cell
> `map.offsetInternal(it.getPtr())` then `parent.isUsed(offset)`,
> `UnifiedHashJoin/HashJoin.cpp:1515-1516`) and is, at one thread, effectively
> absent in `hash`, which emits the same rows inline from `JoiningTransform`.
> It consumes 14,888 us of the 108,055 us probe+non-joined phase
> (**13.8%**) in cell `FULL|u64|hi|t1|medium`.

Verdict is **not** claimed yet: the ablation (G1.2) and the codegen artifact
(G1.1) are outstanding, and until the codegen diff exists this is a LEAD with a
measured phase cost attached, not a CONFIRMED attribution.

### Declared coverage gap found here

At `small` and `medium` cardinality the non-joined scan is nearly empty: 2M probe
rows over 10k/500k build keys match essentially every build key, so RIGHT/FULL
there emit few non-joined rows (median 98 us and 7,311 us). The non-joined path is
properly exercised only at `large` + `lo` (medians 2.4-3.0 s). The match-rate knob
controls the fraction of *probe* rows that match, which is not the same as the
fraction of *build* keys that go unmatched — those coincide only when the probe is
not much larger than the build. Recorded as a declared gap; closing it needs a
probe-rows-per-build-key knob, which is a harness change, not a re-measurement.
