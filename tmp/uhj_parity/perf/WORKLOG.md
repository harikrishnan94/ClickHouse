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
