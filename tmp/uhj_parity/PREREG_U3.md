# PREREG_U3 — pre-registration for Unit 2 alignments

Gate U2-PRE: each entry is committed **before** the commit that implements it, so `git log` order
is checkable. Each entry states the expected *structural* outcome and the condition that would
refute it. No entry uses a performance outcome — performance is not an acceptance criterion in this
mission.

---

## M2 — restore baseline `finalizePerRowFlags(JoinUsedFlags & source, size_t)` signature

**Divergence.** `UnifiedHashJoin/JoinUsedFlags.h:93` declares `finalizePerRowFlags(size_t num_blocks)`
and merges from `this`. Baseline `HashJoin/JoinUsedFlags.h:90` declares
`finalizePerRowFlags(JoinUsedFlags & source, size_t num_blocks)` and merges from `source`.

**Why avoidable.** The `source` parameter exists so `ConcurrentHashJoin.cpp:893` can merge a shard's
flags into a common object. UHJ has no shards, but the baseline `hash` path itself calls it as a
self-merge (`HashJoin.cpp:2380`: `used_flags->finalizePerRowFlags(*used_flags, ...)`). Nothing about
bucketing forces the signature change.

**Change.** Restore the baseline signature and parameter use in
`UnifiedHashJoin/JoinUsedFlags.h`; update the single call site
`UnifiedHashJoin/HashJoin.cpp:2407` to the baseline's self-merge shape.

**Expected structural outcome.** The `finalizePerRowFlags` hunk disappears from the
`JoinUsedFlags.h` section of `U3_normdiff.txt`; residual line count for that file drops.

**Refute condition.** The hunk survives normalization, or the call site cannot be written in the
baseline shape without a further UHJ-only change.

---

## M3 — run `doDebugAsserts()` on the public byte-count path

**Divergence.** Baseline `HashJoin.cpp:533-538` calls `doDebugAsserts()` at the top of
`getTotalByteCount()`. UHJ's `getTotalByteCount()` (`HashJoin.cpp:666-670`) takes `blocks_mutex` and
delegates to `getTotalByteCountUnlocked()`, which does not assert. External callers therefore lose
the accounting check the baseline gives them.

**Why avoidable.** The assert is unsafe only *without* the lock. UHJ's public entry point already
holds `blocks_mutex`, so it can assert exactly where the baseline does. Bucketing does not force the
omission.

**Change.** Call `doDebugAsserts()` in UHJ's public `getTotalByteCount()` under the lock it holds,
leaving `getTotalByteCountUnlocked()` (the build-hot-path variant) unchanged.

**Expected structural outcome.** `rg -n "doDebugAsserts" src/Interpreters/UnifiedHashJoin/HashJoin.cpp`
shows a call reachable from `getTotalByteCount`, matching the baseline's call site set.

**Refute condition.** The assert fires under normal operation, indicating UHJ's accounting genuinely
cannot satisfy the baseline invariant — that would make this UNSETTLED, not aligned.

---

## M4 — rename `UNIFIED_KEYGETTER_RANGE_IMPL` back to `KEYGETTER_RANGE_IMPL`

**Divergence.** `UnifiedHashJoin/KeyGetter.h` renames the macro; baseline uses
`KEYGETTER_RANGE_IMPL` (`HashJoin/KeyGetter.h:270`).

**Why avoidable.** Verified the macro is `#undef`'d immediately after use in *both* copies
(`HashJoin/KeyGetter.h:284`), so no redefinition clash is possible. This is unlike
`APPLY_FOR_JOIN_VARIANTS`, which is not `#undef`'d and is used from four other translation units —
that one stays renamed (FORK-MECHANICAL F1).

**Change.** Rename the macro and its 8 invocations and the `#undef` back to the baseline spelling.

**Expected structural outcome.** The `KEYGETTER_RANGE_IMPL` hunks disappear from the `KeyGetter.h`
section of `U3_normdiff.txt`; the file still compiles, proving no clash.

**Refute condition.** A redefinition or "macro redefined" diagnostic appears at build time, which
would reclassify this as FORK-MECHANICAL.

---

## M6 — admit `Unified::HashJoin` to the `optimize_read_in_order` gate

**Divergence.** `ExpressionAnalyzer.cpp:2255`:
```cpp
join_allow_read_in_order = typeid_cast<HashJoin *>(join.get()) && !join_has_delayed_stream;
```
A `Unified::HashJoin` is a distinct type, so the cast fails and `unified_hash` silently loses
`optimize_read_in_order` in the legacy-analyzer path. `hash` keeps it.

**Why avoidable.** Nothing about bucketing affects left-side block ordering. UHJ's probe path
(`joinBlock` -> `joinBlockImpl`) is textually the baseline's, and `pipelineType()` is identical, so
UHJ preserves left block order exactly as `hash` does. `supportParallelJoin()==true` resizes the
**right** stream only, which read-in-order does not depend on. The `join_has_delayed_stream`
conjunct is computed from `needStreamWithNonJoinedRows()` and is algorithm-independent, so it still
gates UHJ exactly as it gates `hash`.

**Baseline choice.** Two baseline-faithful shapes exist: `hash` enables read-in-order,
`parallel_hash` does not (a `ConcurrentHashJoin` also fails the cast). Per the mission's
reconciling rule this is the serial in-memory join, so `hash` is the shape to match.

**Change.** Extend the gate to accept `Unified::HashJoin` as well.

**Expected structural outcome.** `EXPLAIN PLAN` for an ORDER-BY-on-primary-key query joined with
`join_algorithm='unified_hash'` under the legacy analyzer shows the same read-in-order plan shape as
the identical query with `join_algorithm='hash'`, and both return identical rows.

**Refute condition.** Result rows differ between `hash` and `unified_hash` for that query, or the
plan shows a sort that `hash` does not have — either would mean UHJ does not in fact preserve the
ordering property the gate assumes, making this UNSETTLED rather than aligned.

---

## M1 — port the bucket-partitioned parallel non-joined path

**User decision (2026-08-02):** fix M1. Recorded before implementing.

**Divergence.** `parallel_hash` emits non-joined RIGHT/FULL rows from several pipeline streams;
`unified_hash` emits them from one. UHJ overrides neither
`supportParallelNonJoinedBlocksProcessing()` (inherits `false`, `IJoin.h:158`) nor the 5-arg
`getNonJoinedBlocks(..., bucket_idx, num_buckets)` (inherits the partition-ignoring default,
`IJoin.h:170`), and its `NotJoinedHash` has no `isBucketInRange`/`isBlockInRange` filter and no
`bucket_idx != 0` nullmap guard.

**Why avoidable, not EXCLUDED.** `ConcurrentHashJoin.cpp:555-560` ("Two-level maps: partition
buckets across pipeline streams") delegates to the baseline's bucket-partitioned
`HashJoin::getNonJoinedBlocks` (`HashJoin.cpp:1520`). The baseline reaches that path *because* the
map is two-level. UHJ's maps are always two-level, so TwoLevel is the enabler here, not the cause of
the gap.

**Scope honesty.** This changes no query results: the same non-joined rows with the same values are
emitted either way. It changes how many streams emit them. It is being done to remove a capability
divergence, not for throughput, and no benchmark will be used to accept it.

**Design, and the trap found while studying it.** The baseline gates its partitioned branch on
`if constexpr (requires { it.getBucket(); map.numBuckets(); })`. Copying that verbatim into UHJ
would be **silently wrong**: `numBuckets()` is declared `requires(isFixedStorage())`
(`TwoLevelHashTable.h:77-82`) and UHJ runs every map in runtime mode (`BITS_FOR_BUCKET == -1`), so
the clause is **false for every UHJ map** and all of them would fall into the unfiltered branch —
each stream emitting every non-joined row, i.e. duplicates. The port must key on an accessor that
exists in runtime mode.

Partitioning is therefore by **iteration** bucket, which is what `iteratorAt`/`getBucket` actually
index (`TwoLevelHashTable.h:733` "Indexes the iteration partition, not the bucket partition"):
- `JoinHashMap`/`JoinHashMapWithSavedHash` (RuntimeStorage): `iterationBuckets() == num_buckets`,
  so streams get a genuine disjoint partition.
- `JoinFixedHashMap` (FixedRangeStorage, used by `key8`/`key16`/`range*`): `iterationBuckets()`
  is `1` (`TwoLevelHashTable.h:397-398`), so every cell is in iteration-bucket 0 — stream 0 emits
  all of them and the others emit none. Serial, but correct.
Correctness only needs a disjoint, complete cover of the cells, which both cases give.

**Change.** In `UnifiedHashJoin/HashJoin.{h,cpp}`: add the 5-arg `getNonJoinedBlocks` override and a
delegating 3-arg one; give `NotJoinedHash` `bucket_idx`/`num_buckets` plus
`isBucketInRange`/`isBlockInRange`, the bucket-skip loop, and the stream-0 nullmap guard; override
`supportParallelNonJoinedBlocksProcessing()` with `ConcurrentHashJoin`'s predicate
(`ConcurrentHashJoin.cpp:525-530`).

**Expected structural outcome.** `unified_hash` and `parallel_hash` both report
`supportParallelNonJoinedBlocksProcessing() == true` for a RIGHT/FULL join with keys, and a RIGHT
and a FULL join return exactly the same rows under `unified_hash` as under `hash` at several
`max_threads`.

**Refute condition.** Any row duplicated or lost in a RIGHT/FULL join under `unified_hash` versus
`hash` — that is what a wrong partition looks like, and it would mean the iteration-bucket cover is
not disjoint/complete. Would make this UNSETTLED, and the change must be reverted, not patched over.

---

## M5 — stop `clone()` propagating `stats_collecting_params`

**User decision (2026-08-02):** fix M5, using the split proposed and not objected to.

**Divergence.** UHJ's `clone()` (`HashJoin.h:215-221`) forwards `stats_collecting_params` and
`max_threads`; the baseline's (`HashJoin.h:129-134`) forwards neither its stats nor its map-shape
knob (`use_two_level_maps`).

**Split.**
- `stats_collecting_params` -> **MATERIAL**, drop it to match the baseline. Affects only hash-table
  size hints; no result change.
- `max_threads` -> **EXCLUDED under E3**. It exists in UHJ's constructor solely to size buckets, so
  it is the exact analogue of the baseline's `use_two_level_maps`, and it stays. Dropping it would
  leave a clone that still reports `supportParallelJoin() == true` while having a single bucket.

**Change.** Pass `StatsCollectingParams{}` in UHJ's `clone()`, keep `max_threads`.

**Expected structural outcome.** UHJ's `clone()` forwards no stats, matching the baseline's
argument set modulo the bucket knob.

**Refute condition.** A build error showing `max_threads` cannot be passed without also naming
stats — would mean the two are not separable and the split is wrong.

---

## W1 — gratuitous blank-line differences  [RETROSPECTIVE — see honesty note]

**Honesty note.** This entry is written **after** commit `b99fa90b5b0`, which made the change. It is
therefore *not* a valid Gate U2-PRE pre-registration and is not counted as one. It exists because an
independent verifier correctly flagged that `b99fa90b5b0` modified source with no PREREG entry.
Recorded as a gap rather than backdated.

**Divergence.** Four whitespace-only differences: a doubled blank line before the `namespace Unified`
opener in four headers, and one missing blank line in the non-joined map scan.

**Why it was not pre-registered.** It was found *by* the attribution tooling as the residue left
after every classified cause was accounted for, and treated as part of running the gate rather than
as a separate alignment. That reasoning was wrong: it changed tracked source, so it needed an entry.

**Materiality of the gap:** nil in substance — whitespace only, no token changed, and the same build
and tests covered it. The process gap is real and is reported as such in `REPORT_U3.md`.

---

## L5 — `Interpreters//HashJoin` double-slash include typo  [NOT ALIGNED, deliberately]

Raised by the independent verifier. Baseline `HashJoin/HashJoinMethods.h:4` reads
`#include <Interpreters//HashJoin/JoinUsedFlags.h>` (double slash); the fork spells it correctly.
This is an avoidable divergence in which **the fork is the correct side**. Aligning it would mean
re-introducing a typo into the fork, so it is deliberately left, classified LEAD, not MATERIAL.
Fixing the baseline instead is out of scope (the mission forbids touching unrelated baseline code).
