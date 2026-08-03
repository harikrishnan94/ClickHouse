# Task: implement the `unified_hash` fixes identified by the cost-analysis mission

## Context

`unified_hash` (`src/Interpreters/UnifiedHashJoin/`) is intended to replace both `hash`
(`src/Interpreters/HashJoin/`) and `parallel_hash` (`src/Interpreters/ConcurrentHashJoin.{h,cpp}`)
with one algorithm that is optimal single- and multi-threaded. A previous session profiled
all three and produced a cost map. You are implementing the fixes it identified.

**Read first** (they are the specification for this work):

- `tmp/uhj_parity/perf2/REPORT.md` — the loop table (§3), lock table (§4), necessity
  analysis (§5) and the ranked handoff (§11). Sections §3.1, §3.3 and §4 cover every item
  below.
- `tmp/uhj_parity/perf2/WORKLOG.md` — entries F4 (lock measurements), F6 (the key-packing
  finding), F7 (the ablation that failed and why), F8 (hardware counters).
- `tmp/uhj_parity/perf2/codegen/P1_G2_probe_and_gather.md` and
  `codegen/K1_composite_keygetter.md` — the disassembly-level evidence.

**What the analysis established, in one paragraph.** At one thread on used-flag kinds
(`RIGHT`/`FULL`/`SEMI`/`ANTI`) `unified_hash` is ~6.5% slower than `hash`, and hardware
counters show this is extra *instructions* executed, not extra cache misses — the probe's
`Prober` state does not stay in registers. At 64 threads with fixed-width composite keys
its build burns 21-38% more CPU than `parallel_hash`, because `HashMethodKeysFixed`'s
constructor packs the **whole block's** keys and `unified_hash` constructs one **per
bucket**. Its locking is already better than `parallel_hash`'s and should not be
destabilised. Halving the bucket count was tried and is **worse** — do not go there.

**Branch and scope**

- Work directly on `uhj-parity`. Do not create a branch, do not push, do not open a PR.
- Never rebase or amend; add new commits.
- **Do not modify `src/Interpreters/HashJoin/` or `src/Interpreters/ConcurrentHashJoin.{h,cpp}`.**
  Not for purity — they are the A side of the A/B measurement, and changing them makes the
  before/after numbers meaningless.
- `src/Common/HashTable/TwoLevelHashTable.h` **is** in scope, but it is also compiled into
  `Aggregator` and `parallel_hash`. After editing it, prove the comparators are unchanged
  with `python3 tmp/uhj_parity/perf2/symdiff.py --before <old binary> --after <new binary>
  --expect-changed-regex 'DB::Unified::'`. If baseline symbols show up as changed, find out
  why before continuing.

**HEAD has moved since the analysis was done — check this before starting.** A concurrent
session committed `02a534167f1 "uhj: specialize sole-bucket Prober at per-block level"`,
which routes probe blocks through a compile-time `Prober<has_sole>` so the one-thread path
takes the flat `offsetInternal`. That is most or all of task P4 below. **Re-read
`TwoLevelHashTable.h` at current HEAD and confirm what is actually still broken before
writing code.** Every file:line reference below is from the earlier tree and may have moved.

**Build:** `ninja -C build/reldeb clickhouse`, no `-j`, redirect to a log in the build dir,
have a subagent summarise it. **Server:**
`CH_BIN=/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse tmp/uhj_parity/perf/start_server.sh`
(port 9111, HTTP 8121). Never touch anything on port 9000.

---

## Goal

Make `unified_hash` faster on the cells where it currently loses, without regressing the
cells where it wins, then measure and report the result.

---

## Tasks

### 1. B22 — stop packing the whole block's keys once per bucket

The largest single cost found. `HashMethodKeysFixed`'s constructor calls `packFixedBatch`
when `usePreparedKeys` holds (no nullable/LowCardinality keys, `sizeof(Key) <= 16`, all key
sizes in {1,2,4,8,16}), and `fillFixedBatch` sizes its output by `column->size()` — the
whole block, ignoring the selector (`src/Interpreters/AggregationCommon.h:58-76`).

`unified_hash` constructs a key getter **once per bucket** inside
`insertFromBlockImplTypeCase` (`UnifiedHashJoin/HashJoinMethodsImpl.h:356`) plus once for
the scatter pass (`:166`), so at 64 threads it packs the same block 129 times. `hash` packs
it once.

Construct the key getter **once per block**, before the bucket loop in `insertIntoBuckets`
(`UnifiedHashJoin/HashJoin.cpp:117-217`), and pass it into `insertFromBlockImpl` by const
reference. It is safe to share across buckets and threads: `prepared_keys` is written only
in the constructor and read-only afterwards. Confirm that for the specific key getters in
play (`HashMethodKeysFixed`, `HashMethodOneNumber`, `HashMethodString`,
`HashMethodSingleLowCardinalityColumn`) rather than assuming it — if any of them keeps
mutable per-call state, that one must stay per-bucket and you should say so.

Expect this to be the bulk of the composite-key build win. `u64` keys should be unaffected
(they use `HashMethodOneNumber`, which packs nothing) — that is a useful control.

### 2. P3 — keep the `Prober`'s routing state out of memory

On the multi-bucket path, `Prober::find` writes `state.routed` and `state.routed_prefix`
per row and `offsetInternal` reads them back, and the codegen shows `shift`/`max_bucket`/
`prefix` being reloaded from the stack each row. Result: dependent-load chain depth 3 → 5,
two spill stores and nine spill reloads per row against the baseline's zero and one.

Restructure so the per-row loop does not round-trip through the `Prober` object. Options,
in rough order of preference:

- have `find` return the bucket (or the prefix) alongside the lookup result, and have
  `offsetInternal` take it as an argument, so nothing is stored;
- or split the per-block invariants (`buckets`, `prefix`, `shift`, `max_bucket`) into a
  small by-value struct the loop can keep in registers, leaving only the per-row values as
  locals in the caller.

Verify with disassembly, not by reading: `python3 tmp/uhj_parity/perf2/codegen.py` has
`dis` / `count` / `mca` subcommands and reports dependent-load-chain depth and spill counts.
The target is that the unified probe loop's spill count and dep-load depth approach the
baseline's.

### 3. P4 — no `offsetInternal` work when the offset is not needed, and none on the sole path

Two parts.

- **Sole path:** likely already done by `02a534167f1`. Confirm from the disassembly that at
  one bucket `offsetInternal` compiles to the flat form (a subtraction, not a prefix
  lookup). If it does, record that and move on.
- **Offset not needed at all:** `unified_hash` already templates `needs_offset` from
  `JoinFeatures<...>::need_flags` (`UnifiedHashJoin/HashJoinMethods.h:90`), which is a real
  advantage over the baseline's hardcoded `use_offset = true`
  (`HashJoin/KeyGetter.h:19`). Check the multi-bucket `find` does not still load
  `prefix[bucket]` when `needs_offset` is false — if it does, template it away.

### 4. Prefix sums: compute at post-build, then use `offsetInternalUnsafe` unconditionally

`BucketPrefixSums::offset` guards its computation with `std::call_once`
(`TwoLevelHashTable.h:106-137`), so every probe and non-joined-scan call pays a
once-flag check. `offsetInternalUnsafe` (`:182`, `:284`) skips it and assumes the prefix
array is already built.

`unified_hash` already precomputes via `freezeMapsForProbing`
(`UnifiedHashJoin/HashJoin.cpp:2087-2098`, also called from `runPostBuildPhase`). Make that
guarantee complete and then switch `unified_hash`'s call sites to the unsafe form:

- audit every path that can reach `offsetInternal` on a `unified_hash` map — probe,
  non-joined scan, and anything after `runPostBuildPhase` rewrites the maps (fixed-map
  conversion, reranging, runtime filters) — and make sure `freezeMapsForProbing` has run on
  the map in each case;
- add a debug-only assertion that the prefix array is computed, so a future path that
  forgets to freeze fails loudly in debug rather than silently reading zeros;
- leave the safe `offsetInternal` in place for `parallel_hash`, which has no freeze step.

### 5. B16 / A3 — get the byte accounting out of the bucket critical section

Inside the per-bucket lock, `insert_bucket` calls `getBucketBufferSizeInBytes` and
`pools[bucket]->allocatedBytes()` twice each and then does a relaxed
`bucket_bytes.fetch_add` (`UnifiedHashJoin/HashJoin.cpp:135-152`). Only the insert itself
needs the lock.

Accumulate the delta in a per-thread local and reduce it at build finish, or drop the
incremental accounting entirely and rely on `recomputeBucketBytes`, which already exists and
already runs at `onBuildPhaseFinish`. Whichever you choose, keep the reported byte total
correct at every point an external caller can observe it — check who reads `bucket_bytes`
and when (`getTotalByteCount`, the size-limit checks) before changing the timing.

### 6. L3 — shrink the `blocks_mutex` critical section

Measured hold time is p50 2.90 µs, of which the part that actually needs the lock
(`StoredColumnsIndex::add`) is 32 ns. The rest is
`assertBlocksHaveEqualStructureAllowReplicated`, `doDebugAsserts()` and
`JoinCommon::getCurrentQueryMemoryUsage()` (`UnifiedHashJoin/HashJoin.cpp:877-899`).

Move those three above the `lock_guard`. The structure assertion reads
`data->sample_block`, which is immutable after construction; the memory-usage read is a
process-wide query. Keep inside the lock only the `data->columns` append, the
`StoredColumnsIndex::add`, and the `allocated_size` update. Note `add` takes its own mutex,
so the nesting is worth a look while you are there.

### 7. P1 — confirm, do not implement separately

P1 is the probe main loop; its cost is P2/P3/P4 inlined into it. There is nothing to change
here beyond tasks 2-4. After those land, re-disassemble the probe loop and report its
instruction count, spill count and dependent-load depth against the baseline's, so we know
how much of the gap closed.

---

## Then: measure and report

Re-measure with the same harness the analysis used, so the numbers are comparable.

**Before you change anything**, build and keep the current-HEAD binary aside as `B_old`:

```
ninja -C build/reldeb clickhouse
cp build/reldeb/programs/clickhouse tmp/uhj_parity/perf2/bin/clickhouse.bold
```

Then implement, rebuild, and keep the result as `B_new`.

**Cells to measure.** The full matrix is 144 cells and takes about 12 minutes at 7 reps:

```
python3 tmp/uhj_parity/perf/sweep.py --reps 7 --run-tag fix1
```

Use a **fresh `--run-tag` per run** — query ids are derived from it, and re-using a tag in a
new process regenerates identical ids, which makes any readback that groups
`system.*` logs by `query_id` silently sum two runs.

The cells that should move:

| task | cells |
| --- | --- |
| B22 | `comp` key at 16 and 64 threads, build phase. `u64` is the control and should not move |
| P3, P4, prefix sums | 1 thread, used-flag kinds (`RIGHT`, `FULL`, `LEFT SEMI`, `LEFT ANTI`), probe phase |
| B16, L3 | build phase at 16 and 64 threads, all key types |
| all | nothing at 16/64 threads should get slower — `unified_hash` currently wins every wall-time cell there and must keep doing so |

**Report a table with three columns per cell: A / B_old / B_new**, where A is the
comparator (`hash` at 1 thread, `parallel_hash` at 16 and 64), B_old is `unified_hash`
before your changes and B_new after. Report wall and CPU, and the build / probe / non-joined
phase split where the phase is the thing that moved. Give the percentage change B_new vs
B_old, and B_new vs A.

Useful extras from the previous session, all re-runnable:

- `python3 tmp/uhj_parity/perf2/kscale.py run --tag ... && ... fit --tag ...` — build cost
  per row against partition count, the sharpest view of task 1.
- `KIND=FULL TAG=... REPS=7 bash tmp/uhj_parity/perf2/perfstat.sh` — hardware counters via
  `perf stat`, for tasks 2-4. Note `perf stat -p <server pid>` does **not** work here (it
  misses per-query threads), and ClickHouse's `metrics_perf_events_enabled` underflows;
  `perfstat.sh` uses the one route that works.
- `python3 tmp/uhj_parity/perf2/lockmeas.py` — needs the instrumentation patch from commit
  `79a9eee2619`, which is reverted; `git cherry-pick 79a9eee2619` if you want lock numbers
  for tasks 5-6, and revert it again before you finish.

**Correctness.** These must pass on the final tree:

```
bash tests/queries/0_stateless/04658_unified_hash_join_equivalence.sh    # compare to .reference
UHJ_PORT=9111 bash tmp/uhj_parity/run_04659.sh                          # prints OK
```

04659's own `.sh` fails on a client argument quirk unrelated to the join — use the
`run_04659.sh` wrapper, which is the documented workaround.

**Commit** each task separately with a message saying what changed, why, and what the
measurement showed. If a task turns out not to be worth doing — for example if
`02a534167f1` already fixed P4, or if a key getter cannot be shared across buckets — say so
in the commit or the report rather than forcing it through.
