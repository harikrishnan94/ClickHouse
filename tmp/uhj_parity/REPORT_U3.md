# REPORT_U3 — UHJ divergence reduction (excluding compulsory TwoLevel)

Branch `uhj-parity`. **Not pushed. No pull request.** No commits on `master`, no rebase, no
force-push, no git config change.

## Per-unit verdict

| Unit | Verdict | Basis |
| --- | --- | --- |
| 1 Exhaustive inventory | **green** | Gate U1-INV: `INVENTORY_U3.md` + `U3_rawdiff.txt` + `U3_normdiff.txt`. Every divergence classified, zero unlabeled. |
| 2 Align MATERIAL items | **green** | 6 MATERIAL items + 1 cosmetic, each pre-registered then closed by a structural proof. |
| 3 Divergence-empty proof | **green** | Gate U3-EMPTY: `AVOIDABLE_MATERIAL=0`; residual attribution leaves 0 unattributed hunks. |
| 4 Correctness regression | **green** | `04658` OK / JOB_EXIT=0, `04659` OK / JOB_EXIT=0 on the final binary. |
| 5 Independent verification | see below | |

**Authorization flags:** user instructed "Fix both M1 and M5" after being told plainly that neither
changes query results. The de-duplication of the ~6000 identical forked lines was raised and
explicitly left out of scope.

**Risk-accepted LEADs:** none blocking. L1–L4 are documented as no-behavior-change or out-of-scope.

**HIGH-IMPACT assumptions:**
1. `PARALLEL_BUILD`-caused differences are excluded on the same ground as TwoLevel, because the
   shared-map parallel build exists only by virtue of bucket-indexed locks, which the mission
   excludes explicitly. Counted separately so the judgment is visible and reversible.
2. `FORK-MECHANICAL` (2 items) sits outside `AVOIDABLE_MATERIAL`: not TwoLevel-required, but not
   removable without deleting the fork. Raised with the user; not folded into EXCLUDED.

## Evidence matrix — keyed to divergence gates, never to benchmarks

| Gate | Invocation | Raw result | Verdict |
| --- | --- | --- | --- |
| U1-INV | `U3_normdiff.sh`; `INVENTORY_U3.md` | 3420 raw → 1464 residual in 6 files; 36/42 files identical modulo the fork wrapper | green |
| U1-DISC (M6) | `EXPLAIN PLAN`, in-memory join, before/after rebuild | `Read type` **Default → InOrder**; `parallel_hash` stays Default (negative control) | CONFIRM |
| U1-DISC (M1) | `EXPLAIN PIPELINE` `NonJoinedBlocksTransform` count, RIGHT join, `max_threads=8` | hash 0 / parallel_hash 8 / **unified_hash 0 → 8** | CONFIRM |
| U2-PRE | `git log` ordering | `cab6730b83a` < `8de627d44f5`; `b9cbfabc408` < `58fbde3ae1d`; `9f0b669b7fc` < `5362055b4ed` | green |
| U2-ALIGN M2 | changed-line filter on normalized diff | no `+`/`-` line names `finalizePerRowFlags` | closed |
| U2-ALIGN M3 | `rg -c 'doDebugAsserts\(\);'` | base 9 / UHJ 9 | closed |
| U2-ALIGN M4 | build + changed-line filter | `NINJA_EXIT=0` (no clash), macro only in context | closed |
| U2-ALIGN M6 | before/after control above + row equality | 200 ordered rows identical to `hash` | closed |
| U2-ALIGN M1 | `m1_nonjoined.sh` | 30/30 cases identical to `hash`; FAIL=0 | closed |
| U2-ALIGN M5 | source inspection + build | `clone()` forwards no stats; `NINJA_EXIT=0` | closed |
| U3-EMPTY | `U3_attribute.sh` | TWOLEVEL 1232 / PARALLEL_BUILD 195 / FORK-MECHANICAL 18 / **UNATTRIBUTED 0**; `AVOIDABLE_MATERIAL=0` | green |
| U4 | `04658`, `run_04659.sh` on final binary | both `OK`, `JOB_EXIT=0` | green |

**Explicit non-gates.** No `bench_*.sh` was run in this mission. No wall or CPU number appears in
this matrix, and none was used to accept or reject any unit. The prior campaign's "stop on two-level
CPU residual" story was treated as orientation only and played no part in any decision here.

## What was aligned, and the honest scope of each

| Item | What changed | Result-visible? |
| --- | --- | --- |
| M1 | UHJ gained `supportParallelNonJoinedBlocksProcessing` and the bucket-partitioned 5-arg `getNonJoinedBlocks`; `NotJoinedHash` gained bucket/block range filters and the stream-0 nullmap guard | **No.** Same rows, same values; only the number of emitting streams changes. |
| M2 | Baseline `finalizePerRowFlags(JoinUsedFlags &, size_t)` signature restored | No |
| M3 | `doDebugAsserts` restored to the public `getTotalByteCount` | No (debug-build assert coverage) |
| M4 | `UNIFIED_KEYGETTER_RANGE_IMPL` → `KEYGETTER_RANGE_IMPL` | No |
| M5 | `clone()` stops forwarding `stats_collecting_params` | **No.** Sizing hints only. |
| M6 | `unified_hash` admitted to the `optimize_read_in_order` gate | No — same rows; plan reads in primary-key order as `hash` does |
| W1 | 4 gratuitous blank lines | No |

## Two near-misses worth recording

1. **M1 would have silently duplicated rows** if the baseline's
   `requires { it.getBucket(); map.numBuckets(); }` guard had been copied verbatim. `numBuckets()` is
   `requires(isFixedStorage())` and every UHJ map is runtime-sized, so the guard is false throughout
   and all maps would have fallen into the unfiltered branch. Caught by reading
   `TwoLevelHashTable.h` before writing the port. The M1 test therefore compares **full sorted row
   sets**, not just aggregates, because `count()`/`sum()` can cancel a duplication/loss pair.
2. **A grep-based "proof" gave a false REFUTED** for M2/M4 by matching context lines. Corrected to a
   changed-line filter. Both the wrong run and the correction are in `WORKLOG.md`.

## Not addressed — the larger divergence

`UnifiedHashJoin` is 7466 lines and **36 of 42 files are byte-identical** to `HashJoin` modulo the
namespace wrapper: ~6000 lines of pure duplication. That exceeds the 1445 lines of behavioral
divergence and is untouchable by item-by-item alignment; it needs the two trees merged (e.g.
`HashJoin` templated over a map policy). Raised with the user and left out of scope by agreement.

## Final inventory summary

```
AVOIDABLE_MATERIAL=0
EXCLUDED=17 groups / 100 hunks / 1427 changed lines
FORK_MECHANICAL=2 items / 9 hunks / 18 changed lines
LEAD=4      UNSETTLED=0      UNATTRIBUTED=0
```
