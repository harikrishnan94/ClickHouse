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

---

## Unit 5 — independent verification

Two independent graders, neither of which saw the implementer's reasoning.

### Verifier 1 (static) — FIX-THEN-RESHIP

Its shell was blocked by an environment gate, so it could not execute the runtime gates. It
compensated with a full static review: it read **every hunk** of the normalized diff and
reconstructed git state from `.git` refs and reflog directly.

Findings and disposition:

| # | Finding | Severity | Disposition |
| --- | --- | --- | --- |
| 1 | Runtime gates (U4, adversarial M1) not executed in its environment | blocking for SHIP | Addressed by commissioning Verifier 2 with working tools |
| 2 | **W1 (`b99fa90b5b0`) changed source with no PREREG entry** — Gate U2-PRE technically unmet for it | minor gate | **Accepted as a real gap.** Recorded in `PREREG_U3.md` as an explicitly retrospective entry marked *not* a valid pre-registration, rather than backdated. Substance is nil (whitespace only); the process gap stands. |
| 3 | **`UnifiedHashJoin/joinDispatch.h` was never diffed** — `U3_normdiff.sh` iterated only baseline files, so the fork's one extra file was compared to nothing, undercutting the "exhaustive whole-directory diff" claim | minor method | **Fixed.** The script now also diffs fork-only files against their true baseline (`src/Interpreters/joinDispatch.h`). Result: no diff section, i.e. identical modulo the wrapper — now proven by tooling rather than by manual inspection. |
| 4 | Attribution regexes in `U3_attribute.sh` are broad, so `UNATTRIBUTED=0` may be a regex property | cosmetic | **Tested, not just noted** — see below. |
| 5 | The fork *fixes* a baseline `Interpreters//HashJoin` double-slash include typo | cosmetic | Recorded as **L5**, deliberately not aligned: the fork is the correct side, and aligning would mean re-introducing a typo. |

It explicitly reported **no banned move** and, having read every hunk, **no avoidable divergence
mislabeled as excluded** — the check that matters most.

### Response to finding 4 — the attribution was re-run with narrow markers

Rather than argue the regexes are fine, `U3_attribute_strict.sh` re-runs the same hunk-level
attribution keeping **only** unambiguous TwoLevel markers (dropping `lock`, `offset`, `cells`,
`pools`, `#include`, `max_threads`, `atomic`, ...):

```
STRICT_TWOLEVEL  hunks=69   changed_lines=1098
RESIDUE          hunks=40   changed_lines=347  -> tmp/uhj_parity/U3_strict_residue.txt
```

All 347 residue lines were then read. Every one is: atomics/`blocks_mutex`/`*Unlocked()` accessors
(shared-map parallel build), removal of `ConcurrentHashJoin`-only entry points
(`joinScatteredBlock`, `addBlockToJoin(Selector)`, `getUsedFlags`/`setUsedFlags`,
`hasNonJoinedRows`/`updateNonJoinedRowsStatus`), the F1 macro rename, `BuildResult`/`new_keys`
plumbing (E7), or bucket-addressed build prefetch (E11). One apparent deletion was checked rather
than assumed: the `memory_usage_before_adding_blocks` initialisation shows as removed at
`HashJoin.cpp:650`, and was confirmed **relocated** to `HashJoin.cpp:872-873` inside the
`blocks_mutex` section, with the same four usages as the baseline — not lost.

So `UNATTRIBUTED=0` does not depend on the broad markers.

### Verifier 2 (runtime) — adversarial execution

Commissioned specifically to close finding 1: independent adversarial RIGHT/FULL row-set
comparisons (odd `max_threads`, tiny/empty sides, all-NULL keys, skew, LowCardinality, multi-column
and direct-addressed keys, no-key joins, spill variants, `join_use_nulls`) plus re-running `04658`
and `04659` on the current binary. Result recorded in `WORKLOG.md`.
