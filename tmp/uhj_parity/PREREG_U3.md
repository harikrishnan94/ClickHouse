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
