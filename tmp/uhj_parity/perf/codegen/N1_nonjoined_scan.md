# N1 — Codegen comparison: post-join non-joined scan, `unified_hash` vs `hash`

Cross-implementation diff (both sides live in the **same** binary — this is not before/after).

| | |
|---|---|
| Binary | `/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse` (RelWithDebInfo, 4.9 GB) |
| Build ID | `b7980f6e38fd7fccc6cb883a140c0c0a1b4dbe78` |
| Arch | aarch64 (Neoverse-V2 host); disassembly is AArch64 |
| Tool | `.claude/tools/analyze-assembly.py` with `llvm-objdump`/`llvm-nm` 22.1.8 |
| Workload cell it explains | `FULL \| u64 \| hi \| t1 \| medium`, 85,728 non-joined rows, `max_threads=1` |

Tooling note: `llvm-readobj` is not installed on this host and `analyze-assembly.py` requires it
for `--file-headers`. I added a shim at `tmp/uhj_parity/perf/bin/llvm-readobj` that synthesizes the
three fields the tool parses (format, machine, endianness) from `readelf -h`, plus the build ID from
`readelf -n`. Nothing under `src/` was touched. `llvm-mca` is unavailable, so there is **no**
microarchitectural throughput modelling here — every number below is a static instruction count.

---

## (a) C++-level difference

The key type of the measured cell is `UInt64`, so both sides use their `key64` variant, and the two
variants are **structurally different map families**:

| | `hash` | `unified_hash` |
|---|---|---|
| `key64` map type | `HashMap<UInt64, Mapped, HashCRC32<UInt64>>` — single-level (`src/Interpreters/HashJoin/HashJoin.h:313`) | `JoinHashMap<UInt64, ...>` = `TwoLevelHashMap<..., BITS_FOR_BUCKET>` with `BITS_FOR_BUCKET == -1` → `RuntimeStorage` (`src/Interpreters/UnifiedHashJoin/HashJoin.h:56,101-103,376`) |
| scan loop | `src/Interpreters/HashJoin/HashJoin.cpp:1445-1463` (the `else`, single-level branch — **no** bucket filtering) | `src/Interpreters/UnifiedHashJoin/HashJoin.cpp:1513-1526` (bucket-partitioned loop, always) |
| `offsetInternal` | `src/Common/HashTable/HashTable.h:1499-1504` | `src/Common/HashTable/TwoLevelHashTable.h:906-912` → `RuntimeStorage::offsetInternal` at `:276-282` → `BucketPrefixSums::offset` at `:111-116` |

The decisive lines. Baseline (`HashTable.h:1499-1504`):

```cpp
size_t offsetInternal(ConstLookupResult ptr) const
{
    if (ptr->isZero(*this))
        return 0;
    return ptr - buf + 1;
}
```

Unified (`TwoLevelHashTable.h:906-912`) — the re-hash is real, and it is on the non-fixed-range branch,
which is the branch a `HashMapTable` takes:

```cpp
size_t offsetInternal(ConstLookupResult ptr) const
{
    if constexpr (isFixedRangeStorage())
        return impls.offsetInternal(ptr);
    else
        return impls.offsetInternal(ptr, getBucketFromHash(bucketRoutingHash(ptr->getKey(), ptr->getHash(*this))));
}
```

and (`TwoLevelHashTable.h:111-116`) — the `std::call_once` per call is also real:

```cpp
template <typename BucketAt>
size_t offset(UInt32 bucket_count, BucketAt && bucket_at, size_t buck, size_t cell_offset)
{
    std::call_once(compute_once, [&] { compute(bucket_count, bucket_at); });
    return offsetUnsafe(buck, cell_offset);
}
```

**Both reported source claims are confirmed.** Two refinements matter:

1. `RuntimeStorage::offsetInternal` (`:276-282`) is the one instantiated here, not `FixedStorage`
   (`:174-180`); the two are textually near-identical, so the reported behaviour is unchanged.
2. `HashJoin::runPostBuildPhase` calls `computeBucketPrefix` after the build
   (`UnifiedHashJoin/HashJoin.cpp:2093`), but per the class comment at `TwoLevelHashTable.h:105-110`
   that path deliberately leaves `compute_once` **unarmed**. So the first `offset` call still runs the
   `std::call_once` body; every later call takes the once-flag fast path. This is why the answer to
   "is a `call_once` executed per cell" is *no* while the answer to "is `call_once` machinery on the
   per-cell path" is *yes* — see (c).

Counterpart status: there is no "no counterpart" case here. Both sides run the same
`NotJoinedHash::fillColumns` shape; they differ entirely in what `map.offsetInternal(it.getPtr())` and
the iterator advance compile to.

**Configuration fact that sharpens all of this**: `bucketCountForThreads(1) == 1`
(`UnifiedHashJoin/HashJoin.cpp:66-71`). At `max_threads=1` the unified map has exactly **one** bucket,
so every routing computation below provably yields the constant 0 at runtime, and every bucket-indexed
load provably resolves to index 0 — none of it is specialized away, because the bucket count is a
runtime value (`BITS_FOR_BUCKET == -1`).

---

## (b) Assembly-level difference, quantified

Symbols analysed (both resolved with `resolution confidence: high`):

| side | symbol | address | size |
|---|---|---|---|
| `hash` | `DB::NotJoinedHash::fillColumns<HashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList, HashCRC32<unsigned long>, ...>>>` | `0x142a1b80` | 1864 B / 466 insns |
| `unified_hash` | `DB::Unified::NotJoinedHash::fillColumns<TwoLevelHashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList, HashCRC32<unsigned long>, ..., HashMapTable, -1>>>` | `0x165e8780` | 2440 B / 610 insns |
| callee (unified only) | `TwoLevelHashTable<...>::RuntimeStorage::offsetInternal(HashMapCell<...> const*, unsigned long) const` | `0x165f0cc0` | 236 B / 59 insns |
| thunk (unified only) | `std::__call_once_proxy<... BucketPrefixSums::offset<...>::{lambda()#1}...>` | `0x165f0880` | 280 B / 70 insns |

Hot path = one non-joined cell on the **emitting** path (flag clear → row collected → advance to the
next non-empty cell in the same bucket). Address ranges counted:

- baseline: `0x142a1fe0`–`0x142a2078` (labels `L23`, `L24`, collect, size check, `L25`)
- unified caller: `0x165e8b24`–`0x165e8b28`, `0x165e8b34`–`0x165e8ba0`, `0x165e8ba4`–`0x165e8bac`,
  `0x165e8bbc`–`0x165e8bd4`, `0x165e8bf8`–`0x165e8c3c`, `0x165e8c68`–`0x165e8c70`
- unified callee fast path: `0x165f0cc0`–`0x165f0d28` + `0x165f0d48`–`0x165f0d7c`
  (the `b.eq` at `0x165f0d28` skips the `std::__call_once` call once the flag is set)

`nop` padding excluded. Counts produced by a script over the tool's labeled disassembly.

| metric (per non-joined cell) | `hash` | `unified_hash` | delta |
|---|---:|---:|---:|
| **hot-path instruction count** | **39** | **102** (61 caller + 41 callee) | **+63 / ×2.6** |
| loads | 15 | 28 (18 + 10) | +13 |
| stores | 1 | 8 (1 + 7) | +7 |
| dependent-load chain depth, structural | 4 | 4 | 0 |
| dependent-load chain depth, **as scheduled** | **4** | **7** | **+3** |
| branches | 6 | 16 (11 + 5) | +10 |
| spills/reloads (stack traffic) | 0 st / 2 ld | 7 st / 5 ld | +7 st / +3 ld |
| calls **executed** | 1 (`collect`) | 2 (`offsetInternal`, `collect`) | +1 |
| `call_once`/guard-style call **present** on the path | no | **yes** (`std::__call_once`, `0x165f0d44`) | — |
| `call_once`/guard-style call **executed** per cell | no | **no** (branch-skipped) | — |

Sub-path breakdown for the offset computation alone (the part that actually differs):

| offset computation only | `hash` | `unified_hash` |
|---|---:|---:|
| instructions | 12 | 8 (caller) + 41 (callee) = **49** |
| loads | 5 | 2 + 10 = **12** |
| stores | 0 | 0 + 7 = **7** |
| calls | 0 (fully inlined) | 1 |

**Dependent-load chain, traced by hand from the instruction order.** Baseline: `ldr x9,[x20,#0x8]`
(`&parent`) → `ldr x9,[x9,#0xb0]` (`used_flags`) → `ldp x8,x12,[x9,#0x30]` (`per_offset_flags.begin`)
→ `ldarb w8,[x8]` = depth 4; the offset's own loads (cell key, `map.buf`) are depth 1 and issue in
parallel with that chain. Unified, **as the compiler actually scheduled it**, the two chains are
serialized by the call boundary because the flags-vector walk is emitted *after* the `bl`
(`0x165e8b68` onward): `ldr x8,[x1]` (cell key) → `crc32cx` → [call] → `ldp x8,x9,[x21,#0x28]`
(`prefix.begin`) → `ldr x8,[x8,x19,lsl #3]` (`prefix[buck]`) → [ret] → `ldr x8,[x20,#0x8]` →
`ldr x8,[x8,#0x138]` → `ldp x8,x9,[x8,#0x30]` → `ldarb` = **7 loads deep**. Structurally the flags
chain does not depend on the offset and could overlap; it does not in this build.

**Inlining decisions that differ:**

1. `offsetInternal` is **fully inlined** on the baseline — 12 instructions, zero calls, no stack frame.
   On the unified side it is **not inlined**: a real out-of-line function with a 0x70-byte frame that
   saves and restores `x19`–`x22`, `x29`, `x30`.
2. `BucketPrefixSums::offset` *is* inlined into `offsetInternal`, but its `std::call_once` is not: it
   leaves a `bl std::__call_once` plus a separate 280-byte `__call_once_proxy` thunk. Building the
   closure that call needs is what forces the frame and the spills in point 1 — **inferred**, not
   measured: no `-Rpass-missed` data was collected, but the 4 closure stores at `0x165f0cf8`–`0x165f0d1c`
   sit unconditionally *before* the once-flag check at `0x165f0d20`, which is exactly the shape that
   defeats inlining.
3. Unified emits 4 call sites to an out-of-line
   `TwoLevelHashTable<...>::beginOfNextNonEmptyBucket(unsigned long&)`. The baseline has no counterpart —
   a single-level map's advance is a straight-line pointer walk. This fires once per bucket, not per
   cell, so it is excluded from the per-cell counts above.
4. Whole-function size: 610 insns / 2440 B vs 466 insns / 1864 B (+31% instructions).

Note on symbol names: identical-code folding merged the `RowRefList`, `RowRef` and
`unique_ptr<SortedLookupVectorBase>` instantiations of `offsetInternal`, so `llvm-objdump` labels the
call target with whichever name it picked. I verified `0x165f0cc0` carries the `RowRefList` name too, so
this is one folded body, not a mis-resolution.

---

## (c) The specific instruction-sequence delta on the per-cell path

Per non-joined cell, `unified_hash` executes all of the following, which the baseline does not:

**1. A full CRC32 re-hash of the cell key, plus bucket routing (5 net-new instructions, caller).**

```
165e8b48:  ldr     x8, [x1]              ; ptr->getKey()   (baseline loads this too, for isZero)
165e8b4c:  mov     w9, #-1               ; HashCRC32 seed
165e8b50:  crc32cx w8, w9, x8            ; ptr->getHash(*this)  <-- FULL RE-HASH, net-new
165e8b54:  ldp     w10, w9, [x22, #0x4]  ; max_bucket, shift    <-- net-new load
165e8b58:  lsr     x8, x8, x9            ; net-new
165e8b5c:  and     w2, w10, w8           ; bucket               <-- net-new; provably 0 at t1
165e8b60:  mov     x0, x22
```

The cell is a `HashMapCell` (no saved hash), so this is a genuine recomputation of a hash the build
already computed and discarded. At `max_threads=1` the result is masked down to the constant 0.

**2. A non-inlined call (41 instructions on the fast path) where the baseline has 6 inlined ones.**
Baseline computes the offset as `sub / cmp / asr / csinc` off two already-loaded values. Unified calls
`RuntimeStorage::offsetInternal`, which per cell executes:

```
165f0cc0:  sub  sp, sp, #0x70            ; frame
165f0cc4:  stp  x29, x30, [sp, #0x40]    ; 3 callee-save pair stores (6 registers)
165f0cc8:  stp  x22, x21, [sp, #0x50]
165f0ccc:  stp  x20, x19, [sp, #0x60]
...
165f0ce0:  b.hs <ret 0>                  ; bounds check: buck < buckets.size()
165f0cf0:  add  x8, x8, x2, lsl #7       ; &buckets[buck]  (128-byte Impl stride)
165f0cf8:  str  x0, [sp, #0x8]           ; --- std::call_once closure, built UNCONDITIONALLY ---
165f0cfc:  ldr  w10, [x0], #0x40         ;     load num_buckets; x0 now = &once_flag
165f0d08:  ldr  x22, [x8, #0x20]         ; buckets[buck].buf   (dependent load)
165f0d10:  stp  x9, x11, [sp, #0x10]     ;     closure tuple
165f0d18:  stur w10, [x29, #-0x14]       ;     closure capture
165f0d1c:  str  x8, [sp, #0x20]          ;     closure capture
165f0d20:  ldapr x8, [x0]                ; ACQUIRE LOAD of the std::once_flag   <-- per cell
165f0d24:  cmn  x8, #0x1
165f0d28:  b.eq <fast path>              ; skip the call once initialised
165f0d38:  adr  x2, <__call_once_proxy<...BucketPrefixSums::offset...>>   ; present, not executed
165f0d44:  bl   <std::__call_once>                                        ; present, not executed
165f0d48:  ldp  x8, x9, [x21, #0x28]     ; prefix.begin/end
165f0d54:  b.hs <abort>                  ; SECOND bounds check: buck < prefix.size()
165f0d5c:  ldr  x8, [x8, x19, lsl #3]    ; prefix[buck]        (dependent load)
165f0d58:  sub  x9, x20, x22             ; ptr - buf      \
165f0d60:  asr  x9, x9, #4               ;   / 16          |  the whole of the baseline's work
165f0d64:  add  x9, x9, #0x1             ;   + 1           |
165f0d68:  add  x0, x9, x8               ; + prefix[buck] /
165f0d6c:  ldp  ... x3 + add sp + ret    ; 3 stack reloads, frame teardown
```

Net-new per cell relative to the baseline, from this callee alone: **1 call/return pair, 7 stack
stores, 3 stack reloads, 2 extra bounds checks, one acquire-load of a `std::once_flag`, and 3 extra
dependent loads** (`buckets.begin` → `buckets[buck].buf`, `prefix.begin` → `prefix[buck]`).

**3. A bucket-aware iterator, ~+10 instructions per cell.** The end test is a `(bucket, ptr)` pair
comparison rather than one pointer compare, and the bucket-exhaustion check at `0x165e8bf8`–`0x165e8c3c`
(18 instructions) must bounds-check the bucket index and re-derive `buckets[bucket].buf` and its
size-degree through an indexed load, where the baseline (`0x142a2050`–`0x142a2078`, 11 instructions)
reads them straight off the sole table. Plus the per-cell stream filter at `0x165e8c68` (3 instructions
at 1 stream; it becomes `udiv` + `msub` — integer division — when `num_streams >= 2`).

Aggregate: **102 instructions per non-joined cell vs 39**. At 85,728 emitted rows that is ~5.4M extra
instructions on the scan itself, before counting the cells that *are* used (which pay items 1–3 in full
and skip only `collect`). This is a consistent shape for the reported 14,888 us `NonJoinedBlocksTransform`
against a baseline whose `NonJoinedBlocksTransform` elapsed is 0, and for the deficit tracking join kind
(RIGHT +5.27%, FULL +4.79% down to INNER +0.64%) — the kinds that scan used flags pay it, the kinds that
do not, do not.

---

## (d) Measured vs inferred

**Measured** (disassembled from this binary, counted by script or read directly off the listing):

- Every instruction/load/store/branch/call count in the table in (b), and every address, opcode and
  symbol quoted in (c).
- `offsetInternal` is out-of-line on the unified side (`0x165f0cc0`, 236 B, 59 insns) and fully inlined
  on the baseline (0 calls in the per-cell range).
- `crc32cx` at `0x165e8b50` is on the unified per-cell path; there is no `crc32*` anywhere in the
  baseline function.
- `std::__call_once` is called at `0x165f0d44` inside `offsetInternal`, with
  `__call_once_proxy<... BucketPrefixSums::offset<...>::{lambda()#1}>` as its callback — the mangled
  callback name at `0x165f0d38` names `BucketPrefixSums::offset` explicitly, which is how the
  `TwoLevelHashTable.h:114` `std::call_once` is tied to this exact code.
- The `ldapr`/`cmn`/`b.eq` sequence at `0x165f0d20`–`0x165f0d28` skips that call once the flag is set.
- `__cxa_guard_acquire` / `pthread_once` appear **nowhere** in any of the four disassembled functions.
- Both sides' `key64` map types, and `bucketCountForThreads(1) == 1`, read from source.

**Inferred** (reasoned, not measured):

- That the `std::call_once` closure construction is *the reason* `offsetInternal` was not inlined.
  Plausible from the code shape; no `-Rpass` / remarks data was collected to prove it.
- The "as scheduled" dependent-load depth of 7 is a reading of the compiler's instruction order, not a
  pipeline simulation. `llvm-mca` is unavailable on this host, so no throughput or latency number is
  claimed and none should be read into the instruction counts.
- The mapping from +63 instructions/cell to the 14,888 us and to the 4.79–5.27% wall deltas is a
  consistency argument, not an attribution measurement. Nothing here was profiled.

**Could not resolve / did not do:**

- No `llvm-mca` analysis (`llvm-mca` is not installed) — so no cycle, port-pressure or IPC estimate.
- No `--source` interleaving was used; source attribution is by symbol name and by matching the
  instruction sequence to the source, not by DWARF line tables.
- The `beginOfNextNonEmptyBucket` callee was identified by its call sites but not separately
  disassembled and counted, because it is per-bucket (one bucket at `max_threads=1`) and therefore
  not on the per-cell path.
- Per-cell counts assume the emitting, same-bucket-advance, next-cell-non-empty case. Runs of empty
  cells add 6 instructions each on both sides (`0x142a2080` / `0x165e8be0`), which is a wash.

---

## Bottom line

> **Does a `call_once`/guard call appear on the per-cell path? YES — present, NO — not executed.**
> `std::__call_once` is emitted at `0x165f0d44` inside `RuntimeStorage::offsetInternal`, which unified
> calls once per non-joined cell. It is branch-skipped after the first cell by the once-flag check at
> `0x165f0d20`. What *is* paid per cell is the machinery around it: an acquire-load of the
> `std::once_flag`, 4 unconditional stack stores building the closure the call would need, and the
> out-of-line call/frame/spill that carrying all that prevented the compiler from inlining away.
> `__cxa_guard_acquire` and `pthread_once` appear nowhere.

The single largest removable item is not the `call_once` itself but the fact that `offsetInternal`
stopped being a pointer subtraction. `offsetInternalUnsafe` (`TwoLevelHashTable.h:916-922`) already
exists, skips the once check, and has the precondition this scan satisfies — `computeBucketPrefix` runs
in `runPostBuildPhase` before any non-joined scan. It would not remove the CRC32 re-hash or the
bucket-indexed loads, which need the iterator to carry its bucket instead of recovering it from the key.
