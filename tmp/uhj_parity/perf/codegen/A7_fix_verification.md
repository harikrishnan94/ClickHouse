# A7 — Codegen validity check: did `offsetInternalAtBucket` actually remove the re-hash and the `call_once`?

Gate G1.4-style check. A performance result for this fix is only interpretable if the targeted
instructions genuinely left the binary. This is a **before/after diff of two binaries**, unlike
`N1_nonjoined_scan.md`, which was a cross-implementation diff inside one binary.

| | |
|---|---|
| BEFORE binary | `tmp/uhj_parity/perf/bin/clickhouse.pristine`, build ID `b7980f6e38fd7fccc6cb883a140c0c0a1b4dbe78` |
| AFTER binary | `build/reldeb/programs/clickhouse`, build ID `d115153fc6a930dd1ffaeab37b57b525218905a8` |
| Arch | aarch64 (Neoverse-V2 host); AArch64 disassembly |
| Tool | `.claude/tools/analyze-assembly.py` with `llvm-objdump`/`llvm-nm` 22.1.8 |
| Not available | `llvm-mca`. **No cycle, IPC or port-pressure number is claimed anywhere below.** Every quantity is a static instruction count. |
| Workload cell | `FULL \| u64 \| hi \| t1 \| medium`, 85,728 non-joined rows, `max_threads=1` |

Both build IDs were confirmed with `readelf -n`. The BEFORE build ID matches the one `N1` names,
so this is the same binary `N1` analysed.

Working files (disassembly, counting scripts, raw diffs) are under `codegen/logs/`.

---

## Bottom line

> **The fix did what it claimed.** On the unified non-joined scan's per-cell path the `crc32cx`
> re-hash is gone, the call to `RuntimeStorage::offsetInternal` is gone, and with it the entire
> `std::call_once` apparatus — the once-flag acquire load, the closure stores, the
> `__call_once_proxy` thunk. The offset is now 16 inlined instructions. Per emitted non-joined cell
> the hot path drops from **102 to 69 instructions (-32%)**, and the executed call count drops from
> 2 to 1.
>
> **No collateral damage.** Across all 645,013 text symbols in the binary, exactly 18 changed size,
> and all 18 are `DB::Unified::NotJoinedHash::fillColumns` instantiations. Zero `DB::Aggregator`,
> zero `DB::ConcurrentHashJoin`, zero baseline `DB::HashJoin` symbols changed.

---

## (a) The change under test

Uncommitted working-tree diff, in full (`git diff --stat`):

```
 src/Common/HashTable/BucketPartitionedTable.h |  5 +++++
 src/Common/HashTable/TwoLevelHashTable.h      | 22 ++++++++++++++++++++++
 src/Interpreters/UnifiedHashJoin/HashJoin.cpp |  6 +++++-
 tmp/uhj_parity/bench_parallel.sh              |  6 +++---
 tmp/uhj_parity/bench_serial.sh                |  4 ++--
```

`TwoLevelHashTable.h` gains one method (nothing existing is modified):

```cpp
size_t ALWAYS_INLINE offsetInternalAtBucket(ConstLookupResult ptr, size_t iteration_bucket) const
{
    if constexpr (isFixedRangeStorage())
        return impls.offsetInternalUnsafe(ptr);
    else
        return impls.offsetInternalUnsafe(ptr, iteration_bucket);
}
```

`BucketPartitionedTable.h` gains the matching concept requirement. `UnifiedHashJoin/HashJoin.cpp`
changes one line in `NotJoinedHash::fillColumns`, `map.offsetInternal(it.getPtr())` becoming
`map.offsetInternalAtBucket(it.getPtr(), it.getBucket())`.

**Provenance caveat, stated up front.** I did not independently confirm which commit the pristine
binary was built from; I was given it as "unmodified". The whole-binary symbol comparison in (d) is
what makes that credible: if the pristine had been built from a materially different source state,
far more than 18 functions would differ. Every difference found between the two binaries is
explainable by this diff alone.

---

## (b) Symbol resolution

The scan is a function template, so the `UInt64`-key instantiation is the one that matters for the
measured cell. Both binaries resolve **the same two candidates, in the same order**, for the regex
`NotJoinedHash::fillColumns<TwoLevelHashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList, HashCRC32`:

| # | symbol | BEFORE | AFTER |
|---|---|---|---|
| 1 | `DB::NotJoinedHash::fillColumns<TwoLevelHashMapTable<..., HashMapTable, **8**>>` — baseline `HashJoin`, two-level branch (this is what a `parallel_hash` shard runs) | `0x142a4d40`, 2640 B, 660 insns | `0x142a4740`, 2640 B, 660 insns |
| 2 | `DB::Unified::NotJoinedHash::fillColumns<TwoLevelHashMapTable<..., HashMapTable, **-1**>>` — the fix target | `0x165e8780`, 2440 B, 610 insns | `0x165e81c0`, 2536 B, 634 insns |

The BEFORE numbers for #2 (`0x165e8780`, 2440 B, 610 insns) reproduce `N1` exactly, and the BEFORE
callee `RuntimeStorage::offsetInternal` resolves at `0x165f0cc0`, 236 B, 59 insns — also exactly as
`N1` reports. Addresses moved in the AFTER binary and were re-resolved by symbol, not assumed.

Both resolutions report `Resolution confidence: high`.

---

## (c) Question 1 — is the targeted code gone?

### `crc32cx` re-hash — **GONE**

`grep crc32` over the full disassembly of the scan function:

| | BEFORE | AFTER |
|---|---|---|
| `crc32*` occurrences in the whole function | **1** (`crc32cx w8, w9, x8` at `0x165e8b50`) | **0** |

The BEFORE sequence `N1` names is present exactly as described — `ldr x8,[x1]` / `mov w9,#-1` /
`crc32cx w8,w9,x8` / `ldp w10,w9,[x22,#0x4]` / `lsr` / `and w2,w10,w8` (bucket routing) — and the
whole sequence is absent from the AFTER binary. The AFTER path reads `it.getBucket()`, already live
in `x8`, instead.

### `call_once` machinery — **GONE from this path**

| | BEFORE | AFTER |
|---|---|---|
| `bl <...RuntimeStorage::offsetInternal...>` in the scan | 1 | **0** |
| `ldapr` / `ldar` / `call_once` / `__cxa_guard` / `pthread_once` anywhere in the scan function | 0 (they lived in the callee) | **0** |
| in the BEFORE callee: once-flag acquire load | `ldapr x8, [x0]` at `0x165f0d20` | callee no longer called |
| in the BEFORE callee: `__call_once_proxy` address | `adr x2, <__call_once_proxy<...BucketPrefixSums::offset...>>` at `0x165f0d38` | — |
| in the BEFORE callee: the call | `bl <std::__call_once>` at `0x165f0d44` | — |

Every specific address and instruction `N1` claimed for the BEFORE binary was re-verified here and
is correct. All of it is unreachable from the AFTER scan.

`ldarb` — the acquire load of the per-row **used flag** — occurs 2× in both binaries. That is the
join's own semantics and is correctly untouched; it is not once-flag machinery.

### Is the offset now inlined? — **YES, fully**

AFTER, `0x165e8588`–`0x165e85c4`, 16 instructions, no call, no frame, no spill:

```
ldp  x9, x10, [x22, #0x10]   ; buckets.begin / buckets.end
sub / cmp / b.hs             ; bounds check: bucket < buckets.size()
ldr  x10, [x1]               ; cell key
cbz  x10, <zero-cell>        ; isZero -> offset 0
ldp  x10, x11, [x22, #0x28]  ; prefix.begin / prefix.end
sub / cmp / b.hs             ; bounds check: bucket < prefix.size()
add  x9, x9, x8, lsl #7      ; &buckets[bucket]      (128-byte Impl stride)
ldr  x8, [x10, x8, lsl #3]   ; prefix[bucket]
ldr  x9, [x9, #0x20]         ; buckets[bucket].buf
sub  x9, x1, x9              ; ptr - buf
add  x8, x8, x9, asr #4      ; + (ptr - buf) / 16
add  x8, x8, #0x1            ; + 1
```

This is `offsetInternalUnsafe` inlined at the call site. Call count for the whole function drops
`bl` 50 → 49 — exactly one call removed. The 4 out-of-line `beginOfNextNonEmptyBucket` call sites
are present and unchanged in both (they fire once per bucket, not per cell, so they stay off the
per-cell counts).

### The change is surgical

A normalized diff of the two disassemblies (addresses stripped, label numbers collapsed, `nop`
padding removed) over 610/634 instructions shows the **only** semantic difference is the offset
block above. Everything else differs solely in GOT/data-slot immediates and one string-literal
address — consequences of the data segment moving, not codegen changes.

---

## (d) Question 2 — the quantitative table

**Hot path definition** (identical to `N1`, so the BEFORE column is directly comparable): one
non-joined cell on the **emitting** path — flag clear, row collected, advance to the next non-empty
cell in the same bucket, `num_streams == 1`. `nop` padding excluded. The address ranges `N1` used
were mapped 1:1 onto the AFTER binary by structural correspondence, block by block; I verified each
mapped block begins and ends on the same instruction. Counted by script
(`codegen/logs/count.py`) over the tool's emitted disassembly.

Ranges counted — BEFORE caller `0x165e8b24`-`8b28`, `8b34`-`8ba0`, `8ba4`-`8bac`, `8bbc`-`8bd4`,
`8bf8`-`8c3c`, `8c68`-`8c70`; BEFORE callee `0x165f0cc0`-`0d28` + `0d48`-`0d7c` (the `b.eq` at
`0x165f0d28` skips the `std::__call_once` once the flag is set); AFTER caller `0x165e8564`-`8568`,
`8574`-`85dc`, `861c`-`863c`, `8640`-`8648`, `8658`-`8670`, `8698`-`86dc`, `8708`-`8710`.

### Per non-joined cell, emitting path

| metric | BEFORE (61 caller + 41 callee) | AFTER | delta |
|---|---:|---:|---|
| **hot-path instruction count** | **102** | **69** | **−33 / −32%** |
| loads | 28 | 21 | −7 |
| stores | 8 | 1 | −7 |
| branches (excluding calls) | 16 | 14 | −2 |
| **calls executed** | **2** | **1** | **−1** |
| spills/reloads (stack traffic) | 7 st / 5 ld | **0 st / 2 ld** | −7 st / −3 ld |
| dependent-load chain depth, structural | 4 | 4 | 0 |
| dependent-load chain depth, **as scheduled** | **7** | **4** | **−3** |
| `call_once` machinery **present** on the path | **yes** | **no** | removed |
| `call_once` **executed** per cell | no (branch-skipped) | n/a — gone | — |
| `crc32cx` re-hash per cell | **yes** | **no** | removed |
| out-of-line frame set up per cell | yes (0x70 bytes, 6 callee-saves) | **no** | removed |

### Offset computation alone (the part that differs)

| | BEFORE | AFTER |
|---|---:|---:|
| instructions | 8 (caller) + 41 (callee) = **49** | **16** |
| loads | 2 + 10 = 12 | 5 |
| stores | 0 + 7 = 7 | **0** |
| calls | 1 | **0** |
| branches | 1 + 5 = 6 | 3 |

The remaining 3 branches are the two bounds checks (`bucket < buckets.size()`,
`bucket < prefix.size()`) and the `isZero` test — all present in the source and all cheap,
correctly-predicted forward branches.

### Dependent-load chain

Traced by hand from instruction order, **not** a pipeline simulation (`llvm-mca` unavailable).

BEFORE, as the compiler scheduled it, the two chains were serialized by the call boundary — the
flags walk is emitted after the `bl` and reuses the same register the callee clobbers:
`ldr x8,[x1]` (cell key) → `crc32cx` → [call] → `ldp x8,x9,[x21,#0x28]` (prefix.begin) →
`ldr x8,[x8,x19,lsl #3]` (prefix[buck]) → [ret] → `ldr x8,[x20,#0x8]` → `ldr x8,[x8,#0x138]` →
`ldp x8,x9,[x8,#0x30]` → `ldarb` = **7 loads deep**.

AFTER there is no call boundary, and the two chains are independent, both terminating at the
`ldarb`. Offset chain: `prefix.begin` → `prefix[buck]` → `ldarb` = 3. Flags chain: `&parent` →
`used_flags` → `per_offset_flags.begin` → `ldarb` = **4**. Max = **4**, which is the same depth
`N1` measured for the single-level baseline `hash`.

### Inlining decisions

| | BEFORE | AFTER |
|---|---|---|
| `offsetInternal` / `offsetInternalAtBucket` | **not inlined** — out-of-line body at `0x165f0cc0`, 236 B / 59 insns, 0x70-byte frame saving `x19`–`x22`, `x29`, `x30` | **fully inlined** — 16 instructions, no call, no frame, no spill |
| `BucketPrefixSums::offset` | inlined into `offsetInternal`, but its `std::call_once` was not — left a `bl std::__call_once` plus a separate 280-byte `__call_once_proxy` thunk | **not on the path at all** — `offsetInternalUnsafe` does not consult the once-flag |
| `beginOfNextNonEmptyBucket` | 4 out-of-line call sites (per-bucket) | unchanged, 4 out-of-line call sites |
| whole-function size | 2440 B / 610 insns | 2536 B / 634 insns (+96 B / +24 insns) |

The whole function grew while the per-cell path shrank, which is exactly what inlining a callee
into several call sites looks like: the 41-instruction callee body is gone from the binary, and a
16-instruction version is pasted in.

### Context: how close is this to the single-level baseline?

`N1` measured the baseline `hash` single-level path at **39** instructions per cell. This fix takes
unified from 102 to 69. The remaining +30 is the bucket-aware iterator, the bucket-exhaustion
check, the bounds checks and the stream filter — none of which this change targeted. That number is
**quoted from `N1`, not re-measured in this run.**

---

## (e) Question 3 — regression check

`TwoLevelHashTable.h` is shared with `parallel_hash` (via `DB::ConcurrentHashJoin`, whose shards are
two-level baseline `HashJoin` objects) and with `DB::Aggregator`. The fix only **added** a method,
so nothing else should have changed. Rather than hand-pick two symbols, I compared **every function
in both binaries**.

### Whole-binary function-size comparison

Built from the tool's cached `llvm-nm` symbol tables (`~/.cache/analyze-assembly/<build-id>.symbols.raw`,
format `addr size type name`), comparing the multiset of sizes per mangled name over all `T`/`t`
symbols with nonzero size.

| | count |
|---|---:|
| distinct text symbol names, BEFORE / AFTER | 645,013 / 644,977 |
| names present in both with a **changed size** | **18** |
| names **only in BEFORE** (removed) | 36 |
| names **only in AFTER** (added) | **0** |

**All 18 changed symbols are `DB::Unified::NotJoinedHash::fillColumns<TwoLevelHashMapTable<...>>`
instantiations** — the `UInt64`, `UInt32`, `UInt128`, `UInt256`, `string_view` key variants × the
`RowRefList` / `RowRef` / `unique_ptr<SortedLookupVectorBase>` mapped variants. Each grew by 64, 96
or 108 bytes. Nothing else in the binary changed size.

**The 36 removed symbols are all dead-stripped consequences of this fix**, not changes to other
code:

- 18 × `TwoLevelHashTable<...>::RuntimeStorage::offsetInternal(cell const*, unsigned long)` —
  236/248/276 bytes each. The 236-byte one is precisely the callee disassembled at `0x165f0cc0`.
- 18 × `std::__call_once_proxy<... BucketPrefixSums::offset ... {lambda()#1}>` — 280 bytes each,
  matching the thunk size `N1` reports.

One `offsetInternal` body and one `__call_once_proxy` thunk per instantiation, all of which lost
their only caller. Corroborating count: `__call_once_proxy` symbols mentioning `TwoLevelHashTable`
go from **36 in BEFORE to 18 in AFTER**; the surviving 18 belong to the other `offsetInternal(ptr)`
call sites, which this change deliberately left alone.

Net `.text` accounting: −9,660 bytes removed, +1,688 bytes of growth in the 18 scans, **−7,972
bytes** overall.

### Byte-level verification of specific non-unified symbols

Size being equal does not prove the bytes are equal, so I disassembled six non-unified symbols out
of both binaries and diffed them. Absolute addresses were normalized away (the whole `.text` shifts
because 36 functions were stripped), but each branch/call target keeps its `<symbol>` annotation, so
a retargeted call would still show.

| symbol | insns B / A | opcode-stream diff | branch/call-target diff | remaining differing instructions |
|---|---|---:|---:|---|
| `DB::NotJoinedHash::fillColumns<TwoLevelHashMapTable<u64,...,HashMapTable,8>>` (**the `parallel_hash` shard scan**) | 660 / 660 | **0** | **0** | 4 data-address immediates |
| `DB::Aggregator::convertToBlockImplFinal<AggregationMethodOneNumber<UInt64, TwoLevelHashMapTable<...>>>` | 347 / 347 | **0** | **0** | 2 (one GOT-slot reference) |
| `DB::Aggregator::convertToBlockImplNotFinal<AggregationMethodOneNumber<..., TwoLevelHashMapTable<...>>>` | 317 / 317 | **0** | **0** | **0 — byte-identical** |
| `DB::Aggregator::writeToTemporaryFileImpl<AggregationMethodOneNumber<..., TwoLevelHashMapTable<...>>>` | 927 / 927 | **0** | **0** | 13 data-address immediates |
| `DB::ConcurrentHashJoin::dispatchBlock` | 865 / 865 | **0** | **0** | 11 data-address immediates |
| `DB::ConcurrentHashJoin::addBlockToJoin` | 769 / 769 | **0** | **0** | 13 data-address immediates |

In every case: the **opcode stream is identical instruction for instruction**, every branch and
call target is identical, and every differing line is an `adrp`/`add`/`ldr`/`ldp`/`mov` whose only
change is the immediate offset of a static-data reference. Those shift because `.rodata` moved when
the 36 functions were stripped. This is relocation, not recompilation.

**Answer: no non-unified symbol changed.** The regression check is clean.

---

## (f) Measured vs inferred

**Measured** (disassembled from these two binaries, counted by script or read directly off the
listing):

- Both build IDs, via `readelf -n`.
- Every instruction, load, store, branch, call and spill count in (d), for both columns.
- Every address, opcode and symbol name quoted in (c) and (d).
- `crc32*` count in the scan function: 1 BEFORE, 0 AFTER.
- `bl` count in the scan function: 50 BEFORE, 49 AFTER; the removed one is
  `RuntimeStorage::offsetInternal`, identified by its mangled name.
- `ldapr` / `ldar` / `call_once` / `__cxa_guard` / `pthread_once` count in the AFTER scan function: 0.
- The BEFORE callee's `ldapr x8,[x0]` (`0x165f0d20`), `adr x2, <__call_once_proxy>` (`0x165f0d38`)
  and `bl <std::__call_once>` (`0x165f0d44`) — i.e. `N1`'s central claims, independently re-verified.
- The whole-binary symbol counts, the 18/36/0 changed/removed/added split, the identity of all 36
  removed symbols, and the byte-level opcode-stream comparisons in (e).
- The uncommitted source diff.

**Inferred** (reasoned, not measured):

- That the removed `offsetInternal` and `__call_once_proxy` bodies were dead-stripped *because* they
  lost their only caller. Strongly supported (the counts line up one-per-instantiation, and the
  AFTER scans no longer reference them) but I did not trace the linker's reachability analysis.
- That the residual `adrp`/`add`/`ldr` immediate differences in (e) are relocations caused by the
  data segment moving. Supported by the opcode streams and call targets being identical; I did not
  dump and compare the referenced data.
- That the pristine binary corresponds to the AFTER source minus exactly this diff. Argued from the
  whole-binary comparison, not from build provenance.
- The "as scheduled" dependent-load depths (7 → 4) are a reading of the compiler's instruction
  order, not a pipeline simulation.

**Could not resolve / did not do:**

- No `llvm-mca` (not installed on this host) — **no cycle, latency, IPC or port-pressure estimate is
  offered, and none should be read into the instruction counts.** A −32% static instruction count on
  this path is not a −32% time claim.
- No `-Rpass` / optimization-remark data was collected, so "`ALWAYS_INLINE` is why it inlined" is a
  reasonable reading of the source plus the result, not a compiler-confirmed fact.
- No `--source` interleaving; source attribution is by symbol name and by matching instruction
  sequences to the source, not by DWARF line tables.
- The `beginOfNextNonEmptyBucket` callee was identified by its call sites in both binaries and
  confirmed unchanged in count, but not separately disassembled and counted — it is per-bucket
  (one bucket at `max_threads=1`) and therefore off the per-cell path.
- Nothing here was profiled. This artifact says the intended instructions left the binary. It does
  **not** say how much wall time that is worth; that is for the A7 performance measurement to
  report, and this check is what makes that measurement attributable to this change.
