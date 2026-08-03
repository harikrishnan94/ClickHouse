# P1 (probe row loop) and G2 (result-gather row loop): codegen comparison, `hash` vs `unified_hash`

Binary: `tmp/uhj_parity/perf2/bin/clickhouse.ref` (aarch64, Neoverse-V2, RelWithDebInfo).
Both trees are present in the **same** binary — baseline in namespace `DB::`, unified in
`DB::Unified::` — so every number below is a comparison of two functions in one linked image,
with no build-to-build variance to explain away.

Tools: `tmp/uhj_parity/perf2/codegen.py` (`syms` / `dis` / `count` / `mca`), `llvm-nm --print-size`,
plus two small read-only helpers written for this unit and kept next to the artefact:
`tmp/uhj_parity/perf2/tmp/backedges.py` (lists loop back-edges in a `dis` artefact) and
`tmp/uhj_parity/perf2/tmp/normdiff.py` (namespace-insensitive instruction-stream diff).
Nothing under `src/` was modified.

---

## 0. Headline

1. **G2's "inlining difference" is not a codegen difference. It is refuted.**
   `LazyOutput::buildOutputFromBlocks<true>` exists out of line on **both** sides, at the **same
   size** (`0x634`), and disassembles to **397 instructions with zero positional mismatches** after
   namespace normalisation. `HashJoinResult::generateBlock` is `0x1cf4` on both sides, 1853
   instructions on both sides. The gather loop is byte-for-byte the same code, and `llvm-mca`
   prices it identically (6.09 cycles/iteration, IPC 3.78, Block RThroughput 5.2 on both sides).
   The profiler's asymmetric attribution is an artefact of the sample-aggregation filter, traced
   to its exact cause in §4.3.

2. **P1 is a real and large codegen difference.** The unified probe loop is
   **+49 instructions (123 → 172, +40%)**, **+15 loads**, **+9 spill reloads**, and its
   dependent-load-chain depth goes **3 → 5**. On the modelled hot path `llvm-mca` prices the
   unified iteration at **12.30 vs 8.19 cycles** (no-prefetch) and **15.79 vs 10.41 cycles**
   (prefetch), i.e. **+50%** in both configurations. Both figures are **lower bounds** (§3.4).

3. **The single most important instruction-level difference** is that
   `TwoLevelHashTable::Prober` does not stay in registers. Its `shift`, `max_bucket`, `prefix` and
   `routed_prefix` fields are spilled to the stack and reloaded on **every row**, and the resulting
   `routed = buckets + bucket` pointer inserts an extra level into the dependent-load chain ahead
   of the two loads (`mask`, `buf`) that the baseline reads at fixed offsets from a register it
   already holds. See §3.5.

---

## 1. Which instantiations, and how they were chosen

Not chosen by guesswork: the mission's own profiler output
(`tmp/uhj_parity/perf2/results/samples_u0a.jsonl`, collected by `enumerate.py collect`) records the
exact demangled symbol for every sample. The P1 pair below is the top
`HashMethodOneNumber<..., unsigned long, ...>` `joinRightColumns` entry in each algo's aggregate.

| | baseline (`hash`) | unified (`unified_hash`) |
|---|---|---|
| samples | 4484 | 5870 |
| address | `0x14349f00` | `0x168ade80` |
| `llvm-nm` size | `0x598` (1432 B) | `0x744` (1860 B) |
| map | `HashMapTable<unsigned long, HashMapCell<..., HashCRC32<unsigned long>, ...>>` | `TwoLevelHashMapTable<unsigned long, HashMapCell<..., HashCRC32<unsigned long>, ...>, ..., HashMapTable, -1>` |
| template tail | `false, true, DB::AddedColumns<true>, std::pair<unsigned long, unsigned long>` | `false, true, DB::Unified::AddedColumns<true>, std::pair<unsigned long, unsigned long>` |

So on both sides: `need_filter = false`, `fast_path = true`, `AddedColumns<true>`,
`Selector = std::pair<size_t, size_t>` (a contiguous row range, not `Indexes`), `KIND = FULL`,
`STRICTNESS = All`. The overload is the **single-map** one
(`joinRightColumns(KeyGetter &, const Map *, ...)`, `flag_per_row = false`), confirmed by the
argument list in the demangled name. The two sides differ in exactly one template argument that
matters — the map type — which is the intended comparison.

`need_filter = false` is forced by `FULL`/`All` (`need_replication` is true), so no other
`need_filter` variant could have run; both `fast_path` and both `Selector` variants exist in the
binary and their sizes are listed in `tmp/uhj_parity/perf2/tmp/pick_p1.py` output, but only the
pair above received samples.

Symbol resolution was pinned before disassembly:

```
$ python3 tmp/uhj_parity/perf2/codegen.py syms --binary .../clickhouse.ref --symbol '<regex>'
[0] 0x14349f00  unsigned long DB::HashJoinMethods<(DB::JoinKind)3, ...
[0] 0x168ade80  unsigned long DB::Unified::HashJoinMethods<(DB::JoinKind)3, ...
```

Both regexes match exactly one symbol each (`matches=1`), so there is no index ambiguity.

**Source.** `src/Interpreters/HashJoin/HashJoinMethodsImpl.h:594-631` versus
`src/Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h:704-741`.

---

## 2. How the loop bodies were isolated

`tmp/backedges.py` lists every branch whose target is a lower address, with the implied span. The
row loop is the one whose back-edge target is the block containing the induction variable update
(`add x25, x25, #1; cmp x25, x26`), and whose span covers the `crc32cx` key hash.

### 2.1 P1 baseline — `0x1434a1a8 .. 0x1434a3d8`

Rotated loop. The latch is `<L14>` at `0x1434a1d8`:

```
<L14> 1434a1d8: ldr  x9, [x19, #0xc0]        ; added_columns.offsets_to_replicate
      1434a1dc: add  x27, x8, x27            ; current_offset += n
      1434a1e0: str  x27, [x9, x25, lsl #3]  ; offsets_to_replicate[i] = current_offset
      1434a1e4: add  x25, x25, #0x1          ; ++i
      1434a1e8: cmp  x25, x26                ; i < rows
      1434a1ec: b.eq <L7>                    ; loop exit
<L15> 1434a1f0: tbz  w28, #0x0, <L18>        ; loop entry: `if constexpr (can_prefetch)`
```

The lowest in-loop address is `<L11>` `0x1434a1a8` (reached by `b <L11>` at `0x1434a344`); the
highest is `0x1434a3d8` (`b <L19>`, the tail of the prefetch-distance recalibration block `<L27>`).
Everything from `0x1434a3dc` on is cold: `__libcpp_verbose_abort`, `throw_alloc_error`,
`__throw_out_of_range` and the `system_error` construction — the out-of-range and allocation-failure
paths, not part of any iteration that returns.

### 2.2 P1 unified — `0x168ae200 .. 0x168ae4f0`

Same shape. Latch `<L18>` at `0x168ae208` (`add x25, x25, #1; cmp x25, x26; b.eq <L13>`), entry
`<L19>` at `0x168ae220`. Lowest in-loop address `<L17>` `0x168ae200`, highest `0x168ae4f0`
(`b <L20>`, tail of the same prefetch recalibration block). Cold from `0x168ae4f4`.

### 2.3 G2 — the gather loop

The C++ is `LazyOutput::buildOutputFromBlocks<true>`,
`src/Interpreters/HashJoin/AddedColumns.cpp:254-258` (and the identical
`src/Interpreters/UnifiedHashJoin/AddedColumns.cpp`): the inner
`for (const UInt64 ref_word : refsOf(*row_ref_i))` that pushes one `(block pointer, row number)`
pair per output row. That is the per-output-row loop, and it maps onto a single clean cycle
`<L14> -> <L15> -> <L16> -> <L17> -> <L14>`, contiguous in address order:

| side | range | back-edge |
|---|---|---|
| baseline | `0x144807c4 .. 0x14480818` | `0x14480818: b <L14>` -> `0x144807c4` |
| unified | `0x1676d204 .. 0x1676d258` | `0x1676d258: b <L14>` -> `0x1676d204` |

The two ranges are the same offsets within their functions (`0x1676cf80 - 0x14480540 = 0x22eca40`,
and `0x1676d204 - 0x144807c4 = 0x22eca40`). 22 instructions, **no calls**, single path — this is
the one block in this unit that `llvm-mca` models without any caveat.

The loop body, baseline (unified is identical, see §4.2):

```
<L14> 144807c4: mov   x23, x28
      144807c8: cbz   x24, <L15>
      144807cc: ldr   x23, [x24]                 ; ref word from the batch
<L15> 144807d0: ldp   x22, x9, [sp, #0x38]       ; many_columns end/cap
      144807d4: ubfx  x19, x23, #32, #31         ; refWordBlockNo
      144807d8: ldr   x21, [x20, #0x20]          ; stored_columns data
      144807dc: cmp   x22, x9
      144807e0: b.hs  <L20>                      ; grow (out of loop)
      144807e4: ldr   x8, [x21, x19, lsl #3]     ; stored_columns[block_no]
      144807e8: str   x8, [x22], #0x8            ; many_columns.emplace_back
<L16> 144807ec: ldp   x19, x9, [sp, #0x50]       ; row_nums end/cap
      144807f0: str   x22, [sp, #0x38]
      144807f4: cmp   x19, x9
      144807f8: b.hs  <L21>                      ; grow (out of loop)
      144807fc: str   w23, [x19], #0x4           ; row_nums.emplace_back(refWordRowNo)
<L17> 14480800: str   x19, [sp, #0x50]
      14480804: cbz   x24, <L18>
      14480808: add   x24, x24, #0x8             ; ++ref_word
      1448080c: cmp   x24, x27
      14480810: b.eq  <L19>                      ; end of this RowRefList batch
      14480814: cbz   x24, <L13>
      14480818: b     <L14>
```

---

## 3. P1 — results

### 3.1 Static instruction classes over the **whole** loop body

`codegen.py count`, full loop range (all paths, including the cold-ish reserve and prefetch-tuning
blocks that are inside the loop):

```
P1 base FULL loop  insns= 123 loads= 29 stores= 5 branches= 28 calls= 4 spill(st/ld)=0/2 dep_load_depth=3  [nops excluded 18]
P1 uni  FULL loop  insns= 172 loads= 44 stores= 7 branches= 36 calls= 4 spill(st/ld)=2/9 dep_load_depth=5  [nops excluded 17]
```

| metric | baseline | unified | delta |
|---|---:|---:|---:|
| instructions | 123 | 172 | **+49 (+40%)** |
| loads | 29 | 44 | +15 |
| stores | 5 | 7 | +2 |
| branches | 28 | 36 | +8 |
| calls | 4 | 4 | 0 |
| spill stores / reloads | 0 / 2 | 2 / 9 | **+2 / +7** |
| dependent-load-chain depth | 3 | 5 | **+2** |

### 3.2 Static instruction classes over the modelled **hot path**

Hot path = one row that is not skipped, finds a match on the **first** probe (no collision-chain
iteration), and does not trigger the `offsets_to_replicate` reserve. Two configurations, because
`use_prefetch` is decided at runtime by `shouldUseJoinPrefetch` and the profile does not record
which way it went — so both are reported rather than one being asserted.

```
P1 base hot NOPREFETCH  insns= 45 loads= 12 stores= 2 branches= 11 calls= 2 spill(st/ld)=0/1 dep_load_depth=3
P1 uni  hot NOPREFETCH  insns= 65 loads= 20 stores= 3 branches= 13 calls= 2 spill(st/ld)=1/6 dep_load_depth=5

P1 base hot PREFETCH    insns= 59 loads= 15 stores= 2 branches= 14 calls= 2 spill(st/ld)=0/1 dep_load_depth=3
P1 uni  hot PREFETCH    insns= 85 loads= 25 stores= 3 branches= 15 calls= 2 spill(st/ld)=1/8 dep_load_depth=5
```

Address ranges used (execution order; `codegen.py` emits them in address order, see §3.4):

- baseline, no prefetch: `0x1434a1f0:0x1434a1f0`, `0x1434a280:0x1434a2ac`, `0x1434a2c0:0x1434a2c4`,
  `0x1434a348:0x1434a3a4`, `0x1434a1d8:0x1434a1ec`
- baseline, prefetch: `0x1434a1f0:0x1434a1f8`, `0x1434a238:0x1434a26c`, `0x1434a288:0x1434a2ac`,
  `0x1434a2c0:0x1434a2c4`, `0x1434a348:0x1434a3a4`, `0x1434a1d8:0x1434a1ec`
- unified, no prefetch: `0x168ae220:0x168ae224`, `0x168ae2e0:0x168ae2f8`, `0x168ae384:0x168ae3b8`,
  `0x168ae3c0:0x168ae3c4`, `0x168ae3e0:0x168ae3e4`, `0x168ae400:0x168ae47c`, `0x168ae208:0x168ae21c`
- unified, prefetch: `0x168ae220:0x168ae22c`, `0x168ae26c:0x168ae298`, `0x168ae350:0x168ae3b8`,
  `0x168ae3c0:0x168ae3c4`, `0x168ae3e0:0x168ae3e4`, `0x168ae400:0x168ae47c`, `0x168ae208:0x168ae21c`

### 3.3 `llvm-mca`, `-mcpu=neoverse-v2`, 100 iterations

| block | cycles/iter | IPC | Block RThroughput | uops/iter | limiting resource |
|---|---:|---:|---:|---:|---|
| P1 base, no prefetch | **8.19** | 5.37 | 7.8 | 47 | dispatch width (6/cyc); most-pressed port `V2UnitL01` 5.00 |
| P1 uni, no prefetch  | **12.30** | 5.20 | 11.7 | 70 | dispatch width; `V2UnitL01` 8.02 |
| P1 base, prefetch    | **10.41** | 5.57 | 10.2 | 61 | dispatch width; `V2UnitL01` 6.13 |
| P1 uni, prefetch     | **15.79** | 5.32 | 15.2 | 91 | dispatch width; `V2UnitL01` 10.22 |
| P1 uni, no prefetch, `sole != nullptr` | 10.59 | 5.38 | 10.2 | 61 | dispatch width; `V2UnitL01` 6.67 |

`llvm-mca` reports *"No resource or data dependency bottlenecks discovered"* for every block, and
Block RThroughput equals uops/iteration ÷ 6 in every case (47/6 = 7.83, 70/6 = 11.67, 61/6 = 10.17,
91/6 = 15.17). **The limiter is the 6-wide dispatch/rename, not a functional unit** — which means
the +40% instruction count translates almost linearly into cycles. The most-pressed *port* group is
`V2UnitL01`, the two load/store pipes, consistent with the extra work being loads.

Unified is **+50.2%** cycles without prefetch and **+51.7%** with prefetch.

The last row is the unified `Prober` fast path (`sole != nullptr`: the two-level table routed to a
single bucket, `TwoLevelHashTable.h:556-557`). Even on that path unified is +29% over baseline
without prefetch, because `Prober::offsetInternal` still pays its spill reload (§3.5c). The
profiled configuration is a real two-level map, so the `sole == nullptr` rows are the ones to
compare; the fast-path row is included so the comparison cannot be accused of picking the worst
unified path.

Full reports: `tmp/uhj_parity/perf2/codegen/logs/mca/P1_{base,uni}_{noprefetch,prefetch}.txt`,
`P1_uni_noprefetch_sole.txt`.

### 3.4 Caveats that travel with these numbers

- **The cycle counts are LOWER BOUNDS.** Both hot paths contain two `bl`s that `llvm-mca` cannot
  model and that `codegen.py` therefore drops, reporting the fact:
  `bl <...__shared_weak_countD2Ev>` (an ICF-folded name, see §4.4) and
  `bl <DB::…AddedColumns<true>::appendFromBlock(unsigned long, bool)>`. The modelled number is the
  cost of the loop body **excluding both callees**. The same two calls appear on **both** sides at
  the same point, so the *delta* is not distorted by the omission — but the absolute per-row cost
  is higher than stated on both sides.
- **Block order is address order, not execution order.** `codegen.py`'s range selection sorts by
  address, so in the assembled block the loop latch (lowest address) is placed first rather than
  last. This can only affect a latency-limited result; since every block came back
  dispatch-width-limited with no dependency bottleneck found, the headline cycles/iteration is not
  sensitive to it. The dependent-load-chain depths in §3.1/§3.2 are computed by `codegen.py count`
  on the same address-ordered scan and are documented by the tool as not following control flow.
- **Branches leaving the block were retargeted** to the block end so the extract assembles; every
  such rewrite is listed in the tool output preserved in the mca reports. This changes nothing
  about resource usage.
- The hot path assumes a **first-probe hit**. Collision-chain iterations (`<L20>` baseline,
  `<L25>`/`<L29>` unified) are excluded on both sides; they are 7 instructions per extra probe on
  both sides and do not differ.

### 3.5 The instruction-sequence delta

**(a) The lookup: fixed offsets from a live register vs a computed bucket pointer with spill reloads.**

Baseline (`<L19>`/`<L20>`, `0x1434a288`): `x22` is the map, live in a register for the whole loop.

```
1434a288: add  x8, x8, x25
1434a28c: ldr  x8, [x9, x8, lsl #3]   ; key
1434a290: cbz  x8, <L21>              ; zero-key cell
1434a294: crc32cx w9, w24, x8         ; hash
1434a298: ldr  x10, [x22, #0x48]      ; map->mask        <- fixed offset off a live reg
1434a29c: ldr  x11, [x22, #0x20]      ; map->buf         <- fixed offset off a live reg
1434a2a0: and  x9, x10, x9
1434a2a4: lsl  x12, x9, #4
1434a2a8: ldr  x12, [x11, x12]        ; probe
```

Unified (`<L23>`/`<L28>`, `0x168ae2ec`) — this is `TwoLevelHashTable::Prober::find`,
`src/Common/HashTable/TwoLevelHashTable.h:554-563`:

```
168ae2ec: add  x9, x9, x25
168ae2f0: ldr  x8, [x8, x9, lsl #3]   ; key
168ae2f4: crc32cx w9, w23, x8         ; hash
168ae2f8: cbz  x20, <L28>             ; if (sole) ...    <- extra branch, every row
<L28>
168ae384: ldp  x10, x11, [sp, #0x10]  ; SPILL RELOAD: Prober::shift, Prober::max_bucket
168ae388: lsr  x10, x9, x10           ; hash >> shift
168ae38c: and  x10, x10, x11          ; & max_bucket     -> bucket
168ae390: ldr  x11, [sp]              ; SPILL RELOAD: Prober::prefix
168ae394: add  x21, x28, x10, lsl #7  ; routed = buckets + bucket   <- new pointer
168ae398: ldr  x10, [x11, x10, lsl #3]; prefix[bucket]
168ae39c: str  x10, [sp, #0x28]       ; SPILL STORE: Prober::routed_prefix
168ae3a0: cbz  x8, <L31>
168ae3a4: ldr  x10, [x21, #0x48]      ; routed->mask     <- offset off a COMPUTED pointer
168ae3a8: and  x11, x10, x9
168ae3ac: ldr  x9, [x21, #0x20]       ; routed->buf      <- offset off a COMPUTED pointer
168ae3b0: lsl  x12, x11, #4
168ae3b4: ldr  x12, [x9, x12]         ; probe
```

Net, per row, on the lookup alone: **+1 branch, +3 stack accesses (2 reloads via `ldp`+`ldr`, 1
store), +1 load of `prefix[bucket]`, +3 ALU (`lsr`, `and`, `add …lsl #7`)**. And structurally: the
baseline's `mask`/`buf` loads issue as soon as the loop starts, because `x22` never changes; the
unified ones cannot issue until `x21` is computed, which needs the hash, which needs the key load.
That is the +2 on the dependent-load-chain depth.

The `cbz x20` also **duplicates the collision-probe loop** — the unified function contains two
copies (`<L25>` at `0x168ae320` and `<L29>` at `0x168ae3c0`) against the baseline's one (`<L20>`),
which is a large part of the `0x598 -> 0x744` growth in function size.

**(b) The prefetch, when enabled, pays the bucket computation a second time.** Baseline `<L17>`
(`0x1434a24c`) is 8 instructions: load key, `crc32cx`, `and` with `[x22,#0x48]`, `lsl`, `prfm`.
Unified `<L27>` (`0x168ae350`) is 13, and repeats the whole `lsr`/`and`/`add …lsl #7` routing plus
its own `ldp x11, x12, [sp, #0x10]` spill reload
(`TwoLevelHashTable.h:588`, `buckets[(… >> shift) & max_bucket].prefetchByHash(key_hash)`).
That is why the prefetch delta (+5.38 cycles) is larger than the no-prefetch delta (+4.11 cycles).

**(c) `offsetInternal`: one `add` vs eight instructions including a spill reload.**

Baseline, `<L23>` at `0x1434a348` — for a single-level table the slot index is already in `x9`, so
the global cell offset is one increment:

```
1434a348: add  x24, x11, x9, lsl #4   ; &cell
1434a34c: add  x8,  x9,  #0x1         ; offsetInternal = place + 1
```

Unified, `<L32>` at `0x168ae400` — `Prober::offsetInternal`,
`src/Common/HashTable/TwoLevelHashTable.h:571-575`, must recover the offset within the routed
bucket and then add the bucket's prefix, which it has to reload from the stack:

```
168ae400: ldr   x8, [x21, #0x20]      ; routed->buf
168ae404: ldr   x9, [x22]             ; cell key (to test the zero cell)
168ae408: ldr   x11, [sp, #0x28]      ; SPILL RELOAD: routed_prefix
168ae40c: sub   x8, x22, x8
168ae410: cmp   x9, #0x0
168ae414: asr   x8, x8, #4
168ae418: csinc x9, xzr, x8, eq       ; offset_in_bucket
168ae420: add   x11, x9, x11          ; routed_prefix + offset_in_bucket
168ae424: cmp   x9, #0x0
168ae42c: csel  x9, xzr, x11, eq      ; offset_in_bucket ? … : 0
```

**+8 instructions and +3 loads per matched row**, one of them a spill reload. This is paid on the
`sole != nullptr` fast path too, which is why that path is still +29%.

**(d) `setUsed` is identical.** `ldrb`/`tbnz`/`mov`/`strb` against `used_flags`, same four
instructions at `0x1434a364` and `0x168ae43c`. It is not part of the delta.

---

## 4. G2 — results, and the disposal of the inlining hypothesis

### 4.1 Symbol sizes (`llvm-nm --defined-only --demangle --print-size`)

| symbol | baseline | unified |
|---|---|---|
| `HashJoinResult::generateBlock` | `0x14a1a180` size **`0x1cf4`** (7412 B), `T` | `0x16c29e80` size **`0x1cf4`** (7412 B), `T` |
| `LazyOutput::buildOutputFromBlocks<true>` | `0x14480540` size **`0x634`** (1588 B), `W` | `0x1676cf80` size **`0x634`** (1588 B), `W` |
| `LazyOutput::buildOutputFromBlocks<false>` | `0x1447fac0` size `0x3dc` (988 B), `W` | `0x1676c500` size `0x3dc` (988 B), `W` |
| `LazyOutput::buildOutput` | `0x1447f8c0` size `0x1e8` (488 B), `T` | `0x1676c300` size `0x1e8` (488 B), `T` |
| `LazyOutput::buildOutputFromBlocksLimitAndOffset` | `0x1447fec0` size `0x65c` (1628 B), `T` | `0x1676c900` size `0x65c` (1628 B), `T` |

Answering the mission's questions directly:

- **Does `DB::LazyOutput::buildOutputFromBlocks<true>` exist as a separate symbol in the baseline?**
  **Yes.** `0x14480540`, weak, 1588 bytes.
- **Does `DB::Unified::HashJoinResult::generateBlock` exist, and how big is it versus the
  baseline's?** **Yes**, and it is **exactly the same size**: `0x1cf4` on both sides.
- **Is the baseline symbol much larger and the unified one much smaller?** **No. Every one of the
  five sizes is identical to the byte.** There is no inlining difference to see in the sizes.

None of these addresses is shared with another symbol, so none of them is an ICF alias — checked by
looking up every symbol at each of the six addresses.

### 4.2 Instruction streams

`tmp/normdiff.py` strips addresses, rewrites `N2DB7Unified`/`DB::Unified::` to the baseline
spelling, and compares position by position:

```
G2_base_bofb.asm: 397 insns
G2_uni_bofb.asm:  397 insns
positional mismatches: 0 of 397          <-- buildOutputFromBlocks<true>

G2_base_generateBlock.asm: 1853 insns
G2_uni_generateBlock.asm:  1853 insns
positional mismatches: 35 of 1853        <-- generateBlock

G2_base_buildOutput.asm: 122 insns
G2_uni_buildOutput.asm:  122 insns
positional mismatches: 3 of 122          <-- buildOutput
```

`buildOutputFromBlocks<true>` — the function that holds the gather loop — is **identical**, not
merely the same size. The 35 mismatches in `generateBlock` and the 3 in `buildOutput` are all call
operands only: `bl X` versus `bl __AArch64ADRPThunk_X` (range-extension thunks chosen by the linker
from where each copy landed in a 5 GB image), plus one `std::__hash_table::find` instantiated over
`basic_string_view` on one side and over `__hash_value_type<basic_string_view, size_t>` on the
other. Not one of them is in a loop body, and neither function's instruction count changes.

Both `generateBlock`s reach the gather through the same two out-of-line hops: 208 calls each, 60
unique targets each, and the call-target sets are equal after namespace normalisation.
`buildOutputFromBlocks` is **not** called from `generateBlock` on either side (zero references in
either disassembly); `generateBlock` calls `LazyOutput::buildOutput`, and `buildOutput` calls
`buildOutputFromBlocks<false>`, `buildOutputFromBlocks<true>`,
`buildOutputFromBlocksLimitAndOffset` and `buildOutputFromRowRefLists` — the same four dispatch
targets on both sides (`AddedColumns.cpp:89-106`). Nothing is inlined on either side, and
`buildOutput` at 488 bytes could not contain the 1588-byte callee in any case.

### 4.3 So why did the profiler attribute the samples differently?

Not codegen. The aggregation filter. `enumerate.py:106-140` (`leaf_in_join_symbols`) does not record
the leaf frame — it walks the stack from the leaf and records **the innermost frame that matches
`IN_JOIN_MARKERS`** (`loops.py:23-45`):

```python
for f in frames:
    if in_join.search(f):
        agg[f] = agg.get(f, 0) + int(n)
        break
```

The marker list contains the blanket pattern `r"DB::Unified::"`, which matches
`DB::Unified::LazyOutput::buildOutputFromBlocks<true>`. It contains **no** pattern that matches the
baseline twin: the demangled name
`void DB::LazyOutput::buildOutputFromBlocks<true>(unsigned long, std::vector<COW<DB::IColumn>::mutable_ptr<DB::IColumn>, …>&, unsigned long const*, unsigned long const*)`
contains none of `DB::HashJoin`, `AddedColumns`, `HashJoinResult`, `RowRefList`, `HashMapTable`,
`ColumnsHashing` or any other marker — `LazyOutput` is simply not in the list.

So a baseline sample whose true leaf is `buildOutputFromBlocks<true>` is pushed up the stack until
it reaches `DB::HashJoinResult::generateBlock`, which does match (`HashJoinResult`), while the
identical unified sample stops at the leaf. The observation is one-sided attribution granularity,
and the numbers say so themselves: baseline `generateBlock` 34144, unified
`buildOutputFromBlocks<true>` 34257 plus `generateBlock` 182 — the same ~34.2k samples, split one
frame apart. `parallel_hash` shows the same baseline-side rollup (41519 in `generateBlock`,
nothing in `buildOutputFromBlocks`), as it must, since it runs the same `DB::` code.

**This is a defect in the sample aggregation, not in either join implementation.** Adding
`LazyOutput` to `IN_JOIN_MARKERS` would make the two sides comparable. It is worth noting that this
same asymmetry — one blanket `DB::Unified::` marker against an enumerated list of baseline names —
will roll up *every* baseline frame whose class name is not in the list, so G2 is unlikely to be
the only place it bites.

### 4.4 The gather loop, priced

Because §4.2 proves the code is identical, the measurements are identical, and both were run rather
than one being asserted:

```
G2 base gather loop  insns= 22 loads= 5 stores= 4 branches= 7 calls= 0 spill(st/ld)=2/2 dep_load_depth=2
G2 uni  gather loop  insns= 22 loads= 5 stores= 4 branches= 7 calls= 0 spill(st/ld)=2/2 dep_load_depth=2
```

| block | cycles/iter | IPC | Block RThroughput | uops/iter | limiting resource |
|---|---:|---:|---:|---:|---|
| G2 baseline | **6.09** | 3.78 | 5.2 | 31 | dispatch width (31/6 = 5.17); `V2UnitL01` 3.50 |
| G2 unified  | **6.09** | 3.78 | 5.2 | 31 | dispatch width; `V2UnitL01` 3.50 |

`llvm-mca` reports *"No resource or data dependency bottlenecks discovered"* for both. **No calls
were dropped from this block**, so unlike P1 these are not lower bounds — they are the modelled
cost of the loop body as written. The five retargeted branches (the two vector-growth exits, the
batch-end exit and the two `cbz x24`) are listed in the reports; they leave the block in the
modelled steady state and do not change its resource usage.

**Instruction-sequence delta between the sides: none.** Zero of 22 instructions differ.

Full reports: `tmp/uhj_parity/perf2/codegen/logs/mca/G2_{base,uni}_gather.txt`.

### 4.5 An incidental finding: ICF makes some disassembly operands lie

`llvm-objdump` prints, inside **baseline** `DB::HashJoinResult::generateBlock`, a call to
`DB::Unified::LazyOutput::buildJoinGetOutput`, and inside **baseline** `DB::LazyOutput::buildOutput`
a call to `DB::Unified::LazyOutput::buildOutputFromRowRefLists`; the baseline P1 loop likewise calls
something printed as `DB::Unified::AddedColumns<true>::appendFromBlock`. These are identical-code-
folded pairs (the census in `tmp/uhj_parity/perf2/codegen/icf_census.json` records 31 such pairs,
the first entry being `DB::AddedColumns<true>::applyLazyDefaults` folded with its unified twin), and
the disassembler prints whichever name the symbol table offers first for the shared address. The
`bl <__AArch64ADRPThunk__ZNSt3__119__shared_weak_countD2Ev>` in both P1 hot paths is the same
phenomenon. **A cross-namespace call in this binary's disassembly is not evidence that one tree
calls into the other**; a claim of that kind has to be checked against the ICF census first.

---

## 5. Artefacts

- disassembly: `tmp/uhj_parity/perf2/codegen/logs/{P1_base,P1_uni,G2_base_generateBlock,G2_uni_generateBlock,G2_base_bofb,G2_uni_bofb,G2_base_buildOutput,G2_uni_buildOutput}.asm`
- `llvm-mca` reports: `tmp/uhj_parity/perf2/codegen/logs/mca/*.txt`
- helpers: `tmp/uhj_parity/perf2/tmp/{backedges.py,normdiff.py,pick_p1.py}`
- symbol sizes: `tmp/uhj_parity/perf2/tmp/nm_sizes.txt`
