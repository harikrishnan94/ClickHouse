# G-disasm-build — AMAC build-insert ring steady-loop comparison (U2.3, PREREG-006d)

Date: 2026-07-27.
Comparison: instruction-SEMANTICS equivalence of the STEADY LOOP (not byte identity), per the
prereg criterion: identical prefetch opcodes+localities (write-intent `prfm pstl1keep`),
comparable per-visit load/store/branch counts, no per-visit reloads of policy fields
(frame-copy SSA held), inlined refill, bare `++`-style advance (tail-padded grower:
increment + compare-against-bufsize + select-zero).

Binaries (both aarch64):
- CANDIDATE: `tmp/chj_amac/bins/uncommitted-amacbuild.tmp.bin`
  sha256 `9166ec8df10d2119759a6d00cee2e09b93caa0150df9819c5693567a71573132` (verified).
  Code: `src/Interpreters/HashJoin/AmacBuild.cpp` (`AmacBuildInsertPolicy`, explicit
  `amacBuildInsert` instantiations), `src/Interpreters/HashJoin/AmacRing.h` (`amacRun` — inlined
  into the instantiations), `src/Interpreters/HashJoin/ResumableHashMap.h`
  (`TailPaddedHashTableGrower`, `cursor*` API).
- REFERENCE: `tmp/chj_amac/bins/clickhouse-ahj-cf465cfbe23.bin` (branch `ahj`).
  Code: `ahj:src/Interpreters/PartitionedHashJoin/PartitionedHashJoinBuild.cpp`
  (`AmacBuildInsertPolicy` in anon namespace, `insertSectionImpl`),
  `ahj:src/Interpreters/PartitionedHashJoin/AmacRing.h`,
  `ahj:src/Interpreters/PartitionedHashJoin/PartitionedJoinMaps.h` (identical
  `TailPaddedHashTableGrower`).

Method: `llvm-nm-22 --defined-only --print-size --demangle` for symbol location;
`llvm-objdump-22 -d --start-address/--stop-address` for ranged disassembly (no
analyze-assembly.py, no symbol cache). Loop identification by back-edge analysis
(`cmp …, #0x20` sweep counters) cross-checked against the source structure; family
identification inside the reference's mega-symbol by the mangled map type of each ring's
inlined `resize` call and by the key-fetch pattern (offsets-pair `ldp` = `key_string`,
fixed-stride `madd` = `key_fixed_string`, prepared-keys 32-byte block = `keys256`).
Counts below were taken by hand from the listings (the asmdiff.py `b*`-prefix branch
classifier was not used).

In the candidate each anchor is a standalone weak symbol
(`amacBuildInsert<KeyGetter, ResumableHashMap<…>, selector_is_range>` → `run_ring`
lambda `operator()<PosT>`). In the reference all build rings are inlined into ONE local
symbol: `DB::PartitionedHashJoin::insertLeafSection(...)::$_0::operator()<PartitionedJoinMapsTemplate<DB::RowRefList>>`
at `0x16871380`, size `0x17ea0`. Anchors compare the `PosT=UInt32` rings on both sides
(the common dispatch); the candidate's `selector_is_range=true` instantiation is used, which
matches the reference's single-leaf `RowRef(block_no, row)` path most closely (the selector is
only touched in the refill, never in the step).

Disassembly files (full listings): `tmp/chj_amac/disasm/cand_key64_range_u32.asm`,
`cand_keys256_range_u32.asm`, `cand_key_string_range_u32.asm`, `ahj_rowreflist_lambda.asm`.

---

## Anchor 1 — key64 / RowRefList (`HashMapCell<UInt64, RowRefList, HashCRC32<UInt64>>`, `MapsAll`)

Symbols:
- CANDIDATE: `amacBuildInsert<HashMethodOneNumber<PairNoInit<u64,RowRefList>,…>, ResumableHashMap<HashMapTable<u64, HashMapCell<u64,RowRefList,HashCRC32<u64>>, HashCRC32<u64>, TailPaddedHashTableGrower<8>, Allocator<true,true>>>, true>::…operator()<unsigned int>`
  — addr `0x140f4bc0`, size `0xc8c`. Steady sweep `0x140f4e80..0x140f5208`
  (advance back-edge `0x140f4fa4`, sweep back-edge `0x140f51e0`, outer `full && next<rows`
  re-entry `0x140f51fc`). All 10 `prfm` in the symbol are `pstl1keep`.
- REFERENCE: inside `insertLeafSection $_0<…RowRefList>` (`0x16871380`+`0x17ea0`); key64
  `PosT=u32` ring steady sweep at `0x16886cb8..~0x16887144` (advance back-edge `0x16886d34`);
  identified by the inlined `HashTable<unsigned long, HashMapCell<unsigned long, RowRefList,
  HashCRC32<unsigned long>>…TailPaddedHashTableGrower<8>, ZeroingHashTableAllocator>::resize`
  and the direct one-`ldr` column key fetch (`HashMethodOneNumber`; the `HashMethodKeysFixed`
  twin loops with the same map type contain `memcpy` key assembly and were excluded).

Steady-loop body, hot (collision/advance) path — CANDIDATE:

```
140f4e80: ldr  w9,  [x14, x28, lsl #2]   ; pos  = ring.pos[s]        (u32)
140f4e84: ldr  w1,  [x24, x28, lsl #2]   ; row  = ring.row[s]
140f4e88: add  x8,  x26, x9, lsl #4      ; cell = cells + pos*16
140f4e8c: ldr  x10, [x23, x1, lsl #3]    ; key  = key_col[row]
140f4e90: ldr  x11, [x8]                 ; cell->key
140f4e94: cbz  x11, 140f4ee0             ; empty -> fused claim
140f4e98: cmp  x11, x10
140f4e9c: b.ne 140f4f80                  ; mismatch -> advance
   ...                                    ; equal -> duplicate append (below)
140f4f80: ldr  x8,  [x19, #0x58]         ; grower.precalculated_buf_size (map, x19)
140f4f84: add  x10, x9, #0x1
140f4f88: cmp  x10, x8
140f4f8c: csinc x8, xzr, x9, eq          ; next = pos+1, wrap to 0 only at pad end
140f4f90: str  w8,  [x14, x28, lsl #2]   ; ring.pos[s] = next
140f4f94: lsl  x9,  x8, #4
140f4f98: prfm pstl1keep, [x26, x9]      ; write-intent, L1, keep
140f4f9c: add  x28, x28, #0x1            ; ++s
140f4fa0: cmp  x28, #0x20
140f4fa4: b.ne 140f4e80
```

Same path — REFERENCE (key64 u32 ring):

```
16886cb8: ldr  w9,  [x22, x28, lsl #2]   ; pos
16886cbc: ldr  w1,  [x19, x28, lsl #2]   ; row
16886cc0: add  x8,  x15, x9, lsl #4      ; cell
16886cc4: ldr  x10, [x12, x1, lsl #3]    ; key
16886cc8: ldr  x11, [x8]                 ; cell->key
16886ccc: cbz  x11, 16886cec             ; empty -> fused claim
16886cd0: cmp  x11, x10
16886cd4: b.ne 16886d10                  ; mismatch -> advance
   ...
16886d10: ldr  x8,  [x14, #0x58]         ; grower buf_size
16886d14: add  x10, x9, #0x1
16886d18: cmp  x10, x8
16886d1c: csinc x8, xzr, x9, eq
16886d20: str  w8,  [x22, x28, lsl #2]
16886d24: lsl  x9,  x8, #4
16886d28: prfm pstl1keep, [x15, x9]
16886d2c: add  x28, x28, #0x1
16886d30: cmp  x28, #0x20
16886d34: b.ne 16886cb8
```

Per-visit table (advance = the hot collision visit; hit = duplicate append; claim = empty-cell insert):

| path    | metric              | CANDIDATE                            | REFERENCE                                  |
|---------|---------------------|--------------------------------------|--------------------------------------------|
| advance | instructions        | 18                                   | 18                                         |
| advance | loads               | 5 (pos,row,key,cell key,buf_size)    | 5 (identical set)                          |
| advance | stores              | 1 (pos)                              | 1 (pos)                                    |
| advance | branches            | 3 (cbz, b.ne, back-edge)             | 3 (identical)                              |
| advance | prfm                | 1 × `pstl1keep`                      | 1 × `pstl1keep`                            |
| hit     | extra loads         | 2 spill (block-word, rows) + call    | 1–3 (locators spill, narrow decode) + call |
| hit     | out-of-line calls   | `RowRefList::insert` (dup append)    | `RowRefList::insert` (same)                |
| claim   | loads / stores      | 3 / 4 (key `stp`, m_size, mapped)    | 4–5 / 4 (adds locators+narrow tests)       |

Both advances are the bare tail-pad `++`: `add/cmp/csinc` against `precalculated_buf_size` —
IDENTICAL. Both reload only the map's grower field (`[map+0x58]`) per advance — required in
both designs because claims store through the same map object (`++m_size`); not a policy field.
The candidate's hot path contains zero stack loads (frame-copy SSA held: cells `x26`, key
column `x23`, skip bytes `x27`, `next`/`rows` in registers). Refill (skip-scan, zero-key
synchronous emplace, `crc32cx w, w(-1), x` seed hash, seed `pstl1keep`) is fully inlined in the
steady body on both sides.

Differences, classified:
1. Duplicate/claim completion: reference tests `locators`/`narrow_locators` (2 branches,
   1–2 spill loads, `lsr/and/orr` decode) to form the ref word; candidate ORs a precomputed
   `(block_no|0x80000000)<<32` constant. → justified (ii), completion paths only, not per-visit.
2. `m_size` at `[map+0x18]` (candidate) vs `[map+0x20]` (reference): `ZeroingHashTableAllocator`
   vs `Allocator<true,true>` base layout. → justified (iii).
3. Reference drain-and-grow re-seeds through `start` in slot order (skip + zero-key checks
   inline); candidate collects (row, slot) pairs, `bl std::__sort`, slot-preserving `reseed`.
   Cold growth path inside the sweep, reached only on `DoneNeedsGrow`. → justified (i)
   (deliberate fix of the ahj re-seed slot-movement bug; see `tmp/chj_amac/U23_DRAFT_NOTES.md`).
4. Register numbering and spill-slot layout differ throughout. → justified (iii).

UNEXPLAINED: none.

---

## Anchor 2 — keys256 / RowRefList (`HashMapCell<UInt256, RowRefList, UInt256HashCRC32>`, `MapsAll`)

Symbols:
- CANDIDATE: `amacBuildInsert<HashMethodKeysFixed<PairNoInit<wide::integer<256,u32>,RowRefList>,…>, ResumableHashMap<HashMapTable<UInt256, HashMapCell<UInt256,RowRefList,UInt256HashCRC32>, UInt256HashCRC32, TailPaddedHashTableGrower<8>,…>>, true>::…operator()<unsigned int>`
  — addr `0x1412cb00`, size `0x1338`. Steady sweep `0x1412cf40..0x1412d7c8` (advance
  back-edge `0x1412d2d4`); drain `0x1412d880..0x1412ddfc`. All 6 `prfm` are `pstl1keep`.
  16 `crc32cx` = 4 seed sites × 4-limb `UInt256HashCRC32` chain, all inline.
- REFERENCE: keys256 `PosT=u32` ring steady sweep at `0x16881800..0x16882104` inside the
  mega-symbol (identified by the inlined `HashTable<wide::integer<256,…>,…UInt256HashCRC32…>::resize`
  at `0x16881b4c`); the `PosT=u64` twin is at `0x1687b3f4..0x1687bd28`.

Steady-loop body, per-visit prologue + advance — CANDIDATE:

```
1412cf40: ldp  x8, x9, [x26, #0x80]      ; getter.prepared_keys begin/end (BY-REFERENCE getter)
1412cf44: ldr  w23, [x20, x24, lsl #2]   ; row
1412cf48: str  x22, [sp, #0x90]          ; spill next
1412cf4c: str  x24, [sp, #0x78]          ; spill s
1412cf50: cmp  x9, x8
1412cf54: b.eq 1412d2e0                  ; no prepared keys -> column-gather path
1412cf58: add  x8, x8, x23, lsl #5       ; &prepared_keys[row] (32 B)
1412cf5c: ldr  q0, [x8]      ; stur q0, [x29,#-0xb0]   ; 32-byte key bounced
1412cf64: ldr  q0, [x8,#0x10]; stur q0, [x29,#-0xa0]   ;   through the stack
1412cf6c: ldr  x24, [sp, #0x78]          ; reload s
1412cf78: ldp  x14, x12, [x29, #-0xb0]   ; key limbs 0,1
1412cf7c: ldr  w9,  [x8, x24, lsl #2]    ; pos (x8 = sp+0xb0 pos array)
1412cf80: ldr  x8,  [sp, #0xa8]          ; cells (spilled)
1412cf84: umaddl x8, w9, w10, x8         ; cell = cells + pos*40
1412cf88: ldp  x11, x10, [x29, #-0xa0]   ; key limbs 2,3
1412cf8c: ldr  x13, [x8]                 ; cell limb0
1412cf90: cbz  x13, 1412d420             ; -> limbs1-3 zero check -> claim
1412cf94: ldr  x21, [sp, #0x80]          ; (skip_bytes reload for the completion path)
1412cf98: cmp  x13, x14 ; b.ne 1412d2a0  ; per-limb early-exit equality
1412cfa0: ldr  x13, [x8,#0x8]  ; cmp/b.ne
1412cfac: ldr  x12, [x8,#0x10] ; cmp/b.ne
1412cfb8: ldr  x11, [x8,#0x18] ; cmp/b.ne
   ...
1412d2a0: ldr  x8,  [x28, #0x58]         ; grower buf_size (x28 = map)
1412d2a4: add  x10, x9, #0x1 ; cmp ; csinc x8, xzr, x9, eq
1412d2b8: str  w8,  [x10, x24, lsl #2]   ; pos
1412d2bc: mul  x9,  x8, x9(#40)
1412d2c0: ldr  x8,  [sp, #0xa8]          ; cells reload
1412d2c4: prfm pstl1keep, [x8, x9]
1412d2c8: ldp  x20, x22, [sp, #0x88]     ; restore row-array base, next
1412d2cc: add  x24, x24, #0x1 ; cmp #0x20 ; b.ne 1412cf40
```

Same path — REFERENCE (u32 ring, `0x16881800`): identical structure — prepared-span test per
visit (`ldr [sp,#0x3d8]`/`ldr [sp,#0x3d0]`, from spills), 32-byte key bounce (`ldr q0`×2 +
`stur q0`×2 + `ldp`×2), `s` spill/reload, cells from `[sp,#0x70]`, `umaddl` ×40, cell limb0
`cbz` → out-of-line limbs1–3 zero check, per-limb `ldr/cmp/b.ne` early-exit equality (the
reference's own u64 twin uses `ccmp` fusion instead — intra-reference variance showing this is
scheduling, not design), advance `ldr [x28,#0x98]` buf_size + `add/cmp/csinc` + `str w` +
`mul` ×40 + cells reload + `prfm pstl1keep` + one spill restore.

Per-visit table (advance path, mismatch on limb0):

| metric        | CANDIDATE                     | REFERENCE                         |
|---------------|-------------------------------|-----------------------------------|
| instructions  | ~40                           | ~40                               |
| loads         | 18 (incl. 2 span, 6 key-limb, 2 cells-spill, buf_size, 2 restore) | 16 (same set, 1 restore) |
| stores        | 5 (2 spills, 2 `stur q`, pos) | 5 (identical set)                 |
| branches      | 4 (span b.eq, cbz, limb0 b.ne, back-edge) | 4 (identical)         |
| prfm          | 1 × `pstl1keep`               | 1 × `pstl1keep`                   |
| full-equality | 4 limb loads both sides       | 4 limb loads both sides           |

Differences, classified:
1. Per-visit prepared-keys span load and stack-bounced key exist in BOTH: the `KeysFixed`
   getter is not copyable, so both policies run by reference (`copy_into_frame = false` in both
   sources) — the frame-copy SSA criterion does not apply to this anchor by design, and the
   codegen consequence is identical on both sides. → equivalent (same design constraint).
2. Load-count delta (18 vs 16): one extra `ldp` restore + `skip_bytes` reload placement in the
   candidate vs `mov` re-materialization in the reference. → justified (iii).
3. Grower offsets `[x28,#0x58]` vs `[x28,#0x98]`, `m_size` `0x30` vs reference layout: cell
   size/allocator-dependent map layout. → justified (iii).
4. Equality lowering: per-limb early-exit branches (candidate and reference-u32) vs `ccmp`
   chains (reference-u64). Same 4 loads; the reference itself differs between its own two
   `PosT` variants. → justified (iii).
5. Ref-word machinery and drain re-seed order on completion/growth paths, as anchor 1.
   → justified (ii), (i).

UNEXPLAINED: none.

---

## Anchor 3 — key_string / RowRefList (`HashMapCellWithSavedHash<string_view, RowRefList, DefaultHash<string_view>>`, `MapsAll`)

Symbols:
- CANDIDATE: `amacBuildInsert<HashMethodString<PairNoInit<string_view,RowRefList>,…>, ResumableHashMap<HashMapTable<string_view, HashMapCellWithSavedHash<string_view,RowRefList,DefaultHash<string_view>>, DefaultHash<string_view>, TailPaddedHashTableGrower<8>,…>>, true>::…operator()<unsigned int>`
  — addr `0x140fc540`, size `0x124c`. Steady sweep `0x140fc900..0x140fd0c8` (back-edges
  `0x140fcf3c`, `0x140fd0c4`). All 6 `prfm` are `pstl1keep`. No `bcmp`/`memcmp` anywhere in
  the symbol.
- REFERENCE: key_string `PosT=u32` ring steady sweep at `0x1687c858..0x1687cfac` in the
  mega-symbol — identified by the offsets-pair fetch `ldp …, [x8,#-0x8]` (the
  `key_fixed_string` twins at `0x16876664`/`0x1687c060` use fixed-stride `madd` instead)
  and the `HashMapCellWithSavedHash<string_view…>::resize` inside.

Steady-loop body, per-visit prologue + advance — CANDIDATE:

```
140fc900: ldr  w23, [x19, x26, lsl #2]   ; row
140fc904: add  x8,  x16, x23, lsl #3     ; &offsets[row]   (offsets base x16: REGISTER)
140fc908: ldp  x9, x10, [x8, #-0x8]      ; offsets[row-1], offsets[row]
140fc90c: subs x24, x10, x9 ; b.mi throw ; len
140fc914: ldr  w8,  [x0, x26, lsl #2]    ; pos = ring.pos[s]
140fc918: ldr  x20, [x2, x26, lsl #3]    ; hash = ring.hash[s]  (saved-hash ring, NO recompute)
140fc91c: add  x1,  x17, x9              ; data = chars_base + prev   (chars base x17: REGISTER)
140fc920: add  x21, x22, x8, lsl #5      ; cell = cells + pos*32
140fc924: ldr  x11, [x21, #0x8]          ; cell->size (zero test)
140fc928: cbz  x11, 140fc980             ; empty -> claim (persist-once)
140fc92c: ldr  x9,  [x21, #0x18] ; cmp x9, x20  ; b.ne advance   ; saved-hash prefilter
140fc938: cmp  x11, x24          ; b.ne advance                  ; size test
140fc940: ldr  x9,  [x21]        ; inline size-classed memequalWide (<8 / 8-16 / SIMD cmeq loop)
   ...
140fd0a0: ldr  x9,  [x15, #0x58]         ; grower buf_size (map x15: REGISTER)
140fd0a4: add  x10, x8, #0x1 ; cmp ; csinc x8, xzr, x8, eq
140fd0b0: str  w8,  [x0, x26, lsl #2]    ; pos
140fd0b4: lsl  x9,  x8, #5
140fd0b8: prfm pstl1keep, [x22, x9]
140fd0bc: add  x26, x26, #0x1 ; cmp #0x20 ; b.ne 140fc900
```

Same path — REFERENCE (u32 ring):

```
1687c858: ldr  w22, [x27, x28, lsl #2]   ; row
1687c85c: ldr  x8,  [sp, #0xa8]          ; offsets base: SPILL RELOAD per visit
1687c860: add  x8,  x8, x22, lsl #3
1687c864: ldp  x20, x8, [x8, #-0x8]      ; offsets pair
1687c868: subs x23, x8, x20 ; b.mi throw
1687c870: ldr  w24, [x15, x28, lsl #2]   ; pos
1687c874: ldr  x25, [x16, x28, lsl #3]   ; hash (saved-hash ring)
1687c878: add  x19, x17, x24, lsl #5     ; cell
1687c87c: ldr  x8,  [x19, #0x8]          ; cell->size
1687c880: cbz  x8, 1687c8fc              ; empty -> claim
1687c884: ldr  x9,  [x19, #0x18]
1687c888: cmp  x8, x23 ; ccmp x9, x25, #0x0, eq ; b.ne advance   ; fused size+hash prefilter
1687c894: ldr  x8, [sp,#0x90] ; ldr x0,[x19] ; … ; bl bcmp       ; byte compare OUT OF LINE
   ...
1687c8c8: ldr  x25, [sp, #0x60]          ; map: SPILL RELOAD per advance
1687c8d0: ldr  x8,  [x25, #0x58]         ; buf_size
1687c8d4: cmp ; csinc x8, xzr, x24, eq
1687c8dc: str  w8,  [x15, x28, lsl #2]
1687c8e0: lsl  x9,  x8, #5
1687c8e4: prfm pstl1keep, [x17, x9]
1687c8e8: add  x28, x28, #0x1
1687c8ec: ldr  x14, [sp, #0x80]          ; pool: SPILL RELOAD per advance
1687c8f0: cmp  x28, #0x20 ; b.ne 1687c858
```

Per-visit table (advance path, saved-hash/size mismatch):

| metric       | CANDIDATE                                  | REFERENCE                                       |
|--------------|--------------------------------------------|-------------------------------------------------|
| instructions | ~20                                        | ~24                                             |
| loads        | 8 (row, 2 offsets, pos, hash, size, saved_hash, buf_size) | 11 (same 8 + offsets-base, map, pool spill reloads) |
| stores       | 1 (pos)                                    | 1 (pos)                                         |
| branches     | 4 (b.mi, cbz, b.ne, back-edge)             | 4 (b.mi, cbz, fused-ccmp b.ne, back-edge)       |
| prfm         | 1 × `pstl1keep`                            | 1 × `pstl1keep`                                 |
| hash         | from ring (`ring.hash[s]`), no recompute   | from ring, no recompute                         |
| hit byte-cmp | inline size-classed `memequalWide` (SIMD)  | `bl bcmp` after prefilter (out-of-line)         |
| claim        | inline arena bump + `memcpy` persist-once + `stp` data/size + `stp` mapped/saved_hash | same + ref-word machinery |

Differences, classified:
1. Reference reloads the offsets base, the map pointer, and the pool pointer from stack spills
   per visit; the candidate holds all three in registers (the frame-copied getter and the
   standalone small symbol leave enough registers). Candidate strictly better; consequence of
   the reference's single 98 KB symbol carrying every family's state in one frame.
   → justified (iii) (surrounding symbol layout/register allocation).
2. Reference performs the confirmed-hit byte compare via out-of-line `bl bcmp` (with spill
   save/restore around the call); the candidate inlines `memequalWide` (SIMD `cmeq` loop for
   >16 B, scalar for <16 B) with no call. Candidate strictly better; both sides only reach this
   after the saved-hash + size prefilter. The prereg "no out-of-line call" criterion targets
   the refill — inlined in both. → equivalent (candidate strictly better; different memequal
   lowering, no criterion regressed).
3. Prefilter lowering: candidate = 2 × `cmp/b.ne` (hash, then size); reference = `cmp+ccmp+b.ne`
   fused. Same loads, same short-circuit semantics. → justified (iii).
4. Zero-test on `cell->size` (`[cell+0x8]`), persist-once claim (`keyHolderPersistKey` = arena
   bump + `memcpy`, exactly once at the claim), `stp xzr, hash` mapped/saved-hash store:
   identical on both sides. Ref-word machinery on completion paths: → justified (ii).
   `m_size`/arena field offsets differ: → justified (iii). Drain re-seed: → justified (i).

UNEXPLAINED: none.

---

## Criterion checklist (all three anchors)

| criterion | result |
|---|---|
| Prefetch opcode+locality | PASS — every ring prefetch on both sides is write-intent `prfm pstl1keep` (candidate symbols: 10/6/6 sites, 100%; reference mega-symbol: 116 × `pstl1keep` in rings; its 6 × `pldl1keep` belong to the sequential fallback's read prefetcher, outside every ring) |
| Fused claim | PASS — both claim the empty cell and write the mapped value in the same visit (`stp key` + mapped store before the next dispatch); no batched read-then-act anywhere |
| Per-visit load/store/branch counts | PASS — advance path identical for key64 (5/1/3 both), within ±2 for keys256 (18/5/4 vs 16/5/4, spill-restore scheduling), candidate ahead for key_string (8/1/4 vs 11/1/4) |
| No per-visit policy-field reloads (frame-copy SSA) | PASS — candidate key64/key_string hot paths contain zero stack loads (cells, key-column bases, skip bytes, next/rows register-resident); keys256 is by-reference in BOTH designs (`KeysFixed` not copyable) and both show the same per-visit getter-span/cells reloads; the only per-visit object load in the copyable anchors is the map grower's `buf_size` — a map field mutated by claims, reloaded identically in both |
| Inlined refill | PASS — skip-scan, zero-key synchronous emplace, seed hash (`crc32cx` chains / inline string hash) and seed prefetch are inline in the steady body on both sides; the only calls in any steady loop are `RowRefList::insert` (duplicate append — both sides), the reference's `bcmp` (hit path — candidate inlines instead), and the cold growth path (`resize`, candidate's `__sort`) |
| Bare `++`-style advance | PASS — `add/cmp/csinc` against `precalculated_buf_size` (increment + compare, select 0 only at the pad end), IDENTICAL sequence in all six loops |

Known justified divergences observed exactly where expected: (i) drain/re-seed (candidate:
slot-preserving, row-sorted `reseed`; reference: slot-order `start` re-seed — the candidate
fixed a real reference bug; growth path only, outside the hot visit), (ii) reference-only
leaf-locator/refWord machinery (locators/narrow-locators tests + decode on completion paths;
candidate uses `RowRef(block_no, row)` directly), (iii) symbol layout/register allocation
(map field offsets from allocator/cell layout, spill-slot pressure of the reference's single
98 KB mega-symbol vs the candidate's standalone per-anchor symbols, `ccmp` vs early-exit
compare selection — the reference differs between its own two `PosT` variants on this).

Two candidate-favoring differences documented (not flags): inline `memequalWide` vs reference
`bl bcmp` on the string hit path; fewer per-visit spill reloads in the string and key64 loops.

G-DISASM-BUILD: PASS (0 unexplained)
