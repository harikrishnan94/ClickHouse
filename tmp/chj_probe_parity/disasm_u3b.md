# G-disasm: flat descriptor find of the routed `parallel_hash` probe (u3b vs ahj)

Date: 2026-07-28. Method: `llvm-nm-22` + `llvm-objdump-22` with address ranges (no tooling cache).
Scratch (full disassembly of every region below): `tmp/chj_probe_parity/disasm_scratch/`.

- CANDIDATE: `tmp/chj_amac/bins/uncommitted-u3b.tmp.bin` (aarch64, BuildID `778d3031d3f9...`)
- REFERENCE: `tmp/chj_amac/bins/clickhouse-ahj-cf465cfbe23.bin` (aarch64, BuildID `b3017809078f...`)

Anchors are Inner/All (`(DB::JoinKind)0, (DB::JoinStrictness)3` — source-verified in
`src/Core/Joins.h`: `Inner = 0`, `All = 3`; the task brief said `(JoinKind)1`, which is `Left` —
the Inner/All instantiations were used as specified by the criterion text). `MapsAll` demangles
as `HashJoin::MapsTemplate<RowRefList>`. Both sides compared on the continuous-range selector,
`need_filter` (Inner), no skip, non-precomputed — the steady-state plain arm:

| anchor | candidate symbol (lambda) | addr/size | reference symbol (lambda) | addr/size |
|---|---|---|---|---|
| key64 | `RoutedHashJoinMethods<0,3,MapsAll>::joinRightColumns<HashMethodOneNumber<u64>, ResumableHashMap<HashMap<u64,RowRefList,HashCRC32<u64>,TailPaddedHashTableGrower<8>>>, AddedColumns<true>, pair<u64,u64>>::loop::operator()<true,false,false,true>` | `0x15a4f780` / `0x4bc` | `PartitionedHashJoin::routedJoinRightColumns<0,3,MapsAll,HashMethodOneNumber<u64>, ResumableHashMap<...HashCRC32<u64>...>>::flat_loop::operator()<true,false,true>` | `0x16939d00` / `0x46c` |
| keys256 | same, `HashMethodKeysFixed<UInt256>` / `UInt256HashCRC32` map | `0x15a77780` / `0xa90` | same, `HashMethodKeysFixed<UInt256>` | `0x16951b40` / `0x858` |

Cell layouts (identical on both sides): key64 cell = 16 B (`u64` key @0, `RowRefList` word @8);
keys256 cell = 40 B (32 B key @0, mapped @0x20). Neither cell stores a saved hash, so a
saved-hash prefilter must be (and is) absent on both sides. `tail_pad = 64` cells on both.

## Verdicts {#verdicts}

- **key64: PASS** — 0 unexplained deltas.
- **keys256: PASS** — 0 unexplained deltas.

## Anchor 1: key64 {#anchor-key64}

### Candidate walk (u3b, `0x15a4f9d8..0x15a4fa4c`) {#cand-key64}

```
; row: slot id + key + zero-key gate
15a4f9d8: ldrb  w9, [x8, x4]          ; slot = slot_ids[ind]   (UInt8)
15a4f9e8: ldr   x8, [x8, x4, lsl #3]  ; key = key_data[ind]
15a4f9ec: cbz   x8, 0x15a4fb60        ; isZeroKey -> map-resolved find (cold)
; hash + descriptor (NO map header touched)
15a4f9f0: ldr   x10, [x19, #0x68]     ; &flat_descs   (closure reload, see D1)
15a4f9f4: crc32cx w11, w24, x8        ; hash = crc32c(-1, key)   [ONE op]
15a4f9f8: ldr   x10, [x10]            ; flat_descs
15a4f9fc: add   x9, x10, x9, lsl #4   ; &desc[slot]  (16-B SlotMapDesc)
15a4fa00: ldp   x9, x10, [x9]         ; {buf, mask}  single paired load
15a4fa04: and   x11, x10, x11         ; pos = hash & mask
15a4fa08: add   x27, x9, x11, lsl #4  ; cell = buf + pos*16
15a4fa0c: ldr   x11, [x27]            ; cell->key
15a4fa10: cbz   x11, MISS
15a4fa14: add   x10, x9, x10, lsl #4
15a4fa18: add   x10, x10, #0x410      ; pad_end = buf + (mask+1+64)*16  [wrap bound]
; steady-state visit
15a4fa20: cmp   x11, x8               ; keyEquals (full key, no hash prefilter)
15a4fa24: b.eq  FOUND(0x15a4fa60)
15a4fa28: add   x27, x27, #0x10       ; ++cell (sizeof(cell)=16)
15a4fa2c: cmp   x27, x10              ; wrap compare vs pad_end
15a4fa30: b.eq  WRAP(0x15a4fa40)
15a4fa34: ldr   x11, [x27]
15a4fa38: cbnz  x11, 0x15a4fa20       ; isZero check
15a4fa3c: b     MISS
WRAP:
15a4fa40: mov   x27, x9               ; cell = buf
15a4fa44: ldr   x11, [x9]
15a4fa48: cbnz  x11, 0x15a4fa20
15a4fa4c: b     MISS
FOUND(0x15a4fa60): strb filter[i]; push matched_rows; ldr x1,[x27,#0x8]  ; mapped
                   bl AddedColumns<true>::appendFromBlock ; then RowRefList row-count
                   decode (tbnz #63 / lsr #48 cmp 0x7fff / ldur w,[+1]) for offsets
```

### Reference walk (ahj, `0x16939fac..0x16939ff8`) {#ref-key64}

```
; invariants hoisted to callee-saved regs before the loop:
;   x21=range base, x22=leaf_ids, x23=descs, x24=key data, w26=-1
16939fb4: ldrh  w9, [x22, x8, lsl #1] ; leaf = leaf_ids[ind]   (UInt16)
16939fb8: ldr   x8, [x24, x8, lsl #3] ; key
16939fbc: cbz   x8, 0x1693a00c        ; isZero(key) -> map-resolved find (cold)
16939fc0: add   x10, x23, x9, lsl #4  ; &desc[leaf]
16939fc4: crc32cx w9, w26, x8         ; hash = crc32c(-1, key)   [ONE op]
16939fc8: ldp   x10, x11, [x10]       ; {buf, mask}  single paired load
16939fcc: and   x9, x11, x9           ; pos = hash & mask
16939fd0: lsl   x12, x9, #4
16939fd4: ldr   x12, [x10, x12]       ; buf[pos].key
16939fd8: cbz   x12, MISS
16939fdc: add   x11, x11, #0x40       ; wrap bound = mask + 64 (last index)
; steady-state visit
16939fe0: cmp   x12, x8               ; keyEquals (full key, no hash prefilter)
16939fe4: b.eq  FOUND(0x1693a040)
16939fe8: cmp   x9, x11               ; wrap compare vs mask+64
16939fec: csinc x9, xzr, x9, eq       ; pos = wrap ? 0 : pos+1   [branchless]
16939ff0: lsl   x12, x9, #4
16939ff4: ldr   x12, [x10, x12]
16939ff8: cbnz  x12, 0x16939fe0       ; isZero check
16939ffc: b     MISS
FOUND(0x1693a040): add x26, x10, x9, lsl #4 ; cell
                   strb filter[i]; push matched_rows; ldr x1,[x26,#0x8]
                   bl AddedColumns<true>::appendFromBlock ; identical count decode
```

### key64 comparison {#key64-comparison}

| criterion | candidate | reference | verdict |
|---|---|---|---|
| (1) address gen | `ldp {buf,mask}` from `desc[slot]`; no map-header load anywhere on the find path | identical (`ldp` from `desc[leaf]`) | match |
| (2) hash | 1x `crc32cx`, seed `-1` | 1x `crc32cx`, seed `-1` | match |
| (3) walk | zero-check `cbz`/`cbnz`, full-key `cmp`, `+16` advance, wrap compare present | same elements | match |
| wrap point | `cell == buf+(mask+1+64)*16` -> `buf` (`+0x410` = 16*65) | `pos == mask+64` -> `0` | identical cell sequence |
| per-visit | 7 insns, 1 load, 0 stores, 3 branches | 7 insns, 1 load, 0 stores, 2 branches | within ±1, explained (D2/D3) |
| flag_base | absent (Inner/All: no offset math at all — compiled out) | absent | match |
| prefetch arm | `crc32cx` + `ldp` desc + `prfm pldl1keep` at home cell | identical incl. `pldl1keep` | match |
| emit | inline `processMatch`: filter store, `matched_rows` push (capacity-checked), `bl appendFromBlock`, packed-count decode | instruction-identical shape | match |

## Anchor 2: keys256 {#anchor-keys256}

Both sides pack the key identically first: prepared-keys fast arm (`ldr q0` x2 from
`keys + ind*32`, staged via stack, reloaded as `x11,x8,x9,x10`) with the same per-column
`rbit`/`clz` size-switch + `memcpy` fallback; both route the all-zero key to the map-resolved
find before the walk (`cbnz`/`cbz` chain over the four words on both sides).

### Candidate walk (u3b, `0x15a77c70..0x15a77d3c`) {#cand-keys256}

```
15a77c70: ldr   x12, [x21, #0x68]     ; &flat_descs (closure reload, D1)
15a77c74: mov   w13, #-0x1
15a77c78: mov   w16, #0x28            ; sizeof(cell) = 40
15a77c7c: crc32cx w13, w13, x11       ; 4x chained crc32c over the 32-B key
15a77c80: ldr   x12, [x12]
15a77c84: crc32cx w13, w13, x8
15a77c88: add   x12, x12, x2, lsl #4  ; &desc[slot]
15a77c8c: crc32cx w13, w13, x9
15a77c90: ldp   x12, x14, [x12]       ; {buf, mask}
15a77c94: crc32cx w13, w13, x10
15a77c98: madd  x15, x14, x16, x12
15a77c9c: and   x13, x14, x13         ; pos = hash & mask
15a77ca0: umaddl x19, w13, w16, x12   ; cell = buf + pos*40
15a77ca4: add   x13, x15, #0xa28      ; pad_end = buf + (mask+1+64)*40  (0xa28 = 40*65)
15a77ca8: ldr   x14, [x19]            ; word0
15a77cac: cbnz  x14, CMP(0x15a77d00)
15a77cb0: b     ZCHK(0x15a77cd4)      ; word0==0: check words 1..3 for zero-cell/miss
ADV(0x15a77cc0):
15a77cc0: add   x19, x19, #0x28       ; ++cell
15a77cc4: cmp   x19, x13              ; wrap compare vs pad_end
15a77cc8: b.eq  WRAP(0x15a77d30)      ; -> cell = buf
15a77ccc: ldr   x14, [x19]
15a77cd0: cbnz  x14, CMP
ZCHK:     ldr/cbnz [x19,#0x8],[x19,#0x10],[x19,#0x18] ; all zero -> MISS
CMP(0x15a77d00):
15a77d00: cmp   x14, x11 ; b.ne ADV   ; keyEquals: word0, then sequential
15a77d08: ldr   x14, [x19, #0x8]  ; cmp x8  ; b.ne ADV      early-out per word
15a77d14: ldr   x14, [x19, #0x10] ; cmp x9  ; b.ne ADV
15a77d20: ldr   x14, [x19, #0x18] ; cmp x10 ; b.ne ADV
15a77d2c: b     FOUND(0x15a77d40)     ; mapped at [x19,#0x20]; same emit as key64
```

### Reference walk (ahj, `0x16952018..0x169520b4`) {#ref-keys256}

```
16952018: mov   w12, #-0x1
1695201c: ldp   x22, x13, [sp, #0x30] ; leaf_ids + descs base (stack spill reload)
16952020: crc32cx w12, w12, x11       ; 4x chained crc32c over the 32-B key
16952024: crc32cx w12, w12, x8
16952028: add   x13, x13, x14, lsl #4 ; &desc[leaf]
1695202c: crc32cx w12, w12, x9
16952030: crc32cx w14, w12, x10
16952034: ldp   x12, x15, [x13]       ; {buf, mask}
16952038: and   x14, x15, x14         ; pos = hash & mask
1695203c: add   x13, x15, #0x40       ; wrap bound = mask + 64 (last index)
16952040: madd  x19, x14, x26, x12    ; cell = buf + pos*40   (x26 = 40)
16952044: ldr   x15, [x19]            ; word0
16952048: cbnz  x15, CMP(0x16952090)
1695204c: b     ZCHK(0x16952078)
ADV(0x16952060):
16952060: cmp   x14, x13              ; wrap compare vs mask+64
16952064: add   x14, x14, #0x1        ; ++pos
16952068: b.eq  WRAP(0x169520b4)      ; -> pos = 0
1695206c: madd  x19, x14, x26, x12    ; recompute cell address
16952070: ldr   x15, [x19]
16952074: cbnz  x15, CMP
ZCHK:     ldp x16,x17,[x19,#0x8]; ldr x18,[x19,#0x18]; cmp/ccmp/ccmp -> MISS if zero
CMP(0x16952090):
16952090: cmp   x15, x11 ; b.ne ADV   ; keyEquals: word0, then fused tail
16952098: ldp   x16, x17, [x19, #0x8] ; ldr x18, [x19, #0x18]
169520a0: cmp x16,x8 ; ccmp x17,x9 ; ccmp x18,x10 ; b.ne ADV
169520b0: b     FOUND                 ; mapped at [x19,#0x20]; same emit as key64
```

### keys256 comparison {#keys256-comparison}

| criterion | candidate | reference | verdict |
|---|---|---|---|
| (1) address gen | `ldp {buf,mask}` from `desc[slot]`; no map header | identical | match |
| (2) hash | 4x chained `crc32cx`, seed `-1`, interleaved with desc loads | identical | match |
| (3) walk | word0 zero-check, full 4-word keyEquals (no hash prefilter), `+40` advance, wrap compare present | same elements | match |
| wrap point | `cell == buf+(mask+1+64)*40` (`+0xa28` = 40*65) -> `buf` | `pos == mask+64` -> `0` | identical cell sequence |
| per-visit (word0 mismatch) | 7 insns, 1 load, 0 stores, 3 branches | 8 insns, 1 load, 0 stores, 3 branches | within ±1, explained (D3) |
| flag_base | absent | absent | match |
| prefetch arm | 4x `crc32cx` + `ldp` + `umull` + `prfm pldl1keep` | identical | match |
| emit | mapped at `[cell,#0x20]`, inline filter/matched/`appendFromBlock`/count decode | instruction-identical shape | match |

## Explained deltas (all structural, none semantic) {#explained-deltas}

- **D1 — per-row closure reloads (candidate)**: the candidate's `loop` lambda reloads its
  snapshots (`selector` base, `slot_ids`, key-column/prepared-keys pointer, `flat_descs`,
  `added_columns`) through the closure each row (~5–9 extra L1 loads/row in key64), because the
  snapshots are captured by reference and `appendFromBlock` is an opaque call. ahj's key64
  `flat_loop` hoists them into callee-saved registers (`x21..x26`); in keys256 ahj is itself
  register-starved and reloads from stack spills (`[sp,#0x30]`, `[sp,#0x40]`) per row, so the
  gap there is ~2–3 loads. Per-visit walk counts are unaffected. This is the one measurable
  codegen (not semantics) gap, and it is per-row, not per-visit.
- **D2 — key64 wrap idiom**: candidate `cmp`+`b.eq` to a wrap block (statically never-taken in
  steady state) vs ahj `csinc` (branchless). Same wrap point; ±1 branch.
- **D3 — advance idiom**: candidate walks by pointer (`add cell, #sizeof`), ahj by index
  (`add pos, #1` + `madd` address recompute in keys256, `lsl` fold in key64): ±1 ALU op.
- **D4 — keys256 keyEquals tail**: candidate sequential `ldr`+`cmp`+`b.ne` early-out per word;
  ahj `ldp`/`ldr` + `cmp`+`ccmp`+`ccmp` fused. Same 3 tail loads; ahj 2 fewer branches on a
  full match, candidate exits earlier on a word1 mismatch. First-word mismatch (the common
  visit) is identical: 1 `cmp` + 1 `b.ne`, no tail loads.
- **D5 — slot id width**: candidate `slot_ids` is `UInt8` (`ldrb`), ahj `leaf_ids` is `UInt16`
  (`ldrh`). One load either way.
- **D6 — zero-key cold path**: candidate outlines it into a `processMatch` call
  (`0x15a4bb40`); ahj keeps it inline. Cold on both sides (guarded by the pre-walk zero-key
  branch).
- **D7 — ahj-only dead re-check**: ahj's two-exit `while` + post-loop `!isZero` re-test emits a
  redundant zero-or-found disambiguation block (`0x169520e0`) reachable only on the
  word0==0 probe-key path; the candidate's single-loop break structure has no equivalent.
  Never executed on the hot path.

## Bottom line {#bottom-line}

Both anchors: the candidate's flat descriptor find is instruction-semantics equivalent to
ahj's `flat_loop` — descriptor-pair (`ldp {buf,mask}`) addressing with zero map-header
chasing, identical crc32c hashing (1 op key64 / 4 ops keys256), identical wrap-aware walk over
the same 40/16-byte cells with the same `mask+1+64` wrap point, no saved-hash prefilter on
either side (correct: neither cell stores one), no `flag_base` on Inner/All, and an
instruction-identical inline emit. The only systematic codegen difference is D1 (per-row
closure reloads), which does not touch the per-visit walk.
