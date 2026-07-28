# u4b disassembly parity: did the `AmacWalk` axis perturb the BARE ring walk?

Date: 2026-07-28. Read-only binary analysis with `llvm-nm-22` / `llvm-objdump-22` (aarch64).
Scratch artifacts: `/mnt/ch/ClickHouse/tmp/chj_probe_parity/disasm_scratch2/`.

## Verdicts {#verdicts}

| Check | Verdict |
|---|---|
| Bare steady-walk instruction sequence unchanged (visit / miss / advance) | **PASS** — 0 shape-diff lines in all 4 variant pairings |
| No new compare/branch anywhere in the bare walk | **PASS** — advance is still `add #0x10; str; prfm; ring-idx cmp #0x20`; no wrap compare exists in the bare body (`#0x410` occurs 0 times) |
| `wrap_aware` sibling exists in AFTER and contains the wrap compare | **PASS** — separate instantiation, 4 wrap-compare sites (3 steady copies + drain), `+0xd4` bytes over bare |

## Binaries and symbols {#binaries-and-symbols}

Instantiation under test: `amacFindPass<HashMethodOneNumber<PairNoInit<UInt64, RowRefList>, ...>,
ResumableHashMap<HashMapTable<UInt64, HashMapCell<UInt64, RowRefList, HashCRC32<UInt64>, ...>,
TailPaddedHashTableGrower<8>>>, need_flags=false, selector_is_range=true[, walk]>` (the MapsAll key64 shape).

| Binary | Symbol | Address | Size |
|---|---|---|---|
| BEFORE `tmp/chj_amac/bins/uncommitted-u4a.tmp.bin` | `amacFindPass<..., false, true>` (ring fully inlined) | `0x15b17e00` | `0xbb0` |
| AFTER `tmp/chj_amac/bins/uncommitted-u4b.tmp.bin` | `amacFindPass<..., false, true, (AmacWalk)0>` (wrapper) | `0x15b26e00` | `0x2d0` |
| AFTER | `amacRun<AmacFindPolicy<..., (AmacWalk)0>, 32>` (ring body) | `0x15b27100` | `0x9a0` |
| AFTER | `amacFindPass<..., false, true, (AmacWalk)1>` (wrapper) | `0x15b27ac0` | `0x2d0` |
| AFTER | `amacRun<AmacFindPolicy<..., (AmacWalk)1>, 32>` (ring body) | `0x15b27dc0` | `0xa74` |

BEFORE has 80 `amacFindPass` symbols, none mentioning `AmacWalk`; AFTER has exactly 2x (160), all carrying
`(DB::AmacWalk)0/1` — the axis doubled the instantiation set as designed.

Structural note (the one non-trivial codegen shift, outside the walk): in BEFORE the inliner folded
`amacRun` into `amacFindPass` (one 0xbb0 body); in AFTER it kept `amacRun` standalone
(wrapper `0x2d0` + body `0x9a0` ~= `0xc70`, roughly the BEFORE body plus call overhead). The wrapper's 4
`bl amacRun` sites run once per 8192-row chunk — not on the per-visit path.

## Method {#method}

Both regions disassembled fully by nm address range, mangled annotations stripped, addresses replaced by
relative labels, registers canonicalized by first appearance, then register numbers erased entirely
("shape" form) and diffed per matched control-flow block (`normalize.py` in the scratch dir). BEFORE
specializes the steady loop into 4 copies keyed by (`skip_data`, `slot_ids`) nullness; AFTER-bare has 3
(the two `skip_data != nullptr` copies merged). Pairings diffed: A(skip,slots)->1, C(skip,noslots)->1,
B(noskip,slots)->2, D(noskip,noslots)->3, plus 4 fill-loop pairs and the drain.

## The bare steady walk — unchanged {#bare-steady-walk-unchanged}

Every variant pairing's walk (visit + miss + advance blocks) diffs to zero in shape form. Real-register
side-by-side of the richest variant:

```
BEFORE u4a (variant A, inlined)          AFTER u4b (amacRun bare, variant 1)
--- visit ---                            --- visit ---
182fc: ldr  x14, [x26, x12, lsl #3]      2751c: ldr  x17, [x15, x11, lsl #3]  ; cell = ring.cell[s]
18300: ldr  x15, [x14]                   27520: ldr  x18, [x17]               ; cell->key
18304: cbz  x15, 18340                   27524: cbz  x18, 27580               ; isZero -> miss
18308: ldr  x16, [x24, x12, lsl #3]      27528: ldr  x0,  [x22, x11, lsl #3]  ; ring.key[s]
1830c: cmp  x15, x16                     2752c: cmp  x18, x0                  ; keyEquals
18310: b.ne 18440                        27530: b.ne 27680                    ; mismatch -> advance
18314: ldr  x14, [x14, #0x8]             27534: ldr  x17, [x17, #0x8]         ; mapped word
18318: ldrh w15, [x28, x12, lsl #1]      27538: ldrh w18, [x21, x11, lsl #1]  ; row
1831c: cmp  x9, x23                      2753c: cmp  x9, x19
18320: str  xzr, [x26, x12, lsl #3]      27540: str  xzr, [x15, x11, lsl #3]  ; deactivate
18324: str  x14, [x10, x15, lsl #3]      27544: str  x17, [x20, x18, lsl #3]  ; found_word[row]
18328: b.lo 18354                        27548: b.lo 27594                    ; refill
--- advance (the bare ++cell) ---        --- advance (the bare ++cell) ---
18440: add  x15, x14, #0x10              27680: add  x18, x17, #0x10          ; ++cell (16B cell)
18444: str  x15, [x26, x12, lsl #3]      27684: str  x18, [x15, x11, lsl #3]
18448: prfm pldl1keep, [x14, #0x10]      27688: prfm pldl1keep, [x17, #0x10]
1844c: add  x12, x12, #0x1               2768c: add  x11, x11, #0x1           ; ++s
18450: cmp  x12, #0x20                   27690: cmp  x11, #0x20               ; ring wrap (pre-existing)
18454: b.ne 182fc                        27694: b.ne 2751c
18458: b    182e0                        27698: b    27500
```

Identical opcode-for-opcode; only register allocation and addresses shift. No new compare or branch:
the sole `cmp` in the advance is the ring-index `#0x20`, present in both. No `crc32cx` in the step in
either build (the key64 cell keeps no saved hash and `keyEquals` ignores it); `crc32cx` appears only in
the admit (`start`) blocks, in both builds. The miss block (`ldrh` row; `cmp`; `str xzr` x2; `b.hs`) is
also shape-identical. The drain loop diffs only in its external exit target.

Non-walk deltas in the AFTER-bare body (all on once-per-admitted-row or once-per-ring-sweep paths,
never per visit):
- The two `skip_data != nullptr` steady copies merged into one: the refill entry gained a single
  loop-invariant `cbz x26` (slot-ids null test) dispatching between two refill bodies, and the
  no-slot admit reuses the slot admit with `slot = 0` (`mov x17, xzr`).
- Admit-block scheduling permutations (same instruction multiset), and one `ldr`+`ldr` -> `ldp`
  fusion of the descriptor load in the no-slot fill admit (37 -> 36 instructions).
- The merged copy's hit-exhaust bookkeeping tail is duplicated inline (`eor/and/sub/add/cmp/b`)
  instead of `mov` + shared-tail jump.

## The `wrap_aware` sibling — the axis is real {#wrap-aware-sibling}

`amacRun<AmacFindPolicy<..., (AmacWalk)1>, 32>` at `0x15b27dc0` (`0xa74` = bare + `0xd4`). Its advance
recovers the bounds from the descriptor and wraps exactly at `pad_end = buf + mask + 1 + tail_pad`
(`tail_pad = 64`, 16-byte cells -> `(1 + 64) * 16 = 0x410`):

```
28340: ldrh w18, [x23, x11, lsl #1]      ; slot = ring.slot[s]
28344: add  x17, x17, #0x10              ; ++cell
28348: add  x0,  x20, x18, lsl #4        ; &slot_descs[slot]
2834c: ldp  x18, x0, [x0]                ; buf, mask
28350: add  x0,  x18, x0, lsl #4         ; buf + mask*16
28354: add  x0,  x0, #0x410              ; pad_end
28358: cmp  x17, x0                      ; the wrap compare
2835c: b.eq 283d4                        ; wrap -> cell = buf
28360: str  x17, [x15, x11, lsl #3]
28364: prfm pldl1keep, [x17]
...
283d4: str  x18, [x15, x11, lsl #3]      ; wrapped: cell = buf
283d8: prfm pldl1keep, [x18]
```

The wrap compare appears at 4 sites (`0x28358`, `0x28558`, `0x286f8`, `0x287bc`) — one per advance copy
(3 steady + drain) — and at 0 sites in the bare body and in the whole BEFORE function.

## Artifacts {#artifacts}

`disasm_scratch2/`: `u4{a,b}_syms.txt` (demangled nm), `u4a_findpass_full.clean.asm`,
`u4b_findpass_wrapper.asm`, `u4b_amacrun_{bare,wrap}.clean.asm`, `normalize.py`,
`w_before_{A..D}.txt` / `w_after_{1..3}.txt` (normalized walk blocks; pairwise diffs are empty),
`s_{a,b}_*.txt` + `d_*.diff` (phase-level shape diffs).
