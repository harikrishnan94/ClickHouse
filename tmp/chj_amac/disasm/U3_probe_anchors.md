# G-disasm-probe — Unit 3 AMAC probe FIND ring, 3 anchors vs `ahj` reference

Date: 2026-07-27. PREREG-007 (d) in `tmp/chj_amac/PREREG.md`.

- CANDIDATE: `tmp/chj_amac/bins/uncommitted-u3.tmp.bin` (sha256 `2f941e41ad4c2fa0fc170e56...`),
  symbols `DB::amacFindPass<...>` from `src/Interpreters/HashJoin/AmacProbe.cpp`
  (`AmacFindPolicy` + `amacRun` of `AmacRing.h`, fully inlined into the pass).
- REFERENCE: `tmp/chj_amac/bins/clickhouse-ahj-cf465cfbe23.bin` (sha256 `c8260c682b78...`),
  symbols `DB::amacRun<DB::RoutedAmacFindPolicy<...>, 32ul>` from
  `ahj:src/Interpreters/PartitionedHashJoin/PartitionedHashJoinProbeImpl.h`.

Method: `/usr/local/bin/llvm-nm-22 --defined-only --print-size` + `c++filt` for symbol
selection, `/usr/local/bin/llvm-objdump-22 -d --start-address/--stop-address` for the six
ranges (no `analyze-assembly.py` cache). Anchor arm: RowRefList mapped, `need_flags=false`,
`selector_is_range=true` — the canonical steady loop; the flag arm differs only by the
documented once-per-matched-row `recordHit` descriptor load. In BOTH binaries the linker's
ICF folded the byte-identical `RowRef` and `RowRefList` flagless instantiations onto one
address (2 nm symbols per address), so each analyzed range IS the RowRefList anchor.
Full disassembly kept next to this report: `u3_{key64,keys256,key_string}.asm`,
`ahj_{key64,keys256,key_string}.asm`.

Source cross-check: the candidate collision advance is a bare `++cell`
(`AmacProbe.cpp` line 219; this branch's maps are tail-padded, `static_assert
is_tail_padded_linear_grower`), exactly like `ahj` — the ring slot carries the resolved
`const Cell *`; the `SlotMapDesc {buf,mask}` is read only at admit (`start`) and, in the
flagged arm only, in `recordHit`. No desc load exists on the collision path in either build.

## Global census (all six functions) {#global-census}

| function | `prfm pldl1keep` | `prfum pldl1keep` | `pldl3`/`pldl2`/store-intent | out-of-line calls |
|---|---|---|---|---|
| u3 key64 | 13 | 0 | 0 | `memset` (ring init) |
| u3 keys256 | 4 | 4 | 0 | `memset`, 2x `memcpy` (admit pack fallback), `__libcpp_verbose_abort` (guard) |
| u3 key_string | 4 | 4 | 0 | `memset`, `__libcpp_verbose_abort` (guard) |
| ahj key64 | 11 | 0 | 0 | `memset` |
| ahj keys256 | 4 | 4 | 0 | `memset`, 2x `memcpy`, `__libcpp_verbose_abort` |
| ahj key_string | 4 | 4 | 0 | `memset`, 2x `bcmp` (!), `__libcpp_verbose_abort` |

Every prefetch in every function is read-intent, locality-3 = `pldl1keep`. The historical
`pldl3keep` defect is absent from both builds. `prfum` is the unscaled-immediate encoding of
the same hint (the second-line offsets `+0x27/+0x3f/+0x4f` are not `prfm`-scalable); locality
and intent identical. The 13-vs-11 key64 site count is loop-specialization cloning (below),
not a semantic difference. Candidate-only extra: one tail-call to
`ProfileEvents::increment` after the pass (outside all ring loops), per source.

## Anchor 1 — key64 / RowRefList (cell `HashMapCell<UInt64, RowRefList, HashCRC32<UInt64>>`, 16 B = single line) {#anchor-key64}

| side | symbol (abridged) | addr | size |
|---|---|---|---|
| candidate | `amacFindPass<HashMethodOneNumber<PairNoInit<UInt64, RowRefList>, const RowRefList, UInt64, false, true, false>, ResumableHashMap<HashMapTable<UInt64, HashMapCell<UInt64, RowRefList, HashCRC32<UInt64>>, ..., TailPaddedHashTableGrower<8>>>, false, true>` | `0x15b2a500` | `0xbb0` |
| ahj | `amacRun<RoutedAmacFindPolicy<HashMethodOneNumber<...same...>, const ResumableHashMap<...same...>, false, true>, 32ul>` | `0x16938340` | `0x9a0` |

Candidate steady visit (no-skip/single-slot clone, `0x15b2ae98`; abridged):

```
ldr  x14, [x27, x12, lsl #3]      ; cell = ring.cell[s]
ldr  x15, [x14]                   ; cell->key
cbz  x15, miss
ldr  x16, [x20, x12, lsl #3]      ; ring.key[s]  (stored at admit)
cmp  x15, x16
b.ne collide
ldr  x14, [x14, #0x8]             ; mapped word (RowRefList.word)
ldrh w15, [x23, x12, lsl #1]      ; ring.row[s]
str  xzr, [x27, x12, lsl #3]      ; deactivate
str  x14, [x10, x15, lsl #3]      ; found_word[row]
b.lo refill                       ; inlined admit: key load, crc32cx, desc ldp/ldr,
                                  ;   4 ring stores, prfm pldl1keep [buf, pos*16]
collide:                          ; 0x15b2afa0
add  x15, x14, #0x10              ; ++cell (bare, 16 B)
str  x15, [x27, x12, lsl #3]
prfm pldl1keep, [x14, #0x10]
```

ahj steady visit (`0x16938ad8`, same clone): instruction-for-instruction the same shape —
`ldr/ldr/cbz/ldr/cmp/b.ne`, hit `ldr [cell,#0x8]` + `ldrh` row + 2 stores + inlined refill
(`crc32cx` at admit only, desc `ldp`/`ldr [x24{,#0x8}]`, `prfm pldl1keep`), collision
`add #0x10` + `str` + `prfm pldl1keep [cell,#0x10]`.

Per-visit table (steady loop):

| event | loads | stores | branches | prefetch |
|---|---|---|---|---|
| collision (both) | 3 (ring.cell, cell key, ring.key) | 1 (ring.cell) | 3 | 1x `pldl1keep` next cell (single line, cell 16 B <= 24 B) |
| hit incl. inlined refill (both) | 5 + 3 admit (key col, desc buf+mask) | 2 + 4 admit (ring key/row/slot/cell) | ~6 | 1x `pldl1keep` home cell |
| hash in visit | none — `crc32cx` only at admit in BOTH (keyEquals of a non-saved-hash cell ignores the hash; dead hash eliminated identically) | | | |

No `[sp]` loads on the steady visit path in either build (all bases register-carried); the
only stack reload is the spilled `slot_maps`/`leaf_maps` pointer on the unlikely zero-key
sync path inside `start`. Inlined refill: yes, both. Drain: slot-indexed loop with
`isActive` check and same collision step, both.

Differences: (a) zero-key sync path reads the map object at `[map]`/`[map,#0x10]`
(candidate) vs `[map,#0x8]`/`[map,#0x18]` (ahj) — map-object layout differs across the
branches; outside the ring; **justified (i)**. (b) `slot_ids` are `UInt64` (`ldr`, `lsl #3`)
vs ahj `leaf_ids` `UInt16` (`ldrh`, `lsl #1`) — slot-scheme vs leaf-descriptor machinery;
**justified (i)**. (c) candidate symbol contains the 8192-row chunk loop, policy
construction from arguments, and the event tail-call; ahj loads its policy fields from `x0`
and is chunked by its caller — inlining boundary; **justified (ii)/(iii)**. (d) candidate
clones the steady loop x4 on (`skip_data`, `slot_ids`) nullness, ahj x3 + one runtime check;
per-path semantics identical; **justified (iii)**. Verdict: **equivalent**.

## Anchor 2 — keys256 / RowRefList (cell `HashMapCell<UInt256, RowRefList, UInt256HashCRC32>`, 40 B = two lines) {#anchor-keys256}

| side | symbol (abridged) | addr | size |
|---|---|---|---|
| candidate | `amacFindPass<HashMethodKeysFixed<PairNoInit<wide::integer<256, unsigned>, RowRefList>, ...>, ResumableHashMap<HashMapTable<wide::integer<256, unsigned>, HashMapCell<...>, ...>>, false, true>` | `0x15b40b40` | `0xb20` |
| ahj | `amacRun<RoutedAmacFindPolicy<HashMethodKeysFixed<...same...>, ..., false, true>, 32ul>` | `0x16950380` | `0x890` |

Candidate steady visit (`0x15b410a0`):

```
ldr x8, [x27, x22, lsl #3]        ; cell
add x16, x24, x22, lsl #5         ; &ring.key[s]  (32-B stride)
ldr x9, [x8]        ; limb0        cbz -> empty-check of remaining limbs
ldr x10, [x16]      ; stored limb0 cmp/b.ne collide
...limbs 1..3: ldr/ldr/cmp/b.ne x3
ldr x8, [x8, #0x20]               ; mapped word
ldrh/str xzr/str found_word; b refill
collide:                           ; 0x15b41320
add  x9, x8, #0x28                ; ++cell (bare, 40 B)
str  x9, [x27, x22, lsl #3]
prfm  pldl1keep, [x8, #0x28]      ; next cell line 1
prfum pldl1keep, [x8, #0x4f]      ; next cell last byte -> line 2
```

Admit (refill, `0x15b41378`/`0x15b41478`): prepared-keys fast path = ONE `ldp q0, q1`
32-byte load of the pre-packed key (generic per-column gather with `memcpy` fallback only
when the getter has no packed area — identical block exists in ahj); `crc32cx` chain x4;
desc `ldp` -> `umaddl` x40; `stp q0, q1` into `ring.key[s]`; `prfm pldl1keep [cell]` +
`prfum pldl1keep [cell, #0x27]` (home cell, both lines). ahj admit (`0x16950440`,
`0x16950500`): the same sequence, including the identical `str q`/4x `ldr` stack bounce for
the SIMD->GPR zero-check, `prfm [x9]` + `prfum [x9, #0x27]`.

ahj steady visit (`0x1695075c`): same limb ladder, same hit (`ldr [cell,#0x20]`), same
collision `+0x28` with `prfm [x8,#0x28]` + `prfum [x8,#0x4f]`.

Per-visit table (steady loop):

| event | loads | stores | branches | prefetch |
|---|---|---|---|---|
| collision (both) | 2..9 (ring.cell + up to 4 cell-limb + up to 4 stored-limb, early-out) | 1 | up to 6 | 2x `pldl1keep` (lines 1+2 of next cell) |
| hit (both) | ladder + mapped + row | 2 | ~7 | — |
| admit (both) | 1x32 B packed key (fast path), desc pair | ring key (`stp q`) + row + slot/leaf + cell | — | 2x `pldl1keep` (lines 1+2 of home cell) |
| key re-pack per visit | **none in either** — the visit compares against `ring.key`; the pack (or 32-B load) happens once at admit. The historical K2 per-visit re-pack defect is absent. | | | |

Differences: (a) spill placement — candidate keeps `slot_descs` in a register but carries
the chunk bound and `active` count in stack slots reloaded on completion paths (never on
the common visit path); ahj keeps `active` in a register but reloads `leaf_descs` and four
ring/result bases from the stack around every refill. Both spill once per completed ROW,
neither per visit; **justified (iii)**. (b) same zero-key map-layout and id-width deltas as
anchor 1; **justified (i)**. Verdict: **equivalent**.

## Anchor 3 — key_string / RowRefList (cell `HashMapCellWithSavedHash<std::string_view, RowRefList, DefaultHash>`, 32 B = two lines, ring carries the hash) {#anchor-key-string}

| side | symbol (abridged) | addr | size |
|---|---|---|---|
| candidate | `amacFindPass<HashMethodString<PairNoInit<std::string_view, RowRefList>, const RowRefList, true, false, true, false>, ResumableHashMap<HashMapTable<std::string_view, HashMapCellWithSavedHash<...>, ...>>, false, true>` | `0x15b2d2c0` | `0xb68` |
| ahj | `amacRun<RoutedAmacFindPolicy<HashMethodString<...same...>, ..., false, true>, 32ul>` | `0x1693cf80` | `0x88c` |

Candidate steady visit (`0x15b2d6fc`):

```
ldr x12, [x25, x10, lsl #3]       ; cell
ldr x13, [x12, #0x8]              ; cell key.size
cbz x13, miss                     ; zero sentinel
ldr x14, [x24, x10, lsl #3]       ; ring.hash[s]   (saved at admit)
ldr x15, [x12, #0x18]             ; cell saved hash
cmp x15, x14 ; b.ne collide       ; saved-hash prefilter, no recompute
add x14, x20, x10, lsl #4         ; &ring.key[s] (string_view, 16 B)
ldr x15, [x14, #0x8] ; cmp x13    ; length compare
ldr x15, [x12] ; ldr x14, [x14]   ; data pointers
<inlined size-switched equality: cmeq/shrn SIMD >=16 B, scalar ladders below>
hit: ldr x12, [x12, #0x10]        ; mapped word -> found_word, deactivate, refill
collide:                           ; 0x15b2db00
add x13, x12, #0x20               ; ++cell (bare, 32 B)
str; prfm pldl1keep, [x12, #0x20] ; prfum pldl1keep, [x12, #0x3f]
```

Admit: string view from the offsets pair (`ldp`, `subs` size + `b.mi` sanity-abort guard —
ahj has the same guard), inline `StringRefHash` (mul mixes < 8 B, `crc32cx` word loop >= 8 B
— identical block in ahj at `0x1693d2c0`), then `stp {data,size}` into `ring.key`, `str`
hash into `ring.hash`, desc `ldp`, cell = `buf + (h & mask) * 32`, `prfm [cell]` +
`prfum [cell, #0x1f]`.

ahj steady visit (`0x1693d320`): identical through the saved-hash prefilter and length
compare, then diverges in FORM: it saves six registers, calls out-of-line `bcmp`, restores
nine (two `ldp` + `ldr` from stack), and reloads its `ring.key` base from `[sp, #0x70]` on
every hash-passing visit because the call clobbers it. Collision and prefetch identical
(`+0x20`, `prfm [x25,#0x20]` + `prfum [x25,#0x3f]`).

Per-visit table (steady loop):

| event | candidate | ahj |
|---|---|---|
| collision | 5 loads / 1 store / 3 branches / 2x `pldl1keep` | same |
| hash-passing visit (compare) | + length + inlined byte-equality, zero `[sp]` traffic, no call | + length + `bl bcmp` + 6 reg saves + 9 restores + 1 per-visit `[sp]` reload |
| hit | + mapped `[cell,#0x10]`, row, 2 stores, inlined refill | same semantics after `bcmp` returns |
| hash recompute per visit | none — ring-carried in BOTH (`RingWithHash`) | none |

Difference: the byte-equality is inlined in the candidate vs an out-of-line `bcmp` (plus its
register save/restore and a per-visit stack reload) in the reference. Same semantics —
byte equality over `{data,size}` after the saved-hash and length prefilters — with strictly
fewer per-visit instructions on the candidate; a codegen (inlining/regalloc) difference in
the candidate's favor, **justified (iii)**. All other deltas as anchors 1-2 (**justified
(i)/(iii)**). Verdict: **equivalent** (candidate-favorable).

## Criterion checklist {#criterion-checklist}

- Read-intent `prfm pldl1keep`, never `pldl3keep`: PASS — 56/56 prefetch instructions across
  all six functions are `pldl1keep` (`prfm` or unscaled `prfum` encoding); zero `pldl3`,
  `pldl2`, or store-intent hints.
- Second-line prefetch for cells > 24 B: PASS — keys256 (40 B): `[cell]+[cell,#0x27]` at
  admit, `[+0x28]+[+0x4f]` on advance; key_string (32 B): `[cell]+[cell,#0x1f]` /
  `[+0x20]+[+0x3f]`; key64 (16 B): single line, correctly no second prefetch. Identical in
  both binaries at every site (admit, steady collision, drain collision).
- Resolved `const Cell *` in the slot, no per-visit buf/mask re-resolution: PASS — the
  collision advance is a bare `add cell, cell, #{0x10,0x28,0x20}` in all six functions
  (tail-padded maps on this branch, same as `ahj`); `SlotMapDesc`/`LeafMapDesc` is read at
  admit only (the flagged arm's `recordHit` desc load is the documented exception and is
  outside this flagless anchor arm).
- Stored keys packed once at admit (K2 defect): PASS — keys256 visits compare against
  `ring.key` (interleaved limb loads); the pack (1x `ldp q0,q1` fast path or generic
  gather + `memcpy` fallback) exists only on the admit path in both builds.
- No per-visit policy-field reloads (frame-copy SSA): PASS — candidate steady visit paths
  are `[sp]`-free on all three anchors (spilled driver state is touched only on
  once-per-row completion paths, mirroring equivalent-or-heavier spills in the reference;
  ahj key_string even reloads its ring-key base per visit around `bcmp`).
- Inlined refill: PASS — the admit sequence is inline in the steady loop of all six
  functions; the only in-loop calls anywhere are ahj's `bcmp` (reference-side) and the
  shared `memcpy` pack fallback at admit.
- Ring-carried saved hash for saved-hash cells only: PASS — key_string rings carry and
  compare the hash (no recompute per visit); key64/keys256 compute `crc32cx` at admit only
  in both builds.
- Justified divergences observed and documented: (i) slot scheme vs leaf-descriptor
  machinery (id width `UInt64` vs `UInt16`, map-object field offsets on the sync zero-key
  path, no `flag_base_data` in the candidate's slot-local flag scheme — flagless arm
  unaffected); (ii) drain/boundary structure (candidate symbol carries the 8192-row chunk
  loop, policy construction, and the one-per-pass `ProfileEvents` tail-call; slot-preserving
  drain in both); (iii) register allocation/spill placement, steady-loop clone count,
  `prfm` vs `prfum` encodings, inlined equality vs `bcmp`, SIMD vs scalar compare ladders.

UNEXPLAINED differences: none.

G-DISASM-PROBE: PASS (0 unexplained)
