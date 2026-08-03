# X1 — cross-tree, same-binary codegen parity: `DB::` against `DB::Unified::`

One binary: `tmp/uhj_parity/perf2/bin/clickhouse.ref` (aarch64, Neoverse-V2 target).
Tool: `tmp/uhj_parity/perf2/xtree.py`; the exact invocations are in
`tmp/uhj_parity/perf2/xtree_groups.sh`; per-group output in `codegen/xtree/<group>.log`
with full normalised instruction listings in `codegen/xtree/<group>.txt`.

Symbol table: `tmp/nm_ref_demangled.txt`, produced from `clickhouse.ref` with
`../perf/bin/llvm-nm --defined-only --demangle --print-size`. It is byte-identical to the
pre-existing `tmp/nm_sizes.txt` (416348067 bytes both), which confirms that cache was
also taken from `clickhouse.ref`.

## Summary table {#summary}

`identical` means opcode-identical after normalisation. `aligned` numbers come from a
`difflib` alignment of the two instruction sequences, which separates "same code, shifted"
from "different code"; `positional` is the naive index-by-index count and is reported too
because it is what the task asked for. `bag` is the multiset overlap after replacing every
immediate and field offset by `#IMM` — it answers "are these the same instructions in a
different order".

| group | baseline insns (bytes) | unified insns (bytes) | identical | differing positions (positional / aligned-replaced+ins+del) | bag overlap | one-phrase difference |
|---|---|---|---|---|---|---|
| B11/B12 insert, two-level vs two-level | 325 (1300) | 353 (1412) | no | 340 / 224+6+2 | 62.6% | unified range-checks the extra `bucket` argument and computes the bucket count at run time, then hoists the sub-table pointer out of the per-row loop that the baseline re-derives from the hash |
| B11/B12 insert, baseline FLAT vs unified two-level | 311 (1244) | 353 (1412) | no | 342 / 212+5+3 | 61.8% | as above plus the whole two-level indirection, which the flat baseline map does not have |
| P6 `addFoundRowAll` | 196 (784) | 196 (784) | **yes** | 0 | 100% | none |
| P8/P0 `joinBlockImpl` | 687 (2748) | 687 (2748) | no | 66 / 33+1+10 | 98.1% | field offsets only, except one block where the unified side reads two `HashJoin` fields with `ldar` (acquire) that the baseline reads with plain `ldr` |
| N1/N3/N4/N7 `NotJoinedHash::fillColumns` | 660 (2640) | 634 (2536) | no | 580 / 330+168+242 | 67.9% | different bucket-iteration helper: baseline calls `const_iterator::operator++`, unified calls `beginOfNextNonEmptyBucket` (4 sites), rest is block ordering |
| N6 `fillNullsFromBlocks` | 394 (1576) | 394 (1576) | no | 6 / 6+0+0 | 100% | six `HashJoin` member offsets; instruction sequence otherwise identical |
| B2/B3/B4 `addBlockToJoin(Block const&, Selector, bool)` | 3456 (13824) | 2086 (8344) | no | 3405 / 2819+13+32 | 49.9% | not comparable: the baseline inlines the whole 40-arm `joinDispatch` switch, the unified outlines it into `static_for_impl` (+19 helpers, 19694 instructions), and the unified takes a `std::mutex` the baseline does not |
| B2/B3/B4 `addBlockToJoin(Block const&, bool)` entry overload | 142 (568) | 142 (568) | **yes** | 0 | 100% | none |

Call-target normalisation, per group (number of branch/call annotations whose resolved
target name contained `Unified::` and was normalised; and range thunks followed):

| group | Unified-normalised targets (base / unified) | thunks followed (base / unified) | calls whose target still differs |
|---|---|---|---|
| B11/B12 two-level | 3 / 4 | 16 / 17 | 4 positional, 0 in the callee multiset |
| B11/B12 flat | 3 / 4 | 16 / 17 | 7 positional, 0 in the callee multiset |
| P6 | 6 / 7 | 3 / 3 | 0 |
| P8/P0 | 1 / 15 | 19 / 30 | 3 positional, 0 in the callee multiset |
| N1/N3/N4/N7 | 46 / 46 | 22 / 23 | 15 positional; multiset shows `const_iterator::operator++` (baseline, 1) against `beginOfNextNonEmptyBucket` (unified, 4) |
| N6 | 34 / 34 | 16 / 16 | 0 |
| B2/B3/B4 (Selector overload) | 34 / 32 | 127 / 125 | 54 positional; multiset shows 40 baseline `insertFromBlockImpl` calls against 1 unified `static_for_impl` call, and 4 `mutex::lock` / 5 `mutex::unlock` unified-only |
| B2/B3/B4 entry overload | 0 / 18 | 2 / 2 | 0 |

## Exact symbols used {#symbols}

### B11/B12 — build insert, JoinKind 0 (Inner), JoinStrictness 3 (All), `MapsTemplate<RowRefList>`, UInt64 key

Baseline, two-level map, `0x14a2d640`, size 1300:

```
void DB::HashJoinMethods<(DB::JoinKind)0, (DB::JoinStrictness)3, DB::HashJoin::MapsTemplate<DB::RowRefList>>::insertFromBlockImplTypeCase<DB::ColumnsHashing::HashMethodOneNumber<PairNoInit<unsigned long, DB::RowRefList>, DB::RowRefList, unsigned long, false, true, false>, TwoLevelHashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList, HashCRC32<unsigned long>, HashTableNoState, PairNoInit<unsigned long, DB::RowRefList>>, HashCRC32<unsigned long>, TwoLevelHashTableGrower<8ul>, Allocator<true, true>, HashMapTable, 8>, DB::ColumnVector<unsigned long>>(DB::HashJoin&, TwoLevelHashMapTable<...>&, std::__1::vector<DB::IColumn const*, ...> const&, std::__1::vector<unsigned long, ...> const&, unsigned int, DB::ColumnVector<unsigned long> const&, DB::PODArray<char8_t, 4096ul, Allocator<false, false>, 63ul, 64ul> const*, DB::JoinCommon::JoinMask const&, DB::Arena&, bool&, bool&)
```

Baseline, flat map, `0x14a21a40`, size 1244: same name with
`HashMapTable<unsigned long, HashMapCell<...>, HashCRC32<unsigned long>, HashTableGrowerWithPrecalculation<8ul>, Allocator<true, true>>`
in place of the `TwoLevelHashMapTable<...>`.

Unified, `0x16c30580`, size 1412:

```
void DB::Unified::HashJoinMethods<(DB::JoinKind)0, (DB::JoinStrictness)3, DB::Unified::HashJoin::MapsTemplate<DB::RowRefList>>::insertFromBlockImplTypeCase<DB::ColumnsHashing::HashMethodOneNumber<PairNoInit<unsigned long, DB::RowRefList>, DB::RowRefList, unsigned long, false, false, false>, TwoLevelHashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList, HashCRC32<unsigned long>, HashTableNoState, PairNoInit<unsigned long, DB::RowRefList>>, HashCRC32<unsigned long>, HashTableGrowerWithPrecalculation<8ul>, Allocator<true, true>, HashMapTable, -1>, DB::ColumnVector<unsigned long>>(DB::Unified::HashJoin&, TwoLevelHashMapTable<...>&, unsigned long, std::__1::vector<DB::IColumn const*, ...> const&, std::__1::vector<unsigned long, ...> const&, unsigned int, DB::ColumnVector<unsigned long> const&, DB::PODArray<char8_t, 4096ul, Allocator<false, false>, 63ul, 64ul> const*, DB::JoinCommon::JoinMask const&, DB::Arena&, DB::Unified::BuildResult&)
```

Three template/signature differences beyond the namespace, all expected and all visible in
the codegen:

* the extra `unsigned long` (the `bucket`) parameter;
* the two-level grower: baseline `TwoLevelHashTableGrower<8ul>, ..., 8` (256 buckets, a
  compile-time constant), unified `HashTableGrowerWithPrecalculation<8ul>, ..., -1`
  (bucket count known only at run time);
* the key getter's fifth flag: baseline `HashMethodOneNumber<..., false, true, false>`,
  unified `..., false, false, false` (the baseline's unconditional cell offset, loop P4);
* the two `bool&` out-parameters replaced by `DB::Unified::BuildResult&`.

### P6 — `addFoundRowAll`

Baseline `0x16198840`, size 784:

```
void DB::addFoundRowAll<TwoLevelHashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList, HashCRC32<unsigned long>, HashTableNoState, PairNoInit<unsigned long, DB::RowRefList>>, HashCRC32<unsigned long>, TwoLevelHashTableGrower<8ul>, Allocator<true, true>, HashMapTable, 8>, false, true, DB::AddedColumns<false>>(TwoLevelHashMapTable<...>::mapped_type const&, DB::AddedColumns<false>&, unsigned long&, DB::KnownRowsHolder<true>&, DB::JoinStuff::JoinUsedFlags*, bool)
```

Unified `0x17f42ac0`, size 784: the same with `DB::Unified::addFoundRowAll`,
`HashTableGrowerWithPrecalculation<8ul>, ..., -1`, `DB::Unified::AddedColumns<false>`,
`DB::Unified::KnownRowsHolder<true>`, `DB::Unified::JoinStuff::JoinUsedFlags*`.

### P8/P0 — `joinBlockImpl`

Baseline `0x14a38880`, size 2748:

```
DB::HashJoinMethods<(DB::JoinKind)0, (DB::JoinStrictness)3, DB::HashJoin::MapsTemplate<DB::RowRefList>>::joinBlockImpl(DB::HashJoin const&, DB::ScatteredBlock, DB::Block const&, std::__1::vector<DB::HashJoin::MapsTemplate<DB::RowRefList> const*, std::__1::allocator<DB::HashJoin::MapsTemplate<DB::RowRefList> const*>> const&, bool)
```

Unified `0x16c48300`, size 2748: the same name with `DB::Unified::` throughout. The size
histogram of all 60 `joinBlockImpl` instantiations is identical between the trees
(30 × 0x28c, 12 × 0xa9c, 12 × 0xabc, 5 × 0xb08, 1 × 0xb28), and every name matches
one-to-one after deleting `Unified::`.

### N1/N3/N4/N7 — `NotJoinedHash::fillColumns`

Baseline `0x142a4740`, size 2640:

```
unsigned long DB::NotJoinedHash::fillColumns<TwoLevelHashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList, HashCRC32<unsigned long>, HashTableNoState, PairNoInit<unsigned long, DB::RowRefList>>, HashCRC32<unsigned long>, TwoLevelHashTableGrower<8ul>, Allocator<true, true>, HashMapTable, 8>>(TwoLevelHashMapTable<...> const&, std::__1::vector<COW<DB::IColumn>::mutable_ptr<DB::IColumn>, std::__1::allocator<COW<DB::IColumn>::mutable_ptr<DB::IColumn>>>&)
```

Unified `0x165e81c0`, size 2536: the same with `DB::Unified::NotJoinedHash::fillColumns`
and `HashTableGrowerWithPrecalculation<8ul>, ..., -1`.

### N6 — `fillNullsFromBlocks`

`DB::NotJoinedHash::fillNullsFromBlocks(std::__1::vector<COW<DB::IColumn>::mutable_ptr<DB::IColumn>, std::__1::allocator<COW<DB::IColumn>::mutable_ptr<DB::IColumn>>>&, unsigned long&)`
at `0x1429e740` and the `DB::Unified::` counterpart at `0x165e5100`, both size 1576. The
only two symbols in the binary matching `fillNullsFromBlocks`.

### B2/B3/B4 — `addBlockToJoin`

`DB::HashJoin::addBlockToJoin(DB::Block const&, DB::detail::Selector, bool)` at
`0x14289c00`, size 13824, against
`DB::Unified::HashJoin::addBlockToJoin(DB::Block const&, DB::detail::Selector, bool)` at
`0x165d4f00`, size 8344. Also the two-argument entry overload
`DB::HashJoin::addBlockToJoin(DB::Block const&, bool)` at `0x142899c0` and
`DB::Unified::HashJoin::addBlockToJoin(DB::Block const&, bool)` at `0x165d4cc0`, both
size 568.

Nothing was inlined away: every symbol in the task list resolved. No "could not resolve"
entries.

## Findings per group {#findings}

### P6 `addFoundRowAll` — identical {#p6}

196 instructions each, byte-identical after normalisation, and the instruction bag is 100%
common with no immediates stripped. The two trees emit the same code for the
`RowRefList` walk. This is the strongest possible cross-tree result and it confirms the
`loops.py` note "verified textually identical between trees" at the machine level.

Getting here required two fixes to the normalisation, both of which had been producing
false differences (see "Normalisation" below): self-referential branches, and range
thunks. Before them this group reported 40 differing positions, every one of which was an
artefact.

### N6 `fillNullsFromBlocks` — identical modulo six field offsets {#n6}

394 instructions each, same size, same callee multiset (46 calls to 17 targets on both
sides), and the aligned diff is 6 replacements and nothing else. All six are one
`HashJoin`/`TableJoin` member offset apart:

```
[  23] base    ldr x9, [x8, #0xc0]        unified ldr x9, [x8, #0x148]
[  26] base    ldr x10, [x9, #0xb8]       unified ldr x10, [x9, #0xc0]
[  27] base    ldp x12, x13, [x9, #0xa0]  unified ldp x12, x13, [x9, #0xa8]
[  43] base    ldr x9, [x8, #0xc0]        unified ldr x9, [x8, #0x148]
[  45] base    ldp x8, x10, [x9, #0xa0]   unified ldp x8, x10, [x9, #0xa8]
[  48] base    ldp x10, x9, [x9, #0xb8]   unified ldp x10, x9, [x9, #0xc0]
```

The `#0xc0` → `#0x148` pair recurs in every group below: it is the offset of the same
member of the `HashJoin` object, which sits 0x88 bytes later in the unified class.

### P8/P0 `joinBlockImpl` — identical except two acquire loads {#p8p0}

687 instructions each, identical size, identical callee multiset (76 calls to 25 targets
each, zero targets on one side only). Of the 33 aligned replacements, 11 are pure
immediate/offset changes, 2 differ only in register choice, and 5 have a different opcode.
Those 5 all sit in one basic block near index 518 which builds the result state:

```
base     ldr x8, [x20, #0xc0]      unified  add x9, x20, #0x2f0
base     ldr x9, [x8, #0x150]      unified  add x8, x9, #0xf8   ; then  ldar x8, [x8]
base     ldr x8, [x8, #0x140]      unified  add x9, x9, #0x108  ; then  ldar x9, [x9]
```

The unified side reads two `HashJoin` fields with `ldar` (load-acquire) where the baseline
reads them with a plain `ldr`; the opcode histogram delta is exactly `ldar +2, nop -2,
ldr -1, add +1`. Interpretation: those two members are atomic in the unified tree and
plain in the baseline. On Neoverse-V2 an `ldar` that hits L1 is a few cycles dearer than
an `ldr` and blocks subsequent loads from being reordered ahead of it, but this block runs
once per probe block, not per row, so the per-row probe path (P1, P3) is unaffected.
Everything else in this 687-instruction function is offsets.

### B11/B12 insert — differs, as expected, and the extra work is one-time {#b11}

The maps genuinely differ, so a difference was expected. The useful question is where it
lands, and the answer is favourable: the unified side pays a one-time prologue and then
runs a *shorter* per-row loop.

Baseline two-level 325 instructions, unified 353, delta +28. What the extra instructions do:

1. Bounds-check the new `bucket` argument against the bucket vector, five instructions
   that have no baseline counterpart:

```
   U  ldp x26, x8, [x24, #0x10]   ; begin/end of the bucket vector
   U  sub x8, x8, x26
   U  asr x9, x8, #7              ; / 128 == number of sub-tables
   U  cmp x23, x9                 ; bucket vs count
   U  b.hs <SELF+0x4dc>           ; out of range -> cold path
```

2. Re-derive the bucket count at run time where the baseline had it as a literal. The
   baseline sums the per-bucket sizes with a fixed trip count, `mov w11, #0x100`
   (256 buckets, from the `..., 8` template argument); the unified loop derives the range
   from the same vector (`and x9, x8, #ADDR`, `add x12, x26, #0xc0`) and needs an extra
   remainder loop afterwards (`add x10, x26, x9, lsl #7 ... ldrb w11, [x9], #0x80;
   subs x8, x8, #0x1; lsl x11, x10, x11; add x20, x11, x20; b.ne`), because the range is
   no longer a whole number of unrolled pairs. This accounts for most of the 6 inserted
   and 86 insert/delete positions.

3. In exchange, the per-row body loses three instructions. The baseline re-derives the
   sub-table from the hash on every row:

```
   B  lsr x10, x9, #17            ; getBucketFromHash
   B  and x10, x10, #0x7f80       ; * sizeof(sub-table)
   B  add x10, x11, x10
   B  ldr x11, [x10, #0x48]       ; mask
   B  ldr x10, [x10, #0x20]       ; buf
```

   while the unified computed the sub-table address once before the loop
   (`add x23, x26, x23, lsl #7`, from the passed-in `bucket`) and the body is just:

```
   U  ldr x10, [x23, #0x48]
   U  and x9, x10, x9
   U  ldr x10, [x23, #0x20]
```

   The unified body does add a `cbz x20, <SELF+0x408>` null check on the fetched column
   pointer that the baseline does not have.

So the +28 instructions are per-*block* setup and the per-*row* difference is -3 (bucket
derivation) +1 (null check) in the unified's favour. The callee multisets are the same
size (32 baseline / 35 unified calls to 20 targets each) with no target on one side only
apart from two extra self-calls and one `__libcpp_verbose_abort` on the unified cold path
— consistent with the added range check.

Against the baseline *flat* map (311 instructions) the picture is the same plus the
two-level indirection itself: 42 more instructions, 35 with a different opcode.

### N1/N3/N4/N7 `fillColumns` — differs, and it is a different bucket-iteration API {#n1}

660 baseline against 634 unified instructions, positional diff 580, aligned diff 330
replaced / 168 inserted / 242 deleted, instruction bag overlap 67.9%. Two facts pin down
what the difference is.

The two trees walk the buckets through *different* helpers, and that is the one real
callee difference:

```
baseline  1 x  bl <TwoLevelHashTable<unsigned long, HashMapCell<unsigned long, DB::RowRef, ...>,
                    ..., TwoLevelHashTableGrower<8ul>, Allocator<true, true>>, 8, void>
                    ::const_iterator::operator++()>
unified   4 x  bl <TwoLevelHashTable<unsigned long, HashMapCell<unsigned long, DB::RowRef, ...>,
                    ..., HashTableGrowerWithPrecalculation<8ul>, Allocator<true, true>>, -1, void>
                    ::beginOfNextNonEmptyBucket(unsigned long&) const>
```

Neither call appears on the other side (baseline: 1 `operator++`, 0
`beginOfNextNonEmptyBucket`; unified: 0 and 4; 43 against 49 `bl` in total). The baseline
advances with the standard two-level `const_iterator`; the unified skips directly to the
next non-empty bucket. That is a genuine algorithmic difference in the non-joined scan, not
a compilation artefact, and it is consistent with the `loops.py` note that the unified N1
uses `offsetInternalAtBucket`.

What is *not* different: the run-time bucket arithmetic. `udiv`/`msub` pairs occur exactly
five times on each side — the modulo by a run-time divisor for the per-stream bucket-range
split (loop N4) is present identically in both trees.

Everything else is block ordering and register allocation (the frame is 0x110 bytes in the
baseline against 0xf0 in the unified), with 33 aligned positions where the opcode itself
differs and 17 more where a branch differs only in ICF alias naming. The remaining
one-side-only callees are a cold-path `_Unwind_Resume` and ICF aliases (see the caveat
below).

### B2/B3/B4 `addBlockToJoin` — not comparable at whole-symbol level {#b234}

The two-argument entry overload is **opcode-identical**, 142 instructions each. The
`Selector` overload, into which B2, B3 and B4 are inlined, is not comparable:

* baseline 3456 instructions, unified 2086, delta -1370;
* the baseline makes **40** direct `bl` calls to
  `DB::HashJoinMethods<(JoinKind)k, (JoinStrictness)s, MapsTemplate<...>>::insertFromBlockImpl`
  instantiations — the whole `joinDispatch` switch is inlined into `addBlockToJoin`;
* the unified makes **one** call to
  `static_for_impl<int, 0, bool DB::Unified::joinDispatch<...>>` (`0x1662be80`, 24272
  bytes) and has 19 outlined `func_wrapper`/`static_for_impl` helpers totalling 78776
  bytes (19694 instructions) that mention `DB::Unified::HashJoin::addBlockToJoin`; the
  baseline has **zero** such helpers;
* the unified takes a lock the baseline does not: 4 `std::mutex::lock` and 5
  `std::mutex::unlock` calls against 0 and 0;
* the baseline's opcode histogram has 40 more each of `mul`, `madd`, `asr`, `movk` and
  `b.hs` — the arithmetic of the 40 inlined dispatch arms.

The 919 aligned-identical instructions are scattered in runs of at most 18, so they do not
constitute a recognisable shared region: the frames differ throughout (baseline addresses
locals off `x29` with negative offsets, unified off `sp` with positive ones, a consequence
of the much larger baseline frame), which is why 2819 positions are classified as
replaced. Verdict for B2/B3/B4: **no codegen-level parity statement can be made from this
symbol pair.** To get evidence for those three loops the comparison would have to be made
against the unified `static_for_impl`/`func_wrapper` bodies, or the loops would have to be
isolated by line-table region rather than by symbol.

## Normalisation {#normalisation}

`xtree.py` reuses every rule from `symdiff.body`:

* hex tokens of five or more digits become `ADDR`, shorter ones are left alone (a short
  hex is a real immediate; normalising it once hid the A-K1 ablation — WORKLOG F7);
* llvm-objdump's `<symbol>` operand annotation is stripped except on branch and call
  opcodes;
* the short immediate of the `add`/load/store paired with a preceding `adrp` into the same
  register becomes `#RELOC_LO`.

Four rules are specific to the cross-tree case. The count of items normalised by each is
reported per group so that none of them can quietly hide a real difference.

1. **`Unified::` deleted from resolved call-target names.** Only the `Unified`
   namespace component is deleted, so `DB::Unified::HashJoin` becomes `DB::HashJoin` and
   matches the baseline name. Deleting the whole `DB::Unified::` would yield `HashJoin`
   and make every such callee differ again. Counts per group are in the second table
   above (3–46 per side). A call whose target still differs after this is reported
   separately, both positionally and as a callee-multiset difference; the only two groups
   with a genuine multiset difference are B2/B3/B4 (the dispatch outlining and the mutex)
   and N1/N3/N4/N7 (`const_iterator::operator++` against `beginOfNextNonEmptyBucket`).
   Both are real, and both were found this way rather than from the positional diff.

2. **Targets resolved through the symbol table, not read from objdump's text.** objdump
   prints mangled names, and a unified mangled name is *not* the baseline one with a
   substring removed — the substitution indices shift — so textual deletion on the mangled
   form would not work. Resolving the branch target address to a demangled name makes the
   two sides comparable and incidentally handles identical-code folding, since the whole
   set of names at the target address is available.

3. **Self-references rendered as `SELF`.** A branch inside the function under comparison,
   or a tail call to itself, names the function, and that name embeds the map type, which
   legitimately differs (`TwoLevelHashTableGrower<8ul>, ..., 8` against
   `HashTableGrowerWithPrecalculation<8ul>, ..., -1`). Without this rule every loop
   back-edge reported as a different call target: P6 showed 40 such differences and is in
   fact identical.

4. **Linker range thunks followed.** The unified tree is linked tens of megabytes from the
   baseline tree, so a call the baseline reaches directly may need an
   `__AArch64ADRPThunk_...` from the unified side. That is placement, not code, so the
   thunk is followed to its target (both the one-instruction `b <target>` form and the
   `adrp`/`add`/`br` form). Thunk counts per group are in the second table; P8/P0 needed
   19 on the baseline side and 30 on the unified side, and without this rule reported 11
   spurious differing calls.

### A bug found in `symdiff.py` {#symdiff-bug}

`symdiff.body` detects the opcode with `t.split(" ")[0]`, but llvm-objdump separates the
mnemonic from the operands with a **tab**. The opcode therefore came out as
`"bl\t0x1428a1e0"`, never matched the branch pattern, and the documented rule "keep the
`<symbol>` annotation on branch and call opcodes" never fired — `symdiff.py` has been
stripping callee names everywhere. For a same-symbol diff that is only conservative (it
can miss a retargeted call whose instruction encoding is unchanged), so no earlier
conclusion is invalidated in the unsafe direction, but for the cross-tree diff the callee
name is the whole point. `xtree.py` collapses whitespace before the opcode test. Fixing
this in `symdiff.py` is left alone deliberately: it is outside this task's scope and would
change the baseline of earlier validity checks.

### Caveat: ICF aliasing in the callee multiset {#icf-caveat}

Identical-code folding puts several unrelated names on one address, so the callee
histogram picks one deterministically (a name containing `::` first, then the shortest).
When the two trees call two *different* folded groups, the chosen names can look unrelated
in a way that overstates the difference — this is why `DB::QuantileLevels<double`,
`evp_mac_from_algorithm` and `std::__1::filesystem::__space` appear in some
one-side-only lists. The positional diff does not have this problem: it compares the whole
name set at the target address and classifies a pair whose sets intersect as
"alias naming only" (10, 6, 17 and 52 such positions in the four groups that have any).

## Reproducing {#reproducing}

```bash
cd tmp/uhj_parity/perf2
bash xtree_groups.sh                      # all groups, ~5 minutes, writes codegen/xtree/
python3 xtree.py --base '<regex>' [--unified '<regex>'] --list   # resolve symbols first
python3 tmp/adiff.py codegen/xtree/<group>.txt --structural      # full aligned diff
```

`xtree.py` exits 0 when the pair is opcode-identical, 1 when it differs, 2 when a side
cannot be resolved, and 3 when the regexes are ambiguous (it then lists the candidates).
