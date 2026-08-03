# G5.3 inertness proof for the `unified_hash` fix set

`before` = `bin/clickhouse.bold` (HEAD `7de421147f1`, unmodified)
`after`  = `bin/clickhouse.bnew` (the fix set)

Two shared headers were edited and both are compiled into the comparators:
`src/Common/HashTable/TwoLevelHashTable.h` (also `Aggregator`, also `parallel_hash` through
the baseline `HashJoin`) and `src/Common/ColumnsHashingImpl.h` (also `Aggregator`, `Set`,
and both baselines' key getters). So the A side of every A/B number in the report has to be
shown unperturbed.

## The first attempt was not inert, and the tool caught it

`findKeyImpl` was first rewritten to route the offset through a new `findAndOffset` helper
on every path. That is semantically identical for a table without `findWithOffset`, and it
still changed the generated code:

```
resized OUTSIDE, non-thunk: 42
  18  TwoLevelHashTable<..., -1, ...>::Prober<false> lambdas   (unified's own - allowed)
  14  unsigned long DB::HashJoinMethods<...>::joinRightColumns  (BASELINE - not allowed)
  10  DB::Aggregator::executeImplBatch / mergeStreamsImpl       (not allowed)
```

Mostly *smaller*, which is worse than useless: an accidental improvement to the comparator
would have eaten part of the measured win. The `always_inline` helper is inlined early, but
the resulting IR is not identical to the straight-line original, and downstream layout and
register-allocation decisions followed it.

Fixed by making the fused-offset path a branch of its own and leaving the pre-existing
path textually untouched:

```cpp
if constexpr (has_fused_offset_lookup<Data, Key>) { ...new... }
else { ...the original body, verbatim... }
```

## After the fix

```
$ python3 tmp/uhj_parity/perf2/symdiff.py \
    --before tmp/uhj_parity/perf2/bin/clickhouse.bold \
    --after  tmp/uhj_parity/perf2/bin/clickhouse.bnew \
    --expect-changed-regex 'DB::Unified::|::Prober<' \
    --byte-compare 'DB::ConcurrentHashJoin::addBlockToJoin' \
    --byte-compare 'DB::HashJoin::addBlockToJoin' \
    --byte-compare 'fillFixedBatch<unsigned long'

  resized : 9617  (9481 allowed, 0 OUTSIDE non-thunk)
  removed : 2131  (1950 allowed, 0 OUTSIDE non-thunk)
  added   : 2687  (2399 allowed, 200 OUTSIDE non-thunk - see below)

  byte-compare 'DB::ConcurrentHashJoin::addBlockToJoin': 1 symbols, 1 opcode-identical
  byte-compare 'DB::HashJoin::addBlockToJoin'          : 2 symbols, 2 opcode-identical
  byte-compare 'fillFixedBatch<unsigned long'          : 4 symbols, 4 opcode-identical
```

Plus a direct check of the comparator's probe loop, which is the symbol the one-thread
deficit lives in: **16,680** baseline `DB::HashJoinMethods<...>::joinRightColumns`
instantiations exist, **none changed size**, and a random sample of 24 (seed 7) is
**24/24 opcode-identical**.

**Why the allowed regex includes `::Prober<`.** Unified's maps are
`TwoLevelHashMap<..., BITS_FOR_BUCKET=-1>`, and their symbol names carry no `DB::Unified::`
prefix - the *table* is a `Common/` template. `-1` (runtime bucket count) is used by
`Unified::HashJoin` and by nothing else in the tree (`rg 'BITS_FOR_BUCKET'`), and `Prober`
exists only under `requires(isRuntimeStorage())`, so every `::Prober<` symbol is unified's.

**The 200 added symbols are all additions, not modifications**, and all of one shape:
`std::__shared_ptr_emplace<DB::ColumnsHashing::HashMethod...>` control blocks and the
matching `__destroy_at<...>`, which are the type-erased storage `BlockKeyGetter` needs, plus
one `std::list<DB::StoredBlock>::push_back(StoredBlock&&)` from the `L3` change. New code
cannot perturb an existing baseline function, and the resized count confirms none was.

## A third defect in `symdiff.py`, found and fixed

The previous session recorded two (an `adrp` annotation false positive, and a normaliser
that could not see the A-K1 ablation). This is a third, in the same area and in the
false-alarm direction.

An address is materialised as `adrp xN, <page>` plus a second instruction supplying the low
bits. The normaliser recognised the low half only as `add xN, xN, #lo`, and not as a
load *through* the register, `ldr xM, [xN, #lo]` - which is how a GOT entry is read. Because
the binary's data section moved, that made
`DB::HashJoin::addBlockToJoin` report **35 differing instructions in a function of identical
length whose source had not changed**; 32 of the 35 had an `adrp` on the base register within
the preceding three instructions, the other 3 within a longer window. Every one was a GOT
offset.

Fixed by matching the base register inside brackets as well, keeping the register paired
until something other than a paired use redefines it. The fix cannot hide a field-offset
change off a long-lived register (`ldr x10, [x22, #0x48]`), because such a register was
never set by an `adrp`.

Re-validated against the known-good case so the fix is not just permissive: on
`clickhouse.ref -> clickhouse.ak1` (ablation A-K1) the tool still reports **GREEN**, still
sees the ablation through `--expect-differ`, and still calls both baselines opcode-identical.
That ablation's whole footprint is one instruction (`mov w9, #0x2` -> `#0x1`), so a
normaliser that had gone blind would have failed there.

Linker range-extension thunks (`__AArch64ADRPThunk_*`) are now counted separately rather
than as "outside": they have no source of their own, and any edit that moves code churns
hundreds of them.
