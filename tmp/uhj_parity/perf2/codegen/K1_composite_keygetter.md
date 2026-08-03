# K1 — the composite-key `HashMethodKeysFixed` constructor

Binary: `tmp/uhj_parity/perf2/bin/clickhouse.ref` (aarch64, host Neoverse-V2).
Tools: `tmp/uhj_parity/perf2/codegen.py` (`syms`/`dis`/`count`/`mca`), `tmp/uhj_parity/perf/bin/llvm-nm`.
Nothing under `src/` was modified.

The headline result is that the enumeration's per-block label for this constructor is wrong in a
way that matters: the constructor is a **per-row (O(rows)) key-packing pass over the whole block**,
it is **re-run once per build shard/bucket over the same full block**, and the profile attributes
its callees' cycles to it. The two specialisations (`need_offset = true` for the baseline trees,
`false` for unified) are **instruction-for-instruction identical**; the difference between the
algorithms is entirely how many times the function is called, not what it compiles to.

---

## 1. The template parameters, and what the last one is

`src/Common/ColumnsHashing/HashMethod.h:389-397` (reached from `src/Common/ColumnsHashing.h`, which
includes `ColumnsHashing/HashMethod.h`; `ColumnsHashingImpl.h` holds the `HashMethodBase` /
`BaseStateKeysFixed` bases, not this declaration):

```
template <
    typename Value,                        // 1  hash-table cell value  (PairNoInit<UInt128, RowRefList>)
    typename Key,                          // 2  packed key type        (wide::integer<128,uint> = UInt128)
    typename Mapped,                       // 3  mapped type            (RowRefList / RowRefList const)
    bool has_nullable_keys_ = false,        // 4
    bool has_low_cardinality_ = false,      // 5
    bool use_cache = true,                  // 6
    bool need_offset = false>               // 7  <-- the trailing true/false in the profile
struct HashMethodKeysFixed
```

So the observed profile symbols decode as
`<Value=PairNoInit<UInt128,RowRefList>, Key=UInt128, Mapped=RowRefList, has_nullable_keys=false,
has_low_cardinality=false, use_cache=false, need_offset=true|false>`.

The final parameter is **`need_offset`** — VERIFIED, and the mission's guess about where it comes
from is also verified:

* baseline: `src/Interpreters/HashJoin/KeyGetter.h:19` `constexpr bool use_offset = true;`, passed at
  `KeyGetter.h:212-215` (`keys128`) and `:258-261` (`two_level_keys128`) as the 7th argument. It is a
  *constant*: every baseline join, `hash` and `parallel_hash` alike, instantiates `need_offset=true`.
* unified: `src/Interpreters/UnifiedHashJoin/HashJoinMethods.h:90`
  `static constexpr bool needs_offset = JoinFeatures<KIND, STRICTNESS, MapsTemplate>::need_flags;`,
  threaded through `UnifiedHashJoin/KeyGetter.h:225-228` and `:269-276`. For this cell
  (INNER / ALL) `JoinFeatures::need_flags` (`UnifiedHashJoin/JoinFeatures.h:40`) resolves through
  `MapGetter<Inner, All, *>::flagged` (`UnifiedHashJoin/joinDispatch.h:47`) which is `false` — hence
  `need_offset=false`, hence a second specialisation.

Note also parameter 6, `use_cache=false`, in *both* — no join uses the hash cache.

`need_offset` is consumed only in `HashMethodBase`'s `FindResult` (whether a matched cell's global
offset is produced/carried). It has **no effect on the constructor body** — see §3.

## 2. What the constructor does: O(rows), not O(columns)

`src/Common/ColumnsHashing/HashMethod.h:435-512`. With `has_low_cardinality=false` the
low-cardinality block (`:438-454`) is compiled out, so the body is:

1. `Base(key_columns)` — `BaseStateKeysFixed` copies the `ColumnRawPtrs` vector (one heap allocation
   + `memcpy`), and `key_sizes(key_sizes_)` copies the `Sizes` vector (a second heap allocation +
   `memcpy`). Both O(columns).
2. `if (usePreparedKeys(key_sizes)) packFixedBatch(keys_size, Base::getActualColumns(), key_sizes, prepared_keys);`
   — **`HashMethod.h:456-459`. This is the O(rows) work.**
3. The `#if defined(__SSSE3__)` shuffle-mask branch (`:461-511`) — **does not exist on aarch64**, so
   on this host `packFixedBatch` is the *only* prepared path; when `usePreparedKeys` is false the
   constructor does nothing but the two vector copies and `getKeyHolder` falls back to per-row
   `packFixed` (`:537`).

**Exact condition for the O(rows) work** — `usePreparedKeys`, `HashMethod.h:423-433`: false if
`has_low_cardinality || has_nullable_keys || sizeof(Key) > 16`, else false if any `key_sizes[i]` is
not one of 1/2/4/8/16, else **true**. For this cell (two non-nullable `UInt64` columns,
`key_sizes = {8,8}`, `sizeof(UInt128) == 16`) it is **true**, so every construction packs.

**What the packing costs** — `packFixedBatch` (`src/Interpreters/AggregationCommon.h:80-89`) calls
`fillFixedBatch<T, Key>` five times (T = UInt128, UInt64, UInt32, UInt16, UInt8); each
(`AggregationCommon.h:57-76`) scans the columns for `key_sizes[i] == sizeof(T)` and for each match
does `out.resize_fill(num_rows)` (a `memset` of `16 * num_rows` bytes) followed by
`fillFixedBatch<T, sizeof(Key)/sizeof(T)>` (`:40-49`) — **a scalar `num_rows`-iteration strided copy
loop**. For `{8,8}` that is one `memset` of `16·R` bytes plus **two `R`-iteration loops**, R = rows.

Two further facts that make this worse than "O(rows) once per block":

* `num_rows` comes from `column->size()` (`AggregationCommon.h:65`) — **the whole block**, not the
  selector. A `ScatteredBlock` shard/bucket that owns 1/64th of a block still packs all of it.
* The constructor is created per *shard/bucket*, not per block. See §5.

**Consequently the enumeration's `per block` multiplicity for this loop is wrong.** The honest
formula is `2·R + memset(16·R)` per construction, and `num_shards`(or `num_buckets`)`+1`
constructions per build block — see §5 for the multiplicity table.

### 2a. Why the profile shows the cost *in the constructor* rather than in `fillFixedBatch`

`fillFixedBatch` is **not** inlined into the constructor (five `bl`s — §3), and there are **zero**
samples on any `fillFixedBatch` symbol in `results/samples_u0a.jsonl` for any cell. That is not a
contradiction: `enumerate.py:106-140` does not aggregate the true leaf. It walks the stack outward
from the leaf and credits **the innermost frame matching `IN_JOIN_MARKERS`** (`loops.py:23-45`).
`DB::fillFixedBatch<...>`, `memset`, `memcpy` and `operator new` match none of those markers, while
`ColumnsHashing` matches the constructor — so all of the packing loop's, the `memset`'s and the two
allocations' cycles are credited to the constructor frame. The 2695 / 5039 samples are therefore
**the whole packing pass**, which is exactly consistent with §2 and with the multiplicity in §5, and
not consistent with "644 bytes of vector copying is 45% of the join".

## 3. Codegen of the two specialisations

### Symbols and sizes (`llvm-nm --print-size --demangle --defined-only`)

| specialisation (Value=PairNoInit<UInt128,RowRefList>, Key=UInt128) | address | size |
|---|---|---|
| `Mapped=RowRefList, …,false,false,false,true>::HashMethodKeysFixed` (baseline build) | `0x13ab1280` | `0x284` = 644 B |
| `Mapped=RowRefList, …,false,false,false,false>::HashMethodKeysFixed` (unified build) | `0x16c4c6c0` | `0x284` = 644 B |
| `Mapped=RowRefList const, …,true>::HashMethodKeysFixed` (baseline probe) | `0x1437f740` | `0x284` = 644 B |
| `Mapped=RowRefList const, …,false>::HashMethodKeysFixed` (unified probe) | `0x16caa1c0` | `0x284` = 644 B |
| `~HashMethodKeysFixed` (both) | `0x13ab1540` / `0x16c4d5c0` | `0x88` = 136 B |
| `DB::fillFixedBatch<UInt64, UInt128>` (**one** copy, shared by all of the above) | `0x0e3bbb40` | `0x1ac` = 428 B |

All four constructor instantiations are the same size, and **no ICF folding is involved**: exactly
one symbol name resolves to each of `0x13ab1280`, `0x16c4c6c0`, `0x16caa1c0`, `0x1437f740`,
`0x0e3bbb40` in the full `llvm-nm` dump. So each is genuinely its own 644-byte copy of the same code.

### How the ranges were chosen

`dis --size 0x284` (the exact `llvm-nm` size) for each constructor. Each starts with a `b <L0>` +
7 `nop` patch pad; `<L0>` is the real prologue. The straight-line path runs prologue → the two
vector copies → the `usePreparedKeys` scan `<L4>` → the five `fillFixedBatch` calls → epilogue
`b <L7>`; everything from `+0x1f8` on is the cold landing-pad/`_Unwind_Resume` tail, which is
excluded.

* `ctor_true` full path: `0x13ab12a0:0x13ab1454` (symbol `+0x20 … +0x1d4`).
* `ctor_false` full path: `0x16c4c6e0:0x16c4c894` (the same `+0x20 … +0x1d4`).
* `usePreparedKeys` per-column loop `<L4>`: `0x13ab13a0:0x13ab13c0` / `0x16c4c7e0:0x16c4c800`.
* the real per-row loop, `fillFixedBatch<UInt64,UInt128>` `<L7>`: `0x0e3bbc80:0x0e3bbc8c`
  (`ldr x9,[x10],#8 ; subs x24,x24,#1 ; str x9,[x8],#0x10 ; b.ne`), plus its full path
  `0x0e3bbb60:0x0e3bbc94` for context.

### The two specialisations are the same code

`diff` of the two disassemblies with addresses stripped: **the only differences are call targets**
— the `false` copy is laid out far from its callees, so its 9 calls go through
`__AArch64ADRPThunk__Znwm` / `__AArch64ADRPThunk__ZN2DB14fillFixedBatch…` / `__AArch64ADRPThunk__ZdlPvm`
range-extension thunks, while the `true` copy calls `_Znwm` / `fillFixedBatch…` / `_ZdlPvm`
directly. Same instructions, same registers, same order, same count. Both call the **same single**
`fillFixedBatch` instances. `need_offset` changes only `FindResult`, which the constructor does not
touch, so this is expected — and it means **there is no codegen penalty for unified's
specialisation**; the second instantiation costs I-cache footprint (644 B + 136 B) and nothing else.

### Counts (`codegen.py count`; nops excluded, classes per the inherited counter)

| range | insns | loads | stores | branches | calls | spill st/ld | dep-load depth |
|---|---|---|---|---|---|---|---|
| `ctor_true` full path `13ab12a0:13ab1454` | 107 | 9 | 21 | 18 | 9 | 6 / 5 | 2 |
| `ctor_false` full path `16c4c6e0:16c4c894` | 107 | 9 | 21 | 18 | 9 | 6 / 5 | 2 |
| `usePreparedKeys` loop (either) | 9 | 1 | 0 | 2 | 0 | 0 / 0 | 1 |
| **`fillFixedBatch` per-row loop `<L7>`** | **4** | **1** | **1** | **1** | **0** | 0 / 0 | 1 |
| `fillFixedBatch` full path | 74 | 15 | 10 | 17 | 3 | 6 / 0 | 4 |

### llvm-mca (`-mcpu=neoverse-v2`, 100 iterations, `-bottleneck-analysis`)

| block | cycles/iter | IPC | uOps/cycle | Block RThroughput | limiting resource | caveat |
|---|---|---|---|---|---|---|
| `ctor_true` full path | 24.06 | 4.24 | 5.69 | 22.8 | `V2UnitL01` (13.5/iter, both L/S pipes) | **LOWER BOUND** — 9 calls dropped |
| `ctor_false` full path | 24.06 | 4.24 | 5.69 | 22.8 | `V2UnitL01` (13.5/iter) | **LOWER BOUND** — 9 calls dropped |
| `usePreparedKeys` loop | 2.10 | 4.76 | 5.24 | 1.8 | `V2UnitFlg`/`V2UnitD` | 1 branch retargeted to block end |
| **`fillFixedBatch` per-row loop** | **1.56** | 3.21 | 5.13 | **1.3** | `V2UnitL01` (0.76+0.76) | none; block self-contained |

mca reported "No resource or data dependency bottlenecks discovered" for every block; the limiting
resource above is read off the `Resource pressure per iteration` table (`V2UnitL01` = the two
load/store pipes) rather than from a bottleneck verdict.

Caveats travelling with these numbers, all emitted by the tool:

* Both constructor blocks: **9 calls dropped** (`bl _Znwm` ×2, `bl memcpy` ×2, `bl fillFixedBatch` ×5)
  — mca cannot model callees, so **24.06 cycles is a lower bound** on the constructor and excludes
  precisely the expensive part (two mallocs, two memcpys and the whole O(rows) packing).
  Three branches leaving the range (`<L8>`, `<L9>`, `<L7>`) were retargeted to the block end, which
  changes nothing about straight-line resource use.
* Both constructor blocks needed **one hand fixup** beyond what `codegen.py` does: the line
  `adrp x10, 0x26e4000 <…symbol…>` is rewritten by the tool to `adrp x10, 0x26e4000 .Lblockend`,
  which does not assemble; the stray operand was removed (`tmp/ctor_{true,false}_fixed.s`). Nothing
  else was edited. The `usePreparedKeys` and `fillFixedBatch` blocks assembled unmodified.
* The per-row loop number (1.56 cyc/element) is the trustworthy one: 4 instructions, no calls,
  self-contained. It is a *throughput* figure and does not include the `memset` of `16·R` bytes that
  `resize_fill` performs once per `fillFixedBatch` call.

The loop is scalar and cannot be vectorised as written: the store is `str x9,[x8],#0x10` — a
16-byte-strided single 8-byte store into the packed `UInt128` array — so each key half is written
by its own pass over the block, with a full `memset` of the destination first.

## 4. Why `hash` "does not pay it": not inlining

**"`hash` inlined the constructor" is NOT supported. It is false.**

* `hash` and `parallel_hash` are the *same* implementation (`DB::HashJoin`) with the same
  `constexpr bool use_offset = true`, so they instantiate the *same* specialisation and, since there
  is exactly one symbol at `0x13ab1280` and no ICF fold, they execute **literally the same
  out-of-line 644-byte function**. It cannot be inlined for one caller and not the other: it is one
  address, reached by `bl` from `insertFromBlockImplTypeCase` in both.
* The `hash` cell does have samples in it — 130 in the `RowRefList` (build) instantiation plus 113 in
  the `RowRefList const` (probe) one. It is the same code, sampled ~20× less often.
* Symbol sizes are equal across all four instantiations (644 B), and `insertFromBlockImplTypeCase`
  is present as its own hot symbol in all three cells (445 / 954 / 872 samples) — i.e. nothing was
  absorbed into it.

The samples did not "go somewhere else" either: **the constructor is called far fewer times for
`hash`.** §5.

## 5. The real variable: how many times the constructor runs per block

| algorithm | constructions per build block | why | file:line |
|---|---|---|---|
| `hash` | **1** | `HashJoin::addBlockToJoin` → one `insertFromBlockImpl` for the whole block | `HashJoin/HashJoin.cpp:643,802-813`; `HashJoin/HashJoinMethodsImpl.h` `insertFromBlockImplTypeCase` |
| `parallel_hash` | **`slots` (= 64 here)** | `ConcurrentHashJoin::dispatchBlock` splits the block into one `ScatteredBlock` per slot; on the zero-copy path (`scatterBlocksWithSelector`) all slots **share the full columns**, and each slot's `HashJoin::addBlockToJoin(block, selector)` passes the *full* `key_columns` to its own `insertFromBlockImpl` | `ConcurrentHashJoin.cpp:309,351,711-757`; `HashJoin/HashJoin.cpp:643,806` |
| `unified_hash` | **`num_buckets + 1` (= 129 here)** | `scatterByBucket` builds one key getter for the routing pass, then `insertIntoBuckets` calls `insertFromBlockImpl` **once per bucket**, each with the same full `key_columns` | `UnifiedHashJoin/HashJoin.cpp:973-988`, `:117-152`; `UnifiedHashJoin/HashJoinMethodsImpl.h:91-122,166,356` |

`num_buckets = std::bit_ceil(max_threads) * BUCKETS_PER_THREAD` with `BUCKETS_PER_THREAD = 2`
(`UnifiedHashJoin/HashJoin.cpp:66-74`, `HashJoin.h:67`) → **128** at 64 threads; `parallel_hash`'s
`slots` is the thread count → **64**. Because `fillFixedBatch` sizes by `column->size()`, each of
those constructions packs the *entire* block (`AggregationCommon.h:65-66`), so the total packing work
per build block is `mult · (memset(16·R) + 2·R copies)`.

Predicted ratio unified : parallel = 129 : 64 = **2.02**. Observed, INNER|comp|hi|t64|large:
5039 : 2695 = **1.87**. Predicted parallel : hash = 64 : 1; observed 2695 : 130 = **20.7** — the
right direction and order, but not matched; I did not instrument call counts, and the three cells
differ in block counts and in probe-side scattering (the probe side is *not* re-scattered when the
two-level map is used, `ConcurrentHashJoin.cpp:474`, which is why the probe-side `const` constructor
stays small everywhere: 113 / 123 / 115 samples). Treat the multiplicity column as a code-reading
prediction corroborated by the unified:parallel ratio, not as a measurement.

## 6. C++-level difference, and what it says about the fix

There is **no** C++-level difference in the constructor between `need_offset=true` and
`need_offset=false`; the parameter only reaches `FindResult`. The whole cost is
`packFixedBatch`-per-construction, which is shared code, plus how often a construction happens.

Two independent inefficiencies, both algorithmic and neither one a codegen problem:

1. **Packing the whole block per shard/bucket.** `usePreparedKeys` pays `O(rows)` once per key
   getter, but a shard's getter only ever reads `prepared_keys[row]` for its own selector rows
   (`HashMethod.h:527-528`), so `mult - 1` copies of the array (and its `memset`) are wasted. Packing
   once per block and sharing it — or sizing/filling by the selector — removes a factor of 64/128
   on the build side.
2. **Two heap allocations per construction** for the `ColumnRawPtrs` and `Sizes` copies, also
   `mult` times per block (visible as `V2UnitL01`-bound 24 cycles *plus* the two dropped `_Znwm`
   calls).

Unified is hit ~2× harder than `parallel_hash` purely because it chose 2 buckets per thread where
`parallel_hash` uses 1 slot per thread; `hash` escapes because it has one map and one construction
per block.

---

## Appendix A — raw `count` output

```
ctor_true  full-path                     insns= 107 loads=  9 stores= 21 branches= 18 calls= 9 spill(st/ld)=6/5 dep_load_depth=2  [nops excluded 3]
ctor_false full-path                     insns= 107 loads=  9 stores= 21 branches= 18 calls= 9 spill(st/ld)=6/5 dep_load_depth=2  [nops excluded 3]
ctor_true  usePreparedKeys loop          insns=   9 loads=  1 stores=  0 branches=  2 calls= 0 spill(st/ld)=0/0 dep_load_depth=1  [nops excluded 0]
ctor_false usePreparedKeys loop          insns=   9 loads=  1 stores=  0 branches=  2 calls= 0 spill(st/ld)=0/0 dep_load_depth=1  [nops excluded 0]
fillFixedBatch per-row loop              insns=   4 loads=  1 stores=  1 branches=  1 calls= 0 spill(st/ld)=0/0 dep_load_depth=1  [nops excluded 0]
fillFixedBatch full-path                 insns=  74 loads= 15 stores= 10 branches= 17 calls= 3 spill(st/ld)=6/0 dep_load_depth=4  [nops excluded 4]
```

## Appendix B — raw `mca` summaries

```
== ctor_true  (0x13ab12a0:0x13ab1454)
iterations=100 instructions=10200 total_cycles=2406 total_uops=13700 ipc=4.24
block_rthroughput=22.8 upc=5.69 cycles_per_iteration=24.06
NOTE branch at 0x13ab12d8 -> L8 outside range, retargeted to block end
NOTE branch at 0x13ab132c -> L9 outside range, retargeted to block end
NOTE branch at 0x13ab1454 -> L7 outside range, retargeted to block end
NOTE TOTAL 9 call(s) dropped -> LOWER BOUND
NOTE 1 hand fixup: "adrp x10, 0x26e4000 .Lblockend" -> "adrp x10, 0x26e4000"

== ctor_false (0x16c4c6e0:0x16c4c894)
iterations=100 instructions=10200 total_cycles=2406 total_uops=13700 ipc=4.24
block_rthroughput=22.8 upc=5.69 cycles_per_iteration=24.06
NOTE branch at 0x16c4c718 -> L8 outside range, retargeted to block end
NOTE branch at 0x16c4c76c -> L9 outside range, retargeted to block end
NOTE branch at 0x16c4c894 -> L7 outside range, retargeted to block end
NOTE TOTAL 9 call(s) dropped -> LOWER BOUND
NOTE 1 hand fixup, as above

== ctor usePreparedKeys loop (0x13ab13a0:0x13ab13c0)
instructions=1000 total_cycles=210 cycles_per_iteration=2.1 ipc=4.76
block_rthroughput=1.8 total_uops=1100
NOTE branch at 0x13ab13b8 targets L6 outside the range; retargeted to block end

== fillFixedBatch<UInt64,UInt128> per-row loop (0x0e3bbc80:0x0e3bbc8c)
instructions=500 total_cycles=156 cycles_per_iteration=1.56 ipc=3.21
block_rthroughput=1.3 total_uops=800
"No resource or data dependency bottlenecks discovered."
Instruction info:
 2  4  0.33  *      ldr x9, [x10], #8
 1  1  0.33         subs x24, x24, #1
 3  1  0.50     *   str x9, [x8], #16
 1  1  0.50         b.ne .L7
```

## Appendix C — artefacts

* `codegen/logs/K1_ctor_true.asm`, `codegen/logs/K1_ctor_false.asm`,
  `codegen/logs/K1_ffb_u64_into_u128.asm`
* `codegen/mca/K1_ctor_true.txt`, `K1_ctor_false.txt`, `K1_ctor_sizeloop.txt`, `K1_ffb_loop.txt`
* `tmp/ctor_true_fixed.s`, `tmp/ctor_false_fixed.s` (the assembled mca inputs, with the single
  `adrp` fixup)
* `tmp/nm_all.txt` (full `llvm-nm --print-size --defined-only` dump used for the ICF check)
