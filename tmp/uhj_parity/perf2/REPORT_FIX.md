# `unified_hash` fix set: what changed, and what it measured

Host: aarch64 **Neoverse-V2** (Graviton4), 96 CPUs. Build `build/reldeb` (RelWithDebInfo).
Binaries kept: `bin/clickhouse.bold` (HEAD `7de421147f1`, the before), `bin/clickhouse.bnew2`
(the after). Baselines untouched and proved untouched (§6). Nothing pushed, no PR.

Sweeps: `bold1` and `bnew2`, both `sweep.py --reps 7`, 144 cells, ~12 min each, fresh run
tag per run. Every cell measures its comparator in the **same** run as `unified_hash`, so
`A_old` vs `A_new` is a drift control: median **+0.00%** wall, **+0.01%** CPU.

Raw tables: `results/fix_ab_tables.txt` (per-cell, all four metrics),
`results/kscale_kfix_fit.txt`, `logs/g53_inertness_fix1.md`.

---

## 1. Headline

**The 64-thread composite-key build deficit is closed and reversed, and it is the one
thing in this set that moved a cell.** `unified_hash`'s build phase on `comp` keys at 64
threads goes from **+36.9% over `parallel_hash` to −55.8% under it**; whole-query CPU on the
same cells goes from **+7.5% to −29.2%**. That was the largest cost the analysis found and
the top item in its handoff.

**The one-thread deficit did not move, and the reason is worth more than the attempt.** The
analysis pointed task P3 at it: the probe's `Prober` spilling its routing state per row.
That spilling is real and is now gone — but it exists only on the **multi-bucket** handle,
and a one-thread join has one bucket, so its probe takes `Prober<true>`, whose emitted code
is **byte-identical before and after** (§4). The one-thread deficit was never reachable from
P3. It stands at **+6.1% probe / +4.6% wall** on used-flag kinds, essentially unchanged from
+6.2% / +4.6%.

**Nothing regressed where `unified_hash` was winning.** At 16 and 64 threads there is
**not one cell** where `unified_hash` is slower than its comparator on wall time, before or
after, and no cell moved more than +2.5% (the one that did, `RIGHT|u64|hi|t64`, is still
−45.8% under `parallel_hash`).

**Two of the seven tasks turned out not to need doing, and one needed undoing.** Task 4
(prefix sums) was already correct; task P4's sole-path half was already done by
`02a534167f1`; and the first version of the key-getter hoist perturbed the comparator's own
codegen, which the validity gate caught before any number was believed (§6).

---

## 2. The table: A / B_old / B_new

`A` is the cell's comparator (`hash` at 1 thread, `parallel_hash` at 16/64), measured in the
same run as `B`. Medians over 7 interleaved repetitions.

### 2.1 The cells that moved — `comp` key, build phase (ms of processor time)

| cell | A_old | A_new | B_old | B_new | B_new/B_old | B_old/A | **B_new/A** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `INNER\|comp\|hi\|t64` | 4313 | 4301 | 5903 | **1899** | **−67.8%** | +36.9% | **−55.8%** |
| `INNER\|comp\|lo\|t64` | 4284 | 4292 | 5905 | **1900** | **−67.8%** | +37.8% | **−55.7%** |
| `LEFT\|comp\|hi\|t64` | 4269 | 4287 | 5900 | **1915** | −67.5% | +38.2% | −55.3% |
| `RIGHT\|comp\|hi\|t64` | 4921 | 4853 | 6019 | **1940** | −67.8% | +22.3% | −60.0% |
| `FULL\|comp\|hi\|t64` | 4781 | 4811 | 6001 | **1985** | −66.9% | +25.5% | −58.7% |
| `LEFT-SEMI\|comp\|hi\|t64` | — | — | 6155 | **1738** | −71.8% | — | — |
| `INNER\|comp\|hi\|t16` | 1704 | 1700 | 2036 | **1137** | −44.2% | +19.5% | −33.1% |
| `LEFT\|comp\|hi\|t16` | 1692 | 1684 | 2050 | **1100** | −46.3% | +21.2% | −34.7% |
| `FULL\|comp\|hi\|t16` | 1861 | 1904 | 2133 | **1221** | −42.8% | +14.6% | −35.9% |

Wall time on the same cells:

| cell | A_old | A_new | B_old | B_new | B_new/B_old | B_old/A | **B_new/A** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `INNER\|comp\|hi\|t64` | 226 | 232 | 190 | **127** | −33.2% | −15.9% | **−45.3%** |
| `INNER\|comp\|lo\|t64` | 186 | 193 | 160 | **96** | −40.0% | −14.0% | −50.3% |
| `FULL\|comp\|hi\|t64` | 491 | 491 | 320 | **253** | −20.9% | −34.8% | −48.5% |
| `LEFT-ANTI\|comp\|hi\|t64` | — | — | 145 | **73** | −49.7% | — | — |
| `INNER\|comp\|hi\|t16` | 384 | 383 | 370 | **312** | −15.7% | −3.6% | −18.5% |
| `FULL\|comp\|hi\|t16` | 703 | 714 | 650 | **593** | −8.8% | −7.5% | −16.9% |

### 2.2 The controls that must not move — `u64` and `str`, build phase

`u64` goes through `HashMethodOneNumber` and `str` through `HashMethodString`; neither packs
anything, so neither should see any of the B22 win. They do not:

| group | B_new/B_old build | B_new/B_old wall | B_new/A wall (was) |
| --- | ---: | ---: | --- |
| `u64`, t64 | **+0.1%** | +0.0% | −43.1% (−43.2%) |
| `u64`, t16 | −0.7% | −0.7% | −7.9% (−7.3%) |
| `str`, t64 | −0.3% | −0.7% | −41.7% (−41.7%) |
| `str`, t16 | −1.2% | −1.1% | −5.9% (−6.0%) |

### 2.3 The cells the probe fixes were aimed at — 1 thread, used-flag kinds

| metric | B_new/B_old | B_old/A | **B_new/A** |
| --- | ---: | ---: | ---: |
| probe phase | **−0.1%** | +6.2% | **+6.1%** |
| wall | +0.0% | +4.6% | +4.6% |
| CPU | −0.1% | +4.2% | +4.4% |

Unchanged. §4 explains why, from the disassembly rather than from the timings.

### 2.4 Whole matrix

| group | wall B_new/B_old | CPU B_new/B_old | cells >+2% wall | cells slower than A |
| --- | ---: | ---: | ---: | ---: |
| t1 (72 cells) | +0.0% | +0.1% | 5 | 42 (was 42) |
| t16 (36 cells) | **−1.5%** | −1.1% | 0 | **0** |
| t64 (36 cells) | **−0.9%** | −0.7% | 1 | **0** |

The five t1 cells above +2% are all tiny: 16→17 ms, 46→47 ms. `wall_ms` is
`query_duration_ms`, an integer, so one tick of quantisation on a 16 ms cell is +6.2%. Their
CPU (microseconds, unquantised) moves +0.4% to +2.7%.

---

## 3. Task 1 (B22) — the strongest single result, and it inverts its own discriminator

`kscale.py` fits `ns_per_build_row = a + b·K` across `max_threads` in {4,8,16,32,64} at fixed
30M-row data, where `K` is the partition count. `b` is the per-partition cost. This is the
measurement that established B22, and it is the sharpest view of the fix:

| key | algo | `b` before (ns/row/partition) | `b` after | change |
| --- | --- | ---: | ---: | --- |
| `comp` | **`unified_hash`** | **1.1808** | **0.2080** | **−82%** |
| `comp` | `parallel_hash` (untouched) | 1.7459 | 1.7986 | +3% (drift) |
| `comp` | `hash` (untouched, K≡1) | 0.0000 | 0.0000 | — |
| `u64` | `unified_hash` (control, packs nothing) | 0.1577 | 0.1449 | −8% |
| `u64` | `parallel_hash` | 0.5746 | 0.6105 | +6% (drift) |

At 64 threads with composite keys, build cost per row: **202.5 → 71.99 ns**, against
`parallel_hash`'s 162.6 → 163.8. So `unified_hash` moves from **+24.6% over** `parallel_hash`
to **−56.1% under** it, on a measurement that is independent of the 144-cell sweep and fails
differently from it.

**The pre-registered discriminator now fails, and that is the result.** PREREG P1.2 registered
"the `comp` per-partition coefficient is at least 4x the `u64` one" as the signature of the
packing cost. It was **7.49** before. It is **1.43** now, and the scorer prints `NOT MET`. The
test that established the cost can no longer find it. The residual `b = 0.208` for `comp` is
now within a factor 1.4 of the `u64` arm, i.e. what is left is the ordinary per-partition
cost of having more sub-tables, not packing.

**Why it was there.** `HashMethodKeysFixed`'s constructor calls `packFixedBatch` when
`usePreparedKeys` holds, and `fillFixedBatch` sizes its output by `column->size()` — the whole
block, taking no notice of the selector. The build built one getter per bucket plus one for
the scatter pass, so at 64 threads (`K = 2·bit_ceil(64) = 128`) it packed the same block 129
times. It now builds one per block.

**Sharing is safe, checked getter by getter rather than assumed.** All state in
`HashMethodOneNumber`, `HashMethodOneNumberInRange`, `HashMethodString`,
`HashMethodFixedString`, `HashMethodKeysFixed` and `HashMethodHashed` is written in the
constructor, and `getKeyHolder` is `const`. `HashMethodBase`'s last-element cache is the one
mutable member and every join key getter instantiates it with `use_cache = false`, so nothing
writes it. **One getter does keep mutable per-call state and it is called out rather than
waved through:** `LowCardinalityKeyGetterForJoin` has per-dictionary probe caches — but only
`findKey` touches them and the build calls `emplaceKey`, so it is shared on the build path for
that reason and the probe path is not affected by this change. A getter with
`has_pre_computed_hashes` would keep per-call state; no join getter is one, and
`scatterByBucketTypeCase` already static_asserts it.

**A follow-up the first version needed.** Sharing was initially applied to every getter. The
grouped roll-up then showed one-thread `build_us` at **+2.6%** median, and while the A/A
control puts that inside the noise of *that* metric (sd 14–37% at one thread — see §7), the
design was wrong on its own terms: a shared getter has to live behind a pointer, and for a
getter that latches two column pointers there is nothing to share to pay for it.
`HashMethodKeysFixed` now declares `reads_whole_block_at_construction` and only such a getter
is shared. One-thread `build_us` went **+2.6% → +0.3%** on that change alone, which is
consistent with the indirection having been the cause, but the per-cell noise cannot carry
that claim and I am not making it — the change rests on the necessity argument.

---

## 4. Tasks 2 and 3 (P3, P4) — the routing state is gone; the one-thread deficit is not

**What changed.** `Prober::find` used to store the routed bucket and its prefix per row so
that a later `offsetInternal` could answer about them. The compiler cannot separate a
`size_t` field from the prefix array it is read out of, so those two stores forced `shift`,
`max_bucket` and `prefix` to be reloaded per row as well — the 2 spill stores / 9 spill
reloads and dependent-load depth 3→5 the analysis measured. `find` now stores nothing and a
caller that wants the offset asks for both at once (`findWithOffset`). `Prober::offsetInternal`
is **removed**, not kept as a slower alternative, so a caller that cannot use the fused form
fails to compile rather than silently paying for the old path.

**What it did to the emitted code.** The `withProber` lambda is outlined per specialisation,
so `Prober<true>` and `Prober<false>` can be compared directly (`FULL`, `u64` key,
`AddedColumns<true>`, contiguous selector):

| specialisation | insns | loads | stores | spill st/ld | dep-load depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Prober<true>` (sole) before | 378 | 80 | 31 | 25 / 21 | 8 |
| `Prober<true>` (sole) after | 378 | 80 | 31 | 25 / 21 | 8 |
| `Prober<false>` (routed) before | 398 | 87 | 36 | 27 / 23 | **11** |
| `Prober<false>` (routed) after | **388** | **84** | **32** | **26 / 21** | **8** |

Whole-symbol counts, not the isolated row loop, so they include the per-block setup — stated
because the analysis's numbers were for the isolated loop and these are not directly
comparable to them. The direction is unambiguous on the routed path and **exactly zero on the
sole path**. A second, smaller routed instantiation went the other way (207 → 220 insns, depth
5 → 7), so the codegen effect is real but not uniform across instantiations.

**Why the one-thread cells did not move.** `withProber` picks `Prober<true>` when the table
has one bucket, and a one-thread join has one bucket. So every one-thread probe row runs the
specialisation whose code is byte-identical before and after. There was nothing for P3 to fix
there. The analysis attributed the one-thread deficit partly to P3's spilling; that
attribution was wrong, and the disassembly above is what shows it. The deficit's remaining
candidates are the ones the analysis already listed as unmeasured: the B2/B3/B4 dispatch
outlining (the baseline inlines a 40-arm dispatch in 3456 instructions; unified outlines it
into 19 helpers), and the per-block scaffolding.

**Where the routed path did move, it moved a little.** 16/64-thread probe phase: −1.1% and
−0.6% median across all keys, −1.6% / −1.3% on `str`. Small, and in the right direction.

**P4, both halves.** The sole-path half was already done by `02a534167f1`, confirmed from the
disassembly above: at one bucket the offset is the flat form and there is no per-row sole
branch. The "offset not needed at all" half is now free rather than templated away — `find`
never touches `prefix`, so a join whose `needs_offset` is false never reads it. The one
remaining read is `bucketPrefix()` in the handle's constructor, once per probe block, which at
`max_block_size = 65409` is not worth a template parameter.

---

## 5. Task 4 (prefix sums) — nothing to do, and here is the audit

The brief asked for `unified_hash`'s call sites to be switched to `offsetInternalUnsafe`.
They already were, and the debug assertion it asked for already exists. The audit, path by
path:

| path | how it reaches the prefix sums | pays the `call_once`? |
| --- | --- | --- |
| probe, per row | `Prober` → `bucketPrefix()` at construction, then `prefix[bucket]` | no |
| non-joined scan, per cell | `offsetInternalAtBucket` → `offsetInternalUnsafe` → `offsetUnsafe` | no |
| direct-addressed maps (`key8`/`key16`, post-conversion ranges) | `FixedRangeStorage`, one flat buffer, no prefix at all | no |
| `parallel_hash`, `hash` | `TwoLevelHashTable::offsetInternal` → `prefix_sums.offset` | yes, unchanged |

Both unsafe entry points already `chassert(computed)`, and `bucketPrefix()` does too, so a
future path that forgets to freeze trips in debug rather than reading zeros. `freezeMapsForProbing`
runs at `onBuildPhaseFinish`, at the end of `runPostBuildPhase` (after the fixed-map conversion,
the reranging and the runtime filters), and in the `StorageJoin` takeover path
(`reuseJoinedData`), which covers every way a map can reach a probe. The only change made was
to write down in `BucketPrefixSums` which caller is which, so the next reader does not have to
re-derive it.

---

## 6. Validity: the comparators are untouched, and the first attempt was not

Full proof in `logs/g53_inertness_fix1.md`. Two shared headers were edited
(`Common/HashTable/TwoLevelHashTable.h`, `Common/ColumnsHashingImpl.h`, plus a
code-free constant in `Common/ColumnsHashing/HashMethod.h`), all three compiled into the
comparators and into `Aggregator`.

**The first version was not inert, and the gate caught it.** Routing the offset through a
shared `always_inline` helper on every path is semantically identical and still changed the
generated code of **14 baseline `DB::HashJoinMethods::joinRightColumns` instantiations** and
**10 `Aggregator` functions** — mostly making them *smaller*, which would have quietly eaten
part of the win being measured. Restructured so the fused path is a branch of its own and the
pre-existing path is textually untouched.

**After that:** zero baseline symbols resized, zero removed;
`DB::HashJoin::addBlockToJoin`, `DB::ConcurrentHashJoin::addBlockToJoin` and
`fillFixedBatch` opcode-identical; and of the **16,680** baseline `joinRightColumns`
instantiations, none changed size and a random sample of 24 is 24/24 opcode-identical. The
200 added symbols outside the allowed region are all `shared_ptr` control blocks and
destructors for the type-erased getter plus one `std::list<StoredBlock>::push_back` — new
code, which cannot perturb an existing function.

`git diff 0945a745399 -- src/Interpreters/HashJoin/ src/Interpreters/ConcurrentHashJoin.{h,cpp}`
is empty.

**A third defect in `symdiff.py`, in the false-alarm direction.** Its normaliser recognised
the low half of an `adrp` address pair only as `add xN, xN, #lo`, not as a load through the
register (`ldr xM, [xN, #lo]`), which is how a GOT entry is read. That reported
`DB::HashJoin::addBlockToJoin` as 35 differing instructions in a function of identical length
whose source had not changed. Fixed, and re-validated against the A-K1 ablation — whose entire
footprint is one instruction — so the fix has not blinded it. Linker range-extension thunks are
now counted separately; they have no source and any edit that moves code churns hundreds.

---

## 7. What the instrument can and cannot see

The sweep runs an A/A control: the same binary under two labels, interleaved exactly like a
real A/B. It is the right yardstick for the small numbers above.

| cell | metric | A/A sd | A/A delta |
| --- | --- | ---: | ---: |
| `INNER\|u64\|hi\|t1\|medium` | wall | 2.7–7.4% | +1.8% |
| | **build** | **14–37%** | +2.3% |
| | probe | 1.2–6.0% | +1.3% |
| `INNER\|u64\|hi\|t64\|large` | wall | 1.2–1.9% | −1.5% |
| | build | 2.6–4.1% | +0.9% |
| | probe | 0.5–1.0% | −0.3% |

So: the `comp` build results (−43% to −72%) and the `kscale` coefficient (−82%) are orders of
magnitude outside the noise. The one-thread numbers, and the ±1% moves on `u64`/`str`, are
**not** resolvable by this harness — which is why §3 does not claim the sharing gate as a
measured win and §9 lists the two tasks whose benefit was not separably measured.

---

## 8. Correctness

All on the final binary `bin/clickhouse.bnew2`:

- `bash tests/queries/0_stateless/04658_unified_hash_join_equivalence.sh` → matches `.reference`
- `UHJ_PORT=9111 bash tmp/uhj_parity/run_04659.sh` → `OK`
- 80 existing `tests/queries/0_stateless/*join*.sql` run under `--join_algorithm=hash` and
  `--join_algorithm=unified_hash`: **0 mismatches**. Added because 04658/04659 do not cover
  ASOF, `LowCardinality`, OR-clauses or the non-joined scan, all of which the shared key
  getter and the `Prober` change reach.

---

## 9. What was not achieved, plainly

1. **The one-thread deficit is untouched** (+4.6% wall on used-flag kinds, before and after).
   P3 could not have touched it and now demonstrably did not — §4. The remaining candidates are
   the B2/B3/B4 dispatch outlining and the per-block scaffolding, neither measured. Closing it
   needs an isolated per-phase counter run, which is item 2 of the previous handoff and is
   still open.
2. **Tasks 5 (B16/A3) and 6 (L3) have no separable measured benefit.** Both are granularity
   fixes with *measured granularity* — 32 ns of a 2.90 µs critical section, and `K·B·C` versus
   `B·C` round trips of one contended cache line — but the build-phase timings at 16 and 64
   threads on `u64` and `str` move by less than the A/A noise floor, so I cannot attribute any
   part of the improvement to them. They are committed on the necessity argument, and each
   commit message says so rather than claiming a number.
3. **The routed probe codegen improvement is not uniform.** One instantiation improved on
   every count, another grew. Whole-symbol counts, not the isolated loop; a proper isolation
   (back-edge analysis per instantiation, as `codegen/P1_G2_probe_and_gather.md` did by hand)
   was not redone.
4. **`recomputeBucketBytes` timing was audited, not stress-tested.** The byte total is
   published once per block rather than once per bucket; every reader I found
   (`getTotalByteCount`, `getTotalByteCountUnlocked` from the size-limit check) runs after the
   block is fully in. A reader added between the buckets of one block would now see a staler
   number, and nothing enforces that.
5. **The non-joined-scan coverage gap the previous mission recorded is still there.** The
   match-rate knob controls matching *probe* rows, not unmatched *build* keys, so N1–N7 remain
   under-exercised. `nonjoined_us` moves by ≤1.2% everywhere here, which is consistent with
   "nothing changed" and also with "not exercised".

---

## 10. Commits

| commit | task | what |
| --- | --- | --- |
| `c620756a1ff` | tooling | `symdiff.py` normaliser: the `ldr` form of an `adrp` pair; thunks counted separately |
| `8c76032670e` | 6 / L3 | hold `blocks_mutex` only for what another build thread can see |
| `455bbc0574d` | 5 / B16+A3 | publish the bucket byte delta once per block, not once per bucket |
| `e2932a8cc69` | 1 / B22 | one key getter per block instead of one per bucket |
| `7f034e24ec9` | 2,3,4 / P3+P4 | report a matched cell's offset from the lookup, not after it |
| `fba9bf00c89` | 1 / B22 | share a block's key getter only when construction reads the block |

Task 7 (confirm P1) needed no code and is reported in §4.

---

## 11. Amendment: where the residual losses actually are

Added after §1-§10, from a loss analysis of the `bnew2` sweep (`losses.py`). It corrects one
number in §2.3 and one inherited framing.

### 11.1 A harness limitation that makes §2.3's probe number wrong

**`hash` has no `NonJoinedBlocksTransform`.** It emits a RIGHT/FULL join's non-joined rows
inside its single `JoiningTransform`. Confirmed against the pipeline: the processor is present
in **0 of 24** `hash` RIGHT/FULL cells at one thread, and in **12 of 12** `parallel_hash`
cells at both 16 and 64 threads, and in all 24 unified cells.

So `harness.py`'s `probe` and `nonjoined` columns are **not comparable between `hash` and
`unified_hash`**, and §2.3's "+6.1% probe" was comparing unified's probe against `hash`'s
probe *plus* its non-joined emission. That flatters unified on exactly the kinds that have a
non-joined scan: on `FULL|u64|hi|t1|medium` unified's probe reads **−4.7%** while its
`nonjoined` column has no comparator at all and its wall is **+8.5%**.

The apples-to-apples quantity is `probe + nonjoined`. Corrected:

| metric, 1 thread, used-flag kinds | B_old/A | B_new/A |
| --- | ---: | ---: |
| probe + non-joined (corrected) | +7.5% | **+7.2%** |
| probe alone (§2.3, not comparable) | +6.2% | +6.1% |

The conclusion of §4 is unchanged - the deficit did not move - but its size is 7.2%, not 6.1%.
`parallel_hash` does have the transform, so nothing at 16 or 64 threads is affected.

### 11.2 The loss count, by phase

Comparable cells only (`LEFT SEMI`/`LEFT ANTI` above one thread have no comparator). A cell
"loses" when it is more than 2% above its comparator:

| phase | t1 | t16 | t64 |
| --- | ---: | ---: | ---: |
| **probe + non-joined** | **70 / 72** | 1 / 24 | 1 / 24 |
| build | 20 / 72 (median +0.9%, inside a 14-37% A/A sd - not signal) | 1 / 24 | 0 / 24 |
| wall | 61 / 72 | **0 / 24** | **0 / 24** |

So: **the residual loss is one regime and one phase — the probe at one thread.** The build
side is settled at every thread count, and above one thread there is nothing left.

### 11.3 Its structure: a fixed part plus a proportional part

Per **probe row** rather than as a percentage, which is what separates the two:

| group | A ns/row | B ns/row | delta ns | delta % |
| --- | ---: | ---: | ---: | ---: |
| `LEFT ANTI` | 12.6 | 13.5 | +1.15 | **+10.9%** |
| `LEFT SEMI` | 12.3 | 13.8 | +0.95 | +8.6% |
| `INNER` | 26.6 | 27.9 | +1.18 | +4.5% |
| `LEFT` | 31.6 | 32.9 | +1.53 | +4.9% |
| `RIGHT` | 39.0 | 41.9 | +2.27 | +5.8% |
| `FULL` | 41.7 | 44.4 | +2.14 | +5.5% |
| small (10 k keys, L2-resident) | 17.3 | 18.3 | +0.95 | +6.8% |
| medium (500 k keys) | 45.3 | 48.3 | +2.44 | +6.3% |
| `u64` / `str` / `comp` | 18.2 / 37.6 / 24.5 | 19.4 / 39.2 / 25.6 | +1.31 / +1.57 / +1.42 | +8.4 / +5.5 / +6.4% |

Least squares over all 72 one-thread cells:

```
delta_ns = 0.48 + 0.0422 * baseline_ns          R^2 = 0.645
```

**A fixed 0.48 ns per probe row — about 1.5 cycles — plus 4.2% of whatever the row already
costs.** Those two numbers account for the whole 4.5%-10.9% spread:

- the **proportional 4.2%** is the same effect the previous mission measured with hardware
  counters as **+6.7% instructions/row at −1.3% IPC** on the whole query (F8), and it is why
  the percentage barely moves between cardinalities (6.8% vs 6.3%) or key types. It is not
  memory: a 10 k-key table is L2-resident and a 500 k-key table is not, and the percentage is
  the same in both;
- the **fixed 0.48 ns** is why the percentage is worst on `LEFT SEMI`/`LEFT ANTI`. Those are
  `MapsOne` kinds that emit at most one row per left row and never walk a `RowRefList`, so
  their baseline per-row cost is the smallest in the matrix (12.3 ns) and a constant is
  3.9% of it. On the dearest cells it is 1.1%.

`RIGHT`/`FULL` carry the largest *absolute* overhead (+2.1 to +2.3 ns/row), about +1 ns/row
more than `INNER` on the same data. They are the kinds that also run the non-joined scan and
maintain per-offset used flags, and the non-joined scan is the one loop group the previous
mission left UNSETTLED (N1/N3/N4/N7: unified walks buckets through
`beginOfNextNonEmptyBucket`, 4 calls, where the baseline uses `const_iterator::operator++`,
1 call). That is now the best-supported lead for the `RIGHT`/`FULL` half of the residual, and
it is a different lead from the proportional 4.2%.

### 11.4 What this means for the next attempt

The one-thread deficit is **not** one thing:

1. **~4.2% proportional**, present on every kind, key and cardinality, and not memory-bound.
   Needs per-phase instruction attribution, not more timing. The `Prober` is not it (§4).
2. **~0.5 ns/row fixed**, visible only where per-row work is small (`SEMI`/`ANTI`). A
   per-row constant on the cheapest path.
3. **~1 ns/row extra on `RIGHT`/`FULL` specifically**, coincident with the non-joined scan and
   the per-offset flags. Separable by a harness extension that varies the *unmatched build
   key* fraction, which is exactly the coverage gap recorded as assumption 2 and still open.

Fixing the harness so `hash`'s non-joined work is attributable (§11.1) is a prerequisite for
3, and is cheap: the phase map needs a `hash`-specific entry, or the comparison needs to use
`probe + nonjoined` throughout, as this section does.

### 11.5 The multi-threaded side: no losses, and one coverage hole

Of the 48 cells at 16 and 64 threads that have a comparator, **none loses on wall time or on
CPU**. The margins:

| metric | best | p25 | median | p75 | worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| wall | −52.0% | −45.0% | **−27.0%** | −7.9% | **−2.0%** |
| CPU | −33.1% | −15.6% | −12.1% | −6.2% | **−0.3%** |
| build | −60.0% | −33.4% | −20.5% | −4.2% | +2.1% |
| probe + non-joined | −12.8% | −9.7% | −7.1% | −3.1% | +4.9% |

**Three cells of 48 exceed +2% on any phase, and all three still win the whole query:**

| cell | wall | CPU | build | probe+nj |
| --- | ---: | ---: | ---: | ---: |
| `INNER\|str\|lo\|t16\|large` | −2.0% | −0.3% | −1.3% | **+4.9%** |
| `INNER\|str\|lo\|t64\|large` | −39.2% | −7.3% | −15.0% | **+3.4%** |
| `INNER\|str\|hi\|t16\|large` | −6.5% | −6.2% | **+2.1%** | −8.3% |

**Why those two probe cells, and why `lo`.** `unified_hash`'s probe advantage over
`parallel_hash` is concentrated in the *matched-row* path, and splitting the 48 cells by match
rate shows it directly:

| | A ns/row | delta ns/row | delta % |
| --- | ---: | ---: | ---: |
| match `hi` (90% of probe rows match) | 198.6 | **−19.07** | **−9.7%** |
| match `lo` (10% match) | 129.4 | −4.83 | −3.3% |

`parallel_hash` probes one merged 256-bucket map and recovers a matched cell's bucket by
re-hashing it, plus the `std::call_once` check in `BucketPrefixSums::offset`, once per matched
row; `unified_hash` reads a precomputed prefix (§5). On a **miss** there is no offset to
produce, so that advantage is simply absent while the fixed per-row overhead of §11.3 remains.
`INNER` with `match=lo` is the leanest loop in the multi-threaded matrix - 90% of rows miss,
and an `INNER` miss emits nothing at all - so it is the one place the overhead shows through.
It is the same mechanism as `LEFT SEMI`/`LEFT ANTI` at one thread: smallest denominator.

**Why the build cells are all `str` at 16 threads.** `HashMethodString` packs nothing, so B22
gave it nothing, and what remains of `unified_hash`'s build advantage there is the locking
difference, which is smaller at 16 threads than at 64 (`str` build vs `parallel_hash`: −1.4% at
t16 against −19.5% at t64). These are cells where the two implementations' builds are near
parity to begin with, so ±2% is where they sit.

**The real multi-threaded gap is coverage, not loss.** The other **24 of 48** cells at 16/64 -
every `LEFT SEMI` and `LEFT ANTI` cell - have **no comparator at all**:
`allowParallelHashJoin()` is false for SEMI/ANTI, so `parallel_hash` cannot run them. For a
quarter of the multi-threaded matrix `unified_hash` is the only parallel implementation, so
there is nothing to lose to - and equally nothing is measured. The alternative a user actually
gets for those kinds today is `hash`, with a serial build, and that comparison has never been
run at 16 or 64 threads. It is a one-setting harness change (`comparator_for` returning `hash`
where `parallel_hash` is unavailable) and it is the cheapest open item in this report.

### 11.6 Correction: the fixed per-row cost is the two-level layer, and `unified_hash` pays a third of what the baseline's own two-level map pays

§11.3 fitted a fixed 0.48 ns/row against `hash` at one thread and §11.5 carried it forward as
though it were a `unified_hash` property. That was wrong, and the premise behind it is false:
**`hash` and `unified_hash` do not use the same table.**

```
$ rg -n 'use_two_level_maps' src/Interpreters/HashJoin/ src/Interpreters/ConcurrentHashJoin.cpp
ConcurrentHashJoin.cpp:230:   /*use_two_level_maps*/ true
HashJoin/HashJoin.h:115:      bool use_two_level_maps_ = false,
```

| implementation | map |
| --- | --- |
| `hash` (the one-thread comparator) | **single-level** `HashMapTable` |
| `parallel_hash` (the 16/64 comparator) | two-level, **fixed 256** buckets |
| `unified_hash` | two-level, **runtime** bucket count (1 at one thread) |

So the one-thread deficit is a partitioned map measured against an unpartitioned one, and the
fixed term is a candidate cost of *being two-level at all* - which `parallel_hash` also pays.

**Settings-only experiment, no rebuild.** `parallel_hash` at `max_threads=1` is the baseline's
own code with a two-level map and one slot; `hash` at `max_threads=1` is the same code with a
single-level map. The difference between them is the two-level layer, measured with
`unified_hash` out of the picture entirely:

| cell | `hash` | `parallel_hash` | `unified_hash` | par/hash | uni/hash | uni/par |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `INNER\|u64\|hi\|t1\|small` | 19.1 | 21.3 | 19.7 | **+11.7%** | +3.3% | −7.6% |
| `INNER\|u64\|hi\|t1\|medium` | 35.8 | 39.9 | 37.1 | **+11.4%** | +3.6% | −7.0% |
| `INNER\|str\|hi\|t1\|small` | 30.8 | 34.7 | 31.7 | +12.7% | +2.8% | −8.8% |
| `INNER\|str\|hi\|t1\|medium` | 94.9 | 113.9 | 97.8 | **+20.0%** | +3.0% | −14.1% |
| `FULL\|u64\|hi\|t1\|small` | 22.4 | 23.9 | 23.4 | +6.7% | +4.6% | −2.0% |
| `FULL\|u64\|hi\|t1\|medium` | 46.3 | 56.6 | 50.2 | +22.0% | +8.3% | −11.3% |
| `FULL\|str\|hi\|t1\|small` | 36.5 | 39.1 | 37.9 | +7.4% | +3.9% | −3.3% |
| `FULL\|str\|hi\|t1\|medium` | 125.8 | 136.2 | 130.1 | +8.2% | +3.4% | −4.5% |
| **median** | | | | **+11.6%** | **+3.5%** | **−7.3%** |

ns per probe row, `probe + nonjoined`, 7 interleaved reps, algorithm order rotated. The four
`INNER` rows carry the conclusion on their own and are free of any non-joined-scan asymmetry,
since `INNER` has no non-joined scan.

**So the per-row overhead is the two-level layer, and `unified_hash` pays about a third of what
the baseline's own two-level implementation pays for it.** `parallel_hash` at one slot still has
256 buckets: every probe row routes, lands in one of 256 sub-tables rather than one, and a
matched row's offset costs a re-hash of the cell plus the `call_once` check.
`unified_hash` at one bucket takes `Prober<true>` - `sole->find`, the flat `offsetInternal`, no
routing at all - which is why it is +3.5% and not +11.6%.

This also explains §11.5's match-rate split from the other side: unified's advantage over
`parallel_hash` is concentrated in matched rows because that is where `parallel_hash`'s
two-level offset path is dear, so on a miss-dominated loop the advantage shrinks to nearly
nothing and the residual layer cost shows.

**What it changes about the handoff.** The one-thread deficit is not a `Prober` problem and not
a micro-optimisation problem - §4 already showed the sole-path code is byte-identical before and
after this series, and §11.3's proportional 4.2% is unexplained. The structural lever is to stop
being two-level when there is one bucket: let `num_buckets == 1` select a single-level storage
rather than a one-bucket `RuntimeStorage`. The measured headroom is the **+3.5%** above, i.e.
most of the one-thread probe deficit on these cells, and it is a bigger lever than anything in
this fix list. It is also a real design question rather than a free win, because
`StorageJoin` hands one `RightTableData` to joins built with different `max_threads`, so the
storage kind cannot simply follow the thread count of whoever probes.

Raw: `/tmp/twolevel_probe.py` in the session; re-runnable against the harness with
`H.Cell(kind, key, match, 1, card)` and `settings_for(cell, "parallel_hash")`.

### 11.7 How strongly can "`unified_hash` beats `parallel_hash`" be stated?

**Inside the measured space the claim is stronger than "almost always" - it is every cell.**
Of the 48 cells at 16 and 64 threads that have a `parallel_hash` comparator:

| margin | wall | CPU |
| --- | ---: | ---: |
| unified faster by >20% | 24 / 48 | 8 / 48 |
| faster by 10-20% | 8 / 48 | 22 / 48 |
| faster by 5-10% | 14 / 48 | 12 / 48 |
| faster by 3-5% | 1 / 48 | 5 / 48 |
| within ±3% (parity) | 1 / 48 | 1 / 48 |
| **slower by >3%** | **0 / 48** | **0 / 48** |

median −27.0% wall / −12.1% CPU; worst cell −2.0% / −0.3% (`INNER|str|lo|t16|large`, the same
cell as §11.5); best −52.0% / −33.1%.

**Peak memory, never analysed before this section, points the same way:**

| | median | best | worst | cells >+10% |
| --- | ---: | ---: | ---: | ---: |
| t16 | **−15.0%** | −26.8% | −7.6% | 0 / 24 |
| t64 | **−23.6%** | −48.4% | −7.2% | 0 / 24 |
| t1 (vs `hash`) | +0.1% | −4.7% | +39.5% | 10 / 72 |

The t1 outliers are all `small` cells whose absolute peak is 0.01 GiB, so the percentage is on a
10 MB base. At 16 and 64 threads there is no cell where `unified_hash` uses more memory.

And at **one thread** with `parallel_hash` forced (§11.6, 8 cells x 7 reps): 115 ms vs 118 ms wall,
115.1 vs 118.3 ms CPU, −7.3% on probe+non-joined. So at every thread count measured
`unified_hash` is at least as fast, never slower.

**What stops it being a blanket claim.** Five gaps, in the order that matters:

1. **A third of the multi-threaded matrix has no comparison at all.** `allowParallelHashJoin()`
   is false for SEMI/ANTI, so 24 of the 72 cells at 16/64 have no `parallel_hash` to be faster
   than. Practically that favours `unified_hash` - it is the only parallel option there - but it
   is a capability statement, not a speed one.
2. **Only `large` cardinality is measured above one thread.** `THREAD_CARDS` pairs one thread
   with small/medium and 16/64 with large only, so a 16-thread join over a 20k-row build side -
   a common shape - is entirely unmeasured. This is the cheapest gap to close and the one most
   likely to contain a surprise, because the fixed per-row costs of §11.3/§11.6 matter most when
   the table is cache-resident.
3. **Uniform keys, no skew.** The bucket count `K` is load-bearing for contention (ablation
   A-K1), and skew is exactly where a bucket layout differs from a slot layout.
4. **Whole feature families are unexercised**: residual/additional filters (loop family P9,
   UNSETTLED), `LowCardinality` keys, the direct-addressed `key8`/`key16` maps, long
   `RowRefList` chains (the matrix has 2 rows per key), ASOF, multi-disjunct OR joins, nullable
   keys, `StorageJoin` reuse, and the spill/external paths.
5. **One microarchitecture** (aarch64 Neoverse-V2), one host, single-tenant.

**A sentence that is defensible as written:** *on a 30M-row build side with uniform keys at 16
and 64 threads, `unified_hash` was faster than `parallel_hash` in all 48 measured cells - median
27% less wall time, 12% less CPU and 24% less peak memory at 64 threads - and slower in none.*

That generalises with some confidence because the mechanism is measured, not inferred:
`parallel_hash` fails a `try_lock` 309 times per successful acquisition against
`unified_hash`'s 0.70 and spends 26% more time inside critical sections; it merges every slot
into slot 0 at build finish where `unified_hash` shares one map from the start; its scatter
copies columns where `unified_hash` routes row indices; and its probe pays a re-hash plus a
`call_once` per matched row for an offset that `unified_hash` reads from a precomputed prefix
(§11.6: +11.6% over a single-level map against `unified_hash`'s +3.5%). None of those depend on
the key type or the data distribution.

---

## 12. The shape sweep: the "faster in every cell" claim is FALSE

§11.7 named the untested region - a small or large build side at many threads, since
`THREAD_CARDS` pins the many-threaded points to one build size - and said it was the likeliest
place to hold a surprise. It does. **`unified_hash` is up to 1.9x SLOWER than `parallel_hash`.**

`shapes.py`, 240 cells, `INNER JOIN`, 5 interleaved repetitions with the algorithm order
rotated, one build and one probe table read as range scans so a size is a prefix rather than a
different table, match rate held at 50% for every size and shape:

| axis | values |
| --- | --- |
| build rows | 2^17 (131 072) .. 2^26 (67.1M), powers of two - 10 points |
| probe rows | 1x, 2x, 4x the build rows |
| shape | `narrow` (nothing gathered), `rpay` (4 UInt64 + String gathered from the build side), `lpay` (same carried from the probe side), `uniq` (unique build keys, so all three promote ALL to RightAny and take `MapsOne`) |
| threads | 16, 64 |

**Answers agree in all 240 cells**, so the timings compare the same work.

### 12.1 Result

| group | n | wall median | wall worst | CPU median | cells >2% slower |
| --- | ---: | ---: | ---: | ---: | ---: |
| ALL | 240 | −13.7% | **+93.6%** | −5.8% | **27 / 240** |
| threads=16 | 120 | −2.2% | **+93.6%** | −2.7% | **27 / 120** |
| threads=64 | 120 | −20.9% | −7.8% | −9.7% | **0 / 120** |

By build size, at both thread counts pooled:

| build rows | wall median | wall worst | cells >2% slower |
| --- | ---: | ---: | ---: |
| 131 072 | −19.1% | −5.9% | 0 / 24 |
| 262 144 | −14.7% | +0.0% | 0 / 24 |
| 1 048 576 | −12.2% | +0.0% | 0 / 24 |
| 4 194 304 | −5.7% | +4.8% | 5 / 24 |
| 16 777 216 | −10.1% | +3.9% | 3 / 24 |
| 33 554 432 | −12.6% | +37.7% | 3 / 24 |
| **67 108 864** | **−3.6%** | **+93.6%** | **12 / 24** |

So the small build side at many threads - the shape §11.7 flagged - is **fine** (0 losses below
1M rows). The failure is the opposite corner: **a large build side at a low-to-middling thread
count.** Worst cells, all at 16 threads: `uniq|b67108864|x4` **+93.6%**, `uniq|b67108864|x2`
+86.9%, `uniq|b67108864|x1` +48.7%, `narrow|b67108864|x4` +40.3%.

`uniq` is the worst shape (12 of 60) and that is a clue rather than a quirk: at the same build
*rows* it has twice the distinct *keys* of the two-rows-per-key shapes, so it is really the
largest key count in the matrix.

### 12.2 It is the probe, and it is a function of the bucket count `K`

Phase split of the worst cell (`uniq|b67108864|x4|t16`, ms of processor time):

| | wall | build | probe | peak mem |
| --- | ---: | ---: | ---: | ---: |
| `parallel_hash` | 708 | 1903 | 4556 | 2.61 GiB |
| `unified_hash` | 1371 | 2828 | **14466** | 2.61 GiB |
| delta | **+93.6%** | +48.6% | **+217.5%** | −0.3% |

The probe dominates. A CPU profile confirms it is the probe row loop itself and not
scaffolding: `joinRightColumns` holds **15 359** leaf samples for `unified_hash` against
**4 683** for `parallel_hash`, a factor of **3.3**, and the two instantiations are structurally
identical (`JoinKind::Inner`, `RightAny`, `MapsTemplate<RowRefList>`, `HashMethodOneNumber`,
`AddedColumns<true>`). The only visible difference in the two symbols is the grower -
`TwoLevelHashTableGrower<8ul>` against unified's single-level one - which affects growth, not
lookup.

**The penalty is set by `K`, not by the thread count.** `K = 2 * bit_ceil(threads)` is a step
function, so different thread counts share a `K` - and cells that share a `K` show the same
penalty (67.1M distinct keys, probe 1x, 7 reps, order rotated):

| threads | K | wall (uni vs par) | **probe (uni vs par)** |
| ---: | ---: | ---: | ---: |
| 8 | 16 | +16.7% | +5.6% |
| 16 | 32 | **+51.7%** | **+123.5%** |
| 24 | **64** | +23.9% | **+39.0%** |
| 32 | **64** | +25.4% | **+40.6%** |
| 48 | **128** | −27.2% | **−2.7%** |
| 64 | **128** | −35.8% | **−3.3%** |

K=64 at 24 and at 32 threads: +39.0% and +40.6%. K=128 at 48 and at 64 threads: −2.7% and
−3.3%. `parallel_hash`'s own probe cost is flat across the whole scan (1057 -> 1510 ms), so this
is not a thread-count effect that both would share.

Two things follow. First, **the sweep's "all losses at 16 threads" is an artefact of measuring
only 16 and 64**: 8, 24 and 32 threads lose on this size too, and the crossover for 67.1M keys
sits between K=64 and K=128, i.e. between 32 and 48 threads. Second, the relation is
**non-monotone** - K=16 is nearly free (+5.6%), K=32 is catastrophic (+123%), K=64 is bad
(+40%), K=128 is a small win. Neither "bigger sub-tables are worse" nor "more sub-tables are
worse" predicts that shape.

### 12.3 The mechanism is NOT identified, and this reproduces a known unchased lead

I can say where it is (the probe row loop), what selects it (`K`, at large key counts) and that
it is not the build, not memory (peak is within 0.3%) and not a different instantiation. I
cannot say why K=32 is three times worse than K=128 at the same total capacity - all four K
values give a clean power-of-two capacity per bucket, the routing bits (top of the CRC32) and
the placement bits (low bits) do not overlap at any K, and the load factors are equal.

This is the same bimodality the previous mission recorded in `WORKLOG.md` F7 and explicitly did
not chase:

> **LEAD, not used for acceptance:** the ablation's effect on `unified_hash` is non-monotone in
> thread count (+36%, −22%, +19%, −15%, +90% for `u64` at 4/8/16/32/64) while the controls are
> flat. Something bimodal is happening in the bucket layout that neither the packing story nor
> the contention story predicts on its own. Not chased.

That lead was seen on the build phase of a 30M-row join and looked like a curiosity. On a
67M-key build side it is a **1.9x wall-time regression on the probe**, and it is now the largest
open defect in `unified_hash` - considerably larger than anything the fix set in §1-§10
addressed.

### 12.4 What can honestly be claimed now

- **True, and now well supported across 10 build sizes, 3 probe multiplicities and 4 payload
  shapes:** at 64 threads `unified_hash` beats `parallie_hash` in **120 of 120** cells, median
  −20.9% wall, worst −7.8%.
- **True:** below about 1M build rows `unified_hash` wins at every thread count measured.
- **FALSE:** "faster in every cell". At 67.1M rows and 16 threads it is up to **+93.6%** slower,
  and the losing region extends over 8-32 threads.
- The claim in §11.7 must therefore be narrowed to the thread count as well as the build size:
  *at 64 threads, across build sides from 131 k to 67 M rows, probe multiplicities of 1-4x and
  four payload shapes, `unified_hash` was faster than `parallel_hash` in all 120 measured cells.*

Also fixed while doing this: `sweep.py` loses 0.6-0.7% of its runs (11 of 1848 in `bold1`, 13 of
1848 in `bnew2`) because `SYSTEM FLUSH LOGS` races the `query_log` write for the query that just
finished - always the last query of a cell. Symmetric between runs and harmless to a median over
7 repetitions, but `shapes.py` retries the readback instead and loses none.

Raw: `results/shapes.jsonl`, `results/shapes_s1_report.txt`, `logs/shapes_s1.log`.

### 12.5 A 15-second reproduction

`tmp/uhj_parity/perf2/repro_k32.sh` - self-contained `clickhouse local`, no server and no
tables, so it runs against any build:

```
bash tmp/uhj_parity/perf2/repro_k32.sh [path/to/clickhouse] [reps]
```

Default shape: 67.1M distinct build keys, 268M probe rows (4x) at a 50% match rate,
`max_threads=16` so `unified_hash` gets K = 32 buckets. It checks that both algorithms return
the same answer before timing anything, interleaves the repetitions with the order rotated, and
prints the median wall time of each.

```
median wall  parallel_hash      886 ms   unified_hash     1569 ms   unified is +77.1%
REGRESSION REPRODUCED
```

Two consecutive runs on the same binary: +77.1% and +76.9%. Total runtime 15 s.

**The control matters as much as the repro.** `THREADS=64 bash repro_k32.sh ...` gives K = 128
and prints `-16.6%`, i.e. `not reproduced` - so a change that merely makes everything slower
cannot be mistaken for a fix. `THREADS` is the knob that selects K:

| `THREADS` | K | expected |
| ---: | ---: | --- |
| 8 | 16 | roughly parity |
| **16** | **32** | **worst, about +77%** |
| 24 or 32 | 64 | about +25% wall |
| 48 or 64 | 128 | `unified_hash` wins by 17-36% |

`KEYS` and `MULT` are also settable. Note that the effect needs the full key count: holding
keys-per-bucket at 2.1M and shrinking the data does **not** reproduce it (§12.2), so a smaller
input is not a faster repro - `MULT` is the knob for trading runtime against signal, and
`MULT=1` still shows about +48% in 9 s.

**One lead checked and discarded, recorded so it is not chased twice.** The ProfileEvent
`HashJoinPreallocatedElementsInHashTables` reads 67,108,864 for `parallel_hash` and 0 for
`unified_hash` on this query, which looks like unified failing to preallocate its hash tables.
It is not: `rg HashJoinPreallocatedElementsInHashTables src/` shows the counter is incremented
only in `ConcurrentHashJoin.cpp`, so plain `hash` reports 0 as well, and unified's
`MapsTemplate::create(type, buckets, reserve)` does pass `reserve` down to
`RuntimeStorage::reserveBuckets`. The counter is uninstrumented outside `parallel_hash`, not a
missing reserve.
