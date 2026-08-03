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
