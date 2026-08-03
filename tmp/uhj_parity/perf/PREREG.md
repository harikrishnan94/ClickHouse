# PREREG — pre-registered predictions

Rule: an entry here must exist **before** the change or measurement it predicts.
The ordering is checkable in git history and the verifier checks it. Orientation,
profiling and exploratory runs before pre-registering are orientation, not
acceptance evidence.

Each entry states: what is expected, the exact gate invocation that will prove it,
and what result would refute it. A prediction/result mismatch is a finding to
investigate, never to rationalize afterwards. Results are appended under the entry
after the run, never edited into the prediction.

---

## P0 — Unit 0: the measurement instrument

Registered before the harness exists. Written against commit `0945a745399` +
worklog entries E0/E1 only.

### P0.0 Declared constants (fixed now, so they cannot be tuned to a result)

- **Noise band**: an effect is "no result" if it is within `max(5%, 1 sample
  stdev)` of the comparator's median. No claim may be made inside it.
- **Runs per point**: 7 timed runs, plus 1 untimed warmup, **A and B interleaved**
  (run order `A B A B ...`, never all-A-then-all-B).
- **Statistic**: median and sample stdev (n-1) over the 7 runs.
- **Metrics**: wall = `query_log.query_duration_ms`; CPU =
  `query_log.ProfileEvents['UserTimeMicroseconds']`; phases = summed
  `processors_profile_log.elapsed_us` for `FillingRightJoinSide` (build),
  `JoiningTransform` (probe), `DelayedJoinedBlocksWorkerTransform` +
  `DelayedJoinedBlocksTransform` (non-joined).
- **G0.5 reconciliation tolerance, declared up front**: two separate claims.
  (a) *Accounting identity*: `build + probe + nonjoined + other` must equal the sum
  of all processors' `elapsed_us` **exactly** (it is a partition of the same set;
  any mismatch is a harness bug, tolerance 0).
  (b) *Independent cross-check*: for the INNER cells, a build-only query variant
  (probe side reduced to 1 row) must reproduce the full query's
  `FillingRightJoinSide` figure within **max(15%, 1 stdev)**. 15% rather than the
  5% noise band because the two queries genuinely differ (no probe-side
  interleaving, different memory pressure); a looser band here is honest, and it
  is a validity check on the phase source, not a source of any reported number.
- **Pinned settings, recorded in every run**: `enable_join_runtime_filters=0`,
  `max_bytes_before_external_join=0`, `max_block_size=65409`,
  `max_joined_block_size_rows=65409`, `query_plan_join_swap_table=0`,
  `parallel_hash_join_threshold=0`, `parallel_non_joined_rows_processing=1`,
  `log_processors_profiles=1`, `max_threads=<T>`, `join_algorithm='<exactly one>'`.

### P0.1 Prediction — the deficit does NOT reproduce everywhere

I predict, before running, that the matrix will **not** show a uniform
`unified_hash` deficit, and I am recording the shape I expect so that a
convenient result cannot be claimed as a confirmation afterwards:

1. At **1 thread**, `unified_hash` is at parity with `hash` (inside the noise band)
   on build-dominated INNER, or slower by a small single-digit percentage — because
   `bucketCountForThreads(1) = 1`, so there is exactly one bucket and no scatter
   pass. Any 1-thread deficit must therefore come from residual per-operation
   indirection, not from bucket count or cache footprint.
2. At **16/64 threads**, `unified_hash` is slower than `parallel_hash` on
   **build CPU** by a clearly-beyond-noise margin.
3. At **16/64 threads**, `unified_hash` is **faster** on at least one probe-bound
   cell. The snapshot reports this and it is a wanted result, not an anomaly.
4. **RIGHT/FULL with a non-joined scan** is `unified_hash`'s worst cell at 16/64
   threads.

**Refuted if**: the sweep shows `unified_hash` uniformly slower across all cells
including 1-thread probe-bound and 16-thread probe-bound (that would mean either a
global regression I have not identified, or an instrument that is measuring
something other than the join — and G0.3's A/A calibration is what tells the two
apart).

### P0.2 Prediction — inherited LEAD (1) points the wrong way

> **SUPERSEDED by P0.2R below.** The sub-table counts in this entry are wrong
> (4096 counts empty bucket *objects*, not populated partitions). The entry is
> kept unedited because it is a pre-registration; the corrected version follows.

Inherited LEAD (1) says the 16-thread build CPU excess is `unified_hash`'s
"per-bucket sub-table cache footprint at `BUCKETS_PER_THREAD = 2`". From E1 I have
already established from code, before measuring, that at 16 threads
`unified_hash` has **32** sub-tables and `parallel_hash` has **4096**. I therefore
predict this lead will be **REFUTED**, and that the real 16/64-thread candidates
are per-operation costs that a sharded map does not pay at all:

- per-bucket **mutex acquire/release** on a *shared* map during build insert;
- the **global cell-offset prefix sum** (`offsetInternal`) used to index RIGHT/FULL
  used flags, where `parallel_hash`'s flags are shard-local and flat.

**Refuted if**: an ablation that removes the bucket-count effect (e.g. forcing
`BUCKETS_PER_THREAD = 1`, or raising it) moves the 16-thread build CPU gap by more
than the noise band in the direction LEAD (1) predicts, while a lock-removal
ablation does not. That is the discriminating probe, and it belongs to Unit 1 —
recorded here only so the prediction predates the evidence.

### P0.2R Prediction (CORRECTED) — LEAD (1) may be real with the OPPOSITE sign

Registered after the requester rejected P0.2's arithmetic and before any
measurement. Correction derived and evidenced in WORKLOG E3.

`ConcurrentHashJoin.cpp:589` sets a row's shard from the bucket it will land in
(`getBucketFromHash(hash) & (num_shards - 1)`), so each of the 256 bucket indices
is owned by exactly one shard and only **256** sub-tables are ever populated. The
4096 figure counted empty bucket objects. Populated partitions:

| max_threads | `unified_hash` | `parallel_hash` |
| --- | --- | --- |
| 16 | 32 | 256 |
| 64 | 128 | 256 |

`unified_hash` therefore splits the same rows into **8x fewer** partitions at 16
threads, making each of its sub-tables **8x larger**. Revised prediction:

- LEAD (1) **as worded** ("too many sub-tables to miss the cache on") is still
  expected **REFUTED**: `unified_hash` has fewer, not more.
- But a sub-table cache effect may be real with the **opposite sign** — too *few*
  buckets, each too large to stay cache-resident during its insert window.
- **Discriminating probe, prediction registered before it runs:** sweep
  `BUCKETS_PER_THREAD` upward (2 -> 8 -> 32). If the "too few, too large" mechanism
  is real, the 16-thread build CPU gap **shrinks monotonically** as the count
  rises. If LEAD (1) as worded were right, the gap would **grow**. If neither, the
  gap is flat and the cause is per-operation (lock / offset), not footprint.
  Those three outcomes are mutually exclusive, which is what makes this a
  discriminating probe rather than a confirmation.

### P0.7 Prediction — the direct-addressed conversion is a `unified_hash` WIN

Registered before measurement, from WORKLOG E4. `canConvertToFixedHashMap`
(`UnifiedHashJoin/HashJoin.cpp:2077-2081`) requires `key32`/`key64`, which
`parallel_hash` can never satisfy, while `unified_hash` converts at any thread
count. I predict the `dense` cells at 16/64 threads show `unified_hash`
**faster** than `parallel_hash` by a large, beyond-noise margin on probe.

**Refuted if** `unified_hash` is at parity or slower there — which would mean the
conversion either does not fire or does not pay, and the harness assertion that
conversion fired is what tells those apart.

This axis exists because the dense-key range would otherwise have fired the
conversion silently inside the `small` cells and turned a quarter of the matrix
into an array-lookup comparison labelled as a hash-join comparison. The other
cardinalities now use keys spread over the whole `UInt64` range so the conversion
cannot fire, and the harness asserts per cell that it did not.

### P0.3 Gate invocations (the harness must make each of these runnable)

Each gate is a subcommand of `tmp/uhj_parity/perf/harness/uhjbench.py`, exiting
non-zero on failure and printing the raw evidence. Copy-paste re-runnable:

| Gate | Invocation | Passes iff |
| --- | --- | --- |
| G0.1 | `python3 harness/uhjbench.py gate-algo` | every cell's profiled verification run shows the expected symbol-namespace fingerprint and no foreign one; build-side `input_rows` equals the expected build-table row count (no swap) |
| G0.2 | `python3 harness/uhjbench.py gate-checksum` | for every cell, all measured algorithms return identical `(count, sum(cityHash64(all output cols)), groupBitXor(...))` |
| G0.3 | `python3 harness/uhjbench.py gate-aa` | the A/A delta is inside the noise band for at least one 1-thread and one 64-thread cell, on both wall and CPU |
| G0.4 | `python3 harness/uhjbench.py gate-signal` | the 16-thread build-bound CPU excess and the 16-thread RIGHT-non-joined wall excess reproduce in direction, and in rough magnitude |
| G0.5 | `python3 harness/uhjbench.py gate-phases` | (a) the phase partition is exact; (b) the build-only cross-check agrees within max(15%, 1 stdev) |
| G0.6 | `python3 harness/uhjbench.py gate-coverage` | every declared cell is MEASURED or SKIPPED-with-reason; zero silently missing |
| G0.7 | `python3 harness/uhjbench.py deficit-map` | emits the full classification; this is the stop condition, not a pass/fail |

### P0.4 Prediction — G0.3 A/A will pass, and this is the load-bearing one

I predict the A/A delta will be inside the noise band. **If it is not**, every A/B
from this harness is invalid and the correct action is to fix the instrument
(pin threads, drop caches consistently, increase run count, interleave harder),
**not** to widen the noise band until A/A fits. Widening the band to make A/A pass
is explicitly a banned move and I am recording that here so it is checkable.

### P0.5 Prediction — G0.4 known-signal recovery, and its confound

I predict the two snapshot effects reproduce in **direction**. I explicitly do
**not** predict they reproduce in magnitude, and I am recording the reason before
running rather than as an excuse afterwards: per worklog D1, the snapshot was
produced by a binary that predates commit `5362055b4ed`, which changes
`unified_hash`'s non-joined path — one of the two effects G0.4 tests. So:

- If **direction** reproduces for both, G0.4 is green.
- If the RIGHT-non-joined magnitude differs substantially, that is attributable to
  `5362055b4ed` and is a finding, not a gate failure — but I must then demonstrate
  the instrument still has power, by showing it resolves the *other* known effect
  and passes A/A.
- If **neither** effect reproduces in direction, G0.4 is **red** and Unit 1 does
  not start: an instrument that cannot see effects known to exist has no power to
  find new ones.

### P0.6 Expected SKIPs, declared in advance

- Any cell whose join kind is not supported by an algorithm. From
  `allowParallelHashJoin`, kinds outside {Left, Inner, Right, Full} cannot use
  `parallel_hash`; `LEFT SEMI`/`LEFT ANTI` carry `kind() == Left` and so are
  expected to be *supported*. If a SEMI/ANTI cell turns out to silently run as
  `hash` at 16/64 threads, G0.1 will catch it and the cell is SKIPPED with that
  reason rather than reported as a comparison.
- `RIGHT`/`FULL` at the large cardinality emit a very large non-joined result;
  if any such cell exceeds a wall-clock budget of 120 s per run it is SKIPPED with
  the recorded reason and the cardinality it failed at, not silently shrunk.

---

## Results appended below after each gate runs

(none yet — no gate has been run at the time of writing)

---

## P1 — Unit 1: attribution

Registered **before** any ablation patch exists and before the full deficit map was
read. The candidate list comes from the code inventory
(`artifacts/CANDIDATE_INVENTORY.md`), whose claims are LEADS, not evidence.

Which cells get attributed is decided by Gate G0.7, not by this list. If a
candidate's cell turns out not to be slower, the candidate is not investigated and
that is recorded — hunting a cause for a gap that does not exist is the failure
mode this ordering prevents.

### P1.0 Method, fixed now

- Every claim carries a G1.1 codegen artifact **unconditionally**, whatever its
  verdict.
- Every ablation is predicted (direction + rough magnitude) here **before** it is
  built and run.
- A null result is VOID unless a codegen diff confirms the targeted instructions
  actually left the binary (G1.4). An unvalidated null is never evidence that an
  operation is cheap.
- Ablation binaries are built into a **separate build directory** so the pristine
  `build/reldeb` binary stays available as the unmodified side of every
  before/after codegen diff, and so no ablation can leak into the delivered tree.
- `llvm-mca` is **not available** on this host (`llvm-objdump`/`llvm-nm` v22 are).
  Throughput analysis of loop bodies is therefore out of reach; instruction counts,
  loads/stores, branch and spill density, and inlining decisions are all still
  available and are what the gate actually requires. Recorded as a tooling
  limitation, not silently skipped.

### P1.1 Candidate operations and their pre-registered predictions

| # | Operation | Phase | Multiplicity | Prediction | Refuted if |
| --- | --- | --- | --- | --- | --- |
| C1 | per-bucket **mutex acquire/release** in `insertIntoBuckets` (`UnifiedHashJoin/HashJoin.cpp:187,207`); no counterpart in `hash`, ~`slots` per block in `parallel_hash` | BUILD | per bucket per block per clause (32/block at 16t, 128/block at 64t) | removing the lock loop shrinks the 16/64t build gap **beyond the noise band** | gap unmoved *and* codegen confirms the lock is gone -> REFUTED, cost is elsewhere |
| C2 | **bucket-count trade** via `BUCKETS_PER_THREAD` (2 -> 8 -> 32) | BUILD | once per join, per-row consequences | the three-way outcome discriminates: gap **shrinks** monotonically => UHJ's sub-tables are too few/too large; gap **grows** => inherited LEAD (1) as worded; gap **flat** => cause is per-operation, not footprint | any outcome is informative; this probe cannot fail, only decide |
| C3 | per-row **bucket routing** in `Prober::find` + the `prefix[bucket]` load (`TwoLevelHashTable.h:554-563`) | PROBE | per row | at `max_threads=1` UHJ takes the `sole` short-circuit, so this costs ~0 and the 1-thread probe gap is **not** explained by routing | a 1-thread probe gap that survives the `sole` path means routing is not the mechanism |
| C4 | **`use_offset` asymmetry**: baseline hardcodes `use_offset = true` (`HashJoin/KeyGetter.h:19`) so `parallel_hash` pays a re-hash + `std::call_once` per matched row; UHJ computes no offset for INNER/LEFT | PROBE | per matched row | this is a UHJ **advantage** and explains a large part of why UHJ is *faster* at 16/64t on INNER. Grafting `use_offset=false` into the baseline should **erase most of UHJ's probe lead** | if the baseline gets no faster, UHJ's probe lead is elsewhere |
| C5 | per-cell `offsetInternal` + `std::call_once` in the **non-joined scan** (`UnifiedHashJoin/HashJoin.cpp:1515`) | POST-JOIN | per populated cell | RIGHT/FULL low-match at 16/64t is UHJ's worst shape; swapping to the unsafe variant closes a beyond-noise part of it | gap unmoved with codegen confirming removal -> REFUTED |
| C6 | `per_offset_flags` **sized from summed bucket capacities** (`P7`), wider and sparser than the baseline's | PROBE + POST-JOIN | per matched row (cache) | shows as raised `LLC-load-misses`/row with flat-or-lower IPC, not as raised instructions/row | counters showing raised instructions at flat cache behaviour contradict the layout story -> REFUTED or downgraded |
| C7 | per-block **bucket scatter** (2 row passes + `num_buckets` index-column allocations) | BUILD | per block per clause | contributes to the 64t build gap more than the 16t one, since allocations scale with `num_buckets` | flat across thread counts => not allocation-driven |

C1, C4, C5 are ablatable in isolation. C3, C6 are **not** (they are the
partitioning and its consequence) and are expected to route to SUPPORTED via
G1.1 + G1.3 rather than to CONFIRMED. C2 is a knob sweep, not an ablation, and its
result bounds rather than removes.

### P1.2 Prediction about the shape of the answer

From the pilot slice (INNER|u64|hi), `unified_hash` was at parity at 1 thread and
**faster** at 16 and 64 threads. I therefore predict, before reading the full map,
that the deficit is **not** general, and that the attribution table will be
dominated by RIGHT/FULL non-joined shapes rather than by build or probe of INNER.

**Refuted if** the full sweep shows broad slowness across INNER/LEFT at 16/64
threads, which would mean the pilot slice was unrepresentative and C4's advantage
does not generalise.

If the deficit genuinely does not reproduce in most cells, the honest deliverable
is a deficit map that says so plus attribution of the cells where it does — not a
manufactured gap. That outcome is explicitly acceptable and is recorded here in
advance so it cannot later be dressed up as a disappointment or padded out.
