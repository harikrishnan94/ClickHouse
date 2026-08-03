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
