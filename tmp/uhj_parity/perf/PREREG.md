# PRE-REGISTRATION

Written **before** the change it predicts. Git history is the proof of ordering:
this file is committed before the harness / ablation it refers to. Orientation,
profiling and exploratory runs done before an entry are orientation, not
acceptance evidence, and are logged in `WORKLOG.md` instead.

A prediction/result mismatch is a finding to investigate, never to rationalise
afterwards. Where a prediction turned out wrong, the entry stays as written and
the outcome is recorded beneath it.

---

## P0 — Unit 0: the measurement instrument

**Registered:** before writing `sweep.py` / `matrix.py` (commit ordering is the check).

### What is being built

A harness that measures, for every cell of a declared matrix, the wall time and
CPU time of `unified_hash` against the correct comparator (`hash` at 1 thread,
`parallel_hash` at 16 and 64 threads), split into build / probe / non-joined
phases, and classifies each cell slower / faster / within noise.

### Declared matrix

| Axis | Levels |
| --- | --- |
| join kind | `INNER`, `LEFT`, `RIGHT`, `FULL`, plus `LEFT SEMI` and `LEFT ANTI` |
| key type | one representative per key-getter family enumerated from `KeyGetter.h` / `HashJoinMethods*.h` (see `artifacts/CANDIDATE_INVENTORY.md`); at minimum single fixed-width (`UInt64` -> `key64`), string (`String` -> `key_string`), multi-column composite (`(UInt64, UInt64)` -> `keys128`) |
| thread count | 1, 16, 64 |
| cardinality (distinct build keys) | 1 thread: small = 10,000 and medium = 1,000,000. 16/64 threads: large = 50,000,000 (under the 100M cap) |
| match rate | high = 0.9, low = 0.1 (fraction of probe rows finding a match) |

Cardinality is expressed as the **build table's row count with unique keys**, so
that "distinct build-side keys" and "hash table size" are the same number and one
probe row can produce at most one output row. Recorded as a deliberate call: it
makes the cache-footprint axis clean at the cost of not exercising duplicate-key
`RowRefList` chains, which is registered as a **known coverage gap** in the report.

Probe row counts: 2,000,000 at 1 thread; 50,000,000 at 16/64 threads.

Cells = 6 kinds x 3 key types x 2 match rates x 4 (thread, cardinality) pairs = **144**.

### Pinned settings, recorded in every run

`enable_join_runtime_filters=0`, `max_bytes_before_external_join=0`,
`max_block_size=65409`, `max_joined_block_size_rows=65409`,
`query_plan_join_swap_table=0`, `parallel_hash_join_threshold=0`,
`parallel_non_joined_rows_processing=1`, and `join_algorithm` set to exactly one
algorithm per run.

`join_algorithm` is set to a **single** algorithm rather than a list. Reason,
established from `TableJoin.cpp:1293-1307` and `PlannerJoins.cpp:1244-1257`:
`allowParallelHashJoin` returns false unless `PARALLEL_HASH` is in the list, so a
single-valued `join_algorithm='hash'` cannot be silently upgraded to
`ConcurrentHashJoin` — which it demonstrably could be otherwise, since
`use_parallel_hash` is true whenever the RHS size estimate is missing.

### Predictions (direction and rough magnitude), before any measurement

| # | Prediction | Would refute it |
| --- | --- | --- |
| P0-a | Gate G0.3 (A/A) reports a delta **inside** the noise band `max(5%, 1 stdev)` in both the 1-thread and 64-thread calibration cells. | An A/A delta outside the band => the instrument measures scheduling drift, and every A/B from it is void until fixed. |
| P0-b | Some cells show **no deficit**, and at least one cell shows `unified_hash` **ahead** — most likely a probe-bound cell at 16 threads, per the inherited snapshot. | Every cell slower would itself be suspicious and would prompt a check for a systematic harness bias (e.g. ordering, warm-up). |
| P0-c | At **1 thread** the deficit, if present, is **small** (single-digit %), because `bucketCountForThreads(1) = 1` leaves `unified_hash` with one bucket and therefore no bucket-footprint or locking cost — only residual per-operation indirection. | A large (>15%) 1-thread deficit would mean the one-bucket path carries far more than indirection, redirecting Unit 1 away from the lock/offset candidates. |
| P0-d | At **16/64 threads** the RIGHT/FULL non-joined shapes show the **largest** relative deficits, and build-bound INNER shows a CPU excess larger than its wall excess. | Deficits concentrated instead in probe-bound INNER would refute both inherited leads and redirect Unit 1 to the probe lookup path. |
| P0-e | Gate G0.4 recovers both recorded snapshot effects in **direction**. Magnitude is registered as **uncertain**: the snapshot was produced by a binary predating `5362055b4ed` (see WORKLOG D1), which changed `unified_hash`'s non-joined path — exactly one of the two effects. | If direction is not recovered, the two candidate explanations ("instrument lacks power" vs "the code changed under the snapshot") must be separated before G0.4 is scored; scoring it green without separating them is a banned move. |

### Gate invocations that will prove or refute the above

Each is copy-paste re-runnable from the repo root.

| Gate | Invocation | Expected result |
| --- | --- | --- |
| G0.1 | `python3 tmp/uhj_parity/perf/gates.py g01` | Every cell has a symbol-level proof row; 0 cells with `algo_verdict != requested`. |
| G0.2 | `python3 tmp/uhj_parity/perf/gates.py g02` | 0 checksum mismatches across algorithms per cell. |
| G0.3 | `python3 tmp/uhj_parity/perf/gates.py g03` | A/A delta within `max(5%, 1 stdev)` for the 1-thread and 64-thread calibration cells. |
| G0.4 | `python3 tmp/uhj_parity/perf/gates.py g04` | Direction of both recorded effects recovered; magnitude reported, not asserted. |
| G0.5 | `python3 tmp/uhj_parity/perf/gates.py g05` | Phase accounting identity exact; build-only cross-check within the declared tolerance (below). |
| G0.6 | `python3 tmp/uhj_parity/perf/gates.py g06` | Declared matrix enumerated; every cell MEASURED or SKIPPED-with-reason; 0 missing. |
| G0.7 | `python3 tmp/uhj_parity/perf/gates.py g07` | Deficit map emitted for every measured cell on both wall and CPU. |

### Declared up front, so they cannot be tuned afterwards

- **Noise band:** an effect is "no result" if it is within `max(5%, 1 sample stdev
  of the comparator's runs)`. At least **7** runs per point; median and sample
  stdev reported. A and B runs are **interleaved** (round-robin per repetition),
  not batched, so drift cannot masquerade as an effect.
- **G0.5 tolerance:** two separate checks. (i) The accounting identity
  `build + probe + nonjoined + other == sum(all processors' elapsed_us)` must hold
  **exactly** (it is a partition of the same set; any drift means rows were lost).
  (ii) The **independent** cross-check: build phase measured by a build-only query
  variant must agree with `FillingRightJoinSide` elapsed from the full query to
  within **20%**. 20% rather than the noise band because the two differ
  structurally (the build-only variant still runs the probe pipeline over a
  near-empty left side); a disagreement larger than 20% means
  `FillingRightJoinSide` is not the build phase and every build percentage is void.
- **Phase sources:** build = `FillingRightJoinSide`, probe = `JoiningTransform`,
  non-joined = `DelayedJoinedBlocksTransform` + `DelayedJoinedBlocksWorkerTransform`,
  all from `system.processors_profile_log` with `log_processors_profiles=1`.
  Verified present and non-trivial in a pilot (WORKLOG E2).
- **G0.1 mechanism:** a dedicated **assertion run per cell**, separate from the 7
  timed runs so it cannot perturb them, with the CPU query profiler on, asserting
  from `system.trace_log` demangled stacks:
  `parallel_hash` iff frames match `DB::ConcurrentHashJoin`;
  `unified_hash` iff frames match `DB::Unified::HashJoin` and no
  `DB::ConcurrentHashJoin`; `hash` iff frames match `DB::HashJoin::` and neither of
  the other two. This is positive identification of the code that actually ran,
  not an echo of the requested setting. Verified to discriminate in a pilot
  (WORKLOG E2: 437/453, 414/430, 0/0 respectively).

### Known SKIPs predicted in advance, with reasons

- `LEFT SEMI` and `LEFT ANTI` at 16 and 64 threads have **no `parallel_hash`
  comparator**: `allowParallelHashJoin` (`TableJoin.cpp:1301-1303`) returns false
  for any kind other than Left/Inner/Right/Full, so requesting `parallel_hash`
  silently yields plain `hash`. These cells are recorded SKIPPED for the
  `parallel_hash` comparison, with `unified_hash` vs `hash` still measured and
  reported as context. Predicting this here so that discovering it later cannot be
  presented as a finding.
