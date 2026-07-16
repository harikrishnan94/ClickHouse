# INDEPENDENT_DESIGN — cooperative WaveJoin probe for `RadixHashJoin`

Sealed 2026-07-16, derived exclusively from:
- the corrected formal contract `src/Interpreters/RadixHashJoin/WaveJoinProbe.tla`
  (independent-verifier verdict SHIP),
- the public surfaces `RadixHashJoin.h`, `IJoin.h` (`IJoinResult`, `IBlocksStream`),
- the callers (`JoiningTransform` drive discipline: one `joinBlock` then one
  `IJoinResult::next` per work quantum per lane; one shared delayed-blocks stream
  pulled concurrently by the delayed-worker transforms; `PlannerJoins` constructor
  wiring), and
- the existing tests (gtest + stateless 04508–04512).

`RadixHashJoin.cpp` has not been opened before this document was committed. After
the design checkpoint, the old implementation is only mapped for compatibility
constraints and removed/replaced; nothing of its coordination design may be
retrofitted into this one.

## 0. Roles and vocabulary

- **Worker** — any executor thread currently inside one of this join's probe
  entry points: `joinBlock(block, lane)`, an `IJoinResult::next()` it returned,
  or the delayed-blocks stream's `next()`. Workers are the *same* scheduled
  probe threads in every phase; there are no dedicated producer or consumer
  crews, no probe-side thread pool, no output queue, no reorder buffer, and no
  central scheduler. (The build-side radix pool is untouched — build contract.)
- **Wave** — the single shared probe window. Exactly one exists at a time; it is
  either **active** (accepting admissions) or **sealed** (being drained). TLA:
  `st.phase = "active"` vs the drain phases.
- **Job** — one unit of drain work: per-pass `pre`, per-block `scatter`,
  per-pass `refalloc`/`refine` (only when `PL > 1`), per-leaf `probe`. A leaf is
  the smallest probe task.

## 1. Shared state (one `Wave` instance, epoch-reused)

```
struct Wave {
    // ---- admission (active phase) ----
    std::atomic<UInt64>  reserved_bytes;     // TLA st.mem: admitted + in-flight
    std::atomic<UInt32>  inflight;           // reservations not yet appended
    std::atomic<bool>    seal_requested;     // TLA st.crossed
    std::mutex           admit_mutex;        // guards `admitted` only (append)
    std::vector<Admitted> admitted;          // TLA st.queue: routed blocks + histograms

    // ---- drain machine ----
    std::atomic<UInt8>   phase;              // Active, Pre, Scatter, RefAlloc, Refine, Probe
    std::atomic<UInt32>  stage_next;         // claim cursor of the CURRENT stage
    std::atomic<UInt32>  stage_remaining;    // completions outstanding in the stage
    // arenas: per-pass chunks (a0), per-leaf chunks (a1), exact-sized; plus the
    // per-(block,pass) base offsets and per-(pass,leaf) base offsets computed by
    // pre/refalloc from the admission-time histograms.

    // ---- failure / cancellation ----
    std::mutex           primary_mutex;      // first-exception-wins slot
    std::exception_ptr   primary;            // TLA st.primary
    std::atomic<bool>    cancelled;          // TLA st.cancelled

    // ---- lifecycle ----
    std::atomic<UInt64>  epoch;              // wave generation (ABA guard)
    std::atomic<UInt32>  participants;       // workers currently inside a drain step
    std::mutex           completion_mutex;   // pairs with completion_cv only
    std::condition_variable completion_cv;   // sealed-tail / phase-transition wait
};
```

Everything else (leaf `HashJoin` tables, plan constants `P0`, `PL`, budget) is
read-only during probing.

## 2. Admission: Reserve / Admit (TLA `Reserve`, `Admit`, `Seal`, `EOFSeal`)

`joinBlock(block, lane)` for a post-build call:

1. **Scan (lane-local, no locks):** hash + route every row (pass id, leaf id),
   build the block's leaf-granular histogram, using per-lane scratch indexed by
   `lane` (no locking; lanes are stable per stream).
2. **Reserve (the atomic admission check):**
   `old = reserved_bytes.fetch_add(bytes)` *conditionally*: the reserve loop
   reads the counter and only attempts the add while `current < BUDGET`
   (compare-exchange), matching the spec's atomic conditional add. Outcomes:
   - `old < BUDGET` — reservation granted (`inflight++`). If
     `old + bytes >= BUDGET`, this worker is the **crossing admission** and sets
     `seal_requested = true`. Overshoot is bounded by construction:
     `reserved_bytes <= BUDGET + max_block_bytes` (the single crossing block).
   - counter already `>= BUDGET` — the wave is sealed or sealing; do **not**
     admit; go help (step 5).
3. **Admit:** append `{routed block, histogram}` to `admitted` under
   `admit_mutex` (append only — hashing already done), then `inflight--`.
4. **Seal hand-off:** the worker whose `inflight--` reaches 0 while
   `seal_requested` is set performs the *bounded phase transition*
   active→Pre: initializes the Pre stage cursor (`stage_next = 0`,
   `stage_remaining = P0`) and publishes `phase = Pre`. No admission or
   next-wave scanning can overlap a drain: reserve/admit are only possible
   while `phase == Active` and the counter is below `BUDGET`.
5. **Help or finish:** the call returns a `WaveJoinResult` (this worker's own
   result object; see §5). If this worker still holds an unadmitted block
   (its reserve was refused because the wave is sealed), the result keeps the
   block as **call-local pending input** and its `next()` calls first help
   drain the sealed wave, then admit into the fresh wave after completion.
   No global pending queue exists; pending input lives only inside the active
   call, so at EOF every block has been admitted.

EOF: the **first** delayed-blocks pull performs `EOFSeal` (CAS Active→Pre when
input is exhausted and the wave is non-empty) — same machine, no second
algorithm. An empty final wave means `getDelayedBlocks` has nothing to do.

## 3. Drain: claim / execute / finish (TLA `Claim`, `Finish*`, barriers)

The **worker step**, executed inside `next()` pulls (bounded work per pull):

```
step():
  if cancelled: unwind (release owned task), rethrow primary when visible
  switch (phase):
    Active:   no drain work; return
    Pre|Scatter|RefAlloc|Refine:
              k = stage_next.fetch_add(1)
              if k < stage_size: execute job k; on finish,
                  if (--stage_remaining == 0): advance_stage()   // last finisher
              else: stage exhausted -> bounded wait for phase change (§6)
    Probe:    k = stage_next.fetch_add(1)
              if k < leaf_count: run leaf task k, emitting through OWN result
              else: no claimable leaf -> §6
```

- **Claims are exactly-once** by the stage cursor `fetch_add` (TLA
  `UnownedClaimable` + `Claim`); **completions are exactly-once** by
  `stage_remaining` (TLA done-sets). A worker owns at most one task at a time
  (TLA `wk[w].job`).
- **Barriers are decentralized**: the last finisher of a stage advances the
  phase (TLA's unattributed barrier transitions) — there is no coordinator
  thread. Stage order: Pre → Scatter → (RefAlloc → Refine when `PL > 1`,
  else direct transfer a0→a1) → Probe → CompleteWave.
- **Job bodies** (all derived from admission-time histograms; no rescanning):
  - `pre(p)`: sum per-block histograms over pass `p`'s leaf range; allocate
    arena0[p] exactly; compute per-(block,pass) base offsets (prefix sums in
    queue order) — this is what makes the scatter *stable*.
  - `scatter(b)`: copy block `b`'s rows into their pass arenas at
    `base[b][p] + running offset` — write ranges of distinct blocks are
    disjoint by construction (TLA `RaceFree`/`RankInjective`); no
    synchronization on the arena.
  - `refalloc(p)`: prefix sums of leaf counts within pass `p`; allocate
    arena1 chunks for `p`'s leaves.
  - `refine(p)`: single worker per pass scatters arena0[p] into its leaf
    chunks with running per-leaf cursors — stable because arena0[p] is in
    queue order and the pass is owned by one worker.
  - `probe(l)`: probe leaf `l`'s arena chunk against leaf `HashJoin` `l` by
    delegation (correctness by construction, as the build side contract
    already relies on); emit the produced blocks through the executing
    worker's own result (§5). Under ordinary skew a large leaf simply keeps
    its one owner busy while all other leaves remain claimable — remaining
    work stays distributed; the tail is the unavoidable one.
- **CompleteWave**: the last probe finisher releases the wave: frees arenas
  and admitted input copies (input copies are actually released earlier, at
  the Scatter→next barrier, TLA `liveEntries`), resets counters/cursors,
  `epoch++`, publishes `phase = Active` with release semantics, and
  broadcasts `completion_cv`. Exactly-once by being the unique
  `stage_remaining 1→0` transition of the Probe stage.

## 4. Memory accounting (TLA `MemAccounted`, `MemBound`, `CrossedSound`)

- `reserved_bytes` accounts **exactly** the admitted wave bytes plus in-flight
  reservations. `BUDGET` (derived from `probe_buffer_{fraction,min,max}_bytes`
  as before) is the **admission/sealing threshold only**. The invariant the
  implementation maintains is `reserved_bytes <= BUDGET + max_block_bytes`,
  where the overshoot is the single crossing block.
- Explicitly **outside** `BUDGET` (documented, never claimed otherwise): drain
  arenas (exact-sized from histograms), hash/route columns, per-worker input
  blocks in flight, allocator overhead, and output blocks.

## 5. Output contract (TLA `out[w]`, `OutputJustified`, `FinalRefinement`)

- Workers emit **their own** result blocks: a leaf task's output goes only
  into the executing worker's `WaveJoinResult` (or the delayed stream pull
  that ran it). There is no shared output buffer of completed results.
- **Bounded per-task continuation** (explicitly distinguished from result
  reordering): a leaf probe may produce several output blocks (block-size
  caps); the worker's result holds the leaf's in-progress inner result and
  yields one block per `next()` call until the leaf is exhausted, then the
  task completes. This is the only buffered output: the block(s) required for
  the current call to return.
- `WaveJoinResult::next()` returns `is_last` when: the worker holds no task
  and no pending input, and either the wave it participated in has completed
  or nothing is claimable *and* its own admitted block's wave has been fully
  handed over (remaining owned tasks finish in their owners' quanta).
  Correctness is exact output-multiset equality over all lanes plus the
  delayed stream; per-lane split and global order are unconstrained.
- Pre-build (schema/header path) `joinBlock` keeps its existing delegation
  behavior (public behavior preserved).

## 6. The only waits, and work conservation (TLA `ParticipationLive`)

A worker never idles while compatible sealed-wave work is claimable — the
step loop claims before it ever considers waiting. The only blocking wait is
`completion_cv` with predicate (phase changed ∨ wave completed ∨ cancelled),
taken exclusively when:

- the current stage's cursor is exhausted but the stage is not complete
  (bounded phase transition: waiting on the last finishers), or
- the wave is sealed, nothing is claimable, and this worker still has pending
  input to admit (sealed tail).

Both are the contract's allowed idle reasons. `notify_all` happens on every
phase advance, wave completion, and cancellation. A worker whose result has
no pending input and no task does not wait at all — it returns `is_last` and
leaves the tail to the owners (keeps `joinBlock`+`next` quanta bounded and
lanes responsive, which is what the concurrency gtest pins).

**Fairness mapping discharge** (verifier round-1 finding 4): liveness in the
spec assumes each participating worker keeps being scheduled. In ClickHouse,
`JoiningTransform` keeps calling `next()` until `is_last` on every lane, and
the delayed-worker transforms keep pulling the shared stream until it
finishes; a worker that stops being scheduled mid-task only exists on
pipeline teardown, which is handled by §7's destructor rule — so no task can
be silently stranded.

## 7. Cancellation, failure, cleanup (TLA `Fault*`, `Release*`, `Teardown`)

- **First exception wins**: any throwing job execution (or admission-time
  failure) stores its `exception_ptr` into `primary` under `primary_mutex`
  only if empty, then sets `cancelled = true` (release) and broadcasts.
  Later exceptions are recorded losers (dropped), never overwrite.
- **Cancellation is visible**: every claim loop and wait predicate checks
  `cancelled` (acquire); no new task or admission starts after it.
- **Owned work unwinds safely**: the throwing/cancelled owner releases its
  task state; `participants` tracks workers inside drain steps, and the last
  participant to leave a cancelled wave performs teardown — freeing whatever
  is still live exactly once (TLA `Teardown`, `FreedOnce`).
- **Propagation**: every entry point (`joinBlock`, `next`, delayed stream)
  rethrows `primary` once cancellation is visible; the query fails with the
  first error. No fallback path anywhere: nothing substitutes defaults or
  degrades to another algorithm.
- **Early caller destruction**: destroying a `WaveJoinResult` that still owns
  an incomplete task or pending input poisons the wave (cooperative cancel
  without an error unless one is already primary). Rationale: the executor
  only abandons a result before `is_last` during pipeline teardown; poisoning
  guarantees no silent multiset corruption (dropped task output) can ever
  look like success. A result destroyed with no task/pending is a no-op.
- **Exactly-once release**: arenas and the admitted-input copies each have a
  single release site per wave epoch (barrier or teardown), guarded by the
  unique last-finisher / last-participant transitions.

## 8. Delayed-blocks path (thin adapter, same machine)

`getDelayedBlocks()` returns a stream whose `nextImpl()` = `EOFSeal`-if-first
+ the same worker step loop, returning one output block per pull, empty when
the final wave completed (or rethrowing on failure). It shares every
mechanism above; there is no second algorithm for the final partial wave.
`hasDelayedBlocks()` stays `true`.

## 9. TLA ↔ C++ correspondence (summary)

| TLA | C++ |
| --- | --- |
| `st.mem`, `Reserve`, crossing admission | `reserved_bytes` conditional fetch-add, `seal_requested` |
| `st.queue`, `Admit` | `admitted` append under `admit_mutex`, `inflight` |
| `Seal`/`EOFSeal` | last admitter with `seal_requested` / first delayed pull |
| `wk[w].job`, `Claim`, `Finish*` | stage cursor `fetch_add` + task locals + `stage_remaining` |
| barriers | last-finisher `advance_stage` |
| `out[w]` | the worker's own `WaveJoinResult` / stream pull |
| `primary`, `cancelled` | `primary` slot under mutex, `cancelled` atomic |
| `Release*`, `StopWorker`, `Teardown` | unwind + `participants` last-out teardown |
| `CompleteWave` | last probe finisher resets epoch, broadcasts |
| fairness (WF per worker) | executor drive discipline + §7 destructor rule |

## 10. Compatibility constraints to verify when the old file is opened

Only these things may be taken from the old implementation: the public entry
points' signatures and pre-build behavior, the build-side structures the probe
must read (leaf `HashJoin` array, plan constants, scatter kernels), profile
event names, and the settings plumbing. The coordination design above is
final and is not to be adjusted toward the old one.
