# Design: probe-side win-or-parity for `ConcurrentHashJoin` (`phj-ph`)

Status: REV 3 — awaiting requester approval. No implementation before approval.
Rev 3 change (requester): item 7 added — `AmacWalk` policy axis
{bare, wrap_aware}: the wrapped-chain term leaves the engagement gate;
wrapped plans select a wrap-aware ring instantiation per `joinBlock`
(zero per-row cost) instead of disengaging; plus U5 threshold-boundary
validation cells (engage-as-much-as-profitable companion).
Rev 2 changes (requester): (a) narrow slot-ids adopted on the zero-copy
build scatter path NOW (widen only at the `IColumn::scatter` boundary on
the copying path); (b) scratch-pool rationale documented (atomic parking
= per-lane array with minimal sync; lanes ≠ slots, collisions legal);
(c) probe regime map table added (family × loop × why); (d) ASOF ring
build-parallelism statement added (zero build-side impact). Confirmed
earlier: item 6 dropped; fleet 8× m8g.24xlarge; gate = Probe-event
medians + guards.
Date: 2026-07-28. Base: `phj-ph` @ `21f6d804339`. Baseline arm for all
comparisons: `concurrent-hash-join-profile-events` @ `a05f3ee81ff` (saved
binary `tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin`, sha in
MANIFEST). AHJ mechanism donor: `/mnt/ch/ClickHouse-ahj` @ `cf465cfbe23`
(saved reference binary for disasm anchors).

## Goal and gate

Implement six probe-side mechanisms (user items 1–5 plus item 7, the
`AmacWalk` policy; item 6 dropped by requester decision, see §Item 6) so
that the probe-side (`joinBlock`) cost of EVERY measured cell is a win
or parity vs the two-level baseline.

**G-probe-perf (must-hold):** for every probe cell in the frozen matrix
(probe grid, kind/strictness set, size/thread ladders, hit-rate, dup-heavy,
mixed-ON — every cell whose claim touches `joinBlock`; plus the 20 prior
loss cells), per-cell median of thread-summed
`ConcurrentHashJoinProbeMicroseconds` (the whole probe phase: dispatch +
lookup + emit) satisfies B ≤ A × (1 + band), where band = per-cell noise
band from A/A calibration (max(3%, measured spread)), with the established
duration floors (≥200 ms per cell, ≥2M probe rows/thread). Wall time is a
secondary sanity check on probe-dominated cells. `ProbeDispatch`/
`ProbeLookup` reported per cell for attribution.

**Guards (must-hold):**
- G-build-guard: build cells (wall + `Build*` events) in-band — item 1
  changes the build scatter route, it must not regress builds out of band.
  (The pre-existing ring-independent BuildInsert gap at S4/S5 is NOT in
  scope and not made worse-than-current.)
- G-parity (636-case harness incl. force-engagement staged counters),
  G-order (Native block checks + 03448/03711 ×10 + baseline power check),
  G-tests (differential vs baseline binary; candidate failures ⊆ baseline).
- G-disasm: flat-loop and ASOF-ring anchors instruction-equivalent vs the
  saved ahj reference binary (llvm-objdump ranges, same method as the
  prior mission).
- Honest-red rule: a cell that stays red after ≤5 pre-registered fix
  cycles is reported red, never silently accepted.

Process per commit: PREREG entry → implement → G-build → G-parity →
(G-order/G-tests where relevant) → hygiene loop (2 report-only subagents +
fixer + re-gates + follow-up commit) → never amend/rebase/push. Evidence
tree: `tmp/chj_probe_parity/`.

## Item 1 — Dedicated route hash (build+probe matched pair)

**What.** Replace `calculateHashes` (the slot map's own hash: CRC32C
numerics / DefaultHash strings / SipHash-128 `hashed`) + `hashToSelector`
with the AHJ route-word family: one cheap 32-bit fold per row, slot =
`word >> (32 - bits)`, `bits = log2(slots) ≤ 8`. Deliberately decorrelated
from the map hash (`routeWord` = CRC-32 ISO `__crc32d(-1U, key)` on ARM /
golden-ratio multiply-shift elsewhere, vs the map's CRC-32C): the
route-decorrelation invariant from commit `844ee1a82dd` is preserved by
polynomial/function split instead of bit-window split.

**Port surface.** New `src/Interpreters/HashJoin/JoinSlotRouting.{h,cpp}`:
- primitives (~50 inline lines from ahj `Columns/ColumnsScatter.h:80-134`):
  `routeWord`, `mixStep`, `foldBytes` (8-byte steps + constant-size tail
  switch), `finalizeRoute`;
- `computeJoinSlotRoutes` port of ahj `JoinRouteHashing.cpp:100-214`:
  single-numeric fast path (widths 1/2/4/8), all-fixed width-8 unrolled
  fold (2/3/4 columns), `ColumnString` byte fold, live-LC fold via
  `getDataAt` value bytes, generic per-column `IColumn::computeHashInto`
  + `mixStep` accumulation (exists on phj-ph, `IColumn.h:417`);
- two sinks: probe (`UInt8 * slot_ids`) and build (`IColumn::Selector &`)
  over ONE shared fold implementation — the build/probe bit-identical
  contract stated in comments (fold change requires touching both sides in
  lockstep).
- `key8`/`key16` keep their current low-bit-of-key routing (FixedHashMap,
  `place = key`); the fold applies to all other families. ASOF folds the
  equality prefix only (existing `getDispatchKeyShape` retained).

**Fused with: single key preparation.** `ConcurrentHashJoin::joinBlock`
constructs the block's `JoinOnKeyColumns` ONCE (materialize + nullable
unwrap + join mask), routes over its prepared `key_columns`, stores the
vector in `RoutedJoinResult`, and `HashJoin::joinRoutedBlock` takes it by
`&&` instead of re-preparing (verified: one instance per block, refcounted
holders survive the lazy `next()` gap, `HashJoinResult` chunking never
re-creates it). This deletes today's double materialization (const/LC/
sparse unwrap ×2, null-map OR-merge ×2). LC map types: fold reads value
bytes (`getDataAt`) — representation-independent, so the live-LC probe
column and the build side's columns produce identical words by
construction; the `saved_hash` consistency risk flagged in review does not
apply because the fold never uses the getter's hash.

**Build side (revised per requester change a).** `addBlockToJoin` route
switches to the same fold, and the fold's build sink emits a NARROW
slot-id array now: element width picked from `num_shards`
(UInt8/16/32/64 menu; `num_shards ≤ 256` today, so UInt8 always — the
wider types future-proof the sink template, they compile but are dead).
`scatterBlocksWithSelector` (the zero-copy path) consumes the narrow
array directly — verified it only reads `selector[i]` to bucket row
indexes into per-shard `ScatteredBlock::Indexes`, no `IColumn` API
involved. `scatterBlocksByCopying` widens once into a temporary
`IColumn::Selector` at the `IColumn::scatter` call boundary (core
signature requires UInt64): one 8 B/row temporary, written once and read
once per column, paid only on the narrow-row copying path. A core
`IColumn::scatter` overload for narrow selectors is noted as a future
option, not in scope.

**Effects (measured targets).** ProbeDispatch bill collapses on all
multi-slot cells: `mixed` S5 ~25.0 thread-s and S2/S3 ~6.2–6.3 s →
one cheap fold pass (no serialization, no SipHash); key64 S4 ~1.1 s,
fixstr/k128 S2 1.5–1.9 s shrink to fold+write. Lookup hash count becomes
1 per row — equal to baseline. Double materialization gone (LC/const-key
workloads gain extra).

**Risks + prereg checks.** (a) Slot distribution changes → pre-registered
slot-balance check (per-slot row counts + max chain proxy on key64
sequential/random, strings, `mixed`, ≥16 slots) before accepting; (b) x86
`routeWord` is multiply-shift (Fibonacci hashing, known-good but
unmeasured here) — acceptance fleet is ARM; x86 gets a spot-check only if
an x86 box is available, else recorded NOT-CLAIMED; (c) build scatter
changes → G-build-guard + full parity/tests rerun; (d) stats-hint cache
keys unaffected (route not part of the key) — verified during U1.

## Item 2 — Pooled per-lane ProbeScratch + lane plumbing

**What.** Port AHJ's lane concept and scratch pool:
- `IJoin::joinBlock(Block, size_t lane)` defaulted virtual (forwards to
  lane-less; no other join touched) — ahj `IJoin.h:139` pattern;
- `JoiningTransform` gains `stream_index` ctor param (pipeline-builder
  loop index, `QueryPipelineBuilder.cpp` joinPipelinesRightLeft), passes
  it per `joinBlock` call; lanes stable per stream; out-of-range/collision
  tolerated by contract (totals transform = lane 0 collision is legal);
- `ConcurrentHashJoin::ProbeScratch { PaddedPODArray<UInt8> slot_ids;
  PaddedPODArray<UInt64> found_word; PaddedPODArray<UInt64> found_offset; }`
  with `2 × num_slots_hint` atomic parking slots (exchange-out /
  CAS-back, mutexed pool fallback, `invalid_lane` for legacy entries,
  freed in dtor) — ahj `PartitionedHashJoin.cpp:631-666` semantics;
- **phj-ph lifetime nuance (differs from AHJ):** the lookup is lazy
  (first `next()`), so the scratch is owned by `RoutedJoinResult` and
  CAS-released in its destructor, not at `joinBlock` exit. Safe: one
  in-flight result per lane; collisions degrade to the pool.
- `slot_ids` narrows to `UInt8` end-to-end on the probe
  (`RoutedProbeContext::slot_ids`, routed loops, AMAC admit) — slots ≤ 256.

**Effects.** Removes per-block allocation of 8 B/row selector +
16 B/row AMAC arrays; route storage 8× smaller (8→1 B/row). Targets
the S1/S2 small-map floor (+3–5%) and shaves scratch traffic at
DRAM-bound S5. Acquisition stays lazy: sub-256-row blocks and slots==1
allocate nothing.

**Why atomic parking instead of a plain per-lane array (requester
question b).** The parking table IS "an array of one scratch per lane" —
`acquire` is a single `exchange(nullptr)` and `release` a single
`compare_exchange`, nothing heavier. The synchronization cannot be
dropped and the bound cannot be 256, for verified reasons:
- Lanes are NOT join slots. `num_shards ≤ 256` bounds slots; a lane is a
  probe PIPELINE STREAM index (`max_streams`), unrelated and not bounded
  by 256 on large hosts. The table is sized `2 × num_threads` at
  construction (never resized — lock-free indexing licenses that), and
  out-of-range lanes are a documented `IJoin` contract, not a bug — they
  take the fallback.
- Same-lane CONCURRENT `joinBlock` calls are legal pipeline shapes, not
  hypotheticals: the totals `JoiningTransform` is built without a stream
  index and lands on lane 0, colliding with stream 0; `FilledJoinStep`
  builds every stream as lane 0; plan-time header probes come in
  lane-less. A bare `std::optional<Scratch>` per lane is a data race
  (UB) in exactly those shapes. The exchange/CAS pair makes a collision
  degrade to the mutexed pool ("never lost, never double-owned") instead
  of corrupting.
- phj-ph adds a lifetime reason: the lookup is lazy, so the scratch is
  owned by `RoutedJoinResult` until its destructor — the release point is
  detached from `joinBlock`'s scope, which a scoped-optional design
  cannot express.
The mutexed pool vector is the safety valve for collision losers and
out-of-range lanes; steady state never touches it.

## Item 3 — Flat descriptor fused loop (non-AMAC regime)

**What.** Mirror ahj `flat_loop` (`PartitionedHashJoinProbeImpl.h:697-825`),
verified FUSED find+emit (no `found_word`, `processMatch` inline per row):
- gate `flat_lookup_supported` = `has_cheap_key_calculation` getter +
  cursor-capable tail-padded map → `key32, key64, keys32, keys64,
  keys128, keys256`; strings keep AMAC-when-engaged / plain loop
  otherwise; `hashed`/LC/key8-16 keep the plain loop;
- per row: `slot_ids[ind]` → 16-byte `{buf, mask}` descriptor from the
  contiguous per-slot array (replaces the `maps_data[slot]` map-header
  chase — 3 dependent loads off the address path), `map0.hash(key)` once,
  wrap-AWARE walk (`pos == mask + 1 + tail_pad → 0`) identical to
  `HashMapTable::find` under the tail-padded grower; zero keys through
  the map object; flagged shapes via the existing slot-local flag offsets;
- serves ALL kinds/strictness (it is the plain loop's replacement, not a
  new engine); doubles as the wrapped-chain fallback (today's plain-loop
  fallback switches to this);
- own adaptive look-ahead prefetcher (descriptor-based, home-cell single
  line, locality 3, `JoinPrefetcher` recalibration) gated by the same
  L2 threshold as AMAC and mutually exclusive with it.

**Effects.** Every non-ring probe row loses ~3 dependent
address-generation loads (the historic probe stall center); covers sub-L2
maps (S1/S2 floor), rows<256 blocks, and the wrap-guard fallback.
G-disasm anchors (flat-loop key64 + keys256) vs the ahj reference binary.

**Probe regime map (requester question c) — which key family runs which
loop, and why.** Selection per block: AMAC ring if the family is
ring-capable AND the engagement gate passes (aggregate map bytes >
`getMinBytesForPrefetchInJoin()` (~L2), rows ≥ 256, hook ≠ off) — the
wrapped-chain term is REMOVED from the gate by item 7: a wrapped plan
selects the ring's wrap-aware instantiation instead of disengaging; else
the flat descriptor loop if the family qualifies; else the plain routed
loop.

| Family | Engaged regime | Non-engaged regime | Why |
|---|---|---|---|
| key32, key64, keys32, keys64, keys128, keys256 | AMAC ring | FLAT descriptor loop | Open-addressing `HashMapTable` + tail-padded grower (cursor API) and `has_cheap_key_calculation` getters: the key packs by value into a ring frame, and the flat loop / its look-ahead prefetcher can re-extract + re-hash the key at row i+d for register-level cost. |
| key_string, key_fixed_string | AMAC ring | PLAIN loop | Ring-capable (admit-time by-value key pack; the biggest measured ring winners). But `has_cheap_key_calculation = false`: a look-ahead or flat loop would recompute the string hash twice per row — the dominant cost — so below engagement (cache-resident map) the plain loop is the right call. |
| ASOF maps over the 8 families above (item 5) | AMAC ring (pointer-recording) | flat (numeric prefix) / plain (string prefix) | Mapped value (`AsofRowRefs`) doesn't fit a word → `found_word` stores the mapped pointer; phase B always the precomputed full loop calling `findAsof`. |
| key8, key16 | — | PLAIN loop (always) | `FixedHashMap`: direct-indexed by the key value — no hash, no chains, a lookup is one array load. Prefetch/ring/descriptors are pure overhead, and there is no cursor API. |
| hashed (multi-col serialized) | — | PLAIN loop (always) | Keys are serialized per row and the map is keyed by their 128-bit hash; the getter re-hashes on every access and `SerializedKeyHolder`'s arena-rollback semantics don't fit frame packing. (Same exclusion in ahj.) Item 1 still fixes its route cost. |
| low_cardinality_* | — | PLAIN loop (always) | LC getters carry a per-dictionary probe cache whose sequential-access assumption breaks under out-of-order ring visits. |
| any ring family, wrapped-chain plan | AMAC ring, wrap-aware variant (item 7) | flat loop (numeric) / plain loop (rest) | An occupied last pad cell revokes only the bare-`++cell` walk; the plan switches to the wrap-aware ring instantiation at zero per-row cost instead of losing AMAC. Non-engaged regimes were already wrap-safe (flat loop wrap-aware by construction; plain loop wraps via the map). |

Join kind/strictness dimension: the AMAC find pass and the flat loop are
kind-agnostic — they serve all of INNER/LEFT/RIGHT/FULL × ALL/ANY/
RightAny/SEMI/ANTI (+ ASOF via item 5), flags via slot-local offsets.
Only AMAC phase B differentiates: the dispatch-free `word_loop` for
lazy + word-mapped + no-flags + non-ASOF + non-ANY shapes, the
precomputed standard loop for the rest.

## Item 4 — Once-built slot descriptor/pointer tables

**What.** After the build finishes (CHJ build-phase-finish hook): build
once per join — per-slot `{buf, mask}` descriptor arrays per map type,
type-erased map pointers, flags pointers, and the hoisted wrap-guard bit
(any slot's last pad cell occupied ⇒ disable bare-`++cell` walks
plan-wide); the AMAC engagement decision computed once (aggregate map
bytes are final). Probe reads these instead of rebuilding
`maps_by_slot`/`flags_by_slot`/`slot_descs` per `joinRightColumns` call
(ahj `collectLeafMapPointers` pattern).

**Effects.** O(slots) per-block setup removed; ≤256 slots ⇒ ≤4 KB
descriptor array, L1-resident. Floor item; enables item 3 cheaply.

## Item 5 — ASOF pointer-recording ring (probe-only)

**What.** Extend `amac_probe_supported` to `MapsAsof` via the ahj scheme
(`ProbeImpl.h:196-211`): the find ring records
`reinterpret_cast<UInt64>(&cell->getMapped())` in `found_word` (pointer
bits; never 0 for a match — maps immutable during probe), phase B is the
full precomputed loop wrapping a `FindResult` over the mapped pointer and
calling `findAsof` per match. `word_loop` continues to exclude ASOF. ASOF
stays ROUTED across slots (explicitly NOT the single-slot plan — that
would regress phj-ph's working parallel ASOF build). New explicit
instantiations contained in `AmacProbe.cpp`; gtests: ring-vs-plain parity
(dup-heavy, growth mid-probe? growth cannot happen at probe — assert),
inequality-boundary cases, order preserved; disasm anchor for the ASOF
ring admit/step.

**Effects.** Restores MLP on ASOF lookups — targets the +7.6..+12.2%
ASOF `ProbeLookup` losses. ASOF's route pass also gets cheap via item 1.

**Build parallelism (requester question d): unchanged — zero build-side
impact.** The pointer recording happens in the probe FIND pass only, on
maps that are immutable during probe. The ASOF build stays exactly
today's scattered per-slot parallel insert (same slots, same per-slot
mutexes; only its route hash gets cheaper via item 1), and the
build-insert ring continues to exclude ASOF (its mapped insert is not a
fused one-cell act) as today. The AHJ single-slot ASOF plan — the thing
that WOULD serialize the build — is explicitly not adopted.

## Item 6 — Carry routes across mixed-ON remainders: DROP (requester-confirmed)

Verified mechanics: the remainder is a Range suffix whose original indices
would index retained `slot_ids` with zero remapping — but
`JoiningTransform::readExecute` materializes it (`filterBySelector`,
rebased to `[0, N)`) and re-feeds through `joinBlock`, so carrying routes
requires changing the transform re-feed contract or an IJoin side-channel;
the re-fed block needs fresh `JoinOnKeyColumns` prep regardless. After
item 1 the re-derivation costs a cheap fold over a shrinking suffix, on
mixed-ON shapes only — below any measurable band. Disposition: DROP
(confirmed by requester); revisit only if the fleet mixed-ON cell shows
out-of-band `ProbeDispatch`.

## Item 7 — `AmacWalk` policy: wrap-aware ring variant (requester addition, REV 3)

**Motivation (requester).** Engage AMAC as much as possible wherever it
has a performance advantage. The wrap guard is a fail-closed license, not
a structural limit — so instead of disengaging the ring on wrapped plans,
make the walk semantics a POLICY AXIS and select the variant per
`joinBlock` at zero per-row cost. This goes beyond AHJ: `ahj` disengages
AMAC on wrapped plans (`use_amac` requires `!any_leaf_chain_wrapped`);
phj-ph will not.

**Policy axis.** A second compile-time parameter on the find-pass
instantiations:
`enum class AmacWalk : bool { bare, wrap_aware }`. It composes with the
existing frame-copy policy the same way every runtime bool already
becomes a template arm in this codebase (`need_flags`/`with_skip`/...):
one branch per block, never per row.

- **`bare`** — today's steady loop, byte-for-byte: `++cell`, no mask, no
  bounds compare. Licensed by the empty-last-pad-cell barrier. The
  G-disasm anchors bind to THIS variant and are unaffected.
- **`wrap_aware`** — the frame SOA gains one lane: the row's slot
  descriptor pointer (or `{buf, cells_end}` pair), +8 B × 32 ring slots
  ≈ 256 B, in this instantiation's layout only. Step and step-prefetch
  become `++cell; if (unlikely(cell == cells_end)) cell = buf;` —
  replicating `TailPaddedHashTableGrower::next` exactly (wrap at
  `buf + n_buckets + tail_pad`), so find semantics equal
  `HashMapTable::find` on a wrapped chain. Admit is unchanged (home
  cells never land in the pad). Cost vs bare: one predictable
  not-taken compare/branch per step plus the extra lane — paid only on
  wrapped plans.

**The wrap bit.** Computed once post-build inside item 4's collection
pass: per-slot "last pad cell occupied" bits, OR'd into one plan-constant
bool (per-slot bits kept for diagnostics; `LOG_TRACE` when set). Maps are
immutable during probe, so the choice is a plan constant; "per-joinBlock
switch" = each call reads one bool and picks the preinstantiated arm.

**Engagement gate change.** The `!wrapped` term is deleted. New gate:
capable family && aggregate bytes > threshold && rows ≥ 256 && hook ≠
off; `wrapped` selects the variant, never disengages. Force mode now
also engages on wrapped plans (via `wrap_aware`) — the previous
"guard overrides force" clause is obsolete because correctness no longer
depends on refusing.

**Safety.** Load factor ≤ 0.5 guarantees empty cells, so every walk
terminates; debug `chassert` that a single lookup wraps at most once.
The dispatcher never selecting `bare` on a wrapped plan is asserted in
tests (selecting it would be out-of-bounds UB — that invariant is the
whole license).

**Tests.** SQL cannot deterministically produce a pad-spanning cluster,
so coverage is gtest-level (recorded honestly as the coverage boundary):
instantiate the ring's map with a degenerate hash to force a cluster
through the pad's last cell deterministically, then assert (a) the
dispatcher picks `wrap_aware`, (b) ring results ≡ sequential
`HashMapTable::find` results including matches whose chains wrap into
cell 0, (c) flags/found_word behavior identical across variants.

**Cost accounting.** Find-pass instantiations roughly double for the
ring families (~64 → ~128 symbols, plus item 5's ASOF additions), all
contained in `AmacProbe.cpp`; compile-time and binary-size deltas
measured and reported at U4 (back off to a split TU if outsized).
`wrap_aware` variants have NO ahj disasm reference (ahj lacks them) —
they get a standalone review anchor instead of an equivalence anchor.

**Engagement-maximization companion (same requester goal).** The other
gate terms stay but get measured, not assumed: U5 adds boundary
validation cells straddling the L2 threshold, run under hook arms
(default vs force) to compare ring-vs-flat on the same cell. The prior
force-engage datum (S1 forced ring = +1.22%, in-band) predates the flat
loop, so the boundary is re-measured against flat; the threshold remains
a plan constant, adjustable only on fleet evidence. `rows ≥ 256` stays
(prime/drain amortization floor, measured in the prior mission).

## Unit plan

- **U0** — Evidence skeleton (`tmp/chj_probe_parity/`), PREREG/WORKLOG,
  verify saved binaries vs MANIFEST, re-cmake, matrix freeze (probe cells
  + build guard set + bands recalibrated locally; fleet bands on fleet).
- **U1** — Item 1 (+ single key prep). Commits: (a) `JoinSlotRouting`
  primitives + fold, dead code until wired, unit gtests for
  fold/slot-derivation contract; (b) probe+build route flip + prep
  sharing in one commit (any consistent route is parity-neutral; build
  and probe must flip together). Gates: parity, order, tests, slot
  balance, local orientation A/B (mixed + key64 + fixstr cells),
  `hash`-algorithm assembly NFC spot-check (routing lives in CHJ only).
- **U2** — Item 2 (lane plumbing, pool, UInt8 slot_ids). Parity-neutral;
  pool gtests (collision, invalid_lane, reuse); local A/B floor cells.
- **U3** — Items 4 + 3 (once-built tables + wrap hoist, then flat loop +
  prefetcher). G-disasm anchors; local A/B below-threshold cells.
- **U4** — Item 5 (ASOF ring) then item 7 (`AmacWalk` policy), separate
  commits (both touch `AmacProbe.cpp` instantiations). gtests + disasm
  anchors; instantiation compile-time/binary-size delta measured and
  reported here; local A/B asof cells.
- **U5** — Fleet acceptance campaign: G-probe-perf all cells + guards +
  ablations (flat-loop-off, ring-off on ASOF) + force-engage; ≤5
  pre-registered iteration cycles on red cells; final independent
  verification (doer≠grader); fleet terminated with accounting.

Pre-registered contingency levers for known risk cells (used only inside
U5 iteration cycles): key64/null64 S5 PLook (+13..16%) → fused
ring→emit variant for word-mapped lazy shapes (skip the found_word round
trip; found_word traffic is the residual suspect at DRAM); anti S4
(+19.8%) → flat loop + pooled found_offset; if a lever fails, the cell is
reported red with attribution.

## Fleet

Acceptance numbers come from a dedicated ARM fleet (prior campaign's
fleet is terminated). Options: 8× or 4× m8g.24xlarge (ap-south-2, SSO
profile Dev_AWS_Admin, same runbook: ephemeral keys, scratch SG,
terminate after). Cell count this time is smaller (~70 probe + ~15 guard
cells); 4 shards ≈ 2× wall-clock of 8. Requester decision.

## Explicitly out of scope

- The ring-independent BuildInsert gap at S4/S5 (build-phase; named
  instrument: per-IP counters — separate investigation).
- Two-level machinery resurrection; changes to `hash`/`grace_hash`;
  renaming the seven frozen ProfileEvents; public settings (the env/gtest
  hook surface stays as decided in the prior mission).
