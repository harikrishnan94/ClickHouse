# Hygiene report — reduce-complexity + humanize, report-only

Commits reviewed (branch `phj-ph`, both at tip, working tree clean for all reviewed files):

- `7ffe3f801c2` — Correct the order gate's squash oracle to baseline-differential
  (`tmp/chj_amac/order/run_order.sh`, `tmp/chj_amac/order/SELFTEST.md`)
- `5b276c5fb88` — Make the parallel_hash probe order-preserving with an AMAC find ring
  (23 files; new `HashJoinRoutedMethods{.h,Impl.h}`, 6 `RoutedHashJoin*.cpp` TUs,
  `AmacProbe.{h,cpp}`, `ConcurrentHashJoin` flip, `HashJoin`/`KeyGetter`/`ProfileEvents` edits, gtests)

Mode: report-only (per task). No file outside this report was modified. Tree state: 0 findings
applied.

Constraints honored: no finding proposes a change to the steady-loop codegen of the ring or the
routed loop (disassembly-gated by `tmp/chj_amac/disasm/U3_probe_anchors.md`); the
baseline-differential oracle design is taken as settled; `ahj`-inherited comment wording is
treated as deliberate. Line numbers cite the committed files (identical to the working tree).

---

## 1. reduce-complexity × 7ffe3f801c2 — CLEAN

Base: `7ffe3f801c2~1` (= `837247e57fc`). Scanned: the whole diff — the argument parser rewrite
(`for`→`while`/`shift`), `parse_baseline_reference`, the verdict-loop reclassification, the
summary/final-line changes, and the SELFTEST §11 addition. No finding survived.

Strongest candidates and what killed them:

- **Reclassification isolation** — the `POWER` computation was a candidate for entanglement with
  the new `verdict` rewriting; refuted: `run_order.sh:805` reads the raw `${CHECK_RESULT[$name]}`,
  not the reclassified `verdict`, and reclassification is additionally dead in `--expect-fail`
  mode (`run_order.sh:781` guards on `EXPECT_FAIL = 0`; the flag combination is rejected at
  `run_order.sh:140-146`). The two modes are cleanly separated.
- **Parse-all-verdicts vs squash-only** — `parse_baseline_reference` records every per-check
  verdict though only `_squash` names are consulted; kept: the extra verdicts feed the
  fail-closed validation (no-parseable-lines / no-squash-verdicts / conflicting-duplicates
  FATALs, `run_order.sh:170-196`) — inherent to the adversarial fail-closed spec, not accretion.
- **`ref_verdict` computed before the sibling test** (`run_order.sh:785`) — one line of
  micro-ordering; killed as a nitpick.
- The regex contract was verified end-to-end: the reference parser's pattern matches the
  emission format at `run_order.sh:531` (`log "check $name (T=$threads): $result [$final]
  rows=$rows"`), and `logs/gate_002b_baseline.log` parses to 17 verdicts (`ok=9 fail=8`),
  matching the CORRECTION block's claim.

---

## 2. reduce-complexity × 5b276c5fb88

Base: `5b276c5fb88~1` (= `7ffe3f801c2`). Two report-only findings; the primary new surface
(routed methods + find ring, 30 + 64 instantiations, the `precomputed`/`word_loop` seams) is
intentional, perf- and disassembly-gated design and was not second-guessed.

### Finding R1 — Type `RoutedProbeContext` on `Map` and delete the `maps_untyped` adapter; its justifying comment is false

- **Where:** `src/Interpreters/HashJoin/HashJoinMethods.h:26-28` (comment), `:33`
  (`const void * const * maps_by_slot`); `src/Interpreters/HashJoin/HashJoinRoutedMethodsImpl.h:157-163`
  (the `maps_untyped` copy); `src/Interpreters/HashJoin/HashJoinMethodsImpl.h:950-954`
  (the `static_cast<const Map *>` cast-back in `map_for_row`).
- **Shape:** adapter between two halves of the same change (both introduced by this diff) →
  unify the shapes, delete the adapter. Plus stale narrative: the comment says the maps are
  untyped "because the context is built before the map-type dispatch" — false in the final
  state. Receipt: `grep -rn RoutedProbeContext src/` returns exactly one construction site,
  `HashJoinRoutedMethodsImpl.h:164`, which is inside `joinRightColumnsRouted<KeyGetter, Map,...>`
  — *after* `switchJoinRightColumns`' `APPLY_FOR_JOIN_VARIANTS` dispatch, with `Map` in scope;
  the sole consumer `joinRightColumnsWithAdditionalFilter<KeyGetter, Map>` is also templated on
  `Map` (`HashJoinMethods.h:220-230`). Nothing in the final code needs the erasure.
- **Impact:** cognitive load — the comment teaches a reader a control-flow ordering that does
  not exist, and the void-cast round trip plus the per-block `std::vector<const void *>` copy
  must be re-verified as benign by every reader of the mixed-ON path.
- **Cleaner shape:** `template <typename Map> struct RoutedProbeContext` with
  `const Map * const * maps_by_slot`; construct from `maps_by_slot.data()` directly (deleting
  `maps_untyped`, 4 lines); `map_for_row` loses the cast; the default argument becomes
  `const RoutedProbeContext<Map> * routed = nullptr`. All edits confined to the three in-diff
  files. Codegen-safety: this touches only the additional-filter (mixed-ON) path in
  `joinRightColumnsWithAdditionalFilter` — not `joinRightColumns`' routed loop and not
  `amacFindPass`, so none of the three gated probe anchors is affected; a typed load compiles
  identically to the cast anyway.
- **Behavior preservation:** type-level only. Coverage: the mixed-ON routed path is exercised
  by the shape-matrix smoke (`tmp/u3_smoke/shape_matrix.out` — inner/left/right/full mixed-ON,
  per `U3_DRAFT_NOTES.md`) and the parity gate, but by no checked-in gtest — hence demoted to a
  suggestion rather than a remove-verdict edit.
- **Severity × disposition:** suggestion / before the eventual PR (at minimum the comment must
  be corrected — see humanize H1, which this finding subsumes).
- **Status:** report-only (user-requested report-only run).

### Finding R2 — Build the `slot_joins` pointer vector once per join, not once per probe block

- **Where:** `src/Interpreters/ConcurrentHashJoin.cpp:383-405` — `RoutedJoinResult` owns
  `std::vector<const HashJoin *> slot_joins` (`:385`) and refills it from `hash_joins` in its
  constructor (`:400-403`), i.e. per `joinBlock` call.
- **Shape:** derivable state / adapter between the change's two halves — `joinRoutedBlock`
  wants `const std::vector<const HashJoin *> &`, `ConcurrentHashJoin` has
  `std::vector<std::shared_ptr<InternalHashJoin>>`; the adaptation runs per block instead of
  once. Receipt: `hash_joins` is fully populated in the constructor (`:189`, `:217`) and never
  resized afterwards; elements are only moved out in the destructor's parallel-teardown path
  (`:250`), after which no result may be alive under either shape — the deleted
  `ConcurrentHashJoinResult` relied on exactly the same lifetime by holding
  `const std::vector<...> &`.
- **Impact:** cognitive load (a reader must argue the per-result copy's lifetime safety that a
  reference to a stable member states directly) plus one ≤256-entry allocation per probe block.
  Off the steady loop entirely (constructor code), so no anchor interaction.
- **Cleaner shape:** a `std::vector<const HashJoin *>` member of `ConcurrentHashJoin` filled
  once at the end of the constructor; `RoutedJoinResult` holds a `const &` to it (the exact
  pattern of the deleted `ConcurrentHashJoinResult`); its constructor loop disappears.
- **Behavior preservation:** none observable; covered by all 10 gtests and the parity gate.
- **Severity × disposition:** nit / fine as a follow-up.
- **Status:** report-only (user-requested report-only run).

### Dropped candidates (commit 5b276c5fb88)

- Ring `slot` array written in the flagless arm (`AmacProbe.cpp`, `start`) — constraint-blocked:
  the disasm gate counts the 4 admit stores (key/row/slot/cell) as matching the `ahj` reference;
  removing the store for `!need_flags` changes gated steady-loop codegen.
- `selectDispatchBlock` forward declaration (`ConcurrentHashJoin.cpp:342-343`) — needed:
  definition at `:621`, probe use at `:439`, and a second (build-side) consumer at `:715`.
- `total_map_bytes` computed in both `joinRightColumnsRouted` and `joinRightColumns` — different
  paths, 3 lines each; below any duplication threshold.
- `expectedRows` vs `expectedRowsForProbe` in the gtest — deliberate test duplication; both
  widely used; tests favor clarity over DRY.
- `RoutedHashJoinMethods::joinBlockImpl` re-stating `HashJoinMethods::joinBlockImpl`'s
  `AddedColumns`/remainder/`HashJoinResult` scaffolding — intentional parallel surface (the
  scatter-side original must keep serving plain `HashJoin`); unification is a signature-crossing
  restructure with the highest regression weight, no wrong-abstraction evidence.
- `chassert(onexprs.size() == 1)` followed by a `for` over `onexprs`
  (`HashJoinRoutedMethodsImpl.h:24-30`) — mirrors the reference implementation for diffability.
- `RingBase::isActive`/`deactivate`/`rowAt`/`may_grow`/`copy_into_frame` — policy-contract
  conformance consumed by `amacRun` (`AmacRing.h:141-142`, `:168`, `:197-241`); exempt.
- Per-block `maps_by_slot`/`flags_by_slot` vector builds in `switchJoinRightColumns` — the
  typed pointers depend on the dispatched map type; caching would add cross-call state to a
  const probe path for no measured cost.
- `[[maybe_unused]]` on `slot_joins` in the routed `joinRightColumns` — genuinely used under
  `if constexpr` (`HashJoinRoutedMethodsImpl.h:257`, `amacEnabled`).

---

## 3. humanize × 7ffe3f801c2

The oracle-correction prose was checked claim-by-claim against the logs it cites: the
side-by-side table matches (e.g. `inner_all_k_squash` baseline `FAIL 23/83` vs candidate
`FAIL 27/83`), re-proof (a)/(b) final lines match `logs/gate_u3_order2.log` /
`logs/gate_baseline_normal.log` verbatim, and (a)'s "8 × SOURCE-ARTIFACT, zero NOT-RECLASSIFIED"
matches (grep counts 8 and 0). Two findings.

### Finding H1 — SELFTEST §11 cites re-proof (c) for the teeth-on-scatter claim, but (c) runs on the baseline

- **Where:** `tmp/chj_amac/order/SELFTEST.md:483` — "On scatter binaries the squash checks still
  fail for the real cross-piece reason, so the `--expect-fail` power check keeps its teeth
  (re-proof (c) below)."
- **Defect:** re-proof (c) (`SELFTEST.md:502-508`) is `--expect-fail` on the *baseline* binary
  (`logs/gate_002b_baseline2.log`) — a valid unchanged-power-mode consistency re-run (it
  reproduces the pre-correction `gate_002b_baseline.log` run exactly), but it contains no
  scatter binary, so it cannot carry the "still fails scatter binaries" claim the sentence
  attaches it to. The actual scatter evidence is the pre-correction run
  `logs/gate_002b_candidate.log` (final line `ORDER POWER-CHECK OK`, verified), which stays
  valid precisely because power mode is unchanged — and `run_order.sh:461-463` cites it
  correctly. A reviewer walking the citation chain hits a receipt that does not support the
  sentence.
- **Fix (text-only):** at `SELFTEST.md:483`, cite both facts to their own receipts:
  "...keeps its teeth: the pre-correction scatter run `logs/gate_002b_candidate.log` still
  applies because power mode is code-unchanged (re-proof (c) below re-runs the baseline
  power check and reproduces its pre-correction verdict)." Optionally retitle (c) from
  "unchanged power mode" to "power mode unchanged: baseline re-run reproduces
  `gate_002b_baseline.log`". The commit message carries the same shorthand ("the power check
  keeps its teeth"); immutable, and defensible given `gate_002b_candidate.log` — file-level fix
  only.
- **Severity:** suggestion / before the gate chain is next audited.

### Finding H2 — The mechanism + "WHY THIS IS A CORRECTION" argumentation is duplicated near-verbatim between the script and SELFTEST §11

- **Where:** `tmp/chj_amac/order/run_order.sh:436-464` (CORRECTION block: mechanism paragraph,
  evidence list, WHY paragraph at `:451-457`) vs `tmp/chj_amac/order/SELFTEST.md:415-427` and
  `:477-484`.
- **Defect:** one fact, two owners — the two copies have already drifted slightly (the
  "three independent layers" enumeration is phrased differently), and every future oracle
  change must be argued in both places. "One comment per fact."
- **Fix (text-only):** keep the *operational* rule and the fail-closed conditions in the script
  (they make the gate self-describing at the point of use) and replace the mechanism/WHY
  argumentation there with a two-line pointer to `SELFTEST.md` §11, which owns the evidence
  table and re-proofs. Counterweight, honestly stated: a fully self-describing gate script is a
  deliberate property of this harness — this is the author's call.
- **Severity:** nit / question for the author.

Not findings: the comment density matches the harness's established dialect; the fail-closed
error messages are load-bearing, not theater; the `Co-Authored-By` trailer is mandated
attribution and stays (disclosure gate).

---

## 4. humanize × 5b276c5fb88

The new C++ comments were spot-verified against the code they annotate (mapped-word encoding vs
`RowRef::encode`/`RowRefList`, slots ≤ 256 claim vs `ConcurrentHashJoin.cpp:177`, ring-contract
fields vs `AmacRing.h`, `preservesLeftBlockOrder` inherited default vs `IJoin.h:153`): accurate
except the one below. Three findings.

### Finding H1 — False comment: `RoutedProbeContext` is *not* "built before the map-type dispatch"

- **Where:** `src/Interpreters/HashJoin/HashJoinMethods.h:26-28`.
- **Defect:** the stated reason for the untyped `maps_by_slot` is contradicted by the code —
  the only construction site (`HashJoinRoutedMethodsImpl.h:164`) sits inside the already
  map-typed `joinRightColumnsRouted`. The comment reads like the residue of an earlier draft
  where the context was built higher up. A comment that misdescribes control flow is worse than
  none.
- **Fix:** apply reduce-complexity R1 (type the struct; the sentence disappears), or minimally
  rewrite to the true, defensible rationale — e.g. "kept untyped so the non-template struct can
  be declared once here and threaded through `joinRightColumnsWithAdditionalFilter` as a plain
  pointer; the typed consumer casts back" — dropping the false "before the dispatch" clause.
- **Severity:** issue (for the comment) / before the eventual PR.

### Finding H2 — Collateral says "5 new gtests"; the commit adds 6

- **Where:** `tmp/chj_amac/WORKLOG.md:445` ("5 new gtests") and `tmp/chj_amac/U3_DRAFT_NOTES.md:22`
  ("5 new tests").
- **Defect:** receipt accuracy. `git show 5b276c5fb88~1` has 4 `TEST(` macros in
  `gtest_concurrent_hash_join_amac.cpp`; the committed file has 10 — 6 new
  (`ProbeRingParityUIntKeys`, `ProbeRingParityStringKeys`, `ProbeRingRightAllNonJoinedParity`,
  `ProbeRingFullAllNonJoinedParity`, `ProbeRingRightAnySetUsedOnce`, `ProbeEmitsInLeftRowOrder`).
  The "10/10 gtests" figures elsewhere are consistent with 10 total. Possibly a stale count from
  before `ProbeEmitsInLeftRowOrder` landed, or a "5 scenarios" reading (Right/Full share
  `runFlaggedShapeParity`) — either way the plain reading is wrong, and these files are the
  mission's evidence trail.
- **Fix:** "6 new gtests" / "6 new tests" in both files (the commit message's "10/10 gtests" is
  fine as-is).
- **Severity:** nit / before the collateral is next relied on.

### Finding H3 — The "copy the word in the same visit" rationale is written twice

- **Where:** `src/Interpreters/HashJoin/AmacProbe.h:26-29` (the `amac_mapped_fits_word`
  contract comment) and `src/Interpreters/HashJoin/AmacProbe.cpp:19-23` (the `AmacFindPolicy`
  doc) — the "by the time [the emit] reaches the row, the cell line has usually left the cache
  and re-reading it through a recorded pointer would be a second random miss per row" sentence
  is near-verbatim in both.
- **Defect:** one fact, two comments in the same commit (this is internal duplication, not the
  deliberate `ahj` inheritance). They will drift.
- **Fix (text-only):** keep the full rationale on the header contract (its natural owner — it
  justifies the by-value word design) and shorten the policy doc's phase-A paragraph to
  "…copied by value into `found_word` (0 = no match; see `amac_mapped_fits_word` in
  `AmacProbe.h` for why the copy, not a pointer, is recorded)…".
- **Severity:** nit / fine as a follow-up.

### Dropped candidates (humanize, commit 5b276c5fb88)

- Comment density of the new files — matches the branch dialect (`AmacBuild.h`, `AmacRing.h`);
  every sampled comment carries a fact the code cannot (measurements, invariants, contracts).
- `ahj`-inherited comment wording — deliberate per task constraints.
- Assertion-message asymmetry (`ProbeRingParityUIntKeys` explains its two `EXPECT_EQ`s,
  `ProbeRingParityStringKeys`' identical assertions are bare) — the first occurrence documents
  the contract; duplicating the messages would be padding.
- `ASSERT_TRUE(a == b)` instead of `ASSERT_EQ` — mirrors the file's pre-existing tests (avoids
  printing multi-hundred-thousand-row containers on failure).
- `/// Do not hold memory for join_on_keys anymore` (`HashJoinRoutedMethodsImpl.h:55`) —
  verbatim mirror of the reference implementation (`HashJoinMethodsImpl.h:187`); deliberate
  diffability.
- Robustness both directions: no theater found (the new `chassert`s and the single
  `LOGICAL_ERROR` throw mirror the originals); no missing edge case found — `rows == 0`,
  single-slot (`slot_ids == nullptr`), zero-key sentinel, empty-string key, remainder re-feed
  and the chain-wrap guard are all explicitly handled and mostly test-pinned.
- Commit message and trailer — dense-with-receipts style matches the branch's history;
  `Co-Authored-By` is mandated attribution and stays.
