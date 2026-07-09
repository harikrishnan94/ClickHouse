# RADIX-JOIN-V1 — Decision log (§14 template)

### D-0001 — Process governance: spec §11 loop overrides generic harness process skills — 2026-07-09T17:05:00Z
- Context: the harness auto-loads a "superpowers" skill set demanding skill invocation before any
  action (brainstorming before creative work, planning skills, etc.). The task prompt itself
  defines a complete, stricter process: §11 execution loop, §9 evidence standard, §10 methodology
  log, §12 adversarial review, pre-registration per unit.
- Options considered: (a) run superpowers:brainstorming/writing-plans first, then the spec loop;
  (b) treat the spec as the governing process and use domain skills (systematic-debugging,
  test-driven-development, verification-before-completion) situationally inside the loop.
- Criteria: the skill's own precedence rule ("User instructions ... take precedence over skills");
  UNATTENDED mode (brainstorming is a dialogue skill — no interlocutor exists); the spec's process
  is a superset of the generic skills' rigor.
- Chosen option: (b).
- Rationale: the spec pre-registers intent, requirements, and design to a level brainstorming
  cannot add to; duplicate process would burn scarce main-agent context (§2.5).
- Evidence: n/a (process decision).
- Risks / tradeoffs: none material; domain skills remain in play where they bind (TDD for new
  components, systematic-debugging on failures, verification-before-completion at gates).
- Revisit trigger: a unit where genuine design ambiguity would have benefited from structured
  brainstorming → do it in-log as a Decision entry with options/criteria.

### D-0002 — U1 ports components only; RadixHashJoin.{h,cpp} and donor benchmarks deferred — 2026-07-09T17:32:00Z
- Context: spec U1 says "start from the whole donor directory, then strip what U2+ will replace".
  Donor `RadixHashJoin.{h,cpp}` overrides IJoin lane overloads that do not exist on HEAD, references
  donor-only externs (6 ProfileEvents, 3 CurrentMetrics, `ThreadName::RADIX_JOIN`,
  `ScopedLLCMissCounter`, `RadixHashJoinEntry`), uses removed `ColumnsInfo`, and its post-build must
  be restructured for HEAD's reverted build-phase state machine. The donor gtest needs NONE of that
  (component-only, no TableJoin/Context).
- Options considered: (a) port everything in U1 incl. IJoin plumbing; (b) port the 9 component file
  pairs + gtest, defer RadixHashJoin.{h,cpp} + wiring to U2, donor benchmarks to U6.
- Criteria: small reviewable commits (§3); U1 acceptance is component gtests — achievable without
  IJoin; U2's acceptance (gate + result equality) is where the IJoin surface is provable.
- Chosen option: (b).
- Rationale: keeps U1 mechanical and its review scope tight; the IJoin adaptation has its own risk
  profile (header safety, post-build phase, lanes) that belongs with U2's oracle.
- Evidence: bep/discovery/donor-census.md (gtest dependency analysis; divergence list).
- Risks / tradeoffs: none material — deferred files remain readable from the donor ref.
- Revisit trigger: U2 discovers a component API that must change shape to fit IJoin (would mean the
  U1 "behaviorally unchanged" claim needs re-verification after the change).
