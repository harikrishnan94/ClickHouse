# WORKLOG — AMAC build/probe + order-preserving probe for `ConcurrentHashJoin`

Mission: implement (1) AMAC rings for build-insert and probe-lookup and
(2) an order-preserving routed probe for `ConcurrentHashJoin`
(`join_algorithm = 'parallel_hash'`) on branch `phj-ph`. Baseline arm for all
comparisons: branch `concurrent-hash-join-profile-events` @ `a05f3ee81ff`.
Approved plan: see `PREREG.md` header and the per-unit entries below.

Conventions: entries are per iteration (goal / done / verified-how / plan
changes), amended forward, UTC dates. Raw outputs inline when small, else
path + sha256. Secrets redacted as [REDACTED].

---

## 2026-07-27 — Session start, planning phase

Goal: produce the approved implementation plan.

Done:
- 3 Explore agents (phj-ph code state; `ahj` reference design; history docs +
  environment) and 3 Plan agents (routed probe; AMAC rings; harness/fleet/
  process). Findings recorded in the approved plan file
  (`/home/ubuntu/.claude/plans/mission-amac-zany-hammock.md`).
- Verified myself (not delegated): (a) `hashToSelector` on `phj-ph` routes by
  `hashes[i] & (num_shards - 1)` (ConcurrentHashJoin.cpp:522-530) — the same
  low bits `HashTableGrowerWithPrecalculation::place` uses (HashTable.h:301)
  → per-slot home-cell clustering; the two-level baseline routed by
  `getBucketFromHash` = bits 24..31 (TwoLevelHashTable.h:54); (b) upstream
  `master` shares one `StoredColumnsIndex` across slots
  (master ConcurrentHashJoin.cpp:232-237) — revert `69bf5c26c9f` dropped it.

Requester decisions recorded:
1. Fleet = launch 8× m8g.24xlarge (Dev_AWS_Admin, ap-south-2), terminate
   after the campaign.
2. AMAC diagnostic hook = C++-only (env `CLICKHOUSE_JOIN_AMAC` off/auto/force
   + gtest setters). No public setting. Consequence: no stateless SQL test
   for off/force paths; gtests carry off-path coverage.
3. Unit order = AMAC first; order fix lands with the AMAC probe (its
   in-order emit is the fix).
4. Join maps adopt the tail-padded grower (ring disassembly must replicate
   `ahj`); new gate G-hash-inband (12 `hash`-algorithm A/B cells).
5. New gate G-disasm: instruction-semantics equivalence of ring steady loops
   vs an `ahj`-HEAD reference binary (6 anchors).

Plan approved by requester via plan-mode approval (1 revision cycle).

## 2026-07-27 — U1.1: evidence skeleton

Goal: create `tmp/chj_amac/` skeleton (this file, PREREG.md, .gitignore) and
commit it.

Done: directories {bins, parity, order, hygiene, fleet/{ssh,results},
tests_srv}; .gitignore excluding binaries/keys/server dirs/logs; this
worklog; PREREG.md with U1 pre-registration.

Verified: `git status --porcelain` clean before staging (raw: empty output);
HEAD `6cdee22a4554e3935e26165837dedb2b3eb2362a`;
`concurrent-hash-join-profile-events` = `a05f3ee81ff8411759637fa367aad62e72726e71`;
`ahj` = `cf465cfbe23a14f982d1bc36510f3e311ce6379f`.

Deviation: the per-commit hygiene loop's G-build/G-parity re-run
does not apply to this commit — no source changed and the parity harness does
not exist yet (it is U1.5's deliverable). Hygiene report subagents still run.
