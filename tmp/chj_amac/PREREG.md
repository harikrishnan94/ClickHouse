# PREREG — pre-registrations for gated actions

Approved plan: `/home/ubuntu/.claude/plans/mission-amac-zany-hammock.md`
(user-approved 2026-07-27). Entries are committed BEFORE the implementing
change or gated run; ordering is auditable in `git log` because this file is
committed on-branch.

Template:

    ## PREREG-<n> — <gate or decision> — <UTC date> — written at HEAD <hash>
    Expectation: <one falsifiable, quantified sentence>
    Invocation: <exact command line(s)>
    Refutation criterion: <exact output/condition that falsifies>
    Action on refutation: <stop | descope | investigate with named next probe>

---

## PREREG-001 — Unit 1 reference builds — 2026-07-27 — written at HEAD 6cdee22a455

Expectation: a baseline binary built in a hardlinked-submodule worktree at
`concurrent-hash-join-profile-events` (`a05f3ee81ff`) and an `ahj` reference
binary at `cf465cfbe23`, both RelWithDebInfo clang-22 aarch64 with the
candidate's cmake flags, build with zero errors and report the expected
`GIT_HASH` after a fresh configure; sha256 of each recorded in
`bins/MANIFEST.tsv` is the identity of record (embedded GIT_HASH is
configure-time-stale by prior finding and is secondary).

Invocation:
  git worktree add --no-checkout /mnt/ch/ClickHouse-concurrent-hash-join-profile-events concurrent-hash-join-profile-events
  (create-worktree skill hardlink recipe), then
  ninja -C <worktree>/build/reldeb clickhouse > <worktree>/build/reldeb/build_baseline_a05f3ee.log 2>&1
  (same for /mnt/ch/ClickHouse-ahj @ ahj, log build_ahj_cf465cf.log);
  subagent analyzes each log.

Refutation criterion: any build error, or a worktree whose
`git rev-parse HEAD` differs from the pinned hash, or missing submodule pins
that cannot be resolved by `git -C /mnt/ch/ClickHouse submodule update --init`.

Action on refutation: stop and diagnose before any Unit 2 work; a baseline
binary from any other source is NOT acceptable.

## PREREG-002a — Unit 1 harness gate G-parity (declared up front) — 2026-07-27 — written at HEAD 6cdee22a455

Expectation: on the pre-change candidate (HEAD 6cdee22a455) vs baseline, the
full query matrix (~600-900 queries; 10 key families incl.
embedded/terminating-zero and empty strings; kinds INNER/LEFT/RIGHT/FULL ×
valid strictnesses incl. RightAny and non-equi-ON MapsAll variants;
join_use_nulls × {0,1}; dup-heavy builds; threads {4,32}) prints `PARITY OK`
— ORDER BY-normalized results are byte-identical between arms (the probe
scatter reorders rows but not multisets).

Invocation:
  bash tmp/chj_amac/parity/run_parity.sh tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin tmp/chj_amac/bins/clickhouse-candidate-6cdee22a455.bin

Refutation criterion: any parity divergence on the UNCHANGED candidate — that
would mean the harness or the scatter design is broken (investigate before
proceeding).

Action on refutation: stop; fix the harness (or report the pre-existing
product defect) before any Unit 2 commit.

## PREREG-002b — Unit 1 harness gate G-order power check (declared up front) — 2026-07-27 — written at HEAD 6cdee22a455

Expectation: the per-block tag-monotonicity check at max_threads=96 FAILS
against BOTH the baseline binary and the pre-Unit-3 candidate (both scatter
probes).

Invocation:
  bash tmp/chj_amac/order/run_order.sh <bin>            # power check: run on both arms

Refutation criterion: the order check passing on either scatter binary.

Action on refutation: stop — the oracle is too weak and must be strengthened
before it may gate anything; fix the harness (or report the pre-existing
product defect) before any Unit 2 commit.

## PREREG-002c — Unit 1 harness gate noise band (declared up front) — 2026-07-27 — written at HEAD 6cdee22a455

Expectation: same-binary A/A on 6 representative cells verdicts TIE on every
cell; the per-shape band is frozen as max(3%, observed same-binary spread)
before any perf claim; a deliberate A≠B selftest must produce a non-TIE
verdict.

Invocation:
  python3 tmp/chj_amac/fleet_ab.py sweep --local --aa   # noise band mode

Refutation criterion: any A/A cell outside its band, or the deliberate A≠B
selftest verdicting TIE.

Action on refutation: stop; fix the harness (or report the pre-existing
product defect) before any Unit 2 commit.

## PREREG-003 — env facts for perf venues — 2026-07-27 — written at HEAD 6cdee22a455

Local orientation host (never acceptance evidence): aarch64 Graviton, 96
cores (`uname -m` verified by exploration agent; lscpu digest to be recorded
at first fleet_ab --local run). Acceptance venue: 8× m8g.24xlarge ARM fleet,
ap-south-2, launched in Unit 4; host list + per-shard `uname -m`/`lscpu`
digests will be appended here at launch, per mission requirement.
