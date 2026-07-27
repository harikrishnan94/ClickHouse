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

## PREREG-004 — Unit 2 commit 1: slot-route decorrelation — 2026-07-27 — written at HEAD 91469b6b22e

Expectation: deriving the scatter selector from the hash's bits 24+ for
chained-map types (mirroring the two-level baseline's `getBucketFromHash`
distribution; `FixedHashMap` key8/16 keep low-bit routing) — applied to
build dispatch and probe dispatch together in one commit — leaves join
results byte-identical (G-parity stays green) and improves BOTH
`ConcurrentHashJoinBuildInsertMicroseconds`- and
`ConcurrentHashJoinProbeLookupMicroseconds`-attributed time outside the 3%
band on DRAM-resident cells at 128 slots (local orientation A/B, pre-fix
candidate `75d431b1d74` vs post-fix candidate, cells
`key64:probe.inner_all.S3.T96`, `str:probe.inner_all.S3.T96`,
`key64:build.inner_all.S3.T96`), because today slot routing and in-map
placement share low bits, clustering each slot's home cells on 1/slots of
positions (expected cluster ≈ load×slots cells with linear probing).

Invocation:
  bash tmp/chj_amac/parity/run_parity.sh tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin tmp/chj_amac/bins/clickhouse-candidate-<postfix>.bin
  python3 tmp/chj_amac/fleet_ab.py sweep --local --arm-a tmp/chj_amac/bins/clickhouse-candidate-75d431b1d74.bin --arm-b tmp/chj_amac/bins/clickhouse-candidate-<postfix>.bin --cells "key64:probe.inner_all.S3.T96,str:probe.inner_all.S3.T96,key64:build.inner_all.S3.T96" --calibration tmp/chj_amac/fleet/calibration_rows.json
  (arm B = post-fix; verdict LOSS for arm A = improvement by the fix)

Refutation criterion: any parity divergence; or no probe cell improves
outside the 3% band (the mechanism story is then wrong — investigate with a
chain-length probe before proceeding); build-side improvement is expected
but not required for acceptance of this commit (dispatch hash cost may mask
it — record whatever is measured).

Action on refutation: parity divergence → stop and fix before anything
else; no perf effect → keep the change only if parity-neutral AND
harmless, downgrade the "correlation defect" claim in REPORT.md to
refuted-lead, and investigate the chain-length hypothesis with a dedicated
probe before Unit 2 commit 2.

## PREREG-005 — Unit 2 commit 2: resumable cursor layer + tail-padded grower — 2026-07-27 — written at HEAD 844ee1a82dd

Expectation: rebinding the 8 chained join-map members (`key32`, `key64`,
`key_string`, `key_fixed_string`, `keys32`, `keys64`, `keys128`, `keys256`)
to `ResumableHashMap<HashMapTable<..., TailPaddedHashTableGrower<>, ...>>`
(grower rebind ONLY — the allocator stays `HashTableAllocator`; `ahj`'s
`ZeroingHashTableAllocator` is a separate recorded lead) and extracting
`applyBuildRowToMapped` shared by `Inserter` leaves both algorithms
result-identical (G-parity green) and `hash` performance in-band: the only
sequential-path codegen change is the walk advance (`next()` becomes
increment + compare-to-bufSize instead of increment + mask), and the local
orientation A/B of `join_algorithm='hash'` cells {key64,str,k256} × S3 ×
T{1,96} between the pre-change candidate (`844ee1a82dd`) and the post-change
build must verdict TIE on every cell (3% band).

Invocation:
  ninja -C build/reldeb clickhouse > build/reldeb/build_cursorlayer.log 2>&1
  bash tmp/chj_amac/parity/run_parity.sh tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin <post-change build>
  python3 tmp/chj_amac/fleet_ab.py sweep --local --arm-a bins/clickhouse-candidate-844ee1a82dd.bin --arm-b <post-change snapshot> --cells "key64:probe.inner_all.S3.T96.hash,str:probe.inner_all.S3.T96.hash,k256:probe.inner_all.S3.T96.hash,key64:probe.inner_all.S3.T1.hash" --calibration tmp/chj_amac/fleet/calibration_rows.json
  python3 .claude/tools/analyze-assembly.py --before bins/clickhouse-candidate-844ee1a82dd.bin --after <post-change snapshot> "<joinRightColumns/insertFromBlockImplTypeCase anchor symbols>"

Refutation criterion: any parity divergence; any `hash` cell losing outside
the 3% band; the assembly diff of the sequential insert/probe loops showing
changes beyond the walk-advance pattern and grower field layout (e.g. new
spills, extra loads in the per-row body).

Action on refutation: parity divergence or out-of-band `hash` loss → stop,
revert the grower rebind to the standard grower (mask-carried cursor
fallback documented in the plan), and surface the tradeoff to the requester
(their decision 4 traded `hash` neutrality for disassembly fidelity on the
explicit condition the in-band gate holds); unexplained assembly changes →
investigate before any ring work builds on the layer.

## PREREG-003 — env facts for perf venues — 2026-07-27 — written at HEAD 6cdee22a455

Local orientation host (never acceptance evidence): aarch64 Graviton, 96
cores (`uname -m` verified by exploration agent; lscpu digest to be recorded
at first fleet_ab --local run). Acceptance venue: 8× m8g.24xlarge ARM fleet,
ap-south-2, launched in Unit 4; host list + per-shard `uname -m`/`lscpu`
digests will be appended here at launch, per mission requirement.
