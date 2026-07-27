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

## PREREG-006 — Unit 2 commit 3: AMAC build-insert ring — 2026-07-27 — written at HEAD 60b8d1684a1

Expectation: the AMAC build-insert ring (32-slot SOA ring driver `amacRun` +
`AmacBuildInsertPolicy`, ported as ideas from `ahj`), engaged under
`parallel_hash` only (process hook `CLICKHOUSE_JOIN_AMAC` ∈ {0/off, 1/auto
default, force}; auto predicate: cursor-capable map type AND map buffer
bytes > `getMinBytesForPrefetchInJoin()` AND section rows ≥ 256), (a) keeps
G-parity green with the force pass asserting
`ConcurrentHashJoinAmacBuildRows` > 0 in exactly the 8 cursor-capable
families (lcstr and mixed are excluded getters and must stay at 0); (b) new
gtests (growth-resume mid-ring, duplicate-heavy ring-vs-sequential parity,
string-key persist-once) are green on the candidate and the negative proof
holds (baseline binary lacks the counters; hook=0 cannot engage); (c) the
local orientation A/B (candidate-60b8d1684a1 vs post-ring snapshot) on
{key64,str}:build.inner_all.S4.T96 improves
`ConcurrentHashJoinBuildInsertMicroseconds` outside the 3% band with wall
not regressing outside band, and S1/S2 build cells stay in-band with the
auto predicate disengaging (counters 0) — per the `ahj` lead of a 1.09-1.12x
warm-insert win on DRAM-resident maps; (d) G-disasm-build: the ring steady
loop's instruction semantics match the `ahj` reference binary on the 3 build
anchors (key64/RowRefList, keys256, key_string) — write-intent `pstl1keep`
prefetch, fused claim, no policy-field reloads in the steady loop.

Invocation:
  ninja -C build/reldeb clickhouse > build/reldeb/build_amacbuild.log 2>&1
  bash tmp/chj_amac/parity/run_parity.sh bins/clickhouse-baseline-a05f3ee81ff.bin <post-ring snapshot>   # force pass now REQUIRED green
  <gtest binary> --gtest_filter='*ConcurrentHashJoinAmac*'
  python3 tmp/chj_amac/fleet_ab.py sweep --local --arm-a bins/clickhouse-candidate-60b8d1684a1.bin --arm-b <post-ring snapshot> --cells "key64:build.inner_all.S4.T96,str:build.inner_all.S4.T96,key64:build.inner_all.S2.T96,key64:build.inner_all.S4.T1" --calibration tmp/chj_amac/fleet/calibration_rows.json
  python3 .claude/tools/analyze-assembly.py <post-ring snapshot> "<amacRun build instantiation>" (+ same on bins/clickhouse-ahj-cf465cfbe23.bin; normalized comparison)

Refutation criterion: any parity divergence or force-pass family engaging
that must not (or failing to engage that must); any gtest failure; S4 build
cells showing NO `BuildInsert` improvement outside the band while engaged,
or ANY cell (incl. S1/S2) losing wall outside the band; steady-loop
disassembly showing policy reloads, spills, or a missing/wrong-locality
prefetch vs the `ahj` anchors.

Action on refutation: parity/gtest red → stop and fix before commit; no
perf win while engaged → run the codegen checklist from the `ahj` lessons
(refill inlining, frame-copy SSA, prefetch encodings) BEFORE touching the
predicate; wall loss on small cells → tighten the auto predicate and re-run;
disassembly divergence → fix codegen, never reinterpret the anchor.

## PREREG-007 — Unit 3: order-preserving routed probe + AMAC probe find ring — 2026-07-27 — written at HEAD 8d9bf852d51

Expectation: after the routed probe (per-row slot derivation over the
original left block, shared `StoredColumnsIndex`, per-slot flags with
slot-local offsets, in-order emit) and the AMAC probe find ring (out-of-
order find into per-row `found_word` scratch + dispatch-free `word_loop` /
precomputed-loop emit, `ConcurrentHashJoinAmacProbeRows` event) land:
(a) G-parity green with the force pass asserting BOTH sides —
`force-pass: engaged 8/8+2x0 (build,probe)`;
(b) G-order green BY CONSTRUCTION on the flipped candidate:
`run_order.sh` prints `ORDER OK` (all 17 checks incl. genuine RIGHT/FULL
per-block monotone at `max_threads=96`, T=1 global), stateless 03448 and
03711 pass 10/10 (they fail 10/10 on today's scatter candidate — the flip
must convert them), and the baseline power check still fails per-block
(oracle keeps teeth);
(c) probe orientation A/B (arm A = `candidate-7e64a6cf4d5`, arm B =
post-Unit-3 snapshot) on {key64,str}:probe.inner_all.{S2,S3}.T96 and
key64:probe.inner_all.S3.T1: NO cell loses outside the 3% band, with any
win carried by `ConcurrentHashJoinProbeLookupMicroseconds`; the ordered
probe without the ring may cost — the ring must close it (the `ahj` lead:
the ring won at every depth once the stored-key/prefetch fixes were in);
(d) G-disasm-probe: the find-ring steady loops match the `ahj` reference on
3 anchors (key64, keys256, key_string × RowRefList) — read-intent
`pldl1keep` prefetch plus the second cache line for cells wider than 24
bytes, resolved-`Cell*` slots, no per-visit policy reloads;
(e) new probe gtests (flagged shapes RIGHT/FULL, `setUsedOnce`, ring-vs-Off
parity) green in both hook arms;
(f) compile-time and binary-size delta of the new TUs measured and reported.

Invocation:
  bash tmp/chj_amac/parity/run_parity.sh <baseline.bin> <post-U3 snapshot> --require-engagement
  bash tmp/chj_amac/order/run_order.sh <post-U3 snapshot>
  (cd tests && CLICKHOUSE_PORT_TCP=19310 CLICKHOUSE_PORT_HTTP=18310 ./clickhouse-test --test-runs 10 03448_analyzer_array_join_alias_in_join_using_bug 03711_read_in_order_through_join)
  bash tmp/chj_amac/order/run_order.sh bins/clickhouse-baseline-a05f3ee81ff.bin --expect-fail
  python3 tmp/chj_amac/fleet_ab.py sweep --local --arm-a bins/clickhouse-candidate-7e64a6cf4d5.bin --arm-b <post-U3 snapshot> --cells "key64:probe.inner_all.S2.T96,key64:probe.inner_all.S3.T96,str:probe.inner_all.S2.T96,str:probe.inner_all.S3.T96,key64:probe.inner_all.S3.T1" --calibration tmp/chj_amac/fleet/calibration_rows.json
  <disasm agent over the 3 probe anchors vs bins/clickhouse-ahj-cf465cfbe23.bin>
  build/reldeb/src/unit_tests_dbms --gtest_filter='*Amac*' (both env arms)

Refutation criterion: any parity divergence or wrong-side engagement; any
order check red on the flipped candidate or any of the 20 stateless runs
failing; the baseline power check passing (oracle broken); any probe A/B
cell losing outside the band vs `candidate-7e64a6cf4d5`; unexplained disasm
divergence; binary growth beyond ~2% unaccounted.

Action on refutation: order red → the flip commit does not land until
fixed; a probe loss vs pre-Unit-3 → the `ahj` codegen checklist (stored
keys packed once, buf/mask at admit, prefetch localities, refill inlining)
BEFORE any predicate change, and if the loss stands the ring/routed probe
does not ship for that family (excluded-measured-loss, force-engage
discriminator in Unit 4); disasm divergence → fix codegen, never
reinterpret the anchor.

## PREREG-003 — env facts for perf venues — 2026-07-27 — written at HEAD 6cdee22a455

Local orientation host (never acceptance evidence): aarch64 Graviton, 96
cores (`uname -m` verified by exploration agent; lscpu digest to be recorded
at first fleet_ab --local run). Acceptance venue: 8× m8g.24xlarge ARM fleet,
ap-south-2, launched in Unit 4; host list + per-shard `uname -m`/`lscpu`
digests will be appended here at launch, per mission requirement.

APPENDIX (launched 2026-07-27, per requester authorization): 8×
m8g.24xlarge, ap-south-2c, security group `sg-0426a4e0a113e0985`, ephemeral
keypair `fleet/ssh/id_ed25519` (cloud-init injected). Shards (index /
instance id / private ip): 0 i-0fe67352e5989edf7 172.31.31.21;
1 i-034fe227c0c563e19 172.31.29.85; 2 i-0aacde025557ca905 172.31.18.3;
3 i-0e97d0836ce2798f2 172.31.20.240; 4 i-09c676297620ec10a 172.31.18.14;
5 i-075f553f3fadcad69 172.31.30.126; 6 i-0cc94834290c24619 172.31.16.216;
7 i-00dc41ff4d2cf8008 172.31.25.54. All `uname -m` = aarch64, CPU
`Neoverse-V2` (full lscpu per shard in `fleet/smoke_shard*.log`). Deployed
arms (sha256 verified on every shard against `bins/MANIFEST.tsv`):
baseline `/home/ubuntu/chj/clickhouse-base` = `0d32ef1c96e6d378aa20d3ab...`
(`a05f3ee81ff`); candidate `/home/ubuntu/chj/clickhouse-cand` =
`dc8b1f17e5a7fcce614c8d26...` (`5b276c5fb88`). Teardown owed at campaign
end: `fleet/teardown.sh` (instances + SG), accounting into REPORT.md.
