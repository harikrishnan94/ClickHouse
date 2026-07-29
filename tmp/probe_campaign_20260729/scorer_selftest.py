#!/usr/bin/env python3
"""Prove `probe_ab_report.py`'s checks can FAIL, and that each metric is scored on its own.

A gate that cannot go red proves nothing, so every case below is a hand-built
fixture with a known right answer, fed through the real scorer as a subprocess
(exit code and stdout are the contract, exactly as the campaign gates use it).

Cases:
  1  arm B genuinely slower on probe_cost only      -> LOSS probe_cost, TIE projection_cost
  2  arm B genuinely slower on projection_cost only -> TIE probe_cost, LOSS projection_cost
  3  arm B genuinely faster on both                 -> WIN on both
  4  identical arms                                 -> TIE on both, --aa-control exits 0
  5  identical arms with --band-override 0          -> --aa-control MUST exit non-zero
  6  dispatch+lookup > probe total (negative residual) -> --check-decomposition exits non-zero
  7  a gather event present on arm B only           -> --check-decomposition exits non-zero
  8  a gather event present on BOTH arms            -> --check-decomposition exits 0 (symmetric)
  9  path event zero on one run                     -> --check-path-event exits non-zero
 10  a foreign algorithm's path event non-zero      -> --check-path-event exits non-zero
 11  join_algorithm recorded as 'hash'              -> --check-path-event exits non-zero
 12  a .hash cell id                                -> --check-path-event exits non-zero
 13  fewer than min-runs valid runs                 -> NO-VERDICT, and --expect-cells fails
 14  harness-voided cell (duration floor)           -> NO-VERDICT with the floor reason, never TIE
 15  cross-arm checksum mismatch                    -> NO-VERDICT, never scored
 16  jbmt-shaped input, arm B slower on probe_cost  -> LOSS (proves the jbmt loader)

Usage: python3 scorer_selftest.py   (exit 0 = every case behaved as required)
"""
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SCORER = HERE / "probe_ab_report.py"


def fleet_row(cell, arm, role, run, dispatch, lookup, probe_total, *, valid=True,
              invalid_reason=None, checksum="123", extra=None, algo="parallel_hash",
              path_event=1000, rows=1000, expected_rows=1000):
    events = {"ConcurrentHashJoinBuildMicroseconds": path_event,
              "ConcurrentHashJoinProbeMicroseconds": probe_total,
              "ConcurrentHashJoinProbeDispatchMicroseconds": dispatch,
              "ConcurrentHashJoinProbeLookupMicroseconds": lookup}
    events.update(extra or {})
    return {"cell": cell, "cell_axes": {"algo": algo, "family": "key64"}, "arm": arm,
            "arm_role": role, "run": run, "block": run // 5, "direction": run // 5,
            "protocol": "abba4-pfill", "valid": valid, "invalid_reason": invalid_reason,
            "duration_us": 1_000_000, "events": events, "rows": rows,
            "expected_rows": expected_rows, "checksum": checksum, "nonce": "1_1",
            "binary_sha256": "deadbeef"}


def write_fleet(path, cells, *, runs=6):
    """cells: {cell_id: {"A": (disp, look, total), "B": (...), **overrides}}"""
    with open(path, "w") as fh:
        for cell, spec in cells.items():
            for role, arm in (("A", spec.get("name_a", "armA")), ("B", spec.get("name_b", "armB"))):
                disp, look, total = spec[role]
                for run in range(runs):
                    # a deterministic 1 us wobble keeps pstdev > 0 without moving the median
                    wob = 1 if run % 2 else 0
                    fh.write(json.dumps(fleet_row(
                        cell, arm, role, run, disp + wob, look, total + wob,
                        valid=spec.get("valid", True),
                        invalid_reason=spec.get("invalid_reason"),
                        checksum=spec.get(f"checksum_{role}", "123"),
                        extra=spec.get(f"extra_{role}") or spec.get("extra"),
                        algo=spec.get("algo", "parallel_hash"),
                        path_event=(0 if spec.get("zero_path_on") == (role, run) else 1000),
                    )) + "\n")


def write_jbmt(path, unit, a_events, b_events, *, runs=5):
    def arm(events):
        return {"binary": "/bin/x", "port": 9005, "binary_sha256": "abc",
                "algorithms": {"parallel_hash": {
                    "status": "OK", "fallback_runs": 0,
                    "durations_ms": [10] * runs,
                    "events_per_run": [dict(events, **{
                        "ConcurrentHashJoinProbeDispatchMicroseconds":
                            events["ConcurrentHashJoinProbeDispatchMicroseconds"] + (i % 2)}) 
                        for i in range(runs)],
                    "row_count": 1000, "checksum": 42, "runs": runs, "warmups": 4}}}
    row = {"unit_id": unit, "unit": "cell", "tool_version": "jbmt-v2",
           "meta": {"unit_id": unit, "threads": 96}, "status": "OK",
           "expected_rows_closed_form": None,
           "algorithms_measured": ["parallel_hash"], "lead_arm": "baseline",
           "arms": {"baseline": arm(a_events), "candidate": arm(b_events)},
           "wall_seconds": 1.0}
    pathlib.Path(path).write_text(json.dumps(row) + "\n")


def ev(dispatch, lookup, total, **extra):
    d = {"ConcurrentHashJoinBuildMicroseconds": 1000,
         "ConcurrentHashJoinProbeMicroseconds": total,
         "ConcurrentHashJoinProbeDispatchMicroseconds": dispatch,
         "ConcurrentHashJoinProbeLookupMicroseconds": lookup}
    d.update(extra)
    return d


def run(args):
    p = subprocess.run([sys.executable, str(SCORER)] + args, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


FAILURES = []


def expect(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def verdict_of(out, cell, metric):
    """Read a verdict out of the TSV the scorer writes, not out of prose."""
    lines = [l.split("\t") for l in out.strip().splitlines()]
    head = lines[0]
    col = head.index(f"{metric}_verdict")
    for row in lines[1:]:
        if row[0] == cell:
            return row[col]
    return None


def main():
    scratch = HERE / "tmp"
    scratch.mkdir(exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="scorer_selftest_", dir=scratch))
    print(f"fixtures in {tmp}")

    # ---- 1/2/3: each metric scored independently, in both directions
    f = tmp / "dirs.jsonl"
    write_fleet(f, {
        # probe_cost 100k->200k (LOSS), projection_cost 10k->10k (TIE)
        "key64:probe.slower_probe.S3.T96": {"A": (0, 100_000, 110_000), "B": (0, 200_000, 210_000)},
        # probe_cost equal (TIE), projection_cost 10k->40k (LOSS)
        "key64:probe.slower_proj.S3.T96": {"A": (0, 100_000, 110_000), "B": (0, 100_000, 140_000)},
        # both faster (WIN/WIN)
        "key64:probe.faster_both.S3.T96": {"A": (0, 100_000, 120_000), "B": (0, 50_000, 55_000)},
    })
    tsv = tmp / "dirs.tsv"
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--metric", "both", "--out-tsv", str(tsv), "--quiet-report"])
    t = tsv.read_text()
    expect("1 arm B slower on probe_cost only -> LOSS probe_cost",
           verdict_of(t, "key64:probe.slower_probe.S3.T96", "probe_cost") == "LOSS", out)
    expect("1 ... and TIE on projection_cost (no netting)",
           verdict_of(t, "key64:probe.slower_probe.S3.T96", "projection_cost") == "TIE", out)
    expect("2 arm B slower on projection_cost only -> LOSS projection_cost",
           verdict_of(t, "key64:probe.slower_proj.S3.T96", "projection_cost") == "LOSS", out)
    expect("2 ... and TIE on probe_cost",
           verdict_of(t, "key64:probe.slower_proj.S3.T96", "probe_cost") == "TIE", out)
    expect("3 arm B faster -> WIN on both",
           verdict_of(t, "key64:probe.faster_both.S3.T96", "probe_cost") == "WIN"
           and verdict_of(t, "key64:probe.faster_both.S3.T96", "projection_cost") == "WIN", out)

    # ---- 4/5: A/A control, and its power to go red when the band is removed
    f = tmp / "aa.jsonl"
    write_fleet(f, {f"key64:probe.aa{i}.S3.T96": {"A": (10, 100_000, 110_000),
                                                  "B": (10, 100_050, 110_050)}
                    for i in range(9)})
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--metric", "both", "--aa-control", "--quiet-report"])
    expect("4 identical-ish arms pass --aa-control", rc == 0, out)
    rc0, out0 = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                     "--metric", "both", "--aa-control", "--band-override", "0",
                     "--quiet-report"])
    expect("5 --band-override 0 turns the A/A control RED (gate has power)", rc0 != 0, out0)

    # ---- 6: negative residual
    f = tmp / "neg.jsonl"
    write_fleet(f, {"key64:probe.neg.S3.T96": {"A": (0, 100_000, 110_000),
                                               "B": (90_000, 100_000, 110_000)}})
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--check-decomposition", "--quiet-report"])
    expect("6 negative residual fails --check-decomposition", rc != 0, out)
    expect("6 ... and the offending row is printed", "projection_cost" in out and "< 0" in out, out)

    # ---- 7/8: gather-event symmetry
    f = tmp / "asym.jsonl"
    write_fleet(f, {"key64:probe.asym.S3.T96": {
        "A": (0, 100_000, 110_000), "B": (0, 100_000, 110_000),
        "extra_B": {"HashJoinResultBuildOutputMicroseconds": 5_000}}})
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--check-decomposition", "--quiet-report"])
    expect("7 gather event on arm B only fails --check-decomposition", rc != 0, out)
    f = tmp / "sym.jsonl"
    write_fleet(f, {"key64:probe.sym.S3.T96": {
        "A": (0, 100_000, 110_000), "B": (0, 100_000, 110_000),
        "extra": {"HashJoinResultBuildOutputMicroseconds": 5_000}}})
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--check-decomposition", "--quiet-report"])
    expect("8 gather event on BOTH arms passes (symmetric decomposition)", rc == 0, out)
    expect("8 ... and is reported as splittable", "can be split per side" in out, out)

    # ---- 9/10/11/12: only parallel_hash
    f = tmp / "zeropath.jsonl"
    write_fleet(f, {"key64:probe.zp.S3.T96": {"A": (0, 100_000, 110_000),
                                              "B": (0, 100_000, 110_000),
                                              "zero_path_on": ("B", 3)}})
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--check-path-event", "--quiet-report"])
    expect("9 zero path event on one run fails --check-path-event", rc != 0, out)
    f = tmp / "foreign.jsonl"
    write_fleet(f, {"key64:probe.fp.S3.T96": {
        "A": (0, 100_000, 110_000), "B": (0, 100_000, 110_000),
        "extra": {"PartitionedHashJoinBuildMicroseconds": 7}}})
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--check-path-event", "--quiet-report"])
    expect("10 a foreign algorithm's path event fails --check-path-event", rc != 0, out)
    f = tmp / "hashalgo.jsonl"
    write_fleet(f, {"key64:probe.ha.S3.T96": {"A": (0, 100_000, 110_000),
                                              "B": (0, 100_000, 110_000), "algo": "hash"}})
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--check-path-event", "--quiet-report"])
    expect("11 join_algorithm='hash' fails --check-path-event", rc != 0, out)
    f = tmp / "hashcell.jsonl"
    write_fleet(f, {"key64:probe.inner_all.S2.T96.hash": {"A": (0, 100_000, 110_000),
                                                          "B": (0, 100_000, 110_000)}})
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--check-path-event", "--quiet-report"])
    expect("12 a .hash cell id fails --check-path-event", rc != 0, out)

    # ---- 13: too few runs
    f = tmp / "fewruns.jsonl"
    write_fleet(f, {"key64:probe.few.S3.T96": {"A": (0, 100_000, 110_000),
                                               "B": (0, 100_000, 110_000)}}, runs=3)
    tsv = tmp / "few.tsv"
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--expect-cells", "1", "--out-tsv", str(tsv), "--quiet-report"])
    expect("13 fewer than min-runs -> NO-VERDICT",
           verdict_of(tsv.read_text(), "key64:probe.few.S3.T96", "probe_cost") == "NO-VERDICT", out)
    expect("13 ... and --expect-cells 1 goes red", rc != 0, out)

    # ---- 14: harness-voided cell is NO-VERDICT with the reason, never TIE
    f = tmp / "voided.jsonl"
    write_fleet(f, {"key64:build.void.S2.T96": {
        "A": (0, 100_000, 110_000), "B": (0, 100_000, 110_000), "valid": False,
        "invalid_reason": "below-duration-floor (arm A median 24.5 ms < 200 ms)"}})
    tsv = tmp / "void.tsv"
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--out-tsv", str(tsv)])
    t = tsv.read_text()
    expect("14 floor-voided cell -> NO-VERDICT, not TIE",
           verdict_of(t, "key64:build.void.S2.T96", "probe_cost") == "NO-VERDICT", out)
    expect("14 ... with the harness's own floor reason printed",
           "below-duration-floor" in t and "below-duration-floor" in out, out)

    # ---- 15: cross-arm checksum mismatch is never scored
    f = tmp / "checksum.jsonl"
    write_fleet(f, {"key64:probe.cs.S3.T96": {
        "A": (0, 100_000, 110_000), "B": (0, 100_000, 110_000),
        "checksum_A": "111", "checksum_B": "222"}})
    tsv = tmp / "cs.tsv"
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--out-tsv", str(tsv), "--quiet-report"])
    t = tsv.read_text()
    expect("15 cross-arm checksum mismatch -> NO-VERDICT",
           verdict_of(t, "key64:probe.cs.S3.T96", "probe_cost") == "NO-VERDICT", out)
    expect("15 ... with the mismatch as the reason", "checksum" in t, out)

    # ---- 16: the jbmt loader scores a genuine LOSS too
    f = tmp / "jbmt.jsonl"
    write_jbmt(f, "D1000000_K0_mb1_mp16_h1.0_bp8_pp8_T16",
               ev(10, 100_000, 110_000), ev(10, 200_000, 210_000))
    tsv = tmp / "jbmt.tsv"
    rc, out = run(["--results", str(f), "--arm-a", "baseline", "--arm-b", "candidate",
                   "--out-tsv", str(tsv), "--check-decomposition", "--check-path-event",
                   "--quiet-report"])
    t = tsv.read_text()
    expect("16 jbmt-shaped input: arm B slower on probe_cost -> LOSS",
           verdict_of(t, "D1000000_K0_mb1_mp16_h1.0_bp8_pp8_T16", "probe_cost") == "LOSS", out)
    expect("16 ... and its checks pass on well-formed jbmt rows", rc == 0, out)

    # ---- 17: --compare-order must be able to report AND enforce an order flip
    base = {"key64:probe.flip.S3.T96": {"A": (0, 100_000, 110_000), "B": (0, 100_000, 110_000)},
            "key64:probe.stable.S3.T96": {"A": (0, 100_000, 110_000), "B": (0, 100_000, 110_000)}}
    f1, f2 = tmp / "order1.jsonl", tmp / "order2.jsonl"
    write_fleet(f1, base)
    flipped = dict(base)
    flipped["key64:probe.flip.S3.T96"] = {"A": (0, 100_000, 110_000), "B": (0, 200_000, 210_000)}
    write_fleet(f2, flipped)
    rc, out = run(["--results", str(f1), "--compare-order", str(f2), "--arm-a", "armA",
                   "--arm-b", "armB", "--metric", "both", "--quiet-report"])
    expect("17 --compare-order prints the flipping cell", "ORDER-EFFECT" in out, out)
    expect("17 ... and exits 0 without --fail-on-order-effect (G1-b as specified)", rc == 0, out)
    rc, out = run(["--results", str(f1), "--compare-order", str(f2), "--arm-a", "armA",
                   "--arm-b", "armB", "--metric", "both", "--fail-on-order-effect",
                   "--quiet-report"])
    expect("17 ... and --fail-on-order-effect turns it RED (gate has power)", rc != 0, out)
    rc, out = run(["--results", str(f1), "--compare-order", str(f1), "--arm-a", "armA",
                   "--arm-b", "armB", "--metric", "both", "--fail-on-order-effect",
                   "--quiet-report"])
    expect("17 ... and stays green when the two orders agree", rc == 0, out)

    # ---- 18: --expect-unit-set is a SET comparison, unlike the count-only --expect-cells
    f = tmp / "setcheck.jsonl"
    write_fleet(f, {"key64:probe.a.S3.T96": {"A": (0, 100_000, 110_000), "B": (0, 100_000, 110_000)},
                    "key64:probe.b.S3.T96": {"A": (0, 100_000, 110_000), "B": (0, 100_000, 110_000)}})
    want = tmp / "want.json"
    want.write_text(json.dumps([{"cell": "key64:probe.a.S3.T96"}, {"cell": "key64:probe.b.S3.T96"}]))
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--expect-unit-set", f"{want}:cell", "--quiet-report"])
    expect("18 exact unit set passes --expect-unit-set", rc == 0, out)
    wrong = tmp / "wrong.json"
    wrong.write_text(json.dumps([{"cell": "key64:probe.a.S3.T96"}, {"cell": "key64:probe.ZZZ.S3.T96"}]))
    rc, out = run(["--results", str(f), "--arm-a", "armA", "--arm-b", "armB",
                   "--expect-unit-set", f"{wrong}:cell", "--quiet-report"])
    expect("18 ... same COUNT but one wrong id goes RED (a count could not catch this)",
           rc != 0 and "MISSING" in out and "EXTRA" in out, out)

    # ---- 19: a jbmt unit the harness abandoned must appear as NO-VERDICT, never vanish
    f = tmp / "overbudget.jsonl"
    pathlib.Path(f).write_text(json.dumps({
        "unit_id": "tpch__pathological__T16__tiera", "unit": "query", "tool_version": "jbmt-v2",
        "meta": {"unit_id": "tpch__pathological__T16__tiera"}, "status": "OVER_BUDGET",
        "reason": "arm baseline warmup 0 took 538.6s > unit-time-budget 30s; unit skipped",
        "arms": {"baseline": {"binary": "/bin/a", "port": 9005, "binary_sha256": "x",
                              "algorithms": {}},
                 "candidate": {"binary": "/bin/b", "port": 9006, "binary_sha256": "y",
                               "algorithms": {}}}}) + "\n")
    tsv = tmp / "ob.tsv"
    rc, out = run(["--results", str(f), "--arm-a", "baseline", "--arm-b", "candidate",
                   "--out-tsv", str(tsv)])
    t = tsv.read_text()
    expect("19 an OVER_BUDGET unit is present as NO-VERDICT, not dropped",
           verdict_of(t, "tpch__pathological__T16__tiera", "probe_cost") == "NO-VERDICT", out)
    expect("19 ... carrying the harness's own reason",
           "OVER_BUDGET" in t and "unit-time-budget" in t, out)

    print(f"\nscorer_selftest: {'PASS' if not FAILURES else 'FAIL'} "
          f"({len(FAILURES)} case(s) failed)")
    for name in FAILURES:
        print(f"  FAILED: {name}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
