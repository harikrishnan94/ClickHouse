#!/usr/bin/env python3
"""Frozen protected-cell suite for the WaveJoin campaign (Unit 0 / Unit 3).

Reuses bep/tools/join_mergetree_bench.py builders/parsers verbatim (import).
Protected cells (user ruling 2026-07-16, recorded in PREREG.md):
  A = D=268435456 m=1 ratio=2 hit=1 bp=pp=1   (v1 plans 16384 leaves)
  C = D=268435456 m=1 ratio=4 hit=1 bp=pp=7   (v1 plans 32768 leaves)
  x T in {1,16,32,64,96}  ->  10 cells.

Modes:
  --binary B                 self-paired (baseline-vs-baseline): both arms run
                             the same binary; the within-pair log-ratio SE is
                             the null-distribution scale of the Unit-3 gate
                             statistic. --reps N == N self-pairs.
  --binary-a X --binary-b Y  paired A/B (Unit 3): A = candidate, B = baseline.

Protocol (frozen; any later edit to this file is a register amendment):
  per shape: idle+foreign-process guard, count assertion + fingerprints
  (radix per arm + parallel_hash reference) at T96, then cells T96->T1;
  per cell: N position-balanced pairs (order A,B on even pair index, B,A on
  odd); per session: one `clickhouse local` process, 1 warmup + 1 timed
  query, wall = client --time of the timed query; engagement checked per
  timed run (LeafGroupBuilds; exact expected count per arm). Radix
  ProfileEvents are recorded as DIAGNOSTICS ONLY and never enter a verdict.
  Integrity snapshot of /mnt/data/join_bench_data checked pre/per-shape/post.
Band: max(1%, 3 * SE(within-pair log-ratios)) per cell, over all its pairs.
"""

import argparse
import datetime
import decimal
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/ch/ClickHouse/bep/tools")
import join_mergetree_bench as jmb  # noqa: E402

REPO = "/mnt/ch/ClickHouse"
BASE = f"{REPO}/tmp/wave-join-impl"
SCRATCH = f"{BASE}/chscratch"
DATA_DIR = "/mnt/data/join_bench_data"
INTEGRITY_REF = f"{BASE}/integrity_S0.txt"
MAX_MEMORY = 100_000_000_000
ORACLE_THREADS = 96
ORACLE_TIMEOUT = 1800.0

# FROZEN cell table. Do not edit after the Unit-0 prereg commit.
SHAPES = {
    "A": dict(cardinality=268435456, ratio="2", hit_rate="1", bp=1, pp=1,
              expected_leaves=16384),
    "C": dict(cardinality=268435456, ratio="4", hit_rate="1", bp=7, pp=7,
              expected_leaves=32768),
}
THREADS = (96, 64, 32, 16, 1)  # execution order within a shape: cheap first
CELLS = [f"{s}_T{t}" for s in ("A", "C") for t in THREADS]
SESSION_TIMEOUT_S = {"A_T1": 2400.0, "C_T1": 5400.0, "C_T16": 1200.0}
DEFAULT_SESSION_TIMEOUT_S = 900.0
# Pre-declared: more pairs where single-pair disturbances historically reach
# ~40% (T64/T96). Band is computed over ALL recorded pairs of a cell.
HIGH_T_PAIRS = 9

REPORT_EVENTS = (
    "RadixHashJoinBuildMicroseconds",
    "RadixHashJoinLeafGroupBuilds",
    "RadixHashJoinLeafGroupBuildMicroseconds",
    "RadixHashJoinProbeMicroseconds",
    "RadixHashJoinProbePackHashRouteMicroseconds",
    "RadixHashJoinProbeCollectMatchesMicroseconds",
    "RealTimeMicroseconds",
    "SelectedRows",
    "SelectedBytes",
)


def log(msg):
    print(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}",
          flush=True)


_METADATA = None


def get_metadata(binary):
    global _METADATA
    if _METADATA is None:
        _METADATA = jmb.read_metadata(binary, SCRATCH)
        if _METADATA is None:
            raise SystemExit("bench dataset not found/invalid via scratch dir")
    return _METADATA


def make_point(spec, binary):
    points = jmb.validate_points(
        get_metadata(binary),
        [spec["cardinality"]],
        [1],
        [decimal.Decimal(spec["ratio"])],
        [decimal.Decimal(spec["hit_rate"])],
        [spec["bp"]],
        [spec["pp"]],
    )
    assert len(points) == 1
    return points[0]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def integrity_snapshot():
    out = subprocess.run(
        ["find", DATA_DIR, "-printf", "%p %s %T@\n"],
        stdout=subprocess.PIPE, check=True,
    ).stdout.decode()
    return "".join(sorted(out.splitlines(keepends=True)))


def check_integrity(label, out_row):
    cur = integrity_snapshot()
    if not os.path.exists(INTEGRITY_REF):
        raise SystemExit(f"integrity reference {INTEGRITY_REF} missing; "
                         f"run --init-integrity-ref during bootstrap first")
    with open(INTEGRITY_REF) as f:
        ref = f.read()
    ok = cur == ref
    out_row({"kind": "integrity", "label": label, "ok": ok,
             "snapshot_sha256": hashlib.sha256(cur.encode()).hexdigest()})
    if not ok:
        path = f"{BASE}/integrity_VIOLATION_{label}.txt"
        with open(path, "w") as f:
            f.write(cur)
        raise SystemExit(f"INTEGRITY VIOLATION at {label}: /mnt/data snapshot "
                         f"differs from {INTEGRITY_REF} — dump at {path}. STOP.")


def foreign_processes():
    """Heavy compute processes not belonging to this suite run.

    Exact process names via pgrep -x (a plain -f pattern would match e.g.
    an ssh session whose command line carries a @clickhouse.com user), plus
    command-line matches for the bench/test drivers, excluding transport
    daemons and this process tree.
    """
    lines = []
    for name in ("clickhouse", "ninja", "clang++", "ld.lld"):
        out = subprocess.run(["pgrep", "-ax", name],
                             stdout=subprocess.PIPE, check=False).stdout.decode()
        lines.extend(out.splitlines())
    out = subprocess.run(
        ["pgrep", "-af", "join_mergetree_bench|clickhouse-test"],
        stdout=subprocess.PIPE, check=False,
    ).stdout.decode()
    lines.extend(l for l in out.splitlines()
                 if "tailscaled" not in l and "pgrep" not in l
                 and "suite.py" not in l)
    own = str(os.getpid())
    return [l for l in lines if l.split()[0] != own]


def guard(label):
    busy = foreign_processes()
    if busy:
        raise SystemExit(f"HOST BUSY at {label}, refusing to measure:\n"
                         + "\n".join(busy))


def wait_for_idle(max_wait=600.0):
    start = time.time()
    while True:
        load = os.getloadavg()[0]
        if load < 1.0:
            return load
        if time.time() - start > max_wait:
            raise SystemExit(f"host not idle after {max_wait}s (load {load:.2f})")
        time.sleep(15)


def fingerprint_projection(point):
    cols = []
    for i in range(point.probe_payload_columns):
        cols.append((f"p.p_p{i}", f"fp_p{i}"))
    for i in range(point.build_payload_columns):
        cols.append((f"b.b_p{i}", f"fb_p{i}"))
    if not cols:
        cols = [("toUInt8(0)", "fmatch")]
    inner_proj = ", ".join(f"{expr} AS {alias}" for expr, alias in cols)
    hash_args = ", ".join(alias for _, alias in cols)
    return inner_proj, hash_args


def fingerprint_query(point, algorithm):
    inner_proj, hash_args = fingerprint_projection(point)
    inner = f"SELECT {inner_proj} {jmb._join_from(point)}"
    settings = jmb._settings(algorithm, ORACLE_THREADS, MAX_MEMORY)
    return (f"SELECT count() AS c, sum(cityHash64({hash_args})) AS h "
            f"FROM ({inner}) {settings} FORMAT TSV")


def run_fingerprint(binary, point, algorithm):
    sql = fingerprint_query(point, algorithm)
    rc, stdout, stderr = jmb._run_local(
        binary, SCRATCH, sql + ";\n", timeout=ORACLE_TIMEOUT, profile_events=True)
    if rc != 0:
        raise SystemExit(f"fingerprint failed rc={rc}: {stderr[-2000:]}")
    packets = jmb.parse_profile_events(stderr, expected_packets=1)
    reason = jmb.fallback_reason(algorithm, packets)
    if reason is not None:
        raise SystemExit(f"fingerprint fell back ({algorithm}): {reason}")
    c, h = stdout.decode().strip().split("\t")
    return int(c), h


def run_assertions(binary, point):
    sql = jmb.assertion_query(point, ORACLE_THREADS, MAX_MEMORY)
    rc, stdout, stderr = jmb._run_local(binary, SCRATCH, sql,
                                        timeout=ORACLE_TIMEOUT)
    if rc != 0:
        raise SystemExit(f"assertion failed rc={rc}: {stderr[-2000:]}")
    got = jmb.parse_assertion_output(stdout)
    expected = (point.probe_rows, point.build_rows, point.output_rows)
    if got != expected:
        raise SystemExit(f"count assertion failed: got {got}, expected {expected}")
    return got


def run_timed_session(binary, point, threads, timeout, expected_leaves):
    sql = jmb.measurement_script(point, "radix_join", threads, MAX_MEMORY, runs=1)
    rc, _, stderr = jmb._run_local(binary, SCRATCH, sql, timeout=timeout,
                                   profile_events=True)
    if rc != 0:
        raise SystemExit(f"timed session failed rc={rc}: {stderr[-2000:]}")
    packets = jmb.parse_timed_profile_events(stderr, runs=1)
    reason = jmb.fallback_reason("radix_join", packets)
    if reason is not None:
        raise SystemExit(f"timed run fell back: {reason}")
    packet = packets[0]
    leaves = packet.get("RadixHashJoinLeafGroupBuilds", 0)
    if expected_leaves is not None and leaves != expected_leaves:
        raise SystemExit(f"engagement: leaf builds {leaves} != expected "
                         f"{expected_leaves} (plan-shape regression)")
    row = {"wall_us": packet[jmb.WALL_TIME_EVENT]}
    for name in REPORT_EVENTS:
        row[name] = packet.get(name, 0)
    return row


def cell_pairs(cell, base_pairs, high_t_pairs):
    threads = int(cell.split("_T")[1])
    return max(base_pairs, high_t_pairs) if threads >= 64 else base_pairs


def summarize(cell, rows):
    walls = [r["wall_us"] for r in rows]
    by_arm = {"A": [r["wall_us"] for r in rows if r["arm"] == "A"],
              "B": [r["wall_us"] for r in rows if r["arm"] == "B"]}
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], {})[r["arm"]] = r["wall_us"]
    logs = [math.log(p["A"] / p["B"]) for p in pairs.values()
            if "A" in p and "B" in p]
    se = statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) >= 2 else None
    band = max(0.01, 3 * se) if se is not None else None
    return {
        "kind": "summary", "cell": cell, "n_runs": len(walls),
        "n_pairs": len(logs),
        "median_us": statistics.median(walls),
        "median_us_A": statistics.median(by_arm["A"]) if by_arm["A"] else None,
        "median_us_B": statistics.median(by_arm["B"]) if by_arm["B"] else None,
        "min_us": min(walls), "max_us": max(walls),
        "stdev_us": statistics.stdev(walls) if len(walls) >= 2 else None,
        "pair_log_ratios": logs,
        "geomean_ratio": math.exp(statistics.fmean(logs)) if logs else None,
        "se": se, "band": band,
    }


def dump_sql(binary):
    os.makedirs(f"{BASE}/sql", exist_ok=True)
    written = []
    for shape, spec in SHAPES.items():
        point = make_point(spec, binary)
        for t in THREADS:
            path = f"{BASE}/sql/{shape}_T{t}.sql"
            with open(path, "w") as f:
                f.write(jmb.measurement_script(point, "radix_join", t,
                                               MAX_MEMORY, runs=1))
            written.append(path)
        for name, sql in (
            ("assert", jmb.assertion_query(point, ORACLE_THREADS, MAX_MEMORY)),
            ("fingerprint_radix", fingerprint_query(point, "radix_join")),
            ("fingerprint_parallel_hash",
             fingerprint_query(point, "parallel_hash")),
        ):
            path = f"{BASE}/sql/{shape}_{name}.sql"
            with open(path, "w") as f:
                f.write(sql + "\n")
            written.append(path)
    for path in written:
        print(f"{sha256_file(path)}  {os.path.relpath(path, REPO)}")


def load_completed(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done, []
    rows = []
    with open(out_path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                if row.get("kind") == "run":
                    done.add((row["cell"], row["pair"], row["arm"]))
    return done, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", help="self-paired mode: one binary, both arms")
    ap.add_argument("--binary-a", help="A/B mode: candidate binary")
    ap.add_argument("--binary-b", help="A/B mode: baseline binary")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--reps", "--pairs", dest="pairs", type=int, default=5,
                    help="pairs per cell (a rep == one position-balanced pair)")
    ap.add_argument("--high-t-pairs", type=int, default=HIGH_T_PAIRS,
                    help="pairs at T64/T96 (max of this and --reps)")
    ap.add_argument("--out", help="JSONL output (required for measurement runs)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--expect-sha256-a")
    ap.add_argument("--expect-sha256-b")
    ap.add_argument("--allow-same-binary", action="store_true")
    ap.add_argument("--expected-leaves-a", type=int, default=-1,
                    help="-1 = frozen per-shape value; 0 = only require >=1")
    ap.add_argument("--expected-leaves-b", type=int, default=-1)
    ap.add_argument("--init-integrity-ref", action="store_true",
                    help="bootstrap: freeze current /mnt/data snapshot as S0")
    ap.add_argument("--dump-sql", action="store_true",
                    help="write rendered SQL to sql/ and print sha256s; no runs")
    args = ap.parse_args()

    if args.init_integrity_ref:
        with open(INTEGRITY_REF, "w") as f:
            f.write(integrity_snapshot())
        print(f"wrote {INTEGRITY_REF}")
        return

    if args.binary and (args.binary_a or args.binary_b):
        ap.error("--binary is exclusive with --binary-a/--binary-b")
    if not args.binary and not (args.binary_a and args.binary_b):
        ap.error("need --binary or both --binary-a and --binary-b")
    mode = "self-paired" if args.binary else "ab"
    binary_a = args.binary or args.binary_a
    binary_b = args.binary or args.binary_b

    if args.dump_sql:
        dump_sql(binary_a)
        return

    if not args.out:
        ap.error("--out is required for measurement runs")
    sha_a, sha_b = sha256_file(binary_a), sha256_file(binary_b)
    if args.expect_sha256_a and sha_a != args.expect_sha256_a:
        raise SystemExit(f"binary A sha256 {sha_a} != expected")
    if args.expect_sha256_b and sha_b != args.expect_sha256_b:
        raise SystemExit(f"binary B sha256 {sha_b} != expected")
    if mode == "ab" and sha_a == sha_b and not args.allow_same_binary:
        raise SystemExit("A and B binaries are byte-identical; mispointed path? "
                         "(--allow-same-binary for deliberate null runs)")

    cells = CELLS if args.cells == "all" else args.cells.split(",")
    for c in cells:
        if c not in CELLS:
            raise SystemExit(f"unknown cell {c}; valid: {','.join(CELLS)}")

    done, prior_rows = (load_completed(args.out) if args.resume
                        else (set(), []))
    if not args.resume and os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; use --resume or a new path")

    outf = open(args.out, "a")

    def out_row(row):
        row.setdefault("ts", time.time())
        outf.write(json.dumps(row) + "\n")
        outf.flush()

    load = wait_for_idle()
    guard("start")
    out_row({"kind": "header", "mode": mode, "cells": cells,
             "pairs": args.pairs, "high_t_pairs": args.high_t_pairs,
             "binary_a": {"path": binary_a, "sha256": sha_a,
                          "size": os.path.getsize(binary_a)},
             "binary_b": {"path": binary_b, "sha256": sha_b,
                          "size": os.path.getsize(binary_b)},
             "loadavg": load, "nproc": os.cpu_count(),
             "cmdline": sys.argv})
    check_integrity("pre", out_row)

    exp_a = {s: (v["expected_leaves"] if args.expected_leaves_a == -1
                 else (None if args.expected_leaves_a == 0
                       else args.expected_leaves_a))
             for s, v in SHAPES.items()}
    exp_b = {s: (v["expected_leaves"] if args.expected_leaves_b == -1
                 else (None if args.expected_leaves_b == 0
                       else args.expected_leaves_b))
             for s, v in SHAPES.items()}

    all_rows = [r for r in prior_rows if r.get("kind") == "run"]
    for shape, spec in SHAPES.items():
        shape_cells = [c for c in cells if c.startswith(f"{shape}_")]
        if not shape_cells:
            continue
        guard(f"shape-{shape}")
        point = make_point(spec, binary_a)
        log(f"shape {shape}: oracles at T{ORACLE_THREADS}")
        counts = run_assertions(binary_a, point)
        out_row({"kind": "assertion", "shape": shape, "probe": counts[0],
                 "build": counts[1], "joined": counts[2], "ok": True})
        fps = {}
        fps[("radix_join", "A")] = run_fingerprint(binary_a, point, "radix_join")
        fps[("parallel_hash", "A")] = run_fingerprint(binary_a, point,
                                                      "parallel_hash")
        if mode == "ab":
            fps[("radix_join", "B")] = run_fingerprint(binary_b, point,
                                                       "radix_join")
        if len(set(fps.values())) != 1:
            out_row({"kind": "fingerprint", "shape": shape, "ok": False,
                     "values": {f"{a}/{arm}": [c, h]
                                for (a, arm), (c, h) in fps.items()}})
            raise SystemExit(f"FINGERPRINT MISMATCH shape {shape}: {fps}")
        (c0, h0) = next(iter(fps.values()))
        out_row({"kind": "fingerprint", "shape": shape, "ok": True,
                 "count": c0, "hash": h0,
                 "arms": sorted(f"{a}/{arm}" for (a, arm) in fps)})
        for cell in shape_cells:
            threads = int(cell.split("_T")[1])
            timeout = SESSION_TIMEOUT_S.get(cell, DEFAULT_SESSION_TIMEOUT_S)
            n_pairs = cell_pairs(cell, args.pairs, args.high_t_pairs)
            log(f"cell {cell}: {n_pairs} pairs, timeout {timeout:.0f}s")
            for pair in range(n_pairs):
                order = ("A", "B") if pair % 2 == 0 else ("B", "A")
                for position, arm in enumerate(order):
                    if (cell, pair, arm) in done:
                        continue
                    binary = binary_a if arm == "A" else binary_b
                    exp = exp_a[shape] if arm == "A" else exp_b[shape]
                    t0 = time.time()
                    row = run_timed_session(binary, point, threads, timeout, exp)
                    row.update({"kind": "run", "cell": cell, "shape": shape,
                                "threads": threads, "pair": pair, "arm": arm,
                                "position": position,
                                "binary_sha256": sha_a if arm == "A" else sha_b,
                                "session_s": round(time.time() - t0, 3)})
                    out_row(row)
                    all_rows.append(row)
                    log(f"  {cell} pair {pair} arm {arm}: "
                        f"wall {row['wall_us'] / 1e6:.3f}s "
                        f"(session {row['session_s']:.1f}s)")
        check_integrity(f"post-shape-{shape}", out_row)

    for cell in cells:
        rows = [r for r in all_rows if r["cell"] == cell]
        if rows:
            summary = summarize(cell, rows)
            out_row(summary)
            log(f"summary {cell}: median {summary['median_us'] / 1e6:.3f}s "
                f"n={summary['n_runs']} band={summary['band']}")

    check_integrity("post", out_row)
    sha_a_end = sha256_file(binary_a)
    out_row({"kind": "footer", "status": "complete",
             "binary_a_sha256_end": sha_a_end,
             "binary_b_sha256_end": sha256_file(binary_b),
             "binary_stable": sha_a_end == sha_a})
    outf.close()
    log("suite complete")


if __name__ == "__main__":
    main()
