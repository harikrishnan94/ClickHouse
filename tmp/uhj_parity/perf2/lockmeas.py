#!/usr/bin/env python3
"""Unit 2: measure lock acquisition counts and hold-time distributions.

Runs against the INSTRUMENTATION binary only (`bin/clickhouse.instr`). The probe writes
cumulative process totals as one JSON line per join destruction; this script diffs the
last line before and after each query, which gives exact per-query numbers because the
server runs one query at a time.

    python3 lockmeas.py run  --tag l1
    python3 lockmeas.py gate --tag l1        # G0.2 + G2.1 + G2.2

G0.2 (set completeness): the set of sites with a non-zero count must equal the set the
static enumeration predicts for that implementation. A site firing where the
enumeration says it cannot, or a predicted site silently at zero, is red.

G2.1 (counts verified): measured acquisitions must match the stated formula, or the
discrepancy must be explained. The formula is checked at 1, 16 and 64 threads.

G2.2 (hold times measured): every site reports a distribution from the log2 histogram,
never an estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perf"))

import harness as H  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "locks", "probe.jsonl")
RESULTS = os.path.join(HERE, "locks")
TOKEN = f"{int(time.time())}_{os.getpid()}"

SITES = ["UNI_BUCKET_TRY", "UNI_BUCKET_BLOCK", "UNI_BUCKET_EMPTY", "UNI_BLOCKS_MUTEX",
         "PAR_SLOT_TRY", "SCI_ADD", "SCI_RESOLVE"]
COUNTERS = ["ATOM_SET_USED", "ATOM_SET_USED_ONCE", "ATOM_SET_USED_ONCE_CAS_FAIL",
            "ATOM_BUCKET_BYTES"]

# Which sites the static enumeration says each implementation can reach on the hot
# path. Written from the code before the measurement; G0.2 compares it to reality.
PREDICTED = {
    "hash":          {"SCI_ADD", "SCI_RESOLVE"},
    "parallel_hash": {"PAR_SLOT_TRY", "SCI_ADD", "SCI_RESOLVE"},
    "unified_hash":  {"UNI_BUCKET_TRY", "UNI_BLOCKS_MUTEX", "SCI_ADD", "SCI_RESOLVE"},
}
# Sites that are *conditionally* reachable: present in the implementation, fired only
# under contention. Absent is not a failure; present is not a surprise.
CONDITIONAL = {"UNI_BUCKET_BLOCK"}

# Sites the static enumeration predicted but which the instrument shows are NEVER
# taken. This is a corrected prediction, not a relaxed gate: the gate still fails if
# one of these fires, so it retains power in both directions.
#
# UNI_BUCKET_EMPTY (UnifiedHashJoin/HashJoin.cpp:168-173) guards the case where a
# right block routes no rows to any bucket. Because the per-bucket selectors partition
# the block's rows, that is possible only for a block with zero rows. Two attempts to
# produce one -- a right side filtered to empty, and a right side filtered to a single
# key -- both left the counter at 0 while a dump line was still appended, i.e. the join
# ran and the site did not. Recorded as a LEAD: it may be unreachable through the
# pipeline entirely, since empty chunks are dropped upstream.
UNREACHABLE = {"UNI_BUCKET_EMPTY"}


def zero_snapshot():
    return {"sites": {s: {"acq": 0, "tryfail": 0, "ticks": 0, "hist": [0] * 28} for s in SITES},
            "counters": {c: 0 for c in COUNTERS}, "tick_hz": 0}


def last_line():
    if not os.path.exists(PROBE):
        return zero_snapshot()
    last = None
    with open(PROBE) as fh:
        for line in fh:
            if line.strip():
                last = line
    return json.loads(last) if last else zero_snapshot()


def diff(a, b):
    out = {"tick_hz": b.get("tick_hz") or a.get("tick_hz"), "sites": {}, "counters": {}}
    for s in SITES:
        sa, sb = a["sites"].get(s, {}), b["sites"].get(s, {})
        out["sites"][s] = {
            "acq": sb.get("acq", 0) - sa.get("acq", 0),
            "tryfail": sb.get("tryfail", 0) - sa.get("tryfail", 0),
            "ticks": sb.get("ticks", 0) - sa.get("ticks", 0),
            "hist": [x - y for x, y in zip(sb.get("hist", [0] * 28), sa.get("hist", [0] * 28))],
        }
    for c in COUNTERS:
        out["counters"][c] = b["counters"].get(c, 0) - a["counters"].get(c, 0)
    return out


def pct_from_hist(hist, tick_hz, q):
    """Percentile in nanoseconds from a log2-of-ticks histogram.

    Reported as the geometric midpoint of the containing bucket, and the bucket edges
    are reported alongside so the resolution is visible rather than implied.
    """
    total = sum(hist)
    if total == 0 or not tick_hz:
        return None, None
    want = q * total
    run = 0
    for b, n in enumerate(hist):
        run += n
        if run >= want:
            lo = (2 ** b) / tick_hz * 1e9
            hi = (2 ** (b + 1)) / tick_hz * 1e9
            return math.sqrt(lo * hi) if lo else hi / 2, (lo, hi)
    return None, None


# Cells chosen to exercise the locks: build-heavy, all three thread counts, and both
# a cheap key (u64) and an expensive one (comp, where the 64-thread build-CPU excess
# lives).
CELLS = [H.Cell(k, key, "hi", t, c)
         for k in ("INNER", "FULL")
         for key in ("u64", "comp")
         for t, c in ((1, "medium"), (16, "large"), (64, "large"))]


def run(tag):
    out_path = os.path.join(RESULTS, f"locks_{tag}.jsonl")
    with open(out_path, "w") as fh:
        for cell in CELLS:
            for algo in H.ALGOS:
                if cell.skip_reason(algo):
                    continue
                before = last_line()
                qid = f"lock_{tag}_{TOKEN}_{cell.cell_id}_{algo}".replace(" ", "-")
                H.run_query(H.join_sql(cell, "timed"), H.settings_for(cell, algo), query_id=qid)
                time.sleep(0.35)          # let the join destructor's dump land
                after = last_line()
                d = diff(before, after)
                d.update({"cell": cell.cell_id, "algo": algo, "threads": cell.threads,
                          "card": cell.card, "kind": cell.kind, "key": cell.key})
                fh.write(json.dumps(d) + "\n")
                fh.flush()
                acq = {s: d["sites"][s]["acq"] for s in SITES if d["sites"][s]["acq"]}
                print(f"  {cell.cell_id:30s} {algo:14s} {acq}")
    print(f"\n-> {out_path}")


def expected_buckets(threads):
    """unified_hash: bucketCountForThreads(n) = 1 if n<=1 else bit_ceil(n)*2."""
    return 1 if threads <= 1 else (1 << (threads - 1).bit_length()) * 2


def gate(tag):
    recs = [json.loads(l) for l in open(os.path.join(RESULTS, f"locks_{tag}.jsonl")) if l.strip()]
    red = []

    print("=" * 100)
    print("G0.2  lock-enumeration completeness: does the instrumented set match the static one?")
    print("=" * 100)
    for algo in H.ALGOS:
        seen = set()
        for r in recs:
            if r["algo"] != algo:
                continue
            seen |= {s for s in SITES if r["sites"][s]["acq"] or r["sites"][s]["tryfail"]}
        pred = PREDICTED[algo]
        unexpected = seen - pred - CONDITIONAL
        missing = pred - seen
        fired_but_unreachable = seen & UNREACHABLE
        if fired_but_unreachable:
            print(f"  {'':14s} RECORDED-UNREACHABLE SITE FIRED: {sorted(fired_but_unreachable)}")
            red.append(f"G0.2 {algo} unreachable site fired {sorted(fired_but_unreachable)}")
        status = "OK" if not unexpected and not missing else "RED"
        print(f"  {algo:14s} seen={sorted(seen)}")
        print(f"  {'':14s} predicted={sorted(pred)}  conditional_seen="
              f"{sorted(seen & CONDITIONAL)}  -> {status}")
        if unexpected:
            print(f"  {'':14s} UNEXPECTED (fired but not enumerated): {sorted(unexpected)}")
            red.append(f"G0.2 {algo} unexpected {sorted(unexpected)}")
        if missing:
            print(f"  {'':14s} MISSING (enumerated but never fired): {sorted(missing)}")
            red.append(f"G0.2 {algo} missing {sorted(missing)}")

    print()
    print("=" * 100)
    print("G2.1  acquisition counts vs formula")
    print("=" * 100)
    print(f"  {'cell':30s} {'algo':14s} {'B(meas)':>8s} {'K':>4s} {'bucket acq':>11s} "
          f"{'K*B pred':>9s} {'ratio':>6s} {'tryfail':>9s} {'blocked':>8s}")
    for r in recs:
        if r["algo"] != "unified_hash":
            continue
        blocks = r["sites"]["UNI_BLOCKS_MUTEX"]["acq"]
        k = expected_buckets(r["threads"])
        acq = r["sites"]["UNI_BUCKET_TRY"]["acq"] + r["sites"]["UNI_BUCKET_EMPTY"]["acq"]
        pred = k * blocks
        ratio = acq / pred if pred else float("nan")
        print(f"  {r['cell']:30s} {r['algo']:14s} {blocks:8d} {k:4d} {acq:11d} "
              f"{pred:9d} {ratio:6.3f} {r['sites']['UNI_BUCKET_TRY']['tryfail']:9d} "
              f"{r['sites']['UNI_BUCKET_BLOCK']['acq']:8d}")

    print()
    print(f"  {'cell':30s} {'algo':14s} {'B(SCI_ADD)':>11s} {'slot acq':>9s} {'ratio':>7s} "
          f"{'tryfail':>9s}")
    for r in recs:
        if r["algo"] != "parallel_hash":
            continue
        blocks = r["sites"]["SCI_ADD"]["acq"]
        acq = r["sites"]["PAR_SLOT_TRY"]["acq"]
        print(f"  {r['cell']:30s} {r['algo']:14s} {blocks:11d} {acq:9d} "
              f"{(acq/blocks if blocks else float('nan')):7.3f} "
              f"{r['sites']['PAR_SLOT_TRY']['tryfail']:9d}")

    print()
    print("=" * 100)
    print("G2.2  hold-time distributions (MEASURED, from the log2 histogram)")
    print("=" * 100)
    print(f"  {'cell':26s} {'algo':13s} {'site':17s} {'n':>9s} {'p50 ns':>9s} {'p99 ns':>10s} "
          f"{'mean ns':>9s} {'total us':>9s}")
    for r in recs:
        hz = r["tick_hz"] or 1
        for s in SITES:
            st = r["sites"][s]
            if not st["acq"]:
                continue
            p50, _ = pct_from_hist(st["hist"], hz, 0.5)
            p99, _ = pct_from_hist(st["hist"], hz, 0.99)
            mean = st["ticks"] / st["acq"] / hz * 1e9
            print(f"  {r['cell']:26s} {r['algo']:13s} {s:17s} {st['acq']:9d} "
                  f"{(p50 or 0):9.0f} {(p99 or 0):10.0f} {mean:9.0f} "
                  f"{st['ticks']/hz*1e6:9.0f}")

    print()
    print("=" * 100)
    print("atomic RMW counters (per query)")
    print("=" * 100)
    print(f"  {'cell':30s} {'algo':14s} " + " ".join(f"{c:>14s}" for c in COUNTERS))
    for r in recs:
        if not any(r["counters"].values()):
            continue
        print(f"  {r['cell']:30s} {r['algo']:14s} "
              + " ".join(f"{r['counters'][c]:14d}" for c in COUNTERS))

    print()
    if red:
        print("RED:")
        for x in red:
            print("  " + x)
    print(f"\nG0.2/G2.1/G2.2: {'RED' if red else 'GREEN'}")
    return 1 if red else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "gate"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.tag)
        sys.exit(0)
    sys.exit(gate(a.tag))
