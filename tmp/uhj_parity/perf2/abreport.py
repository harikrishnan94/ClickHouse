#!/usr/bin/env python3
"""A / B_old / B_new table over two sweep run tags.

`A` is the cell's comparator (`hash` at one thread, `parallel_hash` above it),
`B_old` is `unified_hash` from the before-tag and `B_new` from the after-tag.
Medians across the timed repetitions; the comparator is reported from BOTH tags
so that a drifting machine shows up as a moving `A` rather than as a fake win.

Reads only, judges nothing beyond arithmetic: it prints the percentages and
leaves the verdict to the report.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

PERF2 = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(PERF2, "..", "perf", "results", "runs.jsonl")

METRICS = ("wall_ms", "cpu_us", "build_us", "probe_us", "nonjoined_us")


def load(path, tags):
    """(tag, cell_id, algo) -> {metric: median over timed reps}."""
    per = {}
    order = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("purpose") != "timed" or rec.get("run_tag") not in tags:
                continue
            if "wall_ms" not in rec:
                continue
            key = (rec["run_tag"], rec["cell_id"], rec["algo"])
            if key not in per:
                per[key] = {m: [] for m in METRICS}
                if rec["cell_id"] not in order:
                    order.append(rec["cell_id"])
            for m in METRICS:
                if m in rec:
                    per[key][m].append(float(rec[m]))
    out = {k: {m: statistics.median(v) for m, v in d.items() if v} for k, d in per.items()}
    return out, order


def pct(new, old):
    if not old:
        return None
    return 100.0 * (new - old) / old


def fmt_pct(v):
    return "     -" if v is None else f"{v:+6.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="run tag of the B_old sweep")
    ap.add_argument("--after", required=True, help="run tag of the B_new sweep")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--metric", default="wall_ms", choices=METRICS)
    ap.add_argument("--filter", default="", help="substring filter on cell_id")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    data, order = load(args.runs, {args.before, args.after})
    if not data:
        print("no timed records for those tags", file=sys.stderr)
        return 1

    cells = [c for c in order if args.filter in c]
    metric = args.metric
    scale = 1000.0 if metric.endswith("_us") else 1.0

    rows = []
    for cell in cells:
        algos = {a for (t, c, a) in data if c == cell}
        comparator = next((a for a in ("hash", "parallel_hash") if a in algos), None)
        a_old = data.get((args.before, cell, comparator), {}).get(metric)
        a_new = data.get((args.after, cell, comparator), {}).get(metric)
        b_old = data.get((args.before, cell, "unified_hash"), {}).get(metric)
        b_new = data.get((args.after, cell, "unified_hash"), {}).get(metric)
        if b_old is None or b_new is None:
            continue
        rows.append({
            "cell": cell, "comparator": comparator,
            "A_old": a_old, "A_new": a_new, "B_old": b_old, "B_new": b_new,
            "d_new_vs_old": pct(b_new, b_old),
            "d_new_vs_A": pct(b_new, a_new) if a_new else None,
            "d_old_vs_A": pct(b_old, a_old) if a_old else None,
            "d_A_drift": pct(a_new, a_old) if a_old and a_new else None,
        })

    if args.json:
        print(json.dumps(rows, indent=1))
        return 0

    unit = "ms" if scale == 1000.0 or metric.endswith("_ms") else ""
    print(f"metric={metric} (in {unit or 'raw'}); A = cell comparator; "
          f"before={args.before} after={args.after}")
    print(f"{'cell':38s} {'cmp':14s} {'A_old':>9s} {'A_new':>9s} {'B_old':>9s} {'B_new':>9s} "
          f"{'B_new/B_old':>11s} {'B_old/A':>8s} {'B_new/A':>8s} {'A drift':>8s}")
    for r in rows:
        def f(v):
            return "        -" if v is None else f"{v / scale:9.1f}"
        print(f"{r['cell']:38s} {str(r['comparator']):14s} "
              f"{f(r['A_old'])} {f(r['A_new'])} {f(r['B_old'])} {f(r['B_new'])} "
              f"{fmt_pct(r['d_new_vs_old']):>11s} {fmt_pct(r['d_old_vs_A']):>8s} "
              f"{fmt_pct(r['d_new_vs_A']):>8s} {fmt_pct(r['d_A_drift']):>8s}")

    moved = [r["d_new_vs_old"] for r in rows if r["d_new_vs_old"] is not None]
    drift = [r["d_A_drift"] for r in rows if r["d_A_drift"] is not None]
    worse_vs_A = [r for r in rows if r["d_new_vs_A"] is not None and r["d_new_vs_A"] > 0]
    print()
    print(f"cells={len(rows)}  B_new vs B_old: median {statistics.median(moved):+.2f}% "
          f"min {min(moved):+.2f}% max {max(moved):+.2f}%")
    if drift:
        print(f"comparator drift A_new vs A_old: median {statistics.median(drift):+.2f}% "
              f"min {min(drift):+.2f}% max {max(drift):+.2f}%")
    print(f"cells where B_new is slower than A: {len(worse_vs_A)} of {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
