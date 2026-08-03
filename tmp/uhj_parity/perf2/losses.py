#!/usr/bin/env python3
"""Where `unified_hash` still loses to its comparator, per phase and per cell factor.

A "loss" is a cell where `unified_hash`'s median exceeds the comparator's by more than
`--band` percent, both measured in the same sweep run so machine drift cancels. The band
defaults to 2%, which is roughly the A/A noise floor for wall and CPU at 16/64 threads and
well below it for one-thread `build_us` - so the one-thread build column is reported but
should not be read as signal (see `--band` and the A/A table in REPORT_FIX.md).

Prints, per phase: the loss count by thread count, then a cross-tabulation over the cell
factors (kind, key, cardinality, match rate) so a pattern shows up as a concentration
rather than as a list.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys

import abreport as R

PHASES = ("wall_ms", "cpu_us", "build_us", "probe_us", "nonjoined_us")


def rows_for(runs, tag, metric):
    data, order = R.load(runs, {tag})
    out = []
    for cell in order:
        algos = {a for (t, c, a) in data if c == cell}
        cmpr = next((a for a in ("hash", "parallel_hash") if a in algos), None)
        if cmpr is None:
            continue                      # SEMI/ANTI above one thread: no comparator exists
        a = data.get((tag, cell, cmpr), {}).get(metric)
        b = data.get((tag, cell, "unified_hash"), {}).get(metric)
        if not a or not b:
            continue
        kind, key, match, threads, card = cell.split("|")
        out.append({"cell": cell, "kind": kind, "key": key, "match": match,
                    "threads": int(threads[1:]), "card": card, "cmp": cmpr,
                    "A": a, "B": b, "pct": 100.0 * (b - a) / a})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default=R.RUNS)
    ap.add_argument("--band", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    print(f"tag={args.tag}  a cell 'loses' when unified_hash is more than {args.band}% above "
          f"its comparator (hash at t1, parallel_hash at t16/t64)\n")

    for metric in PHASES:
        rows = rows_for(args.runs, args.tag, metric)
        if not rows:
            continue
        losses = [r for r in rows if r["pct"] > args.band]
        print(f"=== {metric}: {len(losses)} losing cells of {len(rows)} comparable ===")
        for t in (1, 16, 64):
            at = [r for r in rows if r["threads"] == t]
            lo = [r for r in at if r["pct"] > args.band]
            if not at:
                continue
            med = statistics.median(r["pct"] for r in at)
            worst = max(at, key=lambda r: r["pct"])
            print(f"   t{t:<3d} {len(lo):3d}/{len(at):3d} lose   median {med:+6.1f}%   "
                  f"worst {worst['pct']:+6.1f}% ({worst['cell']})")
        if losses:
            for factor in ("kind", "key", "card", "match"):
                tot = collections.Counter(r[factor] for r in rows)
                cnt = collections.Counter(r[factor] for r in losses)
                cells = "  ".join(f"{v}={cnt.get(v, 0)}/{tot[v]}" for v in sorted(tot))
                print(f"      by {factor:5s}: {cells}")
            print(f"      worst {min(args.top, len(losses))}:")
            for r in sorted(losses, key=lambda r: -r["pct"])[:args.top]:
                print(f"        {r['pct']:+6.1f}%  {r['cell']:34s} A={r['A']:10.1f} B={r['B']:10.1f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
