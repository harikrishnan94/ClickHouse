#!/usr/bin/env python3
"""Grouped roll-up of the A / B_old / B_new table, by the groups the fixes target.

Reads the same records `abreport.py` does. Prints, per group, the median and the
extremes of B_new vs B_old and of B_new vs A, plus how many cells regressed - so
that "nothing at 16/64 threads got slower" is checked rather than asserted.
"""

from __future__ import annotations

import argparse
import statistics
import sys

import abreport as R

USED_FLAG_KINDS = ("RIGHT", "FULL", "LEFT-SEMI", "LEFT-ANTI")


def cell_parts(cell):
    kind, key, match, threads, card = cell.split("|")
    return kind, key, match, int(threads[1:]), card


def collect(data, order, metric):
    rows = []
    for cell in order:
        algos = {a for (t, c, a) in data if c == cell}
        cmpr = next((a for a in ("hash", "parallel_hash") if a in algos), None)
        get = lambda tag, algo: data.get((tag, cell, algo), {}).get(metric)  # noqa: E731
        b_old, b_new = get("BEFORE", "unified_hash"), get("AFTER", "unified_hash")
        if b_old is None or b_new is None:
            continue
        kind, key, match, threads, card = cell_parts(cell)
        rows.append({
            "cell": cell, "kind": kind, "key": key, "threads": threads, "card": card,
            "cmp": cmpr, "A_old": get("BEFORE", cmpr), "A_new": get("AFTER", cmpr),
            "B_old": b_old, "B_new": b_new,
        })
    return rows


def show(label, rows, metric):
    rows = [r for r in rows if r["B_old"]]
    if not rows:
        print(f"{label:52s}  (no cells)")
        return
    dbb = [R.pct(r["B_new"], r["B_old"]) for r in rows]
    dba = [R.pct(r["B_new"], r["A_new"]) for r in rows if r["A_new"]]
    oa = [R.pct(r["B_old"], r["A_old"]) for r in rows if r["A_old"]]
    worse = [r for r in rows if R.pct(r["B_new"], r["B_old"]) > 2.0]
    med = f"{statistics.median(dbb):+6.1f}"
    rng = f"[{min(dbb):+6.1f} .. {max(dbb):+6.1f}]"
    va = f"{statistics.median(dba):+6.1f}" if dba else "     -"
    vo = f"{statistics.median(oa):+6.1f}" if oa else "     -"
    print(f"{label:52s} n={len(rows):3d}  B_new/B_old {med}% {rng}   "
          f"vs A: old {vo}% new {va}%   >+2%: {len(worse)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--runs", default=R.RUNS)
    args = ap.parse_args()

    raw, order = R.load(args.runs, {args.before, args.after})
    data = {}
    for (tag, cell, algo), v in raw.items():
        data[("BEFORE" if tag == args.before else "AFTER", cell, algo)] = v

    for metric in ("wall_ms", "cpu_us", "build_us", "probe_us", "nonjoined_us"):
        rows = collect(data, order, metric)
        print(f"\n=== {metric} ===")
        for t in (1, 16, 64):
            show(f"all keys, t{t}", [r for r in rows if r["threads"] == t], metric)
        for key in ("u64", "str", "comp"):
            for t in (16, 64):
                show(f"{key} key, t{t}", [r for r in rows if r["key"] == key and r["threads"] == t], metric)
        show("t1, used-flag kinds (RIGHT/FULL/SEMI/ANTI)",
             [r for r in rows if r["threads"] == 1 and r["kind"] in USED_FLAG_KINDS], metric)
        show("t1, INNER/LEFT", [r for r in rows if r["threads"] == 1 and r["kind"] not in USED_FLAG_KINDS], metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
