#!/usr/bin/env python3
"""Render the PREREG 10-cell table from a suite.py JSONL output."""
import json
import sys

CELL_ORDER = [f"{s}_T{t}" for s in ("A", "C") for t in (96, 64, 32, 16, 1)]


def main(path):
    summaries = {}
    header = footer = None
    for line in open(path):
        row = json.loads(line)
        if row.get("kind") == "summary":
            summaries[row["cell"]] = row
        elif row.get("kind") == "header":
            header = row
        elif row.get("kind") == "footer":
            footer = row
    if header:
        print(f"binary A: {header['binary_a']['path']} sha256 "
              f"{header['binary_a']['sha256']}")
    if footer:
        print(f"footer: status={footer.get('status')} "
              f"binary_stable={footer.get('binary_stable')}")
    print()
    print("| cell | n pairs | median (s) | min | max | stdev | SE(log) | band |")
    print("|------|---------|------------|-----|-----|-------|---------|------|")
    for cell in CELL_ORDER:
        s = summaries.get(cell)
        if not s:
            print(f"| {cell} | (missing) | | | | | | |")
            continue
        fmt = lambda v, d=3: (f"{v / 1e6:.{d}f}" if v is not None else "-")
        se = f"{s['se']:.5f}" if s.get("se") is not None else "-"
        band = f"{s['band'] * 100:.2f}%" if s.get("band") is not None else "-"
        stdev = fmt(s.get("stdev_us"))
        print(f"| {cell} | {s['n_pairs']} | {fmt(s['median_us'])} | "
              f"{fmt(s['min_us'])} | {fmt(s['max_us'])} | {stdev} | {se} | {band} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/mnt/ch/ClickHouse/tmp/wave-join-impl/baseline_u0.jsonl")
