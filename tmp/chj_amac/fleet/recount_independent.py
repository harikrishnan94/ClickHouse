#!/usr/bin/env python3
"""Independent recount of the fleet_ab verdicts, plus an ABAB-leader sensitivity check.

This deliberately does NOT import fleet_ab. The verdict rule is re-implemented
from the harness's documented semantics so that a disagreement with
`fleet_ab.py report` is visible rather than impossible:

  * a cell needs both arms present and every row valid, else INVALID
  * a cell needs >= MIN_VERDICT_RUNS (5) valid runs per arm, else INSUFFICIENT
  * verdict compares median(duration_us) of B against A, with a band of
    max(3%, per-cell relative spread) - the same band the harness prints per cell,
    which is read back out of the report text rather than recomputed, so the
    comparison is against the harness's own band and not a guess

The sensitivity pass answers a specific defect found in verification: because
`run_sweep_stealing.py` runs one cell per `fleet_ab.py sweep` invocation,
`cell_index` is always 0, so fleet_ab's `order_pair = (0, 1) if cell_index % 2
== 0 else (1, 0)` never flipped and arm A led the pair in all 93 cells. The
within-cell interleave is still strict ABAB (A at even positions, B at odd), so
the residual exposure is a first-position effect on one of each arm's ten runs.
The pass recomputes every verdict with the first A/B pair dropped; a verdict that
survives that is not an artifact of who led.

Usage: recount_independent.py 'RESULTS_GLOB' [REPORT_TXT]
"""
import collections
import glob
import json
import re
import statistics
import sys

MIN_VERDICT_RUNS = 5
DEFAULT_BAND = 0.03


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bands_from_report(path):
    """Read the per-cell band the harness itself used, so the band is not my guess."""
    bands = {}
    if not path:
        return bands
    for line in open(path):
        m = re.match(r"^CELL (\S+) verdict=\S+ .*band=([0-9.]+)%", line)
        if m:
            bands[m.group(1)] = float(m.group(2)) / 100.0
    return bands


def freshest_attempt(rows):
    """Keep only the newest attempt per (cell, arm_role, run), keyed by nonce."""
    newest = {}
    for r in rows:
        key = (r["cell"], r["arm_role"], r["run"])
        prev = newest.get(key)
        if prev is None or r.get("recorded_at", "") >= prev.get("recorded_at", ""):
            newest[key] = r
    return list(newest.values())


def verdicts(rows, bands, drop_first_pair=False):
    by_cell = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if drop_first_pair and r.get("run") == 0:
            continue
        by_cell[r["cell"]][r["arm_role"]].append(r)
    out = {}
    for cell, roles in by_cell.items():
        allrows = [r for rs in roles.values() for r in rs]
        if any(not r.get("valid") for r in allrows) or set(roles) != {"A", "B"}:
            out[cell] = ("INVALID", None)
            continue
        if any(len(rs) < MIN_VERDICT_RUNS for rs in roles.values()):
            out[cell] = ("INSUFFICIENT", None)
            continue
        med = {k: statistics.median([r["duration_us"] for r in rs]) for k, rs in roles.items()}
        diff = (med["B"] - med["A"]) / med["A"]
        band = bands.get(cell, DEFAULT_BAND)
        if abs(diff) <= band:
            v = "TIE"
        elif diff < 0:
            v = "WIN"
        else:
            v = "LOSS"
        out[cell] = (v, diff * 100)
    return out


def summarize(name, v):
    c = collections.Counter(x[0] for x in v.values())
    print(f"{name}: cells={len(v)} " + " ".join(
        f"{k.lower()}={c[k]}" for k in ("WIN", "TIE", "LOSS", "INVALID", "INSUFFICIENT")))
    return c


def main():
    pattern = sys.argv[1]
    report = sys.argv[2] if len(sys.argv) > 2 else None
    rows = freshest_attempt(load(pattern))
    bands = bands_from_report(report)
    print(f"rows={len(rows)} bands_read_from_report={len(bands)}")

    print()
    print("--- independent recount (no fleet_ab import) ---")
    full = verdicts(rows, bands)
    summarize("all 10 runs", full)

    print()
    print("--- ABAB leader check: who led the pair in each cell ---")
    lead = collections.Counter()
    for r in rows:
        if r.get("run") == 0 and r.get("position") == 0:
            lead[r["arm_role"]] += 1
    print(f"arm leading at run 0 (position 0): {dict(lead)}")
    print("within-cell positions for one cell (should strictly alternate A,B,A,B,...):")
    sample = sorted([r for r in rows if r["cell"] == sorted({x['cell'] for x in rows})[0]],
                    key=lambda r: r.get("position", -1))
    print("  " + " ".join(f"{r['position']}:{r['arm_role']}" for r in sample[:12]))

    print()
    print("--- sensitivity: drop the first A/B pair, recompute every verdict ---")
    dropped = verdicts(rows, bands, drop_first_pair=True)
    summarize("runs 1..9 only", dropped)
    flips = [(c, full[c][0], dropped[c][0]) for c in sorted(full)
             if c in dropped and full[c][0] != dropped[c][0]]
    print(f"verdicts that change when the leading pair is dropped: {len(flips)}")
    for c, a, b in flips:
        print(f"  {c:<48}{a} -> {b}  (diff {full[c][1]:+.2f}% -> {dropped[c][1]:+.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
