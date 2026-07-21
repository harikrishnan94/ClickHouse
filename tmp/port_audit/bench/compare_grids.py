#!/usr/bin/env python3
"""Compare two grid runs (PREREG.md protocol): per point, partitioned_hash wall
median/min, the declared phase counter, and the acceptance verdict.

Usage: compare_grids.py <base_tag> <cand_tag> <phase_event> [--algo partitioned_hash]
Band per point (PREREG amendment): max(3%, 2*(median-min)/median) of the BASE run.
"""

import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
POINTS = ["V", "A", "B", "C", "D", "E"]


def parse(tag: str, point: str, algo: str):
    path = BENCH / f"{tag}_{point}.log"
    if not path.exists():
        return None
    text = path.read_text()
    row = re.search(rf"^{algo}\s+(\S+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)", text, re.M)
    events_m = re.search(rf"^PhaseEvents {algo}: (.*)$", text, re.M)
    if not row or not events_m:
        return None
    events = dict(kv.split("=") for kv in events_m.group(1).split())
    verify = row.group(2)
    return {
        "status": row.group(1),
        "verify": verify,
        "median": float(row.group(3)),
        "min": float(row.group(4)),
        "events": {k: int(v) for k, v in events.items()},
    }


def main():
    base_tag, cand_tag, phase_event = sys.argv[1], sys.argv[2], sys.argv[3]
    algo = "partitioned_hash"
    print(f"point  base_med  cand_med  wall_delta  band     phase_base  phase_cand  phase_delta  verdict")
    improved_any = False
    regressed_any = False
    for p in POINTS:
        b = parse(base_tag, p, algo)
        c = parse(cand_tag, p, algo)
        if not b or not c:
            print(f"{p:5}  MISSING DATA (base={bool(b)} cand={bool(c)})")
            continue
        band = max(0.03, 2 * (b["median"] - b["min"]) / b["median"])
        wall_delta = (c["median"] - b["median"]) / b["median"]
        pb = b["events"].get(phase_event, 0)
        pc = c["events"].get(phase_event, 0)
        phase_delta = (pc - pb) / pb if pb else float("nan")
        verdict = []
        if wall_delta < -band:
            verdict.append("WALL-IMPROVED")
        if wall_delta > band:
            verdict.append("WALL-REGRESSED")
            regressed_any = True
        if pb and phase_delta < -band:
            verdict.append("PHASE-IMPROVED")
        if wall_delta < -band and pb and phase_delta < -band:
            improved_any = True
        flags = []
        if b["status"] != "OK" or c["status"] != "OK":
            flags.append("STATUS!")
        if "FAIL" in (b["verify"], c["verify"]) or "ERROR" in (b["verify"], c["verify"]):
            flags.append("VERIFY!")
        print(
            f"{p:5}  {b['median']:8.0f}  {c['median']:8.0f}  {wall_delta:+9.1%}  {band:6.1%}"
            f"  {pb:10d}  {pc:10d}  {phase_delta:+10.1%}  {' '.join(verdict + flags) or 'in-band'}"
        )
    print()
    print(f"ACCEPT (>=1 point wall+phase improved beyond band AND no wall regression beyond band): "
          f"{'YES' if improved_any and not regressed_any else 'NO'}"
          f"  [improved_any={improved_any} regressed_any={regressed_any}]")


if __name__ == "__main__":
    main()
