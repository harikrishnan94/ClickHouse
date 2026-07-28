#!/usr/bin/env python3
"""Unit 5 analysis for the phj-ph A/B campaign: fleet_ab verdicts, the probe-event
gate metrics, and the U5 changed-verdict comparison.

Everything is recomputed from the raw JSONL rows using fleet_ab's own
`cell_verdicts`, so the numbers here and the numbers in
`fleet_ab.py report` come from one scoring function - no second implementation
that could disagree. The U5 side is recomputed the same way from U5's raw rows
rather than lifted from its REPORT.md, because U5 scored its own campaign with a
different script (`gate_verdicts.py`) over a different cell set; comparing its
published labels against fleet_ab labels would compare two scoring functions and
call the difference a changed verdict.

Usage:
  analyze_phj_ph.py --now 'DIR/results.shard*.jsonl' --u5 'DIR/results.shard*.jsonl'
"""
import argparse
import collections
import glob
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import fleet_ab  # noqa: E402

PROBE_LOOKUP = "ConcurrentHashJoinProbeLookupMicroseconds"
AMAC_PROBE = "ConcurrentHashJoinAmacProbeRows"
AMAC_BUILD = "ConcurrentHashJoinAmacBuildRows"
AMAC_GROWTH = "ConcurrentHashJoinBuildRingGrowths"


def load(patterns):
    files = []
    for p in patterns:
        files += sorted(glob.glob(p))
    rows = []
    for f in files:
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    return files, rows


def verdicts(rows):
    return fleet_ab.cell_verdicts(fleet_ab.dedup_last_attempt(rows))


def median_event(rows, role, event):
    vals = []
    for r in rows:
        if r.get("arm_role") != role or not r.get("valid"):
            continue
        ev = r.get("events") or {}
        if event in ev:
            vals.append(ev[event])
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", nargs="+", required=True)
    ap.add_argument("--u5", nargs="+")
    args = ap.parse_args()

    now_files, now_rows = load(args.now)
    nv = verdicts(now_rows)
    print(f"=== THIS CAMPAIGN ({len(now_files)} files, {len(now_rows)} rows) ===")
    counts = collections.Counter(v["verdict"] for v in nv.values())
    print(f"cells={len(nv)} " + " ".join(f"{k.lower()}={counts[k]}" for k in
          ("WIN", "TIE", "LOSS", "INVALID", "INSUFFICIENT")))
    shas = collections.Counter((r.get("arm_role"), str(r.get("binary_sha256"))[:12]) for r in now_rows)
    print("arm -> binary sha256 prefix:", dict(shas))

    # Which planned cells produced no rows at all (attempted but unmeasurable).
    planned = json.load(open(pathlib.Path(__file__).resolve().parent / "matrix.json"))["measured_plan"]["cells"]
    missing = [c for c in planned if c not in nv]
    print(f"planned={len(planned)} with_rows={len(nv)} no_rows={len(missing)}: {missing}")

    # --- probe-event gate metrics -----------------------------------------
    by_cell = collections.defaultdict(list)
    for r in fleet_ab.dedup_last_attempt(now_rows):
        by_cell[r["cell"]].append(r)

    engaged = notengaged = nocounter = 0
    eng_detail = []
    for cell, rs in sorted(by_cell.items()):
        b = [r for r in rs if r.get("arm_role") == "B"]
        vals = [(r.get("engagement") or {}).get(AMAC_PROBE) for r in b]
        vals = [v for v in vals if v is not None]
        if not vals:
            nocounter += 1
            eng_detail.append((cell, None))
        elif max(vals) > 0:
            engaged += 1
            eng_detail.append((cell, max(vals)))
        else:
            notengaged += 1
            eng_detail.append((cell, 0))
    print()
    print("--- probe-event gate metric (candidate arm B) ---")
    print(f"{AMAC_PROBE}: engaged(>0) in {engaged} cells; zero in {notengaged}; "
          f"counter absent in {nocounter} (of {len(by_cell)})")

    # Probe-side phase attribution: the event a probe-side win must be carried by.
    print()
    print(f"--- {PROBE_LOOKUP} median per arm, probe-side cells ---")
    print(f"{'cell':<48}{'verdict':<9}{'A(us)':>14}{'B(us)':>14}{'delta%':>9}")
    lookup_rows = []
    for cell, rs in sorted(by_cell.items()):
        if ":probe" not in cell:
            continue
        a = median_event(rs, "A", PROBE_LOOKUP)
        b = median_event(rs, "B", PROBE_LOOKUP)
        v = nv.get(cell, {}).get("verdict", "?")
        if a is None or b is None:
            continue
        d = (b - a) / a * 100 if a else float("nan")
        lookup_rows.append((cell, v, a, b, d))
        print(f"{cell:<48}{v:<9}{a:>14.0f}{b:>14.0f}{d:>+9.2f}")
    if lookup_rows:
        imp = [r for r in lookup_rows if r[4] < 0]
        print(f"probe-side cells with {PROBE_LOOKUP} lower on the candidate: "
              f"{len(imp)}/{len(lookup_rows)}")

    # --- U5 comparison ----------------------------------------------------
    if args.u5:
        u5_files, u5_rows = load(args.u5)
        uv = verdicts(u5_rows)
        u5counts = collections.Counter(v["verdict"] for v in uv.values())
        print()
        print(f"=== U5 PRECEDENT ({len(u5_files)} files, {len(u5_rows)} rows), "
              "rescored with fleet_ab.cell_verdicts ===")
        print(f"cells={len(uv)} " + " ".join(f"{k.lower()}={u5counts[k]}" for k in
              ("WIN", "TIE", "LOSS", "INVALID", "INSUFFICIENT")))
        u5shas = collections.Counter((r.get("arm_role"), str(r.get("binary_sha256"))[:12]) for r in u5_rows)
        print("arm -> binary sha256 prefix:", dict(u5shas))

        both = sorted(set(nv) & set(uv))
        only_now = sorted(set(nv) - set(uv))
        only_u5 = sorted(set(uv) - set(nv))
        print()
        print(f"--- cell-set overlap: both={len(both)} only-now={len(only_now)} only-U5={len(only_u5)} ---")
        if only_now:
            print("  only in this campaign:", only_now)
        if only_u5:
            print("  only in U5:", only_u5)

        changed = [(c, uv[c]["verdict"], nv[c]["verdict"]) for c in both
                   if uv[c]["verdict"] != nv[c]["verdict"]]
        print()
        print(f"--- CHANGED VERDICTS: {len(changed)} of {len(both)} shared cells ---")
        print(f"{'cell':<48}{'U5':<14}{'now':<14}{'U5 diff%':>10}{'now diff%':>11}")
        for c, o, n in changed:
            print(f"{c:<48}{o:<14}{n:<14}"
                  f"{uv[c].get('diff_pct', float('nan')):>10.2f}"
                  f"{nv[c].get('diff_pct', float('nan')):>11.2f}")
        print()
        print("--- transition matrix (U5 -> now) ---")
        for (o, n), k in sorted(collections.Counter(
                (uv[c]["verdict"], nv[c]["verdict"]) for c in both).items()):
            print(f"  {o:<13} -> {n:<13} {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
