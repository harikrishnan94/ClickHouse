#!/usr/bin/env python3
"""Compare JOB suite passes: baseline vs uhj, under default and stats-disabled settings.

Metric matches the versions benchmark: min of the hot tries (tries[1:]).
Noise band: max(5%, 1 stdev of the baseline's own 5 hot tries).
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

OUT = Path("/mnt/data/uhj_versions_bench/job_study")


def load(arm: str, variant: str):
    p = OUT / f"job_{arm}_{variant}.tsv"
    if not p.exists():
        return None
    rows = {}
    for line in p.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        q = int(parts[0])
        vals = []
        for v in parts[1:]:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(None)
        rows[q] = vals
    return rows


def hot(vals):
    if not vals or len(vals) < 2:
        return None
    h = vals[1:]
    if any(x is None for x in h):
        return None
    return min(h)


def hot_all(vals):
    if not vals or len(vals) < 2:
        return []
    h = vals[1:]
    return [x for x in h if x is not None]


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def compare(variant: str):
    b = load("baseline", variant)
    u = load("uhj", variant)
    if not b or not u:
        print(f"[{variant}] missing data (baseline={bool(b)} uhj={bool(u)})")
        return None
    common = sorted(set(b) & set(u))
    regs, imps, noresult, nulls = [], [], [], []
    bg_list, ug_list = [], []
    for q in common:
        bh, uh = hot(b[q]), hot(u[q])
        if bh is None or uh is None:
            nulls.append(q)
            continue
        tries = hot_all(b[q])
        sd = statistics.stdev(tries) if len(tries) >= 2 else 0.0
        band = max(0.05, sd / bh if bh else 0.05)
        rel = (uh - bh) / bh
        bg_list.append(bh)
        ug_list.append(uh)
        rec = (q, bh, uh, rel, band)
        if abs(rel) <= band:
            noresult.append(rec)
        elif rel > 0:
            regs.append(rec)
        else:
            imps.append(rec)
    bg, ug = geomean(bg_list), geomean(ug_list)
    delta = (ug / bg - 1) if bg and ug else None
    print(f"\n===== JOB, variant={variant} =====")
    print(f"queries compared: {len(common)}  nulls: {len(nulls)}")
    print(f"geomean baseline={bg:.5f}s  uhj={ug:.5f}s  delta={delta*100:+.2f}%")
    print(f"outside noise: {len(regs)} regressions, {len(imps)} improvements; {len(noresult)} NO_RESULT")
    if regs:
        print("top regressions:")
        for q, bh, uh, rel, band in sorted(regs, key=lambda r: -r[3])[:15]:
            print(f"   q{q:<4} {bh*1000:8.1f}ms -> {uh*1000:8.1f}ms  {rel*100:+8.1f}%  (band +-{band*100:.1f}%)")
    if imps:
        print("top improvements:")
        for q, bh, uh, rel, band in sorted(imps, key=lambda r: r[3])[:10]:
            print(f"   q{q:<4} {bh*1000:8.1f}ms -> {uh*1000:8.1f}ms  {rel*100:+8.1f}%  (band +-{band*100:.1f}%)")
    return {"variant": variant, "geo_base": bg, "geo_uhj": ug, "delta": delta,
            "regs": regs, "imps": imps, "n": len(common)}


def cold_vs_warm(arm: str, variant: str):
    """A cold run much slower/faster than the hot ones is the signature of a plan flip."""
    rows = load(arm, variant)
    if not rows:
        return
    flips = []
    for q, vals in rows.items():
        if not vals or vals[0] is None:
            continue
        h = hot(vals)
        if not h:
            continue
        # Plan flips show up as hot runs SLOWER than the cold run (as on tpch q8).
        if h > vals[0] * 1.5 and (h - vals[0]) > 0.02:
            flips.append((q, vals[0], h))
    if flips:
        print(f"\n[{arm}/{variant}] queries whose HOT runs are >1.5x SLOWER than the cold run "
              f"(plan-flip signature): {len(flips)}")
        for q, c, h in sorted(flips, key=lambda r: -(r[2] / r[1]))[:10]:
            print(f"   q{q:<4} cold {c*1000:8.1f}ms  hot {h*1000:8.1f}ms  x{h/c:.1f}")
    else:
        print(f"\n[{arm}/{variant}] no hot-slower-than-cold plan-flip signature")


def main():
    res = {}
    for variant in ("default", "nostats"):
        r = compare(variant)
        if r:
            res[variant] = r
    for arm in ("baseline", "uhj"):
        for variant in ("default", "nostats"):
            cold_vs_warm(arm, variant)
    if "default" in res and "nostats" in res:
        d, n = res["default"], res["nostats"]
        print("\n===== verdict =====")
        print(f"delta with statistics ON  (as benchmarked): {d['delta']*100:+.2f}%")
        print(f"delta with statistics OFF (same plan both) : {n['delta']*100:+.2f}%")
        if n["delta"] is not None and abs(n["delta"]) > 0.05:
            print("=> regression SURVIVES the statistics control: it is an engine difference.")
        else:
            print("=> regression does NOT survive: it was a plan/statistics artifact.")


if __name__ == "__main__":
    main()
