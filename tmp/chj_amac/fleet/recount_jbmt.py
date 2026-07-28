#!/usr/bin/env python3
"""Independent, candidate-centric recount of a two-arm jbmt result set.

Two reasons this exists rather than quoting `report-ab`:

1. **Orientation.** `report-ab` labels verdicts from the *reference* arm's point
   of view - `join_bench_mt.py:1492` is `("win" if va < vb else "loss")` with
   `va` = arm A = baseline - so its "win" means the BASELINE was faster. Every
   other suite in this campaign is reported candidate-centrically. Quoting
   `report-ab`'s counts directly would invert the result, so they are recomputed
   and relabelled here, and the raw `report-ab` output is kept alongside.
2. **Independence.** This does not import `join_bench_mt`; the rule is
   re-implemented from its documented semantics (tie when the difference is
   within max(5%, 1 stdev) of the arms' run-to-run spread), so a disagreement
   with the harness is visible rather than impossible.

Usage: recount_jbmt.py 'RESULTS_GLOB' [--axis wall|memory]
"""
import collections
import glob
import json
import statistics
import sys

TIE_FRAC = 0.05


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def arm_stat(row, arm, algorithm, axis):
    """Per-arm metric for one algorithm, read straight out of the row."""
    entry = (((row.get("arms") or {}).get(arm) or {}).get("algorithms") or {}).get(algorithm)
    if not entry or entry.get("status") != "OK":
        return None, None, entry
    if axis == "wall":
        return entry.get("median_duration_ms"), entry.get("stdev_duration_ms") or 0.0, entry
    mem = entry.get("memories_bytes") or []
    return (entry.get("median_memory_bytes"),
            statistics.pstdev(mem) if len(mem) > 1 else 0.0, entry)


def main():
    pattern = sys.argv[1]
    axis = "wall"
    if "--axis" in sys.argv:
        axis = sys.argv[sys.argv.index("--axis") + 1]
    rows = load(pattern)
    print(f"rows={len(rows)}")
    print("statuses:", dict(collections.Counter(r.get("status") for r in rows)))
    print("tool_versions:", dict(collections.Counter(r.get("tool_version") for r in rows)))
    print("lead_arm distribution:", dict(collections.Counter(r.get("lead_arm") for r in rows)))

    shas = collections.defaultdict(set)
    for r in rows:
        for name, a in (r.get("arms") or {}).items():
            shas[name].add(str(a.get("binary_sha256"))[:12])
    print("arm -> binary sha256 prefixes:", {k: sorted(v) for k, v in shas.items()})

    algos = collections.Counter()
    for r in rows:
        for a in r.get("algorithms_measured") or []:
            algos[a] += 1
    print("algorithms measured:", dict(algos))

    counts = collections.Counter()
    ratios = []
    detail = []
    skipped = collections.Counter()
    fallback = {}
    ALGO = "parallel_hash"
    for r in rows:
        if r.get("status") != "OK":
            skipped[r.get("status")] += 1
            continue
        va, sa, ea = arm_stat(r, "baseline", ALGO, axis)
        vb, sb, eb = arm_stat(r, "candidate", ALGO, axis)
        for e in (ea, eb):
            if e and e.get("fallback_runs"):
                fallback[r["unit_id"]] = e.get("fallback_runs")
        if not va or not vb:
            skipped["no-metric"] += 1
            continue
        # The documented noise band: 5% of the LARGER median, or the larger
        # per-arm stdev, whichever is bigger.
        band = max(TIE_FRAC * max(va, vb), max(sa, sb))
        ratio = vb / va
        if abs(vb - va) <= band:
            v = "TIE"
        elif vb < va:
            v = "WIN"       # candidate faster / smaller
        else:
            v = "LOSS"
        counts[v] += 1
        ratios.append(ratio)
        detail.append((ratio, r["unit_id"], v))

    print()
    print(f"--- candidate-centric {axis} verdicts (WIN = candidate better) ---")
    print(f"units scored={sum(counts.values())} "
          + " ".join(f"{k.lower()}={counts[k]}" for k in ("WIN", "TIE", "LOSS")))
    if skipped:
        print("not scored:", dict(skipped))
    print("units with any fallback_runs > 0:", fallback if fallback else "none")
    if ratios:
        print(f"median ratio candidate/baseline = {statistics.median(ratios):.3f} "
              f"(>1 means the candidate is slower/larger)")
    detail.sort()
    print()
    print("10 biggest candidate WINS (lowest ratio):")
    for ratio, uid, v in detail[:10]:
        print(f"  {ratio:.3f}  {uid}")
    print("10 biggest candidate LOSSES (highest ratio):")
    for ratio, uid, v in detail[-10:][::-1]:
        print(f"  {ratio:.3f}  {uid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
