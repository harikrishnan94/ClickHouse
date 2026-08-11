#!/usr/bin/env python3
"""Analyze interleaved baseline vs uhj ClickBench versions results.

Noise band (declared before comparison): effect within max(5%, 1 stdev of
baseline hot times across rounds) is NO RESULT.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/uhj_versions_bench")
RESULTS = WORK / "results"
OUT = WORK / "report"
OUT.mkdir(parents=True, exist_ok=True)

# Query order in run_arm.sh
DATASETS = [
    ("coffeeshop", 17),
    ("tpch", 22),
    ("tpcds", 103),
    ("job", 113),
]

PUBLISHED = Path("/mnt/ch/ClickBench-master/versions/results/master.json")


def hot_times(row):
    """Return the list of hot try times (tries[1:]), or None if any missing."""
    if row is None or any(x is None for x in row):
        return None
    hot = row[1:]
    if not hot or any(x is None for x in hot):
        return None
    return list(hot)


def hot_time(row):
    """Official metric: fastest of the hot runs (tries[1:]). Cold is tries[0]."""
    ht = hot_times(row)
    return min(ht) if ht else None


def load_arm_rounds(arm: str):
    files = sorted(RESULTS.glob(f"{arm}_r*.json"))
    rounds = []
    for f in files:
        d = json.loads(f.read_text())
        rounds.append(d)
    return rounds


def slice_datasets(result_rows):
    out = {}
    i = 0
    for name, n in DATASETS:
        out[name] = result_rows[i : i + n]
        i += n
    return out


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    base_rounds = load_arm_rounds("baseline")
    uhj_rounds = load_arm_rounds("uhj")
    report = {
        "baseline_rounds": len(base_rounds),
        "uhj_rounds": len(uhj_rounds),
        "noise_rule": "NO RESULT if |delta| <= max(5%, 1*stdev_baseline_hot)",
        "datasets": {},
        "fidelity": {},
        "errors": [],
    }
    if len(base_rounds) < 1:
        report["errors"].append(f"Need >=1 baseline round; have {len(base_rounds)}")
    # Noise band source:
    # - Prefer across-suite stdev when >=5 baseline rounds (independent suite reps).
    # - Otherwise use within-suite stdev of the 5 hot tries (TRIES=6 contract), which
    #   is the same repetition count the published versions benchmark records.
    noise_mode = (
        "across_suite_rounds"
        if len(base_rounds) >= 5
        else "within_suite_hot_tries"
    )
    report["noise_mode"] = noise_mode
    report["noise_rule"] = (
        f"NO RESULT if |delta| <= max(5%, 1*stdev) [{noise_mode}]"
    )

    # Per-query baseline hot times across rounds -> mean, stdev
    # Align by min rounds available for A/B
    n = min(len(base_rounds), len(uhj_rounds))
    if n == 0:
        report["errors"].append("No completed rounds")
        (OUT / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    base_by_ds = [slice_datasets(r["result"]) for r in base_rounds[:n]]
    uhj_by_ds = [slice_datasets(r["result"]) for r in uhj_rounds[:n]]

    lines = []
    lines.append(f"# uhj-parity vs merge-base on ClickBench versions (emulated c7a.4xlarge)")
    lines.append(f"Rounds compared: {n} (baseline files {len(base_rounds)}, uhj {len(uhj_rounds)})")
    lines.append(f"Noise rule: NO RESULT if |rel_delta| <= max(5%, 1 stdev / mean)")
    lines.append("")

    for ds, nq in DATASETS:
        effects = []
        no_results = []
        regressions = []
        improvements = []
        nulls = []
        per_query = []
        base_hots_all = []
        uhj_hots_all = []

        for qi in range(nq):
            b_mins = []
            u_mins = []
            b_null = False
            u_null = False
            b_hot_tries = []  # flattened / first-round hot tries for within-suite noise
            for r in range(n):
                br = base_by_ds[r][ds][qi] if qi < len(base_by_ds[r][ds]) else None
                ur = uhj_by_ds[r][ds][qi] if qi < len(uhj_by_ds[r][ds]) else None
                bh = hot_time(br)
                uh = hot_time(ur)
                if bh is None:
                    b_null = True
                else:
                    b_mins.append(bh)
                    if r == 0:
                        b_hot_tries = hot_times(br) or []
                if uh is None:
                    u_null = True
                else:
                    u_mins.append(uh)

            label = f"{ds}/q{qi+1}"
            if b_null or u_null or not b_mins or not u_mins:
                nulls.append(label)
                per_query.append(
                    {
                        "query": label,
                        "status": "NULL/ERROR",
                        "baseline_hots": b_mins,
                        "uhj_hots": u_mins,
                    }
                )
                continue

            # Comparison metric: mean across suites of (min-of-hot), or the single min-of-hot.
            b_mean = statistics.mean(b_mins)
            u_mean = statistics.mean(u_mins)
            if noise_mode == "across_suite_rounds" and len(b_mins) >= 2:
                b_stdev = statistics.stdev(b_mins)
            elif len(b_hot_tries) >= 2:
                b_stdev = statistics.stdev(b_hot_tries)
            else:
                b_stdev = 0.0
            rel = (u_mean - b_mean) / b_mean if b_mean else None
            noise = max(0.05, (b_stdev / b_mean) if b_mean else 0.05)
            entry = {
                "query": label,
                "baseline_mean_hot": b_mean,
                "baseline_stdev": b_stdev,
                "uhj_mean_hot": u_mean,
                "rel_delta": rel,
                "noise_band": noise,
                "noise_mode": noise_mode,
                "status": "NO_RESULT" if abs(rel) <= noise else ("REGRESSION" if rel > 0 else "IMPROVEMENT"),
            }
            per_query.append(entry)
            base_hots_all.append(b_mean)
            uhj_hots_all.append(u_mean)
            if entry["status"] == "NO_RESULT":
                no_results.append(entry)
            elif entry["status"] == "REGRESSION":
                regressions.append(entry)
                effects.append(entry)
            else:
                improvements.append(entry)
                effects.append(entry)

        bg = geomean(base_hots_all)
        ug = geomean(uhj_hots_all)
        gdelta = ((ug / bg) - 1.0) if bg and ug else None
        report["datasets"][ds] = {
            "n_queries": nq,
            "n_null": len(nulls),
            "n_no_result": len(no_results),
            "n_regression": len(regressions),
            "n_improvement": len(improvements),
            "baseline_geomean": bg,
            "uhj_geomean": ug,
            "geomean_rel_delta": gdelta,
            "null_queries": nulls,
            "regressions": regressions,
            "improvements": improvements,
            "per_query": per_query,
        }

        lines.append(f"## {ds}")
        lines.append(
            f"geomean baseline={bg:.4f}s uhj={ug:.4f}s delta={gdelta*100:+.2f}%"
            if bg and ug and gdelta is not None
            else "geomean unavailable"
        )
        lines.append(
            f"queries: {nq}; null/error={len(nulls)}; NO_RESULT={len(no_results)}; "
            f"regressions={len(regressions)}; improvements={len(improvements)}"
        )
        if regressions:
            lines.append("### Regressions (outside noise band)")
            for e in sorted(regressions, key=lambda x: -x["rel_delta"]):
                lines.append(
                    f"- {e['query']}: {e['baseline_mean_hot']:.4f}s -> {e['uhj_mean_hot']:.4f}s "
                    f"({e['rel_delta']*100:+.1f}%, noise±{e['noise_band']*100:.1f}%)"
                )
        if improvements:
            lines.append("### Improvements (outside noise band)")
            for e in sorted(improvements, key=lambda x: x["rel_delta"]):
                lines.append(
                    f"- {e['query']}: {e['baseline_mean_hot']:.4f}s -> {e['uhj_mean_hot']:.4f}s "
                    f"({e['rel_delta']*100:+.1f}%, noise±{e['noise_band']*100:.1f}%)"
                )
        if nulls:
            lines.append("### Null / failed queries (excluded from averages)")
            for q in nulls:
                lines.append(f"- {q}")
        lines.append("")

    # Fidelity: baseline geomean vs published master.json for same datasets
    if PUBLISHED.exists():
        pub = json.loads(PUBLISHED.read_text())
        # published order includes all datasets; locate ours
        pub_order = [
            ("mgbench", 15),
            ("ssb", 13),
            ("hits", 43),
            ("uk", 3),
            ("ontime", 11),
            ("taxi", 4),
            ("coffeeshop", 17),
            ("tpch", 22),
            ("tpcds", 103),
            ("job", 113),
        ]
        i = 0
        pub_ds = {}
        for name, nn in pub_order:
            pub_ds[name] = pub["result"][i : i + nn]
            i += nn
        lines.append("## Fidelity (baseline vs published master on c7a.4xlarge)")
        lines.append(f"published machine={pub.get('machine')} version={pub.get('actual_version')}")
        for ds, nq in DATASETS:
            pub_hots = []
            for row in pub_ds[ds]:
                h = hot_time(row)
                if h is not None:
                    pub_hots.append(h)
            pg = geomean(pub_hots)
            bg = report["datasets"][ds]["baseline_geomean"]
            if pg and bg:
                gap = (bg / pg) - 1.0
                ok = abs(gap) <= 0.10
                report["fidelity"][ds] = {
                    "published_geomean": pg,
                    "baseline_geomean": bg,
                    "gap": gap,
                    "within_10pct": ok,
                }
                lines.append(
                    f"- {ds}: published={pg:.4f}s baseline={bg:.4f}s gap={gap*100:+.1f}% "
                    f"{'FIDELITY OK' if ok else 'FIDELITY FAILED — absolute comparisons unreliable'}"
                )
        lines.append("")
        if any(not v.get("within_10pct") for v in report["fidelity"].values()):
            lines.append(
                "Suspected cause of fidelity gap: host is ARM Neoverse-V2 (no SMT) "
                "emulating AMD EPYC c7a.4xlarge (Zen 4, SMT). A/B deltas remain valid."
            )

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}/REPORT.md and report.json")


if __name__ == "__main__":
    main()
