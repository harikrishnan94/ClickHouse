#!/usr/bin/env python3
"""Judge one run of `sweep.py`.

After Stage 1 there is no within-binary fused-vs-split A/B. Correctness is
`diff_goldens.py`. This report still asserts the probe family from stacks (A1) and
keeps A2–A5 structure so Stage 6's cross-binary `uhj_pre` / `uhj_post` arms plug in
without rewriting the gates.

Usage:  python3 ab_report.py --tag ab2 [--metric wall_ms] [--test-arm uhj_ship]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import harness as H

RUNS = os.path.join(H.PERF_DIR, "results", "runs.jsonl")


def hdr(name, text):
    print(f"\n=== {name}: {text} " + "=" * max(0, 74 - len(name) - len(text)))


def load(path, tag):
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("run_tag") == tag:
                rows.append(r)
    return rows


def by(rows, *keys):
    out = collections.defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


# --------------------------------------------------------------------------


def a1_arm_assertion(rows):
    hdr("A1", "the arm that ran is the arm that was asked for (symbol-level proof)")
    asserts = [r for r in rows if r.get("purpose") == "assert" and r.get("algo") in H.ARMS]
    bad, unknown = [], []
    for r in asserts:
        chk = r.get("algo_check") or {}
        # Re-judged from the raw counts rather than read off the record, so a record written
        # before `judge_counts` gained its sample floor is held to the current rule.
        chk = {**chk, **H.judge_counts(chk)} if "total_samples" in chk else chk
        # The stacks prove the probe FAMILY, not the batch length -- every split arm runs
        # the same `probeTwoPhase`. `verify_arm.py --scan` is what covers the length.
        # SEMI/ANTI are forced fused regardless of the arm, so the kind decides too.
        expect = H.expected_probe(r["algo"], r["kind"])
        if chk.get("verdict") != "unified_hash":
            # A query too fast to collect CPU samples proves nothing either way (the tiny
            # special cells finish in ~1 ms); it lands in the unknown bucket, not wrong-arm.
            # With plenty of samples but no UHJ marker, something genuinely else ran.
            if chk.get("verdict") == "UNKNOWN" and chk.get("total_samples", 0) < 20:
                unknown.append((r["cell_id"], r["algo"], chk.get("total_samples")))
            else:
                bad.append((r["cell_id"], r["algo"], f"algorithm={chk.get('verdict')}"))
        elif chk.get("probe_verdict") == "UNKNOWN":
            unknown.append((r["cell_id"], r["algo"], chk.get("total_samples")))
        elif chk.get("probe_verdict") != expect:
            bad.append((r["cell_id"], r["algo"], f"probe={chk.get('probe_verdict')}"))
    # A batch length that never left the harness would make two arms the same measurement,
    # so check the length the record claims against the length the arm is defined with.
    # Batch length is constexpr; records still carry it for Stage 6 continuity.
    mislabelled = [r for r in rows if r.get("algo") in H.ARMS and "probe_batch_rows" in r
                   and r["probe_batch_rows"] != H.batch_of(r["algo"])]
    print(f"assertion runs: {len(asserts)}   wrong arm: {len(bad)}   "
          f"no probe symbol in any stack: {len(unknown)}   "
          f"mislabelled batch length: {len(mislabelled)}")
    if mislabelled:
        bad.append((mislabelled[0]["cell_id"], mislabelled[0]["algo"], "batch length"))
    for cell_id, arm, what in bad[:10]:
        print(f"    WRONG  {cell_id:44s} asked={arm} got {what}")
    for cell_id, arm, n in unknown[:10]:
        print(f"    UNKNOWN {cell_id:44s} asked={arm} cpu_samples={n}")
    # A cell too short to collect a stack inside the probe is not evidence of the wrong
    # arm, but it is not evidence of the right one either, so it is reported apart.
    return "RED" if bad else ("AMBER" if unknown else "GREEN")


def a2_answers(rows):
    hdr("A2", "arms return the same answer (n/a with a single shipping arm)")
    if len(H.ARMS) < 2:
        print("single shipping arm — cross-arm answer check deferred to Stage 6 / goldens")
        return "GREEN"
    mismatch = []
    for purpose in ("checksum", "timed"):
        groups = by([r for r in rows if r.get("purpose") == purpose and r.get("algo") in H.ARMS],
                    "cell_id")
        for (cell_id,), rs in sorted(groups.items()):
            outs = {r["algo"]: {x.get("output") for x in rs if x["algo"] == r["algo"]}
                    for r in rs}
            flat = set()
            for v in outs.values():
                flat |= v
            if len(flat) > 1:
                mismatch.append((purpose, cell_id, outs))
    print(f"cells where the arms disagree: {len(mismatch)}")
    for purpose, cell_id, outs in mismatch[:10]:
        print(f"    MISMATCH [{purpose}] {cell_id}: {outs}")
    return "RED" if mismatch else "GREEN"


def a3_aa(rows, metric):
    hdr("A3", f"A/A calibration on {metric}: the same arm under two labels")
    aa = [r for r in rows if r.get("purpose") == "aa" and metric in r]
    verdicts = []
    for (cell_id,), rs in sorted(by(aa, "cell_id").items()):
        labels = sorted({r["algo"] for r in rs})
        if len(labels) != 2:
            print(f"  {cell_id}: RED -- {len(labels)} labels")
            verdicts.append("RED")
            continue
        a = [r[metric] for r in rs if r["algo"] == labels[0]]
        b = [r[metric] for r in rs if r["algo"] == labels[1]]
        verdict, pct, band = H.classify(a, b)
        ok = "GREEN" if verdict == "within_noise" else "RED"
        verdicts.append(ok)
        print(f"  {cell_id:34s} {pct:+6.1f}%  band +-{band:.1f}%  {verdict}  [{ok}]")
    return "RED" if "RED" in verdicts else ("GREEN" if verdicts else "AMBER")


def a4_ab(rows, metric, test_arm=None):
    test_arm = test_arm or H.TEST_ARM
    hdr("A4", f"{test_arm} against {H.BASELINE_ARM} on {metric} "
              f"(positive = {test_arm} is slower)")
    timed = [r for r in rows if r.get("purpose") == "timed" and metric in r
             and r.get("algo") in H.ARMS]
    per_cell = {}
    for (cell_id,), rs in by(timed, "cell_id").items():
        base = [r[metric] for r in rs if r["algo"] == H.BASELINE_ARM]
        test = [r[metric] for r in rs if r["algo"] == test_arm]
        if not base or not test:
            continue
        verdict, pct, band = H.classify(base, test)
        meta = rs[0]
        per_cell[cell_id] = {
            "verdict": verdict, "pct": pct, "band": band,
            "kind": meta["kind"], "key": meta["key"], "match": meta["match"],
            "threads": meta["threads"], "card": meta["card"],
            "base": H.median(base), "test": H.median(test),
        }

    n = len(per_cell)
    if not n:
        print("no comparable cells")
        return "AMBER", per_cell
    # Stage 6 budget: absolute per-cell median within 2% of pre (positive = post slower).
    budget = 2.0
    over = [c for c, d in per_cell.items() if d["pct"] > budget]
    under = [c for c, d in per_cell.items() if d["pct"] < -budget]
    within = [c for c, d in per_cell.items() if abs(d["pct"]) <= budget]
    slower = [c for c, d in per_cell.items() if d["verdict"] == "slower"]
    faster = [c for c, d in per_cell.items() if d["verdict"] == "faster"]
    pcts = sorted(d["pct"] for d in per_cell.values())
    print(f"cells: {n}   slower(noise-band): {len(slower)}   faster(noise-band): {len(faster)}   "
          f"within noise: {n - len(slower) - len(faster)}")
    print(f"Stage-6 2% budget: within={len(within)}  >2% slower={len(over)}  >2% faster={len(under)}")
    print(f"median delta: {H.median(pcts):+.1f}%   worst: {pcts[-1]:+.1f}%   "
          f"best: {pcts[0]:+.1f}%")

    def group(name, keyfn):
        print(f"\n  by {name}:")
        buckets = collections.defaultdict(list)
        for d in per_cell.values():
            buckets[keyfn(d)].append(d["pct"])
        for k in sorted(buckets, key=lambda x: str(x)):
            v = sorted(buckets[k])
            print(f"    {str(k):16s} n={len(v):3d}  median {H.median(v):+6.1f}%  "
                  f"worst {v[-1]:+6.1f}%  best {v[0]:+6.1f}%")

    group("join kind", lambda d: d["kind"])
    group("key type", lambda d: d["key"])
    group("match rate", lambda d: d["match"])
    group("threads x card", lambda d: f"t{d['threads']}/{d['card']}")

    print(f"\n  worst 15 cells for {test_arm}:")
    for c, d in sorted(per_cell.items(), key=lambda kv: -kv[1]["pct"])[:15]:
        flag = " OVER2%" if d["pct"] > budget else ""
        print(f"    {d['pct']:+7.1f}%  band +-{d['band']:4.1f}%  {c:44s} "
              f"{d['base']:9.1f} -> {d['test']:9.1f}  {d['verdict']}{flag}")
    print(f"\n  best 10 cells for {test_arm}:")
    for c, d in sorted(per_cell.items(), key=lambda kv: kv[1]["pct"])[:10]:
        print(f"    {d['pct']:+7.1f}%  band +-{d['band']:4.1f}%  {c:44s} "
              f"{d['base']:9.1f} -> {d['test']:9.1f}  {d['verdict']}")
    # Budget gate: systematic >2% regression fails; a few noisy cells are summarised.
    if H.median(pcts) > budget or len(over) > max(3, n // 10):
        return "RED", per_cell
    if over:
        return "AMBER", per_cell
    return "GREEN", per_cell


def a5_batch_curve(rows, metric):
    """The question the scan exists to answer: is the split's cost a function of the batch?

    Reported per key type, because the 8192 run showed the split's sign is set by how
    expensive one lookup is, and a batch effect that only moves the cheap keys is a
    different conclusion from one that moves all of them.
    """
    hdr("A5", f"batch-length curve on {metric}, per cell median vs {H.BASELINE_ARM}")
    per_arm = {}
    for arm in H.TEST_ARMS:
        _, per_cell = a4_ab_quiet(rows, metric, arm)
        per_arm[arm] = per_cell
    cells = set.intersection(*(set(p) for p in per_arm.values())) if per_arm else set()
    if not cells:
        print("no cells measured on every arm")
        return

    def line(label, sel):
        picked = [c for c in cells if sel(per_arm[H.TEST_ARMS[0]][c])]
        if not picked:
            return
        out = []
        for arm in H.TEST_ARMS:
            out.append(H.median([per_arm[arm][c]["pct"] for c in picked]))
        print(f"  {label:22s} n={len(picked):3d}  " +
              "  ".join(f"b{H.batch_of(a)}={v:+6.1f}%" for a, v in zip(H.TEST_ARMS, out)))

    line("ALL", lambda d: True)
    print()
    for key in sorted({d["key"] for p in per_arm.values() for d in p.values()}):
        line(f"key={key}", lambda d, k=key: d["key"] == k)
    print()
    for tc in sorted({f"t{d['threads']}/{d['card']}"
                      for p in per_arm.values() for d in p.values()}):
        line(tc, lambda d, t=tc: f"t{d['threads']}/{d['card']}" == t)


def a4_ab_quiet(rows, metric, test_arm):
    """`a4_ab` without the printing, for the curve."""
    timed = [r for r in rows if r.get("purpose") == "timed" and metric in r
             and r.get("algo") in H.ARMS]
    per_cell = {}
    for (cell_id,), rs in by(timed, "cell_id").items():
        base = [r[metric] for r in rs if r["algo"] == H.BASELINE_ARM]
        test = [r[metric] for r in rs if r["algo"] == test_arm]
        if not base or not test:
            continue
        verdict, pct, band = H.classify(base, test)
        meta = rs[0]
        per_cell[cell_id] = {"verdict": verdict, "pct": pct, "band": band,
                             "kind": meta["kind"], "key": meta["key"],
                             "match": meta["match"], "threads": meta["threads"],
                             "card": meta["card"]}
    return "GREEN", per_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--path", default=RUNS)
    ap.add_argument("--metric", default="wall_ms")
    ap.add_argument("--test-arm", default="", help="report one arm only (default: all)")
    args = ap.parse_args()

    rows = load(args.path, args.tag)
    test_arms = [args.test_arm] if args.test_arm else list(H.TEST_ARMS)
    print(f"tag={args.tag}  records={len(rows)}  "
          f"baseline={H.BASELINE_ARM}  test={', '.join(test_arms) or '(none — Stage 6)'}")
    if not rows:
        print("nothing to report")
        return 1

    v = {
        "A1": a1_arm_assertion(rows),
        "A2": a2_answers(rows),
        "A3": a3_aa(rows, args.metric),
    }
    for arm in test_arms:
        v["A4"], per_cell = a4_ab(rows, args.metric, arm)
        if args.metric != "cpu_us":
            a4_ab(rows, "cpu_us", arm)
    if len(test_arms) > 1:
        a5_batch_curve(rows, args.metric)
        if args.metric != "cpu_us":
            a5_batch_curve(rows, "cpu_us")
    elif not test_arms:
        print("\nA4/A5 skipped: no TEST_ARMS until Stage 6 cross-binary A/B")

    hdr("SUMMARY", "")
    for k in sorted(v):
        print(f"  {k}: {v[k]}")
    return 0 if "RED" not in v.values() else 1


if __name__ == "__main__":
    sys.exit(main())
