#!/usr/bin/env python3
"""Unit 0 gates. Judges the sweep; never produces a measurement.

Kept separate from `sweep.py` on purpose: the thing that decides whether numbers
are trustworthy must not be the thing that produced them.

Every gate exits non-zero when red and prints the raw evidence that made it red,
so a reviewer can re-run one invocation and see the same thing.

    python3 gates.py g01 | g02 | g03 | g04 | g05 | g06 | g07 | all
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import harness as H

RUNS_PATH = os.path.join(H.PERF_DIR, "results", "runs.jsonl")


def load(path=RUNS_PATH):
    if not os.path.exists(path):
        sys.exit(f"RED: no results at {path}; run sweep.py first")
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def timed(recs):
    return [r for r in recs if r.get("purpose") == "timed" and "error" not in r]


def by(recs, *keys):
    d = collections.defaultdict(list)
    for r in recs:
        d[tuple(r.get(k) for k in keys)].append(r)
    return d


def hdr(name, desc):
    print(f"\n{'=' * 78}\n{name}: {desc}\n{'=' * 78}")


# --------------------------------------------------------------------------


def g01(recs):
    """The measured algorithm is the requested algorithm."""
    hdr("G0.1", "measured algorithm == requested algorithm (symbol-level proof)")
    asserts = [r for r in recs if r.get("purpose") == "assert"]
    if not asserts:
        print("RED: no assertion runs found")
        return False
    bad, unknown = [], []
    for r in asserts:
        chk = r.get("algo_check") or {}
        verdict = chk.get("verdict")
        if verdict == "UNKNOWN":
            unknown.append(r)
        elif verdict != r["algo"]:
            bad.append(r)
    print(f"assertion runs: {len(asserts)}")
    print(f"  verdict == requested : {len(asserts) - len(bad) - len(unknown)}")
    print(f"  MISMATCH             : {len(bad)}")
    print(f"  UNKNOWN (no samples) : {len(unknown)}")
    for r in bad[:15]:
        print(f"    MISMATCH {r['cell_id']} requested={r['algo']} "
              f"actual={r['algo_check'].get('verdict')} counts={r['algo_check']}")
    for r in unknown[:10]:
        print(f"    UNKNOWN  {r['cell_id']} requested={r['algo']} "
              f"samples={r['algo_check'].get('total_samples')}")
    # UNKNOWN is red too: a cell with no samples has no proof, and "no proof" is
    # not "proof of correctness".
    ok = not bad and not unknown
    print(f"\nG0.1 {'GREEN' if ok else 'RED'}")
    return ok


def g02(recs):
    """All algorithms return identical results per cell."""
    hdr("G0.2", "results agree across algorithms (full-output-column checksum)")
    chk = [r for r in recs if r.get("purpose") == "checksum" and "error" not in r]
    groups = by(chk, "cell_id")
    mismatches = []
    for cell_id, rs in sorted(groups.items()):
        outs = {r["algo"]: r.get("output", "") for r in rs}
        distinct = set(outs.values())
        if len(distinct) > 1:
            mismatches.append((cell_id, outs))
    print(f"cells with checksum runs: {len(groups)}")
    print(f"cells where algorithms disagree: {len(mismatches)}")
    for cell_id, outs in mismatches[:15]:
        print(f"    MISMATCH {cell_id}")
        for a, o in outs.items():
            print(f"       {a:14s} {o}")
    # Also cross-check the weak checksum carried by every timed run.
    weak_bad = []
    for cell_id, rs in by(timed(recs), "cell_id").items():
        outs = {r["algo"] for r in rs}
        vals = {r.get("output") for r in rs}
        if len(outs) > 1 and len(vals) > 1:
            weak_bad.append((cell_id, vals))
    print(f"cells where the per-run weak checksum disagrees: {len(weak_bad)}")
    for cell_id, vals in weak_bad[:10]:
        print(f"    WEAK-MISMATCH {cell_id} {vals}")
    ok = not mismatches and not weak_bad and bool(groups)
    print(f"\nG0.2 {'GREEN' if ok else 'RED'}")
    return ok


def g03(recs):
    """A/A calibration: same algorithm, two labels, must land inside the band."""
    hdr("G0.3", "A/A calibration -- can the instrument tell an effect from drift?")
    aa = [r for r in recs if r.get("purpose") == "aa" and "error" not in r]
    if not aa:
        print("RED: no A/A runs found")
        return False
    ok = True
    for (cell_id,), rs in sorted(by(aa, "cell_id").items()):
        labels = sorted({r["algo"] for r in rs})
        if len(labels) != 2:
            print(f"RED {cell_id}: expected 2 labels, got {labels}")
            ok = False
            continue
        a = [r for r in rs if r["algo"] == labels[0]]
        b = [r for r in rs if r["algo"] == labels[1]]
        for metric in ("wall_ms", "cpu_us"):
            av = [r[metric] for r in a]
            bv = [r[metric] for r in b]
            verdict, pct, band = H.classify(av, bv)
            good = verdict == "within_noise"
            ok &= good
            print(f"  {cell_id:34s} {metric:8s} n={len(av)}/{len(bv)} "
                  f"A={H.median(av):10.1f} B={H.median(bv):10.1f} "
                  f"delta={pct:+6.2f}% band=+-{band:.2f}% -> "
                  f"{verdict} {'OK' if good else 'FAIL'}")
    print("\nNOTE: if this is RED the correct action is to fix the instrument, "
          "never to widen the band until it fits.")
    print(f"\nG0.3 {'GREEN' if ok else 'RED'}")
    return ok


def g04(recs):
    """Known-signal recovery: can the instrument see effects already known real?"""
    hdr("G0.4", "known-signal recovery (direction of the two recorded snapshot effects)")
    print("Recorded snapshot effects, from the prior mission (LEAD-grade: produced")
    print("by a binary predating 5362055b4ed -- see WORKLOG D1). Direction is what")
    print("is scored; magnitude is reported, not asserted.\n")
    print("  (a) 16-thread build-bound INNER: unified_hash CPU excess ~ +25%")
    print("  (b) 16-thread RIGHT with non-joined: unified_hash wall excess ~ +40%\n")
    checks = [
        ("(a) 16t INNER build CPU", "INNER|u64|hi|t16|large", "build_us", +1),
        ("(b) 16t RIGHT wall", "RIGHT|u64|lo|t16|large", "wall_ms", +1),
    ]
    ok = True
    for label, cell_id, metric, expect_sign in checks:
        rs = [r for r in timed(recs) if r["cell_id"] == cell_id]
        if not rs:
            print(f"  {label}: RED -- cell {cell_id} not measured")
            ok = False
            continue
        base_algo = H.comparator_for(rs[0]["threads"])
        bv = [r[metric] for r in rs if r["algo"] == base_algo]
        uv = [r[metric] for r in rs if r["algo"] == "unified_hash"]
        if not bv or not uv:
            print(f"  {label}: RED -- missing arm ({base_algo}={len(bv)}, uhj={len(uv)})")
            ok = False
            continue
        verdict, pct, band = H.classify(bv, uv)
        got_sign = 0 if verdict == "within_noise" else (1 if pct > 0 else -1)
        good = got_sign == expect_sign
        ok &= good
        print(f"  {label:28s} cell={cell_id}")
        print(f"      {base_algo}={H.median(bv):.1f} unified={H.median(uv):.1f} "
              f"delta={pct:+.1f}% band=+-{band:.1f}% -> {verdict} "
              f"{'DIRECTION OK' if good else 'DIRECTION NOT RECOVERED'}")
    if not ok:
        print("\n  Before scoring this RED, the two candidate explanations must be")
        print("  separated: (i) the instrument lacks power, (ii) the code changed")
        print("  since the snapshot. G0.3 passing is the evidence for (ii) over (i).")
    print(f"\nG0.4 {'GREEN' if ok else 'RED'}")
    return ok


def g05(recs):
    """Phase split reconciles: exact identity, plus an independent cross-check."""
    hdr("G0.5", "phase split reconciles")
    t = timed(recs)
    bad_identity = []
    for r in t:
        lhs = r["build_us"] + r["probe_us"] + r["nonjoined_us"] + r["other_us"]
        if lhs != r["total_proc_us"]:
            bad_identity.append((r["cell_id"], r["algo"], lhs, r["total_proc_us"]))
    print(f"(i) accounting identity build+probe+nonjoined+other == total, tolerance 0")
    print(f"    timed runs checked: {len(t)}   violations: {len(bad_identity)}")
    for b in bad_identity[:10]:
        print(f"      {b}")

    print(f"\n(ii) independent cross-check: build-only query vs "
          f"FillingRightJoinSide in the full query, tolerance 20%")
    bo = [r for r in recs if r.get("purpose") == "buildonly" and "error" not in r]
    groups = by(bo, "cell_id", "algo")
    rows, bad_cross = [], []
    for (cell_id, algo), rs in sorted(groups.items()):
        full = [r["build_us"] for r in t
                if r["cell_id"] == cell_id and r["algo"] == algo]
        if not full:
            continue
        b_only = H.median([r["build_us"] for r in rs])
        b_full = H.median(full)
        if b_full <= 0:
            continue
        dev = abs(b_only - b_full) / b_full * 100.0
        rows.append((cell_id, algo, b_only, b_full, dev))
        if dev > 20.0:
            bad_cross.append((cell_id, algo, b_only, b_full, dev))
    print(f"    cross-checked (cell, algo) pairs: {len(rows)}   over tolerance: {len(bad_cross)}")
    for cell_id, algo, bo_us, bf_us, dev in rows[:12]:
        flag = "OK" if dev <= 20.0 else "OVER"
        print(f"      {cell_id:34s} {algo:14s} buildonly={bo_us:9.0f} "
              f"full={bf_us:9.0f} dev={dev:5.1f}% {flag}")

    # (iii) An origin that fails differently from both of the above:
    # ConcurrentHashJoin instruments its own build with
    # ProfileEvents['ConcurrentHashJoinBuildMicroseconds'], measured inside the
    # implementation rather than by the pipeline. Where it agrees with
    # FillingRightJoinSide, the phase source is confirmed by a mechanism that
    # shares no machinery with the processor accounting.
    print(f"\n(iii) independent instrumentation cross-check (parallel_hash only): "
          f"ConcurrentHashJoinBuildMicroseconds vs FillingRightJoinSide, tolerance 20%")
    iii_rows, iii_bad = [], []
    for (cell_id,), rs in sorted(by([r for r in t if r["algo"] == "parallel_hash"],
                                    "cell_id").items()):
        internal = [r["ch_build_us"] for r in rs if r.get("ch_build_us")]
        if not internal:
            continue
        mi, mf = H.median(internal), H.median([r["build_us"] for r in rs])
        if mf <= 0:
            continue
        dev = abs(mi - mf) / mf * 100.0
        iii_rows.append((cell_id, mi, mf, dev))
        if dev > 20.0:
            iii_bad.append((cell_id, mi, mf, dev))
    if iii_rows:
        devs = [r[3] for r in iii_rows]
        print(f"    cells checked: {len(iii_rows)}   over tolerance: {len(iii_bad)}   "
              f"median dev {H.median(devs):.1f}%  max {max(devs):.1f}%")
        for cell_id, mi, mf, dev in iii_bad[:8]:
            print(f"      OVER {cell_id:38s} internal={mi:10.0f} processor={mf:10.0f} "
                  f"dev={dev:.1f}%")
    else:
        print("    no ch_build_us recorded (older runs.jsonl); re-run sweep to populate")

    # Report, not assert: how much of the query the join phases actually are.
    shares = [(r["build_us"] + r["probe_us"] + r["nonjoined_us"]) / r["total_proc_us"]
              for r in t if r["total_proc_us"] > 0]
    if shares:
        print(f"\n    join phases as a share of all processor time: "
              f"median {H.median(shares) * 100:.1f}% "
              f"(min {min(shares) * 100:.1f}%, max {max(shares) * 100:.1f}%)")
    ok = not bad_identity and not bad_cross and bool(rows)
    print(f"\nG0.5 {'GREEN' if ok else 'RED'}")
    return ok


def g06(recs):
    """Coverage: every declared cell measured or explicitly skipped."""
    hdr("G0.6", "coverage -- every declared cell measured or SKIPPED with a reason")
    declared = {c.cell_id: c for c in H.all_cells()}
    measured = by(timed(recs), "cell_id")
    skipped = collections.defaultdict(list)
    for r in recs:
        if r.get("purpose") == "SKIPPED":
            skipped[r["cell_id"]].append(r)
    errored = {r["cell_id"] for r in recs if r.get("purpose") == "CELL_ERROR"}

    missing, partial = [], []
    for cell_id, cell in declared.items():
        want = {a for a in (H.comparator_for(cell.threads), "unified_hash")
                if cell.skip_reason(a) is None}
        got = {r["algo"] for r in measured.get((cell_id,), [])}
        if not got:
            missing.append(cell_id)
        elif not want.issubset(got):
            partial.append((cell_id, sorted(want - got)))

    print(f"declared cells      : {len(declared)}")
    print(f"cells with timings  : {len(measured)}")
    print(f"cells fully covered : {len(declared) - len(missing) - len(partial)}")
    print(f"cells MISSING       : {len(missing)}")
    print(f"cells PARTIAL       : {len(partial)}")
    print(f"cells with errors   : {len(errored)}")
    print(f"(cell,algo) SKIPPED with reason: {sum(len(v) for v in skipped.values())}")
    reasons = collections.Counter(r["skip_reason"] for v in skipped.values() for r in v)
    for reason, n in reasons.items():
        print(f"    {n:4d} x {reason}")
    for c in missing[:15]:
        print(f"    MISSING {c}")
    for c, algos in partial[:15]:
        print(f"    PARTIAL {c} missing={algos}")
    ok = not missing and not partial
    print(f"\nG0.6 {'GREEN' if ok else 'RED'}")
    return ok


def g07(recs, show_all=False):
    """The deficit map. Not pass/fail -- it is the mission's stop condition."""
    hdr("G0.7", "deficit map -- unified_hash slower / faster / within noise, per cell")
    t = timed(recs)
    rows = []
    for (cell_id,), rs in by(t, "cell_id").items():
        threads = rs[0]["threads"]
        base_algo = H.comparator_for(threads)
        base = [r for r in rs if r["algo"] == base_algo]
        uhj = [r for r in rs if r["algo"] == "unified_hash"]
        if not base or not uhj:
            continue
        entry = {"cell_id": cell_id, "threads": threads, "comparator": base_algo,
                 "n": min(len(base), len(uhj))}
        for metric in ("wall_ms", "cpu_us", "build_us", "probe_us", "nonjoined_us"):
            bv = [r[metric] for r in base]
            uv = [r[metric] for r in uhj]
            verdict, pct, band = H.classify(bv, uv)
            entry[metric] = {"base": H.median(bv), "uhj": H.median(uv),
                             "pct": pct, "band": band, "verdict": verdict}
        rows.append(entry)

    rows.sort(key=lambda e: -e["wall_ms"]["pct"])
    counts = collections.Counter()
    for e in rows:
        counts[("wall", e["wall_ms"]["verdict"])] += 1
        counts[("cpu", e["cpu_us"]["verdict"])] += 1

    print(f"cells classified: {len(rows)}\n")
    print("WALL: " + "  ".join(f"{k[1]}={v}" for k, v in sorted(counts.items()) if k[0] == "wall"))
    print("CPU : " + "  ".join(f"{k[1]}={v}" for k, v in sorted(counts.items()) if k[0] == "cpu"))

    print(f"\n{'cell':44s} {'cmp':>13s} {'wall%':>8s} {'cpu%':>8s} "
          f"{'build%':>8s} {'probe%':>8s} {'nonjn%':>8s}  verdict(wall)")
    print("-" * 118)
    shown = rows if show_all else [e for e in rows
                                   if e["wall_ms"]["verdict"] != "within_noise"
                                   or e["cpu_us"]["verdict"] != "within_noise"]
    for e in shown:
        print(f"{e['cell_id']:44s} {e['comparator']:>13s} "
              f"{e['wall_ms']['pct']:+8.1f} {e['cpu_us']['pct']:+8.1f} "
              f"{e['build_us']['pct']:+8.1f} {e['probe_us']['pct']:+8.1f} "
              f"{e['nonjoined_us']['pct']:+8.1f}  {e['wall_ms']['verdict']}")
    if not show_all:
        print(f"\n({len(rows) - len(shown)} cells within noise on both wall and CPU "
              f"omitted; --all to show)")

    out = os.path.join(H.PERF_DIR, "results", "deficit_map.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nfull map written to {out}")
    print("\nG0.7 EMITTED (stop condition: every 'slower' cell must be attributed "
          "in Unit 1 or returned UNSETTLED)")
    return True


GATES = {"g01": g01, "g02": g02, "g03": g03, "g04": g04,
         "g05": g05, "g06": g06}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=list(GATES) + ["g07", "all"])
    ap.add_argument("--runs", default=RUNS_PATH)
    ap.add_argument("--all", action="store_true", help="g07: show every cell")
    args = ap.parse_args()
    recs = load(args.runs)

    if args.gate == "g07":
        g07(recs, args.all)
        return 0
    if args.gate == "all":
        results = {name: fn(recs) for name, fn in GATES.items()}
        g07(recs, args.all)
        print(f"\n{'=' * 78}\nSUMMARY")
        for name, good in results.items():
            print(f"  {name}: {'GREEN' if good else 'RED'}")
        allgreen = all(results.values())
        print(f"UNIT 0 GATES: {'ALL GREEN' if allgreen else 'RED -- Unit 1 must not start'}")
        return 0 if allgreen else 1
    return 0 if GATES[args.gate](recs) else 1


if __name__ == "__main__":
    sys.exit(main())
