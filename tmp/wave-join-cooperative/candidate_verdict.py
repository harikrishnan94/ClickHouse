#!/usr/bin/env python3
"""Gate 10 — preregistered performance verdict (FROZEN before implementation).

Compares the candidate A/B suite run (candidate arm A, in-run baseline arm B)
against the FROZEN Unit-0 baseline (tmp/wave-join-impl/baseline_u0.jsonl) using
the frozen per-cell medians and noise bands from tmp/wave-join-impl/PREREG.md.

Verdict rules (all preregistered; every failure or null is a FAILURE):
  1. Candidate file integrity: footer status=complete and binary_stable=true;
     every integrity/assert/fingerprint row ok; arm B sha256 equals the frozen
     baseline sha in every run row; every run row engaged with the exact
     expected leaf count (A: 16384, C: 32768 — probe-only change, build plan
     unchanged; preregistered in WORKLOG.md).
  2. Pair validity: for every cell, the expected number of position-balanced
     pairs (9 at T96/T64, 5 elsewhere), no missing arm, no duplicate
     (pair, arm) row.
  3. Baseline identity: medians recomputed from baseline_u0.jsonl must equal
     the frozen PREREG medians (guards against a wrong/edited file).
  4. GOAL (per cell, all 10): candidate median must BEAT the frozen baseline
     median by MORE than the cell's frozen band:
         cand_median < base_median * (1 - band)
  5. FLOOR (per cell): candidate median must not exceed the frozen baseline
     median by more than the band (reported separately; implied by GOAL).
  6. SCALING (per shape A and C): the T16->T96 scaling ratio
     S = median(T16)/median(T96) must improve beyond the combined
     preregistered noise of the two cells:
         S_cand > S_base * (1 + band_T16 + band_T96)
  7. Environment drift: the in-run baseline arm B median deviating from the
     frozen baseline median by more than 2x the cell band is reported as
     ENV-DRIFT and fails the verdict (result is UNSETTLED, not a pass).

Exit 0 only if every check passes.  Anything missing, null, or unexpected is
reported and fails.
"""

import argparse
import json
import math
import sys

FROZEN_SHA = "4b55481c22d025ae364d36df39cd662bd986fd5878e711d89e1d76b08ea59cce"

# Frozen 10-cell table from tmp/wave-join-impl/PREREG.md (medians in seconds,
# bands as fractions).  Any edit here is a register amendment requiring user
# sign-off.
FROZEN = {
    "A_T96": (1.343, 0.0137, 9),
    "A_T64": (1.368, 0.0183, 9),
    "A_T32": (2.179, 0.0218, 5),
    "A_T16": (3.289, 0.0107, 5),
    "A_T1": (61.678, 0.0108, 5),
    "C_T96": (7.835, 0.0117, 9),
    "C_T64": (7.939, 0.0100, 9),
    "C_T32": (9.268, 0.0100, 5),
    "C_T16": (14.195, 0.0350, 5),
    "C_T1": (254.125, 0.0159, 5),
}
CELLS = list(FROZEN.keys())
EXPECTED_LEAVES = {"A": 16384, "C": 32768}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def load(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"FAIL: {path}:{i}: invalid JSON ({e})")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    failures = []
    warnings = []

    def fail(msg):
        failures.append(msg)

    base_rows = load(args.baseline)
    cand_rows = load(args.candidate)

    # --- 1. candidate file integrity -------------------------------------
    footers = [r for r in cand_rows if r.get("kind") == "footer"]
    if len(footers) != 1:
        fail(f"candidate: expected exactly 1 footer row, found {len(footers)}")
    else:
        ftr = footers[0]
        if ftr.get("status") != "complete":
            fail(f"candidate: footer status={ftr.get('status')!r} != 'complete'")
        if ftr.get("binary_stable") is not True:
            fail("candidate: footer binary_stable is not true")

    for r in cand_rows:
        k = r.get("kind")
        if k in ("integrity", "assert", "fingerprint") and r.get("ok") is not True:
            fail(f"candidate: {k} row not ok: {json.dumps(r)[:200]}")

    cand_runs = [r for r in cand_rows if r.get("kind") == "run"]
    if not cand_runs:
        fail("candidate: no run rows at all")
    for r in cand_runs:
        cell, arm = r.get("cell"), r.get("arm")
        if cell not in FROZEN or arm not in ("A", "B"):
            fail(f"candidate: unexpected run row cell={cell} arm={arm}")
            continue
        if arm == "B" and r.get("binary_sha256") != FROZEN_SHA:
            fail(f"candidate: {cell} pair {r.get('pair')} arm B binary sha "
                 f"{r.get('binary_sha256')} != frozen baseline sha")
        shape = r.get("shape")
        leaves = r.get("RadixHashJoinLeafGroupBuilds")
        if leaves != EXPECTED_LEAVES.get(shape):
            fail(f"candidate: {cell} pair {r.get('pair')} arm {arm} leaf count "
                 f"{leaves} != expected {EXPECTED_LEAVES.get(shape)} (engagement)")
        if not isinstance(r.get("wall_us"), (int, float)) or r["wall_us"] <= 0:
            fail(f"candidate: {cell} pair {r.get('pair')} arm {arm}: invalid wall_us "
                 f"{r.get('wall_us')!r}")

    # --- 2. pair validity --------------------------------------------------
    for cell in CELLS:
        exp_pairs = FROZEN[cell][2]
        rows = [r for r in cand_runs if r.get("cell") == cell]
        seen = {}
        for r in rows:
            key = (r.get("pair"), r.get("arm"))
            if key in seen:
                fail(f"candidate: {cell}: duplicate row for pair/arm {key}")
            seen[key] = r
        for p in range(exp_pairs):
            for arm in ("A", "B"):
                if (p, arm) not in seen:
                    fail(f"candidate: {cell}: missing pair {p} arm {arm} "
                         f"(expected {exp_pairs} pairs)")

    # --- 3. baseline identity -----------------------------------------------
    base_runs = [r for r in base_rows if r.get("kind") == "run"]
    base_median = {}
    for cell in CELLS:
        walls = [r["wall_us"] / 1e6 for r in base_runs if r.get("cell") == cell]
        m = median(walls)
        base_median[cell] = m
        if m is None:
            fail(f"baseline: no run rows for {cell}")
            continue
        frozen_m = FROZEN[cell][0]
        if abs(m - frozen_m) / frozen_m > 0.001:
            fail(f"baseline: recomputed median for {cell} = {m:.3f}s does not "
                 f"match frozen PREREG median {frozen_m:.3f}s — wrong file?")

    # --- 4/5/7. per-cell verdicts -------------------------------------------
    cand_median = {}
    inrun_b_median = {}
    print(f"{'cell':7} {'base(s)':>9} {'cand(s)':>9} {'delta':>8} {'band':>6} "
          f"{'goal':>5} {'floor':>5} {'inrunB(s)':>9} {'drift':>6}")
    for cell in CELLS:
        frozen_m, band, _ = FROZEN[cell]
        a = median([r["wall_us"] / 1e6 for r in cand_runs
                    if r.get("cell") == cell and r.get("arm") == "A"])
        b = median([r["wall_us"] / 1e6 for r in cand_runs
                    if r.get("cell") == cell and r.get("arm") == "B"])
        cand_median[cell], inrun_b_median[cell] = a, b
        if a is None or b is None:
            fail(f"{cell}: missing candidate ({a}) or in-run baseline ({b}) median")
            print(f"{cell:7} {frozen_m:9.3f} {'NULL':>9}")
            continue
        delta = (frozen_m - a) / frozen_m
        goal = a < frozen_m * (1 - band)
        floor = a <= frozen_m * (1 + band)
        drift = abs(b - frozen_m) / frozen_m
        drift_bad = drift > 2 * band
        print(f"{cell:7} {frozen_m:9.3f} {a:9.3f} {delta:+7.2%} {band:6.2%} "
              f"{'PASS' if goal else 'FAIL':>5} {'ok' if floor else 'FAIL':>5} "
              f"{b:9.3f} {drift:6.2%}")
        if not goal:
            fail(f"{cell}: GOAL failed — candidate median {a:.3f}s does not beat "
                 f"frozen {frozen_m:.3f}s by more than the band {band:.2%} "
                 f"(needed < {frozen_m * (1 - band):.3f}s)")
        if not floor:
            fail(f"{cell}: FLOOR failed — candidate median {a:.3f}s exceeds "
                 f"frozen {frozen_m:.3f}s by more than the band")
        if drift_bad:
            fail(f"{cell}: ENV-DRIFT — in-run baseline arm median {b:.3f}s deviates "
                 f"{drift:.2%} from frozen {frozen_m:.3f}s (> 2x band {band:.2%}); "
                 f"result UNSETTLED")

    # --- 6. scaling ----------------------------------------------------------
    for shape in ("A", "C"):
        c16, c96 = f"{shape}_T16", f"{shape}_T96"
        if cand_median.get(c16) is None or cand_median.get(c96) is None:
            fail(f"scaling {shape}: missing candidate medians")
            continue
        s_base = FROZEN[c16][0] / FROZEN[c96][0]
        s_cand = cand_median[c16] / cand_median[c96]
        combined = FROZEN[c16][1] + FROZEN[c96][1]
        ok = s_cand > s_base * (1 + combined)
        print(f"scaling {shape}: T16/T96 base={s_base:.3f} cand={s_cand:.3f} "
              f"combined_noise={combined:.2%} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            fail(f"scaling {shape}: candidate T16->T96 scaling {s_cand:.3f} does not "
                 f"improve on baseline {s_base:.3f} beyond combined noise {combined:.2%}")

    print()
    if failures:
        print(f"VERDICT: FAIL ({len(failures)} failure(s))")
        for f_ in failures:
            print(f"  - {f_}")
        sys.exit(1)
    print("VERDICT: PASS — every cell beats its frozen baseline median by more than "
          "its frozen band, no cell is slower, and both shapes improve T16->T96 "
          "scaling beyond the combined preregistered noise.")
    sys.exit(0)


if __name__ == "__main__":
    main()
