#!/usr/bin/env python3
"""Ablation A1 — the separate non-joined scan. Pre-registered in PREREG P1.3.

Four arms per cell, all interleaved, 7 reps:

    hash          / parallel_non_joined_rows_processing = 1   (baseline, as measured in Unit 0)
    hash          / parallel_non_joined_rows_processing = 0   (control: must not move)
    unified_hash  / parallel_non_joined_rows_processing = 1   (as measured in Unit 0)
    unified_hash  / parallel_non_joined_rows_processing = 0   (ABLATED: inline path)

The baseline never overrides `IJoin::supportParallelNonJoinedBlocksProcessing`
(`IJoin.h:158` returns false), so its two arms are the same pipeline and give a
free control on whether the setting has any side effect of its own.

Exits non-zero unless the structural check A1-d holds: with the setting off,
`NonJoinedBlocksTransform` must be GONE from the executed pipeline. Without that,
a null result is VOID rather than evidence that the operation is cheap.

    python3 ablate_a1.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time

import harness as H

SAFE = re.compile(r"[^A-Za-z0-9_.-]")
_seq = itertools.count()
_TOKEN = f"{os.getpid():x}{int(time.time()) & 0xffff:04x}"

# 1-thread cells: the deficit A1 is meant to explain.
# 16/64-thread cells: the reverse-direction prediction A1-c.
CELLS = [
    H.Cell("FULL", "u64", "hi", 1, "medium"),
    H.Cell("RIGHT", "u64", "hi", 1, "medium"),
    H.Cell("RIGHT", "u64", "hi", 1, "small"),
    H.Cell("FULL", "u64", "hi", 1, "small"),
    H.Cell("RIGHT", "u64", "lo", 16, "large"),
    H.Cell("RIGHT", "u64", "lo", 64, "large"),
]


def qid(*parts):
    return SAFE.sub("_", "-".join(str(p) for p in parts)) + f".{_TOKEN}.{next(_seq)}"


def arms_for(cell):
    base = H.comparator_for(cell.threads)
    return [(base, 1), (base, 0), ("unified_hash", 1), ("unified_hash", 0)]


def run_arm(cell, algo, pnj, rep, tag):
    q = qid(tag, cell.cell_id, algo, f"pnj{pnj}", rep)
    settings = H.settings_for(cell, algo, {"parallel_non_joined_rows_processing": pnj})
    H.run_query(H.join_sql(cell, "timed"), settings, query_id=q)
    return q


def transform_present(query_id, name="NonJoinedBlocksTransform"):
    n = H.run_query(
        f"SELECT count() FROM system.processors_profile_log "
        f"WHERE query_id = '{query_id}' AND name = '{name}' FORMAT TSV").strip()
    return int(n or 0) > 0


def joining_output_rows(query_id):
    n = H.run_query(
        f"SELECT sum(output_rows) FROM system.processors_profile_log "
        f"WHERE query_id = '{query_id}' AND name = 'JoiningTransform' FORMAT TSV").strip()
    return int(n or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(H.PERF_DIR, "results", "ablate_a1.jsonl"))
    args = ap.parse_args()
    tag = "a1"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = {}
    struct = {}

    with open(args.out, "w") as fh:
        for cell in CELLS:
            arms = arms_for(cell)
            pending = []
            for algo, pnj in arms:                       # warm-up, discarded
                run_arm(cell, algo, pnj, -1, tag)
            for rep in range(args.reps):
                order = arms[rep % len(arms):] + arms[:rep % len(arms)]   # rotate
                for algo, pnj in order:
                    pending.append((run_arm(cell, algo, pnj, rep, tag), algo, pnj, rep))
            H.flush_logs()
            for q, algo, pnj, rep in pending:
                rec = {"cell_id": cell.cell_id, "threads": cell.threads,
                       "algo": algo, "pnj": pnj, "rep": rep, "query_id": q}
                try:
                    rec.update(H.read_run(q))
                except H.QueryError as exc:
                    rec["error"] = str(exc)[:300]
                fh.write(json.dumps(rec) + "\n")
                results.setdefault((cell.cell_id, algo, pnj), []).append(rec)
            # A1-d structural check, on one representative query per arm
            for algo, pnj in arms:
                q = [p[0] for p in pending if p[1] == algo and p[2] == pnj][0]
                struct[(cell.cell_id, algo, pnj)] = (transform_present(q),
                                                     joining_output_rows(q))
            print(f"done {cell.cell_id}", flush=True)

    # ---- A1-d: did the ablation take effect? ----
    print(f"\n{'=' * 100}\nA1-d STRUCTURAL CHECK — NonJoinedBlocksTransform present? "
          f"(must be False when pnj=0 for unified_hash)\n{'=' * 100}")
    print(f"{'cell':34s} {'algo':14s} {'pnj':>3}  {'NonJoined?':>10}  {'JoiningTransform out_rows':>26}")
    d_ok = True
    for (cid, algo, pnj), (present, jrows) in sorted(struct.items()):
        print(f"{cid:34s} {algo:14s} {pnj:>3}  {str(present):>10}  {jrows:>26,}")
        if algo == "unified_hash" and pnj == 0 and present:
            d_ok = False
    print(f"\nA1-d: {'PASS - ablation took effect' if d_ok else 'FAIL - ablation did NOT take effect; any null is VOID'}")

    # ---- A1-a / A1-b / A1-c ----
    print(f"\n{'=' * 100}\nA1-a/b/c — deltas vs the baseline arm in the same cell\n{'=' * 100}")
    print(f"{'cell':34s} {'metric':8s} {'base(pnj=1)':>12} {'uhj pnj=1':>11} {'uhj pnj=0':>11} "
          f"{'gap before':>11} {'gap after':>10} {'band':>7}  verdict")
    for cell in CELLS:
        cid = cell.cell_id
        base_algo = H.comparator_for(cell.threads)
        for metric in ("wall_ms", "cpu_us"):
            def vals(a, p):
                return [r[metric] for r in results.get((cid, a, p), []) if metric in r]
            b1, u1, u0 = vals(base_algo, 1), vals("unified_hash", 1), vals("unified_hash", 0)
            b0 = vals(base_algo, 0)
            if not (b1 and u1 and u0):
                continue
            _, before, band = H.classify(b1, u1)
            verdict_after, after, _ = H.classify(b1, u0)
            ctrl_verdict, ctrl, _ = H.classify(b1, b0)
            note = ""
            if ctrl_verdict != "within_noise":
                note = f"  !! A1-b CONTROL MOVED {ctrl:+.1f}% -> confounded"
            print(f"{cid:34s} {metric:8s} {H.median(b1):12.1f} {H.median(u1):11.1f} "
                  f"{H.median(u0):11.1f} {before:+10.1f}% {after:+9.1f}% {band:6.1f}%  "
                  f"{verdict_after}{note}")

    print(f"\nwrote {args.out}")
    return 0 if d_ok else 1


if __name__ == "__main__":
    sys.exit(main())
