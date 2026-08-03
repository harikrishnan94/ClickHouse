#!/usr/bin/env python3
"""Run the Unit 0 sweep and write raw per-run records.

Measures, never judges: every classification lives in `gates.py`, so the thing
that produces the numbers is not the thing that decides whether they are good.

Per cell, in order:
  1. one warm-up run per algorithm, discarded
  2. one ASSERTION run per algorithm with the CPU profiler on (Gate G0.1),
     kept out of the timed set so profiling cannot perturb a measurement
  3. one CHECKSUM run per algorithm (Gate G0.2), likewise untimed
  4. REPS interleaved timed runs, algorithm order rotated each repetition

Interleaving with rotation is the point: running all of A then all of B lets
machine drift masquerade as an effect, and always running A first lets any
first-position penalty do the same.
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

RESULTS_DIR = os.path.join(H.PERF_DIR, "results")
RUNS_PATH = os.path.join(RESULTS_DIR, "runs.jsonl")

SAFE = re.compile(r"[^A-Za-z0-9_.-]")

_seq = itertools.count()


def qid(*parts) -> str:
    """Query ids must be globally unique: the server rejects a duplicate that is
    still running, and a reused id would also make the log readback ambiguous."""
    return SAFE.sub("_", "-".join(str(p) for p in parts)) + f".{next(_seq)}"


def algos_for(cell: H.Cell, with_context: bool) -> list[str]:
    """Which algorithms to measure in this cell.

    Always `unified_hash` and its comparator. The third algorithm is context the
    mission welcomes but does not require, so it is opt-in.
    """
    out = [H.comparator_for(cell.threads), "unified_hash"]
    if with_context:
        third = [a for a in H.ALGOS if a not in out]
        out += third
    return [a for a in out if cell.skip_reason(a) is None]


def one_run(cell, algo, purpose, rep, run_tag, extra_settings=None):
    """Execute one query and return its query_id; readback happens after a flush."""
    mode = {"timed": "timed", "warmup": "timed", "assert": "timed",
            "checksum": "checksum", "buildonly": "buildonly"}[purpose]
    sql = H.join_sql(cell, mode)
    settings = H.settings_for(cell, algo, extra_settings)
    if purpose == "assert":
        # CPU-time profiler: samples only while the query burns CPU, so the stacks
        # are the join's own, not idle pipeline threads'.
        settings["query_profiler_cpu_time_period_ns"] = 1_000_000
        settings["query_profiler_real_time_period_ns"] = 0
    q = qid(run_tag, cell.cell_id, algo, purpose, rep)
    out = H.run_query(sql, settings, query_id=q)
    return q, out.strip()


def sweep_cell(cell, reps, run_tag, with_context, do_buildonly, fh):
    algos = algos_for(cell, with_context)
    pending = []   # (query_id, algo, purpose, rep, output)

    for algo in algos:
        one_run(cell, algo, "warmup", 0, run_tag)          # discarded
        pending.append((*one_run(cell, algo, "assert", 0, run_tag), algo, "assert", 0))
        pending.append((*one_run(cell, algo, "checksum", 0, run_tag), algo, "checksum", 0))
        if do_buildonly:
            pending.append((*one_run(cell, algo, "buildonly", 0, run_tag), algo, "buildonly", 0))

    for rep in range(reps):
        order = algos[rep % len(algos):] + algos[:rep % len(algos)]   # rotate
        for algo in order:
            pending.append((*one_run(cell, algo, "timed", rep, run_tag), algo, "timed", rep))

    H.flush_logs()

    for q, out, algo, purpose, rep in pending:
        rec = {
            "run_tag": run_tag, "cell_id": cell.cell_id, "kind": cell.kind,
            "key": cell.key, "match": cell.match, "threads": cell.threads,
            "card": cell.card, "algo": algo, "purpose": purpose, "rep": rep,
            "query_id": q, "output": out,
        }
        try:
            rec.update(H.read_run(q))
        except H.QueryError as exc:
            rec["error"] = str(exc)[:500]
        if purpose == "assert":
            try:
                rec["algo_check"] = H.assert_algorithm(q)
            except H.QueryError as exc:
                rec["algo_check"] = {"verdict": "ERROR", "error": str(exc)[:300]}
        fh.write(json.dumps(rec) + "\n")
    fh.flush()

    # Record the skips explicitly, so Gate G0.6 sees a reason rather than a hole.
    for algo in H.ALGOS:
        reason = cell.skip_reason(algo)
        if reason:
            fh.write(json.dumps({
                "run_tag": run_tag, "cell_id": cell.cell_id, "kind": cell.kind,
                "key": cell.key, "match": cell.match, "threads": cell.threads,
                "card": cell.card, "algo": algo, "purpose": "SKIPPED",
                "skip_reason": reason,
            }) + "\n")
    fh.flush()


def aa_cells():
    """Gate G0.3 calibration cells: at least one 1-thread and one 64-thread."""
    return [
        H.Cell("INNER", "u64", "hi", 1, "medium"),
        H.Cell("INNER", "u64", "hi", 64, "large"),
    ]


def sweep_aa(cell, reps, run_tag, fh):
    """Run the SAME algorithm under two labels, interleaved exactly like a real A/B.

    If this reports a significant delta, the instrument is measuring drift and
    every A/B it produced is void -- which is the only reason to trust the rest.
    """
    labels = ["unified_hash#A", "unified_hash#B"]
    pending = []
    for lab in labels:
        one_run(cell, "unified_hash", "warmup", 0, run_tag)
    for rep in range(reps):
        order = labels[rep % 2:] + labels[:rep % 2]
        for lab in order:
            q = qid(run_tag, "AA", cell.cell_id, lab, rep)
            sql = H.join_sql(cell, "timed")
            H.run_query(sql, H.settings_for(cell, "unified_hash"), query_id=q)
            pending.append((q, lab, rep))
    H.flush_logs()
    for q, lab, rep in pending:
        rec = {"run_tag": run_tag, "cell_id": cell.cell_id, "threads": cell.threads,
               "card": cell.card, "algo": lab, "purpose": "aa", "rep": rep, "query_id": q}
        try:
            rec.update(H.read_run(q))
        except H.QueryError as exc:
            rec["error"] = str(exc)[:500]
        fh.write(json.dumps(rec) + "\n")
    fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--run-tag", default=time.strftime("s%Y%m%d-%H%M%S"))
    ap.add_argument("--with-context-algo", action="store_true",
                    help="also measure the third algorithm where it runs")
    ap.add_argument("--buildonly-subset", type=int, default=8,
                    help="how many cells also get a build-only run (G0.5 cross-check)")
    ap.add_argument("--filter", default="", help="substring filter on cell_id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=RUNS_PATH)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cells = [c for c in H.all_cells() if args.filter in c.cell_id]
    if args.limit:
        cells = cells[:args.limit]

    # Build-only cross-check subset: spread across thread counts and join kinds
    # rather than taken from the front of the list, so it validates the phase
    # source in the regimes that actually differ.
    step = max(1, len(cells) // max(1, args.buildonly_subset))
    buildonly_ids = {cells[i].cell_id for i in range(0, len(cells), step)}

    print(f"run_tag={args.run_tag} cells={len(cells)} reps={args.reps} "
          f"buildonly_cells={len(buildonly_ids)}", flush=True)

    t_start = time.time()
    with open(args.out, "a") as fh:
        for cell in aa_cells():
            t0 = time.time()
            sweep_aa(cell, args.reps, args.run_tag, fh)
            print(f"[AA] {cell.cell_id}  {time.time()-t0:5.1f}s", flush=True)

        for i, cell in enumerate(cells, 1):
            t0 = time.time()
            try:
                sweep_cell(cell, args.reps, args.run_tag, args.with_context_algo,
                           cell.cell_id in buildonly_ids, fh)
                status = "ok"
            except H.QueryError as exc:
                status = f"ERROR {str(exc)[:160]}"
                fh.write(json.dumps({
                    "run_tag": args.run_tag, "cell_id": cell.cell_id, "kind": cell.kind,
                    "key": cell.key, "match": cell.match, "threads": cell.threads,
                    "card": cell.card, "purpose": "CELL_ERROR", "error": str(exc)[:800],
                }) + "\n")
                fh.flush()
            el = time.time() - t_start
            print(f"[{i:3d}/{len(cells)}] {cell.cell_id:44s} {time.time()-t0:6.1f}s "
                  f"elapsed={el/60:5.1f}m  {status}", flush=True)

    print(f"SWEEP_DONE run_tag={args.run_tag} total={(time.time()-t_start)/60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
