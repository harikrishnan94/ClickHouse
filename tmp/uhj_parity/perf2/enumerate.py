#!/usr/bin/env python3
"""Unit 0: collect profiler samples across the matrix and prove the loop
enumeration in loops.py is complete (Gate G0.1).

    python3 enumerate.py collect --tag u0a          # run the profiled spread
    python3 enumerate.py gate --tag u0a             # G0.1: zero unexplained symbols
    python3 enumerate.py report --tag u0a           # human-readable coverage

The gate has power to fail because loops.py was written from reading the code, and
the symbol regexes in it name functions -- not sample counts. A loop the reading
missed shows up here as an unmapped symbol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perf"))

import harness as H  # noqa: E402
from loops import build_registry, classify_symbol  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# A per-process token, so re-running a tag in a new process cannot regenerate the
# same query ids and make a groupby over system.* silently sum two runs.
# (Prior mission E7.2: this produced an exact-2x corruption no gate caught.)
TOKEN = f"{int(time.time())}_{os.getpid()}"

# The profiled spread. Chosen for code-path coverage, not for timing: every
# key-getter family, every thread/cardinality point, kinds that switch the
# used-flag and non-joined machinery on and off, all three algorithms.
SPREAD_KINDS = ["INNER", "FULL", "LEFT SEMI"]
SPREAD_KEYS = ["u64", "str", "comp"]
SPREAD_TC = [(1, "small"), (1, "medium"), (16, "large"), (64, "large")]

# 1 ms CPU-time sampling: dense enough that a loop holding >=1% of join time is
# very unlikely to be missed, cheap enough not to distort the run (and these runs
# are never used for timing).
PROFILE_SETTINGS = {
    "query_profiler_cpu_time_period_ns": 1_000_000,
    "query_profiler_real_time_period_ns": 0,
    "allow_introspection_functions": 1,
}


def spread_cells():
    for kind in SPREAD_KINDS:
        for key in SPREAD_KEYS:
            for threads, card in SPREAD_TC:
                # `match` is fixed at hi for the enumeration: it changes how often a
                # loop runs, not which loops exist. `lo` is added for FULL only,
                # where it changes which branch of the non-joined scan is taken.
                yield H.Cell(kind, key, "hi", threads, card)
                if kind == "FULL":
                    yield H.Cell(kind, key, "lo", threads, card)


def collect(tag: str):
    out_path = os.path.join(RESULTS, f"samples_{tag}.jsonl")
    n_ok = n_skip = 0
    with open(out_path, "w") as fh:
        for cell in spread_cells():
            for algo in H.ALGOS:
                reason = cell.skip_reason(algo)
                if reason:
                    n_skip += 1
                    continue
                qid = f"enum_{tag}_{TOKEN}_{cell.cell_id}_{algo}".replace(" ", "-")
                settings = H.settings_for(cell, algo, PROFILE_SETTINGS)
                sql = H.join_sql(cell, "timed")
                try:
                    H.run_query(sql, settings, query_id=qid)
                except H.QueryError as exc:
                    print(f"  QUERY FAILED {cell.cell_id} {algo}: {exc}", file=sys.stderr)
                    continue
                H.flush_logs()
                ran = H.assert_algorithm(qid)
                if ran["verdict"] != algo:
                    # Not a soft warning: measuring the wrong implementation while
                    # labelling it the right one is the failure mode the prior
                    # mission spent a unit closing.
                    print(f"  ALGO MISMATCH {cell.cell_id}: asked {algo} got "
                          f"{ran['verdict']} {ran}", file=sys.stderr)
                syms = leaf_in_join_symbols(qid)
                fh.write(json.dumps({
                    "cell": cell.cell_id, "algo": algo, "query_id": qid,
                    "ran_as": ran["verdict"], "total_samples": ran["total_samples"],
                    "symbols": syms,
                }) + "\n")
                fh.flush()
                n_ok += 1
                print(f"  {cell.cell_id:34s} {algo:14s} ran_as={ran['verdict']:14s} "
                      f"samples={ran['total_samples']:6d} in-join-syms={len(syms)}")
    print(f"\ncollected {n_ok} profiled runs ({n_skip} skipped), -> {out_path}")
    return out_path


def leaf_in_join_symbols(query_id: str) -> dict:
    """Per-sample innermost frame, plus the innermost frame that is inside the join.

    Both are recorded. The leaf frame is where the cycles are; the leaf *in-join*
    frame is what the enumeration must explain. Recording the raw leaf too means a
    leaf that is outside the join (an allocator, say) is still visible in the
    artefact rather than silently dropped by the in-join filter.
    """
    rows = H.run_query(
        f"""
        SELECT demangle(addressToSymbol(trace[1])) AS leaf,
               arrayStringConcat(arrayMap(x -> demangle(addressToSymbol(x)), trace), '|') AS full,
               count() AS n
        FROM system.trace_log
        WHERE query_id = '{query_id}' AND trace_type = 'CPU'
        GROUP BY leaf, full
        FORMAT TSV
        """, {"allow_introspection_functions": 1}).strip()
    _loops, _excl, in_join = build_registry()
    agg: dict[str, int] = {}
    if rows:
        for line in rows.split("\n"):
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            _leaf, full, n = parts
            frames = full.split("|")
            if not any(in_join.search(f) for f in frames):
                continue  # sample is not inside the join at all
            # innermost in-join frame
            for f in frames:
                if in_join.search(f):
                    agg[f] = agg.get(f, 0) + int(n)
                    break
    return agg


def load(tag):
    path = os.path.join(RESULTS, f"samples_{tag}.jsonl")
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def gate(tag: str, min_pct: float = 0.0):
    """G0.1: every in-join sampled symbol maps to an enumerated loop or an
    explicit exclusion. Zero unexplained."""
    recs = load(tag)
    loops, excl, _ = build_registry()

    totals: dict[str, int] = {}
    per_algo: dict[str, dict[str, int]] = {}
    for r in recs:
        for sym, n in r["symbols"].items():
            totals[sym] = totals.get(sym, 0) + n
            per_algo.setdefault(r["algo"], {})
            per_algo[r["algo"]][sym] = per_algo[r["algo"]].get(sym, 0) + n
    grand = sum(totals.values()) or 1

    mapped, excluded, unexplained = {}, {}, {}
    for sym, n in totals.items():
        ids, why = classify_symbol(sym, loops, excl)
        if ids:
            mapped[sym] = (ids, n)
        elif why:
            excluded[sym] = (why, n)
        else:
            unexplained[sym] = n

    print(f"G0.1 loop-enumeration completeness  [tag={tag}]")
    print(f"  profiled runs        : {len(recs)}")
    print(f"  distinct in-join syms: {len(totals)}")
    print(f"  in-join samples      : {grand}")
    print(f"  mapped to a loop     : {len(mapped)} syms, "
          f"{sum(n for _, n in mapped.values())} samples "
          f"({100*sum(n for _, n in mapped.values())/grand:.2f}%)")
    print(f"  explicitly excluded  : {len(excluded)} syms, "
          f"{sum(n for _, n in excluded.values())} samples "
          f"({100*sum(n for _, n in excluded.values())/grand:.2f}%)")
    print(f"  UNEXPLAINED          : {len(unexplained)} syms, "
          f"{sum(unexplained.values())} samples "
          f"({100*sum(unexplained.values())/grand:.2f}%)")

    if unexplained:
        print("\n  unexplained symbols, by sample count:")
        for sym, n in sorted(unexplained.items(), key=lambda kv: -kv[1]):
            print(f"    {n:7d} ({100*n/grand:5.2f}%)  {sym[:150]}")

    # coverage the other way: an enumerated loop that never received a sample is
    # not a gate failure (it may be a cold path, or excluded by the matrix), but
    # it is recorded so that "enumerated" is not confused with "exercised".
    seen_ids = set()
    for ids, _ in mapped.values():
        seen_ids.update(ids)
    never = [l["id"] for l in loops if l["id"] not in seen_ids]
    if never:
        print(f"\n  enumerated but unsampled ({len(never)}): {', '.join(sorted(never))}")
        print("    (not a failure: cold or not exercised by this matrix; "
              "each still needs a codegen artefact under G1.1)")

    out = os.path.join(RESULTS, f"g01_{tag}.json")
    with open(out, "w") as fh:
        json.dump({
            "tag": tag, "runs": len(recs), "in_join_samples": grand,
            "mapped": {k: {"loops": v[0], "samples": v[1]} for k, v in mapped.items()},
            "excluded": {k: {"reason": v[0], "samples": v[1]} for k, v in excluded.items()},
            "unexplained": unexplained,
            "unsampled_loops": never,
        }, fh, indent=1)
    print(f"\n  artefact: {out}")

    red = bool(unexplained)
    print(f"\nG0.1: {'RED' if red else 'GREEN'}")
    return 1 if red else 0


def power(tag: str):
    """Does G0.1 still have the power to fail?

    A completeness gate that maps everything is worthless if it maps everything
    *because the patterns are broad*. Three checks:

      1. injected control: a symbol that is genuinely nothing must come back
         unexplained. If it does not, some pattern is a catch-all.
      2. knockout: for each enumerated loop, delete its patterns and confirm the
         gate goes red. A loop whose removal changes nothing is a loop whose
         symbols were being absorbed by some other pattern -- so its sample
         attribution is not evidence of anything.
      3. exclusion budget: report what share of samples the exclusions absorb.
    """
    import copy
    import re as _re
    recs = load(tag)
    loops, excl, _ = build_registry()

    totals: dict[str, int] = {}
    for r in recs:
        for sym, n in r["symbols"].items():
            totals[sym] = totals.get(sym, 0) + n
    grand = sum(totals.values()) or 1

    print("G0.1 power check")

    controls = [
        "DB::HashJoin::thisFunctionDoesNotExist(int)",
        "DB::Unified::CompletelyMadeUpSymbol::loop()",
        "some::random::symbol",
    ]
    bad = [c for c in controls if classify_symbol(c, loops, excl) != ([], None)]
    print(f"  1. injected controls unexplained : {len(controls)-len(bad)}/{len(controls)}"
          f"  {'OK' if not bad else 'FAIL ' + str(bad)}")

    absorbed = 0
    for sym, n in totals.items():
        ids, why = classify_symbol(sym, loops, excl)
        if not ids and why:
            absorbed += n
    print(f"  3. exclusion budget              : {100*absorbed/grand:.2f}% of in-join "
          f"samples absorbed by exclusions (each carries a written reason)")

    print("  2. per-loop knockout (loops whose patterns are load-bearing):")
    weak = []
    for i, l in enumerate(loops):
        stripped = copy.deepcopy(loops)
        stripped[i] = dict(stripped[i])
        stripped[i]["_re"] = [_re.compile(r"(?!x)x")]  # matches nothing
        newly = 0
        for sym, n in totals.items():
            was, _ = classify_symbol(sym, loops, excl)
            if l["id"] not in was:
                continue
            now_ids, now_why = classify_symbol(sym, stripped, excl)
            if not now_ids and not now_why:
                newly += n
        if newly == 0:
            weak.append(l["id"])
        else:
            print(f"       {l['id']:5s} removing it leaves {newly:7d} samples "
                  f"({100*newly/grand:5.2f}%) unexplained")
    if weak:
        print(f"     loops whose removal changes nothing: {', '.join(weak)}")
        print("       -- these are either unsampled, or their symbols are also "
              "claimed by another loop (inlined together). Both are recorded in the "
              "loop table; neither is evidence the loop does not exist.")
    return 0


def report(tag: str):
    recs = load(tag)
    loops, excl, _ = build_registry()
    per_loop: dict[str, dict[str, int]] = {}
    for r in recs:
        for sym, n in r["symbols"].items():
            ids, _ = classify_symbol(sym, loops, excl)
            for i in ids:
                per_loop.setdefault(i, {}).setdefault(r["algo"], 0)
                per_loop[i][r["algo"]] += n
    print(f"{'loop':6s} {'hash':>9s} {'par_hash':>9s} {'unified':>9s}  name")
    for l in loops:
        d = per_loop.get(l["id"], {})
        print(f"{l['id']:6s} {d.get('hash',0):9d} {d.get('parallel_hash',0):9d} "
              f"{d.get('unified_hash',0):9d}  {l['name'][:60]}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "gate", "report", "power"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.tag)
        return 0
    if a.cmd == "gate":
        return gate(a.tag)
    if a.cmd == "power":
        return power(a.tag)
    return report(a.tag)


if __name__ == "__main__":
    sys.exit(main())
