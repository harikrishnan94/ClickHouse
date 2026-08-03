#!/usr/bin/env python3
"""A sound replacement for the inherited symbol-level algorithm assertion.

The inherited rule (perf/harness.py:347-378) is: the run is `unified_hash` if any
sampled frame demangles to something containing `DB::Unified::HashJoin`, tested after
`DB::ConcurrentHashJoin`. It reports 264/264 agreement in the prior mission, and it is
nevertheless **unsound**, which this mission found the moment it profiled a `hash` run:

    ALGO MISMATCH INNER|u64|hi|t1|small: asked hash got unified_hash
        {'parallel_hash': 0, 'unified_hash': 1, 'hash': 13, ...}

The cause is identical-code folding, confirmed independently of the profiler:

    $ llvm-nm --defined-only -C clickhouse.ref | grep canRemoveColumnsFromLeftBlock
    0000000014289180 T DB::Unified::HashJoin::canRemoveColumnsFromLeftBlock(DB::TableJoin const&)
    0000000014289180 T DB::HashJoin::canRemoveColumnsFromLeftBlock(DB::TableJoin const&)

One address, two names. `addressToSymbol` picks one arbitrarily, so a `hash` run
genuinely does report a `DB::Unified::HashJoin` frame, and a presence test on a
1-in-13 ghost frame decides the verdict.

The fix is to ignore frames whose symbol is ICF-ambiguous. The ambiguous set is
computed from the binary by icf_census.py, so it is derived from the mechanism rather
than tuned against the samples.

    python3 algoassert.py recheck --tag u0a
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perf"))

import harness as H  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, "codegen", "icf_census.json")

MARKERS = [("parallel_hash", "DB::ConcurrentHashJoin"),
           ("unified_hash", "DB::Unified::HashJoin"),
           ("hash", "DB::HashJoin::")]


def ghost_names_for_markers():
    """Names living at a cross-tree ICF-folded address that also contain a marker.

    Small by construction -- only folded addresses contribute -- so it can be
    inlined into the SQL predicate.
    """
    with open(CENSUS) as fh:
        census = json.load(fh)
    out = set()
    for name in census["ghost_names"]:
        if any(m in name for _, m in MARKERS):
            out.add(name)
    return sorted(out)


def assert_algorithm(query_id: str, ghosts: list[str]) -> dict:
    ghost_sql = ", ".join("'" + g.replace("'", "\\'") + "'" for g in ghosts) or "''"
    counts = {}
    for label, marker in MARKERS:
        n = H.run_query(
            f"""
            SELECT countIf(arrayExists(
                       x -> position(demangle(addressToSymbol(x)), '{marker}') > 0
                            AND NOT has([{ghost_sql}], demangle(addressToSymbol(x))),
                       trace))
            FROM system.trace_log
            WHERE query_id = '{query_id}' AND trace_type = 'CPU' FORMAT TSV
            """, {"allow_introspection_functions": 1}).strip()
        counts[label] = int(n or 0)
    # Ordered, as before: parallel_hash's shards ARE baseline HashJoin objects, so
    # ConcurrentHashJoin must be tested first.
    for label, _ in MARKERS:
        if counts[label] > 0:
            counts["verdict"] = label
            break
    else:
        counts["verdict"] = "UNKNOWN"
    return counts


def recheck(tag: str):
    path = os.path.join(HERE, "results", f"samples_{tag}.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    ghosts = ghost_names_for_markers()
    print(f"ICF-ambiguous marker-bearing names excluded from the test: {len(ghosts)}")
    for g in ghosts:
        print(f"    {g[:120]}")
    print()

    old_bad, new_bad, n = [], [], 0
    for r in recs:
        want = r["algo"]
        if r["ran_as"] != want:
            old_bad.append((r["cell"], want, r["ran_as"]))
        got = assert_algorithm(r["query_id"], ghosts)
        n += 1
        if got["verdict"] != want:
            new_bad.append((r["cell"], want, got))
    print(f"runs rechecked                       : {n}")
    print(f"mismatches under the INHERITED rule  : {len(old_bad)}")
    for c, w, g in old_bad:
        print(f"    {c:34s} asked {w:14s} got {g}")
    print(f"mismatches under the CORRECTED rule  : {len(new_bad)}")
    for c, w, g in new_bad:
        print(f"    {c:34s} asked {w:14s} got {g}")

    out = os.path.join(HERE, "results", f"algoassert_{tag}.json")
    with open(out, "w") as fh:
        json.dump({"tag": tag, "runs": n, "ghosts_excluded": ghosts,
                   "inherited_rule_mismatches": old_bad,
                   "corrected_rule_mismatches": new_bad}, fh, indent=1)
    print(f"\nartefact: {out}")
    print(f"\nG0.1b algorithm identity: {'GREEN' if not new_bad else 'RED'}")
    return 1 if new_bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["recheck"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    sys.exit(recheck(a.tag))
