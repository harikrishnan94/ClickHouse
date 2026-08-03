#!/usr/bin/env python3
"""Identical-code-folding census between the baseline and unified join trees.

Two uses, both load-bearing for this mission.

1. **Codegen evidence, of the strongest available kind.** When the linker folds
   `DB::HashJoin::f` and `DB::Unified::HashJoin::f` onto one address, the two are not
   merely "textually identical" -- they are the *same instructions*, and the codegen
   delta for that loop is exactly zero. No disassembly, no counting and no `llvm-mca`
   run can improve on that, and none is needed: mca on one side is mca on the other
   by construction.

2. **A correction to the inherited algorithm assertion.** `addressToSymbol` maps an
   address to one name. For a folded address that name is arbitrary, so a `hash` run
   can and does report a `DB::Unified::HashJoin` frame. The prior mission's rule
   ("unified iff any Unified frame appears") therefore mis-identifies runs. The ghost
   set computed here is what makes the rule sound again.

    python3 icf_census.py --binary <path> [--out <json>]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NM = os.path.join(HERE, "..", "perf", "bin", "llvm-nm")

BASELINE_RE = re.compile(r"\bDB::(HashJoin|ConcurrentHashJoin|JoinStuff|NotJoinedHash|"
                         r"HashJoinMethods|AddedColumns|HashJoinResult|KnownRowsHolder)\b")
UNIFIED_RE = re.compile(r"\bDB::Unified::")


def symbols(binary):
    out = subprocess.run([NM, "--defined-only", "--demangle", binary],
                         capture_output=True, text=True, check=True).stdout
    by_addr = collections.defaultdict(list)
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3:
            continue
        addr, kind, name = parts
        if kind.upper() not in ("T", "W"):
            continue
        try:
            a = int(addr, 16)
        except ValueError:
            continue
        by_addr[a].append(name)
    return by_addr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.path.join(HERE, "bin", "clickhouse.ref"))
    ap.add_argument("--out", default=os.path.join(HERE, "codegen", "icf_census.json"))
    a = ap.parse_args()

    by_addr = symbols(a.binary)

    folded_pairs = []      # addresses holding BOTH a unified and a baseline join name
    unified_only = []      # unified join symbols with no baseline twin at the address
    ghost_names = set()    # every name living at a cross-tree folded address

    n_unified = 0
    for addr, names in by_addr.items():
        u = [n for n in names if UNIFIED_RE.search(n)]
        if not u:
            continue
        n_unified += len(u)
        b = [n for n in names if not UNIFIED_RE.search(n) and BASELINE_RE.search(n)]
        if b:
            folded_pairs.append({"addr": hex(addr), "unified": u, "baseline": b})
            ghost_names.update(names)
        else:
            unified_only.append({"addr": hex(addr), "unified": u,
                                 "other_names_at_addr": [n for n in names
                                                         if not UNIFIED_RE.search(n)]})

    res = {
        "binary": a.binary,
        "unified_symbols_total": n_unified,
        "folded_with_baseline_twin": len(folded_pairs),
        "unified_only_addresses": len(unified_only),
        "ghost_names_count": len(ghost_names),
        "folded_pairs": folded_pairs,
        "ghost_names": sorted(ghost_names),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)

    print(f"binary                      : {a.binary}")
    print(f"DB::Unified:: text symbols  : {n_unified}")
    print(f"addresses folded with a")
    print(f"  baseline join twin (ICF)  : {len(folded_pairs)}"
          f"   <- codegen delta provably ZERO for these")
    print(f"unified-only addresses      : {len(unified_only)}"
          f"   <- these are where a codegen difference can live")
    print(f"ghost names (unsafe for the")
    print(f"  algorithm assertion)      : {len(ghost_names)}")
    print(f"\nartefact: {a.out}")

    print("\nsample of folded pairs (proof of zero codegen delta):")
    for p in folded_pairs[:12]:
        print(f"  {p['addr']}  {p['unified'][0][:100]}")
        print(f"  {'':18s}== {p['baseline'][0][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
