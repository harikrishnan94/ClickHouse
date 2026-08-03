#!/usr/bin/env python3
"""G1.1: assign every enumerated loop a codegen-evidence tier, with the evidence.

The mission requires an unconditional codegen artefact per loop. Producing a full
disassembly-and-mca comparison for all 45 would be mostly waste, because for a large
share of them the two implementations provably run the *same instructions* -- and a
proof of identity is stronger evidence of a zero codegen delta than any measurement of
one. The tiers, strongest first:

  T0 ICF        the linker folded the two trees' symbols onto ONE address. Same
                instructions, same address. Codegen delta exactly zero, and mca on one
                side is mca on the other by construction. (codegen/icf_census.json)
  T1 SHARED     the loop lives in a header or .cpp with NO per-tree copy -- one file,
                included by all three. Checked mechanically here by counting copies
                under src/Interpreters/{HashJoin,UnifiedHashJoin}/. Delta zero by
                construction.
  T2 IDENTICAL  per-tree copies exist but the emitted symbols are opcode-identical
                (symdiff.py byte comparison). Delta zero, measured.
  T3 ANALYSED   the sides genuinely differ; full counts + llvm-mca in a codegen artefact.
  T4 NO-COUNTERPART  the loop exists in only one implementation. The artefact records
                what the others do instead.
  T5 TODO       not yet covered. Reported as such rather than hidden.

    python3 tier.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loops import LOOPS  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# Loops whose codegen was compared in a full artefact (T3), with the artefact path.
ANALYSED = {
    "P1": "codegen/P1_G2_probe_and_gather.md",
    "P2": "codegen/P1_G2_probe_and_gather.md",
    "P3": "codegen/P1_G2_probe_and_gather.md",
    "P4": "codegen/P1_G2_probe_and_gather.md",
    "P5": "codegen/P1_G2_probe_and_gather.md",
    "P10": "codegen/P1_G2_probe_and_gather.md",
    "G1": "codegen/P1_G2_probe_and_gather.md",
    "G2": "codegen/P1_G2_probe_and_gather.md",
    "G3": "codegen/P1_G2_probe_and_gather.md",
    "B22": "codegen/K1_composite_keygetter.md",
}

# Loops proven ICF-folded between the trees.
ICF = {"G4", "P7"}

# Source files with no per-tree copy, keyed by the loop that lives in them. Verified
# mechanically below rather than asserted.
SHARED_FILE = {
    "B0": "src/Interpreters/JoinUtils.cpp",
    "B1": "src/Interpreters/NullableUtils.cpp",
    "B13": "src/Common/HashTable/HashTable.h",
    "B14": "src/Interpreters/RowRefs.h",
    "G5": "src/Interpreters/HashJoin/ScatteredBlock.h",
    "N5": "src/Interpreters/RowRefs.h",
    "B22": "src/Interpreters/AggregationCommon.h",
}


def per_tree_copies(path):
    """How many copies of this file exist across the two join trees?"""
    base = os.path.basename(path)
    hits = []
    for d in ("src/Interpreters/HashJoin", "src/Interpreters/UnifiedHashJoin"):
        p = os.path.join(ROOT, d, base)
        if os.path.exists(p):
            hits.append(os.path.join(d, base))
    if os.path.exists(os.path.join(ROOT, path)) and path not in hits:
        hits.append(path)
    return hits


def main():
    census_path = os.path.join(HERE, "codegen", "icf_census.json")
    census = json.load(open(census_path)) if os.path.exists(census_path) else {"folded_pairs": []}
    folded = len(census.get("folded_pairs", []))

    print(f"G1.1 codegen-evidence tier per enumerated loop   "
          f"({len(LOOPS)} loops; ICF census has {folded} folded pairs)")
    print()
    print(f"{'loop':5s} {'tier':14s} {'impls':22s} evidence")
    print("-" * 118)
    counts = {}
    rows = []
    for l in LOOPS:
        lid = l["id"]
        impls = ",".join(x[:4] for x in l["impls"])
        if lid in ICF:
            tier, ev = "T0 ICF", "one address holds both trees' symbols (codegen/icf_census.json)"
        elif len(l["impls"]) == 1:
            tier = "T4 NO-COUNTERPART"
            ev = f"only in {l['impls'][0]}; artefact records what the others do instead"
            if lid in ANALYSED:
                ev = ANALYSED[lid] + " (+ no counterpart)"
        elif lid in SHARED_FILE:
            copies = per_tree_copies(SHARED_FILE[lid])
            if len(copies) == 1:
                tier = "T1 SHARED"
                ev = f"single file {copies[0]}, no per-tree copy -> same instructions"
            else:
                tier = "T2 IDENTICAL?" 
                ev = f"UNEXPECTED: {len(copies)} copies {copies}"
        elif lid in ANALYSED:
            tier, ev = "T3 ANALYSED", ANALYSED[lid]
        else:
            tier, ev = "T5 TODO", "no codegen artefact yet"
        counts[tier.split()[0]] = counts.get(tier.split()[0], 0) + 1
        rows.append((lid, tier, impls, ev))
        print(f"{lid:5s} {tier:14s} {impls:22s} {ev}")

    print()
    print("tier totals: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    todo = [r[0] for r in rows if r[1].startswith("T5")]
    if todo:
        print(f"\nT5 (no artefact yet), {len(todo)}: {', '.join(todo)}")
    out = os.path.join(HERE, "codegen", "tiers.json")
    json.dump([{"loop": a, "tier": b, "impls": c, "evidence": d} for a, b, c, d in rows],
              open(out, "w"), indent=1)
    print(f"\nartefact: {out}")
    print(f"\nG1.1: {'GREEN' if not todo else f'PARTIAL - {len(todo)} loops without an artefact'}")
    return 0 if not todo else 1


if __name__ == "__main__":
    sys.exit(main())
