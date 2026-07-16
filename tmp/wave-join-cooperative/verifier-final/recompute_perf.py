#!/usr/bin/env python3
"""Independent recompute of perf medians/deltas from raw JSONL evidence."""
import json, statistics, sys

def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

def samples(recs):
    """Return {(cell, arm): [wall_s, ...]}, plus oracle summaries."""
    out = {}
    oracles = {"assertion_fail": 0, "fingerprint_fail": 0, "integrity_fail": 0,
               "assertions": 0, "fingerprints": 0, "integrity": 0}
    engagement = {}
    for r in recs:
        k = r.get("kind")
        if k == "assertion":
            oracles["assertions"] += 1
            if not r.get("ok"):
                oracles["assertion_fail"] += 1
        elif k == "fingerprint":
            oracles["fingerprints"] += 1
            if not r.get("ok"):
                oracles["fingerprint_fail"] += 1
        elif k == "integrity":
            oracles["integrity"] += 1
            if not r.get("ok"):
                oracles["integrity_fail"] += 1
        elif "wall_us" in r:
            cell = r.get("cell") or r.get("kind")
            arm = r.get("arm") or r.get("binary") or "?"
            out.setdefault((cell, arm), []).append(r["wall_us"] / 1e6)
            leaves = r.get("RadixHashJoinLeafGroupBuilds")
            if leaves is not None:
                engagement.setdefault((cell, arm), set()).add(leaves)
    return out, oracles, engagement

def med(v):
    return statistics.median(v)

for path in sys.argv[1:]:
    print(f"=== {path} ===")
    recs = load(path)
    hdr = [r for r in recs if r.get("kind") == "header"]
    if hdr:
        h = hdr[0]
        print("header binary_a sha:", h.get("binary_a", {}).get("sha256", "")[:16],
              "binary_b sha:", h.get("binary_b", {}).get("sha256", "")[:16])
    # dump unknown keys of one sample record for schema confidence
    for r in recs:
        if "wall_us" in r:
            print("sample keys:", sorted(r.keys()))
            break
    s, oracles, engagement = samples(recs)
    print("oracles:", oracles)
    for (cell, arm), vals in sorted(s.items()):
        e = sorted(engagement.get((cell, arm), []))
        print(f"{cell:>8} arm={arm:<12} n={len(vals)} median={med(vals):.3f} vals={[round(v,3) for v in vals]} leaves={e}")
