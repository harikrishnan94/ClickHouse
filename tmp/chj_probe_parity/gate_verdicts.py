#!/usr/bin/env python3
"""Gate verdicts per PREREG 007: per-cell median thread-summed
ConcurrentHashJoinProbeMicroseconds, B <= A * (1 + band); band = max(3%, A/A
spread rule folded into the report's own band logic - here: 3% floor, widened
to the cell's pooled run spread fraction when that exceeds 3%). Wall verdict
printed alongside as the sanity column."""
import json, statistics, sys, glob

PROBE = "ConcurrentHashJoinProbeMicroseconds"
rows = []
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else
                          "/mnt/ch/ClickHouse/tmp/chj_probe_parity/fleet_results/results.shard*.jsonl")):
    for line in open(f):
        rows.append(json.loads(line))

cells = {}
for r in rows:
    if not r.get("valid"):
        cells.setdefault(r["cell"], {}).setdefault("invalid", []).append(r.get("invalid_reason"))
        continue
    arm = r["arm_role"] if "arm_role" in r else r["arm"]
    cells.setdefault(r["cell"], {}).setdefault(arm, {"probe": [], "wall": []})
    cells[r["cell"]][arm]["probe"].append(r["events"].get(PROBE, 0))
    cells[r["cell"]][arm]["wall"].append(r["duration_us"])

def spread_frac(vals):
    med = statistics.median(vals)
    return (max(vals) - min(vals)) / med if med else 0.0

wins = ties = losses = invalid = 0
loss_cells = []
for cell in sorted(cells):
    d = cells[cell]
    arms = [k for k in d if k not in ("invalid",)]
    if len(arms) != 2:
        invalid += 1
        print(f"{cell:48s} INVALID ({d.get('invalid', ['no data'])[:1]})")
        continue
    is_build_guard = ":build." in cell
    a_name = [a for a in arms if "base" in a.lower() or a == "A"][0]
    b_name = [a for a in arms if a != a_name][0]
    a, b = d[a_name], d[b_name]
    if min(len(a["probe"]), len(b["probe"])) < 5:
        invalid += 1
        print(f"{cell:48s} INSUFFICIENT")
        continue
    if is_build_guard:
        # guard rule: wall in-band (Build-event detail lives in the full report)
        pa, pb = statistics.median(a["wall"]), statistics.median(b["wall"])
    else:
        pa, pb = statistics.median(a["probe"]), statistics.median(b["probe"])
    band = max(0.03, spread_frac(a["probe"]), spread_frac(b["probe"]) if False else 0)
    band = max(0.03, spread_frac(a["probe"]))
    diff = (pb - pa) / pa if pa else 0.0
    wa, wb = statistics.median(a["wall"]), statistics.median(b["wall"])
    wdiff = (wb - wa) / wa if wa else 0.0
    if is_build_guard:
        v = "GUARD-OK" if abs(diff) <= band or diff <= 0 else "GUARD-RED"
        if v == "GUARD-OK": ties += 1
        else: losses += 1; loss_cells.append((cell + " [guard-wall]", diff, band))
        print(f"{cell:48s} {v:9s} wall A={pa/1e6:8.2f}s B={pb/1e6:8.2f}s diff={diff*100:+6.2f}% band={band*100:4.1f}%")
        continue
    if diff <= -band: v = "WIN"; wins += 1
    elif diff <= band: v = "TIE"; ties += 1
    else: v = "LOSS"; losses += 1; loss_cells.append((cell, diff, band))
    print(f"{cell:48s} {v:4s} probe A={pa/1e6:10.2f}s B={pb/1e6:10.2f}s diff={diff*100:+6.2f}% band={band*100:4.1f}% | wall {wdiff*100:+6.2f}%")

print(f"\nGATE RESULT: win={wins} tie={ties} loss={losses} invalid/insufficient={invalid}")
if loss_cells:
    print("PROBE-RED CELLS:")
    for c, d, b in loss_cells:
        print(f"  {c}  +{d*100:.2f}% (band {b*100:.1f}%)")
