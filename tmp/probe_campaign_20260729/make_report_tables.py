#!/usr/bin/env python3
"""Emit REPORT.md's per-suite tables straight from a scorer TSV.

Hand-transcribing numbers into prose is how this campaign produced two wrong
tables (a cell id that was never in the plan, and a T16/T96 pair swapped), both
caught by verifiers rather than by me. Every table a reviewer is likely to quote
is generated here instead, so the report cannot disagree with the TSV it cites.

Usage: make_report_tables.py <scorer.tsv> <label> [--top N]
"""
import argparse
import csv
import statistics


def ms(x):
    return "-" if x in (None, "") else f"{float(x) / 1000:,.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("label")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.tsv), delimiter="\t"))
    scored = [r for r in rows if r["probe_cost_verdict"] != "NO-VERDICT"]

    def total(key):
        return sum(float(r[key]) for r in scored if r[key])

    print(f"### {args.label}: generated from `{args.tsv}`\n")
    print(f"Units in file: {len(rows)} · scored: {len(scored)} · "
          f"NO-VERDICT: {len(rows) - len(scored)}\n")

    print("| | `probe_cost` | `projection_cost` |")
    print("| --- | --- | --- |")
    cells = {}
    for m in ("probe_cost", "projection_cost"):
        c = {v: sum(1 for r in scored if r[f"{m}_verdict"] == v) for v in ("WIN", "TIE", "LOSS")}
        a, b = total(f"{m}_median_a_us"), total(f"{m}_median_b_us")
        med = statistics.median(float(r[f"{m}_delta_pct"]) for r in scored)
        cells[m] = (c, a, b, med)
    print(f"| verdicts | {len(scored)} | {len(scored)} |")
    print("| **WIN / TIE / LOSS** | " + " | ".join(
        f"**{cells[m][0]['WIN']} / {cells[m][0]['TIE']} / {cells[m][0]['LOSS']}**"
        for m in ("probe_cost", "projection_cost")) + " |")
    print("| aggregate | " + " | ".join(
        f"{ms(cells[m][1])} ms → {ms(cells[m][2])} ms (**{(cells[m][2] - cells[m][1]) / cells[m][1] * 100:+.1f} %**)"
        for m in ("probe_cost", "projection_cost")) + " |")
    print("| median per-unit delta | " + " | ".join(
        f"**{cells[m][3]:+.1f} %**" for m in ("probe_cost", "projection_cost")) + " |")

    # The two directly measured quantities that are NOT verdict metrics but must be visible.
    pt_a, pt_b = total("probe_total_a_us"), total("probe_total_b_us")
    w_a, w_b = total("wall_a_us"), total("wall_b_us")
    faster = sum(1 for r in scored if float(r["wall_b_us"]) < float(r["wall_a_us"]))
    slower = sum(1 for r in scored if float(r["wall_b_us"]) > float(r["wall_a_us"]))
    print(f"\n**Recorded, never a verdict** (this campaign verdicts only the two metrics above):")
    print(f"\n| measured quantity | arm A | arm B | delta |")
    print("| --- | --- | --- | --- |")
    print(f"| `ConcurrentHashJoinProbeMicroseconds` (the probe total the two metrics sum to) | "
          f"{ms(pt_a)} ms | {ms(pt_b)} ms | **{(pt_b - pt_a) / pt_a * 100:+.2f} %** |")
    print(f"| wall clock (`query_duration_ms`) | {ms(w_a)} ms | {ms(w_b)} ms | "
          f"**{(w_b - w_a) / w_a * 100:+.2f} %** |")
    print(f"\nPer-unit wall clock: **{slower} of {len(scored)} units slower**, {faster} faster, "
          f"{len(scored) - faster - slower} equal.")

    # Where the probe_cost improvement actually comes from.
    net = sum(float(r["probe_cost_delta_us"]) for r in scored)
    imp = sorted(scored, key=lambda r: float(r["probe_cost_delta_us"]))
    print(f"\n**Concentration of the `probe_cost` aggregate** (net {ms(net)} ms):\n")
    print("| improving units | their d(`probe_cost`) | share of net | their d(probe total) | their d(wall) |")
    print("| --- | --- | --- | --- | --- |")
    for n in (5, 20):
        top = imp[:n]
        dpc = sum(float(r["probe_cost_delta_us"]) for r in top)
        dtot = sum(float(r["probe_total_b_us"]) - float(r["probe_total_a_us"]) for r in top)
        dw = sum(float(r["wall_b_us"]) - float(r["wall_a_us"]) for r in top)
        print(f"| top {n} | {ms(dpc)} ms | {dpc / net * 100:.1f} % | **{ms(dtot)} ms** | "
              f"**{ms(dw)} ms** |")

    for m, short in (("probe_cost", "probe"), ("projection_cost", "projection")):
        L = sorted((r for r in scored if r[f"{m}_verdict"] == "LOSS"),
                   key=lambda r: -float(r[f"{m}_delta_pct"]))
        print(f"\n**Worst `{m}` regressions** ({len(L)} total):\n")
        print(f"| unit | {short} A (ms) | {short} B (ms) | delta | band | dispatch A→B | lookup A→B | probe total A→B |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in L[:args.top]:
            print(f"| `{r['unit']}` | {ms(r[f'{m}_median_a_us'])} | {ms(r[f'{m}_median_b_us'])} | "
                  f"**{float(r[f'{m}_delta_pct']):+.1f} %** | {float(r[f'{m}_band_pct']):.1f} % | "
                  f"{ms(r['dispatch_a_us'])} → {ms(r['dispatch_b_us'])} | "
                  f"{ms(r['lookup_a_us'])} → {ms(r['lookup_b_us'])} | "
                  f"{ms(r['probe_total_a_us'])} → {ms(r['probe_total_b_us'])} |")

    opp = [r for r in scored if {r["probe_cost_verdict"], r["projection_cost_verdict"]} == {"WIN", "LOSS"}]
    print(f"\n**Units whose two metrics move in opposite directions: {len(opp)}** "
          f"(never netted; each appears in both lists).")

    nv = [r for r in rows if r["probe_cost_verdict"] == "NO-VERDICT"]
    if nv:
        print(f"\n**NO-VERDICT units ({len(nv)}), with the harness's own reason:**\n")
        print("| unit | reason |")
        print("| --- | --- |")
        for r in sorted(nv, key=lambda r: r["unit"]):
            print(f"| `{r['unit']}` | {r['probe_cost_reason']} |")


if __name__ == "__main__":
    main()
