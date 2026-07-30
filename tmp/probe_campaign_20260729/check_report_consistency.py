#!/usr/bin/env python3
"""Check REPORT.md's headline numbers against the scorer TSVs it cites.

Three separate wrong numbers reached this report by being typed rather than
derived — a cell id that was never in the plan, a T16/T96 swap, and a 36.0 s
value printed as "3xx s" — each caught by a verifier rather than by me. Tables
are now generated (`make_report_tables.py`); this checks the prose too, so the
report cannot drift from its own data without the check going red.

Usage: check_report_consistency.py   (exit 0 = consistent)
"""
import csv
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPORT = (HERE / "REPORT.md").read_text()
# The verification section quotes the defects it fixed, so a withdrawn string is
# legitimate there and only there.
BODY = REPORT.split("## 6. Independent verification")[0]

SUITES = (
    ("fleet ABBA", "reports/fleet_abba.tsv", (68, 4, 6, 2, 5, 71)),
    ("fleet BAAB", "reports/fleet_baab.tsv", (68, 4, 6, 2, 5, 71)),
    ("real tier a", "reports/jbmt_real_a.tsv", (161, 51, 156, 43, 127, 198)),
    ("real tier b", "reports/jbmt_real_b.tsv", (164, 50, 151, 49, 126, 190)),
)
# Numbers the report states, which must be re-derivable from the TSVs.
AGGREGATES = {
    ("real tier a", "probe_cost"): -8.8, ("real tier a", "projection_cost"): 2.8,
    ("real tier b", "probe_cost"): -8.8, ("real tier b", "projection_cost"): 10.0,
    ("fleet ABBA", "probe_cost"): -35.2, ("fleet ABBA", "projection_cost"): 26.7,
    ("fleet BAAB", "probe_cost"): -35.2, ("fleet BAAB", "projection_cost"): 26.6,
}
MUST_APPEAR = ["+7.34 %", "+10.02 %", "9 of 20 are slower", "32-assertion",
               "36.0 s", "+5.01", "+211.7 %", "disjoint", "1.00×"]
# Withdrawn claims and corrected figures: must not survive in the report BODY.
MUST_NOT_APPEAR_IN_BODY = [
    "3xx s", "24-assertion", "33 assertions", "35.9 s",
    "k128:probe.inner_all.S4.T96", "str:probe.inner_all.S2.T1` −5.0",
    "the same two\nunits offend", "measured twice\nover independently",
    "scales with materialized output**", "Blocked on validity",
]

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name:<52} {detail}")
    if not ok:
        fails.append(name)


for label, tsv, expect in SUITES:
    rows = [r for r in csv.DictReader(open(HERE / tsv), delimiter="\t")
            if r["probe_cost_verdict"] != "NO-VERDICT"]
    got = tuple(sum(1 for r in rows if r[f"{m}_verdict"] == v)
                for m in ("probe_cost", "projection_cost") for v in ("WIN", "TIE", "LOSS"))
    check(f"{label}: W/T/L tallies", got == expect, f"{got}")
    check(f"{label}: tallies sum to the scored count",
          sum(got[:3]) == len(rows) == sum(got[3:]), f"{len(rows)} scored")
    for m in ("probe_cost", "projection_cost"):
        a = sum(float(r[f"{m}_median_a_us"]) for r in rows)
        b = sum(float(r[f"{m}_median_b_us"]) for r in rows)
        pct = (b - a) / a * 100
        want = AGGREGATES[(label, m)]
        check(f"{label}: {m} aggregate ≈ {want:+.1f}%", abs(pct - want) < 0.06, f"{pct:+.2f}%")
    # No cell may be netted: a unit must be scored on both metrics or neither.
    mixed = [r for r in csv.DictReader(open(HERE / tsv), delimiter="\t")
             if (r["probe_cost_verdict"] == "NO-VERDICT") != (r["projection_cost_verdict"] == "NO-VERDICT")]
    check(f"{label}: both metrics scored or neither", not mixed, f"{len(mixed)} mixed")

# Every cell/unit id quoted in the report must exist in one of the TSVs.
known = set()
for _, tsv, _ in SUITES:
    known |= {r["unit"] for r in csv.DictReader(open(HERE / tsv), delimiter="\t")}
quoted = set(re.findall(r"`([a-z0-9_]+:[a-z0-9_.]+|[a-z]+__[A-Za-z0-9_]+__tier[ab])`", BODY))
ghosts = {q for q in quoted if q not in known and "__" in q or (":" in q and q not in known)}
check("no ghost cell/unit ids quoted in the report body", not ghosts, f"{sorted(ghosts)[:3]}")

for s in MUST_APPEAR:
    check(f'report states "{s}"', s in REPORT)
for s in MUST_NOT_APPEAR_IN_BODY:
    check(f'report body free of "{s[:34]}"', s not in BODY)

print(f"\nREPORT CONSISTENCY: {'PASS' if not fails else 'FAIL'} ({len(fails)} problem(s))")
for f in fails:
    print(f"  FAILED: {f}")
sys.exit(1 if fails else 0)
