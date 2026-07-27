#!/usr/bin/env python3
"""check_matrix.py -- the G-coverage gate: join fleet/matrix.json with results
JSONL files and fleet/dispositions.json, print per-disposition counts, and
end with the machine-checkable final line '<N> undispositioned' (the gate is
green iff N == 0).

Rules:
- Universe = matrix.json's 1800 base cells (see matrix_gen.py docstring for
  why modifier/.hash cells are not universe members).
- A universe cell counts as undispositioned when it has no entry in the
  dispositions file OR its entry fails validation:
    * MEASURED requires, for that exact cell id in the results (rows of the
      LAST attempt nonce per (cell, arm, host) win, via fleet_ab
      dedup_last_attempt -- imported, so gate and driver cannot drift):
        (a) both arm roles A and B present, each with >= MIN_RUNS (5) VALID
            runs;
        (b) the two arms ran DIFFERENT binaries (binary_sha256 must differ),
            unless the disposition entry carries "aa_acceptable": true --
            no plan cell does;
        (c) every valid run's cell_axes.threads_effective equals the cell
            id's nominal T (a threads-override row is NOT evidence for the
            nominal cell);
        (d) exactly one settings_fingerprint per arm (no mixed-settings
            pooling).
    * INFERRED requires a 'from' cell that itself passes the full MEASURED
      validation above (the 'from' cell may be a modifier/.hash auxiliary
      cell, which carries no disposition of its own), plus a non-empty
      'rule'.
    * PARITY-ONLY / EXCLUDED-INVALID / NOT-CLAIMED require non-empty
      'evidence'.
- Dispositions naming cells outside the universe are warned about and
  ignored (they cannot turn the gate green or red).
- A missing results file is a HARD ERROR (fail-closed), never a warning.

Dispositions file format (fleet/dispositions.json):
  {"<cell>": {"disposition": "MEASURED|INFERRED|PARITY-ONLY|EXCLUDED-INVALID|NOT-CLAIMED",
              "evidence": "...", "from": "<cell>", "rule": "...",
              "aa_acceptable": false}, ...}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

FLEET_DIR = pathlib.Path(__file__).resolve().parent

# Row semantics (freshest-attempt dedup, run-count floor, fail-closed results
# loading) are fleet_ab's; import them so the coverage gate can never drift
# from the driver it audits (same pattern as matrix_gen.py).
sys.path.insert(0, str(FLEET_DIR.parent))
import fleet_ab  # noqa: E402

MIN_RUNS = fleet_ab.MIN_VERDICT_RUNS
DISPOSITIONS = ("MEASURED", "INFERRED", "PARITY-ONLY", "EXCLUDED-INVALID", "NOT-CLAIMED")


def load_results_rows(spec: str | None) -> list[dict]:
    # Only the optional-spec arm (no --results given) is local.
    return fleet_ab.load_result_rows(spec) if spec else []


def nominal_threads(cell: str) -> int | None:
    rest = cell.partition(":")[2]
    parts = rest.split(".")
    if len(parts) > 3 and parts[3].startswith("T") and parts[3][1:].isdigit():
        return int(parts[3][1:])
    return None


def collect_arm_stats(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """cell -> arm_role -> {runs, shas, fps, bad_threads} over VALID rows of
    the freshest attempt."""
    stats: dict[str, dict[str, dict]] = {}
    for r in fleet_ab.dedup_last_attempt(rows):
        if not r.get("valid"):
            continue
        role = r.get("arm_role") or r.get("arm")
        st = stats.setdefault(r["cell"], {}).setdefault(
            role, {"runs": set(), "shas": set(), "fps": set(), "bad_threads": 0})
        st["runs"].add(r.get("run"))
        st["shas"].add(r.get("binary_sha256") or "(missing)")
        st["fps"].add(r.get("settings_fingerprint") or "(missing)")
        nominal = nominal_threads(r["cell"])
        if nominal is None or (r.get("cell_axes") or {}).get("threads_effective") != nominal:
            st["bad_threads"] += 1
    return stats


def cell_measured_in_results(cell: str, stats: dict, aa_acceptable: bool) -> tuple[bool, str]:
    arms = stats.get(cell, {})
    if len(arms) < 2:
        return False, f"results have {len(arms)} arm(s) for {cell}, need >= 2"
    reasons: list[str] = []
    for role in sorted(arms):
        st = arms[role]
        if len(st["runs"]) < MIN_RUNS:
            reasons.append(f"arm {role}: {len(st['runs'])} valid runs (need >= {MIN_RUNS})")
        if len(st["shas"]) != 1:
            reasons.append(f"arm {role}: mixes binaries {sorted(s[:12] for s in st['shas'])}")
        if len(st["fps"]) != 1:
            reasons.append(f"arm {role}: mixes settings fingerprints")
        if st["bad_threads"]:
            reasons.append(f"arm {role}: {st['bad_threads']} run(s) with "
                           f"threads_effective != nominal T of {cell}")
    all_shas = set().union(*(st["shas"] for st in arms.values()))
    if not aa_acceptable and len(all_shas) < 2:
        reasons.append("both arms ran the SAME binary (A/A rows are not A/B evidence; "
                       "set \"aa_acceptable\": true only for explicitly-A/A cells)")
    if reasons:
        return False, "; ".join(reasons)
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(FLEET_DIR / "matrix.json"))
    parser.add_argument("--results", help="results JSONL file(s), comma-separated")
    parser.add_argument("--dispositions", default=str(FLEET_DIR / "dispositions.json"))
    args = parser.parse_args()

    matrix = json.loads(pathlib.Path(args.matrix).read_text())
    universe = [entry["cell"] for entry in matrix["universe"]["cells"]]
    universe_set = set(universe)

    dispositions: dict[str, dict] = {}
    dpath = pathlib.Path(args.dispositions)
    if dpath.exists():
        dispositions = json.loads(dpath.read_text())
    else:
        print(f"NOTE: no dispositions file at {dpath}; every universe cell is undispositioned")

    for cell in sorted(set(dispositions) - universe_set):
        print(f"WARNING: disposition for non-universe cell ignored: {cell}")

    stats = collect_arm_stats(load_results_rows(args.results))

    counts = {d: 0 for d in DISPOSITIONS}
    undispositioned = 0
    issues: list[str] = []
    for cell in universe:
        entry = dispositions.get(cell)
        if entry is None:
            undispositioned += 1
            continue
        disp = entry.get("disposition")
        if disp not in DISPOSITIONS:
            issues.append(f"{cell}: unknown disposition {disp!r}")
            undispositioned += 1
            continue
        aa_ok = bool(entry.get("aa_acceptable"))
        if disp == "MEASURED":
            ok, why = cell_measured_in_results(cell, stats, aa_ok)
            if not ok:
                issues.append(f"{cell}: MEASURED unsupported -- {why}")
                undispositioned += 1
                continue
        elif disp == "INFERRED":
            src = entry.get("from")
            rule = entry.get("rule")
            if not src or not rule:
                issues.append(f"{cell}: INFERRED requires 'from' and 'rule'")
                undispositioned += 1
                continue
            ok, why = cell_measured_in_results(src, stats, aa_ok)
            if not ok:
                issues.append(f"{cell}: INFERRED from-cell not measured -- {why}")
                undispositioned += 1
                continue
        else:
            if not entry.get("evidence"):
                issues.append(f"{cell}: {disp} requires non-empty 'evidence'")
                undispositioned += 1
                continue
        counts[disp] += 1

    print("disposition counts: " + " ".join(f"{d}={counts[d]}" for d in DISPOSITIONS)
          + f" UNDISPOSITIONED={undispositioned}")
    for issue in issues:
        print(f"ISSUE: {issue}")
    print(f"{undispositioned} undispositioned")
    return 0 if undispositioned == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
