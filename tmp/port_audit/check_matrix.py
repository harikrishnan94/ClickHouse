#!/usr/bin/env python3
"""G0 gate: verify PORT_MATRIX.md covers both pinned commit ranges exactly.

Reads the pinned SHAs from the matrix header (so it keeps working even if the
branches move), then checks:
  1. every commit in git rev-list AHJ..RBM and AHJ..PHJ5 appears in exactly one
     matrix row's Commits cell;
  2. every commit listed in a row belongs to one of the two pinned ranges;
  3. every row has a valid disposition and non-empty mechanism and evidence;
  4. row IDs are unique.
Exit 0 iff all hold; else exit 1 and print the violations (orphans first).

Commits cell syntax: comma/space-separated hex prefixes (>= 8 chars), and/or
`@file:<path>` which expands to one full hash per line from <path> (relative to
the repo root).
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATRIX = Path(__file__).resolve().parent / "PORT_MATRIX.md"

VALID_DISPOSITIONS = {
    # Phase A
    "already-present", "port-candidate", "not-applicable", "process-artifact",
    # Phase B lifecycle
    "approved", "deferred-by-requester", "ported", "rejected-by-measurement",
}


def rev_list(a: str, b: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "rev-list", f"{a}..{b}"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    return out


def main() -> int:
    if not MATRIX.exists():
        print(f"FAIL: {MATRIX} does not exist")
        return 1
    text = MATRIX.read_text()

    shas = {}
    for name in ("AHJ_SHA", "RBM_SHA", "PHJ5_SHA"):
        m = re.search(rf"{name}\s*[=:]\s*`?([0-9a-f]{{40}})`?", text)
        if not m:
            print(f"FAIL: pinned {name} not found in matrix header")
            return 1
        shas[name] = m.group(1)

    range_rbm = rev_list(shas["AHJ_SHA"], shas["RBM_SHA"])
    range_phj5 = rev_list(shas["AHJ_SHA"], shas["PHJ5_SHA"])
    universe = {h: None for h in range_rbm + range_phj5}  # hash -> row id

    errors = []

    # Parse table rows: | ID | Source | Commits | Mechanism | Disposition | Evidence |
    rows = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            continue
        if cells[0] in ("ID", "---", "") or set(cells[0]) <= {"-", ":"}:
            continue
        rows.append(cells)

    if not rows:
        print("FAIL: no data rows parsed from the matrix table")
        return 1

    seen_ids = set()
    for cells in rows:
        row_id, source, commits_cell, mechanism, disposition, evidence = cells
        if row_id in seen_ids:
            errors.append(f"duplicate row id: {row_id}")
        seen_ids.add(row_id)

        if disposition not in VALID_DISPOSITIONS:
            errors.append(f"row {row_id}: invalid disposition '{disposition}'")
        if not mechanism.strip():
            errors.append(f"row {row_id}: empty mechanism")
        if not evidence.strip():
            errors.append(f"row {row_id}: empty evidence")

        if commits_cell.strip() == "diff-derived":
            # Mechanism surfaced only by the full-tree diff cross-check; its
            # commits are legitimately owned by another row. Coverage checks
            # do not apply; disposition/evidence checks above still do.
            continue

        tokens: list[str] = []
        for filem in re.finditer(r"@file:(\S+)", commits_cell):
            fpath = REPO / filem.group(1)
            if not fpath.exists():
                errors.append(f"row {row_id}: @file {fpath} missing")
                continue
            tokens += fpath.read_text().split()
        plain = re.sub(r"@file:\S+", " ", commits_cell)
        tokens += re.findall(r"\b[0-9a-f]{8,40}\b", plain)

        if not tokens:
            errors.append(f"row {row_id}: no commits listed")

        for tok in tokens:
            matches = [h for h in universe if h.startswith(tok)]
            if not matches:
                errors.append(
                    f"row {row_id}: commit {tok} not in either pinned range")
            elif len(matches) > 1:
                errors.append(f"row {row_id}: ambiguous prefix {tok}")
            else:
                h = matches[0]
                if universe[h] is not None and universe[h] != row_id:
                    errors.append(
                        f"commit {h[:12]} in two rows: {universe[h]} and {row_id}")
                universe[h] = row_id

    orphans = [h for h, owner in universe.items() if owner is None]
    for h in orphans:
        subj = subprocess.run(
            ["git", "-C", str(REPO), "log", "-1", "--format=%s", h],
            capture_output=True, text=True).stdout.strip()
        errors.insert(0, f"ORPHAN commit {h[:12]} ({subj}) in no row")

    if errors:
        print(f"FAIL: {len(errors)} violation(s); "
              f"ranges: rbm={len(range_rbm)} phj5={len(range_phj5)}, rows={len(rows)}")
        for e in errors:
            print("  " + e)
        return 1

    print(f"OK: {len(range_rbm)} rbm + {len(range_phj5)} phj5 commits covered by "
          f"{len(rows)} rows, all dispositions valid, all evidence non-empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
