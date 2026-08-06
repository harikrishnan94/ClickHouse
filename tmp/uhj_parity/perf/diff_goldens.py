#!/usr/bin/env python3
"""Re-run Stage 0 golden queries against a later UHJ binary.

The shipping settings are assembled explicitly instead of using the old arm helper. This is
intentional: Stage 1 removes the probe-batch setting, while these goldens must remain usable
after that removal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import harness as H
import verify_fused_output as V

DEFAULT_GOLDENS = Path(H.REPO_ROOT) / "tmp" / "uhj_parity" / "stage0" / "goldens.jsonl"


def _parse_checksum(raw: str) -> tuple[int, int]:
    fields = raw.strip().split()
    if len(fields) != 2:
        raise ValueError(f"expected two checksum fields, got {raw!r}")
    try:
        return int(fields[0]), int(fields[1])
    except ValueError as exc:
        raise ValueError(f"non-integer checksum result: {raw!r}") from exc


def _shipping_settings(record: dict) -> dict:
    settings = dict(H.PINNED_SETTINGS)
    settings["join_algorithm"] = "unified_hash"
    settings["max_threads"] = int(record.get("threads", 1))
    settings.update(record.get("extra_settings") or {})
    return settings


def _load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as golden_file:
        for line_number, line in enumerate(golden_file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            for field in ("cell_id", "group", "kind", "sql", "arm_fused", "arm_split"):
                if field not in record:
                    raise ValueError(f"{path}:{line_number}: missing {field!r}")
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("goldens", nargs="?", type=Path, default=DEFAULT_GOLDENS)
    parser.add_argument("--filter", default="", help="only check cell IDs containing this text")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    try:
        records = _load_records(args.goldens)
    except (OSError, ValueError) as exc:
        print(f"GOLDEN_LOAD_FAILURE: {exc}", file=sys.stderr)
        return 1

    records = [record for record in records if args.filter in record["cell_id"]]
    if args.limit:
        records = records[:args.limit]
    if not records:
        print("GOLDEN_LOAD_FAILURE: no records selected", file=sys.stderr)
        return 1

    V.ensure_special_tables()
    mismatches = []
    for index, record in enumerate(records, 1):
        expected = tuple(record["arm_fused"][field] for field in ("cnt", "chk"))
        split_expected = tuple(record["arm_split"][field] for field in ("cnt", "chk"))
        if expected != split_expected:
            mismatches.append(
                f"{record['cell_id']}: stored arms disagree "
                f"fused={expected} split={split_expected}"
            )
            continue

        try:
            current = _parse_checksum(
                H.run_query(record["sql"], _shipping_settings(record))
            )
        except (H.QueryError, ValueError) as exc:
            mismatches.append(f"{record['cell_id']}: query failed: {exc}")
            continue

        if current != expected:
            mismatches.append(
                f"{record['cell_id']}: expected={expected} current={current}"
            )
        if index % 25 == 0 or index == len(records):
            print(f"checked {index}/{len(records)} cells", flush=True)

    if mismatches:
        for mismatch in mismatches[:20]:
            print(f"MISMATCH {mismatch}", file=sys.stderr)
        print(f"GOLDENS_MISMATCH count={len(mismatches)}")
        return 1

    print(f"GOLDENS_MATCH cells={len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
