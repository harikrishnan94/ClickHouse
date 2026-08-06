#!/usr/bin/env python3
"""Run the Stage 0 UHJ correctness gate and record durable goldens.

After Stage 1 the shipping path is a single configuration (constexpr batch, compile-time
fused/split). Re-recording still writes both `arm_fused` and `arm_split` fields so the
Stage 0 golden schema stays readable by `diff_goldens.py`; both fields hold the same
shipping-path checksum.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import harness as H

FUSED_KINDS = ["LEFT", "FULL"]                       # take the new path
CONTROL_KINDS = ["INNER", "RIGHT", "LEFT SEMI", "LEFT ANTI"]   # must not
SHIP_ARM = H.BASELINE_ARM

STAGE0_DIR = Path(H.REPO_ROOT) / "tmp" / "uhj_parity" / "stage0"
GOLDENS_PATH = STAGE0_DIR / "goldens.jsonl"
SUMMARY_PATH = STAGE0_DIR / "goldens_summary.txt"


@dataclass(frozen=True)
class Case:
    cell_id: str
    group: str
    kind: str
    sql: str
    threads: int
    extra_settings: dict


def cells(kinds):
    for kind in kinds:
        for key in H.KEY_TYPES:
            for match in H.MATCH_RATES:
                for threads, card in H.THREAD_CARDS:
                    yield H.Cell(kind, key, match, threads, card)


def _tsv(sql: str) -> str:
    return f"{sql} FORMAT TSV"


def matrix_cases():
    for kinds in (FUSED_KINDS, CONTROL_KINDS):
        for cell in cells(kinds):
            yield Case(
                cell_id=cell.cell_id,
                group="matrix",
                kind=cell.kind,
                sql=_tsv(H.join_sql(cell, "checksum")),
                threads=cell.threads,
                extra_settings={},
            )


def _special_cases():
    common = {
        "left": "uhj_stage0_filter_left",
        "right": "uhj_stage0_filter_right",
        "asof_left": "uhj_stage0_asof_left",
        "asof_right": "uhj_stage0_asof_right",
        "nullable_left": "uhj_stage0_nullable_left",
        "nullable_right": "uhj_stage0_nullable_right",
        "dup_left": "uhj_stage0_dup_left",
        "dup_right": "uhj_stage0_dup_right",
        "af_left": "uhj_stage0_af_left",
        "af_right": "uhj_stage0_af_right",
    }
    # Full-column checksums over the medium MergeTree tables. SEMI/ANTI deliberately hash
    # the RIGHT columns too (N21: SEMI LEFT appends right columns; ANTI emits defaults) -
    # the 144-cell matrix is blind to that output because Cell.has_right_cols is False.
    med_left, med_right = "p_medium_hi", "b_medium"
    med_agg = ("count() AS cnt, "
               "sum(cityHash64(l.k, l.s, l.a, l.b, r.k, r.s, r.a, r.b, r.v)) AS chk")
    # Same checksum shape for tables without the string column (nullable / dup / filter).
    tiny_agg = ("count() AS cnt, "
                "sum(cityHash64(l.k, l.a, l.b, l.v, r.k, r.a, r.b, r.v)) AS chk")
    # 2 disjuncts stay key64 (k and a are both 1:1 with k, so dedup bounds the output).
    # The 3rd disjunct uses the string column: b = k % 1000 fans out to ~1000 rows per
    # probe row at medium cardinality (1.8B-row result - measured), and no other UInt64
    # column is 1:1 with k. Mixed key kinds merge to the generic `hashed` map
    # (mergeJoinMethods), which is the multi-disjunct-with-string path worth covering.
    on2 = "l.k = r.k OR l.a = r.a"
    on3 = "l.k = r.k OR l.a = r.a OR l.s = r.s"
    # Multi-disjunct is only supported for INNER / LEFT / LEFT SEMI: RIGHT, FULL, ANY and
    # ANTI throw "Expected to have only one join clause" (assertHasOneOnExpr), and LEFT
    # multi on NULLABLE keys is rejected the same way (measured on the base binary).
    multi_kinds = ["INNER", "LEFT", "LEFT SEMI"]
    filter_checksum = (
        "SELECT count() AS cnt, "
        "sum(cityHash64(l.k, l.a, l.b, l.v, r.k, r.a, r.b, r.v)) AS chk "
        f"FROM {common['left']} AS l "
        "INNER JOIN "
        f"{common['right']} AS r "
        "ON l.k = r.k OR l.a = r.a"
    )
    add_filter_checksum = (
        "SELECT count() AS cnt, "
        "sum(cityHash64(l.k, l.a, l.b, l.v, r.k, r.a, r.b, r.v)) AS chk "
        f"FROM {common['left']} AS l "
        "INNER JOIN "
        f"{common['right']} AS r "
        "ON l.k = r.k AND l.a < r.a"
    )
    asof_checksum = (
        "SELECT count() AS cnt, "
        "sum(cityHash64(l.k, l.ts, l.v, r.k, r.ts, r.v)) AS chk "
        f"FROM {common['asof_left']} AS l "
        "ASOF INNER JOIN "
        f"{common['asof_right']} AS r "
        "ON l.k = r.k AND l.ts >= r.ts"
    )
    asof_left_checksum = (
        "SELECT count() AS cnt, "
        "sum(cityHash64(l.k, l.ts, l.v, r.k, r.ts, r.v)) AS chk "
        f"FROM {common['asof_left']} AS l "
        "ASOF LEFT JOIN "
        f"{common['asof_right']} AS r "
        "ON l.k = r.k AND l.ts >= r.ts"
    )
    any_checksum = (
        "SELECT count() AS cnt, "
        "sum(cityHash64(l.k, l.a, l.b, l.v, r.k, r.a, r.b, r.v)) AS chk "
        f"FROM {common['left']} AS l "
        "LEFT ANY JOIN "
        f"{common['right']} AS r "
        "ON l.k = r.k"
    )
    nullable_checksum = (
        "SELECT count() AS cnt, "
        "sum(cityHash64(l.k, l.a, l.b, l.v, r.k, r.a, r.b, r.v)) AS chk "
        "FROM uhj_stage0_nullable_left AS l "
        "LEFT JOIN uhj_stage0_nullable_right AS r "
        "ON l.k = r.k AND l.a < r.a"
    )
    cases = [
        Case("multi|filter|t1", "multi", "INNER", _tsv(filter_checksum), 1, {}),
        Case("addfilter|filter|t1", "addfilter", "INNER", _tsv(add_filter_checksum), 1, {}),
        Case("asof_inner|asof|t1", "asof_inner", "ASOF INNER", _tsv(asof_checksum), 1, {}),
        Case("asof_left|asof|t1", "asof_left", "ASOF LEFT", _tsv(asof_left_checksum), 1, {}),
        Case("any|filter|t1", "any", "LEFT ANY", _tsv(any_checksum), 1, {}),
        Case(
            "nullable_filter|nullable|t1",
            "nullable_filter",
            "LEFT",
            _tsv(nullable_checksum),
            1,
            {"join_use_nulls": 1},
        ),
    ]

    # --- Stage 0b additions -------------------------------------------------
    # Multi-disjunct on the medium MergeTree tables (2 and 3 disjuncts).
    for group, on in (("multi2", on2), ("multi3", on3)):
        for kind in multi_kinds:
            cases.append(Case(
                f"{group}|{kind}|t1|medium", group, kind,
                _tsv(f"SELECT {med_agg} FROM {med_left} AS l {kind} JOIN {med_right} AS r ON {on}"),
                1, {}))

    # SEMI LEFT / LEFT ANTI selecting RIGHT columns, single clause (the Stage 2 emit
    # changes exactly this; the matrix never hashes right columns for these kinds).
    for kind in ("LEFT SEMI", "LEFT ANTI"):
        cases.append(Case(
            f"semi_right|{kind}|t1|medium", "semi_right", kind,
            _tsv(f"SELECT {med_agg} FROM {med_left} AS l {kind} JOIN {med_right} AS r ON l.k = r.k"),
            1, {}))

    # Per-clause skip (N3): the k disjunct carries a null map, the a disjunct does not.
    # LEFT and RIGHT both throw assertHasOneOnExpr for multi-disjunct on nullable keys
    # (RIGHT only "worked" un-pinned because query_plan_join_swap_table rewrote it).
    for kind in ("INNER", "LEFT SEMI"):
        cases.append(Case(
            f"nullable_multi|{kind}|t1", "nullable_multi", kind,
            _tsv(f"SELECT {tiny_agg} FROM {common['nullable_left']} AS l {kind} JOIN "
                 f"{common['nullable_right']} AS r ON {on2}"),
            1, {}))

    # C13: high-duplication multi join + small max_joined_block_rows -> the probe stops
    # mid-block and joinBlockImpl splits. Deterministic (count, checksum) under splitting.
    cases.append(Case(
        "c13_split|INNER|t1", "c13_split", "INNER",
        _tsv(f"SELECT count() AS cnt, sum(cityHash64(l.k, l.a, l.b, l.v, r.k, r.a, r.b, r.v)) AS chk "
             f"FROM {common['dup_left']} AS l INNER JOIN {common['dup_right']} AS r ON {on2}"),
        1, {"max_joined_block_size_rows": 1024}))

    # Additional filter: flag_per_row true (RIGHT / FULL / multi) and false with a small
    # max_joined_block_rows (early exit + offsets/filter resize path), plus a medium-sized
    # INNER on the dedicated af tables where the filter is non-degenerate (~50% pass).
    cases.append(Case(
        "addfilter_right|RIGHT|t1", "addfilter_extra", "RIGHT",
        _tsv(f"SELECT {tiny_agg} FROM {common['left']} AS l RIGHT JOIN {common['right']} AS r "
             "ON l.k = r.k AND l.a < r.a"), 1, {}))
    cases.append(Case(
        "addfilter_full|FULL|t1", "addfilter_extra", "FULL",
        _tsv(f"SELECT {tiny_agg} FROM {common['left']} AS l FULL JOIN {common['right']} AS r "
             "ON l.k = r.k AND l.a < r.a"), 1, {}))
    cases.append(Case(
        "addfilter_multi|INNER|t1", "addfilter_extra", "INNER",
        _tsv(f"SELECT {tiny_agg} FROM {common['left']} AS l INNER JOIN {common['right']} AS r "
             "ON (l.k = r.k OR l.a = r.a) AND l.b < r.b"), 1, {}))
    cases.append(Case(
        "addfilter_limit8|INNER|t1", "addfilter_extra", "INNER",
        _tsv(f"SELECT {tiny_agg} FROM {common['left']} AS l INNER JOIN {common['right']} AS r "
             "ON l.k = r.k AND l.a < r.a"), 1, {"max_joined_block_size_rows": 8}))
    cases.append(Case(
        "addfilter_med|INNER|t1|medium", "addfilter_extra", "INNER",
        _tsv(f"SELECT count() AS cnt, sum(cityHash64(l.k, l.a, l.v, r.k, r.a, r.v)) AS chk "
             f"FROM {common['af_left']} AS l INNER JOIN {common['af_right']} AS r "
             "ON l.k = r.k AND l.a < r.a"), 1, {}))
    return cases


def _special_table_sql() -> list[str]:
    return [
        "DROP TABLE IF EXISTS uhj_stage0_asof_left",
        "DROP TABLE IF EXISTS uhj_stage0_asof_right",
        "DROP TABLE IF EXISTS uhj_stage0_filter_left",
        "DROP TABLE IF EXISTS uhj_stage0_filter_right",
        "DROP TABLE IF EXISTS uhj_stage0_nullable_left",
        "DROP TABLE IF EXISTS uhj_stage0_nullable_right",
        "DROP TABLE IF EXISTS uhj_stage0_dup_left",
        "DROP TABLE IF EXISTS uhj_stage0_dup_right",
        "DROP TABLE IF EXISTS uhj_stage0_af_left",
        "DROP TABLE IF EXISTS uhj_stage0_af_right",
        "CREATE TABLE uhj_stage0_asof_left (k UInt64, ts UInt64, v UInt64) ENGINE=Memory",
        "CREATE TABLE uhj_stage0_asof_right (k UInt64, ts UInt64, v UInt64) ENGINE=Memory",
        "CREATE TABLE uhj_stage0_filter_left (k UInt64, a UInt64, b UInt64, v UInt64) ENGINE=Memory",
        "CREATE TABLE uhj_stage0_filter_right (k UInt64, a UInt64, b UInt64, v UInt64) ENGINE=Memory",
        "CREATE TABLE uhj_stage0_nullable_left (k Nullable(UInt64), a UInt64, b UInt64, v UInt64) ENGINE=Memory",
        "CREATE TABLE uhj_stage0_nullable_right (k Nullable(UInt64), a UInt64, b UInt64, v UInt64) ENGINE=Memory",
        "CREATE TABLE uhj_stage0_dup_left (k UInt64, a UInt64, b UInt64, v UInt64) ENGINE=Memory",
        "CREATE TABLE uhj_stage0_dup_right (k UInt64, a UInt64, b UInt64, v UInt64) ENGINE=Memory",
        "CREATE TABLE uhj_stage0_af_left (k UInt64, a UInt64, v UInt64) ENGINE=Memory",
        "CREATE TABLE uhj_stage0_af_right (k UInt64, a UInt64, v UInt64) ENGINE=Memory",
        (
            "INSERT INTO uhj_stage0_asof_left VALUES "
            "(1, 10, 100), (1, 20, 200), (2, 15, 300), (3, 5, 400)"
        ),
        (
            "INSERT INTO uhj_stage0_asof_right VALUES "
            "(1, 4, 11), (1, 12, 12), (1, 22, 13), "
            "(2, 10, 21), (2, 20, 22), (4, 1, 41)"
        ),
        (
            "INSERT INTO uhj_stage0_filter_left VALUES "
            "(1, 10, 100, 1000), (1, 20, 200, 2000), "
            "(2, 5, 50, 3000), (3, 1, 30, 4000)"
        ),
        (
            "INSERT INTO uhj_stage0_filter_right VALUES "
            "(1, 5, 101, 11), (1, 15, 201, 12), (1, 25, 99, 13), "
            "(2, 6, 51, 21), (2, 4, 49, 22), (4, 1, 31, 41)"
        ),
        (
            "INSERT INTO uhj_stage0_nullable_left VALUES "
            "(1, 10, 100, 1000), (NULL, 20, 200, 2000), "
            "(2, 5, 50, 3000), (3, 1, 30, 4000)"
        ),
        (
            "INSERT INTO uhj_stage0_nullable_right VALUES "
            "(1, 5, 101, 11), (NULL, 15, 201, 12), "
            "(2, 6, 51, 21), (2, 4, 49, 22), (4, 1, 31, 41)"
        ),
        # C13 split cell: 8 keys x 250 duplicate rows on the right; each left row matches
        # 250 rows per disjunct (k == a on both sides, so the two disjuncts dedup to the
        # same 250). With max_joined_block_size_rows=1024 the probe stops mid-block.
        (
            "INSERT INTO uhj_stage0_dup_left "
            "SELECT number % 8 AS k, number % 8 AS a, number AS b, number * 3 AS v "
            "FROM numbers(2048)"
        ),
        (
            "INSERT INTO uhj_stage0_dup_right "
            "SELECT intDiv(number, 250) AS k, intDiv(number, 250) AS a, number AS b, number * 5 AS v "
            "FROM numbers(2000)"
        ),
        # Real-sized additional-filter tables: k 1:1 unique on the right, `a` independent
        # of k on both sides so `l.a < r.a` passes ~50% of matched pairs (on b_medium/
        # p_medium every column is a function of k, so any l-r comparison is degenerate).
        (
            "INSERT INTO uhj_stage0_af_right "
            "SELECT number AS k, (number * 2654435761) % 100003 AS a, number * 5 AS v "
            "FROM numbers(100000)"
        ),
        (
            "INSERT INTO uhj_stage0_af_left "
            "SELECT if(number % 10 < 9, (number * 6364136223846793005) % 100000, "
            "100000 + number) AS k, (number * 2246822519) % 100003 AS a, number * 7 AS v "
            "FROM numbers(200000)"
        ),
    ]


def ensure_special_tables(http_ports: list[int] | None = None) -> None:
    """Create Stage 0 Memory tables. Default: harness.HTTP_PORT only.

    Stage 6 A/B passes every arm's HTTP port so multi/addfilter/ASOF cells exist on both servers.
    """
    ports = [H.HTTP_PORT] if http_ports is None else list(http_ports)
    for port in ports:
        for sql in _special_table_sql():
            H.run_query(sql, http_port=port)


def _parse_checksum(raw: str) -> tuple[int, int]:
    fields = raw.strip().split()
    if len(fields) != 2:
        raise ValueError(f"expected two checksum fields, got {raw!r}")
    try:
        return int(fields[0]), int(fields[1])
    except ValueError as exc:
        raise ValueError(f"non-integer checksum result: {raw!r}") from exc


def _run_case(case: Case) -> dict:
    raw = H.run_query(
        case.sql,
        H.settings_for(case, SHIP_ARM, extra=case.extra_settings),
    )
    count, checksum = _parse_checksum(raw)
    values = {"cnt": count, "chk": checksum}
    # Schema keeps both names from Stage 0; both are the shipping path after Stage 1.
    return {"arm_fused": values, "arm_split": dict(values)}


def _record(case: Case, values: dict) -> dict:
    return {
        "cell_id": case.cell_id,
        "group": case.group,
        "kind": case.kind,
        "threads": case.threads,
        "extra_settings": case.extra_settings,
        "sql": case.sql,
        "arm_fused": values["arm_fused"],
        "arm_split": values["arm_split"],
    }


def _write_summary(group_counts: dict[str, int], total: int, elapsed: float) -> None:
    lines = [
        "Stage 0 UHJ checksum goldens",
        f"total_cells={total}",
        *(f"{group}={count}" for group, count in sorted(group_counts.items())),
        "arms=uhj_ship (arm_fused=arm_split=shipping)",
        "arm_agreement=n/a (single shipping path)",
        f"elapsed_seconds={elapsed:.1f}",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    started = time.time()
    temporary_path = GOLDENS_PATH.with_suffix(".jsonl.tmp")
    group_counts: dict[str, int] = {}
    total = 0
    try:
        STAGE0_DIR.mkdir(parents=True, exist_ok=True)
        ensure_special_tables()
        all_cases = [*matrix_cases(), *_special_cases()]
        matrix_count = sum(case.group == "matrix" for case in all_cases)
        if matrix_count != 144:
            raise AssertionError(f"expected the existing 144-cell matrix, got {matrix_count}")
        required_groups = {
            "multi",
            "addfilter",
            "asof_inner",
            "asof_left",
            "any",
            "nullable_filter",
            # Stage 0b additions
            "multi2",
            "multi3",
            "semi_right",
            "nullable_multi",
            "c13_split",
            "addfilter_extra",
        }
        actual_groups = {case.group for case in all_cases if case.group != "matrix"}
        if actual_groups != required_groups:
            raise AssertionError(
                f"unexpected Stage 0 special groups: {sorted(actual_groups)}"
            )

        with temporary_path.open("w", encoding="utf-8") as golden_file:
            for case in all_cases:
                values = _run_case(case)
                golden_file.write(json.dumps(_record(case, values), sort_keys=True) + "\n")
                golden_file.flush()
                total += 1
                group_counts[case.group] = group_counts.get(case.group, 0) + 1
                if total % 25 == 0 or total == len(all_cases):
                    print(f"recorded {total}/{len(all_cases)} cells", flush=True)

        os.replace(temporary_path, GOLDENS_PATH)
        _write_summary(group_counts, total, time.time() - started)
        print(f"\nGOLDENS_PATH={GOLDENS_PATH}")
        print(f"SUMMARY_PATH={SUMMARY_PATH}")
        print("STAGE0_GOLDENS_RECORDED")
        print("STAGE0B_GOLDENS_RECORDED")
        print("FUSED_OUTPUT_CORRECT")
        return 0
    except (AssertionError, H.QueryError, ValueError) as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        print(f"STAGE0_FAILURE: {exc}", file=sys.stderr)
        print("FUSED_OUTPUT_WRONG")
        return 1


if __name__ == "__main__":
    sys.exit(main())
