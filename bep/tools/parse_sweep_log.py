#!/usr/bin/env python3
"""Combine `join_mergetree_bench.py run` logs into one CSV results table.

The benchmark tool has no structured-output mode; it prints a human-readable
table per point. This script parses that printed format from one or more log
files (one per `--threads` value, matched from the filename `..._<N>.log`)
and writes a single combined CSV with one row per (point, algorithm).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass


LABEL_RE = re.compile(
    r"^Point: D=(?P<d>\d+) m=(?P<m>\d+) ratio=(?P<ratio>[\d.]+) "
    r"hit=(?P<hit>[\d.]+) bp=(?P<bp>\d+) pp=(?P<pp>\d+)"
    r"(?: N_p=(?P<n_p>\d+) n_hit=(?P<n_hit>\d+))?\s*$"
)
INVALID_LABEL_RE = re.compile(
    r"^Point: D=(?P<d>\d+) m=(?P<m>\d+) ratio=(?P<ratio>[\d.]+) "
    r"hit=(?P<hit>[\d.]+) bp=(?P<bp>\d+) pp=(?P<pp>\d+)\s*$"
)
INVALID_RE = re.compile(r"^INVALID: (?P<detail>.*)$")
VERIFICATION_RE = re.compile(r"^Verification: (?P<status>\S+) \((?P<detail>.*)\)$")
WINNER_RE = re.compile(
    r"^Winner: (?P<winner>excluded|tie|partitioned_hash|parallel_hash)"
    r"(?: \((?P<speedup>[\d.]+)x\))?$"
)
THREADS_FROM_NAME_RE = re.compile(r"threads_(\d+)\.log$")

ROW_HEADERS = (
    "algorithm",
    "status",
    "verify",
    "median_ms",
    "min_ms",
    "build_ms",
    "probe_ms",
    "partitions",
    "leaf_rows",
    "chj_build_ms",
    "selected_rows",
    "selected_bytes",
    "peak_mem_mb",
)
ROW_HEADERS_BY_LENGTH = {
    len(ROW_HEADERS): ROW_HEADERS,
}

OUTPUT_COLUMNS = (
    "threads",
    "D",
    "ratio",
    "bp",
    "pp",
    "point_status",
    "verification",
    "winner",
    "speedup",
    "partitioned_status",
    "partitioned_median_ms",
    "partitioned_peak_mem_mb",
    "partitioned_build_ms",
    "partitioned_probe_ms",
    "parallel_status",
    "parallel_median_ms",
    "parallel_peak_mem_mb",
    "parallel_build_ms",
    "detail",
)


@dataclass
class PointResult:
    threads: int
    cardinality: int
    ratio: str
    bp: int
    pp: int
    point_status: str
    verification: str
    winner: str
    speedup: str
    partitioned_status: str
    partitioned_median_ms: str
    partitioned_peak_mem_mb: str
    partitioned_build_ms: str
    partitioned_probe_ms: str
    parallel_status: str
    parallel_median_ms: str
    parallel_peak_mem_mb: str
    parallel_build_ms: str
    detail: str


def threads_from_filename(path: str) -> int:
    match = THREADS_FROM_NAME_RE.search(path)
    if not match:
        raise ValueError(
            f"cannot infer --threads value from filename {path!r}; "
            "expected it to end in threads_<N>.log, or pass --threads explicitly"
        )
    return int(match.group(1))


def _split_table_row(line: str) -> list[str]:
    return [field for field in re.split(r"\s{2,}", line.strip()) if field != ""]


def iter_point_blocks(lines: list[str]) -> Iterator[list[str]]:
    block: list[str] | None = None
    for line in lines:
        if line.startswith("Point: "):
            if block is not None:
                yield block
            block = [line]
        elif block is not None:
            block.append(line)
    if block is not None:
        yield block


def parse_block(block: list[str], *, threads: int) -> PointResult | None:
    header = block[0]
    match = LABEL_RE.match(header) or INVALID_LABEL_RE.match(header)
    if not match:
        return None
    cardinality = int(match.group("d"))
    ratio = match.group("ratio")
    bp = int(match.group("bp"))
    pp = int(match.group("pp"))

    invalid_detail = None
    verification = ""
    winner = ""
    speedup = ""
    partitioned_status = ""
    partitioned_median = ""
    partitioned_peak_mem = ""
    partitioned_build = ""
    partitioned_probe = ""
    parallel_status = ""
    parallel_median = ""
    parallel_peak_mem = ""
    parallel_build = ""

    for line in block[1:]:
        invalid_match = INVALID_RE.match(line)
        if invalid_match:
            invalid_detail = invalid_match.group("detail")
            continue
        verification_match = VERIFICATION_RE.match(line)
        if verification_match:
            verification = (
                f"{verification_match.group('status')} "
                f"({verification_match.group('detail')})"
            )
            continue
        winner_match = WINNER_RE.match(line)
        if winner_match:
            winner = winner_match.group("winner")
            speedup = winner_match.group("speedup") or ""
            continue
        fields = _split_table_row(line)
        row_headers = ROW_HEADERS_BY_LENGTH.get(len(fields))
        if row_headers is None:
            continue
        row = dict(zip(row_headers, fields))
        if row["algorithm"] == "partitioned_hash":
            partitioned_status = row["status"]
            partitioned_median = row["median_ms"]
            partitioned_peak_mem = row.get("peak_mem_mb", "")
            partitioned_build = row.get("build_ms", "")
            partitioned_probe = row.get("probe_ms", "")
        elif row["algorithm"] == "parallel_hash":
            parallel_status = row["status"]
            parallel_median = row["median_ms"]
            parallel_peak_mem = row.get("peak_mem_mb", "")
            parallel_build = row.get("chj_build_ms", "")

    if invalid_detail is not None:
        point_status = "INVALID"
    elif not partitioned_status and not parallel_status:
        # An incomplete trailing block (log still being written mid-run).
        return None
    else:
        point_status = "OK"

    return PointResult(
        threads=threads,
        cardinality=cardinality,
        ratio=ratio,
        bp=bp,
        pp=pp,
        point_status=point_status,
        verification=verification,
        winner=winner,
        speedup=speedup,
        partitioned_status=partitioned_status,
        partitioned_median_ms=partitioned_median,
        partitioned_peak_mem_mb=partitioned_peak_mem,
        partitioned_build_ms=partitioned_build,
        partitioned_probe_ms=partitioned_probe,
        parallel_status=parallel_status,
        parallel_median_ms=parallel_median,
        parallel_peak_mem_mb=parallel_peak_mem,
        parallel_build_ms=parallel_build,
        detail=invalid_detail or "",
    )


def parse_log(path: str, *, threads: int | None) -> list[PointResult]:
    resolved_threads = threads if threads is not None else threads_from_filename(path)
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    results = []
    for block in iter_point_blocks(lines):
        result = parse_block(block, threads=resolved_threads)
        if result is not None:
            results.append(result)
    return results


def write_csv(results: list[PointResult], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)
        for result in results:
            writer.writerow(
                (
                    result.threads,
                    result.cardinality,
                    result.ratio,
                    result.bp,
                    result.pp,
                    result.point_status,
                    result.verification,
                    result.winner,
                    result.speedup,
                    result.partitioned_status,
                    result.partitioned_median_ms,
                    result.partitioned_peak_mem_mb,
                    result.partitioned_build_ms,
                    result.partitioned_probe_ms,
                    result.parallel_status,
                    result.parallel_median_ms,
                    result.parallel_peak_mem_mb,
                    result.parallel_build_ms,
                    result.detail,
                )
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="one or more sweep log files")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "override the --threads value for ALL given logs "
            "(default: parsed per-file from a trailing threads_<N>.log)"
        ),
    )
    parser.add_argument("--output", required=True, help="combined CSV output path")
    args = parser.parse_args(argv)

    all_results: list[PointResult] = []
    for log_path in args.logs:
        results = parse_log(log_path, threads=args.threads)
        print(f"{log_path}: parsed {len(results)} points", file=sys.stderr)
        all_results.extend(results)

    write_csv(all_results, args.output)
    print(f"wrote {len(all_results)} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
