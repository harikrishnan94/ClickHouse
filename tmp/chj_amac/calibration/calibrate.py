#!/usr/bin/env python3
"""Empirical calibration driver for the parallel_hash size->rows ladder.

For each (family, D) measurement point:
  - CREATE a Memory build table with D deterministic distinct-ish keys
  - run a count()-only parallel_hash probe (1000 probe rows) with a log_comment tag
  - parse the per-shard "Join data is built" trace lines from the server log:
      the data-carrying slot's byte count == HashJoin::getTotalByteCount of the
      merged two-level map (map buffers + pool + stored blocks; stored is ~0 for
      count()-only probes), which we take as the aggregate hash-map bytes
  - read memory_usage and ConcurrentHashJoinBuildMicroseconds from system.query_log
  - read the build table's own total_bytes from system.tables
  - DROP the table

Appends one JSON object per point to measurements.jsonl.
"""

import json
import re
import subprocess
import sys
import time

BIN = "/mnt/ch/ClickHouse/tmp/chj_amac/bins/clickhouse-baseline-a05f3ee81ff.bin"
HOST, PORT = "127.0.0.1", "19710"
LOG = "/mnt/ch/ClickHouse/tmp/chj_amac/calibration/srv/server.log"
OUT = "/mnt/ch/ClickHouse/tmp/chj_amac/calibration/measurements.jsonl"

SETTINGS = (
    "join_algorithm='parallel_hash', max_threads=32, "
    "collect_hash_table_stats_during_joins=0, max_bytes_before_external_join=0, "
    "max_bytes_ratio_before_external_join=0, parallel_hash_join_threshold=1"
)

# family -> (columns DDL, build select exprs, probe select exprs, join ON condition, expected map rows fn)
FAMILIES = {
    "key32": (
        "k UInt32",
        "toUInt32(number) AS k",
        "toUInt32(number) AS k",
        "t.k = b.k",
        lambda d: d,
    ),
    "key64": (
        "k UInt64",
        "number AS k",
        "number AS k",
        "t.k = b.k",
        lambda d: d,
    ),
    "str": (
        "k String",
        "concat(lpad(hex(number), 16, '0'), lpad(toString(number % 100000000), 8, '0')) AS k",
        "concat(lpad(hex(number), 16, '0'), lpad(toString(number % 100000000), 8, '0')) AS k",
        "t.k = b.k",
        lambda d: d,
    ),
    "strzero": (
        "k String",
        "concat(unhex('00'), lpad(hex(number), 16, '0'), lpad(toString(number % 10000000), 7, '0')) AS k",
        "concat(unhex('00'), lpad(hex(number), 16, '0'), lpad(toString(number % 10000000), 7, '0')) AS k",
        "t.k = b.k",
        lambda d: d,
    ),
    "fixstr": (
        "k FixedString(16)",
        "toFixedString(lpad(hex(number), 16, '0'), 16) AS k",
        "toFixedString(lpad(hex(number), 16, '0'), 16) AS k",
        "t.k = b.k",
        lambda d: d,
    ),
    "k128": (
        "k1 UInt64, k2 UInt64",
        "number AS k1, bitXor(number, 12345) AS k2",
        "number AS k1, bitXor(number, 12345) AS k2",
        "t.k1 = b.k1 AND t.k2 = b.k2",
        lambda d: d,
    ),
    "k256": (
        "k1 UInt64, k2 UInt64, k3 UInt64, k4 UInt64",
        "number AS k1, number + 1 AS k2, number + 2 AS k3, number + 3 AS k4",
        "number AS k1, number + 1 AS k2, number + 2 AS k3, number + 3 AS k4",
        "t.k1 = b.k1 AND t.k2 = b.k2 AND t.k3 = b.k3 AND t.k4 = b.k4",
        lambda d: d,
    ),
    "null64": (
        "k Nullable(UInt64)",
        "if(number % 10 = 0, NULL, number) AS k",
        "toNullable(number) AS k",
        "t.k = b.k",
        lambda d: d - (d + 9) // 10,  # NULL keys are not inserted into the map
    ),
    "lcstr": (
        "k LowCardinality(String)",
        "toLowCardinality(concat('s', lpad(toString(number % 100000), 6, '0'))) AS k",
        "concat('s', lpad(toString(number % 100000), 6, '0')) AS k",
        "t.k = b.k",
        lambda d: d,  # rows, not distinct keys: MapsAll counts total rows
    ),
    "mixed": (
        "k1 UInt32, k2 String",
        "toUInt32(number) AS k1, concat(lpad(hex(number), 16, '0'), lpad(toString(number % 100000000), 8, '0')) AS k2",
        "toUInt32(number) AS k1, concat(lpad(hex(number), 16, '0'), lpad(toString(number % 100000000), 8, '0')) AS k2",
        "t.k1 = b.k1 AND t.k2 = b.k2",
        lambda d: d,
    ),
}

UNIT = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
BUILT_RE = re.compile(r"Join data is built, ([\d.]+) (\w+) and (\d+) rows")


def client(query, timeout=600):
    p = subprocess.run(
        [BIN, "client", "--host", HOST, "--port", PORT, "--query", query],
        capture_output=True, text=True, timeout=timeout,
    )
    if p.returncode != 0:
        raise RuntimeError(f"query failed ({p.returncode}): {p.stderr.strip()}\n{query[:300]}")
    return p.stdout.strip()


def measure(family, d, tag_extra=""):
    cols, build_expr, probe_expr, cond, exp_rows = FAMILIES[family]
    t0 = time.time()
    client("DROP TABLE IF EXISTS calib_build")
    client(
        f"CREATE TABLE calib_build ({cols}) ENGINE = Memory AS "
        f"SELECT {build_expr} FROM numbers_mt({d}) "
        f"SETTINGS max_threads = 32, max_insert_threads = 32",
        timeout=900,
    )
    insert_s = time.time() - t0
    table_bytes = int(client(
        "SELECT total_bytes FROM system.tables WHERE database = currentDatabase() AND name = 'calib_build'"
    ))

    with open(LOG, "rb") as f:
        f.seek(0, 2)
        mark = f.tell()

    tag = f"calib:{family}:{d}{tag_extra}"
    t1 = time.time()
    cnt = client(
        f"SELECT count() FROM (SELECT {probe_expr} FROM numbers(1000)) t "
        f"INNER JOIN calib_build b ON {cond} "
        f"SETTINGS {SETTINGS}, log_comment = '{tag}'",
        timeout=900,
    )
    join_s = time.time() - t1

    time.sleep(0.3)
    with open(LOG, "rb") as f:
        f.seek(mark)
        window = f.read().decode("utf-8", errors="replace")

    built = []
    for m in BUILT_RE.finditer(window):
        built.append((int(float(m.group(1)) * UNIT[m.group(2)]), int(m.group(3))))
    data_slots = [(b, r) for b, r in built if r > 0]
    map_bytes = sum(b for b, r in data_slots)
    map_rows = sum(r for b, r in data_slots)
    skeleton_bytes = sum(b for b, r in built if r == 0)

    client("SYSTEM FLUSH LOGS")
    row = client(
        "SELECT memory_usage, ProfileEvents['ConcurrentHashJoinBuildMicroseconds'] "
        f"FROM system.query_log WHERE log_comment = '{tag}' AND type = 'QueryFinish' "
        "ORDER BY event_time_microseconds DESC LIMIT 1"
    ).split("\t")
    peak_mem, build_us = int(row[0]), int(row[1])
    client("DROP TABLE IF EXISTS calib_build")

    rec = {
        "family": family, "D": d, "tag": tag,
        "map_bytes": map_bytes, "map_rows": map_rows,
        "expected_rows": exp_rows(d),
        "built_lines": len(built), "data_slots": len(data_slots),
        "skeleton_bytes": skeleton_bytes,
        "peak_memory": peak_mem, "table_bytes": table_bytes,
        "build_us": build_us, "probe_count": int(cnt),
        "insert_s": round(insert_s, 1), "join_s": round(join_s, 1),
    }
    ok_rows = rec["map_rows"] == rec["expected_rows"]
    ok_path = build_us > 0
    rec["ok"] = ok_rows and ok_path and len(built) == 32
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(
        f"{family:8s} D={d:<12,d} map={map_bytes/1024**2:10.2f} MiB "
        f"rows={map_rows:>12,d} (exp {rec['expected_rows']:,d}) peak={peak_mem/1024**2:9.1f} MiB "
        f"tbl={table_bytes/1024**2:8.1f} MiB build_us={build_us} ok={rec['ok']} "
        f"[ins {insert_s:.1f}s join {join_s:.1f}s]",
        flush=True,
    )
    return rec


if __name__ == "__main__":
    points = []
    for arg in sys.argv[1:]:
        fam, d = arg.split("=")
        points.append((fam, int(d)))
    for fam, d in points:
        try:
            measure(fam, d)
        except Exception as e:
            print(f"FAILED {fam} D={d}: {e}", flush=True)
