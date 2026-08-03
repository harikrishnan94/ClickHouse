#!/usr/bin/env python3
"""Shared harness for the `unified_hash` perf-attribution mission.

Holds the three things every other script needs to agree on: how to talk to the
measurement server, the declared benchmark matrix, and the SQL for a cell.

Nothing here measures anything; `sweep.py` measures and `gates.py` judges.
Keeping the matrix in one place is what lets Gate G0.6 assert coverage against a
single source of truth rather than against whatever happened to get run.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field

# --------------------------------------------------------------------------
# Server / client
# --------------------------------------------------------------------------

HTTP_HOST = os.environ.get("UHJ_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("UHJ_HTTP_PORT", "8121"))
BASE_URL = f"http://{HTTP_HOST}:{HTTP_PORT}/"

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PERF_DIR, "..", "..", ".."))

# HTTP rather than spawning `clickhouse client` per query: a client process costs
# tens of milliseconds to start, which is the same order as the effects being
# measured. HTTP also lets us set query_id per query, which the trace_log-based
# algorithm assertion (Gate G0.1) needs.


class QueryError(RuntimeError):
    pass


def run_query(sql: str, settings: dict | None = None, query_id: str | None = None,
              timeout: int = 3600) -> str:
    params = dict(settings or {})
    if query_id:
        params["query_id"] = query_id
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, data=sql.encode("utf-8"), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise QueryError(f"HTTP {exc.code} for query_id={query_id}: {body[:2000]}") from None


def flush_logs() -> None:
    run_query("SYSTEM FLUSH LOGS")


def scalar(sql: str, settings: dict | None = None) -> str:
    return run_query(sql, settings).strip()


# --------------------------------------------------------------------------
# Pinned settings — recorded in every run, per the mission's comparison rules
# --------------------------------------------------------------------------

# `join_algorithm` is deliberately set to a SINGLE algorithm, never a list.
# TableJoin.cpp:1293-1307 `allowParallelHashJoin` returns false unless
# PARALLEL_HASH is in the list, which is what stops `join_algorithm='hash'` from
# being silently upgraded to ConcurrentHashJoin -- PlannerJoins.cpp:1246 would
# otherwise do exactly that whenever the RHS size estimate is unavailable.
PINNED_SETTINGS = {
    "enable_join_runtime_filters": 0,
    "max_bytes_before_external_join": 0,
    "max_block_size": 65409,
    "max_joined_block_size_rows": 65409,
    "query_plan_join_swap_table": 0,
    "parallel_hash_join_threshold": 0,
    "parallel_non_joined_rows_processing": 1,
    "log_processors_profiles": 1,
    "max_memory_usage": 0,
    "max_execution_time": 0,
}

ALGOS = ("hash", "parallel_hash", "unified_hash")

# --------------------------------------------------------------------------
# Data sizing
# --------------------------------------------------------------------------

# Cardinality == build-table row count, with UNIQUE keys. Rationale recorded in
# PREREG P0: it makes "distinct build-side keys" and "hash table size" the same
# number and keeps output size linear in the probe side. The cost is that
# duplicate-key RowRefList chains are not exercised; that is a declared coverage
# gap, not an oversight.
# DISTINCT build-side keys. Build tables hold ROWS_PER_KEY rows per key, so build
# rows = card * ROWS_PER_KEY. The mission's own wording separates these two
# quantities ("cardinalities (distinct build-side keys) ... capped at 1M build
# rows"), and they must stay separate here for the reason in WORKLOG E5.2.
CARDS = {
    "small": 10_000,        # fits L2 (Neoverse-V2: 2 MiB/core)
    "medium": 500_000,      # 1M build rows, the mission's 1-thread cap
    "large": 15_000_000,    # 30M build rows, well under the 100M cap
}

# Two rows per key, not one. With one row per key every implementation logs
# "Promoting join strictness to RightAny, because all values in the right table
# are unique" and the whole matrix silently measures MapsOne instead of the
# MapsAll/RowRefList path that INNER/LEFT/RIGHT/FULL JOIN actually mean. That
# would also have made the per-bucket-arena candidate unfalsifiable, since
# nothing would ever chain. See WORKLOG E5.2.
ROWS_PER_KEY = 2

PROBE_ROWS = {
    1: 2_000_000,
    16: 30_000_000,
    64: 30_000_000,
}

MATCH_RATES = {"hi": 0.9, "lo": 0.1}

# (thread count, cardinality) pairs. Small/medium at 1 thread, large at 16/64,
# exactly as the mission's matrix requirement states, plus the `dense` family
# which is measured at every thread count because the conversion asymmetry it
# probes only exists against `parallel_hash` (which can never convert).
THREAD_CARDS = [(1, "small"), (1, "medium"), (16, "large"), (64, "large")]

KINDS = ["INNER", "LEFT", "RIGHT", "FULL", "LEFT SEMI", "LEFT ANTI"]

# Key types: one representative per key-getter family the implementations
# dispatch to. Populated from artifacts/CANDIDATE_INVENTORY.md.
#   name -> (join condition template, human description, expected map variant)
KEY_TYPES = {
    "u64":    ("l.k = r.k",                        "single fixed-width UInt64",     "key64"),
    "str":    ("l.s = r.s",                        "variable-length String",        "key_string"),
    "comp":   ("l.a = r.a AND l.b = r.b",          "two-column composite UInt64",   "keys128"),
}


def comparator_for(threads: int) -> str:
    """`unified_hash` is compared against `hash` at 1 thread and `parallel_hash` above."""
    return "hash" if threads == 1 else "parallel_hash"


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    kind: str
    key: str
    match: str
    threads: int
    card: str

    @property
    def cell_id(self) -> str:
        k = self.kind.replace(" ", "-")
        return f"{k}|{self.key}|{self.match}|t{self.threads}|{self.card}"

    @property
    def build_table(self) -> str:
        return f"b_{self.card}"

    @property
    def probe_table(self) -> str:
        return f"p_{self.card}_{self.match}"

    @property
    def has_right_cols(self) -> bool:
        return self.kind not in ("LEFT SEMI", "LEFT ANTI")

    def skip_reason(self, algo: str) -> str | None:
        """Why this (cell, algo) combination cannot be measured as labelled.

        The one real case, pre-registered before it was hit: allowParallelHashJoin
        (TableJoin.cpp:1301-1303) returns false for any kind outside
        Left/Inner/Right/Full, so requesting `parallel_hash` for SEMI/ANTI yields
        plain `hash`. Measuring it would compare `unified_hash` against something
        that is not `parallel_hash` while labelling it `parallel_hash`.
        """
        if algo == "parallel_hash" and self.kind in ("LEFT SEMI", "LEFT ANTI"):
            return ("no parallel_hash comparator: allowParallelHashJoin() is false for "
                    "SEMI/ANTI (TableJoin.cpp:1301-1303), so the request silently "
                    "yields plain `hash`")
        return None


def all_cells() -> list[Cell]:
    """The declared matrix. Single source of truth for Gate G0.6."""
    cells = []
    for kind in KINDS:
        for key in KEY_TYPES:
            for match in MATCH_RATES:
                for threads, card in THREAD_CARDS:
                    cells.append(Cell(kind, key, match, threads, card))
    return cells


# --------------------------------------------------------------------------
# SQL
# --------------------------------------------------------------------------

def join_sql(cell: Cell, mode: str = "timed") -> str:
    """SQL for one cell.

    mode="timed"     cheap order-independent aggregates; these double as a
                     continuously-checked weak checksum on every timed run.
    mode="checksum"  strong full-output-column checksum for Gate G0.2. Run
                     separately from the timed runs so its cost never enters a
                     measurement.
    mode="buildonly" probe side reduced to (almost) nothing, so the query is
                     dominated by the build. The independent origin that
                     validates `FillingRightJoinSide` as the build phase (G0.5).
    """
    on = KEY_TYPES[cell.key][0]
    kind = cell.kind
    build, probe = cell.build_table, cell.probe_table

    if mode == "buildonly":
        # WHERE 0 on the probe side rather than LIMIT 0: LIMIT 0 lets the planner
        # prune the join entirely, which would measure nothing at all.
        left = f"(SELECT * FROM {probe} WHERE k < 0)"
    else:
        left = probe

    if mode == "checksum":
        if cell.has_right_cols:
            agg = ("count() AS cnt, "
                   "sum(cityHash64(l.k, l.s, l.a, l.b, r.k, r.s, r.a, r.b, r.v)) AS chk")
        else:
            agg = "count() AS cnt, sum(cityHash64(l.k, l.s, l.a, l.b)) AS chk"
    else:
        if cell.has_right_cols:
            agg = "count() AS cnt, sum(l.k) AS s1, sum(r.v) AS s2"
        else:
            agg = "count() AS cnt, sum(l.k) AS s1, toUInt64(0) AS s2"

    return (f"SELECT {agg} FROM {left} AS l "
            f"{kind} JOIN {build} AS r ON {on}")


def settings_for(cell: Cell, algo: str, extra: dict | None = None) -> dict:
    s = dict(PINNED_SETTINGS)
    s["join_algorithm"] = algo
    s["max_threads"] = cell.threads
    if extra:
        s.update(extra)
    return s


# --------------------------------------------------------------------------
# Per-run measurement readback
# --------------------------------------------------------------------------

PHASE_PROCESSORS = {
    "build": ("FillingRightJoinSide",),
    "probe": ("JoiningTransform",),
    "nonjoined": ("DelayedJoinedBlocksTransform", "DelayedJoinedBlocksWorkerTransform"),
}


def read_run(query_id: str) -> dict:
    """Pull wall, CPU and the phase split for one executed query.

    Wall is the server-side `query_duration_ms`, not a client-side stopwatch, so
    client process startup and HTTP framing cannot leak into it.
    """
    row = run_query(
        f"""
        SELECT query_duration_ms,
               ProfileEvents['UserTimeMicroseconds'],
               ProfileEvents['SystemTimeMicroseconds'],
               memory_usage,
               result_rows
        FROM system.query_log
        WHERE query_id = '{query_id}' AND type = 'QueryFinish'
        LIMIT 1 FORMAT TSV
        """
    ).strip()
    if not row:
        raise QueryError(f"no query_log row for {query_id}")
    wall_ms, user_us, sys_us, mem, rrows = row.split("\t")

    proc = run_query(
        f"""
        SELECT name, sum(elapsed_us)
        FROM system.processors_profile_log
        WHERE query_id = '{query_id}'
        GROUP BY name FORMAT TSV
        """
    ).strip()
    per_name = {}
    if proc:
        for line in proc.split("\n"):
            name, us = line.split("\t")
            per_name[name] = per_name.get(name, 0) + int(us)

    phases = {}
    for phase, names in PHASE_PROCESSORS.items():
        phases[phase] = sum(per_name.get(n, 0) for n in names)
    total_proc = sum(per_name.values())
    # `other` closes the accounting identity by construction; G0.5 asserts the
    # identity holds exactly, which catches rows going missing between the two
    # log tables.
    phases["other"] = total_proc - sum(phases.values())
    phases["total_proc"] = total_proc

    return {
        "wall_ms": float(wall_ms),
        "user_us": int(user_us),
        "sys_us": int(sys_us),
        "cpu_us": int(user_us) + int(sys_us),
        "memory_usage": int(mem),
        "result_rows": int(rrows),
        **{f"{k}_us": v for k, v in phases.items()},
    }


ALGO_SYMBOL_RULES = [
    # Order matters: `parallel_hash`'s shards ARE baseline DB::HashJoin objects,
    # so ConcurrentHashJoin must be tested before the baseline marker.
    ("parallel_hash", "DB::ConcurrentHashJoin"),
    ("unified_hash", "DB::Unified::HashJoin"),
    ("hash", "DB::HashJoin::"),
]


def assert_algorithm(query_id: str) -> dict:
    """Gate G0.1: identify from demangled stacks which implementation actually ran.

    Positive identification of executing code, not an echo of the requested
    setting -- EXPLAIN, Settings['join_algorithm'] and ProfileEvents were all
    tried and rejected (WORKLOG E2.2).
    """
    counts = {}
    for label, marker in ALGO_SYMBOL_RULES:
        n = run_query(
            f"""
            SELECT countIf(arrayExists(x -> position(demangle(addressToSymbol(x)), '{marker}') > 0, trace))
            FROM system.trace_log
            WHERE query_id = '{query_id}' AND trace_type = 'CPU' FORMAT TSV
            """
        ).strip()
        counts[label] = int(n or 0)
    total = run_query(
        f"SELECT count() FROM system.trace_log WHERE query_id='{query_id}' AND trace_type='CPU' FORMAT TSV"
    ).strip()
    counts["total_samples"] = int(total or 0)

    if counts["parallel_hash"] > 0:
        verdict = "parallel_hash"
    elif counts["unified_hash"] > 0:
        verdict = "unified_hash"
    elif counts["hash"] > 0:
        verdict = "hash"
    else:
        verdict = "UNKNOWN"
    counts["verdict"] = verdict
    return counts


# --------------------------------------------------------------------------
# Statistics — the declared noise band
# --------------------------------------------------------------------------

def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


NOISE_FLOOR_FRAC = 0.05


def classify(base_vals, test_vals):
    """Classify test against base under the declared band: max(5%, 1 sample stdev).

    Returns (verdict, pct_delta, band_pct). Positive pct means test is SLOWER.
    """
    b, t = median(base_vals), median(test_vals)
    if b <= 0:
        return "NO_DATA", float("nan"), float("nan")
    pct = (t - b) / b * 100.0
    band_pct = max(NOISE_FLOOR_FRAC * 100.0, stdev(base_vals) / b * 100.0)
    if abs(pct) <= band_pct:
        return "within_noise", pct, band_pct
    return ("slower" if pct > 0 else "faster"), pct, band_pct
