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
              timeout: int = 3600, http_port: int | None = None) -> str:
    params = dict(settings or {})
    if query_id:
        params["query_id"] = query_id
    port = HTTP_PORT if http_port is None else int(http_port)
    url = f"http://{HTTP_HOST}:{port}/?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, data=sql.encode("utf-8"), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise QueryError(f"HTTP {exc.code} for query_id={query_id}: {body[:2000]}") from None


def flush_logs(http_port: int | None = None) -> None:
    """Flush one server, or every distinct arm port when `http_port` is None."""
    if http_port is not None:
        run_query("SYSTEM FLUSH LOGS", http_port=http_port)
        return
    ports = {HTTP_PORT}
    for arm in ARMS.values():
        if arm.get("http_port") is not None:
            ports.add(int(arm["http_port"]))
    for port in sorted(ports):
        run_query("SYSTEM FLUSH LOGS", http_port=port)


def scalar(sql: str, settings: dict | None = None, http_port: int | None = None) -> str:
    return run_query(sql, settings, http_port=http_port).strip()


def http_port_for(arm_or_algo: str) -> int | None:
    """Per-arm HTTP port override, or None to use harness.HTTP_PORT."""
    arm = ARMS.get(arm_or_algo)
    if not arm:
        return None
    port = arm.get("http_port")
    return None if port is None else int(port)


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
# Arms — what this sweep compares
# --------------------------------------------------------------------------

# Stage 1+: one shipping configuration. The probe-batch length is constexpr
# `PROBE_BATCH_ROWS = 8192` and fused-vs-split is `if constexpr (split_can_pay)`, so there is
# no within-binary fused/split A/B any more. Correctness is `diff_goldens.py` against Stage 0.
#
# Stage 6: cross-binary timing A/B via `uhj_pre` / `uhj_post` on distinct HTTP ports
# (`UHJ_PRE_HTTP_PORT` / `UHJ_POST_HTTP_PORT`). Same algo/expect/settings shape; one query in
# flight across both servers (sweep interleaves arms).
PROBE_BATCH_ROWS = 8192

PRE_HTTP_PORT = int(os.environ.get("UHJ_PRE_HTTP_PORT", "8122"))
POST_HTTP_PORT = int(os.environ.get("UHJ_POST_HTTP_PORT", "8121"))

ARMS = {
    "uhj_pre": {
        "algo": "unified_hash",
        # Default for kinds where `split_can_pay`; ALWAYS_FUSED_KINDS override via expected_probe.
        "expect": "uhj_split",
        "settings": {},
        "http_port": PRE_HTTP_PORT,
    },
    "uhj_post": {
        "algo": "unified_hash",
        "expect": "uhj_split",
        "settings": {},
        "http_port": POST_HTTP_PORT,
    },
}

BASELINE_ARM = "uhj_pre"
TEST_ARM = "uhj_post"
TEST_ARMS = (TEST_ARM,)
AB_ARMS = (BASELINE_ARM, TEST_ARM)

# Join kinds whose whole per-row emit is a filter bit. `if constexpr (!split_can_pay)` forces
# them onto EmitSink: SEMI LEFT and every ANTI. The arm assertion has to know that.
ALWAYS_FUSED_KINDS = {"LEFT SEMI", "LEFT ANTI", "RIGHT ANTI", "FULL ANTI", "INNER ANTI"}


def expected_probe(arm: str, kind: str) -> str:
    """The probe family a cell of this kind must show on the shipping (or Stage 6) path."""
    if kind in ALWAYS_FUSED_KINDS or " ANTI" in kind:
        return "uhj_fused"
    return ARMS[arm]["expect"]


def batch_of(arm: str) -> int:
    """Shipping batch length. Kept for ab_report labels; no longer an arm setting."""
    _ = arm
    return PROBE_BATCH_ROWS

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


def comparator_for(threads: int) -> str:  # noqa: ARG001 - kept: the arm is thread-independent
    """The baseline arm. Both arms are `unified_hash`, so the comparator no longer
    depends on the thread count the way an `unified_hash` vs `hash`/`parallel_hash`
    comparison did."""
    return BASELINE_ARM


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

    def skip_reason(self, arm_or_algo: str) -> str | None:
        """Why this (cell, arm) combination cannot be measured as labelled.

        Nothing is skipped in the fused-vs-split comparison: both arms are
        `unified_hash`, so every kind in the matrix runs on both. The one case that
        did apply when the comparator was `parallel_hash` is kept for the sibling
        scripts that still measure it: allowParallelHashJoin (TableJoin.cpp:1301-1303)
        returns false for any kind outside Left/Inner/Right/Full, so requesting
        `parallel_hash` for SEMI/ANTI silently yields plain `hash`, and measuring it
        would compare against something that is not `parallel_hash`.
        """
        if arm_or_algo == "parallel_hash" and self.kind in ("LEFT SEMI", "LEFT ANTI"):
            return ("no parallel_hash comparator: allowParallelHashJoin() is false for "
                    "SEMI/ANTI (TableJoin.cpp:1301-1303), so the request silently "
                    "yields plain `hash`")
        return None


def all_cells() -> list[Cell]:
    """The declared 144-cell matrix. Single source of truth for Gate G0.6."""
    cells = []
    for kind in KINDS:
        for key in KEY_TYPES:
            for match in MATCH_RATES:
                for threads, card in THREAD_CARDS:
                    cells.append(Cell(kind, key, match, threads, card))
    return cells


# --------------------------------------------------------------------------
# Stage 4+/6 special timed cells — multi-disjunct / additional-filter / ASOF
# --------------------------------------------------------------------------
# The 144-cell matrix is single-disjunct with no ASOF and no additional-filter
# cell, so Stages 4–5 are invisible to it. These three reuse the tiny Stage 0
# Memory tables (see verify_fused_output.ensure_special_tables) so they stay
# cheap enough to sit in every Stage 6 A/B sweep.

@dataclass(frozen=True)
class SpecialTimedCell:
    """One timed coverage cell outside the 144-cell matrix."""

    cell_id: str
    kind: str
    group: str
    timed_sql: str
    threads: int = 1
    key: str = "special"
    match: str = "n/a"
    card: str = "tiny"
    extra_settings: dict = field(default_factory=dict)

    @property
    def has_right_cols(self) -> bool:
        return True

    def skip_reason(self, arm_or_algo: str) -> str | None:
        if arm_or_algo == "parallel_hash":
            return ("no parallel_hash comparator for Stage 4 special timed cells "
                    "(multi/addfilter/ASOF are unified_hash coverage only)")
        return None


def special_timed_cells() -> list[SpecialTimedCell]:
    """Tiny-table multi/addfilter/ASOF cells plus real-sized Stage 0b timed cells."""
    left, right = "uhj_stage0_filter_left", "uhj_stage0_filter_right"
    asof_l, asof_r = "uhj_stage0_asof_left", "uhj_stage0_asof_right"
    af_l, af_r = "uhj_stage0_af_left", "uhj_stage0_af_right"
    filter_agg = "count() AS cnt, sum(l.k) AS s1, sum(r.v) AS s2"
    asof_agg = "count() AS cnt, sum(l.k) AS s1, sum(r.v) AS s2"
    return [
        SpecialTimedCell(
            cell_id="multi|filter|t1|timed",
            kind="INNER",
            group="multi",
            timed_sql=(f"SELECT {filter_agg} FROM {left} AS l "
                       f"INNER JOIN {right} AS r ON l.k = r.k OR l.a = r.a"),
        ),
        SpecialTimedCell(
            cell_id="addfilter|filter|t1|timed",
            kind="INNER",
            group="addfilter",
            timed_sql=(f"SELECT {filter_agg} FROM {left} AS l "
                       f"INNER JOIN {right} AS r ON l.k = r.k AND l.a < r.a"),
        ),
        SpecialTimedCell(
            cell_id="asof_inner|asof|t1|timed",
            kind="ASOF INNER",
            group="asof_inner",
            timed_sql=(f"SELECT {asof_agg} FROM {asof_l} AS l "
                       f"ASOF INNER JOIN {asof_r} AS r "
                       f"ON l.k = r.k AND l.ts >= r.ts"),
        ),
        # Stage 0b real-sized timed cells. The tiny cells above cannot see a regression
        # (4-row tables); these run the same shapes at matrix size. The b-clause is
        # avoided on the MergeTree tables (b = k % 1000 fans out to ~1000 matches per
        # probe row at medium cardinality).
        SpecialTimedCell(
            cell_id="multi2all|u64|t1|medium|timed",
            kind="INNER",
            group="multi",
            timed_sql=("SELECT count() AS cnt, sum(l.k) AS s1, sum(r.v) AS s2 "
                       "FROM p_medium_hi AS l INNER JOIN b_medium AS r "
                       "ON l.k = r.k OR l.a = r.a"),
        ),
        SpecialTimedCell(
            cell_id="multi2semi|u64|t1|medium|timed",
            kind="LEFT SEMI",
            group="multi",
            timed_sql=("SELECT count() AS cnt, sum(l.k) AS s1, sum(r.v) AS s2 "
                       "FROM p_medium_hi AS l LEFT SEMI JOIN b_medium AS r "
                       "ON l.k = r.k OR l.a = r.a"),
        ),
        SpecialTimedCell(
            cell_id="addfilter|u64|t1|medium|timed",
            kind="INNER",
            group="addfilter",
            timed_sql=(f"SELECT count() AS cnt, sum(l.k) AS s1, sum(r.v) AS s2 "
                       f"FROM {af_l} AS l INNER JOIN {af_r} AS r "
                       "ON l.k = r.k AND l.a < r.a"),
        ),
        # Short-circuit fold at scale: one large cell at 16 threads.
        SpecialTimedCell(
            cell_id="multi2semi|u64|t16|large|timed",
            kind="LEFT SEMI",
            group="multi",
            threads=16,
            timed_sql=("SELECT count() AS cnt, sum(l.k) AS s1, sum(r.v) AS s2 "
                       "FROM p_large_hi AS l LEFT SEMI JOIN b_large AS r "
                       "ON l.k = r.k OR l.a = r.a"),
        ),
    ]


def all_timed_cells() -> list[Cell | SpecialTimedCell]:
    """144-cell matrix plus the Stage 4+/6 special timed cells."""
    return [*all_cells(), *special_timed_cells()]


# --------------------------------------------------------------------------
# SQL
# --------------------------------------------------------------------------

def join_sql(cell: Cell | SpecialTimedCell, mode: str = "timed") -> str:
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
    if isinstance(cell, SpecialTimedCell):
        # Tiny Stage 0 Memory tables: one SQL covers timed/checksum/buildonly. Strong
        # checksums for these shapes live in the Stage 0 goldens, not the sweep.
        return cell.timed_sql

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


def settings_for(cell: Cell | SpecialTimedCell, arm_or_algo: str, extra: dict | None = None) -> dict:
    """Settings for one measured run. `arm_or_algo` is an arm name from `ARMS` when
    measuring the shipping (or Stage 6 binary) path, and a bare `join_algorithm` value for
    the sibling scripts that still compare implementations."""
    s = dict(PINNED_SETTINGS)
    if arm_or_algo in ARMS:
        s["join_algorithm"] = ARMS[arm_or_algo]["algo"]
        s.update(ARMS[arm_or_algo].get("settings") or {})
    else:
        s["join_algorithm"] = arm_or_algo
    s["max_threads"] = cell.threads
    if isinstance(cell, SpecialTimedCell):
        s.update(cell.extra_settings)
    if extra:
        s.update(extra)
    return s


# --------------------------------------------------------------------------
# Per-run measurement readback
# --------------------------------------------------------------------------

PHASE_PROCESSORS = {
    "build": ("FillingRightJoinSide",),
    "probe": ("JoiningTransform",),
    # `NonJoinedBlocksTransform` is the RIGHT/FULL non-joined scan. The
    # `DelayedJoinedBlocks*` pair is the delayed-blocks plumbing and reads 0 on
    # these queries; an earlier version of this mapping listed only those two and
    # therefore reported a zero non-joined phase for every RIGHT and FULL cell.
    # The accounting identity in G0.5 did not catch it, because zero is a
    # perfectly valid member of a partition -- see WORKLOG E7.
    "nonjoined": ("NonJoinedBlocksTransform",
                  "DelayedJoinedBlocksTransform", "DelayedJoinedBlocksWorkerTransform"),
}


def read_run(query_id: str, http_port: int | None = None) -> dict:
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
               result_rows,
               ProfileEvents['ConcurrentHashJoinBuildMicroseconds'],
               ProfileEvents['ConcurrentHashJoinProbeMicroseconds']
        FROM system.query_log
        WHERE query_id = '{query_id}' AND type = 'QueryFinish'
        LIMIT 1 FORMAT TSV
        """,
        http_port=http_port,
    ).strip()
    if not row:
        raise QueryError(f"no query_log row for {query_id}")
    (wall_ms, user_us, sys_us, mem, rrows, ch_build, ch_probe) = row.split("\t")

    proc = run_query(
        f"""
        SELECT name, sum(elapsed_us)
        FROM system.processors_profile_log
        WHERE query_id = '{query_id}'
        GROUP BY name FORMAT TSV
        """,
        http_port=http_port,
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
        # parallel_hash's own internal timers: an origin for the phase split that
        # fails differently from the pipeline's processor accounting (G0.5 iii).
        "ch_build_us": int(ch_build),
        "ch_probe_us": int(ch_probe),
        **{f"{k}_us": v for k, v in phases.items()},
    }


ALGO_SYMBOL_RULES = [
    # Order matters: `parallel_hash`'s shards ARE baseline DB::HashJoin objects,
    # so ConcurrentHashJoin must be tested before the baseline marker.
    ("parallel_hash", "DB::ConcurrentHashJoin"),
    ("unified_hash", "DB::Unified::HashJoin"),
    ("hash", "DB::HashJoin::"),
]

# Which probe loop ran, read out of the same stacks. Every arm is `unified_hash`; these
# markers identify the compile-time choice (`if constexpr (split_can_pay)`): `probeTwoPhase`
# for kinds that pay for a second pass, `EmitSink` for SEMI LEFT / ANTI. Stage 6's
# cross-binary A/B still needs this positive evidence the intended code ran.
#
# The markers are the two outer functions: `consumeProbeBatch` is inlined into
# `probeTwoPhase` and never appears, while `probeTwoPhase` and `EmitSink` both do.
PROBE_SYMBOL_RULES = [
    ("uhj_split", "DB::Unified::probeTwoPhase"),
    ("uhj_fused", "DB::Unified::EmitSink"),
]



def assert_algorithm(query_id: str, http_port: int | None = None) -> dict:
    """Gate G0.1: identify from demangled stacks which implementation actually ran.

    Positive identification of executing code, not an echo of the requested
    setting -- EXPLAIN, Settings['join_algorithm'] and ProfileEvents were all
    tried and rejected (WORKLOG E2.2).
    """
    counts = {}
    for label, marker in ALGO_SYMBOL_RULES + PROBE_SYMBOL_RULES:
        n = run_query(
            f"""
            SELECT countIf(arrayExists(x -> position(demangle(addressToSymbol(x)), '{marker}') > 0, trace))
            FROM system.trace_log
            WHERE query_id = '{query_id}' AND trace_type = 'CPU' FORMAT TSV
            """,
            http_port=http_port,
        ).strip()
        counts[label] = int(n or 0)
    total = run_query(
        f"SELECT count() FROM system.trace_log WHERE query_id='{query_id}' AND trace_type='CPU' FORMAT TSV",
        http_port=http_port,
    ).strip()
    counts["total_samples"] = int(total or 0)
    counts.update(judge_counts(counts))
    return counts


def judge_counts(counts: dict) -> dict:
    """Turn raw per-marker sample counts into a verdict. Separate from the query so that
    records written by an earlier version can be re-judged without re-running anything.

    A marker has to carry a real share of the stacks, not merely appear once. Symbol
    attribution is not exact: a return address can resolve into a neighbouring symbol, and
    identical code folding gives one address several names. In the `ab2` run exactly one
    sample of 238 in LEFT|str|lo|t1|medium resolved to `DB::ConcurrentHashJoin` against 149
    for `DB::Unified::HashJoin`, and because `parallel_hash` is tested first that single
    sample declared the whole query `parallel_hash`. The ordering is still required
    (parallel_hash's shards ARE baseline HashJoin objects, so it must be tested first); what
    was missing is that a stray sample must not outrank a plurality.
    """
    floor = max(2, int(0.05 * counts.get("total_samples", 0)))

    verdict = "UNKNOWN"
    for label in ("parallel_hash", "unified_hash", "hash"):
        if counts.get(label, 0) >= floor:
            verdict = label
            break

    split, fused = counts.get("uhj_split", 0), counts.get("uhj_fused", 0)
    if split >= floor and fused >= floor:
        probe = "CONFLICT"                         # one query cannot have run both loops
    elif split >= floor:
        probe = "uhj_split"
    elif fused >= floor:
        probe = "uhj_fused"
    else:
        probe = "UNKNOWN"

    return {"verdict": verdict, "probe_verdict": probe, "sample_floor": floor}


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
