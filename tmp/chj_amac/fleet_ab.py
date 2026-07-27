#!/usr/bin/env python3
"""fleet_ab.py -- two-ARM interleaved A/B benchmark driver for the AMAC +
order-preserving-probe mission (Unit 1 harness).

An ARM is {name, binary_path, server_env, settings_overlay}. A CELL is
  family:side.group.size.T<threads>[.anti][.dup16][.h50|.h05][.jun][.statson][.hash]
with family in {key32,key64,str,strzero,fixstr,k128,k256,null64,lcstr,mixed},
side in {build,probe}, group in {inner_all,left_all,rf_all,any,semi_anti,asof},
size in {S1..S5} (aggregate map bytes 1MB/32MB/1GB/4GB/16GB; DEFAULT row
counts are UNCALIBRATED until a --calibration file exists), threads in
{1,48,96}. `.anti` (semi_anti group only) instantiates ANTI LEFT JOIN instead
of SEMI LEFT JOIN -- MATRIX.md block 4 measures both. The trailing `.hash`
modifier marks an algo_override='hash' cell (grower-change in-band gate; both
arms run join_algorithm='hash').

Per cell: Memory-engine build/probe tables are filled identically on both
arms' servers; 4 untimed warmups per arm; N timed runs (default 10) strict
ABAB with the leading arm alternating by cell index. Per timed run: rowcount
must equal the closed form, the checksum must be stable within an arm and
equal across arms, and the ProfileEvents path assertion must hold -- any
violation marks the run INVALID and the cell FAILED (never averaged over).
Timing = query_log event_time_microseconds - query_start_time_microseconds.

Subcommands: plan, sweep (--local | remote via --ssh-host), report, selftest.
Remote mode is structurally shared with local mode (one Server interface)
but is UNTESTED until the fleet exists -- see SELFTEST.md.

Aggressively vendored from
/mnt/data/jbmt_results/jbmt-sweep-20260724/join_memory_bench.py
(byte-identical to `git show ahj:bep/tools/join_memory_bench.py`): server
lifecycle + readiness polling, ssh plumbing, log_comment query_log
extraction with exact-suffix tag matching, key/payload expression style,
FILL_SETTINGS, closed-form output-row validation, LPT shard planning,
completed-cells resume, and the fail-closed --check-events pattern.
Stdlib-only Python by policy.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import platform
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time

BASE_DIR = pathlib.Path(__file__).resolve().parent          # tmp/chj_amac
FLEET_DIR = BASE_DIR / "fleet"
RESULTS_DIR = FLEET_DIR / "results"

# The seven shared ProfileEvents (present in BOTH arms; the timing contract).
SHARED_EVENTS = (
    "ConcurrentHashJoinBuildMicroseconds",
    "ConcurrentHashJoinBuildDispatchMicroseconds",
    "ConcurrentHashJoinBuildInsertMicroseconds",
    "ConcurrentHashJoinBuildMergeMicroseconds",
    "ConcurrentHashJoinProbeMicroseconds",
    "ConcurrentHashJoinProbeDispatchMicroseconds",
    "ConcurrentHashJoinProbeLookupMicroseconds",
)

# ---------------------------------------------------------------------------
# FUTURE (Units 2-3) contract. The PRIMARY copy lives in parity/parity_gen.py
# (which also serves it to shell via --print-contract); this is fleet_ab's
# standalone copy because deploy.sh ships this file alone to shards, and
# order/run_order.sh carries a third (bash cannot import). On any contract
# change update all three: grep CLICKHOUSE_JOIN_AMAC under tmp/chj_amac.
# `selftest` cross-checks this copy against parity_gen when the parity
# harness is present on the host. The counters exist in NO binary yet: every
# assertion that depends on them AUTO-DETECTS availability via system.events
# and prints a loud SKIPPED line when absent; --require-engagement upgrades
# absence to failure once Unit 2 lands.
AMAC_ENGAGEMENT_EVENTS = (
    "ConcurrentHashJoinAmacBuildRows",
    "ConcurrentHashJoinAmacBuildRingGrowths",
    "ConcurrentHashJoinAmacProbeRows",
)
# Per-side gating counters (the build ring lands in Unit 2 before the probe
# ring in Unit 3): a side counts as present only when ALL of its gating
# counters exist in system.events. RingGrowths may be legitimately zero and
# is informational: it never gates a side.
AMAC_ASSERT_SIDES = {
    "build": ("ConcurrentHashJoinAmacBuildRows",),
    "probe": ("ConcurrentHashJoinAmacProbeRows",),
}
AMAC_ENV_VAR = "CLICKHOUSE_JOIN_AMAC"  # values: 0/off, 1/auto, force; read by the server process at start
# ---------------------------------------------------------------------------

# Positive execution-path assertions for parallel_hash cells. A run reporting
# zero build time on the total build event is a red flag, not something to
# average into a median (pattern vendored from join_memory_bench.py
# PATH_ASSERTION_EVENT).
#
# The dispatch/merge SUB-phases are deliberately RECORD-ONLY (2026-07-27,
# measured on the baseline a05f3ee binary, key64 S1 A/A cells):
#   - ProbeDispatchMicroseconds = 0 at T=4: baseline `joinBlock` skips the
#     scatter when `twoLevelMapIsUsed()` (emplace_back, sub-us) and small
#     per-block dispatches round to 0 us;
#   - BuildDispatchMicroseconds = 0 at T=1 (single slot: pass-through);
#   - BuildMergeMicroseconds = 1 us at T=1 (one timer tick from zero).
# Asserting them would mass-invalidate legitimate runs. They are still
# recorded in every row and in report's attribution table. The asserted set
# is exactly the claim surface: total build + insert always, total probe +
# lookup for probe-side cells.
PATH_ASSERT_BUILD_POSITIVE = (
    "ConcurrentHashJoinBuildMicroseconds",
    "ConcurrentHashJoinBuildInsertMicroseconds",
)
PATH_ASSERT_PROBE_POSITIVE = (
    "ConcurrentHashJoinProbeMicroseconds",
    "ConcurrentHashJoinProbeLookupMicroseconds",
)

WARMUP_RUNS = 4        # untimed warmups per (cell, arm); JIT rationale vendored from join_memory_bench.py
DEFAULT_TIMED_RUNS = 10
MIN_VERDICT_RUNS = 5   # below this many valid runs/arm a cell is INSUFFICIENT, never WIN/TIE/LOSS

# Local mode fixed ports (assigned non-default ports; a stray server on the
# default ports can never be measured by accident -- every client call passes
# an explicit --port).
LOCAL_PORTS = {"A": {"tcp": 19510, "http": 18510}, "B": {"tcp": 19520, "http": 18520}}
LOCAL_SRV_DIRS = {"A": FLEET_DIR / "srv_a", "B": FLEET_DIR / "srv_b"}

# ---------------------------------------------------------------------------
# Cell grammar
# ---------------------------------------------------------------------------

FAMILIES = ("key32", "key64", "str", "strzero", "fixstr", "k128", "k256", "null64", "lcstr", "mixed")
SIDES = ("build", "probe")
GROUPS = ("inner_all", "left_all", "rf_all", "any", "semi_anti", "asof")
SIZES = ("S1", "S2", "S3", "S4", "S5")
THREADS = (1, 48, 96)

SIZE_BYTES = {"S1": 1 << 20, "S2": 32 << 20, "S3": 1 << 30, "S4": 4 << 30, "S5": 16 << 30}

# Instantiation of the group axis (documented decisions):
#   rf_all    -> FULL JOIN (superset of RIGHT: exercises right-side emission
#                and unmatched-right accounting in one query)
#   any       -> ANY LEFT JOIN
#   semi_anti -> SEMI LEFT JOIN; a cell with the .anti modifier instantiates
#                ANTI LEFT JOIN instead (MATRIX.md block 4 measures both)
#   asof      -> ASOF JOIN (inner) on key equality + l.ts >= r.ts
GROUP_JOIN_CLAUSE = {
    "inner_all": "INNER JOIN",
    "left_all": "LEFT JOIN",
    "rf_all": "FULL JOIN",
    "any": "ANY LEFT JOIN",
    "semi_anti": "SEMI LEFT JOIN",
    "asof": "ASOF JOIN",
}


@dataclasses.dataclass(frozen=True)
class Cell:
    family: str
    side: str
    group: str
    size: str
    threads: int
    anti: bool = False          # semi_anti only: ANTI LEFT JOIN instead of SEMI LEFT JOIN
    dup16: bool = False
    hit_pct: int = 100          # 100 | 50 (.h50) | 5 (.h05)
    jun: bool = False           # join_use_nulls = 1
    statson: bool = False       # collect_hash_table_stats_during_joins = 1
    algo: str = "parallel_hash"  # 'hash' for .hash algo-override cells

    @property
    def cell_id(self) -> str:
        s = f"{self.family}:{self.side}.{self.group}.{self.size}.T{self.threads}"
        if self.anti:
            s += ".anti"
        if self.dup16:
            s += ".dup16"
        if self.hit_pct == 50:
            s += ".h50"
        elif self.hit_pct == 5:
            s += ".h05"
        if self.jun:
            s += ".jun"
        if self.statson:
            s += ".statson"
        if self.algo == "hash":
            s += ".hash"
        return s

    def axes(self) -> dict:
        return {
            "family": self.family, "side": self.side, "group": self.group,
            "size": self.size, "threads": self.threads, "anti": self.anti,
            "dup16": self.dup16, "hit_pct": self.hit_pct, "jun": self.jun,
            "statson": self.statson, "algo": self.algo,
        }

    @property
    def shape_key(self) -> str:
        """Band-file lookup key: the cell without size/threads/modifiers."""
        return f"{self.family}:{self.side}.{self.group}"


def parse_cell(cell_id: str) -> Cell:
    head, _, rest = cell_id.partition(":")
    if head not in FAMILIES:
        raise ValueError(f"cell {cell_id!r}: unknown family {head!r}")
    parts = rest.split(".")
    if len(parts) < 4:
        raise ValueError(f"cell {cell_id!r}: expected side.group.size.T<threads>")
    side, group, size, tpart = parts[0], parts[1], parts[2], parts[3]
    if side not in SIDES:
        raise ValueError(f"cell {cell_id!r}: unknown side {side!r}")
    if group not in GROUPS:
        raise ValueError(f"cell {cell_id!r}: unknown group {group!r}")
    if size not in SIZES:
        raise ValueError(f"cell {cell_id!r}: unknown size {size!r}")
    if not tpart.startswith("T") or not tpart[1:].isdigit():
        raise ValueError(f"cell {cell_id!r}: bad thread part {tpart!r}")
    threads = int(tpart[1:])
    kw = dict(anti=False, dup16=False, hit_pct=100, jun=False, statson=False, algo="parallel_hash")
    for mod in parts[4:]:
        if mod == "anti":
            kw["anti"] = True
        elif mod == "dup16":
            kw["dup16"] = True
        elif mod == "h50":
            kw["hit_pct"] = 50
        elif mod == "h05":
            kw["hit_pct"] = 5
        elif mod == "jun":
            kw["jun"] = True
        elif mod == "statson":
            kw["statson"] = True
        elif mod == "hash":
            kw["algo"] = "hash"
        else:
            raise ValueError(f"cell {cell_id!r}: unknown modifier {mod!r}")
    cell = Cell(head, side, group, size, threads, **kw)
    if cell.anti and cell.group != "semi_anti":
        raise ValueError(f"cell {cell_id!r}: .anti is a semi_anti-group variant only")
    if cell.dup16 and cell.group not in ("inner_all", "left_all"):
        # ANY/SEMI/ASOF pick one of several duplicate matches; which one is
        # implementation-defined, so the cross-arm checksum would not be a
        # fair oracle. Restricting dup16 keeps the oracle exact.
        raise ValueError(f"cell {cell_id!r}: dup16 requires inner_all or left_all")
    if cell.cell_id != cell_id:
        raise ValueError(f"cell {cell_id!r}: non-canonical spelling (canonical: {cell.cell_id})")
    return cell


# ---------------------------------------------------------------------------
# Key families: deterministic, bijective key derivation.
# Unlike join_memory_bench.py (intHash64, disjointness verified empirically),
# keys here are y = (rank * ODD) xor SALT -- bijective per column, so the
# hit domain [0, U) and the miss domain [MISS_OFFSET, MISS_OFFSET + Np) are
# PROVABLY disjoint and closed forms stay exact at any scale.
# ---------------------------------------------------------------------------

GOLDEN64 = 0x9E3779B97F4A7C15
ODD32 = 2654435761
MISS_OFFSET64 = 1 << 40
MISS_OFFSET32 = 1 << 31   # key32 ranks must stay < 2^32 pre-mix
PROBE_TS = 1 << 62        # probe ts >> any build ts, so ASOF always matches the max-ts build row


def _salt(family: str, tag: str, bits: int = 64) -> int:
    # Per-(family, column) salt so distinct families never coincidentally
    # share a keyspace (vendored idea: join_memory_bench.py key_id_salt).
    digest = hashlib.sha256(f"fleet_ab:{family}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << bits) - 1)


def _v64(rank_expr: str, salt: int) -> str:
    return f"bitXor(toUInt64({rank_expr}) * {GOLDEN64}, {salt})"


def _v32(rank_expr: str, salt: int) -> str:
    return f"bitXor(toUInt32({rank_expr}) * {ODD32}, toUInt32({salt}))"


def _hex16(v_expr: str) -> str:
    # leftPad: hex() of UInt64 trims leading zero bytes (still injective, but
    # variable length); pad to a fixed 16 chars for FixedString compatibility.
    return f"leftPad(lower(hex({v_expr})), 16, '0')"


@dataclasses.dataclass(frozen=True)
class Family:
    name: str
    col_types: tuple  # tuple of SQL types, one per key column k0..kN
    cost_weight: float  # only rank order matters (LPT balancing)
    map_bytes_per_row: int  # DEFAULT (UNCALIBRATED) aggregate-map bytes per build row
    miss_offset: int

    @property
    def key_columns(self):
        return tuple(f"k{i}" for i in range(len(self.col_types)))

    def key_exprs(self, rank_expr: str):
        n = self.name
        if n == "key32":
            return [_v32(rank_expr, _salt(n, "k0", 32))]
        if n == "key64":
            return [_v64(rank_expr, _salt(n, "k0"))]
        if n == "str":
            return [f"lower(hex({_v64(rank_expr, _salt(n, 'k0'))}))"]
        if n == "strzero":
            # Embedded and terminating zero bytes, still injective via hex.
            return [
                f"concat({_hex16(_v64(rank_expr, _salt(n, 'k0')))}, char(0), "
                f"{_hex16(_v64(rank_expr, _salt(n, 'k0b')))}, char(0))"
            ]
        if n == "fixstr":
            return [f"toFixedString({_hex16(_v64(rank_expr, _salt(n, 'k0')))}, 16)"]
        if n == "k128":
            return [_v64(rank_expr, _salt(n, f"k{i}")) for i in range(2)]
        if n == "k256":
            return [_v64(rank_expr, _salt(n, f"k{i}")) for i in range(4)]
        if n == "null64":
            # Nullable type, no actual NULLs: exercises the nullable key path
            # while keeping the closed form exact. The .jun modifier flips
            # join_use_nulls instead of injecting NULLs.
            return [f"toNullable({_v64(rank_expr, _salt(n, 'k0'))})"]
        if n == "lcstr":
            return [f"toLowCardinality(lower(hex({_v64(rank_expr, _salt(n, 'k0'))})))"]
        if n == "mixed":
            return [
                _v64(rank_expr, _salt(n, "k0")),
                f"lower(hex({_v64(rank_expr, _salt(n, 'k1'))}))",
            ]
        raise ValueError(f"unknown family {n}")


FAMILY_SPECS = {
    "key32":   Family("key32",   ("UInt32",), 1.0, 24, MISS_OFFSET32),
    "key64":   Family("key64",   ("UInt64",), 1.0, 32, MISS_OFFSET64),
    "str":     Family("str",     ("String",), 2.5, 80, MISS_OFFSET64),
    "strzero": Family("strzero", ("String",), 2.8, 112, MISS_OFFSET64),
    "fixstr":  Family("fixstr",  ("FixedString(16)",), 1.8, 56, MISS_OFFSET64),
    "k128":    Family("k128",    ("UInt64", "UInt64"), 1.6, 48, MISS_OFFSET64),
    "k256":    Family("k256",    ("UInt64", "UInt64", "UInt64", "UInt64"), 2.6, 80, MISS_OFFSET64),
    "null64":  Family("null64",  ("Nullable(UInt64)",), 1.2, 40, MISS_OFFSET64),
    "lcstr":   Family("lcstr",   ("LowCardinality(String)",), 2.2, 96, MISS_OFFSET64),
    "mixed":   Family("mixed",   ("UInt64", "String"), 2.8, 112, MISS_OFFSET64),
}
assert tuple(FAMILY_SPECS) == FAMILIES

# ---------------------------------------------------------------------------
# Sizing: size -> build rows. DEFAULTS ARE UNCALIBRATED (rough bytes-per-row
# estimates); a --calibration JSON file {family: {size: build_rows}} overrides
# them and flips rows_source to 'calibration-file' in every output row.
# ---------------------------------------------------------------------------

PROBE_ROWS_FACTOR = {"probe": 4.0, "build": 0.25}
ROWS_QUANTUM = 80  # divisible by 16 (dup16) and by 20 (h05 exactness)

# Probe-side duration floor. The PREREG-002c A/A run (fleet/results/
# noise_band_002c.jsonl) showed that cells whose timed query runs below
# ~200 ms are scheduler-jitter-bound at high thread counts (S2xT96 medians
# of 23-42 ms produced an 11% same-binary "LOSS"), while the one cell above
# the floor was tight (-0.17%). Probe-side cells therefore get at least
# PER_THREAD_MIN_PROBE_ROWS probe rows per thread, and every measured cell
# is additionally checked against MIN_CELL_DURATION_US after the run
# (fail-closed: a too-fast cell can never produce a verdict). Build-side
# cells cannot inflate their build rows without changing the map size, so
# small-size build cells at high T are expected to trip the floor and be
# re-dispositioned with recorded rationale.
PER_THREAD_MIN_PROBE_ROWS = 2_000_000
MIN_CELL_DURATION_US = 200_000


@dataclasses.dataclass(frozen=True)
class CellShape:
    build_rows: int
    probe_rows: int
    unique_keys: int
    dup: int
    hits: int
    expected_rows: int
    rows_source: str  # 'default-uncalibrated' | 'calibration-file'


def load_calibration(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(pathlib.Path(path).read_text())


def load_band_file(path: str) -> dict:
    """Band files store FRACTIONS (0.03 = 3%), keyed by cell id or shape_key.
    Fail-closed unit guard: a value like 3.0 would be a 300% band, silently
    turning every verdict into TIE and neutering the A/B gate."""
    band = json.loads(pathlib.Path(path).read_text())
    for key, value in band.items():
        if float(value) > 0.5:
            raise SystemExit(
                f"ERROR: band file {path}: {key} = {value}; "
                "band file value looks like a percentage; store fractions")
    return band


def resolve_shape(cell: Cell, calibration: dict) -> CellShape:
    fam = FAMILY_SPECS[cell.family]
    cal = calibration.get(cell.family, {}).get(cell.size)
    if cal is not None:
        build_rows, source = int(cal), "calibration-file"
    else:
        build_rows = SIZE_BYTES[cell.size] // fam.map_bytes_per_row
        source = "default-uncalibrated"
    build_rows = max(ROWS_QUANTUM * 52, build_rows // ROWS_QUANTUM * ROWS_QUANTUM)  # floor 4160 rows
    dup = 16 if cell.dup16 else 1
    unique = build_rows // dup
    probe_rows = int(build_rows * PROBE_ROWS_FACTOR[cell.side])
    if cell.side == "probe":
        probe_rows = max(probe_rows, cell.threads * PER_THREAD_MIN_PROBE_ROWS)
    probe_rows = max(ROWS_QUANTUM * 52, probe_rows // 20 * 20)
    hits = probe_rows * cell.hit_pct // 100
    assert hits * 100 == probe_rows * cell.hit_pct, "hit count must be exact (probe_rows % 20 == 0)"
    if cell.family == "key32":
        # key32 mixes in 32-bit space; ranks must stay below the miss offset
        # and miss ranks below 2^32 for the hit/miss domains to be disjoint.
        assert unique <= MISS_OFFSET32 and probe_rows + MISS_OFFSET32 < (1 << 32), \
            f"{cell.cell_id}: too many rows for the 32-bit key domain"
    expected = expected_output_rows(cell, unique, dup, probe_rows, hits)
    return CellShape(build_rows, probe_rows, unique, dup, hits, expected, source)


def expected_output_rows(cell: Cell, unique: int, dup: int, probe_rows: int, hits: int) -> int:
    """Closed forms (validated per run; discipline vendored from
    join_memory_bench.py expected_output_rows).

    Build: rank = number % U over Nb rows -> every key appears exactly `dup`
    times. Probe: hit rows (number < hits) cycle rank = number % U; miss rows
    take rank = number + MISS_OFFSET (bijective mix keeps domains disjoint).
    """
    probed_unique = min(hits, unique)
    if cell.group == "inner_all":
        return hits * dup
    if cell.group == "left_all":
        return hits * dup + (probe_rows - hits)
    if cell.group == "rf_all":  # FULL JOIN
        return hits * dup + (probe_rows - hits) + (unique - probed_unique) * dup
    if cell.group == "any":  # ANY LEFT: exactly one output row per probe row
        return probe_rows
    if cell.group == "semi_anti":  # SEMI LEFT: matched probe rows; ANTI LEFT: unmatched
        return (probe_rows - hits) if cell.anti else hits
    if cell.group == "asof":  # inner ASOF, probe ts >= every build ts: one match per hit row
        return hits
    raise ValueError(f"no closed form for group {cell.group}")


# ---------------------------------------------------------------------------
# SQL builders
# ---------------------------------------------------------------------------

BUILD_TABLE = "bench.build_t"
PROBE_TABLE = "bench.probe_t"

# Vendored verbatim from join_memory_bench.py FILL_SETTINGS: single-threaded
# deterministic fills with a fixed block size, so both arms' Memory tables are
# block-for-block identical.
FILL_SETTINGS = (
    "SETTINGS min_insert_block_size_rows = 0, min_insert_block_size_bytes = 0, "
    "max_block_size = 57344, max_threads = 1, max_insert_threads = 1"
)


def table_ddl(table: str, fam: Family, payload_prefix: str) -> str:
    cols = [f"{name} {type_}" for name, type_ in zip(fam.key_columns, fam.col_types)]
    cols.append("ts UInt64")
    cols.append(f"{payload_prefix}0 UInt64")
    return (
        f"DROP TABLE IF EXISTS {table}; "
        f"CREATE TABLE {table} ({', '.join(cols)}) ENGINE = Memory SETTINGS compress = false;"
    )


def build_fill_sql(cell: Cell, shape: CellShape) -> str:
    fam = FAMILY_SPECS[cell.family]
    key_select = ", ".join(
        f"{expr} AS {name}" for expr, name in zip(fam.key_exprs("rank"), fam.key_columns)
    )
    payload = f"{_v64('number', _salt(cell.family, 'build_payload'))} AS b_p0"
    return (
        f"INSERT INTO {BUILD_TABLE} SELECT {key_select}, number AS ts, {payload} "
        f"FROM (SELECT number, number % {shape.unique_keys} AS rank FROM numbers({shape.build_rows})) "
        f"{FILL_SETTINGS};"
    )


def probe_fill_sql(cell: Cell, shape: CellShape) -> str:
    fam = FAMILY_SPECS[cell.family]
    key_select = ", ".join(
        f"{expr} AS {name}" for expr, name in zip(fam.key_exprs("rank"), fam.key_columns)
    )
    payload = f"{_v64('number', _salt(cell.family, 'probe_payload'))} AS p_p0"
    rank = f"if(number < {shape.hits}, number % {shape.unique_keys}, number + {fam.miss_offset})"
    return (
        f"INSERT INTO {PROBE_TABLE} SELECT {key_select}, toUInt64({PROBE_TS}) AS ts, {payload} "
        f"FROM (SELECT number, {rank} AS rank FROM numbers({shape.probe_rows})) "
        f"{FILL_SETTINGS};"
    )


def _format_setting(key: str, value) -> str:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float)):
        return f"{key} = {value}"
    if value in ("false", "true"):
        return f"{key} = {value}"
    return f"{key} = '{value}'"


def timed_settings(cell: Cell, arm: "Arm", threads: int) -> dict:
    s = {
        "join_algorithm": cell.algo,
        "max_threads": threads,
        "query_plan_join_swap_table": "false",
        "enable_analyzer": 1,
        "enable_join_runtime_filters": 0,
        "max_bytes_before_external_join": 0,
        "max_bytes_ratio_before_external_join": 0,
        "collect_hash_table_stats_during_joins": 1 if cell.statson else 0,
        "max_execution_time": 600,
        "join_use_nulls": 1 if cell.jun else 0,
    }
    s.update(arm.settings_overlay)
    return s


def settings_fingerprint(settings: dict) -> str:
    # log_comment is intentionally excluded (it carries the per-run nonce).
    canon = "\n".join(f"{k}={settings[k]}" for k in sorted(settings))
    return hashlib.sha256(canon.encode()).hexdigest()


def checksum_expr(cell: Cell) -> str:
    """NULL-aware checksum over every output column, as (isNull, ifNull-
    toString) pairs -- the parity harness pattern. sum(cityHash64(*)) is
    BLIND to rows containing a NULL: cityHash64 propagates NULL and sum
    skips it (measured on the baseline binary, 2026-07-27), so LEFT/FULL
    unmatched rows with join_use_nulls=1 -- and null64's Nullable key
    default in unmatched rows even without it -- would vanish from the
    oracle (review finding 8). Applied to EVERY cell (uniform, A/B-
    symmetric), not only .jun cells."""
    fam = FAMILY_SPECS[cell.family]
    cols = [f"l.{c}" for c in (*fam.key_columns, "ts", "p_p0")]
    cols += [f"r.{c}" for c in (*fam.key_columns, "ts", "b_p0")]
    parts = []
    for c in cols:
        parts.append(f"isNull({c})")
        parts.append(f"ifNull(toString({c}), '')")
    return f"sum(cityHash64({', '.join(parts)}))"


def join_query_sql(cell: Cell, settings: dict, log_comment: str) -> str:
    fam = FAMILY_SPECS[cell.family]
    on = " AND ".join(f"l.{c} = r.{c}" for c in fam.key_columns)
    if cell.group == "asof":
        on += " AND l.ts >= r.ts"
    parts = [_format_setting(k, v) for k, v in settings.items()]
    parts.append(f"log_comment = '{log_comment}'")
    clause = "ANTI LEFT JOIN" if cell.anti else GROUP_JOIN_CLAUSE[cell.group]
    return (
        f"SELECT count() AS row_count, {checksum_expr(cell)} AS checksum "
        f"FROM {PROBE_TABLE} AS l {clause} {BUILD_TABLE} AS r ON {on} "
        f"SETTINGS {', '.join(parts)} FORMAT JSONEachRow"
    )


# ---------------------------------------------------------------------------
# Arms and servers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Arm:
    name: str
    role: str            # 'A' (baseline) | 'B' (candidate)
    binary_path: str     # local path (local mode) or remote path (remote mode)
    server_env: dict
    settings_overlay: dict
    binary_sha256: str = ""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def abs_path(p: str) -> str:
    """argparse type for LOCAL binary paths: resolve to absolute at parse time.
    The server child is spawned with cwd=<srv_dir>, so a relative path that
    resolves for the client (invocation cwd) would NOT resolve for the server
    (review finding 4). Remote paths (--remote-bin-*) are never resolved here."""
    return str(pathlib.Path(p).expanduser().resolve())


def parse_kv_list(items) -> dict:
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"expected K=V, got {item!r}")
        out[key] = value
    return out


def detect_amac(server) -> frozenset:
    """The subset of AMAC engagement counters present in system.events. AMAC
    lands in stages (build ring in Unit 2, probe ring in Unit 3), so
    detection is per counter — requiring the full contract would blind the
    recording for every stage before the last one (the gate_amacbuild
    incident). Works on either transport (needs only server.sql_json)."""
    names = ", ".join(f"'{n}'" for n in AMAC_ENGAGEMENT_EVENTS)
    rows = server.sql_json(
        f"SELECT name FROM system.events WHERE name IN ({names}) "
        "SETTINGS system_events_show_zero_values = 1 FORMAT JSONEachRow")
    return frozenset(r["name"] for r in rows)


class LocalServer:
    """One arm's server on this host. Started from INSIDE its scratch dir
    (preprocessed_configs land in the server CWD), fully managed within one
    script invocation: start -> poll SELECT 1 -> use -> stop by PID (own
    session group; never by pattern). Identity = sha256 of /proc/<pid>/exe
    (the embedded VERSION_GITHASH is configure-time-stale and MUST NOT be
    used). Lifecycle pattern vendored from join_memory_bench.py
    start_server/ensure_server_stopped, adapted to direct child management.
    """

    # sha cache: binary path -> sha256, so per-cell restarts do not re-hash a
    # multi-GB binary; /proc/<pid>/exe is inode-compared against the binary
    # and only hashed in full when the inode differs (then required to match).
    _sha_cache: dict = {}

    def __init__(self, arm: Arm, srv_dir: pathlib.Path, tcp_port: int, http_port: int):
        self.arm = arm
        self.srv_dir = srv_dir
        self.tcp_port = tcp_port
        self.http_port = http_port
        self.proc: subprocess.Popen | None = None
        self.proc_exe_sha256 = ""
        self.amac_available: frozenset | None = None

    def _write_configs(self) -> pathlib.Path:
        self.srv_dir.mkdir(parents=True, exist_ok=True)
        config = self.srv_dir / "config.xml"
        config.write_text(_server_config_text(str(self.srv_dir), self.tcp_port, self.http_port))
        (self.srv_dir / "users.xml").write_text(_server_users_text())
        return config

    def wipe_data(self) -> None:
        data = self.srv_dir / "data"
        if data.exists():
            shutil.rmtree(data)

    def start(self, timeout: float = 120.0) -> None:
        if self._port_in_use():
            raise RuntimeError(
                f"port {self.tcp_port} already accepts connections BEFORE start -- a foreign "
                f"server is running; refusing to measure it (fail-closed)"
            )
        config = self._write_configs()
        env = dict(os.environ)
        # Watchdog off: the Popen pid IS the server, so stop-by-PID is exact
        # and no watchdog can respawn a killed child.
        env["CLICKHOUSE_WATCHDOG_ENABLE"] = "0"
        env.update(self.arm.server_env)
        with open(self.srv_dir / "stdout.log", "ab") as out:
            self.proc = subprocess.Popen(
                [self.arm.binary_path, "server", "-C", str(config)],
                cwd=str(self.srv_dir), env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                start_new_session=True,
            )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server (arm {self.arm.name}) exited rc={self.proc.returncode} during startup; "
                    f"tail: {self._log_tail()}"
                )
            rc, _, _ = self.sql("SELECT 1 FORMAT Null", timeout=10)
            if rc == 0:
                self._verify_exe()
                self.amac_available = detect_amac(self)
                return
            time.sleep(0.5)
        raise RuntimeError(f"server (arm {self.arm.name}) not ready within {timeout}s; tail: {self._log_tail()}")

    def _port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex(("127.0.0.1", self.tcp_port)) == 0

    def _log_tail(self) -> str:
        try:
            lines = (self.srv_dir / "server.err.log").read_text(errors="replace").splitlines()
            return " | ".join(lines[-5:])
        except OSError:
            return "(no server.err.log)"

    def binary_sha(self) -> str:
        """sha256 of the arm's on-disk binary (cached); also used BEFORE any
        server start so sweep resume can require resumed rows to match the
        exact binaries of THIS invocation (review finding 2)."""
        sha = LocalServer._sha_cache.get(self.arm.binary_path)
        if sha is None:
            sha = sha256_file(self.arm.binary_path)
            LocalServer._sha_cache[self.arm.binary_path] = sha
        self.arm.binary_sha256 = sha
        return sha

    def _verify_exe(self) -> None:
        exe = f"/proc/{self.proc.pid}/exe"
        expected = self.binary_sha()
        st_exe, st_bin = os.stat(exe), os.stat(self.arm.binary_path)
        if (st_exe.st_dev, st_exe.st_ino) == (st_bin.st_dev, st_bin.st_ino):
            self.proc_exe_sha256 = expected  # same inode -> byte-identical
        else:
            self.proc_exe_sha256 = sha256_file(exe)
            if self.proc_exe_sha256 != expected:
                raise RuntimeError(
                    f"/proc/{self.proc.pid}/exe sha {self.proc_exe_sha256[:16]} != "
                    f"binary {self.arm.binary_path} sha {expected[:16]} (arm {self.arm.name})"
                )
        self.arm.binary_sha256 = expected

    def sql(self, sql: str, timeout: float | None = 120):
        argv = [self.arm.binary_path, "client", "--host", "127.0.0.1",
                "--port", str(self.tcp_port), "--multiquery"]
        try:
            proc = subprocess.run(argv, input=sql.encode(), stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return 124, b"", f"client timed out after {timeout}s"
        return proc.returncode, proc.stdout, proc.stderr.decode("utf-8", "replace").strip()

    def sql_json(self, sql: str, timeout: float | None = 120) -> list[dict]:
        rc, stdout, stderr = self.sql(sql, timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"query failed (rc={rc}): {stderr or 'no diagnostic'}\nSQL: {sql}")
        return [json.loads(line) for line in stdout.decode().splitlines() if line.strip()]

    def stop(self, timeout: float = 30.0) -> None:
        if self.proc is None:
            return
        proc, self.proc = self.proc, None
        if proc.poll() is not None:
            return
        proc.terminate()  # SIGTERM to the exact PID we spawned
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # own session (start_new_session)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)


class RemoteServer:
    """Remote-mode server over ssh (jbmt pattern, vendored from
    join_memory_bench.py run_ssh/start_server). UNTESTED until the fleet
    exists -- structurally shares the LocalServer interface so all sweep
    logic is transport-agnostic. Stop is by the PID captured at launch ONLY:
    no pattern-matching fallback of any kind, because a pgrep pattern on a
    shared host can match a concurrent operator's server (review finding 10).
    """

    def __init__(self, arm: Arm, ssh_host: str, ssh_key: str, remote_dir: str,
                 tcp_port: int, http_port: int):
        self.arm = arm
        self.ssh_host = ssh_host
        self.ssh_key = ssh_key
        self.remote_dir = remote_dir.rstrip("/")
        self.tcp_port = tcp_port
        self.http_port = http_port
        self.pid: int | None = None
        self.proc_exe_sha256 = ""
        self.amac_available: frozenset | None = None

    def _ssh_base(self):
        # Vendored from join_memory_bench.py ssh_base.
        return ["ssh", "-i", self.ssh_key, "-o", "StrictHostKeyChecking=accept-new",
                "-o", "BatchMode=yes", self.ssh_host]

    def run_ssh(self, remote_cmd: str, input_bytes: bytes | None = None, timeout: float | None = 120):
        try:
            proc = subprocess.run(self._ssh_base() + [remote_cmd],
                                  input=input_bytes if input_bytes is not None else b"",
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return 124, b"", f"ssh timed out after {timeout}s"
        return proc.returncode, proc.stdout, proc.stderr.decode("utf-8", "replace").strip()

    def wipe_data(self) -> None:
        rc, _, err = self.run_ssh(f"rm -rf {shlex.quote(self.remote_dir + '/data')}")
        if rc != 0:
            raise RuntimeError(f"remote data wipe failed (arm {self.arm.name}): {err}")

    def binary_sha(self) -> str:
        """sha256 of the deployed remote binary; fetched BEFORE any server
        start so sweep resume can require resumed rows to match the exact
        binaries of THIS invocation (review finding 2). Fail-closed: an ssh
        or hashing failure raises instead of resuming blind."""
        if not self.arm.binary_sha256:
            rc, stdout, err = self.run_ssh(f"sha256sum {shlex.quote(self.arm.binary_path)}", timeout=300)
            if rc != 0:
                raise RuntimeError(f"remote binary hash failed (arm {self.arm.name}): {err}")
            self.arm.binary_sha256 = stdout.decode().split()[0]
        return self.arm.binary_sha256

    def start(self, timeout: float = 120.0) -> None:
        d = self.remote_dir
        config_text = _server_config_text(d, self.tcp_port, self.http_port)
        users_text = _server_users_text()
        rc, _, err = self.run_ssh(
            f"mkdir -p {shlex.quote(d)} && cat > {shlex.quote(d + '/config.xml')}",
            input_bytes=config_text.encode())
        if rc != 0:
            raise RuntimeError(f"remote config write failed: {err}")
        rc, _, err = self.run_ssh(f"cat > {shlex.quote(d + '/users.xml')}", input_bytes=users_text.encode())
        if rc != 0:
            raise RuntimeError(f"remote users write failed: {err}")
        env_exports = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in {"CLICKHOUSE_WATCHDOG_ENABLE": "0", **self.arm.server_env}.items()
        )
        # Fire-and-forget launch capturing $! (see join_memory_bench.py
        # start_server for why the ssh call must not wait on the child).
        script = (
            f"cd {shlex.quote(d)} && nohup setsid env {env_exports} "
            f"{shlex.quote(self.arm.binary_path)} server -C {shlex.quote(d + '/config.xml')} "
            f"</dev/null >{shlex.quote(d + '/stdout.log')} 2>&1 & echo $!"
        )
        rc, stdout, err = self.run_ssh(script, timeout=30)
        if rc != 0 or not stdout.strip().isdigit():
            raise RuntimeError(f"remote server launch failed (rc={rc}): {err}")
        self.pid = int(stdout.strip())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rc, _, _ = self.sql("SELECT 1 FORMAT Null", timeout=15)
            if rc == 0:
                self.proc_exe_sha256 = self._remote_exe_sha()
                self.amac_available = detect_amac(self)
                return
            time.sleep(1.0)
        raise RuntimeError(f"remote server (arm {self.arm.name}) not ready within {timeout}s")

    def _remote_exe_sha(self) -> str:
        rc, stdout, err = self.run_ssh(f"sha256sum /proc/{self.pid}/exe", timeout=300)
        if rc != 0:
            raise RuntimeError(f"remote exe hash failed: {err}")
        sha = stdout.decode().split()[0]
        if not self.arm.binary_sha256:
            rc, stdout, err = self.run_ssh(f"sha256sum {shlex.quote(self.arm.binary_path)}", timeout=300)
            if rc != 0:
                raise RuntimeError(f"remote binary hash failed: {err}")
            self.arm.binary_sha256 = stdout.decode().split()[0]
        if sha != self.arm.binary_sha256:
            raise RuntimeError(
                f"/proc/{self.pid}/exe sha {sha[:16]} != deployed binary sha "
                f"{self.arm.binary_sha256[:16]} (arm {self.arm.name})")
        return sha

    def sql(self, sql: str, timeout: float | None = 650):
        # Vendored from join_memory_bench.py run_remote_sql.
        client_cmd = f"{shlex.quote(self.arm.binary_path)} client --port {self.tcp_port} --multiquery"
        return self.run_ssh(client_cmd, input_bytes=sql.encode(), timeout=timeout)

    def sql_json(self, sql: str, timeout: float | None = 650) -> list[dict]:
        rc, stdout, stderr = self.sql(sql, timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"remote query failed (rc={rc}): {stderr or 'no diagnostic'}\nSQL: {sql}")
        return [json.loads(line) for line in stdout.decode().splitlines() if line.strip()]

    def stop(self, timeout: float = 30.0) -> None:
        # PID-captured-at-launch ONLY; never pattern-kill (review finding 10:
        # a pgrep -f fallback on a shared default remote_dir can kill a
        # concurrent operator's server). self.pid is None only when start
        # never launched a child (start either sets it or raises).
        if self.pid is None:
            return
        pid, self.pid = self.pid, None
        rc, _, err = self.run_ssh(
            f"kill {pid} 2>/dev/null; sleep 2; "
            f"if kill -0 {pid} 2>/dev/null; then kill -9 {pid}; fi; "
            f"! kill -0 {pid} 2>/dev/null",
            timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"remote server pid {pid} (arm {self.arm.name}) still alive after "
                               f"SIGTERM+SIGKILL: {err}")


def _server_config_text(d: str, tcp_port: int, http_port: int) -> str:
    # One config for BOTH transports (the module docstring's shared-interface
    # claim). Minimal: explicit query_log (engagement/timing extraction
    # depends on it), NO keeper/zookeeper section, loopback only, assigned
    # non-default ports. Only configured ports are opened.
    return f"""<clickhouse>
    <logger>
        <level>warning</level>
        <log>{d}/server.log</log>
        <errorlog>{d}/server.err.log</errorlog>
    </logger>
    <listen_host>127.0.0.1</listen_host>
    <tcp_port>{tcp_port}</tcp_port>
    <http_port>{http_port}</http_port>
    <path>{d}/data/</path>
    <tmp_path>{d}/data/tmp/</tmp_path>
    <user_files_path>{d}/data/user_files/</user_files_path>
    <mlock_executable>false</mlock_executable>
    <query_log>
        <database>system</database>
        <table>query_log</table>
        <flush_interval_milliseconds>1000</flush_interval_milliseconds>
    </query_log>
    <user_directories>
        <users_xml>
            <path>users.xml</path>
        </users_xml>
    </user_directories>
</clickhouse>
"""


def _server_users_text() -> str:
    return """<clickhouse>
    <profiles><default/></profiles>
    <users>
        <default>
            <password></password>
            <networks><ip>127.0.0.1</ip></networks>
            <profile>default</profile>
            <quota>default</quota>
            <access_management>1</access_management>
        </default>
    </users>
    <quotas><default/></quotas>
</clickhouse>
"""


# ---------------------------------------------------------------------------
# Per-cell execution
# ---------------------------------------------------------------------------


def path_assertion(cell: Cell, events: dict) -> tuple[bool, str]:
    if cell.algo == "hash":
        nonzero = [n for n in SHARED_EVENTS if events.get(n, 0) != 0]
        if nonzero:
            return False, f"hash cell must not touch parallel_hash: nonzero {nonzero}"
        return True, ""
    zero = [n for n in PATH_ASSERT_BUILD_POSITIVE if events.get(n, 0) <= 0]
    if cell.side == "probe":
        zero += [n for n in PATH_ASSERT_PROBE_POSITIVE if events.get(n, 0) <= 0]
    if zero:
        return False, f"path assertion: expected > 0 for {zero}"
    return True, ""


def flush_and_extract(server, log_comment_base: str, timeout: float = 120) -> list[dict]:
    rc, _, stderr = server.sql("SYSTEM FLUSH LOGS;", timeout=60)
    if rc != 0:
        raise RuntimeError(f"SYSTEM FLUSH LOGS failed: {stderr}")
    # duration_us is the mission's timing contract: event_time_microseconds -
    # query_start_time_microseconds; query_duration_ms recorded as cross-check.
    return server.sql_json(
        "SELECT log_comment, query_id, query_duration_ms, "
        "toUnixTimestamp64Micro(event_time_microseconds) - "
        "toUnixTimestamp64Micro(query_start_time_microseconds) AS duration_us, "
        "ProfileEvents FROM system.query_log "
        f"WHERE log_comment LIKE '{log_comment_base}%' AND type = 'QueryFinish' "
        "ORDER BY event_time_microseconds FORMAT JSONEachRow",
        timeout=timeout,
    )


def rows_by_run_tag(ql_rows: list[dict], runs: int) -> tuple[dict | None, str]:
    """Exact-suffix tag matching, vendored from join_memory_bench.py
    run_one_algorithm: '|run1' must not match '|run12'; exactly one query_log
    row per timed tag or the whole arm's extraction is invalid."""
    by_tag: dict[str, list[dict]] = {}
    for r in ql_rows:
        tag = r["log_comment"].rsplit("|", 1)[-1]
        if tag.startswith("run") and tag[3:].isdigit():
            by_tag.setdefault(tag, []).append(r)
    expected = [f"run{i}" for i in range(runs)]
    counts = {t: len(v) for t, v in by_tag.items()}
    if sorted(by_tag) != sorted(expected) or any(n != 1 for n in counts.values()):
        return None, f"expected exactly one query_log row per tag {expected}, found {counts}"
    return {t: by_tag[t][0] for t in expected}, ""


def _events_from_row(r: dict) -> dict:
    pe = r.get("ProfileEvents") or {}
    return {name: int(pe.get(name, 0)) for name in SHARED_EVENTS}


def _engagement_from_row(r: dict, available: frozenset | None) -> dict | None:
    """Record the counters the binary actually has (per detect_amac); absent
    ones stay out of the dict so a reader can distinguish "not implemented
    yet" from "implemented and zero"."""
    if not available:
        return None
    pe = r.get("ProfileEvents") or {}
    return {name: int(pe.get(name, 0)) for name in AMAC_ENGAGEMENT_EVENTS if name in available}


def run_cell(cell: Cell, cell_index: int, arms: list[Arm], servers: list, shape: CellShape,
             runs: int, warmups: int, threads: int, shard, host: str, arch: str,
             results_fh) -> dict:
    """Fill both arms' tables identically, warm up, run strict-ABAB timed
    invocations, validate, extract query_log timing/events, append one JSONL
    row per timed run. Returns {'ok': bool, 'reason': str}."""
    fam = FAMILY_SPECS[cell.family]
    nonce = f"{int(time.time())}_{os.getpid()}"
    status = {"ok": True, "reason": ""}

    def fail(reason: str) -> dict:
        status["ok"] = False
        status["reason"] = reason
        print(f"  CELL FAILED: {cell.cell_id}: {reason}")
        return status

    # 1. Identical fills on both arms.
    for arm, srv in zip(arms, servers):
        for sql in ("CREATE DATABASE IF NOT EXISTS bench;",
                    table_ddl(BUILD_TABLE, fam, "b_p"),
                    table_ddl(PROBE_TABLE, fam, "p_p"),
                    build_fill_sql(cell, shape),
                    probe_fill_sql(cell, shape)):
            rc, _, stderr = srv.sql(sql, timeout=3600)
            if rc != 0:
                return fail(f"fill failed on arm {arm.name}: {stderr}")
        for table, want in ((BUILD_TABLE, shape.build_rows), (PROBE_TABLE, shape.probe_rows)):
            n = int(srv.sql_json(f"SELECT count() AS n FROM {table} FORMAT JSONEachRow")[0]["n"])
            if n != want:
                return fail(f"arm {arm.name}: {table} has {n} rows, expected {want}")

    # 2. Warmups per arm; capture the per-arm checksum.
    checksums: dict[str, int] = {}
    for arm, srv in zip(arms, servers):
        settings = timed_settings(cell, arm, threads)
        base = f"{cell.cell_id}|{nonce}|{arm.name}"
        for w in range(warmups):
            sql = join_query_sql(cell, settings, f"{base}|warmup{w}")
            rc, stdout, stderr = srv.sql(sql, timeout=650)
            if rc != 0:
                return fail(f"warmup {w} failed on arm {arm.name}: {stderr}")
            row = json.loads(stdout.decode().strip().splitlines()[0])
            if int(row["row_count"]) != shape.expected_rows:
                return fail(
                    f"warmup {w} arm {arm.name}: row_count {row['row_count']} != closed form {shape.expected_rows}")
            cs = int(row["checksum"])
            if arm.name not in checksums:
                checksums[arm.name] = cs
            elif checksums[arm.name] != cs:
                return fail(f"warmup {w} arm {arm.name}: checksum changed within arm")
    if len(set(checksums.values())) != 1:
        return fail(f"cross-arm checksum mismatch: {checksums}")

    # 3. Timed runs, strict ABAB; leading arm alternates by cell index.
    order_pair = (0, 1) if cell_index % 2 == 0 else (1, 0)
    positions: dict[tuple, int] = {}  # (arm_name, run) -> position 0..2*runs-1
    pos = 0
    for i in range(runs):
        for slot in order_pair:
            arm, srv = arms[slot], servers[slot]
            settings = timed_settings(cell, arm, threads)
            sql = join_query_sql(cell, settings, f"{cell.cell_id}|{nonce}|{arm.name}|run{i}")
            rc, stdout, stderr = srv.sql(sql, timeout=650)
            if rc != 0:
                return fail(f"run {i} failed on arm {arm.name}: {stderr}")
            row = json.loads(stdout.decode().strip().splitlines()[0])
            if int(row["row_count"]) != shape.expected_rows or int(row["checksum"]) != checksums[arm.name]:
                return fail(
                    f"run {i} arm {arm.name}: row_count={row['row_count']} (expected {shape.expected_rows}) "
                    f"checksum={row['checksum']} (warmup {checksums[arm.name]})")
            positions[(arm.name, i)] = pos
            pos += 1

    # 4. Extraction + per-run records.
    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cell_ok = True
    all_rows = []
    for arm, srv in zip(arms, servers):
        settings = timed_settings(cell, arm, threads)
        fp = settings_fingerprint(settings)
        base = f"{cell.cell_id}|{nonce}|{arm.name}"
        ql_rows = flush_and_extract(srv, base)
        by_tag, err = rows_by_run_tag(ql_rows, runs)
        for i in range(runs):
            rec = {
                "cell": cell.cell_id,
                "cell_axes": {**cell.axes(), "threads_effective": threads},
                "arm": arm.name,
                "arm_role": arm.role,
                "run": i,
                "position": positions[(arm.name, i)],
                "nonce": nonce,
                "valid": True,
                "invalid_reason": None,
                "duration_us": None,
                "query_duration_ms": None,
                "events": None,
                "engagement": None,
                "rows": shape.expected_rows,
                "expected_rows": shape.expected_rows,
                "checksum": str(checksums[arm.name]),
                "rows_build": shape.build_rows,
                "rows_probe": shape.probe_rows,
                "rows_source": shape.rows_source,
                "shard": shard,
                "host": host,
                "arch": arch,
                "binary_sha256": arm.binary_sha256,
                "proc_exe_sha256": srv.proc_exe_sha256,
                "settings_fingerprint": fp,
                "recorded_at": recorded_at,
            }
            if by_tag is None:
                rec["valid"] = False
                rec["invalid_reason"] = err
                cell_ok = False
            else:
                ql = by_tag[f"run{i}"]
                rec["duration_us"] = int(ql["duration_us"])
                rec["query_duration_ms"] = int(ql["query_duration_ms"])
                events = _events_from_row(ql)
                rec["events"] = events
                rec["engagement"] = _engagement_from_row(ql, srv.amac_available)
                ok, why = path_assertion(cell, events)
                if not ok:
                    rec["valid"] = False
                    rec["invalid_reason"] = why
                    cell_ok = False
            all_rows.append(rec)
    # Duration floor (fail-closed): a cell whose per-arm median runs faster
    # than MIN_CELL_DURATION_US is jitter-bound and can never yield a verdict.
    for role in sorted({r["arm_role"] for r in all_rows}):
        durations = sorted(r["duration_us"] for r in all_rows
                           if r["arm_role"] == role and r["valid"] and r["duration_us"] is not None)
        if durations and durations[len(durations) // 2] < MIN_CELL_DURATION_US:
            median_ms = durations[len(durations) // 2] / 1000
            why = f"below-duration-floor (arm {role} median {median_ms:.1f} ms < {MIN_CELL_DURATION_US // 1000} ms)"
            for r in all_rows:
                if r["valid"]:
                    r["valid"] = False
                    r["invalid_reason"] = why
            cell_ok = False
            break
    for rec in all_rows:
        results_fh.write(json.dumps(rec) + "\n")
    results_fh.flush()
    if not cell_ok:
        return fail("one or more runs INVALID (see invalid_reason in results)")
    return status


# ---------------------------------------------------------------------------
# Results loading / resume / verdicts
# ---------------------------------------------------------------------------


def load_result_rows(results_spec: str) -> list[dict]:
    rows = []
    for part in str(results_spec).split(","):
        path = pathlib.Path(part.strip())
        if not path.exists():
            # HARD error (review finding 5): a silently-skipped file makes a
            # report/resume lie about coverage. Fresh-start resume is handled
            # by completed_cells' own existence check, never here.
            raise SystemExit(f"ERROR: results file missing: {path} (fail-closed; fix --results)")
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dedup_last_attempt(rows: list[dict]) -> list[dict]:
    """Keep only rows of the LAST attempt nonce per (cell, arm_role, host): a
    cell re-run appends a fresh nonce and the freshest attempt wins WHOLESALE.
    Nonce-keyed (review finding 6) so that shrinking --runs can never leave a
    longer earlier attempt's tail rows (run >= new runs) pooled in."""
    last_nonce: dict[tuple, object] = {}
    for r in rows:
        last_nonce[(r["cell"], r.get("arm_role"), r.get("host"))] = r.get("nonce")
    keyed: dict[tuple, dict] = {}
    for r in rows:
        if last_nonce[(r["cell"], r.get("arm_role"), r.get("host"))] == r.get("nonce"):
            keyed[(r["cell"], r.get("arm_role"), r["run"], r.get("host"))] = r
    return list(keyed.values())


def expected_row_filters(cells, arms: list["Arm"]) -> dict:
    """What a row must match to count for THIS invocation (review finding 2):
    the cell's nominal threads, each arm role's binary sha256, and each arm
    role's settings fingerprint."""
    expected = {}
    for cell in cells:
        expected[cell.cell_id] = {
            "threads": cell.threads,
            "shas": {arm.role: arm.binary_sha256 for arm in arms},
            "fps": {arm.role: settings_fingerprint(timed_settings(cell, arm, cell.threads))
                    for arm in arms},
        }
    return expected


def completed_cells(results_path: pathlib.Path, runs: int, expected: dict) -> set:
    """Resume: a cell is complete when both roles have `runs` distinct VALID
    runs (freshest attempt) whose rows MATCH the current invocation --
    binary_sha256 per role, settings_fingerprint per role, and
    threads_effective == the cell's nominal threads (review finding 2:
    threads-override or different-binary rows must never satisfy resume).
    Vendored pattern: join_memory_bench.py _load_completed_cells."""
    if not results_path.exists():
        return set()
    rows = dedup_last_attempt(load_result_rows(str(results_path)))
    per_cell: dict[str, dict[str, set]] = {}
    for r in rows:
        exp = expected.get(r["cell"])
        role = r.get("arm_role")
        if exp is None or not r.get("valid") or role not in ("A", "B"):
            continue
        if (r.get("cell_axes") or {}).get("threads_effective") != exp["threads"]:
            continue
        if r.get("binary_sha256") != exp["shas"][role]:
            continue
        if r.get("settings_fingerprint") != exp["fps"][role]:
            continue
        per_cell.setdefault(r["cell"], {}).setdefault(role, set()).add(r["run"])
    done = set()
    for cell, roles in per_cell.items():
        if set(roles) == {"A", "B"} and all(len(v) >= runs for v in roles.values()):
            done.add(cell)
    return done


def cell_verdicts(rows: list[dict], band_file: dict | None = None,
                  min_runs: int = MIN_VERDICT_RUNS) -> dict[str, dict]:
    """Per-cell pooled per-arm medians of duration_us over valid runs, spread
    (pstdev), verdict win/tie/loss with band = max(3%, per-shape band from the
    band file, observed relative spread). WIN means the role-B (candidate)
    median is lower than role-A beyond the band. A cell with fewer than
    `min_runs` valid runs on either arm is INSUFFICIENT -- never WIN/TIE/LOSS
    (review finding 7)."""
    rows = dedup_last_attempt(rows)
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    out = {}
    for cell_id, cell_rows in sorted(by_cell.items()):
        invalid = [r for r in cell_rows if not r.get("valid")]
        by_role: dict[str, list[dict]] = {}
        for r in cell_rows:
            if r.get("valid"):
                by_role.setdefault(r["arm_role"], []).append(r)
        entry: dict = {
            "cell": cell_id,
            "n_valid": {role: len(v) for role, v in by_role.items()},
            "n_invalid": len(invalid),
            "arm_names": {r["arm_role"]: r["arm"] for r in cell_rows},
            "uncalibrated": any(r.get("rows_source") == "default-uncalibrated" for r in cell_rows),
        }
        if invalid or set(by_role) != {"A", "B"}:
            entry["verdict"] = "INVALID"
            entry["invalid_reasons"] = sorted({r.get("invalid_reason") or "missing arm" for r in invalid}) \
                or [f"arms present: {sorted(by_role)}"]
            out[cell_id] = entry
            continue
        if any(len(v) < min_runs for v in by_role.values()):
            entry["verdict"] = "INSUFFICIENT"
            entry["min_runs"] = min_runs
            out[cell_id] = entry
            continue
        med, spread = {}, {}
        for role, rs in by_role.items():
            durations = [r["duration_us"] for r in rs]
            med[role] = statistics.median(durations)
            spread[role] = statistics.pstdev(durations) if len(durations) > 1 else 0.0
        rel_spread = max(
            (spread[role] / med[role]) if med[role] else 0.0 for role in ("A", "B")
        )
        shape = None
        try:
            shape = parse_cell(cell_id).shape_key
        except ValueError:
            pass
        file_band = 0.0
        if band_file:
            file_band = float(band_file.get(cell_id, band_file.get(shape or "", 0.0)))
        band_frac = max(0.03, file_band, rel_spread)
        diff = med["B"] - med["A"]
        band_abs = band_frac * max(med["A"], med["B"])
        if abs(diff) <= band_abs:
            verdict = "TIE"
        elif diff < 0:
            verdict = "WIN"
        else:
            verdict = "LOSS"
        # Per-phase attribution from the 7 shared events (medians per arm).
        phases = {}
        for name in SHARED_EVENTS:
            m = {role: statistics.median([r["events"][name] for r in by_role[role]]) for role in ("A", "B")}
            phases[name] = {"A": m["A"], "B": m["B"], "delta": m["B"] - m["A"]}
        engagement = {}
        for role in ("A", "B"):
            vals = [r.get("engagement") for r in by_role[role]]
            if all(v is not None for v in vals) and vals:
                engagement[role] = {
                    n: statistics.median([v[n] for v in vals]) for n in AMAC_ENGAGEMENT_EVENTS
                }
            else:
                engagement[role] = None
        entry.update({
            "verdict": verdict,
            "median_us": med,
            "spread_us": spread,
            "rel_spread": rel_spread,
            "band_frac": band_frac,
            "diff_pct": (diff / med["A"] * 100.0) if med["A"] else 0.0,
            "phases": phases,
            "engagement": engagement,
        })
        out[cell_id] = entry
    return out


# ---------------------------------------------------------------------------
# Subcommand: plan
# ---------------------------------------------------------------------------


def cell_cost_estimate(cell: Cell, calibration: dict) -> float:
    """Deterministic relative cost for LPT shard balancing (vendored idea:
    join_memory_bench.py cell_cost_estimate): touched rows x family weight x
    executions x 2 arms; accuracy beyond rank order is unnecessary."""
    shape = resolve_shape(cell, calibration)
    rows = shape.build_rows + shape.probe_rows + shape.expected_rows
    executions = WARMUP_RUNS + DEFAULT_TIMED_RUNS
    return rows * FAMILY_SPECS[cell.family].cost_weight * executions * 2


def load_cells_file(path: str) -> list[str]:
    p = pathlib.Path(path)
    text = p.read_text()
    if p.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict) and "measured_plan" in data:  # matrix.json
            ids = list(data["measured_plan"]["cells"]) + list(data["hash_inband"]["cells"])
            return ids
        if isinstance(data, list):
            return [c if isinstance(c, str) else c["cell"] for c in data]
        raise ValueError(f"unrecognized cells file structure: {path}")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def default_plan_cells() -> list[str]:
    # The frozen plan is DATA, not code: regenerating it on the fly would
    # silently substitute the current generator's plan for the registered one
    # (MATRIX.md freeze), and a deployed shard has no generator at all.
    matrix_path = FLEET_DIR / "matrix.json"
    if not matrix_path.exists():
        raise SystemExit(
            f"ERROR: {matrix_path} missing (frozen plan; fail-closed). Ship fleet/matrix.json "
            "next to the driver or pass --cells/--cells-file; regenerate only deliberately "
            "via fleet/matrix_gen.py.")
    return load_cells_file(str(matrix_path))


def lpt_assignment(cells: list[Cell], shards: int, calibration: dict) -> tuple[dict, list]:
    """Deterministic LPT: heaviest cell first onto the least-loaded shard
    (vendored from join_memory_bench.py build_plan). Single-sourced on
    purpose: `plan` (the published shard map) and `sweep --shard` (what a
    shard actually runs) must partition identically. Returns
    (cell_id -> shard, per-shard loads)."""
    loads = [0.0] * shards
    assignment: dict[str, int] = {}
    for cell in sorted(cells, key=lambda c: (-cell_cost_estimate(c, calibration), c.cell_id)):
        shard = loads.index(min(loads))
        assignment[cell.cell_id] = shard
        loads[shard] += cell_cost_estimate(cell, calibration)
    return assignment, loads


def plan_command(args) -> int:
    calibration = load_calibration(args.calibration)
    cell_ids = load_cells_file(args.cells_file) if args.cells_file else default_plan_cells()
    cells = [parse_cell(c) for c in cell_ids]
    shards = args.shards
    if shards < 1:
        raise SystemExit(f"plan: --shards must be >= 1, got {shards}")
    if shards > len(cells):
        raise SystemExit(f"plan: --shards {shards} > {len(cells)} cells; refusing -- some shards "
                         f"would be empty and load_balance meaningless (review finding 9)")
    assignment, loads = lpt_assignment(cells, shards, calibration)
    plan = [
        {"cell": c.cell_id, "shard": assignment[c.cell_id],
         "est_cost": cell_cost_estimate(c, calibration)}
        for c in cells
    ]
    out = json.dumps(plan, indent=2)
    if args.out:
        pathlib.Path(args.out).write_text(out + "\n")
        print(f"wrote {len(plan)} cells to {args.out}")
    else:
        print(out)
    for shard in range(shards):
        members = [p for p in plan if p["shard"] == shard]
        print(f"shard {shard}: cells={len(members)} est_cost={sum(m['est_cost'] for m in members):.3e}",
              file=sys.stderr)
    balance = max(loads) / min(loads) if min(loads) > 0 else float("inf")
    print(f"FLEET_AB PLAN RESULT: cells={len(plan)} shards={shards} load_balance={balance:.3f} -> OK")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: sweep
# ---------------------------------------------------------------------------


def build_arms(args) -> list[Arm]:
    env_a = parse_kv_list(getattr(args, "env_a", None))
    env_b = parse_kv_list(getattr(args, "env_b", None))
    set_a = parse_kv_list(getattr(args, "set_a", None))
    set_b = parse_kv_list(getattr(args, "set_b", None))
    bin_a = args.arm_a
    bin_b = args.arm_b
    if getattr(args, "aa", False):
        if bin_b is None:
            bin_b = bin_a
        elif bin_b != bin_a:
            raise SystemExit("--aa requires the same binary on both arms (omit --arm-b)")
    if bin_a is None or bin_b is None:
        raise SystemExit("sweep requires --arm-a and --arm-b (or --aa)")
    name_a = args.name_a or ("aaA" if args.aa else "armA")
    name_b = args.name_b or ("aaB" if args.aa else "armB")
    return [
        Arm(name_a, "A", bin_a, env_a, set_a),
        Arm(name_b, "B", bin_b, env_b, set_b),
    ]


def make_servers(args, arms: list[Arm]) -> list:
    if args.local:
        return [
            LocalServer(arms[0], LOCAL_SRV_DIRS["A"], LOCAL_PORTS["A"]["tcp"], LOCAL_PORTS["A"]["http"]),
            LocalServer(arms[1], LOCAL_SRV_DIRS["B"], LOCAL_PORTS["B"]["tcp"], LOCAL_PORTS["B"]["http"]),
        ]
    if not args.ssh_host or not args.ssh_key:
        raise SystemExit("remote mode requires --ssh-host and --ssh-key (or pass --local)")
    if not args.remote_bin_a or not args.remote_bin_b:
        raise SystemExit("remote mode requires --remote-bin-a/--remote-bin-b")
    arms[0].binary_path = args.remote_bin_a
    arms[1].binary_path = args.remote_bin_b
    print("NOTE: remote mode is UNTESTED (no fleet yet); see SELFTEST.md")
    return [
        RemoteServer(arms[0], args.ssh_host, args.ssh_key, args.remote_dir + "/srv_a",
                     LOCAL_PORTS["A"]["tcp"], LOCAL_PORTS["A"]["http"]),
        RemoteServer(arms[1], args.ssh_host, args.ssh_key, args.remote_dir + "/srv_b",
                     LOCAL_PORTS["B"]["tcp"], LOCAL_PORTS["B"]["http"]),
    ]


def sweep_cells_from_args(args) -> list[Cell]:
    if args.cells:
        return [parse_cell(c) for c in args.cells.split(",")]
    ids = default_plan_cells()
    if args.shard is not None:
        if not 0 <= args.shard < args.shards:
            raise SystemExit(f"sweep: --shard {args.shard} out of range for --shards {args.shards}")
        if args.shards > len(ids):
            raise SystemExit(f"sweep: --shards {args.shards} > {len(ids)} cells; refusing "
                             f"(empty shards; review finding 9)")
        calibration = load_calibration(args.calibration)
        cells = [parse_cell(c) for c in ids]
        assignment, _ = lpt_assignment(cells, args.shards, calibration)
        return [c for c in cells if assignment[c.cell_id] == args.shard]
    return [parse_cell(c) for c in ids]


def sweep_command(args) -> int:
    arms = build_arms(args)
    cells = sweep_cells_from_args(args)
    calibration = load_calibration(args.calibration)
    runs = args.runs
    warmups = args.warmups
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = pathlib.Path(args.results) if args.results else (
        RESULTS_DIR / f"sweep_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl")
    host = socket.gethostname()
    arch = platform.machine()

    servers = make_servers(args, arms)
    # Resolve each arm's deployed-binary sha BEFORE the resume decision:
    # resumed rows must match the exact binaries, settings fingerprints, and
    # nominal threads of THIS invocation (review finding 2).
    for srv in servers:
        srv.binary_sha()
    done = completed_cells(results_path, runs, expected_row_filters(cells, arms))
    todo = [(i, c) for i, c in enumerate(cells) if c.cell_id not in done]
    print(f"sweep: {len(cells)} cells planned, {len(cells) - len(todo)} already complete "
          f"(resume via {results_path}), {len(todo)} to run")
    if args.require_engagement and done:
        # A fully-resumed sweep must not bypass the engagement gate (review
        # finding 3): every resumed candidate-arm row must carry engagement.
        resumed = [r for r in dedup_last_attempt(load_result_rows(str(results_path)))
                   if r["cell"] in done and r.get("arm_role") == "B" and r.get("valid")]
        lacking = sorted({r["cell"] for r in resumed if r.get("engagement") is None})
        if lacking:
            print("FLEET_AB SWEEP RESULT: cells_run=0 cells_ok=0 cells_failed=0 -> FAIL "
                  f"(--require-engagement: resumed rows lack AMAC engagement for: {lacking})")
            return 1
    # Same wipe policy on BOTH transports (review finding 10: remote data
    # dirs must not accumulate across sweeps).
    for srv in servers:
        srv.wipe_data()

    amac_skip_printed = set()
    cells_ok, cells_failed = 0, 0
    try:
        with open(results_path, "a") as fh:
            for cell_index, cell in todo:
                threads = cell.threads
                shape = resolve_shape(cell, calibration)
                print(f"=== cell {cell_index + 1}/{len(cells)}: {cell.cell_id} "
                      f"rows_build={shape.build_rows} rows_probe={shape.probe_rows} "
                      f"expected_rows={shape.expected_rows} threads={threads} "
                      f"rows_source={'DEFAULT-UNCALIBRATED' if shape.rows_source == 'default-uncalibrated' else 'calibration-file'} ===")
                # Fresh servers per cell (fleet protocol: restart per cell).
                for srv in servers:
                    srv.stop()
                for srv in servers:
                    srv.start()
                for arm, srv in zip(arms, servers):
                    if not srv.amac_available and arm.name not in amac_skip_printed:
                        print(f"SKIPPED: AMAC engagement counters absent in system.events "
                              f"(arm={arm.name}); engagement recorded as null")
                        amac_skip_printed.add(arm.name)
                if args.require_engagement and not servers[1].amac_available:
                    print("FLEET_AB SWEEP RESULT: cells_run=0 cells_ok=0 cells_failed=0 -> FAIL "
                          "(--require-engagement: candidate arm lacks AMAC counters)")
                    return 1
                t0 = time.monotonic()
                status = run_cell(cell, cell_index, arms, servers, shape, runs, warmups,
                                  threads, args.shard, host, arch, fh)
                wall = time.monotonic() - t0
                if status["ok"]:
                    cells_ok += 1
                    print(f"  cell OK ({wall:.1f}s)")
                else:
                    cells_failed += 1
    finally:
        for srv in servers:
            try:
                srv.stop()
            except Exception as ex:  # noqa: BLE001 -- teardown must not mask the sweep error
                print(f"WARNING: server stop failed: {ex}", file=sys.stderr)

    verdict_line = ""
    if args.aa:
        rows = [r for r in load_result_rows(str(results_path))
                if r["cell"] in {c.cell_id for c in cells}]
        verdicts = cell_verdicts(rows, min_runs=runs)
        nontie = {c: v["verdict"] for c, v in verdicts.items() if v["verdict"] != "TIE"}
        aa_pass = not nontie and len(verdicts) == len(cells) and cells_failed == 0
        for c, v in sorted(verdicts.items()):
            if v["verdict"] == "TIE":
                print(f"  A/A {c}: TIE (diff {v['diff_pct']:+.2f}%, band {v['band_frac'] * 100:.1f}%)")
            else:
                print(f"  A/A {c}: {v['verdict']} <- must be TIE")
        verdict_line = (f"FLEET_AB AA RESULT: cells={len(verdicts)} "
                        f"tie={len(verdicts) - len(nontie)} nontie={len(nontie)} -> "
                        f"{'PASS' if aa_pass else 'FAIL'}")
    ok = cells_failed == 0
    print(f"FLEET_AB SWEEP RESULT: cells_run={len(todo)} cells_ok={cells_ok} "
          f"cells_failed={cells_failed} results={results_path} -> {'PASS' if ok else 'FAIL'}")
    if verdict_line:
        print(verdict_line)
        return 0 if (ok and "PASS" in verdict_line) else 1
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------


def report_command(args) -> int:
    rows = load_result_rows(args.results)
    if not rows:
        print(f"FLEET_AB REPORT RESULT: cells=0 win=0 tie=0 loss=0 invalid=0 (no rows in {args.results})")
        return 1
    band_file = load_band_file(args.band_file) if args.band_file else None
    verdicts = cell_verdicts(rows, band_file, min_runs=args.min_runs)
    counts = {"WIN": 0, "TIE": 0, "LOSS": 0, "INVALID": 0, "INSUFFICIENT": 0}
    uncal = 0
    for cell_id, v in sorted(verdicts.items()):
        counts[v["verdict"]] += 1
        if v.get("uncalibrated"):
            uncal += 1
        if v["verdict"] == "INVALID":
            print(f"CELL {cell_id} verdict=INVALID reasons={v.get('invalid_reasons')}")
            continue
        if v["verdict"] == "INSUFFICIENT":
            print(f"CELL {cell_id} verdict=INSUFFICIENT valid_runs={v['n_valid']} "
                  f"(need >= {args.min_runs}/arm for a verdict; --min-runs overrides)")
            continue
        names = v["arm_names"]
        print(
            f"CELL {cell_id} verdict={v['verdict']} "
            f"A[{names.get('A')}]={v['median_us']['A']:.0f}us B[{names.get('B')}]={v['median_us']['B']:.0f}us "
            f"diff={v['diff_pct']:+.2f}% band={v['band_frac'] * 100:.1f}% "
            f"spread(A/B)={v['spread_us']['A']:.0f}/{v['spread_us']['B']:.0f}us "
            f"runs={v['n_valid'].get('A', 0)}/{v['n_valid'].get('B', 0)}"
            + (" [UNCALIBRATED-SIZE]" if v.get("uncalibrated") else "")
        )
        if not args.no_phases:
            print("  phase attribution (median us per arm):")
            print(f"    {'event':<50} {'A':>12} {'B':>12} {'delta':>12}")
            for name in SHARED_EVENTS:
                p = v["phases"][name]
                print(f"    {name:<50} {p['A']:>12.0f} {p['B']:>12.0f} {p['delta']:>+12.0f}")
            for role in ("A", "B"):
                eng = v["engagement"][role]
                if eng is not None:
                    print(f"    engagement[{role}]: " + ", ".join(f"{k}={eng[k]:.0f}" for k in AMAC_ENGAGEMENT_EVENTS))
    if uncal:
        print(f"WARNING: {uncal} cell(s) used DEFAULT-UNCALIBRATED size->rows mapping; "
              "medians are orientation only until a calibration file exists")
    print(
        f"FLEET_AB REPORT RESULT: cells={len(verdicts)} win={counts['WIN']} tie={counts['TIE']} "
        f"loss={counts['LOSS']} invalid={counts['INVALID']} insufficient={counts['INSUFFICIENT']} "
        f"uncalibrated={uncal}"
    )
    return 0 if counts["INVALID"] == 0 and counts["INSUFFICIENT"] == 0 else 1


# ---------------------------------------------------------------------------
# Subcommand: selftest
# ---------------------------------------------------------------------------


def selftest_command(args) -> int:
    if not args.local:
        raise SystemExit("selftest currently supports --local only")
    if not args.bin:
        raise SystemExit("selftest requires --bin")
    all_ok = True
    events_summary = "not-run"
    verdict_summary = "not-run"

    # Units 2-3 contract consistency: parity/parity_gen.py holds the PRIMARY
    # copy of the constants block; this file's copy must match it wherever the
    # parity harness is present (deployed shards carry fleet_ab.py alone).
    parity_dir = BASE_DIR / "parity"
    if (parity_dir / "parity_gen.py").exists():
        sys.path.insert(0, str(parity_dir))
        import parity_gen  # noqa: PLC0415
        mismatches = [
            name for name, ours, primary in (
                ("AMAC_ENGAGEMENT_EVENTS", AMAC_ENGAGEMENT_EVENTS, tuple(parity_gen.AMAC_ENGAGEMENT_EVENTS)),
                ("AMAC_ASSERT_SIDES", AMAC_ASSERT_SIDES, parity_gen.AMAC_ASSERT_SIDES),
                ("AMAC_ENV_VAR", AMAC_ENV_VAR, parity_gen.AMAC_ENV_VAR),
                ("SHARED_EVENTS", SHARED_EVENTS, tuple(parity_gen.SHARED_PROFILE_EVENTS)),
            ) if ours != primary
        ]
        if mismatches:
            print(f"contract-check: FAIL -- diverges from parity/parity_gen.py (primary) in: {mismatches}")
            all_ok = False
        else:
            print("contract-check: constants match parity/parity_gen.py (primary copy)")
    else:
        print("contract-check: SKIPPED (parity/parity_gen.py not present on this host)")

    if args.check_events:
        arm = Arm("selftest", "A", args.bin, {}, {})
        srv = LocalServer(arm, FLEET_DIR / "srv_selftest", LOCAL_PORTS["A"]["tcp"], LOCAL_PORTS["A"]["http"])
        srv.wipe_data()
        try:
            srv.start()
            # Fail-closed event existence check, vendored from
            # join_memory_bench.py selftest_check_events.
            names = ", ".join(f"'{n}'" for n in SHARED_EVENTS)
            rows = srv.sql_json(
                f"SELECT name FROM system.events WHERE name IN ({names}) "
                "SETTINGS system_events_show_zero_values = 1 FORMAT JSONEachRow")
            found = {r["name"] for r in rows}
            missing = [n for n in SHARED_EVENTS if n not in found]
            print(f"check-events: {len(found)}/{len(SHARED_EVENTS)} shared events present in system.events")
            if missing:
                print(f"check-events: MISSING (fail-closed): {missing}")
                all_ok = False
            # AMAC lands per SIDE (build ring before probe ring), so presence
            # is reported per counter -- a build-only binary must never be
            # summarized as all-PRESENT (nor as absent).
            amac = srv.amac_available or frozenset()
            for name in AMAC_ENGAGEMENT_EVENTS:
                print(f"check-events: AMAC counter {name}: {'PRESENT' if name in amac else 'ABSENT'}")
            sides_present = [s for s, gating in AMAC_ASSERT_SIDES.items()
                             if all(e in amac for e in gating)]
            if args.require_amac is not None:
                requested = [s for s in args.require_amac.split(",") if s]
                unknown = sorted(set(requested) - set(AMAC_ASSERT_SIDES))
                if unknown:
                    raise SystemExit(f"--require-amac: unknown side(s) {unknown}; "
                                     f"known sides: {list(AMAC_ASSERT_SIDES)}")
                for side in requested:
                    if side not in sides_present:
                        missing = [e for e in AMAC_ASSERT_SIDES[side] if e not in amac]
                        print(f"check-events: FAIL (--require-amac: side '{side}' counters absent: {missing})")
                        all_ok = False
            if args.forbid_amac and amac:
                print(f"check-events: FAIL (--forbid-amac set but counters present: {sorted(amac)})")
                all_ok = False
            if sides_present:
                amac_summary = ",".join(sides_present)
            elif amac:
                amac_summary = "partial"  # some counter present, no complete side
            else:
                amac_summary = "absent"
            events_summary = f"events={len(found)}/{len(SHARED_EVENTS)} amac={amac_summary}"
        finally:
            srv.stop()

    if args.verdict_selftest:
        # Deliberate A != B: same binary, arm B's settings_overlay halves
        # max_threads. The verdict machinery must produce a non-TIE verdict,
        # proving it CAN fail (band/verdict power check). The cell is a REAL
        # nominal-T4 cell id (grammar accepts any T; not a plan cell) -- the
        # old hidden --threads-override poisoned resume/MEASURED evidence.
        cell_id = "key64:probe.inner_all.S2.T4"
        ns = argparse.Namespace(
            local=True, arm_a=args.bin, arm_b=args.bin, aa=False,
            env_a=[], env_b=[], set_a=[], set_b=["max_threads=2"],
            name_a="vsA", name_b="vsB", cells=cell_id,
            shard=None, shards=1, calibration=None, runs=5, warmups=2,
            require_engagement=False,
            results=str(RESULTS_DIR / f"verdict_selftest_{int(time.time())}.jsonl"),
            ssh_host=None, ssh_key=None, remote_bin_a=None, remote_bin_b=None,
            remote_dir=None,
        )
        rc = sweep_command(ns)
        rows = load_result_rows(ns.results)
        verdicts = cell_verdicts(rows)
        v = verdicts.get(cell_id, {})
        verdict = v.get("verdict", "MISSING")
        print(f"verdict-selftest: A(T4) vs B(T2) -> {verdict} "
              f"(diff {v.get('diff_pct', 0):+.2f}%, band {v.get('band_frac', 0) * 100:.1f}%)")
        if rc != 0 or verdict in ("TIE", "MISSING", "INVALID"):
            print("verdict-selftest: FAIL -- expected a non-TIE verdict from a deliberate A != B pair")
            all_ok = False
        verdict_summary = f"verdict={verdict}"

    print(f"FLEET_AB SELFTEST RESULT: {events_summary} {verdict_summary} -> {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def add_common_sweep_args(p) -> None:
    p.add_argument("--local", action="store_true", help="two servers on this host (no ssh)")
    p.add_argument("--arm-a", type=abs_path, help="arm A (baseline) binary path (resolved to absolute)")
    p.add_argument("--arm-b", type=abs_path, help="arm B (candidate) binary path (resolved to absolute)")
    p.add_argument("--env-a", action="append", metavar="K=V", help="arm A server env (repeatable)")
    p.add_argument("--env-b", action="append", metavar="K=V", help="arm B server env (repeatable)")
    p.add_argument("--set-a", action="append", metavar="K=V", help="arm A settings overlay (repeatable)")
    p.add_argument("--set-b", action="append", metavar="K=V", help="arm B settings overlay (repeatable)")
    p.add_argument("--name-a", help="arm A name (default armA/aaA)")
    p.add_argument("--name-b", help="arm B name (default armB/aaB)")
    p.add_argument("--cells", help="comma-separated explicit cell ids")
    p.add_argument("--aa", action="store_true",
                   help="A/A mode: same binary both arms; all verdicts must be TIE")
    p.add_argument("--runs", type=int, default=DEFAULT_TIMED_RUNS)
    p.add_argument("--warmups", type=int, default=WARMUP_RUNS)
    p.add_argument("--calibration", help="JSON {family: {size: build_rows}} overriding the UNCALIBRATED defaults")
    p.add_argument("--results", help="append results JSONL here (default: fleet/results/sweep_<ts>.jsonl)")
    p.add_argument("--require-engagement", action="store_true",
                   help="fail if the candidate arm lacks the AMAC engagement counters "
                        "(applies to resumed rows too)")
    # NOTE: the hidden --threads-override flag was REMOVED (review findings
    # 1-2): its rows counted for the nominal-T cell id and poisoned resume /
    # MEASURED evidence. Selftests use real low-T cell ids (e.g. ...T4)
    # instead; the cell grammar accepts any T, the plan cells are unaffected.
    # Remote (jbmt pattern; UNTESTED until the fleet exists).
    p.add_argument("--shard", type=int, help="run only this shard's cells (remote fleet mode)")
    p.add_argument("--shards", type=int, default=8, help="total shard count for --shard")
    p.add_argument("--ssh-host", help="user@ip of the shard host (remote mode)")
    p.add_argument("--ssh-key", help="ssh private key path (remote mode)")
    p.add_argument("--remote-bin-a", help="arm A binary path on the remote host")
    p.add_argument("--remote-bin-b", help="arm B binary path on the remote host")
    p.add_argument("--remote-dir", default="/home/ubuntu/fleet_ab", help="remote scratch dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="emit the LPT shard assignment")
    p_plan.add_argument("--shards", type=int, required=True)
    p_plan.add_argument("--cells-file", help="cells list (text, JSON list, or matrix.json); default: matrix measured + hash-inband")
    p_plan.add_argument("--calibration")
    p_plan.add_argument("--out", help="write plan JSON here instead of stdout")
    p_plan.set_defaults(handler=plan_command)

    p_sweep = sub.add_parser("sweep", help="run the two-arm interleaved A/B sweep")
    add_common_sweep_args(p_sweep)
    p_sweep.set_defaults(handler=sweep_command)

    p_report = sub.add_parser("report", help="per-cell verdicts + phase attribution from results JSONL")
    p_report.add_argument("--results", required=True, help="results file(s), comma-separated to merge")
    p_report.add_argument("--band-file",
                          help="JSON {cell_id or family:side.group: band FRACTION, e.g. 0.03 = 3%%};"
                               " values > 0.5 are rejected as mistaken percentages")
    p_report.add_argument("--no-phases", action="store_true", help="suppress the per-phase table")
    p_report.add_argument("--min-runs", type=int, default=MIN_VERDICT_RUNS,
                          help=f"valid runs/arm below this -> INSUFFICIENT (default {MIN_VERDICT_RUNS})")
    p_report.set_defaults(handler=report_command)

    p_self = sub.add_parser("selftest", help="event-existence and verdict-power self-checks")
    p_self.add_argument("--local", action="store_true")
    p_self.add_argument("--bin", type=abs_path, help="binary to self-test against (resolved to absolute)")
    p_self.add_argument("--check-events", action="store_true")
    p_self.add_argument("--require-amac", nargs="?", const="build,probe", default=None, metavar="SIDES",
                        help="fail-closed unless every listed side's gating AMAC counters are present"
                             " (comma list of: build, probe; bare flag = all sides)")
    p_self.add_argument("--forbid-amac", action="store_true",
                        help="fail-closed if ANY AMAC counter is present")
    p_self.add_argument("--verdict-selftest", action="store_true",
                        help="deliberate A != B pair must verdict non-TIE")
    p_self.set_defaults(handler=selftest_command)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
