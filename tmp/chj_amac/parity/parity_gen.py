#!/usr/bin/env python3
"""parity_gen.py — deterministic query-matrix generator for the parallel_hash
parity harness (Unit 1 of the AMAC + order-preserving probe mission).

Emits JSONL cases to --out (default: cases.jsonl next to this script). Each
case is fully self-contained:

    {id, family, shape, kind, strictness, variant, tags, settings,
     ddl: [stmts], fill: [stmts], query, columns, verdict_tsv, verdict_chk}

Verdict queries:
  (a) verdict_tsv: SELECT * FROM (<q>) ORDER BY ALL SETTINGS ... FORMAT TSV
      -- primary: byte-for-byte diff between the two arms.
  (b) verdict_chk: SELECT count(), sum(cityHash64(...)) FROM (<q>) ...
      -- secondary checksum. cityHash64(NULL) returns NULL (verified
      empirically, see SELFTEST.md), which would silently drop rows from the
      sum, so every column is wrapped as the pair
      (isNull(c), ifNull(toString(c), '')) — collision-free and total.

Determinism-by-construction rules (why arms cannot legitimately diverge):
  * idx -> key mappings are INJECTIVE (bijective integer mixers; strings and
    FixedString embed toString(idx); disjoint prefix classes). This matters
    because ...
  * ... every value column is a function of idx (hence of the key): ANY /
    SEMI / RightAny joins pick an *arbitrary* matching row, so any two rows
    sharing a key must be identical in all projected columns.
  * The build-side asof column t is unique per (key, occurrence), so ASOF
    matches are unique. t is only projected in ASOF cases.
  * The final ORDER BY ALL totally orders the projected rows, so multiset
    equality between arms implies byte equality.

The `FORMAT TSV` suffix of both verdict queries is a contract with
run_parity.sh / parity_driver.py, which splice `INTO OUTFILE '<f>' TRUNCATE`
in front of it. Do not reorder the clause.

Verified against build/reldeb aarch64 (see SELFTEST.md):
  * ANY FULL JOIN is NOT_IMPLEMENTED without
    any_join_distinct_right_table_keys=1 -> FULL/ANY exists only as the
    RightAny variant.
  * All other kind x strictness x variant shapes parse and run under
    join_algorithm='parallel_hash', enable_analyzer=1.

Fill statements use max_threads=1/max_insert_threads=1 (pattern lifted from
/mnt/data/jbmt_results/jbmt-sweep-20260724/join_memory_bench.py FILL_SETTINGS)
so Memory-table block order is deterministic.

Python: stdlib only.
"""

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Units 2-3 contract — the ONE constants block. Update here only.
# AMAC lands in STAGES: the BUILD ring (Unit 2) ships before the PROBE ring
# (Unit 3), so a candidate binary may carry either side's counters, both, or
# none. Every consumer must auto-detect availability PER SIDE and print a
# loud SKIPPED line only when NO side is present.
# ---------------------------------------------------------------------------
AMAC_ENV_VAR = "CLICKHOUSE_JOIN_AMAC"  # values: 0/off, 1/auto, force; read by the server process at start
AMAC_ENGAGEMENT_EVENTS = (
    "ConcurrentHashJoinAmacBuildRows",
    "ConcurrentHashJoinAmacBuildRingGrowths",
    "ConcurrentHashJoinAmacProbeRows",
)
# Per-side assertions under CLICKHOUSE_JOIN_AMAC=force. A side is asserted
# only when ALL of its events exist in the candidate binary (staged landing).
# RingGrowths may be legitimately zero, so it is reported but never asserted.
AMAC_ASSERT_BUILD_EVENTS = ("ConcurrentHashJoinAmacBuildRows",)
AMAC_ASSERT_PROBE_EVENTS = ("ConcurrentHashJoinAmacProbeRows",)
AMAC_ASSERT_SIDES = {"build": AMAC_ASSERT_BUILD_EVENTS, "probe": AMAC_ASSERT_PROBE_EVENTS}
# Backward-compat alias (union of both sides). fleet_ab.py cross-checks
# AMAC_ENGAGEMENT_EVENTS / AMAC_ENV_VAR / SHARED_PROFILE_EVENTS against this
# module; keep those names stable.
AMAC_ASSERT_POSITIVE_EVENTS = AMAC_ASSERT_BUILD_EVENTS + AMAC_ASSERT_PROBE_EVENTS
# Families whose join-map getters are cursor-capable: under force, EVERY
# asserted event of EVERY present side must be > 0 for these.
AMAC_EXPECTED_ENGAGE_FAMILIES = (
    "key32", "key64", "string", "fixstr", "keys128", "keys256", "null64", "nullstr",
)
# Families whose getters are excluded from AMAC (lcstr: LowCardinality;
# mixed: hashed/serialized). Under force their asserted events must be == 0 —
# the exclusions are load-bearing, an engaged excluded family is a FAILURE.
AMAC_EXCLUDED_FAMILIES = ("lcstr", "mixed")
# The seven ProfileEvents shared by both arms (informational engagement audit):
SHARED_PROFILE_EVENTS = (
    "ConcurrentHashJoinBuildMicroseconds",
    "ConcurrentHashJoinBuildDispatchMicroseconds",
    "ConcurrentHashJoinBuildInsertMicroseconds",
    "ConcurrentHashJoinBuildMergeMicroseconds",
    "ConcurrentHashJoinProbeMicroseconds",
    "ConcurrentHashJoinProbeDispatchMicroseconds",
    "ConcurrentHashJoinProbeLookupMicroseconds",
)
# ---------------------------------------------------------------------------

DATABASE = "parity_db"

BASE_SETTINGS = [
    ("join_algorithm", "'parallel_hash'"),
    ("max_bytes_before_external_join", "0"),
    ("max_bytes_ratio_before_external_join", "0"),
    ("enable_join_runtime_filters", "0"),
    ("query_plan_join_swap_table", "false"),
    ("enable_analyzer", "1"),
]

FILL_SETTINGS = "SETTINGS max_threads = 1, max_insert_threads = 1"


def mix64(idx: str, xor_const: int) -> str:
    """Bijective UInt64 mixer: odd-multiplier multiplication mod 2^64 (wraps
    silently in ClickHouse, verified) xor a constant."""
    return f"bitXor(({idx}) * 6364136223846793005, {xor_const})"


def mix64i(idx: str, xor_const: int) -> str:
    """Bijective Int64 mixer (toInt64 reinterprets/wraps, produces negatives)."""
    return f"toInt64({mix64(idx, xor_const)})"


def mix32(idx: str) -> str:
    """Bijective UInt32 mixer (odd multiplier mod 2^32)."""
    return f"toUInt32(bitAnd(({idx}) * 2654435761, 4294967295))"


def string_key(idx: str, salt: int) -> str:
    """Injective String key with the mandated edge cases:
    idx=0 -> empty string; idx%4==1 -> embedded zero byte; idx%4==2 ->
    terminating zero byte; else variable-length hash-prefixed string.
    Classes have disjoint first bytes ('z' / digit / 'k' / empty) and every
    non-empty string embeds toString(idx), so the map is injective."""
    return (
        f"multiIf(({idx}) = 0, '', "
        f"({idx}) % 4 = 1, concat('z', char(0), toString({idx})), "
        f"({idx}) % 4 = 2, concat(toString({idx}), char(0)), "
        f"concat('k', toString(intHash64(bitXor(toUInt64({idx}), {salt}))), '-', toString({idx})))"
    )


def fixstr_key(idx: str) -> str:
    """Injective FixedString(16) key (unique '-<idx>' suffix, left-padded)."""
    return (
        f"toFixedString(leftPad(concat(toString(bitAnd(intHash64(toUInt64({idx})), 255)), "
        f"'-', toString({idx})), 16, '0'), 16)"
    )


def null_wrap(idx: str, expr: str) -> str:
    """~10% NULLs, deterministically on idx % 10 == 0."""
    return f"if(({idx}) % 10 = 0, NULL, {expr})"


def v_expr(idx: str, seed: int) -> str:
    """Value column: function of idx (== function of the key, injectivity),
    small range [0,1023] so the non-equi predicate l.v < r.v is selective."""
    return f"bitAnd(intHash64(bitXor(toUInt64({idx}), {seed})), 1023)"


V_BUILD_SEED = 1315423911
V_PROBE_SEED = 2654435761
T_BUILD_SEED = 97531
T_PROBE_SEED = 86028121


def t_build_expr(idx: str, domain: int) -> str:
    """Build asof column: unique per (key, occurrence) — occ*100000 strides
    dominate the 16-bit hash term, and idx -> key is injective."""
    return f"(intDiv(number, {domain}) * 100000 + bitAnd(intHash64(bitXor(toUInt64({idx}), {T_BUILD_SEED})), 65535))"


def t_probe_expr() -> str:
    """Probe asof column: per-row hash in [0, 131071]; each probe row is
    emitted independently in ASOF joins, so per-row variation is safe."""
    return f"bitAnd(intHash64(bitXor(number, {T_PROBE_SEED})), 131071)"


class Shape:
    """One concrete pair of build/probe tables. `key_exprs(idx)` must be
    injective in idx (see module docstring)."""

    def __init__(self, name, family, key_cols, key_exprs, build_rows, build_domain,
                 probe_rows, probe_domain, has_asof=False, dup=False, extra_tags=()):
        self.name = name
        self.family = family
        self.key_cols = key_cols            # [(col_name, declared_type)]
        self.key_exprs = key_exprs          # callable: idx_sql -> [expr per key col]
        self.build_rows = build_rows
        self.build_domain = build_domain
        self.probe_rows = probe_rows
        self.probe_domain = probe_domain
        self.has_asof = has_asof
        self.dup = dup
        self.extra_tags = list(extra_tags)

    @property
    def build_table(self):
        return f"{DATABASE}.b_{self.name}"

    @property
    def probe_table(self):
        return f"{DATABASE}.p_{self.name}"

    def _cols_ddl(self):
        cols = [f"{n} {t}" for n, t in self.key_cols]
        cols.append("v UInt64")
        if self.has_asof:
            cols.append("t UInt64")
        return ", ".join(cols)

    def ddl(self):
        return [
            f"CREATE DATABASE IF NOT EXISTS {DATABASE};",
            f"CREATE TABLE IF NOT EXISTS {self.build_table} ({self._cols_ddl()}) ENGINE = Memory;",
            f"CREATE TABLE IF NOT EXISTS {self.probe_table} ({self._cols_ddl()}) ENGINE = Memory;",
        ]

    def fill(self):
        b_idx = f"(number % {self.build_domain})"
        p_idx = f"(number % {self.probe_domain})"
        b_sel = [f"{e} AS {n}" for (n, _), e in zip(self.key_cols, self.key_exprs(b_idx))]
        b_sel.append(f"{v_expr(b_idx, V_BUILD_SEED)} AS v")
        p_sel = [f"{e} AS {n}" for (n, _), e in zip(self.key_cols, self.key_exprs(p_idx))]
        p_sel.append(f"{v_expr(p_idx, V_PROBE_SEED)} AS v")
        if self.has_asof:
            b_sel.append(f"{t_build_expr(b_idx, self.build_domain)} AS t")
            p_sel.append(f"{t_probe_expr()} AS t")
        return [
            f"TRUNCATE TABLE {self.build_table};",
            f"INSERT INTO {self.build_table} SELECT {', '.join(b_sel)} "
            f"FROM numbers({self.build_rows}) {FILL_SETTINGS};",
            f"TRUNCATE TABLE {self.probe_table};",
            f"INSERT INTO {self.probe_table} SELECT {', '.join(p_sel)} "
            f"FROM numbers({self.probe_rows}) {FILL_SETTINGS};",
        ]


def make_shapes():
    # Family names here map to the fleet vocabulary (fleet_ab.py FAMILIES /
    # MATRIX.md) as: string<->str, keys128<->k128, keys256<->k256; fleet's
    # strzero has NO family here -- its parity evidence is the shapes tagged
    # 'zero-bytes' (string, nullstr, mixed_*), which is what MATRIX.md's
    # "PARITY-ONLY ... all strzero cells" disposition points at. nullstr has
    # no fleet counterpart (parity-only coverage); other names coincide.
    S = []
    # -- key32: UInt32, plus the UInt8 low-cardinality variant to hit key8 --
    S.append(Shape("key32", "key32", [("k", "UInt32")],
                   lambda i: [mix32(i)],
                   24000, 18000, 64000, 36000, has_asof=True))
    S.append(Shape("key8", "key32", [("k", "UInt8")],
                   lambda i: [f"toUInt8({i})"],
                   10000, 200, 50000, 256, has_asof=True,
                   extra_tags=["low-cardinality-key8"]))
    # -- key64: UInt64 and Int64 --
    S.append(Shape("key64u", "key64", [("k", "UInt64")],
                   lambda i: [mix64(i, 0x9E3779B97F4A7C15)],
                   30000, 22500, 80000, 45000, has_asof=True))
    S.append(Shape("key64i", "key64", [("k", "Int64")],
                   lambda i: [mix64i(i, 0x1442695040888963)],
                   30000, 22500, 80000, 45000, has_asof=True))
    # -- string: variable length, embedded zero, terminating zero, empty --
    S.append(Shape("string", "string", [("k", "String")],
                   lambda i: [string_key(i, 1000003)],
                   20000, 15000, 60000, 30000,
                   extra_tags=["zero-bytes", "empty-string", "variable-length"]))
    # -- fixstr --
    S.append(Shape("fixstr", "fixstr", [("k", "FixedString(16)")],
                   lambda i: [fixstr_key(i)],
                   20000, 15000, 60000, 30000))
    # -- keys128 / keys256 --
    S.append(Shape("keys128", "keys128", [("k1", "UInt64"), ("k2", "UInt64")],
                   lambda i: [mix64(i, 0x8B72E7B4C2A5F0D3), mix64(i, 0x3C6EF372FE94F82B)],
                   30000, 22500, 80000, 45000, has_asof=True))
    S.append(Shape("keys256", "keys256",
                   [("k1", "UInt64"), ("k2", "UInt64"), ("k3", "UInt64"), ("k4", "UInt64")],
                   lambda i: [mix64(i, 0x0123456789ABCDEF), mix64(i, 0xFEDCBA9876543210),
                              mix64(i, 0x0F1E2D3C4B5A6978), mix64(i, 0x13198A2E03707344)],
                   20000, 15000, 60000, 30000, has_asof=True))
    # -- nullable --
    S.append(Shape("null64", "null64", [("k", "Nullable(UInt64)")],
                   lambda i: [null_wrap(i, mix64(i, 0x243F6A8885A308D3))],
                   24000, 18000, 64000, 36000, extra_tags=["nulls-10pct"]))
    S.append(Shape("nullstr", "nullstr", [("k", "Nullable(String)")],
                   lambda i: [null_wrap(i, string_key(i, 2000003))],
                   20000, 15000, 60000, 30000,
                   extra_tags=["nulls-10pct", "zero-bytes", "empty-string"]))
    # -- LowCardinality(String): small domain, heavy natural duplication --
    S.append(Shape("lcstr", "lcstr", [("k", "LowCardinality(String)")],
                   lambda i: [f"concat('lc_', toString({i}))"],
                   10000, 500, 50000, 1000))
    # -- mixed composite keys -> hashed/serialized methods --
    S.append(Shape("mixed_us", "mixed", [("k1", "UInt32"), ("k2", "String")],
                   lambda i: [mix32(i), string_key(i, 3000017)],
                   20000, 15000, 60000, 30000,
                   extra_tags=["zero-bytes", "empty-string"]))
    S.append(Shape("mixed_sn", "mixed", [("k1", "String"), ("k2", "Nullable(UInt64)")],
                   lambda i: [string_key(i, 4000037), null_wrap(i, mix64(i, 0xA4093822299F31D0))],
                   20000, 15000, 60000, 30000,
                   extra_tags=["nulls-10pct", "zero-bytes", "empty-string"]))
    # -- duplicate-heavy builds (dup factor 16) for key64/string/keys128 --
    S.append(Shape("key64u_dup16", "key64", [("k", "UInt64")],
                   lambda i: [mix64(i, 0x9E3779B97F4A7C15)],
                   32000, 2000, 80000, 4000, dup=True, extra_tags=["dup16"]))
    S.append(Shape("string_dup16", "string", [("k", "String")],
                   lambda i: [string_key(i, 1000003)],
                   32000, 2000, 80000, 4000, dup=True,
                   extra_tags=["dup16", "zero-bytes", "empty-string"]))
    S.append(Shape("keys128_dup16", "keys128", [("k1", "UInt64"), ("k2", "UInt64")],
                   lambda i: [mix64(i, 0x8B72E7B4C2A5F0D3), mix64(i, 0x3C6EF372FE94F82B)],
                   32000, 2000, 80000, 4000, dup=True, extra_tags=["dup16"]))
    return S


# (kind, strictness, variant) combos. Validity verified empirically on
# build/reldeb (see SELFTEST.md): FULL/ANY exists ONLY as the RightAny
# variant (NOT_IMPLEMENTED otherwise).
STD_COMBOS = [
    ("INNER", "ALL", "std"), ("INNER", "ANY", "std"),
    ("LEFT", "ALL", "std"), ("LEFT", "ANY", "std"),
    ("LEFT", "SEMI", "std"), ("LEFT", "ANTI", "std"),
    ("RIGHT", "ALL", "std"), ("RIGHT", "ANY", "std"),
    ("RIGHT", "SEMI", "std"), ("RIGHT", "ANTI", "std"),
    ("FULL", "ALL", "std"),
]
RIGHTANY_COMBOS = [
    ("INNER", "ANY", "rightany"), ("LEFT", "ANY", "rightany"),
    ("RIGHT", "ANY", "rightany"), ("FULL", "ANY", "rightany"),
]
# Non-equi extra ON condition (l.v < r.v) for ANY/SEMI/ANTI -> MapsAll:
NONEQUI_COMBOS = [
    ("INNER", "ANY", "nonequi"), ("LEFT", "ANY", "nonequi"),
    ("LEFT", "SEMI", "nonequi"), ("LEFT", "ANTI", "nonequi"),
    ("RIGHT", "SEMI", "nonequi"), ("RIGHT", "ANTI", "nonequi"),
]
ASOF_COMBOS = [("INNER", "ASOF", "std"), ("LEFT", "ASOF", "std")]

THREADS_CHOICES = (4, 32)


def render_settings(settings):
    return ", ".join(f"{k} = {v}" for k, v in settings)


def checksum_args(col_names):
    parts = []
    for c in col_names:
        parts.append(f"isNull({c})")
        parts.append(f"ifNull(toString({c}), '')")
    return ", ".join(parts)


def make_case(shape, kind, strictness, variant, jun, threads):
    proj = []
    columns = []
    for n, t in shape.key_cols:
        proj.append(f"l.{n} AS l_{n}")
        columns.append({"name": f"l_{n}", "type": t})
    for n, t in shape.key_cols:
        proj.append(f"r.{n} AS r_{n}")
        columns.append({"name": f"r_{n}", "type": t})
    proj.append("l.v AS l_v")
    columns.append({"name": "l_v", "type": "UInt64"})
    proj.append("r.v AS r_v")
    columns.append({"name": "r_v", "type": "UInt64"})

    on = [f"l.{n} = r.{n}" for n, _ in shape.key_cols]
    if variant == "nonequi":
        on.append("l.v < r.v")
    if strictness == "ASOF":
        proj.append("l.t AS l_t")
        columns.append({"name": "l_t", "type": "UInt64"})
        proj.append("r.t AS r_t")
        columns.append({"name": "r_t", "type": "UInt64"})
        on.append("l.t >= r.t")

    join_spec = f"{strictness} {kind} JOIN"
    query = (
        f"SELECT {', '.join(proj)} FROM {shape.probe_table} AS l "
        f"{join_spec} {shape.build_table} AS r ON {' AND '.join(on)}"
    )

    settings = list(BASE_SETTINGS)
    settings.append(("max_threads", str(threads)))
    settings.append(("join_use_nulls", str(jun)))
    if variant == "rightany":
        settings.append(("any_join_distinct_right_table_keys", "1"))
    settings_sql = render_settings(settings)

    col_names = [c["name"] for c in columns]
    verdict_tsv = f"SELECT * FROM ({query}) ORDER BY ALL SETTINGS {settings_sql} FORMAT TSV"
    verdict_chk = (
        f"SELECT count() AS cnt, sum(cityHash64({checksum_args(col_names)})) AS chk "
        f"FROM ({query}) SETTINGS {settings_sql} FORMAT TSV"
    )

    case_id = f"{shape.name}.{kind.lower()}.{strictness.lower()}.{variant}.jun{jun}.t{threads}"
    tags = [shape.family, shape.name, variant, f"jun{jun}", f"t{threads}",
            "expect-parallel-hash"] + shape.extra_tags
    return {
        "id": case_id,
        "family": shape.family,
        "shape": shape.name,
        "kind": kind,
        "strictness": strictness,
        "variant": variant,
        "tags": tags,
        "settings": {k: v.strip("'") for k, v in settings},
        "ddl": shape.ddl(),
        "fill": shape.fill(),
        "query": query,
        # NOTE: declared base types; join_use_nulls=1 wraps the non-matched
        # side in Nullable at output. Informational only — the runner diffs
        # bytes, it does not interpret types.
        "columns": columns,
        "verdict_tsv": verdict_tsv,
        "verdict_chk": verdict_chk,
    }


def generate():
    cases = []
    for shape in make_shapes():
        if shape.dup:
            combos = list(STD_COMBOS)
        else:
            combos = STD_COMBOS + RIGHTANY_COMBOS + NONEQUI_COMBOS
            if shape.has_asof:
                combos = combos + ASOF_COMBOS
        for jun in (0, 1):
            for ci, (kind, strictness, variant) in enumerate(combos):
                threads = THREADS_CHOICES[(ci + jun) % 2]
                cases.append(make_case(shape, kind, strictness, variant, jun, threads))
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    assert 600 <= len(cases) <= 900, f"case count {len(cases)} outside target 600-900"
    return cases


def print_stats(cases, out=sys.stderr):
    fams = {}
    combos = set()
    threads = {4: 0, 32: 0}
    for c in cases:
        fams[c["family"]] = fams.get(c["family"], 0) + 1
        combos.add((c["kind"], c["strictness"], c["variant"]))
        threads[int(c["settings"]["max_threads"])] += 1
    print(f"cases: {len(cases)}", file=out)
    print(f"families ({len(fams)}): " + ", ".join(f"{k}={v}" for k, v in sorted(fams.items())), file=out)
    print(f"kind-strictness-variant combos: {len(combos)}", file=out)
    print(f"threads: t4={threads[4]} t32={threads[32]}", file=out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=None, help="output JSONL path (default: cases.jsonl next to this script)")
    p.add_argument("--stats", action="store_true", help="print matrix statistics to stderr")
    p.add_argument("--print-contract", action="store_true",
                   help="print the Units 2-3 contract constants (for shell consumption) and exit")
    args = p.parse_args()

    if args.print_contract:
        print(f"AMAC_ENV_VAR={AMAC_ENV_VAR}")
        for e in AMAC_ENGAGEMENT_EVENTS:
            print(f"AMAC_EVENT={e}")
        for e in AMAC_ASSERT_POSITIVE_EVENTS:
            print(f"AMAC_ASSERT_EVENT={e}")  # backward-compat: union of both sides
        for e in AMAC_ASSERT_BUILD_EVENTS:
            print(f"AMAC_ASSERT_BUILD_EVENT={e}")
        for e in AMAC_ASSERT_PROBE_EVENTS:
            print(f"AMAC_ASSERT_PROBE_EVENT={e}")
        for f in AMAC_EXPECTED_ENGAGE_FAMILIES:
            print(f"AMAC_EXPECTED_FAMILY={f}")
        for f in AMAC_EXCLUDED_FAMILIES:
            print(f"AMAC_EXCLUDED_FAMILY={f}")
        for e in SHARED_PROFILE_EVENTS:
            print(f"SHARED_EVENT={e}")
        return 0

    import os
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.jsonl")
    cases = generate()
    with open(out_path, "w") as f:
        for c in cases:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases to {out_path}", file=sys.stderr)
    if args.stats:
        print_stats(cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
