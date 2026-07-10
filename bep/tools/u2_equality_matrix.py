#!/usr/bin/env python3
"""U2 acceptance-test driver: join-algorithm result-equality matrix.

Implements the pre-registered result-equality matrix from bep/prereg.md
(section "## U2", "Pre-registered result-equality matrix"): compares a
candidate `join_algorithm` (default `radix_join`) against a baseline
(default `hash`) on deterministic, purely-SQL-generated data. PASS iff
the outputs of the two runs are byte-identical.

Everything is stdlib-only and needs no server: each (point, algorithm)
run is one `clickhouse local` invocation with all data generated from
`numbers_mt()` subqueries (no rand(), no persistent state).

Matrix axes
-----------
* Key configs (packed width -> concrete types), 8 configs in the full cross:
    4B:  k UInt32
    8B:  k UInt64
    16B: k FixedString(16)            and (k1, k2 UInt64)
    32B: k FixedString(32)            and (k1..k4 UInt64)
    64B: k FixedString(64)            and (k1..k8 UInt64)
  plus a 9th config used only in the 1e8 subset:
    12B: (k1 UInt64, k2 UInt32)
* Duplicates: unique | x8 | skew (see skew formula below).
* Hit rates: 1.0 | 0.5 | 0.05 (exact fractions of probe rows that hit).
* Build sizes: full cross at 1e5 and 1e7; at 1e8 only the pre-registered
  subset {8B unique hit1.0, 8B x8 hit0.05, 64B-FixedString unique hit0.5,
  (UInt64,UInt32) skew hit0.5}.
* Probe rows = 2 x build rows.
* Threads: every 1e5 point runs at max_threads=1 AND max_threads=32
  (baseline is re-run at the same thread count); 1e7/1e8 run at 32.
* Edge cases (1e5 tier only): empty build, empty probe, hit rate 0,
  one-row build, one WITH TOTALS query, one extremes=1 query (full
  output including totals/extremes blocks is byte-compared).

Data-generation formulas (all deterministic)
--------------------------------------------
Build key id per duplicate mode (over build row number in [0, N)):
    unique: key_id = number                                (K = N distinct)
    x8:     key_id = intDiv(number, 8)                     (K = ceil(N/8), each key ~8 times)
    skew:   key_id = intDiv(number, 64) * 8
                     + floor(sqrt(number % 64))            (quadratic bucketing)
  Skew properties: within each block of 64 consecutive build rows there are
  8 distinct keys; key j of the block has multiplicity |[j^2,(j+1)^2) cap [0,64)|
  = {1,3,5,7,9,11,13,15}. Max multiplicity = 15, mean = 8.
  K = intDiv(N-1, 64) * 8 + isqrt((N-1) % 64) + 1 (for N > 0).

Probe key id (over probe row number in [0, 2N)), hit period p in {1, 2, 20}
for hit rates {1.0, 0.5, 0.05} (2N is always divisible by p, so fractions
are exact):
    hit  rows (number % p == 0): key_id = intDiv(number, p) % K   (inside build space)
    miss rows:                   key_id = 2^31 + number           (outside build space:
                                          all build key ids are < 2^31, all miss ids >= 2^31)

Key encodings from key_id (injective within the id domain [0, 2^31 + 2N)):
    UInt32:          toUInt32(key_id)               (all ids < 2^32)
    UInt64:          key_id
    FixedString(W):  toFixedString(rightPad(leftPad(toString(key_id), 10, '0'), W, '.'), W)
                     i.e. 10-digit zero-padded decimal of key_id, right-padded
                     with '.' to width W. Injective (all ids < 1e10).
    m x UInt64:      k_i = key_id * C_i mod 2^64, C_1 = 1 and C_2.. odd 64-bit
                     constants (odd multiplier mod 2^64 is bijective).
    (UInt64,UInt32): k1 = key_id, k2 = toUInt32(bitAnd(key_id * 2654435761, 0xFFFFFFFF)).

Payloads: build `b_p = number * 2654435761`, probe `p_p = number * 2654435761`
(each over its own side's `number`, so duplicate keys carry distinct payloads).

Oracle per point
----------------
    SELECT count(), sum(cityHash64(<keys>, b_p, p_p)), groupBitXor(cityHash64(<keys>, b_p, p_p))
    FROM (<probe>) AS p INNER JOIN (<build>) AS b USING (<keys>)
    SETTINGS join_algorithm='<algo>', max_threads=<t>, ...
All three aggregates are order-independent, so the single-row output must be
byte-identical between the candidate and the baseline.

Exit status: 0 iff every executed point is PASS; 1 otherwise.
"""

import argparse
import concurrent.futures
import datetime
import math
import os
import subprocess
import sys
import threading
import time

DEFAULT_BINARY = "/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse"
MISS_BASE = 2 ** 31  # all build key ids < 2^31; all miss ids >= 2^31
PAYLOAD_MULT = 2654435761  # Knuth multiplicative-hash constant (odd)
DEFAULT_MAX_MEMORY = 20_000_000_000  # ~20 GB per clickhouse-local process

# Odd 64-bit multipliers for the multi-UInt64 key configs (C_1 = 1 keeps k1 = key_id).
ODD64 = [
    1,
    11400714785074694791,  # xxhash PRIME64_1
    14029467366897019727,  # xxhash PRIME64_2
    1609587929392839161,   # xxhash PRIME64_3
    9650029242287828579,   # xxhash PRIME64_4
    2870177450012600261,   # xxhash PRIME64_5
    11400714819323198485,  # splitmix64 golden gamma
    18397679294719823053,  # murmur3 fmix64 c1
]


def fs_cols(width):
    return [("k", lambda kid, w=width:
             f"toFixedString(rightPad(leftPad(toString({kid}), 10, '0'), {w}, '.'), {w})")]


def multi_u64_cols(m):
    cols = [("k1", lambda kid: kid)]
    for i in range(1, m):
        cols.append((f"k{i + 1}",
                     lambda kid, c=ODD64[i]: f"toUInt64({kid} * {c})"))
    return cols


# name -> list of (column_name, expr_builder(key_id_expr) -> SQL)
KEY_CONFIGS = {
    "4B_u32":     [("k", lambda kid: f"toUInt32({kid})")],
    "8B_u64":     [("k", lambda kid: kid)],
    "16B_fs16":   fs_cols(16),
    "16B_2xu64":  multi_u64_cols(2),
    "32B_fs32":   fs_cols(32),
    "32B_4xu64":  multi_u64_cols(4),
    "64B_fs64":   fs_cols(64),
    "64B_8xu64":  multi_u64_cols(8),
    "12B_u64u32": [("k1", lambda kid: kid),
                   ("k2", lambda kid: f"toUInt32(bitAnd({kid} * {PAYLOAD_MULT}, 4294967295))")],
}

FULL_CROSS_CONFIGS = ["4B_u32", "8B_u64", "16B_fs16", "16B_2xu64",
                      "32B_fs32", "32B_4xu64", "64B_fs64", "64B_8xu64"]
DUP_MODES = ["unique", "x8", "skew"]
HIT_RATES = [1.0, 0.5, 0.05]

# Pre-registered 1e8 subset (prereg.md): the last point uses the 9th key config.
SUBSET_1E8 = [
    ("8B_u64", "unique", 1.0),
    ("8B_u64", "x8", 0.05),
    ("64B_fs64", "unique", 0.5),
    ("12B_u64u32", "skew", 0.5),
]

HIT_PERIOD = {1.0: 1, 0.5: 2, 0.05: 20}


def build_key_id_expr(dups):
    if dups == "unique":
        return "toUInt64(number)"
    if dups == "x8":
        return "toUInt64(intDiv(number, 8))"
    if dups == "skew":
        return ("toUInt64(intDiv(number, 64) * 8 "
                "+ toUInt64(floor(sqrt(toFloat64(number % 64)))))")
    raise ValueError(dups)


def distinct_keys(dups, n):
    if n == 0:
        return 0
    if dups == "unique":
        return n
    if dups == "x8":
        return (n + 7) // 8
    if dups == "skew":
        return (n - 1) // 64 * 8 + math.isqrt((n - 1) % 64) + 1
    raise ValueError(dups)


def probe_key_id_expr(hit_rate, k_distinct):
    if k_distinct == 0 or hit_rate == 0.0:
        return f"toUInt64({MISS_BASE} + number)"
    p = HIT_PERIOD[hit_rate]
    if p == 1:
        return f"toUInt64(number % {k_distinct})"
    return (f"toUInt64(if(number % {p} = 0, "
            f"intDiv(number, {p}) % {k_distinct}, {MISS_BASE} + number))")


def side_subquery(cfg_name, key_id_expr, payload_col, rows):
    cols = ", ".join(f"{fn(key_id_expr)} AS {name}"
                     for name, fn in KEY_CONFIGS[cfg_name])
    return (f"SELECT {cols}, number * {PAYLOAD_MULT} AS {payload_col} "
            f"FROM numbers_mt({rows})")


# Extra settings appended to every query's SETTINGS clause (both candidate and
# baseline), set once from --extra-settings in main(). Empty = no extras
# (backward-compatible default).
EXTRA_SETTINGS = ""


def settings_clause(algo, threads, max_memory):
    clause = (f"SETTINGS join_algorithm = '{algo}', max_threads = {threads}, "
              f"max_memory_usage = {max_memory}, enable_analyzer = 1, "
              f"query_plan_join_swap_table = 'false'")
    if EXTRA_SETTINGS:
        clause += ", " + ", ".join(
            s.strip() for s in EXTRA_SETTINGS.split(",") if s.strip())
    return clause


def join_from_clause(cfg_name, build_rows, dups, hit_rate, probe_rows=None):
    key_names = [name for name, _ in KEY_CONFIGS[cfg_name]]
    if probe_rows is None:
        probe_rows = 2 * build_rows
    k = distinct_keys(dups, build_rows)
    probe = side_subquery(cfg_name, probe_key_id_expr(hit_rate, k), "p_p", probe_rows)
    build = side_subquery(cfg_name, build_key_id_expr(dups), "b_p", build_rows)
    using = ", ".join(key_names)
    return (f"FROM ({probe}) AS p INNER JOIN ({build}) AS b USING ({using})",
            key_names)


def oracle_query(cfg_name, build_rows, dups, hit_rate, algo, threads,
                 max_memory, probe_rows=None):
    frm, key_names = join_from_clause(cfg_name, build_rows, dups, hit_rate, probe_rows)
    h = f"cityHash64({', '.join(key_names)}, b_p, p_p)"
    return (f"SELECT count() AS cnt, sum({h}) AS s, groupBitXor({h}) AS x "
            f"{frm} {settings_clause(algo, threads, max_memory)}")


def totals_query(build_rows, algo, threads, max_memory):
    frm, _ = join_from_clause("8B_u64", build_rows, "unique", 0.5)
    h = "cityHash64(k, b_p, p_p)"
    return (f"SELECT intDiv(k, {max(build_rows // 10, 1)}) AS g, "
            f"count() AS c, sum({h}) AS s, groupBitXor({h}) AS x "
            f"{frm} GROUP BY g WITH TOTALS ORDER BY g "
            f"{settings_clause(algo, threads, max_memory)}")


def extremes_query(build_rows, algo, threads, max_memory):
    frm, _ = join_from_clause("8B_u64", build_rows, "unique", 0.5)
    return (f"SELECT cityHash64(k, b_p, p_p) AS h {frm} "
            f"ORDER BY h LIMIT 10 "
            f"{settings_clause(algo, threads, max_memory)}, extremes = 1")


def size_tag(n):
    if n > 0 and 10 ** int(math.log10(n)) == n:
        return f"1e{int(math.log10(n))}"
    return str(n)


class Point:
    """One matrix cell at one thread count. `query(algo)` builds the SQL."""

    def __init__(self, point_id, key_config, dups, hit_rate, build_rows,
                 threads, query_builder):
        self.point_id = point_id
        self.key_config = key_config
        self.dups = dups
        self.hit_rate = hit_rate
        self.build_rows = build_rows
        self.threads = threads
        self.query = query_builder  # algo -> SQL


def enumerate_points(sizes, max_memory):
    points = []

    def add(point_id, cfg, dups, hit, n, threads, builder):
        points.append(Point(point_id, cfg, dups, hit, n, threads, builder))

    for n in sorted(sizes):
        tag = size_tag(n)
        threads_list = [1, 32] if n == 100_000 else [32]
        if n == 100_000_000:
            cells = SUBSET_1E8
        else:
            cells = [(cfg, dups, hit)
                     for cfg in FULL_CROSS_CONFIGS
                     for dups in DUP_MODES
                     for hit in HIT_RATES]
        for cfg, dups, hit in cells:
            for t in threads_list:
                pid = f"{tag}_{cfg}_{dups}_h{hit:.2f}_t{t}"
                add(pid, cfg, dups, hit, n, t,
                    lambda algo, cfg=cfg, n=n, dups=dups, hit=hit, t=t:
                    oracle_query(cfg, n, dups, hit, algo, t, max_memory))

        if n == 100_000:
            for t in threads_list:
                add(f"{tag}_edge_empty_build_t{t}", "8B_u64", "edge:empty_build",
                    0.0, 0, t,
                    lambda algo, t=t: oracle_query(
                        "8B_u64", 0, "unique", 0.0, algo, t, max_memory,
                        probe_rows=200_000))
                add(f"{tag}_edge_empty_probe_t{t}", "8B_u64", "edge:empty_probe",
                    1.0, n, t,
                    lambda algo, n=n, t=t: oracle_query(
                        "8B_u64", n, "unique", 1.0, algo, t, max_memory,
                        probe_rows=0))
                add(f"{tag}_edge_all_miss_t{t}", "8B_u64", "edge:all_miss",
                    0.0, n, t,
                    lambda algo, n=n, t=t: oracle_query(
                        "8B_u64", n, "unique", 0.0, algo, t, max_memory))
                add(f"{tag}_edge_one_row_build_t{t}", "8B_u64", "edge:one_row_build",
                    0.5, 1, t,
                    lambda algo, t=t: oracle_query(
                        "8B_u64", 1, "unique", 0.5, algo, t, max_memory,
                        probe_rows=200_000))
                add(f"{tag}_edge_with_totals_t{t}", "8B_u64", "edge:with_totals",
                    0.5, n, t,
                    lambda algo, n=n, t=t: totals_query(n, algo, t, max_memory))
                add(f"{tag}_edge_extremes_t{t}", "8B_u64", "edge:extremes",
                    0.5, n, t,
                    lambda algo, n=n, t=t: extremes_query(n, algo, t, max_memory))
    return points


def run_clickhouse_local(binary, query, timeout):
    """Returns (rc, stdout_bytes, stderr_text). rc None means timeout."""
    try:
        proc = subprocess.run(
            [binary, "local", "--query", query],
            capture_output=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return None, b"", f"timeout after {timeout}s"


def tsv_escape(s, limit=2000):
    if len(s) > limit:
        s = s[:limit] + "...(truncated)"
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\r", "\\r").replace("\n", "\\n"))


def run_point(point, args):
    t0 = time.monotonic()
    cand_rc, cand_out, cand_err = run_clickhouse_local(
        args.binary, point.query(args.candidate), args.timeout)
    base_rc, base_out, base_err = run_clickhouse_local(
        args.binary, point.query(args.baseline), args.timeout)
    seconds = time.monotonic() - t0

    if cand_rc != 0 or base_rc != 0:
        status = "ERROR"
        cand_repr = (cand_out.decode("utf-8", "replace") if cand_rc == 0
                     else f"rc={cand_rc}: {cand_err.strip()}")
        base_repr = (base_out.decode("utf-8", "replace") if base_rc == 0
                     else f"rc={base_rc}: {base_err.strip()}")
    else:
        status = "PASS" if cand_out == base_out else "FAIL"
        cand_repr = cand_out.decode("utf-8", "replace")
        base_repr = base_out.decode("utf-8", "replace")
    return status, cand_repr, base_repr, seconds


def main():
    ap = argparse.ArgumentParser(
        description="U2 join-algorithm result-equality matrix "
                    "(see bep/prereg.md, section U2).")
    ap.add_argument("--candidate", default="radix_join",
                    help="candidate join_algorithm (default: radix_join)")
    ap.add_argument("--baseline", default="hash",
                    help="baseline join_algorithm (default: hash)")
    ap.add_argument("--sizes", default="1e5,1e7,1e8",
                    help="comma-separated build sizes (default: 1e5,1e7,1e8)")
    ap.add_argument("--filter", default="",
                    help="comma-separated substrings; keep points whose "
                         "point_id contains any of them")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel clickhouse-local point runners (default: 4)")
    ap.add_argument("--out", default="",
                    help="output TSV path (default: "
                         "bep/tools/results/u2_matrix_<candidate>_<timestamp>.tsv)")
    ap.add_argument("--binary", default=DEFAULT_BINARY,
                    help=f"clickhouse binary (default: {DEFAULT_BINARY})")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="per-invocation timeout in seconds (default: 3600)")
    ap.add_argument("--max-memory", type=int, default=DEFAULT_MAX_MEMORY,
                    help=f"max_memory_usage per process (default: {DEFAULT_MAX_MEMORY})")
    ap.add_argument("--extra-settings", default="",
                    help="comma-separated key=value settings appended to every "
                         "query's SETTINGS clause, candidate and baseline alike "
                         "(e.g. 'radix_join_probe_buffer_fraction=0,"
                         "radix_join_probe_buffer_min_bytes=1')")
    ap.add_argument("--list", action="store_true",
                    help="list selected point ids and exit")
    ap.add_argument("--print-query", default="",
                    help="print the candidate SQL of the point with this exact "
                         "point_id and exit")
    args = ap.parse_args()

    sizes = [int(float(s)) for s in args.sizes.split(",") if s.strip()]
    points = enumerate_points(sizes, args.max_memory)
    if args.filter:
        subs = [f.strip() for f in args.filter.split(",") if f.strip()]
        points = [p for p in points if any(s in p.point_id for s in subs)]

    if args.list:
        for p in points:
            print(p.point_id)
        return 0
    if args.print_query:
        for p in points:
            if p.point_id == args.print_query:
                print(p.query(args.candidate))
                return 0
        print(f"no point with id {args.print_query}", file=sys.stderr)
        return 2

    if not points:
        print("no points selected", file=sys.stderr)
        return 2
    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}", file=sys.stderr)
        return 2

    out_path = args.out
    if not out_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results",
            f"u2_matrix_{args.candidate}_{ts}.tsv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    lock = threading.Lock()
    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    done = [0]
    total = len(points)
    t_start = time.monotonic()

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("point_id\tkey_config\tdups\thit_rate\tbuild_rows\tthreads"
                  "\tstatus\tcandidate_result\tbaseline_result\tseconds\n")
        out.flush()

        def work(point):
            status, cand, base, seconds = run_point(point, args)
            with lock:
                counts[status] += 1
                done[0] += 1
                out.write("\t".join([
                    point.point_id, point.key_config, point.dups,
                    f"{point.hit_rate:.2f}", str(point.build_rows),
                    str(point.threads), status,
                    tsv_escape(cand), tsv_escape(base), f"{seconds:.2f}"]) + "\n")
                out.flush()
                print(f"[{done[0]}/{total}] {status:5s} {point.point_id} "
                      f"({seconds:.1f}s)", file=sys.stderr)
            return status

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(work, points))

    elapsed = time.monotonic() - t_start
    print(f"\ncandidate={args.candidate} baseline={args.baseline} "
          f"points={total} PASS={counts['PASS']} FAIL={counts['FAIL']} "
          f"ERROR={counts['ERROR']} elapsed={elapsed:.1f}s\nresults: {out_path}",
          file=sys.stderr)
    return 0 if counts["FAIL"] == 0 and counts["ERROR"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
