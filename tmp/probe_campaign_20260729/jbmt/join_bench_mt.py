#!/usr/bin/env python3
"""MergeTree join benchmark: `partitioned_hash` vs `parallel_hash` on persistent tables.

Successor to `join_memory_bench.py` (imported as a library for the shared key
machinery) with four structural changes:
  1. every table is a real `MergeTree` table (`OPTIMIZE FINAL`-ed, merges
     stopped and part counts asserted stable during timing);
  2. two workload families: REAL dataset joins (TPC-H / TPC-DS / CoffeeShop /
     StackOverflow at tier `a` or `b`; ~190 pairwise join specs extracted from
     the source benchmarks, see `join_bench_mt_queries.json`) and SYNTHETIC
     cells (the controlled-axis grid, including the coverage-gap groups:
     tiny dimensions, huge probe multiplicity, filtered dims, m:n builds,
     `UInt32`/`FixedString(2)` keys, join kinds);
  3. the FULL ProfileEvents map of every timed run is recorded (no curated
     allow-list), so any phase counter in the binary is analyzable later;
  4. peak memory is a first-class scored axis in `report` (win/tie/loss with
     the same noise band as wall time).

Correctness: synthetic INNER cells keep the closed-form expected row count;
everything else is gated on cross-algorithm agreement of (row_count, checksum).
A run where the intended algorithm's path event is zero is recorded as
FALLBACK, never silently measured.

v2 adds A/B ARMS (`--arm NAME=BINARY:PORT`, repeatable): one server per
binary, both up at once, and every unit's timed runs interleaved strict ABAB
across the arms with the leading arm alternating per unit. An arm switch is
just a client invocation against the other port, so interleaving costs
nothing over the old one-binary pass; the offline join of two passes is
replaced by per-arm blocks in each result row (see `report-ab`).
`--algorithms` restricts what is measured (default: both), e.g.
`--algorithms parallel_hash` for a pure A/B of one algorithm across two
binaries - at the price of the cross-algorithm result oracle. The
(row_count, checksum) reference is shared across arms, so a cross-binary
disagreement is INVALID, never scored. Servers are managed by
`join_bench_mt_servers.sh`: the second arm's data root is a `cp -al`
hardlink clone of the snapshot-restored root on the SAME volume (parts are
immutable; shared inodes mean one EBS hydration and one shared page cache).

Deliberately NOT implemented in v1 (recorded so nobody assumes coverage):
Zipf multiplicity distributions, nulls_pct other than {0,10}, the CoffeeShop
SCD-2 band residual (the band spec is carried in the manifest and applied as a
post-join WHERE only), cold-cache measurement.

Only the Python standard library is used. `join_memory_bench.py` and the two
JSON files must sit next to this file on every instance.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
import zlib

TOOL_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))
import join_memory_bench as jmb  # noqa: E402

QUERIES_JSON = TOOL_DIR / "join_bench_mt_queries.json"
SCHEMAS_JSON = TOOL_DIR / "join_bench_mt_schemas.json"

ALGORITHMS = ("partitioned_hash", "parallel_hash")
META_DB = "jbmt_meta"
FINGERPRINTS_TABLE = f"{META_DB}.fingerprints"
SYN_DB = "bench"  # jmb.BUILD_TABLE / jmb.PROBE_TABLE live here
TOOL_VERSION = "jbmt-v2"

# Tier definitions. `so_copies=2` doubles StackOverflow by re-inserting every
# table with all id columns offset by SO_2X_OFFSET (joins stay consistent).
TIERS = {
    "a": {"tpch_sf": 40, "tpcds_sf": 32, "coffeeshop_scale": "500m", "so_copies": 1},
    "b": {"tpch_sf": 100, "tpcds_sf": 64, "coffeeshop_scale": "1b", "so_copies": 2},
    "smoke": {"tpch_sf": 1, "tpcds_sf": 1, "coffeeshop_scale": "500m", "so_copies": 1},
}
SO_2X_OFFSET = 1_000_000_000

COFFEESHOP_BUCKET = "https://clickhouse-datasets.s3.amazonaws.com/coffeeshop"
SO_BUCKET = "https://datasets-documentation.s3.eu-west-3.amazonaws.com/stackoverflow/parquet"
SO_SOURCES = {
    "posts": f"{SO_BUCKET}/posts/*.parquet",
    "users": f"{SO_BUCKET}/users.parquet",
    "votes": f"{SO_BUCKET}/votes/*.parquet",
    "comments": f"{SO_BUCKET}/comments/*.parquet",
    "badges": f"{SO_BUCKET}/badges.parquet",
    "postlinks": f"{SO_BUCKET}/postlinks.parquet",
}
# id-bearing columns per StackOverflow table, offset for the tier-b second copy.
SO_ID_COLS = {
    "posts": ["Id", "AcceptedAnswerId", "OwnerUserId", "LastEditorUserId"],
    "users": ["Id", "AccountId"],
    "votes": ["Id", "PostId", "UserId"],
    "comments": ["Id", "PostId", "UserId"],
    "badges": ["Id", "UserId"],
    "postlinks": ["Id", "PostId", "RelatedPostId"],
}

DUCKDB_URLS = {
    "x86_64": "https://github.com/duckdb/duckdb/releases/download/v1.3.2/duckdb_cli-linux-amd64.zip",
    "aarch64": "https://github.com/duckdb/duckdb/releases/download/v1.3.2/duckdb_cli-linux-arm64.zip",
}

DEFAULT_RUNS = 5
REAL_WARMUPS = 2      # big persistent tables: page cache + JIT settle fast
SYN_WARMUPS = jmb.WARMUP_RUNS  # 4: JIT fires on execution #4 of a cold server


# --------------------------------------------------------------------------
# Execution target: local `clickhouse client` subprocess, or the same via ssh.
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExecTarget:
    binary: str
    port: int = 9000
    ssh_key: str | None = None
    ssh_host: str | None = None

    def client_argv(self, query: str, *, settings: dict | None = None) -> list[str]:
        argv = [self.binary, "client", "--port", str(self.port)]
        for k, v in (settings or {}).items():
            argv += [f"--{k}", str(v)]
        argv += ["--query", query]
        return argv


def sql(target: ExecTarget, query: str, *, timeout: float = 600.0, stdin_file: str | None = None,
        settings: dict | None = None):
    # The client's default receive_timeout (300s) aborts long silent queries
    # (e.g. OPTIMIZE FINAL on a big table sends no packets while merging), so
    # widen it to the caller's subprocess timeout.
    merged = {"receive_timeout": max(int(timeout), 300), **(settings or {})}
    argv = target.client_argv(query, settings=merged)
    if target.ssh_host:
        remote = " ".join(shlex.quote(a) for a in argv)
        if stdin_file:
            remote += f" < {shlex.quote(stdin_file)}"
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        if target.ssh_key:
            argv += ["-i", target.ssh_key]
        argv += [target.ssh_host, remote]
        stdin = None
    else:
        stdin = open(stdin_file, "rb") if stdin_file else None
    try:
        proc = subprocess.run(argv, stdin=stdin, capture_output=True, timeout=timeout)
    finally:
        if stdin:
            stdin.close()
    return proc.returncode, proc.stdout, proc.stderr


def sql_ok(target: ExecTarget, query: str, *, timeout: float = 600.0, purpose: str = "",
           stdin_file: str | None = None, settings: dict | None = None) -> bytes:
    rc, out, err = sql(target, query, timeout=timeout, stdin_file=stdin_file, settings=settings)
    if rc != 0:
        raise RuntimeError(f"SQL failed ({purpose or query[:80]}): rc={rc} stderr={err.decode(errors='replace')[:2000]}")
    return out


def sql_json(target: ExecTarget, query: str, *, timeout: float = 600.0) -> list[dict]:
    out = sql_ok(target, query + " FORMAT JSONEachRow", timeout=timeout)
    return [json.loads(line) for line in out.decode().splitlines() if line.strip()]


def target_from_args(args: argparse.Namespace) -> ExecTarget:
    return ExecTarget(binary=args.binary, port=args.port, ssh_key=getattr(args, "ssh_key", None),
                      ssh_host=getattr(args, "ssh_host", None))


DEFAULT_BINARY = os.environ.get("JBMT_CLICKHOUSE", "clickhouse")


def add_target_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--binary", default=DEFAULT_BINARY,
                   help="clickhouse binary (client mode is invoked on it; env JBMT_CLICKHOUSE overrides)")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--ssh-host", default=None, help="run the client on this host via ssh (sweep on fleet instances)")
    p.add_argument("--ssh-key", default=None)


# --------------------------------------------------------------------------
# Arms: one server per binary; timed runs interleave strict ABAB across arms
# --------------------------------------------------------------------------

DEFAULT_ARM_NAME = "solo"


def add_arm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--arm", action="append", default=None, metavar="NAME=BINARY:PORT",
                   help="benchmark arm (repeatable). With two arms every unit's timed runs are "
                        "interleaved strict ABAB across the arms' servers, which must both be "
                        "running (join_bench_mt_servers.sh). Overrides --binary/--port.")


def add_algorithm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--algorithms", default=",".join(ALGORITHMS), metavar="LIST",
                   help=f"comma list of join algorithms to measure, in run order "
                        f"(default: {','.join(ALGORITHMS)}). Restricting to one drops the "
                        "cross-algorithm (row_count, checksum) oracle, which is the only "
                        "independent correctness check real-suite units have in single-arm mode.")


def algorithms_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    names = tuple(a.strip() for a in getattr(args, "algorithms", "").split(",") if a.strip())
    if not names:
        raise SystemExit("--algorithms must name at least one algorithm")
    unknown = [a for a in names if a not in ALGORITHMS]
    if unknown:
        raise SystemExit(f"unknown algorithm(s) {unknown}; known: {list(ALGORITHMS)}")
    if len(set(names)) != len(names):
        raise SystemExit(f"--algorithms has duplicates: {list(names)}")
    return names


def parse_arm_spec(spec: str) -> tuple[str, str, int]:
    name, eq, rest = spec.partition("=")
    binary, colon, port = rest.rpartition(":")
    if not (eq and colon and name and binary and port.isdigit()):
        raise SystemExit(f"--arm expects NAME=BINARY:PORT, got {spec!r}")
    return name, binary, int(port)


def arms_from_args(args: argparse.Namespace) -> dict[str, ExecTarget]:
    """Ordered {arm name: target}. Without --arm: one legacy arm from --binary/--port."""
    if not getattr(args, "arm", None):
        return {DEFAULT_ARM_NAME: target_from_args(args)}
    arms: dict[str, ExecTarget] = {}
    ports: set[int] = set()
    for spec in args.arm:
        name, binary, port = parse_arm_spec(spec)
        if name in arms or port in ports:
            raise SystemExit(f"duplicate arm name or port: {spec!r}")
        arms[name] = ExecTarget(binary=binary, port=port,
                                ssh_key=getattr(args, "ssh_key", None),
                                ssh_host=getattr(args, "ssh_host", None))
        ports.add(port)
    return arms


_BINARY_SHA_CACHE: dict[tuple[str | None, str], str] = {}


def binary_sha256(target: ExecTarget) -> str:
    """sha256 of the arm's binary (provenance in every result row + the sweep
    resume guard); cached per path so multi-GB binaries are hashed once."""
    cache_key = (target.ssh_host, target.binary)
    if cache_key in _BINARY_SHA_CACHE:
        return _BINARY_SHA_CACHE[cache_key]
    if target.ssh_host:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        if target.ssh_key:
            argv += ["-i", target.ssh_key]
        argv += [target.ssh_host, f"sha256sum {shlex.quote(target.binary)}"]
        proc = subprocess.run(argv, capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"sha256sum of {target.binary} on {target.ssh_host} failed: "
                               f"{proc.stderr.decode(errors='replace')[:500]}")
        sha = proc.stdout.decode().split()[0]
    else:
        path = pathlib.Path(target.binary)
        if not path.exists():
            which = shutil.which(target.binary)
            if which is None:
                raise RuntimeError(f"cannot hash binary {target.binary!r}: not found")
            path = pathlib.Path(which)
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        sha = h.hexdigest()
    _BINARY_SHA_CACHE[cache_key] = sha
    return sha


# --------------------------------------------------------------------------
# Manifest / schema loading
# --------------------------------------------------------------------------


def load_specs() -> list[dict]:
    return json.loads(QUERIES_JSON.read_text())["specs"]


def load_schemas() -> dict:
    return json.loads(SCHEMAS_JSON.read_text())["datasets"]


def real_db(dataset: str, tier: str) -> str:
    return f"jbmt_{dataset}_{tier}"


# --------------------------------------------------------------------------
# Real datasets: DDL, loaders, fingerprints
# --------------------------------------------------------------------------


def create_real_tables(target: ExecTarget, dataset: str, tier: str) -> None:
    schemas = load_schemas()[dataset]
    db = real_db(dataset, tier)
    sql_ok(target, f"CREATE DATABASE IF NOT EXISTS {db}", purpose=f"create db {db}")
    for table, spec in schemas.items():
        cols = ", ".join(f"`{name}` {typ}" for name, typ in spec["columns"])
        sql_ok(
            target,
            f"CREATE TABLE IF NOT EXISTS {db}.{table} ({cols}) "
            f"ENGINE = MergeTree ORDER BY {spec['order_by']}",
            purpose=f"create {db}.{table}",
        )


def fingerprint_sql(db: str, table: str, order_by: str) -> str:
    keys = order_by.strip("()")
    return f"SELECT count() AS rows, sum(cityHash64({keys})) AS key_checksum FROM {db}.{table}"


def finalize_table(target: ExecTarget, dataset: str, tier: str, table: str, *, timeout: float = 7200.0) -> dict:
    db = real_db(dataset, tier)
    order_by = load_schemas()[dataset][table]["order_by"]
    sql_ok(target, f"OPTIMIZE TABLE {db}.{table} FINAL", timeout=timeout, purpose=f"optimize {db}.{table}")
    row = sql_json(target, fingerprint_sql(db, table, order_by), timeout=1800)[0]
    fp = {"db": db, "table": table, "rows": int(row["rows"]), "key_checksum": int(row["key_checksum"])}
    sql_ok(target, f"CREATE DATABASE IF NOT EXISTS {META_DB}")
    sql_ok(
        target,
        f"CREATE TABLE IF NOT EXISTS {FINGERPRINTS_TABLE} "
        "(db String, table String, rows UInt64, key_checksum UInt64, loaded_at DateTime, tool_version String) "
        "ENGINE = ReplacingMergeTree(loaded_at) ORDER BY (db, table)",
    )
    sql_ok(
        target,
        f"INSERT INTO {FINGERPRINTS_TABLE} VALUES "
        f"('{db}', '{table}', {fp['rows']}, {fp['key_checksum']}, now(), '{TOOL_VERSION}')",
    )
    print(f"  {db}.{table}: rows={fp['rows']} key_checksum={fp['key_checksum']}")
    return fp


def stored_fingerprints(target: ExecTarget, db: str) -> dict[str, dict]:
    try:
        rows = sql_json(target, f"SELECT db, table, rows, key_checksum FROM {FINGERPRINTS_TABLE} FINAL WHERE db = '{db}'")
    except RuntimeError:
        return {}
    return {r["table"]: r for r in rows}


def table_load_guard(target: ExecTarget, dataset: str, tier: str, table: str) -> bool:
    """Idempotent per-table load gate: True = already fingerprinted, skip.
    A table without a fingerprint may hold a partial load - truncate it so a
    resumed `load-real` never double-inserts."""
    db = real_db(dataset, tier)
    if table in stored_fingerprints(target, db):
        print(f"  {db}.{table}: already loaded (fingerprint present), skipping")
        return True
    # The truncate is destructive by design (partial loads are discarded), so
    # bypass max_table_size_to_drop - a big partial table must still be droppable.
    sql_ok(target, f"TRUNCATE TABLE {db}.{table}", purpose=f"truncate {db}.{table}",
           settings={"max_table_size_to_drop": 0})
    return False


def fetch_duckdb(workdir: pathlib.Path) -> pathlib.Path:
    import platform
    import zipfile
    exe = workdir / "duckdb"
    if exe.exists():
        return exe
    url = DUCKDB_URLS[platform.machine()]
    workdir.mkdir(parents=True, exist_ok=True)
    zpath = workdir / "duckdb.zip"
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extract("duckdb", workdir)
    exe.chmod(0o755)
    zpath.unlink()
    return exe


def duckdb_generate(duckdb: pathlib.Path, kind: str, sf: int, workdir: pathlib.Path, tables: list[str]) -> None:
    """Generate TPC-H/TPC-DS parquet files with the DuckDB extension generators."""
    outdir = workdir / f"{kind}_sf{sf}"
    outdir.mkdir(parents=True, exist_ok=True)
    if all((outdir / f"{t}.parquet").exists() for t in tables):
        print(f"  parquet already present in {outdir}")
        return
    gen = "dbgen" if kind == "tpch" else "dsdgen"
    copies = "\n".join(
        f"COPY (SELECT * FROM {t}) TO '{outdir}/{t}.parquet' (FORMAT PARQUET);" for t in tables
    )
    script = f"INSTALL {kind}; LOAD {kind}; CALL {gen}(sf={sf});\n{copies}\n"
    dbfile = workdir / f"{kind}_sf{sf}.duckdb"
    proc = subprocess.run([str(duckdb), str(dbfile)], input=script.encode(), capture_output=True, timeout=4 * 3600)
    if proc.returncode != 0:
        raise RuntimeError(f"duckdb {gen} failed: {proc.stderr.decode(errors='replace')[:2000]}")
    dbfile.unlink(missing_ok=True)


def insert_parquet_file(target: ExecTarget, db: str, table: str, path: pathlib.Path) -> None:
    if target.ssh_host:
        raise RuntimeError("load-real only supports local mode (run it on the loader instance)")
    sql_ok(target, f"INSERT INTO {db}.{table} FORMAT Parquet", stdin_file=str(path), timeout=7200,
           purpose=f"insert {table}")


def load_tpc(target: ExecTarget, dataset: str, tier: str, workdir: pathlib.Path, duckdb_path: str | None) -> None:
    sf = TIERS[tier][f"{dataset}_sf"]
    schemas = load_schemas()[dataset]
    # duckdb table names: tpcds `customer` etc. match query-visible names.
    tables = list(schemas)
    duckdb = pathlib.Path(duckdb_path) if duckdb_path else fetch_duckdb(workdir)
    print(f"generating {dataset} sf={sf} via {duckdb}")
    duckdb_generate(duckdb, dataset, sf, workdir, tables)
    db = real_db(dataset, tier)
    for t in tables:
        if table_load_guard(target, dataset, tier, t):
            continue
        pq = workdir / f"{dataset}_sf{sf}" / f"{t}.parquet"
        print(f"  loading {db}.{t} from {pq}")
        insert_parquet_file(target, db, t, pq)
        finalize_table(target, dataset, tier, t)


def load_coffeeshop(target: ExecTarget, tier: str, limit: int | None) -> None:
    scale = TIERS[tier]["coffeeshop_scale"]
    db = real_db("coffeeshop", tier)
    lim = f" LIMIT {limit}" if limit else ""
    d = "CAST('2000-01-01' AS Date) AS synth_date"
    loads = {
        "dim_locations": (
            f"SELECT {d}, CAST(ifNull(record_id,'') AS String) AS record_id, CAST(ifNull(location_id,'') AS String) AS location_id, "
            "CAST(ifNull(city,'') AS String) AS city, CAST(ifNull(state,'') AS String) AS state, "
            "CAST(ifNull(country,'') AS String) AS country, CAST(ifNull(region,'') AS String) AS region "
            f"FROM icebergS3('{COFFEESHOP_BUCKET}/dim_locations/', NOSIGN)"
        ),
        "dim_products": (
            f"SELECT {d}, CAST(ifNull(record_id,'') AS String) AS record_id, CAST(ifNull(product_id,'') AS String) AS product_id, "
            "CAST(ifNull(name,'') AS String) AS name, CAST(ifNull(category,'') AS String) AS category, "
            "CAST(ifNull(subcategory,'') AS String) AS subcategory, CAST(ifNull(standard_cost,0) AS Float64) AS standard_cost, "
            "CAST(ifNull(standard_price,0) AS Float64) AS standard_price, CAST(ifNull(from_date, toDate(0)) AS Date) AS from_date, "
            "CAST(ifNull(to_date, toDate(0)) AS Date) AS to_date "
            f"FROM icebergS3('{COFFEESHOP_BUCKET}/dim_products/', NOSIGN)"
        ),
        "fact_sales": (
            "SELECT CAST(ifNull(order_id,'') AS String) AS order_id, CAST(ifNull(order_date, toDate(0)) AS Date) AS order_date, "
            "CAST(ifNull(time_of_day,'') AS String) AS time_of_day, CAST(ifNull(season,'') AS String) AS season, "
            "CAST(ifNull(month,0) AS Int32) AS month, CAST(ifNull(location_id,'') AS String) AS location_id, "
            "CAST(ifNull(region,'') AS String) AS region, CAST(ifNull(product_name,'') AS String) AS product_name, "
            "CAST(ifNull(quantity,0) AS Int32) AS quantity, CAST(ifNull(sales_amount,0) AS Float64) AS sales_amount, "
            "CAST(ifNull(discount_percentage,0) AS Int32) AS discount_percentage, CAST(ifNull(product_id,'') AS String) AS product_id "
            f"FROM icebergS3('{COFFEESHOP_BUCKET}/fact_sales_{scale}/', NOSIGN){lim}"
        ),
    }
    for table, select in loads.items():
        if table_load_guard(target, "coffeeshop", tier, table):
            continue
        print(f"  loading {db}.{table}")
        sql_ok(target, f"INSERT INTO {db}.{table} {select}", timeout=8 * 3600, purpose=f"load {table}")
        finalize_table(target, "coffeeshop", tier, table)


def load_stackoverflow(target: ExecTarget, tier: str, limit: int | None) -> None:
    db = real_db("stackoverflow", tier)
    schemas = load_schemas()["stackoverflow"]
    copies = TIERS[tier]["so_copies"]
    lim = f" LIMIT {limit}" if limit else ""
    for table, spec in schemas.items():
        if table_load_guard(target, "stackoverflow", tier, table):
            continue
        cols = [name for name, _ in spec["columns"]]
        for copy in range(copies):
            selects = []
            for name, typ in spec["columns"]:
                base = "0" if not typ.startswith(("String", "DateTime", "Bool")) else (
                    "''" if typ == "String" else ("false" if typ == "Bool" else "toDateTime64(0, 3)"))
                expr = f"ifNull(CAST({name} AS Nullable({typ})), {base})"
                if copy > 0 and name in SO_ID_COLS[table]:
                    expr = f"({expr}) + {SO_2X_OFFSET}"
                selects.append(f"{expr} AS {name}")
            print(f"  loading {db}.{table} (copy {copy})")
            sql_ok(
                target,
                f"INSERT INTO {db}.{table} ({', '.join(cols)}) SELECT {', '.join(selects)} "
                f"FROM s3('{SO_SOURCES[table]}', NOSIGN, 'Parquet'){lim}",
                timeout=4 * 3600,
                purpose=f"load {table} copy{copy}",
            )
        finalize_table(target, "stackoverflow", tier, table)


JOB_SRC = "https://event.cwi.nl/da/job/imdb.tgz"


def load_job(target: ExecTarget, tier: str, workdir: pathlib.Path) -> None:
    """JOB/IMDB: canonical Postgres-COPY CSV tarball, re-encoded to RFC 4180 by
    the vendored converter (join_bench_mt_jobcsv.py), streamed into the server.
    Identical data at every tier (the dataset has one fixed size)."""
    import tarfile
    if target.ssh_host:
        raise RuntimeError("load-real only supports local mode (run it on the loader instance)")
    db = real_db("job", tier)
    src_dir = workdir / "job_csv"
    if not (src_dir / "title.csv").exists():
        tgz = workdir / "imdb.tgz"
        if not tgz.exists():
            workdir.mkdir(parents=True, exist_ok=True)
            print(f"  downloading {JOB_SRC}")
            urllib.request.urlretrieve(JOB_SRC, tgz)
        print(f"  extracting {tgz}")
        src_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as tf:
            tf.extractall(src_dir, filter="data")
    converter = TOOL_DIR / "join_bench_mt_jobcsv.py"
    for table, spec in load_schemas()["job"].items():
        if table_load_guard(target, "job", tier, table):
            continue
        csv_path = next(src_dir.rglob(f"{table}.csv"))
        cols = [n for n, _ in spec["columns"] if n != "synth_date"]
        insert = f"INSERT INTO {db}.{table} ({', '.join(cols)}) FORMAT CSV"
        print(f"  loading {db}.{table} from {csv_path}")
        with open(csv_path, "rb") as f:
            conv = subprocess.Popen([sys.executable, str(converter)], stdin=f, stdout=subprocess.PIPE)
            client = subprocess.Popen(
                target.client_argv(insert, settings={"input_format_csv_empty_as_default": "1"}),
                stdin=conv.stdout, stderr=subprocess.PIPE,
            )
            conv.stdout.close()
            _, client_err = client.communicate(timeout=7200)
            conv.wait(timeout=60)
        if conv.returncode != 0:
            raise RuntimeError(f"job csv converter failed for {table}")
        if client.returncode != 0:
            raise RuntimeError(f"insert {table} failed: {client_err.decode(errors='replace')[:2000]}")
        finalize_table(target, "job", tier, table)


def load_real_command(args: argparse.Namespace) -> int:
    target = target_from_args(args)
    workdir = pathlib.Path(args.workdir)
    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        print(f"== load {dataset} tier={args.tier}")
        create_real_tables(target, dataset, args.tier)
        if dataset in ("tpch", "tpcds"):
            load_tpc(target, dataset, args.tier, workdir, args.duckdb)
        elif dataset == "coffeeshop":
            load_coffeeshop(target, args.tier, args.limit_rows)
        elif dataset == "stackoverflow":
            load_stackoverflow(target, args.tier, args.limit_rows)
        elif dataset == "job":
            load_job(target, args.tier, pathlib.Path(args.workdir))
        else:
            print(f"unknown dataset {dataset}", file=sys.stderr)
            return 1
    return 0


def verify_command(args: argparse.Namespace) -> int:
    """Recompute fingerprints and compare with the stored ones (and, if given,
    with a reference JSON from the loader instance - the fleet gate)."""
    target = target_from_args(args)
    schemas = load_schemas()
    reference = json.loads(pathlib.Path(args.reference).read_text()) if args.reference else None
    bad = 0
    for dataset in args.datasets.split(","):
        db = real_db(dataset, args.tier)
        stored = stored_fingerprints(target, db)
        for table, spec in schemas[dataset].items():
            row = sql_json(target, fingerprint_sql(db, table, spec["order_by"]), timeout=1800)[0]
            actual = {"rows": int(row["rows"]), "key_checksum": int(row["key_checksum"])}
            exp = stored.get(table)
            ref = (reference or {}).get(f"{db}.{table}")
            for src_name, src in (("stored", exp), ("reference", ref)):
                if src is None:
                    continue
                if int(src["rows"]) != actual["rows"] or int(src["key_checksum"]) != actual["key_checksum"]:
                    print(f"MISMATCH {db}.{table} vs {src_name}: actual={actual} expected={src}")
                    bad += 1
            if exp is None and ref is None:
                print(f"NO FINGERPRINT for {db}.{table} (not loaded?)")
                bad += 1
        if args.emit:
            out = {f"{db}.{t}": {"rows": v["rows"], "key_checksum": v["key_checksum"]} for t, v in stored.items()}
            pathlib.Path(args.emit).write_text(json.dumps(out, indent=1))
    print("verify: OK" if bad == 0 else f"verify: {bad} problems")
    return 0 if bad == 0 else 1


# --------------------------------------------------------------------------
# Synthetic cells: jmb key machinery + two local key kinds + MergeTree DDL
# --------------------------------------------------------------------------

# Local key configs beyond jmb.KEY_CONFIGS. K10 = single UInt32 (every real
# benchmark joins on 32-bit ints); KF2 = FixedString(2) (TPC-DS state codes).
LOCAL_KEY_CONFIGS = {
    "K10": {"columns": ["k0"], "type": "UInt32", "n_max": 128_000_000},
    "KF2": {"columns": ["k"], "type": "FixedString(2)", "n_max": 676},
}


def local_key_exprs(key: str, rank_col: str) -> list[str]:
    if key == "K10":
        # bijective mod 2^32 (odd multiplier): distinct ranks -> distinct keys.
        return [f"toUInt32(bitAnd(({rank_col}) * 2654435761, 4294967295))"]
    if key == "KF2":
        # bijective for rank < 676: two uppercase letters.
        return [f"toFixedString(concat(char(65 + intDiv({rank_col}, 26) % 26), char(65 + ({rank_col}) % 26)), 2)"]
    raise ValueError(key)


def key_columns(key: str) -> list[str]:
    if key in LOCAL_KEY_CONFIGS:
        return LOCAL_KEY_CONFIGS[key]["columns"]
    return list(jmb.KEY_CONFIGS[key].key_columns)


def key_n_max(key: str) -> int:
    if key in LOCAL_KEY_CONFIGS:
        return LOCAL_KEY_CONFIGS[key]["n_max"]
    return jmb.KEY_CONFIGS[key].n_max


def syn_table_ddl(cell: dict) -> list[str]:
    """MergeTree DDL for one cell. Build sorts by the join key (dims are
    PK-sorted in real life); probe is insertion-ordered (facts are not sorted
    by the probe key)."""
    key = cell["key"]
    if key in LOCAL_KEY_CONFIGS:
        cfg = LOCAL_KEY_CONFIGS[key]
        key_cols = [(c, cfg["type"]) for c in cfg["columns"]]
    else:
        kc = jmb.KEY_CONFIGS[key]
        key_cols = [(name, kc.column_type(nullable_wrap=kc.is_nullable)) for name in kc.key_columns]
    bp_cols = jmb.payload_columns_ddl("b_p", cell["bp"])
    pp_cols = jmb.payload_columns_ddl("p_p", cell["pp"])
    order_by = ", ".join(
        f"assumeNotNull({n})" if t.startswith("Nullable") else n for n, t in key_cols
    )
    stmts = [f"CREATE DATABASE IF NOT EXISTS {SYN_DB}"]
    for table, extra in ((jmb.BUILD_TABLE, bp_cols), (jmb.PROBE_TABLE, pp_cols)):
        cols = ", ".join(f"{n} {t}" for n, t in key_cols + extra)
        ob = f"({order_by})" if table == jmb.BUILD_TABLE else "tuple()"
        allow = " SETTINGS allow_nullable_key = 1" if any(t.startswith("Nullable") for _, t in key_cols) and table == jmb.BUILD_TABLE else ""
        stmts += [f"DROP TABLE IF EXISTS {table}",
                  f"CREATE TABLE {table} ({cols}) ENGINE = MergeTree ORDER BY {ob}{allow}"]
    return stmts


def local_keys_store_ddl(key: str) -> list[str]:
    """keys_store tables for K10/KF2 (jmb prepare-keys covers K0-K9)."""
    cfg = LOCAL_KEY_CONFIGS[key]
    n_max = cfg["n_max"]
    tname = jmb.keys_store_table_name(key)
    cols = ", ".join(f"{c} {cfg['type']}" for c in cfg["columns"])
    exprs = ", ".join(f"{e} AS {c}" for e, c in zip(local_key_exprs(key, "number"), cfg["columns"]))
    # miss domain rows live at rank in [n_max, 2*n_max) like jmb stores.
    return [
        "CREATE DATABASE IF NOT EXISTS keys_store",
        f"DROP TABLE IF EXISTS {tname}",
        f"CREATE TABLE {tname} (rank UInt64, {cols}) ENGINE = MergeTree ORDER BY rank",
        f"INSERT INTO {tname} SELECT number AS rank, {exprs} FROM numbers({n_max})",
        f"INSERT INTO {tname} SELECT number + {n_max} AS rank, "
        + ", ".join(f"{e} AS {c}" for e, c in zip(local_key_exprs(key, f"(number + {n_max})"), cfg["columns"]))
        + f" FROM numbers({n_max})",
        f"OPTIMIZE TABLE {tname} FINAL",
    ]


# Above this, one-INSERT-per-pass degenerates and the batched CROSS JOIN fill is
# used instead. 256 = the legacy sweeps' maximum m_p, so every legacy cell keeps
# byte-parity (identical insertion order) with the Memory-engine runs.
MAX_PROBE_PASSES = 256


def _batched_probe_fill(cell: dict) -> str:
    """One INSERT for all `m_p` probe passes: keys_store rows CROSS JOIN
    numbers(m_p), re-ordered to occurrence-major (`ORDER BY number, rank` -
    every key once per pass, matching the per-pass INSERT layout) so huge-m_p
    tiny-dimension cells do not degenerate into m_p tiny INSERTs. Key columns
    are read as stored (the store holds exactly the derived values; that is
    what `prepare-keys` fingerprints). Restricted to nulls_pct=0 cells - the
    NULL-wrapping variants keep the small-m_p per-pass path."""
    key, D, m_p = cell["key"], cell["D"], cell["m_p"]
    if cell["nulls_pct"] != 0:
        raise ValueError("batched probe fill requires nulls_pct=0")
    if key not in LOCAL_KEY_CONFIGS and jmb.KEY_CONFIGS[key].is_nullable:
        raise ValueError("batched probe fill does not support Nullable keys")
    n_max = key_n_max(key)
    src = jmb.keys_store_table_name(key)
    cols = key_columns(key)
    key_cols = ", ".join(cols)
    pp_names = [n for n, _ in jmb.payload_columns_ddl("p_p", cell["pp"])]
    hit = jmb.decimal_int_product(D, str(cell["h"]))
    parts = []
    if hit > 0:
        pp_sel = jmb.payload_select_exprs("p_p", cell["pp"], "rank", jmb.PROBE_PAYLOAD_SEED)
        pp_part = (", " + ", ".join(pp_sel)) if pp_sel else ""
        parts.append(f"SELECT rank, {key_cols}{pp_part} FROM {src} WHERE rank < {hit}")
    if D - hit > 0:
        pp_sel = jmb.payload_select_exprs("p_p", cell["pp"], f"(rank - toUInt64({n_max}))", jmb.PROBE_PAYLOAD_SEED ^ 0x1)
        pp_part = (", " + ", ".join(pp_sel)) if pp_sel else ""
        parts.append(f"SELECT rank, {key_cols}{pp_part} FROM {src} "
                     f"WHERE rank >= {n_max} AND rank < {n_max + (D - hit)}")
    inner = " UNION ALL ".join(f"({p})" for p in parts) if len(parts) > 1 else parts[0]
    out_cols = ", ".join(cols + pp_names)
    return (
        f"INSERT INTO {jmb.PROBE_TABLE} ({out_cols}) "
        f"SELECT {out_cols} FROM ({inner}) AS src CROSS JOIN numbers({m_p}) AS n ORDER BY n.number, src.rank"
    )


def syn_fill_statements(cell: dict) -> list[str]:
    """INSERT statements for one cell (build then probe). jmb's generators are
    reused verbatim for jmb keys at small m_p (byte-parity with the Memory
    sweeps); huge-m_p cells use the batched single-INSERT fill; K10/KF2 get
    local equivalents (uniform, payload-carrying, miss-domain probe keys)."""
    key = cell["key"]
    D, m_b, m_p = cell["D"], cell["m_b"], cell["m_p"]
    if key not in LOCAL_KEY_CONFIGS:
        cfg = jmb.KEY_CONFIGS[key]
        build = jmb.build_fill_statements(cfg, D, m_b, cell["bp"], cell["nulls_pct"], cell.get("skew_s", 0))
        if m_p > MAX_PROBE_PASSES:
            return build + [_batched_probe_fill(cell)]
        probe = jmb.probe_fill_statements(cfg, D, m_p, cell["h"], cell["pp"], cfg.n_max, cell["nulls_pct"])
        return build + probe
    cfg = LOCAL_KEY_CONFIGS[key]
    n_max = cfg["n_max"]
    if D > n_max:
        raise ValueError(f"{key}: D={D} exceeds n_max={n_max}")
    src = jmb.keys_store_table_name(key)
    key_cols = ", ".join(cfg["columns"])
    bp_sel = jmb.payload_select_exprs("b_p", cell["bp"], "rank", jmb.BUILD_PAYLOAD_SEED)
    bp_part = (", " + ", ".join(bp_sel)) if bp_sel else ""
    stmts = [
        f"INSERT INTO {jmb.BUILD_TABLE} SELECT {key_cols}{bp_part} FROM {src} WHERE rank < {D} ORDER BY rank"
        for _ in range(m_b)
    ]
    if m_p > MAX_PROBE_PASSES:
        return stmts + [_batched_probe_fill(cell)]
    pp_sel = jmb.payload_select_exprs("p_p", cell["pp"], "rank", jmb.PROBE_PAYLOAD_SEED)
    pp_part = (", " + ", ".join(pp_sel)) if pp_sel else ""
    hit = jmb.decimal_int_product(D, str(cell["h"]))
    parts = []
    if hit > 0:
        parts.append(f"SELECT {key_cols}{pp_part} FROM {src} WHERE rank < {hit} ORDER BY rank")
    if D - hit > 0:
        parts.append(f"SELECT {key_cols}{pp_part} FROM {src} WHERE rank >= {n_max} AND rank < {n_max + (D - hit)} ORDER BY rank")
    select = " UNION ALL ".join(f"({p})" for p in parts) if len(parts) > 1 else parts[0]
    stmts += [f"INSERT INTO {jmb.PROBE_TABLE} SELECT * FROM ({select})" for _ in range(m_p)]
    return stmts


def expected_rows_for_kind(cell: dict) -> int | None:
    """Closed-form expected output rows; None when only cross-algorithm
    agreement can be asserted."""
    D, m_b, m_p, h, nulls, skew = cell["D"], cell["m_b"], cell["m_p"], str(cell["h"]), cell["nulls_pct"], cell.get("skew_s", 0)
    kind = cell.get("kind", "INNER")
    if kind == "INNER":
        return jmb.expected_output_rows(D, m_b, m_p, h, nulls, skew)
    if nulls != 0 or skew != 0:
        return None
    hit = jmb.decimal_int_product(D, h)
    probe_rows = D * m_p
    if kind == "LEFT":
        return hit * m_p * m_b + (probe_rows - hit * m_p)
    if kind == "LEFT SEMI":
        return hit * m_p
    if kind == "LEFT ANTI":
        return probe_rows - hit * m_p
    return None


def make_syn_cell(*, D, key, m_b=1, m_p=1, h="1.0", bp=8, pp=8, threads, kind="INNER",
                  nulls_pct=0, skew_s=0, runs=DEFAULT_RUNS, group, note=""):
    kind_tag = "" if kind == "INNER" else "_" + kind.replace(" ", "").lower()
    cell_id = f"D{D}_{key}_mb{m_b}_mp{m_p}_h{h}_bp{bp}_pp{pp}_T{threads}{kind_tag}"
    if nulls_pct:
        cell_id += f"_null{nulls_pct}"
    if skew_s:
        cell_id += f"_skew{skew_s}"
    return dict(unit="cell", unit_id=cell_id, D=D, key=key, m_b=m_b, m_p=m_p, h=h, bp=bp, pp=pp,
                threads=threads, kind=kind, nulls_pct=nulls_pct, skew_s=skew_s, runs=runs,
                group=group, note=note, cost=D * m_b + D * m_p)


LEGACY_CELLS_JSON = TOOL_DIR / "join_bench_mt_legacy_cells.json"


def legacy_cells() -> list[dict]:
    """The 347 cells of the Memory-engine sweeps (regenerated from
    `join_memory_bench.py plan`), replayed on MergeTree. Original cell_ids are
    preserved (incl. `_nulls10` / `_rep` suffixes) so results join across
    harnesses; old group letters live in the note."""
    units = []
    for c in json.loads(LEGACY_CELLS_JSON.read_text()):
        u = make_syn_cell(
            D=c["D"], key=c["key"], m_b=c["m_b"], m_p=c["m_p"], h=c["h"],
            bp=c["bp"], pp=c["pp"], threads=c["threads"],
            nulls_pct=c["nulls_pct"], skew_s=c["skew_s"], runs=c["runs"],
            group="LEGACY", note=f"{'+'.join(c['groups'])};{c['note']}",
        )
        u["unit_id"] = c["cell_id"]
        u["rep"] = c.get("rep", 0)
        units.append(u)
    return units


def synthetic_plan() -> list[dict]:
    M = 1_000_000
    cells: dict[str, dict] = {}

    def add(c):
        cells.setdefault(c["unit_id"], c)

    # LEGACY first: for shapes that also exist in the new groups (the ANCH
    # anchors by design), the legacy entry wins so its runs/rep metadata and
    # group tag are preserved.
    for c in legacy_cells():
        add(c)
    # ANCH - continuity anchors shared with the Memory-engine sweeps.
    for key in ("K0", "K2", "K7"):
        for T in (8, 96):
            add(make_syn_cell(D=32 * M, key=key, threads=T, group="ANCH", note="anchor"))
    # K - tiny/small dimensions x huge probe multiplicity.
    for D, m_p in ((16, 2 * M), (512, 62_500), (4_096, 8_192), (65_536, 1_600)):
        for T in (16, 96):
            add(make_syn_cell(D=D, key="K0", m_p=m_p, threads=T, group="K", note="tiny-dim"))
    for D, m_p in ((16, 2 * M), (512, 62_500)):
        for T in (16, 96):
            add(make_syn_cell(D=D, key="K5", m_p=m_p, threads=T, group="K", note="tiny-dim-string"))
    add(make_syn_cell(D=512, key="K0", m_p=62_500, pp=64, threads=96, group="K", note="tiny-dim-widepp"))
    for T in (16, 96):
        add(make_syn_cell(D=64 * M, key="K0", m_p=4, threads=T, group="K", note="large-D-mp4"))
    # L - filtered dimensions (h < 1 below 32M).
    for D, m_p in ((4_096, 8_192), (65_536, 1_600), (1 * M, 64), (8 * M, 16)):
        for h in ("0.05", "0.3"):
            for T in (16, 96):
                add(make_syn_cell(D=D, key="K0", m_p=m_p, h=h, threads=T, group="L", note="filtered-dim"))
    for T in (16, 96):
        add(make_syn_cell(D=8 * M, key="K1", m_p=8, h="0.1", threads=T, group="L", note="returns-join"))
    # M - m:n cross (JOB FK=FK).
    for m_b in (2, 4, 8):
        for m_p in (4, 16):
            for T in (16, 96):
                add(make_syn_cell(D=1 * M, key="K0", m_b=m_b, m_p=m_p, h="0.9", threads=T, group="M", note="m:n"))
    for m_b in (2, 8):
        for T in (16, 96):
            add(make_syn_cell(D=2 * M, key="K0", m_b=m_b, m_p=8, h="0.9", threads=T, group="M", note="m:n-title"))
    # N - narrow integer keys.
    for D, m_p in ((65_536, 1_600), (1 * M, 64), (32 * M, 4), (128 * M, 1)):
        for T in (8, 96):
            add(make_syn_cell(D=D, key="K10", m_p=m_p, threads=T, group="N", note="uint32-key"))
    # O - short-string / FixedString dims.
    for T in (16, 96):
        add(make_syn_cell(D=64, key="KF2", m_p=500_000, threads=T, group="O", note="state-code"))
        add(make_syn_cell(D=512, key="K4", m_p=62_500, threads=T, group="O", note="string8-dim"))
        add(make_syn_cell(D=4_096, key="K4", m_p=8_192, threads=T, group="O", note="string8-dim"))
    # S - NULL-bearing integer keys (nulls_pct=10 is what the generator supports).
    for D, m_p in ((4 * M, 8), (32 * M, 1)):
        for T in (16, 96):
            add(make_syn_cell(D=D, key="K8", m_p=m_p, nulls_pct=10, threads=T, group="S", note="null-keys"))
    # P - join kinds on three representative shapes.
    for kind in ("LEFT", "LEFT SEMI", "LEFT ANTI"):
        for T in (16, 96):
            add(make_syn_cell(D=32 * M, key="K0", h="0.9", kind=kind, threads=T, group="P", note="kind"))
            add(make_syn_cell(D=65_536, key="K0", m_p=1_600, h="0.3", kind=kind, threads=T, group="P", note="kind"))
            add(make_syn_cell(D=8 * M, key="K1", m_p=8, h="0.1", kind=kind, threads=T, group="P", note="kind"))
    return sorted(cells.values(), key=lambda c: c["unit_id"])


def real_plan(tier: str, threads_ladder=(16, 96)) -> list[dict]:
    units = []
    for spec in load_specs():
        scale = 1.0
        if tier == "b":
            scale = {"tpch": 2.5, "tpcds": 2.0, "coffeeshop": 2.0, "job": 1.0, "stackoverflow": 2.0}[spec["dataset"]]
        for T in threads_ladder:
            units.append(dict(
                unit="query", unit_id=f"{spec['id']}__T{T}__tier{tier}", spec_id=spec["id"],
                dataset=spec["dataset"], tier=tier, threads=T, kind=spec["kind"], runs=DEFAULT_RUNS,
                group=f"real-{spec['dataset']}",
                cost=(spec["probe_rows_tier_a"] + spec["build_rows_tier_a"]) * scale,
            ))
    return units


def assign_shards(units: list[dict], shards: int) -> None:
    loads = [0.0] * shards
    for u in sorted(units, key=lambda u: (-u["cost"], u["unit_id"])):
        s = loads.index(min(loads))
        u["shard"] = s
        loads[s] += u["cost"]


def build_plan(suite: str, tier: str, shards: int) -> list[dict]:
    units = []
    if suite in ("synthetic", "all"):
        units += synthetic_plan()
    if suite in ("real", "all"):
        units += real_plan(tier)
    assign_shards(units, shards)
    return units


def plan_command(args: argparse.Namespace) -> int:
    units = build_plan(args.suite, args.tier, args.shards)
    print(json.dumps({"units": units}, indent=1))
    print(f"-- {len(units)} units ({sum(1 for u in units if u['unit'] == 'cell')} cells, "
          f"{sum(1 for u in units if u['unit'] == 'query')} queries), {args.shards} shards", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------
# Query SQL construction
# --------------------------------------------------------------------------


def join_kind_sql(kind: str) -> str:
    return {"INNER": "INNER JOIN", "LEFT": "LEFT JOIN", "LEFT SEMI": "LEFT SEMI JOIN",
            "LEFT ANTI": "LEFT ANTI JOIN"}[kind]


def checksum_projection(cols_probe: list[str], cols_build: list[str], kind: str) -> str:
    """count + order-insensitive content checksum. LEFT produces NULLs on the
    build side; SEMI/ANTI expose only probe columns."""
    probe = [f"p.{c}" for c in cols_probe]
    if kind in ("LEFT SEMI", "LEFT ANTI"):
        exprs = probe
    elif kind == "LEFT":
        exprs = probe + [f"coalesce(toString(b.{c}), '<null>')" for c in cols_build]
    else:
        exprs = probe + [f"b.{c}" for c in cols_build]
    return f"count() AS row_count, sum(cityHash64({', '.join(exprs)})) AS checksum"


def real_query_sql(spec: dict, tier: str, algorithm: str, threads: int, *, log_comment: str) -> str:
    db = real_db(spec["dataset"], tier)
    on = " AND ".join(f"p.{pk} = b.{bk}" for pk, bk in zip(spec["probe_keys"], spec["build_keys"]))
    build_src = f"{db}.{spec['build_table']}"
    if spec.get("filter"):
        build_src = f"(SELECT * FROM {build_src} WHERE {spec['filter']})"
    where = ""
    if spec.get("band") and spec["kind"] == "INNER":
        band = spec["band"]
        where = f" WHERE p.{band['probe_col']} BETWEEN b.{band['build_lo']} AND b.{band['build_hi']}"
    proj = checksum_projection(spec["probe_projection"], spec["build_projection"], spec["kind"])
    settings = jmb.join_settings(algorithm, threads, profiled=False, log_comment=log_comment)
    return (
        f"SELECT {proj} FROM {db}.{spec['probe_table']} AS p "
        f"{join_kind_sql(spec['kind'])} {build_src} AS b ON {on}{where} "
        f"{settings} FORMAT JSONEachRow"
    )


def syn_query_sql(cell: dict, algorithm: str, *, log_comment: str) -> str:
    keys = key_columns(cell["key"])
    on = " AND ".join(f"p.{c} = b.{c}" for c in keys)
    probe_cols = keys + [n for n, _ in jmb.payload_columns_ddl("p_p", cell["pp"])]
    build_payload = [n for n, _ in jmb.payload_columns_ddl("b_p", cell["bp"])]
    kind = cell["kind"]
    if kind == "INNER":
        proj = "count() AS row_count, sum(cityHash64(*)) AS checksum"  # jmb parity
    else:
        # LEFT makes build-side columns Nullable; SEMI/ANTI expose probe only.
        exprs = [f"p.{c}" for c in probe_cols]
        if kind == "LEFT":
            exprs += [f"coalesce(toString(b.{c}), '<null>')" for c in build_payload]
        proj = f"count() AS row_count, sum(cityHash64({', '.join(exprs)})) AS checksum"
    settings = jmb.join_settings(algorithm, cell["threads"], profiled=False, log_comment=log_comment)
    return (
        f"SELECT {proj} FROM {jmb.PROBE_TABLE} AS p "
        f"{join_kind_sql(kind)} {jmb.BUILD_TABLE} AS b ON {on} "
        f"{settings} FORMAT JSONEachRow"
    )


# --------------------------------------------------------------------------
# Measurement core
# --------------------------------------------------------------------------

FALLBACK_QUERY_TIMEOUT = 900.0


def tables_for_unit(unit: dict) -> list[str]:
    if unit["unit"] == "cell":
        return [jmb.BUILD_TABLE, jmb.PROBE_TABLE]
    spec = spec_by_id(unit["spec_id"])
    db = real_db(spec["dataset"], unit["tier"])
    return [f"{db}.{spec['probe_table']}", f"{db}.{spec['build_table']}"]


_SPECS_CACHE: dict[str, dict] | None = None


def spec_by_id(spec_id: str) -> dict:
    global _SPECS_CACHE
    if _SPECS_CACHE is None:
        _SPECS_CACHE = {s["id"]: s for s in load_specs()}
    return _SPECS_CACHE[spec_id]


def unit_query_sql(unit: dict, algorithm: str, *, log_comment: str) -> str:
    if unit["unit"] == "cell":
        return syn_query_sql(unit, algorithm, log_comment=log_comment)
    return real_query_sql(spec_by_id(unit["spec_id"]), unit["tier"], algorithm, unit["threads"],
                          log_comment=log_comment)


def parts_count(target: ExecTarget, tables: list[str]) -> dict[str, int]:
    counts = {}
    for t in tables:
        db, name = t.split(".")
        rows = sql_json(target, f"SELECT count() AS n FROM system.parts WHERE database = '{db}' AND table = '{name}' AND active")
        counts[t] = int(rows[0]["n"])
    return counts


def stop_merges(target: ExecTarget, tables: list[str]) -> None:
    for t in tables:
        sql_ok(target, f"SYSTEM STOP MERGES {t}", purpose=f"stop merges {t}")


def start_merges(target: ExecTarget, tables: list[str]) -> None:
    for t in tables:
        sql(target, f"SYSTEM START MERGES {t}")


def _query_once(target: ExecTarget, unit: dict, algorithm: str, log_comment: str):
    """One benchmark query. Returns ((row_count, checksum), None) or (None, error)."""
    q = unit_query_sql(unit, algorithm, log_comment=log_comment)
    rc, out, err = sql(target, q, timeout=FALLBACK_QUERY_TIMEOUT)
    if rc != 0:
        return None, f"rc={rc} stderr={err.decode(errors='replace')[:1000]}"
    row = json.loads(out.decode().strip().splitlines()[0])
    return (int(row["row_count"]), int(row.get("checksum") or 0)), None


def _collect_algorithm_stats(target: ExecTarget, algorithm: str, log_base: str, runs: int) -> dict:
    """Harvest one (arm, algorithm)'s timed runs from that arm's query_log:
    durations, memory, the FULL ProfileEvents map, and the start timestamps
    (the ABAB interleaving evidence)."""
    sql_ok(target, "SYSTEM FLUSH LOGS", timeout=120, purpose="flush logs")
    ql = sql_json(
        target,
        "SELECT query_duration_ms, memory_usage, ProfileEvents, log_comment, "
        "toUnixTimestamp64Micro(query_start_time_microseconds) AS start_us "
        f"FROM system.query_log WHERE log_comment LIKE '{log_base}|run%' AND type = 'QueryFinish' "
        "ORDER BY log_comment",
        timeout=120,
    )
    by_tag: dict[str, list[dict]] = {}
    for r in ql:
        tag = r["log_comment"].rsplit("|", 1)[-1]
        if tag.startswith("run") and tag[3:].isdigit():
            by_tag.setdefault(tag, []).append(r)
    expected_tags = [f"run{i}" for i in range(runs)]
    if sorted(by_tag) != sorted(expected_tags) or any(len(v) != 1 for v in by_tag.values()):
        return {"status": "INVALID",
                "reason": f"query_log rows per tag: { {t: len(v) for t, v in by_tag.items()} } != one each of {expected_tags}"}
    rows = [by_tag[t][0] for t in expected_tags]

    durations = [int(r["query_duration_ms"]) for r in rows]
    memories = [int(r["memory_usage"]) for r in rows]
    events_per_run = [{k: int(v) for k, v in (r.get("ProfileEvents") or {}).items()} for r in rows]
    all_names = sorted({n for e in events_per_run for n in e})
    median_events = {n: statistics.median([e.get(n, 0) for e in events_per_run]) for n in all_names}

    path_event = jmb.PATH_ASSERTION_EVENT[algorithm]
    fallback_runs = sum(1 for e in events_per_run if e.get(path_event, 0) == 0)
    jit_us = [e.get("CompileExpressionsMicroseconds", 0) for e in events_per_run]

    return {
        "status": "FALLBACK" if fallback_runs == runs else "OK",
        "fallback_runs": fallback_runs,
        "durations_ms": durations,
        "median_duration_ms": statistics.median(durations),
        "stdev_duration_ms": statistics.pstdev(durations) if len(durations) > 1 else 0.0,
        "memories_bytes": memories,
        "median_memory_bytes": statistics.median(memories),
        "peak_memory_bytes": max(memories),
        "events_per_run": events_per_run,
        "median_events": median_events,
        "run_start_us": [int(r["start_us"]) for r in rows],
        "jit_us_per_run": jit_us,
        "jit_compiled_timed_runs": sum(1 for v in jit_us if v > 0),
    }


def prepare_cell(target: ExecTarget, cell: dict) -> None:
    for stmt in syn_table_ddl(cell):
        sql_ok(target, stmt, purpose="cell ddl")
    for stmt in syn_fill_statements(cell):
        sql_ok(target, stmt, timeout=3600, purpose="cell fill")
    for t in (jmb.BUILD_TABLE, jmb.PROBE_TABLE):
        sql_ok(target, f"OPTIMIZE TABLE {t} FINAL", timeout=3600, purpose=f"optimize {t}")


def check_real_tables(target: ExecTarget, unit: dict) -> str | None:
    spec = spec_by_id(unit["spec_id"])
    db = real_db(spec["dataset"], unit["tier"])
    fps = stored_fingerprints(target, db)
    for t in (spec["probe_table"], spec["build_table"]):
        if t not in fps:
            return f"no fingerprint for {db}.{t} (dataset not loaded on this instance)"
    return None


def measure_unit(arms: dict[str, ExecTarget], unit: dict, *, algorithms: tuple[str, ...] = ALGORITHMS,
                 lead_flip: bool = False, selftest_mutate_cb=None,
                 wrong_expected: bool = False, min_timed_runs: int = 0,
                 unit_time_budget_s: float = 0.0) -> dict:
    """Measure every (arm, algorithm) for one unit. With two arms the timed
    runs are interleaved strict ABAB (leading arm reversed when `lead_flip`);
    both servers must be up, and an arm switch is one client invocation. One
    (row_count, checksum) reference is shared across ALL arms and algorithms,
    so cross-binary disagreement is INVALID, never scored."""
    nonce = uuid.uuid4().hex[:12]
    order = list(arms)
    if lead_flip and len(order) > 1:
        order.reverse()
    result = {"unit_id": unit["unit_id"], "unit": unit["unit"], "nonce": nonce,
              "tool_version": TOOL_VERSION, "meta": unit, "lead_arm": order[0],
              "algorithms_measured": list(algorithms),
              "arms": {name: {"binary": arms[name].binary, "port": arms[name].port,
                              "binary_sha256": binary_sha256(arms[name]), "algorithms": {}}
                       for name in arms}}
    if unit["unit"] == "cell":
        expected_rows = expected_rows_for_kind(unit)
        if wrong_expected and expected_rows is not None:
            expected_rows += 1  # selftest must-fail hook
        for name in order:
            prepare_cell(arms[name], unit)
        warmups = SYN_WARMUPS
    else:
        for name in order:
            missing = check_real_tables(arms[name], unit)
            if missing:
                result.update(status="MISSING_DATA", reason=f"arm {name}: {missing}")
                return result
        expected_rows = None
        warmups = REAL_WARMUPS

    tables = tables_for_unit(unit)
    # `min_timed_runs` raises the sample count without lowering any plan's own count:
    # five timed runs leave the per-arm MEDIAN noisy enough that an A/A control can
    # report a 4% difference between two runs of the same binary, so the probe-phase
    # campaign asks for more samples rather than a wider band.
    runs = max(unit.get("runs", DEFAULT_RUNS), min_timed_runs)
    for name in order:
        stop_merges(arms[name], tables)
    try:
        expected: tuple[int, int] | None = None
        for algorithm in algorithms:
            log_bases = {n: f"jbmt|{unit['unit_id']}|{nonce}|{n}|{algorithm}" for n in order}
            failed: dict[str, str] = {}
            parts_before: dict[str, dict] = {}
            for name in order:
                parts_before[name] = parts_count(arms[name], tables)
                for w in range(warmups):
                    t_warm = time.time()
                    got, err = _query_once(arms[name], unit, algorithm, f"{log_bases[name]}|warmup{w}")
                    warm_s = time.time() - t_warm
                    # Uniform, direction-blind time box: a unit whose single query costs more
                    # than the budget would spend hours of a bounded run on one unit
                    # ((warmups + runs) x arms queries), so it is dropped BEFORE any timed run
                    # rather than half-measured. The decision uses wall clock only - never either
                    # metric - so it cannot favour an arm, and the unit is recorded OVER_BUDGET,
                    # i.e. NO-VERDICT with a stated reason.
                    # The budget is checked BEFORE the error branch: a warmup that fails by
                    # exhausting `max_execution_time` is the most expensive case there is, and
                    # letting it fall through to `break` would charge the run the full timeout on
                    # every arm before abandoning the unit anyway.
                    if unit_time_budget_s and warm_s > unit_time_budget_s:
                        result.update(status="OVER_BUDGET",
                                      reason=f"arm {name} warmup {w} took {warm_s:.1f}s > "
                                             f"unit-time-budget {unit_time_budget_s}s"
                                             + (f" (and failed: {err[:120]})" if err else "")
                                             + "; unit skipped before any timed run")
                        return result
                    if err:
                        failed[name] = f"warmup {w} failed: {err}"
                        break
                    if expected is None:
                        expected = got
                    elif got != expected:
                        failed[name] = f"warmup {w} result {got} != expected {expected}"
                        break
            if algorithm == algorithms[0] and selftest_mutate_cb:
                selftest_mutate_cb()  # selftest hook: dirty a table mid-flight
            for i in range(runs):
                for name in order:
                    if name in failed:
                        continue
                    got, err = _query_once(arms[name], unit, algorithm, f"{log_bases[name]}|run{i}")
                    if err:
                        failed[name] = f"run {i} failed: {err}"
                    elif got != expected:
                        failed[name] = f"run {i} result {got} != expected {expected}"
            for name in order:
                if name in failed:
                    algo_result = {"status": "INVALID", "reason": failed[name]}
                elif expected is None:
                    algo_result = {"status": "INVALID",
                                   "reason": "no reference result established (warmups=0 and no prior algorithm)"}
                else:
                    parts_after = parts_count(arms[name], tables)
                    if parts_after != parts_before[name]:
                        algo_result = {"status": "INVALID",
                                       "reason": f"part counts changed during timing: {parts_before[name]} -> {parts_after}"}
                    else:
                        algo_result = _collect_algorithm_stats(arms[name], algorithm, log_bases[name], runs)
                        if algo_result["status"] != "INVALID":
                            algo_result.update(row_count=expected[0], checksum=expected[1],
                                               runs=runs, warmups=warmups)
                result["arms"][name]["algorithms"][algorithm] = algo_result

        statuses = {f"{n}:{a}": r.get("status")
                    for n in order for a, r in result["arms"][n]["algorithms"].items()}
        invalid_reasons = {f"{n}:{a}": r.get("reason", r.get("status"))
                           for n in order for a, r in result["arms"][n]["algorithms"].items()
                           if r.get("status") == "INVALID"}
        if expected_rows is not None and expected is not None and expected[0] != expected_rows:
            result.update(status="INVALID",
                          reason=f"row_count {expected[0]} != closed-form expected {expected_rows}")
        elif statuses and all(s == "OK" for s in statuses.values()):
            result["status"] = "OK"
        elif invalid_reasons or not statuses:
            result.update(status="INVALID", reason=str(invalid_reasons))
        else:
            result["status"] = "FALLBACK"
        result["expected_rows_closed_form"] = expected_rows
    finally:
        for name in order:
            start_merges(arms[name], tables)
    if len(arms) == 1:
        # legacy readers (v1 report tooling) expect a top-level algorithms map
        result["algorithms"] = result["arms"][order[0]]["algorithms"]
    return result


# --------------------------------------------------------------------------
# Sweep (resumable) and single-unit runners
# --------------------------------------------------------------------------


def completed_unit_ids(results_path: pathlib.Path) -> set[str]:
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") in ("OK", "FALLBACK"):
                done.add(row["unit_id"])
    return done


def append_result(results_path: pathlib.Path, row: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_resume_arms(results_path: pathlib.Path, arms: dict[str, ExecTarget],
                      algorithms: tuple[str, ...] = ALGORITHMS) -> None:
    """Refuse to resume into a results file written with different arms or a
    different algorithm set: mixing either in one file would silently corrupt
    the comparison."""
    if not results_path.exists():
        return
    want = {n: binary_sha256(t) for n, t in arms.items()}
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        got = {n: a.get("binary_sha256") for n, a in (row.get("arms") or {}).items()}
        if got and got != want:
            raise SystemExit(f"{results_path} was written with arms {got}, current arms are {want}; "
                             "use a fresh --results file")
        if not got and len(arms) > 1:
            raise SystemExit(f"{results_path} holds arm-less legacy rows; use a fresh --results file "
                             "for a multi-arm sweep")
        got_algos = row.get("algorithms_measured")
        if got_algos and list(algorithms) != got_algos:
            raise SystemExit(f"{results_path} was written with algorithms {got_algos}, current are "
                             f"{list(algorithms)}; use a fresh --results file")


def sweep_command(args: argparse.Namespace) -> int:
    arms = arms_from_args(args)
    algorithms = algorithms_from_args(args)
    units = build_plan(args.suite, args.tier, args.shards)
    mine = [u for u in units if u["shard"] == args.shard]
    if args.only:
        rx = re.compile(args.only)
        mine = [u for u in mine if rx.search(u["unit_id"])]
    results_path = pathlib.Path(args.results)
    check_resume_arms(results_path, arms, algorithms)
    done = completed_unit_ids(results_path)
    todo = [u for u in mine if u["unit_id"] not in done]
    print("arms: " + ", ".join(f"{n}={t.binary}:{t.port} sha={binary_sha256(t)[:12]}"
                               for n, t in arms.items()))
    print(f"algorithms: {','.join(algorithms)}")
    if len(algorithms) == 1 and len(arms) == 1:
        print(f"WARNING: one algorithm and one arm - real-suite units have no independent "
              f"(row_count, checksum) oracle in this configuration; only within-run stability "
              f"and the part-count guard apply", file=sys.stderr)
    print(f"shard {args.shard}/{args.shards}: {len(mine)} units, {len(mine) - len(todo)} done, {len(todo)} to run")
    failures = 0
    for i, unit in enumerate(todo):
        t0 = time.time()
        # the leading arm alternates by a unit-id hash: deterministic, stable
        # under resume (unlike the position in the shrinking todo list)
        lead_flip = bool(zlib.crc32(unit["unit_id"].encode()) & 1)
        try:
            row = measure_unit(arms, unit, algorithms=algorithms, lead_flip=lead_flip,
                               min_timed_runs=getattr(args, "min_timed_runs", 0),
                               unit_time_budget_s=getattr(args, "unit_time_budget", 0.0))
        except Exception as exc:  # noqa: BLE001 - a sweep must survive one bad unit
            row = {"unit_id": unit["unit_id"], "unit": unit["unit"], "meta": unit,
                   "status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}
        row["wall_seconds"] = round(time.time() - t0, 1)
        append_result(results_path, row)
        if row["status"] not in ("OK", "FALLBACK"):
            failures += 1
        print(f"[{i + 1}/{len(todo)}] {unit['unit_id']}: {row['status']} ({row['wall_seconds']}s)")
    print(f"sweep done: {len(todo)} run, {failures} not OK/FALLBACK")
    return 0 if failures == 0 else 1


def run_unit_command(args: argparse.Namespace) -> int:
    arms = arms_from_args(args)
    algorithms = algorithms_from_args(args)
    units = {u["unit_id"]: u for u in build_plan("all", args.tier, 1)}
    unit = units.get(args.unit_id)
    if unit is None:
        matches = [uid for uid in units if args.unit_id in uid]
        print(f"unit {args.unit_id!r} not in plan; close matches: {matches[:10]}", file=sys.stderr)
        return 1
    row = measure_unit(arms, unit, algorithms=algorithms)
    print(json.dumps(row, indent=1))
    return 0 if row["status"] in ("OK", "FALLBACK") else 1


# --------------------------------------------------------------------------
# Report: wall AND peak memory as scored axes
# --------------------------------------------------------------------------


def load_results_rows(specs: list[str]) -> list[dict]:
    rows: dict[str, dict] = {}
    for spec in specs:
        if any(ch in spec for ch in "*?["):
            import glob as globmod
            paths = [pathlib.Path(p) for p in sorted(globmod.glob(spec))]
        else:
            paths = [pathlib.Path(spec)]
        for path in paths:
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[row["unit_id"]] = row  # later files win
    return list(rows.values())


def arm_algorithms(row: dict, arm: str | None) -> dict | None:
    """The algorithm->stats block to score: an explicit arm's, the only arm's,
    or the legacy top-level block. None when the choice is ambiguous (a
    multi-arm row without --arm)."""
    if arm:
        return ((row.get("arms") or {}).get(arm) or {}).get("algorithms")
    if "algorithms" in row:
        return row["algorithms"]
    row_arms = row.get("arms") or {}
    if len(row_arms) == 1:
        return next(iter(row_arms.values()))["algorithms"]
    return None


def score_axis(row: dict, axis: str, arm: str | None = None) -> tuple[str, float] | None:
    """Return (verdict, ratio parallel/partitioned) for one axis, or None if
    the row cannot be scored. Ratio > 1 means partitioned is better."""
    algos = arm_algorithms(row, arm) or {}
    part, par = algos.get("partitioned_hash"), algos.get("parallel_hash")
    if not part or not par or part.get("status") != "OK" or par.get("status") != "OK":
        return None
    if axis == "wall":
        a, b = part["median_duration_ms"], par["median_duration_ms"]
        sa = part["stdev_duration_ms"]
        sb = par["stdev_duration_ms"]
    else:
        a, b = part["median_memory_bytes"], par["median_memory_bytes"]
        sa = statistics.pstdev(part["memories_bytes"])
        sb = statistics.pstdev(par["memories_bytes"])
    if a <= 0 or b <= 0:
        return None
    ratio = b / a
    if jmb._noise_band_tie(a, b, sa, sb):
        return ("tie", ratio)
    return ("win" if a < b else "loss", ratio)


def report_command(args: argparse.Namespace) -> int:
    rows = load_results_rows(args.results)
    planned = build_plan(args.suite, args.tier, 1)
    planned_ids = {u["unit_id"] for u in planned}
    have = {r["unit_id"] for r in rows}
    lines = []
    lines.append(f"# join_bench_mt report ({args.suite}, tier {args.tier})")
    lines.append("")
    lines.append(f"Planned {len(planned_ids)} units; results for {len(planned_ids & have)}; "
                 f"missing {len(planned_ids - have)}; extraneous {len(have - planned_ids)}.")
    by_status = collections.Counter(r["status"] for r in rows)
    lines.append(f"Statuses: {dict(by_status)}")
    for status in ("INVALID", "ERROR", "MISSING_DATA"):
        bad = [r["unit_id"] for r in rows if r["status"] == status]
        if bad:
            lines.append(f"- {status}: {bad}")
    fallback = [r["unit_id"] for r in rows if r["status"] == "FALLBACK"]
    if fallback:
        lines.append(f"- FALLBACK (intended algorithm not used; excluded from scoring): {fallback}")

    def all_algo_blocks(r: dict) -> list[dict]:
        if r.get("arms"):
            return [a for ab in r["arms"].values() for a in ab.get("algorithms", {}).values()]
        return list(r.get("algorithms", {}).values())

    jit = [r["unit_id"] for r in rows
           for a in all_algo_blocks(r) if a.get("jit_compiled_timed_runs")]
    if jit:
        lines.append(f"- JIT-contaminated timed runs in: {sorted(set(jit))}")
    ambiguous = sum(1 for r in rows if arm_algorithms(r, args.arm) is None)
    if ambiguous:
        lines.append(f"- NOTE: {ambiguous} multi-arm rows not scored (pass --arm NAME; "
                     "cross-arm comparison lives in `report-ab`)")
    single_algo = sorted({",".join(r["algorithms_measured"]) for r in rows
                          if len(r.get("algorithms_measured") or ALGORITHMS) < 2})
    if single_algo:
        lines.append(f"- NOTE: rows measured with a single algorithm ({single_algo}) cannot be "
                     "scored here; this report compares the two algorithms against each other")

    def group_of(r):
        return (r.get("meta") or {}).get("group", "?")

    for axis in ("wall", "peak memory"):
        ax = "wall" if axis == "wall" else "memory"
        lines.append("")
        lines.append(f"## {axis} (ratio = parallel/partitioned, >1 partitioned better)")
        scored = [(r, score_axis(r, ax, args.arm)) for r in rows]
        scored = [(r, s) for r, s in scored if s]
        verdicts = collections.Counter(s[0] for _, s in scored)
        ratios = sorted(s[1] for _, s in scored)
        if not ratios:
            lines.append("no scorable rows")
            continue
        lines.append(f"overall: {dict(verdicts)}; median ratio {statistics.median(ratios):.3f}, "
                     f"p10 {jmb._quantile(ratios, 0.10):.3f}, p90 {jmb._quantile(ratios, 0.90):.3f}")
        by_group: dict[str, list] = collections.defaultdict(list)
        for r, s in scored:
            by_group[group_of(r)].append(s)
        lines.append("")
        lines.append("| group | n | win | tie | loss | median ratio |")
        lines.append("|---|---|---|---|---|---|")
        for g in sorted(by_group):
            ss = by_group[g]
            v = collections.Counter(x[0] for x in ss)
            med = statistics.median([x[1] for x in ss])
            lines.append(f"| {g} | {len(ss)} | {v.get('win', 0)} | {v.get('tie', 0)} | {v.get('loss', 0)} | {med:.3f} |")
        losses = sorted(((s[1], r["unit_id"]) for r, s in scored if s[0] == "loss"))[:15]
        if losses:
            lines.append("")
            lines.append(f"worst {axis} losses: " + ", ".join(f"{uid}={ratio:.2f}" for ratio, uid in losses))
    text = "\n".join(lines) + "\n"
    if args.out:
        pathlib.Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def score_ab(row: dict, algorithm: str, axis: str, arm_a: str, arm_b: str) -> tuple[str, float] | None:
    """(verdict, ratio arm_b/arm_a) for one unit x algorithm x axis; ratio > 1
    and 'win' both mean arm_a is better. None when either arm is not OK."""
    blocks = row.get("arms") or {}
    a = (blocks.get(arm_a) or {}).get("algorithms", {}).get(algorithm)
    b = (blocks.get(arm_b) or {}).get("algorithms", {}).get(algorithm)
    if not a or not b or a.get("status") != "OK" or b.get("status") != "OK":
        return None
    if axis == "wall":
        va, vb = a["median_duration_ms"], b["median_duration_ms"]
        sa, sb = a["stdev_duration_ms"], b["stdev_duration_ms"]
    else:
        va, vb = a["median_memory_bytes"], b["median_memory_bytes"]
        sa, sb = statistics.pstdev(a["memories_bytes"]), statistics.pstdev(b["memories_bytes"])
    if va <= 0 or vb <= 0:
        return None
    ratio = vb / va
    if jmb._noise_band_tie(va, vb, sa, sb):
        return ("tie", ratio)
    return ("win" if va < vb else "loss", ratio)


def report_ab_command(args: argparse.Namespace) -> int:
    """Cross-arm A/B report: per algorithm, wall & peak memory verdicts for
    arm_a vs arm_b - the replacement for offline-joining two one-binary passes."""
    rows = load_results_rows(args.results)
    two_arm = [r for r in rows if len(r.get("arms") or {}) >= 2]
    if not two_arm:
        print("no multi-arm rows in the given results", file=sys.stderr)
        return 1
    names = list(two_arm[0]["arms"])
    arm_a = args.arm_a or names[0]
    arm_b = args.arm_b or next(n for n in names if n != arm_a)
    lines = [f"# join_bench_mt A/B report: {arm_a} vs {arm_b} "
             f"(ratio = {arm_b}/{arm_a}; ratio > 1 and 'win' mean {arm_a} better)", ""]
    by_status = collections.Counter(r["status"] for r in rows)
    lines.append(f"{len(rows)} result rows ({len(two_arm)} multi-arm); statuses: {dict(by_status)}")
    for status in ("INVALID", "ERROR", "MISSING_DATA", "FALLBACK"):
        bad = [r["unit_id"] for r in rows if r["status"] == status]
        if bad:
            lines.append(f"- {status}: {bad}")
    shas = {n: sorted({r["arms"][n]["binary_sha256"][:12] for r in two_arm if n in r["arms"]})
            for n in (arm_a, arm_b)}
    lines.append(f"binaries: {shas}")
    leads = collections.Counter(r.get("lead_arm", "?") for r in two_arm)
    lines.append(f"lead arm distribution (ABAB leader): {dict(leads)}")

    measured = [a for a in ALGORITHMS
                if any(a in (r["arms"].get(arm_a) or {}).get("algorithms", {}) for r in two_arm)]
    for axis in ("wall", "memory"):
        for algorithm in measured:
            lines.append("")
            lines.append(f"## {axis} / {algorithm}")
            scored = [(r, score_ab(r, algorithm, axis, arm_a, arm_b)) for r in two_arm]
            scored = [(r, s) for r, s in scored if s]
            if not scored:
                lines.append("no scorable rows")
                continue
            verdicts = collections.Counter(s[0] for _, s in scored)
            ratios = sorted(s[1] for _, s in scored)
            lines.append(f"n={len(scored)}: {dict(verdicts)}; median ratio {statistics.median(ratios):.3f}, "
                         f"p10 {jmb._quantile(ratios, 0.10):.3f}, p90 {jmb._quantile(ratios, 0.90):.3f}")
            by_group: dict[str, list] = collections.defaultdict(list)
            for r, s in scored:
                by_group[(r.get("meta") or {}).get("group", "?")].append(s)
            lines.append("")
            lines.append("| group | n | win | tie | loss | median ratio |")
            lines.append("|---|---|---|---|---|---|")
            for g in sorted(by_group):
                ss = by_group[g]
                v = collections.Counter(x[0] for x in ss)
                lines.append(f"| {g} | {len(ss)} | {v.get('win', 0)} | {v.get('tie', 0)} | "
                             f"{v.get('loss', 0)} | {statistics.median([x[1] for x in ss]):.3f} |")
            worst = sorted(((s[1], r["unit_id"]) for r, s in scored if s[0] == "loss"))[:15]
            if worst:
                lines.append("")
                lines.append("worst losses: " + ", ".join(f"{uid}={ratio:.2f}" for ratio, uid in worst))
    text = "\n".join(lines) + "\n"
    if args.out:
        pathlib.Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------
# Keys preparation (synthetic) and selftest
# --------------------------------------------------------------------------


def prepare_keys_command(args: argparse.Namespace) -> int:
    target = target_from_args(args)
    keys = args.keys.split(",") if args.keys else sorted(set(list(jmb.KEY_CONFIGS) + list(LOCAL_KEY_CONFIGS)))
    sql_ok(target, "CREATE DATABASE IF NOT EXISTS keys_store")
    for key in keys:
        print(f"preparing keys_store for {key}")
        if key in LOCAL_KEY_CONFIGS:
            for stmt in local_keys_store_ddl(key):
                sql_ok(target, stmt, timeout=3600, purpose=f"keys {key}")
        else:
            cfg = jmb.KEY_CONFIGS[key]
            sql_ok(target, f"DROP TABLE IF EXISTS {jmb.keys_store_table_name(key)}")
            sql_ok(target, jmb.keys_store_create_sql(cfg), purpose=f"create keys {key}")
            sql_ok(target, jmb.keys_store_fill_sql(cfg), timeout=4 * 3600, purpose=f"fill keys {key}")
            sql_ok(target, f"OPTIMIZE TABLE {jmb.keys_store_table_name(key)} FINAL", timeout=3600)
        row = sql_json(target, f"SELECT count() AS n FROM {jmb.keys_store_table_name(key)}")[0]
        print(f"  {key}: {row['n']} rows")
    return 0


def selftest_command(args: argparse.Namespace) -> int:
    arms = arms_from_args(args)
    lead = next(iter(arms))
    target = arms[lead]
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    print("selftest: bootstrapping a small keys_store.k0 (1M hit ranks + 1M miss-domain ranks) per arm")
    for arm_name, t in arms.items():
        sql_ok(t, "CREATE DATABASE IF NOT EXISTS keys_store")
        have = sql_json(t, "SELECT count() AS n FROM system.tables WHERE database = 'keys_store' AND name = 'k0'")
        if int(have[0]["n"]) == 0:
            cfg = jmb.KEY_CONFIGS["K0"]
            sql_ok(t, jmb.keys_store_create_sql(cfg), purpose="selftest keys create")
            hit = ", ".join(f"{e} AS {n}" for e, n in zip(jmb.key_value_exprs(cfg, "number", jmb.KEY_SEED_HIT), cfg.key_columns))
            sql_ok(t, f"INSERT INTO {jmb.keys_store_table_name('K0')} "
                      f"SELECT number AS rank, {hit} FROM numbers(1000000)", timeout=600)
            # miss domain (ranks >= n_max), needed by h < 1.0 proof cells
            miss = ", ".join(
                f"{e} AS {n}"
                for e, n in zip(jmb.key_value_exprs(cfg, f"(number + {cfg.n_max})", jmb.KEY_SEED_MISS), cfg.key_columns)
            )
            sql_ok(t, f"INSERT INTO {jmb.keys_store_table_name('K0')} "
                      f"SELECT number + {cfg.n_max} AS rank, {miss} FROM numbers(1000000)", timeout=600)

    cell = make_syn_cell(D=100_000, key="K0", m_p=4, threads=8, group="SELFTEST", runs=3)

    print("selftest 1: nominal cell must be OK with full events captured")
    nominal_row = measure_unit(arms, cell)
    row = nominal_row
    part = row["arms"][lead]["algorithms"].get("partitioned_hash", {})
    check("status OK", row["status"] == "OK", row.get("reason", ""))
    ev = part.get("median_events", {})
    check("full ProfileEvents captured (uncurated event present)",
          any(k for k in ev if k.startswith("OSCPU") or k == "SelectedRows"),
          f"{len(ev)} event names")
    check("partitioned path event nonzero",
          ev.get(jmb.PATH_ASSERTION_EVENT["partitioned_hash"], 0) > 0)

    print("selftest 2 (must-fail): wrong closed-form expectation is detected")
    row = measure_unit(arms, cell, wrong_expected=True)
    check("wrong expected -> INVALID", row["status"] == "INVALID", row.get("reason", "")[:100])

    print("selftest 3 (must-fail): mutation during timing is detected")
    mutate = lambda: sql_ok(target, f"INSERT INTO {jmb.BUILD_TABLE} SELECT * FROM {jmb.BUILD_TABLE} LIMIT 1")
    row = measure_unit(arms, cell, selftest_mutate_cb=mutate)
    check("mid-run insert -> INVALID (parts or checksum)", row["status"] == "INVALID", row.get("reason", "")[:100])

    print("selftest 4: fallback detection")
    # 4a - soundness: under join_algorithm='hash' NEITHER algorithm's path event
    # may fire; a spurious count would blind the zero-event fallback detector.
    # Checked per arm: the arms may run different binaries.
    for arm_name, t in arms.items():
        prepare_cell(t, cell)
        nonce4 = uuid.uuid4().hex[:8]
        q = (
            "SELECT count() AS row_count "
            f"FROM {jmb.PROBE_TABLE} AS p INNER JOIN {jmb.BUILD_TABLE} AS b ON p.k0 = b.k0 "
            + jmb.join_settings("hash", 8, profiled=False, log_comment=f"jbmt|selftest4a|{nonce4}")
            + " FORMAT JSONEachRow"
        )
        rc, out, err = sql(t, q, timeout=600)
        if rc != 0:
            check(f"plain-hash probe query ran ({arm_name})", False, err.decode(errors="replace")[:200])
        else:
            sql_ok(t, "SYSTEM FLUSH LOGS", timeout=120)
            ql = sql_json(t, "SELECT ProfileEvents FROM system.query_log "
                             f"WHERE log_comment = 'jbmt|selftest4a|{nonce4}' AND type = 'QueryFinish' "
                             "ORDER BY event_time_microseconds DESC LIMIT 1")
            events = {k: int(v) for k, v in (ql[0].get("ProfileEvents") or {}).items()} if ql else {}
            check(f"no spurious path events under plain hash ({arm_name})",
                  all(events.get(ev, 0) == 0 for ev in jmb.PATH_ASSERTION_EVENT.values()))
    # 4b - end to end: a LEFT ANTI cell must come back FALLBACK exactly when the
    # intended algorithm's path event stayed zero (on branches where the
    # algorithms support ANTI it must be OK instead - both are consistent).
    anti = make_syn_cell(D=100_000, key="K0", m_p=4, h="0.9", kind="LEFT ANTI",
                         threads=8, group="SELFTEST", runs=2)
    row = measure_unit(arms, anti)
    consistent = False
    detail = row.get("reason", "")
    if row["status"] in ("OK", "FALLBACK"):
        algo_status = {f"{n}:{a}": r["status"]
                       for n, ab in row["arms"].items() for a, r in ab["algorithms"].items()}
        consistent = (row["status"] == "FALLBACK") == any(s == "FALLBACK" for s in algo_status.values())
        detail = f"unit={row['status']} per-arm-algorithm={algo_status}"
    check("LEFT ANTI unit status consistent with path events", consistent, detail)

    if len(arms) >= 2:
        print("selftest 5: two-arm timed runs interleave strict ABAB")
        for algorithm in ALGORITHMS:
            starts = {n: nominal_row["arms"][n]["algorithms"][algorithm].get("run_start_us")
                      for n in arms}
            if any(not s for s in starts.values()):
                check(f"run timestamps present ({algorithm})", False, str(starts))
                continue
            merged = sorted((ts, n) for n, tss in starts.items() for ts in tss)
            seq = [n for _, n in merged]
            alternates = all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))
            check(f"timed runs alternate arms ({algorithm})", alternates, "".join(seq))

    print("selftest 6 (must-fail): fingerprint mismatch is detected")
    sql_ok(target, "CREATE DATABASE IF NOT EXISTS jbmt_selftest_a")
    sql_ok(target, "DROP TABLE IF EXISTS jbmt_selftest_a.t")
    sql_ok(target, "CREATE TABLE jbmt_selftest_a.t (k UInt64) ENGINE = MergeTree ORDER BY k")
    sql_ok(target, "INSERT INTO jbmt_selftest_a.t SELECT number FROM numbers(1000)")
    fp1 = sql_json(target, fingerprint_sql("jbmt_selftest_a", "t", "k"))[0]
    sql_ok(target, "INSERT INTO jbmt_selftest_a.t VALUES (999999)")
    fp2 = sql_json(target, fingerprint_sql("jbmt_selftest_a", "t", "k"))[0]
    check("fingerprint changes on mutation", fp1 != fp2, f"{fp1} vs {fp2}")
    sql_ok(target, "DROP DATABASE jbmt_selftest_a")

    print("selftest:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Fleet helpers (thin AWS CLI wrappers; operator-driven, never implicit)
# --------------------------------------------------------------------------


def _aws(args: argparse.Namespace, *cmd: str) -> dict | list | None:
    argv = ["aws", *cmd, "--region", args.region, "--output", "json"]
    if args.aws_profile:
        argv += ["--profile", args.aws_profile]
    proc = subprocess.run(argv, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"aws {' '.join(cmd[:3])} failed: {proc.stderr.decode(errors='replace')[:1000]}")
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def fleet_snapshot_command(args: argparse.Namespace) -> int:
    out = _aws(args, "ec2", "create-snapshot", "--volume-id", args.volume_id,
               "--description", f"jbmt data {args.tag}",
               "--tag-specifications", f"ResourceType=snapshot,Tags=[{{Key=Name,Value=jbmt-{args.tag}}}]")
    assert isinstance(out, dict)
    print(json.dumps(out, indent=1))
    print("wait with: aws ec2 wait snapshot-completed --snapshot-ids", out["SnapshotId"])
    return 0


def fleet_volumes_command(args: argparse.Namespace) -> int:
    ids = []
    for az_instance in args.attach.split(","):
        az, instance = az_instance.split(":")
        vol = _aws(args, "ec2", "create-volume", "--snapshot-id", args.snapshot_id,
                   "--availability-zone", az, "--volume-type", "gp3",
                   "--iops", "4000", "--throughput", "1000",
                   "--tag-specifications", f"ResourceType=volume,Tags=[{{Key=Name,Value=jbmt-{args.tag}}}]")
        assert isinstance(vol, dict)
        ids.append((vol["VolumeId"], instance))
        print(f"created {vol['VolumeId']} in {az} for {instance}")
    print("after 'available': attach with aws ec2 attach-volume --device /dev/sdf "
          "--volume-id <vol> --instance-id <instance>")
    print(json.dumps({"volumes": ids}, indent=1))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="join_bench_mt", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="emit the unit plan (cells + query specs) with shard assignment")
    p.add_argument("--suite", choices=("synthetic", "real", "all"), default="all")
    p.add_argument("--tier", choices=tuple(TIERS), default="a")
    p.add_argument("--shards", type=int, default=1)
    p.set_defaults(handler=plan_command)

    p = sub.add_parser("load-real", help="create + load the real datasets on THIS host (loader instance)")
    add_target_args(p)
    p.add_argument("--datasets", default="tpch,tpcds,coffeeshop,stackoverflow,job")
    p.add_argument("--tier", choices=tuple(TIERS), default="a")
    p.add_argument("--workdir", default="/mnt/data/jbmt_work")
    p.add_argument("--duckdb", default=None, help="duckdb CLI path (downloaded automatically if omitted)")
    p.add_argument("--limit-rows", type=int, default=None, help="dev/smoke: LIMIT applied to S3-sourced loads")
    p.set_defaults(handler=load_real_command)

    p = sub.add_parser("verify", help="recompute fingerprints; compare with stored and/or a reference JSON")
    add_target_args(p)
    p.add_argument("--datasets", default="tpch,tpcds,coffeeshop,stackoverflow,job")
    p.add_argument("--tier", choices=tuple(TIERS), default="a")
    p.add_argument("--reference", default=None, help="loads JSON emitted on the loader instance")
    p.add_argument("--emit", default=None, help="write this instance's fingerprints to a JSON file")
    p.set_defaults(handler=verify_command)

    p = sub.add_parser("prepare-keys", help="create keys_store tables (synthetic suite)")
    add_target_args(p)
    p.add_argument("--keys", default=None, help="comma list; default = all (K0-K9 + K10 + KF2)")
    p.set_defaults(handler=prepare_keys_command)

    p = sub.add_parser("sweep", help="run this shard's not-yet-complete units, resumably")
    add_target_args(p)
    add_arm_args(p)
    add_algorithm_args(p)
    p.add_argument("--suite", choices=("synthetic", "real", "all"), default="all")
    p.add_argument("--tier", choices=tuple(TIERS), default="a")
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--only", default=None, help="regex filter on unit ids (debugging/partial runs)")
    p.add_argument("--unit-time-budget", type=float, default=0.0, metavar="SECONDS",
                   help="skip a unit before any timed run if its first warmup exceeds SECONDS "
                        "(status OVER_BUDGET). Uniform and wall-clock only, so it cannot favour "
                        "an arm; keeps one pathological unit from eating a bounded run")
    p.add_argument("--min-timed-runs", type=int, default=0, metavar="N",
                   help="raise every unit's timed-run count to at least N (never lowers it); "
                        "more samples stabilize the per-arm median an A/B verdict rests on")
    p.set_defaults(handler=sweep_command)

    p = sub.add_parser("run-unit", help="run one unit by id (debugging)")
    add_target_args(p)
    add_arm_args(p)
    add_algorithm_args(p)
    p.add_argument("--tier", choices=tuple(TIERS), default="a")
    p.add_argument("unit_id")
    p.set_defaults(handler=run_unit_command)

    p = sub.add_parser("report", help="coverage + wall & memory win/tie/loss report")
    p.add_argument("--results", nargs="+", required=True, help="results jsonl paths/globs")
    p.add_argument("--suite", choices=("synthetic", "real", "all"), default="all")
    p.add_argument("--tier", choices=tuple(TIERS), default="a")
    p.add_argument("--arm", default=None, help="arm name to score (required for multi-arm results)")
    p.add_argument("--out", default=None)
    p.set_defaults(handler=report_command)

    p = sub.add_parser("report-ab", help="cross-arm A/B report (wall & memory per algorithm)")
    p.add_argument("--results", nargs="+", required=True, help="results jsonl paths/globs")
    p.add_argument("--arm-a", default=None, help="reference arm (default: first arm in the results)")
    p.add_argument("--arm-b", default=None, help="compared arm (default: the other arm)")
    p.add_argument("--out", default=None)
    p.set_defaults(handler=report_ab_command)

    p = sub.add_parser("selftest", help="nominal + must-fail proofs against a running server (or two, with --arm)")
    add_target_args(p)
    add_arm_args(p)
    p.set_defaults(handler=selftest_command)

    p = sub.add_parser("fetch-duckdb", help="download the duckdb CLI for this arch")
    p.add_argument("--workdir", default="/mnt/data/jbmt_work")
    p.set_defaults(handler=lambda a: (print(fetch_duckdb(pathlib.Path(a.workdir))), 0)[1])

    p = sub.add_parser("fleet-snapshot", help="EBS-snapshot the loaded data volume")
    p.add_argument("--volume-id", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--region", default="ap-south-2")
    p.add_argument("--aws-profile", default=None)
    p.set_defaults(handler=fleet_snapshot_command)

    p = sub.add_parser("fleet-volumes", help="create per-shard volumes from the snapshot")
    p.add_argument("--snapshot-id", required=True)
    p.add_argument("--attach", required=True, help="comma list az:instance-id")
    p.add_argument("--tag", required=True)
    p.add_argument("--region", default="ap-south-2")
    p.add_argument("--aws-profile", default=None)
    p.set_defaults(handler=fleet_volumes_command)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
