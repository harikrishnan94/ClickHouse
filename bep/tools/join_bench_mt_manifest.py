#!/usr/bin/env python3
"""Generate the real-dataset join-query manifest for `join_bench_mt.py`.

Parses the ClickBench `versions` benchmark queries (TPC-H, TPC-DS, CoffeeShop,
JOB) plus hand-curated StackOverflow docs joins, extracts every distinct
equality-join edge, and emits one benchmark query spec per edge: the pairwise
join with the projections each side actually carries in the source queries.

Inputs (defaults assume the repo layout used during development):
  --clickbench   path to a ClickBench checkout (versions/queries, versions/create/schema)
  --out          output JSON path (default: bep/tools/join_bench_mt_queries.json)

The output is committed next to the harness so instances never need the
ClickBench checkout. Regenerate only when the upstream queries change.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

DATASET_TABLES = {
    "tpch": ["lineitem", "orders", "partsupp", "part", "supplier", "customer", "nation", "region"],
    "tpcds": [
        "call_center", "catalog_page", "catalog_returns", "catalog_sales", "tpcds_customer",
        "customer_address", "customer_demographics", "date_dim", "household_demographics",
        "income_band", "inventory", "item", "promotion", "reason", "ship_mode", "store",
        "store_returns", "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
        "web_sales", "web_site",
    ],
    "job": [
        "aka_name", "aka_title", "cast_info", "char_name", "comp_cast_type", "company_name",
        "company_type", "complete_cast", "info_type", "keyword", "kind_type", "link_type",
        "movie_companies", "movie_info", "movie_info_idx", "movie_keyword", "movie_link",
        "name", "person_info", "role_type", "title",
    ],
    "coffeeshop": ["fact_sales", "dim_locations", "dim_products"],
}

# Approximate table cardinalities used only for shard cost estimates and for
# picking the build side (smaller table). Tier (a) scales; tier (b) entries are
# derived in the harness by the per-dataset scale factor.
TABLE_ROWS_TIER_A = {
    # tpch SF40
    "lineitem": 240_000_000, "orders": 60_000_000, "partsupp": 32_000_000, "part": 8_000_000,
    "customer": 6_000_000, "supplier": 400_000, "nation": 25, "region": 5,
    # tpcds SF32 (facts linear, dims stepwise approximations)
    "store_sales": 92_000_000, "catalog_sales": 46_000_000, "web_sales": 23_000_000,
    "store_returns": 9_200_000, "catalog_returns": 4_600_000, "web_returns": 2_300_000,
    "inventory": 130_000_000, "date_dim": 73_049, "time_dim": 86_400, "item": 102_000,
    "tpcds_customer": 1_000_000, "customer_address": 500_000, "customer_demographics": 1_920_800,
    "household_demographics": 7_200, "store": 300, "warehouse": 10, "ship_mode": 20,
    "reason": 50, "promotion": 1_000, "call_center": 30, "web_site": 30, "web_page": 2_000,
    "catalog_page": 20_000, "income_band": 20,
    # job (fixed IMDB)
    "title": 2_528_312, "name": 4_167_491, "cast_info": 36_244_344, "movie_info": 14_835_720,
    "movie_info_idx": 1_380_035, "movie_keyword": 4_523_930, "movie_companies": 2_609_129,
    "keyword": 134_170, "company_name": 234_997, "char_name": 3_140_339, "aka_name": 901_343,
    "aka_title": 361_472, "person_info": 2_963_664, "complete_cast": 135_086,
    "movie_link": 29_997, "comp_cast_type": 4, "company_type": 4, "info_type": 113,
    "kind_type": 7, "link_type": 18, "role_type": 12,
    # coffeeshop 500m (dims counted from the public bucket)
    "fact_sales": 500_000_000, "dim_locations": 1_000, "dim_products": 26,
    # stackoverflow (April 2024 snapshot)
    "posts": 59_820_000, "users": 22_480_000, "votes": 238_980_000, "comments": 90_380_000,
    "badges": 51_290_000, "postlinks": 6_550_000,
}

# Filtered-dimension variants: joins whose build side is essentially always
# filtered in the source benchmark. One representative filter each; the
# harness runs both the unfiltered and the filtered variant.
CURATED_FILTERS = {
    ("tpcds", "date_dim"): "d_year = 2000",
    ("tpcds", "item"): "i_category = 'Electronics'",
    ("tpcds", "store"): "s_state = 'TN'",
    ("tpch", "part"): "p_size = 15 AND p_type LIKE '%BRASS'",
    ("tpch", "customer"): "c_mktsegment = 'BUILDING'",
    ("job", "info_type"): "info = 'top 250 rank'",
    ("job", "keyword"): "keyword = 'character-name-in-title'",
    ("job", "company_type"): "kind = 'production companies'",
    ("job", "company_name"): "country_code = '[de]'",
    ("job", "title"): "production_year BETWEEN 2005 AND 2010",
}

# Non-INNER variants observed in the source suites (edge -> extra kinds).
# Kind names use ClickHouse SQL: LEFT, LEFT SEMI, LEFT ANTI (probe side left).
CURATED_KINDS = {
    ("tpch", "customer.c_custkey", "orders.o_custkey"): ["LEFT", "LEFT ANTI"],   # Q13, Q22
    ("tpch", "lineitem.l_orderkey", "orders.o_orderkey"): ["LEFT SEMI"],         # Q4 EXISTS
    ("tpcds", "store_returns.sr_ticket_number", "store_sales.ss_ticket_number"): ["LEFT"],
    ("tpcds", "catalog_returns.cr_order_number", "catalog_sales.cs_order_number"): ["LEFT"],
    ("tpcds", "web_returns.wr_order_number", "web_sales.ws_order_number"): ["LEFT"],
    ("tpcds", "customer.c_customer_sk", "store_sales.ss_customer_sk"): ["LEFT SEMI"],  # q35-family EXISTS
}

STACKOVERFLOW_EDGES = [
    # (probe.col, build.col, weight) - probe = bigger table.
    ("votes.PostId", "posts.Id", 4),
    ("comments.PostId", "posts.Id", 3),
    ("posts.OwnerUserId", "users.Id", 4),
    ("badges.UserId", "users.Id", 2),
    ("comments.UserId", "users.Id", 2),
    ("postlinks.RelatedPostId", "posts.Id", 1),
]
STACKOVERFLOW_PROJECTIONS = {
    "posts": ["Id", "OwnerUserId", "Score", "ViewCount", "CommentCount"],
    "users": ["Id", "DisplayName", "Reputation"],
    "votes": ["PostId", "VoteTypeId"],
    "comments": ["PostId", "UserId", "Score"],
    "badges": ["UserId", "Name"],
    "postlinks": ["PostId", "RelatedPostId", "LinkTypeId"],
}


def load_schema(schema_dir: pathlib.Path, table: str) -> dict[str, str]:
    cols: dict[str, str] = {}
    for line in (schema_dir / f"{table}.columns").read_text().splitlines():
        line = line.strip().rstrip(",")
        if line:
            name, typ = line.split(None, 1)
            cols[name] = typ
    return cols


def query_visible_name(table: str) -> str:
    return "customer" if table == "tpcds_customer" else table


def extract_dataset(ds: str, query_file: pathlib.Path, schema_dir: pathlib.Path):
    """Return (edges, per_table_columns, band_edges) for one dataset."""
    schemas: dict[str, dict[str, str]] = {}
    col2tab: dict[str, set[str]] = {}
    for t in DATASET_TABLES[ds]:
        q = query_visible_name(t)
        schemas[q] = load_schema(schema_dir, t)
        for c in schemas[q]:
            col2tab.setdefault(c, set()).add(q)

    edges: collections.Counter = collections.Counter()
    table_cols: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    band_edges: collections.Counter = collections.Counter()

    queries = [q for q in query_file.read_text().split("\n") if q.strip()]
    for q in queries:
        # alias map: FROM comma-lists and explicit JOINs.
        amap: dict[str, str] = {}
        for m in re.finditer(r"\bFROM\s+(.*?)(?:\s+WHERE\b|\s+GROUP\b|\s+ORDER\b|$)", q, re.I | re.S):
            for item in m.group(1).split(","):
                # allow trailing JOIN clauses after the first table: "t a JOIN u b ON ..."
                mm = re.match(r"\s*([a-z_0-9]+)(?:\s+AS\s+([a-z_0-9]+)|\s+(?!JOIN\b|LEFT\b|INNER\b|CROSS\b)([a-z_0-9]+))?", item, re.I)
                if mm and mm.group(1) in schemas:
                    alias = mm.group(2) or mm.group(3) or mm.group(1)
                    amap[alias] = mm.group(1)
                    amap[mm.group(1)] = mm.group(1)
        for m in re.finditer(r"\bJOIN\s+([a-z_0-9]+)(?:\s+AS\s+([a-z_0-9]+)|\s+([a-z_0-9]+))?", q, re.I):
            if m.group(1) in schemas:
                alias = m.group(2) or m.group(3) or m.group(1)
                if alias.upper() in ("ON", "USING"):
                    alias = m.group(1)
                amap[alias] = m.group(1)
                amap[m.group(1)] = m.group(1)

        def resolve(tok: str):
            if "." in tok:
                a, c = tok.split(".", 1)
                t = amap.get(a)
                return (t, c) if t and c in schemas.get(t, {}) else (None, c)
            tabs = col2tab.get(tok)
            if tabs and len(tabs) == 1:
                return (next(iter(tabs)), tok)
            return (None, tok)

        # column references per table (for projection mining)
        for m in re.finditer(r"([a-z_][a-z_0-9]*(?:\.[a-z_0-9]+)?)", q, re.I):
            t, c = resolve(m.group(1))
            if t:
                table_cols[t][c] += 1

        # equality edges
        for m in re.finditer(
            r"([a-z_][a-z_0-9]*(?:\.[a-z_0-9]+)?)\s*=\s*([a-z_][a-z_0-9]*(?:\.[a-z_0-9]+)?)(?![a-z_0-9('])",
            q, re.I,
        ):
            l, r = m.group(1), m.group(2)
            if l.upper() in ("DATE", "INTERVAL") or r.upper() in ("DATE", "INTERVAL"):
                continue
            (lt, lc), (rt, rc) = resolve(l), resolve(r)
            if lt and rt and lt != rt:
                a, b = sorted([(lt, lc), (rt, rc)])
                edges[(f"{a[0]}.{a[1]}", f"{b[0]}.{b[1]}")] += 1

        # SCD-2 band conditions across tables
        for m in re.finditer(
            r"([a-z_][a-z_0-9]*\.[a-z_0-9]+)\s+BETWEEN\s+([a-z_][a-z_0-9]*\.[a-z_0-9]+)\s+AND\s+([a-z_][a-z_0-9]*\.[a-z_0-9]+)",
            q, re.I,
        ):
            (lt, lc), (rt, rc1) = resolve(m.group(1)), resolve(m.group(2))
            (_, rc2) = resolve(m.group(3))
            if lt and rt and lt != rt:
                band_edges[(f"{lt}.{lc}", f"{rt}.{rc1}", f"{rt}.{rc2}")] += 1

    return edges, table_cols, band_edges, schemas


def pick_projection(table: str, key_cols: list[str], col_counter, schema: dict[str, str], limit: int = 3):
    """Join keys + up to `limit` most-referenced payload columns of the table."""
    payload = []
    for col, _n in col_counter.most_common() if col_counter is not None else []:
        if col in key_cols or col not in schema:
            continue
        payload.append(col)
        if len(payload) >= limit:
            break
    if not payload:  # fall back to the first non-key schema column
        payload = [c for c in schema if c not in key_cols][:1]
    return key_cols + payload


# Natural primary keys used as MergeTree ORDER BY. Build-side dimension tables
# sort by their surrogate key (realistic: dims are PK-sorted); facts sort by
# their natural insertion/PK order, NOT by the probe-side join key.
ORDER_BY = {
    # tpch
    "lineitem": "(l_orderkey, l_linenumber)", "orders": "o_orderkey",
    "partsupp": "(ps_partkey, ps_suppkey)", "part": "p_partkey", "supplier": "s_suppkey",
    "customer": "c_custkey", "nation": "n_nationkey", "region": "r_regionkey",
    # tpcds
    "store_sales": "(ss_item_sk, ss_ticket_number)", "catalog_sales": "(cs_item_sk, cs_order_number)",
    "web_sales": "(ws_item_sk, ws_order_number)", "store_returns": "(sr_item_sk, sr_ticket_number)",
    "catalog_returns": "(cr_item_sk, cr_order_number)", "web_returns": "(wr_item_sk, wr_order_number)",
    "inventory": "(inv_date_sk, inv_item_sk, inv_warehouse_sk)", "date_dim": "d_date_sk",
    "time_dim": "t_time_sk", "item": "i_item_sk", "tpcds_customer": "c_customer_sk",
    "customer_address": "ca_address_sk", "customer_demographics": "cd_demo_sk",
    "household_demographics": "hd_demo_sk", "store": "s_store_sk", "warehouse": "w_warehouse_sk",
    "ship_mode": "sm_ship_mode_sk", "reason": "r_reason_sk", "promotion": "p_promo_sk",
    "call_center": "cc_call_center_sk", "web_site": "web_site_sk", "web_page": "wp_web_page_sk",
    "catalog_page": "cp_catalog_page_sk", "income_band": "ib_income_band_sk",
    # coffeeshop (fact sorts by its natural order_date, per the source benchmark)
    "fact_sales": "order_date", "dim_locations": "location_id", "dim_products": "(product_id, from_date)",
    # job: every table keeps its natural id order (FK tables are NOT sorted by
    # their join keys, matching the real dataset's insertion order)
    **{t: "id" for t in DATASET_TABLES["job"]},
}

# StackOverflow schemas are not in ClickBench; faithful-but-minimal types from
# the ClickHouse docs sample dataset (Enum columns widened to their base types
# so tier-b doubling needs no enum bookkeeping).
STACKOVERFLOW_SCHEMAS = {
    "posts": [
        ("Id", "Int32"), ("PostTypeId", "UInt8"), ("AcceptedAnswerId", "UInt32"),
        ("CreationDate", "DateTime64(3)"), ("Score", "Int32"), ("ViewCount", "UInt32"),
        ("Body", "String"), ("OwnerUserId", "Int32"), ("OwnerDisplayName", "String"),
        ("LastEditorUserId", "Int32"), ("LastEditDate", "DateTime64(3)"),
        ("LastActivityDate", "DateTime64(3)"), ("Title", "String"), ("Tags", "String"),
        ("AnswerCount", "UInt16"), ("CommentCount", "UInt8"), ("FavoriteCount", "UInt8"),
        ("ContentLicense", "String"), ("ParentId", "String"),
        ("CommunityOwnedDate", "DateTime64(3)"), ("ClosedDate", "DateTime64(3)"),
    ],
    "users": [
        ("Id", "Int32"), ("Reputation", "UInt32"), ("CreationDate", "DateTime64(3)"),
        ("DisplayName", "String"), ("LastAccessDate", "DateTime64(3)"), ("AboutMe", "String"),
        ("Views", "UInt32"), ("UpVotes", "UInt32"), ("DownVotes", "UInt32"),
        ("WebsiteUrl", "String"), ("Location", "String"), ("AccountId", "Int32"),
    ],
    "votes": [
        ("Id", "UInt32"), ("PostId", "Int32"), ("VoteTypeId", "UInt8"),
        ("CreationDate", "DateTime64(3)"), ("UserId", "Int32"), ("BountyAmount", "UInt8"),
    ],
    "comments": [
        ("Id", "UInt32"), ("PostId", "UInt32"), ("Score", "UInt16"), ("Text", "String"),
        ("CreationDate", "DateTime64(3)"), ("UserId", "Int32"), ("UserDisplayName", "String"),
    ],
    "badges": [
        ("Id", "UInt32"), ("UserId", "Int32"), ("Name", "String"),
        ("Date", "DateTime64(3)"), ("Class", "UInt8"), ("TagBased", "Bool"),
    ],
    "postlinks": [
        ("Id", "UInt64"), ("CreationDate", "DateTime64(3)"), ("PostId", "Int32"),
        ("RelatedPostId", "Int32"), ("LinkTypeId", "UInt8"),
    ],
}
STACKOVERFLOW_ORDER_BY = {
    "posts": "Id", "users": "Id", "votes": "(PostId, Id)",
    "comments": "(PostId, Id)", "badges": "(UserId, Id)", "postlinks": "(PostId, Id)",
}


def emit_schemas(schema_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    datasets: dict[str, dict] = {}
    for ds in ("tpch", "tpcds", "coffeeshop", "job"):
        tables = {}
        for t in DATASET_TABLES[ds]:
            qname = query_visible_name(t)
            cols = load_schema(schema_dir, t)
            tables[qname] = {
                "columns": list(cols.items()),
                "order_by": ORDER_BY[t],
                "rows_tier_a": TABLE_ROWS_TIER_A.get(t, 0),
            }
        datasets[ds] = tables
    datasets["stackoverflow"] = {
        t: {
            "columns": [list(c) for c in cols],
            "order_by": STACKOVERFLOW_ORDER_BY[t],
            "rows_tier_a": TABLE_ROWS_TIER_A.get(t, 0),
        }
        for t, cols in STACKOVERFLOW_SCHEMAS.items()
    }
    out_path.write_text(json.dumps({"version": 1, "datasets": datasets}, indent=1) + "\n")
    n = sum(len(t) for t in datasets.values())
    print(f"wrote {n} table schemas to {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clickbench", default="tmp/clickbench")
    ap.add_argument("--out", default="bep/tools/join_bench_mt_queries.json")
    ap.add_argument("--schemas-out", default="bep/tools/join_bench_mt_schemas.json")
    args = ap.parse_args()

    cb = pathlib.Path(args.clickbench)
    qdir, sdir = cb / "versions/queries", cb / "versions/create/schema"
    if not qdir.is_dir():
        print(f"ClickBench checkout not found at {cb}", file=sys.stderr)
        return 1

    specs = []

    def add_spec(ds, probe, build, *, weight, kinds, schemas, table_cols, band_by_table=None):
        (pt, pc), (bt, bc) = probe, build
        rows_p = TABLE_ROWS_TIER_A.get(pt, 0)
        rows_b = TABLE_ROWS_TIER_A.get(bt, 0)
        # probe = bigger side; swap if needed (keys swap with their tables).
        if rows_b > rows_p:
            (pt, pc, bt, bc) = (bt, bc, pt, pc)
            rows_p, rows_b = rows_b, rows_p
        band = (band_by_table or {}).get(bt)
        key_p, key_b = pc.split(","), bc.split(",")
        proj_p = pick_projection(pt, key_p, table_cols.get(pt), schemas[pt])
        proj_b = pick_projection(bt, key_b, table_cols.get(bt), schemas[bt])
        base_id = f"{ds}__{pt}_{key_p[0]}__{bt}_{key_b[0]}"
        for kind in kinds:
            kind_tag = kind.replace(" ", "_").lower()
            qid = base_id if kind == "INNER" else f"{base_id}__{kind_tag}"
            spec = {
                "id": qid,
                "dataset": ds,
                "kind": kind,
                "probe_table": pt, "probe_keys": key_p, "probe_projection": proj_p,
                "build_table": bt, "build_keys": key_b, "build_projection": proj_b,
                "probe_rows_tier_a": rows_p, "build_rows_tier_a": rows_b,
                "source_weight": weight,
                "filter": None,
                "band": band,
            }
            specs.append(spec)
            filt = CURATED_FILTERS.get((ds, bt))
            if filt and kind == "INNER":
                specs.append({**spec, "id": f"{base_id}__filtered", "filter": filt})

    for ds in ("tpch", "tpcds", "job", "coffeeshop"):
        edges, table_cols, band_edges, schemas = extract_dataset(ds, qdir / f"{ds}.sql", sdir)
        bands = {}
        for (probe_col, lo, hi), _n in band_edges.items():
            bt = lo.split(".")[0]
            bands[bt] = {"probe_col": probe_col.split(".")[1], "build_lo": lo.split(".")[1], "build_hi": hi.split(".")[1]}
        for (a, b), n in sorted(edges.items(), key=lambda kv: -kv[1]):
            at, ac = a.split(".")
            btab, bc = b.split(".")
            # merge composite keys: same table pair appearing with multiple column
            # pairs in the same query family (e.g. (item_sk, ticket_number)).
            kinds = ["INNER"] + CURATED_KINDS.get((ds, a, b), [])
            add_spec(ds, (at, ac), (btab, bc), weight=n, kinds=kinds,
                     schemas=schemas, table_cols=table_cols,
                     band_by_table=bands if ds == "coffeeshop" else None)

    # composite-key merges: replace known 2-column pairs with a single spec
    composite = [
        ("tpcds", "store_sales", ["ss_item_sk", "ss_ticket_number"], "store_returns", ["sr_item_sk", "sr_ticket_number"]),
        ("tpcds", "catalog_sales", ["cs_item_sk", "cs_order_number"], "catalog_returns", ["cr_item_sk", "cr_order_number"]),
        ("tpcds", "web_sales", ["ws_item_sk", "ws_order_number"], "web_returns", ["wr_item_sk", "wr_order_number"]),
        ("tpch", "lineitem", ["l_partkey", "l_suppkey"], "partsupp", ["ps_partkey", "ps_suppkey"]),
    ]
    drop_ids = set()
    for ds, pt, pks, bt, bks in composite:
        for spec in specs:
            if spec["dataset"] == ds and {spec["probe_table"], spec["build_table"]} == {pt, bt} and len(spec["probe_keys"]) == 1:
                drop_ids.add(spec["id"])
        specs.append({
            "id": f"{ds}__{pt}_{pks[0]}__{bt}_{bks[0]}__composite",
            "dataset": ds, "kind": "INNER",
            "probe_table": pt, "probe_keys": pks, "probe_projection": pks,
            "build_table": bt, "build_keys": bks, "build_projection": bks,
            "probe_rows_tier_a": TABLE_ROWS_TIER_A[pt], "build_rows_tier_a": TABLE_ROWS_TIER_A[bt],
            "source_weight": 8, "filter": None, "band": None,
        })
    specs = [s for s in specs if s["id"] not in drop_ids or "composite" in s["id"]]

    # StackOverflow (hand-curated; queries come from ClickHouse docs pages).
    for probe, build, w in STACKOVERFLOW_EDGES:
        pt, pc = probe.split(".")
        bt, bc = build.split(".")
        specs.append({
            "id": f"stackoverflow__{pt}_{pc}__{bt}_{bc}",
            "dataset": "stackoverflow", "kind": "INNER",
            "probe_table": pt, "probe_keys": [pc],
            "probe_projection": STACKOVERFLOW_PROJECTIONS[pt],
            "build_table": bt, "build_keys": [bc],
            "build_projection": STACKOVERFLOW_PROJECTIONS[bt],
            "probe_rows_tier_a": TABLE_ROWS_TIER_A[pt], "build_rows_tier_a": TABLE_ROWS_TIER_A[bt],
            "source_weight": w, "filter": None, "band": None,
        })

    out = pathlib.Path(args.out)
    out.write_text(json.dumps({"version": 1, "specs": specs}, indent=1) + "\n")
    by_ds = collections.Counter(s["dataset"] for s in specs)
    print(f"wrote {len(specs)} query specs to {out}: {dict(by_ds)}")
    emit_schemas(sdir, pathlib.Path(args.schemas_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
