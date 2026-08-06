#!/usr/bin/env python3
"""A shape sweep of the two `unified_hash` probe loops, fused and split.

The 144-cell matrix pins the build side to 30M rows at 16 and 64 threads (`THREAD_CARDS`
pairs the many-threaded points with `large` only), so it never covers the regime where a
fixed per-row cost matters most: a small, cache-resident build side driven by many threads.
That is exactly where the split probe's scratch traffic is least likely to pay for itself,
since there is no memory latency to overlap. This searches that space.

Axes:
  build rows      2^17 (131072) .. 2^26 (67.1M), powers of two - 10 points
  probe multiple  1x, 2x, 4x of the build rows
  shape           narrow / rpay / lpay / uniq  (see SHAPES)
  threads         16, 64            (where `parallel_hash` is the comparator)

One build table and one probe table, both ordered by a row index, so a size is a range
scan rather than a separate table: `WHERE rid < n` prunes marks and reads exactly n rows.
That keeps the read cost proportional to the shape being measured and the data identical
across sizes, which a per-size table would not.

Match rate is held at 50% for every shape and size: the probe key is
`rid % (2 * distinct_build_keys)`, and the build holds `[0, distinct_build_keys)`.

Runs are interleaved with the algorithm order rotated per repetition, exactly as
`sweep.py` does, so machine drift cannot masquerade as an effect.

    python3 shapes.py gen                      # once
    python3 shapes.py run --tag s1 --reps 5
    python3 shapes.py report --tag s1
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "perf"))

import harness as H  # noqa: E402

RESULTS = os.path.join(HERE, "results")

BUILD_TABLE = "sb"
PROBE_TABLE = "sp"
BUILD_ROWS_MAX = 1 << 26          # 67.1M; the next power of two is past the 100M cap
PROBE_ROWS_MAX = 1 << 28          # 4x the largest build side

BUILD_SIZES = [1 << e for e in range(17, 27)]
PROBE_MULTS = [1, 2, 4]
THREADS = [16, 64]

# After Stage 1 the arms are the shipping path in harness.ARMS (Stage 6 adds binary A/B).
ARMS = list(H.AB_ARMS)

# name -> (build key column, rows per key, what the query reads besides the key)
#   narrow  the RowRefList path with nothing gathered from either side
#   rpay    right-side payload gathered: 4 UInt64 and a String per output row
#   lpay    left-side payload carried through instead
#   uniq    unique build keys, so all three implementations promote ALL to RightAny
#           and take the MapsOne path - a different map, not a different data size
SHAPES = {
    "narrow": ("k",  2, "none"),
    "rpay":   ("k",  2, "right"),
    "lpay":   ("k",  2, "left"),
    "uniq":   ("k2", 1, "none"),
}

_seq = itertools.count()
_TOKEN = f"{os.getpid():x}{int(time.time()) & 0xffff:04x}"
SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def qid(*parts):
    return SAFE.sub("_", "-".join(str(p) for p in parts)) + f".{_TOKEN}.{next(_seq)}"


def cell_id(shape, build_rows, mult, threads):
    return f"{shape}|b{build_rows}|x{mult}|t{threads}"


def sql(shape, build_rows, mult):
    key_col, rows_per_key, payload = SHAPES[shape]
    distinct_keys = build_rows // rows_per_key
    probe_rows = build_rows * mult
    key_space = 2 * distinct_keys          # 50% of probe keys have no build row

    right_cols = f"{key_col} AS rk" + (", v1, v2, v3, v4, s" if payload == "right" else "")
    left_cols = f"rid % {key_space} AS lk" + (", w1, w2, w3, w4, t" if payload == "left" else "")

    if payload == "right":
        agg = "count() AS cnt, sum(l.lk) AS s1, sum(r.v1 + r.v4) AS s2, sum(length(r.s)) AS s3"
    elif payload == "left":
        agg = "count() AS cnt, sum(l.lk) AS s1, sum(l.w1 + l.w4) AS s2, sum(length(l.t)) AS s3"
    else:
        agg = "count() AS cnt, sum(l.lk) AS s1"

    return (
        f"SELECT {agg} FROM "
        f"(SELECT {left_cols} FROM {PROBE_TABLE} WHERE rid < {probe_rows}) AS l "
        f"INNER JOIN "
        f"(SELECT {right_cols} FROM {BUILD_TABLE} WHERE rid < {build_rows}) AS r "
        f"ON l.lk = r.rk"
    )


def settings(arm, threads):
    s = dict(H.PINNED_SETTINGS)
    s["join_algorithm"] = H.ARMS[arm]["algo"]
    s.update(H.ARMS[arm]["settings"])
    s["max_threads"] = threads
    # A guard, not a limit anyone should hit: a runaway cell should fail loudly rather
    # than take the host down. 200 GiB of the 317 GiB free at the time of writing.
    s["max_memory_usage"] = 200 * (1 << 30)
    return s


def gen():
    ddl = [
        f"DROP TABLE IF EXISTS {BUILD_TABLE}",
        f"DROP TABLE IF EXISTS {PROBE_TABLE}",
        f"""CREATE TABLE {BUILD_TABLE}
            (rid UInt64, k UInt64, k2 UInt64, v1 UInt64, v2 UInt64, v3 UInt64, v4 UInt64, s String)
            ENGINE = MergeTree ORDER BY rid""",
        f"""CREATE TABLE {PROBE_TABLE}
            (rid UInt64, w1 UInt64, w2 UInt64, w3 UInt64, w4 UInt64, t String)
            ENGINE = MergeTree ORDER BY rid""",
    ]
    for q in ddl:
        H.run_query(q)
    print(f"building {BUILD_TABLE}: {BUILD_ROWS_MAX} rows", flush=True)
    H.run_query(
        f"INSERT INTO {BUILD_TABLE} SELECT number AS rid, intDiv(number, 2) AS k, number AS k2, "
        f"number AS v1, number + 1 AS v2, number + 2 AS v3, number + 3 AS v4, "
        f"concat('r', toString(number)) AS s FROM numbers_mt({BUILD_ROWS_MAX})",
        {"max_insert_threads": 32, "max_memory_usage": 0})
    print(f"building {PROBE_TABLE}: {PROBE_ROWS_MAX} rows", flush=True)
    H.run_query(
        f"INSERT INTO {PROBE_TABLE} SELECT number AS rid, number AS w1, number + 1 AS w2, "
        f"number + 2 AS w3, number + 3 AS w4, concat('l', toString(number)) AS t "
        f"FROM numbers_mt({PROBE_ROWS_MAX})",
        {"max_insert_threads": 32, "max_memory_usage": 0})
    for t in (BUILD_TABLE, PROBE_TABLE):
        H.run_query(f"OPTIMIZE TABLE {t} FINAL", {"max_memory_usage": 0, "receive_timeout": 3600})
        n = H.run_query(f"SELECT count() FROM {t}").strip()
        b = H.run_query(f"SELECT formatReadableSize(sum(bytes_on_disk)) FROM system.parts "
                        f"WHERE table = '{t}' AND active").strip()
        print(f"  {t}: {n} rows, {b} on disk", flush=True)
    return 0


def read_run_retrying(query_id, attempts=6):
    """`SYSTEM FLUSH LOGS` races the query_log write for the query that just finished.

    The inherited `sweep.py` does not retry and loses 0.6-0.7% of its runs this way (11 of
    1848 in `bold1`, 13 of 1848 in `bnew2`) - always the last query of a cell. That is
    small and symmetric between runs, so it does not bias a median over 7 repetitions, but
    there is no reason to accept it: flush again and re-ask.
    """
    last = None
    for i in range(attempts):
        try:
            return H.read_run(query_id)
        except H.QueryError as exc:
            last = exc
            time.sleep(0.25 * (i + 1))
            try:
                H.flush_logs()
            except H.QueryError:
                pass
    return {"error": str(last)[:300]}


def run(tag, reps, out, filt):
    os.makedirs(RESULTS, exist_ok=True)
    cells = [(sh, b, m, t) for sh in SHAPES for b in BUILD_SIZES
             for m in PROBE_MULTS for t in THREADS
             if filt in cell_id(sh, b, m, t)]
    print(f"tag={tag} cells={len(cells)} reps={reps} arms={','.join(ARMS)} "
          f"queries={len(cells) * len(ARMS) * (reps + 1)}", flush=True)
    t_start = time.time()
    with open(out, "a") as fh:
        for i, (sh, b, m, th) in enumerate(cells, 1):
            cid = cell_id(sh, b, m, th)
            q = sql(sh, b, m)
            pend, err = [], None
            try:
                for a in ARMS:                                   # warm the page cache
                    H.run_query(q, settings(a, th), query_id=qid(tag, cid, a, "warm"))
                for rep in range(reps):
                    for a in ARMS[rep % 2:] + ARMS[:rep % 2]:     # rotate
                        i_ = qid(tag, cid, a, rep)
                        out_txt = H.run_query(q, settings(a, th), query_id=i_)
                        pend.append((i_, a, rep, out_txt.strip()))
            except H.QueryError as exc:
                err = str(exc)[:400]
            H.flush_logs()
            for i_, a, rep, res in pend:
                rec = {"tag": tag, "cell_id": cid, "shape": sh, "build_rows": b,
                       "mult": m, "threads": th, "algo": a, "rep": rep,
                       "query_id": i_, "output": res}
                rec.update(read_run_retrying(i_))
                fh.write(json.dumps(rec) + "\n")
            if err:
                fh.write(json.dumps({"tag": tag, "cell_id": cid, "error": err}) + "\n")
            fh.flush()
            print(f"[{i:3d}/{len(cells)}] {cid:26s} elapsed={(time.time()-t_start)/60:5.1f}m"
                  f"{'  ERROR ' + err if err else ''}", flush=True)
    print(f"SHAPES_DONE tag={tag} total={(time.time()-t_start)/60:.1f}m")
    return 0


def load(path, tag):
    per = {}
    outs = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("tag") != tag or "wall_ms" not in r:
            continue
        k = (r["cell_id"], r["algo"])
        per.setdefault(k, {"wall_ms": [], "cpu_us": [], "memory_usage": []})
        for m in per[k]:
            if m in r:
                per[k][m].append(float(r[m]))
        outs.setdefault(r["cell_id"], {}).setdefault(r["algo"], set()).add(r.get("output"))
    return {k: {m: statistics.median(v) for m, v in d.items() if v} for k, d in per.items()}, outs


def report(tag, path):
    med, outs = load(path, tag)
    cells = sorted({c for c, _ in med})
    rows = []
    for c in cells:
        a, b = med.get((c, H.BASELINE_ARM)), med.get((c, H.TEST_ARM))
        if not a or not b:
            continue
        sh, br, mu, th = c.split("|")
        rows.append({"cell": c, "shape": sh, "build": int(br[1:]), "mult": int(mu[1:]),
                     "threads": int(th[1:]),
                     "w": 100 * (b["wall_ms"] - a["wall_ms"]) / a["wall_ms"],
                     "c": 100 * (b["cpu_us"] - a["cpu_us"]) / a["cpu_us"],
                     "m": 100 * (b["memory_usage"] - a["memory_usage"]) / a["memory_usage"]})

    # answers must agree, or a timing comparison is meaningless
    bad = [c for c, d in outs.items() if len(d) == 2 and len(set.union(*d.values())) != 1]
    print(f"tag={tag}: {len(rows)} comparable cells; "
          f"answer mismatches between the two arms: {len(bad)}")
    for c in bad[:5]:
        print(f"   MISMATCH {c}: {outs[c]}")
    print()
    print(f"{'group':34s} {'n':>3s} {'wall med':>9s} {'wall worst':>11s} "
          f"{'cpu med':>8s} {'mem med':>8s}  slower>2%")

    def show(label, g):
        if not g:
            return
        w = sorted(r["w"] for r in g)
        print(f"{label:34s} {len(g):3d} {statistics.median(w):+9.1f} {w[-1]:+11.1f} "
              f"{statistics.median(r['c'] for r in g):+8.1f} "
              f"{statistics.median(r['m'] for r in g):+8.1f}  "
              f"{sum(1 for x in w if x > 2):3d}/{len(g)}")

    show("ALL", rows)
    print()
    for th in THREADS:
        show(f"threads={th}", [r for r in rows if r["threads"] == th])
    print()
    for sh in SHAPES:
        show(f"shape={sh}", [r for r in rows if r["shape"] == sh])
    print()
    for mu in PROBE_MULTS:
        show(f"probe={mu}x build", [r for r in rows if r["mult"] == mu])
    print()
    for b in BUILD_SIZES:
        show(f"build={b:>9d} rows", [r for r in rows if r["build"] == b])
    losers = sorted([r for r in rows if r["w"] > 2], key=lambda r: -r["w"])
    print(f"\ncells where {H.TEST_ARM} is more than 2% SLOWER on wall than "
          f"{H.BASELINE_ARM}: {len(losers)}")
    for r in losers[:25]:
        print(f"   {r['w']:+7.1f}% wall  {r['c']:+7.1f}% cpu  {r['cell']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "run", "report"])
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--filter", default="")
    ap.add_argument("--out", default=os.path.join(RESULTS, "shapes.jsonl"))
    a = ap.parse_args()
    if a.cmd == "gen":
        return gen()
    if a.cmd == "run":
        return run(a.tag, a.reps, a.out, a.filter)
    return report(a.tag, a.out)


if __name__ == "__main__":
    sys.exit(main())
