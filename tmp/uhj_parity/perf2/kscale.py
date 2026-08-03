#!/usr/bin/env python3
"""PREREG P1.2 -- the per-partition key-getter packing, as a K-scaling natural experiment.

Sweeps `max_threads` at fixed data and fits build cost per build row against the
partition count. `comp` (two UInt64 -> a packed UInt128 key) goes through
`HashMethodKeysFixed`, whose constructor packs the whole block; `u64` goes through
`HashMethodOneNumber`, which packs nothing. Both arms see the same change in thread
count and therefore the same change in contention, so the difference between their
per-partition coefficients isolates the packing.

Runs on the delivered (uninstrumented) binary: thread count is a setting, so nothing is
rebuilt and nothing can leak into the tree.

    python3 kscale.py run  --tag k1 --reps 7
    python3 kscale.py fit  --tag k1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perf"))

import harness as H  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TOKEN = f"{int(time.time())}_{os.getpid()}"

THREADS = [4, 8, 16, 32, 64]
KEYS = ["u64", "comp"]
BUILD_ROWS = {"small": 20_000, "medium": 1_000_000, "large": 30_000_000}


def partitions(algo, t):
    """Partition count each implementation uses at `t` threads.

    unified_hash: bucketCountForThreads(n) = 1 if n<=1 else bit_ceil(n)*2
    parallel_hash: slots = min(threads, 256)
    hash: one partition (flat map, one insert per block)
    """
    if algo == "unified_hash":
        return 1 if t <= 1 else (1 << (t - 1).bit_length()) * 2
    if algo == "parallel_hash":
        return min(t, 256)
    return 1


def run(tag, reps):
    out = os.path.join(RESULTS, f"kscale_{tag}.jsonl")
    with open(out, "w") as fh:
        for key in KEYS:
            for t in THREADS:
                cell = H.Cell("INNER", key, "hi", t, "large")
                # Interleave algorithms within a repetition rather than running all of
                # one then all of the other, so drift cannot masquerade as an effect.
                for rep in range(reps):
                    for algo in H.ALGOS:
                        if cell.skip_reason(algo):
                            continue
                        qid = f"ks_{tag}_{TOKEN}_{key}_t{t}_{algo}_r{rep}"
                        H.run_query(H.join_sql(cell, "timed"),
                                    H.settings_for(cell, algo), query_id=qid)
                        # The log entry is queued asynchronously, so a single
                        # FLUSH LOGS immediately after the query can race it.
                        for attempt in range(8):
                            H.flush_logs()
                            try:
                                m = H.read_run(qid)
                                break
                            except H.QueryError:
                                if attempt == 7:
                                    raise
                                time.sleep(0.3)
                        m.update({"key": key, "threads": t, "algo": algo, "rep": rep,
                                  "K": partitions(algo, t),
                                  "build_rows": BUILD_ROWS["large"]})
                        fh.write(json.dumps(m) + "\n")
                        fh.flush()
                print(f"  {key:5s} t={t:3d} done ({reps} reps x 3 algos)")
    print(f"-> {out}")


def fit(tag):
    recs = [json.loads(l) for l in open(os.path.join(RESULTS, f"kscale_{tag}.jsonl"))]
    print(f"{'key':5s} {'algo':14s} {'t':>3s} {'K':>4s} {'build_us(med)':>14s} "
          f"{'stdev%':>7s} {'ns/buildrow':>12s} {'cpu_us(med)':>12s}")
    series = {}
    for key in KEYS:
        for algo in H.ALGOS:
            for t in THREADS:
                v = [r for r in recs if r["key"] == key and r["algo"] == algo and r["threads"] == t]
                if not v:
                    continue
                b = [x["build_us"] for x in v]
                c = [x["cpu_us"] for x in v]
                med = H.median(b)
                sd = H.stdev(b)
                nspr = med * 1000.0 / v[0]["build_rows"]
                print(f"{key:5s} {algo:14s} {t:3d} {v[0]['K']:4d} {med:14.0f} "
                      f"{(100*sd/med if med else 0):7.1f} {nspr:12.2f} {H.median(c):12.0f}")
                series.setdefault((key, algo), []).append((v[0]["K"], nspr, sd * 1000.0 / v[0]["build_rows"]))

    print()
    print("least-squares fit of  ns_per_build_row = a + b*K   (b = per-partition cost)")
    print(f"{'key':5s} {'algo':14s} {'a (ns/row)':>12s} {'b (ns/row/part)':>17s} {'R^2':>7s} {'n':>3s}")
    fits = {}
    for (key, algo), pts in sorted(series.items()):
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        a = my - b * mx
        ss_tot = sum((y - my) ** 2 for y in ys)
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        fits[(key, algo)] = (a, b, r2)
        print(f"{key:5s} {algo:14s} {a:12.2f} {b:17.4f} {r2:7.3f} {n:3d}")

    print()
    print("PREREG P1.2 predictions, scored:")
    for algo in ("unified_hash", "parallel_hash", "hash"):
        fu = fits.get(("u64", algo))
        fc = fits.get(("comp", algo))
        if not fu or not fc:
            continue
        ratio = fc[1] / fu[1] if fu[1] else float("inf")
        print(f"  {algo:14s} b(comp)={fc[1]:.4f}  b(u64)={fu[1]:.4f}  ratio={ratio:.2f}")
    print()
    print("  (1) comp+unified has b > 0 outside noise            : "
          f"b={fits.get(('comp','unified_hash'),(0,0,0))[1]:.4f} ns/row/partition")
    ru = fits.get(("comp", "unified_hash"), (0, 0, 0))[1]
    uu = fits.get(("u64", "unified_hash"), (0, 1, 0))[1]
    print(f"  (2) comp coefficient >= 4x the u64 one              : "
          f"ratio={ru/uu if uu else float('inf'):.2f}  "
          f"{'MET' if uu and ru/uu >= 4 else 'NOT MET'}")
    for key in KEYS:
        f = fits.get((key, "hash"))
        if f:
            print(f"  (3) hash shows no K term ({key:4s})                  : b={f[1]:.4f} "
                  f"(K is constant 1 for hash, so this fit is degenerate by construction)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "fit"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.tag, a.reps)
        sys.exit(0)
    sys.exit(fit(a.tag))
