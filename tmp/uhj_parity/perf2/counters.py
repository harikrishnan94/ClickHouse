#!/usr/bin/env python3
"""PREREG P3.2 / gate G3.2: hardware counters on the one-thread probe deficit.

Uses ClickHouse's own `metrics_perf_events_enabled=1`, which exposes per-query
`PerfCPUCycles` / `PerfInstructions` / `PerfCacheMisses` / `PerfDataTLBMisses` in
`system.query_log`.

Why not external `perf stat -p <server pid>`: it was tried first and is WRONG here.
`-p` follows only the threads that exist when it attaches, and ClickHouse spawns new
threads per query, so the join's work is never counted. It reported 178,633 cycles for a
query that actually executes ~385 million -- a 2000x undercount that looked plausible
enough to use. The built-in counters are per-query-thread by construction.

The probe phase is isolated by subtraction: `probe = full - buildonly`, where buildonly
reduces the probe side with `WHERE k < 0` (not `LIMIT 0`, which would let the planner
prune the join entirely).

    python3 counters.py run  --tag c2 --reps 7
    python3 counters.py gate --tag c2
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perf"))
import harness as H

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
# Probe-dominated by construction rather than by subtraction. The `full - buildonly`
# subtraction was tried first and had to be abandoned: ClickHouse's own perf-counter
# deltas UNDERFLOW on short queries (values of ~2^64 appear), and the buildonly runs are
# short, so 0 of 7 unified buildonly samples were usable. Instead the build side is made
# small (20k rows) against a 2M-row probe, so the probe dominates the query outright.
# Deviation from PREREG P3.2's stated method, documented rather than silent.
CELL = H.Cell("INNER", "u64", "hi", 1, "small")
PROBE_ROWS = 2_000_000
BUILD_ROWS = 20_000
SANE = 1e15   # counter values above this are UInt64 underflow, not measurements
TOKEN = f"{int(time.time())}_{os.getpid()}"
EVENTS = ["PerfCPUCycles", "PerfInstructions", "PerfCacheMisses", "PerfCacheReferences",
          "PerfDataTLBMisses", "PerfStalledCyclesBackend", "PerfBranchMisses",
          "PerfMinEnabledTime", "PerfMinEnabledRunningTime"]


def read(qid):
    sel = ", ".join(f"ProfileEvents['{e}']" for e in EVENTS)
    for attempt in range(8):
        H.flush_logs()
        row = H.run_query(f"SELECT query_duration_ms, {sel} FROM system.query_log "
                          f"WHERE query_id='{qid}' AND type='QueryFinish' LIMIT 1 FORMAT TSV").strip()
        if row:
            f = row.split("\t")
            return dict(zip(["wall_ms"] + EVENTS, [float(x) for x in f]))
        time.sleep(0.3)
    raise RuntimeError(f"no query_log row for {qid}")


def run(tag, reps):
    path = os.path.join(RESULTS, f"counters_{tag}.jsonl")
    with open(path, "w") as fh:
        for rep in range(reps):
            for algo in ("hash", "unified_hash"):        # interleaved within a rep
                for mode in ("timed",):
                    qid = f"cnt_{tag}_{TOKEN}_{algo}_{mode}_{rep}"
                    H.run_query(H.join_sql(CELL, mode),
                                H.settings_for(CELL, algo, {"metrics_perf_events_enabled": 1}),
                                query_id=qid)
                    d = read(qid); d.update(algo=algo, mode=mode, rep=rep)
                    fh.write(json.dumps(d) + "\n"); fh.flush()
            print(f"  rep {rep} done")
    print(f"-> {path}")


def gate(tag):
    recs = [json.loads(l) for l in open(os.path.join(RESULTS, f"counters_{tag}.jsonl"))]
    probe, mult, nclean = {}, {}, {}
    for algo in ("hash", "unified_hash"):
        probe[algo] = {}
        # a rep is usable only if EVERY counter in it is sane; dropping per-counter
        # would mix reps and break the IPC ratio
        reps = [r for r in recs if r["algo"] == algo and r["mode"] == "timed"
                and all(0 < r[k] < SANE for k in EVENTS[:7])]
        nclean[algo] = (len(reps), len([r for r in recs if r["algo"] == algo]))
        for k in EVENTS:
            probe[algo][k] = H.median([r[k] for r in reps])
        f = [r["PerfMinEnabledRunningTime"] / r["PerfMinEnabledTime"]
             for r in reps if r["PerfMinEnabledTime"]]
        mult[algo] = H.median(f) if f else float("nan")

    print("G3.2 hardware counters, whole query on a PROBE-DOMINATED cell")
    print(f"  cell {CELL.cell_id}: {BUILD_ROWS:,} build rows vs {PROBE_ROWS:,} probe rows")
    print(f"  usable reps (all counters sane): hash {nclean['hash'][0]}/{nclean['hash'][1]}, "
          f"unified {nclean['unified_hash'][0]}/{nclean['unified_hash'][1]}")
    print(f"  counter multiplexing fraction (running/enabled): "
          f"hash {mult['hash']:.2f}, unified {mult['unified_hash']:.2f} "
          f"-- kernel-scaled; both measured identically, so the RATIO is fair")
    print()
    print(f"  {'counter':26s} {'hash':>13s} {'unified':>13s} {'delta%':>9s}")
    d = {}
    for k in EVENTS[:7]:
        a, b = probe["hash"][k] / PROBE_ROWS, probe["unified_hash"][k] / PROBE_ROWS
        d[k] = (b - a) / a * 100 if a else float("nan")
        print(f"  {k:26s} {a:13.4f} {b:13.4f} {d[k]:8.1f}%")
    ipc_h = probe["hash"]["PerfInstructions"] / probe["hash"]["PerfCPUCycles"]
    ipc_u = probe["unified_hash"]["PerfInstructions"] / probe["unified_hash"]["PerfCPUCycles"]
    ipc_d = (ipc_u - ipc_h) / ipc_h * 100
    print(f"  {'IPC':26s} {ipc_h:13.3f} {ipc_u:13.3f} {ipc_d:8.1f}%")
    print()
    ins = d["PerfInstructions"]; cm = d["PerfCacheMisses"]; tlb = d["PerfDataTLBMisses"]
    print("PREREG P3.2 scoring:")
    print(f"  instructions/row up >10%        : {ins:7.1f}%   {'MET' if ins > 10 else 'NOT MET'}")
    print(f"  |IPC delta| < 10%               : {ipc_d:7.1f}%   {'MET' if abs(ipc_d) < 10 else 'NOT MET'}")
    print(f"  |cache-misses/row delta| < 20%  : {cm:7.1f}%   {'MET' if abs(cm) < 20 else 'NOT MET'}")
    print(f"  (dTLB-misses/row delta)         : {tlb:7.1f}%")
    print()
    if ins > 10 and abs(ipc_d) < 10 and abs(cm) < 20:
        v = ("INSTRUCTION-COUNT cause -- the codegen story is SUPPORTED by the counters. "
             "More work per row at essentially unchanged memory behaviour.")
    elif ipc_d < -10 and abs(ins) < 10:
        v = "CACHE-FOOTPRINT cause -- the codegen story is REFUTED; candidate A5 is right."
    elif ins > 10 and (ipc_d < -10 or abs(cm) >= 20):
        v = ("BOTH signatures present -- registered in advance as possible. Report the "
             "split as unresolved rather than picking one.")
    else:
        v = "NEITHER signature clean -- UNSETTLED."
    print(f"  VERDICT: {v}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "gate"]); ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()
    (run(a.tag, a.reps) if a.cmd == "run" else gate(a.tag))
