#!/usr/bin/env python3
"""Check that the shipping probe path is identifiable from executing code.

After Stage 1 there is no within-binary fused-vs-split setting: batch length is constexpr
`PROBE_BATCH_ROWS` and fused vs split is `if constexpr (split_can_pay)`. This script proves
from CPU stacks that an INNER cell hits `probeTwoPhase` and a LEFT SEMI cell hits `EmitSink`.

    python3 verify_arm.py

`--scan` is retained as a no-op notice: the batch length is no longer a runtime knob.
"""

from __future__ import annotations

import argparse
import sys
import time

import harness as H

# One split_can_pay cell and one always-fused cell so both compile-time paths show up.
CELLS = [
    ("uhj_split", H.Cell("INNER", "u64", "hi", 1, "medium")),
    ("uhj_fused", H.Cell("LEFT SEMI", "u64", "hi", 1, "medium")),
]

MARKERS = [
    "DB::Unified::probeTwoPhase",
    "DB::Unified::EmitSink",
    "DB::Unified::consumeProbeBatch",
    "DB::Unified::HashJoin",
]


def profile_run(cell: H.Cell, tag: str) -> str:
    qid = f"verifyarm-{tag}-{int(time.time())}"
    settings = H.settings_for(cell, H.BASELINE_ARM)
    settings["query_profiler_cpu_time_period_ns"] = 1_000_000
    settings["query_profiler_real_time_period_ns"] = 0
    H.run_query(H.join_sql(cell, "timed"), settings, query_id=qid)
    return qid


def counts_for(qid: str) -> dict:
    out = {}
    for marker in MARKERS:
        n = H.run_query(
            f"SELECT countIf(arrayExists(x -> position(demangle(addressToSymbol(x)), "
            f"'{marker}') > 0, trace)) FROM system.trace_log "
            f"WHERE query_id = '{qid}' AND trace_type = 'CPU' FORMAT TSV").strip()
        out[marker] = int(n or 0)
    out["total"] = int(H.run_query(
        f"SELECT count() FROM system.trace_log WHERE query_id='{qid}' "
        f"AND trace_type='CPU' FORMAT TSV").strip() or 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true",
                    help="legacy: batch length is constexpr; prints a notice and exits")
    args = ap.parse_args()

    if args.scan:
        print(f"BATCH_LENGTH_CONSTEXPR={H.PROBE_BATCH_ROWS} "
              "(unified_hash_join_probe_batch_rows removed in Stage 1; nothing to scan)")
        return 0

    qids = {expect: profile_run(cell, expect) for expect, cell in CELLS}
    H.flush_logs()
    time.sleep(2)
    H.flush_logs()

    table = {expect: counts_for(q) for expect, q in qids.items()}
    width = max(len(m) for m in MARKERS) + 2
    labels = [expect for expect, _ in CELLS]
    print(f"{'marker':{width}s} " + " ".join(f"{a:>13s}" for a in labels))
    for marker in MARKERS + ["total"]:
        print(f"{marker:{width}s} " + " ".join(f"{table[a][marker]:13d}" for a in labels))

    print()
    ok = True
    for expect, cell in CELLS:
        verdict = H.assert_algorithm(qids[expect])
        hit = verdict["probe_verdict"] == expect
        print(f"{cell.kind:14s} expect={expect:9s} verdict={verdict['verdict']:12s} "
              f"probe={verdict['probe_verdict']:9s} samples={verdict['total_samples']:4d}  "
              f"{'ok' if hit else 'WRONG'}")
        ok &= hit
    print("DISCRIMINATOR_OK" if ok else "DISCRIMINATOR_BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
