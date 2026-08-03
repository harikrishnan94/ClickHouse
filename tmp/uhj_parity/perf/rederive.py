#!/usr/bin/env python3
"""Recompute the phase split in an existing runs.jsonl from the raw log.

Used when the phase->processor mapping was wrong but the measurements were not.
The timings in `runs.jsonl` are what was measured and are left untouched; only
the derived `build_us` / `probe_us` / `nonjoined_us` / `other_us` fields are
recomputed from `system.processors_profile_log`, which still holds the raw rows.

Re-running the sweep instead would also change the timings, which would mean
re-measuring to fix a bookkeeping error -- and would quietly discard the numbers
the gates were already run against.

    python3 rederive.py results/runs.jsonl
"""

from __future__ import annotations

import json
import os
import sys

import harness as H


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H.PERF_DIR, "results", "runs.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    qids = [r["query_id"] for r in recs if r.get("query_id")]
    print(f"records={len(recs)} with query_id={len(qids)}")

    # One query for the whole sweep rather than one per run: ~2500 round trips
    # would take longer than the sweep did.
    H.flush_logs()
    rows = H.run_query(
        """
        SELECT query_id, name, sum(elapsed_us)
        FROM system.processors_profile_log
        WHERE event_date >= today() - 1
        GROUP BY query_id, name FORMAT TSV
        """
    )
    per_q: dict[str, dict[str, int]] = {}
    for line in rows.strip().split("\n"):
        if not line:
            continue
        qid, name, us = line.split("\t")
        per_q.setdefault(qid, {})[name] = per_q.setdefault(qid, {}).get(name, 0) + int(us)
    print(f"query_ids found in processors_profile_log: {len(per_q)}")

    changed = missing = 0
    for r in recs:
        q = r.get("query_id")
        if not q or q not in per_q:
            if q and "build_us" in r:
                missing += 1
            continue
        names = per_q[q]
        phases = {p: sum(names.get(n, 0) for n in ns)
                  for p, ns in H.PHASE_PROCESSORS.items()}
        total = sum(names.values())
        phases["other"] = total - sum(phases.values())
        phases["total_proc"] = total
        new = {f"{k}_us": v for k, v in phases.items()}
        if any(r.get(k) != v for k, v in new.items()):
            changed += 1
        r.update(new)

    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    print(f"rewritten: {path}")
    print(f"records with changed phase fields: {changed}")
    print(f"records whose query_id is no longer in the log (left as-is): {missing}")

    nz = sum(1 for r in recs if r.get("purpose") == "timed" and r.get("nonjoined_us", 0) > 0)
    tot = sum(1 for r in recs if r.get("purpose") == "timed")
    print(f"timed runs with a non-zero non-joined phase: {nz}/{tot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
