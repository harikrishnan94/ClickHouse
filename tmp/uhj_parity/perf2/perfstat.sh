#!/usr/bin/env bash
# G3.2: hardware counters via `perf stat` LAUNCHING the process, so perf follows every
# thread the join creates. Two earlier routes were tried and are unsound on this host:
#   * `perf stat -p <server pid>` follows only threads existing at attach time, and
#     ClickHouse spawns per-query threads -- it undercounted by ~2000x.
#   * ClickHouse's own `metrics_perf_events_enabled` UInt64-underflows its per-thread
#     deltas; only 2 of 15 reps were usable.
# Data is a local reproduction of cell INNER|u64|hi|t1|medium (same SCATTER constant,
# same 3,600,000 output rows as the server-side cell).
set -u
CH=build/reldeb/programs/clickhouse
LO=tmp/uhj_parity/perf2/tmp/lo
OUT=tmp/uhj_parity/perf2/results/perfstat_${TAG:-raw}.csv
EV=cycles,instructions,cache-misses,cache-references,dTLB-load-misses,branch-misses
SET="--max_threads=1 --parallel_hash_join_threshold=0 --enable_join_runtime_filters=0 --max_bytes_before_external_join=0 --query_plan_join_swap_table=0 --max_block_size=65409 --max_joined_block_size_rows=65409"
Q="SELECT count() AS cnt, sum(l.k) AS s1, sum(r.v) AS s2 FROM p_medium_hi AS l ${KIND:-INNER} JOIN b_medium AS r ON l.k = r.k"
: > "$OUT"
for rep in $(seq 1 "${REPS:-7}"); do
  for algo in hash unified_hash control; do      # interleaved within each rep
    if [ "$algo" = control ]; then
      perf stat -x, -e $EV -o /dev/stdout -- $CH local --path $LO --query "SELECT 1" >/dev/null 2>tmp/uhj_parity/perf2/tmp/p.err
    else
      perf stat -x, -e $EV -o /dev/stdout -- $CH local --path $LO $SET --join_algorithm=$algo --query "$Q" >/dev/null 2>tmp/uhj_parity/perf2/tmp/p.err
    fi
    perf stat -x, -e $EV -- true >/dev/null 2>/dev/null
    # re-run capturing perf's own stderr, which is where -x, output goes
    if [ "$algo" = control ]; then
      perf stat -x, -e $EV -- $CH local --path $LO --query "SELECT 1" >/dev/null 2>tmp/uhj_parity/perf2/tmp/p.csv
    else
      perf stat -x, -e $EV -- $CH local --path $LO $SET --join_algorithm=$algo --query "$Q" >/dev/null 2>tmp/uhj_parity/perf2/tmp/p.csv
    fi
    while IFS=, read -r val _ ev rest; do
      [ -n "${ev:-}" ] && echo "$rep,$algo,$ev,$val" >> "$OUT"
    done < tmp/uhj_parity/perf2/tmp/p.csv
  done
  echo "  rep $rep done" >&2
done
echo "-> $OUT" >&2
