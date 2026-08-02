#!/usr/bin/env bash
# OPERATIONAL ONLY. These numbers were NOT used to accept or reject any work in the U3 mission;
# performance was explicitly a non-gate there. This exists to answer "where does unified_hash stand"
# after the alignment, covering the two shapes the existing harness does not: a probe-bound join,
# and a RIGHT join that actually emits non-joined rows (the path M1 changed).
set -uo pipefail
CH_BIN="${CH_BIN:-/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse}"
CH=("$CH_BIN" client --port "${UHJ_PORT:-9101}")
RUNS="${RUNS:-7}"

median() { sort -n | awk '{a[NR]=$1} END {if(NR%2==1)print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}'; }
stdev()  { awk '{a[NR]=$1;s+=$1} END {if(NR<2){print 0;exit} m=s/NR; for(i=1;i<=NR;i++){d=a[i]-m;v+=d*d} print sqrt(v/(NR-1))}'; }

bench() {
    local label="$1" algo="$2" threads="$3" query="$4"; shift 4
    local wf cf; wf=$(mktemp); cf=$(mktemp)
    for i in $(seq 1 "$RUNS"); do
        local qid="u3x-${label}-${algo}-${i}-$(date +%s%N)" s e
        s=$(date +%s%N)
        "${CH[@]}" --enable_join_runtime_filters=0 --max_bytes_before_external_join=0 \
                   --max_threads="$threads" --join_algorithm="$algo" "$@" \
                   --query_id "$qid" -q "$query" > /dev/null
        e=$(date +%s%N)
        echo $(( (e - s) / 1000000 )) >> "$wf"
        "${CH[@]}" -q "SYSTEM FLUSH LOGS query_log" > /dev/null
        "${CH[@]}" -q "SELECT ProfileEvents['UserTimeMicroseconds'] FROM system.query_log
                       WHERE type='QueryFinish' AND query_id='$qid'
                       ORDER BY event_time_microseconds DESC LIMIT 1" >> "$cf"
    done
    printf 'RESULT case=%s algo=%-14s threads=%-3s wall_median_ms=%-7s wall_stdev=%-9s cpu_median_us=%-9s cpu_stdev=%s\n' \
        "$label" "$algo" "$threads" "$(median <"$wf")" "$(stdev <"$wf")" "$(median <"$cf")" "$(stdev <"$cf")"
    rm -f "$wf" "$cf"
}

# Probe-bound: small right (build) side, large left (probe) side.
PROBE_Q="SELECT count() FROM (SELECT number % 1000000 AS k FROM numbers_mt(40000000)) AS l
         INNER JOIN (SELECT number AS k FROM numbers(1000000)) AS r ON l.k = r.k"

# RIGHT join leaving ~15M non-joined right rows, so the non-joined emission path dominates.
# query_plan_join_swap_table=0 keeps the optimizer from rewriting it to a LEFT join, which would
# leave the parallel non-joined path dormant for every algorithm.
RIGHT_Q="SELECT count() FROM (SELECT number AS k FROM numbers(5000000)) AS l
         RIGHT JOIN (SELECT number AS k FROM numbers_mt(20000000)) AS r ON l.k = r.k"

echo "BENCH_EXTRA begin $(date -Is) runs=$RUNS"
for a in hash unified_hash;          do bench probe_bound "$a" 1  "$PROBE_Q"; done
for a in parallel_hash unified_hash; do bench probe_bound "$a" 16 "$PROBE_Q"; done
for a in hash unified_hash;          do bench right_nonjoined "$a" 1  "$RIGHT_Q" --query_plan_join_swap_table=0; done
for a in parallel_hash unified_hash; do bench right_nonjoined "$a" 16 "$RIGHT_Q" --query_plan_join_swap_table=0; done
echo "BENCH_EXTRA_DONE"
