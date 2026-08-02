#!/usr/bin/env bash
# U1-DISC: sampling Real profiler; summarize top frames mentioning mutex/lock vs insert.
set -euo pipefail
CH_BIN="${CH_BIN:-/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse}"
PORT="${UHJ_PORT:-9101}"
CH=("$CH_BIN" client --host 127.0.0.1 --port "$PORT")
OUTDIR=/mnt/ch/ClickHouse/tmp/uhj_parity
LABEL="$1"   # e.g. serial_hash
ALGO="$2"
THREADS="$3"
BUILD_ROWS="$4"
PROBE_ROWS="$5"

QID="disc-${LABEL}-$(date +%s%N)"
QUERY="SELECT count() FROM (SELECT number AS k FROM numbers(${PROBE_ROWS})) AS l
  INNER JOIN (SELECT number AS k FROM numbers_mt(${BUILD_ROWS})) AS r ON l.k = r.k"

echo "PROBE label=$LABEL algo=$ALGO threads=$THREADS qid=$QID"
"${CH[@]}" --join_algorithm="$ALGO" --max_threads="$THREADS" \
  --max_bytes_before_external_join=0 --enable_join_runtime_filters=0 \
  --query_profiler_real_time_period_ns=10000000 \
  --query_id "$QID" -q "$QUERY" >/dev/null

"${CH[@]}" -q "SYSTEM FLUSH LOGS"

# Top 12 stacks by samples (symbol demangled, truncated)
"${CH[@]}" --allow_introspection_functions=1 -q "
SELECT
  count() AS samples,
  arrayStringConcat(
    arrayMap(x -> demangle(addressToSymbol(x)), arraySlice(trace, 1, 8)),
    ' | ') AS stack_top
FROM system.trace_log
WHERE query_id = '$QID' AND trace_type = 'Real'
GROUP BY trace
ORDER BY samples DESC
LIMIT 12
FORMAT PrettyCompact
" | tee "$OUTDIR/probe_${LABEL}_stacks.txt"

# Aggregate keyword shares
TOTAL=$("${CH[@]}" -q "SELECT count() FROM system.trace_log WHERE query_id='$QID' AND trace_type='Real'")
MUTEX=$("${CH[@]}" --allow_introspection_functions=1 -q "
SELECT count() FROM system.trace_log
WHERE query_id='$QID' AND trace_type='Real'
  AND arrayExists(x -> positionCaseInsensitive(demangle(addressToSymbol(x)), 'mutex') > 0
                    OR positionCaseInsensitive(demangle(addressToSymbol(x)), 'pthread_mutex') > 0
                    OR positionCaseInsensitive(demangle(addressToSymbol(x)), 'std::lock') > 0
                    OR positionCaseInsensitive(demangle(addressToSymbol(x)), 'scoped_lock') > 0
                    OR positionCaseInsensitive(demangle(addressToSymbol(x)), 'lock_guard') > 0, trace)
")
INSERT=$("${CH[@]}" --allow_introspection_functions=1 -q "
SELECT count() FROM system.trace_log
WHERE query_id='$QID' AND trace_type='Real'
  AND arrayExists(x -> positionCaseInsensitive(demangle(addressToSymbol(x)), 'insertFromBlock') > 0
                    OR positionCaseInsensitive(demangle(addressToSymbol(x)), 'insertAll') > 0
                    OR positionCaseInsensitive(demangle(addressToSymbol(x)), 'insertOne') > 0, trace)
")
echo "SUMMARY label=$LABEL total=$TOTAL mutex_related=$MUTEX insert_related=$INSERT"
echo "SUMMARY label=$LABEL total=$TOTAL mutex_related=$MUTEX insert_related=$INSERT" | tee -a "$OUTDIR/probe_summaries.txt"
