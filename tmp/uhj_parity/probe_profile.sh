#!/usr/bin/env bash
# U1-DISC: Real profiler; lock share excludes cond_wait false positives.
# Lock samples require pthread_mutex_lock/unlock or std::__1::lock / scoped_lock / lock_guard
# as a *frame symbol*, not a mere template parameter containing the substring "mutex".
set -euo pipefail
CH_BIN="${CH_BIN:-/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse}"
PORT="${UHJ_PORT:-9101}"
CH=("$CH_BIN" client --host 127.0.0.1 --port "$PORT")
OUTDIR=/mnt/ch/ClickHouse/tmp/uhj_parity
LABEL="$1"
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

# Frame is a lock acquisition/release symbol (not condition_variable wait).
# Match on mangled-friendly demangled names that are actual lock ops.
LOCK_PRED="
  arrayExists(x ->
    (
      position(demangle(addressToSymbol(x)), 'pthread_mutex_lock') > 0
      OR position(demangle(addressToSymbol(x)), 'pthread_mutex_unlock') > 0
      OR position(demangle(addressToSymbol(x)), '__pthread_mutex_lock') > 0
      OR position(demangle(addressToSymbol(x)), 'std::__1::mutex::lock') > 0
      OR position(demangle(addressToSymbol(x)), 'std::__1::mutex::unlock') > 0
      OR position(demangle(addressToSymbol(x)), 'std::__1::lock') > 0
      OR position(demangle(addressToSymbol(x)), 'scoped_lock') > 0
      OR position(demangle(addressToSymbol(x)), 'lock_guard') > 0
    )
    AND position(demangle(addressToSymbol(x)), 'condition_variable') = 0
    AND position(demangle(addressToSymbol(x)), 'pthread_cond') = 0
  , trace)
"

INSERT_PRED="
  arrayExists(x ->
    position(demangle(addressToSymbol(x)), 'insertFromBlock') > 0
    OR position(demangle(addressToSymbol(x)), 'insertAll') > 0
    OR position(demangle(addressToSymbol(x)), 'insertOne') > 0
  , trace)
"

TOTAL=$("${CH[@]}" -q "SELECT count() FROM system.trace_log WHERE query_id='$QID' AND trace_type='Real'")
LOCK=$("${CH[@]}" --allow_introspection_functions=1 -q "
SELECT count() FROM system.trace_log
WHERE query_id='$QID' AND trace_type='Real' AND ($LOCK_PRED)
")
INSERT=$("${CH[@]}" --allow_introspection_functions=1 -q "
SELECT count() FROM system.trace_log
WHERE query_id='$QID' AND trace_type='Real' AND ($INSERT_PRED)
")
# Discriminating: lock frames under insert path
LOCK_IN_INSERT=$("${CH[@]}" --allow_introspection_functions=1 -q "
SELECT count() FROM system.trace_log
WHERE query_id='$QID' AND trace_type='Real' AND ($LOCK_PRED) AND ($INSERT_PRED)
")

echo "SUMMARY label=$LABEL total=$TOTAL lock_ops=$LOCK insert_related=$INSERT lock_in_insert=$LOCK_IN_INSERT"
echo "SUMMARY label=$LABEL total=$TOTAL lock_ops=$LOCK insert_related=$INSERT lock_in_insert=$LOCK_IN_INSERT" | tee -a "$OUTDIR/probe_summaries_v2.txt"
