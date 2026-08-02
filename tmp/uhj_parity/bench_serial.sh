#!/usr/bin/env bash
# U1 serial bench: unified_hash vs hash on the non-parallel build path.
# Reports per-run wall_ms and UserTimeMicroseconds; prints median + sample stdev.
# Requires: server on UHJ_PORT (default 9101), binary at build/reldeb.
set -euo pipefail

CH_BIN="${CH_BIN:-/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse}"
PORT="${UHJ_PORT:-9101}"
CH=("$CH_BIN" client --port "$PORT")
RUNS="${RUNS:-5}"
BUILD_ROWS="${BUILD_ROWS:-10000000}"
PROBE_ROWS="${PROBE_ROWS:-100000}"  # build-bound: small left, large right

COMMON=(
  --enable_join_runtime_filters=0
  --max_bytes_before_external_join=0
  --max_threads=1
  --max_block_size=65505
)

QUERY="SELECT count() FROM (SELECT number AS k FROM numbers(${PROBE_ROWS})) AS l
  INNER JOIN (SELECT number AS k FROM numbers(${BUILD_ROWS})) AS r ON l.k = r.k"

median() {
  # stdin: one number per line → median (average of two middles if even)
  sort -n | awk '
    { a[NR]=$1 }
    END {
      if (NR==0) { print "NA"; exit }
      if (NR%2==1) print a[(NR+1)/2]
      else print (a[NR/2]+a[NR/2+1])/2
    }'
}

stdev() {
  awk '
    { a[NR]=$1; s+=$1 }
    END {
      if (NR<2) { print 0; exit }
      m=s/NR; v=0
      for (i=1;i<=NR;i++) { d=a[i]-m; v+=d*d }
      print sqrt(v/(NR-1))
    }'
}

run_algo() {
  local algo="$1"
  local wall_file cpu_file
  wall_file=$(mktemp)
  cpu_file=$(mktemp)
  echo "# algo=$algo runs=$RUNS build_rows=$BUILD_ROWS probe_rows=$PROBE_ROWS" >&2
  for i in $(seq 1 "$RUNS"); do
    local qid="u1s-${algo}-${i}-$(date +%s%N)"
    local start end wall_ms cpu_us
    start=$(date +%s%N)
    "${CH[@]}" "${COMMON[@]}" --join_algorithm="$algo" --query_id "$qid" -q "$QUERY" >/dev/null
    end=$(date +%s%N)
    wall_ms=$(( (end - start) / 1000000 ))
    "${CH[@]}" -q "SYSTEM FLUSH LOGS query_log" >/dev/null
    cpu_us=$("${CH[@]}" -q "SELECT ProfileEvents['UserTimeMicroseconds'] FROM system.query_log WHERE type='QueryFinish' AND query_id='$qid' ORDER BY event_time_microseconds DESC LIMIT 1")
    echo "$wall_ms" >> "$wall_file"
    echo "$cpu_us" >> "$cpu_file"
    echo "  run=$i wall_ms=$wall_ms cpu_us=$cpu_us" >&2
  done
  local med_w std_w med_c std_c
  med_w=$(median < "$wall_file")
  std_w=$(stdev < "$wall_file")
  med_c=$(median < "$cpu_file")
  std_c=$(stdev < "$cpu_file")
  echo "RESULT algo=$algo wall_median_ms=$med_w wall_stdev_ms=$std_w cpu_median_us=$med_c cpu_stdev_us=$std_c"
  echo "RAW_WALL algo=$algo $(tr '\n' ' ' < "$wall_file")"
  echo "RAW_CPU algo=$algo $(tr '\n' ' ' < "$cpu_file")"
  rm -f "$wall_file" "$cpu_file"
}

echo "BENCH_SERIAL begin port=$PORT binary=$CH_BIN $(date -Is)"
echo "SETTINGS max_threads=1 enable_join_runtime_filters=0 max_bytes_before_external_join=0"
"${CH[@]}" -q "SELECT version()" >/dev/null
run_algo hash
run_algo unified_hash
echo "BENCH_SERIAL_DONE"
