#!/usr/bin/env bash
# Reproduce the `unified_hash` probe regression against `parallel_hash` (REPORT_FIX.md §12).
#
# Self-contained: `clickhouse local`, no server and no tables, so it runs against any build.
#
#   bash repro_k32.sh [path/to/clickhouse] [reps]
#
# What it measures. 67.1M distinct build keys, 67.1M probe rows at a 50% match rate, at
# `max_threads=16`, which gives `unified_hash` K = 2*bit_ceil(16) = 32 buckets. Expect
# `unified_hash` about +45..55% on wall and about +120% on the probe phase. The comparison is
# interleaved with the algorithm order rotated, so drift cannot masquerade as the effect.
#
# The knob that matters is `--max_threads`, because it sets K. Measured on the reference build,
# 67.1M keys, probe 1x, 7 reps (probe-phase CPU, unified vs parallel_hash):
#
#     threads   8 -> K= 16 :   +5.6%
#     threads  16 -> K= 32 : +123.5%   <- worst, and what this script runs
#     threads  24 -> K= 64 :  +39.0%
#     threads  32 -> K= 64 :  +40.6%
#     threads  48 -> K=128 :   -2.7%
#     threads  64 -> K=128 :   -3.3%
#
# Two thread counts that share a K show the same penalty, so the driver is K and not the thread
# count. It is NOT keys-per-bucket: holding that at 2.1M and shrinking the data gives -15.5%
# (K=4), -3.2% (K=8), -2.6% (K=16) and +47.6% (K=32) on wall, so a smaller/faster repro at the
# same keys-per-bucket does NOT reproduce it. The whole 67.1M-key table is needed.
set -euo pipefail

CH="${1:-/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse}"
REPS="${2:-5}"
THREADS="${THREADS:-16}"
KEYS="${KEYS:-67108864}"          # distinct build keys; 50% of probe keys match one
MULT="${MULT:-4}"                 # probe rows = KEYS * MULT

SETTINGS=(
  --max_threads="$THREADS"
  --parallel_hash_join_threshold=0
  --query_plan_join_swap_table=0
  --enable_join_runtime_filters=0
  --max_bytes_before_external_join=0
  --max_block_size=65409
  --max_joined_block_size_rows=65409
)

PROBE=$((KEYS * MULT))

QUERY="SELECT count() AS cnt, sum(l.lk) AS s
       FROM (SELECT number % $((KEYS * 2)) AS lk FROM numbers_mt($PROBE)) AS l
       INNER JOIN (SELECT number AS rk FROM numbers_mt($KEYS)) AS r
       ON l.lk = r.rk"

run_one() {  # $1 = algorithm -> prints elapsed milliseconds
  local t0 t1
  t0=$(date +%s%N)
  "$CH" local "${SETTINGS[@]}" --join_algorithm="$1" -q "$QUERY" > /tmp/repro_k32.out
  t1=$(date +%s%N)
  echo $(( (t1 - t0) / 1000000 ))
}

echo "binary : $CH"
echo "shape  : $KEYS distinct build keys, $PROBE probe rows (${MULT}x), 50% match, INNER JOIN"
echo "threads: $THREADS  ->  unified_hash buckets K = $(python3 -c "
import math;t=$THREADS;print(1 if t<=1 else 2*(1<<(t-1).bit_length()))")"
echo

declare -A ANS
for a in parallel_hash unified_hash; do
  run_one "$a" > /dev/null                       # warm up
  ANS[$a]=$(cat /tmp/repro_k32.out)
done
if [ "${ANS[parallel_hash]}" != "${ANS[unified_hash]}" ]; then
  echo "ANSWERS DIFFER -- the two algorithms are not computing the same thing:"
  echo "  parallel_hash: ${ANS[parallel_hash]}"
  echo "  unified_hash : ${ANS[unified_hash]}"
  exit 1
fi
echo "answers agree: ${ANS[parallel_hash]}"
echo

declare -A T
for a in parallel_hash unified_hash; do T[$a]=""; done
for ((rep = 0; rep < REPS; rep++)); do
  if (( rep % 2 == 0 )); then order=(parallel_hash unified_hash); else order=(unified_hash parallel_hash); fi
  for a in "${order[@]}"; do
    ms=$(run_one "$a")
    T[$a]="${T[$a]} $ms"
    printf "  rep %d  %-14s %6s ms\n" "$rep" "$a" "$ms"
  done
done

echo
python3 - "${T[parallel_hash]}" "${T[unified_hash]}" <<'PY'
import statistics, sys
par = [float(x) for x in sys.argv[1].split()]
uni = [float(x) for x in sys.argv[2].split()]
p, u = statistics.median(par), statistics.median(uni)
print(f"median wall  parallel_hash {p:8.0f} ms   unified_hash {u:8.0f} ms   "
      f"unified is {100 * (u - p) / p:+.1f}%")
print("REGRESSION REPRODUCED" if (u - p) / p > 0.15 else
      "not reproduced at this threshold - check --max_threads and the build")
PY
