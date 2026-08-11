#!/usr/bin/env bash
# Interleaved A/B driver: baseline, uhj, baseline, uhj, ... for ROUNDS rounds.
# Load datasets once on baseline (shared on-disk server data is NOT shared —
# each arm has its own server_dir; we load into both on first pass, or load
# once per arm on round 1).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
ROUNDS="${ROUNDS:-5}"
export TRIES="${TRIES:-6}"
export QUERY_TIMEOUT="${QUERY_TIMEOUT:-300}"
export LOAD_DATASETS="${LOAD_DATASETS:-coffeeshop tpch tpcds job}"
LOG="${WORK}/logs/interleaved.log"
mkdir -p "${WORK}/logs"

{
  echo "=== interleaved start $(date -Is) rounds=${ROUNDS} ==="
  for r in $(seq 1 "${ROUNDS}"); do
    for arm in baseline uhj; do
      echo "========== ROUND ${r} ARM ${arm} $(date -Is) =========="
      # Round 1: load+bench. Later rounds: data already on disk for that arm —
      # re-start server and bench only (tables persist in server_${arm}/data).
      if [ "${r}" = 1 ]; then
        PHASE=all ARM="${arm}" ROUND="${r}" bash "${HERE}/run_arm.sh"
      else
        PHASE=bench ARM="${arm}" ROUND="${r}" bash "${HERE}/run_arm.sh"
      fi
      echo "========== DONE ROUND ${r} ARM ${arm} $(date -Is) =========="
    done
  done
  echo "=== interleaved done $(date -Is) ==="
} 2>&1 | tee -a "${LOG}"
echo "INTERLEAVED_OK"
