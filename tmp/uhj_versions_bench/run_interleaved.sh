#!/usr/bin/env bash
# Interleaved A/B driver: load once into shared data dir, then
# baseline, uhj, baseline, uhj, ... for ROUNDS rounds (bench only).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
ROUNDS="${ROUNDS:-5}"
export TRIES="${TRIES:-6}"
export QUERY_TIMEOUT="${QUERY_TIMEOUT:-600}"
export LOAD_DATASETS="${LOAD_DATASETS:-coffeeshop tpch tpcds job}"
LOG="${WORK}/logs/interleaved.log"
mkdir -p "${WORK}/logs"

{
  echo "=== interleaved start $(date -Is) rounds=${ROUNDS} ==="

  # One shared load (baseline binary; table format is identical across arms).
  if [ ! -f "${WORK}/server_shared/.load_complete" ]; then
    echo "========== LOAD (shared) $(date -Is) =========="
    # Drop any partial per-arm data from earlier attempts.
    rm -rf "${WORK}/server_baseline/data" "${WORK}/server_uhj/data" 2>/dev/null || true
    PHASE=load ARM=baseline ROUND=0 bash "${HERE}/run_arm.sh"
    touch "${WORK}/server_shared/.load_complete"
    echo "========== LOAD DONE $(date -Is) =========="
  else
    echo "shared load already present; skipping load"
  fi

  for r in $(seq 1 "${ROUNDS}"); do
    for arm in baseline uhj; do
      echo "========== ROUND ${r} ARM ${arm} $(date -Is) =========="
      PHASE=bench ARM="${arm}" ROUND="${r}" bash "${HERE}/run_arm.sh"
      echo "========== DONE ROUND ${r} ARM ${arm} $(date -Is) =========="
    done
  done
  echo "=== interleaved done $(date -Is) ==="
} 2>&1 | tee -a "${LOG}"
echo "INTERLEAVED_OK"
