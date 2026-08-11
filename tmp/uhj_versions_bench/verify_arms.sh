#!/usr/bin/env bash
# After a uhj round: confirm every Select that used a join went through UnifiedHashJoin,
# and dump settings/dataset checksums for arm comparability.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/verify"
mkdir -p "${OUT}"

# Diff settings between latest baseline and uhj dumps
{
  echo "=== settings dumps ==="
  ls -1 "${WORK}/logs"/*.settings.tsv 2>/dev/null || true
  echo "--- latest baseline vs uhj settings ---"
  b=$(ls -1t "${WORK}/logs"/baseline_r*.settings.tsv 2>/dev/null | head -1 || true)
  u=$(ls -1t "${WORK}/logs"/uhj_r*.settings.tsv 2>/dev/null | head -1 || true)
  if [ -n "$b" ] && [ -n "$u" ]; then
    echo "baseline=$b"
    echo "uhj=$u"
    diff -u "$b" "$u" || true
  fi
  echo
  echo "=== join_algorithm files ==="
  for f in "${WORK}/logs"/*.join_algorithm.txt; do
    [ -f "$f" ] || continue
    echo "$(basename "$f"): $(tr -d '\n' < "$f")"
  done
  echo
  echo "=== EXPLAIN Algorithm lines (uhj arm) ==="
  rg -n 'Algorithm:' "${WORK}/logs"/explain_uhj_* 2>/dev/null || true
  echo
  echo "=== EXPLAIN Algorithm lines (baseline arm) ==="
  rg -n 'Algorithm:' "${WORK}/logs"/explain_baseline_* 2>/dev/null || true
} | tee "${OUT}/arm_comparability.txt"

# Dataset checksums (native.zst sources — shared across arms)
{
  echo "=== dataset sha256 ==="
  cd "${WORK}/data"
  sha256sum *.zst | sort
} | tee "${OUT}/dataset_checksums.txt"
