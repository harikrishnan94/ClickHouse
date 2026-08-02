#!/usr/bin/env bash
# Hot-path divergence inventory: UHJ vs HashJoin / ConcurrentHashJoin.
# Prints file:symbol (or file:line) candidates; also documents the exact diff/rg cmds.
set -euo pipefail
REPO="${REPO:-/mnt/ch/ClickHouse}"
OUT="${1:-/mnt/ch/ClickHouse/tmp/uhj_parity/diff_inventory_out.txt}"

{
  echo "# diff_inventory $(date -Is)"
  echo "# work tip: $(git -C "$REPO" rev-parse --short HEAD) $(git -C "$REPO" log -1 --format='%s')"
  echo
  echo "## Commands used"
  echo "diff -u src/Interpreters/HashJoin/HashJoinMethodsImpl.h src/Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h | head"
  echo "rg -n 'insertFromBlockImplTypeCase|bucket_locks|blocks_mutex|parallel_build|TwoLevel|max_threads|createInMemoryHashJoin' ..."
  echo

  echo "## Comparison A — serial build: HashJoin vs UnifiedHashJoin"
  echo "### insertFromBlockImplTypeCase signature / locking"
  echo "HashJoin (no locks, single Arena&):"
  rg -n 'insertFromBlockImplTypeCase|Arena & pool|lock_guard|scoped_lock|parallel_build|bucket' \
    "$REPO/src/Interpreters/HashJoin/HashJoinMethodsImpl.h" | head -40 || true
  echo
  echo "UnifiedHashJoin (bucket_locks / blocks_mutex on parallel path; two-level byte accounting on serial):"
  rg -n 'insertFromBlockImplTypeCase|bucket_locks|blocks_mutex|parallel_build|impls\[0\]|scoped_lock' \
    "$REPO/src/Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h" | head -60 || true
  echo

  echo "### Map type / TwoLevel"
  echo "HashJoin chooseMethod / use_two_level_maps:"
  rg -n 'use_two_level_maps|TwoLevelHashMap|chooseMethod' \
    "$REPO/src/Interpreters/HashJoin/HashJoin.cpp" "$REPO/src/Interpreters/HashJoin/HashJoin.h" | head -40 || true
  echo "UHJ unconditional TwoLevel / num_buckets:"
  rg -n 'TwoLevel|num_buckets|bucketCountForThreads|BUCKETS_PER_THREAD|max_threads' \
    "$REPO/src/Interpreters/UnifiedHashJoin/HashJoin.h" "$REPO/src/Interpreters/UnifiedHashJoin/HashJoin.cpp" | head -50 || true
  echo

  echo "### Plumbing: createInMemoryHashJoin / SpillingHashJoin / PlannerJoins"
  rg -n 'createInMemoryHashJoin|max_threads|UnifiedHashJoin|ConcurrentHashJoin|SpillingHashJoin' \
    "$REPO/src/Interpreters/InMemoryHashJoin.cpp" \
    "$REPO/src/Interpreters/SpillingHashJoin.cpp" \
    "$REPO/src/Planner/PlannerJoins.cpp" | head -80 || true
  echo

  echo "## Comparison B — parallel build: ConcurrentHashJoin vs UnifiedHashJoin"
  echo "ConcurrentHashJoin addBlockToJoin / scatter / per-shard mutex:"
  rg -n 'addBlockToJoin|scatter|mutex|try_to_lock|HashJoin\(' \
    "$REPO/src/Interpreters/ConcurrentHashJoin.cpp" | head -50 || true
  echo
  echo "UHJ parallel build path (bucket_locks sizing / insert):"
  rg -n 'bucket_locks|addBlockToJoin|insertFromBlockImpl|num_buckets|build_mutex|supportParallelJoin' \
    "$REPO/src/Interpreters/UnifiedHashJoin/HashJoin.cpp" | head -60 || true
  echo

  echo "## Symbol-level diffstat (HashJoinMethodsImpl.h)"
  if command -v diff >/dev/null; then
    diff -u \
      "$REPO/src/Interpreters/HashJoin/HashJoinMethodsImpl.h" \
      "$REPO/src/Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h" \
      | awk '/^@@/ || /^[-+][^-+]/ {print}' | head -120 || true
  fi
  echo
  echo "DIFF_INVENTORY_DONE"
} | tee "$OUT"

echo "Wrote $OUT"
