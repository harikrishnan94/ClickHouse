#!/bin/bash
# Run the pre-registered 6-point grid (PREREG.md) with a given tag and binary.
# Usage: run_grid.sh <tag> <binary> [points...]   (default: all 6 points)
set -u
TAG="$1"; BIN="$2"; shift 2
POINTS=("$@")
[ ${#POINTS[@]} -eq 0 ] && POINTS=(V A B C D E)
cd /mnt/ch/ClickHouse
mkdir -p tmp/port_audit/bench
echo "=== grid tag=$TAG binary=$BIN start $(date -u +%FT%TZ)"
uptime
run_point() {
  local name="$1" c="$2" r="$3" bp="$4" pp="$5" t="$6"
  local log="tmp/port_audit/bench/${TAG}_${name}.log"
  echo "--- point $name (c=$c r=$r bp=$bp pp=$pp t=$t) -> $log"
  python3 bep/tools/join_mergetree_bench.py run --path /mnt/data/join_bench_data \
    --binary "$BIN" \
    --cardinalities "$c" --multiplicities 1 --ratios "$r" --hit-rates 1.0 \
    --build-payload-columns "$bp" --probe-payload-columns "$pp" \
    --threads "$t" --runs 5 > "$log" 2>&1
  echo "point $name exit=$?"
}
for p in "${POINTS[@]}"; do
  case "$p" in
    V) run_point V 4194304 1 1 1 32 ;;
    A) run_point A 8388608 4 1 1 32 ;;
    B) run_point B 524288000 1 1 1 32 ;;
    C) run_point C 524288000 1 4 4 32 ;;
    D) run_point D 67108864 4 0 0 32 ;;
    E) run_point E 67108864 4 1 1 96 ;;
    *) echo "unknown point $p"; exit 2 ;;
  esac
done
echo "=== grid tag=$TAG done $(date -u +%FT%TZ)"
uptime
