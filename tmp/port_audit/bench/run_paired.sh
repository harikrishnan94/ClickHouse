#!/bin/bash
# Point-interleaved paired grid (PREREG amendment 2): per point, run the base
# binary then the candidate binary back-to-back.
# Usage: run_paired.sh <round_tag> <base_binary> <cand_binary> [points...]
set -u
TAG="$1"; BASE_BIN="$2"; CAND_BIN="$3"; shift 3
POINTS=("$@")
[ ${#POINTS[@]} -eq 0 ] && POINTS=(V A B C D E)
cd /mnt/ch/ClickHouse
echo "=== paired grid tag=$TAG start $(date -u +%FT%TZ)"; uptime
run_one() {
  local side="$1" bin="$2" name="$3" c="$4" r="$5" bp="$6" pp="$7" t="$8"
  local log="tmp/port_audit/bench/${TAG}_${side}_${name}.log"
  python3 bep/tools/join_mergetree_bench.py run --path /mnt/data/join_bench_data \
    --binary "$bin" \
    --cardinalities "$c" --multiplicities 1 --ratios "$r" --hit-rates 1.0 \
    --build-payload-columns "$bp" --probe-payload-columns "$pp" \
    --threads "$t" --runs 5 > "$log" 2>&1
  echo "point $name $side exit=$?"
}
run_pair() {
  run_one base "$BASE_BIN" "$@"
  run_one cand "$CAND_BIN" "$@"
}
for p in "${POINTS[@]}"; do
  case "$p" in
    V) run_pair V 4194304 1 1 1 32 ;;
    A) run_pair A 8388608 4 1 1 32 ;;
    B) run_pair B 524288000 1 1 1 32 ;;
    C) run_pair C 524288000 1 4 4 32 ;;
    D) run_pair D 67108864 4 0 0 32 ;;
    E) run_pair E 67108864 4 1 1 96 ;;
    *) echo "unknown point $p"; exit 2 ;;
  esac
done
echo "=== paired grid tag=$TAG done $(date -u +%FT%TZ)"; uptime
