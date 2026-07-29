#!/bin/bash
# jbmt A/A control: the baseline binary as BOTH arms, over 10 synthetic units
# spanning all ten key families (K0..K9).
#
# jbmt measures whatever server listens on an arm's port, so an A/A needs a
# THIRD server running the baseline (port 9007) alongside arm A's (9005) - two
# arms pointing at one port would not exercise the ABAB interleave at all.
# Its data root is a hardlink clone of arm A's root taken after `keys_store` was
# filled, so all three servers hold identical bytes.
#
# Usage: jbmt_aa.sh
set -euo pipefail

CAMP=/mnt/ch/ClickHouse/tmp/probe_campaign_20260729
ROOT=/mnt/data/probe_camp_jbmt
BIN_A=$CAMP/bins/clickhouse-baseline-a05f3ee81ff.bin
SRV=$CAMP/jbmt/join_bench_mt_servers.sh

mkdir -p "$ROOT/c" "$CAMP/results/aa_jbmt"
cd "$CAMP/jbmt"

if [ ! -d "$ROOT/c/data" ]
then
    echo "=== stop arm A so its root is quiescent for cloning $(date -u +%FT%TZ)"
    "$SRV" stop "$ROOT/a/scratch" || true
    sleep 5
    echo "=== clone arm A -> A/A control root $(date -u +%FT%TZ)"
    "$SRV" clone "$ROOT/a/data" "$ROOT/c/data"
    echo "=== restart arm A $(date -u +%FT%TZ)"
    "$SRV" start armA "$BIN_A" "$ROOT/a/data" 9005 8125 "$ROOT/a/scratch"
fi

echo "=== start A/A control server (baseline) on 9007 $(date -u +%FT%TZ)"
"$SRV" start aaC "$BIN_A" "$ROOT/c/data" 9007 8127 "$ROOT/c/scratch"

echo "=== server identity per port (independent of the arm's client hash) ==="
for p in 9005 9006 9007; do
    printf 'port %s GIT_HASH: ' "$p"
    "$BIN_A" client --port "$p" -q "SELECT value FROM system.build_options WHERE name = 'GIT_HASH'"
done

echo "=== jbmt A/A sweep (10 units, K0..K9) $(date -u +%FT%TZ)"
python3 join_bench_mt.py sweep \
    --suite synthetic --shards 1 --shard 0 \
    --results "$CAMP/results/aa_jbmt/results.jsonl" \
    --only "$(cat "$CAMP/logs/aa_jbmt_regex.txt")" \
    --algorithms parallel_hash \
    --arm "aaA=$BIN_A:9005" --arm "aaB=$BIN_A:9007"
echo "=== A/A sweep done $(date -u +%FT%TZ)"
