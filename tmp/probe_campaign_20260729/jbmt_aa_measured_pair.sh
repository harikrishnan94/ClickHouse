#!/bin/bash
# A/A control on the EXACT pair of server instances the measured run used.
#
# The first real-suite A/A compared ports 9005 and 9007; the measured sweep
# compares 9005 against 9006. Same construction, but not the same instances, so
# the earlier control bounded the channel only by analogy. This runs the baseline
# binary on BOTH 9005 and 9006 - i.e. arm B's own server instance and data root,
# with the candidate binary swapped out - so a per-instance or per-port bias in
# the measured channel would show up as a non-TIE cell.
#
# It also answers the verifier's port-swap question for the real suite: arm A is
# the LOWER port here as in the measured run, and `--swap` reverses the arm-to-port
# assignment so a fixed per-port offset would flip sign.
#
# Restores the candidate on 9006 on the way out, so the measured configuration is
# exactly as it was.
#
# Usage: jbmt_aa_measured_pair.sh [--swap]
set -euo pipefail

CAMP=/mnt/ch/ClickHouse/tmp/probe_campaign_20260729
ROOT=/mnt/data/probe_camp_jbmt
BIN_A=$CAMP/bins/clickhouse-baseline-a05f3ee81ff.bin
BIN_B=$CAMP/bins/clickhouse-phjph-fa5667f2da7.bin
SRV=$CAMP/jbmt/join_bench_mt_servers.sh
SWAP=${1:-}
OUT=$CAMP/results/aa_real_pair${SWAP:+_swap}

restore_candidate() {
    echo "=== restoring the candidate on 9006 $(date -u +%FT%TZ)"
    "$SRV" stop "$ROOT/b/scratch" || true
    sleep 5
    "$SRV" start armB "$BIN_B" "$ROOT/b/data" 9006 8126 "$ROOT/b/scratch"
    "$BIN_A" client --port 9006 -q "SELECT 'restored 9006 GIT_HASH ' || value FROM system.build_options WHERE name = 'GIT_HASH'"
}
trap restore_candidate EXIT

mkdir -p "$OUT"
cd "$CAMP/jbmt"

echo "=== swap the BASELINE binary onto arm B's own server (port 9006) $(date -u +%FT%TZ)"
"$SRV" stop "$ROOT/b/scratch" || true
sleep 5
"$SRV" start aaB6 "$BIN_A" "$ROOT/b/data" 9006 8126 "$ROOT/b/scratch"

for p in 9005 9006; do
    printf 'port %s now runs sha256: ' "$p"
    "$BIN_A" client --port "$p" -q "SELECT 1" >/dev/null
    pid=$(cat "$ROOT/$([ "$p" = 9005 ] && echo a || echo b)/scratch/server.pid")
    sha256sum "/proc/$pid/exe" | cut -c1-64
done

if [ -n "$SWAP" ]
then
    echo "=== A/A on the measured pair, arm->port assignment SWAPPED (aaA=9006, aaB=9005)"
    ARMS=(--arm "aaA=$BIN_A:9006" --arm "aaB=$BIN_A:9005")
else
    echo "=== A/A on the measured pair (aaA=9005, aaB=9006), same assignment as the measured run"
    ARMS=(--arm "aaA=$BIN_A:9005" --arm "aaB=$BIN_A:9006")
fi

python3 -u join_bench_mt.py sweep --suite real --tier a --shards 1 --shard 0 \
    --results "$OUT/results.jsonl" \
    --only "$(cat "$CAMP/logs/aa_real_regex.txt")" \
    --algorithms parallel_hash --min-timed-runs 11 --unit-time-budget 30 \
    "${ARMS[@]}"
echo "=== done $(date -u +%FT%TZ)"
