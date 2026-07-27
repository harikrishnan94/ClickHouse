#!/bin/bash
# Pull per-shard results + logs from the fleet into fleet/results/.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
KEY=$HERE/ssh/id_ed25519
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
REMOTE_DIR=/home/ubuntu/chj
mkdir -p "$HERE/results"

while IFS=$'\t' read -r shard iid ip az; do
    scp -q $SSH_OPTS "ubuntu@$ip:$REMOTE_DIR/results.shard$shard.jsonl" \
        "$HERE/results/" </dev/null 2>/dev/null || echo "shard $shard: no results yet"
    scp -q $SSH_OPTS "ubuntu@$ip:$REMOTE_DIR/sweep.shard$shard.log" \
        "$HERE/results/" </dev/null 2>/dev/null || true
done < "$HERE/hosts.tsv"
ls -la "$HERE/results/"
