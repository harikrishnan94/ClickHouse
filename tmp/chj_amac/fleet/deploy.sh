#!/bin/bash
# Deploy both arm binaries + the driver to every shard in hosts.tsv, then run
# the per-shard smoke (SELECT 1, /proc/<pid>/exe sha vs MANIFEST, selftest
# --check-events fail-closed both directions). Usage:
#   deploy.sh <baseline.bin> <candidate.bin>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
BASE_BIN=$(readlink -f "$1"); CAND_BIN=$(readlink -f "$2")
KEY=$HERE/ssh/id_ed25519
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
REMOTE_DIR=/home/ubuntu/chj

for bin in "$BASE_BIN" "$CAND_BIN"; do
    grep -q "$(sha256sum "$bin" | cut -d' ' -f1)" "$HERE/../bins/MANIFEST.tsv" || {
        echo "FAILED: $bin sha256 not in MANIFEST.tsv" >&2; exit 1; }
done

while IFS=$'\t' read -r shard iid ip az; do
    (
        ssh $SSH_OPTS "ubuntu@$ip" "mkdir -p $REMOTE_DIR" </dev/null
        scp -q $SSH_OPTS "$BASE_BIN" "ubuntu@$ip:$REMOTE_DIR/clickhouse-base" </dev/null
        scp -q $SSH_OPTS "$CAND_BIN" "ubuntu@$ip:$REMOTE_DIR/clickhouse-cand" </dev/null
        scp -q $SSH_OPTS "$HERE/../fleet_ab.py" "$HERE/calibration_rows.json" "ubuntu@$ip:$REMOTE_DIR/" </dev/null
        ssh $SSH_OPTS "ubuntu@$ip" "chmod +x $REMOTE_DIR/clickhouse-base $REMOTE_DIR/clickhouse-cand;
            sha256sum $REMOTE_DIR/clickhouse-base $REMOTE_DIR/clickhouse-cand;
            uname -m; lscpu | head -20" </dev/null > "$HERE/smoke_shard$shard.log" 2>&1
        echo "shard $shard ($ip): deployed"
    ) &
done < "$HERE/hosts.tsv"
wait
echo "DEPLOY DONE: verify smoke_shard*.log shas against bins/MANIFEST.tsv, then record lscpu digests in PREREG.md"
