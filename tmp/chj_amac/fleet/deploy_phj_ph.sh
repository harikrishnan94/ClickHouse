#!/bin/bash
# Deploy both arm binaries + the fleet_ab driver + the frozen plan to every shard
# in hosts.phj_ph.tsv, then record the per-shard smoke digest.
#
# Same contract as fleet/deploy.sh (unedited, it is the prior campaign's record):
# both binaries must appear in bins/MANIFEST.tsv by sha256 or this fails closed;
# fleet_ab.py resolves the frozen plan at <its dir>/fleet/matrix.json, so
# matrix.json and band_local.json go to $REMOTE_DIR/fleet/.
#
# Usage: deploy_phj_ph.sh <baseline.bin> <candidate.bin>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
BASE_BIN=$(readlink -f "$1"); CAND_BIN=$(readlink -f "$2")
KEY=$HERE/ssh_phj_ph/id_ed25519
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15"
REMOTE_DIR=/home/ubuntu/chj
HOSTS=${HOSTS:-$HERE/hosts.phj_ph.tsv}

for bin in "$BASE_BIN" "$CAND_BIN"; do
    grep -q "$(sha256sum "$bin" | cut -d' ' -f1)" "$HERE/../bins/MANIFEST.tsv" || {
        echo "FAILED: $bin sha256 not in MANIFEST.tsv" >&2; exit 1; }
done

while IFS=$'\t' read -r shard iid ip az; do
    [ -n "${shard:-}" ] || continue
    (
        ssh $SSH_OPTS "ubuntu@$ip" "mkdir -p $REMOTE_DIR/fleet" </dev/null
        scp -q $SSH_OPTS "$BASE_BIN" "ubuntu@$ip:$REMOTE_DIR/clickhouse-base" </dev/null
        scp -q $SSH_OPTS "$CAND_BIN" "ubuntu@$ip:$REMOTE_DIR/clickhouse-cand" </dev/null
        scp -q $SSH_OPTS "$HERE/../fleet_ab.py" "$HERE/calibration_rows.json" "ubuntu@$ip:$REMOTE_DIR/" </dev/null
        scp -q $SSH_OPTS "$HERE/matrix.json" "$HERE/band_local.json" "ubuntu@$ip:$REMOTE_DIR/fleet/" </dev/null
        ssh $SSH_OPTS "ubuntu@$ip" "chmod +x $REMOTE_DIR/clickhouse-base $REMOTE_DIR/clickhouse-cand;
            sha256sum $REMOTE_DIR/clickhouse-base $REMOTE_DIR/clickhouse-cand;
            uname -m; nproc; lscpu | head -20; free -g | head -2; df -h / | tail -1" \
            </dev/null > "$HERE/smoke_phjph_shard$shard.log" 2>&1
        echo "shard $shard ($ip): deployed"
    ) &
done < "$HOSTS"
wait
echo "DEPLOY DONE: verify smoke_phjph_shard*.log shas against bins/MANIFEST.tsv"
