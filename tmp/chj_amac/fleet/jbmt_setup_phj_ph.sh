#!/bin/bash
# Stage the jbmt-v2 harness and mount the snapshot-cloned data volume on every
# shard. Light-touch by design: it mounts and stages, and does NOT start any
# server and does NOT warm-read the device, because Unit 2's in-memory join
# benchmark is still measuring on these hosts and a 1536 GiB read would perturb
# it. The warm read is a separate step (jbmt_hydrate_phj_ph.sh) run after Unit 2.
#
# Usage: jbmt_setup_phj_ph.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
KEY=$HERE/ssh_phj_ph/id_ed25519
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15"
JBMT_SRC=${JBMT_SRC:-/mnt/data/jbmt_results/jbmt-sweep-20260724}
HOSTS=${HOSTS:-$HERE/hosts.phj_ph.tsv}
VOLS=${VOLS:-$HERE/volumes.phj_ph.tsv}

# The harness files that must sit next to join_bench_mt.py on every instance.
FILES="join_bench_mt.py join_bench_mt_servers.sh join_bench_mt_jobcsv.py
       join_bench_mt_legacy_cells.json join_bench_mt_queries.json
       join_bench_mt_schemas.json join_memory_bench.py"

while IFS=$'\t' read -r shard vol iid ip; do
    [ -n "${shard:-}" ] || continue
    (
        # Device by volume-id serial, not a guessed /dev/nvmeXn1 ordering.
        serial=$(echo "$vol" | tr -d '-')
        ssh $SSH_OPTS "ubuntu@$ip" "set -e
            dev=\$(ls /dev/disk/by-id/*${serial}* 2>/dev/null | head -1)
            [ -n \"\$dev\" ] || { echo 'FAILED: no device for $vol'; exit 1; }
            dev=\$(readlink -f \"\$dev\")
            echo \"device: \$dev\"
            sudo mkdir -p /mnt/data
            mountpoint -q /mnt/data || sudo mount \"\$dev\" /mnt/data
            mountpoint -q /mnt/data || { echo 'FAILED: mount'; exit 1; }
            findmnt -no FSTYPE,SIZE,USED /mnt/data
            sudo chown -R ubuntu:ubuntu /mnt/data/jbmt_server 2>/dev/null || true
            ls /mnt/data/ | head -20
            ls /mnt/data/jbmt_server/ 2>/dev/null | head -20
            mkdir -p /home/ubuntu/jbmt" </dev/null > "$HERE/jbmt_setup_shard$shard.log" 2>&1
        # shellcheck disable=SC2086
        scp -q $SSH_OPTS $(for f in $FILES; do echo "$JBMT_SRC/$f"; done | tr '\n' ' ') \
            "ubuntu@$ip:/home/ubuntu/jbmt/" </dev/null
        ssh $SSH_OPTS "ubuntu@$ip" "cd /home/ubuntu/jbmt && chmod +x join_bench_mt_servers.sh &&
            grep -m1 'TOOL_VERSION' join_bench_mt.py &&
            sha256sum join_bench_mt.py join_bench_mt_servers.sh | cut -c1-16 | tr '\n' ' ' && echo &&
            python3 join_bench_mt.py plan --suite real --tier a 2>&1 | tail -1 &&
            python3 join_bench_mt.py plan --suite synthetic 2>&1 | tail -1" \
            </dev/null >> "$HERE/jbmt_setup_shard$shard.log" 2>&1
        echo "shard $shard ($ip): jbmt staged"
    ) &
done < "$VOLS"
wait
echo "JBMT SETUP DONE: check jbmt_setup_shard*.log for device, mount, TOOL_VERSION, plan counts"
