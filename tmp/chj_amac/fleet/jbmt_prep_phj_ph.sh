#!/bin/bash
# Prepare every shard for the two-arm jbmt sweeps (Units 3 and 4).
#
# Per host, in this order and for these reasons:
#   1. warm-read the data files. EBS volumes cloned from a snapshot lazy-load
#      their blocks, and the prior campaign's *first* touch of a tier-b table
#      blew the harness's fixed 600 s budget for that reason. Reading the files
#      once (not the whole 1536 GiB device - only ~417 GiB is used) pulls them
#      from S3 now, on nobody's measurement clock.
#   2. start ONE server on the original data root and run `prepare-keys` for
#      K0-K9, which the 347 legacy cells read and which the sweep does not
#      create. Done pre-clone, so step 4 gives both arms the same keys at zero
#      extra bytes; per-arm preparation would double time and disk.
#   3. stop that server: `clone` refuses a data root with a live `status` file.
#   4. hardlink-clone data -> data_b and start both arms (A=baseline on 9005,
#      B=candidate on 9006), which is the topology the two-arm ABAB sweep needs.
#   5. `verify` both tiers against the snapshot's own loads.<tier>.json
#      reference, so a corrupted or partially-hydrated clone is caught before it
#      can be measured.
#
# Nothing here alters the measurement protocol; it only makes the venue ready.
# Usage: jbmt_prep_phj_ph.sh [--skip-keys]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
KEY=$HERE/ssh_phj_ph/id_ed25519
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
HOSTS=${HOSTS:-$HERE/hosts.phj_ph.tsv}
BIN_A=/home/ubuntu/chj/clickhouse-base
BIN_B=/home/ubuntu/chj/clickhouse-cand
ROOT=/mnt/data/jbmt_server
KEYS=${KEYS:-K0,K1,K2,K3,K4,K5,K6,K7,K8,K9}
SKIP_KEYS=""
[ "${1:-}" = "--skip-keys" ] && SKIP_KEYS=1

while IFS=$'\t' read -r shard iid ip az; do
    [ -n "${shard:-}" ] || continue
    log=$HERE/jbmt_prep_shard$shard.log
    remote_script=$(cat <<REMOTE
set -euo pipefail
cd /home/ubuntu/jbmt
echo "=== [\$(date -u +%FT%TZ)] shard $shard prep start ==="

echo "--- warm-read data files (EBS lazy-load hydration) ---"
# -P 64, not -P 8: at 8-way the first pass ran at ~93 MB/s (measured from
# /proc/diskstats), i.e. IOPS/lazy-load-latency bound rather than throughput
# bound on a gp3 volume provisioned for 1000 MB/s. Snapshot first-touch reads
# are dominated by per-block S3 fetch latency, which only concurrency hides.
sudo find $ROOT/data -type f -print0 | sudo xargs -0 -P 64 -n 16 cat > /dev/null
echo "warm read done: \$(date -u +%FT%TZ)"
df -h /mnt/data | tail -1

if [ -z "${SKIP_KEYS}" ]
then
    echo "--- prepare-keys $KEYS on the pre-clone data root ---"
    if [ ! -f $ROOT/keys_done ]
    then
        ./join_bench_mt_servers.sh start PREP $BIN_A $ROOT/data 9005 8125 $ROOT/arm_prep
        python3 join_bench_mt.py prepare-keys --binary $BIN_A --port 9005 --keys $KEYS
        touch $ROOT/keys_done
        ./join_bench_mt_servers.sh stop $ROOT/arm_prep
    else
        echo "keys already prepared (keys_done marker present)"
    fi
    echo "prepare-keys done: \$(date -u +%FT%TZ)"
    df -h /mnt/data | tail -1
fi

echo "--- clone + start both arms ---"
[ -e $ROOT/data_b ] || ./join_bench_mt_servers.sh clone $ROOT/data $ROOT/data_b
./join_bench_mt_servers.sh start baseline  $BIN_A $ROOT/data   9005 8125 $ROOT/arm_a
./join_bench_mt_servers.sh start candidate $BIN_B $ROOT/data_b 9006 8126 $ROOT/arm_b

echo "--- binary identity as the servers actually run them ---"
for p in 9005 9006; do
    pid=\$(grep -oE '[0-9]+' \$( [ \$p = 9005 ] && echo $ROOT/arm_a/server.pid || echo $ROOT/arm_b/server.pid ))
    echo "port \$p pid \$pid sha256 \$(sudo sha256sum /proc/\$pid/exe | cut -c1-16)"
done

echo "--- verify both tiers against the snapshot reference ---"
python3 join_bench_mt.py verify --tier a --binary $BIN_A --port 9005 --reference $ROOT/loads.a.json 2>&1 | tail -3
python3 join_bench_mt.py verify --tier b --binary $BIN_A --port 9005 --reference $ROOT/loads.b.json 2>&1 | tail -3
echo "=== [\$(date -u +%FT%TZ)] shard $shard prep OK ==="
REMOTE
)
    # Detached on the instance: prepare-keys is a multi-hour step and an ssh
    # drop must not kill it half-way through a 1.024B-row INSERT.
    printf '%s\n' "$remote_script" > "$HERE/prep_remote_shard$shard.sh"
    # Skip a shard whose prep is already running, so a relaunch after a driver
    # fault cannot start a second warm read / prepare-keys on the same host.
    # The bracket keeps the pattern from matching the very ssh shell that runs
    # it: `pgrep -f 'bash prep.sh'` matched its own command line on every host
    # and reported all 8 as already-running when only shard 0 was.
    if ssh $SSH_OPTS "ubuntu@$ip" "pgrep -f 'bash pre[p].sh' >/dev/null" </dev/null 2>/dev/null
    then
        echo "shard $shard ($ip): prep already running, left alone"
        continue
    fi
    scp -q $SSH_OPTS "$HERE/prep_remote_shard$shard.sh" "ubuntu@$ip:/home/ubuntu/jbmt/prep.sh" </dev/null
    # Double-fork: `bash -c 'cmd &'` leaves the child as the ssh shell's direct
    # child when setsid does not fork (the background child is not a group
    # leader), and the shell then blocks in wait() - so ssh never returns. The
    # extra subshell is what actually detaches it. (Same fix fleet_ab.py carries.)
    ssh $SSH_OPTS "ubuntu@$ip" "cd /home/ubuntu/jbmt && rm -f prep.done &&
        (setsid nohup bash -c 'bash prep.sh > prep.log 2>&1; echo \$? > prep.done' \
         </dev/null >/dev/null 2>&1 &) && echo launched" </dev/null
    echo "shard $shard ($ip): prep launched (detached)"
done < "$HOSTS"
echo "JBMT PREP LAUNCHED on all shards; poll with jbmt_prep_poll_phj_ph.sh"
