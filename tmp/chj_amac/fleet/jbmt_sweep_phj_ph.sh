#!/bin/bash
# Launch one two-arm jbmt sweep per shard, detached on the instance, and return
# immediately. The harness runs ON each instance (not driven over ssh per query)
# so no ssh round-trip lands inside a timed run.
#
# Both arms must already be resident (jbmt_prep_phj_ph.sh). Arm names are
# `baseline` and `candidate`, which is what G7's `report-ab --arm-a/--arm-b`
# expects. `--algorithms parallel_hash` everywhere: partitioned_hash does not
# exist in these binaries, so asking for it would produce FALLBACK rows rather
# than data. The single-algorithm warning about losing the cross-*algorithm*
# oracle is expected here and is not a defect: with two arms the cross-*arm*
# (row_count, checksum) reference is the correctness oracle.
#
# Run counts, warmups and the 600 s budget are harness constants with no CLI
# knob; nothing here overrides them.
#
# Usage: jbmt_sweep_phj_ph.sh <label> <suite> <tier> [--only-file FILE]
#   e.g. jbmt_sweep_phj_ph.sh syn synthetic a --only-file legacy_rx.txt
#        jbmt_sweep_phj_ph.sh real_a real a
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
KEY=$HERE/ssh_phj_ph/id_ed25519
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15"
HOSTS=${HOSTS:-$HERE/hosts.phj_ph.tsv}
LABEL=${1:?label}
SUITE=${2:?suite}
TIER=${3:?tier}
ONLY_FILE=""
if [ "${4:-}" = "--only-file" ]; then ONLY_FILE=${5:?only file}; fi
BIN_A=/home/ubuntu/chj/clickhouse-base
BIN_B=/home/ubuntu/chj/clickhouse-cand
# NSHARDS is the PLAN partitioning, which is not the same as the number of hosts
# being launched right now: a subset of hosts may be launched while the rest are
# still finishing an earlier suite, and the plan must still be cut the same way
# for every shard. Defaults to the host count, but an explicit NSHARDS wins.
NSHARDS=${NSHARDS:-$(grep -c . "$HOSTS")}

while IFS=$'\t' read -r shard iid ip az; do
    [ -n "${shard:-}" ] || continue
    only_arg=""
    if [ -n "$ONLY_FILE" ]
    then
        scp -q $SSH_OPTS "$ONLY_FILE" "ubuntu@$ip:/home/ubuntu/jbmt/only.$LABEL.txt" </dev/null
        only_arg="--only \"\$(cat /home/ubuntu/jbmt/only.$LABEL.txt)\""
    fi
    ssh $SSH_OPTS "ubuntu@$ip" "set -e
        cd /home/ubuntu/jbmt
        rm -f sweep.$LABEL.done
        setsid nohup bash -c '
            python3 join_bench_mt.py sweep \
                --arm baseline=$BIN_A:9005 --arm candidate=$BIN_B:9006 \
                --algorithms parallel_hash \
                --suite $SUITE --tier $TIER \
                --shards $NSHARDS --shard $shard \
                --results /home/ubuntu/jbmt/results.$LABEL.shard$shard.jsonl \
                $only_arg > /home/ubuntu/jbmt/sweep.$LABEL.shard$shard.log 2>&1
            echo \$? > /home/ubuntu/jbmt/sweep.$LABEL.done
        ' </dev/null >/dev/null 2>&1 &
        sleep 1; echo launched" </dev/null
    echo "shard $shard ($ip): $LABEL sweep launched"
done < "$HOSTS"
echo "JBMT SWEEP $LABEL LAUNCHED on $NSHARDS shards (detached; poll sweep.$LABEL.done)"
