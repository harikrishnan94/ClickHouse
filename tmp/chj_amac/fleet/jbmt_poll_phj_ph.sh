#!/bin/bash
# Poll a detached jbmt sweep and pull the per-shard results back.
# Usage: jbmt_poll_phj_ph.sh <label> [--collect-only]
# Prints one line per shard: done-marker, unit progress, result row count.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
KEY=$HERE/ssh_phj_ph/id_ed25519
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=10"
HOSTS=${HOSTS:-$HERE/hosts.phj_ph.tsv}
LABEL=${1:?label}
OUT=$HERE/jbmt_results_phj_ph
mkdir -p "$OUT"

alldone=1
while IFS=$'\t' read -r shard iid ip az; do
    [ -n "${shard:-}" ] || continue
    info=$(ssh $SSH_OPTS "ubuntu@$ip" "cd /home/ubuntu/jbmt 2>/dev/null || exit 9
        d=\$(cat sweep.$LABEL.done 2>/dev/null || echo -)
        n=\$(wc -l < results.$LABEL.shard$shard.jsonl 2>/dev/null || echo 0)
        p=\$(grep -aoE 'unit [0-9]+/[0-9]+' sweep.$LABEL.shard$shard.log 2>/dev/null | tail -1)
        t=\$(tail -c 400 sweep.$LABEL.shard$shard.log 2>/dev/null | tr '\n' ' ' | tail -c 160)
        echo \"rc=\$d rows=\$n \$p | \$t\"" </dev/null 2>/dev/null)
    echo "shard $shard: $info"
    case "$info" in *"rc=-"*) alldone=0 ;; esac
    scp -q $SSH_OPTS "ubuntu@$ip:/home/ubuntu/jbmt/results.$LABEL.shard$shard.jsonl" \
        "$OUT/results.$LABEL.shard$shard.jsonl" </dev/null 2>/dev/null
    scp -q $SSH_OPTS "ubuntu@$ip:/home/ubuntu/jbmt/sweep.$LABEL.shard$shard.log" \
        "$OUT/sweep.$LABEL.shard$shard.log" </dev/null 2>/dev/null
done < "$HOSTS"
echo "collected into $OUT ; all_done=$alldone"
[ "$alldone" = 1 ] || exit 2
