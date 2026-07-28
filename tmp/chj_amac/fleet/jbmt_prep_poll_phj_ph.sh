#!/bin/bash
# Poll the detached per-shard jbmt prep and pull its log back.
# Usage: jbmt_prep_poll_phj_ph.sh
# Exits 0 only when every shard has written prep.done with rc 0.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
KEY=$HERE/ssh_phj_ph/id_ed25519
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=10"
HOSTS=${HOSTS:-$HERE/hosts.phj_ph.tsv}

alldone=1; anyfail=0
while IFS=$'\t' read -r shard iid ip az; do
    [ -n "${shard:-}" ] || continue
    info=$(ssh $SSH_OPTS "ubuntu@$ip" "cd /home/ubuntu/jbmt 2>/dev/null || exit 9
        rc=\$(cat prep.done 2>/dev/null || echo -)
        last=\$(grep -aE '^(---|===|warm read done|prepare-keys done|  K[0-9]:|arm .* up|cloned)' prep.log 2>/dev/null | tail -1)
        free=\$(df -h /mnt/data | tail -1 | awk '{print \$4}')
        echo \"rc=\$rc free=\$free | \$last\"" </dev/null 2>/dev/null)
    echo "shard $shard: $info"
    case "$info" in
        *"rc=-"*) alldone=0 ;;
        *"rc=0"*) : ;;
        *) anyfail=1 ;;
    esac
    scp -q $SSH_OPTS "ubuntu@$ip:/home/ubuntu/jbmt/prep.log" "$HERE/jbmt_prep_shard$shard.log" </dev/null 2>/dev/null
done < "$HOSTS"
echo "all_done=$alldone any_failed=$anyfail"
[ "$alldone" = 1 ] && [ "$anyfail" = 0 ] || exit 2
exit 0
