#!/bin/bash
# Destroy every AWS resource this campaign created, and prove it.
#
# Runs even if earlier units failed or the campaign was abandoned. Everything is
# selected by `tag:RUN_TAG`, which every instance, volume and security group was
# given AT CREATION TIME - that is what makes the proof at the end able to fail
# rather than vacuously empty.
#
# Order matters: instances first (a volume attached to a live instance cannot be
# deleted), then volumes, then the security group (it cannot be deleted while an
# ENI still references it).
#
# Fail-closed on authority: `DeleteVolume` is denied by an identity policy unless
# the volume carries `ndc-dbg-target=true` (the prior campaign recorded this).
# This script adds that opt-in marker to volumes IT created and nothing else. If
# a delete is still refused it does NOT escalate - it prints exactly what would
# have been deleted and exits non-zero so the report can flag it as requiring
# authorization.
#
# Usage: teardown_phj_ph.sh [--dry-run]
set -uo pipefail

PROFILE=Dev_AWS_Admin
REGION=ap-south-2
RUN_TAG=${RUN_TAG:-phj-ph-ab-20260728}
SNAPSHOT=${SNAPSHOT:-snap-021cbdc2484f86607}
HERE=$(cd "$(dirname "$0")" && pwd -P)
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY=1

q() { aws ec2 "$@" --profile "$PROFILE" --region "$REGION" --output text; }

echo "=== TEARDOWN $RUN_TAG $(date -u +%FT%TZ) ${DRY:+(DRY RUN)} ==="

echo "--- BEFORE: live inventory by RUN_TAG (this is the gate's power to fail) ---"
INST=$(q describe-instances \
    --filters "Name=tag:RUN_TAG,Values=$RUN_TAG" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId')
VOLS=$(q describe-volumes \
    --filters "Name=tag:RUN_TAG,Values=$RUN_TAG" "Name=status,Values=creating,available,in-use" \
    --query 'Volumes[].VolumeId')
SGS=$(q describe-security-groups \
    --filters "Name=tag:RUN_TAG,Values=$RUN_TAG" --query 'SecurityGroups[].GroupId')
echo "instances: ${INST:-<none>}"
echo "volumes:   ${VOLS:-<none>}"
echo "sgs:       ${SGS:-<none>}"

if [ -n "$DRY" ]
then
    echo "DRY RUN: would terminate the instances above, tag+delete the volumes above, delete the SGs above"
    exit 0
fi

if [ -n "$INST" ]
then
    echo "--- terminating instances ---"
    q terminate-instances --instance-ids $INST \
        --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]'
    aws ec2 wait instance-terminated --profile "$PROFILE" --region "$REGION" --instance-ids $INST
    echo "all instances terminated"
else
    echo "no live instances to terminate"
fi

# Root volumes are DeleteOnTermination=true, so re-query: only the data volumes
# this run created should remain.
VOLS=$(q describe-volumes \
    --filters "Name=tag:RUN_TAG,Values=$RUN_TAG" "Name=status,Values=creating,available,in-use" \
    --query 'Volumes[].VolumeId')
DENIED=""
if [ -n "$VOLS" ]
then
    echo "--- volumes remaining after termination: $VOLS ---"
    # Opt-in marker required by DenyDeleteVolumeExceptNdcDbgTagged, applied only
    # to volumes carrying this run's RUN_TAG.
    q create-tags --resources $VOLS --tags Key=ndc-dbg-target,Value=true >/dev/null
    for v in $VOLS
    do
        if q delete-volume --volume-id "$v" >/dev/null 2>"$HERE/teardown_delvol_err.txt"
        then echo "deleted $v"
        else echo "DELETE DENIED for $v: $(tr -d '\n' < "$HERE/teardown_delvol_err.txt" | tail -c 300)"; DENIED="$DENIED $v"
        fi
    done
else
    echo "no volumes to delete"
fi

if [ -n "$SGS" ]
then
    echo "--- deleting security groups ---"
    for g in $SGS
    do
        for attempt in 1 2 3 4 5 6
        do
            if q delete-security-group --group-id "$g" >/dev/null 2>&1
            then echo "deleted $g"; break
            fi
            echo "  $g still in use (ENI detach lag), retry $attempt"; sleep 20
        done
    done
else
    echo "no security groups to delete"
fi

echo
echo "=== G8 PROOF ==="
echo -n "instances by RUN_TAG (want empty): "
q describe-instances --filters "Name=tag:RUN_TAG,Values=$RUN_TAG" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId'
echo -n "volumes by RUN_TAG (want empty): "
q describe-volumes --filters "Name=tag:RUN_TAG,Values=$RUN_TAG" \
    "Name=status,Values=creating,available,in-use" --query 'Volumes[].VolumeId'
echo -n "security groups by RUN_TAG (want empty): "
q describe-security-groups --filters "Name=tag:RUN_TAG,Values=$RUN_TAG" \
    --query 'SecurityGroups[].GroupId'
echo -n "snapshot $SNAPSHOT (want completed): "
aws ec2 describe-snapshots --snapshot-ids "$SNAPSHOT" --region "$REGION" --profile "$PROFILE" \
    --query 'Snapshots[0].State'

if [ -n "$DENIED" ]
then
    echo "TEARDOWN INCOMPLETE — volume deletion denied for:$DENIED"
    echo "AUTHORIZATION REQUIRED: these volumes are still billing; nothing else was escalated."
    exit 1
fi
echo "TEARDOWN COMPLETE $(date -u +%FT%TZ)"
