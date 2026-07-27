#!/bin/bash
# Terminate the fleet and delete the scratch SG. Run ONLY after the final
# independent-verification spot re-runs are done. Prints the accounting the
# REPORT needs (all instance ids + final states + SG deletion).
set -euo pipefail
PROFILE=Dev_AWS_Admin
REGION=ap-south-2
HERE=$(cd "$(dirname "$0")" && pwd -P)

IDS=$(cut -f2 "$HERE/hosts.tsv" | tr '\n' ' ')
[ -n "$IDS" ] || { echo "FAILED: hosts.tsv empty"; exit 1; }
aws ec2 terminate-instances --profile "$PROFILE" --region "$REGION" --instance-ids $IDS \
    --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
aws ec2 wait instance-terminated --profile "$PROFILE" --region "$REGION" --instance-ids $IDS
echo "all instances terminated"

SG=$(cat "$HERE/sg_id.txt")
aws ec2 delete-security-group --profile "$PROFILE" --region "$REGION" --group-id "$SG"
echo "SG $SG deleted"
aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" --instance-ids $IDS \
    --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text
echo "TEARDOWN COMPLETE $(date -u +%FT%TZ) — copy this output into REPORT.md accounting"
