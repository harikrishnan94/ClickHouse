#!/bin/bash
# Create one per-shard data volume from the jbmt snapshot for every host in
# hosts.phj_ph.tsv, attach it at /dev/sdf, and record the mapping.
#
# Why not `join_bench_mt.py fleet-volumes`: that helper tags only
# `Name=jbmt-<tag>` and cannot add RUN_TAG, but G8's teardown proof filters on
# `tag:RUN_TAG` and this campaign requires the tag to exist *at creation time* -
# a volume created untagged and tagged a second later is invisible to the proof
# if the run dies in between. The volume shape here is byte-for-byte the shape
# the helper hard-codes (gp3, 4000 IOPS, 1000 MBps, from the same snapshot), so
# the measurement environment is the prior campaign's; only the tagging differs.
# The snapshot itself is read-only to this run and is never modified.
set -euo pipefail

PROFILE=Dev_AWS_Admin
REGION=ap-south-2
SNAPSHOT=${SNAPSHOT:-snap-021cbdc2484f86607}
RUN_TAG=${RUN_TAG:-phj-ph-ab-20260728}
HERE=$(cd "$(dirname "$0")" && pwd -P)
HOSTS=${HOSTS:-$HERE/hosts.phj_ph.tsv}
OUT=$HERE/volumes.phj_ph.tsv

# Fail closed if the source snapshot is not ready.
STATE=$(aws ec2 describe-snapshots --profile "$PROFILE" --region "$REGION" \
        --snapshot-ids "$SNAPSHOT" --query 'Snapshots[0].State' --output text)
[ "$STATE" = completed ] || { echo "FAILED: snapshot $SNAPSHOT state=$STATE" >&2; exit 1; }
echo "snapshot $SNAPSHOT state=$STATE"

: > "$OUT"
while IFS=$'\t' read -r shard iid ip az; do
    [ -n "${shard:-}" ] || continue
    VOL=$(aws ec2 create-volume --profile "$PROFILE" --region "$REGION" \
          --snapshot-id "$SNAPSHOT" --availability-zone "$az" \
          --volume-type gp3 --iops 4000 --throughput 1000 \
          --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=jbmt-phjph-shard$shard},{Key=Purpose,Value=chj-phjph-perf},{Key=Owner,Value=harikrishnan},{Key=RUN_TAG,Value=$RUN_TAG}]" \
          --query 'VolumeId' --output text)
    printf "%s\t%s\t%s\t%s\n" "$shard" "$VOL" "$iid" "$ip" >> "$OUT"
    echo "shard $shard: created $VOL in $az for $iid"
done < "$HOSTS"

VOLS=$(cut -f2 "$OUT" | tr '\n' ' ')
aws ec2 wait volume-available --profile "$PROFILE" --region "$REGION" --volume-ids $VOLS
echo "all volumes available"

while IFS=$'\t' read -r shard vol iid ip; do
    aws ec2 attach-volume --profile "$PROFILE" --region "$REGION" \
        --device /dev/sdf --volume-id "$vol" --instance-id "$iid" \
        --query '[VolumeId,InstanceId,State]' --output text
done < "$OUT"

aws ec2 wait volume-in-use --profile "$PROFILE" --region "$REGION" --volume-ids $VOLS
echo "VOLUMES READY: $(wc -l < "$OUT") attached at /dev/sdf; mapping in $OUT; RUN_TAG=$RUN_TAG"
