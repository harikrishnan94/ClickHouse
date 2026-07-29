#!/bin/bash
# Launch the 8-shard fleet for the probe-phase A/B campaign and deploy both arms.
#
# Split out from the sweeps so the sweeps can be gated one at a time (the A/A
# control has to be scored green before the measured sweeps are acceptance
# evidence). Because this script exits with the fleet still up, the money
# protection is a watchdog: a detached timer that tears down THIS run's tag
# after WATCHDOG_HOURS whether or not the orchestrating session is still alive.
#
# The teardown guard only ever fires for a RUN_TAG that this script's own
# launch created: the pre-existing tag is captured first and a teardown is
# refused while run_tag.txt still holds it, so a fleet belonging to somebody
# else can never be torn down here.
#
# Usage: fleet_launch_deploy.sh
set -euo pipefail

BASE=/mnt/data/fleet_ab
CAMP=/mnt/ch/ClickHouse/tmp/probe_campaign_20260729
BIN_A=$CAMP/bins/clickhouse-baseline-a05f3ee81ff.bin
BIN_B=$CAMP/bins/clickhouse-phjph-fa5667f2da7.bin
WATCHDOG_HOURS=${WATCHDOG_HOURS:-10}

export AWS_PROFILE=${AWS_PROFILE:-Dev_AWS_Admin}
export AWS_REGION=${AWS_REGION:-ap-south-2}
export SHARDS=${SHARDS:-8}

cd "$BASE"
mkdir -p "$CAMP/logs"

echo "=== preflight $(date -u +%FT%TZ)"
for f in "$BIN_A" "$BIN_B"; do
    [ -s "$f" ] || { echo "FAILED: missing $f" >&2; exit 1; }
done
aws sts get-caller-identity --profile "$AWS_PROFILE" --region "$AWS_REGION" --output text >/dev/null \
    || { echo "FAILED: no usable AWS credentials for $AWS_PROFILE" >&2; exit 1; }

PRE_TAG=$(cat fleet/run_tag.txt 2>/dev/null || true)
echo "pre-existing run_tag: ${PRE_TAG:-<none>} (never torn down by this script)"

echo "=== launch $(date -u +%FT%TZ)"
fleet/launch.sh 2>&1 | tee "$CAMP/logs/fleet_launch.log"
OUR_TAG=$(cat fleet/run_tag.txt)
[ -n "$OUR_TAG" ] && [ "$OUR_TAG" != "$PRE_TAG" ] || {
    echo "FAILED: launch.sh did not record a new RUN_TAG; refusing to continue" >&2
    exit 1
}
echo "$OUR_TAG" > "$CAMP/logs/our_run_tag.txt"
echo "our RUN_TAG: $OUR_TAG"

# Money watchdog: survives this script, this shell and this session.
setsid bash -c "
    sleep $((WATCHDOG_HOURS * 3600))
    cd $BASE
    if [ \"\$(cat fleet/run_tag.txt 2>/dev/null)\" = '$OUR_TAG' ] || \
       [ -n \"\$(AWS_PROFILE=$AWS_PROFILE aws ec2 describe-instances --region $AWS_REGION \
                 --filters Name=tag:$OUR_TAG,Values='*' --query 'Reservations[]' --output text 2>/dev/null)\" ]
    then
        echo \"watchdog firing for $OUR_TAG at \$(date -u +%FT%TZ)\"
        RUN_TAG=$OUR_TAG fleet/teardown.sh
    fi
" > "$CAMP/logs/fleet_watchdog.log" 2>&1 < /dev/null &
echo "watchdog armed for $OUR_TAG (${WATCHDOG_HOURS}h)"

echo "=== deploy $(date -u +%FT%TZ)"
fleet/deploy.sh "$BIN_A" "$BIN_B" 2>&1 | tee "$CAMP/logs/fleet_deploy.log"

echo "=== deployed $(date -u +%FT%TZ)"
cat fleet/deployed.tsv
echo "=== READY: fleet $OUR_TAG is up with both arms deployed"
