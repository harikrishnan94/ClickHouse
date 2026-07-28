#!/bin/bash
# Launch the tagged 8-shard ARM fleet for the phj-ph A/B campaign.
#
# Same shape as fleet/launch.sh (which this campaign may not edit: it is the
# prior campaign's record), with three deliberate differences:
#   1. every instance AND every instance-created volume carries RUN_TAG, which
#      is what the G8 teardown proof filters on. launch.sh tags only
#      Name/Purpose/Owner, so a tag-filtered teardown proof over its fleet
#      would be vacuously empty - i.e. unable to fail.
#   2. instances are launched one at a time with an AZ fallback: the prior jbmt
#      campaign recorded batched run-instances for 24xlarge hitting
#      InsufficientInstanceCapacity while single launches succeeded.
#   3. artifacts land in *.phj_ph.* files so the prior campaign's hosts.tsv,
#      sg_id.txt and launch receipt stay intact as its audit trail.
#
# Root volume is 300 GiB (fleet_ab materializes the synthetic S4/S5 tables on
# it); the jbmt real-suite data arrives later on a separate snapshot-cloned
# volume.
#
# Produces: fleet/ssh_phj_ph/id_ed25519(.pub), fleet/hosts.phj_ph.tsv,
#           fleet/sg_id.phj_ph.txt, fleet/launch_receipt.phj_ph.<n>.json
set -euo pipefail

PROFILE=Dev_AWS_Admin
REGION=ap-south-2
COUNT=${COUNT:-8}
ITYPE=${ITYPE:-m8g.24xlarge}
RUN_TAG=${RUN_TAG:-phj-ph-ab-20260728}
HERE=$(cd "$(dirname "$0")" && pwd -P)
TAG=chj-phjph-shard

# 0. Identity + vCPU quota preflight (L-1216C47A counts this orchestration host too).
aws sts get-caller-identity --profile "$PROFILE" --region "$REGION" >/dev/null
QUOTA=$(aws service-quotas get-service-quota --profile "$PROFILE" --region "$REGION" \
        --service-code ec2 --quota-code L-1216C47A --query 'Quota.Value' --output text)
RUNNING=$(aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" \
        --filters "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[].Instances[].CpuOptions.[CoreCount,ThreadsPerCore]' --output text |
        awk '{s+=$1*$2} END {print s+0}')
NEED=$((COUNT * 96))
echo "quota=$QUOTA running_vcpus=$RUNNING need=$NEED"
if [ "$(printf '%.0f' "$QUOTA")" -lt "$((RUNNING + NEED))" ]
then
    echo "FAILED: vCPU quota headroom insufficient" >&2
    exit 1
fi

# 1. Ephemeral keypair, injected via cloud-init (ec2:ImportKeyPair is SCP-denied).
mkdir -p "$HERE/ssh_phj_ph"
if [ ! -f "$HERE/ssh_phj_ph/id_ed25519" ]
then
    ssh-keygen -t ed25519 -f "$HERE/ssh_phj_ph/id_ed25519" -N '' -C chj-phjph-fleet
fi
cat > "$HERE/userdata.phj_ph.yaml" <<EOF
#cloud-config
ssh_authorized_keys:
  - $(cat "$HERE/ssh_phj_ph/id_ed25519.pub")
EOF

# 2. Scratch security group: SSH only, only from this host.
VPC=$(aws ec2 describe-vpcs --profile "$PROFILE" --region "$REGION" \
      --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
MYIP=$(ec2metadata --local-ipv4 2>/dev/null || hostname -I | awk '{print $1}')
if [ -s "$HERE/sg_id.phj_ph.txt" ]
then
    SG=$(cat "$HERE/sg_id.phj_ph.txt")
    echo "reusing SG $SG"
else
    SG=$(aws ec2 create-security-group --profile "$PROFILE" --region "$REGION" \
         --group-name "chj-phjph-$(date -u +%Y%m%d%H%M)" \
         --description "phj-ph A/B campaign scratch fleet SG (delete at teardown)" \
         --vpc-id "$VPC" \
         --tag-specifications "ResourceType=security-group,Tags=[{Key=RUN_TAG,Value=$RUN_TAG},{Key=Name,Value=$TAG-sg},{Key=Owner,Value=harikrishnan}]" \
         --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --profile "$PROFILE" --region "$REGION" \
        --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32"
    echo "$SG" > "$HERE/sg_id.phj_ph.txt"
fi

# 3. AMI via SSM parameter (Ubuntu 24.04 arm64).
AMI=$(aws ssm get-parameter --profile "$PROFILE" --region "$REGION" \
      --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
      --query 'Parameter.Value' --output text)
echo "ami=$AMI sg=$SG vpc=$VPC"

# 4. Launch one at a time, preferring one AZ so the snapshot clones can follow.
SUBNETS=$(aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" \
          --filters "Name=vpc-id,Values=$VPC" \
          --query 'Subnets[?AvailabilityZone==`ap-south-2a`].SubnetId | [0]' --output text)
for az in ap-south-2b ap-south-2c
do
    extra=$(aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" \
            --filters "Name=vpc-id,Values=$VPC" \
            --query "Subnets[?AvailabilityZone==\`$az\`].SubnetId | [0]" --output text)
    SUBNETS="$SUBNETS $extra"
done

IDS=""
for i in $(seq 0 $((COUNT - 1)))
do
    launched=""
    for subnet in $SUBNETS
    do
        if out=$(aws ec2 run-instances --profile "$PROFILE" --region "$REGION" \
            --instance-type "$ITYPE" --count 1 --image-id "$AMI" \
            --subnet-id "$subnet" --security-group-ids "$SG" \
            --user-data "file://$HERE/userdata.phj_ph.yaml" \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3","Throughput":600,"Iops":6000,"DeleteOnTermination":true}}]' \
            --tag-specifications \
              "ResourceType=instance,Tags=[{Key=Name,Value=$TAG$i},{Key=Purpose,Value=chj-phjph-perf},{Key=Owner,Value=harikrishnan},{Key=RUN_TAG,Value=$RUN_TAG}]" \
              "ResourceType=volume,Tags=[{Key=Name,Value=$TAG$i-root},{Key=Purpose,Value=chj-phjph-perf},{Key=Owner,Value=harikrishnan},{Key=RUN_TAG,Value=$RUN_TAG}]" \
            --output json 2>"$HERE/launch_err.phj_ph.$i.txt")
        then
            echo "$out" > "$HERE/launch_receipt.phj_ph.$i.json"
            launched=$(python3 -c "import json,sys;print(json.load(open('$HERE/launch_receipt.phj_ph.$i.json'))['Instances'][0]['InstanceId'])")
            echo "shard $i: $launched in $subnet"
            break
        fi
        echo "shard $i: $subnet unavailable ($(tr -d '\n' < "$HERE/launch_err.phj_ph.$i.txt" | tail -c 200))" >&2
    done
    [ -n "$launched" ] || { echo "FAILED: no capacity for shard $i in any AZ" >&2; exit 1; }
    IDS="$IDS $launched"
done

# 5. Wait for running + write hosts.phj_ph.tsv.
aws ec2 wait instance-running --profile "$PROFILE" --region "$REGION" --instance-ids $IDS
aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" --instance-ids $IDS \
    --query 'Reservations[].Instances[].[InstanceId,PrivateIpAddress,Placement.AvailabilityZone]' \
    --output text | sort | awk 'BEGIN{i=0} {printf "%d\t%s\t%s\t%s\n", i++, $1, $2, $3}' \
    > "$HERE/hosts.phj_ph.tsv"
cat "$HERE/hosts.phj_ph.tsv"
echo "FLEET LAUNCHED: $(wc -l < "$HERE/hosts.phj_ph.tsv") shards; SG $SG; RUN_TAG $RUN_TAG"
