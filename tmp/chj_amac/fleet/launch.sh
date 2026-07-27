#!/bin/bash
# Launch the 8-shard ARM perf fleet for the AMAC mission (Unit 4 only —
# requester-authorized: 8x m8g.24xlarge via AWS SSO profile Dev_AWS_Admin,
# region ap-south-2, terminate after the campaign). Patterns follow the prior
# fleet runs (tmp/jbmt-sweep/, /mnt/data/inmem_sweep_tri/LIVE.md).
#
# Produces: fleet/ssh/id_ed25519(.pub) (ephemeral, gitignored),
#           fleet/hosts.tsv (shard, instance-id, private-ip, az),
#           fleet/sg_id.txt, fleet/launch_receipt.json
# Every step is fail-closed; nothing here runs before Units 2-3 pass local
# gates. Record hosts.tsv + per-shard lscpu digests in PREREG.md at launch.
set -euo pipefail

PROFILE=Dev_AWS_Admin
REGION=ap-south-2
COUNT=${COUNT:-8}
ITYPE=${ITYPE:-m8g.24xlarge}
HERE=$(cd "$(dirname "$0")" && pwd -P)
TAG=chj-amac-shard

# 0. Identity + vCPU quota preflight (quota L-1216C47A includes m8g AND this
#    orchestration host — prior-fleet lesson).
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

# 1. Ephemeral keypair via cloud-init (ec2:ImportKeyPair is SCP-denied).
mkdir -p "$HERE/ssh"
if [ ! -f "$HERE/ssh/id_ed25519" ]
then
    ssh-keygen -t ed25519 -f "$HERE/ssh/id_ed25519" -N '' -C chj-amac-fleet
fi
PUBKEY=$(cat "$HERE/ssh/id_ed25519.pub")
cat > "$HERE/userdata.yaml" <<EOF
#cloud-config
ssh_authorized_keys:
  - $PUBKEY
EOF

# 2. Scratch security group: SSH only, only from this host.
VPC=$(aws ec2 describe-vpcs --profile "$PROFILE" --region "$REGION" \
      --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
MYIP=$(ec2metadata --local-ipv4 2>/dev/null || hostname -I | awk '{print $1}')
SG=$(aws ec2 create-security-group --profile "$PROFILE" --region "$REGION" \
     --group-name "chj-amac-$(date -u +%Y%m%d%H%M)" \
     --description "AMAC mission scratch fleet SG (delete at teardown)" \
     --vpc-id "$VPC" --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --profile "$PROFILE" --region "$REGION" \
    --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32"
echo "$SG" > "$HERE/sg_id.txt"

# 3. AMI via SSM parameter (Ubuntu 24.04 arm64).
AMI=$(aws ssm get-parameter --profile "$PROFILE" --region "$REGION" \
      --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
      --query 'Parameter.Value' --output text)

# 4. Launch. On InsufficientInstanceCapacity: retry in another AZ by hand.
aws ec2 run-instances --profile "$PROFILE" --region "$REGION" \
    --instance-type "$ITYPE" --count "$COUNT" --image-id "$AMI" \
    --security-group-ids "$SG" --user-data "file://$HERE/userdata.yaml" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","Throughput":600,"DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=Purpose,Value=chj-amac-perf},{Key=Owner,Value=harikrishnan}]" \
    > "$HERE/launch_receipt.json"

# 5. Wait for running + write hosts.tsv.
IDS=$(python3 -c "import json;d=json.load(open('$HERE/launch_receipt.json'));print(' '.join(i['InstanceId'] for i in d['Instances']))")
aws ec2 wait instance-running --profile "$PROFILE" --region "$REGION" --instance-ids $IDS
aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" --instance-ids $IDS \
    --query 'Reservations[].Instances[].[InstanceId,PrivateIpAddress,Placement.AvailabilityZone]' \
    --output text | awk 'BEGIN{i=0} {printf "%d\t%s\t%s\t%s\n", i++, $1, $2, $3}' \
    > "$HERE/hosts.tsv"
cat "$HERE/hosts.tsv"
echo "FLEET LAUNCHED: $(wc -l < "$HERE/hosts.tsv") shards; SG $SG; record hosts.tsv + lscpu in PREREG.md"
