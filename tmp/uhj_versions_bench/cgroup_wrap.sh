#!/usr/bin/env bash
# Emulate the published versions-benchmark machine inside a cgroup.
#
# Target (from versions/results/*.json "machine" field, and run-benchmark.sh
# default): c7a.4xlarge — 16 vCPU, 32 GiB RAM. Published disk is 1000 GB gp3;
# we only constrain CPU+RAM here (page cache counted via memory.max).
#
# On hosts with SMT, whole physical cores are taken with their siblings from
# /sys/devices/system/cpu/cpu*/topology/thread_siblings_list. On this ARM host
# each "core" is a singleton sibling list, so we pin 16 distinct CPUs.
set -euo pipefail

TARGET_VCPU="${TARGET_VCPU:-16}"
TARGET_MEM_BYTES="${TARGET_MEM_BYTES:-$((32 * 1024 * 1024 * 1024))}"  # 32 GiB
CGROOT="${CGROOT:-/sys/fs/cgroup/uhj_versions_bench}"
CGNAME="${CGNAME:-run}"

pick_cpus() {
    # Collect unique sibling groups in numeric CPU order, take enough whole
    # groups to reach TARGET_VCPU threads. Glob order is NOT numeric (cpu10
    # sorts before cpu2), so we sort explicitly.
    local -A seen=()
    local -a groups=()
    local cpu_id sibs key
    while read -r cpu_id; do
        local f="/sys/devices/system/cpu/cpu${cpu_id}/topology/thread_siblings_list"
        [ -f "${f}" ] || continue
        sibs="$(tr -d ' \n' < "${f}")"
        key="$(echo "${sibs}" | tr ',' '\n' | sort -n | paste -sd, -)"
        if [ -z "${seen[$key]+x}" ]; then
            seen[$key]=1
            groups+=("${key}")
        fi
    done < <(find /sys/devices/system/cpu -maxdepth 1 -type d -name 'cpu[0-9]*' -printf '%f\n' \
             | sed 's/^cpu//' | sort -n)

    local -a picked=()
    local g n=0
    for g in "${groups[@]}"; do
        IFS=',' read -ra parts <<<"${g}"
        if (( n + ${#parts[@]} > TARGET_VCPU )); then
            continue
        fi
        picked+=("${parts[@]}")
        n=$((n + ${#parts[@]}))
        (( n >= TARGET_VCPU )) && break
    done
    if (( n != TARGET_VCPU )); then
        echo "cgroup_wrap: could not pick exactly ${TARGET_VCPU} CPUs as whole sibling groups (got ${n})" >&2
        printf ' available groups: %s\n' "${groups[*]:0:20}" >&2
        exit 1
    fi
    # Emit a contiguous range when possible, else a comma list. Counting uses expand.
    local first="${picked[0]}" last="${picked[$((n-1))]}"
    if (( last - first + 1 == n )); then
        echo "${first}-${last}"
    else
        (IFS=,; echo "${picked[*]}")
    fi
}

ensure_controllers() {
    # Enable cpuset + memory on the root if needed, then on our parent.
    local ctrl
    for ctrl in cpuset memory; do
        if ! grep -qw "${ctrl}" /sys/fs/cgroup/cgroup.controllers 2>/dev/null; then
            echo "cgroup_wrap: controller ${ctrl} not available" >&2
            exit 1
        fi
    done
    # Enable on root subtree if not already.
    local cur
    cur="$(cat /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true)"
    for ctrl in cpuset memory; do
        if ! grep -qw "${ctrl}" <<<"${cur}"; then
            echo "+${ctrl}" | sudo tee /sys/fs/cgroup/cgroup.subtree_control >/dev/null
        fi
    done
}

setup_cgroup() {
    local cpus="$1"
    ensure_controllers
    sudo mkdir -p "${CGROOT}"
    # Parent must delegate controllers to children.
    local cur
    cur="$(cat "${CGROOT}/cgroup.subtree_control" 2>/dev/null || true)"
    for ctrl in cpuset memory; do
        if ! grep -qw "${ctrl}" <<<"${cur}"; then
            echo "+${ctrl}" | sudo tee "${CGROOT}/cgroup.subtree_control" >/dev/null
        fi
    done
    local cg="${CGROOT}/${CGNAME}"
    sudo mkdir -p "${cg}"
    echo "${cpus}" | sudo tee "${cg}/cpuset.cpus" >/dev/null
    echo 0 | sudo tee "${cg}/cpuset.mems" >/dev/null
    echo "${TARGET_MEM_BYTES}" | sudo tee "${cg}/memory.max" >/dev/null
    echo 0 | sudo tee "${cg}/memory.swap.max" >/dev/null
    # Also set cpu.max so AsynchronousMetrics CGroup* CPU rows are populated
    # (cpuset alone already limits NumberOfProcessors / max_threads).
    if [ -f "${cg}/cpu.max" ]; then
        echo "$((TARGET_VCPU * 100000)) 100000" | sudo tee "${cg}/cpu.max" >/dev/null
    fi
    echo "${cg}"
}

usage() {
    cat <<EOF
Usage:
  $0 [--verify] [--] <command...>
  $0 --print-cpus
  $0 --print-cg

Runs <command> inside a cgroup pinned to ${TARGET_VCPU} vCPUs and
${TARGET_MEM_BYTES} bytes RAM (swap.max=0).
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then usage; exit 0; fi
if [ "${1:-}" = "--print-cpus" ]; then pick_cpus; exit 0; fi

CPUS="$(pick_cpus)"
CG="$(setup_cgroup "${CPUS}")"

if [ "${1:-}" = "--print-cg" ]; then
    echo "cg=${CG}"
    echo "cpus=$(cat "${CG}/cpuset.cpus")"
    echo "memory.max=$(cat "${CG}/memory.max")"
    echo "memory.swap.max=$(cat "${CG}/memory.swap.max")"
    exit 0
fi

VERIFY=0
if [ "${1:-}" = "--verify" ]; then VERIFY=1; shift; fi
if [ "${1:-}" = "--" ]; then shift; fi

# Enter the cgroup for this process (and thus children).
echo $$ | sudo tee "${CG}/cgroup.procs" >/dev/null

if [ "${VERIFY}" = 1 ]; then
    echo "=== emulation verify ==="
    echo "nproc=$(nproc)"
    echo "--- lscpu -e ---"
    lscpu -e || true
    echo "--- cgroup ---"
    echo "cpuset.cpus=$(cat "${CG}/cpuset.cpus")"
    echo "memory.max=$(cat "${CG}/memory.max")"
    echo "memory.swap.max=$(cat "${CG}/memory.swap.max")"
    echo "self_cgroup=$(cat /proc/self/cgroup)"
    if [ "$#" -eq 0 ]; then
        exit 0
    fi
fi

exec "$@"
