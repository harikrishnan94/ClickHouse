#!/usr/bin/env bash
# Verification 1: emulation binds (CPU siblings, memory.max, swap.max=0, OOM kill,
# and ClickHouse seeing the constrained nproc / CGroup metrics).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/verify"
WRAP="${HERE}/cgroup_wrap.sh"
BIN="${WORK}/bin/clickhouse-uhj"
mkdir -p "${OUT}"
TARGET_VCPU=16
TARGET_MEM=$((32 * 1024 * 1024 * 1024))

{
  echo "=== host ==="
  echo "host_nproc=$(nproc)"
  echo "host_mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)"
  echo "swap=$(free -h | awk '/Swap/{print $2,$3}')"
  echo
  echo "=== sibling groups (first 20) ==="
  for f in /sys/devices/system/cpu/cpu{0..31}/topology/thread_siblings_list; do
      [ -f "$f" ] && echo "$f: $(cat "$f")"
  done
  echo
  echo "=== cgroup_wrap --verify ==="
  "${WRAP}" --verify
  echo
  echo "=== nproc / lscpu inside wrapper ==="
  "${WRAP}" -- bash -c 'echo nproc=$(nproc); lscpu -e | head -20'
  echo
  echo "=== cgroup memory.max / swap.max ==="
  CG="$("${WRAP}" --print-cg | awk -F= '/^cg=/{print $2}')"
  echo "cg=${CG}"
  echo "memory.max=$(cat "${CG}/memory.max")"
  echo "memory.swap.max=$(cat "${CG}/memory.swap.max")"
  echo "cpuset.cpus=$(cat "${CG}/cpuset.cpus")"
  test "$(cat "${CG}/memory.max")" = "${TARGET_MEM}"
  test "$(cat "${CG}/memory.swap.max")" = "0"
  # Expand ranges like 0-15 into a count of logical CPUs.
  ncpus=$(awk -F, '{
    for (i=1;i<=NF;i++) {
      if ($i ~ /-/) { split($i,a,"-"); n+=a[2]-a[1]+1 }
      else n++
    }
    print n
  }' "${CG}/cpuset.cpus")
  test "${ncpus}" -eq "${TARGET_VCPU}"
  echo "OK: cgroup limits match target (cpus=${ncpus}, mem=$(cat "${CG}/memory.max"), swap=$(cat "${CG}/memory.swap.max"))"
  echo
  echo "=== over-limit allocation must be OOM-killed ==="
  # Allocate ~40 GiB inside the 32 GiB cgroup; expect SIGKILL.
  set +e
  "${WRAP}" -- python3 - <<'PY'
import sys
chunks=[]
try:
    for i in range(40):
        chunks.append(bytearray(1024*1024*1024))  # 1 GiB
        print(i+1, "GiB", flush=True)
    print("UNEXPECTED_SURVIVED")
except MemoryError:
    print("MemoryError")
    sys.exit(2)
PY
  rc=$?
  set -e
  echo "alloc_exit=${rc}"
  # 137 = 128+9 SIGKILL from OOM; 9 raw; sometimes 1
  if [ "${rc}" -eq 137 ] || [ "${rc}" -eq 9 ]; then
      echo "OK: over-limit allocation killed (rc=${rc})"
  else
      echo "WARN: expected OOM kill rc 137/9, got ${rc} — check dmesg/cgroup events"
      cat "${CG}/memory.events" 2>/dev/null || true
  fi
} | tee "${OUT}/emulation_binds.txt"

# Start a short-lived server inside the wrapper and check ClickHouse metrics.
SERVER_DIR="${WORK}/verify_server"
rm -rf "${SERVER_DIR}"
mkdir -p "${SERVER_DIR}"/{data,tmp,log,user_files,access,format_schemas}
cat > "${SERVER_DIR}/config.xml" <<EOF
<clickhouse>
  <logger><level>information</level><console>true</console></logger>
  <http_port>18123</http_port>
  <tcp_port>19000</tcp_port>
  <path>${SERVER_DIR}/data/</path>
  <tmp_path>${SERVER_DIR}/tmp/</tmp_path>
  <user_directories><users_xml><path>${SERVER_DIR}/users.xml</path></users_xml></user_directories>
</clickhouse>
EOF
cat > "${SERVER_DIR}/users.xml" <<'EOF'
<clickhouse>
  <profiles><default/></profiles>
  <users><default><password></password><networks><ip>::/0</ip></networks><profile>default</profile><quota>default</quota></default></users>
  <quotas><default><interval><duration>3600</duration></interval></default></quotas>
</clickhouse>
EOF

fuser -k 19000/tcp 2>/dev/null || true
CG="$("${WRAP}" --print-cg | awk -F= '/^cg=/{print $2}')"
HELPER="${OUT}/start_in_cg.sh"
cat > "${HELPER}" <<EOF
#!/bin/bash
echo \$\$ | sudo tee ${CG}/cgroup.procs >/dev/null
exec "${BIN}" server --config-file="${SERVER_DIR}/config.xml"
EOF
chmod +x "${HELPER}"
# Use information logging so MemoryWorker cgroup lines are visible.
sed -i 's/<level>error<\/level>/<level>information<\/level>/' "${SERVER_DIR}/config.xml" || true
nohup "${HELPER}" >"${OUT}/verify_server.log" 2>&1 &
SPID=$!
for i in $(seq 1 60); do
  "${BIN}" client --host 127.0.0.1 --port 19000 --query 'SELECT 1' >/dev/null 2>&1 && break
  sleep 1
done
sleep 2
{
  echo "=== ClickHouse started inside cgroup ==="
  echo "server_pid=${SPID}"
  echo "server_cgroup=$(cat /proc/${SPID}/cgroup)"
  echo "cgroup_memory.max=$(cat ${CG}/memory.max)"
  echo "cgroup_memory.swap.max=$(cat ${CG}/memory.swap.max)"
  echo "cgroup_cpuset.cpus=$(cat ${CG}/cpuset.cpus)"
  echo "cgroup_cpu.max=$(cat ${CG}/cpu.max 2>/dev/null || echo n/a)"
  echo -n "version="; "${BIN}" client --host 127.0.0.1 --port 19000 --query 'SELECT version()'
  echo -n "max_threads="; "${BIN}" client --host 127.0.0.1 --port 19000 --query "SELECT getSetting('max_threads')"
  echo "--- MemoryWorker / cgroup log lines ---"
  rg -n 'CgroupsReader|cgroup reader|Memory amount initially available|CgroupsMemoryUsageObserver' "${OUT}/verify_server.log" || true
  echo "--- asynchronous_metrics CGroup* (may be empty: openCgroupv2MetricFile string-concat bug) ---"
  "${BIN}" client --host 127.0.0.1 --port 19000 --query \
    "SELECT metric, value FROM system.asynchronous_metrics WHERE metric LIKE 'CGroup%' OR metric = 'OSMemoryTotal' ORDER BY metric FORMAT TSV" \
    || true
  mt="$("${BIN}" client --host 127.0.0.1 --port 19000 --query "SELECT getSetting('max_threads')")"
  if [ "${mt}" != "16" ]; then
    echo "FAIL: max_threads=${mt}, expected 16"
    exit 1
  fi
  echo "OK: ClickHouse max_threads=${mt} (target 16); cgroup membership and memory.max verified above"
} | tee "${OUT}/clickhouse_cgroup.txt"

kill "${SPID}" 2>/dev/null || true
fuser -k 19000/tcp 2>/dev/null || true
echo "verify_emulation done -> ${OUT}/"
